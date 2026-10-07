import torch
import torch.nn as nn

from .base import FunRecModel, SubModel, Layer, Dense, init_tensor_
from .utils import (
    concat_group_embedding,
    build_input_layer,
    build_group_feature_embedding_table_dict,
    FeatureEmbedding,
)
from .layers import SampledSoftmaxLayer, UserAttention, GatedFusion, _MultiHeadAttention


class KerasLSTM(Layer):
    """与 Keras LSTM 层数值等价的 LSTM 层（惰性构建，不使用掩码）

    - activation=tanh, recurrent_activation=sigmoid，门顺序 Keras (i, f, c, o) 与 torch (i, f, g, o) 相同
    - kernel: glorot_uniform，recurrent_kernel: 由 recurrent_initializer 指定（Keras 默认 orthogonal），
      bias: zeros 且遗忘门部分为 1（unit_forget_bias=True）
    - 输入为 [batch_size, seq_len, input_dim]
    参数在内部以 torch.nn.LSTM 保存，可通过 load_keras_weights 从 Keras 权重
    （kernel [in, 4u], recurrent_kernel [u, 4u], bias [4u]）加载。

    参数:
        units (int): 隐藏单元数
        return_sequences (bool): 是否返回所有时间步的隐藏状态，否则只返回最后一个时间步
        recurrent_initializer: 循环权重初始化器
    """

    def __init__(
        self,
        units,
        return_sequences=False,
        recurrent_initializer="orthogonal",
        name=None,
        **kwargs,
    ):
        super(KerasLSTM, self).__init__(name=name)
        self.units = units
        self.return_sequences = return_sequences
        self.recurrent_initializer = recurrent_initializer

    def build(self, input_shape):
        in_dim = input_shape[-1]
        u = self.units
        self.lstm = nn.LSTM(
            in_dim, u, num_layers=1, bias=True, batch_first=True
        ).to(self._build_device)
        # Keras LSTM 只有一个 bias，torch 的 bias_hh 固定为 0 且不参与训练（保持参数量与更新方式一致）
        self.lstm.bias_hh_l0.requires_grad_(False)
        kernel = init_tensor_(torch.empty(in_dim, 4 * u), "glorot_uniform")
        recurrent_kernel = torch.empty(u, 4 * u)
        if self.recurrent_initializer == "orthogonal":
            nn.init.orthogonal_(recurrent_kernel)
        else:
            init_tensor_(recurrent_kernel, self.recurrent_initializer)
        bias = torch.zeros(4 * u)
        bias[u : 2 * u] = 1.0
        self.load_keras_weights(kernel, recurrent_kernel, bias)

    @torch.no_grad()
    def load_keras_weights(self, kernel, recurrent_kernel, bias):
        """从 Keras LSTM 权重加载参数"""
        dev = self.lstm.weight_ih_l0.device
        kernel, recurrent_kernel, bias = (
            torch.as_tensor(t, dtype=torch.float32)
            for t in (kernel, recurrent_kernel, bias)
        )
        self.lstm.weight_ih_l0.copy_(kernel.t().to(dev))
        self.lstm.weight_hh_l0.copy_(recurrent_kernel.t().to(dev))
        self.lstm.bias_ih_l0.copy_(bias.to(dev))
        self.lstm.bias_hh_l0.zero_()

    def forward(self, inputs):
        outputs, (last_h, _) = self.lstm(inputs)
        if self.return_sequences:
            return outputs  # [batch_size, seq_len, units]
        return last_h[0]  # [batch_size, units]


class SDMModel(FunRecModel):
    """SDM (Sequential Deep Matching) 的 PyTorch 实现

    - 主模型输出: 每个样本的采样 softmax 损失 [B, 1]（配合 sampledsoftmaxloss 使用）
    - user_tower: 融合长短期兴趣后的用户向量 [B, emb_dim]，输入为 user / raw_hist_seq / raw_hist_seq_long 组特征
    - item_tower: 物品 embedding [B, emb_dim]，输入为物品 id
    """

    def __init__(self, feature_columns, model_config):
        # 从model_config中提取参数
        label_name = model_config.get("label_name", "movie_id")
        dnn_activation = model_config.get("dnn_activation", "tanh")
        emb_dim = model_config.get("emb_dim", 16)
        num_heads = model_config.get("num_heads", 2)
        num_sampled = model_config.get("num_sampled", 20)

        # 从feature_columns中获取item_vocab_size
        item_vocab_size = None
        for fc in feature_columns:
            if fc.name == label_name:
                item_vocab_size = fc.vocab_size
                break

        if item_vocab_size is None:
            raise ValueError(
                f"Could not find vocab_size for label feature '{label_name}'"
            )

        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="sdm")

        self.label_name = label_name
        self.feature_columns = list(feature_columns)
        self.user_feature_columns = [
            fc
            for fc in feature_columns
            if any(
                g in fc.group for g in ["user", "raw_hist_seq", "raw_hist_seq_long"]
            )
        ]
        user_feature_names = [fc.name for fc in self.user_feature_columns]
        # 长期行为序列特征名称（与原实现中按组字典的顺序一致）
        self.long_history_names = [
            fc.name
            for fc in feature_columns
            if "raw_hist_seq_long" in fc.group and fc.emb_name is not None
        ]

        # 构建特征embedding表
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 用户表示
        self.user_dense = Dense(emb_dim, activation=dnn_activation)

        # ------------ 短期兴趣建模 ------------
        # 1. 序列信息学习层（LSTM）
        self.short_history_dense = Dense(emb_dim, activation=dnn_activation)
        self.lstm_layer = KerasLSTM(
            emb_dim, return_sequences=True, recurrent_initializer="glorot_uniform"
        )
        # 2. 多兴趣提取层（多头自注意力）
        self.short_norm = nn.LayerNorm(emb_dim, eps=1e-3)
        self.short_term_attention = _MultiHeadAttention(
            num_heads=num_heads,
            key_dim=emb_dim // num_heads,
            dropout=0.1,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="short_term_attention",
        )
        self.short_term_norm = nn.LayerNorm(emb_dim, eps=1e-3)
        # 3. 用户个性化注意力层
        self.user_attention_short = UserAttention(name="user_attention_short")

        # ------------ 长期兴趣建模 ------------
        self.user_attention_long = nn.ModuleDict(
            {
                name: UserAttention(name=f"user_attention_long_{name}")
                for name in self.long_history_names
            }
        )
        self.long_term_dense = Dense(emb_dim, activation=dnn_activation)

        # ------------ 长短期兴趣融合 ------------
        self.gated_fusion = GatedFusion(name="gated_fusion")

        # ------------ 预测层 ------------
        # 构建采样softmax层
        self.sampled_softmax_layer = SampledSoftmaxLayer(
            item_vocab_size, num_sampled, emb_dim
        )

        # 构建用户模型和物品模型用于评估（与主模型共享参数）
        self.user_tower = SubModel(
            self, "encode_user", user_feature_names, name="user_model"
        )
        self.item_tower = SubModel(self, "encode_item", [label_name], name="item_model")

    def _final_interest(self, group_embedding_feature_dict, inputs):
        """计算融合长短期兴趣后的用户向量 [B, emb_dim]"""
        # 用户表示
        user_emb_concat = concat_group_embedding(
            group_embedding_feature_dict, "user", flatten=False
        )  # [None, 1, dim * num_user_features]
        user_embedding = self.user_dense(user_emb_concat)  # [None, 1, dim]

        # ------------ 短期兴趣建模 ------------
        # 1. 序列信息学习层（LSTM）
        # 获取短期会话序列特征
        short_history_features = group_embedding_feature_dict["raw_hist_seq"]
        short_history_item_embs = []
        short_history_mask = None
        for name, short_history_feature in short_history_features.items():
            short_history_mask = (inputs[name] != 0).float().unsqueeze(
                -1
            )  # [None, max_len, 1]

            # 使用mask
            short_history_item_embs.append(
                short_history_feature * short_history_mask
            )  # [None, max_len, dim]
        short_history_item_emb_concat = torch.cat(
            short_history_item_embs, dim=-1
        )  # [None, max_len, dim * num_short_history_features]
        short_history_item_emb = self.short_history_dense(
            short_history_item_emb_concat
        )  # [None, max_len, dim]

        # 使用LSTM处理序列（与原实现一致，LSTM 不接收掩码）
        sequence_output = self.lstm_layer(short_history_item_emb)

        # 2. 多兴趣提取层（多头自注意力）
        # 对LSTM输出应用多头注意力
        # 注意: 与原实现一致，attention_mask 为最后一个短期序列特征的掩码 [None, max_len, 1]
        # （作用在 query 维度上并在 key 维度上广播）
        norm_sequence_output = self.short_norm(sequence_output)
        sequence_output = self.short_term_attention(
            norm_sequence_output, sequence_output, attention_mask=short_history_mask
        )  # [None, max_len, dim]

        short_term_output = self.short_term_norm(
            sequence_output
        )  # [None, max_len, dim]

        # 3. 用户个性化注意力层
        short_term_interest = self.user_attention_short(
            user_embedding, short_term_output
        )  # [None, 1, dim]

        # ------------ 长期兴趣建模 ------------
        # 从不同特征维度对长期行为进行聚合
        long_history_features = group_embedding_feature_dict["raw_hist_seq_long"]

        long_term_interests = []
        for name, long_history_feature in long_history_features.items():
            long_history_mask = (inputs[name] != 0).float().unsqueeze(
                -1
            )  # [None, max_len_long, 1]
            long_history_item_emb = (
                long_history_feature * long_history_mask
            )  # [None, max_len_long, dim]

            long_term_interests.append(
                self.user_attention_long[name](user_embedding, long_history_item_emb)
            )  # [None, 1, dim]

        long_term_interests_concat = torch.cat(
            long_term_interests, dim=-1
        )  # [None, 1, dim * len(long_history_features)]
        long_term_interest = self.long_term_dense(
            long_term_interests_concat
        )  # [None, 1, dim]

        # ------------ 长短期兴趣融合 ------------
        final_interest = self.gated_fusion(
            [user_embedding, short_term_interest, long_term_interest]
        )  # [None, 1, dim]
        final_interest = torch.squeeze(final_interest, dim=1)  # [None, dim]
        return final_interest

    def encode_user(self, inputs):
        """用户塔: 返回用户向量 [B, emb_dim]"""
        # 仅查询用户塔所需特征的 embedding（与主模型共享 embedding 表）
        group_embedding_feature_dict = build_group_feature_embedding_table_dict(
            self.user_feature_columns,
            inputs,
            self.embedding.embedding_table_dict,
            self.embedding.mean_pooling,
        )
        return self._final_interest(group_embedding_feature_dict, inputs)

    def encode_item(self, inputs):
        """物品塔: 返回物品 embedding [B, emb_dim]"""
        item_table = self.embedding.embedding_table_dict[self.label_name]
        # 获取item嵌入
        return torch.squeeze(item_table(inputs[self.label_name]), dim=1)

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)
        final_interest = self._final_interest(group_embedding_feature_dict, inputs)

        # ------------ 预测层 ------------
        # 获取item嵌入权重
        item_embedding_weight = self.embedding.embedding_table_dict[
            self.label_name
        ].embeddings
        # 计算输出
        output = self.sampled_softmax_layer(
            [item_embedding_weight, final_interest, inputs[self.label_name]]
        )
        return output


def build_sdm_model(feature_columns, model_config):
    """
    构建SDM (Sequential Deep Matching)模型

    返回:
    (model, user_model, item_model): 主模型、用户模型（融合长短期兴趣的用户向量）、物品模型（共享参数）
    """
    model = SDMModel(feature_columns, model_config)
    return model, model.user_tower, model.item_tower
