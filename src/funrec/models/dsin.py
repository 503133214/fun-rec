import torch
import torch.nn as nn

from .base import FunRecModel, Layer, Dense, Embedding
from .utils import (
    build_input_layer,
    parse_group_feature_columns,
    FeatureEmbedding,
    concat_group_embedding,
    concat_func,
)
from .layers import DNNs, PredictLayer, _MultiHeadAttention


def _truncated_normal_initializer(stddev):
    """与 Keras TruncatedNormal(mean=0, stddev) 一致：截断在两倍标准差处"""

    def init(tensor):
        nn.init.trunc_normal_(tensor, 0.0, stddev, -2.0 * stddev, 2.0 * stddev)

    return init


# 在 build_dsin_model 函数之前添加 BiasEncoding 类
class BiasEncoding(Layer):
    """
    DSIN 模型的偏置编码层。

    该层为会话嵌入添加三种类型的偏置：
    1. 会话偏置：每个会话不同（捕获会话级特征）
    2. 位置偏置：会话内每个位置不同（捕获时间模式）
    3. Item偏置：每个嵌入维度不同（捕获特征级偏置）

    公式：BE(k,t,c) = w_k^K + w_t^T + w_c^C
    其中：
    - k：会话索引
    - t：会话内位置索引
    - c：嵌入维度索引
    """

    def __init__(self, sess_max_count, sess_max_len, seed=1024, **kwargs):
        super(BiasEncoding, self).__init__(**kwargs)
        self.sess_max_count = sess_max_count
        self.sess_max_len = sess_max_len
        # 保留 seed 参数以兼容原接口（PyTorch 中使用全局随机数生成器）
        self.seed = seed

    def build(self, input_shape):
        # 从输入形状获取嵌入大小
        # input_shape: [batch_size, sess_max_count, sess_max_len, embedding_dim]
        if len(input_shape) == 4:
            embedding_dim = input_shape[-1]
        else:
            raise ValueError(f"期望4维输入形状，得到 {input_shape}")

        initializer = _truncated_normal_initializer(0.0001)

        # 会话偏置：每个会话不同
        # 形状：[sess_max_count, 1, 1]
        self.sess_bias_embedding = self.add_weight(
            "sess_bias_embedding",
            shape=(self.sess_max_count, 1, 1),
            initializer=initializer,
            trainable=True,
        )

        # 位置偏置：会话内每个位置不同
        # 形状：[1, sess_max_len, 1]
        self.seq_bias_embedding = self.add_weight(
            "seq_bias_embedding",
            shape=(1, self.sess_max_len, 1),
            initializer=initializer,
            trainable=True,
        )

        # Item偏置：每个嵌入维度不同
        # 形状：[1, 1, embedding_dim]
        self.item_bias_embedding = self.add_weight(
            "item_bias_embedding",
            shape=(1, 1, embedding_dim),
            initializer=initializer,
            trainable=True,
        )

    def forward(self, inputs, **kwargs):
        """
        对会话嵌入应用偏置编码。

        参数：
            inputs：会话嵌入张量
                    形状：[batch_size, sess_max_count, sess_max_len, embedding_dim]

        返回：
            偏置编码后的会话嵌入，形状相同
        """
        # 将所有三种偏置添加到输入中
        # 广播将处理维度对齐
        # 形状：[batch_size, sess_max_count, sess_max_len, embedding_dim]
        encoded_inputs = (
            inputs
            + self.sess_bias_embedding  # 会话偏置
            + self.seq_bias_embedding  # 位置偏置
            + self.item_bias_embedding
        )  # Item偏置

        return encoded_inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            "sess_max_count": self.sess_max_count,
            "sess_max_len": self.sess_max_len,
            "seed": self.seed,
        }


class KerasLSTM(Layer):
    """与 tf.keras.layers.LSTM(return_sequences=True) 数值等价的 LSTM 层

    - 参数布局与 Keras 一致: kernel [D, 4u]（glorot_uniform）、recurrent_kernel [u, 4u]（orthogonal）、
      bias [4u]（零初始化，遗忘门部分为 1，即 unit_forget_bias=True），门顺序为 i, f, c, o
    - dropout / recurrent_dropout 与 Keras 一致：每次调用为每个门各生成一个 dropout 掩码，
      在所有时间步上共享（recurrent_dropout > 0 时 Keras 使用 implementation=1，即每个门独立掩码）
    - go_backwards=True 时逆序处理输入，输出序列也为逆序（与 Keras 一致）
    """

    def __init__(self, units, dropout=0.0, recurrent_dropout=0.0, go_backwards=False, name=None, **kwargs):
        super().__init__(name=name)
        self.units = int(units)
        self.dropout = min(1.0, max(0.0, float(dropout)))
        self.recurrent_dropout = min(1.0, max(0.0, float(recurrent_dropout)))
        self.go_backwards = go_backwards

    def build(self, input_shape):
        in_dim = input_shape[-1]
        u = self.units
        self.kernel = self.add_weight("kernel", (in_dim, 4 * u), "glorot_uniform")
        self.recurrent_kernel = self.add_weight(
            "recurrent_kernel", (u, 4 * u), lambda t: nn.init.orthogonal_(t)
        )
        self.bias = self.add_weight("bias", (4 * u,), "zeros")
        with torch.no_grad():
            self.bias[u : 2 * u].fill_(1.0)

    def _dropout_masks(self, like, rate):
        """生成 4 个（每个门一个）在时间步间共享的 dropout 掩码"""
        if not (self.training and 0.0 < rate < 1.0):
            return None
        keep = 1.0 - rate
        return [
            torch.bernoulli(torch.full_like(like, keep)) / keep for _ in range(4)
        ]

    def forward(self, inputs):
        # inputs: [batch_size, T, D] -> [batch_size, T, units]
        batch_size, steps = inputs.shape[0], inputs.shape[1]
        u = self.units
        h = inputs.new_zeros(batch_size, u)
        c = inputs.new_zeros(batch_size, u)
        dp_mask = self._dropout_masks(inputs[:, 0, :], self.dropout)
        rec_dp_mask = self._dropout_masks(h, self.recurrent_dropout)
        k_i, k_f, k_c, k_o = torch.split(self.kernel, u, dim=1)
        r_i, r_f, r_c, r_o = torch.split(self.recurrent_kernel, u, dim=1)
        b_i, b_f, b_c, b_o = torch.split(self.bias, u, dim=0)

        time_steps = range(steps - 1, -1, -1) if self.go_backwards else range(steps)
        outputs = []
        for t in time_steps:
            x = inputs[:, t, :]
            if dp_mask is not None:
                x_i, x_f, x_c, x_o = (x * m for m in dp_mask)
            else:
                x_i = x_f = x_c = x_o = x
            if rec_dp_mask is not None:
                h_i, h_f, h_c, h_o = (h * m for m in rec_dp_mask)
            else:
                h_i = h_f = h_c = h_o = h
            i = torch.sigmoid(x_i @ k_i + b_i + h_i @ r_i)
            f = torch.sigmoid(x_f @ k_f + b_f + h_f @ r_f)
            c = f * c + i * torch.tanh(x_c @ k_c + b_c + h_c @ r_c)
            o = torch.sigmoid(x_o @ k_o + b_o + h_o @ r_o)
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class BidirectionalLSTM(nn.Module):
    """对应 tf.keras.layers.Bidirectional(LSTM(return_sequences=True), backward_layer=LSTM(go_backwards=True))

    后向层输出在时间维度上翻转回正序后与前向输出拼接（merge_mode='concat'）。
    """

    def __init__(self, units, dropout=0.0, recurrent_dropout=0.0, name=None):
        super().__init__()
        self.layer_name = name
        self.forward_layer = KerasLSTM(
            units, dropout=dropout, recurrent_dropout=recurrent_dropout,
            name="session_interaction_lstm_forward",
        )
        self.backward_layer = KerasLSTM(
            units, dropout=dropout, recurrent_dropout=recurrent_dropout,
            go_backwards=True, name="session_interaction_lstm_backward",
        )

    def forward(self, inputs):
        y = self.forward_layer(inputs)
        y_rev = torch.flip(self.backward_layer(inputs), dims=[1])
        return torch.cat([y, y_rev], dim=-1)


class DSINModel(FunRecModel):
    """DSIN (深度会话兴趣网络) 排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        session_feature_list = model_config.get("session_feature_list", ["video_id"])
        sess_max_count = model_config.get("sess_max_count", 5)
        sess_max_len = model_config.get("sess_max_len", 10)
        bias_encoding = model_config.get("bias_encoding", True)
        att_embedding_size = model_config.get("att_embedding_size", 8)
        att_head_num = model_config.get("att_head_num", 2)
        dnn_units = model_config.get("dnn_units", [128, 64, 1])
        dropout_rate = model_config.get("dropout_rate", 0.2)
        try:
            l2_reg = float(model_config.get("l2_reg", 1e-6))
        except Exception:
            l2_reg = 0.000001

        # 为所有特征构建输入定义
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="dsin")
        self.feature_columns = list(feature_columns)

        # 为所有特征构建嵌入表
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 查找用于会话处理的 video_id 嵌入（独立的嵌入表，mask_zero=True）
        # 注: 原 Keras 实现中该掩码在 tf.stack 处被丢弃，后续的多头注意力 / LSTM 均未收到掩码
        self.video_id_embedding_layer = None
        video_emb_dim = None
        for fc in feature_columns:
            if fc.name == "video_id":
                self.video_id_embedding_layer = Embedding(
                    fc.vocab_size,
                    fc.emb_dim,
                    name="video_id_session_embedding",
                    mask_zero=True,
                    l2_reg=l2_reg,
                )
                video_emb_dim = fc.emb_dim
                break

        # 会话输入特征名（按会话顺序）
        self.session_keys = []
        for sess_idx in range(sess_max_count):
            for feature_name in session_feature_list:
                session_key = f"sess_{sess_idx}_{feature_name}"
                if session_key in input_layer_dict:
                    self.session_keys.append(session_key)
        self.session_inputs_found = len(self.session_keys) > 0

        # 如果启用，应用偏置编码
        self.bias_encoder = (
            BiasEncoding(sess_max_count, sess_max_len, seed=1024) if bias_encoding else None
        )

        # DSIN 组件 1：会话兴趣提取（每个会话一个多头注意力 + 层归一化）
        # 注: Keras MultiHeadAttention 输出维度默认与 query 的维度（嵌入维度）相同
        self.session_attention_layers = nn.ModuleList(
            [
                _MultiHeadAttention(
                    num_heads=att_head_num,
                    key_dim=att_embedding_size,
                    dropout=dropout_rate,
                    name=f"session_attention_{sess_idx}",
                )
                for sess_idx in range(sess_max_count)
            ]
        )
        self.session_attention_norms = nn.ModuleList(
            [nn.LayerNorm(video_emb_dim, eps=1e-3) for _ in range(sess_max_count)]
        )

        # DSIN 组件 2：会话兴趣交互（双向 LSTM + 层归一化）
        d_model = att_embedding_size * att_head_num
        self.session_interaction_lstm = BidirectionalLSTM(
            d_model // 2,  # 每个方向的单元数为一半，以保持输出大小一致
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name="session_interaction_lstm",
        )
        self.session_interaction_norm = nn.LayerNorm(2 * (d_model // 2), eps=1e-3)

        # DSIN 组件 3：会话兴趣激活的注意力打分层
        self.attention_score_interests = Dense(1, activation="tanh", name="attention_score_interests")
        self.attention_score_interactions = Dense(
            1, activation="tanh", name="attention_score_interactions"
        )

        # DNN 层与最终预测层
        self.dnn = DNNs(dnn_units, use_bn=True, dropout_rate=dropout_rate)
        self.dsin_output = PredictLayer(name="dsin_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        # 获取常规 DNN 输入（非会话特征）
        # 形状：[batch_size, total_embedding_dim]
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "dnn")

        # 提取会话嵌入
        session_embeddings = []
        for session_key in self.session_keys:
            # 获取会话输入张量
            # 形状：[batch_size, sess_max_len]
            session_input = inputs[session_key]

            # 获取此会话的嵌入
            # 形状：[batch_size, sess_max_len, embedding_dim]
            session_emb = self.video_id_embedding_layer(session_input)
            session_embeddings.append(session_emb)

        # 堆叠会话嵌入
        # 形状：[batch_size, sess_max_count, sess_max_len, embedding_dim]
        session_embeddings_stack = torch.stack(session_embeddings, dim=1)

        # 如果启用，应用偏置编码
        if self.bias_encoder is not None:
            session_embeddings_stack = self.bias_encoder(session_embeddings_stack)

        # DSIN 组件 1：会话兴趣提取
        session_interests = apply_session_interest_extractor(
            session_embeddings_stack,
            self.session_attention_layers,
            self.session_attention_norms,
        )

        # DSIN 组件 2：会话兴趣交互
        session_interactions = apply_session_interest_interaction(
            session_interests,
            self.session_interaction_lstm,
            self.session_interaction_norm,
        )

        # DSIN 组件 3：会话兴趣激活
        # 获取用于激活的目标Item嵌入
        target_item_embedding = get_target_item_embedding_by_prefix(
            group_embedding_feature_dict, "embedding/video_id", self.feature_columns
        )

        # 基于目标Item激活会话兴趣和交互
        activated_session_interests = apply_attention_activation(
            session_interests, target_item_embedding, self.attention_score_interests
        )
        activated_session_interactions = apply_attention_activation(
            session_interactions, target_item_embedding, self.attention_score_interactions
        )

        # 组合所有特征进行最终预测
        all_features = [dnn_inputs]

        # 如果可用，添加会话特征
        if self.session_inputs_found:
            all_features.extend(
                [activated_session_interests, activated_session_interactions]
            )

        # 连接所有特征
        # 形状：[batch_size, total_feature_dim]
        final_inputs = concat_func(all_features, axis=-1)

        # 应用 DNN 层
        # 形状：[batch_size, 1]（最终预测）
        dnn_logits = self.dnn(final_inputs)

        # 最终预测层
        # 形状：[batch_size, 1] 带 sigmoid 激活
        output = self.dsin_output(dnn_logits)
        return output


def build_dsin_model(feature_columns, model_config):
    """
    构建深度会话兴趣网络 (DSIN) 模型用于 CTR 预测。

    DSIN 引入会话概念来更好地建模用户行为序列。
    关键组件：
    1. 会话划分：基于时间间隔将用户行为序列划分为会话
    2. 偏置编码：应用会话、位置和Item偏置（如果启用）
    3. 会话兴趣提取：使用多头注意力提取会话级兴趣
    4. 会话兴趣交互：使用双向 LSTM 建模会话间交互
    5. 会话兴趣激活：应用注意力激活相关的会话兴趣

    参数：
        feature_columns：特征列定义列表
        model_config：包含以下参数的字典：
            - session_feature_list：list，例如 ['video_id']
            - sess_max_count：int
            - sess_max_len：int
            - bias_encoding：bool
            - att_embedding_size：int
            - att_head_num：int
            - dnn_units：list
            - dropout_rate：float
            - l2_reg：float

    返回：
        (model, None, None)：排序模型元组
    """
    model = DSINModel(feature_columns, model_config)
    return model, None, None


def apply_session_interest_extractor(session_embeddings, attention_layers, norm_layers):
    """
    使用多头自注意力应用会话兴趣提取器。

    这是 DSIN 的核心组件，用于提取会话级兴趣。
    对于每个会话，我们应用多头自注意力来捕获
    该会话内Item之间的关系。

    教育说明：
    - 多头注意力允许模型同时关注会话的不同方面
    - 自注意力意味着会话中的每个Item都关注同一会话中的所有其他Item
    - 这捕获了会话内的依赖关系和兴趣
    - 偏置编码可能已应用于输入，以提供位置和会话级信息

    参数：
        session_embeddings：堆叠的会话嵌入（可能已偏置编码）
                           形状：[batch_size, sess_max_count, sess_max_len, embedding_dim]
        attention_layers：每个会话一个多头注意力层（_MultiHeadAttention，
                          num_heads=att_head_num, key_dim=att_embedding_size, dropout=dropout_rate）
        norm_layers：每个会话一个层归一化层

    返回：
        torch.Tensor：会话兴趣张量
                   形状：[batch_size, sess_max_count, embedding_dim]
                   （多头注意力的输出维度与输入嵌入维度相同）
    """
    session_interests = []

    for sess_idx in range(len(attention_layers)):
        # 获取此会话的会话嵌入
        # 形状：[batch_size, sess_max_len, embedding_dim]
        session_emb = session_embeddings[:, sess_idx, :, :]

        # 在此会话内应用多头自注意力
        # 这捕获同一会话中Item之间的关系（原实现中无掩码）
        # 形状：[batch_size, sess_max_len, embedding_dim]
        attention_output = attention_layers[sess_idx](session_emb, session_emb)

        # 应用层归一化以提高训练稳定性
        attention_output = norm_layers[sess_idx](attention_output)

        # 应用平均池化获取会话级表示
        # 这将会话中的所有Item聚合为单一表示
        # 形状：[batch_size, embedding_dim]
        session_interest = torch.mean(attention_output, dim=1)
        session_interests.append(session_interest)

    # 堆叠所有会话兴趣
    # 形状：[batch_size, sess_max_count, embedding_dim]
    session_interests = torch.stack(session_interests, dim=1)

    return session_interests


def apply_session_interest_interaction(session_interests, bilstm_layer, norm_layer):
    """
    使用双向 LSTM 应用会话兴趣交互。

    该组件建模会话之间的时间关系。
    直觉是会话按时间排序，后续会话
    可能受到早期会话的影响。

    教育说明：
    - 双向 LSTM 在前向和后向方向处理会话
    - 这捕获了过去会话如何影响当前会话以及
      未来会话如何为当前会话提供上下文
    - 输出捕获会话间依赖关系

    参数：
        session_interests：会话兴趣张量
                          形状：[batch_size, sess_max_count, embedding_dim]
        bilstm_layer：双向 LSTM 层（BidirectionalLSTM，每个方向 d_model // 2 个单元）
        norm_layer：层归一化层

    返回：
        torch.Tensor：会话交互输出
                   形状：[batch_size, sess_max_count, (d_model // 2) * 2]
    """

    # 应用双向 LSTM 建模会话间的时间依赖关系
    # 前向方向：过去会话如何影响当前会话
    # 后向方向：未来会话如何为当前会话提供上下文
    session_interactions = bilstm_layer(session_interests)

    # 应用层归一化以提高训练稳定性
    session_interactions = norm_layer(session_interactions)

    return session_interactions


def apply_attention_activation(session_features, target_item_embedding, score_layer):
    """
    基于目标Item对会话特征应用注意力激活。

    该组件基于会话特征与目标Item的相关性来激活会话特征。
    直觉是并非所有会话对预测用户对特定Item的兴趣都同等相关。

    教育说明：
    - 注意力机制允许模型关注最相关的会话
    - 目标Item嵌入为我们试图预测的内容提供上下文
    - 这创建了会话特征的加权组合

    参数：
        session_features：会话特征张量
                         形状：[batch_size, sess_max_count, feature_dim]
        target_item_embedding：目标Item嵌入
                              形状：[batch_size, embedding_dim]
        score_layer：计算注意力分数的密集层 Dense(1, activation='tanh')

    返回：
        torch.Tensor：激活的会话特征
                   形状：[batch_size, feature_dim]
    """

    # 获取维度
    sess_max_count = session_features.shape[1]

    # 扩展目标Item嵌入以匹配会话维度
    # 形状：[batch_size, 1, embedding_dim]
    target_expanded = target_item_embedding.unsqueeze(1)

    # 为每个会话重复目标嵌入
    # 形状：[batch_size, sess_max_count, embedding_dim]
    target_repeated = target_expanded.repeat(1, sess_max_count, 1)

    # 将会话特征与目标Item嵌入连接
    # 这为计算注意力分数提供上下文
    # 形状：[batch_size, sess_max_count, feature_dim + embedding_dim]
    combined_features = torch.cat([session_features, target_repeated], dim=-1)

    # 使用密集层计算注意力分数
    # 形状：[batch_size, sess_max_count, 1]
    attention_scores = score_layer(combined_features)

    # 应用 softmax 获取注意力权重
    # 这确保权重在会话间的和为 1
    # 形状：[batch_size, sess_max_count, 1]
    attention_weights = torch.softmax(attention_scores, dim=1)

    # 将注意力权重应用于会话特征
    # 这创建了会话特征的加权组合
    # 形状：[batch_size, feature_dim]
    activated_features = torch.sum(session_features * attention_weights, dim=1)
    return activated_features


def _dnn_group_embedding_names(feature_columns, group_name="dnn"):
    """返回与 group_embedding_feature_dict[group_name] 列表一一对应的、原 Keras 实现中的张量名前缀

    - sparse 特征: 'embedding/<emb_name>/embedding_lookup'
    - mean 聚合的变长特征: 'sequence_mean_pooling_layer'
    """
    names = []
    for fc in parse_group_feature_columns(feature_columns).get(group_name, []):
        if fc.emb_name is None:
            continue
        if isinstance(fc.emb_name, str):
            emb_name = fc.emb_name
        elif isinstance(fc.emb_name, list):
            emb_name = group_name + "/" + fc.name
        else:
            continue
        if fc.type == "sparse":
            names.append("embedding/" + emb_name + "/embedding_lookup")
        elif fc.type == "varlen_sparse" and fc.combiner is not None:
            if "mean" in fc.combiner and "_sequence" not in group_name:
                names.append("sequence_mean_pooling_layer")
    return names


def get_target_item_embedding_by_prefix(group_embedding_feature_dict, prefix, feature_columns=None):
    """
    获取用于注意力激活的目标Item嵌入。

    该函数从特征嵌入中提取目标Item嵌入。
    目标Item是我们试图预测用户兴趣的Item。

    参数：
        group_embedding_feature_dict：分组嵌入特征字典
        prefix：嵌入名前缀，例如 'embedding/video_id'
        feature_columns：特征列列表（PyTorch 张量没有名字，用于还原原实现中按张量名匹配的逻辑）

    返回：
        torch.Tensor：目标Item嵌入
                   形状：[batch_size, embedding_dim]
    """
    dnn_embeddings = group_embedding_feature_dict.get("dnn", [])
    names = _dnn_group_embedding_names(feature_columns or [], "dnn")
    # 查找以 'embedding/video_id' 开头的嵌入
    target_embedding = None
    for name, embedding in zip(names, dnn_embeddings):
        if name.startswith(prefix):
            target_embedding = embedding
            break

    if len(target_embedding.shape) > 2:
        # 如果嵌入有额外维度，取平均值
        target_embedding = torch.mean(target_embedding, dim=1)

    return target_embedding
