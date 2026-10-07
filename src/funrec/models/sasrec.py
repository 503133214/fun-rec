import torch
import torch.nn as nn

from .base import FunRecModel, SubModel
from .utils import build_input_layer, build_embedding_table_dict, add_tensor_func
from .layers import PositionEncodingLayer, NegativeSampleEmbedding, DNNs, _MultiHeadAttention


class SASRecModel(FunRecModel):
    """SASRec 主模型

    forward 输出: 负采样二分类损失（标量），与原模型的 main_loss 一致
    子塔:
        user_tower: 输入 seq_ids，输出序列最后一个位置的 embedding (B, emb_dim)
        all_item_tower: 输入 all_item_input（物品 id 一维数组），输出 (N, emb_dim)
        sampling_item_tower: 输入 neg_sample_ids，输出 (B, 采样数, emb_dim)
    """

    def __init__(self, feature_columns, model_config):
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="sasrec")

        # 从配置中提取参数，设置默认值
        self.max_seq_len = model_config.get("max_seq_len", 200)
        mha_num = model_config.get("mha_num", 2)
        nums_heads = model_config.get("nums_heads", 1)
        dropout = model_config.get("dropout", 0.2)
        activation = model_config.get("activation", "relu")
        pos_emb_trainable = model_config.get("pos_emb_trainable", True)
        pos_initializer = model_config.get("pos_initializer", "glorot_uniform")
        emb_dim = feature_columns[0].emb_dim
        self.emb_dim = emb_dim

        self.embedding_table_dict = build_embedding_table_dict(
            feature_columns, prefix="sasrec/"
        )
        item_dim = self.embedding_table_dict["item_id"].output_dim

        self.position_encoding = PositionEncodingLayer(
            dims=emb_dim,
            max_len=self.max_seq_len,
            trainable=pos_emb_trainable,
            initializer=pos_initializer,
        )

        # 多头注意力 block: LayerNorm -> MultiHeadAttention(因果) -> 残差 -> LayerNorm -> DNNs
        # Keras LayerNormalization 默认 epsilon=1e-3
        self.attention_norms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.residual_norms = nn.ModuleList()
        self.dnns = nn.ModuleList()
        for i in range(mha_num):
            self.attention_norms.append(nn.LayerNorm(item_dim, eps=1e-3))
            self.attention_layers.append(
                _MultiHeadAttention(
                    num_heads=nums_heads,
                    key_dim=emb_dim,
                    dropout=dropout,
                    name=f"{i}_block",
                )
            )
            self.residual_norms.append(nn.LayerNorm(item_dim, eps=1e-3))
            self.dnns.append(
                DNNs(
                    units=[emb_dim, emb_dim],
                    dropout_rate=dropout,
                    activation=activation,
                    name=f"{i}_dnn",
                )
            )
        self.final_norm = nn.LayerNorm(emb_dim, eps=1e-3)

        # 负采样算loss
        self.negative_sampler = NegativeSampleEmbedding(
            vocab_size=feature_columns[0].vocab_size,
            num_sampled=self.max_seq_len,
            sampled_type="uniform",
        )

        # 推理时用户输入和item输入
        user_input_names = [k for k in input_layer_dict.keys() if k in ["seq_ids"]]
        sampling_item_input_names = [
            k for k in input_layer_dict.keys() if k in ["neg_sample_ids"]
        ]
        self.user_tower = SubModel(self, "encode_user", user_input_names, name="user_model")
        # 全量物品评估使用专用的物品输入（all_item_input，一维物品 id 数组）
        self.all_item_tower = SubModel(
            self, "encode_all_item", ["all_item_input"], name="item_model"
        )
        # 评估时需要使用的embedding（采样评估）
        self.sampling_item_tower = SubModel(
            self, "encode_sampling_item", sampling_item_input_names
        )

    def encode_sequence(self, inputs):
        """序列编码，返回 (B, max_len, emb_dim)"""
        seq_ids = inputs["seq_ids"]
        item_table = self.embedding_table_dict["item_id"]
        sequence_embedding = item_table(seq_ids)
        # 原实现中 item_id 嵌入表由 varlen_sparse 特征创建（mask_zero=True），序列掩码随张量隐式传递:
        # Embedding -> Add -> LayerNormalization -> MultiHeadAttention（作为 query/value 掩码）。
        # 运行时 DNNs 内部的 Dense/Activation/Dropout 均会透传掩码，因此每个注意力 block
        # 都同时使用序列掩码（query/value）和因果掩码，这里显式传入。
        seq_mask = item_table.compute_mask(seq_ids)

        position_embedding = self.position_encoding(sequence_embedding)
        # 原始序列emb加上position embedding
        sequence_embedding = add_tensor_func([sequence_embedding, position_embedding])

        seq_len = sequence_embedding.shape[1]
        # use_causal_mask=True: 下三角因果掩码 [1, T, S]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=sequence_embedding.device)
        ).unsqueeze(0)

        # 多头注意力
        for i in range(len(self.attention_layers)):
            sequence_embedding_norm = self.attention_norms[i](sequence_embedding)
            sequence_embedding_output = self.attention_layers[i](
                sequence_embedding_norm,
                sequence_embedding,
                attention_mask=causal_mask,
                query_mask=seq_mask,
                value_mask=seq_mask,
            )
            # 残差连接
            sequence_embedding = add_tensor_func(
                [sequence_embedding, sequence_embedding_output]
            )
            sequence_embedding = self.residual_norms[i](sequence_embedding)
            # FFN
            sequence_embedding = self.dnns[i](sequence_embedding)
        sequence_embedding = self.final_norm(sequence_embedding)
        return sequence_embedding

    def encode_user(self, inputs):
        # 序列的padding在左边，直接拿到序列的最后一个结果即可
        return self.encode_sequence(inputs)[:, -1, :]  # B, emb_dim

    def encode_all_item(self, inputs):
        # 对于全量物品评估，使用单一的物品 id 输入映射到物品嵌入
        ids = inputs["all_item_input"] if isinstance(inputs, dict) else inputs
        return self.embedding_table_dict["item_id"](ids.reshape(-1))  # N, emb_dim

    def encode_sampling_item(self, inputs):
        # 评估时需要使用的embedding（采样评估）
        return self.embedding_table_dict["item_id"](inputs["neg_sample_ids"])

    def forward(self, inputs):
        sequence_embedding = self.encode_sequence(inputs)
        positive_embedding = self.embedding_table_dict["item_id"](inputs["pos_ids"])

        # 获取所有item的索引和权重
        item_embedding_weight = self.embedding_table_dict["item_id"].embeddings

        # 负采样算loss
        negative_embeddings = self.negative_sampler(
            positive_embedding, item_embedding_weight
        )

        # 序列展开成[batch_size, max_len, emb_dim]
        sequence_embedding = sequence_embedding.reshape(
            sequence_embedding.shape[0], -1, self.emb_dim
        )
        pos_item_embedding = positive_embedding.reshape(
            positive_embedding.shape[0], -1, self.emb_dim
        )
        negative_embeddings = negative_embeddings.reshape(
            negative_embeddings.shape[0], -1, self.emb_dim
        )

        pos_logits = torch.sum(pos_item_embedding * sequence_embedding, dim=-1)  # B, max_len
        # 与原实现一致: neg (B, L, D) 与 expand_dims(seq, 1) (B, 1, L, D) 广播相乘后求和，
        # 结果形状为 (B, B, L)，neg_logits[i, j, l] = <neg[j, l], seq[i, l]>
        # 使用 einsum 计算以避免构造 (B, B, L, D) 的中间张量
        neg_logits = torch.einsum(
            "jld,ild->ijl", negative_embeddings, sequence_embedding
        )  # B, B, max_len

        # 创建目标掩码，忽略padding项目（ID=0），维度: [batch_size, maxlen]
        # 这里的掩码是为了计算loss时忽略padding项
        is_target = (inputs["pos_ids"] != 0).float().reshape(-1, self.max_seq_len)

        main_loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * is_target
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * is_target
        ) / torch.sum(is_target)
        return main_loss


def build_sasrec_model(feature_columns, model_config):
    """
    构建SASRec模型 (Self-Attentive Sequential Recommendation)

    参数:
    feature_columns: 特征列配置
    model_config: 模型配置字典，包含:
        - max_seq_len: 最大序列长度 (default: 200)
        - mha_num: 多头注意力层数 (default: 2)
        - nums_heads: 注意力头数 (default: 1)
        - dropout: dropout率 (default: 0.2)
        - activation: 激活函数 (default: 'relu')
        - pos_emb_trainable: 位置编码是否可训练 (default: True)
        - pos_initializer: 位置编码初始化器 (default: 'glorot_uniform')
    """
    model = SASRecModel(feature_columns, model_config)

    # 为评估创建独立的用户和物品模型（与主模型共享参数）
    user_model = model.user_tower
    item_model = model.all_item_tower

    return model, user_model, item_model
