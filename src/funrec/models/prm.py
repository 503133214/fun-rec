import torch
import torch.nn as nn

from .base import FunRecModel, Dense
from .layers import PositionEncodingLayer, TransformerEncoder
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    concat_func,
    add_tensor_func,
)


class PRMModel(FunRecModel):
    """PRM 重排序模型的 PyTorch 实现

    输出: 对每个列表内 max_seq_len 个物品的 softmax 分数，形状为 B x max_seq_len
    """

    def __init__(
        self,
        feature_columns,
        max_seq_len=30,
        transformer_blocks=2,
        nums_head=1,
        dropout_rate=0.1,
        intermediate_dim=64,
        pos_emb_trainable=True,
    ):
        # 输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="prm")
        self.max_seq_len = max_seq_len

        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")
        # item_part 组中的变长序列特征（原 Keras 中其嵌入带有 mask_zero 掩码，
        # 经 Concatenate / Add 传播后隐式传入第一个 TransformerEncoder）
        self.item_seq_feature_names = [
            (fc.name, fc.emb_name)
            for fc in feature_columns
            if "item_part" in fc.group
            and fc.type == "varlen_sparse"
            and isinstance(fc.emb_name, str)
        ]

        self.position_encoding = PositionEncodingLayer(
            dims=feature_columns[0].emb_dim,
            max_len=max_seq_len,
            trainable=pos_emb_trainable,
            initializer="glorot_uniform",
        )

        self.transformer_encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    intermediate_dim,
                    nums_head,
                    dropout_rate,
                    activation="relu",
                    normalize_first=True,
                    is_residual=True,
                )
                for _ in range(transformer_blocks)
            ]
        )

        self.output_dense = Dense(intermediate_dim, activation="tanh")
        self.score_dense = Dense(1)

    def _compute_page_mask(self, inputs):
        """对应 Keras 中 page_embedding 的隐式掩码 (B x max_len)，没有掩码时返回 None"""
        mask = None
        for name, emb_name in self.item_seq_feature_names:
            m = self.embedding.embedding_table_dict[emb_name].compute_mask(inputs[name])
            if m is None:
                continue
            mask = m if mask is None else (mask & m)
        return mask

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        user_part_embedding = concat_group_embedding(
            group_embedding_feature_dict, "user_part"
        )  # B x D
        # 将用户嵌入在序列长度维度上进行复制
        user_part_embedding = user_part_embedding.unsqueeze(1).repeat(
            1, self.max_seq_len, 1
        )  # [B, max_len, D]

        # 物品侧序列特征
        item_part_embedding = concat_group_embedding(
            group_embedding_feature_dict, "item_part", axis=-1, flatten=False
        )  # [B, max_len, K]
        # 稠密输入在原 Keras 中为 float32 Input，这里显式转换
        pv_embeddings = inputs["pv_emb"].float()  # [B, max_len, D_pv]
        item_embeddings = inputs["item_emb"].float()  # [B, max_len, D_item]

        page_embedding = concat_func(
            [user_part_embedding, item_part_embedding, pv_embeddings, item_embeddings],
            axis=-1,
        )  # [B, max_len, dim]

        position_embedding = self.position_encoding(page_embedding)

        enc_inputs = add_tensor_func([page_embedding, position_embedding])

        # 序列掩码: 原 Keras 实现中 item_part 序列嵌入的掩码（mask_zero=True）经 Concatenate
        # （各输入掩码取与）和 Add 传播到 enc_inputs，并通过 _keras_mask 隐式传给第一个
        # TransformerEncoder 内部的 MultiHeadAttention（作为 query/value 掩码）。
        # TransformerEncoder 的输出不再携带掩码，因此之后的 block 不使用掩码
        mask = self._compute_page_mask(inputs)
        for i, encoder in enumerate(self.transformer_encoders):
            enc_inputs = encoder(enc_inputs, mask=mask if i == 0 else None)

        enc_output = self.output_dense(enc_inputs)
        enc_output = self.score_dense(enc_output)
        flat = torch.flatten(enc_output, start_dim=1)
        score_output = torch.softmax(flat, dim=-1)
        # Keras 的 categorical_crossentropy 在 y_pred 直接由 softmax 算子产生时
        # 按 from_logits=True 计算（不裁剪），见 training/loss.py 中的 _keras_logits
        score_output._keras_logits = flat
        score_output._keras_logits_op = "Softmax"
        return score_output


def build_prm_model(feature_columns, model_config):
    """构建与FunRec训练流水线兼容的PRM重排序模型。

    Args:
        feature_columns: FeatureColumn列表
        model_config: 包含参数的字典:
            - max_seq_len: int (默认30)
            - transformer_blocks: int (默认2)
            - nums_head: int (默认1)
            - dropout_rate: float (默认0.1)
            - intermediate_dim: int (默认64)
            - pos_emb_trainable: bool (默认True)

    Returns:
        (model, None, None): 重排序模型元组
    """
    max_seq_len = model_config.get("max_seq_len", 30)
    transformer_blocks = model_config.get("transformer_blocks", 2)
    nums_head = model_config.get("nums_head", 1)
    dropout_rate = model_config.get("dropout_rate", 0.1)
    intermediate_dim = model_config.get("intermediate_dim", 64)
    pos_emb_trainable = model_config.get("pos_emb_trainable", True)

    model = PRMModel(
        feature_columns,
        max_seq_len=max_seq_len,
        transformer_blocks=transformer_blocks,
        nums_head=nums_head,
        dropout_rate=dropout_rate,
        intermediate_dim=intermediate_dim,
        pos_emb_trainable=pos_emb_trainable,
    )
    return model, None, None
