import torch.nn as nn

from .base import FunRecModel, Dense
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    add_tensor_func,
    get_linear_logits,
    pairwise_feature_interactions,
)
from .layers import AttentionPoolingLayer


class AFMModel(FunRecModel):
    """注意力因子分解机（AFM）排序模型（PyTorch 实现）"""

    def __init__(self, feature_columns, model_config):
        attention_factor = model_config.get("attention_factor", 4)
        dropout_rate = model_config.get("dropout_rate", 0.1)
        l2_reg = model_config.get("l2_reg", 1e-4)
        linear_logits = model_config.get("linear_logits", True)

        # 输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="afm")

        self.dropout_rate = dropout_rate
        self.use_linear_logits = linear_logits

        # 分组嵌入 B x N x D
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 每个组: 成对交互的 dropout + 注意力池化层（与原实现一致，每组各自独立）
        group_names = []
        for fc in feature_columns:
            if fc.emb_name is None or fc.type not in ["sparse", "varlen_sparse"]:
                continue
            for g in fc.group:
                if g != "linear" and g not in group_names:
                    group_names.append(g)
        self.pairwise_dropout = nn.ModuleDict(
            {g: nn.Dropout(dropout_rate) for g in group_names}
        )
        self.attention_pooling = nn.ModuleDict(
            {
                g: AttentionPoolingLayer(attention_factor=attention_factor, l2_reg=l2_reg)
                for g in group_names
            }
        )

        # Logits
        self.afm_dense = Dense(1, activation=None)  # B x 1
        if linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)
        # 遵循FM排序输出约定：展平并应用sigmoid
        self.output_dense = Dense(1, activation="sigmoid", name="afm_output")

    def forward(self, inputs):
        # 分组嵌入 B x N x D
        group_embedding_feature_dict = self.embedding(inputs)

        group_feature_dict = {}
        for group_name, _ in group_embedding_feature_dict.items():
            group_feature_dict[group_name] = concat_group_embedding(
                group_embedding_feature_dict, group_name, axis=1, flatten=False
            )  # B x N x D

        # 对每个组的成对交互进行注意力池化
        group_attention_pooling_out = {}
        for group_name, group_feature in group_feature_dict.items():
            group_pairwise = pairwise_feature_interactions(
                group_feature,
                drop_rate=self.dropout_rate,
                dropout=self.pairwise_dropout[group_name],
            )  # B x num_pairs x D
            group_attention_pooling_out[group_name] = self.attention_pooling[group_name](
                group_pairwise
            )  # B x D

        # 跨组求和
        attention_pooling_output = add_tensor_func(
            [
                group_attention_pooling_out[group_name]
                for group_name in group_feature_dict.keys()
            ]
        )  # B x D

        # Logits
        afm_logits = self.afm_dense(attention_pooling_output)  # B x 1

        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)  # B x 1
            afm_logits = add_tensor_func(
                [afm_logits, linear_logit], name="afm_linear_logits"
            )

        # 遵循FM排序输出约定：展平并应用sigmoid
        afm_logits = afm_logits.reshape(afm_logits.shape[0], -1)
        output = self.output_dense(afm_logits)
        # 原实现 sigmoid 之后还有 Flatten，Keras 损失不会走 from_logits 分支，因此不标记 logits
        output = output.reshape(output.shape[0], -1)  # B x 1
        return output


def build_afm_model(feature_columns, model_config):
    """
    构建注意力因子分解机（AFM）排序模型。

    参数:
        feature_columns: FeatureColumn列表
        model_config: 包含参数的字典:
            - attention_factor: int, 注意力隐藏层大小 (默认 4)
            - dropout_rate: float, 成对交互的dropout (默认 0.1)
            - l2_reg: float, 注意力权重的L2正则化 (默认 1e-4)
            - linear_logits: bool, 是否添加线性项 (默认 True)

    返回:
        (model, None, None): 排序模型元组
    """
    model = AFMModel(feature_columns, model_config)
    return model, None, None
