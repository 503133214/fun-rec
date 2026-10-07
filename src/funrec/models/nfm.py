import torch
import torch.nn as nn

from .base import FunRecModel, Dense
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    add_tensor_func,
    get_linear_logits,
)
from .layers import DNNs, BiInteractionPooling


class NFMModel(FunRecModel):
    """神经因子分解机（NFM）排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        dnn_units = model_config.get("dnn_units", [64, 32])
        use_bn = model_config.get("use_bn", True)
        dropout_rate = model_config.get("dropout_rate", 0.1)
        linear_logits = model_config.get("linear_logits", True)

        # 输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="nfm")

        self.use_linear_logits = linear_logits

        # 分组嵌入 B x N x D（'linear' 组不在其中）
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 双交互池化（无参数，所有组共用一个实例；嵌入字典中的组名可能包含
        # parse_group_feature_columns 之外的组，例如 din 聚合产生的 'din_sequence'）
        self.bi_interaction_pooling = BiInteractionPooling(name="bi_interaction_pooling")

        # 深度神经网络
        self.dnn = DNNs(
            units=dnn_units, activation="relu", use_bn=use_bn, dropout_rate=dropout_rate
        )

        # NFM logits
        self.nfm_logits_dense = Dense(1, activation=None)  # B x 1

        if linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)

        self.nfm_output = Dense(1, activation="sigmoid", name="nfm_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        group_feature_dict = {}
        for group_name, _ in group_embedding_feature_dict.items():
            group_feature_dict[group_name] = concat_group_embedding(
                group_embedding_feature_dict, group_name, axis=1, flatten=False
            )  # B x N x D

        # 每个组的双交互池化
        bi_interaction_pooling_out = add_tensor_func(
            [
                self.bi_interaction_pooling(group_feature)
                for group_name, group_feature in group_feature_dict.items()
            ]
        )  # B x D

        # 深度神经网络
        dnn_out = self.dnn(bi_interaction_pooling_out)

        # NFM logits
        nfm_logits = self.nfm_logits_dense(dnn_out)  # B x 1

        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            nfm_logits = add_tensor_func(
                [nfm_logits, linear_logit], name="nfm_linear_logits"
            )

        # 遵循 FM 排序输出约定：扁平化和 sigmoid
        nfm_logits = torch.flatten(nfm_logits, start_dim=1)
        output = self.nfm_output(nfm_logits)
        output = torch.flatten(output, start_dim=1)
        return output


def build_nfm_model(feature_columns, model_config):
    """
    构建神经因子分解机（NFM）排序模型。

    Args:
        feature_columns: FeatureColumn 列表
        model_config: 包含参数的字典：
            - dnn_units: 列表，DNN 隐藏单元（默认 [64, 32]）
            - use_bn: 布尔值，是否使用批量归一化（默认 True）
            - dropout_rate: 浮点数，dropout 率（默认 0.1）
            - linear_logits: 布尔值，是否添加线性项（默认 True）

    Returns:
        (model, None, None): 排序模型元组
    """
    model = NFMModel(feature_columns, model_config)
    return model, None, None
