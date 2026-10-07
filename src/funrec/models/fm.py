"""
因子分解机（FM）排序模型。
"""

import torch
import torch.nn as nn

from .base import FunRecModel, Dense
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    add_tensor_func,
    get_linear_logits,
    parse_group_feature_columns,
)
from .layers import FM


class FMModel(FunRecModel):
    """因子分解机(FM)排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        # 从配置中提取参数，设置默认值
        linear_logits = model_config.get("linear_logits", True)

        # 构建输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="fm")

        self.use_linear_logits = linear_logits

        # 构建特征embedding表
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 每个特征组一个FM层（FM层无参数，组名在构建期即可确定）
        group_names = list(parse_group_feature_columns(feature_columns).keys())
        self.fm_layers = nn.ModuleDict({g: FM(name=g) for g in group_names})

        # 如果需要线性部分，则构建线性逻辑
        if linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)

        self.fm_output = Dense(1, activation="sigmoid", name="fm_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        # 存储FM特征组的字典
        fm_group_feature_dict = {}
        for fm_group_name in group_embedding_feature_dict.keys():
            # 连接同一组内的嵌入特征，形成形状为BxNxD的张量（批次大小x特征数量x嵌入维度）
            concat_fm_feature = concat_group_embedding(
                group_embedding_feature_dict, fm_group_name, axis=1, flatten=False
            )  # BxNxD
            fm_group_feature_dict[fm_group_name] = concat_fm_feature

        # 对每个特征组应用FM层，并将结果相加
        fm_logit = add_tensor_func(
            [
                self.fm_layers[fm_group_name](fm_input)
                for fm_group_name, fm_input in fm_group_feature_dict.items()
            ]
        )

        # 如果需要线性部分，则添加线性逻辑并与FM逻辑相结合
        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            fm_logit = add_tensor_func([fm_logit, linear_logit], name="fm_linear_logits")

        # 展平以确保输出为 (batch_size,) 用于二分类
        fm_logit = torch.flatten(fm_logit, start_dim=1)
        output = self.fm_output(fm_logit)
        output = torch.flatten(output, start_dim=1)  # 确保最终输出为 (batch_size,)
        return output


def build_fm_model(feature_columns, model_config):
    """
    构建因子分解机(FM)模型 - 用于排序

    参数:
    feature_columns: 特征列配置
    model_config: 模型配置字典，包含:
        - linear_logits: 是否添加线性部分，默认为True
    """
    model = FMModel(feature_columns, model_config)

    # 排序模型返回 (model, None, None)，因为没有单独的用户/物品模型
    return model, None, None
