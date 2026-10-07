import torch
import torch.nn as nn

from .base import FunRecModel, Dense, build_dummy_inputs, to_tensor
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    add_tensor_func,
    get_linear_logits,
    get_cross_logits,
)
from .layers import PredictLayer, DNNs


class WideDeepModel(FunRecModel):
    """Wide&Deep 排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        # 从配置中提取参数并设置默认值
        dnn_units = model_config.get("dnn_units", [64, 32])
        dnn_dropout_rate = model_config.get("dnn_dropout_rate", 0.1)
        # 构建输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="wide_deep")

        # 构建特征embedding表
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 确定分组顺序（与原实现遍历 group_embedding_feature_dict 的顺序一致）:
        # 用假输入跑一次嵌入查询得到实际产生的组名（全 0 索引，保证 vocab_size=1 的特征也不越界）
        with torch.no_grad():
            dummy = {
                k: torch.zeros_like(to_tensor(v, torch.device("cpu")))
                for k, v in build_dummy_inputs(feature_columns, batch_size=1).items()
            }
            self.group_names = list(self.embedding(dummy).keys())

        # 深度部分: 每个组一个 DNN + Dense(1)
        self.deep_dnns = nn.ModuleDict(
            {
                g: DNNs(units=dnn_units, activation="relu", dropout_rate=dnn_dropout_rate)
                for g in self.group_names
            }
        )
        self.deep_logit_layers = nn.ModuleDict(
            {g: Dense(1, activation=None) for g in self.group_names}
        )

        # 宽度部分
        self.linear_logits = get_linear_logits(feature_columns)
        self.cross_logits = get_cross_logits(feature_columns)

        self.wide_deep_output = Dense(1, activation="sigmoid", name="wide_deep_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        # 拼接组特征
        group_feature_dict = {}
        for group_name, _ in group_embedding_feature_dict.items():
            group_feature_dict[group_name] = concat_group_embedding(
                group_embedding_feature_dict, group_name, axis=1, flatten=True
            )  # B x (N * D)

        # 深度部分输出
        deep_logits = []
        for group_name, group_feature in group_feature_dict.items():
            deep_out = self.deep_dnns[group_name](group_feature)
            deep_logit = self.deep_logit_layers[group_name](deep_out)  # 保持为 (B, 1)
            deep_logits.append(deep_logit)

        # 宽度部分输出
        linear_logit = self.linear_logits(inputs)
        cross_logit = self.cross_logits(inputs)

        wide_deep_logits = add_tensor_func(deep_logits + [linear_logit, cross_logit])
        # 展平以确保输出为 (batch_size,) 用于二分类
        wide_deep_logits = torch.flatten(wide_deep_logits, start_dim=1)
        output = self.wide_deep_output(wide_deep_logits)
        output = torch.flatten(output, start_dim=1)  # 确保最终输出为 (batch_size,)
        return output


def build_wide_deep_model(feature_columns, model_config):
    """
    构建Wide&Deep模型

    参数:
    feature_columns: 特征列配置
    model_config: 模型配置字典，包含:
        - dnn_units: DNN层单元数 (默认: [64, 32])
        - dnn_dropout_rate: 丢弃概率 (默认: 0.1)
    """
    model = WideDeepModel(feature_columns, model_config)

    # 排序模型返回 (model, None, None)，因为没有单独的用户/物品模型
    return model, None, None
