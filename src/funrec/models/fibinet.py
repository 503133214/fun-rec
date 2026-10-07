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
from .layers import DNNs, SENetLayer, BilinearInteractionLayer


class FiBiNETModel(FunRecModel):
    """FiBiNET（特征重要性和双线性特征交互网络）排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        dnn_units = model_config.get("dnn_units", [64, 32])
        senet_reduction_ratio = model_config.get("senet_reduction_ratio", 3)
        bilinear_type = model_config.get("bilinear_type", "interaction")
        use_bn = model_config.get("use_bn", False)
        dropout_rate = model_config.get("dropout_rate", 0.0)
        linear_logits = model_config.get("linear_logits", True)

        # 输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="fibinet")

        self.use_linear_logits = linear_logits

        # 分组嵌入 B x N x D
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 每个 fibinet 组各自拥有 SENet 和两个双线性交互层（组名在构建期即可确定）
        group_names = list(parse_group_feature_columns(feature_columns).keys())
        self.fibinet_group_names = [g for g in group_names if g.startswith("fibinet")]
        self.senet_layers = nn.ModuleDict(
            {
                g: SENetLayer(reduction_ratio=senet_reduction_ratio)
                for g in self.fibinet_group_names
            }
        )
        # 原始特征的双线性交互
        self.bilinear_layers = nn.ModuleDict(
            {
                g: BilinearInteractionLayer(bilinear_type=bilinear_type)
                for g in self.fibinet_group_names
            }
        )
        # SENet 增强特征的双线性交互
        self.bilinear_senet_layers = nn.ModuleDict(
            {
                g: BilinearInteractionLayer(bilinear_type=bilinear_type)
                for g in self.fibinet_group_names
            }
        )

        # 深度神经网络
        self.dnn = DNNs(
            units=dnn_units, activation="relu", use_bn=use_bn, dropout_rate=dropout_rate
        )

        # FiBiNET logits
        self.fibinet_dense = Dense(1, activation=None)  # B x 1

        if linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)

        self.fibinet_output = Dense(1, activation="sigmoid", name="fibinet_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        interaction_outputs = []
        for group_feature_name in group_embedding_feature_dict.keys():
            if group_feature_name.startswith("fibinet"):
                # 获取特征嵌入 B x N x D
                group_feature = concat_group_embedding(
                    group_embedding_feature_dict,
                    group_feature_name,
                    axis=1,
                    flatten=False,
                )

                # SENet 特征增强
                senet_enhanced_features = self.senet_layers[group_feature_name](
                    group_feature
                )

                # 原始特征的双线性交互
                bilinear_interaction = self.bilinear_layers[group_feature_name](
                    group_feature
                )

                # SENet 增强特征的双线性交互
                bilinear_senet_interaction = self.bilinear_senet_layers[
                    group_feature_name
                ](senet_enhanced_features)

                # 展平交互输出
                bilinear_flat = torch.flatten(bilinear_interaction, start_dim=1)
                bilinear_senet_flat = torch.flatten(
                    bilinear_senet_interaction, start_dim=1
                )

                # 连接所有展平的特征
                interaction_outputs.extend([bilinear_flat, bilinear_senet_flat])

        if len(interaction_outputs) > 1:
            interaction_outputs = torch.cat(interaction_outputs, dim=-1)
        else:
            interaction_outputs = interaction_outputs[0]

        # 深度神经网络
        dnn_out = self.dnn(interaction_outputs)

        # FiBiNET logits
        fibinet_logits = self.fibinet_dense(dnn_out)  # B x 1

        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            fibinet_logits = add_tensor_func(
                [fibinet_logits, linear_logit], name="fibinet_linear_logits"
            )

        # 遵循排序输出约定：展平并应用 sigmoid
        fibinet_logits = torch.flatten(fibinet_logits, start_dim=1)
        output = self.fibinet_output(fibinet_logits)
        output = torch.flatten(output, start_dim=1)
        return output


def build_fibinet_model(feature_columns, model_config):
    """
    构建 FiBiNET（特征重要性和双线性特征交互网络）排序模型。

    Args:
        feature_columns: FeatureColumn 列表
        model_config: 包含参数的字典：
            - dnn_units: list，DNN 隐藏层单元数（默认 [64, 32]）
            - senet_reduction_ratio: int，SENet 压缩比例（默认 3）
            - bilinear_type: str，双线性交互类型（默认 "interaction"）
            - use_bn: bool，是否使用批归一化（默认 False）
            - dropout_rate: float，dropout 比例（默认 0.0）
            - linear_logits: bool，是否添加线性项（默认 True）

    Returns:
        (model, None, None): 排序模型元组
    """
    model = FiBiNETModel(feature_columns, model_config)
    return model, None, None
