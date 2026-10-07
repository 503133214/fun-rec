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
from .layers import DNNs, PNN


class PNNModel(FunRecModel):
    """基于乘积的神经网络 (PNN) 排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        dnn_units = model_config.get("dnn_units", [64, 32])
        product_layer_units = model_config.get("product_layer_units", 8)
        use_inner = model_config.get("use_inner", True)
        use_outer = model_config.get("use_outer", True)
        use_bn = model_config.get("use_bn", False)
        dropout_rate = model_config.get("dropout_rate", 0.0)
        linear_logits = model_config.get("linear_logits", True)

        # 输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="pnn")

        self.use_linear_logits = linear_logits

        # 分组嵌入 B x N x D
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 每个以 "pnn" 开头的组对应一个乘积层（组名在构建期即可确定）
        group_names = [
            g
            for g in parse_group_feature_columns(feature_columns).keys()
            if g.startswith("pnn")
        ]
        self.pnn_layers = nn.ModuleDict(
            {
                g: PNN(
                    units=product_layer_units, use_inner=use_inner, use_outer=use_outer
                )
                for g in group_names
            }
        )

        # 深度神经网络
        self.dnn = DNNs(
            units=dnn_units, activation="relu", use_bn=use_bn, dropout_rate=dropout_rate
        )

        # PNN logits
        self.pnn_logits_dense = Dense(1, activation=None)  # B x 1

        if linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)

        self.pnn_output = Dense(1, activation="sigmoid", name="pnn_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        interaction_outputs = []
        for (
            group_feature_name,
            group_feature_embedding,
        ) in group_embedding_feature_dict.items():
            if group_feature_name.startswith("pnn"):
                # PNN 期望嵌入张量列表
                pnn_out = self.pnn_layers[group_feature_name](
                    concat_group_embedding(
                        group_embedding_feature_dict,
                        group_feature_name,
                        axis=1,
                        flatten=False,
                    )
                )
                interaction_outputs.append(pnn_out)

        if len(interaction_outputs) > 1:
            interaction_outputs = torch.cat(interaction_outputs, dim=-1)
        else:
            interaction_outputs = interaction_outputs[0]

        # 深度神经网络
        dnn_out = self.dnn(interaction_outputs)

        # PNN logits
        pnn_logits = self.pnn_logits_dense(dnn_out)  # B x 1

        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            pnn_logits = add_tensor_func(
                [pnn_logits, linear_logit], name="pnn_linear_logits"
            )

        # 遵循排序输出约定：展平并应用 sigmoid
        pnn_logits = torch.flatten(pnn_logits, start_dim=1)
        output = self.pnn_output(pnn_logits)
        output = torch.flatten(output, start_dim=1)
        return output


def build_pnn_model(feature_columns, model_config):
    """
    构建基于乘积的神经网络 (PNN) 排序模型。

    Args:
        feature_columns: FeatureColumn 列表
        model_config: 包含参数的字典:
            - dnn_units: 列表，DNN 隐藏层单元数 (默认 [64, 32])
            - product_layer_units: 整数，乘积层输出单元数 (默认 8)
            - use_inner: 布尔值，是否使用内积 (默认 True)
            - use_outer: 布尔值，是否使用外积 (默认 True)
            - use_bn: 布尔值，是否使用批量归一化 (默认 False)
            - dropout_rate: 浮点数，dropout 率 (默认 0.0)
            - linear_logits: 布尔值，是否添加线性项 (默认 True)

    Returns:
        (model, None, None): 排序模型元组
    """
    model = PNNModel(feature_columns, model_config)
    return model, None, None
