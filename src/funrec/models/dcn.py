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
from .layers import DNNs, DCN


class DCNModel(FunRecModel):
    """DCN (深度交叉网络) 排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        num_cross_layers = model_config.get("num_cross_layers", 3)
        dnn_units = model_config.get("dnn_units", [64, 32])
        dropout_rate = model_config.get("dropout_rate", 0.1)
        l2_reg = model_config.get("l2_reg", 1e-5)
        linear_logits = model_config.get("linear_logits", True)

        # 输入
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="dcn")

        self.use_linear_logits = linear_logits

        # 分组嵌入
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 为每个组构建交叉和深度组件（组名在构建期即可确定）
        group_names = list(parse_group_feature_columns(feature_columns).keys())
        self.cross_group_names = [g for g in group_names if g.startswith("dcn")]
        self.deep_group_names = [
            g for g in group_names if not g.startswith("dcn") and g.startswith("dnn")
        ]
        self.cross_layers = nn.ModuleDict(
            {
                g: DCN(num_cross_layers=num_cross_layers, l2_reg=l2_reg)
                for g in self.cross_group_names
            }
        )
        self.deep_layers = nn.ModuleDict(
            {
                g: DNNs(
                    units=dnn_units,
                    dropout_rate=dropout_rate,
                    activation="relu",
                    use_bn=False,
                )
                for g in self.deep_group_names
            }
        )

        # 如果需要，添加线性项
        if linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)
        self.dcn_dense = Dense(1, name="dcn_dense")
        self.dcn_output = Dense(1, activation="sigmoid", name="dcn_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        # 为每个组构建交叉和深度组件（按嵌入字典的组顺序遍历，与原实现一致）
        cross_outputs = []
        deep_outputs = []
        for group_feature_name in group_embedding_feature_dict.keys():
            if group_feature_name.startswith("dcn"):
                # 交叉组件: 连接嵌入作为 DCN 输入
                concat_feature = concat_group_embedding(
                    group_embedding_feature_dict, group_feature_name, axis=-1, flatten=True
                )  # B x (N*D)
                cross_out = self.cross_layers[group_feature_name](concat_feature)
                cross_outputs.append(cross_out)

            elif group_feature_name.startswith("dnn"):
                # DNN 组件: 展平嵌入作为 DNN 输入
                concat_feature = concat_group_embedding(
                    group_embedding_feature_dict, group_feature_name, axis=-1, flatten=True
                )  # B x (N*D)
                dnn_out = self.deep_layers[group_feature_name](concat_feature)
                deep_outputs.append(dnn_out)

        # 合并交叉输出
        if len(cross_outputs) > 1:
            cross_logit = add_tensor_func(cross_outputs, name="cross_logits")
        else:
            cross_logit = cross_outputs[0] if cross_outputs else None

        # 合并 DNN 输出
        if len(deep_outputs) > 1:
            deep_logit = add_tensor_func(deep_outputs, name="dnn_logits")
        else:
            deep_logit = deep_outputs[0] if deep_outputs else None

        # 合并所有输出
        dcn_outputs = []
        if cross_logit is not None:
            dcn_outputs.append(cross_logit)
        if deep_logit is not None:
            dcn_outputs.append(deep_logit)

        if len(dcn_outputs) > 1:
            dcn_logits = torch.cat(dcn_outputs, dim=-1)  # dcn_concat
        else:
            dcn_logits = dcn_outputs[0]

        # 如果需要，添加线性项
        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            dcn_logits = self.dcn_dense(dcn_logits)
            dcn_logits = add_tensor_func(
                [dcn_logits, linear_logit], name="dcn_linear_logits"
            )
        else:
            dcn_logits = self.dcn_dense(dcn_logits)

        # 遵循排序输出约定: 展平并 sigmoid
        dcn_logits = torch.flatten(dcn_logits, start_dim=1)
        output = self.dcn_output(dcn_logits)
        output = torch.flatten(output, start_dim=1)
        return output


def build_dcn_model(feature_columns, model_config):
    """
    构建 DCN (深度交叉网络) 排序模型。

    参数:
        feature_columns: FeatureColumn 列表
        model_config: 包含以下参数的字典:
            - num_cross_layers: int, 交叉层数量 (默认 3)
            - dnn_units: list, DNN 隐藏单元 (默认 [64, 32])
            - dropout_rate: float, dropout 率 (默认 0.1)
            - l2_reg: float, L2 正则化 (默认 1e-5)
            - linear_logits: bool, 是否添加线性项 (默认 True)

    返回:
        (model, None, None): 排序模型元组
    """
    model = DCNModel(feature_columns, model_config)
    return model, None, None
