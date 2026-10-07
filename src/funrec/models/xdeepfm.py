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
from .layers import DNNs, CINs


class XDeepFMModel(FunRecModel):
    """xDeepFM (极端深度因子分解机) 排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        dnn_units = model_config.get("dnn_units", [64, 32])
        dnn_dropout_rate = model_config.get("dnn_dropout_rate", 0.1)
        cin_layer_sizes = model_config.get("cin_layer_sizes", [32, 16])
        l2_reg = model_config.get("l2_reg", 1e-5)
        linear_logits = model_config.get("linear_logits", True)

        # 输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="xdeepfm")

        self.use_linear_logits = linear_logits

        # 分组嵌入
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 为每个组构建 DNN 和 CIN 组件（组名在构建期即可确定）
        group_names = list(parse_group_feature_columns(feature_columns).keys())
        self.xdeepfm_group_names = [g for g in group_names if g.startswith("xdeepfm")]
        self.dnn_layers = nn.ModuleDict(
            {
                g: DNNs(
                    units=dnn_units,
                    dropout_rate=dnn_dropout_rate,
                    activation="relu",
                    use_bn=False,
                )
                for g in self.xdeepfm_group_names
            }
        )
        self.dnn_logit_layers = nn.ModuleDict(
            {g: Dense(1, activation=None, name=f"dnn_{g}") for g in self.xdeepfm_group_names}
        )
        self.cin_layers = nn.ModuleDict(
            {g: CINs(cin_layer_sizes, l2_reg=l2_reg) for g in self.xdeepfm_group_names}
        )
        self.cin_logit_layers = nn.ModuleDict(
            {g: Dense(1, activation=None, name=f"cin_{g}") for g in self.xdeepfm_group_names}
        )

        # 如果需要，添加线性项
        if linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)
        self.xdeepfm_output = Dense(1, activation="sigmoid", name="xdeepfm_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        # 为每个组构建 DNN 和 CIN 组件（按嵌入字典的组顺序遍历，与原实现一致）
        dnn_logits = []
        cin_logits = []

        for group_feature_name in group_embedding_feature_dict.keys():
            if group_feature_name.startswith("xdeepfm"):
                # 为 xDeepFM 输入拼接嵌入: B x N x D
                concat_feature = concat_group_embedding(
                    group_embedding_feature_dict, group_feature_name, axis=1, flatten=False
                )  # B x N x D

                # DNN 组件: 为 DNN 输入展平嵌入
                flatten_feature = torch.flatten(concat_feature, start_dim=1)  # B x (N*D)
                dnn_out = self.dnn_layers[group_feature_name](flatten_feature)
                dnn_logit = self.dnn_logit_layers[group_feature_name](dnn_out)
                dnn_logits.append(dnn_logit)

                # CIN 组件: 使用 B x N x D 格式
                cin_out = self.cin_layers[group_feature_name](concat_feature)
                cin_logit = self.cin_logit_layers[group_feature_name](cin_out)
                cin_logits.append(cin_logit)

        # 合并 DNN 输出
        if len(dnn_logits) > 1:
            dnn_combined = add_tensor_func(dnn_logits, name="dnn_logits")
        else:
            dnn_combined = dnn_logits[0] if dnn_logits else None

        # 合并 CIN 输出
        if len(cin_logits) > 1:
            cin_combined = add_tensor_func(cin_logits, name="cin_logits")
        else:
            cin_combined = cin_logits[0] if cin_logits else None

        # 合并所有输出
        xdeepfm_outputs = []
        if dnn_combined is not None:
            xdeepfm_outputs.append(dnn_combined)
        if cin_combined is not None:
            xdeepfm_outputs.append(cin_combined)

        # 如果需要，添加线性项
        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            xdeepfm_outputs.append(linear_logit)

        if len(xdeepfm_outputs) > 1:
            xdeepfm_logits = add_tensor_func(xdeepfm_outputs, name="xdeepfm_logits")
        else:
            xdeepfm_logits = xdeepfm_outputs[0]

        # 遵循排序输出约定：展平并应用 sigmoid
        xdeepfm_logits = torch.flatten(xdeepfm_logits, start_dim=1)
        output = self.xdeepfm_output(xdeepfm_logits)
        output = torch.flatten(output, start_dim=1)
        return output


def build_xdeepfm_model(feature_columns, model_config):
    """
    构建 xDeepFM (极端深度因子分解机) 排序模型。

    Args:
        feature_columns: FeatureColumn 列表
        model_config: 包含参数的字典:
            - dnn_units: 列表，DNN 隐藏层单元数 (默认 [64, 32])
            - dnn_dropout_rate: 浮点数，DNN 的 dropout 率 (默认 0.1)
            - cin_layer_sizes: 列表，CIN 层大小 (默认 [32, 16])
            - l2_reg: 浮点数，L2 正则化 (默认 1e-5)
            - linear_logits: 布尔值，是否添加线性项 (默认 True)

    Returns:
        (model, None, None): 排序模型元组
    """
    model = XDeepFMModel(feature_columns, model_config)
    return model, None, None
