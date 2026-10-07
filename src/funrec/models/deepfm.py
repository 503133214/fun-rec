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
from .layers import FM, DNNs


class DeepFMModel(FunRecModel):
    """DeepFM (深度分解机) 排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        dnn_units = model_config.get("dnn_units", [64, 32])
        dropout_rate = model_config.get("dropout_rate", 0.1)
        linear_logits = model_config.get("linear_logits", True)

        # 输入
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="deepfm")

        self.use_linear_logits = linear_logits

        # 分组嵌入
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 为每个组构建 FM 和 DNN 组件（组名在构建期即可确定）
        group_names = list(parse_group_feature_columns(feature_columns).keys())
        self.deepfm_group_names = [g for g in group_names if g.startswith("deepfm")]
        self.fm_layers = nn.ModuleDict(
            {g: FM(name=f"fm_{g}") for g in self.deepfm_group_names}
        )
        self.dnn_layers = nn.ModuleDict(
            {
                g: DNNs(
                    name=f"dnn_{g}",
                    units=dnn_units + [1],
                    dropout_rate=dropout_rate,
                )
                for g in self.deepfm_group_names
            }
        )

        # 如果需要，添加线性项
        if linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)
        self.deepfm_output = Dense(1, activation="sigmoid", name="deepfm_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        # 为每个组构建 FM 和 DNN 组件（按嵌入字典的组顺序遍历，与原实现一致）
        fm_outputs = []
        dnn_outputs = []

        for group_feature_name in group_embedding_feature_dict.keys():
            if group_feature_name.startswith("deepfm"):
                # FM 组件: 期望 B x N x D 张量
                concat_feature = concat_group_embedding(
                    group_embedding_feature_dict, group_feature_name, axis=1, flatten=False
                )  # B x N x D
                fm_out = self.fm_layers[group_feature_name](concat_feature)
                fm_outputs.append(fm_out)

                # DNN 组件: 展平嵌入作为 DNN 输入
                flatten_feature = torch.flatten(concat_feature, start_dim=1)
                dnn_out = self.dnn_layers[group_feature_name](flatten_feature)
                dnn_outputs.append(dnn_out)

        # 合并 FM 输出
        if len(fm_outputs) > 1:
            fm_logit = add_tensor_func(fm_outputs, name="fm_logits")
        else:
            fm_logit = fm_outputs[0]

        # 合并 DNN 输出
        if len(dnn_outputs) > 1:
            dnn_logit = add_tensor_func(dnn_outputs, name="dnn_logits")
        else:
            dnn_logit = dnn_outputs[0]

        # 如果需要，添加线性项
        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            fm_logit = add_tensor_func([fm_logit, linear_logit], name="fm_linear_logits")

        # 合并 FM 和 DNN 输出
        deepfm_logits = add_tensor_func([fm_logit, dnn_logit], name="deepfm_logits")

        # 遵循排序输出约定: 展平并 sigmoid
        deepfm_logits = torch.flatten(deepfm_logits, start_dim=1)
        output = self.deepfm_output(deepfm_logits)
        output = torch.flatten(output, start_dim=1)
        return output


def build_deepfm_model(feature_columns, model_config):
    """
    构建 DeepFM (深度分解机) 排序模型。

    参数:
        feature_columns: FeatureColumn 列表
        model_config: 包含以下参数的字典:
            - dnn_units: list, DNN 隐藏层单元数 (默认 [64, 32])
            - dropout_rate: float, dropout 率 (默认 0.1)
            - linear_logits: bool, 是否添加线性项 (默认 True)

    返回:
        (model, None, None): 排序模型元组
    """
    model = DeepFMModel(feature_columns, model_config)
    return model, None, None
