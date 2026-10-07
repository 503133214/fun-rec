import torch
import torch.nn as nn

from .base import FunRecModel, Dense
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    parse_group_feature_columns,
    concat_group_embedding,
    add_tensor_func,
    get_linear_logits,
)
from .layers import MultiHeadAttentionLayer


class AutoIntModel(FunRecModel):
    """AutoInt 排序模型（PyTorch 实现）"""

    def __init__(
        self,
        feature_columns,
        num_interaction_layers=2,
        attention_factor=8,
        num_heads=2,
        use_residual=True,
        linear_logits=True,
    ):
        # 输入
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="autoint")

        # 分组嵌入
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 为每个以 "autoint" 开头的组构建 AutoInt 组件
        # （前向时按嵌入字典中的组顺序计算；未产生嵌入的组其惰性层不会被构建，也不含参数）
        self.autoint_group_names = [
            g for g in parse_group_feature_columns(feature_columns) if g.startswith("autoint")
        ]
        self.attention_layers = nn.ModuleDict()
        self.group_dense = nn.ModuleDict()
        for group_feature_name in self.autoint_group_names:
            # 应用多个自注意力层
            self.attention_layers[group_feature_name] = nn.ModuleList(
                [
                    MultiHeadAttentionLayer(
                        attention_dim=attention_factor,
                        num_heads=num_heads,
                        use_residual=use_residual,
                    )
                    for _ in range(num_interaction_layers)
                ]
            )
            # 该组的全连接层
            self.group_dense[group_feature_name] = Dense(
                1, name=f"autoint_dense_{group_feature_name}"
            )

        # 如果需要，添加线性项
        self.use_linear_logits = linear_logits
        if linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)

        # 输出层
        self.output_dense = Dense(1, activation="sigmoid", name="autoint_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        autoint_outputs = []
        for group_feature_name in group_embedding_feature_dict.keys():
            if group_feature_name not in self.attention_layers:
                continue
            # 获取 B x N x D 格式的组特征
            group_feature = concat_group_embedding(
                group_embedding_feature_dict, group_feature_name, axis=1, flatten=False
            )  # B x N x D

            # 应用多个自注意力层
            attention_output = group_feature
            for attention_layer in self.attention_layers[group_feature_name]:
                attention_output = attention_layer(attention_output)

            # 展平注意力输出: B x N x (D * H) -> B x (N * D * H)
            flattened_attention = torch.flatten(attention_output, start_dim=1)

            # 该组的全连接层
            group_output = self.group_dense[group_feature_name](flattened_attention)
            autoint_outputs.append(group_output)

        # 合并 AutoInt 输出
        if len(autoint_outputs) > 1:
            autoint_logits = add_tensor_func(autoint_outputs, name="autoint_logits")
        elif len(autoint_outputs) == 1:
            autoint_logits = autoint_outputs[0]
        else:
            # 如果没有 AutoInt 组，创建零张量
            first = next(iter(inputs.values()))
            autoint_logits = torch.zeros(first.shape[0], 1, device=first.device)

        # 如果需要，添加线性项
        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            final_logits = add_tensor_func(
                [autoint_logits, linear_logit], name="autoint_linear_logits"
            )
        else:
            final_logits = autoint_logits

        # 遵循排序输出约定: 展平并 sigmoid
        final_logits = torch.flatten(final_logits, start_dim=1)
        output = self.output_dense(final_logits)
        # 原实现在 sigmoid 之后还有一次 Flatten，因此不标记为 logits 输出（BCE 按概率并裁剪计算）
        output = torch.flatten(output, start_dim=1)
        return output


def build_autoint_model(feature_columns, model_config):
    """
    构建 AutoInt (自动特征交互学习) 排序模型。

    参数:
        feature_columns: FeatureColumn 列表
        model_config: 包含以下参数的字典:
            - num_interaction_layers: int, 注意力层数量 (默认 2)
            - attention_factor: int, 注意力维度 (默认 8)
            - num_heads: int, 注意力头数量 (默认 2)
            - use_residual: bool, 是否使用残差连接 (默认 True)
            - linear_logits: bool, 是否添加线性项 (默认 True)

    返回:
        (model, None, None): 排序模型元组
    """
    num_interaction_layers = model_config.get("num_interaction_layers", 2)
    attention_factor = model_config.get("attention_factor", 8)
    num_heads = model_config.get("num_heads", 2)
    use_residual = model_config.get("use_residual", True)
    linear_logits = model_config.get("linear_logits", True)

    model = AutoIntModel(
        feature_columns,
        num_interaction_layers=num_interaction_layers,
        attention_factor=attention_factor,
        num_heads=num_heads,
        use_residual=use_residual,
        linear_logits=linear_logits,
    )
    return model, None, None
