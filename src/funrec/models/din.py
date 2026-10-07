import torch
import torch.nn as nn

from .base import FunRecModel, Dense
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    get_linear_logits,
    add_tensor_func,
    parse_din_feature_columns,
    concat_func,
)
from .layers import DNNs, DinAttentionLayer


class DINModel(FunRecModel):
    """DIN (深度兴趣网络) 排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        dnn_units = model_config.get("dnn_units", [128, 64, 1])
        use_linear_logits = model_config.get("linear_logits", True)

        # 输入和嵌入
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="din")

        self.use_linear_logits = use_linear_logits
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 对序列特征进行 DIN 注意力机制（每个序列特征一个注意力层）
        self.din_feature_list = parse_din_feature_columns(feature_columns)
        # 序列特征名 -> 所用嵌入表名（用于还原原实现中隐式传递的掩码）
        self.din_emb_names = {
            fc.name: fc.emb_name
            for fc in feature_columns
            if fc.type == "varlen_sparse" and fc.combiner is not None and "din" in fc.combiner
        }
        self.din_layers = nn.ModuleDict(
            {
                v_name: DinAttentionLayer(name=v_name + "_din_layer")
                for _, v_name in self.din_feature_list
            }
        )

        # DNN 塔
        self.dnn = DNNs(dnn_units, use_bn=True)
        if use_linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)
        self.din_output = Dense(1, activation="sigmoid", name="din_output")

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        # 来自 'dnn' 组的基础 DNN 输入
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "dnn")

        # 对序列特征进行 DIN 注意力机制
        # 注: 原 Keras 实现中掩码由嵌入表隐式传递: 只有当嵌入表由 varlen_sparse 特征首先创建（mask_zero=True）时
        # 注意力层才会收到掩码。默认配置中序列特征与 sparse 特征（如 video_id）共享嵌入表，因此 mask=None
        din_output_list = []
        for k_name, v_name in self.din_feature_list:
            query_feature = group_embedding_feature_dict["din_sequence"][k_name]
            key_feature = group_embedding_feature_dict["din_sequence"][v_name]
            table = self.embedding.embedding_table_dict[self.din_emb_names[v_name]]
            keys_mask = table.compute_mask(inputs[v_name])
            din_output = self.din_layers[v_name](
                [query_feature, key_feature], mask=keys_mask
            )
            din_output_list.append(din_output)
        din_output = concat_func(din_output_list, axis=1, flatten=True)
        dnn_inputs = concat_func([dnn_inputs, din_output], axis=-1)

        # DNN 塔
        dnn_logits = self.dnn(dnn_inputs)
        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            dnn_logits = add_tensor_func(
                [dnn_logits, linear_logit], name="din_linear_logits"
            )

        # 输出: 遵循排序约定以确保 (batch,) 标签兼容性
        final_logits = torch.flatten(dnn_logits, start_dim=1)
        output = self.din_output(final_logits)
        output = torch.flatten(output, start_dim=1)
        return output


def build_din_model(feature_columns, model_config):
    """
    构建 DIN (深度兴趣网络) 排序模型。

    参数:
        feature_columns: FeatureColumn 列表
        model_config: 包含以下参数的字典:
            - dnn_units: list, 隐藏层单元数包括输出大小 (默认 [128, 64, 1])
            - linear_logits: bool, 是否添加线性项 (默认 True)

    返回:
        (model, None, None): 排序模型元组
    """
    model = DINModel(feature_columns, model_config)
    return model, None, None
