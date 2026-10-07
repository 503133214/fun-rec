import torch.nn as nn

from .base import FunRecModel
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
)
from .layers import DNNs, PredictLayer


class SharedBottomModel(FunRecModel):
    """Shared-Bottom 多任务排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        task_names = model_config.get("task_names", ["is_click"])
        share_dnn_units = model_config.get("share_dnn_units", [128, 64])
        task_tower_dnn_units = model_config.get("task_tower_dnn_units", [128, 64])
        dropout_rate = model_config.get("dropout_rate", 0.1)

        # 输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="shared_bottom")

        self.task_names = task_names

        # 分组嵌入
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 构建共享底层DNN层
        self.shared_bottom = DNNs(
            name="shared_bottom", units=share_dnn_units, dropout_rate=dropout_rate
        )

        # 构建任务特定塔
        self.task_towers = nn.ModuleList()
        self.task_outputs = nn.ModuleList()
        for task_name in task_names:
            self.task_towers.append(
                DNNs(
                    name=f"task_tower_{task_name}",
                    units=task_tower_dnn_units + [1],
                    dropout_rate=dropout_rate,
                )
            )
            self.task_outputs.append(PredictLayer(name=f"task_{task_name}"))

    def forward(self, inputs):
        # 分组嵌入
        group_embedding_feature_dict = self.embedding(inputs)

        # 连接不同组的嵌入向量作为网络的输入
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "shared_bottom")

        # 共享底层DNN
        shared_bottom_feature = self.shared_bottom(dnn_inputs)

        # 任务特定塔
        task_output_list = []
        for i in range(len(self.task_names)):
            task_output_logit = self.task_towers[i](shared_bottom_feature)
            task_output_prob = self.task_outputs[i](task_output_logit)
            task_output_list.append(task_output_prob)

        # 输出任务输出列表
        # 注: 与原 Keras 实现一致，只有一个任务时模型输出单个张量（而非长度为 1 的列表）
        if len(task_output_list) == 1:
            return task_output_list[0]
        return task_output_list


def build_shared_bottom_model(feature_columns, model_config):
    """
    构建Shared-Bottom多任务排序模型。

    Args:
        feature_columns: FeatureColumn列表
        model_config: 包含参数的字典:
            - task_names: 列表，任务名称 (默认 ["is_click"])
            - share_dnn_units: 列表，共享底层DNN隐藏单元 (默认 [128, 64])
            - task_tower_dnn_units: 列表，任务塔DNN隐藏单元 (默认 [128, 64])
            - dropout_rate: 浮点数，dropout率 (默认 0.1)

    Returns:
        (model, None, None): 排序模型元组
    """
    model = SharedBottomModel(feature_columns, model_config)
    return model, None, None
