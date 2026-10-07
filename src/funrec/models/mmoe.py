import torch
import torch.nn as nn

from .base import FunRecModel, Dense
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
)
from .layers import DNNs, PredictLayer


class MMoEModel(FunRecModel):
    """多门控专家混合（MMoE）多任务排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        task_names = model_config.get("task_names", ["is_click"])
        expert_nums = model_config.get("expert_nums", 4)
        expert_dnn_units = model_config.get("expert_dnn_units", [128, 64])
        gate_dnn_units = model_config.get("gate_dnn_units", [128, 64])
        task_tower_dnn_units = model_config.get("task_tower_dnn_units", [128, 64])
        dropout_rate = model_config.get("dropout_rate", 0.1)

        # 输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="mmoe")

        self.task_names = task_names
        self.expert_nums = expert_nums

        # 分组嵌入
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 创建多个专家
        self.experts = nn.ModuleList(
            [
                DNNs(name=f"expert_{i}", units=expert_dnn_units, dropout_rate=dropout_rate)
                for i in range(expert_nums)
            ]
        )

        # 定义任务特定的门控网络
        self.gate_dnns = nn.ModuleList()
        self.gate_softmax = nn.ModuleList()
        for i, task_name in enumerate(task_names):
            self.gate_dnns.append(
                DNNs(name=f"task_{i}_gates", units=gate_dnn_units, dropout_rate=dropout_rate)
            )
            self.gate_softmax.append(
                Dense(
                    expert_nums,
                    use_bias=False,
                    activation="softmax",
                    name=f"task_{i}_softmax",
                )
            )

        # 任务塔与预测层
        self.task_towers = nn.ModuleList()
        self.task_outputs = nn.ModuleList()
        for i, task_name in enumerate(task_names):
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
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "mmoe")

        # 创建多个专家
        expert_output_list = [expert(dnn_inputs) for expert in self.experts]
        expert_concat = torch.stack(expert_output_list, dim=1)  # (None, expert_num, dims)

        # 定义任务特定的门控网络
        task_tower_input_list = []
        for i in range(len(self.task_names)):
            gate_output = self.gate_dnns[i](dnn_inputs)
            gate_output = self.gate_softmax[i](gate_output)
            gate_output = torch.unsqueeze(gate_output, dim=-1)  # (None,expert_num, 1)
            gate_expert_output = gate_output * expert_concat
            gate_expert_output = torch.sum(gate_expert_output, dim=1, keepdim=False)
            task_tower_input_list.append(gate_expert_output)

        # 不同任务通过门控融合多个专家
        task_output_list = []
        for i in range(len(self.task_names)):
            task_output_logit = self.task_towers[i](task_tower_input_list[i])
            task_output_prob = self.task_outputs[i](task_output_logit)
            task_output_list.append(task_output_prob)

        # 输出任务输出列表
        # 注: 与原 Keras 实现一致，只有一个任务时模型输出单个张量（而非长度为 1 的列表）
        if len(task_output_list) == 1:
            return task_output_list[0]
        return task_output_list


def build_mmoe_model(feature_columns, model_config):
    """
    构建多门控专家混合（MMoE）多任务排序模型。

    Args:
        feature_columns: FeatureColumn 列表
        model_config: 包含参数的字典：
            - task_names: 列表，任务名称（默认 ["is_click"]）
            - expert_nums: 整数，专家数量（默认 4）
            - expert_dnn_units: 列表，专家 DNN 隐藏单元（默认 [128, 64]）
            - gate_dnn_units: 列表，门控 DNN 隐藏单元（默认 [128, 64]）
            - task_tower_dnn_units: 列表，任务塔 DNN 隐藏单元（默认 [128, 64]）
            - dropout_rate: 浮点数，dropout 率（默认 0.1）

    Returns:
        (model, None, None): 排序模型元组
    """
    model = MMoEModel(feature_columns, model_config)
    return model, None, None
