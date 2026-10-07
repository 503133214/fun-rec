import torch
import torch.nn as nn
import itertools

from .base import FunRecModel, Dense
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
)
from .layers import DNNs, PredictLayer


class CGCNet(nn.Module):
    """CGC结构中的所有参数（任务专家、共享专家、任务门控、共享门控）

    原实现中 cgc_net 在函数式建图时直接创建各层，PyTorch 中需要先在
    __init__ 中创建好子模块，再在 forward 中调用 cgc_net 完成计算。
    is_last: 主要是判断是否是最后一层CGC，如果是的话，就不需要创建共享门控
    """

    def __init__(
        self,
        task_num,
        task_expert_num,
        shared_expert_num,
        task_expert_dnn_units,
        shared_expert_dnn_units,
        task_gate_dnn_units,
        shared_gate_dnn_units,
        leval_name=None,
        is_last=False,
    ):
        super().__init__()
        self.task_num = task_num
        self.task_expert_num = task_expert_num
        self.shared_expert_num = shared_expert_num
        self.leval_name = leval_name
        self.is_last = is_last

        # 创建每个任务的任务专家
        self.task_experts = nn.ModuleList()
        for i in range(task_num):
            task_i_experts = nn.ModuleList()
            for j in range(task_expert_num):
                task_i_experts.append(
                    DNNs(
                        task_expert_dnn_units,
                        name=f"{leval_name}_task_{str(i)}_expert_{str(j)}",
                    )
                )
            self.task_experts.append(task_i_experts)

        # 创建所有任务的共享专家
        self.shared_experts = nn.ModuleList(
            [
                DNNs(shared_expert_dnn_units, name=f"{leval_name}_shared_expert_{str(i)}")
                for i in range(shared_expert_num)
            ]
        )

        # 创建每个任务的融合门控
        fusion_expert_num = task_expert_num + shared_expert_num
        self.task_gate_dnns = nn.ModuleList()
        self.task_gate_softmax = nn.ModuleList()
        for i in range(task_num):
            self.task_gate_dnns.append(
                DNNs(task_gate_dnn_units, name=f"{leval_name}_task_{str(i)}_gate")
            )
            self.task_gate_softmax.append(
                Dense(fusion_expert_num, use_bias=False, activation="softmax")
            )

        # 如果不是最后一层还需要共享门控，用于融合所有任务专家和共享专家作为共享专家下一层的输入
        if not is_last:
            cur_expert_num = task_num * task_expert_num + shared_expert_num
            self.shared_gate_dnn = DNNs(
                shared_gate_dnn_units, name=f"{leval_name}_shared_gate"
            )
            self.shared_gate_softmax = Dense(
                cur_expert_num, use_bias=False, activation="softmax"
            )
        else:
            self.shared_gate_dnn = None
            self.shared_gate_softmax = None

    def forward(self, input_list):
        return cgc_net(
            input_list,
            self.task_num,
            self.task_expert_num,
            self.shared_expert_num,
            None,
            None,
            None,
            None,
            leval_name=self.leval_name,
            is_last=self.is_last,
            cgc_layers=self,
        )


def cgc_net(
    input_list,
    task_num,
    task_expert_num,
    shared_expert_num,
    task_expert_dnn_units,
    shared_expert_dnn_units,
    task_gate_dnn_units,
    shared_gate_dnn_units,
    leval_name=None,
    is_last=False,
    cgc_layers=None,
):
    """CGC结构
    input_list: 每个任务都有一个输入，这些任务的输入都是共享的，为了方便处理，给每个任务都复制了一份
    is_last: 主要是判断是否是最后一层CGC，如果是的话，就不需要把共享部分添加到输出中了
    cgc_layers: CGCNet 模块，保存该层 CGC 的所有参数（PyTorch 中参数需预先创建；
        为 None 时按给定配置新建一个，仅适用于不需要训练/复用参数的场景）
    """
    if cgc_layers is None:
        cgc_layers = CGCNet(
            task_num,
            task_expert_num,
            shared_expert_num,
            task_expert_dnn_units,
            shared_expert_dnn_units,
            task_gate_dnn_units,
            shared_gate_dnn_units,
            leval_name=leval_name,
            is_last=is_last,
        )

    # 每个任务的任务专家
    task_expert_list = []
    for i in range(task_num):
        task_i_expert_list = []
        for j in range(task_expert_num):
            expert_dnn = cgc_layers.task_experts[i][j](input_list[i])
            task_i_expert_list.append(expert_dnn)
        task_expert_list.append(task_i_expert_list)

    # 所有任务的共享专家
    shared_expert_list = []
    for i in range(shared_expert_num):
        expert_dnn = cgc_layers.shared_experts[i](input_list[-1])
        shared_expert_list.append(expert_dnn)

    # 每个任务的融合门控
    task_gate_list = []
    for i in range(task_num):
        gate_dnn = cgc_layers.task_gate_dnns[i](input_list[i])
        gate_dnn = cgc_layers.task_gate_softmax[i](gate_dnn)
        gate_dnn = torch.unsqueeze(gate_dnn, dim=-1)  # (None, gate_num, 1)
        task_gate_list.append(gate_dnn)

    # CGC输出结果
    cgc_output_list = []
    for i in range(task_num):
        cur_experts = task_expert_list[i] + shared_expert_list
        expert_concat = torch.stack(cur_experts, dim=1)  # None, gate_num, dim
        cur_gate = task_gate_list[i]
        task_gate_fusion_dnn = torch.sum(
            cur_gate * expert_concat, dim=1, keepdim=False
        )  # None, dim
        cgc_output_list.append(task_gate_fusion_dnn)

    # 如果不是最后一层还需要更新共享专家下一层的输入，也就是当前层需要融合所有任务和共享专家
    if not is_last:
        cur_experts = (
            list(itertools.chain.from_iterable(task_expert_list)) + shared_expert_list
        )
        expert_concat = torch.stack(cur_experts, dim=1)  # None, cur_expert_num, dim
        shared_gate_dnn = cgc_layers.shared_gate_dnn(input_list[-1])
        shared_gate_dnn = cgc_layers.shared_gate_softmax(
            shared_gate_dnn
        )  # None, cur_expert_num
        shared_gate = torch.unsqueeze(shared_gate_dnn, dim=-1)  # None, cur_expert_num, 1
        shared_gate_fusion_output = torch.sum(
            shared_gate * expert_concat, dim=1, keepdim=False
        )
        cgc_output_list.append(shared_gate_fusion_output)
    return cgc_output_list


class PLEModel(FunRecModel):
    """PLE（渐进式分层提取）多任务排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        # 从model_config中提取参数
        task_names = model_config.get("task_names", ["is_click"])
        ple_level_nums = model_config.get("ple_level_nums", 1)
        task_expert_num = model_config.get("task_expert_num", 4)
        shared_expert_num = model_config.get("shared_expert_num", 2)
        task_expert_dnn_units = model_config.get("task_expert_dnn_units", [128, 64])
        shared_expert_dnn_units = model_config.get("shared_expert_dnn_units", [128, 64])
        task_gate_dnn_units = model_config.get("task_gate_dnn_units", [128, 64])
        shared_gate_dnn_units = model_config.get("shared_gate_dnn_units", [128, 64])
        task_tower_dnn_units = model_config.get("task_tower_dnn_units", [128, 64])
        dropout_rate = model_config.get("dropout_rate", 0.1)

        # 输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="ple")

        self.task_names = task_names
        self.task_num = len(task_names)
        self.ple_level_nums = ple_level_nums

        # 分组嵌入
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 多层CGC，只有最后一层不需要共享门控
        self.cgc_layers = nn.ModuleList()
        for i in range(ple_level_nums):
            self.cgc_layers.append(
                CGCNet(
                    self.task_num,
                    task_expert_num,
                    shared_expert_num,
                    task_expert_dnn_units,
                    shared_expert_dnn_units,
                    task_gate_dnn_units,
                    shared_gate_dnn_units,
                    leval_name=f"cgc_level_{str(i)}",
                    is_last=(i == ple_level_nums - 1),
                )
            )

        # 构建任务专用塔
        self.task_towers = nn.ModuleList()
        self.task_outputs = nn.ModuleList()
        for i in range(self.task_num):
            self.task_towers.append(
                DNNs(
                    name=f"task_tower_{task_names[i]}",
                    units=task_tower_dnn_units + [1],
                    dropout_rate=dropout_rate,
                )
            )
            self.task_outputs.append(PredictLayer(name=f"task_{task_names[i]}"))

    def forward(self, inputs):
        # 分组嵌入
        group_embedding_feature_dict = self.embedding(inputs)

        # 连接不同组的嵌入向量作为网络的输入
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "ple")

        ple_input_list = [dnn_inputs] * (self.task_num + 1)

        for i in range(self.ple_level_nums):
            cgc_output_list = self.cgc_layers[i](ple_input_list)
            if i != self.ple_level_nums - 1:
                ple_input_list = cgc_output_list

        # 任务专用塔
        task_output_list = []
        for i in range(self.task_num):
            task_output_logit = self.task_towers[i](cgc_output_list[i])
            task_output_prob = self.task_outputs[i](task_output_logit)
            task_output_list.append(task_output_prob)

        # 输出任务输出列表
        # 注: 与原实现一致，只有一个任务时模型输出单个张量（而非长度为 1 的列表）
        if len(task_output_list) == 1:
            return task_output_list[0]
        return task_output_list


def build_ple_model(feature_columns, model_config):
    """
    构建PLE（渐进式分层提取）多任务排序模型。

    参数:
        feature_columns: FeatureColumn列表
        model_config: 包含参数的字典:
            - task_names: 列表，任务名称（默认["is_click"]）
            - ple_level_nums: 整数，PLE层数（默认1）
            - task_expert_num: 整数，任务专用专家数量（默认4）
            - shared_expert_num: 整数，共享专家数量（默认2）
            - task_expert_dnn_units: 列表，任务专家DNN隐藏单元（默认[128, 64]）
            - shared_expert_dnn_units: 列表，共享专家DNN隐藏单元（默认[128, 64]）
            - task_gate_dnn_units: 列表，任务门控DNN隐藏单元（默认[128, 64]）
            - shared_gate_dnn_units: 列表，共享门控DNN隐藏单元（默认[128, 64]）
            - task_tower_dnn_units: 列表，任务塔DNN隐藏单元（默认[128, 64]）
            - dropout_rate: 浮点数，dropout率（默认0.1）

    返回:
        (model, None, None): 排序模型元组
    """
    model = PLEModel(feature_columns, model_config)
    return model, None, None
