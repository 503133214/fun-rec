import torch
import torch.nn as nn

from .base import FunRecModel, Dense
from .layers import DNNs
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    get_linear_logits,
    add_tensor_func,
)


class HMoEModel(FunRecModel):
    """分层专家混合（HMoE）排序模型的 PyTorch 实现（参数含义见 build_hmoe_model）"""

    def __init__(self, feature_columns, model_config):
        num_domains = model_config.get("num_domains", 5)
        domain_feature_name = model_config.get("domain_feature_name", "tab")
        share_gate = model_config.get("share_gate", False)
        share_domain_w = model_config.get("share_domain_w", False)
        shared_expert_nums = model_config.get("shared_expert_nums", 5)
        shared_expert_dnn_units = model_config.get("shared_expert_dnn_units", [256, 128])
        gate_dnn_units = model_config.get("gate_dnn_units", [256, 128])
        domain_tower_units = model_config.get("domain_tower_units", [128, 64])
        domain_weight_units = model_config.get("domain_weight_units", [128, 64])
        use_linear_logits = model_config.get("linear_logits", True)

        # 构建输入层字典
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="hmoe")
        self.num_domains = num_domains
        self.domain_feature_name = domain_feature_name
        self.share_gate = share_gate
        self.share_domain_w = share_domain_w
        self.use_linear_logits = use_linear_logits

        # 构建特征嵌入表字典
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 创建多个专家
        self.experts = nn.ModuleList(
            [
                DNNs(shared_expert_dnn_units, name=f"expert_{str(i)}")
                for i in range(shared_expert_nums)
            ]
        )

        if share_gate:
            # 共享Gate
            self.gate_index_list = [0] * num_domains
        else:
            # 注: 与原实现保持一致——原实现对每个域 i 都把第 i 个门控的输出追加 num_domains 次，
            # 而 domain tower 只取列表的前 num_domains 个元素，因此所有域塔实际都使用第 0 个门控的输出，
            # 其余门控不在输出路径上（原 Keras 模型中不包含这些参数）。这里只为实际被使用的门控创建参数。
            gate_index_list = []
            for i in range(num_domains):
                for _ in range(num_domains):
                    gate_index_list.append(i)
            self.gate_index_list = gate_index_list[:num_domains]
        used_gate_index = sorted(set(self.gate_index_list))
        self.gate_dnns = nn.ModuleDict()
        self.gate_softmax = nn.ModuleDict()
        for i in used_gate_index:
            gate_name = "shared_gates" if share_gate else f"domain_{str(i)}_gates"
            self.gate_dnns[str(i)] = DNNs(gate_dnn_units, name=gate_name)
            self.gate_softmax[str(i)] = Dense(
                shared_expert_nums,
                use_bias=False,
                activation="softmax",
                name=f"domain_{i}_softmax",
            )

        # 定义domain tower
        self.domain_towers = nn.ModuleList(
            [DNNs(domain_tower_units) for _ in range(num_domains)]
        )

        # 定义domain权重
        if share_domain_w:
            # 共享domain权重（与原实现一致，共享时不做softmax）
            self.domain_weight_dnns = nn.ModuleList([DNNs(domain_weight_units)])
        else:
            self.domain_weight_dnns = nn.ModuleList(
                [DNNs(domain_weight_units) for _ in range(num_domains)]
            )

        # 生成DNN分支logit
        self.dnn_logits = Dense(1, activation=None, name="dnn_logits")

        # 可选择性地添加线性项
        self.linear_logits = (
            get_linear_logits(feature_columns) if use_linear_logits else None
        )

    def forward(self, inputs):
        domain_input = inputs[self.domain_feature_name]
        # 确保域输入是整数类型，用于比较/掩码操作
        if torch.is_floating_point(domain_input):
            domain_input = torch.round(domain_input).long()
        else:
            domain_input = domain_input.long()

        group_embedding_feature_dict = self.embedding(inputs)

        # 连接不同组的嵌入向量作为各个网络的输入
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "dnn")

        # 多个专家
        expert_output_list = [expert(dnn_inputs) for expert in self.experts]
        expert_concat = torch.stack(expert_output_list, dim=1)  # (None, expert_num, dims)

        # 门控加权专家输出
        gate_expert_output_dict = {}
        for key in self.gate_dnns.keys():
            gate_output = self.gate_dnns[key](dnn_inputs)
            gate_output = self.gate_softmax[key](gate_output)
            gate_output = gate_output.unsqueeze(-1)  # (None,expert_num, 1)
            gate_expert_output = gate_output * expert_concat
            gate_expert_output = torch.sum(gate_expert_output, dim=1, keepdim=False)
            gate_expert_output_dict[int(key)] = gate_expert_output
        domain_tower_input_list = [
            gate_expert_output_dict[idx] for idx in self.gate_index_list
        ]

        # domain tower
        domain_tower_output_list = []
        for i in range(self.num_domains):
            domain_dnn_input = domain_tower_input_list[i]
            task_output = self.domain_towers[i](domain_dnn_input)
            domain_tower_output_list.append(task_output)

        # domain权重
        domain_weight_list = []
        if self.share_domain_w:
            domain_weight = self.domain_weight_dnns[0](dnn_inputs)
            for i in range(self.num_domains):
                domain_weight_list.append(domain_weight)
        else:
            for i in range(self.num_domains):
                domain_weight = self.domain_weight_dnns[i](dnn_inputs)
                domain_weight = torch.softmax(domain_weight, dim=1)
                domain_weight_list.append(domain_weight)

        # 融合domain信息
        domain_output_list = []
        for i in range(self.num_domains):
            domain_weight = domain_weight_list[i]
            domain_tower_output = domain_tower_output_list[i]
            weighted_output = domain_weight * domain_tower_output
            for j in range(self.num_domains):
                if i == j:
                    continue
                # 其他域塔的输出不回传梯度
                grad_output = domain_tower_output_list[j].detach()
                weighted_output = weighted_output + domain_weight_list[i][:, j : j + 1] * grad_output
            domain_mask = torch.squeeze(torch.eq(domain_input, i), dim=-1)
            # 注: 与原实现一致，按域筛选样本后再按域顺序拼接，输出样本顺序为按域分组后的顺序
            domain_output = weighted_output[domain_mask]
            domain_output_list.append(domain_output)
        # 将所有domain的数据拼接成batch
        final_domain_output = torch.cat(domain_output_list, dim=0)
        # 生成DNN分支logit
        dnn_logit = self.dnn_logits(final_domain_output)

        # 可选择性地添加线性项
        if self.linear_logits is not None:
            linear_logit = self.linear_logits(inputs)
            final_logit = add_tensor_func(
                [dnn_logit, linear_logit], name="hmoe_final_logit"
            )
        else:
            final_logit = dnn_logit

        # 二分类CTR的Sigmoid激活
        output = torch.sigmoid(final_logit)
        # 原实现在 sigmoid 后有 Flatten，因此交叉熵按概率计算（不做 logits 标记）
        output = torch.flatten(output, start_dim=1)
        return output


def build_hmoe_model(feature_columns, model_config):
    """
    构建分层专家混合（HMoE）排序模型（多域CTR风格）。

    参数:
        feature_columns: List[FeatureColumn]
        model_config: 包含以下配置的字典:
            - num_domains: int, 域的数量（默认: 5）
            - domain_feature_name: str, 域指示特征名称（默认: 'tab'）
            - share_gate: bool, 门控是否在域间共享（默认: False）
            - share_domain_w: bool, 域权重是否共享（默认: False）
            - shared_expert_nums: int, 共享专家数量（默认: 5）
            - shared_expert_dnn_units: List[int], 专家MLP单元（默认: [256, 128]）
            - gate_dnn_units: List[int], 门控MLP单元（默认: [256, 128]）
            - domain_tower_units: List[int], 域塔单元（默认: [128, 64]）
            - domain_weight_units: List[int], 域权重网络单元（默认: [128, 64]）
            - linear_logits: bool, 添加线性项（默认: True）
    返回:
        (model, None, None): FunRec流水线的排序模型元组
    """
    model = HMoEModel(feature_columns, model_config)
    # 排序模型返回 (model, None, None)
    return model, None, None
