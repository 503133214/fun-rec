import torch
import torch.nn as nn

from .base import FunRecModel
from .layers import (
    DNNs,
    PositionEncodingLayer,
    MetaUnit,
    MetaAttention,
    MetaTower,
    PredictLayer,
    TaskEmbedding,
    _MultiHeadAttention,
)
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    get_linear_logits,
    concat_func,
    add_tensor_func,
)


class M2MModel(FunRecModel):
    """M2M（多场景多任务元学习）排序模型的 PyTorch 实现（参数含义见 build_m2m_model）"""

    def __init__(self, feature_columns, model_config):
        task_name_list = model_config.get(
            "task_names", ["is_click", "long_view", "is_like"]
        )
        domain_group_name = model_config.get("domain_group_name", "domain")
        num_experts = model_config.get("num_experts", 4)
        view_dim = model_config.get("view_dim", 32)
        scenario_dim = model_config.get("scenario_dim", 16)
        meta_tower_depth = model_config.get("meta_tower_depth", 3)
        meta_unit_depth = model_config.get("meta_unit_depth", 3)
        meta_unit_shared = model_config.get("meta_unit_shared", True)
        activation = model_config.get("activation", "leaky_relu")
        dropout = model_config.get("dropout", 0.2)
        l2_reg = model_config.get("l2_reg", 1e-5)
        positon_agg_func = model_config.get("positon_agg_func", "concat")
        pos_emb_trainable = model_config.get("pos_emb_trainable", True)
        pos_initializer = model_config.get("pos_initializer", "glorot_uniform")
        sequence_pooling = model_config.get("sequence_pooling", "mean")
        use_linear_logits = model_config.get("linear_logits", True)

        # 构建输入层字典
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="m2m")
        self.feature_columns = list(feature_columns)
        self.task_name_list = list(task_name_list)
        self.domain_group_name = domain_group_name
        self.view_dim = view_dim
        self.positon_agg_func = positon_agg_func
        self.sequence_pooling = sequence_pooling
        self.use_linear_logits = use_linear_logits
        self._printed_mha_shape = False

        # 构建特征嵌入表字典
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 每个任务一个可学习的任务向量
        self.task_embedding_layers = nn.ModuleList(
            [
                TaskEmbedding(view_dim, name=f"task_emb_{i}")
                for i in range(len(task_name_list))
            ]
        )

        # 序列transformer（combiner 含 'mha' 的变长序列特征，每个特征一组位置编码 + 多头注意力）
        fc_dict = {fc.name: fc for fc in feature_columns}
        mha_feature_names = []
        for fc in feature_columns:
            if (
                fc.type == "varlen_sparse"
                and fc.combiner is not None
                and "mha" in fc.combiner
                and fc.emb_name is not None
                and any(g != "linear" for g in fc.group)
                and fc.name not in mha_feature_names
            ):
                mha_feature_names.append(fc.name)
        self.position_encoding_layers = nn.ModuleDict()
        self.mha_layers = nn.ModuleDict()
        for feat_name in mha_feature_names:
            fc = fc_dict[feat_name]
            self.position_encoding_layers[feat_name] = PositionEncodingLayer(
                dims=fc.emb_dim,
                max_len=fc.max_len,
                trainable=pos_emb_trainable,
                initializer=pos_initializer,
            )
            self.mha_layers[feat_name] = _MultiHeadAttention(
                num_heads=1,
                key_dim=16,
                value_dim=16,
                dropout=0.2,
            )

        # 场景知识表示
        self.scenario_mlp = DNNs(
            [scenario_dim], activation=activation, name="dnn/scenario_mlp"
        )

        # 专家视图表示
        self.expert_mlps = nn.ModuleList(
            [
                DNNs(
                    [view_dim],
                    activation=activation,
                    dropout_rate=dropout,
                    name=f"dnn/expert_{i}_mlp",
                )
                for i in range(num_experts)
            ]
        )

        attention_shared_meta_unit = None
        tower_shared_meta_unit = None
        if meta_unit_shared:
            attention_shared_meta_unit = MetaUnit(
                meta_unit_depth,
                activation,
                dropout,
                l2_reg,
                name="dnn/attention_shared_meta_unit",
            )
            tower_shared_meta_unit = MetaUnit(
                meta_unit_depth,
                activation,
                dropout,
                l2_reg,
                name="dnn/tower_shared_meta_unit",
            )

        self.linear_logits = None
        if use_linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)

        # 构建多任务输出
        self.task_mlps = nn.ModuleList()
        self.meta_attentions = nn.ModuleList()
        self.meta_towers = nn.ModuleList()
        self.tower_logit_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        for task_name in task_name_list:
            # 任务视图表示
            self.task_mlps.append(
                DNNs(
                    [view_dim],
                    activation=activation,
                    dropout_rate=dropout,
                    name=f"dnn/{task_name}_mlp",
                )
            )
            # 元注意力机制
            self.meta_attentions.append(
                MetaAttention(
                    meta_unit=attention_shared_meta_unit,
                    num_layer=meta_unit_depth,
                    activation=activation,
                    dropout=dropout,
                    l2_reg=l2_reg,
                    name=f"dnn/meta_attention_{task_name}",
                )
            )
            # 元塔网络
            self.meta_towers.append(
                MetaTower(
                    meta_unit=tower_shared_meta_unit,
                    num_layer=meta_tower_depth,
                    meta_unit_depth=meta_unit_depth,
                    activation=activation,
                    dropout=dropout,
                    l2_reg=l2_reg,
                    name=f"dnn/meta_tower_{task_name}",
                )
            )
            # 产生原始logits（无激活函数）
            self.tower_logit_layers.append(
                PredictLayer(as_logit=True, name=f"{task_name}/tower_logit")
            )
            # 任务预测输出
            self.output_layers.append(
                PredictLayer(name=f"task_{task_name}_output")
            )
        self.output_names = [f"task_{t}_output" for t in task_name_list]

    def _sequence_mask(self, feat_name, ids):
        """与原 Keras 实现一致: 仅当嵌入表 mask_zero=True（由 varlen_sparse 特征首先创建）时才有掩码"""
        fc = [x for x in self.feature_columns if x.name == feat_name][0]
        if isinstance(fc.emb_name, str) and fc.emb_name in self.embedding.embedding_table_dict:
            return self.embedding.embedding_table_dict[fc.emb_name].compute_mask(ids)
        return None

    def forward(self, inputs):
        # 构建特征嵌入
        group_embedding_feature_dict = self.embedding(inputs)

        # 用于创建常量/广播张量的批次参考张量
        first_input_tensor = inputs[self.input_names[0]]
        batch_size = first_input_tensor.shape[0]
        device = first_input_tensor.device

        task_embedding_list = [
            layer(first_input_tensor) for layer in self.task_embedding_layers
        ]

        # 用户嵌入：优先选择 'user' 组；回退到 'dnn'
        if "user" in group_embedding_feature_dict:
            user_embeddings = concat_group_embedding(group_embedding_feature_dict, "user")
        elif "dnn" in group_embedding_feature_dict:
            user_embeddings = concat_group_embedding(group_embedding_feature_dict, "dnn")
        else:
            # 如果都不存在，使用维度为view_dim的零向量
            user_embeddings = torch.zeros((batch_size, self.view_dim), device=device)

        expert_inputs = (
            concat_group_embedding(group_embedding_feature_dict, "dnn")
            if "dnn" in group_embedding_feature_dict
            else user_embeddings
        )
        transformer_input_dict = group_embedding_feature_dict.get("mha", {})

        # 领域嵌入：取配置组下的第一个嵌入
        if self.domain_group_name in group_embedding_feature_dict:
            domain_group = group_embedding_feature_dict[self.domain_group_name]
            if isinstance(domain_group, list) and len(domain_group) > 0:
                domain_input = domain_group[0]
            elif isinstance(domain_group, dict) and len(domain_group) > 0:
                domain_input = list(domain_group.values())[0]
            else:
                domain_input = None
        else:
            domain_input = None
        if (
            domain_input is not None
            and domain_input.dim() == 3
            and domain_input.shape[1] == 1
        ):
            domain_embeddings = torch.squeeze(domain_input, dim=1)
        elif domain_input is not None:
            domain_embeddings = domain_input
        else:
            domain_embeddings = torch.zeros((batch_size, self.view_dim), device=device)

        # 序列transformer
        mha_output_list = []
        for feat_name, transformer_input in transformer_input_dict.items():
            # 原实现中序列嵌入的掩码随张量隐式传递（Concatenate/Add 会保留掩码），
            # 多头注意力的 query/value/key 均使用该掩码
            mask = self._sequence_mask(feat_name, inputs[feat_name])
            position_embedding = self.position_encoding_layers[feat_name](
                transformer_input
            )
            if self.positon_agg_func == "sum":
                transformer_input = add_tensor_func(
                    [transformer_input, position_embedding]
                )
            elif self.positon_agg_func == "concat":
                transformer_input = torch.cat(
                    [transformer_input, position_embedding], dim=-1
                )
            transformer_output = self.mha_layers[feat_name](
                transformer_input,
                transformer_input,
                transformer_input,
                query_mask=mask,
                value_mask=mask,
                key_mask=mask,
            )
            if self.sequence_pooling == "mean":
                # 与原实现一致: 直接对所有位置求均值（不使用掩码）
                transformer_output = torch.mean(transformer_output, dim=1)
            mha_output_list.append(transformer_output)

        mha_output = None
        if len(mha_output_list) != 0:
            mha_output = concat_func(mha_output_list, axis=-1)
            expert_inputs = torch.cat([expert_inputs, mha_output], dim=-1)
            if not self._printed_mha_shape:
                print("mha shape", tuple(mha_output.shape))
                self._printed_mha_shape = True

        # 场景知识表示
        scenario_inputs = [domain_embeddings, user_embeddings]
        scenario_views = self.scenario_mlp(torch.cat(scenario_inputs, dim=-1))

        # 专家视图表示
        expert_views = [expert(expert_inputs) for expert in self.expert_mlps]
        expert_views = torch.stack(expert_views, dim=1)

        linear_logit = None
        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)

        # 构建多任务输出
        output_list = []
        for i, task_name in enumerate(self.task_name_list):
            # 任务视图表示
            task_embedding = task_embedding_list[i]
            task_views = self.task_mlps[i](task_embedding)

            # 元注意力机制
            attention_output = self.meta_attentions[i](
                [expert_views, task_views, scenario_views]
            )

            # 元塔网络
            tower_output = self.meta_towers[i]([attention_output, scenario_views])
            tower_output = torch.flatten(tower_output, start_dim=1)
            # 产生原始logits（无激活函数）
            tower_logit = self.tower_logit_layers[i](tower_output)

            if self.use_linear_logits:
                tower_logit = add_tensor_func([tower_logit, linear_logit])
            # 确保形状为 (batch, 1)
            tower_logit = torch.flatten(tower_logit, start_dim=1)

            # 任务预测输出
            prediction = self.output_layers[i](tower_logit)
            output_list.append(prediction)

        return output_list


def build_m2m_model(feature_columns, model_config):
    """构建M2M排序模型并遵循FunRec接口约定

    Args:
        feature_columns: 特征列配置
        model_config: 模型参数配置，包含：
            - task_names: 任务名称列表（默认 ["is_click", "long_view", "is_like"]）
            - domain_group_name: 领域/场景分组名称（默认 'domain'）
            - num_experts, view_dim, scenario_dim, meta_tower_depth, meta_unit_depth,
              meta_unit_shared, activation, dropout, l2_reg
            - positon_agg_func, pos_emb_trainable, pos_initializer, sequence_pooling
            - linear_logits: 是否加上线性项（默认 True）
    Returns:
        (model, None, None): 排序模型返回单一主模型
    """
    model = M2MModel(feature_columns, model_config)
    # FunRec排序模型约定：返回 (model, None, None)
    return model, None, None
