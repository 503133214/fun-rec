# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from .base import FunRecModel, Dense
from .layers import APGLayer, PredictLayer
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    get_linear_logits,
    add_tensor_func,
)


def _infer_group_concat_dim(embedding, feature_columns, group_name):
    """用全零的伪输入（batch=1）前向一次嵌入层，推断某组拼接后的维度。

    原 Keras 实现在构图时直接读取 dnn_inputs.shape[-1]；APGLayer 需要在构造时知道 input_dim，
    因此这里在 __init__ 中做一次无梯度的伪前向（只查嵌入表，不创建任何参数）。
    """
    device = next(embedding.parameters()).device
    dummy_inputs = {}
    for fc in feature_columns:
        if fc.type in ["sparse", "varlen_sparse"]:
            dummy_inputs[fc.name] = torch.zeros(1, fc.max_len, dtype=torch.long, device=device)
        else:
            dummy_inputs[fc.name] = torch.zeros(1, fc.dimension, device=device)
    with torch.no_grad():
        group_embedding_feature_dict = embedding(dummy_inputs)
    return concat_group_embedding(group_embedding_feature_dict, group_name).shape[-1]


class APGModel(FunRecModel):
    """APG排序模型（参数含义见 build_apg_model）"""

    def __init__(
        self,
        feature_columns,
        task_name_list,
        scene_group_name,
        apg_dnn_units,
        scene_emb_dim,
        activation,
        dropout,
        use_uv_shared,
        use_mf_p,
        mf_k,
        mf_p,
        linear_logits,
    ):
        # 构建输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="apg")
        self.task_name_list = list(task_name_list)
        self.output_names = list(task_name_list)
        self.scene_group_name = scene_group_name
        self.use_linear_logits = linear_logits

        # 按组构建嵌入表
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 构建APG堆叠层
        input_dim = _infer_group_concat_dim(self.embedding, feature_columns, "dnn")
        self.apg_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i, units in enumerate(apg_dnn_units):
            self.apg_layers.append(
                APGLayer(
                    input_dim=input_dim,
                    output_dim=units,
                    scene_emb_dim=scene_emb_dim,
                    activation=activation,
                    use_uv_shared=use_uv_shared,
                    use_mf_p=use_mf_p,
                    mf_k=mf_k,
                    mf_p=mf_p,
                    name=f"apg_layer_{i}",
                )
            )
            # 每个APG层后的dropout（dropout为0时用恒等映射占位）
            self.dropouts.append(
                nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
            )
            input_dim = units

        # 可选的线性logits
        self.linear_logits = get_linear_logits(feature_columns) if linear_logits else None

        # 任务特定输出
        self.task_logit_layers = nn.ModuleList(
            [Dense(1, use_bias=False, name=f"task_{task_name}_logit") for task_name in self.task_name_list]
        )
        self.predict_layers = nn.ModuleList(
            [PredictLayer(name=task_name) for task_name in self.task_name_list]
        )

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        # 合并DNN嵌入并提取场景嵌入
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "dnn")
        # 场景组应包含至少一个形状为[B, 1, D]的稀疏嵌入张量
        if self.scene_group_name not in group_embedding_feature_dict:
            raise ValueError(f"在特征组中未找到scene_group_name '{self.scene_group_name}'")
        scene_group_list = group_embedding_feature_dict[self.scene_group_name]
        if isinstance(scene_group_list, dict):
            # 选择第一个张量
            scene_tensor = next(iter(scene_group_list.values()))
        else:
            scene_tensor = scene_group_list[0]
        scene_emb = torch.squeeze(scene_tensor, dim=1)

        # APG堆叠层
        x = dnn_inputs
        for apg_layer, dropout in zip(self.apg_layers, self.dropouts):
            x = apg_layer([x, scene_emb])
            x = dropout(x)

        # 可选的线性logits
        linear_logits_tensor = None
        if self.use_linear_logits:
            # 原 Keras get_linear_logits 输出为 B x 1 x 1，与 B x 1 的任务logit经 Add 后为 B x 1 x 1，
            # 这里保持相同形状
            linear_logits_tensor = self.linear_logits(inputs).unsqueeze(-1)

        # 任务特定输出
        task_outputs = []
        for logit_layer, predict_layer in zip(self.task_logit_layers, self.predict_layers):
            task_logit = logit_layer(x)
            if self.use_linear_logits and linear_logits_tensor is not None:
                task_logit = add_tensor_func([task_logit, linear_logits_tensor])
            output = predict_layer(task_logit)
            task_outputs.append(output)

        # 与 Keras 一致: 单输出模型 predict 返回单个数组
        if len(task_outputs) == 1:
            return task_outputs[0]
        return task_outputs


def build_apg_model(feature_columns, model_config):
    """按照FunRec约定构建APG排序模型。

    参数:
        feature_columns: FeatureColumn列表
        model_config: 字典，支持以下键：
            - task_names: 任务名称列表 (默认 ["is_click"])
            - scene_group_name: 提供场景嵌入的场景组名称 (例如 'domain')
            - apg_dnn_units: APG层隐藏单元列表 (默认 [256, 128])
            - scene_emb_dim: 场景嵌入维度 (未使用；从嵌入中推导)
            - activation: APG内部/输出激活函数 (默认 'relu')
            - dropout: 每个APG层后的dropout (默认 0.2)
            - l2_reg: l2正则化 (保留)
            - use_uv_shared: bool，使用UV共享权重 (默认 True)
            - use_mf_p: bool，启用P路径因子分解 (默认 True)
            - mf_k: int，K路径因子分割因子 (默认 4)
            - mf_p: int，P路径因子分割因子 (默认 4)
            - linear_logits: bool，添加线性项 (默认 False)

    返回:
        (model, None, None)
    """
    task_name_list = model_config.get("task_names", ["is_click"])
    scene_group_name = model_config.get("scene_group_name", "domain")
    apg_dnn_units = model_config.get("apg_dnn_units", [256, 128])
    scene_emb_dim = model_config.get("scene_emb_dim", 8)
    activation = model_config.get("activation", "relu")
    dropout = model_config.get("dropout", 0.2)
    l2_reg = model_config.get("l2_reg", 1e-5)  # 保留（原实现未使用）
    use_uv_shared = model_config.get("use_uv_shared", True)
    use_mf_p = model_config.get("use_mf_p", True)
    mf_k = model_config.get("mf_k", 4)
    mf_p = model_config.get("mf_p", 4)
    linear_logits = model_config.get("linear_logits", False)

    model = APGModel(
        feature_columns,
        task_name_list=task_name_list,
        scene_group_name=scene_group_name,
        apg_dnn_units=apg_dnn_units,
        scene_emb_dim=scene_emb_dim,
        activation=activation,
        dropout=dropout,
        use_uv_shared=use_uv_shared,
        use_mf_p=use_mf_p,
        mf_k=mf_k,
        mf_p=mf_p,
        linear_logits=linear_logits,
    )

    # FunRec排序模型约定
    return model, None, None
