"""
Model utilities for building neural networks (PyTorch).

与原 Keras 函数式实现的对应关系:
- build_input_layer: 返回 {特征名: FeatureColumn}，即模型的输入定义（对应 Keras Input）
- build_embedding_table_dict: 构建嵌入表字典（nn.ModuleDict）
- FeatureEmbedding: 嵌入模块，forward(inputs) 返回按组组织的嵌入字典
  （对应 build_group_feature_embedding_table_dict 的返回结果）
- LinearLogits / CrossLogits: 线性部分与二阶交叉部分的 logits 模块
  （对应 get_linear_logits / get_cross_logits）
- 其余函数为纯张量操作，直接在 forward 中调用
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .base import Dense, Embedding
from .layers import SequenceMeanPoolingLayer, BiasOnly


def build_input_layer(feature_columns, prefix=""):
    """构建输入定义

    Args:
        feature_columns (list): 特征列列表
        prefix (str): 前缀（保留参数，与原接口兼容）

    Returns:
        OrderedDict: 键为特征名称，值为对应的 FeatureColumn
    """
    input_layer_dict = OrderedDict()
    for fc in feature_columns:
        input_layer_dict[fc.name] = fc
    return input_layer_dict


def _new_embedding(fc, emb_name, prefix):
    return Embedding(
        input_dim=fc.vocab_size,
        output_dim=fc.emb_dim if "linear" not in prefix else 1,
        name=prefix + emb_name,
        embeddings_initializer=fc.initializer,
        l2_reg=fc.l2_reg,
        trainable=fc.trainable,
        mask_zero=fc.type == "varlen_sparse",
    )


def build_embedding_table_dict(feature_columns, prefix=""):
    """构建嵌入表字典

    Returns:
        nn.ModuleDict: 键为 emb_name，值为 Embedding
    """
    embedding_table_dict = nn.ModuleDict()
    for fc in feature_columns:
        if fc.type in ["sparse", "varlen_sparse"]:
            if isinstance(fc.emb_name, str) and fc.emb_name not in embedding_table_dict:
                embedding_table_dict[fc.emb_name] = _new_embedding(fc, fc.emb_name, prefix)
            elif isinstance(fc.emb_name, list):
                for emb_name in fc.emb_name:
                    if emb_name not in embedding_table_dict:
                        embedding_table_dict[emb_name] = _new_embedding(fc, emb_name, prefix)
    return embedding_table_dict


def parse_group_feature_columns(feature_columns):
    """解析特征列，按组分类

    Args:
        feature_columns (list): 特征列列表

    Returns:
        dict: 特征列按组分类的字典
    """
    group_feature_columns = {}
    for fc in feature_columns:
        for group in fc.group:
            if group == "linear":
                continue
            if group not in group_feature_columns:
                group_feature_columns[group] = []
            group_feature_columns[group].append(fc)
    return group_feature_columns


def get_sequence_mask(ids: torch.Tensor) -> torch.Tensor:
    """变长序列掩码（对应 Keras Embedding(mask_zero=True) 产生的掩码）: B x L 的 bool 张量"""
    return ids != 0


def build_group_feature_embedding_table_dict(
    feature_columns, inputs, embedding_table_dict, mean_pooling=None
):
    """按组查询嵌入

    Args:
        feature_columns (list): 特征列列表
        inputs (dict): {特征名: 张量} 输入字典
        embedding_table_dict (nn.ModuleDict): 由 build_embedding_table_dict 构建的嵌入表
        mean_pooling: 可选的 SequenceMeanPoolingLayer(keep_shape=True) 实例

    Returns:
        OrderedDict: 按组分类的嵌入特征字典。
            - 普通组: list[Tensor]，sparse 特征为 B x 1 x D，mean 聚合的变长特征为 B x 1 x D
            - combiner 为 None 的变长特征组 / din_sequence / dien_sequence / mha: dict[name, Tensor]
              （序列嵌入为 B x L x D。原 Keras 实现中只有当所用嵌入表 mask_zero=True
              （即该 emb_name 的嵌入表由 varlen_sparse 特征首先创建）时才会隐式携带掩码，
              对应掩码为 embedding_table.compute_mask(inputs[name])，否则为 None；
              与 sparse 特征共享嵌入表的序列特征在原实现中没有掩码）
    """
    if mean_pooling is None:
        mean_pooling = SequenceMeanPoolingLayer(keep_shape=True)
    group_feature_columns = parse_group_feature_columns(feature_columns)
    group_embedding_feature_dict = OrderedDict()
    for group_name, group_feature_column in group_feature_columns.items():
        for fc in group_feature_column:
            # 设置为emb_name设置为None的特征没有embedding
            if fc.emb_name is None:
                continue
            if isinstance(fc.emb_name, str):
                embedding_table = embedding_table_dict[fc.emb_name]
            elif isinstance(fc.emb_name, list):
                group_emb_name = group_name + "/" + fc.name
                embedding_table = embedding_table_dict[group_emb_name]
            else:
                continue
            if fc.type == "sparse":
                if group_name not in group_embedding_feature_dict:
                    group_embedding_feature_dict[group_name] = []
                group_embedding_feature_dict[group_name].append(
                    embedding_table(inputs[fc.name])
                )
            elif fc.type == "varlen_sparse":
                ids = inputs[fc.name]
                embed_list = embedding_table(ids)
                if fc.combiner is None:  # 不进行聚合，返回原始序列特征
                    if group_name not in group_embedding_feature_dict:
                        group_embedding_feature_dict[group_name] = {}
                    group_embedding_feature_dict[group_name][fc.name] = embed_list
                    continue

                # 仅对非序列组 (不含 _sequence) 应用 mean 聚合
                if (
                    ("mean" in fc.combiner)
                    and (group_name not in group_embedding_feature_dict)
                    and ("_sequence" not in group_name)
                ):
                    group_embedding_feature_dict[group_name] = []

                if ("mean" in fc.combiner) and ("_sequence" not in group_name):
                    # 与原 Keras 一致: 只有嵌入表本身 mask_zero=True 时才有掩码。
                    # 嵌入表由第一个使用该 emb_name 的特征创建，若与 sparse 特征共享
                    # （如 hist_movie_id 共享 movie_id 表），则没有掩码，池化会包含 padding 位置
                    pooling_emb = mean_pooling(
                        embed_list, mask=embedding_table.compute_mask(ids)
                    )  # Bx1xD
                    group_embedding_feature_dict[group_name].append(pooling_emb)
                if "din" in fc.combiner:
                    if "din_sequence" not in group_embedding_feature_dict:
                        group_embedding_feature_dict["din_sequence"] = {}
                    group_embedding_feature_dict["din_sequence"][fc.name] = embed_list
                    group_embedding_feature_dict["din_sequence"][fc.att_key_name] = (
                        embedding_table_dict[fc.emb_name](inputs[fc.att_key_name])
                    )
                if "mha" in fc.combiner:
                    if "mha" not in group_embedding_feature_dict:
                        group_embedding_feature_dict["mha"] = {}
                    group_embedding_feature_dict["mha"][fc.name] = embed_list
                if "dien" in fc.combiner:
                    if "dien_sequence" not in group_embedding_feature_dict:
                        group_embedding_feature_dict["dien_sequence"] = {}
                    group_embedding_feature_dict["dien_sequence"][fc.name] = embed_list
                    group_embedding_feature_dict["dien_sequence"][fc.att_key_name] = (
                        embedding_table_dict[fc.emb_name](inputs[fc.att_key_name])
                    )
    return group_embedding_feature_dict


class FeatureEmbedding(nn.Module):
    """特征嵌入模块

    在模型 __init__ 中创建，forward 时返回按组组织的嵌入字典:

        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")
        ...
        group_embedding_feature_dict = self.embedding(inputs)
        item_table = self.embedding.embedding_table_dict["movie_id"]
    """

    def __init__(self, feature_columns, prefix=""):
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.embedding_table_dict = build_embedding_table_dict(feature_columns, prefix=prefix)
        self.mean_pooling = SequenceMeanPoolingLayer(keep_shape=True)

    def forward(self, inputs):
        return build_group_feature_embedding_table_dict(
            self.feature_columns, inputs, self.embedding_table_dict, self.mean_pooling
        )


def concat_group_embedding(
    group_embedding_feature_dict, group_name, axis=-1, flatten=True
):
    group_embedding = group_embedding_feature_dict[group_name]
    if isinstance(group_embedding, dict):
        group_embedding_list = list(group_embedding.values())
    else:
        group_embedding_list = group_embedding
    if len(group_embedding_list) == 1:
        if flatten:
            return torch.flatten(group_embedding_list[0], start_dim=1)
        return group_embedding_list[0]
    else:
        concatenated = torch.cat(group_embedding_list, dim=axis)
        if flatten:
            return torch.flatten(concatenated, start_dim=1)
        return concatenated


def pooling_group_embedding(
    group_embedding_feature_dict, group_name, pooling_type="mean", keep_shape=False
):
    """对指定组的嵌入特征进行池化操作

    Args:
        group_embedding_feature_dict (dict): 按组分类的嵌入特征字典
        group_name (str): 需要池化的组名称
        pooling_type (str): 池化类型，默认为'mean'
        keep_shape (bool): 是否保持输出形状，默认为False

    Returns:
        torch.Tensor: 池化后的嵌入特征
    """
    group_embeddings = group_embedding_feature_dict[group_name]  # BxNxD, BxNxD => BxNxDx2
    if isinstance(group_embeddings, dict):
        group_embedding_list = list(group_embeddings.values())
    else:
        group_embedding_list = group_embeddings
    group_embedding = torch.stack(group_embedding_list, dim=-1)
    if pooling_type == "mean":
        pooled_embedding = torch.mean(group_embedding, dim=-1, keepdim=keep_shape)
    return pooled_embedding


def add_tensor_func(tensor_list, name=None):
    """Add tensors together (对应 Keras Add 层).

    Args:
        tensor_list (list): List of tensors to add
        name (str): 保留参数，与原接口兼容

    Returns:
        torch.Tensor: Sum of input tensors
    """
    ndims = [t.dim() for t in tensor_list]
    if len(set(ndims)) > 1 and min(ndims) >= 2:
        # 与 Keras Add(_Merge) 一致: 秩不同时在 axis=1 处扩展低秩张量直到秩相同
        # （如 B x 1 与 B x 1 x 1 相加得到 B x 1 x 1，而不是 torch 广播得到的 B x B x 1）
        max_ndim = max(ndims)
        tensor_list = [
            t.reshape(t.shape[:1] + (1,) * (max_ndim - t.dim()) + t.shape[1:])
            for t in tensor_list
        ]
    out = tensor_list[0]
    for t in tensor_list[1:]:
        out = out + t
    return out


class LinearLogits(nn.Module):
    """线性部分的 logits（对应原 get_linear_logits）

    对 group 中包含 'linear' 的 sparse/varlen_sparse 特征使用 1 维嵌入求和，
    对 dense 特征拼接后接一个 Dense 层。输出形状 B x 1。
    """

    def __init__(self, feature_columns, use_bias=False, name="linear_logits"):
        super().__init__()
        self.linear_embedding_feature_columns = [
            fc
            for fc in feature_columns
            if "linear" in fc.group and fc.type in ["sparse", "varlen_sparse"]
        ]
        self.linear_dense_feature_columns = [
            fc for fc in feature_columns if "linear" in fc.group and fc.type == "dense"
        ]
        self.linear_embedding_table_dict = build_embedding_table_dict(
            self.linear_embedding_feature_columns, prefix="linear/"
        )
        self.mean_pooling = SequenceMeanPoolingLayer()
        if self.linear_dense_feature_columns:
            dense_dim = sum(fc.dimension for fc in self.linear_dense_feature_columns)
            self.linear_dense = Dense(dense_dim, name="linear_dense")
        self.bias = BiasOnly(1) if use_bias else None

    def forward(self, inputs):
        linear_list = []
        for fc in self.linear_embedding_feature_columns:
            ids = inputs[fc.name]
            emb = self.linear_embedding_table_dict[fc.emb_name](ids)  # B x L x 1
            if fc.type == "sparse":
                linear_list.append(emb.reshape(emb.shape[0], -1).sum(dim=1, keepdim=True))
            elif fc.combiner == "mean":
                table = self.linear_embedding_table_dict[fc.emb_name]
                pooled = self.mean_pooling(emb, mask=table.compute_mask(ids))  # B x 1
                linear_list.append(pooled)
        if self.linear_dense_feature_columns:
            dense = torch.cat(
                [inputs[fc.name].float().reshape(inputs[fc.name].shape[0], -1) for fc in self.linear_dense_feature_columns],
                dim=1,
            )
            dense = self.linear_dense(dense)
            linear_list.append(dense.sum(dim=1, keepdim=True))

        if len(linear_list) == 0:
            batch_size = next(iter(inputs.values())).shape[0]
            device = next(iter(inputs.values())).device
            linear_logits = torch.zeros(batch_size, 1, device=device)
        else:
            linear_logits = add_tensor_func(linear_list)  # B x 1
        if self.bias is not None:
            linear_logits = self.bias(linear_logits)
        return linear_logits


def get_linear_logits(feature_columns, use_bias=False, name="linear_logits"):
    """构建线性 logits 模块（在 __init__ 中调用，forward 中以 module(inputs) 使用）"""
    return LinearLogits(feature_columns, use_bias=use_bias, name=name)


def pairwise_feature_interactions(group_feature, drop_rate=0.1, dropout=None, training=False):
    """
    Args:
        group_feature: B x N x D
        drop_rate: dropout rate
        dropout: 可选的 nn.Dropout 模块（推荐在模型中创建并传入）

    Returns:
        B x num_interactions x D
    """
    n = group_feature.shape[1]
    pairwise_interactions = []
    for i in range(n):
        for j in range(i + 1, n):
            # 特征i和特征j的元素级乘积
            pairwise_interactions.append(
                (group_feature[:, i, :] * group_feature[:, j, :]).unsqueeze(1)
            )  # B x 1 x D
    output = torch.cat(pairwise_interactions, dim=1)  # B x num_interactions x D
    if dropout is not None:
        output = dropout(output)
    else:
        output = torch.nn.functional.dropout(output, p=drop_rate, training=training)
    return output


def parse_din_feature_columns(feature_columns):
    din_feature_columns = []
    for fc in feature_columns:
        if fc.type == "varlen_sparse" and "din" in fc.combiner:
            kv_list = [fc.att_key_name, fc.name]
            din_feature_columns.append(kv_list)
    return din_feature_columns


def parse_dien_feature_columns(feature_columns):
    """Parse DIEN feature columns to extract (target, sequence) pairs.

    Args:
        feature_columns: List of feature column specifications

    Returns:
        dien_feature_list: List of (target_feature, sequence_feature) pairs
    """
    dien_feature_columns = []
    for fc in feature_columns:
        if fc.type == "varlen_sparse" and "dien" in fc.combiner:
            kv_list = [fc.att_key_name, fc.name]
            dien_feature_columns.append(kv_list)
    return dien_feature_columns


class CrossLogits(nn.Module):
    """2阶交叉特征的 logits（对应原 get_cross_logits），输出 B x 1

    对 group 包含 'cross' 的 sparse 特征两两组合，使用笛卡尔积索引查询标量权重。
    """

    def __init__(self, feature_columns, use_bias=False, name="cross_logits"):
        super().__init__()
        self.cross_feature_columns = [
            fc for fc in feature_columns if "cross" in fc.group and fc.type == "sparse"
        ]
        self.pairs = []
        self.cross_embeddings = nn.ModuleDict()
        for i in range(len(self.cross_feature_columns)):
            for j in range(i + 1, len(self.cross_feature_columns)):
                fc_i = self.cross_feature_columns[i]
                fc_j = self.cross_feature_columns[j]
                key = f"cross_{fc_i.name}_{fc_j.name}"
                # 创建交叉特征的1D embedding表 (标量权重)，初始化为0
                self.cross_embeddings[key] = Embedding(
                    input_dim=fc_i.vocab_size * fc_j.vocab_size,
                    output_dim=1,
                    embeddings_initializer="zeros",
                    l2_reg=max(fc_i.l2_reg, fc_j.l2_reg),
                    name=key,
                )
                self.pairs.append((key, fc_i, fc_j))
        self.bias = BiasOnly(1) if use_bias else None

    def forward(self, inputs):
        batch = next(iter(inputs.values()))
        cross_weights = []
        for key, fc_i, fc_j in self.pairs:
            feat_i = inputs[fc_i.name].long()  # [B, 1]
            feat_j = inputs[fc_j.name].long()  # [B, 1]
            # combined_index = feat_i * vocab_size_j + feat_j
            combined_index = feat_i * fc_j.vocab_size + feat_j
            cross_weight = self.cross_embeddings[key](combined_index).squeeze(-1)  # [B, 1]
            cross_weights.append(cross_weight)
        if len(cross_weights) == 0:
            return torch.zeros(batch.shape[0], 1, device=batch.device)
        cross_logits = add_tensor_func(cross_weights)
        if self.bias is not None:
            cross_logits = self.bias(cross_logits)
        return cross_logits


def get_cross_logits(feature_columns, use_bias=False, name="cross_logits"):
    """构建 2 阶交叉 logits 模块（在 __init__ 中调用，forward 中以 module(inputs) 使用）"""
    return CrossLogits(feature_columns, use_bias=use_bias, name=name)


def concat_func(x_list, axis, flatten=False):
    if len(x_list) == 1:
        return x_list[0] if not flatten else torch.flatten(x_list[0], start_dim=1)
    concat_output = torch.cat(x_list, dim=axis)
    if flatten:
        concat_output = torch.flatten(concat_output, start_dim=1)
    return concat_output
