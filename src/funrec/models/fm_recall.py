"""
因子分解机（FM）推荐模型。
"""

import torch
import torch.nn as nn

from .base import FunRecModel, SubModel, Dense
from .utils import (
    build_input_layer,
    build_group_feature_embedding_table_dict,
    FeatureEmbedding,
)


class SumPooling(nn.Module):
    """对嵌入特征进行求和聚合的层"""

    def __init__(self, name=None, **kwargs):
        super(SumPooling, self).__init__()
        self.layer_name = name

    def forward(self, inputs):
        # inputs shape: [batch_size, num_features, embedding_dim]
        return torch.sum(inputs, dim=1)  # [batch_size, embedding_dim]


class OnesLayer(nn.Module):
    """生成全1向量的层"""

    def __init__(self, name=None, **kwargs):
        super(OnesLayer, self).__init__()
        self.layer_name = name

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        return torch.ones((batch_size, 1), dtype=torch.float32, device=inputs.device)


class SquareLayer(nn.Module):
    """平方操作层"""

    def __init__(self, name=None, **kwargs):
        super(SquareLayer, self).__init__()
        self.layer_name = name

    def forward(self, inputs):
        return torch.square(inputs)


class SumScalarLayer(nn.Module):
    """将向量求和为标量的层"""

    def __init__(self, name=None, **kwargs):
        super(SumScalarLayer, self).__init__()
        self.layer_name = name

    def forward(self, inputs):
        return torch.sum(inputs, dim=1, keepdim=True)


class ScaleLayer(nn.Module):
    """按常数缩放的层"""

    def __init__(self, scale_factor, name=None, **kwargs):
        super(ScaleLayer, self).__init__()
        self.layer_name = name
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return inputs * self.scale_factor


class FMRecallModel(FunRecModel):
    """FM 双塔召回模型的 PyTorch 实现

    - 主模型输出: FM 匹配分数经 Dense(1, sigmoid) 后的概率 (B x 1)
    - user_tower / item_tower: 与主模型共享参数的用户塔、物品塔子模型，
      分别输出 [1; ∑(v_u * x_u)] 与 [first_term; ∑(v_t * x_t)]，形状均为 B x (1 + embedding_dim)
    """

    def __init__(self, feature_columns, model_config):
        # 从配置中提取参数，设置默认值（原实现中 embedding_dim 仅被读取，嵌入维度由特征列决定）
        embedding_dim = model_config.get("embedding_dim", 8)

        # 构建输入层
        input_layer_dict = build_input_layer(feature_columns)

        # 分离用户和物品的输入
        user_feature_columns = [fc for fc in feature_columns if "user" in fc.group]
        item_feature_columns = [fc for fc in feature_columns if "item" in fc.group]
        user_input_names = [input_layer_dict[fc.name].name for fc in user_feature_columns]
        item_input_names = [input_layer_dict[fc.name].name for fc in item_feature_columns]

        # 训练模型的输入: user_inputs + item_inputs
        super().__init__(
            input_names=user_input_names + item_input_names, name="fm_two_tower_training"
        )
        self.embedding_dim = embedding_dim
        self.feature_columns = list(feature_columns)
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns

        # 构建嵌入特征
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 用户塔各层
        self.user_embedding_sum = SumPooling(name="user_embedding_sum")
        self.ones_vector = OnesLayer(name="ones_vector")

        # 物品塔各层
        self.item_embedding_sum = SumPooling(name="item_embedding_sum")
        # 一阶线性项：为每个物品特征学习一个权重
        self.item_linear_weights = Dense(
            1, activation="linear", use_bias=False, name="item_linear_weights"
        )
        self.item_sum_squared = SquareLayer(name="item_sum_squared")
        self.item_squared = SquareLayer(name="item_squared")
        self.item_squared_sum = SumPooling(name="item_squared_sum")
        self.fm_half_scale = ScaleLayer(0.5, name="fm_half_scale")
        self.fm_interaction_scalar = SumScalarLayer(name="fm_interaction_scalar")

        # 输出层（sigmoid激活用于二分类）；此处不带激活，sigmoid 在 forward 中显式计算以便标记 logits
        self.output_dense = Dense(1, activation=None, name="output")

        # === 构建独立的用户和物品模型（与主模型共享参数） ===
        self.user_tower = SubModel(self, "encode_user", user_input_names, name="user_tower_model")
        self.item_tower = SubModel(self, "encode_item", item_input_names, name="item_tower_model")

    def _group_embeddings(self, feature_columns, inputs, group_name):
        # 仅查询该塔所需特征的 embedding（与主模型共享 embedding 表）
        group_embedding_feature_dict = build_group_feature_embedding_table_dict(
            feature_columns,
            inputs,
            self.embedding.embedding_table_dict,
            self.embedding.mean_pooling,
        )
        return group_embedding_feature_dict.get(group_name, [])

    # === 用户塔：V_user = [1; ∑(v_u * x_u)] ===
    def _user_tower(self, user_embeddings):
        if not user_embeddings:
            raise ValueError("No user embeddings found")

        # 计算用户嵌入向量的和：∑(v_u * x_u)
        # 注意：这里的x_u对于one-hot编码的类别特征来说就是1
        user_concat = torch.cat(
            user_embeddings, dim=1
        )  # [batch_size, num_user_features, embedding_dim]
        user_embedding_sum = self.user_embedding_sum(
            user_concat
        )  # [batch_size, embedding_dim]

        # 构建用户向量：[1; ∑(v_u * x_u)]
        ones_vector = self.ones_vector(user_embedding_sum)  # [batch_size, 1]

        # 拼接：[1; ∑(v_u * x_u)]
        user_vector = torch.cat([ones_vector, user_embedding_sum], dim=1)

        return user_vector

    # === 物品塔：V_item = [first_term; ∑(v_t * x_t)] ===
    def _item_tower(self, item_embeddings):
        if not item_embeddings:
            raise ValueError("No item embeddings found")

        # 计算物品嵌入向量的和：∑(v_t * x_t)
        item_concat = torch.cat(
            item_embeddings, dim=1
        )  # [batch_size, num_item_features, embedding_dim]
        item_embedding_sum = self.item_embedding_sum(
            item_concat
        )  # [batch_size, embedding_dim]

        # 计算一阶线性项：∑(w_t * x_t)
        # 为每个物品特征学习一个权重
        item_linear_weights = self.item_linear_weights(
            item_embedding_sum
        )  # [batch_size, 1]

        # 计算FM二阶交互项：0.5 * ((∑v_t*x_t)² - ∑(v_t²*x_t²))
        # 1. 计算 (∑v_t*x_t)²
        sum_squared = self.item_sum_squared(
            item_embedding_sum
        )  # [batch_size, embedding_dim]

        # 2. 计算 ∑(v_t²*x_t²) = ∑(v_t²)，因为x_t=1对于one-hot特征
        item_squared = self.item_squared(
            item_concat
        )  # [batch_size, num_item_features, embedding_dim]
        squared_sum = self.item_squared_sum(
            item_squared
        )  # [batch_size, embedding_dim]

        # 3. 计算FM交互项：0.5 * (sum_squared - squared_sum)
        fm_interaction_vector = sum_squared - squared_sum  # [batch_size, embedding_dim]
        # 乘以0.5
        fm_interaction_half = self.fm_half_scale(fm_interaction_vector)

        # 4. 聚合FM交互项为标量
        fm_interaction_scalar = self.fm_interaction_scalar(
            fm_interaction_half
        )  # [batch_size, 1]

        # 5. 计算first_term = ∑(w_t*x_t) + FM_interaction
        first_term = item_linear_weights + fm_interaction_scalar  # [batch_size, 1]

        # 6. 构建物品向量：[first_term; ∑(v_t * x_t)]
        item_vector = torch.cat([first_term, item_embedding_sum], dim=1)

        return item_vector

    def encode_user(self, inputs):
        """用户塔: 返回 [1; ∑(v_u * x_u)]，B x (1 + embedding_dim)"""
        return self._user_tower(
            self._group_embeddings(self.user_feature_columns, inputs, "user")
        )

    def encode_item(self, inputs):
        """物品塔: 返回 [first_term; ∑(v_t * x_t)]，B x (1 + embedding_dim)"""
        return self._item_tower(
            self._group_embeddings(self.item_feature_columns, inputs, "item")
        )

    def forward(self, inputs):
        # 构建嵌入特征
        group_embedding_feature_dict = self.embedding(inputs)

        # 构建双塔
        user_representation = self._user_tower(
            group_embedding_feature_dict.get("user", [])
        )  # [batch_size, 1 + embedding_dim]
        item_representation = self._item_tower(
            group_embedding_feature_dict.get("item", [])
        )  # [batch_size, 1 + embedding_dim]

        # 计算FM匹配分数：V_item · V_user^T（内积，等价于 Keras Dot(axes=1)）
        fm_score = torch.sum(
            item_representation * user_representation, dim=1, keepdim=True
        )  # [batch_size, 1]

        # 输出层（sigmoid激活用于二分类）
        logits = self.output_dense(fm_score)
        output = torch.sigmoid(logits)
        # 原实现最后一层为 Dense(activation="sigmoid")（其后无 Flatten），Keras 交叉熵损失会直接使用其 logits
        # （from_logits=True，不裁剪），见 training/loss.py 中的 _keras_logits
        output._keras_logits = logits
        output._keras_logits_op = "Sigmoid"
        return output


def build_fm_recall_model(feature_columns, model_config):
    """
    构建因子分解机(FM)模型 - 双塔结构用于召回
    基于FM的数学分解：MatchScore = V_item · V_user^T

    根据FM的数学推导：
    - 用户向量：V_user = [1; ∑(v_u * x_u)]
    - 物品向量：V_item = [∑w_t*x_t + FM_interaction; ∑(v_t * x_t)]

    Args:
        feature_columns: 特征列配置
        model_config: 模型配置字典，包含:
            - embedding_dim: 嵌入维度 (default: 8)

    Returns:
        Tuple of (training_model, user_model, item_model)
    """
    training_model = FMRecallModel(feature_columns, model_config)
    user_model = training_model.user_tower
    item_model = training_model.item_tower
    return training_model, user_model, item_model
