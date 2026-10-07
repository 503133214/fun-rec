import torch

from .base import FunRecModel, SubModel
from .utils import (
    concat_group_embedding,
    build_input_layer,
    FeatureEmbedding,
    build_group_feature_embedding_table_dict,
)
from .layers import DNNs, PredictLayer


def _l2_normalize(x, axis=1, epsilon=1e-12):
    """L2 归一化（与原 TensorFlow l2_normalize 数值一致）: x * rsqrt(max(sum(x^2), epsilon))"""
    square_sum = torch.sum(x * x, dim=axis, keepdim=True)
    return x * torch.rsqrt(torch.clamp(square_sum, min=epsilon))


class DSSMModel(FunRecModel):
    """双塔模型 (DSSM) 的 PyTorch 实现

    - 主模型输出: 用户/物品向量余弦相似度经 sigmoid 后的概率 (B x 1)
    - user_tower / item_tower: 与主模型共享参数的用户塔、物品塔子模型，输出 L2 归一化后的向量
    """

    def __init__(self, feature_columns, model_config):
        # Extract parameters from config with defaults
        dnn_units = model_config.get("dnn_units", [128, 64, 32])
        dropout_rate = model_config.get("dropout_rate", 0.2)
        # 构建输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="dssm")

        self.feature_columns = list(feature_columns)
        self.user_feature_columns = [fc for fc in feature_columns if "user" in fc.group]
        self.item_feature_columns = [fc for fc in feature_columns if "item" in fc.group]

        # 构建特征embedding表
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 构建用户塔和物品塔
        self.user_dnn = DNNs(
            units=dnn_units, activation="tanh", dropout_rate=dropout_rate, use_bn=True
        )
        self.item_dnn = DNNs(
            units=dnn_units, activation="tanh", dropout_rate=dropout_rate, use_bn=True
        )

        # 构建输出层
        self.dssm_output = PredictLayer(name="dssm_output")

        # 用户塔 / 物品塔子模型（与主模型共享参数）
        user_input_names = [fc.name for fc in self.user_feature_columns]
        item_input_names = [fc.name for fc in self.item_feature_columns]
        self.user_tower = SubModel(self, "encode_user", user_input_names, name="user_model")
        self.item_tower = SubModel(self, "encode_item", item_input_names, name="item_model")

    def _group_feature(self, feature_columns, inputs, group_name):
        # 仅查询该塔所需特征的 embedding（与主模型共享 embedding 表）
        group_embedding_feature_dict = build_group_feature_embedding_table_dict(
            feature_columns,
            inputs,
            self.embedding.embedding_table_dict,
            self.embedding.mean_pooling,
        )
        return concat_group_embedding(
            group_embedding_feature_dict, group_name, axis=1, flatten=True
        )  # B x (N*D)

    def encode_user(self, inputs):
        """用户塔: 返回 L2 归一化后的用户 embedding"""
        user_feature = self._group_feature(self.user_feature_columns, inputs, "user")
        user_tower = self.user_dnn(user_feature)
        # 获取用户的embedding
        return _l2_normalize(user_tower, axis=1)

    def encode_item(self, inputs):
        """物品塔: 返回 L2 归一化后的物品 embedding"""
        item_feature = self._group_feature(self.item_feature_columns, inputs, "item")
        item_tower = self.item_dnn(item_feature)
        # 获取物品的embedding
        return _l2_normalize(item_tower, axis=1)

    def forward(self, inputs):
        # 构建特征embedding
        group_embedding_feature_dict = self.embedding(inputs)

        # 拼接特征
        user_feature = concat_group_embedding(
            group_embedding_feature_dict, "user", axis=1, flatten=True
        )  # B x (N*D)
        item_feature = concat_group_embedding(
            group_embedding_feature_dict, "item", axis=1, flatten=True
        )  # B x (N*D)

        # 用户塔和物品塔
        user_tower = self.user_dnn(user_feature)
        item_tower = self.item_dnn(item_feature)

        # 获取用户和物品的embedding
        user_embedding = _l2_normalize(user_tower, axis=1)
        item_embedding = _l2_normalize(item_tower, axis=1)

        # 计算余弦相似度 (等价于 Keras Dot(axes=1))
        cosine_similarity = torch.sum(
            user_embedding * item_embedding, dim=1, keepdim=True
        )  # B x 1

        # 输出层
        output = self.dssm_output(cosine_similarity)
        return output


def build_dssm_model(feature_columns, model_config):
    """
    构建双塔模型

    参数:
    feature_columns: 特征列配置
    model_config: 模型配置字典，包含:
        - dnn_units: 物品和用户塔的层单元数 (default: [128, 64, 32])
        - dropout_rate: 丢弃概率 (default: 0.2)

    返回:
    (model, user_model, item_model): 主模型、用户塔子模型、物品塔子模型（共享参数）
    """
    model = DSSMModel(feature_columns, model_config)
    user_model = model.user_tower
    item_model = model.item_tower
    return model, user_model, item_model
