"""
FunkSVD 模型实现

这实现了基本的奇异值分解（SVD）协同过滤模型，不包含偏置项，也称为 FunkSVD。
该模型学习用户和物品的潜在因子来预测用户-物品交互。

评分预测：r_ui = p_u^T * q_i
其中 p_u 是用户潜在因子向量，q_i 是物品潜在因子向量。
"""

import torch

from .base import Dense, Embedding, FunRecModel, SubModel
from .utils import build_input_layer


class FunkSVDModel(FunRecModel):
    """FunkSVD 训练模型（PyTorch 实现）

    forward 输出 sigmoid(Dense(p_u · q_i))，形状 B x 1。
    用户塔 / 物品塔通过 encode_user / encode_item 暴露，分别输出用户 / 物品潜在因子。
    """

    def __init__(
        self,
        user_input_names,
        item_input_names,
        user_vocab_size,
        item_vocab_size,
        embedding_dim=8,
        user_id_name="user_id",
        item_id_name="movie_id",
    ):
        super().__init__(
            input_names=list(user_input_names) + list(item_input_names),
            name="funksvd_training",
        )
        self.user_id_name = user_id_name
        self.item_id_name = item_id_name

        # === 用户塔 ===
        # 用户潜在因子
        self.user_factors = Embedding(
            user_vocab_size,
            embedding_dim,
            embeddings_initializer="normal",
            l2_reg=0.02,
            name="user_factors",
        )

        # === 物品塔 ===
        # 物品潜在因子
        self.item_factors = Embedding(
            item_vocab_size,
            embedding_dim,
            embeddings_initializer="normal",
            l2_reg=0.02,
            name="item_factors",
        )

        # 输出层（sigmoid 激活在 forward 中单独计算，以便记录 logits）
        self.output_layer = Dense(1, activation=None, name="output")

        # === 独立的用户和物品模型（与主模型共享参数）===
        self.user_tower = SubModel(self, "encode_user", list(user_input_names), name="user_tower")
        self.item_tower = SubModel(self, "encode_item", list(item_input_names), name="item_tower")

    @staticmethod
    def _flatten(x):
        # 等价于 Keras Flatten: B x L x D -> B x (L*D)
        return x.reshape(x.shape[0], -1)

    def encode_user(self, inputs):
        # 用户潜在因子
        return self._flatten(self.user_factors(inputs[self.user_id_name]))

    def encode_item(self, inputs):
        # 物品潜在因子
        return self._flatten(self.item_factors(inputs[self.item_id_name]))

    def forward(self, inputs):
        user_factors = self.encode_user(inputs)
        item_factors = self.encode_item(inputs)

        # 仅交互项（无偏置项）: user_factors · item_factors（等价于 Keras Dot(axes=1)）
        prediction = torch.sum(user_factors * item_factors, dim=1, keepdim=True)

        # 输出层
        logits = self.output_layer(prediction)
        output = torch.sigmoid(logits)
        # 原实现最后一层为 Dense(activation="sigmoid")，Keras 交叉熵损失会直接使用其 logits
        # （from_logits=True，不裁剪），见 training/loss.py 中的 _keras_logits
        output._keras_logits = logits
        output._keras_logits_op = "Sigmoid"
        return output


def build_funksvd_model(feature_columns, model_config):
    """
    构建 FunkSVD 协同过滤模型 - 用于召回的双塔结构。

    FunkSVD 基于：评分 = 用户因子 · 物品因子^T

    Args:
        feature_columns: 特征列配置
        model_config: 模型配置字典，包含：
            - embedding_dim: 嵌入维度（对应 SVD 中的潜在因子数量）（默认：8）

    Returns:
        返回 (训练模型, 用户模型, 物品模型) 的元组
    """

    # 从配置中提取参数，设置默认值
    embedding_dim = model_config.get("embedding_dim", 8)

    # 构建输入层
    input_layer_dict = build_input_layer(feature_columns)

    # 查找用户和物品 ID 输入
    user_id_name = None
    item_id_name = None
    user_inputs = []
    item_inputs = []
    user_vocab_size = None
    item_vocab_size = None

    for fc in feature_columns:
        if "user" in fc.group:
            user_inputs.append(input_layer_dict[fc.name].name)
            if fc.name == "user_id":
                user_id_name = fc.name
                user_vocab_size = fc.vocab_size
        elif "item" in fc.group:
            item_inputs.append(input_layer_dict[fc.name].name)
            if fc.name in ["movie_id", "item_id"]:
                item_id_name = fc.name
                item_vocab_size = fc.vocab_size

    if user_id_name is None or item_id_name is None:
        raise ValueError("需要 user_id 和 item_id（或 movie_id）输入")

    # === 训练模型 ===
    training_model = FunkSVDModel(
        user_input_names=user_inputs,
        item_input_names=item_inputs,
        user_vocab_size=user_vocab_size,
        item_vocab_size=item_vocab_size,
        embedding_dim=embedding_dim,
        user_id_name=user_id_name,
        item_id_name=item_id_name,
    )

    # === 独立的用户和物品模型 ===
    user_model = training_model.user_tower
    item_model = training_model.item_tower

    return training_model, user_model, item_model
