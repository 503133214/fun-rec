import torch

from .base import FunRecModel
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
)
from .layers import DNNs, PredictLayer


class ESMMModel(FunRecModel):
    """ESMM（全空间多任务模型）排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        task_names = model_config.get("task_names", ["is_click", "is_like"])
        task_tower_dnn_units = model_config.get("task_tower_dnn_units", [128, 64])
        dropout_rate = model_config.get("dropout_rate", 0.1)

        # 输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="esmm")

        self.task_names = task_names

        # 分组嵌入
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # CTR塔（第一个任务）
        self.ctr_dnn = DNNs(
            name="ctr_dnn", units=task_tower_dnn_units + [1], dropout_rate=dropout_rate
        )

        # CVR塔（第二个任务）
        self.cvr_dnn = DNNs(
            name="cvr_dnn", units=task_tower_dnn_units + [1], dropout_rate=dropout_rate
        )

        # 预测层
        self.ctr_output = PredictLayer(name="ctr_output")
        self.cvr_output = PredictLayer(name="cvr_output")

    def forward(self, inputs):
        # 分组嵌入
        group_embedding_feature_dict = self.embedding(inputs)

        # 连接不同组的嵌入向量作为网络的输入
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "dnn")

        # CTR塔 / CVR塔
        ctr_output_logits = self.ctr_dnn(dnn_inputs)
        cvr_output_logits = self.cvr_dnn(dnn_inputs)

        # 应用预测层
        ctr_output_prob = self.ctr_output(ctr_output_logits)
        cvr_output_prob = self.cvr_output(cvr_output_logits)

        # CTCVR = CTR * CVR（ESMM核心思想）
        # 注: 乘积的最后一个算子不是 sigmoid，因此其交叉熵损失按概率计算（裁剪），与原实现一致
        ctcvr_output_prob = torch.multiply(ctr_output_prob, cvr_output_prob)

        # 输出任务输出列表
        return [ctr_output_prob, ctcvr_output_prob]


def build_esmm_model(feature_columns, model_config):
    """
    构建ESMM（全空间多任务模型）排序模型。

    参数:
        feature_columns: FeatureColumn列表
        model_config: 包含以下参数的字典:
            - task_names: 列表，任务名称（默认 ["is_click", "is_like"]）
            - task_tower_dnn_units: 列表，任务塔DNN隐藏单元数（默认 [128, 64]）
            - dropout_rate: 浮点数，丢弃率（默认 0.1）

    返回:
        (model, None, None): 排序模型元组
    """
    model = ESMMModel(feature_columns, model_config)
    return model, None, None
