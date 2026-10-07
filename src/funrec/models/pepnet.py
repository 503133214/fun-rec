import torch.nn as nn

from .base import FunRecModel, Dense
from .layers import EPNet, PPNet, PredictLayer
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    add_tensor_func,
    get_linear_logits,
)


class PEPNetModel(FunRecModel):
    """PEPNet 多任务排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        task_name_list = model_config.get(
            "task_names", ["is_click", "long_view", "is_like"]
        )
        pepnet_dnn_units = model_config.get("pepnet_dnn_units", [128, 64])
        pepnet_activation = model_config.get("pepnet_activation", "relu")
        pepnet_dropout = model_config.get("pepnet_dropout", 0.1)
        l2_reg = model_config.get("l2_reg", 1e-5)
        linear_logits = model_config.get("linear_logits", False)

        # 构建输入层字典
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="pepnet")

        self.task_name_list = list(task_name_list)
        self.use_linear_logits = bool(linear_logits)

        # 构建特征嵌入表字典
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 创建EPNet和PPNet网络结构
        # EPNet负责嵌入个性化
        self.epnet = EPNet(l2_reg, name="dnn/epnet")
        # PPNet负责参数个性化，multiples参数为任务数量
        self.ppnet = PPNet(
            len(task_name_list),
            pepnet_dnn_units,
            pepnet_activation,
            pepnet_dropout,
            l2_reg,
            name="dnn/ppnet",
        )

        if self.use_linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)
            # 每个任务一个无偏置的 Dense(1)，将 PPNet 输出映射为 logit
            self.ppout_dense = nn.ModuleList(
                [Dense(1, use_bias=False) for _ in task_name_list]
            )

        # 为每个任务创建输出层
        self.predict_layers = nn.ModuleList(
            [PredictLayer(name=task_name) for task_name in task_name_list]
        )
        self.output_names = list(task_name_list)

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        # 连接不同组的嵌入向量作为各个网络的输入
        epnet_inputs = concat_group_embedding(group_embedding_feature_dict, "epnet")
        pepnet_inputs = concat_group_embedding(group_embedding_feature_dict, "pepnet")
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "dnn")

        # 通过EPNet处理嵌入
        ep_emb = self.epnet([epnet_inputs, dnn_inputs])
        # 通过PPNet处理经过个性化的嵌入，生成多任务输出
        pp_output = self.ppnet([ep_emb, pepnet_inputs])

        if self.use_linear_logits:
            # 原 Keras 实现中线性 logits 形状为 B x 1 x 1，与 B x 1 相加后得到 B x 1 x 1，
            # 这里保持相同的输出形状
            linear_logits = self.linear_logits(inputs).unsqueeze(1)  # B x 1 x 1
            pp_output_logits = []
            for i, pp in enumerate(pp_output):
                ppout_logit = self.ppout_dense[i](pp)
                task_logit = add_tensor_func([linear_logits, ppout_logit])
                pp_output_logits.append(task_logit)
            pp_output = pp_output_logits

        # 为每个任务创建输出层
        output_list = []
        for i in range(len(self.task_name_list)):
            # 对每个任务使用预测层生成最终输出
            prediction = self.predict_layers[i](pp_output[i])
            output_list.append(prediction)

        # 注: 与原 Keras 实现一致，只有一个任务时模型输出单个张量（而非长度为 1 的列表）
        if len(output_list) == 1:
            return output_list[0]
        return output_list


def build_pepnet_model(feature_columns, model_config):
    """构建PEPNet排序模型并遵循FunRec接口约定

    Args:
        feature_columns: 特征列配置
        model_config: 模型参数配置，包含：
            - task_names: 任务名称列表（默认 ["is_click", "long_view", "is_like"]）
            - pepnet_dnn_units: PPNet中DNN层的隐藏单元数量（默认 [128, 64]）
            - pepnet_activation: PPNet中DNN层的激活函数（默认 'relu'）
            - pepnet_dropout: PPNet中的dropout比例（默认 0.1）
            - l2_reg: L2正则化系数（默认 1e-5）
            - linear_logits: 是否加上线性项（默认 False）
    Returns:
        (model, None, None): 排序模型返回单一主模型
    """
    model = PEPNetModel(feature_columns, model_config)
    # FunRec排序模型约定：返回 (model, None, None)
    return model, None, None
