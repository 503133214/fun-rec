import torch

from .base import FunRecModel, Dense
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    add_tensor_func,
    get_linear_logits,
    concat_func,
)
from .layers import (
    DNNs,
    PredictLayer,
    PartitionedNormalization,
    StarTopologyFCN,
)


class StarModel(FunRecModel):
    """STAR（星形拓扑自适应推荐器）排序模型的 PyTorch 实现（参数含义见 build_star_model）"""

    def __init__(self, feature_columns, model_config):
        num_domains = model_config.get("num_domains", 5)
        domain_feature_name = model_config.get("domain_feature_name", "tab")
        star_dnn_units = model_config.get("star_dnn_units", [128, 64])
        aux_dnn_units = model_config.get("aux_dnn_units", [128, 64])
        star_fcn_activation = model_config.get("star_fcn_activation", "relu")
        dropout = model_config.get("dropout", 0.2)
        l2_reg = model_config.get("l2_reg", 1e-5)
        linear_logits = model_config.get("linear_logits", False)
        # 构建输入层字典
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="star")
        if domain_feature_name not in input_layer_dict:
            raise KeyError(domain_feature_name)
        self.domain_feature_name = domain_feature_name

        # 构建特征嵌入表字典
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # STAR 主网络: 分域归一化 + 星形拓扑全连接网络
        self.fcn_pn_layer = PartitionedNormalization(num_domain=num_domains, name="fcn_pn_layer")
        self.star_fcn_layer = StarTopologyFCN(
            num_domains,
            star_dnn_units,
            star_fcn_activation,
            dropout,
            l2_reg,
            name="star_fcn_layer",
        )
        self.fcn_logits = PredictLayer(activation=None, name="fcn_logits")

        # 辅助网络: 域嵌入 + 特征嵌入，经分域归一化后输入 DNN
        self.aux_pn_layer = PartitionedNormalization(num_domain=num_domains, name="aux_pn_layer")
        self.aux_dnn = DNNs(aux_dnn_units, dropout_rate=dropout)
        self.aux_logits = PredictLayer(activation=None, name="aux_logits")

        self.use_linear_logits = bool(linear_logits)
        self.linear_logits = get_linear_logits(feature_columns) if self.use_linear_logits else None

        self.star_output = Dense(1, activation="sigmoid", name="star_output")

    def forward(self, inputs):
        domain_input = inputs[self.domain_feature_name]

        group_embedding_feature_dict = self.embedding(inputs)

        # 连接不同组的嵌入向量作为各个网络的输入
        domain_embeddings = concat_group_embedding(group_embedding_feature_dict, "domain")
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "dnn")

        fcn_inputs = self.fcn_pn_layer([dnn_inputs, domain_input])
        fcn_output = self.star_fcn_layer([fcn_inputs, domain_input])
        # 注: PredictLayer(activation=None) 在 as_logit=False 时仍会使用 sigmoid（与原实现一致）
        fcn_logit = self.fcn_logits(fcn_output)

        aux_inputs = concat_func([domain_embeddings, dnn_inputs], axis=-1)
        aux_inputs = self.aux_pn_layer([aux_inputs, domain_input])

        aux_output = self.aux_dnn(aux_inputs)
        aux_logit = self.aux_logits(aux_output)

        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            final_logits = add_tensor_func([linear_logit, fcn_logit, aux_logit])
        else:
            final_logits = add_tensor_func([fcn_logit, aux_logit])

        # 将logits转换为概率并展平以匹配标签形状
        final_logits = torch.flatten(final_logits, start_dim=1)
        final_prediction = self.star_output(final_logits)
        # 原实现在 sigmoid 之后还有一个 Flatten，因此 Keras 的 BCE 使用概率（裁剪）计算，不标记 logits
        final_prediction = torch.flatten(final_prediction, start_dim=1)
        return final_prediction


def build_star_model(feature_columns, model_config):
    """
    为FunRec构建STAR（星形拓扑自适应推荐器）排序模型。

    Args:
        feature_columns: FeatureColumn列表
        model_config: 包含参数的字典:
            - num_domains: 整数，域的数量
            - domain_feature_name: 字符串，域指示特征名称
            - star_dnn_units: 列表，STAR FCN隐藏单元 (默认 [128, 64])
            - aux_dnn_units: 列表，辅助DNN隐藏单元 (默认 [128, 64])
            - star_fcn_activation: 字符串，STAR FCN的激活函数 (默认 'relu')
            - dropout: 浮点数，dropout率 (默认 0.2)
            - l2_reg: 浮点数，L2正则化 (默认 1e-5)
            - linear_logits: 布尔值，是否添加线性项 (默认 False)

    Returns:
        (model, None, None): 排序模型元组
    """
    # 构建模型
    model = StarModel(feature_columns, model_config)
    return model, None, None
