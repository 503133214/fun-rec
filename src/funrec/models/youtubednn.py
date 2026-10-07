import torch

from .base import FunRecModel, SubModel
from .utils import (
    concat_group_embedding,
    build_input_layer,
    build_group_feature_embedding_table_dict,
    FeatureEmbedding,
)
from .layers import DNNs, SampledSoftmaxLayer, L2NormalizeLayer, SqueezeLayer


class YouTubeDNNModel(FunRecModel):
    """YouTubeDNN 召回模型的 PyTorch 实现

    - 主模型输出: 每个样本的采样 softmax 损失 [B, 1]（配合 sampledsoftmaxloss 使用）
    - user_tower: L2 归一化后的用户向量 [B, emb_dim]，输入为 user_dnn / raw_hist_seq 组特征
    - item_tower: L2 归一化后的物品 embedding [B, emb_dim]，输入为物品 id
    """

    def __init__(self, feature_columns, model_config):
        # 从model_config中提取参数
        emb_dim = model_config.get("emb_dim", 16)
        neg_sample = model_config.get("neg_sample", 20)
        dnn_units = model_config.get("dnn_units", [32])
        label_name = model_config.get("label_name", "movie_id")

        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="youtubednn")

        self.label_name = label_name
        self.feature_columns = list(feature_columns)

        # 构建特征embedding表
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 获取物品词汇表大小
        item_vocab_size = None
        for fc in feature_columns:
            if fc.name == label_name:
                item_vocab_size = fc.vocab_size
                break

        # 构建用户塔
        self.user_dnn = DNNs(units=dnn_units + [emb_dim], activation="relu", use_bn=False)
        self.user_l2_norm = L2NormalizeLayer(axis=-1)

        # 构建采样softmax层
        self.sampled_softmax_layer = SampledSoftmaxLayer(
            item_vocab_size, neg_sample, emb_dim
        )

        # 物品塔（评估用）
        self.item_squeeze = SqueezeLayer(axis=1)
        self.item_l2_norm = L2NormalizeLayer(axis=-1)

        # 构建用户模型和物品模型用于评估（与主模型共享参数）
        self.user_feature_columns = [
            fc
            for fc in feature_columns
            if "user_dnn" in fc.group or "raw_hist_seq" in fc.group
        ]
        user_feature_names = [fc.name for fc in self.user_feature_columns]
        self.user_tower = SubModel(
            self, "encode_user", user_feature_names, name="user_model"
        )
        self.item_tower = SubModel(self, "encode_item", [label_name], name="item_model")

    def _user_embedding(self, group_embedding_feature_dict):
        """由按组嵌入计算 L2 归一化后的用户向量 [B, emb_dim]"""
        user_feature_embedding = concat_group_embedding(
            group_embedding_feature_dict, "user_dnn"
        )  # B x (D * N)
        if "raw_hist_seq" in group_embedding_feature_dict:
            hist_seq_embedding = concat_group_embedding(
                group_embedding_feature_dict, "raw_hist_seq"
            )  # B x D
            user_dnn_inputs = torch.cat(
                [user_feature_embedding, hist_seq_embedding], dim=1
            )  # B x (D * N + D)
        else:
            user_dnn_inputs = user_feature_embedding  # B x (D * N)

        # 构建用户塔
        user_dnn_output = self.user_dnn(user_dnn_inputs)
        user_dnn_output = self.user_l2_norm(user_dnn_output)
        return user_dnn_output

    def encode_user(self, inputs):
        """用户塔: 返回用户向量 [B, emb_dim]"""
        # 仅查询用户塔所需特征的 embedding（与主模型共享 embedding 表）
        group_embedding_feature_dict = build_group_feature_embedding_table_dict(
            self.user_feature_columns,
            inputs,
            self.embedding.embedding_table_dict,
            self.embedding.mean_pooling,
        )
        return self._user_embedding(group_embedding_feature_dict)

    def encode_item(self, inputs):
        """物品塔: 返回 L2 归一化后的物品 embedding [B, emb_dim]"""
        item_table = self.embedding.embedding_table_dict[self.label_name]
        output_item_embedding = self.item_squeeze(item_table(inputs[self.label_name]))
        output_item_embedding = self.item_l2_norm(output_item_embedding)
        return output_item_embedding

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)
        user_dnn_output = self._user_embedding(group_embedding_feature_dict)

        # 获取物品嵌入表
        item_embedding_table = self.embedding.embedding_table_dict[self.label_name]
        output = self.sampled_softmax_layer(
            [item_embedding_table.embeddings, user_dnn_output, inputs[self.label_name]]
        )
        return output


def build_youtubednn_model(feature_columns, model_config):
    """
    构建YouTubeDNN召回模型

    返回:
    (model, user_model, item_model): 主模型、用户模型、物品模型（共享参数）
    """
    model = YouTubeDNNModel(feature_columns, model_config)
    return model, model.user_tower, model.item_tower
