import torch

from .base import FunRecModel, SubModel
from .utils import (
    concat_group_embedding,
    build_input_layer,
    build_group_feature_embedding_table_dict,
    FeatureEmbedding,
    pooling_group_embedding,
)
from .layers import (
    DNNs,
    CapsuleLayer,
    LabelAwareAttention,
    SampledSoftmaxLayer,
    L2NormalizeLayer,
)


def dynamic_capsule_num(seq_len, k_max):
    """
    动态计算胶囊数量

    参数:
        seq_len: 序列长度 [B, 1]
        k_max: 最大胶囊数量

    返回:
        k_capsule: 动态胶囊数量
    """
    seq_len = torch.squeeze(seq_len, dim=1)  # [B,]
    log_len = torch.log1p(seq_len.to(torch.float32))
    log_2 = torch.log(torch.tensor(2.0, dtype=torch.float32, device=log_len.device))
    k_max = torch.tensor(float(k_max), dtype=torch.float32, device=log_len.device)
    # 最少一个胶囊
    k_capsule = torch.clamp(torch.minimum(k_max, log_len / log_2), min=1.0).to(
        torch.int32
    )
    return k_capsule


class MINDModel(FunRecModel):
    """MIND (Multi-Interest Network with Dynamic routing) 的 PyTorch 实现

    - 主模型输出: 每个样本的采样 softmax 损失 [B, 1]（配合 sampledsoftmaxloss 使用）
    - user_tower: 用户多兴趣向量 [B, k_max, emb_dims]（标签感知注意力之前），
      输入为 user_dnn / raw_hist_seq 组特征 + hist_len
    - item_tower: 物品 embedding [B, emb_dims]，输入为物品 id
    """

    def __init__(self, feature_columns, model_config):
        # 从配置中提取参数并设置默认值
        neg_samples = model_config.get("neg_samples", 50)
        emb_dims = model_config.get("emb_dims", 16)
        max_capsulen_nums = model_config.get("max_capsulen_nums", 4)
        max_seq_len = model_config.get("max_seq_len", 50)
        user_dnn_units = model_config.get("user_dnn_units", [128, 64])

        # 从特征列中获取物品词汇表大小和标签名称
        item_vocab_size = None
        label_name = None
        for fc in feature_columns:
            if "target_item" in fc.group and fc.name == "movie_id":
                item_vocab_size = fc.vocab_size
                label_name = fc.name
                break

        if item_vocab_size is None or label_name is None:
            raise ValueError("无法从特征列中找到物品词汇表大小或标签名称")

        # 获取用户嵌入特征名称列表
        user_emb_feature_name_list = []
        for fc in feature_columns:
            if any(group in ["user_dnn", "raw_hist_seq"] for group in fc.group):
                user_emb_feature_name_list.append(fc.name)
        user_emb_feature_name_list.append("hist_len")  # 添加hist_len

        # 构建输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="mind")

        self.feature_columns = list(feature_columns)
        self.user_feature_columns = [
            fc
            for fc in feature_columns
            if any(group in ["user_dnn", "raw_hist_seq"] for group in fc.group)
        ]
        self.label_name = label_name
        self.max_capsulen_nums = max_capsulen_nums
        self.max_seq_len = max_seq_len

        # 构建特征embedding表
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        self.capsule_layer = CapsuleLayer(
            input_units=emb_dims,
            out_units=emb_dims,
            max_len=max_seq_len,
            k_max=max_capsulen_nums,
        )
        # 再将拼接后的特征降维到胶囊的维度
        self.user_dnn = DNNs(user_dnn_units + [emb_dims])
        # 计算LabelAwareAttention
        self.label_aware_attention = LabelAwareAttention(
            k_max=max_capsulen_nums, pow_p=1.0
        )
        self.l2_normalize = L2NormalizeLayer(axis=-1)
        # 构建采样softmax层
        self.sampled_softmax_layer = SampledSoftmaxLayer(
            vocab_size=item_vocab_size, num_sampled=neg_samples, emb_dim=emb_dims
        )

        # 构建用户模型和物品模型用于评估（与主模型共享参数）
        self.user_tower = SubModel(
            self, "encode_user", user_emb_feature_name_list, name="user_model"
        )
        self.item_tower = SubModel(self, "encode_item", [label_name], name="item_model")

    def build_with(self, sample_features, batch_size: int = 2):
        """创建惰性参数。

        CapsuleLayer 每次前向都会累加更新 routing_logits，而原 Keras 模型在训练前不会做预热前向，
        因此预热前向之后需要把 routing_logits 恢复为预热前的值（首次构建时则重新按初始化分布采样）。
        """
        capsule = self.capsule_layer
        snapshot = capsule.routing_logits.detach().clone() if capsule.built else None
        super().build_with(sample_features, batch_size=batch_size)
        with torch.no_grad():
            if snapshot is not None:
                capsule.routing_logits.copy_(snapshot)
            else:
                capsule.routing_logits.normal_(0.0, capsule.init_std)
        return self

    def _user_embeddings(self, group_embedding_feature_dict, hist_len):
        """计算用户多兴趣向量 [B, k_max, emb_dims]"""
        user_dnn_inputs = concat_group_embedding(
            group_embedding_feature_dict, "user_dnn"
        )
        user_hist_seq_embedding = pooling_group_embedding(
            group_embedding_feature_dict, "raw_hist_seq"
        )

        # 转换成胶囊的数量
        user_dnn_inputs = user_dnn_inputs.unsqueeze(1)
        user_dnn_inputs = user_dnn_inputs.repeat(
            1, self.max_capsulen_nums, 1
        )  # [B, k_max, feat_num x dim]

        # 因为序列是左侧padding，所以mask需要反转
        hist_len = hist_len.reshape(-1, 1)
        sequence_mask = torch.arange(
            self.max_seq_len, device=hist_len.device
        ).unsqueeze(0) < hist_len  # 等价于 sequence_mask(hist_len, max_seq_len)
        sequence_mask = torch.flip(sequence_mask, dims=[1])  # [B, max_seq_len]

        capsule_num = dynamic_capsule_num(hist_len, self.max_capsulen_nums)

        high_capsule = self.capsule_layer(
            [user_hist_seq_embedding, sequence_mask, capsule_num]
        )  # [B,k_max,out_units]

        # 将每个胶囊都拼接上用户的特征
        user_embeddings = torch.cat([user_dnn_inputs, high_capsule], dim=-1)
        # 再将拼接后的特征降维到胶囊的维度
        user_embeddings = self.user_dnn(user_embeddings)
        return user_embeddings

    def encode_user(self, inputs):
        """用户塔: 返回用户多兴趣向量 [B, k_max, emb_dims]"""
        # 仅查询用户塔所需特征的 embedding（与主模型共享 embedding 表）
        group_embedding_feature_dict = build_group_feature_embedding_table_dict(
            self.user_feature_columns,
            inputs,
            self.embedding.embedding_table_dict,
            self.embedding.mean_pooling,
        )
        return self._user_embeddings(group_embedding_feature_dict, inputs["hist_len"])

    def encode_item(self, inputs):
        """物品塔: 返回物品 embedding [B, emb_dims]"""
        item_table = self.embedding.embedding_table_dict[self.label_name]
        # 获取item embedding
        return torch.squeeze(item_table(inputs[self.label_name]), dim=1)

    def forward(self, inputs):
        # 构建特征embedding
        group_embedding_feature_dict = self.embedding(inputs)

        user_embeddings = self._user_embeddings(
            group_embedding_feature_dict, inputs["hist_len"]
        )
        target_item_embedding = pooling_group_embedding(
            group_embedding_feature_dict, "target_item"
        )

        # 计算LabelAwareAttention
        user_embedding_final = self.label_aware_attention(
            (user_embeddings, target_item_embedding)
        )
        user_embedding_final = self.l2_normalize(user_embedding_final)

        # 获取item emb权重
        item_embedding_weight = self.embedding.embedding_table_dict[
            self.label_name
        ].embeddings
        # 计算输出
        output = self.sampled_softmax_layer(
            [item_embedding_weight, user_embedding_final, inputs[self.label_name]]
        )
        return output


def build_mind_model(feature_columns, model_config):
    """
    构建MIND (Multi-Interest Network with Dynamic routing)模型

    参数:
    feature_columns: 特征列配置
    model_config: 模型配置字典，包含:
        - neg_samples: 负采样数量 (默认: 50)
        - emb_dims: embedding维度 (默认: 16)
        - max_capsulen_nums: 最大胶囊数量 (默认: 4)
        - max_seq_len: 最大序列长度 (默认: 50)
        - user_dnn_units: 用户DNN层单元数 (默认: [128, 64])

    返回:
    (model, user_model, item_model): 主模型、用户模型（多兴趣向量）、物品模型（共享参数）
    """
    model = MINDModel(feature_columns, model_config)
    return model, model.user_tower, model.item_tower
