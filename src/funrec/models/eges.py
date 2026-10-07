import os
import logging

logger = logging.getLogger(__name__)

import torch
import networkx as nx
import numpy as np
from tqdm import tqdm

from .base import Embedding, FunRecModel, Layer, SubModel


class SimpleWalker:
    """简化的随机游走器实现"""

    def build_graph(self, session_list):

        logger.debug("构建图...")

        # 提取所有边
        edges = []
        for session in session_list:
            if len(session) > 1:
                for i in range(len(session) - 1):
                    u = int(session[i])
                    v = int(session[i + 1])
                    edges.append((u, v))

        if len(edges) == 0:
            return None, None

        # 创建NetworkX图
        G = nx.Graph()
        G.add_edges_from(edges)

        # 获取所有唯一节点
        nodes = list(G.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}
        reverse_node_map = {i: node for i, node in enumerate(nodes)}

        logger.debug(
            f"图包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边"
        )

        return G, (node_map, reverse_node_map)

    def generate_walks(self, G, num_walks, walk_length):
        """生成随机游走"""
        logger.debug(f"生成随机游走...")

        walks = []
        nodes = list(G.nodes())

        for _ in range(num_walks):
            np.random.shuffle(nodes)
            for node in tqdm(
                nodes,
                desc=f"Walk {_+1}/{num_walks}",
                disable=not logger.isEnabledFor(logging.DEBUG),
            ):
                walk = [node]
                for _ in range(walk_length - 1):
                    curr = walk[-1]
                    neighbors = list(G.neighbors(curr))
                    if len(neighbors) == 0:
                        break
                    walk.append(np.random.choice(neighbors))
                walks.append(walk)

        logger.debug(f"生成的游走序列数量: {len(walks)}")

        return walks


def get_graph_context_all_pairs(walks, window_size):
    """
    从随机游走生成训练样本对
    """
    all_pairs = []
    for walk in tqdm(
        walks, desc="生成训练样本对", disable=not logger.isEnabledFor(logging.DEBUG)
    ):
        for i in range(len(walk)):
            for j in range(
                max(0, i - window_size), min(len(walk), i + window_size + 1)
            ):
                if i != j:
                    all_pairs.append((walk[i], walk[j]))
    logger.debug(f"生成的样本对数量: {len(all_pairs)}")
    return all_pairs


class ItemSpecificAttentionLayer(Layer):
    """
    正确实现EGES论文中描述的加权池化注意力层
    A ∈ R^{|V| × (n+1)} - 每个商品有自己的一组特征权重
    """

    def __init__(self, num_items, **kwargs):
        """
        参数:
            num_items: 商品总数 |V|
        """
        self.num_items = num_items
        super(ItemSpecificAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 商品数量和特征数量
        num_features = input_shape[1]  # n+1

        # 为每个商品创建一组特征权重 A ∈ R^{|V| × (n+1)}
        # RandomNormal() 默认 mean=0, stddev=0.05；L2 正则系数 1e-5
        self.attention_weights = self.add_weight(
            name="attention_weights",
            shape=(self.num_items, num_features),  # |V| x (n+1)
            initializer="random_normal",
            trainable=True,
            regularizer=1e-5,
        )

    def forward(self, inputs, item_indices):
        """
        参数:
            inputs: 特征嵌入 [batch_size, n+1, emb_dim], n 是特征数量，emb_dim 是嵌入维度
            item_indices: 当前批次中每个样本对应的商品索引 [batch_size]
        """

        # 获取每个样本对应的商品特定权重 [batch_size, n+1]
        batch_attention_weights = self.attention_weights[item_indices.long()]

        # 计算 e^(a_v^j)
        exp_attention = torch.exp(batch_attention_weights)  # [batch_size, n+1]

        # 计算每个样本的权重和 [batch_size, 1]
        attention_sum = torch.sum(exp_attention, dim=1, keepdim=True)

        # 归一化权重 [batch_size, n+1]
        normalized_attention = exp_attention / attention_sum

        # 扩展维度用于广播 [batch_size, n+1, 1]
        normalized_attention = normalized_attention.unsqueeze(-1)

        # 应用权重到特征嵌入
        weighted_embedding = inputs * normalized_attention  # [batch_size, n+1, emb_dim]

        # 求和得到最终表示
        output = torch.sum(weighted_embedding, dim=1)  # [batch_size, emb_dim]

        return output, normalized_attention


def generate_negative_samples(train_sample_dict, num_negatives=2):
    """生成负样本"""
    negative_sample_dict = {
        # 重复 movie_id 列 num_negatives次
        "movie_id": np.repeat(train_sample_dict["movie_id"], num_negatives),
        # 重复 context_id 列 num_negatives次
        "context_id": np.repeat(train_sample_dict["context_id"], num_negatives),
        # 重复 genre_id 列 num_negatives次
        "genre_id": np.repeat(train_sample_dict["genre_id"], num_negatives),
    }
    # 打乱负样本字典中的context_id
    np.random.shuffle(negative_sample_dict["context_id"])

    return negative_sample_dict


class EGESModel(FunRecModel):
    """EGES 主模型: 物品侧特征加权聚合后与上下文物品嵌入点积，输出 sigmoid 概率

    输入: {"movie_id": [B], "context_id": [B], "genre_id": [B]}（[B, 1] 会被展平为 [B]）
    输出: [B, 1] 概率
    子塔: self.item_tower（输入为 item_feature_list，输出最终物品嵌入 [B, emb_dim]），
          对应原实现中的 main_model.item_input / main_model.item_embedding
    """

    def __init__(
        self,
        item_feature_list,
        item_vocab_size,
        genre_vocab_size,
        emb_dim=16,
        l2_reg=1e-5,
        use_attention=True,
    ):
        super().__init__(input_names=["movie_id", "context_id", "genre_id"], name="eges")
        self.item_feature_list = list(item_feature_list)
        self.use_attention = use_attention

        # 嵌入层（RandomNormal 初始化 + L2 正则，不使用 mask）
        self.movie_emb_table = Embedding(
            item_vocab_size,
            emb_dim,
            embeddings_initializer="random_normal",
            l2_reg=l2_reg,
            mask_zero=False,
            name="eges_movie_id",
        )
        self.genre_emb_table = Embedding(
            genre_vocab_size,
            emb_dim,
            embeddings_initializer="random_normal",
            l2_reg=l2_reg,
            mask_zero=False,
            name="eges_genre_id",
        )
        self.context_emb_table = Embedding(
            item_vocab_size,
            emb_dim,
            embeddings_initializer="random_normal",
            l2_reg=l2_reg,
            mask_zero=False,
            name="eges_context_id",
        )

        # 加权聚合
        if use_attention:
            self.attention_layer = ItemSpecificAttentionLayer(num_items=item_vocab_size)
        else:
            self.attention_layer = None

        # 便于评估：物品输入与嵌入（与主模型共享参数）
        item_input_names = [k for k in self.input_names if k in self.item_feature_list]
        self.item_tower = SubModel(self, "encode_item", item_input_names, name="item_tower")

    @staticmethod
    def _ids(x):
        # 原实现输入形状为 ()，即 [B]；这里将 [B, 1] 展平为 [B]
        return x.reshape(-1)

    def _lookup(self, feat_name, ids):
        table = {
            "movie_id": self.movie_emb_table,
            "genre_id": self.genre_emb_table,
            "context_id": self.context_emb_table,
        }[feat_name]
        return table(self._ids(ids)).unsqueeze(1)  # [B,1,D]

    def encode_item(self, inputs):
        # 堆叠物品侧特征
        all_feature_embeddings = []
        for feat_name in self.item_feature_list:
            all_feature_embeddings.append(self._lookup(feat_name, inputs[feat_name]))  # [B,1,D]
        stacked_embeddings = torch.cat(all_feature_embeddings, dim=1)  # [B,n,D]

        # 加权聚合
        if self.use_attention:
            # 商品索引（注意力权重用）
            item_indices = self._ids(inputs["movie_id"]).long()
            final_embedding, _ = self.attention_layer(stacked_embeddings, item_indices)
        else:
            final_embedding = torch.mean(stacked_embeddings, dim=1)
        return final_embedding

    def forward(self, inputs):
        final_embedding = self.encode_item(inputs)

        # 与上下文点积
        context_embedding = self._lookup("context_id", inputs["context_id"]).squeeze(1)
        logits = torch.sum(final_embedding * context_embedding, dim=1, keepdim=True)
        output = torch.sigmoid(logits)
        # 原实现最后一层为 Activation("sigmoid")（其后无 Flatten），Keras 交叉熵损失会直接使用其 logits
        # （from_logits=True，不裁剪），见 training/loss.py 中的 _keras_logits
        output._keras_logits = logits
        output._keras_logits_op = "Sigmoid"
        return output


def build_eges_model(feature_columns, model_config):
    """
    构建EGES模型（自包含版本，适配funrec训练/评估流程）

    参数:
        feature_columns: 未使用（EGES使用自定义输入）
        model_config: 包含以下字段：
            - dataset_name: 数据集名称，用于解析字典大小            
            - item_feature_list: 物品侧特征列表（例如 ['movie_id', 'genre_id']）
            - emb_dim: 嵌入维度
            - l2_reg: L2正则
            - use_attention: 是否使用注意力聚合
    返回:
        (main_model, None, item_model)
    """
    from ..config.data_config import DATASET_CONFIG
    from ..data.data_utils import read_pkl_data

    item_feature_list = model_config.get("item_feature_list", ["movie_id", "genre_id"])
    emb_dim = int(model_config.get("emb_dim", 16))
    l2_reg = float(model_config.get("l2_reg", 1e-5))
    use_attention = model_config.get("use_attention", True)
    dataset_name = model_config.get("dataset_name")    

    if dataset_name is None:
        raise ValueError("EGES requires 'dataset_name' in model_config")

    dataset_cfg = dict(DATASET_CONFIG.get(dataset_name, {}))

    feature_dict = read_pkl_data(dataset_cfg["dict_path"])

    # 推断词表大小
    item_vocab_size = feature_dict.get("movie_id") or feature_dict.get("movieId") or 0
    if not isinstance(item_vocab_size, int):
        # 兼容可能的dict形式
        item_vocab_size = len(item_vocab_size) + 1
    genre_vocab_size = feature_dict.get("genres", 0)
    if not isinstance(genre_vocab_size, int):
        genre_vocab_size = len(genre_vocab_size) + 1

    # 主模型（输入: movie_id / context_id / genre_id，输出: sigmoid 概率 [B,1]）
    model = EGESModel(
        item_feature_list=item_feature_list,
        item_vocab_size=item_vocab_size,
        genre_vocab_size=genre_vocab_size,
        emb_dim=emb_dim,
        l2_reg=l2_reg,
        use_attention=use_attention,
    )

    # 物品侧嵌入模型（与主模型共享参数，评估时通过 main_model.item_tower 使用）
    item_model = model.item_tower

    return model, None, item_model
