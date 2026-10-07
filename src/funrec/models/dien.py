"""
DIEN (深度兴趣演化网络) 排序模型实现，用于 FunRec。

在 funrec 中自包含。构建与统一训练/评估管道兼容的单一排序模型（返回 (model, None, None)）。

两个关键层：
1) InterestExtractorLayer: 对行为序列使用 GRU，可选辅助损失
2) InterestEvolutionLayer: 双线性注意力 + AIGRU/AGRU/AUGRU
"""

import torch
import torch.nn as nn

from .base import FunRecModel, Layer, Dense, init_tensor_
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
    get_linear_logits,
    add_tensor_func,
    concat_func,
    parse_dien_feature_columns as _parse_dien_feature_columns,
)
from .layers import DNNs


def _shuffle_batch(x):
    """沿 batch 维随机打乱（对应原 TensorFlow 实现中的 random.shuffle）"""
    return x[torch.randperm(x.shape[0], device=x.device)]


class KerasGRU(Layer):
    """与 Keras GRU 层默认参数一致的 GRU 层（惰性构建）

    - activation=tanh, recurrent_activation=sigmoid, reset_after=True（与 torch.nn.GRU 的计算方式一致）
    - kernel: glorot_uniform, recurrent_kernel: orthogonal, bias: zeros（Keras 默认初始化）
    - 输入为 batch_first 的 [batch_size, seq_len, input_dim]，不使用掩码（与原实现中实际收到的 mask=None 一致）

    参数在内部以 torch.nn.GRU 保存。Keras 门顺序为 (z, r, h)，torch 为 (r, z, n)，
    可通过 load_keras_weights 从 Keras 权重（kernel [in, 3u], recurrent_kernel [u, 3u], bias [2, 3u]）加载。

    参数:
        units (int): 隐藏单元数
        return_sequences (bool): 是否返回所有时间步的隐藏状态，否则只返回最后一个时间步
    """

    def __init__(self, units, return_sequences=False, name=None, **kwargs):
        super(KerasGRU, self).__init__(name=name)
        self.units = units
        self.return_sequences = return_sequences

    @staticmethod
    def _keras_to_torch_gates(w, units):
        # Keras 列顺序 (z, r, h) -> torch 行顺序 (r, z, n)
        z, r, h = w[..., :units], w[..., units : 2 * units], w[..., 2 * units :]
        return torch.cat([r, z, h], dim=-1)

    def build(self, input_shape):
        in_dim = input_shape[-1]
        u = self.units
        self.gru = nn.GRU(in_dim, u, num_layers=1, bias=True, batch_first=True).to(self._build_device)
        kernel = init_tensor_(torch.empty(in_dim, 3 * u), "glorot_uniform")
        recurrent_kernel = torch.empty(u, 3 * u)
        nn.init.orthogonal_(recurrent_kernel)
        bias = torch.zeros(2, 3 * u)
        self.load_keras_weights(kernel, recurrent_kernel, bias)

    @torch.no_grad()
    def load_keras_weights(self, kernel, recurrent_kernel, bias):
        """从 Keras GRU 权重加载参数"""
        u = self.units
        dev = self.gru.weight_ih_l0.device
        kernel, recurrent_kernel, bias = (torch.as_tensor(t, dtype=torch.float32) for t in (kernel, recurrent_kernel, bias))
        self.gru.weight_ih_l0.copy_(self._keras_to_torch_gates(kernel, u).t().to(dev))
        self.gru.weight_hh_l0.copy_(self._keras_to_torch_gates(recurrent_kernel, u).t().to(dev))
        self.gru.bias_ih_l0.copy_(self._keras_to_torch_gates(bias[0], u).to(dev))
        self.gru.bias_hh_l0.copy_(self._keras_to_torch_gates(bias[1], u).to(dev))

    def forward(self, inputs):
        outputs, last_state = self.gru(inputs)
        if self.return_sequences:
            return outputs  # [batch_size, seq_len, units]
        return last_state[0]  # [batch_size, units]


class InterestExtractorLayer(Layer):
    """
    DIEN 模型的兴趣提取层。

    该层使用带辅助损失的 GRU 从行为序列中提取用户兴趣。
    辅助损失通过使用下一个行为来监督当前兴趣状态，帮助 GRU 隐藏状态更好地表示用户兴趣。

    参数:
        hidden_units (int): GRU 中的隐藏单元数 (默认: 128)
        use_auxiliary_loss (bool): 是否使用辅助损失 (默认: True)
        auxiliary_loss_weight (float): 辅助损失的权重 (默认: 0.1)
        dropout_rate (float): 正则化的 dropout 率 (默认: 0.0)

    注: 训练模式下计算得到的（已乘以权重的）辅助损失保存在 self.aux_loss 中，
    由外层模型通过 FunRecModel.add_loss 加入总损失（对应原 Keras 层内的 self.add_loss）。
    """

    def __init__(
        self,
        hidden_units=128,
        use_auxiliary_loss=True,
        auxiliary_loss_weight=0.1,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(InterestExtractorLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.use_auxiliary_loss = use_auxiliary_loss
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.dropout_rate = dropout_rate
        self.aux_loss = None

        # 兴趣提取 GRU
        self.interest_gru = KerasGRU(
            units=hidden_units,
            return_sequences=True,
            name="interest_extractor_gru",
        )

        # 辅助损失组件
        if self.use_auxiliary_loss:
            # 辅助损失预测的 MLP
            self.auxiliary_mlp = nn.Sequential(
                Dense(64, activation="relu", name="aux_dense_1"),
                nn.Dropout(dropout_rate),
                Dense(32, activation="relu", name="aux_dense_2"),
                nn.Dropout(dropout_rate),
                Dense(1, activation="sigmoid", name="aux_output"),
            )
        else:
            self.auxiliary_mlp = None

        # Dropout 层
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def build(self, input_shape):
        """构建方法，如果需要则初始化辅助 MLP。"""
        # 如果存在辅助 MLP 则构建它（即使推理时不使用，其参数也与原实现一样在构建时创建）
        if self.auxiliary_mlp is not None:
            # 辅助 MLP 输入形状: [batch_size, seq_len-1, hidden_units + embedding_dim]
            # embedding_dim 从 input_shape 确定
            if isinstance(input_shape, list) and len(input_shape) > 0:
                embedding_dim = input_shape[0][-1]  # 从行为嵌入获取嵌入维度
            else:
                embedding_dim = 8  # 默认嵌入维度

            aux_input_dim = self.hidden_units + embedding_dim
            # 用一个占位输入前向一次，触发惰性 Dense 的参数创建
            with torch.no_grad():
                self.auxiliary_mlp(torch.zeros(1, aux_input_dim, device=self._build_device))

    def forward(self, inputs, mask=None):
        """
        兴趣提取层的前向传播。

        参数:
            inputs: 包含 [behavior_embeddings, neg_behavior_embeddings (可选)] 的列表
                - behavior_embeddings: 形状为 [batch_size, seq_len, embedding_dim] 的张量
                - neg_behavior_embeddings: 形状为 [batch_size, seq_len, embedding_dim] 的张量 (用于辅助损失)
            mask: 序列的填充掩码（原实现中行为序列与 sparse 特征共享嵌入表，实际收到的 mask 为 None）

        返回:
            interest_states: 形状为 [batch_size, seq_len, hidden_units] 的张量
                表示每个时间步用户兴趣的隐藏状态
        """
        self.aux_loss = None
        behavior_embeddings = inputs[0]  # 形状: [batch_size, seq_len, embedding_dim]

        # 使用 GRU 提取兴趣
        # 形状: [batch_size, seq_len, hidden_units]
        # 注: 与原实现一致，GRU 不使用掩码（原实现中 mask 为 None）
        interest_states = self.interest_gru(behavior_embeddings)

        # 如果启用则应用 dropout
        if self.dropout is not None:
            interest_states = self.dropout(interest_states)

        # 如果启用且在训练模式下计算辅助损失
        if (
            self.use_auxiliary_loss
            and self.training
            and len(inputs) > 1
            and self.auxiliary_mlp is not None
        ):
            neg_behavior_embeddings = inputs[
                1
            ]  # 形状: [batch_size, seq_len, embedding_dim]
            aux_loss = self._compute_auxiliary_loss(
                interest_states, behavior_embeddings, neg_behavior_embeddings, mask
            )
            # 将辅助损失保存，由模型加入总损失
            self.aux_loss = self.auxiliary_loss_weight * aux_loss

        return interest_states

    def _compute_auxiliary_loss(
        self, interest_states, pos_behaviors, neg_behaviors, mask
    ):
        """
        计算兴趣提取的辅助损失。

        辅助损失使用下一个行为来监督当前兴趣状态：
        - 正样本: 当前兴趣应该预测下一个正行为
        - 负样本: 当前兴趣不应该预测下一个负行为

        参数:
            interest_states: 形状为 [batch_size, seq_len, hidden_units] 的张量
            pos_behaviors: 形状为 [batch_size, seq_len, embedding_dim] 的张量
            neg_behaviors: 形状为 [batch_size, seq_len, embedding_dim] 的张量
            mask: 填充掩码

        返回:
            aux_loss: 标量辅助损失
        """
        # 获取兴趣和下一个行为（兴趣排除最后一个时间步）
        # 形状: [batch_size, seq_len-1, hidden_units]
        current_interests = interest_states[:, :-1, :]
        # 形状: [batch_size, seq_len-1, embedding_dim]
        next_pos_behaviors = pos_behaviors[:, 1:, :]
        next_neg_behaviors = neg_behaviors[:, 1:, :]

        # 连接兴趣和行为嵌入作为 MLP 输入
        # 形状: [batch_size, seq_len-1, hidden_units + embedding_dim]
        pos_input = torch.cat([current_interests, next_pos_behaviors], dim=-1)
        neg_input = torch.cat([current_interests, next_neg_behaviors], dim=-1)

        # 预测概率
        # 形状: [batch_size, seq_len-1, 1]
        pos_probs = self.auxiliary_mlp(pos_input)
        neg_probs = self.auxiliary_mlp(neg_input)

        # 计算二元交叉熵损失
        pos_loss = -torch.log(pos_probs + 1e-8)  # 形状: [batch_size, seq_len-1, 1]
        neg_loss = -torch.log(
            1 - neg_probs + 1e-8
        )  # 形状: [batch_size, seq_len-1, 1]

        # 应用掩码（排除填充位置）
        if mask is not None:
            # 为 seq_len-1 调整掩码
            loss_mask = mask[:, 1:]  # 形状: [batch_size, seq_len-1]
            loss_mask = loss_mask.unsqueeze(-1)  # 形状: [batch_size, seq_len-1, 1]
            loss_mask = loss_mask.to(torch.float32)

            pos_loss = pos_loss * loss_mask
            neg_loss = neg_loss * loss_mask

        # 平均损失
        aux_loss = torch.mean(pos_loss + neg_loss)

        return aux_loss


class InterestEvolutionLayer(Layer):
    """
    DIEN 模型的兴趣演化层。

    该层使用双线性注意力结合 GRU 变体来建模用户兴趣随时间的演化。支持三种演化类型：
    1. AIGRU: 基于注意力的输入 GRU
    2. AGRU: 基于注意力的 GRU
    3. AUGRU: 基于注意力的更新 GRU（推荐）

    注意力机制遵循原始 DIEN 论文使用双线性形式：
    a_t = exp(h_t * W * e_a) / sum(exp(h_j * W * e_a))

    参数:
        hidden_units (int): 演化 GRU 中的隐藏单元数 (默认: 128)
        evolution_type (str): 演化机制类型 ('AIGRU', 'AGRU', 'AUGRU') (默认: 'AUGRU')
        dropout_rate (float): Dropout 率 (默认: 0.0)
    """

    def __init__(
        self, hidden_units=128, evolution_type="AUGRU", dropout_rate=0.0, **kwargs
    ):
        super(InterestEvolutionLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.evolution_type = evolution_type
        self.dropout_rate = dropout_rate

        # 验证演化类型
        if evolution_type not in ["AIGRU", "AGRU", "AUGRU"]:
            raise ValueError(
                f"evolution_type 必须是 ['AIGRU', 'AGRU', 'AUGRU'] 之一，得到 {evolution_type}"
            )

        # 如原始 DIEN 论文中的双线性注意力机制
        # 注意力分数: h_t * W * e_a，其中 W 是双线性权重矩阵
        self.bilinear_weight = None  # 将在 build 方法中初始化

        # 演化 GRU（基于 evolution_type 的不同实现）
        if evolution_type == "AIGRU":
            self.evolution_gru = KerasGRU(
                units=hidden_units,
                return_sequences=False,
                name="evolution_gru",
            )
        elif evolution_type == "AGRU":
            # 自定义 AGRU 实现
            self.evolution_gru = self._build_agru()
        elif evolution_type == "AUGRU":
            # 自定义 AUGRU 实现
            self.evolution_gru = self._build_augru()

        # Dropout 层
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def build(self, input_shape):
        """构建方法，初始化双线性权重矩阵。"""
        # 初始化注意力计算的双线性权重矩阵
        # 形状: [hidden_units, embedding_dim]
        # 这将用于: h_t * W * e_a
        if isinstance(input_shape, list) and len(input_shape) > 1:
            # input_shape[0] 是 interest_states 形状，input_shape[1] 是 target_item_embedding 形状
            embedding_dim = (
                input_shape[1][-1] if len(input_shape[1]) > 1 else input_shape[1][0]
            )
        else:
            embedding_dim = 8  # 默认嵌入维度

        self.bilinear_weight = self.add_weight(
            name="bilinear_attention_weight",
            shape=(self.hidden_units, embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def _build_agru(self):
        """构建自定义 AGRU 单元。"""
        return AGRULayer(self.hidden_units, name="agru_evolution")

    def _build_augru(self):
        """构建自定义 AUGRU 单元。"""
        return AUGRULayer(self.hidden_units, name="augru_evolution")

    def forward(self, inputs, mask=None):
        """
        兴趣演化层的前向传播。

        参数:
            inputs: 包含 [interest_states, target_item_embedding] 的列表
                - interest_states: 形状为 [batch_size, seq_len, hidden_units] 的张量
                - target_item_embedding: 形状为 [batch_size, embedding_dim] 的张量
            mask: 序列的填充掩码（原实现中实际收到的 mask 为 None，演化 GRU 不使用掩码）

        返回:
            final_interest: 形状为 [batch_size, hidden_units] 的张量
                最终演化的兴趣表示
        """
        interest_states = inputs[0]  # 形状: [batch_size, seq_len, hidden_units]
        target_item_embedding = inputs[1]  # 形状: [batch_size, embedding_dim]

        # 计算每个兴趣状态与目标物品之间的注意力分数
        # 形状: [batch_size, seq_len, 1]
        attention_scores = self._compute_attention_scores(
            interest_states, target_item_embedding
        )

        # 基于演化类型应用注意力机制
        if self.evolution_type == "AIGRU":
            # 基于注意力的输入 GRU: 将注意力分数与输入相乘
            # 形状: [batch_size, seq_len, hidden_units]
            attended_interests = interest_states * attention_scores
            # 形状: [batch_size, hidden_units]
            final_interest = self.evolution_gru(attended_interests)

        elif self.evolution_type in ["AGRU", "AUGRU"]:
            # 使用内部集成注意力的自定义 GRU 实现
            # 形状: [batch_size, hidden_units]
            final_interest = self.evolution_gru(
                [interest_states, attention_scores], mask=mask
            )

        # 如果启用则应用 dropout
        if self.dropout is not None:
            final_interest = self.dropout(final_interest)

        return final_interest

    def _compute_attention_scores(self, interest_states, target_item_embedding):
        """
        使用双线性形式计算兴趣状态与目标物品之间的注意力分数。

        遵循原始 DIEN 论文: a_t = exp(h_t * W * e_a) / sum(exp(h_j * W * e_a))

        参数:
            interest_states: 形状为 [batch_size, seq_len, hidden_units] 的张量
            target_item_embedding: 形状为 [batch_size, 1, embedding_dim] 或 [batch_size, embedding_dim] 的张量

        返回:
            attention_scores: 形状为 [batch_size, seq_len, 1] 的张量
        """
        # 确保 target_item_embedding 具有正确的形状 [batch_size, embedding_dim]
        if target_item_embedding.dim() == 3:
            # 形状: [batch_size, 1, embedding_dim] -> [batch_size, embedding_dim]
            target_item_embedding = target_item_embedding.squeeze(1)

        # 双线性注意力计算: h_t * W * e_a
        # 步骤 1: h_t * W -> [batch_size, seq_len, embedding_dim]
        h_W = torch.matmul(interest_states, self.bilinear_weight)

        # 步骤 2: (h_t * W) * e_a -> [batch_size, seq_len]
        # 将目标嵌入扩展为 [batch_size, 1, embedding_dim] 以进行广播
        target_expanded = target_item_embedding.unsqueeze(1)

        # 逐元素乘法并在嵌入维度上求和
        # 形状: [batch_size, seq_len, embedding_dim] * [batch_size, 1, embedding_dim] -> [batch_size, seq_len, embedding_dim]
        attention_scores = torch.sum(h_W * target_expanded, dim=2)  # [batch_size, seq_len]

        # 应用 softmax 来归一化注意力分数
        # 形状: [batch_size, seq_len]
        attention_scores = torch.softmax(attention_scores, dim=1)

        # 扩展为 [batch_size, seq_len, 1] 以与模型其余部分保持一致
        attention_scores = attention_scores.unsqueeze(2)

        return attention_scores


class AGRULayer(Layer):
    """
    基于注意力的 GRU 层。

    在 AGRU 中，注意力分数直接替换更新门值。
    这是一种简化的方法，但可能会失去一些表示能力。

    参数:
        units (int): 隐藏单元数
    """

    def __init__(self, units, **kwargs):
        super(AGRULayer, self).__init__(**kwargs)
        self.units = units

        # GRU 单元组件（在 init 中构建它们）
        self.dense_input_reset = Dense(units, name="input_reset")
        self.dense_hidden_reset = Dense(units, name="hidden_reset")
        self.dense_input_candidate = Dense(units, name="input_candidate")
        self.dense_hidden_candidate = Dense(units, name="hidden_candidate")

    def forward(self, inputs, mask=None):
        """
        AGRU 的前向传播。

        参数:
            inputs: [interest_states, attention_scores] 的列表
                - interest_states: 形状为 [batch_size, seq_len, units] 的张量
                - attention_scores: 形状为 [batch_size, seq_len, 1] 的张量
            mask: 填充掩码（未使用，与原实现一致）

        返回:
            final_state: 形状为 [batch_size, units] 的张量
        """
        interest_states, attention_scores = inputs
        batch_size = interest_states.shape[0]
        seq_len = interest_states.shape[1]

        # 初始化隐藏状态
        # 形状: [batch_size, units]
        hidden_state = torch.zeros(
            batch_size, self.units, device=interest_states.device, dtype=interest_states.dtype
        )

        # 逐步处理序列
        for t in range(seq_len):
            current_input = interest_states[:, t, :]  # 形状: [batch_size, units]
            current_attention = attention_scores[:, t, 0]  # 形状: [batch_size]

            # 使用注意力作为更新门的自定义 GRU 步骤
            # 使用标准 GRU 计算重置门和候选状态
            reset_gate = torch.sigmoid(
                self.dense_input_reset(current_input)
                + self.dense_hidden_reset(hidden_state)
            )

            candidate_state = torch.tanh(
                self.dense_input_candidate(current_input)
                + self.dense_hidden_candidate(reset_gate * hidden_state)
            )

            # 使用注意力分数作为更新门
            # 形状: [batch_size, units]
            update_gate = current_attention.unsqueeze(1).expand(-1, self.units)

            # 更新隐藏状态
            hidden_state = (
                1 - update_gate
            ) * hidden_state + update_gate * candidate_state

        return hidden_state


class AUGRULayer(Layer):
    """
    基于注意力的更新 GRU 层。

    AUGRU 保持更新门的多维性质，同时通过注意力分数对其进行缩放。
    这是 DIEN 中推荐的方法。

    参数:
        units (int): 隐藏单元数
    """

    def __init__(self, units, **kwargs):
        super(AUGRULayer, self).__init__(**kwargs)
        self.units = units

        # GRU 单元权重
        self.dense_input_update = Dense(units, name="input_update")
        self.dense_hidden_update = Dense(units, name="hidden_update")
        self.dense_input_reset = Dense(units, name="input_reset")
        self.dense_hidden_reset = Dense(units, name="hidden_reset")
        self.dense_input_candidate = Dense(units, name="input_candidate")
        self.dense_hidden_candidate = Dense(units, name="hidden_candidate")

    def forward(self, inputs, mask=None):
        """
        AUGRU 的前向传播。

        参数:
            inputs: [interest_states, attention_scores] 的列表
                - interest_states: 形状为 [batch_size, seq_len, units] 的张量
                - attention_scores: 形状为 [batch_size, seq_len, 1] 的张量
            mask: 填充掩码（未使用，与原实现一致）

        返回:
            final_state: 形状为 [batch_size, units] 的张量
        """
        interest_states, attention_scores = inputs
        batch_size = interest_states.shape[0]
        seq_len = interest_states.shape[1]

        # 初始化隐藏状态
        # 形状: [batch_size, units]
        hidden_state = torch.zeros(
            batch_size, self.units, device=interest_states.device, dtype=interest_states.dtype
        )

        # 逐步处理序列
        for t in range(seq_len):
            current_input = interest_states[:, t, :]  # 形状: [batch_size, units]
            current_attention = attention_scores[:, t, 0]  # 形状: [batch_size]

            # 标准 GRU 计算
            # 更新门: 形状 [batch_size, units]
            update_gate = torch.sigmoid(
                self.dense_input_update(current_input)
                + self.dense_hidden_update(hidden_state)
            )

            # 重置门: 形状 [batch_size, units]
            reset_gate = torch.sigmoid(
                self.dense_input_reset(current_input)
                + self.dense_hidden_reset(hidden_state)
            )

            # 候选状态: 形状 [batch_size, units]
            candidate_state = torch.tanh(
                self.dense_input_candidate(current_input)
                + self.dense_hidden_candidate(reset_gate * hidden_state)
            )

            # 基于注意力的更新门缩放
            # 形状: [batch_size, units]
            attention_expanded = current_attention.unsqueeze(1).expand(-1, self.units)
            attention_update_gate = attention_expanded * update_gate

            # 使用注意力缩放的更新门更新隐藏状态
            hidden_state = (
                1 - attention_update_gate
            ) * hidden_state + attention_update_gate * candidate_state

        return hidden_state


class DIENModel(FunRecModel):
    """DIEN (深度兴趣演化网络) 排序模型的 PyTorch 实现"""

    def __init__(self, feature_columns, model_config):
        # 解析配置
        dnn_units = model_config.get("dnn_units", [200, 80, 1])
        interest_hidden_units = model_config.get("interest_hidden_units", 64)
        evolution_type = model_config.get("evolution_type", "AUGRU")
        use_auxiliary_loss = model_config.get("use_auxiliary_loss", True)
        auxiliary_loss_weight = model_config.get("auxiliary_loss_weight", 0.1)
        dropout_rate = model_config.get("dropout_rate", 0.1)
        use_linear_logits = model_config.get("linear_logits", True)

        # 构建输入层
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="dien")

        self.use_auxiliary_loss = use_auxiliary_loss
        self.use_linear_logits = use_linear_logits

        # 构建嵌入层
        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")

        # 处理 DIEN 序列特征
        self.dien_feature_list = _parse_dien_feature_columns(feature_columns)

        if len(self.dien_feature_list) == 0:
            raise ValueError(
                "未找到 DIEN 序列特征。请在 combiner 中添加包含 'dien' 的序列特征。"
            )

        # 每个 DIEN 特征对（query, key）的兴趣提取层与兴趣演化层
        self.interest_extractors = nn.ModuleDict()
        self.interest_evolutions = nn.ModuleDict()
        for target_feature, sequence_feature in self.dien_feature_list:
            # 兴趣提取层
            self.interest_extractors[sequence_feature] = InterestExtractorLayer(
                hidden_units=interest_hidden_units,
                use_auxiliary_loss=use_auxiliary_loss,
                auxiliary_loss_weight=auxiliary_loss_weight,
                dropout_rate=dropout_rate,
                name=f"{sequence_feature}_interest_extractor",
            )
            # 兴趣演化层
            self.interest_evolutions[sequence_feature] = InterestEvolutionLayer(
                hidden_units=interest_hidden_units,
                evolution_type=evolution_type,
                dropout_rate=dropout_rate,
                name=f"{sequence_feature}_interest_evolution",
            )

        # 最终 DNN 层
        self.dnn = DNNs(dnn_units, use_bn=True, dropout_rate=dropout_rate)

        # 可选线性项
        if use_linear_logits:
            self.linear_logits = get_linear_logits(feature_columns)

        self.dien_output = Dense(1, activation="sigmoid", name="dien_output")

    def forward(self, inputs):
        # 构建嵌入
        group_embedding_feature_dict = self.embedding(inputs)

        # 获取 DNN 输入（非序列特征）
        dnn_inputs = concat_group_embedding(group_embedding_feature_dict, "dnn")

        # 处理每个 DIEN 特征对（query, key）
        dien_outputs = []
        for target_feature, sequence_feature in self.dien_feature_list:
            # 获取目标物品嵌入（query）
            # 形状: [batch_size, 1, embedding_dim]
            target_embedding = group_embedding_feature_dict["dien_sequence"][target_feature]

            # 获取行为序列嵌入（keys）
            # 形状: [batch_size, seq_len, embedding_dim]
            # 注: 行为序列与 sparse 特征共享嵌入表（mask_zero=False），原实现中没有掩码传入后续层
            behavior_embeddings = group_embedding_feature_dict["dien_sequence"][
                sequence_feature
            ]

            # 为辅助损失生成负样本（简化方法）
            if self.use_auxiliary_loss:
                # 通过打乱行为嵌入创建负样本（沿 batch 维打乱）
                # 形状: [batch_size, seq_len, embedding_dim]
                # 仅在训练时需要（推理时辅助损失不计算，打乱结果不会被使用）
                # 原实现的 random.shuffle 不可导（梯度为 None），负样本路径不向嵌入表回传梯度，故 detach
                if self.training:
                    neg_behavior_embeddings = _shuffle_batch(behavior_embeddings).detach()
                    interest_extractor_inputs = [behavior_embeddings, neg_behavior_embeddings]
                else:
                    interest_extractor_inputs = [behavior_embeddings, behavior_embeddings]
            else:
                interest_extractor_inputs = [behavior_embeddings]

            # 兴趣提取层
            # 从行为序列中提取用户兴趣
            # 形状: [batch_size, seq_len, interest_hidden_units]
            interest_extractor = self.interest_extractors[sequence_feature]
            interest_states = interest_extractor(interest_extractor_inputs)
            if interest_extractor.aux_loss is not None:
                self.add_loss(interest_extractor.aux_loss)

            # 兴趣演化层
            # 基于目标物品使用双线性注意力建模兴趣演化
            # 形状: [batch_size, interest_hidden_units]
            evolved_interest = self.interest_evolutions[sequence_feature](
                [interest_states, target_embedding]
            )

            dien_outputs.append(evolved_interest)

        # 连接所有 DIEN 输出
        if len(dien_outputs) > 1:
            dien_output = concat_func(dien_outputs, axis=1, flatten=True)
        else:
            dien_output = dien_outputs[0]

        # 与其他 DNN 输入结合
        # 形状: [batch_size, dnn_dim + interest_hidden_units]
        final_dnn_inputs = concat_func([dnn_inputs, dien_output], axis=-1)

        # 最终 DNN 层
        dnn_logits = self.dnn(final_dnn_inputs)

        # 可选线性项
        if self.use_linear_logits:
            linear_logit = self.linear_logits(inputs)
            dnn_logits = add_tensor_func(
                [dnn_logits, linear_logit], name="dien_linear_logits"
            )

        # 输出: 确保与排序评估器兼容的 rank-1 预测
        final_logits = torch.flatten(dnn_logits, start_dim=1)
        output = self.dien_output(final_logits)
        output = torch.flatten(output, start_dim=1)
        return output


def build_dien_model(feature_columns, model_config):
    """
    构建 DIEN (深度兴趣演化网络) 模型。

    DIEN 通过两个关键组件捕获用户兴趣演化：
    1. 兴趣提取层: 从行为序列中提取兴趣
    2. 兴趣演化层: 使用双线性注意力建模兴趣随时间的演化

    参数:
        feature_columns: 特征列规范列表
        dnn_units: 最终 DNN 层的隐藏单元列表 (默认: [256, 128, 64, 1])
        interest_hidden_units: 兴趣提取的隐藏单元 (默认: 128)
        evolution_type: 演化机制类型 ('AIGRU', 'AGRU', 'AUGRU') (默认: 'AUGRU')
        use_auxiliary_loss: 是否使用辅助损失 (默认: True)
        auxiliary_loss_weight: 辅助损失的权重 (默认: 0.1)
        dropout_rate: 正则化的 dropout 率 (默认: 0.0)
        linear_logits: 是否包含线性 logits (默认: True)

    返回:
        (model, None, None): 统一接口的排序模型元组
    """
    model = DIENModel(feature_columns, model_config)
    return model, None, None
