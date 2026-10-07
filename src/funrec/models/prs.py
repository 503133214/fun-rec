import numpy as np
import torch
import torch.nn as nn

from .base import FunRecModel, SubModel, Layer, Dense
from .utils import (
    build_input_layer,
    FeatureEmbedding,
    concat_group_embedding,
)


def calculate_estimated_reward(ordering, ctr_scores, next_scores, alpha=0.5, beta=0.5):
    """
    根据PRS FPSA算法计算给定排序的估计奖励。

    Args:
        ordering: 物品ID的元组/列表
        ctr_scores: 物品到CTR分数的映射字典
        next_scores: 物品到Next分数的映射字典
        alpha, beta: 融合系数

    Returns:
        float 估计奖励
    """
    if not ordering:
        return 0.0

    r_ipv = 0.0
    p_expose = 1.0

    for item in ordering:
        p_ctr = ctr_scores[item]
        p_next = next_scores[item]
        r_ipv += p_expose * p_ctr
        p_expose *= p_next

    r_pv = p_expose
    return alpha * r_pv + beta * r_ipv


def fpsa_algorithm(
    items, ctr_scores, next_scores, beam_size=5, max_length=10, alpha=0.5, beta=0.5
):
    """
    快速排列搜索算法（FPSA），基于PRS论文。

    Args:
        items: 物品的可迭代对象
        ctr_scores: 物品到分数的映射字典
        next_scores: 物品到分数的映射字典
        beam_size: 束搜索宽度
        max_length: 输出长度
        alpha, beta: 融合系数

    Returns:
        候选排列的列表（作为元组）
    """
    candidates = [()]  # 从空排序开始
    for _ in range(1, max_length + 1):
        new_candidates = []
        rewards = {}
        for ordering in candidates:
            used = set(ordering)
            for ci in items:
                if ci in used:
                    continue
                new_order = ordering + (ci,)
                r = calculate_estimated_reward(
                    new_order, ctr_scores, next_scores, alpha, beta
                )
                rewards[new_order] = r
                new_candidates.append(new_order)
        # 按奖励进行剪枝
        new_candidates.sort(key=lambda x: rewards[x], reverse=True)
        candidates = new_candidates[:beam_size]
    return candidates


class MaskedLSTM(Layer):
    """与 Keras LSTM(return_sequences=True) 数值等价的 LSTM 层（支持序列掩码）

    - 参数布局与 Keras 一致: kernel [D, 4u]（glorot_uniform）、recurrent_kernel [u, 4u]（orthogonal）、
      bias [4u]（零初始化，遗忘门部分为 1，即 unit_forget_bias=True），门顺序为 i, f, c, o
    - mask: B x L 的 bool 张量。与 Keras 一致，被掩码的时间步不更新隐藏状态和细胞状态（沿用上一步），
      且该时间步输出为 0（Bidirectional 包装的 return_sequences=True 层 zero_output_for_mask=True）
    - go_backwards=True 时逆序处理输入，输出序列也为逆序（与 Keras 一致）
    """

    def __init__(self, units, go_backwards=False, name=None, **kwargs):
        super().__init__(name=name)
        self.units = int(units)
        self.go_backwards = go_backwards

    def build(self, input_shape):
        in_dim = input_shape[-1]
        u = self.units
        self.kernel = self.add_weight("kernel", (in_dim, 4 * u), "glorot_uniform")
        self.recurrent_kernel = self.add_weight(
            "recurrent_kernel", (u, 4 * u), lambda t: nn.init.orthogonal_(t)
        )
        self.bias = self.add_weight("bias", (4 * u,), "zeros")
        with torch.no_grad():
            self.bias[u : 2 * u].fill_(1.0)

    def forward(self, inputs, mask=None):
        # inputs: [B, T, D] -> [B, T, units]
        batch_size, steps = inputs.shape[0], inputs.shape[1]
        u = self.units
        h = inputs.new_zeros(batch_size, u)
        c = inputs.new_zeros(batch_size, u)
        # 输入投影可一次性完成
        x_proj = torch.matmul(inputs, self.kernel) + self.bias  # [B, T, 4u]

        time_steps = range(steps - 1, -1, -1) if self.go_backwards else range(steps)
        outputs = []
        for t in time_steps:
            z = x_proj[:, t, :] + torch.matmul(h, self.recurrent_kernel)
            z_i, z_f, z_c, z_o = torch.split(z, u, dim=-1)
            i = torch.sigmoid(z_i)
            f = torch.sigmoid(z_f)
            c_new = f * c + i * torch.tanh(z_c)
            o = torch.sigmoid(z_o)
            h_new = o * torch.tanh(c_new)
            if mask is not None:
                m = mask[:, t].unsqueeze(-1)
                h = torch.where(m, h_new, h)
                c = torch.where(m, c_new, c)
                outputs.append(torch.where(m, h_new, torch.zeros_like(h_new)))
            else:
                h, c = h_new, c_new
                outputs.append(h_new)
        return torch.stack(outputs, dim=1)


class MaskedBidirectionalLSTM(nn.Module):
    """对应 Keras Bidirectional(LSTM(units, return_sequences=True))

    后向层（go_backwards=True）输出在时间维度上翻转回正序后与前向输出拼接（merge_mode='concat'）。
    """

    def __init__(self, units, name=None):
        super().__init__()
        self.layer_name = name
        self.forward_layer = MaskedLSTM(units, name="forward_lstm")
        self.backward_layer = MaskedLSTM(units, go_backwards=True, name="backward_lstm")

    def forward(self, inputs, mask=None):
        y_fwd = self.forward_layer(inputs, mask=mask)
        y_bwd = torch.flip(self.backward_layer(inputs, mask=mask), dims=[1])
        return torch.cat([y_fwd, y_bwd], dim=-1)


class TimeDistributedMLPHead(nn.Module):
    """TimeDistributed MLP 头: Dense(256, relu) -> Dropout -> Dense(128, relu) -> Dropout -> Dense(1, sigmoid)

    Dense 作用于最后一维，等价于 Keras 的 TimeDistributed(Sequential([...]))
    """

    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.dense_0 = Dense(256, activation="relu")
        self.dropout_0 = nn.Dropout(dropout_rate)
        self.dense_1 = Dense(128, activation="relu")
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dense_2 = Dense(1, activation="sigmoid")

    def forward(self, x):
        x = self.dropout_0(self.dense_0(x))
        x = self.dropout_1(self.dense_1(x))
        return self.dense_2(x)


def _slice_single_example(features_dict, idx):
    single = {}
    for k, v in features_dict.items():
        # 支持列表和np.ndarray输入
        arr = v
        single[k] = arr[idx : idx + 1]
    return single


def _reorder_sequence_feature(arr, order):
    # arr形状：[1, L, ...] 或 [1, L]
    if arr.ndim == 3:
        return arr[:, order, :]
    elif arr.ndim == 2:
        return arr[:, order]
    else:
        return arr


def _prepare_permuted_batch(single_features, candidates, seq_keys):
    # 通过堆叠单个示例的排列副本来构建批次字典
    batch = {}
    num_cand = len(candidates)
    for k, v in single_features.items():
        if k in seq_keys:
            stacked = []
            for perm in candidates:
                stacked.append(_reorder_sequence_feature(v, list(perm)))
            batch[k] = np.concatenate(stacked, axis=0)
        else:
            # 沿批次维度平铺非序列特征
            batch[k] = np.repeat(v, repeats=num_cand, axis=0)
    return batch


class PRSModel(FunRecModel):
    """PRS DPWN 风格重排序模型的 PyTorch 实现

    输出: [ctr_probs, next_probs]，形状均为 [B, max_seq_len]
    """

    def __init__(self, feature_columns, max_seq_len=30, hidden_dim=128, dropout_rate=0.2):
        # 使用现有工具构建输入和嵌入
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="prs")
        self.max_seq_len = max_seq_len
        self.has_pv_emb = "pv_emb" in input_layer_dict
        self.has_item_emb = "item_emb" in input_layer_dict

        self.embedding = FeatureEmbedding(feature_columns, prefix="embedding/")
        # item_part 组中 combiner=None 的变长特征名称（用于计算序列掩码）
        self.item_part_seq_features = [
            (fc.name, fc.emb_name)
            for fc in feature_columns
            if fc.type == "varlen_sparse"
            and fc.combiner is None
            and isinstance(fc.emb_name, str)
            and "item_part" in (fc.group or [])
        ]

        # 使用双向LSTM对物品属性+物品嵌入进行序列编码
        self.sequence_encoder = MaskedBidirectionalLSTM(hidden_dim, name="bidirectional")

        # TimeDistributed MLP头 -> 每个位置的分数
        self.td_ctr_head = TimeDistributedMLPHead(dropout_rate)
        self.td_next_head = TimeDistributedMLPHead(dropout_rate)

        # 收集序列特征名称以支持PRank期间的排列
        self.item_part_feature_names = []
        self.user_part_feature_names = []
        for fc in feature_columns:
            if hasattr(fc, "group") and fc.group is not None:
                if "item_part" in fc.group:
                    self.item_part_feature_names.append(fc.name)
                if "user_part" in fc.group:
                    self.user_part_feature_names.append(fc.name)

        # 密集序列特征名称（如果存在）
        self.seq_dense_feature_names = []
        if self.has_pv_emb:
            self.seq_dense_feature_names.append("pv_emb")
        if self.has_item_emb:
            self.seq_dense_feature_names.append("item_emb")

        # 为推理构建辅助模型（PMatch输入）
        self.ctr_model = SubModel(self, "ctr_forward", self.input_names, name="ctr_model")
        self.next_model = SubModel(self, "next_forward", self.input_names, name="next_model")

    def _sequence_mask(self, inputs):
        """序列掩码: 对应 Keras 中 mask_zero=True 的嵌入经 Concatenate 合并后的掩码

        Keras Concatenate 对没有掩码的输入视为全 1，最终在最后一维上取逻辑与，
        因此掩码为所有（嵌入表 mask_zero=True 的）item_part 序列特征 id != 0 的逻辑与。
        """
        mask = None
        for name, emb_name in self.item_part_seq_features:
            m = self.embedding.embedding_table_dict[emb_name].compute_mask(inputs[name])
            if m is None:
                continue
            mask = m if mask is None else (mask & m)
        return mask

    def forward(self, inputs):
        group_embedding_feature_dict = self.embedding(inputs)

        # 用户特征（向量）和平铺版本
        user_vector = concat_group_embedding(
            group_embedding_feature_dict, "user_part", axis=-1, flatten=True
        )  # [B, D_user]
        user_tiled = user_vector.unsqueeze(1).repeat(1, self.max_seq_len, 1)  # [B, L, D_user]

        # 物品侧序列特征
        item_part_seq = concat_group_embedding(
            group_embedding_feature_dict, "item_part", axis=-1, flatten=False
        )  # [B, L, D_item_part]
        # 稠密输入在原 Keras 中为 float32 Input，这里显式转换
        pv_seq = inputs["pv_emb"].float() if self.has_pv_emb else None  # [B, L, D_pv]
        item_emb_seq = inputs["item_emb"].float() if self.has_item_emb else None  # [B, L, D_item]

        # 构建类似DPWN的网络
        # 使用双向LSTM对物品属性+物品嵌入进行序列编码
        # 注: 原 Keras 实现中 item_part 序列嵌入（mask_zero=True）的掩码经 Concatenate 传入 LSTM，
        # 这里显式传入相同的掩码（被掩码位置输出为 0，状态沿用上一步）
        mask = self._sequence_mask(inputs)
        seq_feature = torch.cat([item_part_seq, item_emb_seq], dim=-1)
        sequence_hidden = self.sequence_encoder(seq_feature, mask=mask)  # [B, L, 2*hidden_dim]

        # 与用户向量和pv嵌入进行特征融合
        fused = torch.cat([sequence_hidden, user_tiled, pv_seq], dim=-1)

        # DPWN点击概率（我们重用CTR头作为训练输出）
        ctr_probs_3d = self.td_ctr_head(fused)  # [B, L, 1]
        next_probs_3d = self.td_next_head(fused)  # [B, L, 1]

        ctr_probs = ctr_probs_3d.squeeze(-1)  # [B, L]
        next_probs = next_probs_3d.squeeze(-1)  # [B, L]

        # 主要训练输出：输出 ctr_probs和next_probs以确保两个头都接收梯度
        # 但在训练时我们只使用ctr_probs作为主要损失
        return [ctr_probs, next_probs]

    def ctr_forward(self, inputs):
        return self(inputs)[0]

    def next_forward(self, inputs):
        return self(inputs)[1]

    def prs_predict(self, features, eval_config=None):
        """
        运行PRS PMatch（FPSA）+ PRank（DPWN）以产生反映最佳排列的每个slate分数。
        返回形状为[B, L]的分数；降序排序产生最终排列。
        """
        item_part_feature_names = self.item_part_feature_names
        seq_dense_feature_names = self.seq_dense_feature_names
        alpha = (eval_config or {}).get("alpha", 0.5)
        beta = (eval_config or {}).get("beta", 0.5)
        beam_size = (eval_config or {}).get("beam_size", 5)
        # 从可用的密集序列确定序列长度
        if "item_emb" in features:
            L = features["item_emb"].shape[1]
        elif "pv_emb" in features:
            L = features["pv_emb"].shape[1]
        else:
            # 回退：使用第一个item_part特征
            any_item_feat = (
                item_part_feature_names[0] if item_part_feature_names else None
            )
            if any_item_feat is None:
                raise ValueError("PRS需要序列特征（item_emb/pv_emb或item_part特征）。")
            L = features[any_item_feat].shape[1]
        max_length = min((eval_config or {}).get("max_length", L), L)

        # 需要排列的键
        seq_keys = set(item_part_feature_names + seq_dense_feature_names)

        B = next(iter(features.values())).shape[0]
        output_scores = np.zeros((B, L), dtype=np.float32)
        # 子模型 predict 结束时会按子模型自身的 training 标志恢复主模型模式，这里记录并在最后恢复
        was_training = self.training

        for i in range(B):
            # 切片单个示例
            single = _slice_single_example(features, i)

            # PMatch：计算每个位置的CTR和Next
            ctr_vec = self.ctr_model.predict(single, verbose=0)[0]  # [L]
            next_vec = self.next_model.predict(single, verbose=0)[0]  # [L]
            items = list(range(L))
            ctr_scores = {idx: float(ctr_vec[idx]) for idx in items}
            next_scores = {idx: float(next_vec[idx]) for idx in items}

            candidates = fpsa_algorithm(
                items=items,
                ctr_scores=ctr_scores,
                next_scores=next_scores,
                beam_size=beam_size,
                max_length=max_length,
                alpha=alpha,
                beta=beta,
            )

            # PRank：使用DPWN点击概率（ctr头）在排列序列上评估候选
            perm_batch = _prepare_permuted_batch(single, candidates, seq_keys)
            model_output = self.predict(perm_batch, verbose=0)  # [ctr_probs, next_probs]
            # 提取CTR概率（第一个输出）
            click_probs_batch = (
                model_output[0] if isinstance(model_output, list) else model_output
            )  # [num_cand, L]
            # 列表奖励 = 每个位置点击概率的总和
            lr_vec = np.sum(click_probs_batch[:, :max_length], axis=1)
            best_idx = int(np.argmax(lr_vec))
            best_perm = list(candidates[best_idx])

            # 构建分数以反映最终顺序：较早排名的分数更高
            scores = np.zeros((L,), dtype=np.float32)
            for rank, pos in enumerate(best_perm):
                scores[pos] = float(L - rank)
            output_scores[i] = scores

        self.train(was_training)
        return output_scores


def build_prs_model(feature_columns, model_config):
    """
    构建与FunRec流水线兼容的PRS DPWN风格重排序模型。

    输入遵循PRM/PRM数据集约定：
      - 用户部分（广播特征）：组'user_part'
      - 物品分类序列：组'item_part'（varlen_sparse，combiner=None）
      - 密集序列特征：'item_emb'和'pv_emb'

    模型架构（类似DPWN）：
      在连接的物品序列嵌入上使用双向LSTM；与平铺的用户向量和pv嵌入融合；
      TimeDistributed MLP输出每个位置的分数，使用sigmoid激活。

    模型对象上附加了 prs_predict（PMatch + PRank 推理）、ctr_model、next_model、
    item_part_feature_names、seq_dense_feature_names 供评估器使用。

    Returns: (model, None, None)
    """
    max_seq_len = model_config.get("max_seq_len", 30)
    hidden_dim = model_config.get("hidden_dim", 128)
    user_dim = model_config.get("user_dim")  # 可选覆盖（原实现中未使用）
    item_dim = model_config.get("item_dim")  # 可选覆盖（原实现中未使用）
    dropout_rate = model_config.get("dropout_rate", 0.2)

    model = PRSModel(
        feature_columns,
        max_seq_len=max_seq_len,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    )
    return model, None, None
