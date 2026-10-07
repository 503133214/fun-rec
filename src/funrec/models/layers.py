"""通用网络层（PyTorch 实现）"""
import math
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Dense, Embedding, Layer, init_tensor_
from .base import get_activation as _base_get_activation


def squash(inputs):
    vec_squared_norm = torch.sum(torch.square(inputs), dim=-1, keepdim=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * inputs
    return vec_squashed


# ---------------------------------------------------------------------------
# 采样 softmax（等价于 TensorFlow 的 sampled_softmax_loss）
# ---------------------------------------------------------------------------

_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _log_uniform_prob(ids: torch.Tensor, range_max: int) -> torch.Tensor:
    """log-uniform(Zipfian) 分布下类别 k 的概率: P(k) = log((k+2)/(k+1)) / log(range_max+1)"""
    ids = ids.to(torch.float64)
    return torch.log((ids + 2.0) / (ids + 1.0)) / math.log1p(range_max)


def _expected_count(prob: torch.Tensor, num_sampled: int, num_tries: int) -> torch.Tensor:
    """与 TF RangeSampler 的 ExpectedCountHelper 一致

    num_tries == num_sampled（非 unique 采样或未发生重复）时为 p * num_sampled，
    否则为 1 - (1 - p)^num_tries（unique 采样的期望出现次数）
    """
    if num_tries == num_sampled:
        return prob * num_sampled
    return -torch.expm1(num_tries * torch.log1p(-prob))


def log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max):
    """等价于 TensorFlow 的 log_uniform_candidate_sampler

    采样方式与 TF LogUniformSampler 一致: value = int(exp(u * log(range_max + 1))) - 1, 再对 range_max 取模。
    unique=True 时不断有放回地采样直到得到 num_sampled 个不同的类别（num_tries 为总采样次数）。

    返回:
        sampled_candidates: [num_sampled] LongTensor
        true_expected_count: 与 true_classes 同形状的 float32 张量
        sampled_expected_count: [num_sampled] float32 张量
    """
    device = true_classes.device
    log_range = math.log1p(range_max)

    def draw(n):
        u = torch.rand(n, dtype=torch.float64, device=device)
        value = torch.exp(u * log_range).long() - 1
        return torch.remainder(value, range_max)

    if not unique:
        sampled = draw(num_sampled)
        num_tries = num_sampled
    else:
        if num_sampled > range_max:
            raise ValueError("unique 采样要求 num_sampled <= range_max")
        pool = draw(max(2 * num_sampled, 64))
        while True:
            # 按采样顺序找出每个值第一次出现的位置
            sorted_vals, perm = torch.sort(pool, stable=True)
            first_sorted = torch.ones_like(sorted_vals, dtype=torch.bool)
            first_sorted[1:] = sorted_vals[1:] != sorted_vals[:-1]
            is_first = torch.zeros_like(first_sorted)
            is_first[perm] = first_sorted
            cnt = torch.cumsum(is_first.long(), dim=0)
            pos = int(torch.searchsorted(cnt, torch.tensor([num_sampled], device=device)).item())
            if pos < pool.numel():
                num_tries = pos + 1
                sampled = pool[:num_tries][is_first[:num_tries]]
                break
            pool = torch.cat([pool, draw(pool.numel())])

    true_expected_count = _expected_count(
        _log_uniform_prob(true_classes, range_max), num_sampled, num_tries
    ).to(torch.float32)
    sampled_expected_count = _expected_count(
        _log_uniform_prob(sampled, range_max), num_sampled, num_tries
    ).to(torch.float32)
    return sampled, true_expected_count, sampled_expected_count


def sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes,
                         remove_accidental_hits=True, num_true=1, sampled_values=None):
    """采样 softmax 损失，忠实复现 TensorFlow 的 sampled_softmax_loss

    - 候选采样: log_uniform_candidate_sampler(unique=True)，与 TF 默认一致
    - logits 减去 log(期望采样次数)（subtract_log_q）
    - remove_accidental_hits=True 时，将与真实标签相同的负样本 logit 加上 -FLT_MAX
    - 返回每个样本的损失，形状 [B]

    Args:
        weights: [num_classes, dim] 类别嵌入矩阵
        biases: [num_classes] 类别偏置
        labels: [B, num_true] 真实类别 id
        inputs: [B, dim] 输入向量（如用户向量）
        sampled_values: 可选 (sampled_candidates, true_expected_count, sampled_expected_count)，
            传入时不再采样（用于测试）
    """
    labels = labels.to(inputs.device).long().reshape(-1, num_true)
    batch_size = labels.shape[0]
    if sampled_values is None:
        sampled_values = log_uniform_candidate_sampler(
            true_classes=labels, num_true=num_true, num_sampled=num_sampled,
            unique=True, range_max=num_classes,
        )
    sampled, true_expected_count, sampled_expected_count = sampled_values
    sampled = sampled.to(inputs.device).long().reshape(-1)
    true_expected_count = true_expected_count.to(inputs.device, inputs.dtype).reshape(batch_size, num_true)
    sampled_expected_count = sampled_expected_count.to(inputs.device, inputs.dtype).reshape(-1)

    all_ids = torch.cat([labels.reshape(-1), sampled])
    all_w = weights[all_ids]
    all_b = biases[all_ids]
    n_true = batch_size * num_true
    true_w = all_w[:n_true].reshape(batch_size, num_true, -1)
    true_b = all_b[:n_true].reshape(batch_size, num_true)
    sampled_w = all_w[n_true:]
    sampled_b = all_b[n_true:]

    # 真实类别 logits: [B, num_true]
    true_logits = torch.sum(inputs.unsqueeze(1) * true_w, dim=-1) + true_b
    # 负样本 logits: [B, num_sampled]
    sampled_logits = torch.matmul(inputs, sampled_w.t()) + sampled_b

    if remove_accidental_hits:
        hits = (labels.unsqueeze(-1) == sampled.view(1, 1, -1)).any(dim=1)  # [B, num_sampled]
        sampled_logits = sampled_logits + hits.to(sampled_logits.dtype) * (-_FLOAT32_MAX)

    true_logits = true_logits - torch.log(true_expected_count)
    sampled_logits = sampled_logits - torch.log(sampled_expected_count)

    out_logits = torch.cat([true_logits, sampled_logits], dim=1)
    log_probs = F.log_softmax(out_logits, dim=1)
    # 标签: 真实类别为 1/num_true，负样本为 0
    loss = -torch.sum(log_probs[:, :num_true], dim=1) / num_true
    return loss


# ---------------------------------------------------------------------------
# Keras 风格 BatchNormalization（作用于最后一维）
# ---------------------------------------------------------------------------


class LastAxisBatchNorm(Layer):
    """与 Keras BatchNormalization(axis=-1) 默认参数一致的批归一化

    momentum=0.99, epsilon=1e-3；训练时用有偏方差归一化并更新滑动均值/方差（与 Keras 非 fused 实现一致），
    推理时使用滑动统计量。支持 2 维及以上输入（对除最后一维外的所有维度统计）。
    """

    def __init__(self, momentum=0.99, epsilon=1e-3, center=True, scale=True, name=None, **kwargs):
        super().__init__(name=name)
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = self.add_weight("gamma", (dim,), "ones", trainable=self.scale)
        self.beta = self.add_weight("beta", (dim,), "zeros", trainable=self.center)
        self.register_buffer("moving_mean", torch.zeros(dim, device=self._build_device))
        self.register_buffer("moving_variance", torch.ones(dim, device=self._build_device))

    def forward(self, inputs):
        if self.training:
            dims = tuple(range(inputs.dim() - 1))
            mean = torch.mean(inputs, dim=dims)
            variance = torch.mean(torch.square(inputs - mean.detach()), dim=dims)
            with torch.no_grad():
                self.moving_mean.mul_(self.momentum).add_(mean.detach() * (1.0 - self.momentum))
                self.moving_variance.mul_(self.momentum).add_(variance.detach() * (1.0 - self.momentum))
        else:
            mean, variance = self.moving_mean, self.moving_variance
        inv = torch.rsqrt(variance + self.epsilon) * self.gamma
        return inputs * inv + (self.beta - mean * inv)


class MeanPoolingLayer(Layer):
    def __init__(self, **kwargs):
        super(MeanPoolingLayer, self).__init__(**kwargs)
        self.eps = 1e-8

    def forward(self, inputs, mask=None, *args, **kwargs):
        # mask 需显式传入（Keras 中由 mask_zero 自动传播）
        mask = mask.to(torch.float32)
        valid_len = torch.sum(mask, dim=-1, keepdim=True)
        mask = mask.unsqueeze(2)
        input_sum = torch.sum(inputs * mask, dim=1, keepdim=False)
        input_mean = input_sum / (valid_len + self.eps)
        input_mean = input_mean.unsqueeze(1)
        return input_mean


class DNNs(Layer):
    def __init__(self, units, use_bn=False, dropout_rate=0.0, out_active=False, activation='relu', last_layer_name=None, **kwargs):
        super(DNNs, self).__init__(**kwargs)
        self.units = units
        self.num_layers = len(units)
        self.out_active = out_active
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.activation = get_activation(activation)
        self.last_layer_name = last_layer_name
        self.dropout = None if dropout_rate <= 0 else nn.Dropout(dropout_rate)

        # Dense 为惰性层（首次调用时根据输入推断维度），因此可以直接在 __init__ 中定义
        # 注: 原实现在 last_layer_name 不为 None 时会在末尾多创建一个从未被调用的 Dense（无参数），此处省略
        dense_layers = []
        for i, unit in enumerate(self.units):
            if self.last_layer_name is not None and i == len(self.units) - 1:
                dense_layers.append(Dense(unit, name=f"{self.last_layer_name}_{i}"))
            else:
                dense_layers.append(Dense(unit))
        self.dense_layers = nn.ModuleList(dense_layers)

        if self.use_bn:
            self.bn_layers = nn.ModuleList([LastAxisBatchNorm() for _ in range(self.num_layers - 1)])
        else:
            self.bn_layers = nn.ModuleList()

    def forward(self, inputs):
        for i in range(self.num_layers):
            inputs = self.dense_layers[i](inputs)
            if self.use_bn and i < self.num_layers - 1:
                inputs = self.bn_layers[i](inputs)
            if i < self.num_layers - 1 or self.out_active:
                inputs = self.activation(inputs)
            if self.dropout is not None and i < self.num_layers - 1:
                inputs = self.dropout(inputs)
        return inputs


class EmbeddingIndex(Layer):
    def __init__(self, index, **kwargs):
        super(EmbeddingIndex, self).__init__(**kwargs)
        self.index = index

    def forward(self, x, **kwargs):
        device = x.device if isinstance(x, torch.Tensor) else None
        return torch.as_tensor(self.index, dtype=torch.long, device=device)


class L2NormalizeLayer(Layer):
    """L2 归一化层"""
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def forward(self, inputs):
        # 与 TensorFlow 的 l2_normalize 一致: x * rsqrt(max(sum(x^2), 1e-12))
        square_sum = torch.sum(torch.square(inputs), dim=self.axis, keepdim=True)
        return inputs * torch.rsqrt(torch.clamp(square_sum, min=1e-12))

    def get_config(self):
        return {"axis": self.axis}


class SqueezeLayer(Layer):
    """squeeze操作"""
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def forward(self, inputs):
        return torch.squeeze(inputs, dim=self.axis)

    def get_config(self):
        return {"axis": self.axis}


class SampledSoftmaxLayer(Layer):
    """采样 softmax 层

    输入: [item_embedding（物品嵌入矩阵 [vocab_size, emb_dim]，如 embedding_table.embeddings）,
           user_emb [B, emb_dim], label_index [B, 1]]
    输出: 每个样本的采样 softmax 损失 [B, 1]
    """
    def __init__(self, vocab_size, num_sampled, emb_dim, **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled

    def build(self, input_shape):
        self.zero_bias = self.add_weight(
            "zero_bias",
            shape=[self.vocab_size],
            initializer="zeros",
            trainable=False,
        )

    def forward(self, inputs):
        item_embedding, user_emb, label_index = inputs
        loss = sampled_softmax_loss(
            weights=item_embedding,
            biases=self.zero_bias,
            labels=label_index,
            inputs=user_emb,
            num_sampled=self.num_sampled,
            num_classes=self.vocab_size
        )
        return loss.unsqueeze(1)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "num_sampled": self.num_sampled,
            "emb_dim": self.emb_dim
        }


class CapsuleLayer(Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3,
                 init_std=1.0, **kwargs):
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len  # 序列的最大长度
        self.k_max = k_max # 兴趣的个数
        self.iteration_times = iteration_times  # 动态路由计算的次数，一般情况下需要计算三次
        self.init_std = init_std
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #  定义一个kmax的routing_logits, 后续通过实际的胶囊数量去做mask 不可训练 注意 trainable 参数
        std = self.init_std
        self.routing_logits = self.add_weight("routing_logits", shape=[1, self.k_max, self.max_len],
                                              initializer=lambda t: t.normal_(0.0, std),
                                              trainable=False)
        self.bilinear_mapping_matrix = self.add_weight("bilinear_mapping_matrix",
                                                       shape=[self.input_units, self.out_units],
                                                       initializer=lambda t: t.normal_(0.0, std))

    def forward(self, inputs, mask=None, **kwargs):
        # [B,max_len,input_units] , [B,max_len] 序列 mask, [B] 每个用户实际的胶囊数量
        behavior_embddings, history_mask, capsule_num  = inputs[0], inputs[1], inputs[2]
        batch_size = behavior_embddings.shape[0]

        # 用户序列mask生成
        mask = history_mask.bool().unsqueeze(1) # [B, 1, max_len]
        mask = mask.expand(-1, self.k_max, -1) # [B, k_max, max_len]

        # 实际有效的胶囊mask生成, 等价于 sequence_mask(capsule_num, k_max)
        capsule_num = capsule_num.reshape(-1)
        capsule_mask = torch.arange(self.k_max, device=capsule_num.device).unsqueeze(0) < capsule_num.unsqueeze(1) # [B, k_max]
        capsule_mask = capsule_mask.unsqueeze(-1).expand(-1, -1, self.max_len) # [B, k_max, max_len]
        capsule_padding = torch.full(capsule_mask.shape, float(-2 ** 31), device=behavior_embddings.device)
        pad = torch.full(mask.shape, float(-2 ** 32 + 1), device=behavior_embddings.device)

        for i in range(self.iteration_times):  # 动态路由的循环迭代
            # routing_logits 不可训练，读取当前值（不参与梯度计算）
            routing_logits = self.routing_logits.detach().clone()
            # 对胶囊进行mask
            mask_routing_logits = torch.where(capsule_mask, routing_logits.expand(batch_size, -1, -1), capsule_padding) # [B, k_max, max_len]
            # 对序列进行mask  [B,k_max,max_len]
            routing_logits_with_padding = torch.where(mask, mask_routing_logits, pad)
            weight = F.softmax(routing_logits_with_padding, dim=-1)  # 操作 softmax 得到 w_ij 可以对比原论文 [B,k_max,max_len]
            # 原文得到High-cat 需要经过 w_ij* S_ij*C_i 此步骤只是计算后面两个参数的点积 [B,max_len,input_units] dot [input_units,out_units] --->  [B,max_len,out_units]
            behavior_embdding_mapping = torch.matmul(behavior_embddings, self.bilinear_mapping_matrix)
            Z = torch.matmul(weight, behavior_embdding_mapping) # 接上一步完成 High-cat 输出计算 [B,k_max,out_units]
            interest_capsules = squash(Z) # [B,k_max,out_units]
            # [B,k_max,out_units]  matual [B,out_units,max_len]  ---->   [B,k_max,max_len]   ---reduce_sum--> [1,k_max,max_len]
            with torch.no_grad():
                delta_routing_logits = torch.sum(
                    torch.matmul(interest_capsules, behavior_embdding_mapping.transpose(1, 2)),
                    dim=0, keepdim=True
                )
                # 与 Keras 的 assign_add 一致: 每次前向（训练和推理）都会更新 routing_logits
                self.routing_logits.add_(delta_routing_logits)
        interest_capsules = interest_capsules.reshape(-1, self.k_max, self.out_units) # 输出兴趣胶囊 [B,k_max,out_units]
        return interest_capsules


class LabelAwareAttention(Layer):
    def __init__(self, k_max, pow_p=1, **kwargs):
        self.k_max = k_max
        self.pow_p = pow_p
        super(LabelAwareAttention, self).__init__(**kwargs)

    def forward(self, inputs, **kwargs):
        keys = inputs[0]
        query = inputs[1]
        weight = torch.sum(keys * query, dim=-1, keepdim=True)
        weight = torch.pow(weight, self.pow_p)  # [x,k_max,1]

        # 如果pow_p 比较大，直接返回最感兴趣的胶囊
        if self.pow_p >= 100:
            idx = torch.argmax(weight, dim=1).squeeze(1)  # [x]
            output = keys[torch.arange(keys.shape[0], device=keys.device), idx]
        else:
            weight = F.softmax(weight, dim=1)
            output = torch.sum(keys * weight, dim=1)

        return output


class GateNU(nn.Module):
    def __init__(self,
                 hidden_units,
                 gamma=2.,
                 l2_reg=0.):
        assert len(hidden_units) == 2
        super(GateNU, self).__init__()
        self.gamma = gamma

        self.dense_layers = nn.ModuleList([
            Dense(hidden_units[0], activation="relu", kernel_regularizer=l2_reg),
            Dense(hidden_units[1], activation="sigmoid", kernel_regularizer=l2_reg)
        ])

    def forward(self, inputs):
        output = self.dense_layers[0](inputs)

        output = self.gamma * self.dense_layers[1](output)

        return output


class EPNet(Layer):
    """Embedding Personalized Network(EPNet)

    Reference:
        PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information
    """
    def __init__(self,
                 l2_reg=0.,
                 **kwargs):
        self.l2_reg = l2_reg

        super(EPNet, self).__init__(**kwargs)
        self.gate_nu = None

    def build(self, input_shape):
        assert len(input_shape) == 2
        shape1, shape2 = input_shape
        self.gate_nu = GateNU(hidden_units=[shape2[-1], shape2[-1]], l2_reg=self.l2_reg)

    def forward(self, inputs, *args, **kwargs):
        domain, emb = inputs

        return self.gate_nu(torch.cat([domain, emb.detach()], dim=-1)) * emb


class PPNet(Layer):
    """Parameter Personalized Network(PPNet)

    Reference:
        PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information
    """
    def __init__(self,
                 multiples,
                 hidden_units,
                 activation,
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        super(PPNet, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.l2_reg = l2_reg

        self.multiples = multiples

        self.dense_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(multiples):
            self.dense_layers.append(
                nn.ModuleList([Dense(units, activation=activation, kernel_regularizer=l2_reg) for units in hidden_units])
            )
            self.dropout_layers.append(
                nn.ModuleList([nn.Dropout(dropout) for _ in hidden_units])
            )
        self.gate_nu = nn.ModuleList()

    def build(self, input_shape):
        self.gate_nu = nn.ModuleList([GateNU([i*self.multiples, i*self.multiples], l2_reg=self.l2_reg
                                             ) for i in self.hidden_units])

    def forward(self, inputs, **kwargs):
        inputs, persona = inputs

        gate_list = []
        for i in range(len(self.hidden_units)):
            gate = self.gate_nu[i](torch.cat([persona, inputs.detach()], dim=-1))
            gate = torch.split(gate, gate.shape[1] // self.multiples, dim=1)
            gate_list.append(gate)

        output_list = []

        for n in range(self.multiples):
            output = inputs

            for i in range(len(self.hidden_units)):
                fc = self.dense_layers[n][i](output)

                output = gate_list[i][n] * fc

                output = self.dropout_layers[n][i](output)

            output_list.append(output)

        return output_list


def _softmax_last_axis(x):
    return F.softmax(x, dim=-1)


class PredictLayer(Layer):
    """预测层

    该层负责将模型的最终输出转换为预测结果。
    支持二分类和多分类任务，可以选择是否输出logits或概率值。

    参数:
        task: 任务类型，'binary'表示二分类，'multiclass'表示多分类
        num_classes: 输出类别数，二分类通常为1，多分类为具体类别数
        as_logit: 是否输出logits，True表示输出原始logits，False则应用激活函数
        use_bias: 是否使用偏置项
    """
    def __init__(self,
                 task: str = 'binary',
                 num_classes: int = 1,
                 as_logit: bool = False,
                 use_bias: bool = False,
                 activation = None,
                 **kwargs):
        if task not in ('binary', 'multiclass'):
            raise ValueError(f"task must be binary or multiclass, but got {task}")
        super(PredictLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.task = task
        self.as_logit = as_logit
        if isinstance(activation, str):
            activation = get_activation(activation)
        self.activation = activation
        if not as_logit:
            if task == "binary":
                self.activation = torch.sigmoid
            elif task == "multiclass":
                self.activation = _softmax_last_axis
        self.dense_layer = None
        self._use_global_bias = False

    def build(self, input_shape):
        if input_shape[-1] != self.num_classes:
            self.dense_layer = Dense(self.num_classes, use_bias=self.use_bias)
        else:
            if self.use_bias:
                # 输入维度与类别数相同时只加一个全局偏置
                self.global_bias = self.add_weight("global_bias", shape=(1,), initializer="zeros")
                self._use_global_bias = True

    def forward(self, inputs, *args, **kwargs):
        output = inputs
        if self.dense_layer is not None:
            output = self.dense_layer(output)
        elif self._use_global_bias:
            output = output + self.global_bias
        if self.activation is not None:
            logits = output
            output = self.activation(output)
            if not self.as_logit:
                # 原实现最后一个算子为 tf.nn.sigmoid/softmax，Keras 交叉熵损失会直接使用其输入 logits
                # （from_logits=True，不裁剪），见 training/loss.py 中的 _keras_logits
                output._keras_logits = logits
                output._keras_logits_op = "Sigmoid" if self.task == "binary" else "Softmax"

        return output

    def get_config(self):
        """获取层配置

        返回层的配置参数，用于序列化和重建层。
        """
        return {
            "task": self.task,
            "num_classes": self.num_classes,
            "as_logit": self.as_logit,
            "use_bias": self.use_bias
        }

    def compute_output_shape(self, input_shape):
        """计算输出形状

        根据输入形状计算输出形状。
        """
        return tuple(input_shape[:-1]) + (self.num_classes,)


class SequenceMeanPoolingLayer(Layer):
    def __init__(self, keep_shape=False, supports_masking=True, **kwargs):
        """序列池化层，用于对变长序列特征进行平均池化操作。

        Args:
            keep_shape (bool): 是否保持输入形状，默认为False。
            supports_masking (bool): 是否支持mask。默认为True。
        """
        super(SequenceMeanPoolingLayer, self).__init__(**kwargs)
        self.keep_shape = keep_shape
        self.supports_masking = supports_masking
        self.eps = 1e-8

    def forward(self, inputs, mask=None, *args, **kwargs):
        """
        前向传播逻辑。

        Args:
            inputs (torch.Tensor): 输入张量，形状为(batch_size, seq_len, embedding_dim)。
            mask (torch.Tensor): 掩码张量，形状为(batch_size, seq_len)。默认为None。
                （原实现由 mask_zero 的 Embedding 隐式传入，这里需要显式传入）

        Returns:
            torch.Tensor: 平均池化后的张量，形状为(batch_size, embedding_dim)。
        """
        if mask is not None:
            mask = mask.to(inputs.dtype)
            mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            inputs = inputs * mask

        if mask is not None:
            valid_len = torch.sum(mask, dim=1)  # (batch_size, 1)
        else:
            valid_len = float(inputs.shape[1])
        output = torch.sum(inputs, dim=1) / (valid_len + self.eps)
        if self.keep_shape:
            output = output.unsqueeze(1)
        return output

    def get_config(self):
        """
        返回层的配置字典。

        Returns:
            dict: 配置字典。
        """
        return {"keep_shape": self.keep_shape, "supports_masking": self.supports_masking}

    def compute_mask(self, inputs, mask=None):
        return None


class BiasOnly(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def forward(self, inputs):
        return inputs + self.bias


class FM(Layer):
    """因子分解机(Factorization Machine)层

    这个层实现了FM的二阶交叉部分，用于捕获特征之间的交互关系。
    FM算法的核心思想是通过隐向量内积来表示特征之间的相互作用。

    计算公式:
    sum_{i=1}^{n}sum_{j=i+1}^{n} <v_i, v_j> x_i x_j
    = 0.5 * (sum(v)^2 - sum(v^2))

    其中v是特征的embedding向量，x是特征值。
    """
    def __init__(self, **kwargs):
        """初始化FM层

        Args:
            **kwargs: 传递给父类的参数
        """
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        """构建层

        Args:
            input_shape: 输入张量的形状，预期为[batch_size, field_num, embedding_size]
        """
        pass

    def forward(self, inputs, **kwargs):
        """前向传播

        Args:
            inputs: 形状为[batch_size, field_num, embedding_size]的张量
            **kwargs: 额外参数

        Returns:
            形状为[batch_size, 1]的张量，表示FM的二阶交互项
        """
        concated_embeds_value = inputs  # shape: [batch_size, field_num, embedding_size]

        # 计算(sum(v))^2，先在field维度上求和，再平方
        square_of_sum = torch.square(torch.sum(concated_embeds_value, dim=1, keepdim=True))  # [batch_size, 1, embedding_size]

        # 计算sum(v^2)，先平方，再在field维度上求和
        sum_of_square = torch.sum(concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)  # [batch_size, 1, embedding_size]

        # 计算FM的二阶交互项: 0.5 * ((sum(v))^2 - sum(v^2))
        cross_term = square_of_sum - sum_of_square  # [batch_size, 1, embedding_size]
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)  # [batch_size, 1]

        return cross_term

    def compute_output_shape(self, input_shape):
        """计算输出形状

        Args:
            input_shape: 输入张量的形状

        Returns:
            输出张量的形状，固定为(batch_size, 1)
        """
        return (None, 1)


class PReLU(Layer):
    """参数化 ReLU，与原实现所用 PReLU 层（默认参数）等价

    alpha 初始化为 0，对除 batch 维以外的每个元素各有一个 alpha（shared_axes=None），
    首次调用时根据输入形状惰性创建。
    f(x) = max(0, x) - alpha * max(0, -x)
    """

    def __init__(self, alpha_initializer='zeros', **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.alpha_initializer = alpha_initializer

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=tuple(input_shape[1:]),
            initializer=self.alpha_initializer,
            name='alpha'
        )

    def forward(self, inputs, *args, **kwargs):
        pos = F.relu(inputs)
        neg = -self.alpha * F.relu(-inputs)
        return pos + neg


def get_activation(activation):
    if activation is None:
        return nn.Identity()
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'prelu':
            return PReLU()
        elif activation.lower() == 'dice':
            return Dice()
        else:
            # 对应原实现中按名称构造的通用激活层
            return _base_get_activation(activation)
    if isinstance(activation, nn.Module):
        return activation
    # 普通可调用对象包装为模块
    return _base_get_activation(activation)


class Dice(Layer):
    def __init__(self, epsilon=1e-3, **kwargs):
        super(Dice, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(1,),
                                     initializer='zeros',
                                     dtype=torch.float32,
                                     name='alpha')

    def forward(self, inputs, *args, **kwargs):
        # 与原实现一致: 训练和推理阶段均使用当前 batch(axis=0) 的均值与(有偏)方差进行归一化
        mean = torch.mean(inputs, dim=0)
        var = torch.var(inputs, dim=0, unbiased=False)
        inputs_normed = (inputs - mean) * torch.rsqrt(var + self.epsilon)
        x_p = torch.sigmoid(inputs_normed)
        return self.alpha * (1.0 - x_p) * inputs + x_p * inputs


class FeedForwardLayer(Layer):
    def __init__(self,
                 hidden_units: List[int],
                 activation: Optional[Union[str, Callable]] = "relu",
                 l2_reg: float = 0.,
                 dropout_rate: float = 0.,
                 use_bn: bool = False,
                 **kwargs
                 ):
        super(FeedForwardLayer, self).__init__(**kwargs)

        self.dense_layers = nn.ModuleList([Dense(i, kernel_regularizer=l2_reg) for i in hidden_units])

        self.activations = nn.ModuleList([get_activation(activation) for _ in hidden_units])

        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout_rate) for _ in hidden_units])
        if use_bn:
            self.bn_layers = nn.ModuleList([LastAxisBatchNorm() for _ in hidden_units])

        self.use_bn = use_bn

    def forward(self, inputs, **kwargs):
        output = inputs

        for i in range(len(self.dense_layers)):
            fc = self.dense_layers[i](output)

            if self.use_bn:
                fc = self.bn_layers[i](fc)

            fc = self.activations[i](fc)

            fc = self.dropout_layers[i](fc)

            output = fc

        return output


class DinAttentionLayer(Layer):
    """ DIN Attention:

    Reference:
        Deep Interest Network for Click-Through Rate Prediction

    Args:
        ffn_hidden_units: 前馈神经网络隐藏层单元数
        ffn_activation: 前馈神经网络激活函数
        query_ffn: 是否使用前馈神经网络对查询输入进行处理，当查询和键的维度不同时必须启用
        query_activation: 查询前馈神经网络激活函数
    """
    def __init__(self,
                 ffn_hidden_units=[80, 40],
                 ffn_activation="dice",
                 query_ffn=False,
                 query_activation="prelu",
                 **kwargs):
        super(DinAttentionLayer, self).__init__(**kwargs)
        self.query_ffn = query_ffn
        self.query_activation = query_activation
        self.query_ffn_layer = None

        self.ffn_layer = FeedForwardLayer(ffn_hidden_units, ffn_activation)
        self.dense = Dense(1)

    def build(self, input_shape):
        assert len(input_shape) == 2 and len(input_shape[0]) == 3 and len(input_shape[1]) == 3
        if self.query_ffn:
            self.query_ffn_layer = Dense(input_shape[1][-1], self.query_activation)

    def forward(self, inputs, mask=None, **kwargs):
        """
        Args:
            inputs: [query, keys]，query: [B, 1, H]，keys: [B, L, H]
            mask: 原实现隐式传入的掩码。可以是 [query_mask, keys_mask] 列表（与原实现一致，取 mask[1]），
                也可以直接传入 keys 的掩码张量 [B, L]（True/1 为有效位置）。None 表示不做掩码。
        """
        # query: [B, 1, H]
        # keys: [B, L, H]
        query, keys = inputs
        query = query.squeeze(1)
        if self.query_ffn_layer is not None:
            query = self.query_ffn_layer(query)

        length = keys.shape[-2]
        query = query.unsqueeze(1)
        att_inputs = torch.cat([query.expand(-1, length, -1),
                                keys, query - keys, query * keys], dim=-1)
        hidden_layer = self.ffn_layer(att_inputs)
        scores = self.dense(hidden_layer)
        scores = scores.reshape(-1, 1, length)

        if isinstance(mask, (list, tuple)):
            mask = mask[1]
        if mask is not None:
            mask = mask.unsqueeze(1)

            scores = scores + (1.0 - mask.to(keys.dtype)) * (-1e9)

        scores = scores / (keys.shape[-1] ** 0.5)
        scores = torch.softmax(scores, dim=-1)

        att_outputs = torch.matmul(scores, keys)

        return att_outputs.squeeze(1)



class PositionEncodingLayer(Layer):
    def __init__(self, dims, max_len, trainable=True, name=None, dtype=None, dynamic=False, initializer='glorot_uniform', **kwargs):
        """可学习的位置编码

        Args:
            dims: 编码维度
            max_len: 最大序列长度
            trainable: 是否可训练，默认为True
            initializer: 初始化方式，可以是'glorot_uniform'(随机初始化)或'sinusoidal'(正弦初始化)
        """
        super().__init__(name=name, **kwargs)
        # 注: 原 TF 实现中 self.dims 被注释掉，导致 'sinusoidal' 初始化会报错；这里保留 dims 以便正弦初始化可用
        self.dims = dims
        self.max_len = max_len
        self.trainable = trainable
        self.initializer = initializer

    def build(self, input_shape):
        # 初始化位置编码
        if self.initializer == 'sinusoidal':
            # 使用正弦函数初始化
            encoded_vec = np.array([pos/np.power(10000, 2*i/self.dims)
                        for pos in range(self.max_len) for i in range(self.dims)])
            encoded_vec[::2] = np.sin(encoded_vec[::2])
            encoded_vec[1::2] = np.cos(encoded_vec[1::2])
            initial_value = torch.as_tensor(encoded_vec.reshape([self.max_len, self.dims]), dtype=torch.float32)

            def initializer(t):
                t.copy_(initial_value.to(t.device))
        else:
            # 使用随机初始化
            initializer = 'glorot_uniform'

        # 创建可学习的权重
        self.pos_embeddings = self.add_weight(
            shape=(self.max_len, input_shape[-1]),
            initializer=initializer,
            trainable=self.trainable,
            name="position_embeddings"
        )

    def forward(self, inputs, *args, **kwargs):
        # 返回位置编码，形状与输入序列相匹配
        batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        # 获取序列长度对应的位置编码
        position_enc = self.pos_embeddings[:seq_len, :]  # [seq_len, dims]
        # 扩展到batch size维度
        position_enc = position_enc.unsqueeze(0)  # [1, seq_len, dims]
        position_enc = position_enc.expand(batch_size, -1, -1)  # [batch_size, seq_len, dims]
        return position_enc

    def get_config(self):
        return {
            'dims': self.dims,
            'max_len': self.max_len,
            'initializer': self.initializer
        }



class NegativeSampleEmbedding(Layer):
    def __init__(self, vocab_size, num_sampled, sampled_type='uniform', trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.sampled_type = sampled_type
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled

    def build(self, input_shape):
        pass

    def forward(self, target_index, embedding_table):
        if self.sampled_type == 'uniform':
            # 在 [1, vocab_size) 中均匀采样负样本 id（0 为 padding）
            sampled_idx = torch.randint(
                1,
                self.vocab_size,
                (target_index.shape[0], self.num_sampled),
                device=embedding_table.device,
                dtype=torch.int64,
            )
        else:
            raise ValueError(f"不支持的采样方式: {self.sampled_type}")

        # 按索引从嵌入表中取出负样本 embedding（梯度可回传到嵌入表）
        neg_sampled_emb = F.embedding(sampled_idx, embedding_table)
        return neg_sampled_emb


class PartitionedNormalization(Layer):
    """分域批归一化（STAR 中的 Partitioned Normalization）

    每个域使用独立的 BatchNormalization（center=False, scale=False，统计量单独维护），
    再乘以 (全局 gamma + 域 gamma)、加上 (全局 beta + 域 beta)。

    输入: [inputs (B x dim), domain_index (B x 1 或 B)]
    """

    def __init__(self,
                 num_domain,
                 name=None,
                 **kwargs):
        super(PartitionedNormalization, self).__init__(name=name)
        self.num_domain = num_domain
        # Keras BatchNormalization 默认 momentum=0.99, epsilon=1e-3
        self.momentum = float(kwargs.get("momentum", 0.99))
        self.epsilon = float(kwargs.get("epsilon", 1e-3))

    def build(self, input_shape):
        assert len(input_shape) == 2 and len(input_shape[1]) <= 2
        dim = input_shape[0][-1]
        self.dim = dim

        self.global_gamma = self.add_weight(
            name="global_gamma",
            shape=[dim],
            initializer=lambda t: t.fill_(0.5),
            trainable=True
        )
        self.global_beta = self.add_weight(
            name="global_beta",
            shape=[dim],
            initializer="zeros",
            trainable=True
        )
        self.domain_gamma = self.add_weight(
                name="domain_gamma",
                shape=[self.num_domain, dim],
                initializer=lambda t: t.fill_(0.5),
                trainable=True
            )
        self.domain_beta = self.add_weight(
                name="domain_beta",
                shape=[self.num_domain, dim],
                initializer="zeros",
                trainable=True
            )
        # 每个域的 BN 滑动统计量（对应 bn_i/moving_mean, bn_i/moving_variance）
        device = self._build_device
        self.register_buffer("moving_mean", torch.zeros(self.num_domain, dim, device=device))
        self.register_buffer("moving_variance", torch.ones(self.num_domain, dim, device=device))

    def _bn(self, x, i):
        """与 Keras BatchNormalization(center=False, scale=False) 一致的归一化

        训练时使用当前批次（该域样本）的有偏方差归一化，并以有偏方差更新滑动方差（与 Keras 一致，
        而 torch.nn.BatchNorm 使用无偏方差更新滑动方差）。
        """
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                decay = 1.0 - self.momentum
                self.moving_mean[i].sub_((self.moving_mean[i] - mean.detach()) * decay)
                self.moving_variance[i].sub_((self.moving_variance[i] - var.detach()) * decay)
        else:
            mean = self.moving_mean[i]
            var = self.moving_variance[i]
        return (x - mean) * torch.rsqrt(var + self.epsilon)

    def forward(self, inputs):
        inputs, domain_index = inputs
        domain_index = domain_index.reshape(-1).long()

        output = inputs
        # compute each domain's BN individually
        for i in range(self.num_domain):
            mask = domain_index == i
            # 当前批次中没有该域的样本: 输出保持不变，且不更新该域的 BN 统计量
            # （原 Keras 实现会用空批次的 NaN 统计量更新滑动均值/方差，此处避免该问题）
            if not bool(mask.any()):
                continue
            # get current domain samples' indices
            indices = torch.nonzero(mask, as_tuple=True)[0]
            single_bn = self._bn(inputs[indices], i)
            single_bn = (self.global_gamma + self.domain_gamma[i]) * single_bn + (self.global_beta + self.domain_beta[i])
            output = output.index_put((indices,), single_bn)

        return output


class StarTopologyFCN(Layer):
    """
    Reference:
        One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction
    """
    def __init__(self,
                 num_domain,
                 hidden_units,
                 activation="relu",
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        super(StarTopologyFCN, self).__init__(name=kwargs.get("name"))
        self.num_domain = num_domain
        self.hidden_units = list(hidden_units)
        self.activation_list = nn.ModuleList([get_activation(activation) for _ in hidden_units])
        self.dropout_list = nn.ModuleList([nn.Dropout(dropout) for _ in hidden_units])
        self.l2_reg = l2_reg

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_shape = input_shape[0]
        device = self._build_device

        # 注: 原实现以单元数命名(shared_bias_{units})，hidden_units 含重复值时参数名冲突，
        # 在 torch 中会导致前一个参数被覆盖而不再注册，因此改为按层序号命名
        self.shared_bias = [
            self.add_weight(
                name=f"shared_bias_{k}",
                shape=[1, i],
                initializer="zeros",
                trainable=True
            ) for k, i in enumerate(self.hidden_units)
        ]
        self.domain_bias_list = nn.ModuleList([
            Embedding(
                self.num_domain,
                output_dim=i,
                embeddings_initializer="zeros"
            ) for i in self.hidden_units
        ]).to(device)

        hidden_units = self.hidden_units.copy()
        hidden_units.insert(0, input_shape[-1])
        self.shared_weights = [
            self.add_weight(
                name=f"shared_weight_{i}",
                shape=[1, hidden_units[i], hidden_units[i+1]],
                initializer="glorot_uniform",
                regularizer=self.l2_reg,
                trainable=True
            ) for i in range(len(hidden_units) - 1)
        ]
        self.domain_weights_list = nn.ModuleList([
            Embedding(
                self.num_domain,
                hidden_units[i] * hidden_units[i + 1],
                embeddings_initializer="glorot_uniform",
                l2_reg=self.l2_reg
            ) for i in range(len(hidden_units) - 1)
        ]).to(device)

    def forward(self, inputs, **kwargs):
        inputs, domain_index = inputs
        domain_index = domain_index.long()
        output = inputs.unsqueeze(1)
        for i in range(len(self.hidden_units)):
            domain_weight = self.domain_weights_list[i](domain_index).reshape(
                [-1] + list(self.shared_weights[i].shape[1:]))
            weight = self.shared_weights[i] * domain_weight
            domain_bias = self.domain_bias_list[i](domain_index).reshape(
                [-1] + list(self.shared_bias[i].shape[1:]))
            bias = self.shared_bias[i] + domain_bias

            fc = torch.matmul(output, weight) + bias.unsqueeze(1)
            output = self.activation_list[i](fc)
            output = self.dropout_list[i](output)

        return output.squeeze(1)


class PNN(Layer):
    """
    Reference:
        Product-based Neural Networks for User Response Prediction
    """
    def __init__(self, units, use_inner=True, use_outer=True):
        super(PNN, self).__init__()
        self.use_inner = use_inner
        self.use_outer = use_outer
        self.units = units  # 原文中D1的大小

    def build(self, input_shape):
        # input_shape[0] : feat_nums x embed_dims
        # 输入为 embedding 列表（每个为 B x 1 x D）；也兼容已拼接的 B x N x D 张量
        if isinstance(input_shape, list):
            self.feat_nums = len(input_shape)
            self.embed_dims = input_shape[0][-1]
        else:
            self.feat_nums = input_shape[1]
            self.embed_dims = input_shape[-1]
        flatten_dims = self.feat_nums * self.embed_dims

        # 线性信号权重，用于产生Z
        self.linear_w = self.add_weight(
            name='linear_w',
            shape=(flatten_dims, self.units),
            initializer='glorot_normal'
        )

        # 内积权重
        if self.use_inner:
            # 优化后的内积权重大小为：D x N
            self.inner_w = self.add_weight(
                name='inner_w',
                shape=(self.units, self.feat_nums),
                initializer='glorot_normal'
            )

        # 外积权重
        if self.use_outer:
            # 优化后的外积权重大小为：D x embed_dim x embed_dim
            self.outer_w = self.add_weight(
                name='outer_w',
                shape=(self.units, self.embed_dims, self.embed_dims),
                initializer='glorot_normal'
            )

    def forward(self, inputs):
        """
        inputs: list, 包含所有特征的embedding矩阵, shape为[B, N, D]
        """
        # 计算线性信号部分的输出
        if isinstance(inputs, (list, tuple)):
            concat_embed = torch.cat(list(inputs), dim=1)  # B x feat_nums x embed_dims
        else:
            concat_embed = inputs
        concat_embed_ = concat_embed.reshape(-1, self.feat_nums * self.embed_dims)  # B x feat_nums * embed_dims
        lz = torch.matmul(concat_embed_, self.linear_w)  # B x units

        # 内积部分
        lp_list = []
        if self.use_inner:
            # 给每一个特征向量乘以权重，并在特征之间的维度上求和（对所有 units 一次性计算）
            delta = torch.einsum("bnd,un->bud", concat_embed, self.inner_w)  # B x units x embed_dims
            # 在特征embedding维度上求二范数
            lp_list.append(torch.sum(delta * delta, dim=-1))  # B x units

        # 外积部分
        if self.use_outer:
            # 将embedding矩阵在特征间的维度上通过求和进行压缩
            feat_sum = torch.sum(concat_embed, dim=1)  # B x embed_dims

            # 求外积 a * a^T
            product = feat_sum.unsqueeze(2) * feat_sum.unsqueeze(1)  # B x embed_dims x embed_dims

            # 将product与外积权重矩阵对应元素相乘再相加
            lp_list.append(torch.einsum("bij,uij->bu", product, self.outer_w))  # B x units

        # 将所有交叉特征拼接到一起，再与lz拼接到一起
        product_out = torch.cat([lz] + lp_list, dim=1)  # [batch_size, units * (1 + use_inner + use_outer)]

        return product_out


class DCN(Layer):
    """
    Reference:
        Deep & Cross Network for Ad Click Predictions
    """

    def __init__(self, num_cross_layers, l2_reg=0.0):
        super(DCN, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # input_shape : batch_size x (feat_nums * embed_dims)
        self.input_dim = input_shape[-1]

        self.ws = [self.add_weight(
            name='cross_weight_{}'.format(i),
            shape=(self.input_dim, 1),
            initializer='glorot_normal',
            regularizer=self.l2_reg,
            trainable=True
        ) for i in range(self.num_cross_layers)]

        self.bs = [self.add_weight(
            name='cross_bias_{}'.format(i),
            shape=(self.input_dim, 1),
            initializer='zeros',
            regularizer=self.l2_reg,
            trainable=True
        ) for i in range(self.num_cross_layers)]

    def forward(self, x_0):
        """
        实现交叉层的计算: x_{l+1} = x_0 * x_l^T * w_l + b_l + x_l

        Args:
            x_0: 原始输入，shape: (batch_size, feature_embedding_dim)
        """
        # x_l 是上一层的输出，shape: (batch_size, feature_embedding_dim)

        x_l = x_0
        for i in range(self.num_cross_layers):
            # 计算 x_l^T * w_l
            xlw = torch.matmul(x_l, self.ws[i])  # (batch_size, 1)

            # 计算 x_0 * (x_l^T * w_l)
            cross_term = x_0 * xlw  # (batch_size, feature_embedding_dim)

            # 计算 x_0 * x_l^T * w_l + b_l + x_l
            x_l = cross_term + self.bs[i].reshape(1, -1) + x_l  # (batch_size, feature_embedding_dim)

        return x_l


class AttentionPoolingLayer(Layer):
    """
    Reference:
        Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks (AFM)
    """
    def __init__(self, attention_factor, l2_reg, **kwargs):
        super(AttentionPoolingLayer, self).__init__(name=kwargs.get("name"))
        self.attention_factor = attention_factor
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # input_shape: (None, n, D)

        # 注意力权重, D x attention_factor
        self.attention_weight = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_factor),
            initializer='glorot_normal',
            regularizer=self.l2_reg,
            trainable=True
        )

        # 注意力偏置, attention_factor x 1
        self.attention_bias = self.add_weight(
            name='attention_bias',
            shape=(self.attention_factor,),
            initializer='zeros',
            trainable=True
        )

        # 注意力投影层, attention_factor x 1
        self.attention_projection = self.add_weight(
            name='attention_projection',
            shape=(self.attention_factor, 1),
            initializer='glorot_normal',
            trainable=True
        )

    def forward(self, inputs, **kwargs):
        # inputs: B x num_interactions x D
        # - a_ij' = h^T \cdot RELU(W \cdot (v_i * v_j) * x_i * x_j + b)
        # - a_ij = softmax(a_ij')
        # - output = \sum_{i=1}^{n} \sum_{j=i+1}^{n} a_ij * (v_i * v_j) * x_i * x_j

        # 注意力权重计算
        weighted_inputs = torch.matmul(inputs, self.attention_weight) + self.attention_bias  # B x num_interactions x attention_factor

        # activation relu
        activation = F.relu(weighted_inputs)  # B x num_interactions x attention_factor

        # 注意力投影
        projected_activation = torch.matmul(activation, self.attention_projection)  # B x num_interactions x 1

        # 注意力权重归一化
        attention_weights = torch.softmax(projected_activation, dim=1)  # B x num_interactions x 1

        # 注意力池化
        return torch.sum(inputs * attention_weights, dim=1)  # B x D


class CINs(Layer):
    def __init__(self, cin_layer_sizes, l2_reg=1e-5, **kwargs):
        """压缩交互网络(CIN)层

        Reference:
            xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

        Args:
            cin_layer_sizes: CIN各层的大小列表
        """
        super(CINs, self).__init__(**kwargs)
        self.cin_layer_sizes = cin_layer_sizes
        self.l2_reg = l2_reg
        # 为每个CIN层创建对应的Dense层（输入维度在首次调用时惰性推断）
        self.dense_layers = nn.ModuleList(
            [
                Dense(
                    layer_size,
                    activation=None,
                    use_bias=False,
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=self.l2_reg,
                    name=f"cin_dense_{k}",
                )
                for k, layer_size in enumerate(self.cin_layer_sizes)
            ]
        )

    def forward(self, inputs, **kwargs):
        """CIN层的前向传播

        Args:
            inputs: 特征embedding表示，shape为[B, field_num, emb_dim]

        Returns:
            CIN的最终输出张量 [B, sum(cin_layer_sizes)]
        """
        pooled_outputs = []
        field_nums = [inputs.shape[1]]  # 初始特征域数量 m

        # 初始化输入 X^0
        cin_layers = [inputs]  # 第0层就是原始特征embedding, x_0

        # 构建CIN网络
        for k, layer_size in enumerate(self.cin_layer_sizes):
            # 创建当前层，更新field_nums记录
            field_nums.append(layer_size)

            # 获取前一层输出X^{k-1}和输入层X^0
            x_k_minus_1 = cin_layers[-1]  # [B, H_{k-1}, D]
            x_0 = cin_layers[0]  # [B, m, D]

            # 获取维度信息
            batch_size = x_0.shape[0]
            embed_dim = x_0.shape[-1]
            x0_field_num, prev_field_num = field_nums[0], field_nums[-2]

            # 为更好地进行特征交互计算，调整张量形状
            # reshape为 [B, H_{k-1}, 1, D]
            x_k_minus_1_expand = x_k_minus_1.unsqueeze(2)
            # reshape为 [B, 1, m, D]
            x_0_expand = x_0.unsqueeze(1)

            # 执行向量级别的特征交互（哈达玛积）
            # Z^k形状为 [B, H_{k-1}, m, D]
            z_k = x_k_minus_1_expand * x_0_expand

            # 重塑张量，便于应用线性变换
            # reshape为 [B, H_{k-1}*m, D]
            z_k_reshape = z_k.reshape(batch_size, prev_field_num * x0_field_num, embed_dim)

            # 通过线性变换压缩特征交互的结果，这相当于原论文中的权重矩阵W
            # 使用Dense层实现线性变换，将H_{k-1}*m维压缩为H_k维
            x_k = self.dense_layers[k](
                z_k_reshape.transpose(1, 2)  # [B, D, H_{k-1}*m] -> [B, D, H_k]
            )

            # 转置回来，得到当前层的输出 [B, H_k, D]
            x_k = x_k.transpose(1, 2)

            # 保存当前层
            cin_layers.append(x_k)

            # 对每个特征图进行求和池化，得到标量输出 [B, H_k]
            pooled_output = torch.sum(x_k, dim=-1)
            pooled_outputs.append(pooled_output)

        # 拼接所有层的池化输出 [B, sum(H_k)]
        final_result = torch.cat(pooled_outputs, dim=1)
        return final_result


class MultiHeadAttentionLayer(Layer):
    """
    Reference:
        AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
    """
    def __init__(self, attention_dim, num_heads, use_residual=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention_dim = attention_dim  # 注意力层的维度 d'
        self.num_heads = num_heads          # 注意力头数量 H
        self.use_residual = use_residual    # 是否使用残差连接

    def build(self, input_shape):
        # input_shape: B x N x D
        self.feat_num = input_shape[1]
        self.embed_dim = input_shape[2]

        # 为每个注意力头创建查询(Query)、键(Key)和值(Value)的权重矩阵
        # （参数以 query_weights_{i} 等名称注册，前向时按名称取用）
        for i in range(self.num_heads):
            self.add_weight(
                name=f'query_weights_{i}',
                shape=[self.embed_dim, self.attention_dim],
                initializer='glorot_uniform',
                trainable=True
            )

            self.add_weight(
                name=f'key_weights_{i}',
                shape=[self.embed_dim, self.attention_dim],
                initializer='glorot_uniform',
                trainable=True
            )

            self.add_weight(
                name=f'value_weights_{i}',
                shape=[self.embed_dim, self.attention_dim],
                initializer='glorot_uniform',
                trainable=True
            )

        # 残差连接的权重矩阵
        if self.use_residual:
            self.residual_weights = self.add_weight(
                name='residual_weights',
                shape=[self.embed_dim, self.attention_dim * self.num_heads],
                initializer='glorot_uniform',
                trainable=True
            )

    @property
    def query_weights(self):
        return [getattr(self, f'query_weights_{i}') for i in range(self.num_heads)]

    @property
    def key_weights(self):
        return [getattr(self, f'key_weights_{i}') for i in range(self.num_heads)]

    @property
    def value_weights(self):
        return [getattr(self, f'value_weights_{i}') for i in range(self.num_heads)]

    def forward(self, inputs):
        # 存储每个注意力头的输出
        head_outputs = []
        query_weights, key_weights, value_weights = self.query_weights, self.key_weights, self.value_weights

        for i in range(self.num_heads):
            # 计算查询、键、值矩阵
            query = torch.einsum('bfe,ea->bfa', inputs, query_weights[i])  # [batch_size, feat_num, attention_dim]
            key = torch.einsum('bfe,ea->bfa', inputs, key_weights[i])      # [batch_size, feat_num, attention_dim]
            value = torch.einsum('bfe,ea->bfa', inputs, value_weights[i])  # [batch_size, feat_num, attention_dim]

            # 计算注意力得分
            attention_score = torch.matmul(query, key.transpose(-1, -2))  # [batch_size, feat_num, feat_num]

            # 对注意力分数进行归一化处理
            attention_weights = torch.softmax(attention_score, dim=-1)  # [batch_size, feat_num, feat_num]

            # 计算注意力加权和
            head_output = torch.matmul(attention_weights, value)  # [batch_size, feat_num, attention_dim]
            head_outputs.append(head_output)

        # 将所有注意力头的输出拼接在一起
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [batch_size, feat_num, attention_dim * num_heads]

        # 应用残差连接
        if self.use_residual:
            # 将原始输入通过线性变换后的结果与注意力输出相加
            residual_input = torch.tensordot(inputs, self.residual_weights, dims=([2], [0]))  # [batch_size, feat_num, attention_dim * num_heads]

            output = F.relu(multi_head_output + residual_input)  # [batch_size, feat_num, attention_dim * num_heads]
        else:
            output = F.relu(multi_head_output)

        return output # B x N x (D * H)


class SENetLayer(Layer):
    """
    Reference:
        FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
    """
    def __init__(self, reduction_ratio=3):
        super(SENetLayer, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # input_shape B x N x D
        self.feat_nums = input_shape[1]
        self.embed_dims = input_shape[2]

        # 计算缩减大小
        self.reduction_size = max(1, int(self.feat_nums // self.reduction_ratio))

        # 定义挤压和激励的FC层
        self.w1 = self.add_weight(
            name='senet_w1',
            shape=(self.feat_nums, self.reduction_size),
            initializer='glorot_normal'
        )

        self.w2 = self.add_weight(
            name='senet_w2',
            shape=(self.reduction_size, self.feat_nums),
            initializer='glorot_normal'
        )

    def forward(self, inputs):
        # 挤压：全局平均池化
        squeeze = torch.mean(inputs, dim=-1)  # batch_size x feat_nums

        # 激活：两个FC层，ReLU和Sigmoid激活
        excitation = torch.matmul(squeeze, self.w1)  # batch_size x reduction_size
        excitation = F.relu(excitation)
        excitation = torch.matmul(excitation, self.w2)  # batch_size x feat_nums
        excitation = torch.sigmoid(excitation)

        # 应用注意力权重
        excitation = excitation.unsqueeze(2)  # batch_size x feat_nums x 1
        reweighted_embed = inputs * excitation  # batch_size x feat_nums x embed_dims

        return reweighted_embed


class BilinearInteractionLayer(Layer):
    """
    Reference:
        FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
    """
    def __init__(self, bilinear_type="interaction"):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type

    def build(self, input_shape):
        # input_shape B x N x D
        self.feat_nums = input_shape[1]
        self.embed_dims = input_shape[2]
        self._w_names = []

        if self.bilinear_type == 'all':
            # 所有特征交互共享一个权重矩阵
            self.W = self.add_weight(
                name='bilinear_w',
                shape=(self.embed_dims, self.embed_dims),
                initializer='glorot_normal'
            )
        elif self.bilinear_type == 'each':
            # 每个特征有自己的权重矩阵
            for i in range(self.feat_nums):
                self.add_weight(
                    name=f'bilinear_w_{i}',
                    shape=(self.embed_dims, self.embed_dims),
                    initializer='glorot_normal'
                )
                self._w_names.append(f'bilinear_w_{i}')
        elif self.bilinear_type == 'interaction':
            # 每个特征交互对有自己的权重矩阵
            self.pairs = []
            for i in range(self.feat_nums):
                for j in range(i+1, self.feat_nums):
                    self.pairs.append((i, j))
                    self.add_weight(
                        name=f'bilinear_w_{i}_{j}',
                        shape=(self.embed_dims, self.embed_dims),
                        initializer='glorot_normal'
                    )
                    self._w_names.append(f'bilinear_w_{i}_{j}')

    @property
    def W_list(self):
        return [getattr(self, n) for n in self._w_names]

    def forward(self, inputs):
        #inputs B x N x D
        interaction_outputs = []

        if self.bilinear_type == 'all':
            # 一个共享的权重矩阵
            vdotw_list = [torch.matmul(inputs[:, i, :], self.W) for i in range(self.feat_nums)]
            for i in range(self.feat_nums):
                for j in range(i+1, self.feat_nums):
                    interaction = vdotw_list[i] * inputs[:, j, :]  # batch_size x embed_dims
                    interaction_outputs.append(interaction)

        elif self.bilinear_type == 'each':
            # 每个特征有自己的权重矩阵
            W_list = self.W_list
            vdotw_list = [torch.matmul(inputs[:, i, :], W_list[i]) for i in range(self.feat_nums)]
            for i in range(self.feat_nums):
                for j in range(i+1, self.feat_nums):
                    interaction = vdotw_list[i] * inputs[:, j, :]  # batch_size x embed_dims
                    interaction_outputs.append(interaction)

        elif self.bilinear_type == 'interaction':
            # 每个特征交互对有自己的权重矩阵
            W_list = self.W_list
            for idx, (i, j) in enumerate(self.pairs):
                vdotw = torch.matmul(inputs[:, i, :], W_list[idx])  # batch_size x embed_dims
                interaction = vdotw * inputs[:, j, :]  # batch_size x embed_dims
                interaction_outputs.append(interaction)

        # 拼接所有交互输出
        if len(interaction_outputs) > 1:
            concat_interact = torch.cat(interaction_outputs, dim=1)  # batch_size x (n_pairs * embed_dims)
        else:
            concat_interact = interaction_outputs[0]

        return concat_interact


class _MultiHeadAttention(Layer):
    """与原 Keras MultiHeadAttention 层数值等价的多头注意力（仅支持本项目用到的参数）

    参数布局与 Keras 一致:
        query/key/value kernel: [D, num_heads, key_dim]，bias: [num_heads, key_dim]
        attention_output kernel: [num_heads, key_dim, D_out]，bias: [D_out]
    掩码语义与 Keras 一致:
        - attention_mask: [B, T, S] (或可广播)，1/True 表示可以关注
        - query_mask / value_mask / key_mask: [B, T] / [B, S]，对应原 Keras 实现中随张量隐式传递的掩码
        - 被屏蔽位置在 softmax 前加上 -1e9
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim else key_dim
        self.dropout_rate = dropout
        self.use_bias = use_bias
        self._output_shape = output_shape
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.dropout_layer = nn.Dropout(dropout)

    def build(self, input_shape):
        # input_shape 为 query 的形状
        q_dim = input_shape[-1]
        self._q_dim = q_dim
        self._built_kv = False

    def _build_kv(self, value_dim_in, key_dim_in):
        h, kd, vd = self.num_heads, self.key_dim, self.value_dim
        out_dim = self._output_shape if self._output_shape else self._q_dim
        self.add_weight("query_kernel", (self._q_dim, h, kd), self.kernel_initializer)
        self.add_weight("key_kernel", (key_dim_in, h, kd), self.kernel_initializer)
        self.add_weight("value_kernel", (value_dim_in, h, vd), self.kernel_initializer)
        self.add_weight("output_kernel", (h, vd, out_dim), self.kernel_initializer)
        if self.use_bias:
            self.add_weight("query_bias", (h, kd), self.bias_initializer)
            self.add_weight("key_bias", (h, kd), self.bias_initializer)
            self.add_weight("value_bias", (h, vd), self.bias_initializer)
            self.add_weight("output_bias", (out_dim,), self.bias_initializer)
        self._built_kv = True

    def forward(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        query_mask=None,
        value_mask=None,
        key_mask=None,
    ):
        if key is None:
            key = value
            if key_mask is None:
                key_mask = value_mask
        if not self._built_kv:
            self._build_kv(value.shape[-1], key.shape[-1])

        q = torch.einsum("abc,cde->abde", query, self.query_kernel)  # [B, T, H, K]
        k = torch.einsum("abc,cde->abde", key, self.key_kernel)  # [B, S, H, K]
        v = torch.einsum("abc,cde->abde", value, self.value_kernel)  # [B, S, H, V]
        if self.use_bias:
            q = q + self.query_bias
            k = k + self.key_bias
            v = v + self.value_bias

        q = q * (1.0 / math.sqrt(float(self.key_dim)))
        scores = torch.einsum("aecd,abcd->acbe", k, q)  # [B, H, T, S]

        # 组合掩码（与 Keras _compute_attention_mask 一致）
        mask = None
        if query_mask is not None:
            mask = query_mask.bool().unsqueeze(-1)  # [B, T, 1]
        if value_mask is not None:
            vm = value_mask.bool().unsqueeze(-2)  # [B, 1, S]
            mask = vm if mask is None else (mask & vm)
        if key_mask is not None:
            km = key_mask.bool().unsqueeze(-2)
            mask = km if mask is None else (mask & km)
        if attention_mask is not None:
            am = attention_mask.bool()
            mask = am if mask is None else (mask & am)
        if mask is not None:
            # 在 head 维度上扩展: [B, 1, T, S]
            mask = mask.unsqueeze(-3)
            scores = scores + (1.0 - mask.to(scores.dtype)) * -1e9

        attention_scores = torch.softmax(scores, dim=-1)
        attention_scores_dropout = self.dropout_layer(attention_scores)
        out = torch.einsum("acbe,aecd->abcd", attention_scores_dropout, v)  # [B, T, H, V]
        out = torch.einsum("abcd,cde->abe", out, self.output_kernel)
        if self.use_bias:
            out = out + self.output_bias
        if return_attention_scores:
            return out, attention_scores
        return out


class TransformerEncoder(Layer):
    def __init__(self,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="relu",
        normalize_first=False,
        is_residual=True,
        return_attention_scores=False,
        trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(name=name)
        self.trainable = trainable
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.normalize_first = normalize_first
        self.is_residual = is_residual
        self.return_attention_scores = return_attention_scores

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)

        # self-attention 相关的层
        self.self_attention_layer = _MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            name="self_attention_layer",
        )
        # Keras LayerNormalization 默认 epsilon=1e-3
        self.attention_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-3)
        self.attention_dropout = nn.Dropout(p=self.dropout)

        # feedforward 相关层
        self.feedforward_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-3)
        self.feedforward_intermediate_dense = Dense(self.intermediate_dim, activation=self.activation)
        self.feedforward_output_dense = Dense(hidden_dim, activation=self.activation)
        self.feedforward_dropout = nn.Dropout(p=self.dropout)

        device = getattr(self, "_build_device", None)
        if device is not None:
            self.attention_layer_norm.to(device)
            self.feedforward_layer_norm.to(device)
        if not self.trainable:
            self.attention_layer_norm.requires_grad_(False)
            self.feedforward_layer_norm.requires_grad_(False)

    def _freeze_if_needed(self):
        # trainable=False 时冻结惰性创建的参数（首次前向后执行一次）
        if not self.trainable and not getattr(self, "_frozen", False):
            self.requires_grad_(False)
            self._frozen = True

    def forward(
            self,
            inputs,
            attention_mask=None,
            training=None,
            return_attention_scores=False,
            mask=None, *args, **kwargs):
        """
        Args:
            inputs: [B, L, D]
            attention_mask: [B, L, L]，1/True 表示可以关注
            training: 保留参数以兼容原接口，实际使用 self.training
            return_attention_scores: 是否计算注意力分数
            mask: [B, L] 序列padding掩码（对应原 Keras 实现中隐式传递给 MultiHeadAttention 的掩码，
                同时作为 query 和 value 的掩码），默认为 None
        """

        # self-attention
        x = inputs
        residual = inputs
        if self.normalize_first:
            x = self.attention_layer_norm(x)

        attention_scores = None
        if return_attention_scores or self.return_attention_scores:
            x, attention_scores = self.self_attention_layer(
                query=x,
                value=x,
                attention_mask=attention_mask,
                return_attention_scores=True,
                query_mask=mask,
                value_mask=mask,
            )
        else:
            x = self.self_attention_layer(
                query=x,
                value=x,
                attention_mask=attention_mask,
                query_mask=mask,
                value_mask=mask,
            )

        x = self.attention_dropout(x)
        if self.is_residual:
            x = x + residual

        if not self.normalize_first:
            x = self.attention_layer_norm(x)

        # feed forward
        residual = x
        if self.normalize_first:
            x = self.feedforward_layer_norm(x)
        x = self.feedforward_intermediate_dense(x)
        x = self.feedforward_output_dense(x)
        x = self.feedforward_dropout(x)

        if self.is_residual:
            x = x + residual

        if not self.normalize_first:
            x = self.feedforward_layer_norm(x)

        self._freeze_if_needed()

        if self.return_attention_scores:
            return x, attention_scores

        return x



class UserAttention(Layer):
    """用户注意力层，使用用户基础表示作为查询向量"""
    def __init__(self, **kwargs):
        super(UserAttention, self).__init__(**kwargs)

    def forward(self, query_vector, key_vectors):
        # 计算注意力分数
        attention_scores = torch.matmul(
            query_vector,  # [batch_size, 1, dim]
            key_vectors.transpose(1, 2)   # [batch_size, dim, seq_len]
        )  # [batch_size, 1, seq_len]

        attention_scores = attention_scores.squeeze(1)  # [batch_size, seq_len]

        # 应用softmax获取注意力权重
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]

        # 加权求和
        context_vector = torch.matmul(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            key_vectors  # [batch_size, seq_len, dim]
        )  # [batch_size, 1, dim]

        return context_vector

class GatedFusion(Layer):
    """门控融合层，用于融合长期和短期兴趣"""
    def __init__(self, **kwargs):
        super(GatedFusion, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[0][-1]
        self.W1 = self.add_weight(
            shape=(dim, dim),
            initializer="glorot_uniform",
            trainable=True,
            name="W1"
        )
        self.W2 = self.add_weight(
            shape=(dim, dim),
            initializer="glorot_uniform",
            trainable=True,
            name="W2"
        )
        self.W3 = self.add_weight(
            shape=(dim, dim),
            initializer="glorot_uniform",
            trainable=True,
            name="W3"
        )
        self.b = self.add_weight(
            shape=(dim,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def forward(self, inputs):
        user_embedding, short_term, long_term = inputs

        # 计算门控向量
        gate = torch.sigmoid(
            torch.matmul(user_embedding, self.W1) +
            torch.matmul(short_term, self.W2) +
            torch.matmul(long_term, self.W3) +
            self.b
        )

        # 融合长短期兴趣
        output = (1 - gate) * long_term + gate * short_term

        return output


class BiInteractionPooling(Layer):
    def __init__(self, **kwargs):
        super(BiInteractionPooling, self).__init__(**kwargs)

    def forward(self, inputs, **kwargs):
        # 双线性交互池化: 1/2 * ((\sum_i x_i*v_i)^2 - \sum_i (x_i*v_i)^2)
        sum_of_embeds = torch.sum(inputs, dim=1, keepdim=False) # B x D
        square_of_sum = torch.square(sum_of_embeds) # B x D

        square_of_embeds = torch.square(inputs) # B x n x D
        sum_of_square = torch.sum(square_of_embeds, dim=1, keepdim=False) # B x D

        bi_interaction = 0.5 * (square_of_sum - sum_of_square) # B x D
        return bi_interaction


class APGLayer(Layer):
    """注意力个性化门控层 (Attention Personalized Gating Layer)

    该层实现了基于场景嵌入的注意力个性化机制，通过共享权重和场景特定权重的组合，
    动态调整输入特征的转换过程，支持矩阵分解和不同的权重共享策略。

    参数:
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        scene_emb_dim: 场景嵌入向量维度
        activation: 输出激活函数名称
        generate_activation: 权重生成网络的激活函数
        inner_activation: 内部层激活函数
        use_uv_shared: 是否使用UV共享权重模式
        mf_k: 矩阵分解中K路径的分割因子
        use_mf_p: 是否使用P路径的矩阵分解
        mf_p: 矩阵分解中P路径的分割因子
    """
    def __init__(self, input_dim, output_dim, scene_emb_dim, activation='relu', generate_activation=None,
                 inner_activation=None, use_uv_shared=True, mf_k=16, use_mf_p=True, mf_p=4, **kwargs):
        super(APGLayer, self).__init__(name=kwargs.get("name"))
        self.input_dim = input_dim                  # 输入特征维度
        self.output_dim = output_dim                # 输出特征维度
        self.scene_emb_dim = scene_emb_dim          # 场景嵌入向量维度
        self.use_uv_shared = use_uv_shared          # 是否使用UV共享权重模式
        self.use_mf_p = use_mf_p                    # 是否使用P路径矩阵分解
        self.mf_k = mf_k                            # K路径矩阵分解分割因子
        self.mf_p = mf_p                            # P路径矩阵分解分割因子

        # 激活函数初始化
        self.activation = get_activation(activation) if activation else None
        self.inner_activation = get_activation(inner_activation) if inner_activation else None

        # 计算矩阵分解维度
        min_dim = min(input_dim, output_dim)
        self.p_dim = math.ceil(min_dim / mf_p) if use_mf_p else None  # P路径维度
        self.k_dim = math.ceil(min_dim / mf_k)                        # K路径维度

        # 场景特定KK权重生成器
        # 用于从场景嵌入生成KK权重矩阵和偏置
        kk_weight_size = self.k_dim * self.k_dim
        self.specific_weight_kk = DNNs([kk_weight_size], activation=generate_activation)
        self.specific_bias_kk = DNNs([self.k_dim], activation=generate_activation)

        # 参数形状已知，直接在构造时创建（先放在 CPU 上，随模型 .to(device) 迁移）
        def _w(name, shape, init):
            return self.add_weight(name, shape, initializer=init, device="cpu")

        # 权重初始化: 共享权重模式或场景特定权重模式
        if use_uv_shared:
            # UV共享权重模式: 使用共享矩阵进行特征转换
            if use_mf_p:
                # P路径矩阵分解: NP -> PK -> KK -> KP -> PM
                self.shared_weight_np = _w('shared_weight_np', (input_dim, self.p_dim), 'glorot_uniform')
                self.shared_bias_np = _w('shared_bias_np', (self.p_dim,), 'zeros')
                self.shared_weight_pk = _w('shared_weight_pk', (self.p_dim, self.k_dim), 'glorot_uniform')
                self.shared_bias_pk = _w('shared_bias_pk', (self.k_dim,), 'zeros')
                self.shared_weight_kp = _w('shared_weight_kp', (self.k_dim, self.p_dim), 'glorot_uniform')
                self.shared_bias_kp = _w('shared_bias_kp', (self.p_dim,), 'zeros')
                self.shared_weight_pm = _w('shared_weight_pm', (self.p_dim, output_dim), 'glorot_uniform')
                self.shared_bias_pm = _w('shared_bias_pm', (output_dim,), 'zeros')
            else:
                # 无P路径矩阵分解: NK -> KK -> KM
                self.shared_weight_nk = _w('shared_weight_nk', (input_dim, self.k_dim), 'glorot_uniform')
                self.shared_bias_nk = _w('shared_bias_nk', (self.k_dim,), 'zeros')
                self.shared_weight_km = _w('shared_weight_km', (self.k_dim, output_dim), 'glorot_uniform')
                self.shared_bias_km = _w('shared_bias_km', (output_dim,), 'zeros')
        else:
            # 场景特定权重模式: NK和KM权重由场景嵌入生成
            nk_weight_size = input_dim * self.k_dim
            km_weight_size = self.k_dim * output_dim
            self.specific_weight_nk = DNNs([nk_weight_size], activation=generate_activation)
            self.specific_bias_nk = DNNs([self.k_dim], activation=generate_activation)
            self.specific_weight_km = DNNs([km_weight_size], activation=generate_activation)
            self.specific_bias_km = DNNs([output_dim], activation=generate_activation)

    def forward(self, inputs):
        """前向传播方法

        参数:
            inputs: 包含两个元素的列表 [x, scene_emb]
                x: 输入特征张量，形状为 (batch_size, input_dim)
                scene_emb: 场景嵌入张量，形状为 (batch_size, scene_emb_dim)

        返回:
            output: 经过注意力个性化门控处理的输出张量，形状为 (batch_size, output_dim)
        """
        x, scene_emb = inputs  # x: 输入特征, scene_emb: 场景嵌入

        # 生成场景特定KK权重矩阵和偏置
        specific_weight_kk = self.specific_weight_kk(scene_emb)  # 形状: (batch_size, k_dim*k_dim)
        specific_weight_kk = torch.reshape(specific_weight_kk, (-1, self.k_dim, self.k_dim))  # 重塑为矩阵
        specific_bias_kk = self.specific_bias_kk(scene_emb)  # KK偏置

        if self.use_uv_shared:
            # UV共享权重模式下的前向传播
            if self.use_mf_p:
                # P路径矩阵分解路径: NP -> PK -> KK -> KP -> PM
                # 1. NP: 输入特征到P维度空间
                output_np = torch.matmul(x, self.shared_weight_np) + self.shared_bias_np
                if self.inner_activation: output_np = self.inner_activation(output_np)

                # 2. PK: P维度到K维度空间
                output_pk = torch.matmul(output_np, self.shared_weight_pk) + self.shared_bias_pk
                if self.inner_activation: output_pk = self.inner_activation(output_pk)

                # 3. KK: 应用场景特定KK权重
                output_kk = torch.matmul(output_pk.unsqueeze(1), specific_weight_kk)
                output_kk = output_kk.squeeze(1) + specific_bias_kk
                if self.inner_activation: output_kk = self.inner_activation(output_kk)

                # 4. KP: K维度到P维度空间
                output_kp = torch.matmul(output_kk, self.shared_weight_kp) + self.shared_bias_kp
                if self.inner_activation: output_kp = self.inner_activation(output_kp)

                # 5. PM: P维度到输出维度
                output = torch.matmul(output_kp, self.shared_weight_pm) + self.shared_bias_pm
            else:
                # 无P路径矩阵分解路径: NK -> KK -> KM
                # 1. NK: 输入特征到K维度空间
                output_nk = torch.matmul(x, self.shared_weight_nk) + self.shared_bias_nk
                if self.inner_activation: output_nk = self.inner_activation(output_nk)

                # 2. KK: 应用场景特定KK权重
                output_kk = torch.matmul(output_nk.unsqueeze(1), specific_weight_kk)
                output_kk = output_kk.squeeze(1) + specific_bias_kk
                if self.inner_activation: output_kk = self.inner_activation(output_kk)

                # 3. KM: K维度到输出维度
                output = torch.matmul(output_kk, self.shared_weight_km) + self.shared_bias_km
        else:
            # 场景特定权重模式下的前向传播: NK -> KK -> KM
            # 1. NK: 生成场景特定NK权重并应用
            specific_weight_nk = self.specific_weight_nk(scene_emb)
            specific_weight_nk = torch.reshape(specific_weight_nk, (-1, self.input_dim, self.k_dim))
            specific_bias_nk = self.specific_bias_nk(scene_emb)

            output_nk = torch.matmul(x.unsqueeze(1), specific_weight_nk)
            output_nk = output_nk.squeeze(1) + specific_bias_nk
            if self.inner_activation: output_nk = self.inner_activation(output_nk)

            # 2. KK: 应用场景特定KK权重
            output_kk = torch.matmul(output_nk.unsqueeze(1), specific_weight_kk)
            output_kk = output_kk.squeeze(1) + specific_bias_kk
            if self.inner_activation: output_kk = self.inner_activation(output_kk)

            # 3. KM: 生成场景特定KM权重并应用
            specific_weight_km = self.specific_weight_km(scene_emb)
            specific_weight_km = torch.reshape(specific_weight_km, (-1, self.k_dim, self.output_dim))
            specific_bias_km = self.specific_bias_km(scene_emb)

            output = torch.matmul(output_kk.unsqueeze(1), specific_weight_km)
            output = output.squeeze(1) + specific_bias_km

        # 应用输出激活函数
        if self.activation: output = self.activation(output)
        return output


class MetaUnit(Layer):
    """
    Reference:
        Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling
    """
    def __init__(self,
                 num_layer,
                 activation="leaky_relu",
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        super(MetaUnit, self).__init__(name=kwargs.get("name"))
        self.num_layer = num_layer
        self.l2_reg = l2_reg

        self.weights_dense = nn.ModuleList()
        self.bias_dense = nn.ModuleList()
        self.activation_list = nn.ModuleList([get_activation(activation) for _ in range(num_layer)])
        self.dropout_list = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layer)])

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_size = input_shape[0][-1]
        self.input_size = input_size

        for i in range(self.num_layer):
            self.weights_dense.append(
                Dense(input_size*input_size, kernel_regularizer=self.l2_reg)
            )
            self.bias_dense.append(
                Dense(input_size, kernel_regularizer=self.l2_reg)
            )

    def forward(self, inputs, **kwargs):
        inputs, scenario_views = inputs

        # [bs, 1, dim]
        squeeze = False
        if inputs.dim() == 2:
            squeeze = True
            inputs = inputs.unsqueeze(1)

        output = inputs
        for i in range(self.num_layer):
            # [bs, dim*dim]
            w = self.weights_dense[i](scenario_views)
            b = self.bias_dense[i](scenario_views)

            # [bs, dim, dim]
            w = torch.reshape(w, [-1, self.input_size, self.input_size])
            b = b.unsqueeze(1)

            # [bs, 1, dim] * [bs, dim, dim] = [bs, 1, dim]
            fc = torch.matmul(output, w) + b

            output = self.activation_list[i](fc)

            output = self.dropout_list[i](output)

        # [bs, dim]
        if squeeze:
            return output.squeeze(1)
        else:
            return output


class MetaAttention(Layer):
    """
    Reference:
        Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling
    """
    def __init__(self,
                 meta_unit=None,
                 num_layer=3,
                 activation="leaky_relu",
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        super(MetaAttention, self).__init__(name=kwargs.get("name"))
        if meta_unit is not None:
            self.meta_unit = meta_unit
        else:
            self.meta_unit = MetaUnit(num_layer, activation, dropout, l2_reg)
        self.dense = Dense(1)

    def forward(self, inputs, **kwargs):
        expert_views, task_views, scenario_views = inputs
        task_views = task_views.unsqueeze(1).expand(-1, expert_views.shape[1], -1)
        # [bs, num_experts, dim]
        meta_unit_output = self.meta_unit([torch.cat([expert_views, task_views], dim=-1), scenario_views])
        # [bs, num_experts, 1]
        score = self.dense(meta_unit_output)
        # [bs, dim]
        output = torch.sum(expert_views * score, dim=1)

        return output


class MetaTower(Layer):
    """
    Reference:
        Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling
    """
    def __init__(self,
                 meta_unit=None,
                 num_layer=3,
                 meta_unit_depth=3,
                 activation="leaky_relu",
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        super(MetaTower, self).__init__(name=kwargs.get("name"))
        if meta_unit is not None:
            self.layers = nn.ModuleList([meta_unit] * num_layer)  # 列表中的每个元素都是同一个meta_unit对象
        else:
            self.layers = nn.ModuleList([MetaUnit(meta_unit_depth, activation, dropout, l2_reg) for _ in range(num_layer)])
        self.activation_list = nn.ModuleList([get_activation(activation) for _ in range(num_layer)])
        self.dropout_list = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layer)])

    def forward(self, inputs, **kwargs):
        inputs, scenario_views = inputs

        output = inputs
        for i in range(len(self.layers)):
            output = self.layers[i]([output, scenario_views])
            output = self.activation_list[i](output)
            output = self.dropout_list[i](output)

        return output


class TaskEmbedding(Layer):
    """
    Reference:
        Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling
    """
    # 创建M2M模型每个任务的可学习向量（不依赖显式的 `task` 特征列），用自定义层以避免Lambda创建变量问题
    def __init__(self, output_dim, **kwargs):
        super().__init__(name=kwargs.get("name"))
        self.emb = Embedding(input_dim=1, output_dim=output_dim, name=f"{self.layer_name}_table")

    def forward(self, ref_tensor):
        batch = ref_tensor.shape[0]
        idx = torch.zeros((batch, 1), dtype=torch.long, device=ref_tensor.device)
        return self.emb(idx).squeeze(1)
