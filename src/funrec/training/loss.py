"""
损失函数 (PyTorch)

所有损失函数签名统一为 loss(y_true, y_pred) -> 标量张量，与原 Keras 损失函数保持一致。
"""

import torch
import torch.nn.functional as F

_EPSILON = 1e-7  # 与 Keras backend.epsilon() 一致


def _align(y_true, y_pred):
    """将标签对齐到预测的形状（Keras 会自动扩展 B -> B x 1）"""
    y_true = y_true.to(y_pred.dtype) if y_true.is_floating_point() or y_pred.is_floating_point() else y_true
    if y_true.shape != y_pred.shape and y_true.numel() == y_pred.numel():
        y_true = y_true.reshape(y_pred.shape)
    return y_true


def _keras_logits(y_pred, op_type):
    """Keras 的 binary/categorical_crossentropy 在 y_pred 直接由 sigmoid/softmax 算子产生时
    （如 PredictLayer 作为最后一层），会取出其输入 logits 按 from_logits=True 计算（不做裁剪）。
    这里约定: 产生概率的层可在输出张量上设置属性 `_keras_logits`（logits）与
    `_keras_logits_op`（"Sigmoid" 或 "Softmax"）以复现该行为
    （之后经过 reshape/flatten 等操作得到的新张量不再携带该属性，与 Keras 相同）。
    """
    logits = getattr(y_pred, "_keras_logits", None)
    if logits is None or getattr(y_pred, "_keras_logits_op", None) != op_type:
        return None
    return logits


def binary_crossentropy(y_true, y_pred):
    """二分类交叉熵，y_pred 为 sigmoid 之后的概率（与 Keras backend.binary_crossentropy 一致）"""
    y_true = _align(y_true.float(), y_pred)
    logits = _keras_logits(y_pred, "Sigmoid")
    if logits is not None and logits.shape == y_pred.shape:
        return F.binary_cross_entropy_with_logits(logits, y_true)
    y_pred = torch.clamp(y_pred, _EPSILON, 1.0 - _EPSILON)
    # Keras: target * log(output + eps) + (1 - target) * log(1 - output + eps)
    loss = -(
        y_true * torch.log(y_pred + _EPSILON)
        + (1.0 - y_true) * torch.log(1.0 - y_pred + _EPSILON)
    )
    return loss.mean()


def categorical_crossentropy(y_true, y_pred):
    """多分类交叉熵，y_true 为 one-hot（或概率分布），y_pred 为 softmax 概率"""
    y_true = _align(y_true.float(), y_pred)
    logits = _keras_logits(y_pred, "Softmax")
    if logits is not None and logits.shape == y_pred.shape:
        # 与 tf.nn.softmax_cross_entropy_with_logits 一致: 数值为 -sum(y * log_softmax(z))，
        # 但其融合算子对 logits 的梯度固定为 softmax(z) - y（假设每行标签和为 1）。
        # 当标签为多热/全零（如 PRM 的点击序列）时该梯度与数学梯度 sum(y)*softmax(z) - y 不同，这里复现 TF 的梯度
        value = -torch.sum(y_true * F.log_softmax(logits, dim=-1), dim=-1)
        surrogate = torch.logsumexp(logits, dim=-1) - torch.sum(y_true * logits, dim=-1)
        return (value.detach() + surrogate - surrogate.detach()).mean()
    y_pred = y_pred / torch.sum(y_pred, dim=-1, keepdim=True)
    y_pred = torch.clamp(y_pred, _EPSILON, 1.0 - _EPSILON)
    return (-torch.sum(y_true * torch.log(y_pred), dim=-1)).mean()


def sparse_categorical_crossentropy(y_true, y_pred):
    """稀疏多分类交叉熵，y_true 为类别索引，y_pred 为 softmax 概率"""
    y_true = y_true.long().reshape(-1)
    y_pred = torch.clamp(y_pred.reshape(y_true.shape[0], -1), _EPSILON, 1.0 - _EPSILON)
    return F.nll_loss(torch.log(y_pred), y_true)


def mean_squared_error(y_true, y_pred):
    y_true = _align(y_true.float(), y_pred)
    return torch.mean((y_pred - y_true) ** 2)


def mean_absolute_error(y_true, y_pred):
    y_true = _align(y_true.float(), y_pred)
    return torch.mean(torch.abs(y_pred - y_true))


def contrastive_loss(y_true, y_pred, temperature=0.1):
    """
    Batch内对比损失 (InfoNCE)。
    参数:
        y_true: 未使用，但为了与Keras接口兼容而保留
        y_pred: 来自模型的余弦相似度分数 (B x B)
        temperature: 温度参数控制分布的集中度
    返回:
        对比损失值
    """
    batch_size = y_pred.shape[0]

    # 通过温度参数缩放相似度
    scaled_sim = y_pred / temperature

    # 为正样本对创建掩码（对角线元素）
    pos_mask = torch.eye(batch_size, device=y_pred.device, dtype=y_pred.dtype)

    # 计算log softmax
    log_softmax = scaled_sim - torch.log(torch.sum(torch.exp(scaled_sim), dim=1, keepdim=True))

    # 计算损失作为选择正样本的负对数似然
    loss = -torch.sum(pos_mask * log_softmax, dim=1)

    return torch.mean(loss)


def sampledsoftmaxloss(y_true, y_pred):
    """因为在模型构建的时候使用了sampled_softmax_loss,这里只需要计算mean就可以了"""
    return torch.mean(y_pred)


def sum_loss(y_true, y_pred):
    return torch.sum(y_pred)


def mean_loss(y_true, y_pred):
    return torch.mean(y_pred)


LOSSES = {
    "binary_crossentropy": binary_crossentropy,
    "bce": binary_crossentropy,
    "categorical_crossentropy": categorical_crossentropy,
    "sparse_categorical_crossentropy": sparse_categorical_crossentropy,
    "mse": mean_squared_error,
    "mean_squared_error": mean_squared_error,
    "mae": mean_absolute_error,
    "mean_absolute_error": mean_absolute_error,
    "sampledsoftmaxloss": sampledsoftmaxloss,
    "contrastive_loss": contrastive_loss,
    "sum_loss": sum_loss,
    "mean_loss": mean_loss,
}


def get_loss(loss):
    """根据名称或 callable 返回损失函数"""
    if callable(loss):
        return loss
    if isinstance(loss, str) and loss in LOSSES:
        return LOSSES[loss]
    raise ValueError(f"不支持的损失函数: {loss}")
