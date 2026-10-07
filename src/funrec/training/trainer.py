"""
模型训练
"""

import importlib
from typing import Dict, Any, List, Tuple, Union, Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..features.feature_column import FeatureColumn
from ..features.processors import apply_training_preprocessing
from ..models.base import get_device, num_samples, slice_features, to_tensor
from .loss import get_loss


def train_model(
    training_config: Dict[str, Any],
    feature_columns: List[FeatureColumn],
    processed_data: Dict[str, Any],
) -> Union[Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module], Any]:
    """
    基于配置和处理后的数据训练模型。
    参数:
        training_config: 训练配置字典，包含以下内容：
            - build_function: 模型构建函数的完整路径 (例如：'funrec.models.dssm.build_dssm_model')
            - model_params: 模型特定参数
            - classical_model: 布尔值，指示是否为经典模型 (默认: False)
            - data_preprocessing: 要应用的预处理规则列表
            - optimizer: 使用的优化器 (默认: 'adam')
            - loss: 损失函数 (默认: ['binary_crossentropy'])
            - metrics: 要跟踪的指标 (默认: ['binary_accuracy'])
            - batch_size: 训练批次大小 (默认: 1024)
            - epochs: 训练轮数 (默认: 1)
            - validation_split: 验证集分割比例 (默认: 0.2)
            - verbose: 训练详细程度 (默认: 1)
        feature_columns: 特征列规范列表
        processed_data: 处理后的数据字典，包含训练/测试特征和标签
    返回:
        对于神经网络模型: 元组 (main_model, user_model, item_model)
        对于经典模型: 训练后的经典模型实例
    """
    # 从配置中获取构建函数路径和参数
    build_function_path = training_config.get(
        "build_function", "funrec.models.dssm.build_dssm_model"
    )
    model_params = training_config.get("model_params", {})
    is_classical = training_config.get("classical_model", False)
    is_external_embedding = training_config.get("embedding_external", False)

    # 解析模块和函数名
    module_path, function_name = build_function_path.rsplit(".", 1)

    # 动态导入并调用构建函数
    module = importlib.import_module(module_path)
    build_function = getattr(module, function_name)

    if is_classical:
        # 经典模型：直接构建和拟合
        model = build_function(feature_columns, model_params)

        # 经典模型：准备交互数据
        # 经典模型期望用户-物品交互：[(user_id, item_id, label), ...]
        train_interactions = []
        # 当特征配置为空时，processed_data已经直接包含训练字典
        train_features = (
            processed_data["train"]["features"]
            if "features" in processed_data["train"]
            else processed_data["train"]
        )
        train_labels = processed_data["train"]["labels"]

        # 从特征中提取用户和物品ID
        # 假设第一个特征是user_id，第二个是item_id
        user_ids = (
            train_features[0]
            if isinstance(train_features, list)
            else train_features["user_id"]
        )
        item_ids = (
            train_features[1]
            if isinstance(train_features, list)
            else train_features["item_id"]
        )

        # 转换为交互格式
        for i in range(len(user_ids)):
            train_interactions.append((user_ids[i], item_ids[i], train_labels[i]))

        # 训练经典模型
        model.fit(train_interactions)

        # 返回模型和None作为用户和物品模型（经典模型）
        return model, None, None

    elif is_external_embedding:
        # 外部嵌入模型（例如Item2Vec）从用户历史序列训练
        # 使用自己的参数签名构建模型
        model = build_function(model_params)

        # 从处理后的数据准备训练序列
        train_features = (
            processed_data["train"]["features"]
            if "features" in processed_data["train"]
            else processed_data["train"]
        )
        # 支持多个潜在键
        hist_key_candidates = [
            "hist_movie_id_list",
            "hist_movie_ids",
            "hist_item_id_list",
            "hist_item_ids",
        ]
        hist_array = None
        for key in hist_key_candidates:
            if key in train_features:
                hist_array = train_features[key]
                break
        if hist_array is None:
            raise ValueError(
                "外部嵌入训练需要训练特征中包含'hist_movie_id_list'（或兼容键）"
            )

        # 转换为token序列列表（过滤填充0）
        try:
            import numpy as np

            if isinstance(hist_array, list):
                train_sequences = [np.array(seq) for seq in hist_array]
            else:
                train_sequences = [hist_array[i] for i in range(len(hist_array))]
            train_hist_sequences = [
                seq[np.where(seq != 0)[0]].tolist() for seq in train_sequences
            ]
        except Exception:
            train_hist_sequences = []
            for seq in hist_array:
                try:
                    train_hist_sequences.append([token for token in seq if token != 0])
                except Exception:
                    train_hist_sequences.append(list(seq))

        model.fit(train_hist_sequences)

        # 以统一元组形式返回
        return model, None, None

    else:
        # 神经网络模型：PyTorch 训练流水线（语义与原 Keras compile/fit 对齐）
        model, user_model, item_model = build_function(feature_columns, model_params)

        optimizer_name = training_config.get("optimizer", "adam")
        optimizer_params = training_config.get("optimizer_params", {})
        loss = training_config.get("loss", ["binary_crossentropy"])
        loss_weights = training_config.get("loss_weights", None)

        # 获取训练参数
        batch_size = training_config.get("batch_size", 1024)
        epochs = training_config.get("epochs", 1)
        validation_split = training_config.get("validation_split", 0.2)
        verbose = training_config.get("verbose", 0)

        # 基于配置应用训练特定的预处理
        # 支持完全准备的字典和原始字典
        if "features" in processed_data["train"]:
            train_features = processed_data["train"]["features"]
            train_labels = processed_data["train"].get("labels")
        else:
            train_features = processed_data["train"]
            train_labels = (
                processed_data["train"].get("labels")
                if isinstance(processed_data["train"], dict)
                else None
            )
        train_features, train_labels = apply_training_preprocessing(
            training_config, train_features, train_labels
        )

        labels_for_fit = train_labels
        if isinstance(labels_for_fit, list) and len(labels_for_fit) == 1:
            labels_for_fit = labels_for_fit[0]

        # 处理多输出模型：如果模型有多个输出但只有一个标签集，复制标签
        if (
            isinstance(loss, list)
            and len(loss) > 1
            and not isinstance(labels_for_fit, list)
        ):
            # 对于PRS等模型：两个输出都使用相同的标签
            labels_for_fit = [labels_for_fit] * len(loss)

        fit_model(
            model,
            train_features,
            labels_for_fit,
            loss=loss,
            loss_weights=loss_weights,
            optimizer=optimizer_name,
            optimizer_params=optimizer_params,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
        )

        return model, user_model, item_model


class KerasAdam(torch.optim.Optimizer):
    """与 tf.keras.optimizers.Adam 数值一致的 Adam

    与 torch.optim.Adam 的区别: Keras 将 epsilon 加在未做偏差修正的 sqrt(v) 上
        alpha = lr * sqrt(1 - beta_2^t) / (1 - beta_1^t)
        var -= alpha * m / (sqrt(v) + epsilon)
    （torch 将 eps 加在修正后的 sqrt(v_hat) 上，梯度很小时（如嵌入表）更新量差异很大）；
    且步数 t 为优化器全局迭代次数（Keras optimizer.iterations），而非每个参数各自的步数。
    """

    def __init__(self, params, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False):
        defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.iterations = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        t = self.iterations + 1
        for group in self.param_groups:
            beta_1, beta_2, eps = group["beta_1"], group["beta_2"], group["epsilon"]
            alpha = group["lr"] * (1.0 - beta_2 ** t) ** 0.5 / (1.0 - beta_1 ** t)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    grad = grad.to_dense()
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["v_hat"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                m.add_((grad - m) * (1.0 - beta_1))
                v.add_((grad * grad - v) * (1.0 - beta_2))
                if group["amsgrad"]:
                    torch.maximum(state["v_hat"], v, out=state["v_hat"])
                    v = state["v_hat"]
                p.sub_((m * alpha) / (v.sqrt() + eps))
        self.iterations += 1
        return loss


def build_optimizer(model: torch.nn.Module, optimizer_name: str = "adam", optimizer_params: Dict[str, Any] = None):
    """构建优化器，参数名兼容 Keras（learning_rate/epsilon/beta_1/beta_2）"""
    params = dict(optimizer_params or {})
    lr = params.pop("learning_rate", params.pop("lr", None))
    trainable = [p for p in model.parameters() if p.requires_grad]
    name = (optimizer_name or "adam").lower()
    if name == "adam":
        kwargs = {
            "lr": 1e-3 if lr is None else lr,
            "beta_1": params.pop("beta_1", 0.9),
            "beta_2": params.pop("beta_2", 0.999),
            "epsilon": params.pop("epsilon", 1e-7),  # Keras 默认 epsilon
            "amsgrad": params.pop("amsgrad", False),
        }
        kwargs.update(params)
        return KerasAdam(trainable, **kwargs)
    if name == "adagrad":
        return torch.optim.Adagrad(trainable, lr=1e-3 if lr is None else lr, initial_accumulator_value=0.1, eps=1e-7, **params)
    if name == "sgd":
        return torch.optim.SGD(trainable, lr=1e-2 if lr is None else lr, **params)
    if name == "rmsprop":
        return torch.optim.RMSprop(trainable, lr=1e-3 if lr is None else lr, alpha=0.9, eps=1e-7, **params)
    raise ValueError(f"不支持的优化器: {optimizer_name}")


def _as_output_list(outputs):
    return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]


def fit_model(
    model,
    features: Dict[str, Any],
    labels: Any,
    loss: Union[str, Callable, List] = "binary_crossentropy",
    loss_weights: Optional[List[float]] = None,
    optimizer: str = "adam",
    optimizer_params: Optional[Dict[str, Any]] = None,
    batch_size: int = 1024,
    epochs: int = 1,
    verbose: int = 0,
    validation_split: float = 0.0,
    shuffle: bool = True,
) -> Dict[str, List[float]]:
    """训练模型（对应 Keras model.compile + model.fit）

    - validation_split: 与 Keras 一致，取数据末尾的比例作为验证集（不打乱）
    - shuffle: 每个 epoch 打乱训练集
    - 总损失 = sum(loss_weight_i * loss_i(y_i, output_i)) + 模型辅助损失(add_loss) + L2 正则
    返回: history 字典 {"loss": [...], "val_loss": [...]}
    """
    device = get_device()
    model.to(device)

    n = num_samples(features)
    if validation_split and 0 < validation_split < 1:
        split_at = int(n * (1.0 - validation_split))
    else:
        split_at = n
    train_idx_all = np.arange(split_at)
    val_idx_all = np.arange(split_at, n)

    label_list = labels if isinstance(labels, (list, tuple)) else [labels]
    label_list = [None if l is None else np.asarray(l) for l in label_list]

    # 先用一个批次数据前向一次，创建惰性参数，再创建优化器
    model.build_with(slice_features(features, train_idx_all[: min(batch_size, len(train_idx_all))]), batch_size=min(batch_size, len(train_idx_all)))
    opt = build_optimizer(model, optimizer, optimizer_params)

    def compute_loss(outputs, batch_labels):
        outputs = _as_output_list(outputs)
        losses = loss if isinstance(loss, (list, tuple)) else [loss] * len(outputs)
        if len(losses) == 1 and len(outputs) > 1:
            losses = list(losses) * len(outputs)
        weights = loss_weights if loss_weights is not None else [1.0] * len(outputs)
        if isinstance(weights, dict):
            names = model.output_names or []
            weights = [weights.get(nm, 1.0) for nm in names]
        total = None
        for i, out in enumerate(outputs):
            fn = get_loss(losses[i])
            y = batch_labels[i] if i < len(batch_labels) else batch_labels[-1]
            term = weights[i] * fn(y, out)
            total = term if total is None else total + term
        return total

    def batch_labels_of(idx):
        res = []
        for l in label_list:
            if l is None:
                res.append(torch.zeros(len(idx), 1, device=device))
            else:
                res.append(to_tensor(l[idx], device))
        return res

    history = {"loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(train_idx_all) if shuffle else train_idx_all
        total_loss, steps = 0.0, 0
        iterator = range(0, len(order), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for start in iterator:
            idx = order[start : start + batch_size]
            batch = model.prepare_inputs(slice_features(features, idx), device)
            outputs = model(batch)
            loss_value = compute_loss(outputs, batch_labels_of(idx))
            for extra in model.pop_extra_losses():
                loss_value = loss_value + extra
            reg = model.regularization_loss()
            if reg is not None:
                loss_value = loss_value + reg
            opt.zero_grad(set_to_none=True)
            loss_value.backward()
            opt.step()
            # 与 Keras 一致: epoch 损失为按样本数加权的批次损失均值
            total_loss += float(loss_value.detach()) * len(idx)
            steps += len(idx)
        history["loss"].append(total_loss / max(steps, 1))

        if len(val_idx_all) > 0:
            model.eval()
            val_total, val_steps = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(val_idx_all), batch_size):
                    idx = val_idx_all[start : start + batch_size]
                    batch = model.prepare_inputs(slice_features(features, idx), device)
                    val_value = compute_loss(model(batch), batch_labels_of(idx))
                    reg = model.regularization_loss()
                    if reg is not None:  # Keras 的 val_loss 同样包含正则损失
                        val_value = val_value + reg
                    val_total += float(val_value) * len(idx)
                    val_steps += len(idx)
            history["val_loss"].append(val_total / max(val_steps, 1))
        if verbose:
            msg = f"Epoch {epoch + 1}/{epochs} - loss: {history['loss'][-1]:.4f}"
            if history["val_loss"]:
                msg += f" - val_loss: {history['val_loss'][-1]:.4f}"
            print(msg)
    model.eval()
    model.history = history
    return history
