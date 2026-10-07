"""
PyTorch 模型基础设施

提供与原 Keras 实现语义对齐的基础组件:
- Layer: 惰性构建的层基类（首次调用时根据输入形状创建参数，等价于 Keras 的 build）
- Dense / Embedding: 与 Keras 默认初始化方式和 L2 正则保持一致的线性层与嵌入层
- FunRecModel: 所有深度模型的基类，提供 predict / add_loss / 正则损失 / 子塔(SubModel)
- SubModel: 与主模型共享参数的子模型（用户塔、物品塔等），用于评估和线上服务
- save_model / load_model: 模型保存与加载
"""

import importlib
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEVICE = None


def get_device() -> torch.device:
    """返回全局计算设备，优先使用 GPU"""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def set_device(device: Union[str, torch.device]) -> None:
    """手动指定全局计算设备，例如 set_device('cpu')"""
    global _DEVICE
    _DEVICE = torch.device(device)


# ---------------------------------------------------------------------------
# 初始化器（与 Keras 默认值保持一致）
# ---------------------------------------------------------------------------


def _fan_in_out(shape):
    if len(shape) < 1:
        return 1, 1
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) == 2:
        return shape[0], shape[1]
    receptive = int(np.prod(shape[:-2]))
    return shape[-2] * receptive, shape[-1] * receptive


def init_tensor_(tensor: torch.Tensor, initializer: Any = "glorot_uniform", shape=None):
    """按照 Keras 命名的初始化器原地初始化张量

    shape 为 Keras 视角下的形状（kernel 为 [in, out]），用于计算 fan_in/fan_out。
    支持: glorot_uniform / glorot_normal / he_uniform / he_normal / lecun_uniform /
    uniform / random_uniform / normal / random_normal / truncated_normal / zeros / ones，
    或任意 callable(tensor)。
    """
    if callable(initializer) and not isinstance(initializer, str):
        with torch.no_grad():
            initializer(tensor)
        return tensor
    name = (initializer or "glorot_uniform").lower()
    shape = tuple(shape) if shape is not None else tuple(tensor.shape)
    fan_in, fan_out = _fan_in_out(shape)
    with torch.no_grad():
        if name in ("glorot_uniform", "xavier_uniform"):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            tensor.uniform_(-limit, limit)
        elif name in ("glorot_normal", "xavier_normal"):
            std = np.sqrt(2.0 / (fan_in + fan_out))
            nn.init.trunc_normal_(tensor, 0.0, std / 0.87962566103423978, -2 * std / 0.87962566103423978, 2 * std / 0.87962566103423978)
        elif name == "he_uniform":
            limit = np.sqrt(6.0 / fan_in)
            tensor.uniform_(-limit, limit)
        elif name == "he_normal":
            std = np.sqrt(2.0 / fan_in) / 0.87962566103423978
            nn.init.trunc_normal_(tensor, 0.0, std, -2 * std, 2 * std)
        elif name == "lecun_uniform":
            limit = np.sqrt(3.0 / fan_in)
            tensor.uniform_(-limit, limit)
        elif name in ("uniform", "random_uniform"):
            tensor.uniform_(-0.05, 0.05)
        elif name in ("normal", "random_normal"):
            tensor.normal_(0.0, 0.05)
        elif name == "truncated_normal":
            nn.init.trunc_normal_(tensor, 0.0, 0.05, -0.1, 0.1)
        elif name == "zeros":
            tensor.zero_()
        elif name == "ones":
            tensor.fill_(1.0)
        else:
            raise ValueError(f"不支持的初始化器: {initializer}")
    return tensor


def get_activation(activation):
    """根据名称返回激活函数模块（None 表示恒等映射）"""
    if activation is None or activation == "linear":
        return nn.Identity()
    if isinstance(activation, nn.Module):
        return activation
    if callable(activation) and not isinstance(activation, str):
        return _FuncActivation(activation)
    name = activation.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "softmax":
        return nn.Softmax(dim=-1)
    if name == "gelu":
        return nn.GELU()
    if name in ("swish", "silu"):
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    if name == "softplus":
        return nn.Softplus()
    if name == "prelu":
        from .layers import PReLU

        return PReLU()
    if name == "dice":
        from .layers import Dice

        return Dice()
    # 其余 Keras 内置激活函数名（与 tf.keras.activations 语义一致）
    if name == "relu6":
        return nn.ReLU6()
    if name == "softsign":
        return nn.Softsign()
    if name == "mish":
        return nn.Mish()
    if name == "exponential":
        return _FuncActivation(torch.exp)
    if name == "hard_sigmoid":
        return _FuncActivation(lambda x: torch.clamp(0.2 * x + 0.5, 0.0, 1.0))
    raise ValueError(f"不支持的激活函数: {activation}")


class _FuncActivation(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


# ---------------------------------------------------------------------------
# 惰性构建层
# ---------------------------------------------------------------------------


def _shape_of(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [_shape_of(v) for v in x]
    if isinstance(x, dict):
        return {k: _shape_of(v) for k, v in x.items()}
    return None


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


class Layer(nn.Module):
    """惰性构建层基类，等价于 Keras Layer 的 build/call 机制

    子类实现:
        build(self, input_shape): 根据首个位置参数的形状创建参数，使用 self.add_weight
        forward(self, inputs, ...): 前向计算

    首次调用时自动执行 build。参数创建在输入所在设备上。
    注意: 含有惰性参数的模型必须先用一个批次数据前向一次（训练器会自动完成），
    之后再创建优化器或加载 state_dict。
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__()
        self.layer_name = name
        self.built = False
        self._l2_weights: List[tuple] = []

    def build(self, input_shape):  # pragma: no cover - 默认无参数
        pass

    def add_weight(
        self,
        name: str,
        shape,
        initializer: Any = "glorot_uniform",
        regularizer: Optional[float] = None,
        trainable: bool = True,
        device=None,
        dtype=torch.float32,
    ) -> nn.Parameter:
        """创建并注册参数（对应 Keras add_weight）

        regularizer: L2 正则系数（float），等价于 tf.keras.regularizers.l2(x)
        """
        shape = tuple(int(s) for s in shape)
        device = device or getattr(self, "_build_device", None) or get_device()
        tensor = torch.empty(shape, device=device, dtype=dtype)
        init_tensor_(tensor, initializer, shape)
        param = nn.Parameter(tensor, requires_grad=trainable)
        self.register_parameter(name.replace("/", "_").replace(".", "_"), param)
        if regularizer:
            self._l2_weights.append((name.replace("/", "_").replace(".", "_"), float(regularizer)))
        return param

    def __call__(self, *args, **kwargs):
        if not self.built:
            first = args[0] if args else next(iter(kwargs.values()), None)
            t = _first_tensor(first)
            self._build_device = t.device if t is not None else get_device()
            self.build(_shape_of(first))
            self.built = True
            # build 中新建的子模块默认处于 train 模式，需与当前层的 train/eval 状态保持一致
            # （否则在 eval 模式下首次前向时，Dropout/BN 等子模块仍按训练模式运行）
            for child in self.children():
                if child.training != self.training:
                    child.train(self.training)
        return super().__call__(*args, **kwargs)

    def regularization_loss(self) -> torch.Tensor:
        loss = None
        for pname, coef in self._l2_weights:
            p = getattr(self, pname)
            term = coef * torch.sum(p * p)
            loss = term if loss is None else loss + term
        return loss


class Dense(Layer):
    """全连接层，等价于 tf.keras.layers.Dense（惰性推断输入维度）

    默认 kernel 初始化 glorot_uniform，bias 初始化为 0，与 Keras 一致。
    支持对任意维度输入的最后一维做线性变换。
    """

    def __init__(
        self,
        units: int,
        activation=None,
        use_bias: bool = True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer: Optional[float] = None,
        bias_regularizer: Optional[float] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.units = int(units)
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = _l2_coef(kernel_regularizer)
        self.bias_regularizer = _l2_coef(bias_regularizer)

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.kernel = self.add_weight(
            "kernel", (in_dim, self.units), self.kernel_initializer, self.kernel_regularizer
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias", (self.units,), self.bias_initializer, self.bias_regularizer
            )
        else:
            self.bias = None

    def forward(self, inputs):
        out = torch.matmul(inputs, self.kernel)
        if self.bias is not None:
            out = out + self.bias
        return self.activation(out)


def _l2_coef(reg):
    """将正则配置统一转换为 L2 系数（float 或 None）"""
    if reg is None:
        return None
    if isinstance(reg, (int, float)):
        return float(reg) if reg > 0 else None
    raise ValueError(f"仅支持 float 类型的 L2 正则系数, 收到 {reg}")


class Embedding(nn.Module):
    """嵌入层，等价于 tf.keras.layers.Embedding

    - 默认 embeddings_initializer='uniform'（U(-0.05, 0.05)），与 Keras 一致
    - l2_reg 对整张嵌入表施加 L2 正则（与 Keras embeddings_regularizer 行为一致）
    - mask_zero=True 时，可通过 compute_mask(ids) 获得 ids != 0 的掩码
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embeddings_initializer="uniform",
        l2_reg: float = 0.0,
        trainable: bool = True,
        mask_zero: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.mask_zero = mask_zero
        self.layer_name = name
        weight = torch.empty(self.input_dim, self.output_dim)
        init_tensor_(weight, embeddings_initializer)
        self.embeddings = nn.Parameter(weight, requires_grad=trainable)
        self.l2_reg = float(l2_reg or 0.0)

    @property
    def weight(self):
        return self.embeddings

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids.long(), self.embeddings)

    def compute_mask(self, ids: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.mask_zero:
            return None
        return ids != 0

    def regularization_loss(self):
        if self.l2_reg > 0 and self.embeddings.requires_grad:
            return self.l2_reg * torch.sum(self.embeddings * self.embeddings)
        return None


# ---------------------------------------------------------------------------
# 模型基类
# ---------------------------------------------------------------------------


def to_tensor(value, device=None) -> torch.Tensor:
    """numpy/list 转换为张量：整数->long，浮点->float32；一维数组扩展为 [N, 1]"""
    device = device or get_device()
    if isinstance(value, torch.Tensor):
        t = value
    else:
        arr = np.asarray(value)
        if arr.dtype == object:
            arr = np.stack([np.asarray(v) for v in arr])
        t = torch.from_numpy(np.ascontiguousarray(arr))
    if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        t = t.long()
    elif t.dtype == torch.bool:
        pass
    else:
        t = t.float()
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    return t.to(device, non_blocking=True)


def num_samples(features) -> int:
    if isinstance(features, dict):
        return len(next(iter(features.values())))
    if isinstance(features, (list, tuple)):
        return len(features[0])
    return len(features)


def slice_features(features, idx):
    if isinstance(features, dict):
        return {k: np.asarray(v)[idx] if not isinstance(v, torch.Tensor) else v[idx] for k, v in features.items()}
    if isinstance(features, (list, tuple)):
        return [np.asarray(v)[idx] for v in features]
    return np.asarray(features)[idx]


class FunRecModel(nn.Module):
    """所有深度推荐模型的基类

    约定:
        - forward(inputs) 接收 {特征名: 张量} 字典，返回张量或张量列表（多任务）
        - input_names: 模型需要的输入特征名称列表（predict 时据此筛选/映射输入）
        - 训练期间可调用 self.add_loss(t) 添加辅助损失（如 DIEN 的辅助损失）
        - 子塔（用户塔、物品塔等）通过 SubModel 暴露，与主模型共享参数
    """

    def __init__(self, input_names: Optional[Sequence[str]] = None, name: Optional[str] = None):
        super().__init__()
        self.input_names = list(input_names) if input_names is not None else None
        self.model_name = name
        self._extra_losses: List[torch.Tensor] = []
        self.output_names: Optional[List[str]] = None

    def __setattr__(self, name, value):
        # 子塔 SubModel 以普通属性保存（不注册为子模块），
        # 否则 parent.train() -> SubModel.train() -> parent.train() 会无限递归
        if isinstance(value, SubModel):
            self._modules.pop(name, None)
            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)

    # ---- 辅助损失 / 正则 ----
    def add_loss(self, loss: torch.Tensor) -> None:
        if self.training:
            self._extra_losses.append(loss)

    def pop_extra_losses(self) -> List[torch.Tensor]:
        losses = []
        for m in self.modules():
            if isinstance(m, FunRecModel) and m._extra_losses:
                losses.extend(m._extra_losses)
                m._extra_losses = []
        return losses

    def regularization_loss(self) -> Optional[torch.Tensor]:
        """收集所有层的 L2 正则损失"""
        total = None
        seen = set()
        for m in self.modules():
            if id(m) in seen or isinstance(m, FunRecModel):
                continue
            seen.add(id(m))
            fn = getattr(m, "regularization_loss", None)
            if fn is None:
                continue
            r = fn()
            if r is not None:
                total = r if total is None else total + r
        return total

    # ---- 输入处理 ----
    def prepare_inputs(self, features, device=None) -> Dict[str, torch.Tensor]:
        """将 numpy 特征转为张量字典，并按 input_names 过滤"""
        if not isinstance(features, dict):
            if self.input_names is None or len(self.input_names) == 0:
                raise ValueError("非字典输入需要模型定义 input_names")
            if isinstance(features, (list, tuple)) and len(self.input_names) == len(features) and len(self.input_names) > 1:
                features = dict(zip(self.input_names, features))
            else:
                features = {self.input_names[0]: features}
        names = self.input_names if self.input_names is not None else list(features.keys())
        out = {}
        for k in names:
            if k in features:
                out[k] = to_tensor(features[k], device)
        missing = [k for k in names if k not in features]
        if missing:
            raise KeyError(f"缺少模型输入特征: {missing}")
        return out

    def _mode_owner(self) -> nn.Module:
        """训练/推理模式的归属模块（子模型的模式跟随主模型）"""
        return self

    # ---- 推理 ----
    @torch.no_grad()
    def predict(self, features, batch_size: int = 256, verbose: int = 0):
        """批量推理，返回 numpy 数组（多输出时返回列表），对应 keras Model.predict"""
        was_training = self._mode_owner().training
        self.eval()
        device = get_device()
        self.to(device)
        n = num_samples(features if isinstance(features, dict) else features)
        outputs = None
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            batch = self.prepare_inputs(slice_features(features, idx), device)
            out = self(batch)
            if isinstance(out, (list, tuple)):
                out = [o.detach().float().cpu().numpy() for o in out]
                if outputs is None:
                    outputs = [[] for _ in out]
                for i, o in enumerate(out):
                    outputs[i].append(o)
            else:
                if outputs is None:
                    outputs = []
                outputs.append(out.detach().float().cpu().numpy())
        if was_training:
            self.train()
        if outputs is None:
            return np.zeros((0,))
        # 标量输出（如 SASRec/HSTU 主模型输出整批损失）按批次拼接为一维数组
        def _cat(arrs):
            return np.concatenate([np.atleast_1d(a) for a in arrs], axis=0)

        if outputs and isinstance(outputs[0], list):
            return [_cat(o) for o in outputs]
        return _cat(outputs)

    def build_with(self, sample_features, batch_size: int = 2):
        """用少量样本前向一次以创建惰性参数"""
        n = num_samples(sample_features)
        idx = np.arange(min(batch_size, n))
        device = get_device()
        self.to(device)
        was_training = self._mode_owner().training
        self.eval()
        with torch.no_grad():
            self(self.prepare_inputs(slice_features(sample_features, idx), device))
        if was_training:
            self.train()
        self.pop_extra_losses()
        return self


class SubModel(FunRecModel):
    """与主模型共享参数的子模型（如用户塔/物品塔）

    Args:
        parent: 主模型
        fn: 前向函数 fn(inputs_dict) -> Tensor，一般为主模型的方法，如 parent.encode_user
        input_names: 子模型需要的输入特征名称
    """

    def __init__(self, parent: nn.Module, fn: Union[str, Callable], input_names: Sequence[str], name: Optional[str] = None):
        super().__init__(input_names=input_names, name=name)
        # 使用列表持有父模型，避免将其注册为子模块导致参数重复
        object.__setattr__(self, "_parent", [parent])
        self._fn_name = fn if isinstance(fn, str) else None
        object.__setattr__(self, "_fn", fn)

    @property
    def parent(self):
        return self._parent[0]

    def forward(self, inputs):
        fn = getattr(self.parent, self._fn) if isinstance(self._fn, str) else self._fn
        return fn(inputs)

    def _mode_owner(self) -> nn.Module:
        return self.parent

    def train(self, mode: bool = True):
        super().train(mode)
        self.parent.train(mode)
        return self

    def to(self, *args, **kwargs):
        self.parent.to(*args, **kwargs)
        return self

    def parameters(self, recurse: bool = True):
        return self.parent.parameters(recurse)

    def state_dict(self, *args, **kwargs):
        return self.parent.state_dict(*args, **kwargs)


def build_dummy_inputs(feature_columns, batch_size: int = 2) -> Dict[str, np.ndarray]:
    """根据特征列构造全零（索引为1）的假输入，用于加载模型前创建惰性参数"""
    dummy = {}
    for fc in feature_columns:
        if fc.type == "dense":
            shape = (batch_size, fc.max_len, fc.dimension) if fc.max_len > 1 else (batch_size, fc.dimension)
            dummy[fc.name] = np.zeros(shape, dtype=np.float32)
        else:
            dummy[fc.name] = np.ones((batch_size, fc.max_len), dtype=np.int64)
    return dummy


def save_model(model: FunRecModel, path: str, build_function: str, feature_columns, model_config: Dict[str, Any], tower: Optional[str] = None, sample_features=None):
    """保存模型（参数 + 重建所需信息）

    Args:
        model: 主模型或子模型（子模型会保存其主模型的参数）
        build_function: 构建函数路径，如 'funrec.models.deepfm.build_deepfm_model'
        tower: None 表示主模型；'user_model' / 'item_model' 表示返回元组中的对应子模型
        sample_features: 可选，一小批样本特征，用于加载时创建惰性参数
    """
    root = model.parent if isinstance(model, SubModel) else model
    if sample_features is not None:
        idx = np.arange(min(2, num_samples(sample_features)))
        sample_features = slice_features(sample_features, idx)
    torch.save(
        {
            "build_function": build_function,
            "feature_columns": feature_columns,
            "model_config": model_config,
            "tower": tower,
            "state_dict": {k: v.detach().cpu() for k, v in root.state_dict().items()},
            "sample_features": sample_features,
        },
        path,
    )


def load_model(path: str, map_location=None):
    """加载由 save_model 保存的模型，返回对应的主模型或子模型（eval 模式）"""
    ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    module_path, fn_name = ckpt["build_function"].rsplit(".", 1)
    build_fn = getattr(importlib.import_module(module_path), fn_name)
    main_model, user_model, item_model = build_fn(ckpt["feature_columns"], ckpt["model_config"])
    sample = ckpt.get("sample_features") or build_dummy_inputs(ckpt["feature_columns"])
    main_model.build_with(sample)
    main_model.load_state_dict(ckpt["state_dict"])
    main_model.to(get_device())
    main_model.eval()
    tower = ckpt.get("tower")
    if tower == "user_model":
        return user_model
    if tower == "item_model":
        return item_model
    return main_model
