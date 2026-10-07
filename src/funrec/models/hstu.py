import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Dense, FunRecModel, Layer, SubModel
from .utils import build_input_layer, build_embedding_table_dict, add_tensor_func
from .layers import PositionEncodingLayer, NegativeSampleEmbedding, DNNs, EmbeddingIndex


def _random_normal_002(tensor):
    """正态分布初始化（均值 0，标准差 0.02），与原实现的 RandomNormal(mean=0, stddev=0.02) 一致"""
    return tensor.normal_(0.0, 0.02)


class HstuLayer(Layer):
    """多头注意力机制的 PyTorch 实现（与原 TF 实现语义一致）

    参数:
      num_units: 注意力的维度大小，如果为None则使用输入的最后一维
      num_heads: 注意力头的数量
      dropout_rate: dropout 比率
      attention_type: 注意力类型，支持 'dot_product', 'relative_position_bias', 'time_interval_bias'
      causality: 是否使用因果掩码
      linear_projection_and_dropout: 是否在输出后添加线性投影和dropout
      args: 包含各种配置参数的对象
      with_qk: 是否返回Q和K

    注意: 与原实现一致，配置项通过 getattr/hasattr 从 args 中读取。当 args 为字典（如
    build_hstu_model 传入的 model_config）时，这些属性读取均返回默认值（即字典中的
    normalize_query / qkv_projection_bias / attention_normalization 等键不会生效）。
    """

    def __init__(
        self,
        num_units=None,
        num_heads=8,
        attention_type="dot_product",
        dropout_rate=0,
        causality=False,
        linear_projection_and_dropout=False,
        args=None,
        with_qk=False,
        **kwargs,
    ):
        super(HstuLayer, self).__init__(**kwargs)
        self.num_units = num_units
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.linear_projection_and_dropout = linear_projection_and_dropout
        self.args = args
        self.with_qk = with_qk

    def build(self, input_shape):
        # 设置注意力维度的默认值
        if isinstance(input_shape, list):
            queries_shape = input_shape[0]
            keys_shape = input_shape[1]
        else:
            queries_shape = input_shape
            keys_shape = input_shape

        if self.num_units is None:
            self.num_units = queries_shape[-1]

        # 初始化器（None 对应 原 Dense 默认的 glorot_uniform）
        qkv_initializer = "glorot_uniform"
        if (
            hasattr(self.args, "qkv_projection_initializer")
            and self.args.qkv_projection_initializer == "normal"
        ):
            print("Set qkv projection initializer to normal")
            qkv_initializer = _random_normal_002

        # 线性投影层
        use_bias = (
            getattr(self.args, "qkv_projection_bias", False) if self.args else False
        )
        self.query_dense = Dense(
            self.num_units,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=qkv_initializer,
            name="query_projection",
        )
        self.key_dense = Dense(
            self.num_units,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=qkv_initializer,
            name="key_projection",
        )

        self.value_projection = (
            getattr(self.args, "value_projection", True) if self.args else True
        )
        if self.value_projection:
            self.value_dense = Dense(
                self.num_units,
                activation=None,
                use_bias=use_bias,
                kernel_initializer=qkv_initializer,
                name="value_projection",
            )

        # 相对位置偏置和时间间隔偏置
        if "relative_position_bias" in self.attention_type:
            self.rel_pos_bias = self.add_weight(
                shape=(2 * getattr(self.args, "maxlen", 50) - 1, self.num_heads),
                initializer="glorot_uniform",
                trainable=True,
                # 注: 参数名不能与 relative_position_bias 方法同名（原权重名为 relative_position_bias）
                name="rel_pos_bias",
            )

        if "time_interval_bias" in self.attention_type:
            max_interval = getattr(
                self.args, "time_interval_attention_max_interval", 1024
            )
            # 注: 原实现中该权重与 time_interval_bias 方法同名（会覆盖方法导致调用失败），
            # 这里将参数命名为 time_interval_bias_table 以避免冲突
            self.time_interval_bias_table = self.add_weight(
                shape=(max_interval + 1, self.num_heads),
                initializer="glorot_uniform",
                trainable=True,
                name="time_interval_bias_table",
            )

        # U投影
        self.u_projection = (
            getattr(self.args, "u_projection", False) if self.args else False
        )
        if self.u_projection:
            u_initializer = "glorot_uniform"
            if (
                hasattr(self.args, "u_projection_initializer")
                and self.args.u_projection_initializer == "normal"
            ):
                u_initializer = _random_normal_002

            u_bias = (
                getattr(self.args, "u_projection_bias", False) if self.args else False
            )
            self.u_dense = Dense(
                self.num_units,
                activation=None,
                use_bias=u_bias,
                kernel_initializer=u_initializer,
                name="u_projection",
            )

        # 输出投影
        if self.linear_projection_and_dropout:
            self.output_dense = Dense(
                self.num_units,
                activation=None,
                kernel_initializer=qkv_initializer,
                name="output_projection",
            )

        # Dropout层
        self.dropout = nn.Dropout(self.dropout_rate)
        # 创建归一化层（与原 LayerNormalization 一致，epsilon=1e-3）
        # 注: 原实现总是创建该层，但只有被调用时才会产生权重；这里仅在会被使用时创建，参数量保持一致
        if hasattr(self.args, "normalize_query") and self.args.normalize_query:
            self.query_norm = nn.LayerNorm(queries_shape[-1], eps=1e-3).to(
                self._build_device
            )
        else:
            self.query_norm = None

    def silu(self, x):
        """SiLU (Swish) 激活函数"""
        return x * torch.sigmoid(x)

    def normalize(self, norm_func, inputs):
        """归一化"""
        return norm_func(inputs)

    def relative_position_bias(self, batch_size, maxlen, num_heads):
        """相对位置偏置"""
        seq_len = self.queries.shape[1]

        # 计算位置索引
        positions = torch.arange(seq_len, device=self.queries.device)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions = relative_positions + maxlen - 1  # 转换为非负索引

        # 获取相应的偏置
        bias = self.rel_pos_bias[relative_positions]
        # 转换形状以适配多头注意力
        bias = bias.permute(2, 0, 1)  # (num_heads, seq_len, seq_len)
        bias = bias.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # (batch_size, num_heads, seq_len, seq_len)
        bias = bias.reshape(
            batch_size * num_heads, seq_len, seq_len
        )  # (batch_size * num_heads, seq_len, seq_len)

        return bias

    def time_interval_bias(self, input_interval, maxlen, max_interval, num_heads):
        """时间间隔偏置"""
        batch_size = self.queries.shape[0]
        seq_len = self.queries.shape[1]

        # 计算时间间隔
        intervals = torch.abs(
            input_interval[:, :, None] - input_interval[:, None, :]
        )  # (batch_size, seq_len, seq_len)
        intervals = torch.clamp(intervals, max=max_interval).long()  # 截断最大间隔

        # 获取相应的偏置
        bias = self.time_interval_bias_table[
            intervals
        ]  # (batch_size, seq_len, seq_len, num_heads)
        bias = bias.permute(0, 3, 1, 2)  # (batch_size, num_heads, seq_len, seq_len)
        bias = bias.reshape(
            batch_size * num_heads, seq_len, seq_len
        )  # (batch_size * num_heads, seq_len, seq_len)

        return bias

    def apply_attention(
        self,
        K,
        V,
        outputs,
        scale_attention=True,
        attention_activation=None,
        attention_normalization=None,
    ):
        """应用注意力机制"""
        # 缩放
        if scale_attention:
            depth = float(K.shape[-1])
            outputs = outputs / torch.sqrt(torch.tensor(depth, device=outputs.device))

        # 因果掩码
        if self.causality:
            # 创建下三角矩阵
            diag_vals = torch.ones_like(outputs[0, :, :])
            tril = torch.tril(diag_vals)  # 下三角为1，上三角为0
            causality_mask = tril.unsqueeze(0).expand(outputs.shape[0], -1, -1)

            # 将上三角部分设置为很小的负数
            paddings = torch.ones_like(causality_mask) * (-(2**32) + 1)
            outputs = torch.where(causality_mask == 0, paddings, outputs)

        # Key掩码
        key_masks = torch.sign(torch.sum(torch.abs(K), dim=-1))  # (h*N, T_k)
        key_masks = key_masks.unsqueeze(1).expand(
            -1, self.queries.shape[1], -1
        )  # (h*N, T_q, T_k)

        # 应用Key掩码
        paddings = torch.ones_like(outputs) * (-(2**32) + 1)
        outputs = torch.where(key_masks == 0, paddings, outputs)

        # 应用激活函数
        if attention_activation == "softmax":
            weights = torch.softmax(outputs, dim=-1)
        else:
            weights = outputs

        # 归一化
        if attention_normalization == "softmax":
            weights = torch.softmax(weights, dim=-1)

        # 应用dropout（与原实现一致: training=True，即推理时同样生效）
        weights = F.dropout(weights, p=self.dropout_rate, training=True)

        # 加权求和
        attention_output = torch.matmul(weights, V)

        return attention_output

    def forward(self, inputs, input_interval=None):
        # 处理输入
        if isinstance(inputs, list):
            self.queries, self.keys = inputs[:2]
        else:
            self.queries = self.keys = inputs

        # 归一化查询
        if hasattr(self.args, "normalize_query") and self.args.normalize_query:
            self.queries = self.normalize(self.query_norm, self.queries)

        # 如果设置了覆写key
        if (
            hasattr(self.args, "overwrite_key_with_query")
            and self.args.overwrite_key_with_query
        ):
            self.keys = self.queries

        # 获取batch_size
        batch_size = self.queries.shape[0]
        head_dim = self.num_units // self.num_heads

        # 线性投影
        Q = self.query_dense(self.queries)  # (N, T_q, C)
        K = self.key_dense(self.keys)  # (N, T_k, C)
        if self.value_projection:
            V = self.value_dense(self.keys)  # (N, T_k, C)
        else:
            V = self.keys

        # 分割并拼接，实现多头
        Q_split = Q.reshape(batch_size, -1, self.num_heads, head_dim)
        Q_split = Q_split.permute(0, 2, 1, 3)
        Q_ = Q_split.reshape(batch_size * self.num_heads, -1, head_dim)

        K_split = K.reshape(batch_size, -1, self.num_heads, head_dim)
        K_split = K_split.permute(0, 2, 1, 3)
        K_ = K_split.reshape(batch_size * self.num_heads, -1, head_dim)

        V_split = V.reshape(batch_size, -1, self.num_heads, head_dim)
        V_split = V_split.permute(0, 2, 1, 3)
        V_ = V_split.reshape(batch_size * self.num_heads, -1, head_dim)

        # 应用SiLU激活
        if (
            hasattr(self.args, "qkv_projection_activation")
            and self.args.qkv_projection_activation == "silu"
        ):
            print("Use SiLU activation on qkv projection")
            Q_, K_, V_ = self.silu(Q_), self.silu(K_), self.silu(V_)

        new_values = 0
        # 不同类型的注意力机制
        if "dot_product" in self.attention_type:
            outputs = torch.matmul(Q_, K_.transpose(1, 2))  # (h*N, T_q, T_k)
            outputs = self.apply_attention(K_, V_, outputs)
            new_values += outputs

        if "relative_position_bias" in self.attention_type:
            print("Add relative position bias")
            maxlen = getattr(self.args, "maxlen", 50)
            attention_bias = self.relative_position_bias(
                batch_size, maxlen, self.num_heads
            )

            if (
                hasattr(self.args, "relative_position_bias_add_item_interaction")
                and self.args.relative_position_bias_add_item_interaction
            ):
                print("Relative position bias add item interaction")
                outputs = torch.matmul(Q_, K_.transpose(1, 2))
                outputs = outputs + attention_bias
            else:
                outputs = attention_bias

            scale_attention = (
                getattr(self.args, "scale_attention", True) if self.args else True
            )
            attention_activation = (
                getattr(self.args, "attention_activation", None) if self.args else None
            )
            attention_normalization = (
                getattr(self.args, "attention_normalization", None)
                if self.args
                else None
            )

            outputs = self.apply_attention(
                K_,
                V_,
                outputs,
                scale_attention=scale_attention,
                attention_activation=attention_activation,
                attention_normalization=attention_normalization,
            )
            new_values += outputs

        if "time_interval_bias" in self.attention_type and input_interval is not None:
            print("Add time interval bias attention")
            maxlen = getattr(self.args, "maxlen", 50)
            max_interval = getattr(
                self.args, "time_interval_attention_max_interval", 1024
            )
            attention_bias = self.time_interval_bias(
                input_interval, maxlen, max_interval, self.num_heads
            )

            if (
                hasattr(self.args, "time_interval_bias_add_item_interaction")
                and self.args.time_interval_bias_add_item_interaction
            ):
                outputs = torch.matmul(Q_, K_.transpose(1, 2))
                outputs = outputs + attention_bias
            else:
                outputs = attention_bias

            scale_attention = (
                getattr(self.args, "scale_attention", True) if self.args else True
            )
            attention_activation = (
                getattr(self.args, "attention_activation", None) if self.args else None
            )
            attention_normalization = (
                getattr(self.args, "attention_normalization", None)
                if self.args
                else None
            )

            outputs = self.apply_attention(
                K_,
                V_,
                outputs,
                scale_attention=scale_attention,
                attention_activation=attention_activation,
                attention_normalization=attention_normalization,
            )
            new_values += outputs

        # 合并多头注意力结果
        outputs_split = new_values.reshape(batch_size, self.num_heads, -1, head_dim)
        outputs_split = outputs_split.permute(0, 2, 1, 3)
        outputs = outputs_split.reshape(batch_size, -1, self.num_units)

        # U投影
        if self.u_projection:
            U = self.u_dense(self.queries)
            U = self.silu(U)
            # 注: 与原实现保持一致（原实现此处 normalize 缺少归一化层参数）
            outputs = U * self.normalize(outputs)

        # 线性投影和dropout
        if self.linear_projection_and_dropout:
            dropout_before = (
                getattr(self.args, "dropout_before_linear_projection", False)
                if self.args
                else False
            )
            if dropout_before:
                outputs = self.dropout(outputs)
            outputs = self.output_dense(outputs)
            if not dropout_before:
                outputs = self.dropout(outputs)

        # 残差连接
        outputs = outputs + self.queries

        if self.with_qk:
            return Q, K
        else:
            return outputs

    def get_config(self):
        return {
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "attention_type": self.attention_type,
            "dropout_rate": self.dropout_rate,
            "causality": self.causality,
            "linear_projection_and_dropout": self.linear_projection_and_dropout,
            "with_qk": self.with_qk,
        }


class HSTUModel(FunRecModel):
    """HSTU 主模型

    forward 输出: 负采样二分类损失（标量），与原模型的 main_loss 一致
    子塔:
        user_tower: 输入 seq_ids(/timestamps)，输出序列最后一个位置的 embedding (B, emb_dim)
        all_item_tower: 输入 all_item_input（物品 id 一维数组），输出 (N, emb_dim)
        sampling_item_tower: 输入 neg_sample_ids，输出 (B, 采样数, emb_dim)
    """

    def __init__(self, feature_columns, model_config):
        input_layer_dict = build_input_layer(feature_columns)
        super().__init__(input_names=list(input_layer_dict.keys()), name="hstu")

        # 从配置中提取参数并设置默认值
        self.max_seq_len = model_config.get("max_seq_len", 50)
        mha_num = model_config.get("mha_num", 2)
        nums_heads = model_config.get("nums_heads", 1)
        dropout = model_config.get("dropout", 0.2)
        activation = model_config.get("activation", "relu")
        pos_emb_trainable = model_config.get("pos_emb_trainable", True)
        pos_initializer = model_config.get("pos_initializer", "glorot_uniform")
        attention_type = model_config.get("attention_type", "dot_product")
        emb_dim = feature_columns[0].emb_dim
        self.emb_dim = emb_dim

        filter_feature_columns = [x for x in feature_columns if x.name != "timestamps"]
        self.embedding_table_dict = build_embedding_table_dict(
            filter_feature_columns, prefix="hstu/"
        )
        item_dim = self.embedding_table_dict["item_id"].output_dim

        self.position_encoding = PositionEncodingLayer(
            dims=emb_dim,
            max_len=self.max_seq_len,
            trainable=pos_emb_trainable,
            initializer=pos_initializer,
        )

        # 多头注意力 block: LayerNorm -> HstuLayer -> 残差 -> LayerNorm -> DNNs
        self.attention_norms = nn.ModuleList()
        self.hstu_layers = nn.ModuleList()
        self.residual_norms = nn.ModuleList()
        self.dnns = nn.ModuleList()
        for i in range(mha_num):
            self.attention_norms.append(nn.LayerNorm(item_dim, eps=1e-3))
            self.hstu_layers.append(
                HstuLayer(
                    num_units=emb_dim,
                    num_heads=nums_heads,
                    attention_type=attention_type,
                    dropout_rate=dropout,
                    causality=True,
                    linear_projection_and_dropout=False,
                    args=model_config,
                    name=f"{i}_block",
                )
            )
            self.residual_norms.append(nn.LayerNorm(item_dim, eps=1e-3))
            self.dnns.append(
                DNNs(
                    units=[emb_dim, emb_dim],
                    dropout_rate=dropout,
                    activation=activation,
                    name=f"{i}_dnn",
                )
            )
        self.final_norm = nn.LayerNorm(emb_dim, eps=1e-3)

        # 负采样算loss
        self.negative_sampler = NegativeSampleEmbedding(
            vocab_size=feature_columns[0].vocab_size,
            num_sampled=self.max_seq_len,
            sampled_type="uniform",
        )

        # 推理时用户输入和item输入
        user_input_names = [
            k for k in input_layer_dict.keys() if k in ["seq_ids", "timestamps"]
        ]
        sampling_item_input_names = [
            k for k in input_layer_dict.keys() if k in ["neg_sample_ids"]
        ]
        self.user_tower = SubModel(self, "encode_user", user_input_names, name="user_model")
        # 全量物品评估使用专用的物品输入
        self.all_item_tower = SubModel(
            self, "encode_all_item", ["all_item_input"], name="item_model"
        )
        self.sampling_item_tower = SubModel(
            self, "encode_sampling_item", sampling_item_input_names
        )

    def encode_sequence(self, inputs):
        """序列编码，返回 (B, max_len, emb_dim)"""
        sequence_embedding = self.embedding_table_dict["item_id"](inputs["seq_ids"])
        position_embedding = self.position_encoding(sequence_embedding)
        # 原始序列emb加上position embedding
        sequence_embedding = add_tensor_func([sequence_embedding, position_embedding])

        input_interval = inputs.get("timestamps")
        # 多头注意力
        for i in range(len(self.hstu_layers)):
            sequence_embedding_norm = self.attention_norms[i](sequence_embedding)
            sequence_embedding_output = self.hstu_layers[i](
                sequence_embedding_norm, input_interval=input_interval
            )
            # 残差连接
            sequence_embedding = add_tensor_func(
                [sequence_embedding, sequence_embedding_output]
            )
            sequence_embedding = self.residual_norms[i](sequence_embedding)
            # 前馈神经网络
            sequence_embedding = self.dnns[i](sequence_embedding)
        sequence_embedding = self.final_norm(sequence_embedding)
        return sequence_embedding

    def encode_user(self, inputs):
        # 序列的padding在左边，直接拿到序列的最后一个结果即可
        return self.encode_sequence(inputs)[:, -1, :]  # B, emb_dim

    def encode_all_item(self, inputs):
        ids = inputs["all_item_input"] if isinstance(inputs, dict) else inputs
        return self.embedding_table_dict["item_id"](ids.reshape(-1))  # N, emb_dim

    def encode_sampling_item(self, inputs):
        # 评估时需要使用的embedding（采样评估）
        return self.embedding_table_dict["item_id"](inputs["neg_sample_ids"])

    def forward(self, inputs):
        sequence_embedding = self.encode_sequence(inputs)
        positive_embedding = self.embedding_table_dict["item_id"](inputs["pos_ids"])

        # 获取所有item的权重
        item_embedding_weight = self.embedding_table_dict["item_id"].embeddings

        # 负采样算loss
        negative_embeddings = self.negative_sampler(
            positive_embedding, item_embedding_weight
        )

        # 序列展开成[batch_size, max_len, emb_dim]
        sequence_embedding = sequence_embedding.reshape(
            sequence_embedding.shape[0], -1, self.emb_dim
        )
        pos_item_embedding = positive_embedding.reshape(
            positive_embedding.shape[0], -1, self.emb_dim
        )
        negative_embeddings = negative_embeddings.reshape(
            negative_embeddings.shape[0], -1, self.emb_dim
        )

        pos_logits = torch.sum(pos_item_embedding * sequence_embedding, dim=-1)  # B, max_len
        # 与原实现一致: neg (B, L, D) 与 expand_dims(seq, 1) (B, 1, L, D) 广播相乘后求和，
        # 结果形状为 (B, B, L)，neg_logits[i, j, l] = <neg[j, l], seq[i, l]>
        # 使用 einsum 计算以避免构造 (B, B, L, D) 的中间张量
        neg_logits = torch.einsum(
            "jld,ild->ijl", negative_embeddings, sequence_embedding
        )  # B, B, max_len

        # 创建目标掩码，忽略padding项目（ID=0），维度: [batch_size, maxlen]
        # 这里的掩码是为了计算loss时忽略padding项
        is_target = (inputs["pos_ids"] != 0).float().reshape(-1, self.max_seq_len)

        main_loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * is_target
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * is_target
        ) / torch.sum(is_target)
        return main_loss


def build_hstu_model(feature_columns, model_config):
    """
    构建HSTU模型 (分层结构化Transformer单元)

    参数:
    feature_columns: 特征列配置
    model_config: 模型配置字典，包含:
        - max_seq_len: 最大序列长度 (默认: 50)
        - mha_num: 多头注意力层数 (默认: 2)
        - nums_heads: 注意力头数 (默认: 1)
        - dropout: dropout率 (默认: 0.2)
        - activation: 激活函数 (默认: 'relu')
        - pos_emb_trainable: 位置编码是否可训练 (默认: True)
        - pos_initializer: 位置编码初始化器 (默认: 'glorot_uniform')
        - attention_type: 注意力类型 (默认: 'dot_product')
    """
    model = HSTUModel(feature_columns, model_config)

    # 为评估创建独立的用户和物品模型（与主模型共享参数）
    user_model = model.user_tower
    item_model = model.all_item_tower

    return model, user_model, item_model
