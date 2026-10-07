# PyTorch 迁移：待完成实验清单

> 分支：`pytorch-migration`（改动尚未提交）
> 状态：**代码迁移已完成**，所有模型均已通过 TF↔PyTorch 数值等价测试与合成数据冒烟测试；
> **尚未在真实数据集上运行任何训练/评估实验**（按要求本阶段只改代码）。本文档列出所有待做实验。

---

## 1. 已完成的工作（供参考）

| 项目 | 说明 |
| --- | --- |
| 框架核心 | `models/base.py`（新增：惰性构建 `Layer`、`Dense`、`Embedding`、`FunRecModel`、`SubModel`、`save_model`/`load_model`）、`models/layers.py`、`models/utils.py`、`training/trainer.py`（`fit_model` 对齐 Keras compile+fit，含 Keras 版 Adam）、`training/loss.py`、`evaluation/evaluator.py` |
| 模型 | 34 个深度模型文件全部改写为 PyTorch，接口（`build_xxx_model` 签名、返回 `(model, user_model, item_model)`、config）保持不变 |
| web_project | 离线训练脚本改用 `save_model` 保存 `.pt`；线上 `resource_manager` 改用 `load_model`；后端依赖改为 torch |
| 依赖/文档 | `requirements.txt`、`pyproject.toml` 改为 `torch>=2.1`；`README.md`/`README_en.md` 增加环境安装说明 |
| 验证（无真实数据） | ① 每个 layer 与模型：拷贝 TF 权重后输出最大误差一般 ≤1e-6，参数量一致，一步训练的梯度一致；② `tmp/agent_scratch/audit/smoke_all.py`：38 个 config 在 CPU/CUDA 上均可构建→训练 2 步→预测→评估；③ `tmp/agent_scratch/audit/roundtrip.py`：保存/加载后预测完全一致 |

运行环境：

```bash
conda create -n funrec-torch --override-channels -c conda-forge python=3.10
conda activate funrec-torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

---

## 2. 运行实验前需要注意的数据问题

1. **数据路径**：仓库根目录 `.env`（已在 .gitignore 中）当前为
   ```
   FUNREC_RAW_DATA_PATH=/mnt/i/dev/Search-Advertise-Recommend/fun-rec/data/dataset
   FUNREC_PROCESSED_DATA_PATH=/home/saber812/dev/fun-rec/tmp
   ```
2. **KuaiRand 预处理内存问题**（影响所有 `kuairand_data` 模型，共 20 个）：
   - 原始代码一次性读入 3.4GB 的 `video_features_statistic_1k.csv`，在 15GB 内存的机器上触发 OOM（进程被 kill，峰值 >15GB）。
   - 已修改 `src/funrec/data/preprocess/kuairand.py`：分块读取并只保留日志中出现过的 `video_id`（后续是 left merge，理论上结果不变）。**该修改尚未在真实数据上验证**，待做：跑通一次预处理，并与原始结果（如有其他机器生成的 pkl）比对。
   - 从 WSL 读取 `/mnt/i`（Windows 盘）上的大文件时，分块读取仍报 `OSError: [Errno 12] Cannot allocate memory`。**建议先把数据集复制到 WSL 本地磁盘**（如 `~/dev/fun-rec/data_local`），再修改 `.env` 的 `FUNREC_RAW_DATA_PATH`。
3. 预处理缓存写入 `FUNREC_PROCESSED_DATA_PATH`，首次运行某个数据集的模型时会自动生成；**不要并行跑同一数据集的多个首次实验**，以免同时预处理导致内存不足或写入冲突。

---

## 3. 待运行的模型实验

运行方式（每个模型单独进程，避免内存累积）：

```python
import funrec
funrec.run_experiment("deepfm")                         # 单模型
funrec.compare_models(["fm", "afm", "nfm", "pnn", "fibinet", "deepfm"])  # 对比组
```

“TF 参考值”取自 `docs/_sources/**/*.rst.txt` 中原 TensorFlow 版本打印的结果。由于随机种子、负采样、Dropout 等不同，**指标只需大致一致**（排序模型 AUC 一般相差 ±0.01 以内可接受；召回指标本身很低、噪声大，只需同一量级）。

### 3.1 排序模型 —— `kuairand_data`

| 模型 | TF 参考值（auc / gauc / val_user） | 文档位置 |
| --- | --- | --- |
| `wide_deep` | 0.5902 / 0.5724 / 928 | chapter_2_ranking/1.wide_and_deep |
| `fm` | 0.5893 / 0.5721 / 928 | chapter_2_ranking/2.feature_crossing/1.second_order |
| `afm` | 0.5867 / 0.5702 / 928 | 同上 |
| `nfm` | 0.5905 / 0.5597 / 928 | 同上 |
| `pnn` | 0.5987 / 0.5748 / 928 | 同上 |
| `fibinet` | 0.6001 / 0.5734 / 928 | 同上 |
| `deepfm` | 0.5953 / 0.5742 / 928 | 同上 |
| `dcn` | 0.6020 / 0.5776 / 928 | chapter_2_ranking/2.feature_crossing/2.higher_order |
| `xdeepfm` | 0.6024 / 0.5769 / 928 | 同上 |
| `autoint` | 0.6048 / 0.5731 / 928 | 同上 |
| `din` | 0.5999 / 0.5654 / 928 | chapter_2_ranking/3.sequence |
| `dien` | 0.5649 / 0.5539 / 928 | 同上 |
| `dsin` | 0.5456 / 0.5521 / 99 | 同上 |
| `hmoe` | 0.5924 / 0.5493 / 217 | chapter_2_ranking/5.multi_scenario/1.multi_tower |
| `star` | 0.6390 / 0.6156 / 693 | 同上 |
| `apg` | 0.6669 / 0.6369 / 217 | chapter_2_ranking/5.multi_scenario/2.dynamic_weight |

### 3.2 多目标 / 多场景模型 —— `kuairand_data`（评估 is_click）

| 模型 | TF 参考值（auc_is_click / gauc_is_click / val_user） | 文档位置 |
| --- | --- | --- |
| `shared_bottom` | 0.5935 / 0.5738 / 928 | chapter_2_ranking/4.multi_objective/1.arch |
| `mmoe` | 0.6013 / 0.5753 / 928 | 同上 |
| `ple` | 0.6047 / 0.5764 / 928 | 同上 |
| `esmm` | 0.5936 / 0.5744 / 928 | chapter_2_ranking/4.multi_objective/2.dependency_modeling |
| `pepnet` | 0.6774 / 0.6247 / 217 | chapter_2_ranking/5.multi_scenario/2.dynamic_weight |
| `m2m` | 0.6681 / 0.6119 / 217 | 同上 |

### 3.3 文档中的对比组（需整体重跑以更新书中表格）

- [ ] `compare_models(['fm', 'afm', 'nfm', 'pnn', 'fibinet', 'deepfm'])` —— 1.second_order
- [ ] `compare_models(['dcn', 'xdeepfm', 'autoint'])` —— 2.higher_order
- [ ] `compare_models(['din', 'dien', 'dsin'])` —— 3.sequence

### 3.4 召回模型

| 模型 | 数据集 | TF 参考值 | 文档位置 |
| --- | --- | --- | --- |
| `dssm` | ml-1m_recall_pos_neg_data | hr@10 0.0161, hr@5 0.0131, ndcg@10 0.0084, ndcg@5 0.0075, p@10 0.0016, p@5 0.0026 | chapter_1_retrieval/3.two_tower/2.dssm |
| `fm_recall` | ml-1m_recall_pos_neg_data | hr@10 0.0467, hr@5 0.0310, ndcg@10 0.0240, ndcg@5 0.0189, p@10 0.0047, p@5 0.0062 | chapter_1_retrieval/3.two_tower/1.fm |
| `funksvd` | ml-1m_recall_pos_neg_data | hr@10 0.0025, hr@5 0.0013, ndcg@10 0.0012, ndcg@5 0.0008, p@10 0.0002, p@5 0.0003 | chapter_1_retrieval/1.cf/4.mf |
| `biassvd` | ml-1m_recall_pos_neg_data | hr@10 0.0275, hr@5 0.0012, ndcg@10 0.0099, ndcg@5 0.0006, p@10 0.0027, p@5 0.0002 | 同上 |
| `youtubednn` | ml-1m_recall_data | hr@10 0.0126, hr@5 0.0041, ndcg@10 0.0050, ndcg@5 0.0023, p@10 0.0013, p@5 0.0008 | chapter_1_retrieval/3.two_tower/3.youtubednn |
| `eges` | ml-1m_recall_data | hr@10 0.0189, hr@5 0.0129, ndcg@10 0.0097, ndcg@5 0.0078, p@10 0.0019, p@5 0.0026 | chapter_1_retrieval/2.i2i/3.eges |
| `mind` | ml-1m_recall_data | hr@10 0.0058, hr@5 0.0012, ndcg@10 0.0020, ndcg@5 0.0006, p@10 0.0006, p@5 0.0002 | chapter_1_retrieval/4.sequence/1.mind |
| `sdm` | ml-1m_recall_data | hr@10 0.0555, hr@5 0.0513, ndcg@10 0.0356, ndcg@5 0.0342, p@10 0.0055, p@5 0.0103 | chapter_1_retrieval/4.sequence/2.sdm |
| `sasrec` | ml-1m_sasrec | **文档中无 TF 参考值**，需在原 TF 版本上补跑 | — |
| `hstu` | ml-1m_sasrec | **文档中无 TF 参考值**，需在原 TF 版本上补跑 | — |

### 3.5 经典模型（代码本身未改动，但训练/评估流水线的公共代码已改，需回归验证）

| 模型 | 数据集 | TF 参考值 | 文档位置 |
| --- | --- | --- | --- |
| `item_cf` | ml_latest_small_classical | hr@10 0.6594, hr@5 0.5459, p@10 0.1444, p@5 0.1826 | chapter_1_retrieval/1.cf/1.itemcf |
| `swing` | ml_latest_small_classical | hr@10 0.6194, hr@5 0.5042, p@10 0.1282, p@5 0.1629 | chapter_1_retrieval/1.cf/2.swing |
| `user_cf` | ml_latest_small_classical | hr@10 0.6912, hr@5 0.5927, p@10 0.1643, p@5 0.2063 | chapter_1_retrieval/1.cf/3.usercf |
| `item2vec` | ml_latest_small_youtubednn | hr@10 0.0082, hr@5 0.0049, ndcg@10 0.0036, ndcg@5 0.0025, p@10 0.0008, p@5 0.0010 | chapter_1_retrieval/2.i2i/2.item2vec |

### 3.6 重排模型 —— `e_commerce_rerank_data`

| 模型 | TF 参考值 | 文档位置 |
| --- | --- | --- |
| `prm` | map@5 0.2207, p@5 0.0926, old_map@5 0.2824, old_p@5 0.1196 | chapter_3_rerank/2.personalized |
| `prs` | **文档中无 TF 参考值**，需在原 TF 版本上补跑 | — |

---

## 4. web_project 待做验证

- [ ] 离线流水线（ML 数据）：`preprocess_retrieval` → `preprocess_ranking` → `train_retrieval` → `train_ranking` → `local_deploy`；确认生成 `saved_models/*.pt`，`deployed_models/model/*/active.json` 指向 `.pt` 文件。
- [ ] 启动线上服务（`docker compose up` 或 `uvicorn`），请求 `/api/v1/recommend`，确认走 YouTubeDNN 召回 + DeepFM 排序而不是兜底策略，并检查 CPU 推理延迟。
- [ ] Docker 构建：确认 CPU 版 torch 能装上；确认新增挂载 `../src:/app/src:ro` 后容器内能 `import funrec`（线上 `load_model` 需要 funrec 源码来重建模型）。
- [ ] 旧的 TF SavedModel 目录已无法加载，需要重新训练并部署。
- 设备：线上默认 CPU，可通过环境变量 `MODEL_DEVICE=cuda` 切换。

---

## 5. 已知的预期差异（排查指标差异时优先检查）

以下行为在 TF 原版中就存在，迁移时**刻意保持一致**，不是迁移引入的问题：

- **hmoe**：每个 batch 内输出按 domain 重新排序（原 `boolean_mask` + concat），与标签顺序不对齐；预测结果依赖 batch_size。
- **hstu**：`model_config` 以 dict 传给 `HstuLayer`，`getattr` 读取全部落到默认值；attention dropout 在推理时也生效；负样本 logits 广播成 B×B×L。
- **sasrec**：负样本 logits 同样广播成 B×B×L，loss 随 batch 大小缩放。
- **mind**：CapsuleLayer 的 routing logits 在每次前向（包括推理）都会更新，评估结果与调用顺序有关。
- **din / dien / dsin / youtubednn / sdm**：序列与 sparse 特征共享 embedding 表，因此 TF 中并没有 mask 传到注意力、GRU 或 pooling 层，padding 位置也参与计算；迁移版保持不变。
- **prm**：只有第一个 transformer block 收到 padding mask。
- **prs**：`prs_predict` 在 `max_length < L` 时会报错（TF 原版同样如此），评估流程不调用它。
- **dssm**：输出为 sigmoid(cosine)，范围只在 [0.27, 0.73]，这也是 TF 指标偏低的原因。
- **star / PartitionedNormalization**：某个 batch 缺失某个 domain 时，TF 会把该 domain 的 BN 滑动统计量写成 NaN；PyTorch 版跳过更新（**有意的差异**，若指标比 TF 好，可能与此有关）。
- **star**：`tab` 特征直接作为 domain 索引，真实数据上其编码值必须在 `[0, num_domains=5)` 内，否则 PyTorch 会报索引越界（首次运行时注意）。
- **随机性**：负采样（sampled softmax、DIEN 辅助损失、EGES 随机游走）使用 torch / numpy RNG，与 TF 的随机流不同，只在分布上一致。
- **Dice**（din 等）：推理时也用当前 batch 统计量，预测结果依赖 `predict` 的 batch_size（PyTorch 版默认 256，Keras 默认 32）。

---

## 6. 未在本次迁移范围内的内容

- [ ] `docs/` 中的书稿代码片段仍为 TensorFlow 写法（如 1.second_order、1.multi_tower、4.mf 等章节，以及 chapter_10_projects 中关于 SavedModel 的描述）。`docs/` 是构建产物（HTML + `_sources`），需在书稿源文件中修改后重新构建。
- [ ] 跑完实验后，用 PyTorch 版的结果更新书中各章节的指标表格。
- [ ] `docs/_sources/chapter_installation` 的安装说明未加 CUDA 版 torch 的安装步骤。

---

## 7. 迁移期间留下的辅助资源（可按需清理）

- `~/miniconda3/envs/funrec-tf-ref`：TensorFlow 2.13（CPU）参考环境，用于与原版对比。
- `/home/saber812/dev/fun-rec-tf-ref`：`master` 分支的 git worktree（原 TF 代码，只读参考）。不需要后可用 `git worktree remove ../fun-rec-tf-ref` 删除。
- `tmp/agent_scratch/`：各 layer/模型的 TF↔PyTorch 数值等价测试脚本，以及 `audit/smoke_all.py`（全部 config 冒烟测试）和 `audit/roundtrip.py`（保存/加载测试）。`tmp/` 已在 .gitignore 中。
- `tmp/port_models_result.json`：各模型迁移与审查的详细报告。
