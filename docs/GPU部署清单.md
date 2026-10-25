# GPU 部署清单（RTX 3080 20G）

> **一句话**：软件这边已经装好，23 号管理员把卡挂上来、驱动一到位，**当天直接能用**，
> 不用在比赛第一天等几个 G 的下载。

---

## 1. 先看清楚：这台机器现在是什么状况

实测（2026-10-21）：

| 项 | 现状 |
|---|---|
| 物理显卡 | **没有 NVIDIA 卡**。PCI 总线上 304 个设备，唯一的显示设备是 ASPEED（服务器的管理显卡） |
| `nvidia-smi` | 不存在 |
| 内核 | 5.19.11-051911-generic（Ubuntu 20.04.5 底座，Pop!_OS 风格内核） |
| 内存/磁盘 | `/home` 剩 68 G（装完深度学习栈后仍有余量） |

**结论**：这是一台虚拟机，GPU 是 23 号由管理员**挂载/直通**进来的。
所以现在能做、且应该做的事只有一件：**把软件全部准备好**。

---

## 2. 管理员需要做的事（就两件）

### 2.1 装上 NVIDIA 驱动，版本 ≥ 525

```bash
nvidia-smi
```

看到 `RTX 3080` 和驱动版本号，就算成功。

**一定要说清楚的两点：**

1. **不需要装完整 CUDA Toolkit。**
   我们的 PyTorch 轮子自带 CUDA 运行时（cu126），装 Toolkit 是白占几个 G。
   只需要**驱动**。

2. **如果是 vGPU（虚拟化切分），必须用厂商配套的 vGPU 驱动。**
   普通桌面驱动在 vGPU 上不工作。这一点要跟管理员确认是"直通"还是"vGPU"。

| 驱动版本 | 能用什么 |
|---|---|
| ≥ 525 | CUDA 12.x（我们的默认选择 cu126） |
| 470–524 | 只有 CUDA 11.8，需要改用 cu118 轮子 |
| < 470 | 都跑不了，必须升级 |

> 脚本会自动读驱动版本挑对应的 CUDA 轮子，但 **≥525 最省事**。

### 2.2 不需要做的

- 不用装 conda（已经有）
- 不用装 CUDA Toolkit
- 不用改现有 `mathorcup` 环境（我们另建了一个）

---

## 3. 我们装了什么

**独立环境 `mathorcup-gpu`**，由 `mathorcup` 克隆而来，所以三个树模型的版本**完全一致**，
对比结果才可信。**`mathorcup` 环境一个包都没动。**

### 三层组件

| 层 | 包 | 干什么 |
|---|---|---|
| **表格**（主力） | torch 2.7.1 (cu126) + torchvision 0.22.1 | 深度学习底座 |
| | **pytabkit 1.7.3** → RealMLP / FT-Transformer | 表格深度模型，2024 NeurIPS《Better by Default》 |
| | **xgboost（pip 版）** | CUDA 加速 |
| | **catboost（pip 版）** | GPU 训练 |
| **视觉**（保险） | timm | 预训练骨干网，迁移学习 |
| | OpenCV | 图像读写与预处理 |
| **文本**（保险） | transformers | 预训练语言模型 |
| | sentence-transformers | 把文本编码成向量，喂给树模型 |

### 为什么单独建环境

torch 会牵动 numpy 等底层依赖。**绝不能为了装 torch 去动那个已经用 88 篇文献、
一整套门控验证过的竞赛环境。** 克隆一份，各过各的。

---

## 4. 23 号当天怎么做（三条命令）

```bash
# 1. 体检。必须全绿才算真的能用 GPU
bash scripts/setup_gpu.sh --check
```

```bash
# 2. 树模型用 GPU（配置一行都不用改）
make train GPU=1 CONFIG=src/configs/xgboost.yaml
```

```bash
# 3. 表格深度模型
make dl CONFIG=src/configs/baseline.yaml PY=/home/eto/miniconda3/envs/mathorcup-gpu/bin/python
```

**如果驱动还没到位**：`bash scripts/setup_gpu.sh --check` 会明确告诉你卡在哪一步，
不会含糊地"看起来装好了"。

---

## 5. 已知的坑（全部实测，不是听说）

### 5.1 🔴 conda 装的 XGBoost 是 CPU-only 构建

在本机实测：

```python
>>> xgboost.build_info()['USE_CUDA']
False
```

参数写对了、程序不报错、结果照出——**就是慢十倍，而你完全看不出来**。
这就是为什么必须 `pip install --force-reinstall xgboost`，而不是相信 conda。

`scripts/gpu_check.py` 专门查这一条，`USE_CUDA=False` 直接判 FAIL。

### 5.2 LightGBM 的 pip 轮子从来没有 CUDA

库文件里的原文：

```
CUDA Tree Learner was not enabled in this build.
Please recompile with CMake option -DUSE_CUDA=1
```

要用 GPU 版得自己编译（CUDA Toolkit + cmake + 匹配 gcc）。**赛期 7 天，不值得。**
LightGBM 在 CPU 上本来就快，GPU 加速交给 XGBoost / CatBoost。

> 所以 `make train GPU=1` 遇到 lightgbm 只会**警告一句**，不会假装成功。

### 5.3 CatBoost 的 GPU 限制

从 `_catboost.so` 里直接读出的报错文案：

| 限制 | 原文 |
|---|---|
| `rsm` 在 GPU 上只支持 pairwise 模式 | `rsm on GPU is supported for pairwise modes only` |
| MVS 采样不支持多分类 GPU | `MVS is not supported for multiclass models on GPU` |
| 不接受多个验证集 | `Multiple eval sets are not supported on GPU` |

我们的 `catboost.yaml` 里有 `rsm: 1.0`，直接上 GPU 会报错。
`src/models.py` 的 `apply_gpu_override()` 会自动把它去掉（1.0 本来就是"不采样特征"，去掉无损失）。

### 5.4 `nvidia-smi` 存在 ≠ 驱动能用

本机现在就有一个 `/usr/bin/nvidia-smi`，但**没有驱动**。没有驱动时它会把
`NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`
打到 **stdout**（不是 stderr）。如果脚本直接拿它当驱动版本号，bash 会把这段
文字当算术表达式求值，报出莫名其妙的 `NVIDIA: unbound variable`。

`scripts/setup_gpu.sh` 现在会校验结果必须是纯数字，否则按「没驱动」处理。
**23 号如果看到这句 warning，说明卡还没挂上，不是脚本坏了。**

### 5.5 参数名不通用（老坑，已有守卫）

LightGBM 的 `num_leaves`、`min_child_samples` 喂给 XGBoost 会被**静默忽略**。
`validate_params()` 已经拦着。GPU 参数同样走这个检查。

---

## 6. "装了" ≠ "在用"

这是整个部署里最重要的一条。

`scripts/gpu_check.py` 不接受"import 成功"当证据，每一条都实测：

| 检查 | 怎么证明 |
|---|---|
| torch 真在 GPU 上算 | 跑 4096³ 矩阵乘，并检查**显存峰值真的涨了**（否则说明数据被搬回 CPU 算了） |
| XGBoost 真在 GPU 上 | `build_info()['USE_CUDA']` + 实际 `device='cuda'` 训练一次 |
| CatBoost 真在 GPU 上 | `task_type='GPU'` 实际训练一次 |
| RealMLP 真在 GPU 上 | 训练前后 `max_memory_allocated()` 的差值 |
| timm 真在 GPU 上 | 模型和输入都 `.cuda()`，跑一次前向 |

而且它自己也要被证伪：

```bash
python scripts/gpu_check.py --falsify
```

把 GPU 藏起来（`CUDA_VISIBLE_DEVICES=""`）再跑一遍，**体检必须变红**。
永远通过的守卫是假守卫——这条规矩抄自 LUAD 项目的审计流程。

---

## 7. 如果赛题是视觉 / 文本

用户提到"有可能有视觉模型或者其他东西"。两条现成的起手式：

### 视觉：迁移学习，不要从头训

```
预训练骨干（timm, ImageNet 权重）
    ├─ 冻结主干 → 提特征 → 接 LightGBM（小数据、快、稳）
    └─ 解冻最后几层 → 微调（数据多时用）
```

**20G 显存跑 ResNet / EfficientNet / ViT-S 绰绰有余。**
小样本下"冻结 + 树模型头"往往比端到端微调更稳，也更省时间。

### 文本：编码成向量，当数值特征用

```
sentence-transformers 编码 → 若干维向量 → 和表格特征拼在一起 → 树模型
```

这条路的好处是**不需要改现有流水线**——文本变成几列数值，`src/features.py` 直接吃。

> 注意：首次用预训练模型要下载权重。国内网络可能卡，
> 必要时设 `export HF_ENDPOINT=https://hf-mirror.com`。

---

## 8. 值不值得：GPU 的收益到底在哪

诚实说，**表格数据上 GPU 的收益主要在调参，不在训练**。

| 用途 | 收益 | 优先级 |
|---|---|---|
| **Optuna 搜参**（XGBoost/CatBoost GPU） | 同样时间能试 **10–50 倍**的组合 | 🔴 最高 |
| **RealMLP** | 多一个异构专家，可能帮融合 | 🟡 中 |
| 视觉/文本 | 只有赛题真的需要才用 | 🟡 视赛题 |
| LightGBM 加速 | 做不到（无 CUDA），且它本来就快 | ⚪ 放弃 |

所以 23 号拿到卡之后，**第一件事不是跑深度学习，是把搜参预算放大**。

---

## 9. 体检清单（打印出来贴墙上）

23 号那天，按顺序打勾：

- [ ] `nvidia-smi` 能看到 RTX 3080、驱动 ≥ 525
- [ ] `bash scripts/setup_gpu.sh --check` 全绿
- [ ] `python scripts/gpu_check.py --falsify` 通过（守卫可被证伪）
- [ ] `make train GPU=1` 在 xgboost 配置上比 CPU 明显快
- [ ] `make dl ... --epochs 8` 能跑通（先小轮数冒烟，再放开）
- [ ] `mathorcup` 环境未被改动：`make check` 仍全绿

最后一条别省。**部署新东西最大的风险是弄坏了原来能用的东西。**
