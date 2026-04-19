# 特征级多模型集成 MIL 框架（EnsembleFeatureMIL）

> **使用说明**：本文档中所有公式均已采用标准 LaTeX 语法，可直接复制粘贴至以下环境：
> - **Word 论文**：在 Word 中按 `Alt + =` 插入公式，切换到"LaTeX 模式"，粘贴后按 Enter 即可自动转换。
> - **LaTeX 论文**：直接复制，已编号公式已用 `equation` 环境包装，行内公式已用 `$...$` 包装。
> - **公式编号**：每块公式右侧带有编号 `(1)`、`(2)` 等，与论文正文中引用一致。

---

## 一、方法概述

本文提出一种**特征级多实例学习集成框架**（Feature-Level Ensemble MIL）。该框架以五种多实例学习（Multiple Instance Learning, MIL）基模型——RRTMIL、AMIL、WiKG、DSMIL 和 S4MIL——为并行特征提取器，分别从同一全切片图像（Whole Slide Image, WSI）的 patch 特征序列中提取 bag 级表征向量，拼接后通过一个可学习的融合分类头完成最终预测。

---

## 二、网络结构

### 2.1 整体架构

```
                    ┌─────────────────────────────────┐
                    │       Patch Features X           │
                    │     (N × d, e.g. N × 512)        │
                    └────────────┬─────────────────────┘
                                 │
          ┌──────────┬───────────┼───────────┬──────────┐
          ▼          ▼           ▼           ▼          ▼
    ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
    │  RRTMIL  ││   AMIL   ││   WiKG   ││  DSMIL   ││  S4MIL   │
    │          ││          ││          ││          ││          │
    │ 区域关系  ││ 注意力MIL ││ 图知识   ││ 双分类器  ││ 状态空间  │
    │ Transformer││ 门控聚合  ││ 图聚合   ││ 实例+袋   ││ S4D序列  │
    └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
         │           │           │           │           │
         ▼           ▼           ▼           ▼           ▼
      h_RRT      h_AMIL       h_WiKG     h_DSMIL      h_S4
     [1, 256]    [1, 256]    [1, 512]    [1, 512]     [1, 512]
         │           │           │           │           │
         └───────────┴───────────┴───────────┴───────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Concatenate           │
                    │  h_fused ∈ R^(1, d_total) │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Fusion Head           │
                    │  Linear(d_total → 512)   │
                    │  ReLU + Dropout(0.3)     │
                    │  Linear(512 → 256)       │
                    │  ReLU + Dropout(0.2)     │
                    │  Linear(256 → n_classes) │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                         logits → softmax
```

### 2.2 五种基模型的特征提取位置与维度

| 基模型 | 架构类型 | 特征提取位置 | 特征维度 |
|--------|---------|-------------|---------|
| RRTMIL | 区域关系 Transformer | 编码器输出经注意力池化后、分类器前 | $d_{\text{RRT}} = 256$ |
| AMIL | 注意力 MIL | 注意力加权求和后 | $d_{\text{AMIL}} = 256$ |
| WiKG | 弱监督图知识聚合 | 图消息传播 + readout + LayerNorm 后 | $d_{\text{WiKG}} = 512$ |
| DSMIL | 双分类器 MIL | BClassifier 输出的袋级表示 | $d_{\text{DSMIL}} = 512$ |
| S4MIL | S4D 状态空间序列 | S4D 块输出后取序列维 max 池化 | $d_{\text{S4}} = 512$ |

五种特征向量拼接后的总维度为：

```latex
\begin{equation}
d_{\text{total}} = d_{\text{RRT}} + d_{\text{AMIL}} + d_{\text{WiKG}} + d_{\text{DSMIL}} + d_{\text{S4}} = 256 + 256 + 512 + 512 + 512 = 2048
\end{equation}
```

---

## 三、数学公式（可直接插入论文）

> 以下公式按论文正文顺序编号，每道公式均为独立块级环境。复制时包含 `\begin{equation}` 和 `\end{equation}` 完整内容。

### 3.1 问题定义：多实例学习包表示

在全切片图像分析中，每张 WSI 被分割为 $N$ 个图像块（patch），构成一个包（bag）。给定输入包的 patch 特征序列：

```latex
\begin{equation}
X = \{ \boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_N \}, \quad \boldsymbol{x}_i \in \mathbb{R}^{d}
\end{equation}
```

其中 $N$ 为 patch 数量，$d$ 为特征维度（通常为 512），$X \in \mathbb{R}^{N \times d}$。每个 patch 的特征 $\boldsymbol{x}_i$ 由预训练视觉编码器（如 ResNet-50 或 CONCH）提取得到。

### 3.2 RRTMIL 特征提取

RRTMIL 通过区域关系 Transformer（Regional Relation Transformer）对 patch 特征进行编码，利用 R-MSA（区域多头自注意力）和 CR-MSA（跨区域多头自注意力）捕获局部与全局空间依赖，最终经注意力池化得到 bag 级表示：

```latex
\begin{equation}
\boldsymbol{h}_{\text{RRT}} = \text{DAttention}\!\left( \text{RRTEncoder}\!\left( \text{Proj}(X) \right) \right) \in \mathbb{R}^{d_{\text{RRT}}}
\end{equation}
```

其中 $\text{Proj}(\cdot)$ 为 LazyLinear 投影层，$\text{RRTEncoder}(\cdot)$ 由多层 TransLayer 组成，$\text{DAttention}(\cdot)$ 为注意力池化函数，输出维度 $d_{\text{RRT}} = 256$。

### 3.3 AMIL 特征提取

AMIL 采用门控注意力机制对实例特征进行加权聚合。首先通过 MLP 将原始特征投影到 256 维：

```latex
\begin{equation}
\boldsymbol{h}_i = \text{ReLU}\!\left( \boldsymbol{W}_p \boldsymbol{x}_i + \boldsymbol{b}_p \right), \quad \boldsymbol{h}_i \in \mathbb{R}^{256}
\end{equation}
```

然后计算每个实例的注意力权重 $a_i$：

```latex
\begin{equation}
a_i = \frac{\exp\!\left( \boldsymbol{w}^\top \Big( \tanh(\boldsymbol{V} \boldsymbol{h}_i) \odot \sigma(\boldsymbol{U} \boldsymbol{h}_i) \Big) \right)}{\sum\limits_{j=1}^{N} \exp\!\left( \boldsymbol{w}^\top \Big( \tanh(\boldsymbol{V} \boldsymbol{h}_j) \odot \sigma(\boldsymbol{U} \boldsymbol{h}_j) \Big) \right)}
\end{equation}
```

其中 $\boldsymbol{V} \in \mathbb{R}^{128 \times 256}$、$\boldsymbol{U} \in \mathbb{R}^{128 \times 256}$、$\boldsymbol{w} \in \mathbb{R}^{128}$ 为可学习参数，$\sigma(\cdot)$ 为 Sigmoid 激活函数，$\odot$ 为逐元素乘积。最终 bag 级特征为注意力加权和：

```latex
\begin{equation}
\boldsymbol{h}_{\text{AMIL}} = \sum_{i=1}^{N} a_i \boldsymbol{h}_i \in \mathbb{R}^{256}
\end{equation}
```

### 3.4 WiKG 特征提取

WiKG 将 patch 视为图中的节点，通过构造 patch 间的相似度邻接矩阵并取 Top-$K$ 邻居来构建稀疏图。节点间注意力分数定义为：

```latex
\begin{equation}
\text{Attn}_{ij} = \frac{(\boldsymbol{W}_{\text{head}} \boldsymbol{x}_i)^\top (\boldsymbol{W}_{\text{tail}} \boldsymbol{x}_j)}{\sqrt{d_h}}
\end{equation}
```

其中 $\boldsymbol{W}_{\text{head}}, \boldsymbol{W}_{\text{tail}} \in \mathbb{R}^{d_h \times d_h}$ 为可学习权重矩阵，$d_h = 512$。对每个节点取 Top-$K$ 个最高分邻居，归一化得到邻居权重：

```latex
\begin{equation}
p_{ij} = \frac{\exp(\text{Attn}_{ij})}{\sum\limits_{k \in \text{TopK}(i)} \exp(\text{Attn}_{ik})}
\end{equation}
```

通过门控知识注意力融合邻居信息：

```latex
\begin{equation}
\boldsymbol{e}_i^{\text{agg}} = \sum_{j \in \text{TopK}(i)} p_{ij} \cdot \boldsymbol{g}_{ij} \odot \boldsymbol{e}_j
\end{equation}
```

其中门控信号 $\boldsymbol{g}_{ij} = \tanh(\boldsymbol{e}_i + \boldsymbol{e}_j)$。最终采用双交互聚合（bi-interaction aggregation）：

```latex
\begin{equation}
\boldsymbol{h}_{\text{WiKG}} = \text{Readout}\!\left( \text{ReLU}(\boldsymbol{W}_1(\boldsymbol{e}_i + \boldsymbol{e}_i^{\text{agg}})) + \text{ReLU}(\boldsymbol{W}_2(\boldsymbol{e}_i \odot \boldsymbol{e}_i^{\text{agg}})) \right) \in \mathbb{R}^{512}
\end{equation}
```

其中 $\text{Readout}(\cdot)$ 为全局注意力池化函数。

### 3.5 DSMIL 特征提取

DSMIL 采用双分类器策略，包含实例分类器（IClassifier）和袋分类器（BClassifier）。实例分类器直接对每个 patch 进行预测：

```latex
\begin{equation}
\boldsymbol{c}_i = \text{IClassifier}(\boldsymbol{x}_i) \in \mathbb{R}^{C}
\end{equation}
```

袋分类器首先选取得分最高的关键实例 $\boldsymbol{x}_m$（其中 $m = \arg\max_i \boldsymbol{c}_i$），然后计算其余实例相对于关键实例的注意力：

```latex
\begin{equation}
A_i = \text{softmax}\!\left( \frac{\boldsymbol{q}(\boldsymbol{x}_i)^\top \boldsymbol{q}(\boldsymbol{x}_m)}{\sqrt{d_q}} \right)
\end{equation}
```

其中 $\boldsymbol{q}(\cdot)$ 为可学习的非线性投影网络。最终袋级表示为注意力加权聚合后通过一维卷积：

```latex
\begin{equation}
\boldsymbol{h}_{\text{DSMIL}} = \text{Conv1D}\!\left( \sum_{i=1}^{N} A_i \cdot \boldsymbol{v}(\boldsymbol{x}_i) \right) \in \mathbb{R}^{C}
\end{equation}
```

其中 $\boldsymbol{v}(\cdot)$ 为值投影网络。

### 3.6 S4MIL 特征提取

S4MIL 引入结构化状态空间模型 S4D（Diagonal State Space Model）对 patch 序列进行建模。S4D 的核心是在频域执行状态空间卷积：

```latex
\begin{equation}
\boldsymbol{y} = \mathcal{F}^{-1}\!\left( \mathcal{F}(\boldsymbol{K}) \odot \mathcal{F}(\boldsymbol{X}) \right) + \boldsymbol{D} \cdot \boldsymbol{X}
\end{equation}
```

其中 $\mathcal{F}(\cdot)$ 为快速傅里叶变换（FFT），$\boldsymbol{K}$ 为 S4D 核函数在时域的离散化表示，$\boldsymbol{D}$ 为跳跃连接参数。最终取序列维度的最大值作为 bag 级表示：

```latex
\begin{equation}
\boldsymbol{h}_{\text{S4}} = \max_{i=1}^{N} \left( \text{GELU}\!\left( \text{Linear}(\boldsymbol{y}_i) \right) \right) \in \mathbb{R}^{512}
\end{equation}
```

### 3.7 特征拼接

五个基模型各自输出的 bag 级特征向量在通道维度上拼接，得到集成表示：

```latex
\begin{equation}
\boldsymbol{h}_{\text{fused}} = \left[ \boldsymbol{h}_{\text{RRT}}; \; \boldsymbol{h}_{\text{AMIL}}; \; \boldsymbol{h}_{\text{WiKG}}; \; \boldsymbol{h}_{\text{DSMIL}}; \; \boldsymbol{h}_{\text{S4}} \right] \in \mathbb{R}^{d_{\text{total}}}
\end{equation}
```

其中 $[\cdot;\cdot]$ 表示向量拼接操作，$d_{\text{total}} = 2048$。

### 3.8 融合分类头

融合头采用两层多层感知机（MLP）结构，将拼接后的集成表示映射为最终的分类输出：

```latex
\begin{equation}
\boldsymbol{h}^{(1)} = \text{ReLU}\!\left( \boldsymbol{W}^{(1)} \boldsymbol{h}_{\text{fused}} + \boldsymbol{b}^{(1)} \right), \quad \boldsymbol{W}^{(1)} \in \mathbb{R}^{512 \times d_{\text{total}}}, \; \boldsymbol{b}^{(1)} \in \mathbb{R}^{512}
\end{equation}
```

```latex
\begin{equation}
\boldsymbol{h}^{(2)} = \text{ReLU}\!\left( \boldsymbol{W}^{(2)} \boldsymbol{h}^{(1)} + \boldsymbol{b}^{(2)} \right), \quad \boldsymbol{W}^{(2)} \in \mathbb{R}^{256 \times 512}, \; \boldsymbol{b}^{(2)} \in \mathbb{R}^{256}
\end{equation}
```

```latex
\begin{equation}
\boldsymbol{z} = \boldsymbol{W}^{(3)} \boldsymbol{h}^{(2)} + \boldsymbol{b}^{(3)}, \quad \boldsymbol{W}^{(3)} \in \mathbb{R}^{C \times 256}, \; \boldsymbol{b}^{(3)} \in \mathbb{R}^{C}
\end{equation}
```

最终通过 Softmax 函数得到类别概率分布：

```latex
\begin{equation}
P_c = \frac{\exp(z_c)}{\sum\limits_{k=1}^{C} \exp(z_k)}, \quad c = 1, 2, \ldots, C
\end{equation}
```

### 3.9 损失函数

训练阶段采用交叉熵损失函数：

```latex
\begin{equation}
\mathcal{L}_{\text{CE}} = - \sum_{c=1}^{C} \mathbb{I}(y = c) \cdot \log P_c
\end{equation}
```

其中 $y \in \{1, 2, \ldots, C\}$ 为真实类别标签，$\mathbb{I}(\cdot)$ 为指示函数，$P_c$ 为模型预测的第 $c$ 类概率。

### 3.10 优化目标（冻结基模型模式）

在冻结基模型模式下，五个基模型的参数固定不变，仅优化融合分类头的参数 $\boldsymbol{\theta}_{\text{fusion}} = \{ \boldsymbol{W}^{(1)}, \boldsymbol{b}^{(1)}, \boldsymbol{W}^{(2)}, \boldsymbol{b}^{(2)}, \boldsymbol{W}^{(3)}, \boldsymbol{b}^{(3)} \}$：

```latex
\begin{equation}
\boldsymbol{\theta}_{\text{fusion}}^{\ast} = \arg\min_{\boldsymbol{\theta}_{\text{fusion}}} \; \mathbb{E}_{(X, y) \sim \mathcal{D}} \left[ \mathcal{L}_{\text{CE}}\!\left( g_{\boldsymbol{\theta}_{\text{fusion}}}(\boldsymbol{h}_{\text{fused}}(X)), y \right) \right]
\end{equation}
```

其中 $\mathcal{D}$ 为训练数据分布，$g_{\boldsymbol{\theta}_{\text{fusion}}}(\cdot)$ 表示融合分类头的映射函数。

---

## 四、论文中可直接引用的公式汇总表

| 编号 | 公式名称 | LaTeX 代码 |
|------|---------|-----------|
| (1) | Patch 特征序列定义 | 见 3.1 节 |
| (2) | RRTMIL bag 级特征 | 见 3.2 节 |
| (3) | AMIL 实例投影 | 见 3.3 节 |
| (4) | AMIL 注意力权重 | 见 3.3 节 |
| (5) | AMIL 加权聚合 | 见 3.3 节 |
| (6) | WiKG 注意力分数 | 见 3.4 节 |
| (7) | WiKG 邻居权重 | 见 3.4 节 |
| (8) | WiKG 门控聚合 | 见 3.4 节 |
| (9) | WiKG 双交互聚合 | 见 3.4 节 |
| (10) | DSMIL 实例预测 | 见 3.5 节 |
| (11) | DSMIL 注意力 | 见 3.5 节 |
| (12) | DSMIL 袋级表示 | 见 3.5 节 |
| (13) | S4MIL 频域卷积 | 见 3.6 节 |
| (14) | S4MIL max 池化 | 见 3.6 节 |
| (15) | 特征拼接 | 见 3.7 节 |
| (16) | 融合头第一层 | 见 3.8 节 |
| (17) | 融合头第二层 | 见 3.8 节 |
| (18) | 融合头输出层 | 见 3.8 节 |
| (19) | Softmax 概率 | 见 3.8 节 |
| (20) | 交叉熵损失 | 见 3.9 节 |
| (21) | 优化目标 | 见 3.10 节 |
| (22) | 总维度 | 见 2.2 节 |

---

## 五、使用说明

### 5.1 环境要求

确保已安装项目所需依赖（参考 ViLa-MIL 项目的 `requirements.txt`）。

### 5.2 第一步：训练五个基模型

进入项目目录：

```bash
cd /Users/zzfly/毕设/vila-mil/ViLa-MIL
```

分别训练五个基模型：

```bash
# RRTMIL
python main_LUSC.py \
    --model_type RRTMIL \
    --exp_code RRTMIL_baseline \
    --results_dir results/RRTMIL \
    --max_epochs 100 \
    --lr 1e-5 \
    --k 4 \
    --early_stopping

# AMIL
python main_LUSC.py \
    --model_type AMIL \
    --exp_code AMIL_baseline \
    --results_dir results/AMIL \
    --max_epochs 100 \
    --lr 1e-5 \
    --k 4 \
    --early_stopping

# WiKG
python main_LUSC.py \
    --model_type WiKG \
    --exp_code WiKG_baseline \
    --results_dir results/WiKG \
    --max_epochs 100 \
    --lr 1e-5 \
    --k 4 \
    --early_stopping

# DSMIL
python main_LUSC.py \
    --model_type DSMIL \
    --exp_code DSMIL_baseline \
    --results_dir results/DSMIL \
    --max_epochs 100 \
    --lr 1e-5 \
    --k 4 \
    --early_stopping

# S4MIL
python main_LUSC.py \
    --model_type S4MIL \
    --exp_code S4MIL_baseline \
    --results_dir results/S4MIL \
    --max_epochs 100 \
    --lr 1e-5 \
    --k 4 \
    --early_stopping
```

每个模型训练完成后，会在对应 `results_dir` 下生成 `s_0_checkpoint.pt`、`s_1_checkpoint.pt`、`s_2_checkpoint.pt`、`s_3_checkpoint.pt`（对应 4 折交叉验证）。

### 5.3 第二步：基模型 Checkpoint（后端自动解析，默认无需手动收集）

训练 **EnsembleFeature** 且**未**传入 `--ensemble_ckpt_dir` 时，`utils/core_utils.py` 会在**每一个训练折** `cur` 上，自动从 ViLa-MIL 根目录下：

- `uploaded_features/best_models.json`（键：`{癌种}:{基模型名}:{mode}`，如 `LUSC:RRTMIL:transformer`）
- `uploaded_features/tasks.json`（根据各基线的 `bestTaskId` 取 `resultsDir`）

解析五个基模型（RRTMIL、AMIL、WiKG、DSMIL、S4MIL）的权重文件路径，并加载：

- `s_{cur}_checkpoint.pt`（与 `main_LUSC.py` 多折训练保存方式一致；亦支持在 `resultsDir` 子目录中 `os.walk` 找到同名文件）

**你需要做的**：在平台（或命令行）上分别训练好五个基模型，使 `best_models.json` 中为当前癌种与 `mode`（如 `transformer`）都写好 `bestTaskId`，且对应任务目录下存在**当前折**的 `s_cur_checkpoint.pt`。

**注意**：

- 五个基模型建议使用**相同折数 `--k`** 训练，否则某一折可能缺少 `s_cur_checkpoint.pt`，自动解析会失败并打印告警（此时该折基线相当于未加载预训练）。
- **可选**：仍支持手动目录 `--ensemble_ckpt_dir`：目录内放 5 个 `.pt`，文件名需包含 `RRTMIL` / `AMIL` / … 子串（与旧版逻辑相同），**优先级高于自动解析**。
- **可选**：`--ensemble_disable_auto_ckpt` 关闭自动解析（仅在不传 `ensemble_ckpt_dir` 时生效）；`--ensemble_best_models_json` / `--ensemble_tasks_json` 可覆盖默认 JSON 路径。

### 5.4 第三步：训练集成模型

集成模型支持两种训练策略（**默认无需 `--ensemble_ckpt_dir`**，依赖上节自动解析）：

#### 策略 A：冻结基模型，仅训练融合头（推荐）

```bash
python main_LUSC.py \
    --model_type EnsembleFeature \
    --exp_code Ensemble_frozen \
    --results_dir results/Ensemble_frozen \
    --max_epochs 100 \
    --lr 1e-3 \
    --k 4 \
    --early_stopping \
    --freeze_base
```

**参数说明**：
- `--freeze_base`：冻结五个基模型的所有参数，仅优化融合分类头
- `--lr 1e-3`：冻结模式下仅训练融合头，可使用较大学习率
- 若需手动权重目录，追加：`--ensemble_ckpt_dir <目录>`

#### 策略 B：端到端微调所有参数

```bash
python main_LUSC.py \
    --model_type EnsembleFeature \
    --exp_code Ensemble_finetune \
    --results_dir results/Ensemble_finetune \
    --max_epochs 100 \
    --lr 1e-5 \
    --k 4 \
    --early_stopping \
    --finetune_ensemble
```

**参数说明**：
- `--finetune_ensemble`：解冻五个基线 MIL，与融合头一起训练（代码中与默认 `--freeze_base` 互斥）
- `--lr 1e-5`：微调模式需使用较小学习率，避免破坏预训练权重
- 同样可追加 `--ensemble_ckpt_dir` 使用手动目录而非自动解析

### 5.5 第四步：查看结果

训练完成后，结果保存在 `results_dir` 下：

```
results/Ensemble_frozen/
├── 0/                          # 第 0 折
│   ├── s_0_checkpoint.pt       # 最优模型权重
│   └── splits_0.csv
├── 1/
├── 2/
├── 3/
└── logging/                    # 训练日志
```

---

## 六、消融实验设计

论文中建议做以下消融实验，以验证各组件的有效性。

### 6.1 逐一排除消融（Leave-One-Out Ablation）

每次去掉一个基模型的特征，观察集成模型性能变化。实现方式：在 `EnsembleFeatureMIL` 的 `forward` 方法中将对应特征替换为零向量。

| 实验编号 | 使用的特征 | 目的 |
|---------|-----------|------|
| Full | RRT + AMIL + WiKG + DSMIL + S4 | 完整集成 |
| w/o RRT | AMIL + WiKG + DSMIL + S4 | 验证 Transformer 特征的贡献 |
| w/o AMIL | RRT + WiKG + DSMIL + S4 | 验证注意力特征的贡献 |
| w/o WiKG | RRT + AMIL + DSMIL + S4 | 验证图结构特征的贡献 |
| w/o DSMIL | RRT + AMIL + WiKG + S4 | 验证双分类器特征的贡献 |
| w/o S4 | RRT + AMIL + WiKG + DSMIL | 验证状态空间特征的贡献 |

论文中可以用柱状图展示各配置的 AUC / F1 分数，直观比较。

### 6.2 训练策略对比

| 实验编号 | 策略 | 学习率 | 可训练参数量 |
|---------|------|--------|-------------|
| Frozen | 冻结基模型 | 1e-3 | 仅融合头（约 1.3M） |
| Finetune | 端到端微调 | 1e-5 | 全部参数 |
| Random Init | 随机初始化基模型 | 1e-5 | 全部参数 |

### 6.3 融合头深度消融

| 融合头结构 | 参数量 |
|-----------|--------|
| 单层：Linear → n_classes | 最少 |
| 两层：Linear → ReLU → Linear → n_classes（默认） | 适中 |
| 三层：Linear → ReLU → Linear → ReLU → Linear → n_classes | 最多 |

---

## 七、与基模型的对比实验

论文中可以设计如下对比表格：

| 模型 | AUC | Accuracy | F1 | 参数量 | 推理时间/张片 |
|------|-----|----------|----|--------|--------------|
| RRTMIL | -- | -- | -- | -- | -- |
| AMIL | -- | -- | -- | -- | -- |
| WiKG | -- | -- | -- | -- | -- |
| DSMIL | -- | -- | -- | -- | -- |
| S4MIL | -- | -- | -- | -- | -- |
| **EnsembleFeature (Ours)** | -- | -- | -- | -- | -- |

> 注意：集成模型的参数量 ≈ 五个基模型参数量之和 + 融合头参数量。推理时间也近似为五个基模型推理时间之和（因为前向传播需依次执行五个基模型）。

---

## 八、论文文字素材（可直接引用）

### 8.1 方法章节示例段落

> **特征级多模型集成 MIL**  全切片图像（WSI）的病理学分析中，不同 MIL 架构从不同视角对 patch 特征进行建模：RRTMIL 通过区域关系 Transformer 捕获局部空间依赖；AMIL 利用门控注意力机制识别关键实例；WiKG 构建 patch 图进行消息传递；DSMIL 采用实例级与袋级双分类器策略；S4MIL 引入状态空间模型处理长序列。然而，单一模型仅利用了某一类归纳偏置，难以充分挖掘 WSI 中的多尺度、多模式信息。
>
> 为此，本文提出一种特征级多模型集成框架。该框架将五个基模型视为并行特征提取器，分别从 patch 特征序列中提取 bag 级表征，拼接后通过一个两层 MLP 融合分类头进行最终预测。集成表示 $\boldsymbol{h}_{\text{fused}}$ 定义为：
>
> $$\boldsymbol{h}_{\text{fused}} = \left[ \boldsymbol{h}_{\text{RRT}}; \boldsymbol{h}_{\text{AMIL}}; \boldsymbol{h}_{\text{WiKG}}; \boldsymbol{h}_{\text{DSMIL}}; \boldsymbol{h}_{\text{S4}} \right]$$
>
> 训练阶段采用两阶段策略：首先独立训练五个基模型至收敛，然后冻结其参数，仅优化融合分类头。该策略避免了端到端训练的高显存需求，同时确保各基模型的特征表示不受其他模型梯度干扰。

### 8.2 实验分析示例段落

> **消融实验分析**  为验证各基模型对集成性能的贡献，我们进行了逐一排除消融实验（表 X）。结果表明，移除任一基模型均导致 AUC 下降，其中 RRTMIL 和 WiKG 的去除影响最为显著，说明 Transformer 捕获的全局空间依赖与图结构建模的局部拓扑信息在该任务中最为关键。同时，完整集成模型在五项指标上均优于任一单一基模型，证明了多视角特征互补的有效性。

---

## 九、常见问题

### Q1：显存不够怎么办？

使用 `--freeze_base` 冻结基模型参数后，反向传播仅需为融合头计算梯度，显存占用大幅下降。如果仍然不够，可以：
1. 减少融合头的隐藏层维度（修改 `EnsembleFeature.py` 中 `fusion_head` 的 512/256 为更小的值）
2. 使用梯度累积（需要自行修改训练循环）

### Q2：没有预训练权重 / 自动解析失败怎么办？

若 `best_models.json` 与 `tasks.json` 中无法为当前癌种、`mode` 和**当前折**解析出五个 `s_{fold}_checkpoint.pt`，则**不会**加载基线预训练（会打印 Warning），等价于五个基线随机初始化再与融合头一起训练，**显存与时间开销大**。

解决：先在平台训练五个基模型并产生 best 记录；或显式传入 `--ensemble_ckpt_dir` 指向含 5 个权重文件的目录；调试可用 `--ensemble_disable_auto_ckpt` 强制不走自动解析（需自行保证训练意图）。

### Q3：如何选择冻结还是微调？

- **冻结模式**：显存需求低、训练快、不易过拟合，适合数据量小的场景。是本科毕设的推荐选择。
- **微调模式**：理论上性能上限更高，但需要更多显存和训练时间，且需小心调节学习率。

### Q4：nll_loss / nll_loss_soft 报错？

这两个函数是生存分析专用的损失函数，在你的项目中可能未定义。标准分类任务走的是 `CrossEntropyLoss` 分支（`elif label is not None`），不会触发 `nll_loss` 调用。如果你的训练触发了 `nll_loss` 报错，请确认 `disc` 和 `staus` 参数是否为 `None`——在标准训练流程中它们应为 `None`。

### Q5：公式在 Word 中粘贴后乱码？

确保在 Word 中：
1. 按 `Alt + =` 插入公式
2. 在公式工具栏中选择"LaTeX"模式（不是"Unicode"）
3. 粘贴后按 Enter 键自动转换
4. 如果仍有问题，检查 Word 版本是否为 Microsoft 365 或 Word 2021+（旧版本可能不支持 LaTeX）

---

## 十、文件结构

本项目新增/修改的文件：

```
ViLa-MIL/
├── models/
│   └── EnsembleFeature.py           ← 集成模型定义
├── utils/
│   ├── core_utils.py              ← EnsembleFeature 训练与加载逻辑
│   └── ensemble_ckpt_resolve.py   ← 从 best_models/tasks 自动解析五基线 checkpoint
├── main_LUSC.py                   ← --freeze_base、--ensemble_ckpt_dir、--finetune_ensemble、自动解析相关参数
└── docs/
    └── EnsembleFeature使用说明.md ← 本文件
```
