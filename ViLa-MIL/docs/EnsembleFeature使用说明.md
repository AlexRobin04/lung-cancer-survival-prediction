# 特征级多模型集成 MIL 框架（EnsembleFeatureMIL）

> **姊妹篇**：[**`EnsembleFeature使用说明-数学预览版.md`**](./EnsembleFeature使用说明-数学预览版.md) — 公式用 `$` / `$$` 写成，在 Cursor / GitHub 等预览里按数学式渲染。  
> **本文件**仍保留 `\begin{equation}` 代码块，便于复制到 Word / LaTeX 论文。

> **使用说明**：本文档中所有公式均已采用标准 LaTeX 语法，可直接复制粘贴至以下环境：
> - **Word 论文**：在 Word 中按 `Alt + =` 插入公式，切换到"LaTeX 模式"，粘贴后按 Enter 即可自动转换。
> - **LaTeX 论文**：直接复制，已编号公式已用 `equation` 环境包装，行内公式已用 `$...$` 包装。
> - **公式编号**：每块公式右侧带有编号 `(1)`、`(2)` 等，与论文正文中引用一致。

---

## 一、方法概述

本文提出一种**特征级多实例学习集成框架**（Feature-Level Ensemble MIL）。该框架以五种多实例学习（Multiple Instance Learning, MIL）基模型——RRTMIL、AMIL、WiKG、DSMIL 和 S4MIL——为并行特征提取器，分别从同一全切片图像（Whole Slide Image, WSI）的 patch 特征序列中提取 bag 级表征向量。

**当前实现（相对早期「裸拼接」的扩展）**：在融合前，对各路 bag 特征先做 **LayerNorm + 线性投影** 到统一维度 \(D\)（默认 `feature_align_dim=512`），以缓解不同基线输出**尺度与分布差异**导致的梯度主导问题。融合阶段支持两种模式——**门控加权融合（`fusion_mode=gate`，默认）**：由五路对齐特征联合经小型 MLP 产生 5 个 softmax 权重，对对齐后的向量加权求和后再送入融合 MLP；**对齐后拼接（`fusion_mode=concat`）**：将五路对齐向量拼接为 \(\mathbb{R}^{5D}\) 后送入融合 MLP，便于与门控模式做消融对照。

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
   (各基线维数   依实现而定，见 2.2 表；此处为原始 bag 维)
         │           │           │           │           │
         ▼           ▼           ▼           ▼           ▼
     LN+Proj    LN+Proj     LN+Proj     LN+Proj     LN+Proj
         │           │           │           │           │
      z_RRT      z_AMIL      z_WiKG     z_DSMIL      z_S4
      各 ∈ R^D   各 ∈ R^D    各 ∈ R^D    各 ∈ R^D     各 ∈ R^D
         │           │           │           │           │
         └───────────┴───────────┴───────────┴───────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼ (默认 gate)                          ▼ (concat 消融)
     ┌─────────────────────┐            ┌─────────────────────┐
     │ cat(z*) → gate_mlp  │            │ Concatenate         │
     │ → softmax(w)∈R^5    │            │  h_cat ∈ R^(5D)     │
     │ h=Σ_i w_i z_i ∈R^D  │            └──────────┬──────────┘
     └──────────┬──────────┘                       │
                │                                   │
                └───────────────┬───────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │    Fusion Head (MLP)    │
                    │  gate: Linear(D→512)…   │
                    │  concat: Linear(5D→512)…│
                    │  ReLU + Dropout …       │
                    │  Linear(256 → n_classes)│
                    └────────────┬────────────┘
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

记各基线**原始** bag 维为 \(d_{\text{RRT}}, d_{\text{AMIL}}, \ldots\)（与代码一致；RRT 维数由 `online_encoder.final_dim` 决定，未必为 256）。**对齐后**统一为 \(D =\) `feature_align_dim`（默认 512），得到 \(\boldsymbol{z}_{\text{RRT}}, \ldots \in \mathbb{R}^{D}\)。

- **`fusion_mode=concat`（消融）**：融合头第一层输入维为 \(5D\)（默认 \(5 \times 512 = 2560\)）。若不做对齐而直接拼接原始 \(\boldsymbol{h}\)，则等价于早期 \(d_{\text{total}} = \sum d_{(\cdot)}\) 形式（如 \(2048\)），当前代码在 concat 模式下使用的是**对齐后**的 \(5D\)。
- **`fusion_mode=gate`（默认）**：先算 \(\boldsymbol{w} \in \mathbb{R}^5\)，再 \(\boldsymbol{h}_{\text{fuse}} = \sum_i w_i \boldsymbol{z}_i \in \mathbb{R}^D\)，融合头第一层输入维为 \(D\)。

---

### 2.3 代码与接口一览（对齐 / 门控）

| 项目 | 说明 |
|------|------|
| 实现文件 | `models/EnsembleFeature.py`：`EnsembleFeatureMIL(..., feature_align_dim=512, fusion_mode="gate"\|"concat")` |
| 命令行 | `main_LUSC.py`：`--ensemble_fusion gate`（默认）或 `concat`；与 `--freeze_base` / `--finetune_ensemble` 等组合使用 |
| HTTP API | `POST /api/training/start` JSON 字段：`ensembleFusion` 或 `ensemble_fusion`，取值 `gate` / `concat`；排队任务写入 `ensembleFusion`，调度启动时带入子进程 |
| Web 前端 | `vila-mil-frontend` 训练页：选择 `EnsembleFeature` 时，「融合方式」下拉框（门控加权 / 对齐后拼接） |
| 推理加载 | `api_server.py` 中根据 checkpoint 是否含 `gate_mlp.*` 权重自动选择 `fusion_mode`，以兼容仅 concat 结构的旧权重 |

**权重兼容性**：门控结构含 `gate_mlp` 与「按 \(D\) 输入」的 `fusion_head`，与**仅裸拼接、无对齐层**的旧 checkpoint **不通用**；升级后需重新训练集成任务，或显式使用 `concat` 并对齐到与旧实验相同的结构定义后再做迁移学习。

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

> **说明**：下式描述的是**原始 bag 向量直接拼接**的理想化记号（各 \(\boldsymbol{h}\) 维数见 2.2 节）。当前代码在融合前增加了 **3.11 节的对齐映射** 与 **3.12–3.13 节的门控/拼接融合**；若论文正文采用「裸拼接」叙述，可将 \(\boldsymbol{h}\) 理解为 \(\boldsymbol{z}\) 之前的原始表征。

五个基模型各自输出的 bag 级特征向量在通道维度上拼接，得到集成表示：

```latex
\begin{equation}
\boldsymbol{h}_{\text{fused}} = \left[ \boldsymbol{h}_{\text{RRT}}; \; \boldsymbol{h}_{\text{AMIL}}; \; \boldsymbol{h}_{\text{WiKG}}; \; \boldsymbol{h}_{\text{DSMIL}}; \; \boldsymbol{h}_{\text{S4}} \right] \in \mathbb{R}^{d_{\text{total}}}
\end{equation}
```

其中 $[\cdot;\cdot]$ 表示向量拼接操作，$d_{\text{total}} = 2048$。

### 3.8 融合分类头

融合头采用两层多层感知机（MLP）结构，将**融合后的向量**（裸拼接时为 \(\boldsymbol{h}_{\text{fused}}\)；门控模式下为 \(\boldsymbol{h}_{\text{fuse}} \in \mathbb{R}^{D}\)；对齐后拼接模式下为 \(\boldsymbol{h}_{\text{cat}} \in \mathbb{R}^{5D}\)）映射为最终的分类输出。以下公式中 \(\boldsymbol{h}_{\text{fused}}\) 可视为上述三种输入之一的占位记号；第一层权重 \(\boldsymbol{W}^{(1)}\) 的列维随实现为 \(d_{\text{total}}\)、\(D\) 或 \(5D\)。

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

在冻结基模型模式下，五个基模型的参数固定不变，仅优化**对齐层、（若启用）门控 MLP、以及融合分类头**的可学习参数；记其合并为 $\boldsymbol{\theta}_{\text{head}}$（实现中即 `requires_grad=True` 的 `align_*`、`gate_mlp`、`fusion_head`）：

```latex
\begin{equation}
\boldsymbol{\theta}_{\text{head}}^{\ast} = \arg\min_{\boldsymbol{\theta}_{\text{head}}} \; \mathbb{E}_{(X, y) \sim \mathcal{D}} \left[ \mathcal{L}_{\text{CE}}\!\left( f_{\boldsymbol{\theta}_{\text{head}}}(X), y \right) \right]
\end{equation}
```

其中 $\mathcal{D}$ 为训练数据分布，$f_{\boldsymbol{\theta}_{\text{head}}}$ 表示从 patch 特征经五基线前向、对齐、（可选）门控与融合 MLP 到 logits 的复合映射；若仅写融合头，可将 $f$ 简记为 $g_{\boldsymbol{\theta}_{\text{fusion}}}(\boldsymbol{h}_{\text{fused}}(\cdot))$，但实现上 \(\boldsymbol{\theta}_{\text{head}}\) 包含对齐与门控参数。

### 3.11 特征对齐（LayerNorm + 投影，实现扩展）

对第 \(k\) 个基模型输出的原始 bag 特征 \(\boldsymbol{h}_k \in \mathbb{R}^{d_k}\)，在拼接或门控前先对齐到公共维度 \(D\)：

```latex
\begin{equation}
\boldsymbol{z}_k = \boldsymbol{W}_k \, \mathrm{LN}(\boldsymbol{h}_k) \in \mathbb{R}^{D}, \quad k \in \{\text{RRT}, \text{AMIL}, \text{WiKG}, \text{DSMIL}, \text{S4}\}
\end{equation}
```

其中 \(\mathrm{LN}(\cdot)\) 为最后一维上的 LayerNorm，\(\boldsymbol{W}_k \in \mathbb{R}^{D \times d_k}\) 为可学习线性投影。冻结五个基线时，\(\{\boldsymbol{W}_k, \mathrm{LN}\}\) 与后续融合模块仍参与训练，从而对预提取特征做**仿射尺度校准**。

### 3.12 门控融合（`fusion_mode=gate`，默认）

将对齐后的向量拼接为 \(\boldsymbol{c} = [\boldsymbol{z}_{\text{RRT}}; \cdots; \boldsymbol{z}_{\text{S4}}] \in \mathbb{R}^{5D}\)，经两层 MLP（实现中为 `gate_mlp`：\(5D \to 128 \to 5\)，含 ReLU 与 Dropout）得到 logits，再对五路做 softmax 得到门控权重：

```latex
\begin{equation}
\boldsymbol{w} = \mathrm{softmax}\!\left( \boldsymbol{W}^{(g)}_2 \, \mathrm{ReLU}\!\left( \boldsymbol{W}^{(g)}_1 \boldsymbol{c} + \boldsymbol{b}^{(g)}_1 \right) + \boldsymbol{b}^{(g)}_2 \right) \in \mathbb{R}^{5}
\end{equation}
```

加权融合与后续分类：

```latex
\begin{equation}
\boldsymbol{h}_{\text{fuse}} = \sum_{k} w_k \, \boldsymbol{z}_k \in \mathbb{R}^{D}, \quad \boldsymbol{z} = g_{\boldsymbol{\theta}_{\text{fusion}}}(\boldsymbol{h}_{\text{fuse}})
\end{equation}
```

门控使不同 WSI 样本可自适应地提高或抑制某一基线的贡献，减轻「某一视角在特定样本上失效却仍被拼接 MLP 强行使用」的问题。

### 3.13 拼接融合（`fusion_mode=concat`，消融）

```latex
\begin{equation}
\boldsymbol{h}_{\text{cat}} = [\boldsymbol{z}_{\text{RRT}}; \cdots; \boldsymbol{z}_{\text{S4}}] \in \mathbb{R}^{5D}, \quad \boldsymbol{z} = g_{\boldsymbol{\theta}_{\text{fusion}}}(\boldsymbol{h}_{\text{cat}})
\end{equation}
```

此时 \(g_{\boldsymbol{\theta}_{\text{fusion}}}\) 的第一层线性映射为 \(\mathbb{R}^{5D} \to \mathbb{R}^{512}\)，与门控模式下 \(\mathbb{R}^{D} \to \mathbb{R}^{512}\) 不同，二者 checkpoint **不可互换**。

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
| (23) | 特征对齐（LN+Proj） | 见 3.11 节 |
| (24) | 门控 softmax 权重 | 见 3.12 节 |
| (25) | 加权融合与拼接消融 | 见 3.12–3.13 节 |

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

### 5.3A 前端训练页：实验思路与操作详解（Training）

以下说明对应 **`vila-mil-frontend` → Training（训练）** 页面；**只有先把 Model 选成 `EnsembleFeature`** 时，才会出现集成与消融相关区块。

已发任务可在 **Dashboard → 训练任务详情** 对话框中查看 **「训练参数与消融（本次任务）」**：系统会合并 `tasks.json` 中的字段与 **启动命令** 里的参数（如 `--ensemble_fusion`、`--ensemble_exclude`、`--early_stopping` 等），便于对照实验记录。

#### （1）实验思路：三块旋钮分别控制什么

| 旋钮 | 对应论文/文档位置 | 含义 |
|------|------------------|------|
| **基线特征子集**（五个勾选 + 快捷 Chip） | 第六节 **6.1**、「用几路特征」 | 某路**取消勾选**＝该路对齐后特征恒为 0，等价于请求里的 **`ensembleExclude`**。五路全关不允许（至少留一路）。 |
| **冻结五个基模型** + **融合方式** | **6.2**（冻结/微调）+ **6.4**（gate/concat） | 勾选冻结＝只训对齐层/门控/融合头（与命令行 `--freeze_base` 一致）；取消冻结并提交时会带 **`finetuneEnsemble`**。下拉选 **门控** 或 **拼接**。 |
| **三种启动方式** | 流程组合 | **Start**：只发 1 个任务，参数=当前页全部选项。**提交所选消融**：在**当前子集**不变的前提下，按勾选批量发 4 类里的若干类（门控/拼接 × 冻结/微调）。**留一法入队（5 项）**：固定发 5 个任务，与「子集勾选」无关，每次只排除一路，用于严格对照表 6.1。 |

三者关系简述：

- **子集勾选**影响：**Start** 和 **提交所选消融**（都会把当前 `ensembleExclude` 带给后端）。
- **留一法入队**不影响子集勾选：它自己每次只排除一路，五路轮流，方便你固定其它超参写论文表。
- **融合方式、是否冻结**影响：**Start**、**留一法**、以及「提交所选消融」里每一项各自的内置组合（批量四项会覆盖你当前下拉里的 gate/concat 与冻结，见下表）。

#### （2）进入页后建议的通用设置顺序

1. **Cancer**：选与五基线 best 记录一致的癌种（如 LUSC）。  
2. **maxEpochs / learningRate / kFolds / weightDecay / 早停 / seed**：所有「单次 / 批量 / 留一」任务**共用**这一套；写论文时建议同一组超参，只改变下面消融维度。  
3. **Model** 选 **`EnsembleFeature`**。  
4. 先决定 **基线特征子集**：  
   - 要做「**全五路**」实验：五路全勾选，或点 Chip **「五路全开」**。  
   - 要做「**只用其中几路**」：按需取消勾选，或点 **「仅 RRT+WiKG+DSMIL」** 等快捷 Chip。  
   - 要做「**只关掉某一路**」：点对应 **「只关 RRTMIL」** 等五个 Chip 之一（等价于只排除该路）。  
5. 再选 **是否冻结五基线**、**融合方式**（门控 / 拼接）。  
6. 最后用 **Start**、**提交所选消融** 或 **留一法入队** 之一发任务。

#### （3）三种按钮分别点谁、勾选什么

**A. 只跑一个配置（日常调试 / 单次实验）**

- 子集、冻结、融合都按上面设好。  
- 点 **Start**。  
- 后端收到：当前 `ensembleFusion`、`finetuneEnsemble`（若未冻结）、以及非空的 `ensembleExclude`（若有路被关）。

**B. 在「当前子集」固定下，对比 6.4×6.2 的四种组合（批量）**

- 先设好 **子集**（例如五路全开）。  
- 在 **「消融实验（批量入队）」** 里勾选想要的行（默认常勾「门控+冻结」「拼接+冻结」）：  
  - 门控 + 冻结五基线  
  - 拼接 + 冻结五基线  
  - 门控 + 端到端微调  
  - 拼接 + 端到端微调  
- 点 **「提交所选消融（N 项）」**。  
- 说明：这 N 个任务**每一项**会自带对应的 `ensembleFusion` 与是否 `finetuneEnsemble`，**与你当前页上的「融合方式」下拉、冻结勾选可以不一致**（以批量表里那一行为准）；但 **子集** 与 **Cancer、超参** 与当前页一致。

**C. 严格做表 6.1「留一」五行（每次少一路）**

- 先设好 **融合方式**（门控或拼接）和 **是否冻结**（这两项会应用到全部 5 个任务）。  
- **不必**先调子集勾选（留一逻辑与「基线特征子集」区块独立）。  
- 点 **「留一法入队（5 项）」**。  
- 后端会依次发：`ensembleExclude` 分别为 `[RRTMIL]`、`[AMIL]`、`[WiKG]`、`[DSMIL]`、`[S4MIL]`（每次只关一路，其余四路仍参与）。

#### （4）与第六节表格的对应关系（方便写论文）

| 文档小节 | 在前端怎么复现 |
|---------|----------------|
| **6.1 Full** | 五路全勾选 + Start（或留一法之外任意提交）。 |
| **6.1 w/o 某基线** | 要么点 **「只关 某模型」** 再 Start；要么直接 **留一法入队** 里对应那一发任务。 |
| **6.2 Frozen** | 勾选「冻结五个基模型」+ Start；或在批量里勾选带「冻结」的两行。 |
| **6.2 Finetune** | 取消冻结 + Start；或批量里勾选带「端到端微调」的行。 |
| **6.4 gate / concat** | 用「融合方式」下拉；或批量里两种各勾一项。 |

**6.3（融合头层数）** 当前前端未暴露；若需改层数需在 `EnsembleFeature.py` 中改结构或加参数后再接 API。

#### （5）队列与排错

- 若已有任务在跑，新点 **Start / 批量 / 留一** 会 **自动入队**，在 **Training Queue** 里按顺序执行。  
- 若提示「至少保留一路基线特征」：说明五路基线勾选全关，请至少勾回一路。  
- 若 API 报 `ensembleExclude` 相关错误：检查是否传了非法键名；合法键为：`RRTMIL`、`AMIL`、`WiKG`、`DSMIL`、`S4MIL`（亦支持 `S4` 作为 S4MIL 别名，以后端为准）。

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
    --freeze_base \
    --ensemble_fusion gate
```

**参数说明**：
- `--freeze_base`：冻结五个基模型的所有参数，仅优化融合分类头
- `--lr 1e-3`：冻结模式下仅训练融合头，可使用较大学习率
- `--ensemble_fusion gate|concat`：**门控加权（默认 `gate`）** 或 **对齐后拼接（`concat`，消融）**，见本文 2.3 节与 3.11–3.13 节
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
    --finetune_ensemble \
    --ensemble_fusion gate
```

**参数说明**：
- `--finetune_ensemble`：解冻五个基线 MIL，与融合头一起训练（代码中与默认 `--freeze_base` 互斥）
- `--lr 1e-5`：微调模式需使用较小学习率，避免破坏预训练权重
- `--ensemble_fusion`：与策略 A 相同；端到端微调时仍可优先使用 `gate`
- 同样可追加 `--ensemble_ckpt_dir` 使用手动目录而非自动解析

**Web / API**：前端训练页在模型为 `EnsembleFeature` 时提供「融合方式」选择；REST 请求体可带 `ensembleFusion`（或 `ensemble_fusion`）为 `gate` / `concat`，与上述命令行含义一致。

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

每次去掉一个基模型的特征，观察集成模型性能变化。实现方式：对齐后对该路特征乘以 0（`ensemble_branch_mask`），可由 **`ensembleExclude`** / 前端「只关某路」或「留一法入队」触发；详见 **5.3A**。

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

### 6.4 门控 vs. 拼接（`ensemble_fusion`）

| 配置 | 说明 |
|------|------|
| `gate`（默认） | 样本自适应五路基线权重，融合头输入维为 \(D\) |
| `concat` | 五路对齐向量直接拼接，融合头输入维为 \(5D\)，用于与门控对照 |

建议在论文表格中同时报告两种设置（固定其它超参、相同五基线 checkpoint），以说明门控是否带来一致增益。

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
> 为此，本文提出一种特征级多模型集成框架。该框架将五个基模型视为并行特征提取器，分别从 patch 特征序列中提取 bag 级表征；在融合前对各支路输出进行归一化与维度对齐，并可采用**门控加权**或**拼接后 MLP** 两种融合方式，再经两层 MLP 融合分类头得到最终预测。记对齐后的支路特征为 $\boldsymbol{z}_k$，则门控模式下有 $\boldsymbol{h}_{\text{fuse}} = \sum_k w_k \boldsymbol{z}_k$，拼接模式下有 $\boldsymbol{h}_{\text{cat}} = [\boldsymbol{z}_{\text{RRT}}; \cdots; \boldsymbol{z}_{\text{S4}}]$（详见正文公式）。
>
> 训练阶段可采用两阶段策略：首先独立训练五个基模型至收敛，再冻结其参数并仅优化对齐层与融合模块；或在资源允许时端到端微调。冻结策略可降低显存占用，并减少不同基线之间的梯度干扰。

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

### Q6：升级对齐 / 门控后旧 checkpoint 无法加载？

对齐层（`align_*`）、门控（`gate_mlp`）与不同 `fusion_mode` 下 `fusion_head` 首层形状均可能变化。**整模** `load_state_dict(strict=True)` 会失败。处理方式：（1）使用当前代码重新训练集成；（2）若必须加载旧权重，需保证 `fusion_mode` 与保存时一致，并仅对五个子网络调用 `load_pretrained`（项目已有接口），对齐与融合头随机初始化后做 warm-start；（3）推理 API 会根据 state_dict 中是否含 `gate_mlp` 自动选择 `gate` 或 `concat`，但仍需与训练时结构一致以免部分层随机。

---

## 十、文件结构

本项目新增/修改的文件（与特征对齐 / 门控相关的要点）：

```
ViLa-MIL/
├── models/
│   └── EnsembleFeature.py           ← 集成模型：各分支 LN+Proj；fusion_mode=gate|concat；gate_mlp
├── utils/
│   ├── core_utils.py                ← 传入 fusion_mode（args.ensemble_fusion）
│   └── ensemble_ckpt_resolve.py     ← 从 best_models/tasks 自动解析五基线 checkpoint
├── main_LUSC.py                     ← --ensemble_fusion gate|concat；以及 --freeze_base、--ensemble_ckpt_dir、--finetune_ensemble 等
├── api_server.py                    ← POST /api/training/start：ensembleFusion；推理加载 Ensemble 时按权重推断 fusion_mode
└── docs/
    └── EnsembleFeature使用说明.md   ← 本文件

vila-mil-frontend/
└── src/components/Training/Training.jsx  ← EnsembleFeature：融合方式下拉框、payload.ensembleFusion
```
