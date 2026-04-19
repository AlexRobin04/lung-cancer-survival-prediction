# 特征级多模型集成 MIL（EnsembleFeatureMIL）— 数学公式预览版

> **本文定位**：与 **[`EnsembleFeature使用说明.md`](./EnsembleFeature使用说明.md)** 内容一致，但第三节起**不用** `` ```latex ... ``` `` 代码块里的 `\begin{equation}`，而改用 **Markdown 原生数学环境**（`$...$` 行内、`$$...$$` 独立显示）。在 **Cursor / VS Code / GitHub** 等支持 MathJax 的预览中，公式会**渲染为可读数学式**，而不是整段「人话解释公式」。
>
> **仍需要整段 `\begin{equation}` 复制到 Word / LaTeX 时**：请用 **原版** 文档第三节的代码块。

---

## 一、方法概述

本文提出一种**特征级多实例学习集成框架**（Feature-Level Ensemble MIL）。该框架以五种 MIL 基模型——RRTMIL、AMIL、WiKG、DSMIL 和 S4MIL——为并行特征提取器，从同一 WSI 的 patch 特征序列中提取 bag 级表征。

**当前实现**：融合前对各路 bag 特征做 **LayerNorm + 线性投影** 到统一维度 $D$（默认 `feature_align_dim=512`）。融合支持 **`fusion_mode=gate`（默认）**：五路对齐特征经 MLP 得 $\boldsymbol{w}\in\mathbb{R}^5$，$\mathrm{softmax}$ 后对 $\boldsymbol{z}_k$ 加权求和；**`fusion_mode=concat`**：拼接为 $\mathbb{R}^{5D}$ 再经融合 MLP。

---

## 二、网络结构与维度

**ASCII 结构图、接口表、兼容性说明**与原版完全相同，请直接打开 **[`EnsembleFeature使用说明.md`](./EnsembleFeature使用说明.md)** 第二节。

五种基线**原始** bag 维记为 $d_{\text{RRT}}, d_{\text{AMIL}}, \ldots$；**对齐后** $\boldsymbol{z}_{\cdot}\in\mathbb{R}^{D}$。**concat** 时融合头首层输入维为 $5D$；**gate** 时为 $D$。

---

## 三、数学公式（预览渲染）

以下每式右侧编号与原版「第四节公式汇总表」一致（其中门控加权与拼接消融在原表中合并为第 25 条，此处拆成 **(25)(26)** 便于阅读）。

### 3.1 问题定义：多实例学习包表示

给定 patch 特征序列：

$$
X = \{ \boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_N \}, \quad \boldsymbol{x}_i \in \mathbb{R}^{d}
\tag{1}
$$

其中 $N$ 为 patch 数，$d$ 为 patch 特征维，$X \in \mathbb{R}^{N \times d}$。

### 3.2 RRTMIL 特征提取

$$
\boldsymbol{h}_{\text{RRT}} = \text{DAttention}\!\left( \text{RRTEncoder}\!\left( \text{Proj}(X) \right) \right) \in \mathbb{R}^{d_{\text{RRT}}}
\tag{2}
$$

### 3.3 AMIL 特征提取

$$
\boldsymbol{h}_i = \text{ReLU}\!\left( \boldsymbol{W}_p \boldsymbol{x}_i + \boldsymbol{b}_p \right), \quad \boldsymbol{h}_i \in \mathbb{R}^{256}
\tag{3}
$$

$$
a_i = \frac{\exp\!\left( \boldsymbol{w}^\top \Big( \tanh(\boldsymbol{V} \boldsymbol{h}_i) \odot \sigma(\boldsymbol{U} \boldsymbol{h}_i) \Big) \right)}{\sum\limits_{j=1}^{N} \exp\!\left( \boldsymbol{w}^\top \Big( \tanh(\boldsymbol{V} \boldsymbol{h}_j) \odot \sigma(\boldsymbol{U} \boldsymbol{h}_j) \Big) \right)}
\tag{4}
$$

$$
\boldsymbol{h}_{\text{AMIL}} = \sum_{i=1}^{N} a_i \boldsymbol{h}_i \in \mathbb{R}^{256}
\tag{5}
$$

### 3.4 WiKG 特征提取

$$
\text{Attn}_{ij} = \frac{(\boldsymbol{W}_{\text{head}} \boldsymbol{x}_i)^\top (\boldsymbol{W}_{\text{tail}} \boldsymbol{x}_j)}{\sqrt{d_h}}
\tag{6}
$$

$$
p_{ij} = \frac{\exp(\text{Attn}_{ij})}{\sum\limits_{k \in \text{TopK}(i)} \exp(\text{Attn}_{ik})}
\tag{7}
$$

$$
\boldsymbol{e}_i^{\text{agg}} = \sum_{j \in \text{TopK}(i)} p_{ij} \cdot \boldsymbol{g}_{ij} \odot \boldsymbol{e}_j
\tag{8}
$$

其中 $\boldsymbol{g}_{ij} = \tanh(\boldsymbol{e}_i + \boldsymbol{e}_j)$。

$$
\boldsymbol{h}_{\text{WiKG}} = \text{Readout}\!\left( \text{ReLU}(\boldsymbol{W}_1(\boldsymbol{e}_i + \boldsymbol{e}_i^{\text{agg}})) + \text{ReLU}(\boldsymbol{W}_2(\boldsymbol{e}_i \odot \boldsymbol{e}_i^{\text{agg}})) \right) \in \mathbb{R}^{512}
\tag{9}
$$

### 3.5 DSMIL 特征提取

$$
\boldsymbol{c}_i = \text{IClassifier}(\boldsymbol{x}_i) \in \mathbb{R}^{C}
\tag{10}
$$

$$
A_i = \text{softmax}\!\left( \frac{\boldsymbol{q}(\boldsymbol{x}_i)^\top \boldsymbol{q}(\boldsymbol{x}_m)}{\sqrt{d_q}} \right)
\tag{11}
$$

$$
\boldsymbol{h}_{\text{DSMIL}} = \text{Conv1D}\!\left( \sum_{i=1}^{N} A_i \cdot \boldsymbol{v}(\boldsymbol{x}_i) \right) \in \mathbb{R}^{C}
\tag{12}
$$

### 3.6 S4MIL 特征提取

$$
\boldsymbol{y} = \mathcal{F}^{-1}\!\left( \mathcal{F}(\boldsymbol{K}) \odot \mathcal{F}(\boldsymbol{X}) \right) + \boldsymbol{D} \cdot \boldsymbol{X}
\tag{13}
$$

$$
\boldsymbol{h}_{\text{S4}} = \max_{i=1}^{N} \left( \text{GELU}\!\left( \text{Linear}(\boldsymbol{y}_i) \right) \right) \in \mathbb{R}^{512}
\tag{14}
$$

### 3.7 特征拼接（理想化「裸拼接」记号）

$$
\boldsymbol{h}_{\text{fused}} = \left[ \boldsymbol{h}_{\text{RRT}}; \; \boldsymbol{h}_{\text{AMIL}}; \; \boldsymbol{h}_{\text{WiKG}}; \; \boldsymbol{h}_{\text{DSMIL}}; \; \boldsymbol{h}_{\text{S4}} \right] \in \mathbb{R}^{d_{\text{total}}}
\tag{15}
$$

典型维数下 $d_{\text{total}} = 256+256+512+512+512 = 2048$（与实现中对齐前讨论一致；当前代码在融合前见原版 **3.11–3.13**）。总维度亦可写为：

$$
d_{\text{total}} = d_{\text{RRT}} + d_{\text{AMIL}} + d_{\text{WiKG}} + d_{\text{DSMIL}} + d_{\text{S4}}
\tag{22}
$$

### 3.8 融合分类头

以下 $\boldsymbol{h}_{\text{fused}}$ 表示融合头输入占位（可为裸拼接向量、门控输出 $\boldsymbol{h}_{\text{fuse}}$ 或拼接对齐后的 $\boldsymbol{h}_{\text{cat}}$），$\boldsymbol{W}^{(1)}$ 的列维随实现为 $d_{\text{total}}$、$D$ 或 $5D$。

$$
\boldsymbol{h}^{(1)} = \text{ReLU}\!\left( \boldsymbol{W}^{(1)} \boldsymbol{h}_{\text{fused}} + \boldsymbol{b}^{(1)} \right), \quad \boldsymbol{W}^{(1)} \in \mathbb{R}^{512 \times d_{\text{total}}}, \; \boldsymbol{b}^{(1)} \in \mathbb{R}^{512}
\tag{16}
$$

$$
\boldsymbol{h}^{(2)} = \text{ReLU}\!\left( \boldsymbol{W}^{(2)} \boldsymbol{h}^{(1)} + \boldsymbol{b}^{(2)} \right), \quad \boldsymbol{W}^{(2)} \in \mathbb{R}^{256 \times 512}, \; \boldsymbol{b}^{(2)} \in \mathbb{R}^{256}
\tag{17}
$$

$$
\boldsymbol{z} = \boldsymbol{W}^{(3)} \boldsymbol{h}^{(2)} + \boldsymbol{b}^{(3)}, \quad \boldsymbol{W}^{(3)} \in \mathbb{R}^{C \times 256}, \; \boldsymbol{b}^{(3)} \in \mathbb{R}^{C}
\tag{18}
$$

$$
P_c = \frac{\exp(z_c)}{\sum\limits_{k=1}^{C} \exp(z_k)}, \quad c = 1, 2, \ldots, C
\tag{19}
$$

### 3.9 损失函数

$$
\mathcal{L}_{\text{CE}} = - \sum_{c=1}^{C} \mathbb{I}(y = c) \cdot \log P_c
\tag{20}
$$

### 3.10 优化目标（冻结基模型模式）

$$
\boldsymbol{\theta}_{\text{head}}^{\ast} = \arg\min_{\boldsymbol{\theta}_{\text{head}}} \; \mathbb{E}_{(X, y) \sim \mathcal{D}} \left[ \mathcal{L}_{\text{CE}}\!\left( f_{\boldsymbol{\theta}_{\text{head}}}(X), y \right) \right]
\tag{21}
$$

### 3.11 特征对齐（LayerNorm + 投影，与原版 3.11 一致）

$$
\boldsymbol{z}_k = \boldsymbol{W}_k \, \mathrm{LN}(\boldsymbol{h}_k) \in \mathbb{R}^{D}, \quad k \in \{\text{RRT}, \text{AMIL}, \text{WiKG}, \text{DSMIL}, \text{S4}\}
\tag{23}
$$

### 3.12 门控融合（`fusion_mode=gate`，与原版 3.12 一致）

记 $\boldsymbol{c} = [\boldsymbol{z}_{\text{RRT}}; \cdots; \boldsymbol{z}_{\text{S4}}] \in \mathbb{R}^{5D}$。

$$
\boldsymbol{w} = \mathrm{softmax}\!\left( \boldsymbol{W}^{(g)}_2 \, \mathrm{ReLU}\!\left( \boldsymbol{W}^{(g)}_1 \boldsymbol{c} + \boldsymbol{b}^{(g)}_1 \right) + \boldsymbol{b}^{(g)}_2 \right) \in \mathbb{R}^{5}
\tag{24}
$$

$$
\boldsymbol{h}_{\text{fuse}} = \sum_{k} w_k \, \boldsymbol{z}_k \in \mathbb{R}^{D}, \quad \boldsymbol{z} = g_{\boldsymbol{\theta}_{\text{fusion}}}(\boldsymbol{h}_{\text{fuse}})
\tag{25}
$$

### 3.13 拼接融合（`fusion_mode=concat`，与原版 3.13 一致）

$$
\boldsymbol{h}_{\text{cat}} = [\boldsymbol{z}_{\text{RRT}}; \cdots; \boldsymbol{z}_{\text{S4}}] \in \mathbb{R}^{5D}, \quad \boldsymbol{z} = g_{\boldsymbol{\theta}_{\text{fusion}}}(\boldsymbol{h}_{\text{cat}})
\tag{26}
$$

> **编号说明**：原版第四节汇总表第 25 条对应「3.12–3.13 门控加权与拼接消融」；本文件将 **(25)** 固定为门控加权输出、**(26)** 为拼接分支，便于在预览中区分两式。

---

## 四、公式汇总表（简）

| 编号 | 名称 | 本节 |
|------|------|------|
| (1)–(21) | 与原版表一致 | §3.1–§3.10 |
| (22) | 总维度 | §3.7（紧随式 (15)） |
| (23) | 对齐 LN+Proj | §3.11 |
| (24)(25) | 门控 softmax 与加权融合 | §3.12 |
| (26) | 拼接融合（原版表并入「25」行时与此对应） | §3.13 |

---

## 五、训练命令、消融、FAQ、文件结构

**第五节起**（训练步骤、前端说明、消融表、论文素材、常见问题、文件树）与原版逐字一致，请阅读 **[`EnsembleFeature使用说明.md`](./EnsembleFeature使用说明.md)** 第五节至第十节。

---

## 预览不显示公式时

1. **Cursor / VS Code**：确认 Markdown 预览支持数学（较新版本默认开启）；可尝试换用内置预览或扩展。  
2. **纯文本编辑器**：仍会看到 `$`、`$$` 与反斜杠命令，此时请用 **原版** `\begin{equation}` 块复制到 Word，或安装支持 MathJax 的预览。
