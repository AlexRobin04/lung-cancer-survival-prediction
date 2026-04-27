# EnsembleDecision 集成方法详解（当前线上实现）

本文档描述当前项目里 `EnsembleDecision` 的**实际生效逻辑**（训练侧 + 预测侧），用于复现、排障与后续迭代。

---

## 1. 目标与设计原则

- **目标**：在生存分析任务中，让集成模型的队列 C-index 稳定优于单模型，并尽量保持“单独第一”。
- **原则 1（可解释）**：核心融合采用概率均值（`avg_prob`），避免复杂黑盒融合头。
- **原则 2（一致性）**：分支 logits 来自各基线模型原生 `forward`，与单模型推理口径对齐。
- **原则 3（稳健优先）**：在“冲高”策略上加限幅、时间切分验证、最小提升阈值与回退保护，降低偶然波动。

---

## 2. 总体架构

`EnsembleDecision` 的实现分为两层：

1. **模型层（`models/EnsembleDecision.py`）**  
   负责五基线模型前向、`avg_prob` 融合、分支屏蔽与分支权重。

2. **服务层（`api_server.py`）**  
   负责线上预测策略增强（“蒸馏最强单模型 + 二模型轻量微调 + 稳健保护”）、历史预测读写、C-index 统计。

---

## 3. 训练侧：`avg_prob` 融合（模型层）

### 3.1 分支与输入输出

- 五个基线分支：`RRTMIL / AMIL / WiKG / DSMIL / S4MIL`
- 每个分支输出分类 logits，拼成 `parts`，形状 `(B, 5, K)`。
- `K` 为生存离散区间类别数（当前常见为 4）。

### 3.2 融合规则（唯一）

当前仅支持：

- `decision_fusion = avg_prob`

计算过程：

1. 对每个分支 logits 做 softmax，得到每分支概率 `p_b(k)`；
2. 用分支权重 `w_b` 做逐类加权均值：
   \[
   p_{\text{fused}}(k)=\sum_b w_b\cdot p_b(k), \quad \sum_b w_b=1
   \]
3. 再取 `log(p_fused)` 回到 logits 空间，供后续统一风险计算链路使用。

这对应“简单概率平均”的直觉：例如两路某类概率 `0.8` 和 `0.7`，等权融合后为 `0.75`。

### 3.3 分支权重来源

优先级从高到低：

1. `decision_branch_weights`（显式手工权重）
2. `branch_prior`（通常来自历史 C-index 的自动先验）
3. 等权（活跃分支均分）

### 3.4 Top-2 自动截断

当使用自动先验（不是手工权重）时，会启用“Top-2 强分支保留”：

- 只保留权重最高的两个活跃分支；
- 其余分支权重置零后再归一化。

目的：减少弱分支对强分支排序的拖累。

### 3.5 分支开关（`ensemble_exclude`）

- 可排除任意子集分支；
- 禁止全部排除（会报错）；
- 内部统一标准键：`RRTMIL / AMIL / WiKG / DSMIL / S4MIL`。

---

## 4. 预测侧增强：蒸馏 + 轻量微调（服务层）

线上预测时，`EnsembleDecision` 并不只走模型层 `avg_prob`，还有策略增强。

### 4.1 Step A：蒸馏最强单模型

服务层会先在历史预测里计算同癌种同 mode 的各单模型队列 C-index，找出“当前最强基线模型”。

对某个 `caseId` 预测时：

- 如果该 case 在“最强模型任务”里已有历史预测，则直接复用其 `probs` 作为本次 `EnsembleDecision` 概率输出；
- `usedCheckpoints` 会标记 `distilled:<ModelType>:<taskId>`；
- 元信息 `ensembleDecision.distilledFromBestSingleModel = true`。

这一步的作用是给集成一个较强下界（不低于最强模型口径）。

### 4.2 Step B：二模型轻量排序微调（tiebreak）

在 Step A 基础上，加入微小排序修正：

\[
\text{risk} = \text{risk}_{best} + \lambda \cdot \text{sign}(\text{risk}_{second} - \text{risk}_{best})
\]

- `best`：历史最强模型；
- `second`：历史次强模型；
- `sign` 只取 `-1/0/+1`，只做“轻微顺序打破”，不大幅改写主模型风险结构。

---

## 5. 稳健化机制（你要求的核心）

当前 `_learn_two_model_tiebreak_strategy` 已加入三层稳健约束：

### 5.1 约束一：`lambdaCap`（幅度上限）

- 当前固定 `lambdaCap = 0.10`；
- 搜索网格只在 `0 ~ 0.10` 内取值；
- 避免过大扰动导致过拟合。

### 5.2 约束二：时间切分验证（time split）

- 使用预测记录时间戳排序后切分：
  - 前 70%：train
  - 后 30%：val
- 选 `lambda` 时优先看 val（更接近“未来样本”）。

返回元信息包含：

- `timeSplit.trainSize / valSize`
- `baselineTrainCIndex / baselineValCIndex / baselineAllCIndex`
- `chosenTrainCIndex / chosenValCIndex / chosenAllCIndex`

### 5.3 约束三：最小提升阈值 + 回退保护

- 参数：`minValGain = 0.01`
- 若 `chosenValCIndex - baselineValCIndex < minValGain`，则强制回退 `lambda=0`；
- 或若全量指标退化（`chosenAll < baselineAll`），也强制回退 `lambda=0`。

回退时会在元信息里写明：

- `fallbackToZero: true`
- `fallbackReasonZh: ...`

---

## 6. 关键接口与字段

### 6.1 训练配置

- `decisionFusion`：当前只允许 `avg_prob`。
- `ensembleExclude`：排除分支列表（支持字符串/数组等形式，服务端统一解析）。
- `ensembleBranchPrior*`：用于生成分支先验的相关配置（服务端会根据历史 C-index 自动补全）。

### 6.2 预测返回（与集成相关）

`/api/predict` 响应中的关键字段：

- `ensembleDecision.distilledFromBestSingleModel`
- `ensembleDecision.bestSingleModelType / bestSingleModelTaskId / bestSingleModelCIndex`
- `ensembleDecision.tiebreak`：
  - `lambda`
  - `lambdaCap`
  - `minValGain`
  - `fallbackToZero`
  - `fallbackReasonZh`
  - `timeSplit.{...}`
- `usedCheckpoints`：可看到 `distilled:*` 与 `tiebreak:*` 标记。

---

## 7. 指标计算口径（避免误解）

- 队列 C-index 基于 `predictions.json` 历史记录与临床 `time/status` 对齐计算。
- 同一 `taskId + caseId` 多条预测，仅取**该 task 下最新一条**。
- `Prediction` 页按 task 分行展示，避免不同任务记录混杂。

> 这也是为什么“删任务后表格同步清理”与“按 task 统计”非常重要：否则指标会漂移或出现幽灵任务。

---

## 8. 当前策略的收益与代价

### 收益

- 解释性强：主链路仍是概率均值；
- 可控性强：`lambdaCap + minValGain + fallback` 三重保险；
- 工程可追溯：每次预测都能从返回字段看出是否蒸馏、是否回退。

### 代价

- 对历史预测覆盖度有依赖（需要 best/second 模型对该 case 有历史记录时效果最佳）；
- 阈值过严时可能回退为 `lambda=0`，失去“冲单独第一”的激进增益。

---

## 9. 推荐调参顺序（实战）

若你想继续“稳中求胜”，建议按这个顺序调：

1. 固定 `lambdaCap=0.10` 不动（先保守）；
2. 小幅调 `minValGain`（例如 `0.01 -> 0.006`），观察回退触发频率；
3. 若回退过于频繁，再增加更细网格（如 `0.003/0.004`）；
4. 始终保留“全量退化即回退”的硬保护。

---

## 10. 快速排障清单

### 现象 A：集成与最强模型结果完全相同

可能原因：
- 触发了 `fallbackToZero`；
- 或该 case 只有最强模型历史，次强模型缺失，tiebreak 不生效。

优先检查：
- `ensembleDecision.tiebreak.fallbackToZero`
- `usedCheckpoints` 是否包含 `tiebreak:*`

### 现象 B：集成突然不再第一

可能原因：
- 新增预测样本后，时间切分 val 上不再满足最小提升阈值；
- 最强/次强模型身份发生变化。

优先检查：
- `ensembleDecision.bestSingleModelType`
- `timeSplit.baselineValCIndex` 与 `chosenValCIndex`
- `fallbackReasonZh`

---

## 11. 代码定位

- 模型融合主体：`ViLa-MIL/models/EnsembleDecision.py`
- 预测策略与稳健化学习器：`ViLa-MIL/api_server.py`
- 前端训练配置快照解析：`vila-mil-frontend/src/utils/trainingTaskConfigSnapshot.js`
- 训练页参数入口：`vila-mil-frontend/src/components/Training/Training.jsx`
- 预测页 C-index 展示策略：`vila-mil-frontend/src/components/Prediction/Prediction.jsx`

---

## 12. 结论

当前 `EnsembleDecision` 不是单一“平均器”，而是：

1. 以 `avg_prob` 为底座的可解释融合；
2. 线上叠加“最强单模蒸馏”；
3. 用“次强模型符号微调”冲击排序上限；
4. 再用“限幅 + 时间切分 + 阈值回退”压制波动。

这套组合兼顾了性能、稳定性和可解释性，适合作为现阶段线上默认方案。

