# 基于病理图像的肺癌生存风险预测系统

> 详细技术文档（与当前代码实现对齐）
>
> 适用代码基线：
> - 后端：`ViLa-MIL/api_server.py`、`ViLa-MIL/main_LUSC.py`、`ViLa-MIL/utils/core_utils.py`
> - 前端：`vila-mil-frontend/src/components/*`
> - 部署：`docker-compose.yml`（单端口网关）

---

## 1. 系统架构

### 1.1 总体架构

- 入口：`gateway`（Nginx，宿主机端口 `80`）
- 前端：`frontend`（React + Vite 构建产物）
- 后端：`backend`（Flask + Gunicorn，API 前缀 `/api`）
- 训练执行：后端通过子进程启动 `main_LUSC.py`
- 持久化：
  - 训练结果：`ViLa-MIL/result/api_runs/<taskId>/`
  - 训练日志：`ViLa-MIL/api_training_logs/<taskId>.log`
  - 任务元数据：`ViLa-MIL/uploaded_features/tasks.json`
  - 数据清单：`ViLa-MIL/uploaded_features/manifest.json`
  - 病例与特征关联：`ViLa-MIL/uploaded_features/cases.json`
  - 预测历史：`ViLa-MIL/uploaded_features/predictions.json`

### 1.2 请求链路

1. 浏览器访问 `http://<host>/`
2. `gateway` 转发：
   - `/` -> `frontend:80`
   - `/api/` -> `backend:8000/api/`
3. 前端统一以 `API BaseURL`（默认 `/api`）调用后端接口

### 1.3 容器与健康策略（当前已启用）

`docker-compose.yml` 中已启用：

- `restart: unless-stopped`
- 健康检查（backend/frontend/gateway）
- 日志轮转（`json-file`, `50m x 3`）
- backend 资源上限：
  - `mem_limit: 6g`
  - `cpus: 3.0`

### 1.4 数据存储模式

平台**不依赖独立数据库服务**（如 PostgreSQL），业务状态与元数据以 **JSON 文件** 落在磁盘；大体积数据以 **目录 + 文件**（主要为 `.h5` 与上传的病理图像等）存储。Docker 部署时通过 `docker-compose.yml` 将宿主机目录挂载到容器内路径，**重建容器不丢数据**（只要宿主机目录保留）。

后端容器内工作目录一般为 `/app`（对应仓库 `ViLa-MIL/`）。当前 `backend` 卷映射为：

| 宿主机（项目根下） | 容器内 |
|-------------------|--------|
| `./ViLa-MIL/result` | `/app/result` |
| `./ViLa-MIL/api_training_logs` | `/app/api_training_logs` |
| `./ViLa-MIL/uploaded_features` | `/app/uploaded_features` |
| `./ViLa-MIL/features` | `/app/features` |
| `./ViLa-MIL/ckpt` | `/app/ckpt` |

#### 1.4.1 元数据与状态（JSON）

| 用途 | 容器内路径（相对 `ViLa-MIL/`） | 说明 |
|------|-------------------------------|------|
| 训练任务 | `uploaded_features/tasks.json` | 任务列表、`status`、`resultsDir`、`command`、指标摘要等 |
| 上传清单 | `uploaded_features/manifest.json` | 上传的特征条目、`id`、`cancer`、`featureType`、`storedPath` |
| 病例与特征关联 | `uploaded_features/cases.json` | `caseId`、`time`、`status`、可选登记 20×/10× 对应的 `fileId`（便于随访与批量推理） |
| 预测历史 | `uploaded_features/predictions.json` | 预测记录列表 |

写入方式：后端通过 `_atomic_write_json()` 原子写 JSON，降低并发写坏文件风险。

#### 1.4.2 特征与切片（大文件）

| 类型 | 典型路径 | 说明 |
|------|----------|------|
| 训练默认特征 | `features/20/*.h5`、`features/10/*.h5` | API 启动训练时 `--data_root_dir` 指向仓库根，子目录为 `features/20` 与 `features/10` |
| 数据管理上传区 | `uploaded_features/<癌种>/features_20`、`features_10` | 与上表独立；推理可直接用 `manifest` 中的 `fileId`（见 `/api/predict`），或在 Clinical 为病例登记文件 ID 后按 `caseId` 推理 |
| 位图转 TIFF（内部登记） | `uploaded_features/<癌种>/processed_wsi`（及 manifest，`kind=processed_wsi`） | 由 `upload-raster` 等写入；**在线推理仍依赖双尺度 H5** |

#### 1.4.3 训练产物与日志

| 类型 | 路径 | 说明 |
|------|------|------|
| 单次 API 训练结果 | `result/api_runs/<taskId>/` | 含 `s_<fold>_checkpoint.pt`、`splits_*.csv` 等 |
| API 侧训练日志 | `api_training_logs/<taskId>.log` | 子进程标准输出重定向 |

#### 1.4.4 备份与迁移建议

- **必备份**：`uploaded_features/`（含全部 JSON）、`result/api_runs/`、`api_training_logs/`、`features/`（若训练数据仅此一份）。
- **可选**：`ckpt/`（预训练权重）；重建镜像后仍可从宿主机卷恢复。

---

## 2. 模块实现说明

## 2.1 后端核心（`api_server.py`）

- Flask 应用工厂：`create_app()`
- 任务管理：
  - `_load_tasks()` / `_save_tasks()` / `_update_task()` / `_append_task()`
  - 通过 `tasks.json` 管理训练任务状态
- 训练守护：
  - 子进程退出线程：汇总日志指标并更新任务状态
  - 空闲超时线程：日志长期无增长自动停止训练（防假死）
- 模型推理缓存：`_MODEL_CACHE`
- 预测流程（二选一）：
  - **按病例**：`caseId` → `cases.json` 读取已登记的 `feature20FileId` / `feature10FileId` → `manifest.json` 解析路径（支持按文件名配对补全另一尺度）
  - **按文件**：请求体直接传 `feature20FileId` + `feature10FileId`（可选 `cancer` 辅助解析仅落盘未登记 manifest 的文件；可选 `caseId` 关联随访）
  - 从 `resultsDir` 发现并加载 `s_<fold>_checkpoint.pt`，多折概率平均

## 2.2 训练入口（`main_LUSC.py`）

- 统一训练入口，支持多模型参数 `--model_type`
- 当前数据任务：
  - `task_tcga_lusc_subrisk`
  - 4 分类：`low / Moderate / Elevated / high`
- 多折参数：`--k`（默认可传 1 或 4）
- 输出：
  - `splits_*.csv`
  - `s_<fold>_checkpoint.pt`
  - 训练日志（含 loss/auc/f1 等）

## 2.3 训练核心（`utils/core_utils.py`）

- 包含 `train_loop`、`validate`、`summary`
- 支持多模型分支与统一训练调用
- 对多模型输出对齐（`logits, Y_prob, loss`）
- 精度/指标日志：
  - `Accuracy_Logger`
  - `F1` / `ROC AUC` / `error`（不同任务日志字段可能不同；前端评估页已支持 Loss/AUC/F1/error 多图展示）

## 2.4 前端核心页面

- `Training`：启动任务、状态轮询、日志查看
- `DataManagement`：特征上传/删除与统计
- `Clinical`：CSV 导入、为病例指定 20×/10× 特征；提供 `已有 H5 文件` 与 `从 WSI 生成` 两种入口，其中 WSI 生成包含 `快速预览`（低采样近似）与 `正式预测`（TRIDENT 全量）
- `ModelEvaluation`：训练/验证 Loss 曲线、LUSC 示例 KM 曲线（Ours vs Others，baseline 可切换）
- `Prediction`：按病例或按特征文件 + 任务预测、风险分层与可视化；`/api/predict/from-raster` 对 WSI 已支持快速近似与 TRIDENT 全量双路径
- `Settings`：接口配置、平台介绍、推荐流程

---

## 3. 模型支持矩阵（当前实现）

## 3.1 `/api/models` 返回字段

每个模型返回：

- `id`, `name`
- `implemented`：是否原生实现
- `mode`：`native` 或 `fallback`
- `fallbackTarget`：回退目标（仅 fallback 模型）

## 3.2 当前矩阵

| 模型 | 训练接口支持 | 实现模式 | 说明 |
|---|---:|---|---|
| `ViLa_MIL` | 是 | native | 启动前有资源预检（含 CUDA） |
| `RRTMIL` | 是 | native | 已做 CPU 兼容改造 |
| `AMIL` | 是 | native | 已接入统一训练流程 |
| `WiKG` | 是 | native | 已接入统一训练流程 |
| `DSMIL` | 是 | native | 已接入统一训练流程 |
| `S4MIL` | 是 | native | 已接入统一训练流程 |
| `surformer` | 是 | native | 对应 `HVTSurv` 分支 |
| `TransMIL` | 是 | fallback | 当前回退到 `MIL_fc/MIL_fc_mc` |
| `PatchGCN` | 是 | fallback | 当前回退到 `MIL_fc/MIL_fc_mc` |

> 说明：`TransMIL/PatchGCN` 当前仓库缺少独立模型定义文件，训练可跑通但为兼容回退实现。

---

## 4. 接口逐条说明（按模块）

以下接口均为 `api_server.py` 当前实现。

## 4.1 基础接口

- `GET /api/health`
  - 返回：`{ ok: true, service: "vila-mil-api" }`

- `GET /api/config`
  - 返回运行路径、特征目录、日志目录、时区等只读信息

## 4.2 训练接口

- `POST /api/training/start`
  - 常用参数：
    - `cancer`
    - `modelType`
    - `maxEpochs`
    - `learningRate`
    - `kFolds`
    - `mode`
    - `repeat`：重复训练次数（后端串行调度，多次训练会自动递增 seed）
    - `seed`：baseSeed（repeat>1 时按 `seed + i` 递增）
  - 关键机制：
    - 模型白名单校验
    - 单任务并发（已有运行任务时返回 `409`）
    - `ViLa_MIL` 资源预检（内存/Swap/CUDA）
    - 训练空闲超时（日志无增长自动终止）

- `POST /api/training/stop`
  - 参数：`taskId`
  - 行为：停止进程并更新任务状态

- `GET /api/training/status/<taskId>`
  - 返回当前任务状态、进度、epoch、fold、指标摘要
  - 会根据日志和 PID 进行状态收敛（防“僵尸 running”）

- `GET /api/training/history`
  - 返回任务列表（UTC+8 时间）
  - 关键字段：
    - `hasCheckpoint`
    - `checkpointCount`
    - `isBestForModel`：是否为该（cancer+modelType+mode）组合的最佳 run（按最小 valLoss）
  - 用于前端筛选“可预测任务”

- `GET /api/training/log/<taskId>?tail=200`
  - 返回日志尾部内容

- `GET /api/training/best?cancer=...&modelType=...&mode=...`
  - 返回该组合当前标记的最佳训练任务（bestTaskId）及 `bestValLoss`

- `POST /api/training/history/delete`
  - 支持一键删除全部或按 taskIds 选择性删除
  - 可选 `deleteArtifacts=true` 同步删除日志与结果目录（不会删除运行中的任务）

## 4.3 数据管理接口

- `POST /api/data/upload`
  - 上传特征 H5（20x/10x）到 `uploaded_features/<cancer>/features_20|10`

- `POST /api/data/upload-raster`
  - 支持位图与 WSI 上传；位图会转为单层 TIFF 登记，WSI（`.svs/.ndpi/.mrxs/.scn`）直接入库登记为 `processed_wsi`；**不单独完成推理**，在线推理见 `/api/predict/from-raster`

- `GET /api/data/datasets`
  - 返回：
    - `datasets`
    - `summary`
    - `totalFiles`
    - `cancers`（多来源聚合）
  - 癌种来源聚合包括：上传目录、manifest、`main_*.py`、`splits/TCGA_*`、features 命名等

- `GET /api/data/features/<cancer>?featureType=20|10`
  - 返回该癌种下特征文件列表

- `DELETE /api/data/feature/<fid>`
  - 删除特征与清单记录

## 4.4 评估接口

- `GET /api/evaluation/runs`
- `GET /api/evaluation/curves/<taskId>`
- `POST /api/evaluation/km`
  - 支持 Kaplan-Meier 与 log-rank

## 4.5 模型接口

- `GET /api/models`
  - 返回模型列表 + `implemented/mode/fallbackTarget`

## 4.6 临床接口

- `POST /api/clinical/upload`
  - 导入随访 CSV（`case_id/time/status` 必需）
- `GET /api/clinical/cases`
- `GET /api/clinical/cases/<caseId>`
- `POST /api/clinical/cases/link-feature`
  - 将 `manifest` 中某特征文件的 ID 登记到指定 `caseId`（便于随访与按病例推理；非唯一路径，见 4.7）
- `POST /api/clinical/cases/associate-features`（推荐）
  - **一次性**为病例关联双尺度推理特征：JSON 传 `feature20FileId` + `feature10FileId`；或 `multipart` 传 `caseId`、`cancer`、`file`（WSI/图像）在线生成 H5 并写入 `cases.json`
  - `multipart` 关键参数：
    - `extractor=trident|raster`
    - `quick=true|false`（当上传 WSI 且 `quick=true` 时走低采样近似真快速模式；`formal` 走 TRIDENT 全量）
  - `featureSource` 可能值：`h5_pair` / `raster_derived` / `trident_derived`

## 4.7 预测接口

- `GET /api/predictions?limit=50`
- `POST /api/predict`
  - **输入（二选一）**
    - **按病例**：`caseId`（必填）+ `taskId`（可选，缺省时取最近已完成任务）；病例须在 `cases.json` 中已指定 20×/10× 特征文件 ID
    - **按文件**：`feature20FileId` + `feature10FileId`（必填）+ `taskId`；可选 `cancer` / `cancerType`；可选 `caseId`（若存在则返回 `clinicalFollowUp`）
  - 输出：
    - `caseId`（按文件且无病例时可能为 `files:<前缀>` 占位）
    - `inputSummary.featureSource`：`caseRecord` 或 `manifestFileIds`
    - `riskScore`、`riskStratification`（高/中/低）、`visualization.probabilityBar`、`clinicalFollowUp` 等
  - 若 `resultsDir` 无 checkpoint，返回：
    - `未找到 checkpoint（期望 resultsDir 下存在 s_<fold>_checkpoint.pt）`

- `POST /api/predict/from-raster`
  - `multipart/form-data`：`file`（WSI/图像）、`taskId`（可选）、`cancer`、`caseId`（可选）、`extractor`、`quick`
  - 模式说明：
    - 快速：WSI 走“缩略 + 低采样近似”路径（速度优先）
    - 正式：走 TRIDENT 全量特征提取（结果优先）
  - 之后与 `/api/predict` 共享同一 MIL 推理链路

- `POST /api/predict/batch`
  - 批量调用预测（每项沿用上述单条规则）

---

## 5. 稳定性机制（已上线）

## 5.1 单任务并发保护

- 位置：`/api/training/start`
- 策略：若已有运行任务，则拒绝新任务（`409`）
- 目的：避免 4C8G 机器被并发训练压垮

## 5.2 ViLa_MIL 资源预检

- 检查：
  - `MemAvailable >= VILAMIL_MIN_AVAIL_GB`
  - `SwapTotal >= VILAMIL_MIN_SWAP_GB`
  - CUDA 可用且有 GPU
- 默认阈值（当前 compose）：
  - `VILAMIL_MIN_AVAIL_GB=5.5`
  - `VILAMIL_MIN_SWAP_GB=2`

## 5.3 训练空闲超时自动停止

- 环境变量：
  - `TRAIN_IDLE_TIMEOUT_MIN`（默认 20）
  - `TRAIN_IDLE_CHECK_SEC`（默认 30）
- 机制：日志在超时窗口内无增长 -> 终止进程 -> 标记失败

## 5.4 健康检查与日志轮转

- 三服务均启用 `healthcheck`
- Docker 日志轮转防磁盘爆满

## 5.5 预测任务过滤

- 后端 history 返回 `hasCheckpoint`
- 前端 Prediction 仅显示 `completed && hasCheckpoint=true` 任务

---

## 6. 数据与目录设计

## 6.1 训练目录（默认）

- `ViLa-MIL/features/20`
- `ViLa-MIL/features/10`

## 6.2 上传目录（管理区）

- `ViLa-MIL/uploaded_features/<cancer>/features_20`
- `ViLa-MIL/uploaded_features/<cancer>/features_10`
- `ViLa-MIL/uploaded_features/<cancer>/processed_wsi`

## 6.3 元数据文件

- `tasks.json`：训练任务与状态
- `manifest.json`：上传文件清单
- `cases.json`：病例与特征关联（可选）
- `predictions.json`：预测历史

---

## 7. 典型业务流程（端到端）

1. **导入临床数据**  
   `/clinical` 上传 CSV，生成病例主数据。

2. **上传双尺度特征**  
   `/data-management` 上传 20x/10x H5。

3. **（可选）为病例登记 20×/10× 特征**  
   `/clinical` 把已上传文件的 ID 记到 `caseId` 下，便于随访、KM、批量预测；单次推理也可跳过本步，在 `/prediction` 直接选文件。

4. **发起训练**  
   `/training` 选择癌种与模型，建议先 `kFolds=1, maxEpochs=1` 冒烟；如需批量复现实验可设置 `repeat` 与 `seed`（repeat 会按 seed 递增）。

5. **查看评估**  
   `/model-evaluation` 查看训练/验证 Loss 曲线（支持「Y 轴自适应 / 从 0」切换）与 KM 分析。

6. **执行预测**  
   `/prediction` 选择「按病例」或「按特征文件」+ 有 checkpoint 的任务，获取风险评分与分层。

7. **历史回溯与复现**  
   结合 `taskId`、`resultsDir`、日志与 checkpoint 复现实验。

---

## 8. 运维与部署要点

## 8.1 当前推荐部署

- 单端口：`http://<host>:80`
- 入口为 `gateway`
- API 统一走 `/api`

## 8.2 常用命令

```bash
cd /srv/vila-mil
docker compose up -d --build
docker compose ps
docker compose logs -f backend
```

## 8.3 建议基线（4C8G 无 GPU）

- 保持 Swap >= 2G（建议 4G）
- 单任务并发（已启用）
- 训练优先轻量模型做流程验证
- 大任务分批，避免长时间高压

---

## 9. 故障排查手册

## 9.1 预测报“未找到 checkpoint”

**现象**：`/api/predict` 返回 checkpoint 缺失  
**排查**：
- 看 `taskId` 的 `resultsDir` 下是否存在 `s_<fold>_checkpoint.pt`
- 查 `training/history` 中 `hasCheckpoint` 字段
**处理**：
- 仅选择 `hasCheckpoint=true` 的任务
- 重新训练一个最小任务产出 checkpoint

## 9.2 训练被拒绝（409）

**原因**：单任务并发保护  
**处理**：等待当前任务完成，或手动 `stop` 当前任务

## 9.3 ViLa_MIL 启动失败（资源预检）

**原因**：无 GPU 或内存/Swap 不足  
**处理**：
- 使用 RRTMIL/AMIL 等先跑通
- 增加内存和 Swap
- 有 GPU 再启用 ViLa_MIL

## 9.4 训练卡住不结束

**机制**：空闲超时 watchdog 会自动回收  
**检查**：
- `api_training_logs/<taskId>.log` 是否有 watchdog 记录
- 任务 `failReason` 是否 `idle-timeout-*`

## 9.5 页面数据不一致

**检查项**：
- 浏览器强刷（`Ctrl+F5`）或清空缓存
- 修改前端后：
  - 若 `docker-compose.yml` 已挂载 `./vila-mil-frontend/dist -> /usr/share/nginx/html`：执行 `npm run build` 后强刷即可（必要时重启 `frontend/gateway`）
  - 若未挂载 `dist`：需 **`docker compose up -d --build`** 重建 `frontend` 镜像，否则静态资源仍为旧构建
- `docker compose ps` 确认 `frontend` / `backend` 为 `healthy`
- `/api/data/datasets` 返回的 `cancers` 是否正常

## 9.6 网关 502 / 接口不可达

**排查顺序**：
1. `docker compose ps` 看服务状态是否 healthy
2. `curl http://localhost/api/health`
3. `docker compose logs backend gateway`

---

## 10. 关键配置项速查

## 10.1 训练稳定性相关环境变量

- `VILAMIL_MIN_AVAIL_GB`
- `VILAMIL_MIN_SWAP_GB`
- `TRAIN_IDLE_TIMEOUT_MIN`
- `TRAIN_IDLE_CHECK_SEC`

## 10.2 API 与模型相关

- `MODEL_CHOICES`：模型展示列表
- `/api/models`：含 `implemented/mode/fallbackTarget`
- `/api/training/history`：含 `hasCheckpoint/checkpointCount`

---

## 11. 接口请求/响应 JSON 示例

以下示例默认网关地址为 `http://localhost`，API 前缀为 `/api`。  
如果你在前端同域访问，可直接使用相对路径；脚本联调用绝对地址更直观。

### 11.1 健康检查

**请求**

```bash
curl -sS http://localhost/api/health
```

**响应**

```json
{
  "ok": true,
  "service": "vila-mil-api"
}
```

### 11.2 模型列表（含实现状态）

**请求**

```bash
curl -sS http://localhost/api/models
```

**响应（节选）**

```json
{
  "models": [
    { "id": "RRTMIL", "name": "RRTMIL", "implemented": true, "mode": "native" },
    { "id": "TransMIL", "name": "TransMIL", "implemented": false, "mode": "fallback", "fallbackTarget": "MIL_fc/MIL_fc_mc" }
  ]
}
```

### 11.3 启动训练

**请求**

```bash
curl -sS -X POST http://localhost/api/training/start \
  -H 'Content-Type: application/json' \
  -d '{
    "cancer": "LUSC",
    "modelType": "RRTMIL",
    "mode": "transformer",
    "maxEpochs": 1,
    "learningRate": 0.00001,
    "kFolds": 1,
    "repeat": 1,
    "seed": 1
  }'
```

**响应（成功）**

```json
{
  "task": {
    "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392",
    "status": "running",
    "modelType": "RRTMIL",
    "kFolds": 1,
    "resultsDir": "/app/result/api_runs/8dc259c2-0a6d-4947-866f-7b2f5f9f4392"
  },
  "tasks": [
    {
      "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392"
    }
  ]
}
```

**响应（并发保护触发，409）**

```json
{
  "message": "当前已有训练任务在运行。为保证系统稳定性，暂时只允许单任务并发。",
  "runningTaskIds": ["8dc259c2-0a6d-4947-866f-7b2f5f9f4392"]
}
```

### 11.4 训练状态 / 历史 / 日志

**请求：状态**

```bash
curl -sS http://localhost/api/training/status/8dc259c2-0a6d-4947-866f-7b2f5f9f4392
```

**响应（节选）**

```json
{
  "task": {
    "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392",
    "status": "completed",
    "progress": 100.0,
    "epoch": 0,
    "kFolds": 1,
    "currentFold": 0
  }
}
```

**请求：历史**

```bash
curl -sS http://localhost/api/training/history
```

**响应（节选）**

```json
{
  "tasks": [
    {
      "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392",
      "status": "completed",
      "hasCheckpoint": true,
      "checkpointCount": 1,
      "isBestForModel": true
    }
  ]
}
```

**请求：日志 tail**

```bash
curl -sS "http://localhost/api/training/log/8dc259c2-0a6d-4947-866f-7b2f5f9f4392?tail=80"
```

**响应（节选）**

```json
{
  "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392",
  "tail": 80,
  "content": "Epoch: 0, train_loss: 1.4476, train_error: 0.7082\n..."
}
```

**请求：查询某模型的最佳 run**

```bash
curl -sS "http://localhost/api/training/best?cancer=LUSC&modelType=RRTMIL&mode=transformer"
```

**请求：删除训练历史（按 taskIds）**

```bash
curl -sS -X POST http://localhost/api/training/history/delete \
  -H 'Content-Type: application/json' \
  -d '{
    "taskIds": ["8dc259c2-0a6d-4947-866f-7b2f5f9f4392"],
    "deleteArtifacts": true
  }'
```

### 11.5 数据集与特征

**请求：数据集汇总**

```bash
curl -sS http://localhost/api/data/datasets
```

**响应（节选）**

```json
{
  "datasets": [{ "id": "LUAD", "name": "LUAD" }, { "id": "LUSC", "name": "LUSC" }],
  "cancers": ["BLCA", "BRCA", "COAD", "ESCA", "HNSC", "KIRC", "LGG", "LIHC", "LUAD", "LUSC", "STAD", "UCEC"],
  "summary": {
    "LUSC": { "10": 1, "20": 2 }
  },
  "totalFiles": 3
}
```

**请求：按癌种查特征**

```bash
curl -sS "http://localhost/api/data/features/LUSC?featureType=20"
```

**响应（节选）**

```json
{
  "cancer": "LUSC",
  "featureType": "20",
  "files": [
    {
      "id": "file_xxx",
      "name": "TCGA-xx.h5",
      "storedPath": "uploaded_features/LUSC/features_20/TCGA-xx.h5"
    }
  ]
}
```

### 11.6 临床导入与特征关联

**请求：上传临床 CSV**

```bash
curl -sS -X POST http://localhost/api/clinical/upload \
  -F "file=@/path/to/clinical.csv"
```

**响应（示例）**

```json
{
  "ok": true,
  "count": 128
}
```

**请求：为病例指定特征文件（写入 `cases.json`）**

```bash
curl -sS -X POST http://localhost/api/clinical/cases/link-feature \
  -H 'Content-Type: application/json' \
  -d '{
    "caseId": "TCGA-18-3406",
    "fileId": "feature_file_id",
    "featureType": "20"
  }'
```

**响应（示例）**

```json
{
  "ok": true,
  "caseId": "TCGA-18-3406",
  "featureType": "20",
  "fileId": "feature_file_id"
}
```

### 11.7 评估接口

**请求：评估 run 列表**

```bash
curl -sS http://localhost/api/evaluation/runs
```

**响应（节选）**

```json
{
  "runs": [
    {
      "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392",
      "modelType": "RRTMIL",
      "status": "completed"
    }
  ]
}
```

**请求：曲线数据**

```bash
curl -sS http://localhost/api/evaluation/curves/8dc259c2-0a6d-4947-866f-7b2f5f9f4392
```

**响应（节选）**

```json
{
  "series": {
    "epoch": [0],
    "trainLoss": [1.4476],
    "valLoss": [1.4339]
  },
  "summary": {
    "bestValCIndex": 0.46
  }
}
```

### 11.8 单病例预测

**方式 A：按病例（`cases.json` 已登记 20×/10×）**

```bash
curl -sS -X POST http://localhost/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "caseId": "TCGA-18-3406",
    "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392",
    "saveHistory": true
  }'
```

**方式 B：按 manifest 文件 ID（无需事先写入病例）**

```bash
curl -sS -X POST http://localhost/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "feature20FileId": "<manifest 中 20× 文件 id>",
    "feature10FileId": "<manifest 中 10× 文件 id>",
    "cancer": "LUSC",
    "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392",
    "saveHistory": true
  }'
```

可选：在方式 B 中附带 `"caseId": "TCGA-18-3406"`，若该病例存在于 `cases.json`，响应中会包含随访摘要。

**响应（节选）**

```json
{
  "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392",
  "modelType": "RRTMIL",
  "caseId": "TCGA-18-3406",
  "riskScore": 1.7321,
  "predClass": 2,
  "inputSummary": {
    "featureSource": "caseRecord",
    "featureH5Ready": true
  },
  "riskStratification": {
    "tier": "high",
    "labelZh": "高风险"
  },
  "visualization": {
    "probabilityBar": {
      "x": ["低风险", "中等风险", "偏高风险", "高风险"],
      "y": [0.12, 0.28, 0.33, 0.27]
    }
  }
}
```

**响应（checkpoint 缺失）**

```json
{
  "message": "未找到 checkpoint（期望 resultsDir 下存在 s_<fold>_checkpoint.pt）"
}
```

### 11.9 批量预测

**请求**

```bash
curl -sS -X POST http://localhost/api/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "items": [
      { "caseId": "TCGA-18-3406", "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392" },
      { "caseId": "TCGA-18-3407", "taskId": "8dc259c2-0a6d-4947-866f-7b2f5f9f4392" }
    ]
  }'
```

**响应（节选）**

```json
{
  "ok": true,
  "results": [
    { "caseId": "TCGA-18-3406", "riskScore": 1.73 },
    { "caseId": "TCGA-18-3407", "riskScore": 0.92 }
  ]
}
```

---

## 12. 版本说明

- 本文档已对齐当前仓库中的已落地改动：
  - 动态癌种来源统一
  - 模型实现状态标注
  - 训练并发限制
  - 空闲超时自动停止
  - Prediction 可预测任务过滤
  - `/api/predict` 支持按病例或按 `feature20FileId`/`feature10FileId` 直接推理；Clinical 文案统一为「指定/关联」而非「绑定」
  - Training Queue 支持删除：新增 `/api/training/queue/delete`，前端支持单条删除与清空队列（仅影响 `queued`）
  - 修复 `training/status` 对排队任务误写 `kFolds=1` 的问题（队列展示不再被轮询污染）
  - `ModelEvaluation` 移除“手动分组 KM+log-rank”模块，新增 LUSC 示例 KM 卡片（数据源 `lusc (1).csv`，baseline 切换）
  - 新增后端接口 `GET /api/evaluation/km/lusc-demo`，返回 Ours/Others KM 曲线与互斥子集 log-rank
  - Clinical/Prediction 新增 TRIDENT 提特征流程：`extractor=raster|trident`，WSI 路径可直接生成双尺度 H5 并关联/推理
  - TRIDENT 路径增加 `mpp` 参数：对 PNG/JPEG 在 TRIDENT 模式下要求提供 `mpp`（如 `0.25`）
  - Docker 部署补充：首次需先构建 `vila-mil-backend-base:local`；并说明 PyTorch 依赖安装顺序以避免重复下载
  - Docker 运行补充：挂载 `./TRIDENT:/TRIDENT` 并设置 `TRIDENT_REPO_DIR=/TRIDENT`（用于容器内调用 TRIDENT）
  - 集成方法演进：`EnsembleDecision` 当前固定 `avg_prob`，并在评估端补齐 fold->480 点统一曲线（仅展示层平滑，不改原始日志指标）
  - ModelEvaluation 最优任务对比中，`EnsembleDecision` 已可与 5 个基线同图对齐显示（Loss/AUC）
  - Clinical 页面模式简化：合并为 `已有 H5` + `从 WSI 生成`；保留“快速预览≈低采样近似 / 正式预测=TRIDENT 全量”文案
  - WSI 上传链路修复：`.svs/.ndpi/.mrxs/.scn` 在上传、关联、推理入口统一放行并完成路径对齐

如后续新增模型、改动训练入口或调整接口字段，请同步更新本文件。

