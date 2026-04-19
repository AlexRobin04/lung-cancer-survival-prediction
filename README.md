# 肺癌生存预测系统（ViLa-MIL Web）

基于 **CVPR 2024 [ViLa-MIL](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ViLa_MIL_Dual-scale_Vision-Language_Multiple_Instance_Learning_for_Whole_Slide_Image_CVPR_2024_paper.pdf)** 的全切片图像（WSI）相关能力，本仓库在原始算法代码之上扩展了 **HTTP API（Flask + gunicorn）** 与 **Web 前端（React + Vite + MUI）**，用于训练任务管理、特征上传、多模型选择与生存/风险相关可视化等一体化流程。

> 原始论文与数据预处理说明见 [`ViLa-MIL/README.md`](./ViLa-MIL/README.md)。本 README 描述的是**本毕设工程化系统**（前后端 + 部署），而非论文复现的逐步教程。

## 功能概览

- **后端 API**：`ViLa-MIL/api_server.py`，训练子进程调用 `main_LUSC.py`；支持多种 MIL / 融合模型（含 `EnsembleFeature` 等）。
- **前端**：`vila-mil-frontend/`，构建后由 Nginx 托管；请求前缀为 **`/api`**（与网关反代一致）。
- **可选 TRIENT**：Docker 中通过卷挂载 `TRIDENT/` 与相关环境变量，用于特征编码等（详见 `docker-compose.yml`）。

## 仓库结构

```
.
├── ViLa-MIL/              # Python 后端：模型、训练脚本、API、Dockerfile
├── vila-mil-frontend/     # React 前端源码与 Nginx 配置
├── deploy/                # 非 Docker 部署：systemd、Nginx、依赖清单、同步脚本
├── TRIDENT/               # （可选）特征编码相关仓库，Compose 中只读挂载
├── docker-compose.yml     # 后端 + 前端静态 + 网关（单端口 80）
├── DOCKER_DEPLOY.md       # Docker Compose 部署步骤（含基础镜像构建）
└── ViLa-MIL/docs/         # 如 EnsembleFeature 使用说明等
```

## 环境要求（摘要）

- **后端**：Python 3、PyTorch 等（完整依赖以 `ViLa-MIL/deploy/requirements-api-*.txt` 与 `deploy/requirements-api.txt` 为准）；WSI/训练相关还需按 `ViLa-MIL/README.md` 准备 OpenSlide 等。
- **前端**：Node.js 20 LTS 推荐（`npm install` / `npm run build`）。
- **权重与数据**：`ckpt/`、`features/`、`uploaded_features/` 等目录中的大文件通常**不随 Git 提交**，需在目标机器上自行放置或挂载（Docker 已在 `docker-compose.yml` 中映射常见路径）。

## 本地开发（快速）

### 前端

```bash
cd vila-mil-frontend
npm install
npm run dev
```

生产构建产物在 `vila-mil-frontend/dist`：可在 **`vila-mil` 仓库根** 直接执行 **`npm run build`**（根目录 `package.json` 会转发到 `vila-mil-frontend`），或 `cd vila-mil-frontend && npm run build`。

默认 Vite 开发服务器；需将 API 代理到本机后端（按项目内 `vite` 配置，一般指向带 `/api` 的后端地址）。

### 后端 API

```bash
cd ViLa-MIL
# 建议在虚拟环境中安装依赖后：
python api_server.py
# 或
gunicorn --bind 127.0.0.1:8000 api_server:app
```

健康检查示例：`curl http://127.0.0.1:8000/api/health`

> 可通过环境变量 **`VILAMIL_PYTHON_BIN`** 指定训练子进程使用的 Python 解释器。

## 部署方式（二选一）

| 方式 | 说明文档 |
|------|-----------|
| **Docker Compose**（推荐一键：网关 `:80`，同源 `/api/*`） | [`DOCKER_DEPLOY.md`](./DOCKER_DEPLOY.md) |
| **宿主机 Nginx + gunicorn + systemd** | [`deploy/README.md`](./deploy/README.md) |

Docker 场景下注意：**首次**需按 `DOCKER_DEPLOY.md` 构建本地基础镜像 `vila-mil-backend-base:local`，再 `docker compose up`；前端需先在宿主机于**仓库根**执行 `npm run build`（或进入 `vila-mil-frontend` 再 build）生成 `vila-mil-frontend/dist`。

## API 与模型

- API 实现：`ViLa-MIL/api_server.py`。
- 当前可选模型标识（与前端下拉一致）包括但不限于：`ViLa_MIL`、`TransMIL`、`AMIL`、`WiKG`、`RRTMIL`、`PatchGCN`、`surformer`、`DSMIL`、`S4MIL`、`EnsembleFeature`（以后端 `MODEL_CHOICES` 为准）。

融合与特征相关说明见：**[`ViLa-MIL/docs/EnsembleFeature使用说明.md`](./ViLa-MIL/docs/EnsembleFeature使用说明.md)**。

## 许可证与致谢

- **ViLa-MIL** 原作者与论文信息见 [`ViLa-MIL/README.md`](./ViLa-MIL/README.md)。
- **CONCH** 等第三方子目录请遵循其自带 `LICENSE` / `README`。
- 本仓库在论文代码基础上完成的界面、API 与部署文档为毕设工程内容；引用论文请注明原始文献。

## 远程仓库

若使用 Git 管理，克隆后请确认 `ViLa-MIL/` 为**普通目录**（完整树），而非未初始化的 submodule；否则远程会缺少后端源码。
