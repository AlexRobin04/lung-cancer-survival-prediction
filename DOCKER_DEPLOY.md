# Docker 部署（当前机器）

> **若后端不在 Docker 里跑**（宿主机直接用 gunicorn / `python` 起 `api_server`），请改看 **[`deploy/README.md`](./deploy/README.md)**：Nginx 反代 + systemd 托管，与 Docker 二选一即可。

本项目可通过 `docker compose` 在单机启动前后端，并通过 **Nginx 网关实现单端口部署**：

- **统一入口（单端口）**：`http://<server-ip>/`
- **后端 API（同源）**：`http://<server-ip>/api/*`（例如 `http://<server-ip>/api/health`）

## 1) 前置条件

- 已安装 Docker 与 Docker Compose 插件：
  - `docker --version`
  - `docker compose version`

> 注意：网关占用 **80 端口**。如果宿主机已安装并运行 Nginx/Apache，会端口冲突。
> 需要先停掉宿主机服务（示例：`systemctl stop nginx`）。

## 2) 启动

### 首次构建：必须先打本地基础镜像

后端 `ViLa-MIL/Dockerfile` 第一行是 `FROM vila-mil-backend-base:local`，该镜像**不在** Docker Hub，需在本机先构建一次：

```bash
cd /srv/vila-mil/ViLa-MIL
docker build -f Dockerfile.base -t vila-mil-backend-base:local .
```

**说明（避免重复下载 PyTorch）**：`Dockerfile.base` 里**先**从 PyTorch 官方 CPU 源安装 `torch/torchvision/torchaudio`，**再** `pip install -r requirements-api-base.txt`。若顺序反了，`timm` 等会先从默认 PyPI 拉一整份 torch，后面再装 CPU 版，等于同类大包下两遍、体积和时间都浪费。

完成后再回到项目根目录执行 `docker compose build backend` 或 `up -d --build`。

### 日常更新后端 / 网关

```bash
cd /srv/vila-mil
# 可选：docker compose pull   # 仅当 compose 里使用可拉取的镜像时；gateway 用官方 nginx 时可拉到
docker compose build backend
docker compose up -d backend gateway
```

若只改动了 `ViLa-MIL/` 代码，一般**不必**重建 `Dockerfile.base`，除非基础依赖（`requirements-api-base.txt` 等）有变。

### 一键起全套（含前端）

当前 compose 已改为：**前端不在 Docker 内构建**，而是直接由 nginx 挂载本地 `vila-mil-frontend/dist`。
这样可避免每次拉取 `node:20-alpine`，降低网络不稳定导致的构建失败。

先在宿主机构建前端静态资源：

```bash
cd /srv/vila-mil/vila-mil-frontend
npm install
npm run build
```

再回项目根目录启动容器：

在项目根目录执行：

```bash
cd /srv/vila-mil
docker compose up -d --build
```

> 说明：`frontend` 现在使用 `nginx:1.27-alpine` 直接托管本地 `dist`；  
> 前端代码更新后，只需重新执行 `npm run build`，必要时 `docker compose restart frontend gateway`。

## 3) 查看状态与日志

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f gateway
```

## 4) 停止

```bash
docker compose down
```

## 5) 说明

- 本机无 GPU，`ViLa_MIL` 已在后端做 CUDA 预检，无 GPU 会拒绝启动（这是保护机制）。
- 训练结果、日志、任务文件通过 volume 挂载到宿主机目录，容器重建后仍保留：
  - `ViLa-MIL/result`
  - `ViLa-MIL/api_training_logs`
  - `ViLa-MIL/uploaded_features`
- 单端口模式下，前端默认同源请求 `/api`，一般不需要额外设置 API BaseURL；
  若你反向代理路径有变化，再去 `Settings` 调整即可。

## 6) TRIDENT（当前推荐配置）

当前 `docker-compose.yml` 已内置以下后端环境变量（`backend.environment`）：

- `TRIDENT_REPO_DIR=/TRIDENT`
- `TRIDENT_PATCH_ENCODER=conch_v15`
- `TRIDENT_ALLOW_RASTER_FALLBACK=0`（纯 TRIDENT，不自动回退到 ResNet）
- `TRIDENT_MAX_WORKERS=0`（稳定优先，降低并发）
- `TRIDENT_FORCE_FEATURE_DIM=512`（将每个尺度特征适配为 512，双尺度拼接为 1024，便于兼容现有训练任务）

同时需保证挂载：

- `./TRIDENT:/TRIDENT:ro`

> 说明：`TRIDENT_FORCE_FEATURE_DIM=512` 是在后端对 TRIDENT 产出的 H5 做维度适配（截断/补零），用于兼容历史 1024 输入模型。

## 7) 网关超时（长任务必配）

TRIDENT 生成特征耗时较长，建议确认 `deploy/nginx-gateway.conf` 的 `/api/` 已配置：

- `proxy_connect_timeout 60s`
- `proxy_send_timeout 3600s`
- `proxy_read_timeout 3600s`
- `send_timeout 3600s`

修改后重启网关：

```bash
cd /srv/vila-mil
docker compose restart gateway
```

## 8) 常见故障快速处理

- `504 Gateway Timeout`
  - 原因：网关超时过短。
  - 处理：按上节配置超时并重启 `gateway`。

- `Torch not compiled with CUDA enabled`
  - 原因：TRIDENT 走到 CUDA 分支但后端是 CPU torch。
  - 处理：已在本项目修复为 CPU 可运行路径；若仍报错，重建并重启 `backend`。

- `ValueError: max_workers must be greater than 0`
  - 原因：TRIDENT 线程池不接受 0 worker。
  - 处理：项目已在 TRIDENT 代码中兼容（该入口自动至少 1），重启后端生效。

- 预测时报维度不匹配（如 `...1536 and 1024...`）
  - 原因：特征维度与任务 checkpoint 不一致。
  - 处理：重新生成特征（确保维度适配已生效）或切换兼容任务。

## 9) 安全组建议（生产）

- 放行 `80/tcp`（Web）
- 放行 `22/tcp`（SSH，建议限制为你的 IP 白名单）
- 如需 HTTPS，可额外放行 `443/tcp`（并在网关层加证书）
