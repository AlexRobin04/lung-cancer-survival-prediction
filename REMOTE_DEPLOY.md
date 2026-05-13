# 远程部署说明

目标示例：**`root@8.130.211.90`**，仓库解压后目录名为 **`vila-mil`**（与 `docker-compose.yml` 同级）。

---

## 0. 上传方式怎么选

| 情况 | 做法 |
|------|------|
| 服务器已装 **`rsync`**，要传 **单个几 GB 的 tar.gz** | **§2.2**（推荐脚本）或 **§2.3**（手写 rsync），**断线后重复同一条命令即可续传** |
| 服务器 **不能装 rsync** / 只有 `scp` | **§2.4** 分卷 + `scp`（脚本 `upload_split_scp.sh`） |
| 网络极差、单文件总失败 | **§2.5** 分卷 + 每卷再用 rsync；或把 **§2.4** 里 `CHUNK` 改小（如 `100M`） |
| 小文件 | 普通 `scp` 即可 |

---

## 1. 本机打包

在**仓库根**（与 `docker-compose.yml` 同级）执行：

**默认 slim**（不含 `.git`、`node_modules`、`ViLa-MIL/result/api_runs`、`ViLa-MIL/api_training_logs`，体积明显更小；**仍含** `TRIDENT`、`ViLa-MIL/ckpt`、`ViLa-MIL/features`、`vila-mil-frontend/dist` 等运行所需内容）：

```bash
cd "/path/to/vila-mil"
chmod +x scripts/package_for_deploy.sh
./scripts/package_for_deploy.sh ~/Desktop
```

生成文件形如：`~/Desktop/vila-mil-deploy-YYYYMMDD-HHMMSS.tar.gz`。下文用 **`$DEPLOY_PKG`** 指代该文件的**绝对路径**（请按你机器上实际文件名替换）。

**完整包**（含 `result/api_runs` 等，可能 **>10 GB**）：

```bash
SLIM=0 ./scripts/package_for_deploy.sh ~/Desktop
```

---

## 2. 上传到服务器

大文件**不要**用整包 `scp` 硬传（易超时、`Connection reset by peer`，且**一般不能续传**）。

### 2.1 先确认：本机、服务器都有 `rsync`

本机（Mac 通常已有）：

```bash
rsync --version
```

服务器上若没有，**SSH 登录服务器后**安装（按系统任选其一）：

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y rsync

# CentOS / RHEL 7
sudo yum install -y rsync

# Rocky / AlmaLinux / Fedora
sudo dnf install -y rsync

# Alpine
apk add --no-cache rsync
```

若报错 **`bash: rsync: 未找到命令`** 且 **`exited with status 127`**：多半是 **远端**未装 `rsync`（`rsync` 会经 SSH 在服务器再调一次 `rsync`）。装好后再从本机传。

若**无法**在服务器安装 `rsync`，请直接用 **§2.4**（分卷 + `scp`，不依赖远端 `rsync`）。

---

### 2.2 推荐：单文件断点续传（`upload_resumable.sh`）

```bash
cd "/path/to/vila-mil"
chmod +x scripts/upload_resumable.sh

export DEPLOY_PKG="$HOME/Desktop/vila-mil-deploy-20260512-181141.tar.gz"   # 改成你的实际文件名

./scripts/upload_resumable.sh \
  "$DEPLOY_PKG" \
  root@8.130.211.90:/root/
```

说明：

- 出现 **`Transfer starting: 1 files`** 后，`-P` 会陆续打出 **进度 / 速度**（大文件可能要很久，属正常）。
- **中断后**：再次执行**完全相同**的 `./scripts/upload_resumable.sh ...` 即可续传。
- 另开终端观察远端文件是否在变大：

```bash
ssh -o ServerAliveInterval=15 root@8.130.211.90 'ls -lh /root/vila-mil-deploy-*.tar.gz'
```

传完后与本机比大小（字节应一致）：

```bash
ls -lh "$DEPLOY_PKG"
ssh root@8.130.211.90 'ls -lh /root/vila-mil-deploy-*.tar.gz'
```

仍频繁掉线时，可加密保活（本机）：

```bash
SSH_OPTS="-o ServerAliveInterval=10 -o ServerAliveCountMax=240 -o TCPKeepAlive=yes" \
  ./scripts/upload_resumable.sh "$DEPLOY_PKG" root@8.130.211.90:/root/
```

---

### 2.3 等价命令：手写 `rsync`（不跑脚本）

```bash
export DEPLOY_PKG="$HOME/Desktop/vila-mil-deploy-20260512-181141.tar.gz"

rsync -avhP --partial --inplace \
  -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=120 -o TCPKeepAlive=yes" \
  "$DEPLOY_PKG" \
  root@8.130.211.90:/root/
```

- **`-P`**：`--partial` + 进度。
- **`--inplace`**：往同一个目标文件续写，适合大单文件续传。

---

### 2.4 服务器无 `rsync`：分卷 + `scp`（可断点重跑）

脚本在**本机**分卷、逐卷 `scp`；**已传满的卷**再次运行会按远程文件大小**自动跳过**。

```bash
cd "/path/to/vila-mil"
chmod +x scripts/upload_split_scp.sh

export DEPLOY_PKG="$HOME/Desktop/vila-mil-deploy-20260512-181141.tar.gz"

# 默认每卷 300MB；网络差可改为：CHUNK=200M 或 CHUNK=100M
./scripts/upload_split_scp.sh "$DEPLOY_PKG" root@8.130.211.90:/root/
```

全部卷上传完成后，在**服务器上合并**（`sort` 保证分卷顺序正确；**`f=` 改成与你的 tar.gz  basename 一致**）：

```bash
ssh root@8.130.211.90 'cd /root && f=vila-mil-deploy-20260512-181141.tar.gz && cat $(ls -1 "${f}.part_"* | sort) > "$f" && ls -lh "$f"'
```

与本机 `ls -lh "$DEPLOY_PKG"` 核对大小无误后，再删远程分卷：

```bash
ssh root@8.130.211.90 'rm -f /root/vila-mil-deploy-*.tar.gz.part_*'
```

（本机 `split` 产生的临时分卷在脚本的临时目录里，脚本结束会清理；若曾手动 `split` 在桌面，请自行删除 `*.part_*`。）

---

### 2.5 网络仍不稳：分卷 + `rsync`（远端需已装 `rsync`）

每卷更小，单次失败成本低：

```bash
cd ~/Desktop
export BASE=vila-mil-deploy-20260512-181141.tar.gz
split -b 500M "$BASE" "${BASE}.part_"
rsync -avhP --partial \
  -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=120" \
  "${BASE}.part_"* root@8.130.211.90:/root/
```

服务器合并（同样建议 `sort`）：

```bash
ssh root@8.130.211.90 'cd /root && f=vila-mil-deploy-20260512-181141.tar.gz && cat $(ls -1 "${f}.part_"* | sort) > "$f" && ls -lh "$f"'
```

---

### 2.6 不推荐：整包 `scp` 传大 tar.gz

仅适合小文件。大包请用 **§2.2～§2.5**。

---

## 3. 服务器解压、前端构建与 Docker 启动

### 3.1 解压部署包

```bash
ssh root@8.130.211.90
cd /root
tar -xzf vila-mil-deploy-*.tar.gz
cd vila-mil
```

以下路径默认以 **`/root/vila-mil`** 为仓库根（与 `docker-compose.yml` 同级）；若你的解压目录不同，请自行替换。

---

### 3.2 前端：先同步完整源码，再装依赖、构建（必读）

Docker 只挂载 **`vila-mil-frontend/dist`**，**不会**在容器里替你跑 `npm build`。若服务器上 **`vila-mil-frontend/` 不完整或只有空目录**，必须在**宿主机**（有 Node 的环境）先准备好 **`dist`**，再 `docker compose up`。

**先把完整前端同步到服务器（任选其一）：**

1. **Git**：在服务器拉全仓库（含 `vila-mil-frontend` 源码）；或  
2. **部署包**：按 **§1** 用 `scripts/package_for_deploy.sh` 打的包，其中已带 `vila-mil-frontend`（含 `src` 等）；或  
3. **rsync / scp**：从本机把完整的 **`vila-mil-frontend/`** 拷到服务器的 **`/root/vila-mil/vila-mil-frontend/`**（需含 `package.json`、`src/` 等，与本地仓库一致）。

**确认存在前端工程后再装依赖、构建**（在仓库根执行，与根目录 `package.json` 的 `--prefix` 一致）：

```bash
cd /root/vila-mil
test -f vila-mil-frontend/package.json && echo "前端目录 OK" || echo "缺少 vila-mil-frontend/package.json，请先同步完整前端"

npm run install:frontend
npm run build
```

若你已在子目录里放好了完整前端，也可以只在子目录操作：

```bash
cd /root/vila-mil/vila-mil-frontend
npm install
npm run build
```

**注意：** 不要在**空的** `vila-mil-frontend` 里执行仓库根目录的 `npm run build`。根目录脚本等价于 `npm run build --prefix vila-mil-frontend`，会去子目录找 `package.json`；若子目录为空或缺文件，会报错或反复失败。

构建完成后应存在目录 **`/root/vila-mil/vila-mil-frontend/dist/`**（内含 `index.html` 等），再启动 Docker：

---

### 3.3 Docker 镜像与 Compose

按 **[DOCKER_DEPLOY.md](./DOCKER_DEPLOY.md)** 首次构建基础镜像并启动：

```bash
cd /root/vila-mil
cd ViLa-MIL
docker build -f Dockerfile.base -t vila-mil-backend-base:local .
cd ..
docker compose build backend
docker compose up -d --build
```

> **平台**：`docker-compose.yml` 中 `platform: linux/amd64` 面向常见 **x86_64** 云主机；若为 **ARM**，需自行调整 `platform` 或镜像构建方式。

---

### 3.4 同步离线特征目录 `ViLa-MIL/features`（可选）

本机已有 **`ViLa-MIL/features`**（约数十 MB 级 H5 等）时，可单独增量同步到服务器，无需重传整个部署包：

```bash
cd "/path/to/vila-mil"
chmod +x scripts/sync_features_to_server.sh
./scripts/sync_features_to_server.sh
```

默认同步到 **`root@8.130.211.90:/root/vila-mil/ViLa-MIL/features/`**。若服务器上的仓库根不是 `/root/vila-mil`，可改环境变量：

```bash
REMOTE=root@8.130.211.90 REMOTE_ROOT=/srv/vila-mil ./scripts/sync_features_to_server.sh
```

等价手写命令：

```bash
rsync -avhP --partial \
  -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=120 -o TCPKeepAlive=yes" \
  "/path/to/vila-mil/ViLa-MIL/features/" \
  root@8.130.211.90:/root/vila-mil/ViLa-MIL/features/
```

---

## 4. 访问与健康检查

- 浏览器：`http://8.130.211.90/`（网关 **80** 端口）。
- 服务器本机：`curl -s http://127.0.0.1/api/health`（若只映射 80，需经网关或改 curl 地址，以你机上实际监听为准）。

---

## 5. 可选：Git 克隆 + 仅同步大目录（免打巨大 tar）

若代码已在 Git 托管，可在服务器 `git clone` 后，用 `rsync` 从本机只同步 **`ViLa-MIL/ckpt`、`ViLa-MIL/features`、`TRIDENT`** 等大目录。**注意**：若在**服务器上**执行 `npm run build` 生成前端，仍需仓库里**完整的 `vila-mil-frontend` 源码**；若你只在服务器上放 **`dist`** 而从不跑 `npm build`，则需保证 `vila-mil-frontend/dist` 与 compose 挂载路径一致（参见 **§3.2**）。
