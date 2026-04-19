# ViLa-MIL 生产部署说明（Nginx + gunicorn）

默认目录约定：**`/srv/vila-mil/`**（可按需改，并同步修改 `nginx-vila-mil.conf` 与 `gunicorn.service`）。

```
/srv/vila-mil/
├── ViLa-MIL/              # 后端仓库
├── vila-mil-frontend/     # 前端仓库
└── venv/                  # Python 虚拟环境
```

## 1. 服务器准备（Ubuntu 示例）

```bash
sudo apt update && sudo apt install -y nginx git python3-venv python3-pip curl
# Node 20 LTS（二选一：nodesource 或 nvm）
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

安全组放行：**22、80**（及 **443** 若上 HTTPS）。

## 2. 拉代码与目录

### 方式 A：本机 rsync 同步（推荐）

在**你自己的 Mac 终端**执行（需已能 `ssh 登录服务器`，可先 `ssh-copy-id` 免密）：

```bash
cd /path/to/毕设
chmod +x deploy/sync-to-server.sh
./deploy/sync-to-server.sh
```

默认 **`root@121.41.39.63`**，远程目录 **`/srv/vila-mil/`**（与当前云主机一致）。仅当要改路径时例如：

```bash
REMOTE_DIR=/opt/vila-mil ./deploy/sync-to-server.sh
```

脚本会排除 `node_modules`、`venv`、前端 `dist` 等，减小体积；同步后在服务器上 `npm ci && npm run build` 与创建 Python venv。

### 方式 B：服务器上 git clone / scp

```bash
sudo mkdir -p /srv/vila-mil
sudo chown $USER:$USER /srv/vila-mil
cd /srv/vila-mil
# 将本仓库放到此目录，结构为：deploy/、ViLa-MIL/、vila-mil-frontend/
```

## 3. Python 与 API

```bash
cd /srv/vila-mil
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r deploy/requirements-api.txt
# 若训练等功能需完整环境，再按你本地 conda 环境补充安装 torch 等
```

> 以下命令均在 `/srv/vila-mil` 且已 `source venv/bin/activate` 的前提下书写；`deploy` 指仓库根目录下的 `deploy` 文件夹。

测试：

```bash
cd /srv/vila-mil/ViLa-MIL
../venv/bin/gunicorn --bind 127.0.0.1:8000 api_server:app
# 另开终端：curl http://127.0.0.1:8000/api/health
```

## 4. systemd 托管 API

```bash
sudo cp /srv/vila-mil/deploy/gunicorn.service /etc/systemd/system/vila-mil-api.service
# 用编辑器确认 WorkingDirectory、ExecStart 中的 venv 路径与 User=www-data
sudo chown -R www-data:www-data /srv/vila-mil/ViLa-MIL/api_training_logs 2>/dev/null || true
sudo systemctl daemon-reload
sudo systemctl enable --now vila-mil-api
sudo systemctl status vila-mil-api
```

若 `User=www-data` 权限不便，可改为你的部署用户，并保证 Nginx 能连上 `127.0.0.1:8000`。

## 5. 前端构建

```bash
cd /srv/vila-mil/vila-mil-frontend
cp ../deploy/env.frontend.production.example .env.production
npm ci
npm run build
```

确认生成 `dist/index.html`。

## 6. Nginx

```bash
sudo cp /srv/vila-mil/deploy/nginx-vila-mil.conf /etc/nginx/sites-available/vila-mil
# 编辑 root 为实际 dist 路径，如 /srv/vila-mil/vila-mil-frontend/dist
sudo ln -sf /etc/nginx/sites-available/vila-mil /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

浏览器访问：`http://<服务器公网IP>/`  
接口：`http://<服务器公网IP>/api/health`

## 7. 环境变量说明

| 变量 | 说明 |
|------|------|
| `VITE_API_BASE_URL=/api` | 前端生产构建用，请求同源 `/api` |
| `VILAMIL_PYTHON_BIN` | `api_server` 启动训练子进程时使用的 Python，建议指向 venv 内 `python` |

## 8. 更新版本

```bash
cd /srv/vila-mil/ViLa-MIL && git pull
cd /srv/vila-mil/vila-mil-frontend && git pull && npm ci && npm run build
sudo systemctl restart vila-mil-api
sudo systemctl reload nginx
```

## 9. HTTPS（可选）

域名解析到服务器后，使用 Let’s Encrypt（certbot）或阿里云 SSL 证书，在 Nginx 中增加 `listen 443 ssl` 与证书路径。

## 10. 说明

- **训练任务**依赖 conda/大量依赖与数据路径时，建议在服务器上复现你本机环境，或仅部署「展示 + 轻量 API」，重训练在本机完成。
- 若代码不在 `/srv/vila-mil`，请在所有配置与 systemd 单元中替换为实际路径。
- **非 Docker 后端**：`WorkingDirectory` 一般为 `ViLa-MIL`（与 `gunicorn api_server:app` 一致）。接口若依赖仓库内文件（例如评估页 **LUSC KM 示例** 读取 `ViLa-MIL/lusc (1).csv`），只要该文件在部署目录中即可，**无需** Docker 卷；更新代码后执行 `sudo systemctl restart vila-mil-api`（或你的 service 名）。
