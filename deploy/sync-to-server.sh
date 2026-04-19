#!/usr/bin/env bash
# 将本地「毕设」目录同步到云主机（需已能用 ssh 登录）
# 默认：root@121.41.39.63 → /srv/vila-mil/
# 用法：
#   chmod +x deploy/sync-to-server.sh && ./deploy/sync-to-server.sh
# 覆盖示例：REMOTE_DIR=/opt/vila-mil ./deploy/sync-to-server.sh
#
# 首次建议：ssh-copy-id root@121.41.39.63

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-121.41.39.63}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/srv/vila-mil}"

# 脚本所在目录 -> 毕设根（deploy 的上一级）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "本地目录: $PROJECT_ROOT"
echo "远程: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
echo ""

ssh -o ConnectTimeout=10 "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"

rsync -avz \
  --progress \
  --human-readable \
  --exclude 'node_modules' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude '.env.local' \
  --exclude 'dist' \
  --exclude '.cursor' \
  -e ssh \
  "${PROJECT_ROOT}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo ""
echo "同步完成。到服务器上执行："
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_DIR}/vila-mil-frontend && npm ci && npm run build"
echo "  并按 deploy/README.md 配置 nginx 与 systemd。"
