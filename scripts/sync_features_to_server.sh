#!/usr/bin/env bash
# 将本机 ViLa-MIL/features 同步到远程服务器（rsync，可重复执行增量更新）。
#
# 用法：
#   ./scripts/sync_features_to_server.sh
#   REMOTE=root@1.2.3.4 REMOTE_ROOT=/root/vila-mil ./scripts/sync_features_to_server.sh
#
# 默认：
#   REMOTE=root@8.130.211.90
#   REMOTE_ROOT=/root/vila-mil   （与 REMOTE_DEPLOY.md 解压路径一致）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/ViLa-MIL/features/"
REMOTE="${REMOTE:-root@8.130.211.90}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/vila-mil}"
DEST="${REMOTE}:${REMOTE_ROOT}/ViLa-MIL/features/"

SSH_BASE="-o ServerAliveInterval=15 -o ServerAliveCountMax=120 -o TCPKeepAlive=yes"
SSH_OPTS="${SSH_OPTS:-$SSH_BASE}"

if [[ ! -d "$SRC" ]]; then
  echo "本地目录不存在: $SRC" >&2
  exit 1
fi

echo "源: $SRC"
echo "目标: $DEST"
echo "（中断后可再次运行同一命令续传）"
exec rsync -avhP --partial \
  -e "ssh $SSH_OPTS" \
  "$SRC" "$DEST"
