#!/usr/bin/env bash
# 大文件断点续传上传到服务器（用 rsync，中断后重复执行同一条命令即可续传）。
#
# 用法：
#   ./scripts/upload_resumable.sh ~/Desktop/vila-mil-deploy-20260512-181141.tar.gz root@8.130.211.90:/root/
#
# 可选环境变量：
#   SSH_OPTS   附加 ssh 参数，默认已含保活，可覆盖或追加，例如：
#   SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=240" ./scripts/upload_resumable.sh ...
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "用法: $0 <本地文件> <user@host:远程路径>" >&2
  echo "示例: $0 ~/Desktop/vila-mil-deploy-xxx.tar.gz root@8.130.211.90:/root/" >&2
  exit 1
fi

SRC="$1"
DEST="$2"

if [[ ! -f "$SRC" ]]; then
  echo "文件不存在: $SRC" >&2
  exit 2
fi

# 保活：降低「长时间只传数据、无交互」被中间设备掐断的概率
DEFAULT_SSH_OPTS="-o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o TCPKeepAlive=yes"
SSH_OPTS="${SSH_OPTS:-$DEFAULT_SSH_OPTS}"

echo "源: $SRC"
echo "目标: $DEST"
echo "SSH: $SSH_OPTS"
echo "（若中断，请直接再次运行本命令即可续传）"
echo

# -a 归档；-v 详情；-h 人类可读
# --partial 保留未完成文件以便续传；-P 等同 --partial --progress
# 已是 .tar.gz 不再 -z，省 CPU
# --timeout=0 不限制单次 I/O 等待（仍受 TCP/SSH 影响）
exec rsync -avhP --partial --inplace \
  -e "ssh $SSH_OPTS" \
  "$SRC" "$DEST"
