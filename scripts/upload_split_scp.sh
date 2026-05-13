#!/usr/bin/env bash
# 将大文件切成若干卷后用 scp 逐卷上传（远程不需要 rsync）。
# 某一卷失败时，重新运行本脚本：已完整上传的卷会自动跳过（按远程文件大小比对）。
#
# 用法：
#   ./scripts/upload_split_scp.sh <本地tar.gz> root@8.130.211.90:/root/
#
# 可选环境变量：
#   CHUNK=300M     每卷大小（默认 300M，网络差可改为 200M / 100M）
#   SSH_OPTS       传给 scp/ssh 的额外参数
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "用法: $0 <本地文件> user@host:远程目录/>" >&2
  echo "示例: $0 ~/Desktop/vila-mil-deploy-xxx.tar.gz root@8.130.211.90:/root/" >&2
  exit 1
fi

SRC="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
DEST_SPEC="$2"

if [[ ! -f "$SRC" ]]; then
  echo "文件不存在: $SRC" >&2
  exit 2
fi

if [[ "$DEST_SPEC" != *:* ]]; then
  echo "目标必须是 user@host:/path/ 形式，且目录以 / 结尾，例如 root@1.2.3.4:/root/" >&2
  exit 2
fi

USER_HOST="${DEST_SPEC%%:*}"
REMOTE_DIR="${DEST_SPEC#*:}"
[[ "$REMOTE_DIR" == */ ]] || REMOTE_DIR="${REMOTE_DIR}/"

CHUNK="${CHUNK:-300M}"
SSH_BASE="-o ServerAliveInterval=15 -o ServerAliveCountMax=120 -o TCPKeepAlive=yes"
SSH_OPTS="${SSH_OPTS:-$SSH_BASE}"

BASE="$(basename "$SRC")"
WORKDIR="$(mktemp -d "/tmp/vila-mil-upload-${BASE}.XXXXXX")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

PART_PREFIX="${WORKDIR}/${BASE}.part_"
echo "分卷目录: $WORKDIR"
echo "每卷大小: $CHUNK"
split -b "$CHUNK" "$SRC" "$PART_PREFIX"

# 本机文件大小（BSD stat）
local_size() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }

# 远程文件大小；不存在则输出 0
remote_size() {
  local rf="$1"
  ssh $SSH_OPTS "$USER_HOST" "test -f '${REMOTE_DIR}${rf}' && wc -c < '${REMOTE_DIR}${rf}' || echo 0" 2>/dev/null | tr -d ' \r'
}

shopt -s nullglob
parts=( "${PART_PREFIX}"* )
n="${#parts[@]}"
if [[ "$n" -eq 0 ]]; then
  echo "未生成分卷" >&2
  exit 3
fi

echo "共 ${n} 卷，开始逐卷 scp ..."
i=0
for part in "${parts[@]}"; do
  i=$((i + 1))
  name="$(basename "$part")"
  lsz="$(local_size "$part")"
  rsz="$(remote_size "$name")"
  if [[ "$rsz" == "$lsz" ]]; then
    echo "[$i/$n] 跳过（远程已完整）: $name"
    continue
  fi
  if [[ "$rsz" != "0" && "$rsz" != "$lsz" ]]; then
    echo "[$i/$n] 远程存在不完整文件，先删除: $name"
    ssh $SSH_OPTS "$USER_HOST" "rm -f '${REMOTE_DIR}${name}'"
  fi
  echo "[$i/$n] 上传: $name (${lsz} bytes)"
  scp $SSH_OPTS "$part" "${USER_HOST}:${REMOTE_DIR}"
done

echo
echo "全部卷已上传。请在服务器合并（按分卷名字排序）："
echo "  ssh $USER_HOST \"cd '$REMOTE_DIR' && cat ${BASE}.part_* > '$BASE' && ls -lh '$BASE'\""
echo
echo "校验可选（本机记录的大小）："
echo "  本地: $(local_size "$SRC") 字节"
echo "合并成功后可在服务器删除分卷: rm -f ${REMOTE_DIR}${BASE}.part_*"
