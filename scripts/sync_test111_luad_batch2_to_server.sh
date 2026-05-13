#!/usr/bin/env bash
# 将 Desktop/test111 中「第二批 20 张」TCGA-LUAD SVS（见 tcga_luad_batch2_20slides_manifest.json）
# 同步到远程：/root/vila-mil/ViLa-MIL/test_LUAD_WSI/
#
# 用法：
#   ./scripts/sync_test111_luad_batch2_to_server.sh
#   SRC_DIR=/path/to/test111 REMOTE=root@1.2.3.4 ./scripts/sync_test111_luad_batch2_to_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${SRC_DIR:-$HOME/Desktop/test111}"
MANIFEST="${SRC_DIR}/tcga_luad_batch2_20slides_manifest.json"
REMOTE="${REMOTE:-root@8.130.211.90}"
REMOTE_DIR="/root/vila-mil/ViLa-MIL/test_LUAD_WSI"

SSH_BASE="-o ServerAliveInterval=15 -o ServerAliveCountMax=120 -o TCPKeepAlive=yes"
SSH_OPTS="${SSH_OPTS:-$SSH_BASE}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "缺少清单: $MANIFEST" >&2
  exit 1
fi

FILES=()
while IFS= read -r line; do
  [[ -n "$line" ]] && FILES+=("$line")
done < <(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
for it in m.get("files", []):
    print(it["file_name"])
' "$MANIFEST")

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "清单中无文件条目" >&2
  exit 1
fi

missing=0
for f in "${FILES[@]}"; do
  if [[ ! -f "${SRC_DIR}/${f}" ]]; then
    echo "本地缺少: ${SRC_DIR}/${f}" >&2
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || exit 2

echo "远程创建目录: ${REMOTE}:${REMOTE_DIR}"
ssh $SSH_OPTS "$REMOTE" "mkdir -p '${REMOTE_DIR}'"

echo "共 ${#FILES[@]} 个文件，开始 rsync（中断可重跑续传）..."
cd "$SRC_DIR"
rsync -avhP --partial \
  -e "ssh $SSH_OPTS" \
  "${FILES[@]}" \
  "${REMOTE}:${REMOTE_DIR}/"

echo "完成。可在服务器检查:"
echo "  ssh $SSH_OPTS $REMOTE 'ls -lh ${REMOTE_DIR}'"
