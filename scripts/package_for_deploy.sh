#!/usr/bin/env bash
# 在仓库根（与 docker-compose.yml 同级）的上一级目录生成 tar.gz，便于 scp 到服务器后解压得到同名目录。
# 用法：
#   ./scripts/package_for_deploy.sh                    # 输出到 ~/Desktop，默认 slim
#   ./scripts/package_for_deploy.sh /path/to/outdir    # 指定输出目录
#   SLIM=0 ./scripts/package_for_deploy.sh             # 完整包（含 result/api_runs，体积可能 >10GB）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"
OUT_DIR="${1:-$HOME/Desktop}"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_FILE="${OUT_DIR%/}/${NAME}-deploy-${STAMP}.tar.gz"
SLIM="${SLIM:-1}"

EXCLUDES=(
  --exclude="${NAME}/.git"
  --exclude="${NAME}/.cursor"
  --exclude="${NAME}/vila-mil-frontend/node_modules"
  --exclude="${NAME}/*/__pycache__"
  --exclude="${NAME}/*/*/__pycache__"
  --exclude="${NAME}/*/*/*/__pycache__"
  --exclude="${NAME}/*.pyc"
)

if [[ "$SLIM" == "1" ]]; then
  EXCLUDES+=(
    --exclude="${NAME}/ViLa-MIL/result/api_runs"
    --exclude="${NAME}/ViLa-MIL/api_training_logs"
  )
  echo "[slim] 将排除 ViLa-MIL/result/api_runs 与 ViLa-MIL/api_training_logs（仅中间产物）。"
  echo "       若需一并打包历史训练目录，请执行: SLIM=0 $0 $*"
else
  echo "[full] 包含 result/api_runs 与 api_training_logs，压缩包可能非常大。"
fi

echo "打包目录: $ROOT"
echo "输出文件: $OUT_FILE"

tar -czf "$OUT_FILE" -C "$PARENT" "${EXCLUDES[@]}" "$NAME"

ls -lh "$OUT_FILE"
echo "完成。上传到服务器示例:"
echo "  scp \"$OUT_FILE\" root@8.130.211.90:/root/"
echo "解压:"
echo "  ssh root@8.130.211.90 'cd /root && tar -xzf $(basename "$OUT_FILE")'"
