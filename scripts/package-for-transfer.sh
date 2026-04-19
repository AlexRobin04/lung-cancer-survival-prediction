#!/usr/bin/env bash
# 构建 Docker 镜像并导出离线包 + 项目源码归档，便于拷贝到另一台机器（如 Mac）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-$ROOT/transfer}"
mkdir -p "$OUT"

STAMP="$(date +%Y%m%d-%H%M%S)"
IMAGES_TAR="$OUT/vila-mil-docker-images-${STAMP}.tar"
SRC_TGZ="$OUT/vila-mil-project-${STAMP}.tar.gz"
README_TXT="$OUT/README-导入说明-${STAMP}.txt"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "==> 构建后端基础镜像 vila-mil-backend-base:local"
  docker build -f "$ROOT/ViLa-MIL/Dockerfile.base" -t vila-mil-backend-base:local "$ROOT/ViLa-MIL"

  echo "==> docker compose build"
  docker compose -f "$ROOT/docker-compose.yml" build
else
  echo "==> SKIP_BUILD=1，跳过镜像构建（使用本机已有镜像）"
fi

echo "==> 导出镜像到 $IMAGES_TAR（体积可能较大，请耐心等待）"
docker save \
  vila-mil-backend-base:local \
  vila-mil-backend \
  vila-mil-frontend \
  nginx:1.27-alpine \
  -o "$IMAGES_TAR"

echo "==> 打包项目源码与数据目录到 $SRC_TGZ（排除 node_modules、__pycache__、.git、transfer）"
# 输出文件不能位于被打包的目录树内；transfer 内为导出物，也不打进源码包以免重复与自引用
PROJ_BASE="$(basename "$ROOT")"
PARENT="$(dirname "$ROOT")"
SRC_TGZ_TMP="$(mktemp "/tmp/${PROJ_BASE}-project-XXXXXX.tar.gz")"
tar -czf "$SRC_TGZ_TMP" \
  --exclude="${PROJ_BASE}/vila-mil-frontend/node_modules" \
  --exclude='__pycache__' \
  --exclude="${PROJ_BASE}/.git" \
  --exclude="${PROJ_BASE}/transfer" \
  -C "$PARENT" \
  "$PROJ_BASE"
mv -f "$SRC_TGZ_TMP" "$SRC_TGZ"

cat > "$README_TXT" << EOF
ViLa-MIL 离线包（生成于 ${STAMP}）

一、在本机（如 Mac）先放好两个文件
  1) $(basename "$IMAGES_TAR")
  2) $(basename "$SRC_TGZ")

二、解压源码
  mkdir -p "/Users/zzfly/毕设"
  tar -xzf $(basename "$SRC_TGZ") -C "/Users/zzfly/毕设"
  （若路径不同，把 /Users/zzfly/毕设 换成你的目录）

三、导入镜像
  docker load -i $(basename "$IMAGES_TAR")

四、进入项目根目录启动
  cd "/Users/zzfly/毕设/$(basename "$ROOT")"
  docker compose up -d

浏览器访问 http://localhost （需本机 80 端口未被占用）

从服务器下载到 Mac（在 Mac 终端执行，把 SERVER 换成你的服务器地址）:
  scp USER@SERVER:$IMAGES_TAR "/Users/zzfly/毕设/"
  scp USER@SERVER:$SRC_TGZ "/Users/zzfly/毕设/"
  scp USER@SERVER:$README_TXT "/Users/zzfly/毕设/"
EOF

ls -lh "$IMAGES_TAR" "$SRC_TGZ" "$README_TXT"
echo "完成。请查看 $README_TXT"
