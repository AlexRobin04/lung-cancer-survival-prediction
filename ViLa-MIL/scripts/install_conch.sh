#!/usr/bin/env bash
# ViLa-MIL 的 ViLa_MIL 模型依赖 Mahmood Lab 的 CONCH Python 包（import conch）。
# 官方说明：https://github.com/mahmoodlab/CONCH —— clone 后 pip install -e .
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONCH_DIR="${CONCH_DIR:-$ROOT/CONCH}"
ENV_NAME="${CONDA_ENV_NAME:-ViLa-MIL}"
PY_BIN="${PY_BIN:-python3}"

cd "$ROOT"
if [[ ! -f "$CONCH_DIR/pyproject.toml" ]]; then
  echo "未找到 CONCH 源码，将克隆到: $CONCH_DIR"
  echo "若 git 报错，可尝试: export GIT_HTTP_VERSION=1.1 后重试，或浏览器下载 zip 解压到该目录。"
  GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-1.1}" git clone --depth 1 https://github.com/mahmoodlab/CONCH.git "$CONCH_DIR"
fi

echo "在 conda 环境 [$ENV_NAME] 中执行: pip install -e $CONCH_DIR"
if command -v conda >/dev/null 2>&1; then
  conda run -n "$ENV_NAME" python -m pip install -U pip
  # 上游 CONCH 有时不支持 PEP660 editable，失败则退化为“直接加入 PYTHONPATH 使用”
  if ! conda run -n "$ENV_NAME" python -m pip install -e "$CONCH_DIR"; then
    echo "pip install -e 失败，将采用 PYTHONPATH 方式使用（训练/推理时会自动追加 CONCH_REPO_PATH）。"
  fi
else
  echo "未检测到 conda，将使用 [$PY_BIN] 安装依赖并采用 PYTHONPATH 方式使用。"
  "$PY_BIN" -m pip install -U pip
fi

# 确保 conch 是可 import 的包（上游 repo 某些版本缺少顶层 __init__.py）
if [[ ! -f "$CONCH_DIR/conch/__init__.py" ]]; then
  cat >"$CONCH_DIR/conch/__init__.py" <<'EOF'
"""
CONCH Python package entrypoint.
"""
EOF
fi

echo "安装 CONCH 运行依赖（transformers/tokenizers/ftfy/regex 等）"
"$PY_BIN" -m pip install -U transformers tokenizers ftfy regex >/dev/null

echo "校验: python -c 'import conch'"
export CONCH_REPO_PATH="$CONCH_DIR"
"$PY_BIN" -c "import sys,os; sys.path.insert(0, os.environ['CONCH_REPO_PATH']); import conch; print('conch OK:', getattr(conch,'__file__','(namespace)'))"
