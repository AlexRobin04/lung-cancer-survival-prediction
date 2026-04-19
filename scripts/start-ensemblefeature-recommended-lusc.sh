#!/usr/bin/env bash
# 推荐的 LUSC EnsembleFeature 训练（与单模型对比更公平）：
# - maxEpochs=400（与常见单模型跑满量一致）
# - earlyStopping=false（避免几十 epoch 就停、指标不可比）
# - finetuneEnsemble=true（五基线解冻 + 融合头端到端，通常比仅训融合头更稳）
#
# 用法：在已启动 API 的机器上执行（Docker 网关默认 80 端口同源 /api）
#   export API_BASE=http://127.0.0.1   # 或 http://<服务器IP>
#   bash scripts/start-ensemblefeature-recommended-lusc.sh

set -euo pipefail
API_BASE="${API_BASE:-http://127.0.0.1}"
URL="${API_BASE%/}/api/training/start"

curl -sS -X POST "$URL" \
  -H 'Content-Type: application/json' \
  -d '{
    "cancer": "LUSC",
    "modelType": "EnsembleFeature",
    "mode": "transformer",
    "maxEpochs": 400,
    "learningRate": 1e-5,
    "kFolds": 4,
    "earlyStopping": false,
    "weightDecay": 1e-5,
    "seed": 1,
    "repeat": 1,
    "finetuneEnsemble": true
  }'
echo
