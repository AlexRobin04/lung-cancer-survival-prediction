#!/usr/bin/env bash
# LUSC 决策级集成（EnsembleDecision）：五路基线需已训练；融合不训练，仅评估并写 checkpoint。
set -euo pipefail
API="${VILAMIL_API:-http://127.0.0.1/api}"
curl -sS -X POST "${API}/training/start" \
  -H 'Content-Type: application/json' \
  -d '{
    "cancer": "LUSC",
    "modelType": "EnsembleDecision",
    "mode": "transformer",
    "maxEpochs": 1,
    "learningRate": 1e-5,
    "kFolds": 4,
    "weightDecay": 1e-5,
    "earlyStopping": false,
    "seed": 1,
    "repeat": 1,
    "decisionFusion": "weighted",
    "ensembleBranchPriorAuto": true,
    "ensembleBranchPriorTemperature": 0.55
  }' | python3 -m json.tool
