#!/usr/bin/env bash
# 从 qcomp 项目根目录启动四个压缩方案的五项全量 batch-64 评测，不重复评测 baseline。
set -euo pipefail
cd /mnt/intern7/xls/projects/qcomp
exec /home/xls/appdata/miniforge3/envs/qwen3-tn/bin/python -u \
  scripts/run_compression_plan.py \
  --selection-json \
    artifacts/layer-selection/20260911T193413/qwen3-8b-mpo-rank96-96layers-20260911T193529.json \
    artifacts/layer-selection/20260911T193413/qwen3-8b-mpo-rank96-96layers-20260911T193540.json \
    artifacts/layer-selection/20260911T193413/qwen3-8b-mpo-rank96-96layers-20260911T193621.json \
    artifacts/layer-selection/20260911T193413/qwen3-8b-mpo-rank96-96layers-20260911T193632.json \
  --eval-config config/compression_evaluation_full.json \
  --model /home/xls/workspace/models/Qwen3-8B \
  --device cuda:0 \
  --eval-batch-size 64 \
  --decomposition-provider tensorly \
  --execution-provider tensorly \
  --decomposition-seed 42 \
  --output artifacts/compression/full-eval-batch64-20260912
