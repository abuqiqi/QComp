#!/usr/bin/env bash
# 后台全量评测按 GSM8K 敏感度选出的 8/16/32/64/96 层 rank-96 嵌套压缩方案。
set -euo pipefail

cd /home/xls/workspace/projects/qcomp

run_timestamp=$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)
run_root="artifacts/compression/qwen3-8b-gsm8k-comparison/$run_timestamp"
log_path="$run_root/nohup.log"
output_path="$run_root/results"
mkdir -p "$run_root"

nohup /home/xls/appdata/miniforge3/envs/qwen3-tn/bin/python -u \
  scripts/run_compression_plan.py \
  --runtime-config config/runtime-qwen3-8b.toml \
  --eval-config config/compression_evaluation_full.json \
  --selection-json \
    artifacts/compression/qwen3-8b-gsm8k-comparison/qwen3-8b-mpo-rank96-8layers-20260914T150853.json \
    artifacts/compression/qwen3-8b-gsm8k-comparison/qwen3-8b-mpo-rank96-16layers-20260914T150858.json \
    artifacts/compression/qwen3-8b-gsm8k-comparison/qwen3-8b-mpo-rank96-32layers-20260914T150906.json \
    artifacts/compression/qwen3-8b-gsm8k-comparison/qwen3-8b-mpo-rank96-64layers-20260914T150909.json \
    artifacts/compression/qwen3-8b-gsm8k-comparison/qwen3-8b-mpo-rank96-96layers-20260914T150917.json \
  --evaluate-baseline \
  --device cuda:0 \
  --eval-batch-size 64 \
  --decomposition-provider tensorly \
  --execution-provider tensorly \
  --decomposition-seed 42 \
  --output "$output_path" \
  >"$log_path" 2>&1 &

pid=$!
echo "Started GSM8K compression comparison: PID=$pid"
echo "Startup log: $log_path"
echo "Results: $output_path"
