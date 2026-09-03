#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/xls/appdata/miniforge3/envs/qwen3-tn/bin/python"
HF_ROOT="/home/xls/workspace/datasets/huggingface"
CONFIG="$REPO_ROOT/configs/experiments/qwen3_8b_compactifai_standard_eval.json"
OUTPUT_ROOT="$REPO_ROOT/results/evaluations/compactifai-standard"
LOG_PATH="$OUTPUT_ROOT/run.log"
PID_PATH="$OUTPUT_ROOT/run.pid"
STATE_PATH="$OUTPUT_ROOT/state.json"
SUMMARY_PATH="$OUTPUT_ROOT/summary.json"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_ROOT"
export HF_DATASETS_CACHE="$HF_ROOT/datasets"
export HF_HUB_CACHE="$HF_ROOT/hub"
export HF_XET_CACHE="$HF_ROOT/xet"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"

is_running() {
  [[ -f "$PID_PATH" ]] || return 1
  local pid
  pid="$(<"$PID_PATH")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

is_complete() {
  [[ -f "$SUMMARY_PATH" ]] || return 1
  "$PYTHON_BIN" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s.get("complete") is True; assert len(s.get("completed_evaluations", [])) == 15' "$SUMMARY_PATH" >/dev/null 2>&1
}

start_run() {
  if is_running; then
    echo "评测已在后台运行，PID=$(<"$PID_PATH")"
    return 0
  fi
  if is_complete; then
    echo "15 项评测已经完整完成，不会重复执行。"
    echo "结果：$OUTPUT_ROOT/comparison.md"
    return 0
  fi
  touch "$LOG_PATH"
  nohup bash "$SCRIPT_PATH" worker >>"$LOG_PATH" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$PID_PATH"
  echo "后台评测已启动，PID=$pid"
  echo "状态：bash scripts/run_compactifai_standard_eval.sh status"
  echo "日志：bash scripts/run_compactifai_standard_eval.sh tail"
}

show_status() {
  if is_running; then
    echo "运行状态：运行中（PID=$(<"$PID_PATH")）"
  elif [[ -f "$STATE_PATH" ]]; then
    "$PYTHON_BIN" -c 'import json,sys; s=json.load(open(sys.argv[1])); print("运行状态："+str(s.get("status", "未知"))+"；阶段："+str(s.get("stage", "未知")))' "$STATE_PATH"
  else
    echo "运行状态：未开始"
  fi
  if [[ -f "$SUMMARY_PATH" ]]; then
    "$PYTHON_BIN" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(f"已完成：{len(s.get('"'"'completed_evaluations'"'"', []))}/{s.get('"'"'expected_evaluations'"'"', 15)}")' "$SUMMARY_PATH"
  fi
  echo "结果：$OUTPUT_ROOT/comparison.md"
  echo "日志：$LOG_PATH"
  if [[ -f "$LOG_PATH" ]]; then
    echo
    tail -n 20 "$LOG_PATH"
  fi
}

follow_log() {
  touch "$LOG_PATH"
  tail -n 50 -f "$LOG_PATH"
}

worker_run() {
  echo "[$(date --iso-8601=seconds)] 开始 CompactifAI 五任务、三模型标准评测"
  [[ -x "$PYTHON_BIN" ]]
  [[ -d /infini-data/Qwen3-8B ]]
  [[ -f artifacts/decompositions/qwen3-8b-compactifai-d96-blocks18-35/index.json ]]
  [[ -f artifacts/finetuned/qwen3-8b-compactifai-d96-blocks18-35-alpaca-600/final/index.json ]]
  "$PYTHON_BIN" -m qwen3_tn.cli.evaluate_suite "$CONFIG"
  echo "[$(date --iso-8601=seconds)] 全部评测完成：$OUTPUT_ROOT/comparison.md"
}

case "${1:-}" in
  start) start_run ;;
  status) show_status ;;
  tail) follow_log ;;
  worker) worker_run ;;
  *)
    echo "用法：bash scripts/run_compactifai_standard_eval.sh {start|status|tail}" >&2
    exit 2
    ;;
esac
