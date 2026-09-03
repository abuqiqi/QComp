#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/xls/appdata/miniforge3/envs/qwen3-tn/bin/python"
HF_ROOT="/home/xls/workspace/datasets/huggingface"
COMPRESSION_CONFIG="$REPO_ROOT/configs/compression/qwen3_8b_compactifai_d96_blocks18_35.json"
EXPERIMENT_CONFIG="$REPO_ROOT/configs/experiments/qwen3_8b_compactifai_d96_blocks18_35_alpaca_600.json"
MODULE_SET="$REPO_ROOT/artifacts/decompositions/qwen3-8b-compactifai-d96-blocks18-35/index.json"
ARTIFACT_ROOT="$REPO_ROOT/artifacts/finetuned/qwen3-8b-compactifai-d96-blocks18-35-alpaca-600"
RUN_JSON="$ARTIFACT_ROOT/run.json"
FINAL_INDEX="$ARTIFACT_ROOT/final/index.json"
RUNTIME_DIR="$REPO_ROOT/results/runs/compactifai-alpaca-native"
LOG_PATH="$RUNTIME_DIR/run.log"
PID_PATH="$RUNTIME_DIR/run.pid"
STATE_PATH="$RUNTIME_DIR/state.json"
DECOMPOSITION_JSON="$RUNTIME_DIR/decomposition-result.json"
VERIFICATION_JSON="$RUNTIME_DIR/offline-verification.json"
REPORT_PATH="$REPO_ROOT/reports/weekly/2026-08-27-compactifai-alpaca-native.md"
PHASE="未启动"

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

mkdir -p "$RUNTIME_DIR"
cd "$REPO_ROOT"

report_update() {
  local status="$1"
  local stage="$2"
  shift 2
  "$PYTHON_BIN" -m qwen3_tn.cli.report \
    --report "$REPORT_PATH" \
    --state-file "$STATE_PATH" \
    --status "$status" \
    --stage "$stage" \
    --artifact-root "$ARTIFACT_ROOT" \
    --log-path "$LOG_PATH" \
    --run-json "$RUN_JSON" \
    --decomposition-json "$DECOMPOSITION_JSON" \
    --verification-json "$VERIFICATION_JSON" \
    "$@"
}

is_running() {
  [[ -f "$PID_PATH" ]] || return 1
  local pid
  pid="$(<"$PID_PATH")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

is_complete() {
  [[ -f "$RUN_JSON" && -f "$FINAL_INDEX" && -f "$VERIFICATION_JSON" ]] || return 1
  "$PYTHON_BIN" -c 'import json,sys; from qwen3_tn.checkpoint import load_module_set_index; run=json.load(open(sys.argv[1])); verification=json.load(open(sys.argv[3])); _,modules=load_module_set_index(sys.argv[2]); assert run["training_metrics"]["global_step"]==600; assert len(modules.modules)==108; assert verification.get("success") is True; assert verification.get("module_count")==108' \
    "$RUN_JSON" "$FINAL_INDEX" "$VERIFICATION_JSON" \
    >/dev/null 2>&1
}

start_run() {
  if is_running; then
    echo "实验已在后台运行，PID=$(<"$PID_PATH")"
    echo "日志：$LOG_PATH"
    return 0
  fi
  if is_complete; then
    report_update success "已完成；没有重复启动"
    echo "实验已经完整完成，不会重复执行。"
    echo "报告：$REPORT_PATH"
    return 0
  fi
  touch "$LOG_PATH"
  local start_time
  start_time="$(date --iso-8601=seconds)"
  nohup bash "$SCRIPT_PATH" worker "$start_time" >>"$LOG_PATH" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$PID_PATH"
  echo "后台任务已启动，PID=$pid"
  echo "查看状态：bash scripts/run_compactifai_alpaca_native.sh status"
  echo "持续看日志：bash scripts/run_compactifai_alpaca_native.sh tail"
}

show_status() {
  if is_running; then
    local pid
    pid="$(<"$PID_PATH")"
    echo "运行状态：运行中（PID=$pid）"
  elif [[ -f "$STATE_PATH" ]]; then
    if "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "running" else 1)' "$STATE_PATH"; then
      report_update failure "后台进程已经退出；自动修正陈旧状态"
    fi
    "$PYTHON_BIN" -c 'import json,sys; s=json.load(open(sys.argv[1])); print("运行状态："+str(s.get("status","未知"))+"；阶段："+str(s.get("stage","未知"))); print("开始："+str(s.get("start_time","尚无"))+"；结束："+str(s.get("end_time","尚无")))' "$STATE_PATH"
  else
    echo "运行状态：未开始"
  fi
  echo "报告：$REPORT_PATH"
  echo "日志：$LOG_PATH"
  if [[ -f "$LOG_PATH" ]]; then
    echo
    echo "最近 20 行日志："
    tail -n 20 "$LOG_PATH"
  fi
}

follow_log() {
  touch "$LOG_PATH"
  tail -n 50 -f "$LOG_PATH"
}

worker_run() {
  local start_time="${1:-$(date --iso-8601=seconds)}"
  PHASE="环境检查"
  trap 'code=$?; trap - EXIT; if [[ $code -ne 0 ]]; then report_update failure "$PHASE" || true; echo "[$(date --iso-8601=seconds)] 失败阶段：$PHASE，退出码：$code"; fi; exit "$code"' EXIT

  echo "[$(date --iso-8601=seconds)] 开始 CompactifAI 风格 Alpaca healing"
  report_update running "$PHASE" --start-time "$start_time"

  [[ -x "$PYTHON_BIN" ]]
  [[ -d /infini-data/Qwen3-8B ]]
  "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); assert p.total_memory < 80*1024**3 + 1024**3; print(f"GPU: {p.name}, {p.total_memory/1024**3:.1f} GiB")'
  "$PYTHON_BIN" -c 'from datasets import load_dataset; d=load_dataset("tatsu-lab/alpaca", split="train"); assert len(d)==52002, len(d); assert {"instruction","input","output","text"} <= set(d.column_names); print(f"Alpaca offline validation: {len(d):,} rows, columns={d.column_names}")'

  PHASE="TT 分解"
  report_update running "$PHASE"
  if [[ -f "$MODULE_SET" ]]; then
    "$PYTHON_BIN" -c 'import sys; from qwen3_tn.checkpoint import load_module_set_index; _,m=load_module_set_index(sys.argv[1]); assert len(m.modules)==108, len(m.modules); print("已有 108-module 分解产物校验通过，跳过分解。")' "$MODULE_SET"
  else
    local decomp_tmp="$RUNTIME_DIR/.decomposition-result.tmp.json"
    "$PYTHON_BIN" -m qwen3_tn.cli.decompose "$COMPRESSION_CONFIG" >"$decomp_tmp"
    "$PYTHON_BIN" -c 'import json,sys; value=json.load(open(sys.argv[1])); aggregate=value["aggregate"]; assert aggregate["target_count"]==108; assert aggregate["dense_parameters"]==2566914048; assert aggregate["tt_parameters"]==218972160' "$decomp_tmp"
    mv "$decomp_tmp" "$DECOMPOSITION_JSON"
  fi

  PHASE="Alpaca TT-only healing"
  report_update running "$PHASE"
  "$PYTHON_BIN" -m qwen3_tn.cli.finetune "$EXPERIMENT_CONFIG"
  [[ -f "$RUN_JSON" && -f "$FINAL_INDEX" ]]

  PHASE="最终模型离线重载和生成"
  report_update running "$PHASE"
  "$PYTHON_BIN" -m qwen3_tn.cli.verify \
    --model-path /infini-data/Qwen3-8B \
    --module-set "$FINAL_INDEX" \
    --output "$VERIFICATION_JSON" \
    --prompt "请用两句话解释张量网络压缩。" \
    --max-new-tokens 16

  PHASE="全部完成"
  report_update success "$PHASE"
  trap - EXIT
  echo "[$(date --iso-8601=seconds)] 实验成功，报告已更新：$REPORT_PATH"
}

case "${1:-}" in
  start)
    start_run
    ;;
  status)
    show_status
    ;;
  tail)
    follow_log
    ;;
  worker)
    shift
    worker_run "$@"
    ;;
  *)
    echo "用法：bash scripts/run_compactifai_alpaca_native.sh {start|status|tail}" >&2
    exit 2
    ;;
esac

