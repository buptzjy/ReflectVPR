#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/data/zhangjingyi/ReflectVPR"
MODEL_DIR="/media/data1/zhangjingyi/.cache/modelscope/Qwen/Qwen3-VL-8B-Instruct"
VLLM_PYTHON="/media/data1/zhangjingyi/miniconda3/envs/hongyb/bin/python"
AGENT_PYTHON="/media/data/zhangjingyi/vismatch/.venv/bin/python"
GPU_ID="${GPU_ID:-1}"
VLLM_PORT="${VLLM_PORT:-22002}"
LIGHTX2V_PORT="${LIGHTX2V_PORT:-8201}"
ICLIGHT_PORT="${ICLIGHT_PORT:-8202}"
RUN_ROOT="${RUN_ROOT:-/data_nvme/zhangjingyi/ReflectVPR/test_qwen3vl8b_nf4_100}"
OUTPUT_ROOT="$RUN_ROOT/output"
TMP_ROOT="$RUN_ROOT/tmp"
SERVICE_LOG_ROOT="$RUN_ROOT/service_logs"
RECORD_ROOT="$OUTPUT_ROOT/_records"
MAIN_LOG="$RUN_ROOT/run.log"
VLLM_LOG="$RUN_ROOT/vllm.log"
GPU_LOG="$RUN_ROOT/gpu_memory.csv"
MODEL_NAME="qwen3-vl-8b-instruct-nf4"

mkdir -p "$OUTPUT_ROOT" "$TMP_ROOT" "$SERVICE_LOG_ROOT" "$RECORD_ROOT"
touch "$MAIN_LOG" "$VLLM_LOG"

exec > >(tee -a "$MAIN_LOG") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

model_ready() {
  local count
  count=$(find "$MODEL_DIR" -maxdepth 1 -type f -name 'model-*-of-00004.safetensors' | wc -l)
  [ "$count" -eq 4 ] && [ -s "$MODEL_DIR/model.safetensors.index.json" ]
}

wait_http() {
  local url="$1"
  local label="$2"
  local timeout="$3"
  local start
  start=$(date +%s)
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "$label ready: $url"
      return 0
    fi
    if [ $(( $(date +%s) - start )) -ge "$timeout" ]; then
      log "$label timed out after ${timeout}s"
      return 1
    fi
    sleep 5
  done
}

VLLM_PID=""
MONITOR_PID=""

cleanup() {
  local rc=$?
  set +e
  if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null
  fi
  if [ -n "$VLLM_PID" ]; then
    kill "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null
  fi
  fuser -k "${LIGHTX2V_PORT}/tcp" >/dev/null 2>&1
  fuser -k "${ICLIGHT_PORT}/tcp" >/dev/null 2>&1
  log "cleanup complete, return_code=$rc"
  exit "$rc"
}
trap cleanup EXIT INT TERM

log "Waiting for complete model weights in $MODEL_DIR"
while ! model_ready; do
  du -sh "$MODEL_DIR" 2>/dev/null || true
  sleep 60
done
log "Model weights complete"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

log "Starting vLLM NF4 service on physical GPU $GPU_ID"
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --dtype bfloat16 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --gpu-memory-utilization 0.22 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1}' \
  --enforce-eager \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

wait_http "http://127.0.0.1:${VLLM_PORT}/health" "vLLM" 1800

echo "timestamp,memory_used_mib,gpu_util_pct" >"$GPU_LOG"
(
  while true; do
    values=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID")
    echo "$(date '+%Y-%m-%d %H:%M:%S.%3N'),$values"
    sleep 0.2
  done
) >>"$GPU_LOG" &
MONITOR_PID=$!

export LIGHTX2V_VRAM_TARGET_GB=0
export ICLIGHT_VRAM_TARGET_GB=0
export ICLIGHT_PORT
export LIGHTX2V_PORT
export ICLIGHT_API_URL="http://127.0.0.1:${ICLIGHT_PORT}/generate"
export LIGHTX2V_API_URL="http://127.0.0.1:${LIGHTX2V_PORT}/generate"
export REFLECTVPR_FORCE_RESTART=1
export REFLECTVPR_START_SERVICES=1
export REFLECTVPR_SERVICE_TIMEOUT=1800
export REFLECTVPR_SERVICE_LOG_DIR="$SERVICE_LOG_ROOT"
export REFLECTVPR_TMP_DIR="$TMP_ROOT"
export REFLECTVPR_DISABLE_MOCK=1
export REFLECTVPR_DISABLE_SERVICE_MOCK=1

export OPENAI_API_KEY=local
export OPENAI_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_MODEL_NAME="$MODEL_NAME"
export OPENAI_VLM_MODEL="$MODEL_NAME"
export OPENAI_TIMEOUT=180
export OPENAI_MAX_RETRIES=1

export REFLECTVPR_MODE=both
export REFLECTVPR_SAMPLE_NUM=100
export REFLECTVPR_SAMPLE_SEED=20260622
export REFLECTVPR_IMAGE_ROOT="/media/data1/chenshunpeng1/datasets/gsv_cities/Images"
export REFLECTVPR_OUTPUT_ROOT="$OUTPUT_ROOT"
export REFLECTVPR_RECORD_ROOT="$RECORD_ROOT"
export REFLECTVPR_EXPERIENCE_JSON="$ROOT/experience_bank_v0.json"
export REFLECTVPR_RESUME=1
export REFLECTVPR_CLEAR_OUTPUT=0
export REFLECTVPR_MAX_ROUNDS=1
export REFLECTVPR_MINE_EXPERIENCE=0
export REFLECTVPR_AGGREGATE_EVERY=10
export REFLECTVPR_CLIP_LOCAL_FILES_ONLY=1

START_EPOCH=$(date +%s)
log "Starting isolated 100-image ReflectVPR test"
cd "$ROOT"
"$AGENT_PYTHON" -u run.py \
  --cities "London,Miami,Chicago,Boston,Phoenix,PRS,Melbourne,Lisbon"
END_EPOCH=$(date +%s)
log "ReflectVPR finished, elapsed_seconds=$((END_EPOCH - START_EPOCH))"

"$AGENT_PYTHON" - "$OUTPUT_ROOT" "$GPU_LOG" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

output_root = Path(sys.argv[1])
gpu_log = Path(sys.argv[2])
records = []
for path in output_root.glob("*/reflect.json"):
    try:
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass

routes = Counter(str(r.get("route", "unknown")) for r in records)
passed = sum(bool(r.get("passed")) for r in records)
failed_generation = sum(bool(r.get("generation_failed")) for r in records)
router_failed = sum(bool(r.get("router_failed")) for r in records)
rounds = Counter(int(r.get("rounds_used", 0) or 0) for r in records)

memory = []
with gpu_log.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        try:
            memory.append(int(row["memory_used_mib"].strip()))
        except Exception:
            pass

summary = {
    "records": len(records),
    "passed": passed,
    "pass_rate": passed / len(records) if records else 0.0,
    "failed_generation": failed_generation,
    "router_failed": router_failed,
    "routes": dict(routes),
    "rounds_used": dict(rounds),
    "gpu_memory_min_mib": min(memory) if memory else None,
    "gpu_memory_max_mib": max(memory) if memory else None,
}
summary_path = output_root.parent / "summary_100.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"summary_path={summary_path}")
PY
