#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/data/zhangjingyi/ReflectVPR"
MODEL_DIR="${QWEN_VLM_MODEL_DIR:-/media/data/zhangjingyi/models/Qwen3-VL-4B-Instruct}"
VLLM_PYTHON="${VLLM_PYTHON:-/media/data1/zhangjingyi/miniconda3/envs/hongyb/bin/python}"
AGENT_PYTHON="${REFLECTVPR_PYTHON_BIN:-/media/data/zhangjingyi/vismatch/.venv/bin/python}"
POST_PYTHON="${POST_PYTHON:-/media/data1/zhangjingyi/miniconda3/bin/python3}"
GPU_ID="${GPU_ID:-3}"
VLLM_PORT="${VLLM_PORT:-23012}"
LIGHTX2V_PORT="${LIGHTX2V_PORT:-8311}"
ICLIGHT_PORT="${ICLIGHT_PORT:-8312}"
RUN_ROOT="${RUN_ROOT:-$ROOT/tmp/strategy_test20}"
OUTPUT_ROOT="$RUN_ROOT/output"
IMAGE_OUTPUT_DIR="$RUN_ROOT/images"
SERVICE_LOG_ROOT="$RUN_ROOT/service_logs"
RECORD_ROOT="$OUTPUT_ROOT/_records"
MAIN_LOG="$RUN_ROOT/run.log"
VLLM_LOG="$RUN_ROOT/vllm.log"
GPU_LOG="$RUN_ROOT/gpu_memory.csv"
SUMMARY_PATH="$RUN_ROOT/summary.json"
CONTACT_SHEET="$RUN_ROOT/original_vs_generated_labeled.jpg"
MODEL_NAME="qwen3-vl-4b-instruct-local"
INPUT_SAMPLE_NUM="${INPUT_SAMPLE_NUM:-36}"
TARGET_IMAGES="${TARGET_IMAGES:-20}"

mkdir -p "$OUTPUT_ROOT" "$IMAGE_OUTPUT_DIR" "$SERVICE_LOG_ROOT" "$RECORD_ROOT"
touch "$MAIN_LOG" "$VLLM_LOG"
exec > >(tee -a "$MAIN_LOG") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

model_ready() {
  "$VLLM_PYTHON" - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = root / "model.safetensors.index.json"
if not index.is_file():
    raise SystemExit(1)
data = json.loads(index.read_text(encoding="utf-8"))
files = sorted(set(data.get("weight_map", {}).values()))
if not files or any(not (root / name).is_file() or (root / name).stat().st_size == 0 for name in files):
    raise SystemExit(1)
PY
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

if ! model_ready; then
  log "Incomplete model weights: $MODEL_DIR"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

log "Starting Qwen3-VL-4B vLLM on physical GPU $GPU_ID"
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.35 \
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
    sleep 0.5
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
export REFLECTVPR_SERVICE_TIMEOUT=1800
export REFLECTVPR_SERVICE_LOG_DIR="$SERVICE_LOG_ROOT"
export REFLECTVPR_TMP_DIR="$RUN_ROOT/service_tmp"
export REFLECTVPR_DISABLE_MOCK=1
export REFLECTVPR_DISABLE_SERVICE_MOCK=1
export REFLECTVPR_DUAL_PROMPT_STRATEGY=test20_v1

export OPENAI_API_KEY=local
export OPENAI_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_MODEL_NAME="$MODEL_NAME"
export OPENAI_VLM_MODEL="$MODEL_NAME"
export OPENAI_TIMEOUT=180
export OPENAI_MAX_RETRIES=1

export REFLECTVPR_SAMPLE_NUM="$INPUT_SAMPLE_NUM"
export REFLECTVPR_SAMPLE_SEED=20260703
export REFLECTVPR_IMAGE_ROOT="/media/data1/chenshunpeng1/datasets/gsv_cities/Images"
export REFLECTVPR_OUTPUT_ROOT="$OUTPUT_ROOT"
export REFLECTVPR_RECORD_ROOT="$RECORD_ROOT"
export REFLECTVPR_EXPERIENCE_JSON="$ROOT/experience_bank_v0.json"
export REFLECTVPR_RESUME=0
export REFLECTVPR_CLEAR_OUTPUT=1
export REFLECTVPR_MAX_ROUNDS=1
export REFLECTVPR_MINE_EXPERIENCE=0
export REFLECTVPR_AGGREGATE_EVERY=10
export REFLECTVPR_CLIP_LOCAL_FILES_ONLY=1

START_EPOCH=$(date +%s)
cd "$ROOT"
log "Planning $INPUT_SAMPLE_NUM images with local Qwen3-VL"
export REFLECTVPR_MODE=plan
export REFLECTVPR_START_SERVICES=0
"$AGENT_PYTHON" -u run.py \
  --cities "London,Miami,Chicago,Boston,Phoenix,PRS,Melbourne,Lisbon"

log "Planning complete; releasing Qwen3-VL GPU memory"
kill "$VLLM_PID"
wait "$VLLM_PID" || true
VLLM_PID=""

log "Starting single-GPU image generation; target_images=$TARGET_IMAGES"
export REFLECTVPR_MODE=generate
export REFLECTVPR_START_SERVICES=1
"$AGENT_PYTHON" -u run.py \
  --cities "London,Miami,Chicago,Boston,Phoenix,PRS,Melbourne,Lisbon"
END_EPOCH=$(date +%s)

"$POST_PYTHON" "$ROOT/scripts/postprocess_strategy20.py" \
  --output-root "$OUTPUT_ROOT" \
  --image-output-dir "$IMAGE_OUTPUT_DIR" \
  --gpu-log "$GPU_LOG" \
  --summary-path "$SUMMARY_PATH" \
  --contact-sheet "$CONTACT_SHEET" \
  --target-images "$TARGET_IMAGES" \
  --elapsed-seconds "$((END_EPOCH - START_EPOCH))"

log "Finished. Images: $IMAGE_OUTPUT_DIR"
log "Summary: $SUMMARY_PATH"
log "Original/generated labeled comparison: $CONTACT_SHEET"
