#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/data/zhangjingyi/ReflectVPR"
AGENT_PYTHON="${REFLECTVPR_PYTHON_BIN:-/media/data/zhangjingyi/vismatch/.venv/bin/python}"
POST_PYTHON="${POST_PYTHON:-/media/data1/zhangjingyi/miniconda3/bin/python3}"
GPU_ID="${GPU_ID:-1}"
LIGHTX2V_PORT="${LIGHTX2V_PORT:-8311}"
ICLIGHT_PORT="${ICLIGHT_PORT:-8312}"
RUN_ROOT="${RUN_ROOT:-$ROOT/tmp/strategy_test20_remote126_gpu${GPU_ID}}"
OUTPUT_ROOT="$RUN_ROOT/output"
IMAGE_OUTPUT_DIR="$RUN_ROOT/images"
SERVICE_LOG_ROOT="$RUN_ROOT/service_logs"
RECORD_ROOT="$OUTPUT_ROOT/_records"
MAIN_LOG="$RUN_ROOT/run.log"
GPU_LOG="$RUN_ROOT/gpu_memory.csv"
SUMMARY_PATH="$RUN_ROOT/summary.json"
CONTACT_SHEET="$RUN_ROOT/original_vs_generated_labeled.jpg"

PLANNER_API_BASE="${REFLECTVPR_PLANNER_API_BASE:-http://10.160.4.126:23002/v1}"
PLANNER_HEALTH_URL="${REFLECTVPR_PLANNER_HEALTH_URL:-http://10.160.4.126:23002/health}"
PLANNER_MODEL="${REFLECTVPR_PLANNER_MODEL:-qwen3-vl-4b-instruct-remote}"

INPUT_SAMPLE_NUM="${INPUT_SAMPLE_NUM:-36}"
TARGET_IMAGES="${TARGET_IMAGES:-20}"

mkdir -p "$OUTPUT_ROOT" "$IMAGE_OUTPUT_DIR" "$SERVICE_LOG_ROOT" "$RECORD_ROOT"
touch "$MAIN_LOG"
exec > >(tee -a "$MAIN_LOG") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
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

MONITOR_PID=""
KEEP_SERVICES="${REFLECTVPR_KEEP_SERVICES:-1}"

cleanup() {
  local rc=$?
  set +e
  if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null
  fi
  if [ "$KEEP_SERVICES" = "1" ]; then
    log "keep services enabled; not killing ports ${LIGHTX2V_PORT}/${ICLIGHT_PORT}"
  else
    fuser -k "${LIGHTX2V_PORT}/tcp" >/dev/null 2>&1
    fuser -k "${ICLIGHT_PORT}/tcp" >/dev/null 2>&1
  fi
  log "cleanup complete, return_code=$rc"
  exit "$rc"
}
trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

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
export ICLIGHT_API_TIMEOUT=1200
export LIGHTX2V_API_TIMEOUT=1200
export LIGHTX2V_API_RETRIES=1
export REFLECTVPR_FORCE_RESTART="${REFLECTVPR_FORCE_RESTART:-0}"
export REFLECTVPR_SERVICE_TIMEOUT="${REFLECTVPR_SERVICE_TIMEOUT:-0}"
export REFLECTVPR_SERVICE_LOG_DIR="$SERVICE_LOG_ROOT"
export REFLECTVPR_TMP_DIR="$RUN_ROOT/service_tmp"
export REFLECTVPR_DISABLE_MOCK=1
export REFLECTVPR_DISABLE_SERVICE_MOCK=1
export REFLECTVPR_DISABLE_TMUX="${REFLECTVPR_DISABLE_TMUX:-0}"
export REFLECTVPR_DUAL_PROMPT_STRATEGY="${REFLECTVPR_DUAL_PROMPT_STRATEGY:-dual_hard_v4}"
export LIGHTX2V_READY_TIMEOUT="${LIGHTX2V_READY_TIMEOUT:-0}"

export REFLECTVPR_PLANNER_API_KEY=local
export REFLECTVPR_PLANNER_API_BASE="$PLANNER_API_BASE"
export REFLECTVPR_PLANNER_MODEL="$PLANNER_MODEL"
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
export REFLECTVPR_GENERATE_IDLE_LIMIT="${REFLECTVPR_GENERATE_IDLE_LIMIT:-720}"
export REFLECTVPR_GENERATE_IDLE_SLEEP="${REFLECTVPR_GENERATE_IDLE_SLEEP:-5}"
export REFLECTVPR_CLIP_LOCAL_FILES_ONLY=1

START_EPOCH=$(date +%s)
cd "$ROOT"

log "Starting persistent generation services on gpu=$GPU_ID while planner runs"
export REFLECTVPR_START_SERVICES=1
bash "$ROOT/start_generation_services.sh"

wait_http "$PLANNER_HEALTH_URL" "Remote planner" 120

log "Starting planner and generator concurrently: planner writes decisions, gpu=$GPU_ID reads ready decisions"
(
  export REFLECTVPR_MODE=plan
  export REFLECTVPR_START_SERVICES=0
  export REFLECTVPR_RESUME=0
  export REFLECTVPR_CLEAR_OUTPUT=1
  "$AGENT_PYTHON" -u run.py \
    --cities "London,Miami,Chicago,Boston,Phoenix,PRS,Melbourne,Lisbon"
) &
PLAN_PID=$!

# Give the planner process a tiny head start to clear/create the run output tree.
sleep 5

set +e
(
  export REFLECTVPR_MODE=generate
  export REFLECTVPR_START_SERVICES=0
  export REFLECTVPR_RESUME=1
  export REFLECTVPR_CLEAR_OUTPUT=0
  "$AGENT_PYTHON" -u run.py \
    --cities "London,Miami,Chicago,Boston,Phoenix,PRS,Melbourne,Lisbon"
)
GENERATE_RC=$?

wait "$PLAN_PID"
PLAN_RC=$?
set -e
if [ "$PLAN_RC" -ne 0 ] || [ "$GENERATE_RC" -ne 0 ]; then
  log "plan/generate failed: plan_rc=$PLAN_RC generate_rc=$GENERATE_RC"
  exit 1
fi
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
