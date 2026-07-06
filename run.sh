#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${REFLECTVPR_PYTHON_BIN:-/media/data/zhangjingyi/vismatch/.venv/bin/python}"
IMAGE_ROOT="${REFLECTVPR_IMAGE_ROOT:-/media/data1/chenshunpeng1/datasets/gsv_cities/Images}"
OUTPUT_ROOT="${REFLECTVPR_OUTPUT_ROOT:-/data_nvme/zhangjingyi/ReflectVPR/output_gsv23_100k}"
EXPERIENCE_JSON="${REFLECTVPR_EXPERIENCE_JSON:-/media/data/zhangjingyi/ReflectVPR/experience_bank_v0.json}"
RUN_ROOT="${REFLECTVPR_RUN_ROOT:-/data_nvme/zhangjingyi/ReflectVPR/runtime_gsv23_100k_a6000}"
LOG_DIR="${REFLECTVPR_LOG_DIR:-$RUN_ROOT/logs}"
TMP_DIR="${REFLECTVPR_TMP_DIR:-$RUN_ROOT/tmp}"
SERVICE_LOG_DIR="${REFLECTVPR_SERVICE_LOG_DIR:-$RUN_ROOT/service_logs}"
OPENAI_MODEL="${REFLECTVPR_OPENAI_MODEL:-qwen3-vl-flash}"

MIN_FREE_GB="${REFLECTVPR_MIN_FREE_GB:-300}"
SERVICE_TIMEOUT="${REFLECTVPR_SERVICE_TIMEOUT:-3600}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$TMP_DIR" "$SERVICE_LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[run] tmux is required for disconnect-safe background runs." >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[run] python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

available_gb=$(df -BG "$OUTPUT_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
if [ "$available_gb" -lt "$MIN_FREE_GB" ]; then
  echo "[run] output filesystem has only ${available_gb}GB free; need at least ${MIN_FREE_GB}GB." >&2
  echo "[run] set REFLECTVPR_OUTPUT_ROOT to a larger SSD path, or lower REFLECTVPR_MIN_FREE_GB if you accept the risk." >&2
  exit 1
fi

GROUP_SPEC="${REFLECTVPR_A6000_GROUPS:-0,1,2}"
IFS=',' read -r -a SELECTED_GROUPS <<< "$GROUP_SPEC"
if [ "${#SELECTED_GROUPS[@]}" -lt 1 ]; then
  echo "[run] REFLECTVPR_A6000_GROUPS must contain at least 1 group id." >&2
  exit 1
fi

GPU_SPEC="${REFLECTVPR_A6000_GPUS:-$GROUP_SPEC}"
IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
if [ "${#GPUS[@]}" -ne "${#SELECTED_GROUPS[@]}" ]; then
  echo "[run] REFLECTVPR_A6000_GPUS must contain the same count as REFLECTVPR_A6000_GROUPS." >&2
  echo "[run] groups=$GROUP_SPEC gpus=$GPU_SPEC" >&2
  exit 1
fi

GROUP_NAMES=(
  "a6000_g0"
  "a6000_g1"
  "a6000_g2"
)

CITIES=(
  "London,Miami,Chicago,Boston,Phoenix,PRS,Melbourne,Lisbon"
  "Rome,Osaka,Minneapolis,Bangkok,TRT,PRG,Barcelona,Madrid"
  "Brussels,MexicoCity,WashingtonDC,OSL,LosAngeles,BuenosAires,Medellin"
)

SAMPLE_NUMS=(
  "34784"
  "34784"
  "30432"
)

ICLIGHT_PORTS=(
  "8102"
  "8112"
  "8122"
)

LIGHTX2V_PORTS=(
  "8101"
  "8111"
  "8121"
)

for slot in "${!SELECTED_GROUPS[@]}"; do
  idx="${SELECTED_GROUPS[$slot]}"
  if ! [[ "$idx" =~ ^[0-2]$ ]]; then
    echo "[run] invalid group id: $idx ; valid ids are 0,1,2" >&2
    exit 1
  fi

  name="${GROUP_NAMES[$idx]}"
  session="reflectvpr_${name}"
  gpu="${GPUS[$slot]}"
  cities="${CITIES[$idx]}"
  sample_num="${SAMPLE_NUMS[$idx]}"
  iclight_port="${ICLIGHT_PORTS[$idx]}"
  lightx2v_port="${LIGHTX2V_PORTS[$idx]}"
  agent_log="$LOG_DIR/${name}_agent.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[run] session already exists: $session"
    echo "      attach: tmux attach -t $session"
    echo "      log:    $agent_log"
    continue
  fi

  echo "[run] starting $session gpu=$gpu cities=$cities sample_num=$sample_num"
  tmux new-session -d -s "$session" \
    "cd '$PWD'; \
     export CUDA_VISIBLE_DEVICES='$gpu'; \
     export ICLIGHT_PORT='$iclight_port'; \
     export LIGHTX2V_PORT='$lightx2v_port'; \
     export ICLIGHT_API_URL='http://127.0.0.1:${iclight_port}/generate'; \
     export LIGHTX2V_API_URL='http://127.0.0.1:${lightx2v_port}/generate'; \
     export REFLECTVPR_FORCE_RESTART='${REFLECTVPR_FORCE_RESTART:-0}'; \
     export REFLECTVPR_MODE='${REFLECTVPR_MODE:-both}'; \
     export REFLECTVPR_CLEAR_OUTPUT='0'; \
     export REFLECTVPR_RESUME='1'; \
     export REFLECTVPR_MAX_ROUNDS='${REFLECTVPR_MAX_ROUNDS:-1}'; \
     export OPENAI_TIMEOUT='${OPENAI_TIMEOUT:-20}'; \
     export OPENAI_MAX_RETRIES='${OPENAI_MAX_RETRIES:-0}'; \
     export OPENAI_MODEL_NAME='$OPENAI_MODEL'; \
     export OPENAI_VLM_MODEL='$OPENAI_MODEL'; \
     export REFLECTVPR_VLM_ROUTER_RETRIES='${REFLECTVPR_VLM_ROUTER_RETRIES:-0}'; \
     export REFLECTVPR_RETRY_ROUTER_FAILED='${REFLECTVPR_RETRY_ROUTER_FAILED:-1}'; \
     export REFLECTVPR_RECORD_ROOT='${REFLECTVPR_RECORD_ROOT:-$OUTPUT_ROOT/_records}'; \
     export REFLECTVPR_AGGREGATE_EVERY='${REFLECTVPR_AGGREGATE_EVERY:-50}'; \
     export REFLECTVPR_GENERATE_IDLE_SLEEP='${REFLECTVPR_GENERATE_IDLE_SLEEP:-5}'; \
     export REFLECTVPR_GENERATE_IDLE_LIMIT='${REFLECTVPR_GENERATE_IDLE_LIMIT:-120}'; \
     export REFLECTVPR_SAMPLE_NUM='$sample_num'; \
     export REFLECTVPR_SAMPLE_SEED='${REFLECTVPR_SAMPLE_SEED:-20260529}'; \
     export REFLECTVPR_IMAGE_ROOT='$IMAGE_ROOT'; \
     export REFLECTVPR_OUTPUT_ROOT='$OUTPUT_ROOT'; \
     export REFLECTVPR_EXPERIENCE_JSON='$EXPERIENCE_JSON'; \
     export REFLECTVPR_TMP_DIR='$TMP_DIR'; \
     export REFLECTVPR_SERVICE_LOG_DIR='$SERVICE_LOG_DIR'; \
     export REFLECTVPR_SERVICE_TIMEOUT='$SERVICE_TIMEOUT'; \
     export REFLECTVPR_CLIP_LOCAL_FILES_ONLY='${REFLECTVPR_CLIP_LOCAL_FILES_ONLY:-1}'; \
     export REFLECTVPR_DISABLE_MOCK='1'; \
     export REFLECTVPR_DISABLE_SERVICE_MOCK='1'; \
     export LIGHTX2V_VRAM_GUARD_FREE_GB='${LIGHTX2V_VRAM_GUARD_FREE_GB:-2}'; \
     export ICLIGHT_AUTO_START='0'; \
     exec '$PYTHON_BIN' run.py --cities '$cities' >>'$agent_log' 2>&1"

  echo "      attach: tmux attach -t $session"
  echo "      log:    $agent_log"
done

cat <<EOF
[run] submitted A6000 jobs.
[run] output: $OUTPUT_ROOT
[run] logs:   $LOG_DIR

Check progress:
  tmux ls
  tail -f $LOG_DIR/a6000_g0_agent.log
  tail -f $LOG_DIR/a6000_g1_agent.log
  tail -f $LOG_DIR/a6000_g2_agent.log

Pause one job:
  tmux send-keys -t reflectvpr_a6000_g0 C-c

Resume:
  bash run.sh
EOF
