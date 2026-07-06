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
RECORD_ROOT="${REFLECTVPR_RECORD_ROOT:-$OUTPUT_ROOT/_records}"
OPENAI_MODEL="${REFLECTVPR_OPENAI_MODEL:-qwen3-vl-flash}"

GROUP_SPEC="${REFLECTVPR_A6000_GROUPS:-1,2}"
GPU_SPEC="${REFLECTVPR_A6000_GPUS:-2,3}"
PLAN_WORKERS_BY_GROUP="${REFLECTVPR_PLAN_WORKERS_BY_GROUP:-2,1}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$TMP_DIR" "$SERVICE_LOG_DIR" "$RECORD_ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[pc] tmux is required." >&2
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

IFS=',' read -r -a SELECTED_GROUPS <<< "$GROUP_SPEC"
IFS=',' read -r -a PLAN_WORKERS_PER_GROUP <<< "$PLAN_WORKERS_BY_GROUP"

if [ "${#PLAN_WORKERS_PER_GROUP[@]}" -ne "${#SELECTED_GROUPS[@]}" ]; then
  echo "[pc] REFLECTVPR_PLAN_WORKERS_BY_GROUP must contain the same count as REFLECTVPR_A6000_GROUPS." >&2
  echo "[pc] groups=$GROUP_SPEC plan_workers_by_group=$PLAN_WORKERS_BY_GROUP" >&2
  exit 1
fi

for slot in "${!SELECTED_GROUPS[@]}"; do
  idx="${SELECTED_GROUPS[$slot]}"
  if ! [[ "$idx" =~ ^[0-2]$ ]]; then
    echo "[pc] invalid group id: $idx ; valid ids are 0,1,2" >&2
    exit 1
  fi
  plan_workers="${PLAN_WORKERS_PER_GROUP[$slot]}"
  if ! [[ "$plan_workers" =~ ^[0-9]+$ ]] || [ "$plan_workers" -lt 1 ]; then
    echo "[pc] invalid planner worker count for group $idx: $plan_workers" >&2
    exit 1
  fi
  name="${GROUP_NAMES[$idx]}"
  cities="${CITIES[$idx]}"
  sample_num="${SAMPLE_NUMS[$idx]}"
  for worker in $(seq 1 "$plan_workers"); do
    session="reflectvpr_plan_${name}_${worker}"
    log="$LOG_DIR/${name}_plan_${worker}.log"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "[pc] planner already exists: $session"
      continue
    fi
    echo "[pc] starting planner $session cities=$cities"
    tmux new-session -d -s "$session" \
      "cd '$PWD'; \
       export CUDA_VISIBLE_DEVICES=''; \
       export REFLECTVPR_MODE='plan'; \
       export REFLECTVPR_START_SERVICES='0'; \
       export REFLECTVPR_RESUME='1'; \
       export REFLECTVPR_CLEAR_OUTPUT='0'; \
       export OPENAI_TIMEOUT='${OPENAI_TIMEOUT:-45}'; \
       export OPENAI_MAX_RETRIES='${OPENAI_MAX_RETRIES:-0}'; \
       export OPENAI_MODEL_NAME='$OPENAI_MODEL'; \
       export OPENAI_VLM_MODEL='$OPENAI_MODEL'; \
       export REFLECTVPR_VLM_ROUTER_RETRIES='${REFLECTVPR_VLM_ROUTER_RETRIES:-0}'; \
       export REFLECTVPR_RETRY_ROUTER_FAILED='${REFLECTVPR_RETRY_ROUTER_FAILED:-1}'; \
       export REFLECTVPR_SAMPLE_NUM='$sample_num'; \
       export REFLECTVPR_SAMPLE_SEED='${REFLECTVPR_SAMPLE_SEED:-20260529}'; \
       export REFLECTVPR_IMAGE_ROOT='$IMAGE_ROOT'; \
       export REFLECTVPR_OUTPUT_ROOT='$OUTPUT_ROOT'; \
       export REFLECTVPR_RECORD_ROOT='$RECORD_ROOT'; \
       export REFLECTVPR_EXPERIENCE_JSON='$EXPERIENCE_JSON'; \
       export REFLECTVPR_MINE_EXPERIENCE='0'; \
       exec '$PYTHON_BIN' run.py --cities '$cities' >>'$log' 2>&1"
  done
done

echo "[pc] starting GPU generators via run.sh groups=$GROUP_SPEC gpus=$GPU_SPEC"
REFLECTVPR_MODE=generate \
REFLECTVPR_A6000_GROUPS="$GROUP_SPEC" \
REFLECTVPR_A6000_GPUS="$GPU_SPEC" \
REFLECTVPR_RECORD_ROOT="$RECORD_ROOT" \
REFLECTVPR_OUTPUT_ROOT="$OUTPUT_ROOT" \
REFLECTVPR_IMAGE_ROOT="$IMAGE_ROOT" \
REFLECTVPR_EXPERIENCE_JSON="$EXPERIENCE_JSON" \
REFLECTVPR_RUN_ROOT="$RUN_ROOT" \
REFLECTVPR_LOG_DIR="$LOG_DIR" \
REFLECTVPR_TMP_DIR="$TMP_DIR" \
REFLECTVPR_SERVICE_LOG_DIR="$SERVICE_LOG_DIR" \
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-45}" \
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-0}" \
REFLECTVPR_OPENAI_MODEL="$OPENAI_MODEL" \
REFLECTVPR_VLM_ROUTER_RETRIES="${REFLECTVPR_VLM_ROUTER_RETRIES:-0}" \
REFLECTVPR_RETRY_ROUTER_FAILED="${REFLECTVPR_RETRY_ROUTER_FAILED:-1}" \
bash run.sh

cat <<EOF
[pc] submitted producer-consumer jobs.
[pc] planners: reflectvpr_plan_<group>_<worker>
[pc] generators: reflectvpr_a6000_g1/g2 by default
[pc] output: $OUTPUT_ROOT
[pc] records: $RECORD_ROOT
[pc] logs: $LOG_DIR
EOF
