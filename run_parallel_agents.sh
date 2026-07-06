#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE_ROOT=${REFLECTVPR_IMAGE_ROOT:-/media/data1/chenshunpeng1/datasets/gsv_cities/Images}
OUTPUT_ROOT=${REFLECTVPR_OUTPUT_ROOT:-/media/data/zhangjingyi/ReflectVPR/output_parallel}
EXPERIENCE_JSON=${REFLECTVPR_EXPERIENCE_JSON:-/media/data/zhangjingyi/ReflectVPR/experience_bank_v0.json}
PER_CITY_SAMPLE=${REFLECTVPR_PER_CITY_SAMPLE:-50}
GPU_LIST=${REFLECTVPR_GPU_LIST:-0,1,2,3}
CITY_GROUP_SPEC=${REFLECTVPR_PARALLEL_GROUPS:-London Phoenix Osaka PRS}
ICLIGHT_BASE_PORT=${REFLECTVPR_ICLIGHT_BASE_PORT:-8102}
LIGHTX2V_BASE_PORT=${REFLECTVPR_LIGHTX2V_BASE_PORT:-8101}
LOG_DIR=${REFLECTVPR_PARALLEL_LOG_DIR:-$OUTPUT_ROOT/parallel_logs}

mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
read -r -a CITY_GROUPS <<< "$CITY_GROUP_SPEC"

if [ "${#GPUS[@]}" -lt "${#CITY_GROUPS[@]}" ]; then
  echo "[parallel] not enough GPUs: groups=${#CITY_GROUPS[@]} gpus=${#GPUS[@]}" >&2
  exit 1
fi

pids=()
for idx in "${!CITY_GROUPS[@]}"; do
  gpu="${GPUS[$idx]}"
  cities="${CITY_GROUPS[$idx]}"
  city_count=$(awk -F, '{print NF}' <<< "$cities")
  sample_num=$((PER_CITY_SAMPLE * city_count))
  iclight_port=$((ICLIGHT_BASE_PORT + idx * 10))
  lightx2v_port=$((LIGHTX2V_BASE_PORT + idx * 10))
  log_file="$LOG_DIR/agent_${idx}_gpu${gpu}_${cities//,/+}.log"

  echo "[parallel] gpu=$gpu cities=$cities sample_num=$sample_num iclight=$iclight_port lightx2v=$lightx2v_port log=$log_file"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export ICLIGHT_PORT="$iclight_port"
    export LIGHTX2V_PORT="$lightx2v_port"
    export ICLIGHT_API_URL="http://127.0.0.1:${iclight_port}/generate"
    export LIGHTX2V_API_URL="http://127.0.0.1:${lightx2v_port}/generate"
    export REFLECTVPR_FORCE_RESTART="${REFLECTVPR_FORCE_RESTART:-0}"
    export REFLECTVPR_CLEAR_OUTPUT=0
    export REFLECTVPR_MAX_ROUNDS="${REFLECTVPR_MAX_ROUNDS:-1}"
    export REFLECTVPR_SAMPLE_NUM="$sample_num"
    export REFLECTVPR_IMAGE_ROOT="$IMAGE_ROOT"
    export REFLECTVPR_OUTPUT_ROOT="$OUTPUT_ROOT"
    export REFLECTVPR_EXPERIENCE_JSON="$EXPERIENCE_JSON"
    export REFLECTVPR_CLIP_LOCAL_FILES_ONLY="${REFLECTVPR_CLIP_LOCAL_FILES_ONLY:-1}"
    export ICLIGHT_AUTO_START=0
    python run.py --cities "$cities"
  ) > "$log_file" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "[parallel] one or more agents failed; check $LOG_DIR" >&2
  exit 1
fi

echo "[parallel] done output=$OUTPUT_ROOT logs=$LOG_DIR"
