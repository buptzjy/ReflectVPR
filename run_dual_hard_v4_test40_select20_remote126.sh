#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/data/zhangjingyi/ReflectVPR"
GPU_ID="${GPU_ID:-1}"

export GPU_ID
export RUN_ROOT="${RUN_ROOT:-$ROOT/tmp/dual_hard_v4_test40_select20_gpu${GPU_ID}}"
export INPUT_SAMPLE_NUM="${INPUT_SAMPLE_NUM:-120}"
export TARGET_IMAGES="${TARGET_IMAGES:-40}"

# Keep generation services hot across repeated tests. Only kill them manually.
export REFLECTVPR_KEEP_SERVICES=1
export REFLECTVPR_FORCE_RESTART="${REFLECTVPR_FORCE_RESTART:-0}"
export REFLECTVPR_DISABLE_TMUX="${REFLECTVPR_DISABLE_TMUX:-0}"
export REFLECTVPR_PREWARM_LIGHTX2V=1
export REFLECTVPR_SERVICE_TIMEOUT="${REFLECTVPR_SERVICE_TIMEOUT:-0}"
export LIGHTX2V_READY_TIMEOUT="${LIGHTX2V_READY_TIMEOUT:-0}"

# Dual-only routing: unsuitable images become skip; suitable images must be dual.
export REFLECTVPR_FORCE_ROUTE=dual
export REFLECTVPR_FORCE_OCCLUSION=vehicle
export REFLECTVPR_TARGET_ROUTE_RATIOS="skip:0,global:0,local:0,dual:1"
export REFLECTVPR_WEATHER_THRESHOLD="${REFLECTVPR_WEATHER_THRESHOLD:-0.55}"
export REFLECTVPR_OCCLUSION_THRESHOLD="${REFLECTVPR_OCCLUSION_THRESHOLD:-0.58}"

# v4 copies the successful pattern: truck/bus + wet reflection band + lower structural cue interruption.
export REFLECTVPR_DUAL_PROMPT_STRATEGY=dual_hard_v4

# Make postprocess use the ImAge env with faiss and avoid sandbox multiprocessing sockets.
export POST_PYTHON="${POST_PYTHON:-/media/data1/zhangjingyi/miniconda3/envs/ImAge/bin/python}"
export REFLECTVPR_IMAGETEST_NUM_WORKERS="${REFLECTVPR_IMAGETEST_NUM_WORKERS:-0}"
export REFLECTVPR_IMAGETEST_GLOBAL_FALLBACK=1

bash "$ROOT/run_strategy_test20_remote126.sh"

"$POST_PYTHON" "$ROOT/scripts/select_top_by_image.py" \
  --output-root "$RUN_ROOT/output" \
  --pair-metrics "$RUN_ROOT/imagetest_distance/pair_metrics.csv" \
  --image-output-dir "$RUN_ROOT/images_top20_passed_by_image" \
  --contact-sheet "$RUN_ROOT/original_vs_generated_labeled_top20_passed_by_image.jpg" \
  --summary-path "$RUN_ROOT/summary_top20_passed_by_image.json" \
  --top-k 20 \
  --passed-only
