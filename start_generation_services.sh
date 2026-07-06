#!/bin/bash
set -euo pipefail

LOG_DIR=${REFLECTVPR_SERVICE_LOG_DIR:-/media/data/zhangjingyi/ReflectVPR/tmp/logs}
TMP_DIR=${REFLECTVPR_TMP_DIR:-/media/data/zhangjingyi/ReflectVPR/tmp}
HEALTHCHECK_PYTHON=${REFLECTVPR_HEALTHCHECK_PYTHON:-/media/data1/zhangjingyi/miniconda3/envs/hongyb/bin/python}
ICLIGHT_WORKDIR=${ICLIGHT_WORKDIR:-/media/data/zhangjingyi/IC-Light}
ICLIGHT_PYTHON=${ICLIGHT_PYTHON:-/media/data1/zhangjingyi/miniconda3/envs/hongyb/bin/python}
LIGHTX2V_WORKDIR=${LIGHTX2V_WORKDIR:-/media/data/zhangjingyi/LightX2V}
LIGHTX2V_PYTHON=${LIGHTX2V_PYTHON:-/media/data1/zhangjingyi/miniconda3/envs/lightx2v/bin/python}
mkdir -p "$LOG_DIR"
mkdir -p "$TMP_DIR"

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export REFLECTVPR_DISABLE_SERVICE_MOCK=${REFLECTVPR_DISABLE_SERVICE_MOCK:-1}

is_port_open() {
  local port="$1"
  "$HEALTHCHECK_PYTHON" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

start_service() {
  local name="$1"
  local port="$2"
  local workdir="$3"
  local python_bin="$4"
  local log_file="$LOG_DIR/${name}_${port}.log"
  local cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"
  local port_env=""
  local lightx2v_guard_free_gb="${LIGHTX2V_VRAM_GUARD_FREE_GB:-0}"

  if [ "$name" = "iclight" ]; then
    port_env="export ICLIGHT_PORT='$port';"
  elif [ "$name" = "lightx2v" ]; then
    port_env="export LIGHTX2V_PORT='$port';"
  fi

  if [ "${REFLECTVPR_FORCE_RESTART:-0}" != "1" ] && is_port_open "$port"; then
    echo "[$name] port $port already open"
    return 0
  fi
  if [ "${REFLECTVPR_FORCE_RESTART:-0}" = "1" ] && is_port_open "$port"; then
    echo "[$name] stopping existing process on port $port"
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    sleep 2
  fi

  echo "[$name] starting on port $port"
  if [ "${REFLECTVPR_DISABLE_TMUX:-0}" != "1" ] && command -v tmux >/dev/null 2>&1; then
    local session="reflectvpr_${name}_${port}"
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
    fi
    tmux new-session -d -s "$session" \
      "cd '$workdir'; export PLATFORM=cuda; export SKIP_PLATFORM_CHECK=True; export CUDA_VISIBLE_DEVICES='$cuda_devices'; $port_env export LIGHTX2V_VRAM_GUARD_FREE_GB='$lightx2v_guard_free_gb'; export REFLECTVPR_TMP_DIR='$TMP_DIR'; export TMPDIR='$TMP_DIR'; export ICLIGHT_OUTPUT_DIR='$TMP_DIR/service_outputs/iclight'; export LIGHTX2V_OUTPUT_DIR='$TMP_DIR/service_outputs/lightx2v'; export REFLECTVPR_DISABLE_SERVICE_MOCK=\${REFLECTVPR_DISABLE_SERVICE_MOCK:-1}; exec '$python_bin' app.py >>'$log_file' 2>&1"
    echo "[$name] tmux=$session log=$log_file"
  else
    nohup bash -c '
    cd "$1"
    export PLATFORM=cuda
    export SKIP_PLATFORM_CHECK=True
    export CUDA_VISIBLE_DEVICES="$3"
    if [ "$5" = "iclight" ]; then
      export ICLIGHT_PORT="$6"
    elif [ "$5" = "lightx2v" ]; then
      export LIGHTX2V_PORT="$6"
    fi
    export LIGHTX2V_VRAM_GUARD_FREE_GB="$7"
    export REFLECTVPR_TMP_DIR="$4"
    export TMPDIR="$4"
    export ICLIGHT_OUTPUT_DIR="$4/service_outputs/iclight"
    export LIGHTX2V_OUTPUT_DIR="$4/service_outputs/lightx2v"
    export REFLECTVPR_DISABLE_SERVICE_MOCK="${REFLECTVPR_DISABLE_SERVICE_MOCK:-1}"
    exec "$2" app.py
    ' _ "$workdir" "$python_bin" "$cuda_devices" "$TMP_DIR" "$name" "$port" "$lightx2v_guard_free_gb" >>"$log_file" 2>&1 &
    echo "[$name] pid=$! log=$log_file"
  fi
}

start_service \
  "iclight" \
  "${ICLIGHT_PORT:-8002}" \
  "$ICLIGHT_WORKDIR" \
  "$ICLIGHT_PYTHON"

start_service \
  "lightx2v" \
  "${LIGHTX2V_PORT:-8001}" \
  "$LIGHTX2V_WORKDIR" \
  "$LIGHTX2V_PYTHON"

if [ "${REFLECTVPR_PREWARM_LIGHTX2V:-1}" = "1" ]; then
  (
    port="${LIGHTX2V_PORT:-8001}"
    timeout="${REFLECTVPR_SERVICE_TIMEOUT:-1800}"
    start_ts=$(date +%s)
    while true; do
      health_json=$(curl -fsS "http://127.0.0.1:${port}/health" 2>/dev/null || true)
      if echo "$health_json" | grep -q '"model_loaded":true'; then
        echo "[lightx2v] prewarm requested on port $port"
        curl -fsS \
          -X POST \
          -H 'Content-Type: application/json' \
          -d '{"infer_steps":4,"guidance_scale":1.0}' \
          "http://127.0.0.1:${port}/prewarm" || true
        break
      fi
      if [ "$timeout" != "0" ] && [ $(( $(date +%s) - start_ts )) -ge "$timeout" ]; then
        echo "[lightx2v] prewarm skipped: service not ready after ${timeout}s"
        break
      fi
      sleep 10
    done
  ) >>"$LOG_DIR/lightx2v_${LIGHTX2V_PORT:-8001}_prewarm.log" 2>&1 &
  echo "[lightx2v] background prewarm watcher pid=$!"
fi

echo "Generation service startup requested."
