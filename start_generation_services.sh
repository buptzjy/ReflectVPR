#!/bin/bash
set -euo pipefail

LOG_DIR=${REFLECTVPR_SERVICE_LOG_DIR:-/media/data/zhangjingyi/ReflectVPR/tmp/logs}
mkdir -p "$LOG_DIR"

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export REFLECTVPR_DISABLE_SERVICE_MOCK=${REFLECTVPR_DISABLE_SERVICE_MOCK:-1}

is_port_open() {
  local port="$1"
  /media/data1/zhangjingyi/miniconda3/envs/hongyb/bin/python - "$port" <<'PY'
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
  local log_file="$LOG_DIR/${name}.log"
  local cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"

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
  if command -v tmux >/dev/null 2>&1; then
    local session="reflectvpr_${name}"
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
    fi
    tmux new-session -d -s "$session" \
      "cd '$workdir'; export PLATFORM=cuda; export SKIP_PLATFORM_CHECK=True; export CUDA_VISIBLE_DEVICES='$cuda_devices'; export REFLECTVPR_TMP_DIR=/media/data/zhangjingyi/ReflectVPR/tmp; export REFLECTVPR_DISABLE_SERVICE_MOCK=\${REFLECTVPR_DISABLE_SERVICE_MOCK:-1}; exec '$python_bin' app.py >>'$log_file' 2>&1"
    echo "[$name] tmux=$session log=$log_file"
  else
    nohup bash -c '
    cd "$1"
    export PLATFORM=cuda
    export SKIP_PLATFORM_CHECK=True
    export CUDA_VISIBLE_DEVICES="$3"
    export REFLECTVPR_TMP_DIR=/media/data/zhangjingyi/ReflectVPR/tmp
    export REFLECTVPR_DISABLE_SERVICE_MOCK="${REFLECTVPR_DISABLE_SERVICE_MOCK:-1}"
    exec "$2" app.py
    ' _ "$workdir" "$python_bin" "$cuda_devices" >>"$log_file" 2>&1 &
    echo "[$name] pid=$! log=$log_file"
  fi
}

start_service \
  "iclight" \
  "8002" \
  "/media/data/zhangjingyi/IC-Light" \
  "/media/data1/zhangjingyi/miniconda3/envs/IC-Light/bin/python"

start_service \
  "lightx2v" \
  "8001" \
  "/media/data/zhangjingyi/LightX2V" \
  "/media/data1/zhangjingyi/miniconda3/envs/lightx2v/bin/python"

echo "Generation service startup requested."
