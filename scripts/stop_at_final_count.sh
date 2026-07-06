#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${REFLECTVPR_OUTPUT_ROOT:-/data_nvme/zhangjingyi/ReflectVPR/output_gsv23_100k}"
TARGET="${REFLECTVPR_STOP_FINAL_COUNT:-50000}"
INTERVAL="${REFLECTVPR_STOP_CHECK_INTERVAL:-60}"
LOG_FILE="${REFLECTVPR_STOP_WATCH_LOG:-/tmp/reflectvpr_stop_at_${TARGET}.log}"
STOP_SERVICES="${REFLECTVPR_STOP_SERVICES:-1}"

count_final() {
  python3 - "$OUTPUT_ROOT" <<'PY'
import sys
from pathlib import Path

base = Path(sys.argv[1])
count = 0
if base.exists():
    for city_dir in base.iterdir():
        if not city_dir.is_dir() or city_dir.name.startswith("_"):
            continue
        for route in ("dual", "global", "local"):
            route_dir = city_dir / route
            if route_dir.is_dir():
                count += sum(1 for _ in route_dir.glob("*__final.jpg"))
print(count)
PY
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

stop_sessions() {
  local sessions patterns session
  patterns='^(reflectvpr_plan_|reflectvpr_a6000_g)'
  if [ "$STOP_SERVICES" = "1" ]; then
    patterns='^(reflectvpr_plan_|reflectvpr_a6000_g|reflectvpr_iclight_|reflectvpr_lightx2v_)'
  fi

  sessions="$(tmux ls 2>/dev/null | awk -F: '{print $1}' | grep -E "$patterns" || true)"
  if [ -z "$sessions" ]; then
    log "no matching tmux sessions to stop"
    return 0
  fi

  log "sending C-c to sessions: $(echo "$sessions" | tr '\n' ' ')"
  while IFS= read -r session; do
    [ -n "$session" ] || continue
    tmux send-keys -t "$session" C-c 2>/dev/null || true
  done <<< "$sessions"

  sleep 20

  sessions="$(tmux ls 2>/dev/null | awk -F: '{print $1}' | grep -E "$patterns" || true)"
  if [ -n "$sessions" ]; then
    log "killing remaining sessions: $(echo "$sessions" | tr '\n' ' ')"
    while IFS= read -r session; do
      [ -n "$session" ] || continue
      tmux kill-session -t "$session" 2>/dev/null || true
    done <<< "$sessions"
  fi
}

log "watch start output=$OUTPUT_ROOT target=$TARGET interval=${INTERVAL}s stop_services=$STOP_SERVICES"
while true; do
  current="$(count_final)"
  log "final_count=$current target=$TARGET"
  if [ "$current" -ge "$TARGET" ]; then
    log "target reached; stopping ReflectVPR sessions"
    stop_sessions
    log "done"
    exit 0
  fi
  sleep "$INTERVAL"
done
