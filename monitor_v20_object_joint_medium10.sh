#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/micro
TRAIN_DIR="$ROOT/experiments/v20_object_joint_medium10"
LOG_DIR="$TRAIN_DIR/logs"
SAVE_DIR="$TRAIN_DIR/checkpoints"
MONITOR_LOG="$LOG_DIR/stall_monitor.log"
TARGET_PATTERN="src.train_v19_object_joint --config /root/autodl-tmp/micro/config_v20_object_joint_medium10.json"
STALL_SECONDS=1800
CHECK_INTERVAL=60

mkdir -p "$LOG_DIR" "$SAVE_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$MONITOR_LOG"
}

latest_activity_ts() {
  python3 - <<'PY'
from pathlib import Path
paths = [
    Path("/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/checkpoints/latest_v19_object_joint.pt"),
    Path("/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/checkpoints/history_v19_object_joint.json"),
]
log_dir = Path("/root/autodl-tmp/micro/experiments/v20_object_joint_medium10/logs")
paths.extend(sorted(log_dir.glob("v20_object_joint_medium10_*.log")))
existing = [p.stat().st_mtime for p in paths if p.exists()]
print(int(max(existing)) if existing else -1)
PY
}

log "stall monitor started; stall_seconds=${STALL_SECONDS}; check_interval=${CHECK_INTERVAL}"

while true; do
  parent_pid="$(pgrep -of "$TARGET_PATTERN" || true)"
  if [[ -n "$parent_pid" ]]; then
    now="$(date +%s)"
    last_ts="$(latest_activity_ts)"
    if [[ "$last_ts" -gt 0 ]]; then
      idle=$(( now - last_ts ))
      if [[ "$idle" -ge "$STALL_SECONDS" ]]; then
        log "detected stalled training; parent_pid=${parent_pid}; idle_seconds=${idle}; killing parent to trigger supervisor restart"
        kill "$parent_pid" || true
        sleep 5
      fi
    fi
  fi
  sleep "$CHECK_INTERVAL"
done
