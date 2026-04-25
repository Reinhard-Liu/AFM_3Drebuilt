#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/micro
RUNNER="$ROOT/run_v19_object_joint_full15_all.sh"
TRAIN_DIR="$ROOT/experiments/v19_object_joint_full15_all"
SAVE_DIR="$TRAIN_DIR/checkpoints"
LOG_DIR="$TRAIN_DIR/logs"
WATCH_LOG="$LOG_DIR/watchdog.log"
CONFIG="$ROOT/config_v19_object_joint_full15_all.json"
LATEST="$SAVE_DIR/latest_v19_object_joint.pt"

mkdir -p "$LOG_DIR" "$SAVE_DIR"

get_target_epochs() {
  python3 - <<'PY'
import json
from pathlib import Path
p = Path("/root/autodl-tmp/micro/config_v19_object_joint_full15_all.json")
cfg = json.loads(p.read_text())
print(int(cfg.get("epochs", 0)))
PY
}

get_latest_epoch() {
  python3 - <<'PY'
from pathlib import Path
import torch
p = Path("/root/autodl-tmp/micro/experiments/v19_object_joint_full15_all/checkpoints/latest_v19_object_joint.pt")
if not p.exists():
    print(-1)
else:
    state = torch.load(p, map_location="cpu", weights_only=False)
    print(int(state.get("epoch", -1)))
PY
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$WATCH_LOG"
}

target_epochs="$(get_target_epochs)"
log "watchdog started; target_epochs=${target_epochs}"

while true; do
  mapfile -t pids < <(pgrep -f "python3 -u -m src.train_v19_object_joint --config /root/autodl-tmp/micro/config_v19_object_joint_full15_all.json" || true)
  if [[ "${#pids[@]}" -eq 0 ]]; then
    latest_epoch="$(get_latest_epoch)"
    if [[ "$latest_epoch" -ge "$target_epochs" ]]; then
      log "training already complete at epoch=${latest_epoch}; watchdog exiting"
      exit 0
    fi
    log "training process missing; latest_epoch=${latest_epoch}; restarting"
    bash "$RUNNER" >> "$WATCH_LOG" 2>&1
    sleep 5
  fi
  sleep 30
done
