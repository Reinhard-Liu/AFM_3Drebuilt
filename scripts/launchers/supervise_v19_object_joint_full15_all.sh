#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$ROOT/configs/config_v19_object_joint_full15_all.json"
TRAIN_DIR="$ROOT/experiments/v19_object_joint_full15_all"
SAVE_DIR="$TRAIN_DIR/checkpoints"
LOG_DIR="$TRAIN_DIR/logs"
SUP_LOG="$LOG_DIR/supervisor.log"
LATEST="$SAVE_DIR/latest_v19_object_joint.pt"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$SUP_LOG"
}

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

run_once() {
  local ts logfile
  ts="$(date -u +%Y%m%d_%H%M%S)"
  logfile="$LOG_DIR/v19_object_joint_full15_all_${ts}.log"
  log "launching training; logfile=${logfile}"
  cd "$ROOT"
  if [[ -f "$LATEST" ]]; then
    env PYTHONUNBUFFERED=1 python3 -u -m src.train_v19_object_joint --config "$CONFIG" --resume_checkpoint "$LATEST" > "$logfile" 2>&1
  else
    env PYTHONUNBUFFERED=1 python3 -u -m src.train_v19_object_joint --config "$CONFIG" > "$logfile" 2>&1
  fi
}

target_epochs="$(get_target_epochs)"
log "supervisor started; target_epochs=${target_epochs}"

while true; do
  latest_epoch="$(get_latest_epoch)"
  if [[ "$latest_epoch" -ge "$target_epochs" ]]; then
    log "training already complete at epoch=${latest_epoch}; supervisor exiting"
    exit 0
  fi

  set +e
  run_once
  status=$?
  set -e

  latest_epoch="$(get_latest_epoch)"
  if [[ "$latest_epoch" -ge "$target_epochs" ]]; then
    log "training reached target epoch=${latest_epoch}; last_status=${status}; supervisor exiting"
    exit 0
  fi

  log "training exited unexpectedly; status=${status}; latest_epoch=${latest_epoch}; restarting in 10s"
  sleep 10
done
