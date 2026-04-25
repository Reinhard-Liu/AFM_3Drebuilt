#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/micro
CONFIG="$ROOT/config_v20_object_joint_medium10.json"
TRAIN_DIR="$ROOT/experiments/v20_object_joint_medium10"
SAVE_DIR="$TRAIN_DIR/checkpoints"
LOG_DIR="$TRAIN_DIR/logs"
LATEST="$SAVE_DIR/latest_v19_object_joint.pt"

mkdir -p "$LOG_DIR" "$SAVE_DIR"
TS=$(date -u +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/v20_object_joint_medium10_${TS}.log"
PIDFILE="$LOG_DIR/v20_object_joint_medium10_${TS}.pid"

cd "$ROOT"
if [[ -f "$LATEST" ]]; then
  nohup env PYTHONUNBUFFERED=1 python3 -u -m src.train_v19_object_joint --config "$CONFIG" --resume_checkpoint "$LATEST" > "$LOGFILE" 2>&1 < /dev/null &
else
  nohup env PYTHONUNBUFFERED=1 python3 -u -m src.train_v19_object_joint --config "$CONFIG" > "$LOGFILE" 2>&1 < /dev/null &
fi

echo $! > "$PIDFILE"
echo "PIDFILE=$PIDFILE"
echo "LOGFILE=$LOGFILE"
