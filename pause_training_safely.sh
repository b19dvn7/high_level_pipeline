#!/usr/bin/env bash
set -euo pipefail

# 1) Snapshot current artifacts to a timestamped folder
STAMP=$(date -u +"%Y%m%d_%H%M%S")
SNAP="runs/snap_${STAMP}"
mkdir -p "$SNAP"

# Copy whatever models exist right now (actors/critics/scaler/gate if present)
# These 'cp' lines will ignore missing files (|| true)
cp -v td3bc_actor_seed*.pt        "$SNAP"/ 2>/dev/null || true
cp -v td3bc_critic*_seed*.pt      "$SNAP"/ 2>/dev/null || true
cp -v scaler.json                 "$SNAP"/ 2>/dev/null || true
cp -v consensus_gate_td3bc.json   "$SNAP"/ 2>/dev/null || true
cp -v rl_duckdb_data_layer.py     "$SNAP"/ 2>/dev/null || true
echo "[snapshot] saved to $SNAP"

# 2) Send a gentle SIGINT (same as pressing Ctrl+C once)
PID=$(pgrep -f train_td3bc_per_ensemble.py || true)
if [[ -n "${PID}" ]]; then
  echo "[stop] sending SIGINT to PID(s): ${PID}"
  kill -SIGINT ${PID}
  # Wait up to ~15s for graceful exit
  for i in {1..15}; do
    sleep 1
    if ! ps -p ${PID} >/dev/null 2>&1; then
      echo "[stop] trainer exited cleanly."
      exit 0
    fi
  done
  # 3) If still running, escalate to SIGTERM (still graceful)
  echo "[stop] still running; sending SIGTERM…"
  kill -SIGTERM ${PID}
  sleep 5
  if ps -p ${PID} >/dev/null 2>&1; then
    echo "[stop] still running; LAST RESORT sending SIGKILL…"
    kill -9 ${PID}
  fi
else
  echo "[stop] no running trainer found."
fi
