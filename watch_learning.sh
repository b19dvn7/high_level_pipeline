#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# how often to evaluate (minutes)
EVERY_MIN=${1:-5}

OUTDIR="runs/eval_watch"
mkdir -p "$OUTDIR"

while true; do
  echo "[watch] snapshot eval…"
  python3 eval_actors_snapshot.py \
    --actors-dir . \
    --gate consensus_gate_td3bc.json \
    --states states.csv \
    --device cpu \
    --H 40 --mode sign --action-th 0.05 --hold-bars 8 --fee-bps 5.0 \
    --outdir "$OUTDIR" || true

  echo
  echo "=== eval progress (last 10) ==="
  if [ -f "$OUTDIR/eval.csv" ]; then
    tail -n 10 "$OUTDIR/eval.csv"
  else
    echo "(no eval yet)"
  fi
  echo

  # simple sparkline of equity_end over time
  if [ -f "$OUTDIR/eval.csv" ]; then
    python3 - <<'PY'
import pandas as pd, os
df = pd.read_csv("runs/eval_watch/eval.csv")
if len(df):
    xs = df['equity_end'].tail(30).fillna(1.0).tolist()
    # text sparkline
    blocks="▁▂▃▄▅▆▇█"
    lo=min(xs); hi=max(xs)
    def scale(x):
        if hi==lo: return 0
        r=(x-lo)/(hi-lo)
        return min(7, max(0, int(r*7)))
    print("[equity spark]", "".join(blocks[scale(x)] for x in xs), f"  last={xs[-1]:.4f}")
PY
  fi

  sleep $((EVERY_MIN*60))
done
