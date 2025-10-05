#!/usr/bin/env bash
set -euo pipefail

# 0) Where we are
cd "$(dirname "$0")"

# 1) Kick off a sweep (same grids you’ve been using; purely offline)
python3 auto_tune_gate.py --states states.csv --device cpu \
  --mode sign --fee-bps 5.0 --H 40 \
  --min-agree-k 2 3 \
  --agree-eps 0.05 0.10 0.15 \
  --min-mag 0.10 0.20 \
  --action-th 0.05 0.10 0.15 \
  --hold-bars 4 8 12

# 2) Figure out the sweep dir that was just created
SWEEP="$(ls -dt runs/sweep_sign_* | head -1)"
echo "[info] using sweep: $SWEEP"

# 3) Show progress (last lines) in a readable way
echo "[info] tailing sweep results (Ctrl-C when you’re done peeking)…"
( tail -n +1 -f "$SWEEP/results.csv" 2>/dev/null | sed -u 's/,/ , /g' ) &
TAILPID=$!

# 4) Wait for the sweep process to finish writing, then kill the tail
#    (If auto_tune exits instantly, the file is already complete; we just sleep a bit for sanity)
sleep 2
kill $TAILPID >/dev/null 2>&1 || true

# 5) Rank & filter (NO live). Tighten/relax filters as you like.
python3 rank_sweep.py --sweep "$SWEEP" --top 12 \
  --min-trades 6 --min-equity 1.01 --min-sharpe 0.05 --max-dd 1.50

# 6) Choose a known-good candidate (8 trades, moderate DD) and backtest it now.
#    If rank_sweep prints a different best tag, swap below accordingly.
BEST_GATE="$SWEEP/gates/gate_k2_eps0.05_mag0.1_th0.05_hold8.csv"
if [ ! -f "$BEST_GATE" ]; then
  # fallback: pick first gate file
  BEST_GATE="$(ls -1 "$SWEEP"/gates/*.csv | head -1)"
fi
echo "[info] backtesting $BEST_GATE …"
python3 backtest_gate_horizon_v3.py \
  --states states.csv --gate "$BEST_GATE" \
  --h 40 --mode sign --action_thresh 0.05 --hold_bars 8 --fee_bps 5.0 --verbose

echo ""
echo "[done] Sweep dir  : $SWEEP"
echo "[done] Backtested : $BEST_GATE"
echo ""
echo "[tip] To eyeball the whole table later:"
echo "  ./view_sweep.py --sweep \"$SWEEP\" --top 12 --show-sat"
