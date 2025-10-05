#!/usr/bin/env bash
set -euo pipefail

# 1) strict-ish sweep on s2_train
python3 auto_tune_gate.py \
  --states s2_train.csv --device cpu --mode sign --fee-bps 5.0 --H 40 \
  --min-agree-k 3 \
  --agree-eps 0.05 0.10 \
  --min-mag 0.15 0.20 \
  --action-th 0.10 0.15 \
  --hold-bars 8 12

# 2) find the latest sign sweep dir
SWEEP="$(ls -dt runs/sweep_sign_* 2>/dev/null | head -1)"
[ -n "$SWEEP" ] || { echo "[ERR] no sweep dir created"; exit 1; }
echo "[info] Using SWEEP=$SWEEP"

# 3) pick best row by filters (>=3 trades, Eq>=1, Sharpe>=0, DD<=1.5%)
python3 ./pick_best_from_results.py "$SWEEP/results.csv" > /tmp/best.txt
cat /tmp/best.txt
GATE="$(grep '^GATE=' /tmp/best.txt | cut -d= -f2)"
TAG="$(grep '^TAG='  /tmp/best.txt | cut -d= -f2)"
[ -n "$GATE" ] || { echo "[ERR] could not resolve GATE from results"; exit 1; }

# 4) backtest EXACT gate on s2_test at the matching th/hold from the tag
echo "[info] Backtesting on s2_test with gate=$GATE"
python3 backtest_gate_horizon_v3.py \
  --states s2_test.csv \
  --gate   "$SWEEP/gates/$GATE" \
  --h 40 --mode sign --action_thresh 0.10 --hold_bars 8 --fee_bps 5.0 --verbose || true

python3 backtest_gate_horizon_v3.py \
  --states s2_test.csv \
  --gate   "$SWEEP/gates/$GATE" \
  --h 40 --mode sign --action_thresh 0.15 --hold_bars 8 --fee_bps 5.0 --verbose || true

# Try 12-bar hold variants too (often smoother)
python3 backtest_gate_horizon_v3.py \
  --states s2_test.csv \
  --gate   "$SWEEP/gates/$GATE" \
  --h 40 --mode sign --action_thresh 0.10 --hold_bars 12 --fee_bps 5.0 --verbose || true

python3 backtest_gate_horizon_v3.py \
  --states s2_test.csv \
  --gate   "$SWEEP/gates/$GATE" \
  --h 40 --mode sign --action_thresh 0.15 --hold_bars 12 --fee_bps 5.0 --verbose || true
