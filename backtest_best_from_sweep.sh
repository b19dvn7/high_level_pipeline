#!/usr/bin/env bash
set -euo pipefail

STATES="${1:-states.csv}"
H=40
FEE=5.0
MODE=sign
TH=0.05
HOLD=8

# pick the latest sweep dir that has a non-empty results.csv
SWEEP=$(ls -td runs/sweep_sign_* 2>/dev/null | while read -r d; do
  [[ -s "$d/results.csv" ]] && echo "$d" && break
done)

if [[ -z "${SWEEP:-}" ]]; then
  echo "[error] no sweep directories with a results.csv" >&2
  exit 1
fi

# read results.csv, strip CRLF, pick first 'sign' row, return LAST field (gate filename)
GATE=$(awk -F, '
  { sub(/\r$/, "", $0) }                               # drop CR if present
  NR==1 { next }                                       # skip header
  $1 ~ /^[[:space:]]*sign[[:space:]]*$/ {              # first column is "sign"
    g=$NF                                              # last column = gate_csv
    sub(/^[[:space:]]+/, "", g); sub(/[[:space:]]+$/, "", g)
    print g; exit
  }' "$SWEEP/results.csv")

if [[ -z "${GATE:-}" ]]; then
  echo "[error] could not find a gate_csv in $SWEEP/results.csv" >&2
  exit 1
fi

GATE_PATH="$SWEEP/gates/$GATE"
if [[ ! -f "$GATE_PATH" ]]; then
  echo "[error] gate file not found: $GATE_PATH" >&2
  exit 1
fi

echo "[run] sweep=$SWEEP"
echo "[run] gate =$GATE_PATH"
echo "[run] states=$STATES H=$H th=$TH hold=$HOLD fee_bps=$FEE mode=$MODE"
python3 backtest_gate_horizon_v3.py \
  --states "$STATES" \
  --gate "$GATE_PATH" \
  --h $H --mode $MODE --action_thresh $TH --hold_bars $HOLD --fee_bps $FEE --verbose
