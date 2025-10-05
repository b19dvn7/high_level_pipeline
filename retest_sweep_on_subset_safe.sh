#!/usr/bin/env bash
set -euo pipefail

SWEEP_SRC="${1:-runs/sweep_sign_20251005_044449}"   # existing working sweep (source of gates)
STATES="${2:-states_train.csv}"                      # subset to test on
OUT="runs/sweep_sign_retest_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/gates"

# Copy gates
cp -v "$SWEEP_SRC"/gates/gate_*.csv "$OUT/gates/" 2>/dev/null || {
  echo "[error] no gates found in $SWEEP_SRC/gates" >&2
  exit 2
}

# small helper: sed-extract first numeric group after KEY=
# usage: val "$(sed_extract "Sharpe_like" "$OUTTXT")"
sed_extract() {
  local key="$1" txt="$2"
  # match KEY=<number possibly signed with dot/exponent> and print the number
  printf "%s" "$txt" | sed -n "s/.*${key}=\([-+0-9.eE]*\).*/\1/p" | head -1
}

# Build results.csv by re-running backtests on the subset
{
  echo "mode,H,fee_bps,min_agree_k,agree_eps,min_mag,action_thresh,linear_scale,hold_bars,rows,trades,trade_winrate,per_bar_hitrate,equity_end,Sharpe_like,MaxDD,CAGR_like,exposure,avg_turnover,avg_trade_pnl,median_trade_pnl,saturation_neg1,saturation_pos1,saturation_mid,gate_csv,tag"
  for g in "$OUT"/gates/gate_*.csv; do
    base="$(basename "$g")"
    # parse params from filename: gate_k2_eps0.05_mag0.1_th0.05_hold8.csv
    if [[ "$base" =~ gate_k([0-9]+)_eps([0-9.]+)_mag([0-9.]+)_th([0-9.]+)_hold([0-9]+)\.csv ]]; then
      K="${BASH_REMATCH[1]}"; EPS="${BASH_REMATCH[2]}"; MAG="${BASH_REMATCH[3]}"; TH="${BASH_REMATCH[4]}"; HOLD="${BASH_REMATCH[5]}"
    else
      echo "[skip] cannot parse $base" >&2
      continue
    fi

    OUTTXT="$(python3 backtest_gate_horizon_v3.py \
      --states "$STATES" --gate "$g" \
      --h 40 --mode sign --action_thresh "$TH" --hold_bars "$HOLD" --fee_bps 5.0 2>&1 || true)"

    rows="$(printf "%s" "$OUTTXT" | sed -n 's/.* rows=\([0-9][0-9]*\).*/\1/p' | head -1)"
    trades="$(printf "%s" "$OUTTXT" | sed -n 's/.* trades=\([0-9][0-9]*\).*/\1/p' | head -1)"
    win="$(printf "%s" "$OUTTXT" | sed -n 's/.*trade_winrate=\([0-9.]\+%\).*/\1/p' | head -1)"
    hit="$(printf "%s" "$OUTTXT" | sed -n 's/.*per_bar_hitrate=\([0-9.]\+%\).*/\1/p' | head -1)"
    eq="$(sed_extract "equity_end" "$OUTTXT")"
    sh="$(sed_extract "Sharpe_like" "$OUTTXT")"
    dd="$(printf "%s" "$OUTTXT" | sed -n 's/.*MaxDD=\([0-9.]\+%\).*/\1/p' | head -1)"
    cagr="$(printf "%s" "$OUTTXT" | sed -n 's/.*CAGR_like=\([\-0-9.]\+%\).*/\1/p' | head -1)"
    exp="$(printf "%s" "$OUTTXT" | sed -n 's/.*exposure=\([0-9.]\+%\).*/\1/p' | head -1)"
    turn="$(printf "%s" "$OUTTXT" | sed -n 's/.*avg_turnover=\([0-9.]\+\)\/bar.*/\1/p' | head -1)"
    avgp="$(sed_extract "avg_trade_pnl" "$OUTTXT")"
    medp="$(sed_extract "median_trade_pnl" "$OUTTXT")"

    rows="${rows:-0}"; trades="${trades:-0}" ; eq="${eq:-0}" ; sh="${sh:-0}" ; turn="${turn:-0}" ; avgp="${avgp:-0}" ; medp="${medp:-0}"
    win="${win:-0.00%}" ; hit="${hit:-0.00%}" ; dd="${dd:-0.00%}" ; cagr="${cagr:-0.00%}" ; exp="${exp:-0.00%}"

    tag="k${K}_eps${EPS}_mag${MAG}_th${TH}_hold${HOLD}"
    echo "sign,40,5.0,$K,$EPS,$MAG,$TH,,${HOLD},${rows},${trades},${win},${hit},${eq},${sh},${dd},${cagr},${exp},${turn},${avgp},${medp},,,,$base,$tag"
    printf "[ok] %-40s eq=%-8s sharpe=%-8s trades=%s\n" "$base" "$eq" "$sh" "$trades" >&2
  done
} > "$OUT/results.csv"

echo "[done] retested to $OUT"
