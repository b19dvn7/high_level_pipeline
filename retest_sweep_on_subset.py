#!/usr/bin/env python3
import argparse, os, re, shutil, subprocess, sys, csv
from pathlib import Path
from datetime import datetime, timezone

RE_NUM  = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
RE_INT  = r'\d+'
def rx(p): return re.compile(p)

# Robust patterns
R = {
  "rows":     rx(rf'rows=({RE_INT})'),
  "trades":   rx(rf'trades=({RE_INT})'),
  "win":      rx(rf'trade_winrate=({RE_NUM})%'),
  "hit":      rx(rf'per_bar_hitrate=({RE_NUM})%'),
  "eq":       rx(rf'equity_end=({RE_NUM})'),
  "sharpe":   rx(rf'Sharpe_like=({RE_NUM})'),
  "maxdd":    rx(rf'MaxDD=({RE_NUM})%'),
  "cagr":     rx(rf'CAGR_like=({RE_NUM})%'),
  "expo":     rx(rf'exposure=({RE_NUM})%'),
  "turn":     rx(rf'avg_turnover=({RE_NUM})/bar'),
  "avgp":     rx(rf'avg_trade_pnl=({RE_NUM})'),
  "medp":     rx(rf'median_trade_pnl=({RE_NUM})'),
}

FN_RX = re.compile(
  r'^gate_k(?P<k>\d+)_eps(?P<eps>[0-9.]+)_mag(?P<mag>[0-9.]+)_th(?P<th>[0-9.]+)_hold(?P<hold>\d+)\.csv$'
)

def g1(rx, s):
    m = rx.search(s)
    return m.group(1) if m else None

def f_or(x, default):
    try: 
        return float(x)
    except:
        return default

def sat_counts(csv_path: Path):
    neg = pos = mid = 0
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 3: 
                continue
            try:
                a = float(row[2])
            except:
                continue
            if a <= -0.99: neg += 1
            elif a >= 0.99: pos += 1
            else: mid += 1
    return neg, pos, mid

def run_backtest(states, gate, H, fee_bps):
    mm = FN_RX.match(gate.name)
    th   = mm.group("th")
    hold = mm.group("hold")
    cmd = [
        "python3", "backtest_gate_horizon_v3.py",
        "--states", states, "--gate", str(gate),
        "--h", str(H), "--mode", "sign",
        "--action_thresh", th, "--hold_bars", hold,
        "--fee_bps", str(fee_bps), "--verbose"
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-src", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--H", type=int, default=40)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path.cwd()
    src = Path(args.sweep_src).resolve()
    gates_src = src / "gates"
    if not gates_src.exists():
        print(f"[error] no gates dir in {src}", file=sys.stderr); return 2

    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = root / "runs" / f"sweep_sign_retest_{ts}"
    gates_out = out_dir / "gates"
    logs_out  = out_dir / "logs"
    gates_out.mkdir(parents=True, exist_ok=True)
    logs_out.mkdir(parents=True, exist_ok=True)

    # copy gates
    copied = 0
    for g in sorted(gates_src.glob("gate_*.csv")):
        shutil.copy2(g, gates_out / g.name); copied += 1
    if copied == 0:
        print(f"[error] no gate_*.csv in {gates_src}", file=sys.stderr); return 2

    results_path = out_dir / "results.csv"
    with results_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "mode","H","fee_bps","min_agree_k","agree_eps","min_mag","action_thresh",
            "linear_scale","hold_bars","rows","trades","trade_winrate","per_bar_hitrate",
            "equity_end","Sharpe_like","MaxDD","CAGR_like","exposure","avg_turnover",
            "avg_trade_pnl","median_trade_pnl","saturation_neg1","saturation_pos1",
            "saturation_mid","gate_csv","tag"
        ])

        for g in sorted(gates_out.glob("gate_*.csv")):
            base = g.name
            mm = FN_RX.match(base)
            if not mm:
                print(f"[skip] cannot parse params from {base}", file=sys.stderr)
                continue

            K    = int(mm.group("k"))
            EPS  = float(mm.group("eps"))
            MAG  = float(mm.group("mag"))
            TH   = float(mm.group("th"))
            HOLD = int(mm.group("hold"))

            # saturation
            sneg, spos, smid = sat_counts(g)

            # backtest + raw log
            try:
                out = run_backtest(args.states, g, args.H, args.fee_bps)
            except subprocess.CalledProcessError as e:
                out = e.output

            (logs_out / f"{base}.log").write_text(out)

            rows    = int(f_or(g1(R["rows"],   out), 0))
            trades  = int(f_or(g1(R["trades"], out), 0))
            win     = f_or(g1(R["win"],   out), 0.0)   # already stripped %
            hit     = f_or(g1(R["hit"],   out), 0.0)
            eq      = f_or(g1(R["eq"],    out), 0.0)
            sharpe  = f_or(g1(R["sharpe"],out), 0.0)
            maxdd   = f_or(g1(R["maxdd"], out), 0.0)
            cagr    = f_or(g1(R["cagr"],  out), 0.0)
            expo    = f_or(g1(R["expo"],  out), 0.0)
            turn    = f_or(g1(R["turn"],  out), 0.0)
            avgp    = f_or(g1(R["avgp"],  out), 0.0)
            medp    = f_or(g1(R["medp"],  out), 0.0)

            tag = f"k{K}_eps{EPS}_mag{MAG}_th{TH}_hold{HOLD}"
            w.writerow([
                "sign", args.H, args.fee_bps, K, EPS, MAG, TH, "",
                HOLD, rows, trades, win, hit, eq, sharpe, maxdd, cagr,
                expo, turn, avgp, medp, sneg, spos, smid, base, tag
            ])
            print(f"[ok] {base:40s} eq={eq:7.4f} sharpe={sharpe:5.2f} trades={trades}  sat: (-1){sneg} mid{smid} (+1){spos}", file=sys.stderr)

    print(f"[done] retested to {out_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
