#!/usr/bin/env python3
import argparse, csv, os, time
from math import isnan

def score(row):
    try:
        equity_end  = float(row["equity_end"] or 1.0)
        sharpe      = float(row["Sharpe_like"] or 0.0)
        maxdd       = float(row["MaxDD"] or 0.0)
        exposure    = float(row["exposure"] or 0.0)
        trades      = int(float(row["trades"] or 0))
    except Exception:
        return -1e9
    if trades == 0: return -1e9
    s  = (equity_end - 1.0) * 100.0
    s += 0.5 * sharpe
    s -= 0.1 * maxdd
    if exposure > 95.0: s -= 1.0
    return s

def read_rows(path):
    if not os.path.exists(path): return []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        return list(rdr)

def fmt(r):
    mode = r["mode"]; gate = r["gate_csv"]
    return (f"{mode:5}   {float(r['equity_end']):.4f}   {float(r['Sharpe_like']):6.2f}   "
            f"{float(r['MaxDD']):6.2f}%   {int(float(r['trades'])):4d}   "
            f"k={r['min_agree_k']} eps={r['agree_eps']} mag={r['min_mag']}  "
            f"{('th='+r['action_thresh']) if r['action_thresh'] else ('ls='+r['linear_scale'])}  "
            f"hold={r['hold_bars']}   {gate}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to sweep .../results.csv")
    ap.add_argument("--n", type=int, default=10, help="top K")
    ap.add_argument("--every", type=float, default=1.0, help="refresh seconds")
    args = ap.parse_args()

    while True:
        rows = read_rows(args.results)
        ranked = sorted([(score(r), r) for r in rows], key=lambda x: x[0], reverse=True)
        os.system("clear")
        print(f"=== LIVE TOP {args.n} ===  ({len(rows)} rows)  {args.results}\n")
        print("mode   equity_end   Sharpe   MaxDD    trds   params...   gate")
        print("-"*120)
        for i, (sc, r) in enumerate(ranked[:args.n]):
            print(f"#{i+1:02}  {fmt(r)}   | score={sc:.3f}")
        print("\n[press Ctrl+C to exit]")
        time.sleep(args.every)

if __name__ == "__main__":
    main()
