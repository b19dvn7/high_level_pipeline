#!/usr/bin/env python3
import csv, sys, os, math

# usage: pick_best_from_results.py runs/<sweep_dir>/results.csv
# prints two lines to stdout:
# GATE=gate_....csv
# TAG=....

if len(sys.argv) != 2:
    print("usage: pick_best_from_results.py PATH_TO_results.csv", file=sys.stderr)
    sys.exit(2)

csv_path = sys.argv[1]
if not os.path.isfile(csv_path):
    print(f"[ERR] results.csv not found: {csv_path}", file=sys.stderr)
    sys.exit(1)

rows = []
with open(csv_path, newline="") as f:
    R = csv.DictReader(f)
    for r in R:
        # Skip partial/blank rows
        if not r.get("gate_csv"): 
            continue
        try:
            trades   = int(float(r.get("trades","0")))
            eq_end   = float(r.get("equity_end","nan"))
            sharpe   = float(r.get("Sharpe_like","nan"))
            maxdd    = float(r.get("MaxDD","nan"))
        except:
            continue

        # Filters (tune as you like)
        if trades < 3:                   # avoid too-few-sample mirages
            continue
        if math.isnan(eq_end) or math.isnan(sharpe) or math.isnan(maxdd):
            continue
        # be conservative on test: want >=1.0 eq, non-negative sharpe, moderate DD
        if eq_end < 1.0 or sharpe < 0.0 or maxdd > 1.5:
            continue

        rows.append(r)

if not rows:
    print("[WARN] No rows passed filters; falling back to 'best equity' among all rows.", file=sys.stderr)
    # fallback: just rank everything we can parse
    alt = []
    with open(csv_path, newline="") as f:
        R = csv.DictReader(f)
        for r in R:
            try:
                eq_end   = float(r.get("equity_end","nan"))
                sharpe   = float(r.get("Sharpe_like","nan"))
                maxdd    = float(r.get("MaxDD","inf"))
                trades   = int(float(r.get("trades","0")))
            except:
                continue
            if r.get("gate_csv"):
                alt.append((eq_end, sharpe, -maxdd, trades, r))
    if not alt:
        print("[ERR] results.csv has no usable rows.", file=sys.stderr)
        sys.exit(1)
    rows = [sorted(alt, key=lambda t:(t[0], t[1], t[2], t[3]), reverse=True)[0][-1]]

# Rank: equity_end desc, Sharpe desc, MaxDD asc, trades desc
def keyfn(r):
    return (
        float(r["equity_end"]),
        float(r["Sharpe_like"]),
        -float(r["MaxDD"]),           # smaller DD is better → negative for descending sort
        float(r["trades"])
    )

rows.sort(key=keyfn, reverse=True)
best = rows[0]
print(f"GATE={best['gate_csv']}")
print(f"TAG={best.get('tag','')}")
