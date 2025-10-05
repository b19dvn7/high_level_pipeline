import os, csv, pathlib, sys

p = pathlib.Path("auto_tune_gate.py")
src = p.read_text()

# We'll locate the start of the old footer by an anchor string
anchor = "Recommended Params (no live execution)"
idx = src.find(anchor)

if idx == -1:
    # fall back: chop last ~120 lines and rebuild a footer anyway
    lines = src.splitlines()
    keep = "\n".join(lines[:-120]) if len(lines) > 120 else src
else:
    # keep everything up to the line BEFORE the anchor’s line
    head = src[:idx]
    # also backtrack to the start of that line
    head = head[:head.rfind("\n")] if "\n" in head else head
    keep = head

safe_footer = r'''
# === Recommended Params (no live execution) ===
# We re-read results.csv we just wrote into out_dir and pick a "best" row to print clean commands.

import csv, os

def _num(x, default=0.0):
    try:
        return float(x)
    except: 
        return default

def _pick_best(results_csv):
    if not os.path.exists(results_csv):
        return None
    with open(results_csv, newline='') as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        return None
    # Prefer higher equity_end, then higher Sharpe_like
    rows.sort(key=lambda r: (_num(r.get('equity_end',0)), _num(r.get('Sharpe_like',0))), reverse=True)
    return rows[0]

try:
    results_csv = os.path.join(out_dir, "results.csv")  # out_dir is defined earlier in the script
    best = _pick_best(results_csv)
    if best is None:
        print("[warn] no best row to print (results.csv empty?).")
    else:
        gate_csv = best.get('gate_csv','')
        tag      = best.get('tag','')
        win      = best.get('trade_winrate','')
        hit      = best.get('per_bar_hitrate','')
        eq       = best.get('equity_end','')
        sh       = best.get('Sharpe_like','')
        dd       = best.get('MaxDD','')
        hold     = best.get('hold_bars','')
        th       = best.get('action_thresh','')
        ls       = best.get('linear_scale','')
        mode     = args.mode

        print("\\n=== Recommended Params (no live execution) ===")
        print(f"Gate CSV: {gate_csv}")
        print(f"Tag     : {tag}")
        print(f"KPI     : Win {win} | Hit {hit} | Eq {eq} | Sharpe {sh} | MaxDD {dd}")

        print("\\n# When you want to backtest this exact gate again:")
        print("python3 backtest_gate_horizon_v3.py \\")
        print(f"  --states {args.states} --gate {os.path.join(out_dir,'gates',gate_csv)} \\")
        if th:
            print(f"  --h {args.H} --mode {mode} --action_thresh {th} --hold_bars {hold} --fee_bps {args.fee_bps}")
        else:
            # linear mode case
            print(f"  --h {args.H} --mode {mode} --linear_scale {ls} --hold_bars {hold} --fee_bps {args.fee_bps}")

except Exception as e:
    print(f"[warn] footer print failed: {e}")
'''.lstrip()

p.write_text(keep + "\n" + safe_footer)
print("[patch] auto_tune_gate.py footer replaced OK.")
