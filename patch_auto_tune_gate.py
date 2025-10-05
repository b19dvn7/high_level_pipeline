import re, sys, pathlib

p = pathlib.Path("auto_tune_gate.py")
src = p.read_text()

# Find the broken footer area and replace with a safe printer.
# We look for the block that mentions "Recommended Params (no live execution)".
pat = re.compile(
    r"(?s)\n\s*#\s*Recommended Params.*?print\(.+?\)\s*$"
)

replacement = r'''
# --- Recommended summary (safe prints) ---
if best is not None:
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
    sweep_dir = out_dir  # where we wrote results

    print("\\n=== Recommended Params (no live execution) ===")
    print(f"Gate CSV: {gate_csv}")
    print(f"Tag     : {tag}")
    print(f"KPI     : Win {win} | Hit {hit} | Eq {eq} | Sharpe {sh} | MaxDD {dd}")

    # Backtest command
    print("\\n# When you want to backtest this exact gate again:")
    print("python3 backtest_gate_horizon_v3.py \\")
    print(f"  --states {args.states} --gate {sweep_dir}/gates/{gate_csv} \\")
    if th:
        print(f"  --h {args.H} --mode {mode} --action_thresh {th} --hold_bars {hold} --fee_bps {args.fee_bps}")
    else:
        print(f"  --h {args.H} --mode {mode} --linear_scale {ls} --hold_bars {hold} --fee_bps {args.fee_bps}")
else:
    print("[warn] no best row computed; results.csv may be empty.")
'''.strip()

new = pat.sub("\n"+replacement, src)
if new == src:
    print("[patch] pattern not found. No change made (maybe already fixed?).")
else:
    p.write_text(new)
    print("[patch] auto_tune_gate.py footer replaced OK.")
