#!/usr/bin/env python3
import argparse, os, re, csv, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--sweep", required=True)
ap.add_argument("--states", default="states.csv")
ap.add_argument("--H", type=int, default=40)
ap.add_argument("--mode", choices=["sign","linear"], default="sign")
ap.add_argument("--fee_bps", type=float, default=5.0)
args = ap.parse_args()

gates_dir = os.path.join(args.sweep, "gates")
if not os.path.isdir(gates_dir):
    print(f"[error] gates dir not found: {gates_dir}", file=sys.stderr); sys.exit(2)

hdr = ["mode","H","fee_bps","min_agree_k","agree_eps","min_mag","action_thresh","linear_scale",
       "hold_bars","rows","trades","trade_winrate","per_bar_hitrate","equity_end","Sharpe_like",
       "MaxDD","CAGR_like","exposure","avg_turnover","avg_trade_pnl","median_trade_pnl",
       "saturation_neg1","saturation_pos1","saturation_mid","gate_csv","tag"]

pat = re.compile(r"^gate_k(?P<k>\d+)_eps(?P<eps>[0-9.]+)_mag(?P<mag>[0-9.]+)_th(?P<th>[0-9.]+)_hold(?P<hold>\d+)\.csv$")
def parse_gate_name(fn):
    m = pat.match(fn)
    return (dict(min_agree_k=int(m["k"]), agree_eps=float(m["eps"]), min_mag=float(m["mag"]),
                 action_thresh=float(m["th"]), hold_bars=int(m["hold"])) if m else None)

def parse_kpis(text):
    import re
    g=lambda rx,default=None,cast=float:(lambda m: default if not m else cast(m.group(1)))(re.search(rx,text))
    return dict(
        rows=g(r"rows=(\d+)",cast=int), trades=g(r"trades=(\d+)",cast=int),
        trade_winrate=g(r"trade_winrate=([0-9.]+)%"), per_bar_hitrate=g(r"per_bar_hitrate=([0-9.]+)%"),
        equity_end=g(r"equity_end=([0-9.]+)"), Sharpe_like=g(r"Sharpe_like=([\-0-9.]+)"),
        MaxDD=g(r"MaxDD=([0-9.]+)%"), CAGR_like=g(r"CAGR_like=([\-0-9.]+)%"),
        exposure=g(r"exposure=([0-9.]+)%"), avg_turnover=g(r"avg_turnover=([0-9.]+)"),
        avg_trade_pnl=g(r"avg_trade_pnl=([\-0-9.eE]+)"), median_trade_pnl=g(r"median_trade_pnl=([\-0-9.eE]+)")
    )

out_csv = os.path.join(args.sweep, "results.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
    for fn in sorted(os.listdir(gates_dir)):
        if not (fn.startswith("gate_") and fn.endswith(".csv")): continue
        meta = parse_gate_name(fn)
        if not meta: print(f"[skip] {fn}"); continue
        gate_path = os.path.join(gates_dir, fn)
        cmd = ["python3","backtest_gate_horizon_v3.py","--states",args.states,"--gate",gate_path,
               "--h",str(args.H),"--mode",args.mode,"--hold_bars",str(meta["hold_bars"]),
               "--fee_bps",str(args.fee_bps)]
        if args.mode=="sign": cmd += ["--action_thresh", str(meta["action_thresh"])]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"[error] backtest failed for {fn}\n{e.output}", file=sys.stderr); continue
        k = parse_kpis(out)
        row = {"mode":args.mode,"H":args.H,"fee_bps":args.fee_bps,
               "min_agree_k":meta["min_agree_k"],"agree_eps":meta["agree_eps"],"min_mag":meta["min_mag"],
               "action_thresh":meta["action_thresh"],"linear_scale":"","hold_bars":meta["hold_bars"],
               "gate_csv":fn,"tag":f"k{meta['min_agree_k']}_eps{meta['agree_eps']}_mag{meta['min_mag']}_th{meta['action_thresh']}_hold{meta['hold_bars']}",
               "saturation_neg1":"","saturation_pos1":"","saturation_mid":""}
        row.update(k); w.writerow(row)
        sys.stdout.write(f"[ok] {fn} -> eq={row['equity_end']} sharpe={row['Sharpe_like']} trades={row['trades']}\n")
print(f"[done] wrote {out_csv}")
