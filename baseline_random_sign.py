#!/usr/bin/env python3
import argparse, csv, random, time, subprocess, sys, os, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--states", default="states.csv")
ap.add_argument("--H", type=int, default=40)
ap.add_argument("--fee-bps", type=float, default=5.0)
ap.add_argument("--hold-bars", type=int, default=8)
args = ap.parse_args()

tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv")
tmp.write("row_idx,agree,action,details\n")
with open(args.states) as f:
    n = sum(1 for _ in f) - 1
for i in range(n):
    a = random.choice([-1.0, 0.0, 1.0])
    tmp.write(f"{i},1,{a},{{}}\n")
tmp.close()

cmd = [sys.executable, "backtest_gate_horizon_v3.py",
       "--states", args.states, "--gate", tmp.name,
       "--h", str(args.H), "--mode", "sign",
       "--action_thresh", "0.05",
       "--hold_bars", str(args.hold_bars),
       "--fee_bps", str(args.fee_bps)]
out = subprocess.check_output(cmd, text=True)
print(out)
os.unlink(tmp.name)
