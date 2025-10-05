#!/usr/bin/env python3
import argparse, json, os, time, subprocess, sys, glob, csv
import numpy as np
import pandas as pd
import torch

def find_actors(actors_dir):
    # pick three seeds if available; otherwise any *.pt
    pts = sorted(glob.glob(os.path.join(actors_dir, "td3bc_actor_seed*.pt")))
    if not pts:
        pts = sorted(glob.glob(os.path.join(actors_dir, "*.pt")))
    return pts[:3]

def run_gate(gate_json, state_csv, device, out_csv):
    # Use your flexible gate to produce actions for states
    cmd = [
        sys.executable, "consensus_gate_td3bc_flexible.py",
        "--gate", gate_json,
        "--state_csv", state_csv,
        "--device", device
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rows = []
    for line in p.stdout:
        if line.startswith("row_idx,") or (len(line)>0 and line.split(",")[0].isdigit()):
            rows.append(line.strip())
    _, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"gate failed: {err}")
    with open(out_csv, "w") as f:
        f.write("\n".join(rows) + "\n")
    return out_csv

def backtest(states_csv, gate_csv, H, mode, fee_bps, action_th=None, hold_bars=None, linear_scale=None):
    cmd = [sys.executable, "backtest_gate_horizon_v3.py",
           "--states", states_csv, "--gate", gate_csv,
           "--h", str(H), "--mode", mode, "--fee_bps", str(fee_bps)]
    if mode == "sign" and action_th is not None:
        cmd += ["--action_thresh", str(action_th)]
    if hold_bars is not None:
        cmd += ["--hold_bars", str(hold_bars)]
    if mode == "linear" and linear_scale is not None:
        cmd += ["--linear_scale", str(linear_scale)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    if p.returncode != 0:
        raise RuntimeError(out)
    # parse KPI lines
    kpi = {"equity_end":None,"sharpe":None,"maxdd":None,"trades":None,"hitrate":None,"winrate":None}
    for ln in out.splitlines():
        if ln.strip().startswith("rows=") and "equity_end=" in ln:
            tok = ln.replace(",", " ").split()
            for t in tok:
                if t.startswith("equity_end="): kpi["equity_end"] = float(t.split("=")[1])
                if t.startswith("Sharpe_like="): kpi["sharpe"] = float(t.split("=")[1])
                if t.startswith("MaxDD="): kpi["maxdd"] = float(t.split("=")[1].replace("%",""))
                if t.startswith("trades="): kpi["trades"] = int(t.split("=")[1])
                if t.startswith("per_bar_hitrate="): kpi["hitrate"] = float(t.split("=")[1].replace("%",""))
                if t.startswith("trade_winrate="): kpi["winrate"] = float(t.split("=")[1].replace("%",""))
    return kpi, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors-dir", default=".")
    ap.add_argument("--gate", default="consensus_gate_td3bc.json")
    ap.add_argument("--states", default="states.csv")
    ap.add_argument("--scaler", default="scaler.json")
    ap.add_argument("--device", choices=["cpu","cuda","mps"], default="cpu")
    ap.add_argument("--H", type=int, default=40)
    ap.add_argument("--mode", choices=["sign","linear"], default="sign")
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--action-th", type=float, default=0.05)
    ap.add_argument("--hold-bars", type=int, default=8)
    ap.add_argument("--linear-scale", type=float, default=0.5)
    ap.add_argument("--outdir", default="runs/eval_watch")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Sanity
    if not os.path.exists(args.states):
        sys.exit(f"[error] states not found: {args.states}")
    if not os.path.exists(args.gate):
        sys.exit(f"[error] gate JSON not found: {args.gate}")

    # Discover actors (not used directly here; gate JSON already references names on disk)
    actors = find_actors(args.actors-dir if hasattr(args, "actors-dir") else args.actors_dir)

    # Build gate actions CSV
    gate_csv = os.path.join(args.outdir, f"gate_eval_{int(time.time())}.csv")
    run_gate(args.gate, args.states, args.device, gate_csv)

    # Backtest
    kpi, raw = backtest(args.states, gate_csv, args.H, args.mode, args.fee_bps,
                        action_th=(args.action_th if args.mode=="sign" else None),
                        hold_bars=args.hold_bars,
                        linear_scale=(args.linear_scale if args.mode=="linear" else None))

    # Append eval CSV
    eval_csv = os.path.join(args.outdir, "eval.csv")
    write_header = not os.path.exists(eval_csv)
    with open(eval_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts","H","mode","fee_bps","action_th","linear_scale","hold_bars",
            "equity_end","sharpe","maxdd","trades","hitrate","winrate","gate_csv"])
        if write_header: w.writeheader()
        w.writerow(dict(
            ts=int(time.time()), H=args.H, mode=args.mode, fee_bps=args.fee_bps,
            action_th=(args.action_th if args.mode=="sign" else ""),
            linear_scale=(args.linear_scale if args.mode=="linear" else ""),
            hold_bars=args.hold_bars,
            equity_end=kpi["equity_end"], sharpe=kpi["sharpe"], maxdd=kpi["maxdd"],
            trades=kpi["trades"], hitrate=kpi["hitrate"], winrate=kpi["winrate"],
            gate_csv=os.path.basename(gate_csv)
        ))

    # Human summary
    print("[eval]")
    print(f" gate_csv = {gate_csv}")
    print(f" equity_end={kpi['equity_end']}  Sharpe_like={kpi['sharpe']}  MaxDD={kpi['maxdd']}%")
    print(f" trades={kpi['trades']}  win%={kpi['winrate']}  hit%={kpi['hitrate']}")
