#!/usr/bin/env python3
# decide_from_latest.py
# Pull the latest state from parquet, normalize with scaler.json, run consensus gate, print action.
import argparse, subprocess, json, numpy as np, sys, os

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr.strip(), file=sys.stderr)
        sys.exit(p.returncode)
    return p.stdout.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # 1) get latest state (single CSV line)
    state_row = run([
        sys.executable, "latest_state_from_parquet.py",
        "--parquet", args.parquet
    ])
    # 2) run live gate with --state_row
    out = run([
        sys.executable, "consensus_gate_live.py",
        "--gate", args.gate,
        "--scaler", args.scaler,
        "--state_row", state_row,
        "--device", args.device
    ])
    # Print only the last line with the decision
    lines = [ln for ln in out.splitlines() if ln and ln[0].isdigit()]
    if not lines:
        print(out)
    else:
        print(lines[-1])

if __name__ == "__main__":
    main()
