#!/usr/bin/env python3
# tune_backtest_raw.py — robust grid search that prints FULL stdout from backtest for each combo.
import argparse, subprocess, sys, itertools

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("".join([
            "\n[CMD]\n", " ".join(cmd),
            "\n[STDERR]\n", p.stderr.strip(), "\n"
        ]), file=sys.stderr)
        sys.exit(p.returncode)
    return p.stdout.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--h", type=int, default=40)
    ap.add_argument("--fee_bps", type=float, default=5.0)
    ap.add_argument("--mode", choices=["sign","linear"], default="sign")
    args = ap.parse_args()

    if args.mode == "sign":
        th_grid   = [0.03, 0.05, 0.08, 0.10, 0.15]
        hold_grid = [0, 4, 8, 12]
        grid = [(th, hold) for th in th_grid for hold in hold_grid]
        for th, hold in grid:
            cmd = [
                sys.executable, "backtest_gate_horizon.py",
                "--states", args.states,
                "--gate", args.gate,
                "--h", str(args.h),
                "--mode", "sign",
                "--action_thresh", str(th),
                "--hold_bars", str(hold),
                "--fee_bps", str(args.fee_bps),
            ]
            out = run(cmd)
            print(f"\n>>> th={th:.3f} hold={hold}")
            print(out)
    else:
        scale_grid = [0.5, 0.75, 1.0, 1.5, 2.0]
        hold_grid  = [0, 4, 8, 12]
        for sc, hold in [(s,h) for s in scale_grid for h in hold_grid]:
            cmd = [
                sys.executable, "backtest_gate_horizon.py",
                "--states", args.states,
                "--gate", args.gate,
                "--h", str(args.h),
                "--mode", "linear",
                "--linear_scale", str(sc),
                "--hold_bars", str(hold),
                "--fee_bps", str(args.fee_bps),
            ]
            out = run(cmd)
            print(f"\n>>> scale={sc:.2f} hold={hold}")
            print(out)

if __name__ == "__main__":
    main()
