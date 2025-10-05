#!/usr/bin/env python3
# compare_gates.py — grid search both gate files and print best config per file.
import argparse, subprocess, sys, itertools, re

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr.strip(), file=sys.stderr); sys.exit(p.returncode)
    return p.stdout.strip()

def parse_summary(out):
    # Extract the last two lines from backtest_gate_horizon.py output
    lines = [ln for ln in out.splitlines() if ln.strip()]
    hdr = next((ln for ln in lines if ln.startswith("[backtest H]")), "")
    met = next((ln for ln in lines if ln.startswith(" equity_end=") or "equity_end=" in ln), "")
    return hdr, met

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--gate_a", required=True, help="first gate csv (e.g., gate_out.csv)")
    ap.add_argument("--gate_b", required=True, help="second gate csv (e.g., gate_out_sign.csv)")
    ap.add_argument("--h", type=int, default=40)
    ap.add_argument("--fee_bps", type=float, default=5.0)
    args = ap.parse_args()

    th_grid = [0.03, 0.05, 0.08, 0.10, 0.15]
    hold_grid = [0, 4, 8, 12]

    best = {}
    for gate in [args.gate_a, args.gate_b]:
        best[gate] = None
        for th, hold in itertools.product(th_grid, hold_grid):
            out = run([sys.executable, "backtest_gate_horizon.py",
                      "--states", args.states, "--gate", gate,
                      "--h", str(args.h), "--mode", "sign",
                      "--action_thresh", str(th), "--hold_bars", str(hold),
                      "--fee_bps", str(args.fee_bps)])
            hdr, met = parse_summary(out)
            # crude score: prioritize Sharpe, then equity_end
            m = re.search(r"equity_end=([\d\.]+)\s+Sharpe_like=([-\d\.]+)\s+MaxDD=([\d\.%]+)", met)
            if not m: continue
            eq = float(m.group(1)); sh = float(m.group(2))
            score = (sh, eq)
            if best[gate] is None or score > best[gate][0]:
                best[gate] = ((sh, eq), th, hold, hdr, met)

    for gate in [args.gate_a, args.gate_b]:
        if best[gate] is None:
            print(f"[{gate}] no result"); continue
        (sh, eq), th, hold, hdr, met = best[gate]
        print(f"\n=== {gate} ===")
        print(f"best: th={th:.3f} hold={hold} | Sharpe_like={sh:.2f} equity_end={eq:.4f}")
        print(hdr); print(met)

if __name__ == "__main__":
    main()
