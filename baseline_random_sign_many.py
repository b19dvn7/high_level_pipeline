#!/usr/bin/env python3
import argparse, subprocess, statistics, re

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=50)
ap.add_argument("--states", default="states.csv")
ap.add_argument("--H", type=int, default=40)
ap.add_argument("--hold_bars", type=int, default=8)
ap.add_argument("--fee_bps", type=float, default=5.0)
args = ap.parse_args()

equities, sharpes, trades = [], [], []

for i in range(args.runs):
    cmd = [
        "./baseline_random_sign.py",
        "--states", args.states,
        "--H", str(args.H),
        "--hold-bars", str(args.hold_bars),
        "--fee-bps", str(args.fee_bps),
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

    m_eq = re.search(r"equity_end=([0-9.]+)", out)
    m_sh = re.search(r"Sharpe_like=([\-0-9.]+)", out)
    m_tr = re.search(r"trades=(\d+)", out)
    if m_eq and m_sh and m_tr:
        equities.append(float(m_eq.group(1)))
        sharpes.append(float(m_sh.group(1)))
        trades.append(int(m_tr.group(1)))

def q(xs, p):
    if not xs: return float("nan")
    s = sorted(xs)
    i = max(0, min(len(s)-1, int(p*(len(s)-1))))
    return s[i]

def mean(xs): return (sum(xs)/len(xs)) if xs else float("nan")

print(f"[random baseline x{args.runs}]")
print(f"trades avg/med: {mean(trades):.2f}/{q(trades,0.50):.0f}")
print(
    "equity_end mean={:.4f}  median={:.4f}  p05={:.4f}  p95={:.4f}".format(
        mean(equities), q(equities,0.50), q(equities,0.05), q(equities,0.95)
    )
)
print(
    "Sharpe mean={:.2f}  median={:.2f}  p05={:.2f}  p95={:.2f}".format(
        mean(sharpes), q(sharpes,0.50), q(sharpes,0.05), q(sharpes,0.95)
    )
)
