#!/usr/bin/env python3
# backtest_gate_horizon.py
# Aligns gate_out.csv with states.csv by row index and simulates PnL on H-step ahead returns.
# Adds holding constraint to cut churn and realistic fees.
#
# Example:
#   python3 backtest_gate_horizon.py --states states.csv --gate gate_out.csv \
#       --h 40 --mode sign --action_thresh 0.08 --hold_bars 8 --fee_bps 5.0
#
import argparse, csv, statistics as st

def read_states(path):
    mids=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        need=["mid","spread","imbalance","mom1","mom3","vol3"]
        hdr=[h.strip().lower() for h in (r.fieldnames or [])]
        if hdr!=need:
            raise SystemExit(f"states header mismatch: got {r.fieldnames}, want {need}")
        for row in r:
            try: mids.append(float(row["mid"]))
            except: pass
    if len(mids)<3: raise SystemExit("not enough rows in states.csv")
    return mids

def read_gate(path):
    acts=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                agree=int(row["agree"])
                action=float(row["action"])
                acts.append(action if agree==1 else 0.0)
            except: pass
    if not acts: raise SystemExit("no rows in gate csv")
    return acts

def metrics(equity):
    rets=[equity[i]/equity[i-1]-1.0 for i in range(1,len(equity))]
    if not rets: return {}
    mu = st.mean(rets)
    sd = st.pstdev(rets) if len(rets)>1 else 0.0
    sharpe = (mu/sd) if sd>1e-12 else 0.0
    peak=equity[0]; maxdd=0.0
    for x in equity:
        if x>peak: peak=x
        dd = peak/x - 1.0
        if dd>maxdd: maxdd=dd
    return {"CAGR_like": equity[-1]/equity[0]-1.0, "Sharpe_like": sharpe, "MaxDD": maxdd, "Ret_mean": mu, "Ret_std": sd}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--h", type=int, default=40, help="forecast horizon (bars) to realize PnL")
    ap.add_argument("--mode", choices=["sign","linear"], default="sign")
    ap.add_argument("--action_thresh", type=float, default=0.08, help="sign mode: trade if |a|>=th")
    ap.add_argument("--linear_scale", type=float, default=1.0, help="linear mode: pos = clip(a*scale, -1, 1)")
    ap.add_argument("--hold_bars", type=int, default=8, help="minimum holding period (bars) before switching/flattening")
    ap.add_argument("--fee_bps", type=float, default=5.0, help="per-change bps")
    args=ap.parse_args()

    mids=read_states(args.states)
    acts=read_gate(args.gate)

    # Need t and t+H → truncate
    n = min(len(acts), len(mids) - args.h)
    acts = acts[:n]; mids = mids[:n + args.h]

    # Compute H-step returns
    rets = [ (mids[t+args.h] - mids[t]) / mids[t] for t in range(n) ]

    # Build intended position (raw) from actions
    raw_pos=[0.0]*n
    if args.mode=="sign":
        th=args.action_thresh
        for t,a in enumerate(acts):
            raw_pos[t] = 1.0 if a>=th else (-1.0 if a<=-th else 0.0)
    else:
        for t,a in enumerate(acts):
            v = max(-1.0, min(1.0, a*args.linear_scale))
            raw_pos[t]=v

    # Enforce min holding
    pos=[0.0]*n
    if n>0:
        pos[0]=raw_pos[0]
        last_change=0
        for t in range(1,n):
            if raw_pos[t] != pos[t-1] and (t - last_change) < args.hold_bars:
                pos[t] = pos[t-1]   # defer change until holding satisfied
            else:
                if raw_pos[t] != pos[t-1]:
                    last_change = t
                pos[t] = raw_pos[t]

    # PnL with fees on position changes
    fee_rate = args.fee_bps / 10000.0
    equity=[1.0]
    trades=0; wins=0
    prev_pos=0.0
    for t in range(n):
        turnover = abs(pos[t] - prev_pos)
        fee = turnover * fee_rate
        r = pos[t]*rets[t] - fee
        equity.append(equity[-1]*(1.0 + r))
        if pos[t]!=0.0 and pos[t]!=prev_pos:
            trades += 1
        if pos[t]!=0.0 and r>0: wins += 1
        prev_pos = pos[t]

    m = metrics(equity)
    winrate = (wins/trades) if trades>0 else 0.0
    print("[backtest H]")
    print(f" rows={n}  trades={trades}  winrate={winrate:.2%}")
    print(f" H={args.h}  hold_bars={args.hold_bars}  mode={args.mode}  th={args.action_thresh if args.mode=='sign' else None}  scale={args.linear_scale if args.mode=='linear' else None}  fee_bps={args.fee_bps}")
    print(f" equity_end={equity[-1]:.4f}  Sharpe_like={m.get('Sharpe_like',0):.2f}  MaxDD={m.get('MaxDD',0):.2%}  CAGR_like={m.get('CAGR_like',0):.2%}")
