#!/usr/bin/env python3
# backtest_gate_from_states.py
# Aligns gate_out.csv with states.csv by row index and simulates PnL on next-step mid returns.
# Position can be "sign" (±1/0 using threshold) or "linear" (position = clipped action).
# Fees charged on position changes (turnover * fee_bps).
#
# Usage:
#   python3 backtest_gate_from_states.py --states states.csv --gate gate_out.csv \
#       --mode sign --action_thresh 0.05 --fee_bps 1.0
#
import argparse, csv, math, statistics as st

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
    mean = st.mean(rets)
    sd   = st.pstdev(rets) if len(rets)>1 else 0.0
    sharpe = mean/sd if sd>1e-12 else 0.0
    peak=equity[0]; dd=0.0; maxdd=0.0
    for x in equity:
        peak=max(peak,x); dd=peak/x-1.0; maxdd=max(maxdd,dd)
    return {"CAGR_like": (equity[-1]/equity[0]-1.0), "Sharpe_like": sharpe, "MaxDD": maxdd,
            "Ret_mean": mean, "Ret_std": sd}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--mode", choices=["sign","linear"], default="sign")
    ap.add_argument("--action_thresh", type=float, default=0.05, help="sign mode: trade if |a|>=th")
    ap.add_argument("--linear_scale", type=float, default=1.0, help="linear mode: pos = clip(a*scale, -1, 1)")
    ap.add_argument("--fee_bps", type=float, default=1.0, help="per-change bps")
    args=ap.parse_args()

    mids=read_states(args.states)
    acts=read_gate(args.gate)
    n=min(len(acts), len(mids)-1)  # need t and t+1
    acts=acts[:n]; mids=mids[:n+1]

    # returns on mid
    rets=[(mids[t+1]-mids[t])/mids[t] for t in range(n)]

    # build positions
    pos=[0.0]*n
    if args.mode=="sign":
        for t,a in enumerate(acts):
            pos[t]= 1.0 if a>=args.action_thresh else (-1.0 if a<=-args.action_thresh else 0.0)
    else:
        for t,a in enumerate(acts):
            v = max(-1.0, min(1.0, a*args.linear_scale))
            pos[t]=v

    # fee on position changes
    fee_rate=args.fee_bps/10000.0
    pnl=[]; equity=[1.0]
    prev_pos=0.0
    wins=0; trades=0
    for t in range(n):
        # turnover fee when position changes
        turn=abs(pos[t]-prev_pos)
        fee = turn*fee_rate
        r   = pos[t]*rets[t] - fee
        pnl.append(r)
        equity.append(equity[-1]*(1.0+r))
        if pos[t]!=0.0:
            trades+=1
            if r>0: wins+=1
        prev_pos=pos[t]

    m=metrics(equity)
    winrate = (wins/trades) if trades>0 else 0.0
    print("[backtest]")
    print(f" rows={n}  trades={trades}  winrate={winrate:.2%}")
    print(f" mode={args.mode}  th={args.action_thresh if args.mode=='sign' else None}  scale={args.linear_scale if args.mode=='linear' else None}  fee_bps={args.fee_bps}")
    print(f" equity_end={equity[-1]:.4f}  Sharpe_like={m.get('Sharpe_like',0):.2f}  MaxDD={m.get('MaxDD',0):.2%}  CAGR_like={m.get('CAGR_like',0):.2%}")

if __name__=="__main__":
    main()
