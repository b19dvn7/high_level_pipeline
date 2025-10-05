#!/usr/bin/env python3
# backtest_linear_deadband.py — linear sizing with deadband on actions (|a|<db -> pos=0)
import argparse, csv, statistics as st

def read_states(p):
    with open(p, newline="") as f:
        r=csv.DictReader(f); mids=[float(row["mid"]) for row in r]
    return mids

def read_gate(p):
    acts=[]
    with open(p, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            agree=int(row["agree"]); a=float(row["action"])
            acts.append(a if agree==1 else 0.0)
    return acts

def metrics(eq):
    rets=[eq[i]/eq[i-1]-1 for i in range(1,len(eq))]
    if not rets: return 0,0,0
    m=st.mean(rets); s=st.pstdev(rets) or 1e-12
    sharpe=m/s
    peak=eq[0]; maxdd=0
    for x in eq:
        if x>peak: peak=x
        dd=peak/x-1
        if dd>maxdd: maxdd=dd
    return sharpe,maxdd,eq[-1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--h", type=int, default=40)
    ap.add_argument("--linear_scale", type=float, default=0.75)
    ap.add_argument("--deadband", type=float, default=0.05)
    ap.add_argument("--hold_bars", type=int, default=8)
    ap.add_argument("--fee_bps", type=float, default=2.0)
    args=ap.parse_args()

    mids=read_states(args.states)
    acts=read_gate(args.gate)
    n=min(len(acts), len(mids)-args.h)
    mids=mids[:n+args.h]; acts=acts[:n]
    rets=[(mids[t+args.h]-mids[t])/mids[t] for t in range(n)]

    # deadbanded linear position
    raw=[0.0]*n
    for t,a in enumerate(acts):
        v = a if abs(a)>=args.deadband else 0.0
        v = max(-1.0, min(1.0, v*args.linear_scale))
        raw[t]=v

    # min-hold on continuous pos (defer changes until hold satisfied)
    pos=[0.0]*n
    if n>0:
        pos[0]=raw[0]; last_change=0
        for t in range(1,n):
            if raw[t]!=pos[t-1] and (t-last_change)<args.hold_bars:
                pos[t]=pos[t-1]
            else:
                if raw[t]!=pos[t-1]: last_change=t
                pos[t]=raw[t]

    fee = args.fee_bps/10000.0
    eq=[1.0]; prev=0.0; trades=0
    for t in range(n):
        turn=abs(pos[t]-prev); r=pos[t]*rets[t] - turn*fee
        eq.append(eq[-1]*(1+r))
        if pos[t]!=prev: trades+=1
        prev=pos[t]

    sharpe,maxdd,eqend = metrics(eq)
    print("[backtest H-linear-deadband]")
    print(f" rows={n} trades={trades} db={args.deadband} scale={args.linear_scale} hold={args.hold_bars} fee_bps={args.fee_bps}")
    print(f" equity_end={eqend:.4f} Sharpe_like={sharpe:.2f} MaxDD={maxdd:.2%}")

if __name__=="__main__":
    main()
