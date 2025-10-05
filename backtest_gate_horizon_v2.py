#!/usr/bin/env python3
# backtest_gate_horizon_v2.py
# - Always prints a result block (or a clear error) and flushes stdout
# - Extra header tolerance for gate CSV
# - Verbose mode shows row counts and a few sample actions/returns

import argparse, csv, statistics as st, sys

def read_states(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        hdr = [h.strip().lower() for h in (r.fieldnames or [])]
        need = ["mid","spread","imbalance","mom1","mom3","vol3"]
        if hdr != need:
            raise SystemExit(f"[error] states header mismatch: got {r.fieldnames}, want {need}")
        mids=[]
        ok=0; bad=0
        for row in r:
            try:
                mids.append(float(row["mid"])); ok+=1
            except:
                bad+=1
        if ok < 3:
            raise SystemExit(f"[error] states has too few usable rows: ok={ok}, bad={bad}")
        return mids, ok, bad

def read_gate(path):
    # Accept headers like: row_idx, agree, action, max_pair_diff, (min_q|details)
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit("[error] gate csv has no header")
        hdr = [h.strip().lower() for h in r.fieldnames]
        if "agree" not in hdr or "action" not in hdr:
            raise SystemExit(f"[error] gate header missing 'agree'/'action': {r.fieldnames}")
        acts=[]
        ok=0; bad=0
        for row in r:
            try:
                agree = int(row.get("agree", "0"))
                action = float(row.get("action","0"))
                acts.append(action if agree==1 else 0.0)
                ok+=1
            except:
                bad+=1
        if ok == 0:
            raise SystemExit("[error] gate has zero usable rows")
        return acts, ok, bad

def metrics(equity):
    rets=[equity[i]/equity[i-1]-1.0 for i in range(1,len(equity))]
    if not rets: return {"Sharpe_like": 0.0, "MaxDD": 0.0, "CAGR_like": 0.0}
    mu = st.mean(rets)
    sd = st.pstdev(rets) if len(rets)>1 else 0.0
    sharpe = (mu/sd) if sd>1e-12 else 0.0
    peak=equity[0]; maxdd=0.0
    for x in equity:
        if x>peak: peak=x
        dd = peak/x - 1.0
        if dd>maxdd: maxdd=dd
    return {"CAGR_like": equity[-1]/equity[0]-1.0, "Sharpe_like": sharpe, "MaxDD": maxdd}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--h", type=int, default=40)
    ap.add_argument("--mode", choices=["sign","linear"], default="sign")
    ap.add_argument("--action_thresh", type=float, default=0.08)
    ap.add_argument("--linear_scale", type=float, default=1.0)
    ap.add_argument("--hold_bars", type=int, default=8)
    ap.add_argument("--fee_bps", type=float, default=5.0)
    ap.add_argument("--verbose", action="store_true")
    args=ap.parse_args()

    mids, s_ok, s_bad = read_states(args.states)
    acts, g_ok, g_bad = read_gate(args.gate)

    n = min(len(acts), len(mids) - args.h)
    if n <= 0:
        print(f"[error] insufficient overlap: len(acts)={len(acts)} len(mids)={len(mids)} H={args.h}", flush=True)
        sys.exit(0)

    # H-step returns
    rets = [ (mids[t+args.h]-mids[t]) / mids[t] for t in range(n) ]

    # intended positions
    raw_pos=[0.0]*n
    if args.mode=="sign":
        th=args.action_thresh
        for t,a in enumerate(acts[:n]):
            raw_pos[t] = 1.0 if a>=th else (-1.0 if a<=-th else 0.0)
    else:
        sc=args.linear_scale
        for t,a in enumerate(acts[:n]):
            v = max(-1.0, min(1.0, a*sc))
            raw_pos[t]=v

    # min holding
    pos=[0.0]*n
    if n>0:
        pos[0]=raw_pos[0]
        last_change=0
        for t in range(1,n):
            if raw_pos[t] != pos[t-1] and (t - last_change) < args.hold_bars:
                pos[t] = pos[t-1]
            else:
                if raw_pos[t] != pos[t-1]:
                    last_change = t
                pos[t] = raw_pos[t]

    fee_rate = args.fee_bps/10000.0
    equity=[1.0]; trades=0; wins=0; prev_pos=0.0
    for t in range(n):
        turnover = abs(pos[t]-prev_pos)
        fee = turnover * fee_rate
        r = pos[t]*rets[t] - fee
        equity.append(equity[-1]*(1.0+r))
        if pos[t]!=prev_pos:
            trades += 1
        if pos[t]!=0.0 and r>0:
            wins += 1
        prev_pos = pos[t]

    m = metrics(equity)
    winrate = (wins/trades) if trades>0 else 0.0

    if args.verbose:
        print(f"[debug] states ok/bad={s_ok}/{s_bad}  gate ok/bad={g_ok}/{g_bad}  n={n}", flush=True)
        # sample few actions/returns
        import itertools
        aa = list(itertools.islice(acts, 0, 5))
        rr = list(itertools.islice(rets, 0, 5))
        print(f"[debug] sample acts={ [round(x,4) for x in aa] }  rets={ [round(x,6) for x in rr] }", flush=True)

    print("[backtest H]", flush=True)
    print(f" rows={n}  trades={trades}  winrate={winrate:.2%}", flush=True)
    print(f" H={args.h}  hold_bars={args.hold_bars}  mode={args.mode}  th={args.action_thresh if args.mode=='sign' else None}  scale={args.linear_scale if args.mode=='linear' else None}  fee_bps={args.fee_bps}", flush=True)
    print(f" equity_end={equity[-1]:.4f}  Sharpe_like={m.get('Sharpe_like',0):.2f}  MaxDD={m.get('MaxDD',0):.2%}  CAGR_like={m.get('CAGR_like',0):.2%}", flush=True)

if __name__=="__main__":
    main()
