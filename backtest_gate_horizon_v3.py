#!/usr/bin/env python3
# backtest_gate_horizon_v3.py
# - H-step ahead PnL
# - Correct per-trade winrate (segment-based), per-bar hitrate
# - Exposure, turnover, avg trade PnL, slippage/fee model
# - Verbose sanity prints

import argparse, csv, statistics as st, sys
from math import isfinite

def read_states(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        hdr = [h.strip().lower() for h in (r.fieldnames or [])]
        need = ["mid","spread","imbalance","mom1","mom3","vol3"]
        if hdr != need:
            raise SystemExit(f"[error] states header mismatch: got {r.fieldnames}, want {need}")
        mids=[]; ok=0; bad=0
        for row in r:
            try:
                x = float(row["mid"])
                if isfinite(x): mids.append(x); ok+=1
                else: bad+=1
            except: bad+=1
        if ok < 3: raise SystemExit(f"[error] states too short: ok={ok}, bad={bad}")
        return mids, ok, bad

def read_gate(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames: raise SystemExit("[error] gate has no header")
        hdr = [h.strip().lower() for h in r.fieldnames]
        if "agree" not in hdr or "action" not in hdr:
            raise SystemExit(f"[error] gate header missing agree/action: {r.fieldnames}")
        acts=[]; ok=0; bad=0
        for row in r:
            try:
                agree = int(row.get("agree","0"))
                a = float(row.get("action","0"))
                acts.append(a if agree==1 else 0.0); ok+=1
            except: bad+=1
        if ok == 0: raise SystemExit("[error] gate has zero usable rows")
        return acts, ok, bad

def segment_trades(pos):
    """Return trade segments as (t_open, t_close, pos_sign). pos_sign is -1,0,+1."""
    segs=[]
    if not pos: return segs
    cur = pos[0]; t0 = 0
    for t in range(1, len(pos)):
        if pos[t] != cur:
            segs.append((t0, t-1, cur))
            t0 = t; cur = pos[t]
    segs.append((t0, len(pos)-1, cur))
    return segs

def metrics_time_series(equity):
    rets=[equity[i]/equity[i-1]-1.0 for i in range(1,len(equity))]
    if not rets: return {"Sharpe_like": 0.0, "MaxDD": 0.0, "CAGR_like": 0.0, "Ret_mean":0.0, "Ret_std":0.0}
    mu = st.mean(rets)
    sd = st.pstdev(rets) if len(rets)>1 else 0.0
    sharpe = mu/sd if sd>1e-12 else 0.0
    peak = equity[0]; maxdd=0.0
    for x in equity:
        peak = max(peak, x)
        dd = peak/x - 1.0
        maxdd = max(maxdd, dd)
    return {"CAGR_like": equity[-1]/equity[0]-1.0, "Sharpe_like": sharpe, "MaxDD": maxdd, "Ret_mean": mu, "Ret_std": sd}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--h", type=int, default=40)
    ap.add_argument("--mode", choices=["sign","linear"], default="sign")
    ap.add_argument("--action_thresh", type=float, default=0.08)
    ap.add_argument("--linear_scale", type=float, default=1.0)
    ap.add_argument("--hold_bars", type=int, default=8)
    ap.add_argument("--fee_bps", type=float, default=5.0, help="bps applied on position changes")
    ap.add_argument("--slip_bps", type=float, default=0.0, help="optional slip bps per trade entry/exit")
    ap.add_argument("--verbose", action="store_true")
    args=ap.parse_args()

    mids, s_ok, s_bad = read_states(args.states)
    acts, g_ok, g_bad = read_gate(args.gate)

    n = min(len(acts), len(mids) - args.h)
    if n <= 0:
        print(f"[error] insufficient overlap: len(acts)={len(acts)} len(mids)={len(mids)} H={args.h}", flush=True)
        sys.exit(0)

    # H-step forward returns
    rets = [ (mids[t+args.h]-mids[t]) / mids[t] for t in range(n) ]

    # Intended positions
    raw=[0.0]*n
    if args.mode=="sign":
        th=args.action_thresh
        for t,a in enumerate(acts[:n]):
            raw[t] = 1.0 if a>=th else (-1.0 if a<=-th else 0.0)
    else:
        sc=args.linear_scale
        for t,a in enumerate(acts[:n]):
            raw[t] = max(-1.0, min(1.0, a*sc))

    # Enforce minimum holding
    pos=[0.0]*n
    if n>0:
        pos[0]=raw[0]
        last_change=0
        for t in range(1,n):
            if raw[t] != pos[t-1] and (t - last_change) < args.hold_bars:
                pos[t] = pos[t-1]
            else:
                if raw[t] != pos[t-1]:
                    last_change = t
                pos[t] = raw[t]

    # Fees/slippage
    fee_rate = args.fee_bps/10000.0
    slip_rate = args.slip_bps/10000.0

    # Per-bar PnL and equity
    equity=[1.0]
    exposure_bars = sum(1 for p in pos if abs(p)>0.0)
    turnover = 0.0
    prev_pos = 0.0
    bar_pnl=[]
    for t in range(n):
        trn = abs(pos[t] - prev_pos)
        turnover += trn
        fee = trn * fee_rate
        r = pos[t]*rets[t] - fee
        equity.append(equity[-1]*(1.0 + r))
        bar_pnl.append(r)
        prev_pos = pos[t]

    # Per-trade (segment) stats
    segs = segment_trades(pos)
    trade_pnls=[]; entries=0
    for (lo, hi, sign) in segs:
        if sign == 0.0: continue
        # entry slip + exit slip (charge once per boundary)
        seg_turn = (1.0 if (lo==0 or pos[lo-1]==0.0) else 0.0) + (1.0 if (hi==n-1 or pos[hi+1]==0.0) else 0.0)
        slip = seg_turn * slip_rate
        seg_ret = sum(bar_pnl[lo:hi+1]) - slip
        trade_pnls.append(seg_ret)
        entries += 1

    m = metrics_time_series(equity)
    per_bar_hitrate = sum(1 for r,p in zip(bar_pnl,pos) if p!=0.0 and r>0) / max(1, sum(1 for p in pos if p!=0.0))
    trade_wins = sum(1 for x in trade_pnls if x>0)
    trade_winrate = trade_wins / max(1, len(trade_pnls))
    avg_trade = (st.mean(trade_pnls) if trade_pnls else 0.0)
    med_trade = (st.median(trade_pnls) if trade_pnls else 0.0)
    exposure = exposure_bars / n
    avg_turnover = turnover / n

    if args.verbose:
        print(f"[debug] states ok/bad={s_ok}/{s_bad} gate ok/bad={g_ok}/{g_bad} n={n}", flush=True)
        print(f"[debug] sample acts={ [round(x,4) for x in acts[:5]] } rets={ [round(x,6) for x in rets[:5]] }", flush=True)

    print("[backtest H]", flush=True)
    print(f" rows={n}  trades={len(trade_pnls)}  trade_winrate={trade_winrate:.2%}  per_bar_hitrate={per_bar_hitrate:.2%}", flush=True)
    print(f" H={args.h} hold_bars={args.hold_bars} mode={args.mode} th={args.action_thresh if args.mode=='sign' else None} scale={args.linear_scale if args.mode=='linear' else None}", flush=True)
    print(f" equity_end={equity[-1]:.4f} Sharpe_like={m['Sharpe_like']:.2f} MaxDD={m['MaxDD']:.2%} CAGR_like={m['CAGR_like']:.2%}", flush=True)
    print(f" exposure={exposure:.2%} avg_turnover={avg_turnover:.4f}/bar avg_trade_pnl={avg_trade:.6f} median_trade_pnl={med_trade:.6f}", flush=True)

if __name__=="__main__":
    main()
