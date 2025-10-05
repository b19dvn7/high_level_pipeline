#!/usr/bin/env python3
import argparse, csv, glob, os, sys
from typing import List

ANSI = {"reset":"\033[0m","bd":"\033[1m","r":"\033[31m","g":"\033[32m","y":"\033[33m","b":"\033[34m","m":"\033[35m","c":"\033[36m","w":"\033[37m"}
def use_color(enabled: bool) -> bool: return enabled and sys.stdout.isatty()
def C(txt, code, enabled=True): return f"{ANSI.get(code,'')}{txt}{ANSI['reset']}" if enabled else str(txt)

COLS = ["mode","H","fee_bps","min_agree_k","agree_eps","min_mag","action_thresh","linear_scale","hold_bars",
        "rows","trades","trade_winrate","per_bar_hitrate","equity_end","Sharpe_like","MaxDD","CAGR_like",
        "exposure","avg_turnover","avg_trade_pnl","median_trade_pnl","saturation_neg1","saturation_pos1",
        "saturation_mid","gate_csv"]

def count_data_rows(path:str)->int:
    try:
        with open(path, 'r', newline='') as f:
            n=0
            for i,_ in enumerate(f): n=i+1
        return max(0, n-1)  # minus header
    except: return 0

def resolve_candidates(sweep_arg: str):
    """Return list of candidate results.csv paths, newest first."""
    cands=[]
    def add(p):
        if os.path.isfile(p): cands.append((os.path.getmtime(p), p))
    if sweep_arg.endswith(".csv") and os.path.isfile(sweep_arg):
        add(sweep_arg)
    elif os.path.isdir(sweep_arg):
        p=os.path.join(sweep_arg,"results.csv"); 
        if os.path.isfile(p): add(p)
    else:
        for m in glob.glob(sweep_arg):
            if os.path.isdir(m):
                p=os.path.join(m,"results.csv"); 
                if os.path.isfile(p): add(p)
            elif m.endswith(".csv") and os.path.isfile(m):
                add(m)
    cands.sort(reverse=True)
    return [p for _,p in cands]

def resolve_results_csv(sweep_arg: str, prefer_non_empty=True):
    cands = resolve_candidates(sweep_arg)
    if not cands:
        raise FileNotFoundError(f"No results.csv found under: {sweep_arg}")
    if not prefer_non_empty:
        return cands[0]
    for p in cands:
        if count_data_rows(p) > 0:
            return p
    # none had data; return newest anyway
    return cands[0]

def read_results(results_csv: str) -> List[dict]:
    with open(results_csv, newline="") as f:
        rdr = csv.DictReader(f); rows=[]
        for r in rdr:
            for k in COLS:
                if k not in r: r[k]=""
            rows.append(r)
        return rows

def ffloat(x, dflt=None):
    try: return float(x)
    except: return dflt

def highlight_kpis(r: dict):
    tw=ffloat(r.get("trade_winrate")); ph=ffloat(r.get("per_bar_hitrate"))
    eq=ffloat(r.get("equity_end")); sh=ffloat(r.get("Sharpe_like")); dd=ffloat(r.get("MaxDD"))
    tw_c="r";   tw_c="y" if tw is not None and tw>=50 else tw_c;   tw_c="g" if tw is not None and tw>=60 else tw_c
    ph_c="r";   ph_c="y" if ph is not None and ph>=50 else ph_c;   ph_c="g" if ph is not None and ph>=55 else ph_c
    eq_c="r";   eq_c="y" if eq is not None and eq>=1.00 else eq_c; eq_c="g" if eq is not None and eq>=1.01 else eq_c
    sh_c="r";   sh_c="y" if sh is not None and sh>=0.1 else sh_c;  sh_c="g" if sh is not None and sh>=0.5 else sh_c
    dd_c="r";   dd_c="y" if dd is not None and dd<=1.0 else dd_c;  dd_c="g" if dd is not None and dd<=0.5 else dd_c
    return {"tw":tw_c,"ph":ph_c,"eq":eq_c,"sh":sh_c,"dd":dd_c}

def fmt_pct(x, p=2):
    v=ffloat(x); return f"{v:.{p}f}%" if v is not None else "-"

def fmt_num(x, p=4):
    v=ffloat(x); return f"{v:.{p}f}" if v is not None else "-"

def print_table(rows: List[dict], top: int, color_on: bool):
    print(C("=== Top Results ===","bd",color_on))
    hdr=["Tag","Trades","Winrate","HitRate","EquityEnd","Sharpe","MaxDD","H","k","eps","mag","th","hold"]
    print(" | ".join(hdr)); print("-"* (len(" | ".join(hdr))))
    for r in rows[:top]:
        tag=f"k{r['min_agree_k']}_eps{r['agree_eps']}_mag{r['min_mag']}_th{r['action_thresh']}_hold{r['hold_bars']}"
        K=highlight_kpis(r)
        line=" | ".join([
            tag, f"{r['trades']}",
            C(fmt_pct(r['trade_winrate']),K["tw"],color_on),
            C(fmt_pct(r['per_bar_hitrate']),K["ph"],color_on),
            C(fmt_num(r['equity_end'],4),K["eq"],color_on),
            C(fmt_num(r['Sharpe_like'],2),K["sh"],color_on),
            C(fmt_num(r['MaxDD'],2)+"%",K["dd"],color_on),
            r["H"], r["min_agree_k"], r["agree_eps"], r["min_mag"], r["action_thresh"], r["hold_bars"]
        ])
        print(line)
    print()

def print_gate_detail(results_csv: str, gate_name: str, color_on: bool):
    rows=read_results(results_csv); m=None
    for r in rows:
        if r.get("gate_csv")==gate_name: m=r; break
    if not m:
        print(C(f"[warn] gate not found: {gate_name}","y",color_on)); return
    K=highlight_kpis(m)
    print(C("=== Backtest KPI (selected gate) ===","bd",color_on))
    print(f"Gate           : {m['gate_csv']}")
    print(f"Mode/H/Fee     : {m['mode']}/H={m['H']}/fee={m['fee_bps']} bps")
    print(f"Consensus      : k={m['min_agree_k']} eps={m['agree_eps']} mag={m['min_mag']}")
    print(f"Exec           : th={m['action_thresh']} hold={m['hold_bars']}")
    print(f"Rows/Trades    : {m['rows']}/{m['trades']}")
    print("KPI            : "
          + f"Win {C(fmt_pct(m['trade_winrate']),K['tw'],color_on)} | "
          + f"Hit {C(fmt_pct(m['per_bar_hitrate']),K['ph'],color_on)} | "
          + f"Eq {C(fmt_num(m['equity_end']),K['eq'],color_on)} | "
          + f"Sharpe {C(fmt_num(m['Sharpe_like'],2),K['sh'],color_on)} | "
          + f"MaxDD {C(fmt_num(m['MaxDD'],2)+'%',K['dd'],color_on)}")
    print()

def print_saturation(results_csv: str, color_on: bool):
    rows=read_results(results_csv)
    if not rows:
        print(C("[empty results]","y",color_on)); return
    r=rows[-1]
    print(C("=== Latest Gate Saturation Snapshot ===","bd",color_on))
    print(f"Gate: {r.get('gate_csv','-')}")
    print(f"neg≈-1: {r.get('saturation_neg1','-')} | pos≈+1: {r.get('saturation_pos1','-')} | mid: {r.get('saturation_mid','-')}")
    print()

def main():
    ap=argparse.ArgumentParser(description="View sweep results (colored KPIs) with empty-safe fallback.")
    ap.add_argument("--sweep", required=True, help="Sweep dir, results.csv, or glob (e.g. 'runs/sweep_sign_*')")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--show-sat", action="store_true")
    ap.add_argument("--gate", help="Show KPI for a specific gate CSV")
    ap.add_argument("--force-color", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--allow-empty", action="store_true", help="Allow showing an empty results.csv")
    args=ap.parse_args()

    try:
        results_csv = resolve_results_csv(args.sweep, prefer_non_empty=not args.allow_empty)
    except FileNotFoundError as e:
        print(f"[error] {e}"); sys.exit(1)

    nrows = count_data_rows(results_csv)
    color_on = use_color(not args.no_color or args.force_color)

    print(C(f"[sweep] {os.path.dirname(results_csv) or '.'}  (rows={nrows})","bd",color_on))
    rows = read_results(results_csv)
    if nrows <= 0:
        print(C("[info] results.csv has no data rows yet.","y",color_on))
        print(C("      Tip: run another sweep, or pass a glob (e.g. --sweep 'runs/sweep_sign_*') and I’ll pick the latest non-empty.","y",color_on))
        sys.exit(0)

    # sort by equity_end desc, then Sharpe desc
    rows_sorted = sorted(rows, key=lambda r: (ffloat(r.get("equity_end"),1.0), ffloat(r.get("Sharpe_like"),0.0)), reverse=True)

    print_table(rows_sorted, args.top, color_on)
    if args.gate: print_gate_detail(results_csv, args.gate, color_on)
    if args.show_sat: print_saturation(results_csv, color_on)

if __name__=="__main__": main()
