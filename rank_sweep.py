#!/usr/bin/env python3
import argparse, csv, os, sys, glob
from collections import namedtuple

Row = namedtuple("Row", [
    "mode","H","fee_bps","min_agree_k","agree_eps","min_mag",
    "action_thresh","linear_scale","hold_bars","rows","trades",
    "trade_winrate","per_bar_hitrate","equity_end","Sharpe_like",
    "MaxDD","CAGR_like","exposure","avg_turnover","avg_trade_pnl",
    "median_trade_pnl","saturation_neg1","saturation_pos1","saturation_mid","gate_csv","tag"
])

def parse_float(x):
    try:
        return float(x)
    except:
        return None

def pct_to_frac(v):
    """
    Accepts:
      - '57.14%'         -> 0.5714
      - '57.14' (0..100) -> 0.5714
      - 0.5714 (0..1)    -> 0.5714
      - 57.14 (0..100)   -> 0.5714
    Returns fraction in [0,1] or None.
    """
    if v is None or v == "":
        return None
    if isinstance(v, str):
        s = v.strip()
        if s.endswith("%"):
            num = parse_float(s[:-1])
            return None if num is None else num/100.0
        num = parse_float(s)
        if num is None:
            return None
    else:
        num = v
    # If it's obviously 0..1 already
    if 0.0 <= num <= 1.0:
        return num
    # If it looks like a percent (0..100], scale down
    if 0.0 <= num <= 100.0:
        return num/100.0
    # Out of range -> None
    return None

def load_results(sweep_dir):
    res = []
    path = os.path.join(sweep_dir, "results.csv")
    if not os.path.exists(path):
        sys.exit(f"[error] results.csv not found in {sweep_dir}")
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        if not rows:
            sys.exit("[error] results.csv is empty")
        for d in rows:
            tag = d.get("tag") or (d.get("gate_csv","").replace("gate_","").replace(".csv",""))
            # numeric fields
            H = parse_float(d.get("H"))
            fee_bps = parse_float(d.get("fee_bps"))
            min_agree_k = parse_float(d.get("min_agree_k"))
            agree_eps = parse_float(d.get("agree_eps"))
            min_mag = parse_float(d.get("min_mag"))
            action_thresh = parse_float(d.get("action_thresh"))
            linear_scale = parse_float(d.get("linear_scale"))
            hold_bars = parse_float(d.get("hold_bars"))
            rows_cnt = parse_float(d.get("rows"))
            trades = parse_float(d.get("trades"))
            equity_end = parse_float(d.get("equity_end"))
            Sharpe_like = parse_float(d.get("Sharpe_like"))
            MaxDD = pct_to_frac(d.get("MaxDD"))
            CAGR_like = pct_to_frac(d.get("CAGR_like"))
            exposure = pct_to_frac(d.get("exposure"))
            avg_turnover = parse_float(d.get("avg_turnover"))
            avg_trade_pnl = parse_float(d.get("avg_trade_pnl"))
            median_trade_pnl = parse_float(d.get("median_trade_pnl"))
            sat_n1 = parse_float(d.get("saturation_neg1") or 0)
            sat_p1 = parse_float(d.get("saturation_pos1") or 0)
            sat_mid = parse_float(d.get("saturation_mid") or 0)
            trade_winrate = pct_to_frac(d.get("trade_winrate"))
            per_bar_hitrate = pct_to_frac(d.get("per_bar_hitrate"))
            # coerce ints where appropriate
            H = int(H) if H is not None else None
            min_agree_k = int(min_agree_k) if min_agree_k is not None else None
            hold_bars = int(hold_bars) if hold_bars is not None else None
            rows_cnt = int(rows_cnt) if rows_cnt is not None else None
            trades = int(trades) if trades is not None else 0
            sat_n1 = int(sat_n1) if sat_n1 is not None else 0
            sat_p1 = int(sat_p1) if sat_p1 is not None else 0
            sat_mid = int(sat_mid) if sat_mid is not None else 0
            res.append(Row(
                mode=d.get("mode","sign"),
                H=H,
                fee_bps=fee_bps or 0.0,
                min_agree_k=min_agree_k or 0,
                agree_eps=agree_eps or 0.0,
                min_mag=min_mag,
                action_thresh=action_thresh,
                linear_scale=linear_scale,
                hold_bars=hold_bars,
                rows=rows_cnt,
                trades=trades,
                trade_winrate=trade_winrate,
                per_bar_hitrate=per_bar_hitrate,
                equity_end=equity_end or 1.0,
                Sharpe_like=Sharpe_like or 0.0,
                MaxDD=MaxDD or 0.0,
                CAGR_like=CAGR_like or 0.0,
                exposure=exposure or 0.0,
                avg_turnover=avg_turnover or 0.0,
                avg_trade_pnl=avg_trade_pnl or 0.0,
                median_trade_pnl=median_trade_pnl or 0.0,
                saturation_neg1=sat_n1,
                saturation_pos1=sat_p1,
                saturation_mid=sat_mid,
                gate_csv=d.get("gate_csv",""),
                tag=tag,
            ))
    return res

def color_enabled(force):
    return True if force else sys.stdout.isatty()

def cfmt(enabled):
    class C:
        def __init__(self, on):
            if not on:
                self.end=self.green=self.red=self.yellow=self.blue=self.cyan=self.magenta=""
                return
            self.end="\033[0m"
            self.green="\033[38;5;82m"
            self.red="\033[38;5;196m"
            self.yellow="\033[38;5;220m"
            self.blue="\033[38;5;39m"
            self.cyan="\033[38;5;51m"
            self.magenta="\033[38;5;207m"
    return C(enabled)

def fmt_pct(frac):
    return "-" if frac is None else f"{frac*100:.2f}%"

def score_row(r, w_eq=100.0, w_sh=2.0, w_dd=50.0, w_exp=0.5):
    # Higher is better: equity↑, Sharpe↑, drawdown↓, exposure↓ (small)
    return (w_eq*(r.equity_end-1.0)) + (w_sh*r.Sharpe_like) - (w_dd*r.MaxDD) - (w_exp*r.exposure)

def parse_best_flags(r):
    k = r.min_agree_k
    eps = r.agree_eps
    mag = r.min_mag
    th  = r.action_thresh
    hold = r.hold_bars
    flags = ["--consensus-mode","sign","--min-agree-k",str(k),"--agree-eps",str(eps)]
    if mag is not None:
        flags += ["--min-mag", str(mag)]
    if th is not None:
        flags += ["--buy_th", str(th), "--sell_th", str(th)]
    if hold is not None:
        flags += ["--hold_bars", str(hold)]
    return flags

def explain_filter_fail(rows, args):
    reasons = {
        "min_trades": 0,
        "min_equity": 0,
        "min_sharpe": 0,
        "max_dd": 0,
    }
    for r in rows:
        if r.trades is None or r.trades < args.min_trades: reasons["min_trades"] += 1
        if r.equity_end is None or r.equity_end < args.min_equity: reasons["min_equity"] += 1
        if r.Sharpe_like is None or r.Sharpe_like < args.min_sharpe: reasons["min_sharpe"] += 1
        if r.MaxDD is None or (r.MaxDD*100.0) > args.max_dd: reasons["max_dd"] += 1
    return reasons

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="runs/sweep_* or quoted glob")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--min-trades", type=int, default=6)
    ap.add_argument("--min-equity", type=float, default=1.0)
    ap.add_argument("--min-sharpe", type=float, default=0.0)
    ap.add_argument("--max-dd", type=float, default=1.5, help="max drawdown in percent")
    ap.add_argument("--force-color", action="store_true")
    args = ap.parse_args()

    # Expand globs, pick latest non-empty
    dirs = sorted([d for g in glob.glob(args.sweep) for d in ([g] if os.path.isdir(g) else [])])
    if not dirs:
        sys.exit(f"[error] no sweep dirs match: {args.sweep}")
    chosen = None; rows = None
    for d in reversed(dirs):
        try:
            rows = load_results(d)
            chosen = d
            break
        except SystemExit:
            continue
    if chosen is None:
        sys.exit("[error] found no non-empty results.csv in provided sweep(s)")

    C = cfmt(color_enabled(args.force_color))
    print(f"[sweep] {chosen}  (rows={len(rows)})")

    # Filter with guardrails
    filt = []
    for r in rows:
        if r.trades is None or r.trades < args.min_trades: continue
        if r.equity_end is None or r.equity_end < args.min_equity: continue
        if r.Sharpe_like is None or r.Sharpe_like < args.min_sharpe: continue
        if r.MaxDD is None or (r.MaxDD*100.0) > args.max_dd: continue
        filt.append(r)

    if not filt:
        why = explain_filter_fail(rows, args)
        print(f"{C.yellow}[warn]{C.end} no rows passed filters "
              f"(min_trades={args.min_trades}, min_equity={args.min_equity}, "
              f"min_sharpe={args.min_sharpe}, max_dd={args.max_dd}%)")
        print(f" breakdown: trades<{args.min_trades}: {why['min_trades']}, "
              f"equity<{args.min_equity}: {why['min_equity']}, "
              f"sharpe<{args.min_sharpe}: {why['min_sharpe']}, "
              f"dd>{args.max_dd}%: {why['max_dd']}")
        sys.exit(0)

    ranked = sorted(filt, key=lambda r: (-score_row(r), -r.Sharpe_like, r.MaxDD, -r.equity_end))[:args.top]

    # Pretty table with *only* the important KPIs
    print("=== Top Results (filtered & ranked) ===")
    print("Tag | Trades | "
          f"{C.cyan}Winrate{C.end} | {C.cyan}HitRate{C.end} | "
          f"{C.green}EquityEnd{C.end} | {C.green}Sharpe{C.end} | {C.red}MaxDD{C.end} | "
          "H | k | eps | mag | th | hold")
    print("-"*98)

    for r in ranked:
        print(
            f"{r.tag} | {r.trades} | "
            f"{C.cyan}{fmt_pct(r.trade_winrate)}{C.end} | "
            f"{C.cyan}{fmt_pct(r.per_bar_hitrate)}{C.end} | "
            f"{C.green}{r.equity_end:.4f}{C.end} | "
            f"{C.green}{r.Sharpe_like:.2f}{C.end} | "
            f"{C.red}{r.MaxDD*100:.2f}%{C.end} | "
            f"{r.H} | {r.min_agree_k} | {r.agree_eps} | {r.min_mag} | {r.action_thresh} | {r.hold_bars}"
        )

    best = ranked[0]
    print("\n=== Recommended Params (no live execution) ===")
    print(f"Gate CSV: {best.gate_csv}")
    print(f"Tag     : {best.tag}")
    print(f"KPI     : Win {fmt_pct(best.trade_winrate)} | Hit {fmt_pct(best.per_bar_hitrate)} | "
          f"Eq {best.equity_end:.4f} | Sharpe {best.Sharpe_like:.2f} | MaxDD {best.MaxDD*100:.2f}%")

    flags = " ".join(parse_best_flags(best))
    print("\n# When you LATER want to test with these thresholds (no live now):")
    print("BACKTEST (sign mode):")
    print(f"  python3 backtest_gate_horizon_v3.py \\")
    print(f"    --states states.csv --gate {os.path.join('runs', os.path.basename(chosen), 'gates', best.gate_csv)} \\")
    print(f"    --h {best.H} --mode sign --action_thresh {best.action_thresh} --hold_bars {best.hold_bars} --fee_bps {best.fee_bps}")

    print("\n# If/when you want consensus flags for later (again, do not run live yet):")
    print(f"  {flags}")

if __name__ == "__main__":
    main()
