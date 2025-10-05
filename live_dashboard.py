#!/usr/bin/env python3
import argparse, csv, os, sys, time, shutil, glob, re
from collections import namedtuple

# ---------- ANSI Palette ----------
def make_palette(bright=True, enabled=True):
    def c(code): 
        return f"\033[{code}m" if enabled else ""
    N = c(0)
    # base colors (dim/bright pairs)
    base = dict(
        R=("31", "91"),
        G=("32", "92"),
        Y=("33", "93"),
        B=("34", "94"),
        M=("35", "95"),
        C=("36", "96"),
        W=("37", "97"),
        K=("30", "90"),
    )
    def pick(key): 
        lo, hi = base[key]
        return c(hi if bright else lo)
    class C:
        reset = N
        bold  = c("1")
        dim   = c("2")
        inv   = c("7")
        R = pick("R"); G = pick("G"); Y = pick("Y")
        B = pick("B"); M = pick("M"); Cx = pick("C")
        W = pick("W"); K = pick("K")
        hdr = pick("Y")  # headers you liked (green boxes) => yellow
        ok  = pick("G"); warn = pick("Y"); bad = pick("R")
        cold = pick("C")
    return C

# ---------- Helpers ----------
def read_results_csv(path):
    rows = []
    if not os.path.exists(path): 
        return rows
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            # normalize numeric fields
            def ffloat(k, default=0.0):
                v = r.get(k, "")
                try: return float(v)
                except: return default
            r["_score"]  = ffloat("score")
            r["_eq_end"] = ffloat("equity_end")
            r["_sharpe"] = ffloat("Sharpe_like")
            r["_maxdd"]  = ffloat("MaxDD")
            r["_trades"] = int(ffloat("trades", 0))
            # optional fields may or may not exist
            r["_winp"]   = ffloat("win%") if "win%" in r else None
            r["_hitp"]   = ffloat("hit%") if "hit%" in r else None
            rows.append(r)
    rows.sort(key=lambda x: x["_score"], reverse=True)
    return rows

def latest_file(globpat):
    files = glob.glob(globpat)
    if not files: return None
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]

def compute_saturation(csv_path):
    neg1 = pos1 = mid = 0
    if not csv_path or not os.path.exists(csv_path): 
        return (0,0,0)
    with open(csv_path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        # expect: row_idx,agree,action,details  (action is col 2)
        for row in rdr:
            try:
                a = float(row[2])
            except:
                continue
            if a <= -0.99: neg1 += 1
            elif a >= 0.99: pos1 += 1
            else: mid += 1
    return (neg1, pos1, mid)

def term_width():
    try:
        return shutil.get_terminal_size().columns
    except:
        return 120

def bar():
    return "-" * min(120, term_width())

# --- Color thresholds for the KPIs you care about ---
def color_value(C, name, val):
    """Return (colored_text, plain_val_str)."""
    # where "val" is numeric, pick color by thresholds
    label = f"{C.W}{name}{C.reset}"
    try:
        x = float(val)
    except:
        return f"{label}={val}", str(val)

    if name in ("trade_winrate", "per_bar_hitrate"):
        # percent in [0,100]
        col = C.bad
        if x >= 55: col = C.ok
        elif x >= 50: col = C.warn
        return f"{label}={col}{x:.2f}%{C.reset}", f"{x:.2f}%"

    if name == "equity_end":
        # >1 good
        if x >= 1.01: col = C.ok
        elif x >= 1.0: col = C.warn
        else: col = C.bad
        return f"{label}={col}{x:.4f}{C.reset}", f"{x:.4f}"

    if name == "Sharpe_like":
        if x >= 0.3: col = C.ok
        elif x >= 0.1: col = C.warn
        else: col = C.bad
        return f"{label}={col}{x:.2f}{C.reset}", f"{x:.2f}"

    if name == "MaxDD":
        # lower is better (input as percent)
        if x <= 0.5: col = C.ok
        elif x <= 1.0: col = C.warn
        else: col = C.bad
        return f"{label}={col}{x:.2f}%{C.reset}", f"{x:.2f}%"

    if name == "exposure":
        # informational
        return f"{label}={C.cold}{x:.2f}%{C.reset}", f"{x:.2f}%"

    if name in ("avg_turnover",):
        return f"{label}={C.cold}{x:.4f}/bar{C.reset}", f"{x:.4f}/bar"

    if name in ("avg_trade_pnl","median_trade_pnl"):
        # faint color
        col = C.cold
        return f"{label}={col}{x:.6f}{C.reset}", f"{x:.6f}"

    # default
    return f"{label}={x}", str(x)

def color_pct(C, x):
    # used for top table mild coloring on win%/hit%
    if x is None: return "-"
    if x >= 55: col = C.ok
    elif x >= 50: col = C.warn
    else: col = C.bad
    return f"{col}{x:.2f}%{C.reset}"

# ---------- Renderers ----------
def print_header(C, txt):
    print(f"{C.hdr}{txt}{C.reset}")

def print_top_table(C, results, topn=12):
    print_header(C, f"-- TOP {topn} (composite score) from results.csv  |  total={len(results)} --")
    hdr = f" #   score   eq_end  Sharpe  MaxDD   trades   win%    hit%   k  eps  mag  th/ls  hold   gate_csv"
    print(hdr)
    print("-"*len(hdr))
    for i, r in enumerate(results[:topn], 1):
        winp = color_pct(C, r.get("_winp"))
        hitp = color_pct(C, r.get("_hitp"))
        k    = r.get("min_agree_k","")
        eps  = r.get("agree_eps","")
        mag  = r.get("min_mag","")
        thls = r.get("action_thresh","") or r.get("th/ls","")
        hold = r.get("hold_bars","")
        gate = r.get("gate_csv","")
        line = (
            f"{i:>2}  "
            f"{r['_score']:>5.3f}  "
            f"{r['_eq_end']:>6.4f}  "
            f"{r['_sharpe']:>6.2f}  "
            f"{r['_maxdd']:>5.2f}%  "
            f"{r['_trades']:>6d}  "
            f"{winp:>7}  {hitp:>7}  "
            f"{k:>2}  {eps:>4}  {mag:>4}  {str(thls):>5}  {str(hold):>4}  {gate}"
        )
        print(line)
    print()

def print_latest_gate_saturation(C, gates_dir):
    latest_gate = latest_file(os.path.join(gates_dir, "*.csv"))
    base = os.path.basename(latest_gate) if latest_gate else "N/A"
    print_header(C, f"-- latest gate saturation  --  {base}")
    n1, p1, mid = compute_saturation(latest_gate) if latest_gate else (0,0,0)
    print(f"neg=-1: {C.ok}{n1}{C.reset}   pos=+1: {C.ok}{p1}{C.reset}   mid: {C.warn}{mid}{C.reset}")
    print()

def print_log_tail(C, sweep_dir, lines=25):
    latest_log = latest_file(os.path.join(sweep_dir, "log_*.txt"))
    base = os.path.basename(latest_log) if latest_log else "N/A"
    print_header(C, f"-- tail of {base} (last {lines}) --")
    if not latest_log:
        print(f"{C.warn}[no logs yet]{C.reset}\n")
        return
    with open(latest_log, "r", errors="ignore") as f:
        tail = f.readlines()[-lines:]
    # show tail *except* the raw 'rows=' KPI line; we'll render it colored later
    for t in tail:
        if t.strip().startswith("rows="):
            continue
        print(f"{C.dim}{t.rstrip()}{C.reset}")
    print()
    return latest_log

def parse_kpis_from_logs(sweep_dir):
    """Find most recent KPI line 'rows=... trade_winrate=... per_bar_hitrate=...' in any log."""
    logs = glob.glob(os.path.join(sweep_dir, "log_*.txt"))
    if not logs: return None
    logs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    pat = re.compile(
        r"rows=(?P<rows>\d+)\s+trades=(?P<trades>\d+)\s+trade_winrate=(?P<win>[\d\.]+)%\s+per_bar_hitrate=(?P<hit>[\d\.]+)%"
    )
    # other KPIs appear on next lines in most of your logs
    pat2 = re.compile(
        r"equity_end=(?P<eq>[\d\.]+)\s+Sharpe_like=(?P<sh>[\-\d\.]+)\s+MaxDD=(?P<dd>[\-\d\.]+)%\s+CAGR_like=(?P<cagr>[\-\d\.]+)%"
    )
    pat3 = re.compile(
        r"exposure=(?P<exp>[\-\d\.]+)%\s+avg_turnover=(?P<to>[\-\d\.]+)/bar\s+avg_trade_pnl=(?P<avgp>[\-\d\.]+)\s+median_trade_pnl=(?P<medp>[\-\d\.]+)"
    )
    for path in logs:
        rows = []
        with open(path, "r", errors="ignore") as f:
            rows = f.readlines()
        # scan backwards to find last KPI trio
        rows_rev = list(reversed(rows))
        got1=got2=got3=None
        for ln in rows_rev:
            if got3 is None:
                m3 = pat3.search(ln); 
                if m3: got3 = m3.groupdict()
                continue
        # continue scans separately to keep last matches
        for ln in rows_rev:
            if got2 is None:
                m2 = pat2.search(ln)
                if m2: got2 = m2.groupdict()
            if got1 is None:
                m1 = pat.search(ln)
                if m1: got1 = m1.groupdict()
            if got1 and got2: break
        if got1 and got2 and got3:
            return {
                "rows": int(got1["rows"]),
                "trades": int(got1["trades"]),
                "trade_winrate": float(got1["win"]),
                "per_bar_hitrate": float(got1["hit"]),
                "equity_end": float(got2["eq"]),
                "Sharpe_like": float(got2["sh"]),
                "MaxDD": float(got2["dd"]),
                "exposure": float(got3["exp"]),
                "avg_turnover": float(got3["to"]),
                "avg_trade_pnl": float(got3["avgp"]),
                "median_trade_pnl": float(got3["medp"]),
            }
    return None

def print_kpi_line(C, k):
    print_header(C, "-- backtest KPIs (colored) --")
    # first row (rows/trades/win/hit)
    parts1 = [
        f"rows={C.W}{k['rows']}{C.reset}",
        f"trades={C.W}{k['trades']}{C.reset}",
        color_value(C, "trade_winrate", k["trade_winrate"])[0],
        color_value(C, "per_bar_hitrate", k["per_bar_hitrate"])[0],
    ]
    print("  " + "  ".join(parts1))
    # second row (equity/sharpe/dd/cagr/exposure)
    parts2 = [
        color_value(C, "equity_end", k["equity_end"])[0],
        color_value(C, "Sharpe_like", k["Sharpe_like"])[0],
        color_value(C, "MaxDD", k["MaxDD"])[0],
        f"CAGR_like={C.cold}{k['exposure']*0+0:.2f}%{C.reset}".replace("0.00%","—"),  # placeholder if you don’t log CAGR here
        color_value(C, "exposure", k["exposure"])[0],
    ]
    print("  " + "  ".join(parts2))
    # third row (turnover/pnls)
    parts3 = [
        color_value(C, "avg_turnover", k["avg_turnover"])[0],
        color_value(C, "avg_trade_pnl", k["avg_trade_pnl"])[0],
        color_value(C, "median_trade_pnl", k["median_trade_pnl"])[0],
    ]
    print("  " + "  ".join(parts3))
    print()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="runs/sweep_* dir")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--refresh", type=float, default=2.0)
    ap.add_argument("--force-color", action="store_true")
    ap.add_argument("--dim-colors", action="store_true")
    args = ap.parse_args()

    sweep_dir = os.path.abspath(args.sweep)
    gates_dir = os.path.join(sweep_dir, "gates")
    results_csv = os.path.join(sweep_dir, "results.csv")
    now_csv = os.path.join(sweep_dir, "now.csv")  # optional live snapshot

    term_supports_color = sys.stdout.isatty()
    C = make_palette(bright=not args.dim_colors, enabled=(term_supports_color or args.force_color))

    try:
        while True:
            os.system("clear")
            print(f"{C.hdr}=== LIVE SWEEP DASHBOARD ==={C.reset}")
            print(f"{sweep_dir}")
            print()

            # now.csv (current trial) if present
            if os.path.exists(now_csv):
                print_header(C, "-- now.csv (current trial, last line) --")
                try:
                    with open(now_csv, "r") as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                        if lines:
                            print(lines[0])
                            if len(lines) > 1:
                                print(lines[-1])
                    print()
                except:
                    print(f"{C.warn}[now.csv unreadable]{C.reset}\n")
            else:
                print_header(C, "-- now.csv (not found) --")
                print()

            # Top table
            results = read_results_csv(results_csv)
            print_top_table(C, results, topn=args.top)

            # Latest gate saturation
            print_latest_gate_saturation(C, gates_dir)

            # Log tail (without raw KPI duplicate)
            latest_log = print_log_tail(C, sweep_dir, lines=25)

            # Colored KPI line from logs
            kpis = parse_kpis_from_logs(sweep_dir)
            if kpis:
                print_kpi_line(C, kpis)
            else:
                print(f"{C.warn}[no KPI line parsed yet]{C.reset}\n")

            print(f"{C.dim}[Ctrl+C to exit]{C.reset}")
            sys.stdout.flush()
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
