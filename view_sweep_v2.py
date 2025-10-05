#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, os, sys, glob
from pathlib import Path

REQ_COLS = ["equity_end", "Sharpe_like", "MaxDD", "trades", "exposure",
            "min_agree_k", "agree_eps", "min_mag", "action_thresh", "linear_scale",
            "hold_bars", "mode", "gate_csv"]

def to_float(x, default=0.0):
    try:
        if x is None or x == "": return default
        return float(x)
    except Exception:
        return default

def to_int(x, default=0):
    try:
        if x is None or x == "": return default
        return int(float(x))
    except Exception:
        return default

def score(row):
    eq = to_float(row.get("equity_end"), 1.0)
    sh = to_float(row.get("Sharpe_like"), 0.0)
    dd = to_float(row.get("MaxDD"), 0.0)
    ex = to_float(row.get("exposure"), 0.0)
    tr = to_int(row.get("trades"), 0)
    if tr == 0:
        return -1e9
    # composite: emphasize equity, then Sharpe; penalize large DD or >95% exposure
    s = (eq - 1.0) * 100.0 + 0.5 * sh - 0.1 * dd - (1.0 if ex > 95.0 else 0.0)
    return s

def latest_sweep_dir(base="runs"):
    # prefer 'sweep_*' then newest lexicographically
    paths = sorted(glob.glob(os.path.join(base, "sweep_*")), reverse=True)
    return Path(paths[0]) if paths else None

def ensure_results_path(sweep_path: Path):
    res = sweep_path / "results.csv"
    if not res.exists():
        sys.exit(f"[error] results.csv not found: {res}")
    return res

def read_results(res_csv: Path):
    with res_csv.open(newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        sys.exit(f"[error] results.csv is empty: {res_csv}")
    # normalize keys (if someone wrote weird header casing)
    rows_norm = []
    for r in rows:
        nr = {k.strip(): v for k, v in r.items()}
        # Backward compatibility: accept alternative keys if present
        if "MaxDD" not in nr and "maxdd" in nr: nr["MaxDD"] = nr["maxdd"]
        if "Sharpe_like" not in nr and "Sharpe" in nr: nr["Sharpe_like"] = nr["Sharpe"]
        if "gate_csv" not in nr and "gate" in nr: nr["gate_csv"] = nr["gate"]
        rows_norm.append(nr)
    return rows_norm

def print_top(rows, n):
    ranked = sorted(rows, key=score, reverse=True)
    n = min(n, len(ranked))
    print(f"=== TOP {n} by composite score ===  ({len(rows)} total rows)\n")
    print(" #  score   eq_end  Sharpe  MaxDD  trades  k  eps    mag    th/ls      hold  gate_csv")
    print("-" * 130)
    for i, r in enumerate(ranked[:n], 1):
        eq = to_float(r.get("equity_end"), 1.0)
        sh = to_float(r.get("Sharpe_like"), 0.0)
        dd = to_float(r.get("MaxDD"), 0.0)
        tr = to_int(r.get("trades"), 0)
        k  = r.get("min_agree_k", "")
        eps = r.get("agree_eps", "")
        mag = r.get("min_mag", "")
        th  = r.get("action_thresh", "")
        ls  = r.get("linear_scale", "")
        tl  = f"th={th}" if (th not in (None, "", "0")) else f"ls={ls}" if ls not in (None, "") else "-"
        hold = r.get("hold_bars", "")
        gate = r.get("gate_csv", "")
        print(f"{i:2d}  {score(r):6.3f}  {eq:.4f}  {sh:6.2f}  {dd:5.2f}%  {tr:6d}  {k:>1}  {str(eps):<5}  {str(mag):<5}  {tl:<9} {str(hold):>4}  {gate}")
    print()

def saturate_counts(gate_csv: Path):
    neg = pos = mid = 0
    try:
        with gate_csv.open() as f:
            header = next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                try:
                    a = float(parts[2])
                except Exception:
                    continue
                if a <= -0.99: neg += 1
                elif a >= 0.99: pos += 1
                else: mid += 1
    except FileNotFoundError:
        return None
    return neg, pos, mid

def list_gates(sweep_dir: Path, show_sat=False, limit=20):
    gdir = sweep_dir / "gates"
    gates = sorted(gdir.glob("gate_*.csv"))
    print(f"=== GATES ({len(gates)}) in {gdir} ===")
    for i, g in enumerate(gates[:limit], 1):
        if show_sat:
            sp = saturate_counts(g)
            if sp is None:
                print(f"{i:3d}. {g.name:<60}  [missing]")
            else:
                n1, p1, md = sp
                print(f"{i:3d}. {g.name:<60}  sat: neg≈-1 {n1:4d}  +1 {p1:4d}  mid {md:4d}")
        else:
            print(f"{i:3d}. {g.name}")
    if len(gates) > limit:
        print(f"... (+{len(gates) - limit} more)")
    print()

def find_log_for_gate(sweep_dir: Path, gate_name: str):
    tag = gate_name.replace("gate_", "").replace(".csv", "")
    # flexible matching
    cands = list(sweep_dir.glob(f"log_{tag}.txt"))
    if cands:
        return cands[0]
    # fallback: any log that contains all param tokens
    tokens = [t for t in tag.split("_") if t]
    for p in sorted(sweep_dir.glob("log_*.txt")):
        name = p.name.replace("log_", "").replace(".txt", "")
        if all(tok in name for tok in tokens):
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=None, help="path to runs/sweep_* folder; if omitted, auto-pick latest")
    ap.add_argument("--top", type=int, default=10, help="top K")
    ap.add_argument("--show-sat", action="store_true", help="show saturation for first 20 gates")
    ap.add_argument("--gate", default=None, help="gate file name (under sweep/gates) to inspect")
    ap.add_argument("--tail", type=int, default=30, help="tail lines for the gate log")
    args = ap.parse_args()

    sweep = Path(args.sweep).resolve() if args.sweep else latest_sweep_dir("runs")
    if not sweep or not sweep.exists():
        sys.exit("[error] could not locate a sweep folder; pass --sweep runs/sweep_* explicitly")

    res = ensure_results_path(sweep)
    rows = read_results(res)

    print(f"[info] sweep: {sweep}")
    print_top(rows, args.top)
    list_gates(sweep, show_sat=args.show_sat)

    if args.gate:
        g = (sweep / "gates" / args.gate)
        if not g.exists():
            sys.exit(f"[error] gate not found under {sweep/'gates'}: {args.gate}")
        sp = saturate_counts(g)
        if sp is None:
            print(f"[error] cannot open gate csv: {g}")
        else:
            n1, p1, md = sp
            print(f"=== {g.name} saturation ===")
            print(f"neg≈-1: {n1}  pos≈+1: {p1}  mid: {md}\n")
        log = find_log_for_gate(sweep, g.name)
        if log and log.exists():
            print(f"=== tail {args.tail} lines of {log.name} ===")
            with log.open() as f:
                lines = f.readlines()
            for ln in lines[-args.tail:]:
                sys.stdout.write(ln)
        else:
            print("[info] no matching log file found for that gate.")
