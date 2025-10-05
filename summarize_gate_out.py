#!/usr/bin/env python3
# summarize_gate_out.py — prints quick stats for gate_out.csv
# CSV format expected: row_idx,agree,action,min_q
import argparse, csv, math, statistics as st

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--bins", type=int, default=11, help="histogram bins")
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "agree": int(row["agree"]),
                    "action": float(row["action"]),
                    "min_q": None if row.get("min_q","") in ("", "None") else float(row["min_q"]),
                })
            except Exception:
                pass

    n = len(rows)
    if n == 0:
        print("[error] no rows")
        return
    agree_cnt = sum(1 for x in rows if x["agree"] == 1)
    nonzero_cnt = sum(1 for x in rows if abs(x["action"]) > 1e-8)
    actions = [x["action"] for x in rows]
    act_min, act_max = min(actions), max(actions)
    act_mean = st.mean(actions)
    act_median = st.median(actions)

    print(f"[stats] rows={n}")
    print(f"[stats] agree_rate = {agree_cnt}/{n} = {agree_cnt/n:.3f}")
    print(f"[stats] nonzero_action_rate = {nonzero_cnt}/{n} = {nonzero_cnt/n:.3f}")
    print(f"[stats] action range = [{act_min:.4f}, {act_max:.4f}]")
    print(f"[stats] action mean/median = {act_mean:.4f} / {act_median:.4f}")

    # simple histogram
    lo, hi = -1.0, 1.0
    step = (hi - lo) / max(1, args.bins - 1)
    edges = [lo + i*step for i in range(args.bins)]
    counts = [0]* (len(edges)-1)
    for a in actions:
        k = min(len(edges)-2, max(0, int((a - lo) / step)))
        counts[k] += 1
    print("[hist] bins (approx):")
    for i in range(len(counts)):
        l, r = edges[i], edges[i+1]
        print(f"  [{l:+.2f},{r:+.2f}) : {counts[i]}")
