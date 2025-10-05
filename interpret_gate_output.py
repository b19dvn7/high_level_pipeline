#!/usr/bin/env python3
# interpret_gate_output.py — turns gate_out.csv into an interpretable summary
# Usage: python3 interpret_gate_output.py --csv gate_out.csv --pos_bins 0.05 0.15 0.30
import argparse, csv, statistics as st, math

def bucket_action(a, bins):
    ab = abs(a)
    for i, b in enumerate(bins):
        if ab < b:
            return i
    return len(bins)

def label_for(i, bins):
    if i==0: return "Tiny (|a| < %.2f)" % bins[0]
    if i==1: return "Small [%.2f, %.2f)" % (bins[0], bins[1])
    if i==2: return "Medium [%.2f, %.2f)" % (bins[1], bins[2])
    return "Large (>= %.2f)" % bins[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pos_bins", nargs="+", type=float, default=[0.05,0.15,0.30],
                    help="abs(action) bucket edges")
    args = ap.parse_args()

    rows=[]
    with open(args.csv, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                agree=int(row["agree"])
                action=float(row["action"])
                maxdiff=float(row.get("max_pair_diff") or row.get("details","{}").split("max_pair_diff': ")[-1].split(",")[0]) if "max_pair_diff" in row or "details" in row else float("nan")
                rows.append((agree, action, maxdiff))
            except:
                pass

    if not rows:
        print("[error] no rows"); return
    n=len(rows)
    agree_n=sum(1 for a,_,__ in rows if a==1)
    print(f"[rows] {n}   [agree=1] {agree_n}  ({agree_n/n:.2%})")

    acts=[a for g,a,_ in rows if g==1]
    if not acts:
        print("[info] no agreed trades → loosen gate or try sign-consensus")
        return

    print("\n=== Action stats (agreed trades only) ===")
    print(f"mean={st.mean(acts):+.4f}  median={st.median(acts):+.4f}  min={min(acts):+.4f}  max={max(acts):+.4f}")

    # Buckets
    bins = sorted(args.pos_bins)
    buckets = {}
    for g,a,_ in rows:
        if g!=1: continue
        k = bucket_action(a, bins)
        buckets.setdefault(k, []).append(a)

    total_agreed=len(acts)
    for k in sorted(buckets):
        arr=buckets[k]
        print(f"- {label_for(k,bins)}: {len(arr)} ({len(arr)/total_agreed:.1%}) | buy={sum(1 for x in arr if x>0)} sell={sum(1 for x in arr if x<0)}")

    # sample most confident buys/sells
    buys=sorted([x for x in acts if x>0], reverse=True)[:5]
    sells=sorted([x for x in acts if x<0])[:5]
    print("\nTop buys:", [round(x,4) for x in buys])
    print("Top sells:", [round(x,4) for x in sells])

    # disagreement snapshot
    diffs=[d for g,_,d in rows if g==1 and not math.isnan(d)]
    if diffs:
        print(f"\nmax_pair_diff: mean={st.mean(diffs):.3f}, 90p={sorted(diffs)[int(0.9*len(diffs))]:.3f}, max={max(diffs):.3f}")
    print("\nLegend:")
    print("  action: signed size in [-1,1] (magnitude ~ conviction).")
    print("  Tiny/Small/Medium/Large buckets help you see if signals are mostly noise or have teeth.")
    print("  If you mostly see Tiny, raise the min magnitude you act on (e.g., trade only if |a|>=0.05).")

if __name__ == "__main__":
    main()
