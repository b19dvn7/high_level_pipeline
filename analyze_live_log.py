#!/usr/bin/env python3
import csv, sys, pathlib, statistics as st
from datetime import datetime

def to_float(x, default=None):
    try: return float(x)
    except: return default

def parse_rows(path):
    rows=[]
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return rows
        for row in r:
            # unify keys across old/new schemas
            ts = row.get("ts") or row.get("time") or ""
            prev = int(float(row.get("prev_pos", "0") or 0))
            nxt  = int(float(row.get("next_pos", "0") or 0))
            raw  = to_float(row.get("raw_action", row.get("action", "0")), 0.0)
            # thresholds: new has buy_th/sell_th; old had th
            buy_th  = to_float(row.get("buy_th", row.get("th", None)), None)
            sell_th = to_float(row.get("sell_th", row.get("th", None)), None)
            hold_bars = int(float(row.get("hold_bars", "0") or 0))
            hold_ctr  = int(float(row.get("hold_counter", row.get("held", "0") or 0)))
            spread = to_float(row.get("spread", None))
            vol3   = to_float(row.get("vol3", None))
            skipped = row.get("skipped_reason", "")
            rows.append(dict(ts=ts, prev=prev, nxt=nxt, raw=raw, buy_th=buy_th,
                             sell_th=sell_th, hold_bars=hold_bars, hold_ctr=hold_ctr,
                             spread=spread, vol3=vol3, skipped=skipped))
    return rows

def summarize(rows, last_n=20):
    if not rows:
        print("[empty] no rows"); return
    # basic counts
    flips = sum(1 for r in rows if r["nxt"] != r["prev"])
    entries = sum(1 for i,r in enumerate(rows) if (r["prev"]==0 and r["nxt"]!=0))
    exits   = sum(1 for i,r in enumerate(rows) if (r["prev"]!=0 and r["nxt"]==0))
    inpos_bars = sum(1 for r in rows if r["nxt"]!=0)
    pos_frac = inpos_bars / len(rows)

    # segment holds
    segs=[]; start=0; cur=rows[0]["nxt"]
    for i in range(1,len(rows)):
        if rows[i]["nxt"] != cur:
            segs.append((start, i-1, cur))
            start = i; cur = rows[i]["nxt"]
    segs.append((start, len(rows)-1, cur))
    holds=[(hi-lo+1) for (lo,hi,sign) in segs if sign!=0]
    avg_hold = st.mean(holds) if holds else 0
    med_hold = st.median(holds) if holds else 0

    # action stats while in position
    acts=[abs(r["raw"]) for r in rows if r["nxt"]!=0]
    act_mean = st.mean(acts) if acts else 0
    act_p95  = (sorted(acts)[int(0.95*(len(acts)-1))] if acts else 0)

    # skips
    skip_counts={}
    for r in rows:
        s=r["skipped"]
        if s:
            skip_counts[s]=skip_counts.get(s,0)+1

    print("[live summary]")
    print(f" rows={len(rows)}  flips={flips}  entries={entries}  exits={exits}")
    print(f" in_position_bars={inpos_bars}  exposure={pos_frac:.2%}")
    print(f" hold_bars(avg/med)={avg_hold:.1f}/{med_hold:.1f}")
    print(f" |action|(mean/p95)={act_mean:.3f}/{act_p95:.3f}")
    if skip_counts:
        print(" skips:", "  ".join(f"{k}={v}" for k,v in skip_counts.items()))
    # last N lines
    print("\n[last {} rows]".format(min(last_n,len(rows))))
    hdr = "ts,prev_pos,next_pos,raw_action,buy_th,sell_th,hold_bars,hold_counter,spread,vol3,skipped_reason"
    print(hdr)
    for r in rows[-last_n:]:
        print("{},{},{},{:.6f},{},{},{},{},{},{}{}".format(
            r["ts"], r["prev"], r["nxt"], r["raw"],
            "" if r["buy_th"] is None else f"{r['buy_th']:.2f}",
            "" if r["sell_th"] is None else f"{r['sell_th']:.2f}",
            r["hold_bars"], r["hold_ctr"],
            "" if r["spread"] is None else f"{r['spread']:.6f}",
            "" if r["vol3"]   is None else f"{r['vol3']:.6f}",
            f",{r['skipped']}" if r["skipped"] else ""
        ))

if __name__ == "__main__":
    p = pathlib.Path("live_signals.csv")
    if not p.exists():
        sys.exit("live_signals.csv not found")
    rows = parse_rows(str(p))
    summarize(rows, last_n=20)
