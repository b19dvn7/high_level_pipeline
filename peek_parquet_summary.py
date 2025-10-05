#!/usr/bin/env python3
# peek_parquet_summary.py
# Minimal, concise Parquet inspector using DuckDB.
# - Prints: file count, a few example paths, compact schema summary (first 30 cols),
#   detected timestamp column + format (ISO vs epoch_ms) with a couple sample values,
#   detected bid/ask price & size columns (prefers *_0), and short recommended DuckDB expressions.
# - Optional: --json writes full details to a JSON file; console stays concise.

import argparse, json, re, sys
from typing import List, Tuple, Optional
import duckdb

def sql_lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

def qident(name: str) -> str:
    return '"' + name.replace('"','""') + '"'

def describe_schema(con, glob: str) -> List[Tuple[str,str]]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_lit(glob)})").fetchall()
    return [(r[0], r[1]) for r in rows]

def list_files(con, glob: str) -> List[str]:
    try:
        q = f"SELECT DISTINCT filename FROM read_parquet({sql_lit(glob)}, filename=true)"
        return [r[0] for r in con.execute(q).fetchall()]
    except Exception:
        return []

def sample_values(con, glob: str, col: str, n: int = 8) -> List[Optional[str]]:
    try:
        q = f"SELECT {qident(col)} FROM read_parquet({sql_lit(glob)}) LIMIT {n}"
        return [r[0] for r in con.execute(q).fetchall()]
    except Exception:
        return []

def to_level(name: str) -> Optional[int]:
    m = re.search(r'_(\d+)$', name)
    return int(m.group(1)) if m else None

def ts_candidates(cols: List[Tuple[str,str]]) -> List[str]:
    cands = []
    for c,_t in cols:
        lc = c.lower()
        if "ts" in lc or "time" in lc or "timestamp" in lc or "datetime" in lc:
            cands.append(c)
    seen=set(); out=[]
    for c in cands:
        if c not in seen:
            seen.add(c); out.append(c)
    return out or ["ts"]

def classify_ts_samples(samples: List[Optional[str]]) -> str:
    if not samples: return "unknown"
    # If most values are numeric-ish with 13 digits → epoch ms
    num_like=0; ms_like=0; iso_like=0; total=0
    for v in samples:
        if v is None: continue
        s = str(v)
        total += 1
        if re.fullmatch(r'\d{10,16}', s or ""):
            num_like += 1
            if len(s) >= 13: ms_like += 1
        if re.search(r'\d{4}-\d{2}-\d{2}[ T]', s):
            iso_like += 1
    if total==0: return "unknown"
    if ms_like >= max(3, total//2): return "epoch_ms"
    if iso_like >= max(3, total//2): return "iso"
    if num_like >= max(3, total//2): return "epoch_like"
    return "unknown"

def pick_candidates(cols: List[Tuple[str,str]], side: str, kind: str) -> List[str]:
    names=[c for c,_ in cols]
    cands=[]
    for n in names:
        ln=n.lower()
        if side not in ln: continue
        if kind=="px":
            if ("px" in ln) or ("price" in ln) or re.search(r'(^|_)(p|prc|price)($|_)', ln):
                cands.append(n)
        else:
            if ("sz" in ln) or ("size" in ln) or ("qty" in ln) or ("quantity" in ln) or re.search(r'(^|_)(q|qty|size|quantity)($|_)', ln):
                cands.append(n)
    cands=list(set(cands))
    cands.sort(key=lambda x: (999999 if to_level(x) is None else to_level(x), x))
    return cands

def coalesce_expr(cols: List[str], default: Optional[str]=None, limit:int=6) -> str:
    parts = [f"CAST({qident(c)} AS DOUBLE)" for c in cols[:limit]]
    if default is not None:
        parts.append(default)
    if not parts:
        return "NULL"
    return "COALESCE(" + ", ".join(parts) + ")"

def build_ts_expr(ts_cols: List[str]) -> str:
    parts=[]
    for c in ts_cols[:4]:
        parts.append(f"TRY_CAST({qident(c)} AS TIMESTAMP)")
    for c in ts_cols[:4]:
        parts.append(f"to_timestamp(CAST(TRY_CAST({qident(c)} AS BIGINT) AS DOUBLE)/1000.0)")
    # minimal common fallbacks
    for c in ["ts_ms","timestamp_ms","time_ms"]:
        parts.append(f"to_timestamp(CAST(TRY_CAST({c} AS BIGINT) AS DOUBLE)/1000.0)")
    return "COALESCE(" + ", ".join(parts) + ")"

def main():
    ap=argparse.ArgumentParser(description="Concise Parquet summary (schema + detected OB fields).")
    ap.add_argument("--parquet", required=True, help="Recursive glob, e.g. /path/**/*.parquet")
    ap.add_argument("--json", default=None, help="Optional: write full report JSON here")
    args=ap.parse_args()

    con=duckdb.connect(database=":memory:")

    files=list_files(con, args.parquet)
    if not files:
        print(f"[info] No files matched: {args.parquet}")
        sys.exit(0)

    cols=describe_schema(con, args.parquet)
    # Detect fields
    ts_cols = ts_candidates(cols)
    # choose best ts candidate by which classifies cleanly
    ts_choice = ts_cols[0]
    ts_samples = sample_values(con, args.parquet, ts_choice, 8)
    ts_format = classify_ts_samples(ts_samples)

    bid_px = pick_candidates(cols, "bid", "px")
    ask_px = pick_candidates(cols, "ask", "px")
    bid_sz = pick_candidates(cols, "bid", "sz")
    ask_sz = pick_candidates(cols, "ask", "sz")

    # Short recommended expressions (trimmed to be readable)
    TS_EXPR   = build_ts_expr(ts_cols)
    BID_PX_EX = coalesce_expr(bid_px, default=None, limit=6)
    ASK_PX_EX = coalesce_expr(ask_px, default=None, limit=6)
    BID_SZ_EX = coalesce_expr(bid_sz, default="1.0", limit=6)
    ASK_SZ_EX = coalesce_expr(ask_sz, default="1.0", limit=6)

    # -------- Console summary (concise) --------
    print(f"[files] matched: {len(files)}")
    for p in files[:5]:
        print("  -", p)
    if len(files) > 5:
        print(f"  ... (+{len(files)-5} more)")

    print("\n[schema] (first 30 columns)")
    for i,(n,t) in enumerate(cols[:30],1):
        print(f"  {i:2d}. {n}: {t}")
    if len(cols) > 30:
        print(f"  ... (+{len(cols)-30} more columns)")

    print("\n[timestamp]")
    print(f"  candidates: {ts_cols[:6]}{' ...' if len(ts_cols)>6 else ''}")
    print(f"  chosen    : {ts_choice}")
    print(f"  format    : {ts_format}")
    if ts_samples:
        show = [str(s) for s in ts_samples[:3]]
        print(f"  samples   : {show}")

    print("\n[top-of-book candidates]")
    print(f"  bid_px: {bid_px[:6]}{' ...' if len(bid_px)>6 else ''}")
    print(f"  ask_px: {ask_px[:6]}{' ...' if len(ask_px)>6 else ''}")
    print(f"  bid_sz: {bid_sz[:6]}{' ...' if len(bid_sz)>6 else ''}")
    print(f"  ask_sz: {ask_sz[:6]}{' ...' if len(ask_sz)>6 else ''}")

    print("\n[recommended DuckDB expressions] (truncated to keep readable)")
    print(f"  TS_EXPR   = {TS_EXPR}")
    print(f"  BID_PX_EX = {BID_PX_EX}")
    print(f"  ASK_PX_EX = {ASK_PX_EX}")
    print(f"  BID_SZ_EX = {BID_SZ_EX}")
    print(f"  ASK_SZ_EX = {ASK_SZ_EX}")

    # -------- Optional JSON with fuller detail --------
    if args.json:
        report = {
            "parquet_glob": args.parquet,
            "file_count": len(files),
            "example_files": files[:10],
            "schema_first30": cols[:30],
            "all_columns_count": len(cols),
            "timestamp": {
                "candidates": ts_cols,
                "chosen": ts_choice,
                "format": ts_format,
                "samples_first3": [str(x) for x in ts_samples[:3]] if ts_samples else []
            },
            "candidates": {
                "bid_px": bid_px,
                "ask_px": ask_px,
                "bid_sz": bid_sz,
                "ask_sz": ask_sz
            },
            "expressions": {
                "TS_EXPR": TS_EXPR,
                "BID_PX_EX": BID_PX_EX,
                "ASK_PX_EX": ASK_PX_EX,
                "BID_SZ_EX": BID_SZ_EX,
                "ASK_SZ_EX": ASK_SZ_EX
            }
        }
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[ok] wrote JSON: {args.json}")

if __name__ == "__main__":
    main()
