#!/usr/bin/env python3
# gen_states_from_parquet.py — makes states.csv from your book snaps
# Usage:
#   python3 gen_states_from_parquet.py --parquet "/path/**/*.parquet" --out states.csv --limit 200
import argparse, duckdb, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default="states.csv")
    ap.add_argument("--limit", type=int, default=200, help="recent rows to export")
    args = ap.parse_args()

    con = duckdb.connect(database=":memory:")
    # robust: union_by_name handles mixed schemas; ts/timestamp both supported
    sql = f"""
WITH raw AS (
  SELECT * FROM read_parquet('{args.parquet}', filename=true, union_by_name=true)
),
proj AS (
  SELECT
    COALESCE(TRY_CAST("ts" AS TIMESTAMP), TRY_CAST("timestamp" AS TIMESTAMP))   AS ts_parsed,
    CAST(bid_px_1 AS DOUBLE) AS bid_px, CAST(ask_px_1 AS DOUBLE) AS ask_px,
    CAST(bid_sz_1 AS DOUBLE) AS bid_sz, CAST(ask_sz_1 AS DOUBLE) AS ask_sz
  FROM raw
),
ord AS (
  SELECT * FROM proj WHERE ts_parsed IS NOT NULL ORDER BY ts_parsed
),
feat AS (
  SELECT
    ts_parsed AS ts,
    (bid_px + ask_px)*0.5 AS mid,
    (ask_px - bid_px)     AS spread,
    CASE WHEN (bid_sz + ask_sz) > 0
         THEN (bid_sz - ask_sz)/NULLIF(bid_sz + ask_sz,0)
         ELSE 0 END       AS imbalance,
    ((bid_px + ask_px)*0.5 - LAG((bid_px + ask_px)*0.5, 1) OVER ()) AS mom1,
    ((bid_px + ask_px)*0.5 - LAG((bid_px + ask_px)*0.5, 3) OVER ()) AS mom3,
    STDDEV_SAMP((bid_px + ask_px)*0.5) OVER (ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS vol3
  FROM ord
)
SELECT mid, spread, imbalance, mom1, mom3, vol3
FROM feat
WHERE mid IS NOT NULL
ORDER BY ts DESC
LIMIT {args.limit}
"""
    df = con.execute(sql).fetchdf()
    # ensure correct order & types
    cols = ["mid","spread","imbalance","mom1","mom3","vol3"]
    df = df[cols].astype("float32")
    df.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
