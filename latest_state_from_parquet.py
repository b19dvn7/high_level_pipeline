#!/usr/bin/env python3
# latest_state_from_parquet.py
# Reads your snaps parquet glob, builds features, prints the LAST row as:
# mid,spread,imbalance,mom1,mom3,vol3
import argparse, duckdb, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help='e.g. "/path/**/*.parquet"')
    ap.add_argument("--mom1_lag", type=int, default=1)
    ap.add_argument("--mom3_lag", type=int, default=3)
    ap.add_argument("--vol_win",  type=int, default=10, help="stddev window (rows)")
    args = ap.parse_args()

    con = duckdb.connect(database=":memory:")
    sql = f"""
WITH raw AS (
  SELECT * FROM read_parquet('{args.parquet}', filename=true, union_by_name=true)
),
proj AS (
  SELECT
    COALESCE(TRY_CAST("ts" AS TIMESTAMP), TRY_CAST("timestamp" AS TIMESTAMP)) AS ts_parsed,
    CAST(bid_px_1 AS DOUBLE) AS bid_px,
    CAST(ask_px_1 AS DOUBLE) AS ask_px,
    CAST(bid_sz_1 AS DOUBLE) AS bid_sz,
    CAST(ask_sz_1 AS DOUBLE) AS ask_sz
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
    ((bid_px + ask_px)*0.5 - LAG((bid_px + ask_px)*0.5, {args.mom1_lag}) OVER ()) AS mom1,
    ((bid_px + ask_px)*0.5 - LAG((bid_px + ask_px)*0.5, {args.mom3_lag}) OVER ()) AS mom3,
    STDDEV_SAMP((bid_px + ask_px)*0.5) OVER (ROWS BETWEEN {args.vol_win-1} PRECEDING AND CURRENT ROW) AS vol3
  FROM ord
)
SELECT mid, spread, imbalance, mom1, mom3, vol3
FROM feat
WHERE mid IS NOT NULL
ORDER BY ts DESC
LIMIT 1
"""
    df = con.execute(sql).fetchdf()
    if df is None or len(df)==0:
        print("[error] no rows produced", file=sys.stderr); sys.exit(2)
    row = df.iloc[0].tolist()
    # print a single CSV line (no header)
    print(",".join(f"{float(x):.10f}" if x is not None else "0.0" for x in row))

if __name__ == "__main__":
    main()
