# rl_duckdb_data_layer.py (v1.5 — reward scaled by 1000x)
# - Timestamp: supports 'ts' or 'timestamp' (TIMESTAMP)
# - L1 fields: bid_px_1 / ask_px_1 / bid_sz_1 / ask_sz_1
# - Features: mid, spread, imbalance, mom1, mom3, vol3
# - Next-state (H bars ahead): mid_p, spread_p, imbalance_p, mom1_p, mom3_p, vol3_p
# - Reward: fwd_ret_raw = (mid_p - mid)/mid ; fwd_ret = 1000 * fwd_ret_raw
# - union_by_name=true to tolerate schema drift across days/files

def build_sql(parquet_glob: str,
              start: str | None,
              end: str | None,
              h: int,
              th: float,
              drop_noise: bool) -> str:
    # Optional time filter
    time_filter = ""
    if start and end:
        time_filter = f"WHERE ts >= TIMESTAMP '{start}' AND ts <= TIMESTAMP '{end}'"
    elif start:
        time_filter = f"WHERE ts >= TIMESTAMP '{start}'"
    elif end:
        time_filter = f"WHERE ts <= TIMESTAMP '{end}'"

    return f"""
WITH raw AS (
  SELECT
    COALESCE(TRY_CAST("ts" AS TIMESTAMP), TRY_CAST("timestamp" AS TIMESTAMP)) AS ts,
    CAST("bid_px_1" AS DOUBLE) AS bid_px_1,
    CAST("ask_px_1" AS DOUBLE) AS ask_px_1,
    CAST("bid_sz_1" AS DOUBLE) AS bid_sz_1,
    CAST("ask_sz_1" AS DOUBLE) AS ask_sz_1
  FROM read_parquet('{parquet_glob}', union_by_name=true)
),
lvl AS (
  SELECT *
  FROM raw
  WHERE ts IS NOT NULL
    AND bid_px_1 IS NOT NULL AND ask_px_1 IS NOT NULL
),
feat AS (
  SELECT
    ts,
    (ask_px_1 + bid_px_1)/2.0 AS mid,
    (ask_px_1 - bid_px_1)     AS spread,
    CASE
      WHEN (bid_sz_1 + ask_sz_1) > 0
      THEN (bid_sz_1 - ask_sz_1) / NULLIF(bid_sz_1 + ask_sz_1, 0)
      ELSE NULL
    END AS imbalance
  FROM lvl
),
lags AS (
  SELECT
    ts, mid, spread, imbalance,
    LAG(mid, 1) OVER (ORDER BY ts) AS mid_l1,
    LAG(mid, 3) OVER (ORDER BY ts) AS mid_l3
  FROM feat
),
rets AS (
  SELECT
    ts, mid, spread, imbalance, mid_l1, mid_l3,
    CASE WHEN mid_l1 IS NULL OR mid_l1 = 0 THEN NULL ELSE (mid - mid_l1)/mid_l1 END AS ret1,
    CASE WHEN mid_l3 IS NULL OR mid_l3 = 0 THEN NULL ELSE (mid - mid_l3)/mid_l3 END AS ret3
  FROM lags
),
vol AS (
  SELECT
    ts, mid, spread, imbalance,
    ret1, ret3,
    /* 3-bar rolling stddev of 1-bar returns */
    STDDEV_SAMP(ret1) OVER (ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS vol3
  FROM rets
),
lead_h AS (
  SELECT
    ts, mid, spread, imbalance, ret1 AS mom1, ret3 AS mom3, vol3,
    /* H-step lookahead */
    LEAD(mid,        {h}) OVER (ORDER BY ts) AS mid_p,
    LEAD(spread,     {h}) OVER (ORDER BY ts) AS spread_p,
    LEAD(imbalance,  {h}) OVER (ORDER BY ts) AS imbalance_p,
    LEAD(ret1,       {h}) OVER (ORDER BY ts) AS mom1_p,
    LEAD(ret3,       {h}) OVER (ORDER BY ts) AS mom3_p,
    LEAD(vol3,       {h}) OVER (ORDER BY ts) AS vol3_p
  FROM vol
),
aug AS (
  SELECT
    ts, mid, spread, imbalance, mom1, mom3, vol3,
    mid_p, spread_p, imbalance_p, mom1_p, mom3_p, vol3_p,
    CASE WHEN mid IS NULL OR mid = 0 OR mid_p IS NULL THEN NULL
         ELSE (mid_p - mid) / mid
    END AS fwd_ret_raw
  FROM lead_h
),
scale AS (
  SELECT
    ts, mid, spread, imbalance, mom1, mom3, vol3,
    mid_p, spread_p, imbalance_p, mom1_p, mom3_p, vol3_p,
    /* Scale reward to strengthen gradients (does not change optimal policy) */
    1000.0 * fwd_ret_raw AS fwd_ret
  FROM aug
),
flt AS (
  SELECT * FROM scale
  {time_filter}
)
SELECT
  ts, mid, spread, imbalance, mom1, mom3, vol3,
  mid_p, spread_p, imbalance_p, mom1_p, mom3_p, vol3_p,
  fwd_ret
FROM flt
ORDER BY ts
"""
