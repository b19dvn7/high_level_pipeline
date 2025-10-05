#!/usr/bin/env python3
# DO NOT EDIT: generated from /home/bigdan7/Documents/TRADING/high_level_pipeline/ob_mapping.json
# Custom, non-destructive data layer (keeps originals intact).

import argparse
from typing import List, Optional, Dict
import duckdb, numpy as np, pandas as pd

TS_EXPR   = r"""COALESCE(TRY_CAST("ts" AS TIMESTAMP), to_timestamp(CAST(TRY_CAST("ts" AS BIGINT) AS DOUBLE)/1000.0), to_timestamp(CAST(TRY_CAST(ts_ms AS BIGINT) AS DOUBLE)/1000.0), to_timestamp(CAST(TRY_CAST(timestamp_ms AS BIGINT) AS DOUBLE)/1000.0), to_timestamp(CAST(TRY_CAST(time_ms AS BIGINT) AS DOUBLE)/1000.0))"""
BID_PX_EX = r"""COALESCE(CAST("bid_px_1" AS DOUBLE), CAST("bid_px_2" AS DOUBLE), CAST("bid_px_3" AS DOUBLE), CAST("bid_px_4" AS DOUBLE), CAST("bid_px_5" AS DOUBLE), CAST("bid_px_6" AS DOUBLE))"""
ASK_PX_EX = r"""COALESCE(CAST("ask_px_1" AS DOUBLE), CAST("ask_px_2" AS DOUBLE), CAST("ask_px_3" AS DOUBLE), CAST("ask_px_4" AS DOUBLE), CAST("ask_px_5" AS DOUBLE), CAST("ask_px_6" AS DOUBLE))"""
BID_SZ_EX = r"""COALESCE(CAST("bid_sz_1" AS DOUBLE), CAST("bid_sz_2" AS DOUBLE), CAST("bid_sz_3" AS DOUBLE), CAST("bid_sz_4" AS DOUBLE), CAST("bid_sz_5" AS DOUBLE), CAST("bid_sz_6" AS DOUBLE), 1.0)"""
ASK_SZ_EX = r"""COALESCE(CAST("ask_sz_1" AS DOUBLE), CAST("ask_sz_2" AS DOUBLE), CAST("ask_sz_3" AS DOUBLE), CAST("ask_sz_4" AS DOUBLE), CAST("ask_sz_5" AS DOUBLE), CAST("ask_sz_6" AS DOUBLE), 1.0)"""

class ReplayBuffer:
    def __init__(self, capacity:int, state_dim:int, act_dim:int=0):
        self.capacity=int(capacity); self.state_dim=int(state_dim); self.act_dim=int(act_dim)
        self.ptr=0; self.full=False
        self.s=np.zeros((self.capacity,self.state_dim),dtype=np.float32)
        self.sp=np.zeros((self.capacity,self.state_dim),dtype=np.float32)
        self.r=np.zeros((self.capacity,1),dtype=np.float32)
        self.d=np.zeros((self.capacity,1),dtype=np.float32)
        self.a=None
        if self.act_dim>0: self.a=np.zeros((self.capacity,self.act_dim),dtype=np.float32)
    def add(self,s,a,r,sp,d):
        i=self.ptr; self.s[i]=s; self.sp[i]=sp; self.r[i]=r; self.d[i]=d
        if self.a is not None and a is not None: self.a[i]=a
        self.ptr=(self.ptr+1)%self.capacity; self.full = self.full or (self.ptr==0)
    def size(self): return self.capacity if self.full else self.ptr
    def sample(self,batch_size:int,rng:np.random.Generator)->Dict[str,np.ndarray]:
        n=self.size(); idx=rng.integers(0,n,size=batch_size)
        out={"s":self.s[idx],"r":self.r[idx],"sp":self.sp[idx],"d":self.d[idx]}
        if self.a is not None: out["a"]=self.a[idx]
        return out

def build_sql(parquet_glob:str,start_ts:Optional[str],end_ts:Optional[str],horizon_rows:int,ret_threshold:float,extra_filters:Optional[str]=None)->str:
    where=[]
    if start_ts: where.append(f"ts_parsed >= TIMESTAMP '{start_ts}'")
    if end_ts:   where.append(f"ts_parsed <= TIMESTAMP '{end_ts}'")
    if extra_filters: where.append(f"({extra_filters})")
    where_clause = "WHERE " + " AND ".join(where) if where else ""
    sql = f"""
WITH raw AS (
  SELECT * FROM read_parquet('{parquet_glob}', filename=true)
),
proj AS (
  SELECT
    {TS_EXPR}   AS ts_parsed,
    {BID_PX_EX} AS bid_px_d,
    {ASK_PX_EX} AS ask_px_d,
    {BID_SZ_EX} AS bid_sz_d,
    {ASK_SZ_EX} AS ask_sz_d
  FROM raw
),
src AS ( SELECT * FROM proj {where_clause} ),
ord AS ( SELECT * FROM src ORDER BY ts_parsed ),
feat AS (
  SELECT
    ts_parsed AS ts,
    (bid_px_d + ask_px_d)*0.5 AS mid,
    (ask_px_d - bid_px_d)     AS spread,
    CASE WHEN (bid_sz_d + ask_sz_d)>0
         THEN (bid_sz_d - ask_sz_d)/NULLIF(bid_sz_d + ask_sz_d,0)
         ELSE 0 END           AS imbalance,
    ((bid_px_d + ask_px_d)*0.5 - LAG((bid_px_d + ask_px_d)*0.5,1) OVER ()) AS mom1,
    ((bid_px_d + ask_px_d)*0.5 - LAG((bid_px_d + ask_px_d)*0.5,3) OVER ()) AS mom3,
    STDDEV_SAMP((bid_px_d + ask_px_d)*0.5) OVER (ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS vol3
  FROM ord
),
lab AS (
  SELECT
    *,
    LEAD(mid, {horizon_rows}) OVER () AS mid_fwd,
    (LEAD(mid, {horizon_rows}) OVER () - mid) / NULLIF(mid,0) AS fwd_ret
  FROM feat
),
final AS (
  SELECT
    ts,
    mid, spread, imbalance, mom1, mom3, vol3,
    LEAD(mid, {horizon_rows}) OVER ()        AS mid_p,
    LEAD(spread, {horizon_rows}) OVER ()     AS spread_p,
    LEAD(imbalance, {horizon_rows}) OVER ()  AS imbalance_p,
    LEAD(mom1, {horizon_rows}) OVER ()       AS mom1_p,
    LEAD(mom3, {horizon_rows}) OVER ()       AS mom3_p,
    LEAD(vol3, {horizon_rows}) OVER ()       AS vol3_p,
    fwd_ret,
    CASE WHEN fwd_ret > {ret_threshold} THEN 1
         WHEN fwd_ret < -{ret_threshold} THEN -1
         ELSE 0 END AS label
  FROM lab
)
SELECT * FROM final
WHERE mid IS NOT NULL AND mid_p IS NOT NULL
"""
    return sql

def _ensure_float(df: pd.DataFrame, cols: List[str])->pd.DataFrame:
    for c in cols:
        if c in df.columns: df[c]=pd.to_numeric(df[c], errors="coerce")
    return df

def stream_into_buffer(con:duckdb.DuckDBPyConnection, sql:str, buffer:ReplayBuffer,
                       chunk_rows:int=200_000, drop_noise_label:bool=False, derive_action:bool=False)->int:
    total=0; cur=con.execute(sql)
    state_cols=["mid","spread","imbalance","mom1","mom3","vol3"]
    next_cols=["mid_p","spread_p","imbalance_p","mom1_p","mom3_p","vol3_p"]
    while True:
        df=cur.fetch_df_chunk(chunk_rows)
        if df is None or len(df)==0: break
        df=_ensure_float(df, list(df.columns))
        if drop_noise_label and "label" in df.columns: df=df[df["label"]!=0]
        df=df.dropna(subset=state_cols+next_cols+["fwd_ret"])
        S=df[state_cols].to_numpy(np.float32)
        SP=df[next_cols].to_numpy(np.float32)
        R=df[["fwd_ret"]].to_numpy(np.float32)
        D=np.zeros((len(df),1),dtype=np.float32)
        if derive_action:
            A=df[["label"]].to_numpy(np.float32) if "label" in df.columns else np.sign(R).astype(np.float32)
        else:
            A=None
        for i in range(len(df)):
            buffer.add(S[i], (A[i] if A is not None else None), R[i], SP[i], D[i])
        total+=len(df)
    return total

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--th", type=float, default=0.0002)
    args=ap.parse_args()
    con=duckdb.connect(database=":memory:")
    sql=build_sql(args.parquet,args.start,args.end,args.h,args.th,None)
    plan=con.execute(f"EXPLAIN {sql}").fetchall()
    print("[DuckDB plan]"); [print(" ",r[0]) for r in plan]
