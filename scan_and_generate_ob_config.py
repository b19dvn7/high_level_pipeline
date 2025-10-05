#!/usr/bin/env python3
"""
scan_and_generate_ob_config.py
One-shot: scan Parquet schema -> infer timestamp & top-of-book bid/ask px/sz -> emit:
  - ob_mapping.json   (what was detected)
  - rl_duckdb_data_layer.py (FULL, ready-to-run, tailored to your schema)

Usage:
  python3 scan_and_generate_ob_config.py \
    --parquet "/abs/path/**/*.parquet" \
    --out-config ob_mapping.json \
    --out-datalayer rl_duckdb_data_layer.py
"""

import argparse, json, re, os
import duckdb
import pandas as pd

# -------- Utilities (version-safe) --------
def quote_ident(c: str) -> str:
    # Double-quote SQL identifier, doubling any inner quotes
    return '"' + c.replace('"', '""') + '"'

def list_columns(con, glob):
    q = f"DESCRIBE SELECT * FROM read_parquet('{glob}')"
    rows = con.execute(q).fetchall()
    return [(r[0], r[1]) for r in rows]  # (name, type)

def sample_values(con, glob, cols, n=20):
    if not cols: return None
    sel = ", ".join([quote_ident(c) for c in cols])
    q = f"SELECT {sel} FROM read_parquet('{glob}') LIMIT {n}"
    try:
        return con.execute(q).fetchdf()
    except Exception:
        return None

def to_level(name: str):
    m = re.search(r'_(\d+)$', name)
    return int(m.group(1)) if m else None

def find_candidates(cols, side, kind):
    names = [c[0] for c in cols]
    cands = []
    for n in names:
        ln = n.lower()
        if side in ln:
            if kind == "px" and ("px" in ln or "price" in ln):
                cands.append(n)
            elif kind == "sz" and ("sz" in ln or "size" in ln or "qty" in ln or "quantity" in ln):
                cands.append(n)
    # fallback if strict missed
    if not cands:
        for n in names:
            ln = n.lower()
            if side in ln:
                if kind == "px" and ("p" in ln or "price" in ln):
                    cands.append(n)
                if kind == "sz" and ("size" in ln or "qty" in ln or "quantity" in ln):
                    cands.append(n)
    cands = list(set(cands))
    cands.sort(key=lambda x: (999999 if to_level(x) is None else to_level(x), x))
    return cands

def ts_candidates(cols):
    out = []
    for c,_ in cols:
        lc = c.lower()
        if "ts" in lc or "time" in lc or "timestamp" in lc or "datetime" in lc:
            out.append(c)
    # keep order but unique
    seen = set(); uniq=[]
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq or ["ts"]  # default guess

def build_ts_expr(cands):
    parts = []
    for c in cands:
        ci = quote_ident(c)
        parts.append(f"TRY_CAST({ci} AS TIMESTAMP)")
    for c in cands:
        ci = quote_ident(c)
        parts.append(f"to_timestamp(CAST(TRY_CAST({ci} AS BIGINT) AS DOUBLE)/1000.0)")
    # common fallbacks
    for c in ["ts_ms","timestamp_ms","time_ms","epoch_ms","t_ms"]:
        parts.append(f"to_timestamp(CAST(TRY_CAST({c} AS BIGINT) AS DOUBLE)/1000.0)")
    parts.append("to_timestamp(CAST(TRY_CAST(CAST(ts AS DOUBLE) AS BIGINT) AS DOUBLE)/1000.0)")
    return "COALESCE(" + ", ".join(parts) + ")"

def build_num_expr(cands, default=None):
    parts = [f"CAST({quote_ident(c)} AS DOUBLE)" for c in cands]
    if default is not None:
        parts.append(default)
    if not parts:
        return "NULL"
    return "COALESCE(" + ", ".join(parts) + ")"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-config", default="ob_mapping.json")
    ap.add_argument("--out-datalayer", default="rl_duckdb_data_layer.py")
    ap.add_argument("--sample", type=int, default=10)
    args = ap.parse_args()

    con = duckdb.connect()

    cols = list_columns(con, args.parquet)
    if not cols:
        raise SystemExit("No columns found. Check your --parquet path/pattern.")

    # preview first few columns just to show something in mapping
    df_preview = sample_values(con, args.parquet, [c for c,_ in cols][:min(12,len(cols))], args.sample)

    # detect
    ts_cands = ts_candidates(cols)
    bid_px_c = find_candidates(cols, "bid", "px")
    ask_px_c = find_candidates(cols, "ask", "px")
    bid_sz_c = find_candidates(cols, "bid", "sz")
    ask_sz_c = find_candidates(cols, "ask", "sz")

    # expressions
    ts_expr   = build_ts_expr(ts_cands)
    bid_px_ex = build_num_expr(bid_px_c)
    ask_px_ex = build_num_expr(ask_px_c)
    bid_sz_ex = build_num_expr(bid_sz_c, default="1.0")
    ask_sz_ex = build_num_expr(ask_sz_c, default="1.0")

    # write mapping JSON
    mapping = {
        "ts_expr": ts_expr,
        "bid_px_expr": bid_px_ex,
        "ask_px_expr": ask_px_ex,
        "bid_sz_expr": bid_sz_ex,
        "ask_sz_expr": ask_sz_ex,
        "columns_detected": {k:v for k,v in cols},
        "ts_candidates": ts_cands,
        "bid_px_candidates": bid_px_c,
        "ask_px_candidates": ask_px_c,
        "bid_sz_candidates": bid_sz_c,
        "ask_sz_candidates": ask_sz_c,
        "preview": (df_preview.to_dict(orient="records") if df_preview is not None else [])
    }
    with open(args.out_config, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[ok] wrote mapping {os.path.abspath(args.out_config)}")

    # emit FULL tailored rl_duckdb_data_layer.py
    code = f"""#!/usr/bin/env python3
# Auto-generated by scan_and_generate_ob_config.py
# Tailored top-of-book mapping for your Parquet schema.

import argparse
from typing import List, Optional, Dict
import duckdb, numpy as np, pandas as pd

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
        out={{"s":self.s[idx],"r":self.r[idx],"sp":self.sp[idx],"d":self.d[idx]}}
        if self.a is not None: out["a"]=self.a[idx]
        return out

def build_sql(parquet_glob:str,start_ts:Optional[str],end_ts:Optional[str],horizon_rows:int,ret_threshold:float,extra_filters:Optional[str]=None)->str:
    where=[]
    if start_ts: where.append(f"ts_parsed >= TIMESTAMP '{{start_ts}}'")
    if end_ts:   where.append(f"ts_parsed <= TIMESTAMP '{{end_ts}}'")
    if extra_filters: where.append(f"({{extra_filters}})")
    where_clause = "WHERE " + " AND ".join(where) if where else ""
    sql = f\"\"\"
WITH raw AS (
  SELECT * FROM read_parquet('{{parquet_glob}}', filename=true)
),
proj AS (
  SELECT
    {ts_expr} AS ts_parsed,
    {bid_px_ex} AS bid_px_d,
    {ask_px_ex} AS ask_px_d,
    {bid_sz_ex} AS bid_sz_d,
    {ask_sz_ex} AS ask_sz_d
  FROM raw
),
src AS ( SELECT * FROM proj {{where_clause}} ),
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
    LEAD(mid, {{horizon_rows}}) OVER () AS mid_fwd,
    (LEAD(mid, {{horizon_rows}}) OVER () - mid) / NULLIF(mid,0) AS fwd_ret
  FROM feat
),
final AS (
  SELECT
    ts,
    mid, spread, imbalance, mom1, mom3, vol3,
    LEAD(mid, {{horizon_rows}}) OVER ()        AS mid_p,
    LEAD(spread, {{horizon_rows}}) OVER ()     AS spread_p,
    LEAD(imbalance, {{horizon_rows}}) OVER ()  AS imbalance_p,
    LEAD(mom1, {{horizon_rows}}) OVER ()       AS mom1_p,
    LEAD(mom3, {{horizon_rows}}) OVER ()       AS mom3_p,
    LEAD(vol3, {{horizon_rows}}) OVER ()       AS vol3_p,
    fwd_ret,
    CASE WHEN fwd_ret > {{ret_threshold}} THEN 1
         WHEN fwd_ret < -{{ret_threshold}} THEN -1
         ELSE 0 END AS label
  FROM lab
)
SELECT * FROM final
WHERE mid IS NOT NULL AND mid_p IS NOT NULL
\"\"\".format(parquet_glob=parquet_glob,start_ts=start_ts,end_ts=end_ts,extra_filters=extra_filters or "",
             horizon_rows=horizon_rows,ret_threshold=ret_threshold,where_clause=where_clause)
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
"""
    with open(args.out_datalayer, "w") as f:
        f.write(code)
    print(f"[ok] wrote data layer {os.path.abspath(args.out_datalayer)}")

    # quick EXPLAIN on the projection itself
    test_sql = f"""
WITH raw AS (SELECT * FROM read_parquet('{args.parquet}', filename=true)),
proj AS (
  SELECT
    {ts_expr} AS ts_parsed,
    {bid_px_ex} AS bid_px_d,
    {ask_px_ex} AS ask_px_d,
    {bid_sz_ex} AS bid_sz_d,
    {ask_sz_ex} AS ask_sz_d
  FROM raw
)
SELECT * FROM proj LIMIT 5
"""
    try:
        plan = con.execute(f"EXPLAIN {test_sql}").fetchall()
        print("[ok] projection EXPLAIN passed.")
    except Exception as e:
        print("[warn] projection EXPLAIN failed — check ob_mapping.json")
        print("      ", e)

if __name__ == "__main__":
    main()
