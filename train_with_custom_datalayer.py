#!/usr/bin/env python3
# Shim runner: use adapters/mexc15s/rl_duckdb_data_layer_custom.py
# Patch it to:
#  - allow both "ts" and "timestamp"
#  - add union_by_name=true so mixed schemas in a glob work

import os, sys, importlib, re

ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "adapters", "mexc15s")
sys.path.insert(0, ADAPTER_DIR)

custom = importlib.import_module("rl_duckdb_data_layer_custom")

# 1) Robust timestamp: handle "ts" OR "timestamp"
custom.TS_EXPR = (
    'COALESCE('
    'TRY_CAST("ts" AS TIMESTAMP), '
    'TRY_CAST("timestamp" AS TIMESTAMP), '
    'to_timestamp(CAST(TRY_CAST("ts" AS BIGINT) AS DOUBLE)/1000.0), '
    'to_timestamp(CAST(TRY_CAST("timestamp" AS BIGINT) AS DOUBLE)/1000.0)'
    ')'
)

# 2) Wrap build_sql to inject union_by_name=true in read_parquet(...)
_old_build_sql = custom.build_sql
def _build_sql_union(*args, **kwargs):
    sql = _old_build_sql(*args, **kwargs)
    # Add union_by_name=true to every read_parquet call; keep filename=true intact
    sql = re.sub(
        r"read_parquet\(\s*'([^']*)'\s*,\s*filename\s*=\s*true\s*\)",
        r"read_parquet('\1', filename=true, union_by_name=true)",
        sql,
        flags=re.IGNORECASE,
    )
    return sql
custom.build_sql = _build_sql_union

# 3) Make trainer import our patched module under the expected name
sys.modules["rl_duckdb_data_layer"] = custom

# 4) Run the trainer with original CLI args
import train_td3bc_per_ensemble as trainer
if hasattr(trainer, "main"):
    trainer.main()
else:
    with open(trainer.__file__, "rb") as f:
        code = compile(f.read(), trainer.__file__, "exec")
        exec(code, {"__name__": "__main__"})
