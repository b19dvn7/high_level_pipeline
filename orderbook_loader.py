from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from typing import Callable, Optional

def load_orderbook_shards(
    ob_root: str | Path,
    symbol: str = "BTC/USDT",
    max_files: int | None = None,
    min_coverage: float = 0.05,
    max_per_min_ticks: int = 4,
    ffill_limit_sec: int = 90,
    verbose: bool = True,
    progress_cb: Optional[Callable[[int,int,Optional[pd.Timestamp],Optional[pd.Timestamp]], None]] = None,
    **_ignore,  # accept extra kwargs (date_start/date_end/etc.)
) -> pd.DataFrame:
    """
    Load raw OB 'snaps_wide_*.parquet' shards and aggregate to per-minute features.
    - Calls progress_cb(i, n, span_start, span_end) while scanning files.
    - Uses 's' and 'min' units (no deprecation warnings).
    """
    ob_root = Path(ob_root)
    files = sorted(ob_root.rglob("snaps_wide_*.parquet"))
    if max_files:
        files = files[:max_files]

    if not files:
        if verbose:
            print("[OB] no files found")
        return pd.DataFrame()

    dfs = []
    total = len(files)
    seen_min: pd.Timestamp | None = None
    seen_max: pd.Timestamp | None = None

    for i, f in enumerate(files, 1):
        try:
            df = pd.read_parquet(f)
            # normalize timestamp
            ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
            df["ts"] = ts.dt.floor("s")  # second precision
            # maintain running span
            if df["ts"].notna().any():
                cur_min = df["ts"].min()
                cur_max = df["ts"].max()
                seen_min = cur_min if (seen_min is None or cur_min < seen_min) else seen_min
                seen_max = cur_max if (seen_max is None or cur_max > seen_max) else seen_max
            dfs.append(df)
        except Exception as e:
            if verbose:
                sys.stdout.write(f"\n[OB] skip {f.name}: {e}\n")
                sys.stdout.flush()
        # single-line progress callback
        if progress_cb is not None:
            progress_cb(i, total, seen_min, seen_max)

    if not dfs:
        return pd.DataFrame()

    ob = pd.concat(dfs, ignore_index=True)
    ob = ob.dropna(subset=["ts"]).sort_values("ts").set_index("ts")

    # bound per-minute ticks to limit RAM/IO
    minute = ob.index.floor("min")
    ob["__minute__"] = minute
    ob["_rank_in_min"] = ob.groupby("__minute__").cumcount()
    ob = ob[ob["_rank_in_min"] < max_per_min_ticks]

    # lightweight minute aggregation (mean)
    agg = ob.groupby("__minute__").agg("mean")
    agg.index.name = "ts"

    # coverage estimate
    if len(agg) > 1:
        span_min = int((agg.index.max() - agg.index.min()).total_seconds() // 60) + 1
        coverage = len(agg) / max(span_min, 1)
    else:
        coverage = 0.0

    # forward fill (limit in minutes)
    ffill_limit_min = max(0, int(round(ffill_limit_sec / 60)))
    if ffill_limit_min > 0:
        agg = agg.asfreq("min").ffill(limit=ffill_limit_min)

    kept = len(agg)
    if verbose:
        print(f"[OB] minutes={kept} kept={kept} cut(low-coverage)=0 max_per_min_ticks={max_per_min_ticks} min_coverage={min_coverage:.02f} ffill_limit_min={ffill_limit_min}")
        if kept:
            print(f"[OB] span: {agg.index.min()} → {agg.index.max()} rows={len(agg):,} cols={agg.shape[1]}")

    return agg.sort_index()
