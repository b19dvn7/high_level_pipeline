#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

from src.data_loader import load_monthly_candles
from src.feature_engineering import build_candle_features
from src.orderbook_loader import load_orderbook_shards

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" in df.columns:
        df = df.dropna(subset=["ts"]).copy()
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.loc[ts.notna()]
        df["ts"] = ts.dt.floor("min")
        df = df.set_index("ts").sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        df = df.copy()
        df.index = idx.floor("min")
        df = df.sort_index()
    else:
        raise ValueError("DataFrame lacks 'ts' column and DatetimeIndex.")
    return df

def _slice_by_dates(df: pd.DataFrame, ds: Optional[str], de: Optional[str]) -> pd.DataFrame:
    if ds:
        df = df[df.index >= pd.to_datetime(ds, utc=True)]
    if de:
        df = df[df.index <= pd.to_datetime(de, utc=True)]
    return df

def create_fusion_dataset(
    use_orderbook: bool,
    symbol: str = "BTC/USDT",
    monthly_root: str = "./data/candles",
    orderbook_root: Optional[str] = None,
    recent_days: Optional[int] = None,
    max_ob_files: Optional[int] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    clip_to_ob: bool = True,
    verbose: bool = True,
    # NEW: gap handling knobs (plumbed through)
    max_ffill_sec: int = 15,
    min_coverage: float = 0.20,
) -> Tuple[pd.DataFrame, str]:
    # Candles
    candles = load_monthly_candles(monthly_root=monthly_root, symbol=symbol, verbose=verbose)
    candles = _ensure_dt_index(candles)

    if date_start or date_end:
        candles = _slice_by_dates(candles, date_start, date_end)
        if verbose: print(f"[FILTER] explicit date → {len(candles):,} rows")
    elif recent_days:
        cutoff = candles.index.max() - pd.Timedelta(days=int(recent_days))
        candles = candles[candles.index >= cutoff]
        if verbose: print(f"[FILTER] recent_days={recent_days} → {len(candles):,} rows")

    if verbose: print("[FEATURES] Engineering candle features...")
    candles_fe = build_candle_features(candles)
    candles_fe = _ensure_dt_index(candles_fe)
    if verbose: print(f"[FEATURES] Generated {candles_fe.shape[1]} candle features")

    if not use_orderbook:
        return candles_fe, "candles"

    if not orderbook_root or not Path(orderbook_root).exists():
        if verbose:
            print(f"[FUSION] OB root missing → candles-only ({orderbook_root})")
        return candles_fe, "candles"

    try:
        ob = load_orderbook_shards(
            ob_root=str(Path(orderbook_root).resolve()),
            symbol=symbol,
            max_files=max_ob_files,
            recent_days=None if (date_start or date_end) else recent_days,
            verbose=verbose,
            max_ffill_sec=max_ffill_sec,
            min_coverage=min_coverage,
        )
    except Exception as e:
        if verbose:
            print(f"[FUSION] OB load failed: {e} → candles-only")
        return candles_fe, "candles"

    if len(ob) == 0:
        if verbose:
            print("[FUSION] OB after resample/QC is empty → candles-only")
        return candles_fe, "candles"

    ob_min, ob_max = ob.index.min(), ob.index.max()
    if clip_to_ob:
        before = len(candles_fe)
        candles_fe = candles_fe[(candles_fe.index >= ob_min) & (candles_fe.index <= ob_max)]
        if verbose:
            print(f"[FUSION] clip_to_ob: {before:,} → {len(candles_fe):,} (candles range to OB span)")

    if date_start or date_end:
        ob = _slice_by_dates(ob, date_start, date_end)
        candles_fe = _slice_by_dates(candles_fe, date_start, date_end)

    if len(candles_fe) == 0:
        if verbose:
            print("[FUSION] No candle rows after clipping/date filters.\n"
                  f"  candle span: {candles.index.min()} → {candles.index.max()}\n"
                  f"  OB span    : {ob_min} → {ob_max}")
        return candles_fe, "candles"

    df = candles_fe.join(ob, how="inner", lsuffix="", rsuffix="_ob")
    if verbose:
        print(f"[FUSION] Joined candles+OB → rows={len(df):,} cols={df.shape[1]}")
        print(f"[FUSION] OB span: {ob_min} → {ob_max}")

    if len(df) == 0:
        if verbose:
            print("[FUSION] ⚠️ Join empty. Likely no timestamp intersection after filters.")
        return candles_fe, "candles"

    return df, "fusion"
