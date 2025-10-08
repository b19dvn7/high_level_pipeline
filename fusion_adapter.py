#!/usr/bin/env python3
"""
Compatibility adapter for create_fusion_dataset.

This imports the project's actual `create_fusion_dataset` (from src.fusion_pipeline)
and wraps it to accept multiple common argument names so callers (tools/preflight,
autotune, scripts) don't need to be edited when argnames drift.

It:
 - maps aliases (recent_days -> days, candles_root -> monthly_root, orderbook_root -> ob_root)
 - accepts both date_start/date-end or days
 - converts string date inputs to pandas.Timestamp
 - uses inspect.signature to only pass supported args to the underlying function
"""

from __future__ import annotations
import sys, inspect
from pathlib import Path
from datetime import datetime
import pandas as pd

# ensure project root is importable when tools are executed from tools/
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

try:
    from src.fusion_pipeline import create_fusion_dataset as _create_fusion_dataset
except Exception as e:
    # graceful fallback so imports don't explode in preflight scripts
    _create_fusion_dataset = None
    _import_error = e
else:
    _import_error = None

def _norm_timestamp(x):
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return x
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.to_datetime(str(x))

def create_fusion_dataset(*,  # force kwargs
                          # common aliases (we accept many and map below)
                          days: int = None,
                          recent_days: int = None,
                          monthly_root: str = None,
                          candles_root: str = None,
                          candles_path: str = None,
                          ob_root: str = None,
                          orderbook_root: str = None,
                          orderbook_path: str = None,
                          max_ob_files: int = None,
                          max_files: int = None,
                          date_start: str = None,
                          date_end: str = None,
                          verbose: bool = False,
                          **extra):
    """
    Compatibility wrapper.

    Will call the real create_fusion_dataset with only parameters it supports.
    """
    if _create_fusion_dataset is None:
        raise ImportError(f"Underlying create_fusion_dataset could not be imported: {_import_error!r}")

    # normalize aliases
    if recent_days is not None and days is None:
        days = recent_days
    if candles_root is None:
        candles_root = monthly_root or candles_path
    if ob_root is None:
        ob_root = orderbook_root or orderbook_path
    if max_ob_files is None:
        max_ob_files = max_files

    kwargs = dict()
    # map back to the most likely param names used in this project
    if days is not None:
        kwargs['days'] = int(days)
    if candles_root is not None:
        kwargs['monthly_root'] = str(candles_root)
    if ob_root is not None:
        # some versions expect 'ob_root' or 'orderbook_root' or 'monthly_root' — we'll pass both if available
        kwargs['ob_root'] = str(ob_root)
    if max_ob_files is not None:
        kwargs['max_ob_files'] = int(max_ob_files)

    if date_start:
        kwargs['date_start'] = _norm_timestamp(date_start)
    if date_end:
        kwargs['date_end'] = _norm_timestamp(date_end)

    # inspect underlying function signature and only pass supported keys
    sig = inspect.signature(_create_fusion_dataset)
    passable = {k: v for k, v in kwargs.items() if k in sig.parameters}

    if verbose:
        print("[adapter] calling create_fusion_dataset with:", passable)

    return _create_fusion_dataset(**passable)
