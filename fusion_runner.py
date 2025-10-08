#!/usr/bin/env python3
"""
FusionRunner — orchestrates safe fusion between candles and orderbook shards.

This:
 - Scans OB shards, infers spans
 - Scans candle data spans
 - Finds the intersection window
 - Picks safe bounds (1-2 day “golden slice”) (or full interval if commanded)
 - Validates OB coverage, drops low coverage windows
 - Calls create_fusion_dataset (adapter) with safe arguments
 - Returns df, dtype, and diagnostics
"""

import sys
from pathlib import Path
from datetime import timedelta
import pandas as pd
import traceback

# ensure project root importable
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from src.fusion_adapter import create_fusion_dataset

def infer_shard_span(path: Path, ts_col="ts"):
    """Try to read minimal ts span from shard file."""
    try:
        # read small sample
        if path.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(path, columns=[ts_col])
        else:
            df = pd.read_csv(path, usecols=[ts_col], parse_dates=[ts_col], nrows=5000)
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna()
        if ts.empty:
            return None, None
        return ts.min(), ts.max()
    except Exception:
        return None, None

def scan_ob_root(ob_root: Path, max_shards: int = 500):
    """Scan OB root recursively, build list of usable shards with spans."""
    exts = {".parquet", ".pq", ".csv", ".json", ".gz"}
    shards = []
    for p in ob_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            a, b = infer_shard_span(p)
            if a is not None and b is not None:
                shards.append((p, a, b))
    # optionally limit number of shards (choose those intersecting central range)
    return shards

def scan_candle_root(c_root: Path):
    """Infer candle data span by sampling a few files."""
    # reuse simple method: read ts from first few parquet/csv in c_root
    exts = {".parquet", ".pq", ".csv", ".gz"}
    spans = []
    for p in c_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                if p.suffix.lower() in (".parquet", ".pq"):
                    df = pd.read_parquet(p, columns=['ts'])
                else:
                    df = pd.read_csv(p, usecols=['ts'], parse_dates=['ts'], nrows=5000)
                ts = pd.to_datetime(df['ts'], utc=True, errors="coerce").dropna()
                spans.append((ts.min(), ts.max()))
            except Exception:
                continue
        if len(spans) >= 5:
            break
    if not spans:
        return None, None
    mins = [a for (a,b) in spans]
    maxs = [b for (a,b) in spans]
    return min(mins), max(maxs)

def pick_safe_interval(ob_shards, candle_span, max_days: int = 2):
    """Pick a “golden slice” intersection interval of at most max_days."""
    if not ob_shards or candle_span[0] is None or candle_span[1] is None:
        return None, None
    # OB overall span
    ob_min = min(a for (_,a,_) in ob_shards)
    ob_max = max(b for (_,_,b) in ob_shards)
    c_min, c_max = candle_span
    # intersection
    start = max(ob_min, c_min)
    end = min(ob_max, c_max)
    if start >= end:
        return None, None
    # bound to max_days
    if (end - start) > timedelta(days=max_days):
        end = start + timedelta(days=max_days)
    return start, end

def run_fusion_safe(candle_root: Path, ob_root: Path,
                    days: int = None,
                    date_start: pd.Timestamp = None,
                    date_end: pd.Timestamp = None,
                    max_ob_files: int = 200,
                    min_coverage: float = 0.05,
                    max_ffill_sec: int = 90,
                    verbose: bool = False):
    """Main entry to run safe fusion with diagnostics."""
    # scan
    ob_shards = scan_ob_root(ob_root, max_shards=max_ob_files)
    candle_span = scan_candle_root(candle_root)
    if verbose:
        print("[scan] detected OB shards:", len(ob_shards))
        print("[scan] candle span:", candle_span)

    # pick intersection
    inter_start, inter_end = pick_safe_interval(ob_shards, candle_span)
    if inter_start is None:
        print("❌ No overlap between candles and OB data.")
        return None, None, {"error":"no_overlap"}

    # use date_start / date_end overrides if provided
    ds = date_start or inter_start
    de = date_end or inter_end

    if verbose:
        print("→ Using fusion interval:", ds, "→", de)

    # prepare adapter kwargs
    kw = dict()
    kw['date_start'] = ds
    kw['date_end'] = de
    kw['max_ob_files'] = max_ob_files
    kw['min_coverage'] = min_coverage
    kw['max_ffill_sec'] = max_ffill_sec
    kw['monthly_root'] = str(candle_root)
    kw['orderbook_root'] = str(ob_root)
    kw['verbose'] = verbose

    try:
        df, dtype = create_fusion_dataset(**kw)
        if verbose:
            print("Fusion output:", dtype, "rows=", len(df) if df is not None else None)
        return df, dtype, {"date_start": ds, "date_end": de}
    except Exception as e:
        print("❌ Fusion failed:", e)
        traceback.print_exc()
        return None, None, {"error":"fusion_failed", "exception": str(e)}

# Example CLI for direct run (optional)
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--candles-root", required=True)
    ap.add_argument("--orderbook-root", required=True)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--date-start", default=None)
    ap.add_argument("--date-end", default=None)
    ap.add_argument("--max-ob-files", type=int, default=200)
    ap.add_argument("--min-coverage", type=float, default=0.05)
    ap.add_argument("--max-ffill-sec", type=int, default=90)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ds = pd.to_datetime(args.date_start) if args.date_start else None
    de = pd.to_datetime(args.date_end) if args.date_end else None
    candle_root = Path(args.candles_root)
    ob_root = Path(args.orderbook_root)

    df, dtype, info = run_fusion_safe(candle_root, ob_root,
                                      days=args.days,
                                      date_start=ds, date_end=de,
                                      max_ob_files=args.max_ob_files,
                                      min_coverage=args.min_coverage,
                                      max_ffill_sec=args.max_ffill_sec,
                                      verbose=args.verbose)
    if df is None:
        print("No fusion result. Info:", info)
    else:
        print("Fusion successful. dtype:", dtype, "rows:", len(df))
