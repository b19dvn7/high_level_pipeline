# src/feature_engineering.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _safe_rolling(s: pd.Series, w: int, fn: str):
    if w <= 1:  # avoid zero division & no-op
        return s.copy()
    r = s.rolling(w, min_periods=max(2, int(w*0.5)))
    if fn == "mean":   return r.mean()
    if fn == "std":    return r.std()
    if fn == "max":    return r.max()
    if fn == "min":    return r.min()
    if fn == "sum":    return r.sum()
    raise ValueError(f"Unknown fn {fn}")

def engineer_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: ts index (UTC), open, high, low, close, volume (float)
    Output: features dataframe (keeps OHLCV + engineered numeric features).
    """
    assert isinstance(df.index, pd.DatetimeIndex), "df must be indexed by DatetimeIndex (UTC)."
    out = df.copy()

    # Basic returns
    out["ret_1"]   = out["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    out["ret_5"]   = out["close"].pct_change(5)
    out["ret_15"]  = out["close"].pct_change(15)
    out["hl_spread"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_spread"] = (out["open"] - out["close"]) / out["close"].replace(0, np.nan)

    # Volatility proxies
    out["vol_30"]   = _safe_rolling(out["ret_1"], 30, "std")
    out["vol_120"]  = _safe_rolling(out["ret_1"], 120, "std")

    # Momentum
    out["mom_30"]   = out["close"] / out["close"].shift(30) - 1.0
    out["mom_120"]  = out["close"] / out["close"].shift(120) - 1.0

    # Moving averages & ratios
    out["ma_20"]  = _safe_rolling(out["close"], 20, "mean")
    out["ma_50"]  = _safe_rolling(out["close"], 50, "mean")
    out["ma_200"] = _safe_rolling(out["close"], 200, "mean")
    out["ma_20_rel"]  = out["close"] / out["ma_20"]  - 1.0
    out["ma_50_rel"]  = out["close"] / out["ma_50"]  - 1.0
    out["ma_200_rel"] = out["close"] / out["ma_200"] - 1.0

    # Volume features
    out["v_ma_20"] = _safe_rolling(out["volume"], 20, "mean")
    out["v_rel_20"] = out["volume"] / out["v_ma_20"].replace(0, np.nan)

    # Clean
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

# -------------------- compat shim: build_candle_features --------------------
# Some scripts (e.g., preflight/fusion pipeline) import `build_candle_features`.
# If it's not defined in this module, create a thin wrapper that either calls an
# existing builder you already have or computes a compact default feature set.
try:
    build_candle_features  # noqa: F401
except NameError:
    import pandas as _pd
    import numpy as _np

    # Try to reuse an existing function name from older codebases if present
    _CANDIDATE_IMPLS = [
        'engineer_candle_features',
        'engineer_candles_features',
        'make_candle_features',
        'make_features',
        'candle_features',
    ]

    _delegate = None
    for _name in _CANDIDATE_IMPLS:
        if _name in globals() and callable(globals()[_name]):
            _delegate = globals()[_name]
            break

    def _default_candle_features(df: _pd.DataFrame) -> _pd.DataFrame:
        """Minimal, fast feature set that doesn't blow up memory."""
        df = df.copy()
        # sanity & ordering
        idx = df.index
        if 'ts' in df.columns:
            df['ts'] = _pd.to_datetime(df['ts'], utc=True, errors='coerce')
            df = df.dropna(subset=['ts']).set_index('ts').sort_index()
        else:
            if not isinstance(df.index, _pd.DatetimeIndex):
                df.index = _pd.to_datetime(df.index, utc=True, errors='coerce')
                df = df.dropna().sort_index()
        # required columns
        needed = ['open','high','low','close','volume']
        for c in needed:
            if c not in df.columns:
                df[c] = _np.nan
        # returns (log)
        def _logret(x): 
            x = x.replace(0, _np.nan)
            return _np.log(x / x.shift(1))
        df['logret_1']  = _logret(df['close'])
        df['logret_5']  = df['logret_1'].rolling(5).sum()
        df['logret_10'] = df['logret_1'].rolling(10).sum()
        df['logret_30'] = df['logret_1'].rolling(30).sum()

        # simple MAs / EMAs
        for w in (5,15,30,60,120):
            df[f'ma_{w}']  = df['close'].rolling(w, min_periods=max(2, w//3)).mean()
        for w in (5,15,30,60,120):
            df[f'ema_{w}'] = df['close'].ewm(span=w, adjust=False, min_periods=max(2, w//3)).mean()

        # volatility proxies
        for w in (30,60):
            df[f'vol_{w}'] = df['logret_1'].rolling(w, min_periods=max(2, w//3)).std()

        # volume features
        for w in (15,30,60):
            df[f'volu_ma_{w}']  = df['volume'].rolling(w, min_periods=max(2, w//3)).mean()
            df[f'volu_std_{w}'] = df['volume'].rolling(w, min_periods=max(2, w//3)).std()

        # spreads & ranges
        df['oc_spread'] = (df['close'] - df['open']) / df['open'].replace(0, _np.nan)
        df['hl_spread'] = (df['high'] - df['low'])  / df['open'].replace(0, _np.nan)

        # Bollinger (20)
        mid = df['close'].rolling(20, min_periods=7).mean()
        std = df['close'].rolling(20, min_periods=7).std()
        df['bb_mid'] = mid
        df['bb_up']  = mid + 2*std
        df['bb_dn']  = mid - 2*std

        # simple time encodings
        if isinstance(df.index, _pd.DatetimeIndex):
            # minute of day in [0, 1440)
            mod = (df.index.view('int64')//10**9) % (24*60*60)
            minute = (mod // 60).astype(int)
            df['min_of_day'] = minute
            # sin/cos daily
            twopi = 2*_np.pi
            df['tod_sin'] = _np.sin(twopi * minute / (24*60))
            df['tod_cos'] = _np.cos(twopi * minute / (24*60))

        # keep numeric only
        out = df.select_dtypes(include='number').copy()
        out = out.dropna(how='all')
        # restore original index type if needed
        out.index = df.index
        return out

    def build_candle_features(df: _pd.DataFrame) -> _pd.DataFrame:
        """Compatibility wrapper for callers expecting `build_candle_features`."""
        if _delegate is not None:
            return _delegate(df)
        return _default_candle_features(df)

    # make it visible to wildcard imports
    try:
        __all__ = list(set(list(globals().get('__all__', [])) + ['build_candle_features']))
    except Exception:
        pass
# ------------------ end compat shim ------------------
