#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autotune.py — robust, fusion-aware, and compatible with your create_fusion_dataset() signature.

Key points:
- No 'ffill_limit_sec' anywhere (matches your local fusion_pipeline API).
- Adds optional fusion args: --orderbook-root, --date-start, --date-end, --max-ob-files, --min-coverage
- Works for both candles-only and fusion (falls back only if fusion fails upstream).
- Keeps memory safer; prints diagnostics.

Examples:
  # Candles-only, last 30 days
  python3 autotune.py --model candles --trials 20 --splits 5 \
    --candles-root ./data/candles --days 30 --deadzone-bps 5 --device cpu

  # Fusion on a known OB window (clip_to_ob=True inside)
  python3 autotune.py --model fusion --trials 20 --splits 5 \
    --candles-root ./data/candles --orderbook-root ./data/order_book \
    --date-start 2025-09-03 --date-end 2025-09-04 \
    --deadzone-bps 5 --device cpu
"""

import os, sys, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

# Ensure project imports (src/...) work when running from repo root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

# LightGBM / Optuna
import lightgbm as lgb
import optuna
from optuna.pruners import HyperbandPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, accuracy_score, classification_report
)

# Local imports
try:
    from src.config import HORIZONS, MODEL_STORAGE_PATH
except Exception:
    # Safe defaults if config import fails
    HORIZONS = [30, 60, 120, 240]
    MODEL_STORAGE_PATH = str(ROOT / "models")

from src.fusion_pipeline import create_fusion_dataset

# ---------- utils ----------
def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" in df.columns:
        df = df.dropna(subset=["ts"]).copy()
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.floor("min")
        df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True).floor("min")
        df = df.dropna().sort_index()
    else:
        df = df.copy()
        df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")).floor("min")
        df = df.sort_index()
    return df

def auto_features(df: pd.DataFrame) -> List[str]:
    feats = []
    for c in df.columns:
        if c == "ts" or c.startswith("label_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats

def make_labels(df: pd.DataFrame, horizon: int) -> Tuple[pd.Series, pd.Series]:
    base = df["close"]
    future = base.shift(-horizon)
    r = (future - base) / base.replace(0, np.nan)
    return (r > 0).astype(int).rename(f"label_{horizon}min"), r.rename(f"ret_{horizon}min")

def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    grid = np.unique(np.concatenate([thr, np.linspace(0.05, 0.95, 19)]))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        yhat = (proba >= t).astype(int)
        f = f1_score(y_true, yhat, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    yhat = (proba >= best_t).astype(int)
    metrics = {
        "threshold": best_t,
        "f1": f1_score(y_true, yhat, zero_division=0),
        "ap": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "acc": accuracy_score(y_true, yhat),
    }
    return best_t, metrics

def _sanitize_goss(params: Dict) -> Dict:
    # GOSS cannot use bagging
    if params.get("boosting_type") == "goss":
        params = params.copy()
        params["bagging_fraction"] = 1.0
        params["bagging_freq"] = 0
    return params

def _take(X: Any, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    else:
        return X[idx]

# ---------- Optuna objective ----------
def tscv_objective(
    trial,
    X: Any, y: Any,
    n_splits: int,
    gap_rows: int,
    device: str,
    fold_hook: Optional[callable] = None
) -> float:

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_jobs": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt","goss"]),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 100.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "max_bin": trial.suggest_int("max_bin", 63, 255),
    }
    if device in ("gpu","cuda"):
        params["device_type"] = device

    params = _sanitize_goss(params)

    # class balance
    pos = float((np.array(y) == 1).sum())
    neg = float((np.array(y) == 0).sum())
    if pos > 0 and neg > 0:
        params["scale_pos_weight"] = max(1.0, neg / pos)
    else:
        params["is_unbalance"] = True

    n_est = trial.suggest_int("n_estimators", 400, 3000)

    splits = np.array_split(np.arange(len(_take(X, np.arange(len(X)))), dtype=int), n_splits + 1)
    aucs, f1s = [], []
    for k in range(1, n_splits + 1):
        tr_idx = np.concatenate(splits[:k])
        va_idx = splits[k]
        if gap_rows > 0 and len(tr_idx) > gap_rows:
            tr_idx = tr_idx[:-gap_rows]  # embargo

        Xtr, ytr = _take(X, tr_idx), _take(y, tr_idx)
        Xva, yva = _take(X, va_idx), _take(y, va_idx)

        model = lgb.LGBMClassifier(**params, n_estimators=n_est, random_state=42)
        model.fit(
            Xtr, ytr, eval_set=[(Xva, yva)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(150, verbose=False)]
        )
        proba = model.predict_proba(Xva)[:, 1]
        auc_k = roc_auc_score(yva, proba)

        # tiny threshold sweep for fold F1
        thr_grid = np.linspace(0.1, 0.9, 9)
        f1_k = 0.0
        for t in thr_grid:
            yhat = (proba >= t).astype(int)
            f1_k = max(f1_k, f1_score(yva, yhat, zero_division=0))

        aucs.append(auc_k); f1s.append(f1_k)

        if fold_hook is not None:
            fold_hook(
                trial_number=int(trial.number)+1, k=k, K=n_splits,
                auc_mean=float(np.nanmean(aucs)), f1_mean=float(np.nanmean(f1s)), params=params
            )

        trial.report(float(np.nanmean(aucs)), step=k)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.nanmean(aucs))

def fit_best_model(X: Any, y: Any, best_params: Dict) -> lgb.LGBMClassifier:
    params = _sanitize_goss(best_params)
    final = lgb.LGBMClassifier(**params, random_state=42)
    final.fit(X, y)
    return final

# ---------- per-horizon ----------
def run_one_horizon(
    df: pd.DataFrame, horizon: int, trials: int, n_splits: int,
    deadzone_bps: Optional[float], device: str
) -> Dict:

    labels, future_ret = make_labels(df, horizon)
    data = df.join(labels).dropna()

    # dead-zone in basis points (bps)
    if deadzone_bps and "close" in data.columns:
        m = abs(future_ret.loc[data.index]) >= (deadzone_bps / 1e4)
        data = data.loc[m.index[m.values]]
        print(f"[DEADZONE] {deadzone_bps} bps → kept {len(data):,} rows")

    n = len(data)
    if n < 200:
        print(f"[WARN] {horizon}min — small sample: {n}")

    feats = auto_features(data)
    if not feats:
        return {"status":"no_features"}

    X_df = data[feats]
    y_series = data[f"label_{horizon}min"].astype(int)

    # progress bar (minimal)
    def _fold_hook(**kw): pass

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=HyperbandPruner()
    )
    gap_rows = int(max(1, horizon))
    study.optimize(
        lambda t: tscv_objective(t, X_df, y_series, n_splits, gap_rows, device, fold_hook=_fold_hook),
        n_trials=trials,
        show_progress_bar=False
    )

    completed = [tr for tr in study.trials if tr.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) == 0:
        print("[WARN] No completed Optuna trials — skipping model fit")
        return {"status": "no_completed_trials", "horizon": horizon, "n_rows": int(n)}

    best_trial = study.best_trial
    best_params = best_trial.params.copy()
    best_params.update({"objective":"binary","metric":"auc","verbosity":-1,"n_jobs":-1})
    if device in ("gpu","cuda"):
        best_params["device_type"] = device

    # holdout = last 15%
    val_end = int(n * 0.85)
    X_train_df, y_train = X_df.iloc[:val_end], y_series.iloc[:val_end]
    X_test_df,  y_test  = X_df.iloc[val_end:], y_series.iloc[val_end:]

    model = fit_best_model(X_train_df, y_train, best_params)
    proba = model.predict_proba(X_test_df)[:, 1]
    thr, metrics = tune_threshold(y_test.values, proba)
    y_pred = (proba >= thr).astype(int)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    # quick diagnostics
    try:
        print("\n[DIAG] proba.describe:")
        print(pd.Series(proba).describe())
        booster = getattr(model, "booster_", None)
        if booster is not None:
            imp = booster.feature_importance(importance_type="gain")
            names = booster.feature_name()
            df_imp = pd.DataFrame({"feature": names, "gain": imp}).sort_values("gain", ascending=False).head(20)
            print("[DIAG] Top 20 features (gain):")
            print(df_imp.to_string(index=False))
    except Exception as e:
        print(f"[DIAG] diagnostics failed: {e}")

    return {
        "status": "ok",
        "horizon": horizon,
        "features": feats,
        "n_rows": int(n),
        "best_params": best_params,
        "threshold": metrics["threshold"],
        "metrics": {
            "test_auc": metrics["roc_auc"],
            "test_ap": metrics["ap"],
            "test_f1": metrics["f1"],
            "test_acc": metrics["acc"],
            "cls_report": report
        },
        "model": model
    }

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["candles","fusion","orderbook"], default="candles")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--symbol", default="BTC/USDT")

    # DATA locations (match your fusion_pipeline signature)
    ap.add_argument("--candles-root", default="./data/candles")
    ap.add_argument("--orderbook-root", default="./data/order_book")

    # Window controls (only those your fusion accepts)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--date-start", type=str, default=None)
    ap.add_argument("--date-end", type=str, default=None)
    ap.add_argument("--max-ob-files", type=int, default=None)
    ap.add_argument("--min-coverage", type=float, default=0.05)
    ap.add_argument("--clip-to-ob", action="store_true", help="Clip candles to OB span (recommended for fusion)")

    # Modeling
    ap.add_argument("--deadzone-bps", type=float, default=None)
    ap.add_argument("--device", choices=["cpu","gpu","cuda"], default="cpu")
    args = ap.parse_args()

    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

    use_ob = (args.model in ["fusion","orderbook"])

    # NOTE: We pass ONLY the kwargs your fusion_pipeline exposes.
    df, data_type = create_fusion_dataset(
        use_orderbook=use_ob,
        symbol=args.symbol,
        monthly_root=args.candles_root,
        orderbook_root=(args.orderbook_root if use_ob else None),
        recent_days=args.days,
        date_start=args.date_start,
        date_end=args.date_end,
        max_ob_files=args.max_ob_files,
        min_coverage=args.min_coverage,
        clip_to_ob=args.clip_to_ob if use_ob else False,
    )

    if df is None or len(df) == 0:
        print("⚠️  No data returned by create_fusion_dataset(). Check date overlap / OB span / filters.")
        sys.exit(1)

    df = ensure_datetime_index(df)

    print("\n================ AUTOTUNE ================\n")
    results = []
    best_overall = None

    for h in HORIZONS:
        print(f"--- Horizon {h} min ---")
        out = run_one_horizon(
            df.copy(), h, trials=args.trials, n_splits=args.splits,
            deadzone_bps=args.deadzone_bps, device=args.device
        )
        if out.get("status") != "ok":
            print(f"[SKIP] {h}min ({out.get('status')})")
            continue

        # persist
        mdl_path = os.path.join(MODEL_STORAGE_PATH, f"autotune_lgbm_{h}min_{data_type}.joblib")
        import joblib
        joblib.dump({
            "model": out["model"],
            "features": out["features"],
            "params": out["best_params"],
            "threshold": out["threshold"],
            "horizon": h,
            "data_type": data_type
        }, mdl_path, compress=3)
        print(f"[SAVE] {mdl_path}")

        params_path = os.path.join(MODEL_STORAGE_PATH, f"autotune_best_params_{h}min_{data_type}.json")
        with open(params_path, "w") as pf:
            json.dump(out["best_params"], pf, indent=2)
        print(f"[PARAMS] saved → {params_path}")
        print("[PARAMS] best:")
        for k,v in sorted(out["best_params"].items()):
            print(f"  {k}: {v}")

        key = (out["metrics"]["test_ap"], out["metrics"]["test_auc"], out["metrics"]["test_f1"])
        if best_overall is None or key > best_overall[0]:
            best_overall = (key, out)

        results.append({
            "horizon": h,
            "auc": out["metrics"]["test_auc"],
            "ap": out["metrics"]["test_ap"],
            "f1": out["metrics"]["test_f1"],
            "acc": out["metrics"]["test_acc"],
            "threshold": out["threshold"]
        })
        print(f"[{h}m] AUC={out['metrics']['test_auc']:.4f}  AP={out['metrics']['test_ap']:.4f}  "
              f"F1={out['metrics']['test_f1']:.4f}  ACC={out['metrics']['test_acc']:.4f}  "
              f"thr={out['threshold']:.3f}\n")

    # summary + leaderboard
    results_sorted = sorted(results, key=lambda r: (r["ap"], r["auc"], r["f1"]), reverse=True)
    print("============== LEADERBOARD (by AP→AUC→F1) ==============")
    for r in results_sorted:
        print(f"{r['horizon']:>4}m | AUC {r['auc']:.4f} | AP {r['ap']:.4f} | F1 {r['f1']:.4f} | "
              f"ACC {r['acc']:.4f} | thr {r['threshold']:.3f}")

    summary = {
        "data_type": data_type,
        "symbol": args.symbol,
        "trials": args.trials,
        "splits": args.splits,
        "results": results_sorted,
        "best": {
            "horizon": best_overall[1]["horizon"] if best_overall else None,
            "metrics": best_overall[1]["metrics"] if best_overall else None,
            "threshold": best_overall[1]["threshold"] if best_overall else None,
            "params": best_overall[1]["best_params"] if best_overall else None
        }
    }
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
    summ_path = os.path.join(MODEL_STORAGE_PATH, f"autotune_summary_{data_type}.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY SAVED] {summ_path}")

if __name__ == "__main__":
    main()
