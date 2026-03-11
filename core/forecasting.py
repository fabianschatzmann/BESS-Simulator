# core/forecasting.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _select_model(model_name: str, params: dict | None = None):
    params = params or {}
    name = str(model_name).lower().strip()

    if name == "ridge":
        return Ridge(**{"alpha": 1.0, **params})
    if name == "lasso":
        return Lasso(**{"alpha": 0.001, "max_iter": 10000, **params})
    if name == "elasticnet":
        return ElasticNet(**{"alpha": 0.001, "l1_ratio": 0.2, "max_iter": 10000, **params})
    if name == "random_forest":
        return RandomForestRegressor(**{"n_estimators": 300, "random_state": 42, "n_jobs": -1, **params})
    if name == "gbrt":
        return GradientBoostingRegressor(**{"random_state": 42, **params})

    raise ValueError(f"Unknown model_name: {model_name}")


def _to_utc_naive(ts: pd.Series) -> pd.Series:
    """
    Converts timestamps to a robust merge key:
    - tz-aware -> convert to UTC and drop tz
    - tz-naive -> keep as-is
    """
    t = pd.to_datetime(ts, errors="coerce")
    try:
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    return t


def prepare_xy(
    features: pd.DataFrame,
    ts_col: str = "ts",
    target_col: str = "price_da",
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Converts feature table to X, y with timestamp.
    """
    df = features.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    drop_cols = drop_cols or []
    base_drop = {ts_col, "date", "day", "block_start", "gate_closure_time", "market"}
    base_drop |= set(drop_cols)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in features.")

    y = pd.to_numeric(df[target_col], errors="coerce")
    ts = df[ts_col].copy()

    X = df.drop(columns=[c for c in base_drop if c in df.columns] + [target_col], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    return X, y, ts


# =============================================================================
# PERFECT FORESIGHT (Upper Bound)
# =============================================================================
def perfect_foresight_predict_all(
    features: pd.DataFrame,
    ts_col: str = "ts",
    target_col: str = "price_da",
) -> dict:
    """
    Perfect foresight upper bound:
      y_pred := y_true

    Returns pred_df with schema compatible to other forecasters:
      ts, ts_key, y_true, y_pred
    """
    df = features.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in features.")

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    y_true = pd.to_numeric(df[target_col], errors="coerce")
    y_pred = y_true.copy()

    pred_df = pd.DataFrame({"ts": ts.values, "y_true": y_true.values, "y_pred": y_pred.values}).sort_values("ts")
    pred_df["ts"] = pd.to_datetime(pred_df["ts"], errors="coerce")
    pred_df["ts_key"] = _to_utc_naive(pred_df["ts"]).dt.floor("H")

    metrics = {
        "MAE": 0.0,
        "RMSE": 0.0,
        "n_predictions": int(np.isfinite(pred_df["y_pred"]).sum()),
        "mode": "perfect_foresight",
    }
    return {"pred_df": pred_df, "metrics": metrics, "model_name": "perfect_foresight"}


# =============================================================================
# FIT-ONCE (not leak-free)
# =============================================================================
def fit_once_predict_all(
    features: pd.DataFrame,
    ts_col: str = "ts",
    target_col: str = "price_da",
    model_name: str = "ridge",
    model_params: dict | None = None,
) -> dict:
    """
    FIT-ONCE mode:
    - Train model once on ALL available data
    - Predict for ALL rows where X is available
    - Returns pred_df with ts, ts_key, y_true, y_pred

    NOTE: Not leak-free (in-sample).
    """
    X, y, ts = prepare_xy(features, ts_col=ts_col, target_col=target_col)

    train_mask = y.notna() & X.notna().all(axis=1)
    if train_mask.sum() < 50:
        raise ValueError(f"Not enough training rows after filtering (got {int(train_mask.sum())}).")

    model = _select_model(model_name, model_params)
    model.fit(X.loc[train_mask].values, y.loc[train_mask].values)

    pred_mask = X.notna().all(axis=1)
    y_pred = np.full(shape=len(X), fill_value=np.nan, dtype=float)
    y_pred[pred_mask.values] = model.predict(X.loc[pred_mask].values)

    pred_df = pd.DataFrame({"ts": ts.values, "y_true": y.values, "y_pred": y_pred}).sort_values("ts")
    pred_df["ts"] = pd.to_datetime(pred_df["ts"], errors="coerce")
    pred_df["ts_key"] = _to_utc_naive(pred_df["ts"]).dt.floor("H")

    eval_mask = np.isfinite(pred_df["y_true"].values) & np.isfinite(pred_df["y_pred"].values)
    if eval_mask.sum() > 0:
        mae = mean_absolute_error(pred_df.loc[eval_mask, "y_true"], pred_df.loc[eval_mask, "y_pred"])
        rmse = float(np.sqrt(mean_squared_error(pred_df.loc[eval_mask, "y_true"], pred_df.loc[eval_mask, "y_pred"])))
    else:
        mae, rmse = np.nan, np.nan

    metrics = {
        "MAE_in_sample": float(mae) if np.isfinite(mae) else np.nan,
        "RMSE_in_sample": float(rmse) if np.isfinite(rmse) else np.nan,
        "n_predictions": int(np.isfinite(pred_df["y_pred"]).sum()),
        "n_eval_points": int(eval_mask.sum()),
    }

    return {"pred_df": pred_df, "metrics": metrics, "model_name": model_name}


# =============================================================================
# ROLLING BACKTEST BY DAY (leak-free, day-causal)
# =============================================================================
def rolling_backtest_by_day(
    features: pd.DataFrame,
    ts_col: str = "ts",
    target_col: str = "price_da",
    model_name: str = "ridge",
    model_params: dict | None = None,
    train_days_min: int = 60,
    retrain_every_days: int = 1,
) -> dict:
    """
    Rolling forecast: for each day D, train on all data < D and predict day D.
    Returns pred_df with ts, ts_key, y_true, y_pred.
    """
    X, y, ts = prepare_xy(features, ts_col=ts_col, target_col=target_col)

    df_all = pd.DataFrame({"ts": ts, "y": y}).copy()
    df_all["day"] = pd.to_datetime(df_all["ts"]).dt.floor("D")

    unique_days = np.sort(df_all["day"].dropna().unique())

    preds = []
    last_fit_day_idx = None
    model = None

    for i, day in enumerate(unique_days):
        train_mask = df_all["day"] < day
        test_mask = df_all["day"] == day

        train_days = pd.to_datetime(df_all.loc[train_mask, "day"]).nunique()
        if train_days < train_days_min:
            continue

        train_ok = train_mask & y.notna() & X.notna().all(axis=1)
        test_ok = test_mask & X.notna().all(axis=1)

        if train_ok.sum() < 50 or test_ok.sum() == 0:
            continue

        if (last_fit_day_idx is None) or ((i - last_fit_day_idx) >= retrain_every_days):
            model = _select_model(model_name, model_params)
            model.fit(X.loc[train_ok].values, y.loc[train_ok].values)
            last_fit_day_idx = i

        y_pred = model.predict(X.loc[test_ok].values)

        tmp = pd.DataFrame({
            "ts": df_all.loc[test_ok, "ts"].reset_index(drop=True),
            "y_true": y.loc[test_ok].reset_index(drop=True),
            "y_pred": pd.Series(y_pred),
            "day": day
        })
        preds.append(tmp)

    if not preds:
        raise ValueError("Backtest produced no predictions. Increase coverage or reduce train_days_min.")

    pred_df = pd.concat(preds, ignore_index=True).sort_values("ts")
    pred_df["ts"] = pd.to_datetime(pred_df["ts"], errors="coerce")
    pred_df["ts_key"] = _to_utc_naive(pred_df["ts"]).dt.floor("H")

    mae = mean_absolute_error(pred_df["y_true"], pred_df["y_pred"])
    rmse = float(np.sqrt(mean_squared_error(pred_df["y_true"], pred_df["y_pred"])))

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "n_predictions": int(len(pred_df)),
        "days_predicted": int(pred_df["day"].nunique()),
    }

    return {"pred_df": pred_df, "metrics": metrics, "model_name": model_name}


# =============================================================================
# NEW: Leak-free prediction at a given cutoff
# =============================================================================
def predict_with_cutoff(
    features: pd.DataFrame,
    *,
    cutoff_ts: pd.Timestamp,
    ts_col: str,
    target_col: str,
    model_name: str,
    model_params: dict | None,
    drop_cols: list[str] | None = None,
    predict_mask: pd.Series | None = None,
    min_train_rows: int = 200,
) -> dict:
    """
    Train using only rows with ts < cutoff_ts, predict rows where predict_mask True (or all rows).
    Leak-free as long as your features themselves are leak-free.
    """
    X, y, ts = prepare_xy(features, ts_col=ts_col, target_col=target_col, drop_cols=drop_cols)
    ts = pd.to_datetime(ts, errors="coerce")

    cutoff_ts = pd.Timestamp(cutoff_ts)

    train_mask = (ts < cutoff_ts) & y.notna() & X.notna().all(axis=1)
    if int(train_mask.sum()) < int(min_train_rows):
        pred_df = pd.DataFrame({"ts": ts.values, "y_true": y.values, "y_pred": np.nan})
        pred_df["ts"] = pd.to_datetime(pred_df["ts"], errors="coerce")
        pred_df["ts_key"] = _to_utc_naive(pred_df["ts"]).dt.floor("H")
        pred_df["cutoff_ts"] = cutoff_ts
        return {"pred_df": pred_df, "metrics": {"train_rows": int(train_mask.sum())}, "model_name": model_name}

    model = _select_model(model_name, model_params)
    model.fit(X.loc[train_mask].values, y.loc[train_mask].values)

    if predict_mask is None:
        predict_mask = X.notna().all(axis=1)
    else:
        predict_mask = predict_mask & X.notna().all(axis=1)

    y_pred = np.full(shape=len(X), fill_value=np.nan, dtype=float)
    if int(predict_mask.sum()) > 0:
        y_pred[predict_mask.values] = model.predict(X.loc[predict_mask].values)

    pred_df = pd.DataFrame({"ts": ts.values, "y_true": y.values, "y_pred": y_pred}).sort_values("ts")
    pred_df["ts"] = pd.to_datetime(pred_df["ts"], errors="coerce")
    pred_df["ts_key"] = _to_utc_naive(pred_df["ts"]).dt.floor("H")
    pred_df["cutoff_ts"] = cutoff_ts

    eval_mask = np.isfinite(pred_df["y_true"].values) & np.isfinite(pred_df["y_pred"].values)
    if int(eval_mask.sum()) > 0:
        mae = mean_absolute_error(pred_df.loc[eval_mask, "y_true"], pred_df.loc[eval_mask, "y_pred"])
        rmse = float(np.sqrt(mean_squared_error(pred_df.loc[eval_mask, "y_true"], pred_df.loc[eval_mask, "y_pred"])))
    else:
        mae, rmse = np.nan, np.nan

    metrics = {
        "MAE": float(mae) if np.isfinite(mae) else np.nan,
        "RMSE": float(rmse) if np.isfinite(rmse) else np.nan,
        "train_rows": int(train_mask.sum()),
        "pred_rows": int(np.isfinite(pred_df["y_pred"]).sum()),
    }
    return {"pred_df": pred_df, "metrics": metrics, "model_name": model_name}


def rolling_cutoff_forecast_hourly(
    features: pd.DataFrame,
    *,
    ts_col: str,
    target_col: str,
    model_name: str,
    model_params: dict | None,
    drop_cols: list[str] | None = None,
    retrain_every_hours: int = 24,
    horizon_hours: int = 24,
    min_train_rows: int = 200,
) -> dict:
    """
    Leak-free rolling forecast for intraday-style decisions:
      - At each decision time t_dec (hourly grid), train on ts < t_dec
      - Predict for ts in [t_dec, t_dec + horizon_hours)
    To keep runtime sane: retrain every N hours.
    """
    df = features.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    X, y, ts = prepare_xy(df, ts_col=ts_col, target_col=target_col, drop_cols=drop_cols)
    ts = pd.to_datetime(ts, errors="coerce").dt.floor("H")

    unique_hours = pd.Index(ts.dropna().unique()).sort_values()
    if unique_hours.empty:
        raise ValueError("No valid timestamps for hourly rolling forecast.")

    y_pred = np.full(shape=len(df), fill_value=np.nan, dtype=float)

    last_fit_idx = None
    model = None

    for i, t_dec in enumerate(unique_hours):
        horizon_mask = (ts >= t_dec) & (ts < (t_dec + pd.Timedelta(hours=int(horizon_hours))))
        if not horizon_mask.any():
            continue

        if (last_fit_idx is None) or ((i - last_fit_idx) >= int(max(1, retrain_every_hours))):
            train_mask = (ts < t_dec) & y.notna() & X.notna().all(axis=1)
            if int(train_mask.sum()) >= int(min_train_rows):
                model = _select_model(model_name, model_params)
                model.fit(X.loc[train_mask].values, y.loc[train_mask].values)
            else:
                model = None
            last_fit_idx = i

        if model is None:
            continue

        pred_mask = horizon_mask & X.notna().all(axis=1)
        if int(pred_mask.sum()) == 0:
            continue

        y_pred[pred_mask.values] = model.predict(X.loc[pred_mask].values)

    pred_df = pd.DataFrame(
        {
            "ts": pd.to_datetime(df[ts_col], errors="coerce"),
            "y_true": pd.to_numeric(df[target_col], errors="coerce"),
            "y_pred": y_pred,
        }
    ).sort_values("ts")
    pred_df["ts_key"] = _to_utc_naive(pred_df["ts"]).dt.floor("H")

    eval_mask = np.isfinite(pred_df["y_true"].values) & np.isfinite(pred_df["y_pred"].values)
    if int(eval_mask.sum()) > 0:
        mae = mean_absolute_error(pred_df.loc[eval_mask, "y_true"], pred_df.loc[eval_mask, "y_pred"])
        rmse = float(np.sqrt(mean_squared_error(pred_df.loc[eval_mask, "y_true"], pred_df.loc[eval_mask, "y_pred"])))
    else:
        mae, rmse = np.nan, np.nan

    metrics = {
        "MAE": float(mae) if np.isfinite(mae) else np.nan,
        "RMSE": float(rmse) if np.isfinite(rmse) else np.nan,
        "n_predictions": int(np.isfinite(pred_df["y_pred"]).sum()),
        "retrain_every_hours": int(retrain_every_hours),
        "horizon_hours": int(horizon_hours),
        "min_train_rows": int(min_train_rows),
    }
    return {"pred_df": pred_df, "metrics": metrics, "model_name": model_name}