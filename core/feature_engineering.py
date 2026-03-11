# core/feature_engineering.py
from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_ts(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    if ts_col not in df.columns:
        raise ValueError(f"Missing timestamp column '{ts_col}'.")
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col]).copy()
    out = out.sort_values(ts_col)
    return out


def add_calendar_features(
    df: pd.DataFrame,
    ts_col: str = "ts",
    add_month: bool = True,
    add_weekend: bool = True,
    add_cyclical_hour: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], errors="coerce")

    out["hour"] = ts.dt.hour.astype(int)
    out["dow"] = ts.dt.dayofweek.astype(int)  # Mon=0..Sun=6
    out["date"] = ts.dt.date
    out["day"] = ts.dt.floor("D")

    if add_cyclical_hour:
        h = out["hour"].to_numpy(dtype=float)
        out["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * h / 24.0)

    if add_weekend:
        out["is_weekend"] = (out["dow"] >= 5).astype(int)

    if add_month:
        m = ts.dt.month.astype(int)
        out["month"] = m
        mm = m.to_numpy(dtype=float)
        out["month_sin"] = np.sin(2 * np.pi * mm / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * mm / 12.0)

    return out


def add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: list[int],
    prefix: str = "",
) -> pd.DataFrame:
    """
    Leakage-safe lag features: uses past values via shift(k).
    """
    out = df.copy()
    s = pd.to_numeric(out[col], errors="coerce")
    for k in lags:
        out[f"{prefix}{col}_lag_{int(k)}"] = s.shift(int(k))
    return out


def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: list[int],
    min_frac: float = 0.25,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Leakage-safe rolling means: window mean shifted by 1 step so current timestep is not included.
    """
    out = df.copy()
    s = pd.to_numeric(out[col], errors="coerce")
    for w in windows:
        w = int(w)
        minp = max(3, int(np.ceil(w * float(min_frac))))
        out[f"{prefix}{col}_rollmean_{w}"] = s.rolling(window=w, min_periods=minp).mean().shift(1)
    return out


def add_market_block_keys(
    df: pd.DataFrame,
    ts_col: str,
    market: str,
    block_hours: int,
    gate_closure_offset_hours: float,
) -> pd.DataFrame:
    """
    Adds generic "block" + "gate closure" keys for later simulation alignment.
    """
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], errors="coerce")

    out["block_start"] = ts.dt.floor(f"{int(block_hours)}H")
    out["block_hour_in_block"] = ((ts - out["block_start"]) / pd.Timedelta(hours=1)).astype(int)
    out["gate_closure_time"] = out["block_start"] - pd.to_timedelta(float(gate_closure_offset_hours), unit="h")

    out["market"] = str(market)
    out["block_hours"] = int(block_hours)
    out["gate_closure_offset_h"] = float(gate_closure_offset_hours)
    return out


# =============================================================================
# Multi-target features (price_da + price_id)
# =============================================================================
def build_feature_frame_multi(
    master: pd.DataFrame,
    *,
    ts_col: str = "ts",
    target_cols: list[str],
    feature_cols: list[str],
    add_calendar: bool = True,
    add_price_history: bool = True,
    price_history_col_map: dict[str, str] | None = None,
    lags: list[int] | None = None,
    roll_windows: list[int] | None = None,
    drop_missing_targets: bool = True,
    drop_missing_features: bool = True,
) -> pd.DataFrame:
    df = ensure_ts(master, ts_col=ts_col)

    required = [ts_col] + list(target_cols) + list(feature_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master is missing required columns: {missing}")

    out = df[[ts_col] + list(target_cols) + list(feature_cols)].copy()
    out = out.rename(columns={ts_col: "ts"})

    if add_calendar:
        out = add_calendar_features(out, ts_col="ts")

    if add_price_history:
        lags = lags or []
        roll_windows = roll_windows or []
        price_history_col_map = price_history_col_map or {}

        for tgt in target_cols:
            hist_src = price_history_col_map.get(tgt, tgt)
            if hist_src not in df.columns:
                raise ValueError(f"price_history_col_map for target '{tgt}' points to '{hist_src}', not in master.")

            if hist_src not in out.columns:
                out[hist_src] = df[hist_src].values

            pref = f"{tgt}__"
            if lags:
                out = add_lag_features(out, col=hist_src, lags=lags, prefix=pref)
            if roll_windows:
                out = add_rolling_features(out, col=hist_src, windows=roll_windows, prefix=pref)

    if drop_missing_targets:
        out = out.dropna(subset=list(target_cols)).copy()

    if drop_missing_features:
        protected = {"ts", "date", "day", "block_start", "gate_closure_time", "market"}
        feat_cols = [c for c in out.columns if c not in protected and c not in set(target_cols)]
        out = out.dropna(subset=feat_cols).copy()

    out = out.sort_values("ts").reset_index(drop=True)
    return out


# =============================================================================
# Backwards-compatible single-target builder
# =============================================================================
def build_feature_frame(
    master: pd.DataFrame,
    ts_col: str,
    target_col: str,
    feature_cols: list[str],
    add_calendar: bool = True,
    add_price_history: bool = True,
    price_history_col: str | None = None,
    lags: list[int] | None = None,
    roll_windows: list[int] | None = None,
    drop_missing_target: bool = True,
    drop_missing_features: bool = True,
) -> pd.DataFrame:
    price_history_col_map = None
    if price_history_col is not None:
        price_history_col_map = {target_col: price_history_col}

    df = build_feature_frame_multi(
        master=master,
        ts_col=ts_col,
        target_cols=[target_col],
        feature_cols=feature_cols,
        add_calendar=add_calendar,
        add_price_history=add_price_history,
        price_history_col_map=price_history_col_map,
        lags=lags,
        roll_windows=roll_windows,
        drop_missing_targets=drop_missing_target,
        drop_missing_features=drop_missing_features,
    )
    return df


def coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != "ts"]
    rep = []
    n = len(df)
    for c in cols:
        miss = int(df[c].isna().sum())
        rep.append({"column": c, "missing_count": miss, "missing_frac": (miss / n) if n else np.nan})
    return pd.DataFrame(rep).sort_values(["missing_frac", "column"], ascending=[False, True])