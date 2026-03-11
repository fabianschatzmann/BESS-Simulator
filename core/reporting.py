# core/reporting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RevenueDiagnostics:
    n: int
    n_missing: int

    sum_da_chf: float
    sum_id_inc_chf: float
    sum_total_chf: float

    share_sign_mismatch_settle: float
    share_sign_mismatch_fc: float

    corr_delta_vs_spread_settle: float
    corr_delta_vs_spread_fc: float

    min_id_inc_chf: float
    max_id_inc_chf: float


def _require_cols(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} missing columns: {missing}")


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    v = pd.concat([a, b], axis=1).dropna()
    if len(v) < 3:
        return float("nan")
    if np.isclose(v.iloc[:, 0].std(ddof=0), 0.0) or np.isclose(v.iloc[:, 1].std(ddof=0), 0.0):
        return float("nan")
    return float(v.iloc[:, 0].corr(v.iloc[:, 1]))


def compute_revenues_da_id_incremental(
    master: pd.DataFrame,
    dispatch: pd.DataFrame,
    *,
    ts_key_col: str = "ts_key",
    price_da_col: str = "price_da",
    price_id_col: str = "price_id",
    price_da_fc_col: str = "price_da_fc",
    price_id_fc_col: str = "price_id_fc",
    p_da_col: str = "p_da_kw",
    p_id_delta_col: str = "p_id_delta_kw",
) -> Tuple[pd.DataFrame, RevenueDiagnostics]:
    """
    Merge master + dispatch on ts_key and compute:
      rev_da_chf       = price_da * (p_da_kw/1000)
      rev_id_inc_chf   = (price_id - price_da) * (p_id_delta_kw/1000)   [incremental vs DA]
      revenue_chf      = rev_da_chf + rev_id_inc_chf

    Realistic mode:
      - NO ex-post clipping.
      - ID incremental can be negative ex-post due to forecast error / market risk.

    Returns:
      out_df: merged frame with computed columns
      diagnostics: sign mismatch and correlation checks (helps detect sign/mapping bugs)
    """
    _require_cols(master, [ts_key_col, price_da_col, price_id_col], "master")
    _require_cols(dispatch, [ts_key_col, p_da_col, p_id_delta_col], "dispatch")

    m = master[[ts_key_col, price_da_col, price_id_col]].copy()
    d = dispatch[[ts_key_col, p_da_col, p_id_delta_col]].copy()

    out = m.merge(d, on=ts_key_col, how="left", validate="one_to_one")

    # Numeric casting
    for c in [price_da_col, price_id_col, p_da_col, p_id_delta_col]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["spread_settle"] = out[price_id_col] - out[price_da_col]
    out["rev_da_chf"] = out[price_da_col] * (out[p_da_col] / 1000.0)
    out["rev_id_inc_chf"] = out["spread_settle"] * (out[p_id_delta_col] / 1000.0)
    out["revenue_chf"] = out["rev_da_chf"] + out["rev_id_inc_chf"]

    # Forecast spread if available
    has_fc = (price_da_fc_col in master.columns) and (price_id_fc_col in master.columns)
    if has_fc:
        mf = master[[ts_key_col, price_da_fc_col, price_id_fc_col]].copy()
        out = out.merge(mf, on=ts_key_col, how="left", validate="one_to_one")
        out["spread_fc"] = out[price_id_fc_col] - out[price_da_fc_col]

    # Diagnostics (settlement)
    valid = out[[price_da_col, price_id_col, p_id_delta_col]].dropna()
    n = int(len(out))
    n_missing = int(n - len(valid))

    spread_sign = np.sign(pd.to_numeric(valid[price_id_col] - valid[price_da_col], errors="coerce"))
    delta_sign = np.sign(pd.to_numeric(valid[p_id_delta_col], errors="coerce"))
    mismatch_settle = (spread_sign * delta_sign) < 0
    share_mismatch_settle = float(mismatch_settle.mean()) if len(valid) else float("nan")
    corr_delta_spread_settle = _safe_corr(valid[p_id_delta_col], valid[price_id_col] - valid[price_da_col])

    # Diagnostics (forecast)
    if has_fc:
        valid_fc = out[["spread_fc", p_id_delta_col]].dropna()
        spread_fc_sign = np.sign(pd.to_numeric(valid_fc["spread_fc"], errors="coerce"))
        delta_fc_sign = np.sign(pd.to_numeric(valid_fc[p_id_delta_col], errors="coerce"))
        mismatch_fc = (spread_fc_sign * delta_fc_sign) < 0
        share_mismatch_fc = float(mismatch_fc.mean()) if len(valid_fc) else float("nan")
        corr_delta_spread_fc = _safe_corr(valid_fc[p_id_delta_col], valid_fc["spread_fc"])
    else:
        share_mismatch_fc = float("nan")
        corr_delta_spread_fc = float("nan")

    diagnostics = RevenueDiagnostics(
        n=n,
        n_missing=n_missing,
        sum_da_chf=_to_float(out["rev_da_chf"].sum(skipna=True)),
        sum_id_inc_chf=_to_float(out["rev_id_inc_chf"].sum(skipna=True)),
        sum_total_chf=_to_float(out["revenue_chf"].sum(skipna=True)),
        share_sign_mismatch_settle=share_mismatch_settle,
        share_sign_mismatch_fc=share_mismatch_fc,
        corr_delta_vs_spread_settle=corr_delta_spread_settle,
        corr_delta_vs_spread_fc=corr_delta_spread_fc,
        min_id_inc_chf=_to_float(out["rev_id_inc_chf"].min(skipna=True)),
        max_id_inc_chf=_to_float(out["rev_id_inc_chf"].max(skipna=True)),
    )
    return out, diagnostics