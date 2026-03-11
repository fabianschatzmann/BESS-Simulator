# core/results.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def compute_results_from_dispatch(
    dispatch_df: pd.DataFrame,
    runtime_s: float = 0.0,
    market_name: str = "day_ahead",
    e_nom_kwh: Optional[float] = None
) -> Dict[str, Any]:
    """
    Results dict compatible with Dashboard:
      results["kpis"], ["revenue_breakdown"], ["top_days"], ["worst_days"], ["timeseries"]

    This version is aligned to your current dispatch schema (screenshot):
      ts, price_da, price_id, p_da_kw, p_id_delta_kw, soc_kwh, soc_pct, rev_da_chf, rev_id_chf, revenue_chf, ...

    IMPORTANT:
      - Revenue is always computed in CHF using MWh: kW * h / 1000 * CHF/MWh
      - Intraday is reported as incremental value vs DA: (price_id - price_da) * delta_energy
    """
    df = dispatch_df.copy()
    if "ts" not in df.columns:
        raise ValueError("dispatch_df needs a 'ts' column.")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    dt_h = 1.0

    # -------------------------
    # SOC normalization for dashboard
    # -------------------------
    if "soc_kwh" in df.columns:
        df["soc_kwh"] = pd.to_numeric(df["soc_kwh"], errors="coerce")
    elif "soc" in df.columns:
        df["soc_kwh"] = pd.to_numeric(df["soc"], errors="coerce")
    else:
        df["soc_kwh"] = np.nan

    if "soc_pct" in df.columns:
        df["soc_pct"] = pd.to_numeric(df["soc_pct"], errors="coerce")
        df["soc"] = df["soc_pct"]
    elif e_nom_kwh is not None and float(e_nom_kwh) > 0:
        en = float(e_nom_kwh)
        df["soc_pct"] = (df["soc_kwh"] / en) * 100.0
        df["soc"] = df["soc_pct"]
    else:
        df["soc_pct"] = np.nan
        if "soc" not in df.columns:
            df["soc"] = df["soc_kwh"]

    # -------------------------
    # Identify power columns
    # -------------------------
    # DA net (kW)
    p_da_col = None
    for c in ["p_da_kw", "p_da_net_kw", "p_net_kw", "p_bess_da_kw"]:
        if c in df.columns:
            p_da_col = c
            break
    if p_da_col is not None:
        df[p_da_col] = pd.to_numeric(df[p_da_col], errors="coerce").fillna(0.0)
    else:
        df["p_da_kw"] = 0.0
        p_da_col = "p_da_kw"

    # ID delta net (kW)
    dp_col = None
    for c in ["p_id_delta_kw", "dp_net_kw", "dp_id_net_kw"]:
        if c in df.columns:
            dp_col = c
            break
    if dp_col is not None:
        df[dp_col] = pd.to_numeric(df[dp_col], errors="coerce").fillna(0.0)
    else:
        df["p_id_delta_kw"] = 0.0
        dp_col = "p_id_delta_kw"

    # Prices (CHF/MWh)
    if "price_da" in df.columns:
        df["price_da"] = pd.to_numeric(df["price_da"], errors="coerce").fillna(0.0)
    else:
        df["price_da"] = 0.0
    if "price_id" in df.columns:
        df["price_id"] = pd.to_numeric(df["price_id"], errors="coerce").fillna(0.0)
    else:
        df["price_id"] = 0.0

    # -------------------------
    # Revenue: DA cashflow
    # -------------------------
    if "rev_da_chf" in df.columns:
        df["rev_da_chf"] = pd.to_numeric(df["rev_da_chf"], errors="coerce").fillna(0.0)
        df["revenue_da_chf_h"] = df["rev_da_chf"]
    else:
        # compute correctly: kW -> MWh
        df["revenue_da_chf_h"] = (df[p_da_col] * dt_h / 1000.0) * df["price_da"]

    annual_rev_da = float(df["revenue_da_chf_h"].sum())

    # -------------------------
    # Revenue: ID incremental value vs DA (your intended definition)
    # -------------------------
    # delta energy in MWh
    df["dp_mwh"] = df[dp_col] * dt_h / 1000.0
    df["spread_settle"] = df["price_id"] - df["price_da"]
    df["revenue_id_chf_h"] = df["spread_settle"] * df["dp_mwh"]

    annual_rev_id = float(df["revenue_id_chf_h"].sum())

    # Total (DA + incremental ID)
    df["revenue"] = df["revenue_da_chf_h"] + df["revenue_id_chf_h"]
    annual_rev_total = float(df["revenue"].sum())

    # -------------------------
    # Cycles/year (simple EFC estimate if possible)
    # -------------------------
    cycles_per_year = 0.0
    if e_nom_kwh is not None and float(e_nom_kwh) > 0:
        en = float(e_nom_kwh)
        # best-effort throughput from charge/discharge columns
        if "p_charge_kw" in df.columns and "p_discharge_kw" in df.columns:
            pch = pd.to_numeric(df["p_charge_kw"], errors="coerce").fillna(0.0).clip(lower=0.0)
            pdis = pd.to_numeric(df["p_discharge_kw"], errors="coerce").fillna(0.0).clip(lower=0.0)
            e_th = float(((pch + pdis) * dt_h).sum())
            cycles_per_year = float(e_th / (2.0 * en + 1e-9))

    # Constraint violations
    violations = 0
    if "constraint_violation" in df.columns:
        violations = int(pd.to_numeric(df["constraint_violation"], errors="coerce").fillna(0).astype(int).sum())

    peak_reduction_kw = 0.0
    effective_rte = 0.0  # optional, keep 0 here

    # Daily ranking
    df["date"] = df["ts"].dt.date
    daily = df.groupby("date", as_index=False)["revenue"].sum().rename(columns={"revenue": "revenue_chf"})
    daily_sorted = daily.sort_values("revenue_chf", ascending=False)
    top_days = daily_sorted.head(10).to_dict(orient="records")
    worst_days = daily_sorted.tail(10).sort_values("revenue_chf", ascending=True).to_dict(orient="records")

    revenue_breakdown = [
        {"market": "Day-Ahead", "revenue_chf": annual_rev_da},
        {"market": "Intraday", "revenue_chf": annual_rev_id},
        {"market": "Total", "revenue_chf": annual_rev_total},
    ]

    kpis = {
        "annual_revenue_chf": annual_rev_total,
        "cycles_per_year": float(cycles_per_year),
        "peak_reduction_kw": float(peak_reduction_kw),
        "effective_rte": float(effective_rte),
        "constraint_violations": int(violations),
        "runtime_s": float(runtime_s),
        "annual_revenue_da_chf": annual_rev_da,
        "annual_revenue_id_chf": annual_rev_id,
    }

    return {
        "kpis": kpis,
        "revenue_breakdown": revenue_breakdown,
        "top_days": top_days,
        "worst_days": worst_days,
        "timeseries": df.drop(columns=["date"], errors="ignore"),
    }