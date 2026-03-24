# pages/5_Dashboard.py
# -*- coding: utf-8 -*-

import io
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from core.scenario_store import load_parquet, load_config
from core.results import compute_results_from_dispatch
from core.reporting import compute_revenues_da_id_incremental

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

# ============================================================
# Helpers
# ============================================================
def _to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _get_batt(cfg: dict, key: str, default=None):
    try:
        return cfg.get("battery", {}).get(key, default)
    except Exception:
        return default


def _get_float(cfg: dict, keys, default=None):
    if not isinstance(keys, list):
        keys = [keys]
    for k in keys:
        v = _get_batt(cfg, k, None)
        try:
            if v is not None:
                return float(v)
        except Exception:
            pass
    return default


def _get_eco_float(cfg: dict, key: str, default=None):
    try:
        v = cfg.get("economics", {}).get(key, default)
        return float(v) if v is not None else default
    except Exception:
        return default


def _get_eco_int(cfg: dict, key: str, default=None):
    try:
        v = cfg.get("economics", {}).get(key, default)
        return int(v) if v is not None else default
    except Exception:
        return default


def _get_total_project_cost_chf(cfg: dict) -> float:
    eco_candidates = [
        "total_project_cost_chf",
        "project_cost_total_chf",
        "total_capex_chf",
        "capex_total_chf",
        "project_capex_chf",
        "capex_chf",
        "investment_cost_chf",
        "total_investment_chf",
    ]

    for key in eco_candidates:
        v = _get_eco_float(cfg, key, default=None)
        try:
            if v is not None and np.isfinite(float(v)) and float(v) > 0:
                return float(v)
        except Exception:
            pass

    top_candidates = [
        "total_project_cost_chf",
        "project_cost_total_chf",
        "total_capex_chf",
        "capex_total_chf",
        "investment_cost_chf",
    ]
    for key in top_candidates:
        try:
            v = cfg.get(key, None)
            if v is not None and np.isfinite(float(v)) and float(v) > 0:
                return float(v)
        except Exception:
            pass

    return 0.0


def _normalize_pct_rate(value, default=0.023) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)

    if np.isnan(v):
        return float(default)

    if v < 0:
        return 0.0

    if v > 1.0:
        v = v / 100.0

    return float(v)


def _ensure_ts_key(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    if "ts_key" not in out.columns:
        if "ts" in out.columns:
            out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
            out["ts_key"] = out["ts"].dt.floor("H")
        else:
            return out

    out["ts_key"] = pd.to_datetime(out["ts_key"], errors="coerce").dt.floor("H")
    out = out.dropna(subset=["ts_key"]).sort_values("ts_key").reset_index(drop=True)

    if "ts" not in out.columns:
        out["ts"] = out["ts_key"]
    else:
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
        out["ts"] = out["ts"].fillna(out["ts_key"])

    try:
        out["ts"] = out["ts"].dt.tz_localize(None)
    except Exception:
        pass

    return out


def _infer_market_from_timeseries(ts: pd.DataFrame) -> dict:
    cols = set(ts.columns)

    has_sdl = (
        any(str(c).startswith("sdl_") for c in cols)
        or ("sdl_total_rev_chf_h" in cols)
        or ("prl_sym_accepted" in cols)
        or ("srl_up_accepted" in cols)
        or ("srl_down_accepted" in cols)
    )

    has_da = (
        ("p_charge_kw" in cols and "p_discharge_kw" in cols)
        or ("p_bess" in cols)
        or ("soc_kwh" in cols)
        or ("price_da" in cols)
    )

    has_id = (
        ("price_id" in cols)
        or ("price_id_fc" in cols)
        or ("p_id_delta_kw" in cols)
        or ("rev_id_inc_chf" in cols)
    )

    return {
        "has_sdl": bool(has_sdl),
        "has_da": bool(has_da),
        "has_id": bool(has_id),
    }


def _compute_sdl_kpis(ts: pd.DataFrame) -> dict:
    k = {}
    if ts is None or ts.empty:
        return k

    if "sdl_total_rev_chf_h" in ts.columns:
        k["sdl_total_revenue_chf"] = float(
            pd.to_numeric(ts["sdl_total_rev_chf_h"], errors="coerce").fillna(0.0).sum()
        )

    if "sdl_total_rev_cap_chf_h" in ts.columns:
        k["sdl_total_revenue_cap_chf"] = float(
            pd.to_numeric(ts["sdl_total_rev_cap_chf_h"], errors="coerce").fillna(0.0).sum()
        )

    if "sdl_total_rev_energy_chf_h" in ts.columns:
        k["sdl_total_revenue_energy_chf"] = float(
            pd.to_numeric(ts["sdl_total_rev_energy_chf_h"], errors="coerce").fillna(0.0).sum()
        )

    for pref in ["prl_sym", "srl_up", "srl_down"]:
        acc = f"{pref}_accepted"
        rev_total = f"{pref}_rev_total_chf_h"
        rev_old = f"{pref}_rev_chf_h"
        rev_cap = f"{pref}_rev_cap_chf_h"
        rev_energy = f"{pref}_rev_energy_chf_h"

        if acc in ts.columns:
            k[f"{pref}_accept_rate"] = float(
                pd.to_numeric(ts[acc], errors="coerce").fillna(0.0).mean()
            )

        if rev_total in ts.columns:
            k[f"{pref}_revenue_chf"] = float(
                pd.to_numeric(ts[rev_total], errors="coerce").fillna(0.0).sum()
            )
        elif rev_old in ts.columns:
            k[f"{pref}_revenue_chf"] = float(
                pd.to_numeric(ts[rev_old], errors="coerce").fillna(0.0).sum()
            )
        else:
            if (rev_cap in ts.columns) or (rev_energy in ts.columns):
                cap_sum = float(
                    pd.to_numeric(ts.get(rev_cap, 0.0), errors="coerce").fillna(0.0).sum()
                )
                en_sum = float(
                    pd.to_numeric(ts.get(rev_energy, 0.0), errors="coerce").fillna(0.0).sum()
                )
                k[f"{pref}_revenue_chf"] = cap_sum + en_sum

    return k


def _efc_cycles_from_dispatch(ts: pd.DataFrame, e_nom_kwh: float, dt_h: float = 1.0) -> float:
    if ts is None or ts.empty:
        return 0.0
    if e_nom_kwh is None or e_nom_kwh <= 0:
        return 0.0
    if "p_charge_kw" not in ts.columns or "p_discharge_kw" not in ts.columns:
        return 0.0

    pch = pd.to_numeric(ts["p_charge_kw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    pdis_raw = pd.to_numeric(ts["p_discharge_kw"], errors="coerce").fillna(0.0)
    pdis = (
        (-pdis_raw).clip(lower=0.0)
        if (pdis_raw <= 0).all() and (pdis_raw.abs().sum() > 0)
        else pdis_raw.abs().clip(lower=0.0)
    )

    e_through_kwh = float(((pch + pdis) * float(dt_h)).sum())
    return float(e_through_kwh / (2.0 * float(e_nom_kwh) + 1e-9))


def _build_financial_projection(
    annual_revenue_chf: float,
    total_project_cost_chf: float,
    opex_chf_per_year: float,
    wacc: float,
    project_lifetime_years: int,
    degradation_pct_per_year: float,
) -> pd.DataFrame:
    capex = float(total_project_cost_chf or 0.0)
    annual_revenue_y1 = float(annual_revenue_chf or 0.0)
    opex = float(opex_chf_per_year or 0.0)
    r = float(wacc or 0.0)
    n = int(project_lifetime_years or 0)
    deg = _normalize_pct_rate(degradation_pct_per_year, default=0.023)

    rows = []
    cum_net = 0.0
    cum_disc = -capex

    for year in range(1, n + 1):
        degradation_factor = (1.0 - deg) ** (year - 1)
        revenue_y = annual_revenue_y1 * degradation_factor
        net_cashflow_y = revenue_y - opex

        if (1.0 + r) != 0:
            discounted_net_cashflow_y = net_cashflow_y / ((1.0 + r) ** year)
        else:
            discounted_net_cashflow_y = 0.0

        cum_net += net_cashflow_y
        cum_disc += discounted_net_cashflow_y

        rows.append(
            {
                "year": year,
                "degradation_factor": degradation_factor,
                "revenue_chf": revenue_y,
                "opex_chf": opex,
                "net_cashflow_chf": net_cashflow_y,
                "discounted_net_cashflow_chf": discounted_net_cashflow_y,
                "cum_net_cashflow_chf": cum_net,
                "cum_discounted_cashflow_after_capex_chf": cum_disc,
            }
        )

    return pd.DataFrame(rows)


def _compute_npv_and_payback(
    annual_revenue_chf: float,
    total_project_cost_chf: float,
    opex_chf_per_year: float,
    wacc: float,
    project_lifetime_years: int,
    degradation_pct_per_year: float = 0.023,
) -> tuple[float, float | None, pd.DataFrame]:
    capex = float(total_project_cost_chf or 0.0)

    proj = _build_financial_projection(
        annual_revenue_chf=annual_revenue_chf,
        total_project_cost_chf=total_project_cost_chf,
        opex_chf_per_year=opex_chf_per_year,
        wacc=wacc,
        project_lifetime_years=project_lifetime_years,
        degradation_pct_per_year=degradation_pct_per_year,
    )

    npv = -capex + float(proj["discounted_net_cashflow_chf"].sum()) if not proj.empty else -capex

    cf_year_1 = float(proj.iloc[0]["net_cashflow_chf"]) if not proj.empty else 0.0
    payback_years = None if (cf_year_1 <= 0 or capex <= 0) else (capex / cf_year_1)

    return float(npv), payback_years, proj


def _pick_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_existing_col_by_patterns(df: pd.DataFrame, prefix: str, patterns: list[str]) -> str | None:
    if df is None or df.empty:
        return None

    cols = [str(c) for c in df.columns]
    pref_cols = [c for c in cols if c.startswith(prefix)]

    for pat in patterns:
        for c in pref_cols:
            if pat in c:
                return c
    return None


def _resolve_sdl_product_cols(df: pd.DataFrame, pref: str) -> dict:
    acc_candidates = [
        f"{pref}_accepted",
        f"{pref}_is_accepted",
        f"{pref}_award",
        f"{pref}_accepted_flag",
        f"{pref}_won",
    ]
    acc_col = _pick_existing_col(df, acc_candidates)
    if acc_col is None:
        acc_col = _pick_existing_col_by_patterns(df, pref, ["accepted", "award", "won"])

    bid_candidates = [
        f"{pref}_bid_chf_per_mw_h",
        f"{pref}_bid_price_chf_per_mw_h",
        f"{pref}_bid_price",
        f"{pref}_bid",
    ]
    bid_col = _pick_existing_col(df, bid_candidates)
    if bid_col is None:
        bid_col = _pick_existing_col_by_patterns(df, pref, ["bid_chf", "bid_price", "_bid"])

    clear_candidates = [
        f"{pref}_clear_cap_chf_per_mw_h",
        f"{pref}_clear_true_chf_per_mw_h",
        f"{pref}_clear_chf_per_mw_h",
        f"{pref}_clearing_chf_per_mw_h",
        f"{pref}_clear_price_chf_per_mw_h",
        f"{pref}_clearing_price_chf_per_mw_h",
        f"{pref}_clear",
        f"{pref}_clearing",
    ]
    clear_col = _pick_existing_col(df, clear_candidates)
    if clear_col is None:
        clear_col = _pick_existing_col_by_patterns(df, pref, ["clear_cap", "clear_true", "clearing", "_clear"])

    settlement_candidates = [
        f"{pref}_settlement_cap_price_chf_per_mw_h",
        f"{pref}_settlement_price_chf_per_mw_h",
        f"{pref}_settlement_chf_per_mw_h",
        f"{pref}_settlement",
    ]
    settlement_col = _pick_existing_col(df, settlement_candidates)
    if settlement_col is None:
        settlement_col = _pick_existing_col_by_patterns(df, pref, ["settlement_cap", "settlement_price", "settlement"])

    rev_candidates = [
        f"{pref}_rev_total_chf_h",
        f"{pref}_rev_chf_h",
        f"{pref}_revenue_chf_h",
    ]
    rev_col = _pick_existing_col(df, rev_candidates)
    if rev_col is None:
        rev_col = _pick_existing_col_by_patterns(df, pref, ["rev_total", "_rev_", "revenue"])

    return {
        "accepted": acc_col,
        "bid": bid_col,
        "clear": clear_col,
        "settlement": settlement_col,
        "rev": rev_col,
    }


def _filter_time(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    return df[(df["ts"] >= pd.Timestamp(start_ts)) & (df["ts"] <= pd.Timestamp(end_ts))].copy()


def _long(df: pd.DataFrame, cols, series_col: str, label_map=None) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=["ts", series_col, "value"])

    out = (
        df[["ts"] + keep]
        .melt(id_vars=["ts"], var_name=series_col, value_name="value")
        .dropna(subset=["value"])
    )

    if label_map:
        out[series_col] = out[series_col].map(lambda x: label_map.get(x, x))

    return out


def _chart(
    df_long: pd.DataFrame,
    title: str,
    y_title: str,
    series_col: str,
    height=240,
    mark: str = "line",
):
    if df_long.empty:
        return (
            alt.Chart(pd.DataFrame({"ts": [], "value": []}))
            .mark_line()
            .encode(
                x=alt.X("ts:T", title="Zeit"),
                y=alt.Y("value:Q", title=y_title),
            )
            .properties(title=title, height=height)
        )

    base = alt.Chart(df_long)
    chart = base.mark_line(interpolate="step-after") if mark == "step" else base.mark_line()

    return (
        chart.encode(
            x=alt.X("ts:T", title="Zeit"),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color(f"{series_col}:N", title="Linie", legend=alt.Legend(orient="right")),
            tooltip=[
                alt.Tooltip("ts:T", title="Zeit"),
                alt.Tooltip(f"{series_col}:N", title="Serie"),
                alt.Tooltip("value:Q", title="Wert", format=",.4f"),
            ],
        )
        .properties(title=title, height=height)
    )


def _prepare_merge_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "ts" not in df.columns:
        return pd.DataFrame(columns=["ts"] + list(cols))

    keep = ["ts"] + [c for c in cols if c in df.columns]
    out = df[keep].copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out


def _nan_series(index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def _first_existing_numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return _nan_series(df.index)


def _valid_text_value(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip()
    return s not in {"", "nan", "None", "<NA>"}


def _positive_charge_series(raw: pd.Series) -> pd.Series:
    s = pd.to_numeric(raw, errors="coerce")
    if s.dropna().empty:
        return s.astype("float64")
    if (s <= 0).all() and (s.abs().sum() > 0):
        return (-s).clip(lower=0.0)
    return s.clip(lower=0.0)


def _positive_discharge_series(raw: pd.Series) -> pd.Series:
    s = pd.to_numeric(raw, errors="coerce")
    if s.dropna().empty:
        return s.astype("float64")
    if (s <= 0).all() and (s.abs().sum() > 0):
        return (-s).clip(lower=0.0)
    return s.abs().clip(lower=0.0)


def _binary_series(raw: pd.Series, default=0.0) -> pd.Series:
    s = pd.to_numeric(raw, errors="coerce")
    if s.dropna().empty:
        return pd.Series(default, index=raw.index if hasattr(raw, "index") else None, dtype="float64")
    return (s.fillna(default) > 0).astype(float)


def _get_product_offer_mw_from_cfg(cfg: dict, pref: str) -> float | None:
    mapping = {
        "prl_sym": ["p_offer_prl_mw", "sdl_p_offer_prl_mw"],
        "srl_up": ["p_offer_srl_up_mw", "sdl_p_offer_srl_up_mw"],
        "srl_down": ["p_offer_srl_down_mw", "sdl_p_offer_srl_down_mw"],
    }
    keys = mapping.get(pref, [])
    return _get_float(cfg, keys, default=None)


def _resolve_sdl_offer_series(sdl_df: pd.DataFrame, pref: str, cfg: dict | None = None) -> pd.Series:
    if sdl_df is None or sdl_df.empty:
        return pd.Series(dtype="float64")

    offer_candidates = {
        "prl_sym": [
            "sdl_p_offer_prl_mw",
            "p_offer_prl_mw",
            "prl_sym_p_offer_mw",
            "prl_sym_offer_mw",
            "prl_offer_mw",
            "prl_sym_effective_offer_mw",
            "prl_sym_nominal_offer_mw",
        ],
        "srl_up": [
            "sdl_p_offer_srl_up_mw",
            "p_offer_srl_up_mw",
            "srl_up_p_offer_mw",
            "srl_up_offer_mw",
            "srl_up_effective_offer_mw",
            "srl_up_nominal_offer_mw",
        ],
        "srl_down": [
            "sdl_p_offer_srl_down_mw",
            "p_offer_srl_down_mw",
            "srl_down_p_offer_mw",
            "srl_down_offer_mw",
            "srl_down_effective_offer_mw",
            "srl_down_nominal_offer_mw",
        ],
    }

    for c in offer_candidates.get(pref, []):
        if c in sdl_df.columns:
            return pd.to_numeric(sdl_df[c], errors="coerce")

    fallback = None
    if isinstance(cfg, dict):
        fallback = _get_product_offer_mw_from_cfg(cfg, pref)

    if fallback is None:
        return pd.Series(np.nan, index=sdl_df.index, dtype="float64")

    return pd.Series(float(fallback), index=sdl_df.index, dtype="float64")


def _compute_sdl_revenue_series(
    sdl_df: pd.DataFrame,
    pref: str,
    cols_resolved: dict,
    cfg: dict | None = None,
) -> pd.Series:
    """
    Prioritaet:
    1. Bereits vorhandene Revenue-Spalte aus dem Optimizer verwenden
    2. Sonst produktspezifisch rekonstruieren:
       - PRL: pay-as-cleared
       - SRL UP / SRL DOWN: pay-as-bid
    """
    if sdl_df is None or sdl_df.empty:
        return pd.Series(dtype="float64")

    acc_col = cols_resolved.get("accepted")
    bid_col = cols_resolved.get("bid")
    clear_col = cols_resolved.get("clear")
    settlement_col = cols_resolved.get("settlement")
    rev_col = cols_resolved.get("rev")

    if rev_col is not None and rev_col in sdl_df.columns:
        return pd.to_numeric(sdl_df[rev_col], errors="coerce").fillna(0.0)

    offer_mw = _resolve_sdl_offer_series(sdl_df, pref, cfg=cfg)

    bid_series = (
        pd.to_numeric(sdl_df[bid_col], errors="coerce")
        if bid_col is not None and bid_col in sdl_df.columns
        else None
    )
    clear_series = (
        pd.to_numeric(sdl_df[clear_col], errors="coerce")
        if clear_col is not None and clear_col in sdl_df.columns
        else None
    )
    settlement_series = (
        pd.to_numeric(sdl_df[settlement_col], errors="coerce")
        if settlement_col is not None and settlement_col in sdl_df.columns
        else None
    )

    if acc_col is not None and acc_col in sdl_df.columns:
        accepted = _binary_series(sdl_df[acc_col], default=0.0)
    else:
        accepted = pd.Series(1.0, index=sdl_df.index, dtype="float64")

    if settlement_series is not None:
        price_per_mw_h = settlement_series
    else:
        if pref == "prl_sym":
            price_per_mw_h = clear_series if clear_series is not None else bid_series
        elif pref in {"srl_up", "srl_down"}:
            price_per_mw_h = bid_series if bid_series is not None else clear_series
        else:
            price_per_mw_h = clear_series if clear_series is not None else bid_series

    if price_per_mw_h is not None and not offer_mw.dropna().empty:
        rev_calc = accepted.fillna(0.0) * offer_mw.fillna(0.0) * price_per_mw_h.fillna(0.0)
        return rev_calc.astype("float64")

    return pd.Series(np.nan, index=sdl_df.index, dtype="float64")


def _build_price_soc_dispatch_view(
    ts_df: pd.DataFrame | None,
    mu_df: pd.DataFrame | None,
    e_nom_kwh: float | None,
) -> pd.DataFrame:
    sources = []
    for df in [ts_df, mu_df]:
        if isinstance(df, pd.DataFrame) and not df.empty and "ts" in df.columns:
            sources.append(pd.to_datetime(df["ts"], errors="coerce"))

    if not sources:
        return pd.DataFrame(columns=["ts"])

    base_ts = (
        pd.concat(sources, ignore_index=True)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    out = pd.DataFrame({"ts": base_ts})

    if isinstance(ts_df, pd.DataFrame) and not ts_df.empty:
        t = _prepare_merge_frame(
            ts_df,
            [
                "price_da",
                "price_fc",
                "price_da_fc",
                "soc_pct",
                "soc_kwh",
                "p_charge_kw",
                "p_discharge_kw",
                "p_bess",
                "price_id",
                "price_id_fc",
                "spread_settle",
                "spread_fc",
                "rev_da_chf",
                "rev_id_inc_chf",
                "p_id_delta_kw",
            ],
        )
        out = out.merge(t, on="ts", how="left")

    if isinstance(mu_df, pd.DataFrame) and not mu_df.empty:
        mu = _prepare_merge_frame(
            mu_df,
            [
                "soc_pct_multiuse",
                "soc_kwh_multiuse",
                "p_charge_multiuse_kw",
                "p_discharge_multiuse_kw",
                "p_bess_multiuse_kw",
                "market_state",
                "market_state_detail",
                "sdl_product_label",
                "rev_multiuse_chf_h",
                "p_da_multiuse_kw",
                "p_id_multiuse_kw",
                "rev_da_multiuse_chf_h",
                "rev_id_multiuse_chf_h",
                "da_active",
                "id_active",
                "active_market",
                "active_energy_market",
                "da_id_market_state",
                "da_id_market_state_detail",
            ],
        )
        out = out.merge(mu, on="ts", how="left")

    out["da_fc_display"] = _first_existing_numeric(out, ["price_fc", "price_da_fc"])
    out["price_da_display"] = _first_existing_numeric(out, ["price_da"])
    out["price_id_display"] = _first_existing_numeric(out, ["price_id"])
    out["price_id_fc_display"] = _first_existing_numeric(out, ["price_id_fc"])

    if "spread_settle" in out.columns:
        out["spread_settle_display"] = pd.to_numeric(out["spread_settle"], errors="coerce")
    else:
        out["spread_settle_display"] = out["price_id_display"] - out["price_da_display"]

    if "spread_fc" in out.columns:
        out["spread_fc_display"] = pd.to_numeric(out["spread_fc"], errors="coerce")
    else:
        out["spread_fc_display"] = out["price_id_fc_display"] - out["da_fc_display"]

    soc_pct_mu = _first_existing_numeric(out, ["soc_pct_multiuse"])
    soc_kwh_mu = _first_existing_numeric(out, ["soc_kwh_multiuse"])
    soc_pct_base = _first_existing_numeric(out, ["soc_pct"])
    soc_kwh_base = _first_existing_numeric(out, ["soc_kwh"])

    out["soc_pct_display"] = soc_pct_mu

    if e_nom_kwh and e_nom_kwh > 0:
        mask = out["soc_pct_display"].isna() & soc_kwh_mu.notna()
        out.loc[mask, "soc_pct_display"] = (soc_kwh_mu[mask] / float(e_nom_kwh)) * 100.0

    mask = out["soc_pct_display"].isna() & soc_pct_base.notna()
    out.loc[mask, "soc_pct_display"] = soc_pct_base[mask]

    if e_nom_kwh and e_nom_kwh > 0:
        mask = out["soc_pct_display"].isna() & soc_kwh_base.notna()
        out.loc[mask, "soc_pct_display"] = (soc_kwh_base[mask] / float(e_nom_kwh)) * 100.0

    pnet_pref = _first_existing_numeric(out, ["p_bess_multiuse_kw", "p_bess"])
    p_charge_pref = _first_existing_numeric(out, ["p_charge_multiuse_kw", "p_charge_kw"])
    p_discharge_pref = _first_existing_numeric(out, ["p_discharge_multiuse_kw", "p_discharge_kw"])

    charge_series = _positive_charge_series(p_charge_pref)
    discharge_series = _positive_discharge_series(p_discharge_pref)

    if pnet_pref.notna().any():
        charge_from_net = (-pnet_pref).clip(lower=0.0)
        discharge_from_net = pnet_pref.clip(lower=0.0)
        charge_series = charge_series.where(charge_series.notna(), charge_from_net)
        discharge_series = discharge_series.where(discharge_series.notna(), discharge_from_net)

    out["p_charge_display_kw"] = charge_series
    out["p_discharge_display_kw"] = discharge_series

    return out


def _build_multiuse_market_state_detail(mu_df: pd.DataFrame, sdl_df: pd.DataFrame | None = None) -> pd.DataFrame:
    mu = _prepare_merge_frame(mu_df, ["market_state", "market_state_detail", "sdl_product_label"])
    if mu.empty:
        return pd.DataFrame(columns=["ts", "market_state_detail_display"])

    out = mu.copy()

    def _normalize_generic_state(state_raw: str) -> str:
        state = str(state_raw or "").strip().upper()
        if state in {"DA_ID", "DA_ID_ONLY"}:
            return "DA_ID"
        if state == "SDL_ONLY":
            return "SDL_ONLY"
        if state == "SDL_PLUS_DA_ID":
            return "SDL_PLUS_DA_ID"
        if state == "SDL":
            return "SDL_ONLY"
        return state_raw

    def _clean_product_text(txt: str) -> str:
        s = str(txt or "").strip()
        if not s:
            return ""
        s_up = s.upper()
        if s_up.startswith("SDL:"):
            s = s.split(":", 1)[1].strip()
        if s_up in {"SDL", "SDL_ONLY", "SDL_PLUS_DA_ID", "DA_ID", "DA_ID_ONLY", "NONE"}:
            return ""
        return s

    def _row_label(row) -> str:
        state = _normalize_generic_state(row.get("market_state", ""))

        detail = str(row.get("market_state_detail")).strip() if _valid_text_value(row.get("market_state_detail")) else ""
        product = str(row.get("sdl_product_label")).strip() if _valid_text_value(row.get("sdl_product_label")) else ""

        generic_values = {
            "",
            "SDL",
            "SDL_ONLY",
            "SDL_PLUS_DA_ID",
            "DA_ID",
            "DA_ID_ONLY",
            "NONE",
        }

        chosen = detail if detail.upper() not in generic_values else ""
        if not chosen and product.upper() not in generic_values:
            chosen = product

        chosen = _clean_product_text(chosen)

        if state == "DA_ID":
            return "DA_ID"

        if state == "SDL_ONLY":
            return chosen if chosen else "SDL"

        if state == "SDL_PLUS_DA_ID":
            if chosen:
                return chosen if "DA/ID" in chosen.upper() else f"{chosen} + DA/ID"
            return "SDL + DA/ID"

        if chosen:
            return chosen

        if sdl_df is not None and isinstance(sdl_df, pd.DataFrame) and not sdl_df.empty:
            sdl_cols = {}
            for pref in ["prl_sym", "srl_up", "srl_down"]:
                res = _resolve_sdl_product_cols(sdl_df, pref)
                if res["accepted"] is not None:
                    sdl_cols[pref] = res["accepted"]

            if sdl_cols:
                sdl = _prepare_merge_frame(sdl_df, list(sdl_cols.values()))
                rename_acc = {col: f"acc_{pref}" for pref, col in sdl_cols.items()}
                sdl = sdl.rename(columns=rename_acc)
                sdl_row = sdl[sdl["ts"] == row["ts"]]
                if not sdl_row.empty:
                    sr = sdl_row.iloc[0]
                    active = []
                    if pd.notna(sr.get("acc_prl_sym")) and float(sr.get("acc_prl_sym")) > 0:
                        active.append("PRL (sym)")
                    if pd.notna(sr.get("acc_srl_up")) and float(sr.get("acc_srl_up")) > 0:
                        active.append("SRL UP")
                    if pd.notna(sr.get("acc_srl_down")) and float(sr.get("acc_srl_down")) > 0:
                        active.append("SRL DOWN")

                    if active:
                        base = " + ".join(active)
                        if state == "SDL_PLUS_DA_ID":
                            return f"{base} + DA/ID"
                        return base

        if _valid_text_value(row.get("market_state")):
            return str(row.get("market_state"))

        return "—"

    out["market_state_detail_display"] = out.apply(_row_label, axis=1)
    return out[["ts", "market_state_detail_display"]]


def _build_da_id_market_state_detail(
    ts_df: pd.DataFrame | None,
    mu_df: pd.DataFrame | None,
) -> pd.DataFrame:
    sources = []
    for df in [ts_df, mu_df]:
        if isinstance(df, pd.DataFrame) and not df.empty and "ts" in df.columns:
            sources.append(pd.to_datetime(df["ts"], errors="coerce"))

    if not sources:
        return pd.DataFrame(columns=["ts", "da_id_market_state_display"])

    base_ts = (
        pd.concat(sources, ignore_index=True)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    out = pd.DataFrame({"ts": base_ts})

    if isinstance(mu_df, pd.DataFrame) and not mu_df.empty:
        mu = _prepare_merge_frame(
            mu_df,
            [
                "p_da_multiuse_kw",
                "p_id_multiuse_kw",
                "rev_da_multiuse_chf_h",
                "rev_id_multiuse_chf_h",
                "da_active",
                "id_active",
            ],
        )
        out = out.merge(mu, on="ts", how="left")

    def _infer_row(row) -> str:
        tol_p = 1e-6
        tol_rev = 1e-6

        p_da = float(row.get("p_da_multiuse_kw", 0.0)) if pd.notna(row.get("p_da_multiuse_kw")) else 0.0
        p_id = float(row.get("p_id_multiuse_kw", 0.0)) if pd.notna(row.get("p_id_multiuse_kw")) else 0.0
        rev_da = float(row.get("rev_da_multiuse_chf_h", 0.0)) if pd.notna(row.get("rev_da_multiuse_chf_h")) else 0.0
        rev_id = float(row.get("rev_id_multiuse_chf_h", 0.0)) if pd.notna(row.get("rev_id_multiuse_chf_h")) else 0.0

        # Primaer: reale Multiuse-Leistungen
        da_active = abs(p_da) > tol_p
        id_active = abs(p_id) > tol_p

        # Sekundaer: falls Leistungen nicht geschrieben werden, auf realisierte Erloese gehen
        if not da_active and abs(rev_da) > tol_rev:
            da_active = True
        if not id_active and abs(rev_id) > tol_rev:
            id_active = True

        # Nur als letzte Rueckfalloption auf explizite Flags gehen
        if not da_active and pd.notna(row.get("da_active")):
            try:
                da_active = float(row.get("da_active")) > 0
            except Exception:
                pass

        if not id_active and pd.notna(row.get("id_active")):
            try:
                id_active = float(row.get("id_active")) > 0
            except Exception:
                pass

        if da_active and id_active:
            return "DA+ID"
        if da_active:
            return "DA"
        if id_active:
            return "ID"
        return "—"

    out["da_id_market_state_display"] = out.apply(_infer_row, axis=1)
    return out[["ts", "da_id_market_state_display"]]

    def _infer_row(row) -> str:
        for c in direct_text_candidates:
            if c in row.index and _valid_text_value(row.get(c)):
                txt = str(row.get(c)).strip().upper()
                if txt in {"DA", "DAY_AHEAD"}:
                    return "DA"
                if txt in {"ID", "INTRADAY"}:
                    return "ID"
                if txt in {"DA+ID", "DA_ID", "DAY_AHEAD+INTRADAY"}:
                    return "DA+ID"
                if "INTRADAY" in txt and "DAY" not in txt and txt != "DA/ID":
                    return "ID"
                if ("DAY-AHEAD" in txt or "DAY_AHEAD" in txt or txt == "DA") and "ID" not in txt:
                    return "DA"
                if "DA/ID" in txt or "DA_ID" in txt:
                    return "DA/ID"

        da_active = False
        id_active = False

        if "da_active" in row.index and pd.notna(row.get("da_active")):
            da_active = bool(float(pd.to_numeric(pd.Series([row.get("da_active")]), errors="coerce").iloc[0]) > 0)

        if "id_active" in row.index and pd.notna(row.get("id_active")):
            id_active = bool(float(pd.to_numeric(pd.Series([row.get("id_active")]), errors="coerce").iloc[0]) > 0)

        da_numeric_candidates = [
            "rev_da_multiuse_chf_h",
            "p_da_multiuse_kw",
            "rev_da_chf",
        ]
        id_numeric_candidates = [
            "rev_id_multiuse_chf_h",
            "p_id_multiuse_kw",
            "rev_id_inc_chf",
            "p_id_delta_kw",
        ]

        tol = 1e-9
        for c in da_numeric_candidates:
            if c in row.index and pd.notna(row.get(c)) and abs(float(row.get(c))) > tol:
                da_active = True
                break

        for c in id_numeric_candidates:
            if c in row.index and pd.notna(row.get(c)) and abs(float(row.get(c))) > tol:
                id_active = True
                break

        state = str(row.get("market_state", "")).strip().upper() if "market_state" in row.index else ""

        if da_active and id_active:
            return "DA+ID"
        if id_active:
            return "ID"
        if da_active:
            return "DA"

        if state in {"DA_ID", "DA_ID_ONLY", "SDL_PLUS_DA_ID"}:
            return "DA/ID"

        if state == "SDL_ONLY":
            return "—"

        return "—"

    out["da_id_market_state_display"] = out.apply(_infer_row, axis=1)
    return out[["ts", "da_id_market_state_display"]]


def _combine_market_state_labels(
    multiuse_label,
    da_id_label,
) -> str:
    def _clean(v) -> str:
        if pd.isna(v):
            return ""
        s = str(v).strip()
        if s in {"", "—", "nan", "None", "<NA>"}:
            return ""
        return s

    def _normalize_energy_label(v: str) -> str:
        s = _clean(v).upper()

        if s in {"DA_ID", "DA+ID", "DA/ID", "DAY_AHEAD+INTRADAY"}:
            return "DA+ID"
        if s in {"DA", "DAY_AHEAD"}:
            return "DA"
        if s in {"ID", "INTRADAY"}:
            return "ID"
        return ""

    mu = _clean(multiuse_label)
    daid = _clean(da_id_label)

    if mu.upper().startswith("SDL:"):
        mu = mu.split(":", 1)[1].strip()

    mu_upper = mu.upper()

    # Fall 1: multiuse_label ist selbst schon ein reiner Energiezustand
    mu_energy = _normalize_energy_label(mu)
    daid_energy = _normalize_energy_label(daid)

    if mu_energy and daid_energy:
        if mu_energy == daid_energy:
            return mu_energy
        if {mu_energy, daid_energy} == {"DA", "ID"}:
            return "DA+ID"

    if mu_energy:
        return mu_energy if not daid_energy else daid_energy if mu_energy == daid_energy else "DA+ID"

    # Fall 2: SDL-Label bereinigen
    mu = (
        mu.replace(" + DA/ID", "")
          .replace(" + DA+ID", "")
          .replace(" + DA_ID", "")
          .replace(" + DA", "")
          .replace(" + ID", "")
          .strip()
    )

    daid = daid_energy

    if mu and daid:
        return f"{mu} + {daid}"
    if mu:
        return mu
    if daid:
        return daid
    return "—"


def _build_dashboard_export_timeseries(
    ts_df: pd.DataFrame | None,
    sdl_df: pd.DataFrame | None,
    mu_df: pd.DataFrame | None,
    e_nom_kwh: float | None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    final_cols = [
        "Zeit [datetime]",
        "Markt pro Stunde / Market-State [-]",
        "Multiuse: Erloes pro Stunde [CHF/h]",
        "Day-Ahead Preis: Settlement [CHF/MWh]",
        "Day-Ahead Preis: Forecast [CHF/MWh]",
        "State of Charge [%]",
        "Batterie-Dispatch: Laden [kW]",
        "Batterie-Dispatch: Entladen [kW]",
        "Intraday Preis: Settlement [CHF/MWh]",
        "Intraday Preis: Forecast [CHF/MWh]",
        "Spread ID-DA: Settlement [CHF/MWh]",
        "Spread ID-DA: Forecast [CHF/MWh]",
        "SDL: PRL (sym): Angebotsleistung [MW]",
        "SDL: PRL (sym): Bid [CHF/MW/h]",
        "SDL: PRL (sym): Clearing [CHF/MW/h]",
        "SDL: PRL (sym): Settlement [CHF/MW/h]",
        "SDL: PRL (sym): Zuschlag [0/1]",
        "SDL: PRL (sym): Erloes pro Stunde [CHF/h]",
        "SDL: SRL UP: Angebotsleistung [MW]",
        "SDL: SRL UP: Bid [CHF/MW/h]",
        "SDL: SRL UP: Clearing [CHF/MW/h]",
        "SDL: SRL UP: Settlement [CHF/MW/h]",
        "SDL: SRL UP: Zuschlag [0/1]",
        "SDL: SRL UP: Erloes pro Stunde [CHF/h]",
        "SDL: SRL DOWN: Angebotsleistung [MW]",
        "SDL: SRL DOWN: Bid [CHF/MW/h]",
        "SDL: SRL DOWN: Clearing [CHF/MW/h]",
        "SDL: SRL DOWN: Settlement [CHF/MW/h]",
        "SDL: SRL DOWN: Zuschlag [0/1]",
        "SDL: SRL DOWN: Erloes pro Stunde [CHF/h]",
    ]

    view = _build_price_soc_dispatch_view(ts_df=ts_df, mu_df=mu_df, e_nom_kwh=e_nom_kwh)
    if view is None or view.empty:
        return pd.DataFrame(columns=final_cols)

    out = pd.DataFrame({"ts": view["ts"]})

    mu_realized = None

    if isinstance(mu_df, pd.DataFrame) and not mu_df.empty:
        mu = _prepare_merge_frame(mu_df, ["rev_multiuse_chf_h"])
        mu_detail = _build_multiuse_market_state_detail(mu_df, sdl_df)
        mu = mu.merge(mu_detail, on="ts", how="left")

        if "rev_multiuse_chf_h" in mu.columns:
            mu["rev_multiuse_chf_h"] = pd.to_numeric(mu["rev_multiuse_chf_h"], errors="coerce")

        mu_export = pd.DataFrame({"ts": mu["ts"]})
        mu_export["multiuse_market_state_display"] = mu.get("market_state_detail_display", pd.NA)
        mu_export["Multiuse: Erloes pro Stunde [CHF/h]"] = (
            mu["rev_multiuse_chf_h"] if "rev_multiuse_chf_h" in mu.columns else pd.NA
        )

        out = out.merge(mu_export, on="ts", how="left")

        mu_realized = mu_export[["ts", "multiuse_market_state_display"]].copy()

        def _normalize_realized_product(v: str) -> str | None:
            if pd.isna(v):
                return None
            s = str(v).strip().upper()
            if not s or s in {"—", "NONE", "DA_ID", "DA+ID", "DA/ID", "SDL", "SDL + DA/ID"}:
                return None

            if "PRL" in s:
                return "prl_sym"
            if "SRL UP" in s:
                return "srl_up"
            if "SRL DOWN" in s:
                return "srl_down"
            return None

        mu_realized["realized_sdl_pref"] = mu_realized["multiuse_market_state_display"].map(_normalize_realized_product)

    da_id_detail = _build_da_id_market_state_detail(ts_df=ts_df, mu_df=mu_df)
    if not da_id_detail.empty:
        out = out.merge(da_id_detail, on="ts", how="left")

    out["Markt pro Stunde / Market-State [-]"] = out.apply(
        lambda row: _combine_market_state_labels(
            row.get("multiuse_market_state_display", pd.NA),
            row.get("da_id_market_state_display", pd.NA),
        ),
        axis=1,
    )

    t_export = pd.DataFrame({"ts": view["ts"]})
    t_export["Day-Ahead Preis: Settlement [CHF/MWh]"] = view.get("price_da_display", pd.NA)
    t_export["Day-Ahead Preis: Forecast [CHF/MWh]"] = view.get("da_fc_display", pd.NA)
    t_export["State of Charge [%]"] = view.get("soc_pct_display", pd.NA)
    t_export["Batterie-Dispatch: Laden [kW]"] = view.get("p_charge_display_kw", pd.NA)
    t_export["Batterie-Dispatch: Entladen [kW]"] = view.get("p_discharge_display_kw", pd.NA)
    t_export["Intraday Preis: Settlement [CHF/MWh]"] = view.get("price_id_display", pd.NA)
    t_export["Intraday Preis: Forecast [CHF/MWh]"] = view.get("price_id_fc_display", pd.NA)
    t_export["Spread ID-DA: Settlement [CHF/MWh]"] = view.get("spread_settle_display", pd.NA)
    t_export["Spread ID-DA: Forecast [CHF/MWh]"] = view.get("spread_fc_display", pd.NA)

    out = out.merge(t_export, on="ts", how="left")

    if isinstance(sdl_df, pd.DataFrame) and not sdl_df.empty:
        sdl_label_map = [
            ("prl_sym", "SDL: PRL (sym)"),
            ("srl_up", "SDL: SRL UP"),
            ("srl_down", "SDL: SRL DOWN"),
        ]

        for pref, label in sdl_label_map:
            cols_resolved = _resolve_sdl_product_cols(sdl_df, pref)
            offer_series = _resolve_sdl_offer_series(sdl_df, pref, cfg=cfg)
            rev_series = _compute_sdl_revenue_series(sdl_df, pref, cols_resolved, cfg=cfg)

            s_export = pd.DataFrame({"ts": sdl_df["ts"]})
            s_export["ts"] = pd.to_datetime(s_export["ts"], errors="coerce")
            s_export = (
                s_export
                .dropna(subset=["ts"])
                .sort_values("ts")
                .drop_duplicates(subset=["ts"], keep="last")
                .reset_index(drop=True)
            )

            s_export[f"{label}: Angebotsleistung [MW]"] = pd.to_numeric(
                offer_series.reindex(s_export.index), errors="coerce"
            )

            if cols_resolved["bid"] is not None and cols_resolved["bid"] in sdl_df.columns:
                s_export[f"{label}: Bid [CHF/MW/h]"] = pd.to_numeric(
                    sdl_df[cols_resolved["bid"]].reindex(s_export.index), errors="coerce"
                )

            if cols_resolved["clear"] is not None and cols_resolved["clear"] in sdl_df.columns:
                s_export[f"{label}: Clearing [CHF/MW/h]"] = pd.to_numeric(
                    sdl_df[cols_resolved["clear"]].reindex(s_export.index), errors="coerce"
                )

            if cols_resolved["settlement"] is not None and cols_resolved["settlement"] in sdl_df.columns:
                s_export[f"{label}: Settlement [CHF/MW/h]"] = pd.to_numeric(
                    sdl_df[cols_resolved["settlement"]].reindex(s_export.index), errors="coerce"
                )

            if cols_resolved["accepted"] is not None and cols_resolved["accepted"] in sdl_df.columns:
                acc_raw = pd.to_numeric(
                    sdl_df[cols_resolved["accepted"]].reindex(s_export.index),
                    errors="coerce",
                ).fillna(0.0)
            else:
                acc_raw = pd.Series(0.0, index=s_export.index, dtype="float64")

            rev_raw = pd.to_numeric(rev_series.reindex(s_export.index), errors="coerce").fillna(0.0)

            if mu_realized is not None and not mu_realized.empty:
                s_export = s_export.merge(
                    mu_realized[["ts", "realized_sdl_pref"]],
                    on="ts",
                    how="left",
                )
                realized_mask = (s_export["realized_sdl_pref"] == pref).fillna(False)
                acc_final = np.where(realized_mask, (acc_raw > 0).astype(float), 0.0)
                rev_final = np.where(realized_mask, rev_raw, 0.0)
                s_export = s_export.drop(columns=["realized_sdl_pref"])
            else:
                acc_final = (acc_raw > 0).astype(float)
                rev_final = rev_raw

            try:
                s_export[f"{label}: Zuschlag [0/1]"] = pd.Series(acc_final, index=s_export.index).round().astype("Int64")
            except Exception:
                s_export[f"{label}: Zuschlag [0/1]"] = pd.Series(acc_final, index=s_export.index)

            s_export[f"{label}: Erloes pro Stunde [CHF/h]"] = pd.Series(
                rev_final,
                index=s_export.index,
                dtype="float64",
            )

            out = out.merge(s_export, on="ts", how="left")

    out = out.rename(columns={"ts": "Zeit [datetime]"})

    for col in final_cols:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[final_cols].copy()
    return out


def _csv_bytes_for_dashboard_export(df: pd.DataFrame) -> bytes:
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    return df.to_csv(index=False).encode("utf-8")


def _fmt_pdf_cell(v):
    if pd.isna(v):
        return ""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, (np.integer, int)):
        return f"{int(v)}"
    if isinstance(v, (np.floating, float)):
        return f"{float(v):,.2f}"
    return str(v)


def _make_dashboard_pdf_bytes(
    *,
    scenario_name: str,
    market_mode: str,
    start_ts,
    end_ts,
    annual_rev_chf: float,
    npv_chf: float,
    payback_str: str,
    cycles_efc: float | None,
    degradation_rate_per_year: float,
    revenue_split_df: pd.DataFrame | None,
    dashboard_export_df: pd.DataFrame | None,
) -> bytes | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
            LongTable,
        )
    except Exception:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h_style = styles["Heading2"]
    body_style = styles["BodyText"]
    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
    )

    def _tbl_style():
        return TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )

    story = []
    story.append(Paragraph(f"Dashboard Report - {scenario_name}", title_style))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            f"Marktmodus: <b>{market_mode}</b> | Zeitraum: <b>{pd.Timestamp(start_ts).strftime('%Y-%m-%d %H:%M')}</b> "
            f"bis <b>{pd.Timestamp(end_ts).strftime('%Y-%m-%d %H:%M')}</b>",
            body_style,
        )
    )
    story.append(
        Paragraph(
            f"Degradation fuer NPV: <b>{degradation_rate_per_year * 100:.2f} %/a</b>",
            body_style,
        )
    )
    story.append(Spacer(1, 5 * mm))

    kpi_rows = [
        ["KPI", "Wert"],
        ["Multiuse / Jahresertrag [CHF]", _fmt_pdf_cell(annual_rev_chf)],
        ["NPV [CHF]", _fmt_pdf_cell(npv_chf)],
        ["Payback [a]", payback_str],
        ["Zyklen / Jahr (EFC)", "—" if cycles_efc is None else _fmt_pdf_cell(cycles_efc)],
    ]
    kpi_table = Table(kpi_rows, colWidths=[80 * mm, 45 * mm])
    kpi_table.setStyle(_tbl_style())
    story.append(Paragraph("KPIs", h_style))
    story.append(kpi_table)
    story.append(Spacer(1, 5 * mm))

    if isinstance(revenue_split_df, pd.DataFrame) and not revenue_split_df.empty:
        rev_df = revenue_split_df.copy()
        rev_data = [["Markt", "Erloes [CHF]"]]
        for _, row in rev_df.iterrows():
            rev_data.append([_fmt_pdf_cell(row.get("Markt")), _fmt_pdf_cell(row.get("Erlös [CHF]"))])

        rev_table = Table(rev_data, colWidths=[90 * mm, 45 * mm])
        rev_table.setStyle(_tbl_style())
        story.append(Paragraph("Ertragsaufteilung", h_style))
        story.append(rev_table)
        story.append(Spacer(1, 5 * mm))

    if isinstance(dashboard_export_df, pd.DataFrame) and not dashboard_export_df.empty:
        if "Markt pro Stunde / Market-State [-]" in dashboard_export_df.columns:
            state_counts = (
                dashboard_export_df["Markt pro Stunde / Market-State [-]"]
                .fillna("—")
                .astype(str)
                .value_counts(dropna=False)
                .reset_index()
            )
            state_counts.columns = ["Market-State", "Anzahl Stunden"]

            state_data = [["Market-State", "Anzahl Stunden"]]
            for _, row in state_counts.iterrows():
                state_data.append([_fmt_pdf_cell(row["Market-State"]), _fmt_pdf_cell(row["Anzahl Stunden"])])

            state_table = Table(state_data, colWidths=[90 * mm, 45 * mm])
            state_table.setStyle(_tbl_style())
            story.append(Paragraph("Verteilung der Market-States", h_style))
            story.append(state_table)
            story.append(Spacer(1, 5 * mm))

        story.append(PageBreak())
        story.append(Paragraph("Dashboard Timeseries - Auszug", h_style))
        story.append(
            Paragraph(
                "Hinweis: Der PDF-Report enthaelt einen kompakten Auszug der Timeseries. "
                "Die vollstaendige Zeitreihe steht weiterhin als CSV-Export zur Verfuegung.",
                small_style,
            )
        )
        story.append(Spacer(1, 3 * mm))

        preview = dashboard_export_df.copy()
        preview = preview.head(120).copy()

        if "Zeit [datetime]" in preview.columns:
            preview["Zeit [datetime]"] = pd.to_datetime(preview["Zeit [datetime]"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

        core_cols = [
            "Zeit [datetime]",
            "Markt pro Stunde / Market-State [-]",
            "Multiuse: Erloes pro Stunde [CHF/h]",
            "Day-Ahead Preis: Settlement [CHF/MWh]",
            "State of Charge [%]",
            "Batterie-Dispatch: Laden [kW]",
            "Batterie-Dispatch: Entladen [kW]",
            "Intraday Preis: Settlement [CHF/MWh]",
        ]
        core_cols = [c for c in core_cols if c in preview.columns]

        if core_cols:
            core_data = [core_cols]
            for _, row in preview[core_cols].iterrows():
                core_data.append([_fmt_pdf_cell(row[c]) for c in core_cols])

            core_table = LongTable(
                core_data,
                repeatRows=1,
                colWidths=[32 * mm, 42 * mm, 22 * mm, 24 * mm, 18 * mm, 20 * mm, 20 * mm, 24 * mm][: len(core_cols)],
            )
            core_table.setStyle(_tbl_style())
            story.append(core_table)
            story.append(Spacer(1, 5 * mm))

        sdl_cols = [
            "Zeit [datetime]",
            "SDL: PRL (sym): Angebotsleistung [MW]",
            "SDL: PRL (sym): Zuschlag [0/1]",
            "SDL: PRL (sym): Erloes pro Stunde [CHF/h]",
            "SDL: SRL UP: Angebotsleistung [MW]",
            "SDL: SRL UP: Zuschlag [0/1]",
            "SDL: SRL UP: Erloes pro Stunde [CHF/h]",
            "SDL: SRL DOWN: Angebotsleistung [MW]",
            "SDL: SRL DOWN: Zuschlag [0/1]",
            "SDL: SRL DOWN: Erloes pro Stunde [CHF/h]",
        ]
        sdl_cols = [c for c in sdl_cols if c in preview.columns]

        if sdl_cols:
            sdl_data = [sdl_cols]
            for _, row in preview[sdl_cols].iterrows():
                sdl_data.append([_fmt_pdf_cell(row[c]) for c in sdl_cols])

            sdl_table = LongTable(
                sdl_data,
                repeatRows=1,
                colWidths=[30 * mm, 18 * mm, 18 * mm, 24 * mm, 18 * mm, 18 * mm, 24 * mm, 18 * mm, 18 * mm, 24 * mm][: len(sdl_cols)],
            )
            sdl_table.setStyle(_tbl_style())
            story.append(Paragraph("SDL-Detailauszug", h_style))
            story.append(sdl_table)

    doc.build(story)
    return buffer.getvalue()


# ============================================================
# Config / Szenario
# ============================================================
sname = st.session_state.get("scenario_name", "BaseCase_2025")

if "scenario_config" not in st.session_state or not isinstance(st.session_state.get("scenario_config"), dict):
    st.session_state["scenario_config"] = load_config(sname) or {}

cfg = st.session_state["scenario_config"]
market_mode = str(cfg.get("market_mode", "CUSTOM"))

# ============================================================
# Load Artifacts
# ============================================================
ts_saved = load_parquet(sname, "results_timeseries")
dispatch_saved = load_parquet(sname, "dispatch")
sdl_ts = load_parquet(sname, "sdl_timeseries")
multiuse_ts = load_parquet(sname, "multiuse_timeseries")

ts_saved = _ensure_ts_key(ts_saved) if isinstance(ts_saved, pd.DataFrame) else None
dispatch_saved = _ensure_ts_key(dispatch_saved) if isinstance(dispatch_saved, pd.DataFrame) else None
sdl_ts = _ensure_ts_key(sdl_ts) if isinstance(sdl_ts, pd.DataFrame) else None
multiuse_ts = _ensure_ts_key(multiuse_ts) if isinstance(multiuse_ts, pd.DataFrame) else None

has_multiuse = (
    isinstance(multiuse_ts, pd.DataFrame)
    and (not multiuse_ts.empty)
    and ("market_state" in multiuse_ts.columns)
)

if (
    (ts_saved is None or ts_saved.empty)
    and (dispatch_saved is None or dispatch_saved.empty)
    and (sdl_ts is None or sdl_ts.empty)
    and (not has_multiuse)
):
    st.warning("Keine Ergebnisse vorhanden. Bitte zuerst Prognose & Dispatch ausfuehren.")
    st.stop()

# ============================================================
# Build 'results' object (Legacy)
# ============================================================
if "results" not in st.session_state:
    if isinstance(ts_saved, pd.DataFrame) and not ts_saved.empty:
        st.session_state["results"] = {
            "kpis": {},
            "revenue_breakdown": [],
            "top_days": [],
            "worst_days": [],
            "timeseries": ts_saved,
        }
    else:
        disp = dispatch_saved
        if disp is None or not isinstance(disp, pd.DataFrame) or disp.empty:
            if isinstance(sdl_ts, pd.DataFrame) and not sdl_ts.empty:
                st.session_state["results"] = {
                    "kpis": {},
                    "revenue_breakdown": [],
                    "top_days": [],
                    "worst_days": [],
                    "timeseries": sdl_ts,
                }
            else:
                st.warning("Keine Ergebnisse vorhanden. Bitte zuerst Prognose & Dispatch ausfuehren.")
                st.stop()
        else:
            e_nom = _get_float(cfg, ["e_nom_kwh", "E_nom_kwh", "e_nom"], default=None)
            looks_like_da = (
                ("p_charge_kw" in disp.columns and "p_discharge_kw" in disp.columns)
                or ("p_bess" in disp.columns)
            )

            if looks_like_da:
                res = compute_results_from_dispatch(
                    dispatch_df=disp,
                    runtime_s=0.0,
                    market_name="day_ahead",
                    e_nom_kwh=e_nom,
                )
                st.session_state["results"] = res
            else:
                st.session_state["results"] = {
                    "kpis": {},
                    "revenue_breakdown": [],
                    "top_days": [],
                    "worst_days": [],
                    "timeseries": disp,
                }

res = st.session_state["results"]
kpi = res.get("kpis", {}) if isinstance(res, dict) else {}
ts = res.get("timeseries") if isinstance(res, dict) else None

if ts is None or not isinstance(ts, pd.DataFrame) or ts.empty:
    ts = ts_saved if isinstance(ts_saved, pd.DataFrame) and not ts_saved.empty else None

if ts is None or not isinstance(ts, pd.DataFrame) or ts.empty:
    ts = multiuse_ts if has_multiuse else sdl_ts

ts = _ensure_ts_key(ts)

flags = _infer_market_from_timeseries(ts)
has_da = flags["has_da"]
has_id = flags["has_id"]
has_sdl = isinstance(sdl_ts, pd.DataFrame) and not sdl_ts.empty

if has_sdl and ("sdl_total_revenue_chf" not in kpi):
    kpi.update(_compute_sdl_kpis(sdl_ts))

# ============================================================
# Zeitspalte & Filter
# ============================================================
st.markdown("---")
with st.expander("Plot-Einstellungen", expanded=True):
    max_points_prices = st.slider("Max. Punkte (nur Preise)", 500, 8760, 2500, 250)
    show_spread_chart = st.toggle("Spread-Chart anzeigen (ID-DA)", value=True)

    tmin, tmax = ts["ts"].min(), ts["ts"].max()
    cA, cB = st.columns(2)
    with cA:
        start_ts = st.datetime_input("Start", value=tmin)
    with cB:
        end_ts = st.datetime_input("Ende", value=tmax)

    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts

ts_filt = _filter_time(ts)
sdl_filt = _filter_time(sdl_ts) if has_sdl else None
mu_filt = _filter_time(multiuse_ts) if has_multiuse else None

# ============================================================
# Optional: DA/ID Revenue Diagnostics
# ============================================================
rev_diag = None
try:
    master_df = load_parquet(sname, "master")
    dispatch_df = load_parquet(sname, "dispatch")
    if (
        isinstance(master_df, pd.DataFrame)
        and isinstance(dispatch_df, pd.DataFrame)
        and (not master_df.empty)
        and (not dispatch_df.empty)
    ):
        _, rev_diag = compute_revenues_da_id_incremental(master=master_df, dispatch=dispatch_df)
except Exception:
    rev_diag = None

# ============================================================
# KPIs
# ============================================================
st.markdown("---")
st.subheader("KPIs")

e_nom_kwh = _get_float(cfg, ["e_nom_kwh", "E_nom_kwh", "e_nom"], default=None)
cycles_source = _build_price_soc_dispatch_view(
    ts_df=ts_filt if isinstance(ts_filt, pd.DataFrame) else None,
    mu_df=mu_filt if isinstance(mu_filt, pd.DataFrame) else None,
    e_nom_kwh=e_nom_kwh,
)

cycles_input = pd.DataFrame({
    "p_charge_kw": pd.to_numeric(cycles_source.get("p_charge_display_kw", pd.Series(dtype="float64")), errors="coerce"),
    "p_discharge_kw": pd.to_numeric(cycles_source.get("p_discharge_display_kw", pd.Series(dtype="float64")), errors="coerce"),
})

cycles_efc = _efc_cycles_from_dispatch(cycles_input, e_nom_kwh=e_nom_kwh, dt_h=1.0)

total_project_cost_chf = _get_total_project_cost_chf(cfg)
opex_chf_per_year = _get_eco_float(cfg, "opex_chf_per_year", default=0.0)
wacc = _get_eco_float(cfg, "wacc", default=0.06)
project_lifetime_years = _get_eco_int(
    cfg,
    "project_lifetime_years",
    default=_get_eco_int(cfg, "asset_life_years", default=15),
)

degradation_pct_per_year_raw = _get_float(cfg, ["degradation_pct_per_year"], default=2.3)
degradation_rate_per_year = _normalize_pct_rate(degradation_pct_per_year_raw, default=0.023)

annual_rev_chf = 0.0
if has_multiuse and isinstance(multiuse_ts, pd.DataFrame) and "rev_multiuse_chf_h" in multiuse_ts.columns:
    annual_rev_chf = float(
        pd.to_numeric(multiuse_ts["rev_multiuse_chf_h"], errors="coerce").fillna(0.0).sum()
    )
elif rev_diag is not None:
    annual_rev_chf = float(rev_diag.sum_total_chf)
elif has_sdl and (not has_da):
    annual_rev_chf = float(kpi.get("sdl_total_revenue_chf", 0.0))
else:
    annual_rev_chf = (
        float(pd.to_numeric(ts_saved.get("revenue_chf", 0.0), errors="coerce").fillna(0.0).sum())
        if isinstance(ts_saved, pd.DataFrame)
        else 0.0
    )

npv_chf, payback_years, fin_proj = _compute_npv_and_payback(
    annual_revenue_chf=annual_rev_chf,
    total_project_cost_chf=total_project_cost_chf,
    opex_chf_per_year=opex_chf_per_year,
    wacc=wacc,
    project_lifetime_years=project_lifetime_years,
    degradation_pct_per_year=degradation_rate_per_year,
)

payback_str = "—" if payback_years is None else f"{payback_years:.2f}"

if has_multiuse:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sum Multiuse (TOTAL) [CHF]", f"{annual_rev_chf:.0f}")
    c2.metric("Sum SDL (TOTAL) [CHF]", f"{float(kpi.get('sdl_total_revenue_chf', 0.0)):.0f}")
    c3.metric("Sum DA+ID (additiv) [CHF]", f"{float(rev_diag.sum_total_chf):.0f}" if rev_diag is not None else "—")
    c4.metric("NPV [CHF]", f"{npv_chf:.0f}")
    c5.metric("Payback [a]", payback_str)
    c6.metric("Zyklen / Jahr (EFC)", f"{float(cycles_efc):.2f}" if cycles_efc is not None else "—")
else:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Jahresertrag [CHF]", f"{annual_rev_chf:.0f}")
    c2.metric("NPV [CHF]", f"{npv_chf:.0f}")
    c3.metric("Payback [a]", payback_str)
    c4.metric("SDL Jahresertrag [CHF]", f"{float(kpi.get('sdl_total_revenue_chf', 0.0)):.0f}")
    c5.metric("Zyklen / Jahr (EFC)", f"{float(cycles_efc):.2f}")
    c6.metric("Modus", market_mode)

st.caption(
    f"Szenario: {sname} | Marktmodus: **{market_mode}** | "
    f"Degradation fuer NPV: **{degradation_rate_per_year * 100:.2f} %/a**"
)

if isinstance(fin_proj, pd.DataFrame) and not fin_proj.empty:
    st.markdown("### Wirtschaftlichkeitsprojektion (mit Degradation)")

    fin_plot = fin_proj.copy()
    fin_plot_long = fin_plot[
        ["year", "revenue_chf", "net_cashflow_chf", "discounted_net_cashflow_chf"]
    ].melt(
        id_vars=["year"],
        var_name="Serie",
        value_name="Wert",
    )

    fin_plot_long["Serie"] = fin_plot_long["Serie"].map(
        {
            "revenue_chf": "Erloes nach Degradation",
            "net_cashflow_chf": "Netto-Cashflow",
            "discounted_net_cashflow_chf": "Diskontierter Netto-Cashflow",
        }
    )

    fin_chart = (
        alt.Chart(fin_plot_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Jahr"),
            y=alt.Y("Wert:Q", title="CHF"),
            color=alt.Color("Serie:N", title="Serie", legend=alt.Legend(orient="right")),
            tooltip=[
                alt.Tooltip("year:O", title="Jahr"),
                alt.Tooltip("Serie:N", title="Serie"),
                alt.Tooltip("Wert:Q", title="Wert", format=",.2f"),
            ],
        )
        .properties(height=280, title="Jaehrliche Erloes- und Cashflow-Entwicklung")
    )
    st.altair_chart(fin_chart, use_container_width=True)

    fin_table = fin_proj.copy()
    fin_table["degradation_pct_vs_y1"] = (1.0 - fin_table["degradation_factor"]) * 100.0

    fin_table = fin_table.rename(
        columns={
            "year": "Jahr",
            "degradation_factor": "Degradation-Faktor [-]",
            "degradation_pct_vs_y1": "Kapazitaets-/Erloesreduktion ggü. Jahr 1 [%]",
            "revenue_chf": "Erloes nach Degradation [CHF]",
            "opex_chf": "OPEX [CHF]",
            "net_cashflow_chf": "Netto-Cashflow [CHF]",
            "discounted_net_cashflow_chf": "Diskontierter Netto-Cashflow [CHF]",
            "cum_net_cashflow_chf": "Kumuliert netto [CHF]",
            "cum_discounted_cashflow_after_capex_chf": "Kumuliert diskontiert nach CAPEX [CHF]",
        }
    )

    st.dataframe(
        fin_table.style.format(
            {
                "Degradation-Faktor [-]": "{:.4f}",
                "Kapazitaets-/Erloesreduktion ggü. Jahr 1 [%]": "{:.2f}",
                "Erloes nach Degradation [CHF]": "{:,.2f}",
                "OPEX [CHF]": "{:,.2f}",
                "Netto-Cashflow [CHF]": "{:,.2f}",
                "Diskontierter Netto-Cashflow [CHF]": "{:,.2f}",
                "Kumuliert netto [CHF]": "{:,.2f}",
                "Kumuliert diskontiert nach CAPEX [CHF]": "{:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# Ertragsaufteilung (nur intern fuer PDF / Export)
# ============================================================
da_sum = 0.0
id_inc_sum = 0.0
daid_sum = 0.0

if rev_diag is not None:
    da_sum = float(rev_diag.sum_da_chf)
    id_inc_sum = float(rev_diag.sum_id_inc_chf)
    daid_sum = float(rev_diag.sum_total_chf)
else:
    if isinstance(ts_saved, pd.DataFrame) and not ts_saved.empty:
        if "rev_da_chf" in ts_saved.columns:
            da_sum = float(pd.to_numeric(ts_saved["rev_da_chf"], errors="coerce").fillna(0.0).sum())
        if "rev_id_inc_chf" in ts_saved.columns:
            id_inc_sum = float(pd.to_numeric(ts_saved["rev_id_inc_chf"], errors="coerce").fillna(0.0).sum())
        daid_sum = da_sum + id_inc_sum

sdl_sum = 0.0
if has_sdl and isinstance(sdl_ts, pd.DataFrame) and not sdl_ts.empty:
    parts = []
    for pref in ["prl_sym", "srl_up", "srl_down"]:
        cols_resolved = _resolve_sdl_product_cols(sdl_ts, pref)
        rev_series = _compute_sdl_revenue_series(sdl_ts, pref, cols_resolved, cfg=cfg)
        if not rev_series.empty:
            parts.append(pd.to_numeric(rev_series, errors="coerce").fillna(0.0))
    if parts:
        sdl_sum = float(np.sum([p.sum() for p in parts]))
    elif "sdl_total_rev_chf_h" in sdl_ts.columns:
        sdl_sum = float(pd.to_numeric(sdl_ts["sdl_total_rev_chf_h"], errors="coerce").fillna(0.0).sum())

mu_sum = 0.0
if (
    has_multiuse
    and isinstance(multiuse_ts, pd.DataFrame)
    and not multiuse_ts.empty
    and "rev_multiuse_chf_h" in multiuse_ts.columns
):
    mu_sum = float(pd.to_numeric(multiuse_ts["rev_multiuse_chf_h"], errors="coerce").fillna(0.0).sum())

rows = []
if daid_sum != 0.0 or has_da or has_id:
    rows.append({"Markt": "Day-Ahead", "Erlös [CHF]": da_sum})
    rows.append({"Markt": "Intraday (inkrementell)", "Erlös [CHF]": id_inc_sum})
    rows.append({"Markt": "DA+ID Total (additiv)", "Erlös [CHF]": daid_sum})
if has_sdl:
    rows.append({"Markt": "SDL Total", "Erlös [CHF]": sdl_sum})
if has_multiuse:
    rows.append({"Markt": "Multiuse TOTAL (Entscheidungsspur)", "Erlös [CHF]": mu_sum})

df_rev = pd.DataFrame(rows)
if not df_rev.empty:
    df_rev["Erlös [CHF]"] = pd.to_numeric(df_rev["Erlös [CHF]"], errors="coerce").fillna(0.0)

# ============================================================
# Zeitreihen
# ============================================================
st.markdown("---")
st.subheader("Zeitreihen")

view_full = _build_price_soc_dispatch_view(
    ts_df=ts_filt if isinstance(ts_filt, pd.DataFrame) else None,
    mu_df=mu_filt if isinstance(mu_filt, pd.DataFrame) else None,
    e_nom_kwh=e_nom_kwh,
)

ts_plot_prices = view_full.copy()
if len(ts_plot_prices) > max_points_prices:
    step = max(1, len(ts_plot_prices) // max_points_prices)
    ts_plot_prices = ts_plot_prices.iloc[::step].copy()

df_price_da = pd.DataFrame(columns=["ts", "serie_da", "value"])
df_soc = pd.DataFrame(columns=["ts", "serie_soc", "value"])
df_dispatch = pd.DataFrame(columns=["ts", "serie_p", "value"])
df_price_id = pd.DataFrame(columns=["ts", "serie_id", "value"])
df_spread = pd.DataFrame(columns=["ts", "serie_spread", "value"])

if not view_full.empty:
    df_price_da = _long(
        ts_plot_prices.rename(
            columns={
                "price_da_display": "price_da",
                "da_fc_display": "price_fc",
            }
        ),
        ["price_da", "price_fc"],
        "serie_da",
        {"price_da": "DA Settlement", "price_fc": "DA Forecast"},
    )

    soc_view = view_full[["ts", "soc_pct_display"]].copy()
    df_soc = _long(soc_view, ["soc_pct_display"], "serie_soc", {"soc_pct_display": "SOC [%]"})

    dispatch_view = view_full[["ts", "p_charge_display_kw", "p_discharge_display_kw"]].copy()
    dispatch_view["p_charge_plot"] = pd.to_numeric(dispatch_view["p_charge_display_kw"], errors="coerce").fillna(0.0)
    dispatch_view["p_discharge_plot"] = -pd.to_numeric(dispatch_view["p_discharge_display_kw"], errors="coerce").fillna(0.0)

    df_dispatch = _long(
        dispatch_view,
        ["p_charge_plot", "p_discharge_plot"],
        "serie_p",
        {"p_charge_plot": "Laden (+)", "p_discharge_plot": "Entladen (-)"},
    )

    df_price_id = _long(
        ts_plot_prices.rename(
            columns={
                "price_id_display": "price_id",
                "price_id_fc_display": "price_id_fc",
            }
        ),
        ["price_id", "price_id_fc"],
        "serie_id",
        {"price_id": "ID Settlement", "price_id_fc": "ID Forecast"},
    )

    if show_spread_chart:
        spread_view = ts_plot_prices.rename(
            columns={
                "spread_settle_display": "spread_settle",
                "spread_fc_display": "spread_fc",
            }
        )
        df_spread = _long(
            spread_view,
            ["spread_settle", "spread_fc"],
            "serie_spread",
            {
                "spread_settle": "Spread settle (ID-DA)",
                "spread_fc": "Spread forecast (ID-DA)",
            },
        )

# ============================================================
# Multiuse Charts
# ============================================================
if has_multiuse and isinstance(mu_filt, pd.DataFrame) and not mu_filt.empty:
    st.markdown("### Multiuse")

    mu_plot = mu_filt.copy()
    mu_plot["_x2"] = mu_plot["ts"] + pd.Timedelta(hours=1)

    mu_state_detail = _build_multiuse_market_state_detail(mu_filt, sdl_filt if has_sdl else None)
    mu_plot = mu_plot.merge(mu_state_detail, on="ts", how="left")

    state_col = "market_state_detail_display" if "market_state_detail_display" in mu_plot.columns else "market_state"
    mu_plot[state_col] = mu_plot[state_col].astype(str)

    state_chart = (
        alt.Chart(mu_plot)
        .mark_rect(opacity=0.75)
        .encode(
            x=alt.X("ts:T", title="Zeit"),
            x2=alt.X2("_x2:T"),
            y=alt.value(0),
            y2=alt.value(40),
            color=alt.Color(
                f"{state_col}:N",
                title="Markt / Produkt",
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("ts:T", title="Zeit"),
                alt.Tooltip(f"{state_col}:N", title="Markt / Produkt"),
            ],
        )
        .properties(height=80, title="Multiuse: Markt pro Stunde (Market-State)")
    )
    st.altair_chart(state_chart, use_container_width=True)

    if "rev_multiuse_chf_h" in mu_plot.columns:
        mu_plot["rev_multiuse_chf_h"] = pd.to_numeric(mu_plot["rev_multiuse_chf_h"], errors="coerce").fillna(0.0)

        mu_rev = (
            alt.Chart(mu_plot)
            .mark_line(interpolate="step-after", strokeWidth=2.0)
            .encode(
                x=alt.X("ts:T", title="Zeit"),
                y=alt.Y("rev_multiuse_chf_h:Q", title="Erloes [CHF/h]"),
                tooltip=[
                    alt.Tooltip("ts:T", title="Zeit"),
                    alt.Tooltip("rev_multiuse_chf_h:Q", title="Erloes", format=",.2f"),
                    alt.Tooltip(f"{state_col}:N", title="Markt / Produkt"),
                ],
            )
            .properties(height=220, title="Multiuse: Erloes pro Stunde (Entscheidungsspur)")
        )
        st.altair_chart(mu_rev, use_container_width=True)

# ============================================================
# DA / ID / Batterie Charts
# ============================================================
st.markdown("### Day-Ahead / Intraday / Batterie")

if len(df_price_da) > 0:
    st.altair_chart(
        _chart(df_price_da, "Day-Ahead Preis: Settlement vs Forecast", "Preis [CHF/MWh]", "serie_da", mark="step"),
        use_container_width=True,
    )

if len(df_soc) > 0:
    st.altair_chart(
        _chart(df_soc, "State of Charge", "SOC [%]", "serie_soc", mark="line"),
        use_container_width=True,
    )

if len(df_dispatch) > 0:
    st.altair_chart(
        _chart(df_dispatch, "Batterie-Dispatch (Leistung)", "Leistung [kW]", "serie_p", mark="step"),
        use_container_width=True,
    )

if len(df_price_id) > 0:
    st.altair_chart(
        _chart(df_price_id, "Intraday Preis: Settlement vs Forecast", "Preis [CHF/MWh]", "serie_id", mark="step"),
        use_container_width=True,
    )

if show_spread_chart and len(df_spread) > 0:
    st.altair_chart(
        _chart(df_spread, "Spread: ID - DA (Settlement vs Forecast)", "Spread [CHF/MWh]", "serie_spread", mark="step"),
        use_container_width=True,
    )

# ============================================================
# SDL Charts
# ============================================================
if has_sdl and isinstance(sdl_filt, pd.DataFrame) and not sdl_filt.empty:
    st.markdown("### Regelenergie (SDL) - Bids, Clearing, Zuschlag, Erloese")

    available_products = []
    for label, pref in [
        ("PRL (sym)", "prl_sym"),
        ("SRL UP", "srl_up"),
        ("SRL DOWN", "srl_down"),
    ]:
        cols_resolved = _resolve_sdl_product_cols(sdl_filt, pref)
        if any(v is not None for v in cols_resolved.values()):
            available_products.append((label, pref, cols_resolved))

    if available_products:
        labels = [x[0] for x in available_products]
        prod_label = st.selectbox("Produkt (SDL)", options=labels, index=0)
        selected = [x for x in available_products if x[0] == prod_label][0]
        _, pref, cols_resolved = selected

        title = f"SDL: {prod_label}"
        acc_col = cols_resolved["accepted"]
        bid_col = cols_resolved["bid"]
        clear_col = cols_resolved["clear"]

        ts_sdl = sdl_filt.copy()
        ts_sdl["_x2"] = ts_sdl["ts"] + pd.Timedelta(hours=1)

        bg_chart = None
        if acc_col is not None:
            bg = ts_sdl[["ts", "_x2", acc_col]].copy()
            bg[acc_col] = pd.to_numeric(bg[acc_col], errors="coerce").fillna(0).astype(int)
            bg["status"] = bg[acc_col].map({1: "Zuschlag", 0: "kein Zuschlag"}).astype(str)

            bg_chart = (
                alt.Chart(bg)
                .mark_rect(opacity=0.22)
                .encode(
                    x=alt.X("ts:T", title="Zeit"),
                    x2=alt.X2("_x2:T"),
                    y=alt.value(0),
                    y2=alt.value(260),
                    color=alt.Color(
                        "status:N",
                        legend=alt.Legend(title="Zuschlag", orient="right"),
                        sort=["kein Zuschlag", "Zuschlag"],
                        scale=alt.Scale(domain=["kein Zuschlag", "Zuschlag"], range=["#d9d9d9", "#ffd54f"]),
                    ),
                    tooltip=[
                        alt.Tooltip("ts:T", title="Zeit"),
                        alt.Tooltip("status:N", title="Status"),
                    ],
                )
                .properties(height=260)
            )

        line_chart = None
        line_cols = []
        rename_map = {}

        if bid_col is not None:
            line_cols.append(bid_col)
            rename_map[bid_col] = "Bid"
        if clear_col is not None:
            line_cols.append(clear_col)
            rename_map[clear_col] = "Clearing"

        if line_cols:
            df_lines = ts_sdl[["ts"] + line_cols].copy().rename(columns=rename_map)

            if "Bid" in df_lines.columns:
                df_lines["Bid"] = pd.to_numeric(df_lines["Bid"], errors="coerce")
            if "Clearing" in df_lines.columns:
                df_lines["Clearing"] = pd.to_numeric(df_lines["Clearing"], errors="coerce")

            value_vars = [c for c in ["Bid", "Clearing"] if c in df_lines.columns]
            df_long_lines = (
                df_lines.melt(id_vars=["ts"], value_vars=value_vars, var_name="Serie", value_name="Wert")
                .dropna(subset=["Wert"])
            )

            if not df_long_lines.empty:
                line_chart = (
                    alt.Chart(df_long_lines)
                    .mark_line(interpolate="step-after", strokeWidth=2.2)
                    .encode(
                        x=alt.X("ts:T", title="Zeit"),
                        y=alt.Y("Wert:Q", title="Preis [CHF/MW/h]"),
                        color=alt.Color(
                            "Serie:N",
                            scale=alt.Scale(domain=["Bid", "Clearing"], range=["#1f77b4", "#d62728"]),
                            legend=alt.Legend(orient="right"),
                        ),
                        tooltip=[
                            alt.Tooltip("ts:T", title="Zeit"),
                            alt.Tooltip("Serie:N"),
                            alt.Tooltip("Wert:Q", format=",.2f"),
                        ],
                    )
                    .properties(height=260, title=f"{title}: Bid vs Clearing + Zuschlagsband")
                )

        if bg_chart is not None and line_chart is not None:
            st.altair_chart(alt.layer(bg_chart, line_chart).resolve_scale(y="shared"), use_container_width=True)
        elif line_chart is not None:
            st.altair_chart(line_chart, use_container_width=True)
        elif bg_chart is not None:
            st.altair_chart(bg_chart, use_container_width=True)

        rev_series = _compute_sdl_revenue_series(ts_sdl, pref, cols_resolved, cfg=cfg)
        if not rev_series.empty:
            ts_sdl["rev_display_chf_h"] = pd.to_numeric(rev_series, errors="coerce").fillna(0.0)

            rev_chart = (
                alt.Chart(ts_sdl)
                .mark_line(interpolate="step-after", strokeWidth=2.0)
                .encode(
                    x=alt.X("ts:T", title="Zeit"),
                    y=alt.Y("rev_display_chf_h:Q", title="Erloes [CHF/h]"),
                    tooltip=[
                        alt.Tooltip("ts:T", title="Zeit"),
                        alt.Tooltip("rev_display_chf_h:Q", title="Erloes", format=",.2f"),
                    ],
                )
                .properties(height=220, title=f"{title}: Erloes pro Stunde")
            )
            st.altair_chart(rev_chart, use_container_width=True)
    else:
        st.info("SDL-Zeitreihen wurden geladen, aber es konnten keine produktbezogenen SDL-Spalten erkannt werden.")

# ============================================================
# Export
# ============================================================
st.markdown("---")
st.subheader("Export")

dashboard_export_df = _build_dashboard_export_timeseries(
    ts_df=ts_filt if isinstance(ts_filt, pd.DataFrame) else None,
    sdl_df=sdl_filt if isinstance(sdl_filt, pd.DataFrame) else None,
    mu_df=mu_filt if isinstance(mu_filt, pd.DataFrame) else None,
    e_nom_kwh=e_nom_kwh,
    cfg=cfg,
)

safe_sname = str(sname).replace(" ", "_")

st.download_button(
    "Dashboard Timeseries herunterladen (CSV)",
    data=_csv_bytes_for_dashboard_export(dashboard_export_df),
    file_name=f"{safe_sname}_dashboard_timeseries.csv",
    mime="text/csv",
    key="dl_dashboard_timeseries_csv",
)

dashboard_pdf_bytes = _make_dashboard_pdf_bytes(
    scenario_name=sname,
    market_mode=market_mode,
    start_ts=start_ts,
    end_ts=end_ts,
    annual_rev_chf=annual_rev_chf,
    npv_chf=npv_chf,
    payback_str=payback_str,
    cycles_efc=cycles_efc,
    degradation_rate_per_year=degradation_rate_per_year,
    revenue_split_df=df_rev,
    dashboard_export_df=dashboard_export_df,
)

if dashboard_pdf_bytes is not None:
    st.download_button(
        "Dashboard Bericht herunterladen (PDF)",
        data=dashboard_pdf_bytes,
        file_name=f"{safe_sname}_dashboard_report.pdf",
        mime="application/pdf",
        key="dl_dashboard_report_pdf",
    )
else:
    st.info("PDF-Export ist in dieser Umgebung nicht verfuegbar.")