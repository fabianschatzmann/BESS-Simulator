# pages/5_Dashboard.py
# -*- coding: utf-8 -*-

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


def _compute_npv_and_payback(
    annual_revenue_chf: float,
    total_project_cost_chf: float,
    opex_chf_per_year: float,
    wacc: float,
    project_lifetime_years: int,
) -> tuple[float, float | None]:
    capex = float(total_project_cost_chf or 0.0)
    opex = float(opex_chf_per_year or 0.0)
    cf = float(annual_revenue_chf or 0.0) - opex

    npv = -capex
    r = float(wacc or 0.0)
    n = int(project_lifetime_years or 0)

    for t in range(1, n + 1):
        npv += cf / ((1.0 + r) ** t) if (1.0 + r) != 0 else 0.0

    payback_years = None if cf <= 0 else (capex / cf)
    return float(npv), payback_years


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
        clear_col = _pick_existing_col_by_patterns(df, pref, ["clear_true", "clearing", "_clear"])

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
    st.warning("Keine Ergebnisse vorhanden. Bitte zuerst **Prognose & Dispatch** ausführen.")
    st.stop()


# ============================================================
# Build 'results' object (Legacy) -> DA/ID bewusst wie ursprünglich
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
                st.warning("Keine Ergebnisse vorhanden. Bitte zuerst **Prognose & Dispatch** ausführen.")
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
cycles_efc = _efc_cycles_from_dispatch(ts_filt, e_nom_kwh=e_nom_kwh, dt_h=1.0) if has_da else 0.0

total_project_cost_chf = _get_eco_float(cfg, "total_project_cost_chf", default=0.0)
opex_chf_per_year = _get_eco_float(cfg, "opex_chf_per_year", default=0.0)
wacc = _get_eco_float(cfg, "wacc", default=0.06)
project_lifetime_years = _get_eco_int(
    cfg,
    "project_lifetime_years",
    default=_get_eco_int(cfg, "asset_life_years", default=15),
)

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

npv_chf, payback_years = _compute_npv_and_payback(
    annual_revenue_chf=annual_rev_chf,
    total_project_cost_chf=total_project_cost_chf,
    opex_chf_per_year=opex_chf_per_year,
    wacc=wacc,
    project_lifetime_years=project_lifetime_years,
)

payback_str = "∞" if payback_years is None else f"{payback_years:.2f}"

if has_multiuse:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Σ Multiuse (TOTAL) [CHF]", f"{annual_rev_chf:.0f}")
    c2.metric("Σ SDL (TOTAL) [CHF]", f"{float(kpi.get('sdl_total_revenue_chf', 0.0)):.0f}")
    c3.metric("Σ DA+ID (additiv) [CHF]", f"{float(rev_diag.sum_total_chf):.0f}" if rev_diag is not None else "—")
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

st.caption(f"Szenario: {sname} | Marktmodus: **{market_mode}**")


# ============================================================
# Ertragsaufteilung
# ============================================================
st.markdown("---")
st.subheader("Ertragsaufteilung")

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
    if "sdl_total_rev_chf_h" in sdl_ts.columns:
        sdl_sum = float(pd.to_numeric(sdl_ts["sdl_total_rev_chf_h"], errors="coerce").fillna(0.0).sum())
    else:
        parts = []
        for pref in ["prl_sym", "srl_up", "srl_down"]:
            for cand in [f"{pref}_rev_total_chf_h", f"{pref}_rev_chf_h"]:
                if cand in sdl_ts.columns:
                    parts.append(pd.to_numeric(sdl_ts[cand], errors="coerce").fillna(0.0))
                    break
        if parts:
            sdl_sum = float(np.sum([p.sum() for p in parts]))

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
df_rev["Erlös [CHF]"] = pd.to_numeric(df_rev["Erlös [CHF]"], errors="coerce").fillna(0.0)

chart_rev = (
    alt.Chart(df_rev)
    .mark_bar()
    .encode(
        x=alt.X("Markt:N", title="Markt"),
        y=alt.Y("Erlös [CHF]:Q", title="Erlös [CHF]"),
        tooltip=[alt.Tooltip("Markt:N"), alt.Tooltip("Erlös [CHF]:Q", format=",.2f")],
    )
    .properties(height=280)
)
st.altair_chart(chart_rev, use_container_width=True)


# ============================================================
# Zeitreihen: Chartdaten vorbereiten
# ============================================================
st.markdown("---")
st.subheader("Zeitreihen")

ts_plot_prices = ts_filt.copy()
if len(ts_plot_prices) > max_points_prices:
    step = max(1, len(ts_plot_prices) // max_points_prices)
    ts_plot_prices = ts_plot_prices.iloc[::step].copy()

df_price_da = pd.DataFrame(columns=["ts", "serie_da", "value"])
df_soc = pd.DataFrame(columns=["ts", "serie_soc", "value"])
df_dispatch = pd.DataFrame(columns=["ts", "serie_p", "value"])
df_price_id = pd.DataFrame(columns=["ts", "serie_id", "value"])
df_spread = pd.DataFrame(columns=["ts", "serie_spread", "value"])

if has_da:
    if "price_da_fc" in ts_plot_prices.columns and "price_fc" not in ts_plot_prices.columns:
        ts_plot_prices["price_fc"] = ts_plot_prices["price_da_fc"]

    df_price_da = _long(
        ts_plot_prices,
        ["price_da", "price_fc"],
        "serie_da",
        {"price_da": "DA Settlement", "price_fc": "DA Forecast"},
    )

    ts_da = ts_filt.copy()
    if "soc_pct" not in ts_da.columns and ("soc_kwh" in ts_da.columns) and e_nom_kwh and e_nom_kwh > 0:
        ts_da["soc_pct"] = (pd.to_numeric(ts_da["soc_kwh"], errors="coerce") / float(e_nom_kwh)) * 100.0

    df_soc = _long(ts_da, ["soc_pct"], "serie_soc", {"soc_pct": "SOC [%]"})

    if "p_bess" in ts_da.columns and ("p_charge_kw" not in ts_da.columns or "p_discharge_kw" not in ts_da.columns):
        pnet = pd.to_numeric(ts_da["p_bess"], errors="coerce").fillna(0.0)
        ts_da["p_discharge_kw"] = pnet.clip(lower=0.0)
        ts_da["p_charge_kw"] = (-pnet).clip(lower=0.0)

    ts_da["p_charge_plot"] = (
        pd.to_numeric(ts_da.get("p_charge_kw", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    ts_da["p_discharge_plot"] = -pd.to_numeric(
        ts_da.get("p_discharge_kw", 0.0), errors="coerce"
    ).fillna(0.0).abs()

    df_dispatch = _long(
        ts_da,
        ["p_charge_plot", "p_discharge_plot"],
        "serie_p",
        {"p_charge_plot": "Laden (+)", "p_discharge_plot": "Entladen (−)"},
    )

if has_id:
    df_price_id = _long(
        ts_plot_prices,
        ["price_id", "price_id_fc"],
        "serie_id",
        {"price_id": "ID Settlement", "price_id_fc": "ID Forecast"},
    )

    if show_spread_chart:
        tmp = ts_plot_prices.copy()

        if ("spread_settle" not in tmp.columns) and ("price_id" in tmp.columns) and ("price_da" in tmp.columns):
            tmp["spread_settle"] = (
                pd.to_numeric(tmp["price_id"], errors="coerce")
                - pd.to_numeric(tmp["price_da"], errors="coerce")
            )

        if ("spread_fc" not in tmp.columns) and ("price_id_fc" in tmp.columns) and ("price_da_fc" in tmp.columns):
            tmp["spread_fc"] = (
                pd.to_numeric(tmp["price_id_fc"], errors="coerce")
                - pd.to_numeric(tmp["price_da_fc"], errors="coerce")
            )

        df_spread = _long(
            tmp,
            ["spread_settle", "spread_fc"],
            "serie_spread",
            {
                "spread_settle": "Spread settle (ID-DA)",
                "spread_fc": "Spread forecast (ID-DA)",
            },
        )


# ============================================================
# Multiuse Charts separat
# ============================================================
if has_multiuse and isinstance(mu_filt, pd.DataFrame) and not mu_filt.empty:
    st.markdown("### Multiuse")

    mu_plot = mu_filt.copy()
    mu_plot["market_state"] = mu_plot["market_state"].astype(str)
    mu_plot["_x2"] = mu_plot["ts"] + pd.Timedelta(hours=1)

    state_chart = (
        alt.Chart(mu_plot)
        .mark_rect(opacity=0.75)
        .encode(
            x=alt.X("ts:T", title="Zeit"),
            x2=alt.X2("_x2:T"),
            y=alt.value(0),
            y2=alt.value(40),
            color=alt.Color(
                "market_state:N",
                title="Markt",
                legend=alt.Legend(orient="right"),
                sort=["DA_ID", "SDL"],
            ),
            tooltip=[
                alt.Tooltip("ts:T", title="Zeit"),
                alt.Tooltip("market_state:N", title="Markt"),
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
                y=alt.Y("rev_multiuse_chf_h:Q", title="Erlös [CHF/h]"),
                tooltip=[
                    alt.Tooltip("ts:T", title="Zeit"),
                    alt.Tooltip("rev_multiuse_chf_h:Q", title="Erlös", format=",.2f"),
                    alt.Tooltip("market_state:N", title="Markt"),
                ],
            )
            .properties(height=220, title="Multiuse: Erlös pro Stunde (Entscheidungsspur)")
        )
        st.altair_chart(mu_rev, use_container_width=True)


# ============================================================
# Obere 5 Charts separat
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
        _chart(df_spread, "Spread: ID − DA (Settlement vs Forecast)", "Spread [CHF/MWh]", "serie_spread", mark="step"),
        use_container_width=True,
    )


# ============================================================
# SDL Charts separat
# ============================================================
if has_sdl and isinstance(sdl_filt, pd.DataFrame) and not sdl_filt.empty:
    st.markdown("### Regelenergie (SDL) – Bids, Clearing, Zuschlag, Erlöse")

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
        rev_col = cols_resolved["rev"]

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

        if rev_col is not None:
            ts_sdl[rev_col] = pd.to_numeric(ts_sdl[rev_col], errors="coerce").fillna(0.0)

            rev_chart = (
                alt.Chart(ts_sdl)
                .mark_line(interpolate="step-after", strokeWidth=2.0)
                .encode(
                    x=alt.X("ts:T", title="Zeit"),
                    y=alt.Y(f"{rev_col}:Q", title="Erlös [CHF/h]"),
                    tooltip=[
                        alt.Tooltip("ts:T", title="Zeit"),
                        alt.Tooltip(f"{rev_col}:Q", title="Erlös", format=",.2f"),
                    ],
                )
                .properties(height=220, title=f"{title}: Erlös pro Stunde")
            )
            st.altair_chart(rev_chart, use_container_width=True)
    else:
        st.info("SDL-Zeitreihen wurden geladen, aber es konnten keine produktbezogenen SDL-Spalten erkannt werden.")


# ============================================================
# Export
# ============================================================
st.markdown("---")
st.subheader("Export")

st.download_button(
    "Primary Timeseries herunterladen (CSV)",
    data=ts.to_csv(index=False).encode("utf-8"),
    file_name="bess_results_timeseries.csv",
    mime="text/csv",
    key="dl_results_timeseries_csv",
)

if has_sdl and isinstance(sdl_ts, pd.DataFrame) and not sdl_ts.empty:
    st.download_button(
        "SDL Timeseries herunterladen (CSV)",
        data=sdl_ts.to_csv(index=False).encode("utf-8"),
        file_name="bess_sdl_timeseries.csv",
        mime="text/csv",
        key="dl_sdl_timeseries_csv",
    )

if has_multiuse and isinstance(multiuse_ts, pd.DataFrame) and not multiuse_ts.empty:
    st.download_button(
        "Multiuse Timeseries herunterladen (CSV)",
        data=multiuse_ts.to_csv(index=False).encode("utf-8"),
        file_name="bess_multiuse_timeseries.csv",
        mime="text/csv",
        key="dl_multiuse_timeseries_csv",
    )