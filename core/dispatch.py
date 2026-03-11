# pages/5_Dashboard.py
# -*- coding: utf-8 -*-
"""
Dashboard (Analyse & Visualisierung)

Konventionen:
- SOC immer in % (soc_pct)
- Batterie-Dispatch: genau zwei Linien
    Laden = positiv
    Entladen = negativ
- Day-Ahead Preise sind stündlich konstant -> Step-Darstellung (Altair: mark_line(interpolate="step-after"))
- Zoom/Pan: nur X-Achse (Y bleibt fix)
- Legenden: pro Diagramm getrennt (nur relevante Linien)

Wichtig:
- SOC wird für die Visualisierung aus dem Dispatch rekonstruiert (Konsistenzcheck),
  um SOC/Dispatch-Mismatches aus alten Files/Indexing/Sign-Konventionen zu eliminieren.
- Downsampling wird NUR für Preisplot angewendet (optional), nicht für SOC/Dispatch.
"""

import streamlit as st
import pandas as pd
import altair as alt

from core.scenario_store import load_parquet, load_config
from core.results import compute_results_from_dispatch


# =========================
# Page setup
# =========================
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

# Scope-sicher
ts_saved = None

# =========================
# Fallback load if session missing
# =========================
if "results" not in st.session_state:
    sname = st.session_state.get("scenario_name", "BaseCase_2025")

    ts_saved = load_parquet(sname, "results_timeseries")
    disp = st.session_state.get("dispatch")
    if disp is None:
        disp = load_parquet(sname, "dispatch")

    if disp is None and ts_saved is None:
        st.warning("Keine Ergebnisse vorhanden. Bitte zuerst **Prognose & Dispatch** ausführen.")
        st.stop()

    cfg = st.session_state.get("scenario_config") or load_config(sname) or {}
    try:
        e_nom = float(cfg.get("battery", {}).get("e_nom_kwh", 0.0))
    except Exception:
        e_nom = None

    if disp is not None:
        res = compute_results_from_dispatch(
            dispatch_df=disp,
            runtime_s=0.0,
            market_name="day_ahead",
            e_nom_kwh=e_nom,
        )

        # NICHT blind überschreiben -> sonst mischst du Runs
        # (nur übernehmen, wenn gleiche Länge & gleiche Zeitstempel)
        if ts_saved is not None and isinstance(res, dict) and isinstance(res.get("timeseries"), pd.DataFrame):
            try:
                tc = None
                for cand in ["ts", "timestamp", "datetime", "time", "ts_local", "ts_utc", "ts_key"]:
                    if cand in ts_saved.columns and cand in res["timeseries"].columns:
                        tc = cand
                        break
                if (
                    tc is not None
                    and len(ts_saved) == len(res["timeseries"])
                    and ts_saved[tc].equals(res["timeseries"][tc])
                ):
                    res["timeseries"] = ts_saved
            except Exception:
                pass

        st.session_state["results"] = res
    else:
        st.session_state["results"] = {
            "kpis": {},
            "revenue_breakdown": [],
            "top_days": [],
            "worst_days": [],
            "timeseries": ts_saved,
        }

res = st.session_state["results"]
kpi = res.get("kpis", {}) if isinstance(res, dict) else {}
rev_breakdown = res.get("revenue_breakdown", []) if isinstance(res, dict) else []
top_days = res.get("top_days", []) if isinstance(res, dict) else []
worst_days = res.get("worst_days", []) if isinstance(res, dict) else []

def _k(key, default=0.0):
    try:
        return kpi.get(key, default)
    except Exception:
        return default

# =========================
# KPI tiles
# =========================
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Jahresertrag [CHF]", f"{_k('annual_revenue_chf', 0.0):.0f}")
c2.metric("Zyklen / Jahr", f"{_k('cycles_per_year', 0.0):.2f}")
c3.metric("Peak-Reduktion [kW]", f"{_k('peak_reduction_kw', 0.0):.1f}")
c4.metric("Effektive RTE [-]", f"{_k('effective_rte', 0.0):.3f}")
c5.metric("Constraint-Verletzungen", f"{int(_k('constraint_violations', 0))}")
c6.metric("Runtime [s]", f"{_k('runtime_s', 0.0):.2f}")

st.markdown("---")

# =========================
# Revenue + Top/Worst days
# =========================
left, right = st.columns([1, 1])

with left:
    st.subheader("Ertragsaufteilung nach Markt")
    if rev_breakdown:
        df_rev = pd.DataFrame(rev_breakdown)
        if "market" in df_rev.columns and "revenue_chf" in df_rev.columns:
            st.bar_chart(df_rev.set_index("market")["revenue_chf"])
        else:
            st.dataframe(df_rev, use_container_width=True)
    else:
        st.info("Noch kein Revenue-Breakdown vorhanden.")

with right:
    st.subheader("Top / Worst Tage")
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Top 10 Tage")
        st.dataframe(pd.DataFrame(top_days), use_container_width=True, height=320)
    with col_b:
        st.caption("Worst 10 Tage")
        st.dataframe(pd.DataFrame(worst_days), use_container_width=True, height=320)

# =========================
# Timeseries
# =========================
st.markdown("---")
st.subheader("Zeitreihen")

ts = res.get("timeseries") if isinstance(res, dict) else None
if ts is None or not isinstance(ts, pd.DataFrame) or ts.empty:
    st.info("Keine Timeseries-Daten vorhanden.")
    st.stop()

ts = ts.copy()

# -------------------------
# Determine time column (deine Dashboard-Logik)
# -------------------------
time_col = None
for cand in ["ts", "timestamp", "datetime", "time", "ts_local", "ts_utc", "ts_key"]:
    if cand in ts.columns:
        time_col = cand
        break
if time_col is None:
    ts["_idx"] = range(len(ts))
    time_col = "_idx"

is_dt = pd.api.types.is_datetime64_any_dtype(ts[time_col])
if is_dt:
    ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
    # Vega/Altair: tz-aware ist oft problematisch -> tz entfernen
    try:
        ts[time_col] = ts[time_col].dt.tz_localize(None)
    except Exception:
        pass

# -------------------------
# Plot settings
# -------------------------
with st.expander("Plot-Einstellungen", expanded=True):
    max_points_prices = st.slider("Max. Punkte (nur Preise)", 500, 8760, 2500, 250)

    if is_dt:
        tmin, tmax = ts[time_col].min(), ts[time_col].max()
        cA, cB = st.columns(2)
        with cA:
            start_ts = st.datetime_input("Start", value=tmin)
        with cB:
            end_ts = st.datetime_input("Ende", value=tmax)
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts
    else:
        start_ts, end_ts = None, None
        st.info("Zeitachse ist nicht datetime – Zeitraumfilter ist deaktiviert.")

# Filter (wichtig: SOC-Rekonstruktion erfolgt auf gefilterter, aber NICHT downsampled Basis!)
ts_filt = ts.copy()
if is_dt and start_ts is not None and end_ts is not None:
    ts_filt = ts_filt[
        (ts_filt[time_col] >= pd.Timestamp(start_ts)) &
        (ts_filt[time_col] <= pd.Timestamp(end_ts))
    ].copy()

# -------------------------
# Helpers
# -------------------------
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

# -------------------------
# Price aliases
# -------------------------
if "price_da" not in ts_filt.columns:
    if "da_price_settle" in ts_filt.columns:
        ts_filt["price_da"] = ts_filt["da_price_settle"]
    elif "da_price" in ts_filt.columns:
        ts_filt["price_da"] = ts_filt["da_price"]

if "price_fc" not in ts_filt.columns:
    if "da_price_fcst" in ts_filt.columns:
        ts_filt["price_fc"] = ts_filt["da_price_fcst"]
    elif "da_price_fc" in ts_filt.columns:
        ts_filt["price_fc"] = ts_filt["da_price_fc"]

# -------------------------
# Ensure dispatch columns exist (derive if necessary)
# -------------------------
if "p_charge_kw" not in ts_filt.columns or "p_discharge_kw" not in ts_filt.columns:
    # derive from p_net_kw (optimizer convention: positive=discharge)
    if "p_net_kw" in ts_filt.columns:
        pnet = _to_num(ts_filt["p_net_kw"])
        ts_filt["p_discharge_kw"] = pnet.clip(lower=0.0)   # >=0 magnitude
        ts_filt["p_charge_kw"] = (-pnet).clip(lower=0.0)   # >=0 magnitude
    elif "p_bess" in ts_filt.columns:
        # fallback: assume p_bess >0 discharge, <0 charge
        p = _to_num(ts_filt["p_bess"])
        ts_filt["p_discharge_kw"] = p.clip(lower=0.0)
        ts_filt["p_charge_kw"] = (-p).clip(lower=0.0)

# -------------------------
# FIX B: Robust sign normalization for SOC reconstruction
# We want:
#   p_charge_mag >= 0
#   p_discharge_mag >= 0 (magnitude), regardless of stored sign convention
# -------------------------
if "p_charge_kw" in ts_filt.columns and "p_discharge_kw" in ts_filt.columns:
    pch_raw = _to_num(ts_filt["p_charge_kw"])
    pdis_raw = pd.to_numeric(ts_filt["p_discharge_kw"], errors="coerce").fillna(0.0)

    # charge magnitude
    pch_mag = pch_raw.clip(lower=0.0)

    # discharge magnitude
    if (pdis_raw <= 0).all() and (pdis_raw.abs().sum() > 0):
        # discharge stored as negative values
        pdis_mag = (-pdis_raw).clip(lower=0.0)
    else:
        # discharge stored as positive magnitudes OR mixed -> safest: abs()
        pdis_mag = pdis_raw.abs().clip(lower=0.0)

    if (pdis_raw < 0).any() and (pdis_raw > 0).any():
        st.warning("Hinweis: p_discharge_kw hat gemischte Vorzeichen. Für SOC-Rekonstruktion wird abs() verwendet.")

    ts_filt["_pch_mag_kw"] = pch_mag
    ts_filt["_pdis_mag_kw"] = pdis_mag

    # Plot convention: Laden positiv, Entladen negativ (zwei Linien)
    ts_filt["p_charge_plot"] = pch_mag
    ts_filt["p_discharge_plot"] = -pdis_mag

# -------------------------
# SOC reconstruction from dispatch (guaranteed consistency)
# -------------------------
sname = st.session_state.get("scenario_name", "BaseCase_2025")
cfg = st.session_state.get("scenario_config") or load_config(sname) or {}

e_nom_kwh = _get_float(cfg, ["e_nom_kwh", "E_nom_kwh", "e_nom"], default=None)
eta_ch = _get_float(cfg, ["eta_ch", "eta_charge", "eta_c"], default=1.0)
eta_dis = _get_float(cfg, ["eta_dis", "eta_discharge", "eta_d"], default=1.0)
soc0_frac = _get_float(cfg, ["soc0", "soc_init", "soc_init_frac"], default=None)

dt = 1.0  # 1h

# initial SOC in kWh (priority)
soc0_kwh = None
if "soc_kwh" in ts_filt.columns:
    try:
        soc0_kwh = float(pd.to_numeric(ts_filt["soc_kwh"].iloc[0], errors="coerce"))
        if pd.isna(soc0_kwh):
            soc0_kwh = None
    except Exception:
        soc0_kwh = None

if soc0_kwh is None and "soc_pct" in ts_filt.columns and e_nom_kwh is not None and e_nom_kwh > 0:
    try:
        soc0_kwh = float(pd.to_numeric(ts_filt["soc_pct"].iloc[0], errors="coerce")) / 100.0 * e_nom_kwh
        if pd.isna(soc0_kwh):
            soc0_kwh = None
    except Exception:
        soc0_kwh = None

if soc0_kwh is None and soc0_frac is not None and e_nom_kwh is not None and e_nom_kwh > 0:
    soc0_kwh = soc0_frac * e_nom_kwh

if soc0_kwh is None:
    soc0_kwh = 0.0

if (
    e_nom_kwh is not None and e_nom_kwh > 0
    and "_pch_mag_kw" in ts_filt.columns
    and "_pdis_mag_kw" in ts_filt.columns
):
    pch = ts_filt["_pch_mag_kw"].values
    pdis = ts_filt["_pdis_mag_kw"].values
    eta_dis_safe = max(float(eta_dis), 1e-9)

    soc = [float(soc0_kwh)]
    for t in range(len(ts_filt)):
        soc_next = soc[-1] + (float(eta_ch) * pch[t] * dt) - ((1.0 / eta_dis_safe) * pdis[t] * dt)
        soc.append(soc_next)

    # Align start-of-hour SOC to ts[t]
    soc_start = pd.Series(soc[:-1], index=ts_filt.index)
    ts_filt["soc_kwh_calc"] = soc_start
    ts_filt["soc_pct_calc"] = (soc_start / float(e_nom_kwh)) * 100.0

    # Use computed SOC for plotting
    ts_filt["soc_pct"] = ts_filt["soc_pct_calc"]

    # Debug flags
    ts_filt["flag_soc_change_without_power"] = (
        (soc_start.diff().abs() > 1e-6) &
        (ts_filt["_pch_mag_kw"] < 1e-6) &
        (ts_filt["_pdis_mag_kw"] < 1e-6)
    )
else:
    # fallback: ensure soc_pct exists if possible
    if "soc_pct" not in ts_filt.columns:
        if "soc" in ts_filt.columns:
            ts_filt["soc_pct"] = _to_num(ts_filt["soc"]) * 100.0
        elif "soc_kwh" in ts_filt.columns and e_nom_kwh is not None and e_nom_kwh > 0:
            ts_filt["soc_pct"] = (_to_num(ts_filt["soc_kwh"]) / float(e_nom_kwh)) * 100.0

# Optional debug: show suspicious rows (should be empty if reconstruction is used and downsampling isn't corrupting)
if "flag_soc_change_without_power" in ts_filt.columns:
    dbg = ts_filt[ts_filt["flag_soc_change_without_power"]].copy()
    if not dbg.empty:
        st.warning("Sanity-Check: SOC-Änderung bei (rekonstruiert) 0 Leistung erkannt. Das deutet auf Zeit-/Dateninkonsistenz hin.")
        cols_show = [time_col, "_pch_mag_kw", "_pdis_mag_kw", "soc_kwh_calc", "soc_pct"]
        cols_show = [c for c in cols_show if c in dbg.columns]
        st.dataframe(dbg[cols_show], use_container_width=True, height=220)

# -------------------------
# FIX A: Downsampling NUR für Preise (optional)
# SOC & Dispatch bleiben stündlich vollständig, damit keine "fehlenden" Leistungsstunden entstehen.
# -------------------------
ts_plot_socdisp = ts_filt.copy()

ts_plot_prices = ts_filt.copy()
if len(ts_plot_prices) > max_points_prices:
    step = max(1, len(ts_plot_prices) // max_points_prices)
    ts_plot_prices = ts_plot_prices.iloc[::step].copy()

# =========================
# Altair helpers
# =========================
x_enc = alt.X(f"{time_col}:T" if is_dt else f"{time_col}:Q", title="Zeit")

def _long(df: pd.DataFrame, cols, series_col: str, label_map=None) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=[time_col, series_col, "value"])
    out = (
        df[[time_col] + keep]
        .melt(id_vars=[time_col], var_name=series_col, value_name="value")
        .dropna(subset=["value"])
    )
    if label_map:
        out[series_col] = out[series_col].map(lambda x: label_map.get(x, x))
    return out

def _chart(df_long: pd.DataFrame, title: str, y_title: str, series_col: str, height=240, mark: str = "line"):
    """
    mark:
      - "line": normal line
      - "step": step-like (Altair via interpolate="step-after")
    """
    if df_long.empty:
        empty = pd.DataFrame({time_col: [], series_col: [], "value": []})
        base = alt.Chart(empty)
        if mark == "step":
            base_mark = base.mark_line(interpolate="step-after")
        else:
            base_mark = base.mark_line()
        return base_mark.encode(
            x=x_enc,
            y=alt.Y("value:Q", title=y_title),
        ).properties(title=title, height=height)

    base = alt.Chart(df_long)
    if mark == "step":
        chart = base.mark_line(interpolate="step-after")
    else:
        chart = base.mark_line()

    return (
        chart.encode(
            x=x_enc,
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color(
                f"{series_col}:N",
                title="Linie",
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                alt.Tooltip(time_col, title="Zeit"),
                alt.Tooltip(f"{series_col}:N", title="Serie"),
                alt.Tooltip("value:Q", title="Wert", format=",.4f"),
            ],
        )
        .properties(title=title, height=height)
    )

# =========================
# Build charts (separate series columns => separate legends)
# =========================

# --- Price chart (STEP) ---
df_price = _long(
    ts_plot_prices,
    ["price_da", "price_fc"],
    series_col="series_price",
    label_map={"price_da": "DA Settlement", "price_fc": "DA Forecast (ML)"},
)
chart_price = _chart(
    df_price,
    "Day-Ahead Preis: Settlement vs Forecast",
    "Preis",
    series_col="series_price",
    mark="step",
)

# --- SOC chart ---
# Empfehlung: als LINE (SOC kann innerhalb der Stunde linear verlaufen).
# Wenn du strikt diskret willst, setze mark="step".
df_soc = _long(
    ts_plot_socdisp,
    ["soc_pct"],
    series_col="series_soc",
    label_map={"soc_pct": "SOC [%]"},
)
chart_soc = _chart(df_soc, "State of Charge", "SOC [%]", series_col="series_soc", mark="line")

# --- Dispatch chart (STEP) ---
df_dispatch = _long(
    ts_plot_socdisp,
    ["p_charge_plot", "p_discharge_plot"],
    series_col="series_dispatch",
    label_map={"p_charge_plot": "Laden (+)", "p_discharge_plot": "Entladen (−)"},
)
chart_dispatch = _chart(
    df_dispatch,
    "Batterie-Dispatch (Leistung)",
    "Leistung [kW]",
    series_col="series_dispatch",
    mark="step",
)

# =========================
# X-only zoom/pan
# =========================
zoom_x = alt.selection_interval(bind="scales", encodings=["x"])

combo = (
    alt.vconcat(chart_price, chart_soc, chart_dispatch)
    .resolve_scale(x="shared")
    .add_params(zoom_x)
)

st.altair_chart(combo, use_container_width=True)

# =========================
# Export
# =========================
st.markdown("---")
st.subheader("Export")
st.download_button(
    "Resultate herunterladen (CSV)",
    data=ts.to_csv(index=False).encode("utf-8"),
    file_name="bess_results_timeseries.csv",
    mime="text/csv",
)
