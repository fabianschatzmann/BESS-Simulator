# pages/4_Prognose_und_Dispatch.py
import os
import time
import streamlit as st
import pandas as pd
import numpy as np

from core.scenario_store import load_parquet, save_parquet, load_config
from core.forecasting import fit_once_predict_all, rolling_backtest_by_day, perfect_foresight_predict_all
from core.results import compute_results_from_dispatch

from core.optimizer import (
    BatteryParams,
    OptimizerSettings,
    EconomicsParams,
    SDLOptimizerSettings,
    optimize_day_ahead_milp,
    optimize_intraday_delta_milp,
    optimize_sdl_only,
)

from core.multiuse import MultiuseSettings, build_multiuse_priority_sdl


st.set_page_config(page_title="Prognose & Dispatch", layout="wide")
st.title("Prognose & Dispatch")
st.caption("Rechnet nur. Parameter kommen aus Szenario_Manager.py (Single Source of Truth).")


# ----------------------------
# Helpers
# ----------------------------
def _ensure_datetime(df: pd.DataFrame, col="ts") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    out = out.dropna(subset=[col]).sort_values(col)
    return out


def _to_utc_naive(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, errors="coerce")
    try:
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    return t


def _scenario_error(msg: str):
    st.error(msg)
    st.stop()


def _ensure_ts_key(m: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_datetime(m, "ts")
    out["ts_key"] = _to_utc_naive(out["ts"]).dt.floor("H")
    return out


def _load_master_best_effort(sname: str) -> tuple[pd.DataFrame, str]:
    if "master_2025" in st.session_state and isinstance(st.session_state["master_2025"], pd.DataFrame):
        return st.session_state["master_2025"], "session_state.master_2025 (Wizard)"
    if "master" in st.session_state and isinstance(st.session_state["master"], pd.DataFrame):
        return st.session_state["master"], "session_state.master"

    m = load_parquet(sname, "master")
    if isinstance(m, pd.DataFrame):
        return m, f"scenario_store: data/scenarios/{sname}/master.parquet"

    _scenario_error("Kein Master gefunden. Bitte zuerst im Datenimport den Master erstellen und speichern.")


def _get_df_from_state_or_disk(state_key: str, sname: str, disk_key: str) -> pd.DataFrame | None:
    obj = st.session_state.get(state_key)
    if isinstance(obj, pd.DataFrame):
        return obj
    obj2 = load_parquet(sname, disk_key)
    if isinstance(obj2, pd.DataFrame):
        return obj2
    return None


def _load_pf_settings(sname: str) -> dict:
    """
    Reads data/scenarios/<scenario>/pf_settings.parquet if exists.
    Returns defaults if missing.
    """
    df = load_parquet(sname, "pf_settings")
    if isinstance(df, pd.DataFrame) and not df.empty:
        row = df.iloc[0].to_dict()

        def _b(x) -> bool:
            if x is None:
                return False
            if isinstance(x, (bool, np.bool_)):
                return bool(x)
            try:
                return bool(int(x))
            except Exception:
                return str(x).strip().lower() in {"true", "yes", "y", "on"}

        return {
            "pf_da": _b(row.get("perfect_forecast_da", 0)),
            "pf_id": _b(row.get("perfect_forecast_id", 0)),
            "pf_horizon_da_h": int(row.get("pf_horizon_da_h", 24) or 24),
            "pf_horizon_id_h": int(row.get("pf_horizon_id_h", 24) or 24),
        }
    return {"pf_da": False, "pf_id": False, "pf_horizon_da_h": 24, "pf_horizon_id_h": 24}


def _scenario_dir(sname: str) -> str:
    return os.path.join("data", "scenarios", sname)


def _save_run_info(
    sname: str,
    *,
    pf_da: bool,
    pf_id: bool,
    pf_horizon_da_h: int,
    pf_horizon_id_h: int,
    market_mode: str,
    forecast_mode: str,
    model_name: str,
    notes: str | None = None,
):
    os.makedirs(_scenario_dir(sname), exist_ok=True)
    out_path = os.path.join(_scenario_dir(sname), "run_info.parquet")

    df = pd.DataFrame(
        [
            {
                "run_ts_utc": pd.Timestamp.utcnow().tz_localize(None),
                "mode_da": "perfect" if pf_da else "ml",
                "mode_id": "perfect" if pf_id else "ml",
                "pf_horizon_da_h": int(pf_horizon_da_h),
                "pf_horizon_id_h": int(pf_horizon_id_h),
                "market_mode": str(market_mode),
                "forecast_mode_cfg": str(forecast_mode),
                "model_name": str(model_name),
                "notes": notes,
            }
        ]
    )
    df.to_parquet(out_path, index=False)


def _show_forecast_mode_banner(pf_da: bool, pf_id: bool, pf_horizon_da_h: int, pf_horizon_id_h: int):
    c1, c2 = st.columns(2)
    if pf_da:
        c1.success("Day-Ahead Prognose: PERFECT (Upper Bound)")
    else:
        c1.warning("Day-Ahead Prognose: ML (leak-free, rolling)")

    if pf_id:
        c2.success("Intraday Prognose: PERFECT (Upper Bound)")
    else:
        c2.warning("Intraday Prognose: ML (leak-free, rolling)")

    st.caption(f"PF-Horizonte (nur gespeichert): DA={int(pf_horizon_da_h)} h | ID={int(pf_horizon_id_h)} h")


def _enrich_results_timeseries_for_multiuse(results_ts: pd.DataFrame, dispatch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ergänzt die DA/ID Results-Timeseries um Spalten, die Multiuse benötigt.
    Merge strikt nur über ts_key.
    """
    if not isinstance(results_ts, pd.DataFrame) or results_ts.empty:
        return pd.DataFrame()

    out = results_ts.copy()

    if "ts_key" not in out.columns:
        if "ts" in out.columns:
            out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
            out["ts_key"] = _to_utc_naive(out["ts"]).dt.floor("H")
        else:
            return out

    d = dispatch_df.copy()
    if "ts_key" not in d.columns:
        if "ts" in d.columns:
            d["ts"] = pd.to_datetime(d["ts"], errors="coerce")
            d["ts_key"] = _to_utc_naive(d["ts"]).dt.floor("H")
        else:
            return out

    keep_cols = [
        "ts_key",
        "price_da",
        "price_id",
        "price_da_fc",
        "price_id_fc",
        "p_da_kw",
        "p_id_delta_kw",
        "p_bess",
        "soc_kwh",
        "soc_pct",
        "soc",
        "rev_da_chf",
        "rev_id_inc_chf",
        "revenue",
        "revenue_chf",
    ]
    keep_cols = [c for c in keep_cols if c in d.columns]

    if not keep_cols:
        return out

    dd = d[keep_cols].drop_duplicates(subset=["ts_key"], keep="last")

    missing = [c for c in dd.columns if c != "ts_key" and c not in out.columns]
    if missing:
        out = out.merge(dd[["ts_key"] + missing], on="ts_key", how="left")

    # Falls Spalten schon existieren, aber leer sind, konservativ mit Dispatch-Werten auffüllen
    common = [c for c in dd.columns if c != "ts_key" and c in out.columns]
    if common:
        tmp = out[["ts_key"] + common].merge(dd[["ts_key"] + common], on="ts_key", how="left", suffixes=("", "__disp"))
        for c in common:
            out[c] = pd.to_numeric(tmp[c], errors="coerce").combine_first(pd.to_numeric(tmp[f"{c}__disp"], errors="coerce"))

    return out


# ----------------------------
# Progress helpers (UI)
# ----------------------------
def _init_progress():
    if "run_progress" not in st.session_state:
        st.session_state["run_progress"] = 0.0
    if "run_stage" not in st.session_state:
        st.session_state["run_stage"] = ""


def _set_progress(p: float, stage: str, bar, text_placeholder):
    p = float(np.clip(p, 0.0, 1.0))
    st.session_state["run_progress"] = p
    st.session_state["run_stage"] = stage
    bar.progress(p, text=f"{int(p*100)}% – {stage}")
    text_placeholder.caption(stage)


_init_progress()
progress_bar = st.progress(st.session_state["run_progress"], text="0% – bereit")
progress_text = st.empty()


# ----------------------------
# scenario & config
# ----------------------------
sname = st.session_state.get("scenario_name", "BaseCase_2025")
cfg = st.session_state.get("scenario_config")
if not isinstance(cfg, dict) or not cfg:
    cfg = load_config(sname) or {}
    st.session_state["scenario_config"] = cfg
if not cfg:
    _scenario_error("Keine Szenario-Konfiguration gefunden. Bitte zuerst im Szenario-Manager definieren.")

batt = cfg.get("battery", {})
eco = cfg.get("economics", {})
model = cfg.get("model", {})
dp = cfg.get("dispatch_policy", {})
mp = cfg.get("market_params", {})
sdl_cfg = cfg.get("sdl", {})
mu_cfg = cfg.get("multiuse", {})

market_mode = cfg.get("market_mode", "CUSTOM")

target_col_da = mp.get("forecast_target_col", "price_da")
target_col_id = mp.get("forecast_target_col_id", "price_id")

model_name = model.get("model_name", "ridge")
forecast_mode = model.get("forecast_mode", "rolling")

id_decision_hour_local = int(mp.get("id_decision_hour_local", 8))  # 08:00 local
id_reopt_every_hours = int(mp.get("id_reopt_every_hours", 24))  # kept (1x/day)

pf = _load_pf_settings(sname)
pf_da = pf["pf_da"]
pf_id = pf["pf_id"]
pf_horizon_da_h = pf["pf_horizon_da_h"]
pf_horizon_id_h = pf["pf_horizon_id_h"]

_show_forecast_mode_banner(pf_da, pf_id, pf_horizon_da_h, pf_horizon_id_h)

# ----------------------------
# Load master/features
# ----------------------------
master, master_src = _load_master_best_effort(sname)
st.session_state["master"] = master

features = st.session_state.get("features")
if features is None:
    features = load_parquet(sname, "features")
if features is not None:
    st.session_state["features"] = features

st.success(f"Master geladen ({master_src}) | rows={len(master):,} cols={len(master.columns):,}")
st.write(f"Szenario: `{sname}` | Marktmodus: **{market_mode}**")

# ----------------------------
# Buttons
# ----------------------------
st.subheader("Berechnung")
col1, col2, col3 = st.columns([1, 1, 1])

if market_mode == "SDL_ONLY":
    run_all = col1.button("🚀 Regelenergie (SDL) rechnen", type="primary")
    run_forecast = col2.button("Nur Prognose", disabled=True)
    run_dispatch = col3.button("Nur Dispatch", disabled=True)
elif market_mode == "DA_ONLY":
    run_all = col1.button("🚀 Prognose + Dispatch (DA)", type="primary")
    run_forecast = col2.button("Nur Prognose (DA)")
    run_dispatch = col3.button("Nur Dispatch (DA)")
elif market_mode == "DA_PLUS_ID":
    run_all = col1.button("🚀 DA + Intraday (FAST, gate closure, leak-free)", type="primary")
    run_forecast = col2.button("Nur Forecasts (DA+ID)")
    run_dispatch = col3.button("Nur Dispatch (DA+ID)")
elif market_mode == "DA_ID_SDL_MULTIUSE":
    run_all = col1.button("🚀 Alle Märkte (Multiuse: SDL priorisiert)", type="primary")
    run_forecast = col2.button("Nur Forecasts (DA+ID)")
    run_dispatch = col3.button("Nur Dispatch (DA+ID + SDL + Multiuse)")
else:
    run_all = col1.button("🚀 Prognose + Dispatch", type="primary")
    run_forecast = col2.button("Nur Prognose")
    run_dispatch = col3.button("Nur Dispatch")

start_time = time.time()

# =============================================================================
# SDL ONLY
# =============================================================================
if market_mode == "SDL_ONLY":
    if not run_all:
        st.info("SDL_ONLY: Starte die Berechnung mit dem Button oben.")
        st.stop()

    _set_progress(0.05, "Vorbereitung: Master (ts_key) normalisieren", progress_bar, progress_text)
    m = _ensure_ts_key(master)

    _set_progress(0.15, "Parameter: Batterie/Economics/SDL Settings", progress_bar, progress_text)
    batt_params = BatteryParams(
        e_nom_kwh=float(batt.get("e_nom_kwh", 1000.0)),
        p_ch_max_kw=float(batt.get("p_ch_max_kw", 500.0)),
        p_dis_max_kw=float(batt.get("p_dis_max_kw", 500.0)),
        eta_ch=float(batt.get("eta_ch", 0.95)),
        eta_dis=float(batt.get("eta_dis", 0.95)),
        soc0=float(batt.get("soc0", 0.5)),
        soc_min=float(batt.get("soc_min", 0.05)),
        soc_max=float(batt.get("soc_max", 0.95)),
    )

    eco_params = EconomicsParams(
        capex_chf_per_kw_power=float(eco.get("capex_chf_per_kw_power", 0.0)),
        capex_chf_per_kwh_energy=float(eco.get("capex_chf_per_kwh_energy", 0.0)),
        fixed_om_chf_per_kw_year=float(eco.get("fixed_om_chf_per_kw_year", 0.0)),
        asset_life_years=int(eco.get("asset_life_years", 15)),
        wacc=float(eco.get("wacc", 0.06)),
        risk_premium_chf_per_mw_h=float(eco.get("risk_premium_chf_per_mw_h", 0.0)),
    )

    gate = (sdl_cfg.get("gate_closure") or {})
    prl_time = str((gate.get("PRL") or {}).get("time", "08:00")).strip()
    srl_time = str((gate.get("SRL") or {}).get("time", "14:30")).strip()

    act = (sdl_cfg.get("activation") or {})
    alpha_up = float(act.get("alpha_srl_up", 0.06))
    alpha_down = float(act.get("alpha_srl_down", 0.04))

    sdl_settings = SDLOptimizerSettings(
        accept_prob_target=float(sdl_cfg.get("accept_prob_target", 0.70)),
        window_days=int(sdl_cfg.get("window_days", 28)),
        tz="Europe/Zurich",
        use_price=str(sdl_cfg.get("use_price", "p_clear_true")),

        # Legacy / fallback
        p_offer_mw=float(sdl_cfg.get("p_offer_mw", 1.0)),

        # NEU: produktspezifische Angebotsleistungen
        p_offer_prl_mw=float(sdl_cfg.get("p_offer_prl_mw", sdl_cfg.get("p_offer_mw", 1.0))),
        p_offer_srl_up_mw=float(sdl_cfg.get("p_offer_srl_up_mw", sdl_cfg.get("p_offer_mw", 1.0))),
        p_offer_srl_down_mw=float(sdl_cfg.get("p_offer_srl_down_mw", sdl_cfg.get("p_offer_mw", 1.0))),

        prl_close_time_local=prl_time,
        srl_close_time_local=srl_time,
        alpha_srl_up=float(alpha_up),
        alpha_srl_down=float(alpha_down),
        min_bid_chf_per_mw=float(sdl_cfg.get("min_bid_chf_per_mw", 0.0) or 0.0),

        # optionale technische SDL-Parameter
        reserve_duration_minutes=float(sdl_cfg.get("reserve_duration_minutes", 5.0) or 5.0),
        soc_buffer_kwh=float(sdl_cfg.get("soc_buffer_kwh", 0.0) or 0.0),
        partial_offer_allowed=bool(sdl_cfg.get("partial_offer_allowed", False)),
    )

    _set_progress(0.35, "SDL Optimierung: Bidding + Merit-Order", progress_bar, progress_text)
    results = optimize_sdl_only(
        master=m,
        batt=batt_params,
        eco=eco_params,
        settings=sdl_settings,
        scenario_market_mode="SDL_ONLY",
    )

    _set_progress(0.80, "Persistenz: dispatch/results_timeseries/sdl_timeseries", progress_bar, progress_text)
    ts_raw = results.get("timeseries", None)
    if isinstance(ts_raw, pd.DataFrame):
        ts_res = ts_raw.copy()
    else:
        ts_res = pd.DataFrame()

    runtime_s = time.time() - start_time

    st.session_state["dispatch"] = ts_res
    save_parquet(sname, "dispatch", ts_res)

    st.session_state["results"] = results
    save_parquet(sname, "results_timeseries", ts_res)

    save_parquet(sname, "sdl_timeseries", ts_res)

    _save_run_info(
        sname,
        pf_da=pf_da,
        pf_id=pf_id,
        pf_horizon_da_h=pf_horizon_da_h,
        pf_horizon_id_h=pf_horizon_id_h,
        market_mode=market_mode,
        forecast_mode=forecast_mode,
        model_name=model_name,
        notes="SDL_ONLY run",
    )

    _set_progress(1.00, "Fertig", progress_bar, progress_text)
    st.success(f"SDL_ONLY fertig. Runtime={runtime_s:.2f}s")
    st.stop()

# =============================================================================
# DA ONLY
# =============================================================================
if market_mode == "DA_ONLY":
    if run_forecast or run_all:
        if features is None:
            _scenario_error("Keine Features gefunden. Bitte Feature Engineering ausführen.")
        feats = _ensure_datetime(features, "ts")

        _set_progress(0.05, "Forecast Day-Ahead: Vorbereitung", progress_bar, progress_text)

        if pf_da:
            res_da = perfect_foresight_predict_all(features=feats, ts_col="ts", target_col=target_col_da)
        else:
            res_da = rolling_backtest_by_day(
                features=feats,
                ts_col="ts",
                target_col=target_col_da,
                model_name=model_name,
                model_params=None,
                train_days_min=int(model.get("train_days_min", 60)),
                retrain_every_days=int(model.get("retrain_every_days", 1)),
            )

        pred_da = res_da["pred_df"].copy()
        pred_da["ts"] = pd.to_datetime(pred_da["ts"], errors="coerce")
        pred_da["ts_key"] = _to_utc_naive(pred_da["ts"]).dt.floor("H")
        pred_da = pred_da.rename(columns={"y_pred": "price_da_fc", "y_true": "price_da_true"})
        pred_da["mode"] = "perfect_foresight" if pf_da else "ml"

        st.session_state["pred_prices_da"] = pred_da
        save_parquet(sname, "pred_prices_da", pred_da)

        _set_progress(0.35, "Forecast Day-Ahead gespeichert", progress_bar, progress_text)
        st.success("Forecast DA fertig: pred_prices_da gespeichert.")

    if not (run_dispatch or run_all):
        st.info("Für DA-Dispatch bitte 'Nur Dispatch (DA)' oder 'Prognose + Dispatch (DA)' drücken.")
        st.stop()

    _set_progress(0.45, "Dispatch DA: Inputs laden (master + pred_prices_da)", progress_bar, progress_text)
    pred_da = _get_df_from_state_or_disk("pred_prices_da", sname, "pred_prices_da")
    if pred_da is None:
        _scenario_error("DA_ONLY: pred_prices_da fehlt. Bitte zuerst den DA-Forecast rechnen.")

    m = _ensure_ts_key(master)
    if "price_da" not in m.columns:
        _scenario_error("Master benötigt 'price_da' für DA_ONLY.")

    m = m.merge(pred_da[["ts_key", "price_da_fc"]], on="ts_key", how="left")

    batt_params = BatteryParams(
        e_nom_kwh=float(batt.get("e_nom_kwh", 1000.0)),
        p_ch_max_kw=float(batt.get("p_ch_max_kw", 500.0)),
        p_dis_max_kw=float(batt.get("p_dis_max_kw", 500.0)),
        eta_ch=float(batt.get("eta_ch", 0.95)),
        eta_dis=float(batt.get("eta_dis", 0.95)),
        soc0=float(batt.get("soc0", 0.5)),
        soc_min=float(batt.get("soc_min", 0.05)),
        soc_max=float(batt.get("soc_max", 0.95)),
    )

    opt_settings = OptimizerSettings(
        timestep_h=1.0,
        forbid_simultaneous=True,
        cycle_penalty_chf_per_kwh=float(dp.get("cycle_penalty_chf_per_kwh", 0.0)),
        mip_gap=0.001,
        time_limit_s=None,
    )

    _set_progress(0.60, "Dispatch DA: Day-Ahead MILP", progress_bar, progress_text)
    sched_da = optimize_day_ahead_milp(
        ts=m["ts"],
        price_forecast=pd.to_numeric(m["price_da_fc"], errors="coerce").fillna(0.0),
        batt=batt_params,
        settings=opt_settings,
    )
    sched_da["ts"] = pd.to_datetime(sched_da["ts"], errors="coerce")
    sched_da["ts_key"] = _to_utc_naive(sched_da["ts"]).dt.floor("H")
    p_da_net = pd.to_numeric(sched_da["p_net_kw"], errors="coerce").fillna(0.0)

    _set_progress(0.80, "Dispatch DA: Timeseries + SOC + Revenues", progress_bar, progress_text)

    price_id_series = (
        pd.to_numeric(m["price_id"], errors="coerce")
        if "price_id" in m.columns
        else pd.Series(np.nan, index=m.index, dtype=float)
    )

    disp = pd.DataFrame(
        {
            "ts": pd.to_datetime(m["ts"], errors="coerce"),
            "ts_key": pd.to_datetime(m["ts_key"], errors="coerce").dt.floor("H"),
            "price_da": pd.to_numeric(m["price_da"], errors="coerce"),
            "price_id": price_id_series,
            "price_da_fc": pd.to_numeric(m["price_da_fc"], errors="coerce"),
            "price_id_fc": np.nan,
            "p_da_kw": p_da_net.values,
            "p_id_delta_kw": 0.0,
        }
    )

    disp["p_bess"] = disp["p_da_kw"]
    disp["p_discharge_kw"] = disp["p_bess"].clip(lower=0.0)
    disp["p_charge_kw"] = (-disp["p_bess"]).clip(lower=0.0)

    soc = [float(batt_params.soc0) * float(batt_params.e_nom_kwh)]
    for t in range(len(disp)):
        pch = float(disp.loc[t, "p_charge_kw"])
        pdis = float(disp.loc[t, "p_discharge_kw"])
        soc_next = soc[-1] + batt_params.eta_ch * pch - (1.0 / batt_params.eta_dis) * pdis
        soc_next = float(
            np.clip(
                soc_next,
                batt_params.soc_min * batt_params.e_nom_kwh,
                batt_params.soc_max * batt_params.e_nom_kwh,
            )
        )
        soc.append(soc_next)

    disp["soc_kwh"] = pd.Series(soc[:-1])
    disp["soc_pct"] = (disp["soc_kwh"] / float(batt_params.e_nom_kwh)) * 100.0
    disp["soc"] = disp["soc_pct"]

    dt_h = 1.0
    disp["rev_da_chf"] = disp["price_da"] * (disp["p_da_kw"] * dt_h / 1000.0)
    disp["spread_settle"] = disp["price_id"] - disp["price_da"]
    disp["rev_id_inc_chf"] = 0.0
    disp["revenue"] = disp["rev_da_chf"]
    disp["revenue_chf"] = disp["revenue"]

    runtime_s = time.time() - start_time
    st.session_state["dispatch"] = disp
    save_parquet(sname, "dispatch", disp)

    res = compute_results_from_dispatch(
        dispatch_df=disp,
        runtime_s=runtime_s,
        market_name="da_only",
        e_nom_kwh=float(batt.get("e_nom_kwh", 1000.0)),
    )
    st.session_state["results"] = res
    save_parquet(sname, "results_timeseries", res.get("timeseries", pd.DataFrame()))

    _save_run_info(
        sname,
        pf_da=pf_da,
        pf_id=pf_id,
        pf_horizon_da_h=pf_horizon_da_h,
        pf_horizon_id_h=pf_horizon_id_h,
        market_mode=market_mode,
        forecast_mode=forecast_mode,
        model_name=model_name,
        notes="DA_ONLY run",
    )

    _set_progress(1.00, "Fertig", progress_bar, progress_text)
    st.success(f"DA_ONLY fertig. Runtime={runtime_s:.2f}s")
    st.stop()

# =============================================================================
# DA + ID (FAST) + optional SDL + Multiuse
# =============================================================================
if market_mode in ("DA_PLUS_ID", "DA_ID_SDL_MULTIUSE"):
    if features is None:
        _scenario_error("Keine Features gefunden. Bitte Feature Engineering ausführen.")
    feats = _ensure_datetime(features, "ts")

    if run_forecast or run_all:
        _set_progress(0.05, "Forecasts: Vorbereitung", progress_bar, progress_text)

        _set_progress(0.10, "Forecast Day-Ahead", progress_bar, progress_text)
        if pf_da:
            res_da = perfect_foresight_predict_all(features=feats, ts_col="ts", target_col=target_col_da)
        else:
            res_da = rolling_backtest_by_day(
                features=feats,
                ts_col="ts",
                target_col=target_col_da,
                model_name=model_name,
                model_params=None,
                train_days_min=int(model.get("train_days_min", 60)),
                retrain_every_days=int(model.get("retrain_every_days", 1)),
            )

        pred_da = res_da["pred_df"].copy()
        pred_da["ts"] = pd.to_datetime(pred_da["ts"], errors="coerce")
        pred_da["ts_key"] = _to_utc_naive(pred_da["ts"]).dt.floor("H")
        pred_da = pred_da.rename(columns={"y_pred": "price_da_fc", "y_true": "price_da_true"})
        pred_da["mode"] = "perfect_foresight" if pf_da else "ml"
        st.session_state["pred_prices_da"] = pred_da
        save_parquet(sname, "pred_prices_da", pred_da)

        _set_progress(0.20, "Forecast Intraday", progress_bar, progress_text)
        if pf_id:
            res_id = perfect_foresight_predict_all(features=feats, ts_col="ts", target_col=target_col_id)
        else:
            res_id = rolling_backtest_by_day(
                features=feats,
                ts_col="ts",
                target_col=target_col_id,
                model_name=model_name,
                model_params=None,
                train_days_min=int(model.get("train_days_min", 60)),
                retrain_every_days=int(model.get("retrain_every_days", 1)),
            )

        pred_id = res_id["pred_df"].copy()
        pred_id["ts"] = pd.to_datetime(pred_id["ts"], errors="coerce")
        pred_id["ts_key"] = _to_utc_naive(pred_id["ts"]).dt.floor("H")
        pred_id = pred_id.rename(columns={"y_pred": "price_id_fc", "y_true": "price_id_true"})
        pred_id["mode"] = "perfect_foresight" if pf_id else "ml"
        st.session_state["pred_prices_id"] = pred_id
        save_parquet(sname, "pred_prices_id", pred_id)

        st.success("Forecasts fertig: pred_prices_da + pred_prices_id gespeichert (FAST).")

    if not (run_dispatch or run_all):
        st.info("Für Dispatch bitte 'Nur Dispatch' oder 'DA+Intraday/Alle Märkte' drücken.")
        st.stop()

    _set_progress(0.30, "Dispatch: Inputs laden (master + pred_prices)", progress_bar, progress_text)
    pred_da = _get_df_from_state_or_disk("pred_prices_da", sname, "pred_prices_da")
    pred_id = _get_df_from_state_or_disk("pred_prices_id", sname, "pred_prices_id")
    if pred_da is None or pred_id is None:
        _scenario_error("DA+ID: pred_prices_da oder pred_prices_id fehlt. Bitte zuerst Forecasts rechnen.")

    m = _ensure_ts_key(master)
    if "price_da" not in m.columns:
        _scenario_error("Master benötigt 'price_da'.")
    if "price_id" not in m.columns:
        _scenario_error("Master benötigt 'price_id'.")

    m = m.merge(pred_da[["ts_key", "price_da_fc"]], on="ts_key", how="left")
    m = m.merge(pred_id[["ts_key", "price_id_fc"]], on="ts_key", how="left")

    batt_params = BatteryParams(
        e_nom_kwh=float(batt.get("e_nom_kwh", 1000.0)),
        p_ch_max_kw=float(batt.get("p_ch_max_kw", 500.0)),
        p_dis_max_kw=float(batt.get("p_dis_max_kw", 500.0)),
        eta_ch=float(batt.get("eta_ch", 0.95)),
        eta_dis=float(batt.get("eta_dis", 0.95)),
        soc0=float(batt.get("soc0", 0.5)),
        soc_min=float(batt.get("soc_min", 0.05)),
        soc_max=float(batt.get("soc_max", 0.95)),
    )

    opt_settings = OptimizerSettings(
        timestep_h=1.0,
        forbid_simultaneous=True,
        cycle_penalty_chf_per_kwh=float(dp.get("cycle_penalty_chf_per_kwh", 0.0)),
        mip_gap=0.001,
        time_limit_s=None,
    )

    _set_progress(0.40, "Dispatch: Day-Ahead MILP", progress_bar, progress_text)
    sched_da = optimize_day_ahead_milp(
        ts=m["ts"],
        price_forecast=pd.to_numeric(m["price_da_fc"], errors="coerce").fillna(0.0),
        batt=batt_params,
        settings=opt_settings,
    )
    sched_da["ts"] = pd.to_datetime(sched_da["ts"], errors="coerce")
    sched_da["ts_key"] = _to_utc_naive(sched_da["ts"]).dt.floor("H")
    p_da_net = pd.to_numeric(sched_da["p_net_kw"], errors="coerce").fillna(0.0)

    _set_progress(0.55, "Dispatch: Intraday Delta (1 MILP pro Tag)", progress_bar, progress_text)

    ts_key = pd.to_datetime(m["ts_key"], errors="coerce").dt.floor("H")
    ts_local = pd.to_datetime(m["ts"], errors="coerce")
    day = ts_local.dt.floor("D")

    spread_fc_full = (
        pd.to_numeric(m["price_id_fc"], errors="coerce").fillna(0.0)
        - pd.to_numeric(m["price_da_fc"], errors="coerce").fillna(0.0)
    )

    dp_net = np.zeros(len(m), dtype=float)
    soc_now = float(batt_params.soc0) * float(batt_params.e_nom_kwh)

    unique_days = pd.Index(day.dropna().unique()).sort_values()

    n_days = max(1, len(unique_days))
    for i_day, d in enumerate(unique_days):
        frac = (i_day + 1) / n_days
        _set_progress(0.55 + 0.20 * frac, f"Intraday Delta: Tag {i_day+1}/{n_days}", progress_bar, progress_text)

        mask_day = day == d
        idx_day = np.where(mask_day.values)[0]
        if len(idx_day) == 0:
            continue

        t_dec = pd.Timestamp(d) + pd.Timedelta(hours=int(id_decision_hour_local))
        open_start = t_dec + pd.Timedelta(hours=1)

        ts_day = pd.to_datetime(m.loc[mask_day, "ts"], errors="coerce")
        open_mask_day = ts_day >= open_start

        if open_mask_day.sum() == 0:
            for k in idx_day:
                p_tot = float(p_da_net.iloc[k] + dp_net[k])
                pch = max(0.0, -p_tot)
                pdis = max(0.0, p_tot)
                soc_now = soc_now + batt_params.eta_ch * pch - (1.0 / batt_params.eta_dis) * pdis
                soc_now = float(
                    np.clip(
                        soc_now,
                        batt_params.soc_min * batt_params.e_nom_kwh,
                        batt_params.soc_max * batt_params.e_nom_kwh,
                    )
                )
            continue

        idx_open = idx_day[open_mask_day.values]
        if len(idx_open) == 0:
            continue

        open_mask = pd.Series([True] * len(idx_open))

        sched_id = optimize_intraday_delta_milp(
            ts=ts_key.iloc[idx_open],
            p_da_base_kw=p_da_net.iloc[idx_open],
            price_spread_fc=spread_fc_full.iloc[idx_open],
            open_mask=open_mask,
            batt=batt_params,
            settings=opt_settings,
            soc0_kwh=soc_now,
            enforce_non_negative_id_value=True,
        )

        dp_net[idx_open] = pd.to_numeric(sched_id["dp_net_kw"], errors="coerce").fillna(0.0).values

        for k in idx_day:
            p_tot = float(p_da_net.iloc[k] + dp_net[k])
            pch = max(0.0, -p_tot)
            pdis = max(0.0, p_tot)
            soc_now = soc_now + batt_params.eta_ch * pch - (1.0 / batt_params.eta_dis) * pdis
            soc_now = float(
                np.clip(
                    soc_now,
                    batt_params.soc_min * batt_params.e_nom_kwh,
                    batt_params.soc_max * batt_params.e_nom_kwh,
                )
            )

    _set_progress(0.80, "Dispatch: Timeseries + SOC + Revenues", progress_bar, progress_text)

    disp = pd.DataFrame(
        {
            "ts": pd.to_datetime(m["ts"], errors="coerce"),
            "ts_key": ts_key,
            "price_da": pd.to_numeric(m["price_da"], errors="coerce"),
            "price_id": pd.to_numeric(m["price_id"], errors="coerce"),
            "price_da_fc": pd.to_numeric(m["price_da_fc"], errors="coerce"),
            "price_id_fc": pd.to_numeric(m["price_id_fc"], errors="coerce"),
            "p_da_kw": p_da_net.values,
            "p_id_delta_kw": dp_net,
        }
    )

    disp["p_bess"] = disp["p_da_kw"] + disp["p_id_delta_kw"]
    disp["p_discharge_kw"] = disp["p_bess"].clip(lower=0.0)
    disp["p_charge_kw"] = (-disp["p_bess"]).clip(lower=0.0)

    soc = [float(batt_params.soc0) * float(batt_params.e_nom_kwh)]
    for t in range(len(disp)):
        pch = float(disp.loc[t, "p_charge_kw"])
        pdis = float(disp.loc[t, "p_discharge_kw"])
        soc_next = soc[-1] + batt_params.eta_ch * pch - (1.0 / batt_params.eta_dis) * pdis
        soc_next = float(
            np.clip(
                soc_next,
                batt_params.soc_min * batt_params.e_nom_kwh,
                batt_params.soc_max * batt_params.e_nom_kwh,
            )
        )
        soc.append(soc_next)

    disp["soc_kwh"] = pd.Series(soc[:-1])
    disp["soc_pct"] = (disp["soc_kwh"] / float(batt_params.e_nom_kwh)) * 100.0
    disp["soc"] = disp["soc_pct"]

    dt_h = 1.0
    disp["rev_da_chf"] = disp["price_da"] * (disp["p_da_kw"] * dt_h / 1000.0)
    disp["spread_settle"] = disp["price_id"] - disp["price_da"]
    disp["rev_id_inc_chf"] = disp["spread_settle"] * (disp["p_id_delta_kw"] * dt_h / 1000.0)
    disp["revenue"] = disp["rev_da_chf"] + disp["rev_id_inc_chf"]
    disp["revenue_chf"] = disp["revenue"]

    runtime_s = time.time() - start_time
    st.session_state["dispatch"] = disp
    save_parquet(sname, "dispatch", disp)

    res = compute_results_from_dispatch(
        dispatch_df=disp,
        runtime_s=runtime_s,
        market_name="da_plus_id",
        e_nom_kwh=float(batt.get("e_nom_kwh", 1000.0)),
    )
    st.session_state["results"] = res
    save_parquet(sname, "results_timeseries", res.get("timeseries", pd.DataFrame()))

    # -------------------------------------------------------------------------
    # OPTIONAL: Wenn alle Märkte aktiv -> SDL rechnen + Multiuse Decision Layer
    # -------------------------------------------------------------------------
    if market_mode == "DA_ID_SDL_MULTIUSE":
        _set_progress(0.82, "SDL zusätzlich rechnen (für Multiuse)", progress_bar, progress_text)

        m_sdl = _ensure_ts_key(master)

        eco_params = EconomicsParams(
            capex_chf_per_kw_power=float(eco.get("capex_chf_per_kw_power", 0.0)),
            capex_chf_per_kwh_energy=float(eco.get("capex_chf_per_kwh_energy", 0.0)),
            fixed_om_chf_per_kw_year=float(eco.get("fixed_om_chf_per_kw_year", 0.0)),
            asset_life_years=int(eco.get("asset_life_years", 15)),
            wacc=float(eco.get("wacc", 0.06)),
            risk_premium_chf_per_mw_h=float(eco.get("risk_premium_chf_per_mw_h", 0.0)),
        )

        gate = (sdl_cfg.get("gate_closure") or {})
        prl_time = str((gate.get("PRL") or {}).get("time", "08:00")).strip()
        srl_time = str((gate.get("SRL") or {}).get("time", "14:30")).strip()

        act = (sdl_cfg.get("activation") or {})
        alpha_up = float(act.get("alpha_srl_up", 0.06))
        alpha_down = float(act.get("alpha_srl_down", 0.04))

        sdl_settings = SDLOptimizerSettings(
            accept_prob_target=float(sdl_cfg.get("accept_prob_target", 0.70)),
            window_days=int(sdl_cfg.get("window_days", 28)),
            tz="Europe/Zurich",
            use_price=str(sdl_cfg.get("use_price", "p_clear_true")),

            # Legacy / fallback
            p_offer_mw=float(sdl_cfg.get("p_offer_mw", 1.0)),

            # NEU: produktspezifische Angebotsleistungen
            p_offer_prl_mw=float(sdl_cfg.get("p_offer_prl_mw", sdl_cfg.get("p_offer_mw", 1.0))),
            p_offer_srl_up_mw=float(sdl_cfg.get("p_offer_srl_up_mw", sdl_cfg.get("p_offer_mw", 1.0))),
            p_offer_srl_down_mw=float(sdl_cfg.get("p_offer_srl_down_mw", sdl_cfg.get("p_offer_mw", 1.0))),

            prl_close_time_local=prl_time,
            srl_close_time_local=srl_time,
            alpha_srl_up=float(alpha_up),
            alpha_srl_down=float(alpha_down),
            min_bid_chf_per_mw=float(sdl_cfg.get("min_bid_chf_per_mw", 0.0) or 0.0),

            # optionale technische SDL-Parameter
            reserve_duration_minutes=float(sdl_cfg.get("reserve_duration_minutes", 5.0) or 5.0),
            soc_buffer_kwh=float(sdl_cfg.get("soc_buffer_kwh", 0.0) or 0.0),
            partial_offer_allowed=bool(sdl_cfg.get("partial_offer_allowed", False)),
        )

        sdl_out = optimize_sdl_only(
            master=m_sdl,
            batt=batt_params,
            eco=eco_params,
            settings=sdl_settings,
            scenario_market_mode=None,
        )

        ts_raw = sdl_out.get("timeseries", None)
        if isinstance(ts_raw, pd.DataFrame):
            sdl_ts = ts_raw.copy()
        else:
            sdl_ts = pd.DataFrame()

        if sdl_ts.empty:
            st.warning("SDL Timeseries ist leer. Multiuse wird ohne SDL-Anteile erstellt (nur DA/ID).")
        else:
            save_parquet(sname, "sdl_timeseries", sdl_ts)

        _set_progress(0.92, "Multiuse Decision Layer bauen (Market-State pro Stunde)", progress_bar, progress_text)

        daid_ts = res.get("timeseries", None)
        if not isinstance(daid_ts, pd.DataFrame) or daid_ts.empty:
            st.warning("DA/ID Results-Timeseries ist leer. Multiuse kann nicht erstellt werden.")
        else:
            daid_ts = _enrich_results_timeseries_for_multiuse(daid_ts, disp)

            if "ts_key" not in daid_ts.columns:
                if "ts" in daid_ts.columns:
                    daid_ts = daid_ts.copy()
                    daid_ts["ts"] = pd.to_datetime(daid_ts["ts"], errors="coerce")
                    daid_ts["ts_key"] = _to_utc_naive(daid_ts["ts"]).dt.floor("H")
                else:
                    _scenario_error("DA/ID Results-Timeseries hat weder ts_key noch ts.")

            if not isinstance(sdl_ts, pd.DataFrame) or sdl_ts.empty:
                sdl_ts = pd.DataFrame({"ts_key": pd.to_datetime(daid_ts["ts_key"], errors="coerce").dt.floor("H")})
                sdl_ts["sdl_total_rev_chf_h"] = 0.0
                sdl_ts["delivery_date_local"] = pd.NaT

            mu_settings = MultiuseSettings(
                require_sdl_acceptance=bool(mu_cfg.get("require_sdl_acceptance", True)),
                use_dynamic_margin=bool(mu_cfg.get("use_dynamic_margin", True)),
                use_opportunity_cost=bool(mu_cfg.get("use_opportunity_cost", True)),
                lookahead_h=int(mu_cfg.get("lookahead_h", 6) or 6),
                weight_spread_vol=float(mu_cfg.get("weight_spread_vol", 1.0) or 1.0),
                weight_soc_edge=float(mu_cfg.get("weight_soc_edge", 8.0) or 8.0),
                weight_id_forecast_uncertainty=float(mu_cfg.get("weight_id_forecast_uncertainty", 0.5) or 0.5),
                weight_sdl_activation=float(mu_cfg.get("weight_sdl_activation", 1.0) or 1.0),
                weight_degradation=float(mu_cfg.get("weight_degradation", 1.0) or 1.0),
                soc_edge_band_pct=float(mu_cfg.get("soc_edge_band_pct", 15.0) or 15.0),
                degradation_chf_per_kwh_throughput=float(mu_cfg.get("degradation_chf_per_kwh_throughput", 0.0) or 0.0),
                lookahead_discount=float(mu_cfg.get("lookahead_discount", 0.90) or 0.90),
                perfect_forecast_upper_bound_mode=True,
                block_hours=4,
                tz_local="Europe/Zurich",

                # technische Realisierungslogik
                enforce_realized_soc=bool(mu_cfg.get("enforce_realized_soc", True)),
                reserve_duration_minutes=float(
                    mu_cfg.get(
                        "reserve_duration_minutes",
                        sdl_cfg.get("reserve_duration_minutes", 5.0),
                    ) or 5.0
                ),
                soc_buffer_kwh=float(
                    mu_cfg.get(
                        "soc_buffer_kwh",
                        sdl_cfg.get("soc_buffer_kwh", 0.0),
                    ) or 0.0
                ),
                allow_partial_sdl_offer=bool(
                    mu_cfg.get(
                        "allow_partial_sdl_offer",
                        sdl_cfg.get("partial_offer_allowed", False),
                    )
                ),
                fallback_eta_ch=float(batt.get("eta_ch", 0.95)),
                fallback_eta_dis=float(batt.get("eta_dis", 0.95)),
            )

            multiuse_df, multiuse_kpis = build_multiuse_priority_sdl(
                results_da_id=daid_ts,
                sdl_timeseries=sdl_ts,
                settings=mu_settings,
                batt={
                    "e_nom_kwh": float(batt.get("e_nom_kwh", 1000.0)),
                    "p_ch_max_kw": float(batt.get("p_ch_max_kw", 500.0)),
                    "p_dis_max_kw": float(batt.get("p_dis_max_kw", 500.0)),
                    "eta_ch": float(batt.get("eta_ch", 0.95)),
                    "eta_dis": float(batt.get("eta_dis", 0.95)),
                    "soc0": float(batt.get("soc0", 0.5)),
                    "soc_min": float(batt.get("soc_min", 0.05)),
                    "soc_max": float(batt.get("soc_max", 0.95)),
                },
            )

            save_parquet(sname, "multiuse_timeseries", multiuse_df)
            st.session_state["multiuse_timeseries"] = multiuse_df

            mu_kpis_df = pd.DataFrame([{k: v for k, v in (multiuse_kpis or {}).items()}])
            save_parquet(sname, "multiuse_kpis", mu_kpis_df)

            st.success(
                "Multiuse erstellt & gespeichert: "
                f"data/scenarios/{sname}/multiuse_timeseries.parquet (Market-State pro Stunde verfügbar)."
            )

    _save_run_info(
        sname,
        pf_da=pf_da,
        pf_id=pf_id,
        pf_horizon_da_h=pf_horizon_da_h,
        pf_horizon_id_h=pf_horizon_id_h,
        market_mode=market_mode,
        forecast_mode=forecast_mode,
        model_name=model_name,
        notes=(
            "DA_PLUS_ID run (FAST gate-closure)"
            if market_mode == "DA_PLUS_ID"
            else "DA_ID_SDL_MULTIUSE run (DA+ID + SDL + Multiuse)"
        ),
    )

    _set_progress(1.00, "Fertig", progress_bar, progress_text)
    st.success(f"{'DA+ID' if market_mode == 'DA_PLUS_ID' else 'Alle Märkte'} fertig. Runtime={runtime_s:.2f}s")
    st.stop()

st.warning("Unbekannter market_mode. Bitte im Szenario-Manager einen gültigen Marktmodus wählen.")