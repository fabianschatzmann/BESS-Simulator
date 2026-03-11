# core/optimizer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, Tuple
import numpy as np
import pandas as pd

try:
    import pulp
except ImportError as e:
    raise ImportError("PuLP ist nicht installiert. Installiere mit: pip install pulp") from e


# =============================================================================
# Common Dataclasses
# =============================================================================
@dataclass
class BatteryParams:
    e_nom_kwh: float
    p_ch_max_kw: float
    p_dis_max_kw: float
    eta_ch: float
    eta_dis: float
    soc_min: float        # fraction (0..1)
    soc_max: float        # fraction (0..1)
    soc0: float           # fraction (0..1)


@dataclass
class OptimizerSettings:
    timestep_h: float = 1.0
    forbid_simultaneous: bool = True
    cycle_penalty_chf_per_kwh: float = 0.0
    mip_gap: Optional[float] = 0.001
    time_limit_s: Optional[int] = None


# =============================================================================
# Day-Ahead MILP
# =============================================================================
def optimize_day_ahead_milp(
    ts: pd.Series,
    price_forecast: pd.Series,
    batt: BatteryParams,
    settings: Optional[OptimizerSettings] = None,
) -> pd.DataFrame:
    if settings is None:
        settings = OptimizerSettings()

    df = pd.DataFrame(
        {"ts": pd.to_datetime(ts, errors="coerce"), "price_fc": pd.to_numeric(price_forecast, errors="coerce")}
    )
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    df["price_fc"] = df["price_fc"].fillna(0.0)

    T = len(df)
    dt = float(settings.timestep_h)

    e_min = batt.soc_min * batt.e_nom_kwh
    e_max = batt.soc_max * batt.e_nom_kwh
    e0 = batt.soc0 * batt.e_nom_kwh

    prob = pulp.LpProblem("DA_BESS_Arbitrage", pulp.LpMaximize)

    p_ch = pulp.LpVariable.dicts("p_ch", range(T), lowBound=0, upBound=batt.p_ch_max_kw, cat="Continuous")
    p_dis = pulp.LpVariable.dicts("p_dis", range(T), lowBound=0, upBound=batt.p_dis_max_kw, cat="Continuous")
    soc = pulp.LpVariable.dicts("soc", range(T + 1), lowBound=e_min, upBound=e_max, cat="Continuous")

    if settings.forbid_simultaneous:
        z = pulp.LpVariable.dicts("z", range(T), lowBound=0, upBound=1, cat="Binary")
        for t in range(T):
            prob += p_ch[t] <= batt.p_ch_max_kw * z[t]
            prob += p_dis[t] <= batt.p_dis_max_kw * (1 - z[t])

    prob += soc[0] == e0

    for t in range(T):
        prob += soc[t + 1] == soc[t] + batt.eta_ch * p_ch[t] * dt - (1.0 / batt.eta_dis) * p_dis[t] * dt

    obj_terms = []
    for t in range(T):
        lam = float(df.loc[t, "price_fc"])
        obj_terms.append(lam * (p_dis[t] * dt - p_ch[t] * dt))
        if settings.cycle_penalty_chf_per_kwh > 0:
            obj_terms.append(-settings.cycle_penalty_chf_per_kwh * (p_ch[t] * dt + p_dis[t] * dt))
    prob += pulp.lpSum(obj_terms)

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        gapRel=settings.mip_gap if settings.mip_gap is not None else None,
        timeLimit=settings.time_limit_s if settings.time_limit_s is not None else None,
    )
    prob.solve(solver)

    out = pd.DataFrame(
        {
            "ts": df["ts"],
            "p_charge_kw": [pulp.value(p_ch[t]) for t in range(T)],
            "p_discharge_kw": [pulp.value(p_dis[t]) for t in range(T)],
            "soc_kwh": [pulp.value(soc[t]) for t in range(T)],
            "soc_kwh_end": [pulp.value(soc[t + 1]) for t in range(T)],
        }
    )
    out["p_net_kw"] = out["p_discharge_kw"] - out["p_charge_kw"]
    return out


# =============================================================================
# Intraday Continuous (Delta MILP, MPC-style)
# =============================================================================
def optimize_intraday_delta_milp(
    *,
    ts: pd.Series,
    p_da_base_kw: pd.Series,
    price_spread_fc: pd.Series,
    open_mask: pd.Series,
    batt: BatteryParams,
    settings: Optional[OptimizerSettings] = None,
    soc0_kwh: Optional[float] = None,
    # Optional soft terminal SOC (to avoid forced trades)
    terminal_soc_target_kwh: Optional[float] = None,
    terminal_soc_penalty_chf_per_kwh: float = 0.0,
    # ID must not be negative in expected value (forecast)
    enforce_non_negative_id_value: bool = True,
    # NEW: "wait for bigger spread" deadband (CHF/MWh). If |spread| <= deadband -> no trade that hour.
    spread_deadband_chf_per_mwh: float = 0.0,
    # NEW: strictly forbid trading against forecast sign per hour
    enforce_hourly_non_negative_id_value: bool = True,
) -> pd.DataFrame:
    """
    Delta optimization vs Day-Ahead baseline:
      p_total = p_da_base + dp

    Objective uses price_spread_fc, typically:
      spread = price_id_fc - price_da_fc   (Forecast vs Forecast, leak-free)

    Gate closure:
      open_mask[t] == False => dp_charge=dp_dis=0 for that step.

    NEW:
      - enforce_non_negative_id_value: Σ spread*dp_net >= 0 over horizon.
      - enforce_hourly_non_negative_id_value: forbids dp in the "wrong direction" per hour:
            if spread >  deadband -> dp_charge=0 (only discharge delta)
            if spread < -deadband -> dp_discharge=0 (only charge delta)
            if |spread|<=deadband -> dp_charge=dp_discharge=0 (wait)
        This is what you want when you say: "wenn Intraday negativ wäre, dann nicht handeln / warten".
    """
    if settings is None:
        settings = OptimizerSettings()

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, errors="coerce"),
            "p_base": pd.to_numeric(p_da_base_kw, errors="coerce").fillna(0.0),
            "spread_fc": pd.to_numeric(price_spread_fc, errors="coerce").fillna(0.0),
            "open": pd.Series(open_mask).astype(bool).values,
        }
    )
    df = df.dropna(subset=["ts"]).reset_index(drop=True)

    # baseline split into magnitudes
    p_base = df["p_base"].values.astype(float)
    p_base_dis = np.clip(p_base, 0.0, None)   # discharge magnitude
    p_base_ch = np.clip(-p_base, 0.0, None)   # charge magnitude

    T = len(df)
    dt = float(settings.timestep_h)

    e_min = batt.soc_min * batt.e_nom_kwh
    e_max = batt.soc_max * batt.e_nom_kwh
    if soc0_kwh is None:
        e0 = batt.soc0 * batt.e_nom_kwh
    else:
        e0 = float(np.clip(float(soc0_kwh), e_min, e_max))

    prob = pulp.LpProblem("ID_Delta_BESS", pulp.LpMaximize)

    dp_ch = pulp.LpVariable.dicts("dp_ch", range(T), lowBound=0, upBound=batt.p_ch_max_kw, cat="Continuous")
    dp_dis = pulp.LpVariable.dicts("dp_dis", range(T), lowBound=0, upBound=batt.p_dis_max_kw, cat="Continuous")
    soc = pulp.LpVariable.dicts("soc", range(T + 1), lowBound=e_min, upBound=e_max, cat="Continuous")

    deadband = float(max(0.0, spread_deadband_chf_per_mwh))

    # Total power limits and gate closure + NEW per-hour sign/deadband constraints
    for t in range(T):
        prob += (p_base_ch[t] + dp_ch[t]) <= batt.p_ch_max_kw
        prob += (p_base_dis[t] + dp_dis[t]) <= batt.p_dis_max_kw

        if not bool(df.loc[t, "open"]):
            prob += dp_ch[t] == 0
            prob += dp_dis[t] == 0
            continue

        if enforce_hourly_non_negative_id_value:
            spread = float(df.loc[t, "spread_fc"])
            # wait zone
            if abs(spread) <= deadband:
                prob += dp_ch[t] == 0
                prob += dp_dis[t] == 0
            elif spread > deadband:
                # forecast says ID > DA => only discharge delta makes sense
                prob += dp_ch[t] == 0
            else:  # spread < -deadband
                # forecast says ID < DA => only charge delta makes sense
                prob += dp_dis[t] == 0

    # forbid simult on total (approx)
    if settings.forbid_simultaneous:
        z = pulp.LpVariable.dicts("z", range(T), lowBound=0, upBound=1, cat="Binary")
        for t in range(T):
            prob += (p_base_ch[t] + dp_ch[t]) <= batt.p_ch_max_kw * z[t]
            prob += (p_base_dis[t] + dp_dis[t]) <= batt.p_dis_max_kw * (1 - z[t])

    prob += soc[0] == e0

    for t in range(T):
        pch_tot = p_base_ch[t] + dp_ch[t]
        pdis_tot = p_base_dis[t] + dp_dis[t]
        prob += soc[t + 1] == soc[t] + batt.eta_ch * pch_tot * dt - (1.0 / batt.eta_dis) * pdis_tot * dt

    # Objective: maximize spread * delta_net
    obj = []
    id_value_terms = []
    for t in range(T):
        spread = float(df.loc[t, "spread_fc"])
        dp_net_expr = (dp_dis[t] - dp_ch[t])
        term = spread * dp_net_expr * dt
        obj.append(term)
        id_value_terms.append(term)

        if settings.cycle_penalty_chf_per_kwh > 0:
            obj.append(-settings.cycle_penalty_chf_per_kwh * (dp_ch[t] * dt + dp_dis[t] * dt))

    # enforce non-negative ID expected value over horizon (kept)
    if enforce_non_negative_id_value:
        prob += pulp.lpSum(id_value_terms) >= 0.0

    # Soft terminal SOC penalty
    soc_end_dev_abs_var = None
    if terminal_soc_target_kwh is not None and float(terminal_soc_penalty_chf_per_kwh) > 0:
        tgt = float(np.clip(float(terminal_soc_target_kwh), e_min, e_max))
        dev_pos = pulp.LpVariable("soc_dev_pos_kwh", lowBound=0, cat="Continuous")
        dev_neg = pulp.LpVariable("soc_dev_neg_kwh", lowBound=0, cat="Continuous")
        prob += (soc[T] - tgt) == (dev_pos - dev_neg)
        soc_end_dev_abs_var = dev_pos + dev_neg
        obj.append(-float(terminal_soc_penalty_chf_per_kwh) * soc_end_dev_abs_var)

    prob += pulp.lpSum(obj)

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        gapRel=settings.mip_gap if settings.mip_gap is not None else None,
        timeLimit=settings.time_limit_s if settings.time_limit_s is not None else None,
    )
    prob.solve(solver)

    out = pd.DataFrame(
        {
            "ts": df["ts"],
            "dp_charge_kw": [pulp.value(dp_ch[t]) for t in range(T)],
            "dp_discharge_kw": [pulp.value(dp_dis[t]) for t in range(T)],
            "soc_kwh": [pulp.value(soc[t]) for t in range(T)],
            "soc_kwh_end": [pulp.value(soc[t + 1]) for t in range(T)],
            "open": df["open"].astype(bool).values,
            "spread_fc": df["spread_fc"].astype(float).values,
        }
    )

    out["dp_charge_kw"] = pd.to_numeric(out["dp_charge_kw"], errors="coerce").fillna(0.0)
    out["dp_discharge_kw"] = pd.to_numeric(out["dp_discharge_kw"], errors="coerce").fillna(0.0)
    out["dp_net_kw"] = out["dp_discharge_kw"] - out["dp_charge_kw"]
    out["id_value_fc_chf_h"] = out["spread_fc"] * out["dp_net_kw"] * float(dt)

    if soc_end_dev_abs_var is not None:
        out["soc_end_dev_abs_kwh"] = float(pulp.value(soc_end_dev_abs_var) or 0.0)
        out["terminal_soc_target_kwh"] = float(terminal_soc_target_kwh)
        out["terminal_soc_penalty_chf_per_kwh"] = float(terminal_soc_penalty_chf_per_kwh)
    else:
        out["soc_end_dev_abs_kwh"] = np.nan
        out["terminal_soc_target_kwh"] = np.nan
        out["terminal_soc_penalty_chf_per_kwh"] = np.nan

    return out


# =============================================================================
# SDL-only (PRL/SRL) – calendar-exact gate closures + economics floor
# =============================================================================
@dataclass
class EconomicsParams:
    capex_chf_per_kw_power: float = 0.0
    capex_chf_per_kwh_energy: float = 0.0
    fixed_om_chf_per_kw_year: float = 0.0
    asset_life_years: int = 15
    wacc: float = 0.06
    risk_premium_chf_per_mw_h: float = 0.0


@dataclass
class CurrencyParams:
    """
    Intern wird in CHF gerechnet.
    Falls Eingangszeitreihen in EUR sind, wird mit fx_eur_to_chf umgerechnet.
    """
    fx_eur_to_chf: float = 1.0


@dataclass
class SDLOptimizerSettings:
    accept_prob_target: float = 0.70
    window_days: int = 28
    tz: str = "Europe/Zurich"
    use_price: Literal["p_clear_true", "p_vwa_true"] = "p_clear_true"
    p_offer_mw: float = 1.0
    prl_close_time_local: str = "08:00"    # D-1
    srl_close_time_local: str = "14:30"    # D-1
    min_periods_frac: float = 0.25

    # Währung der Capacity-Preise in master
    capacity_price_currency: Literal["CHF", "EUR"] = "CHF"
    # Währung der Energy-Preise in master (falls echte Energy-Preis-Zeitreihen existieren)
    energy_price_currency: Literal["CHF", "EUR"] = "CHF"

    # NEU: Erwartungswert-Aktivierungsgrade (nur SRL/aFRR)
    # Interpretation pro Stunde: E_exp = alpha * p_offer_mw * 1h
    alpha_srl_up: float = 0.06
    alpha_srl_down: float = 0.04


def _crf(wacc: float, n_years: int) -> float:
    w = float(wacc)
    n = int(max(1, n_years))
    if w <= 0:
        return 1.0 / n
    return (w * (1 + w) ** n) / ((1 + w) ** n - 1)


def _cost_floor_chf_per_mw_h(
    batt: BatteryParams,
    eco: EconomicsParams,
    p_offer_mw: float,
    offered_hours_per_year: int = 8760,
) -> float:
    p_mw = float(max(0.0, p_offer_mw))
    if p_mw <= 0:
        return 0.0

    crf = _crf(eco.wacc, eco.asset_life_years)
    capex_power_total = float(eco.capex_chf_per_kw_power) * (p_mw * 1000.0)
    capex_energy_total = float(eco.capex_chf_per_kwh_energy) * float(batt.e_nom_kwh)

    annualized_capex = (capex_power_total + capex_energy_total) * crf
    annual_om = float(eco.fixed_om_chf_per_kw_year) * (p_mw * 1000.0)
    annual_total = annualized_capex + annual_om

    return float(annual_total / (p_mw * float(offered_hours_per_year)))


def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    s = str(hhmm).strip()
    if not s or ":" not in s:
        raise ValueError(f"Invalid time string '{hhmm}'. Expected 'HH:MM'.")
    hh = int(s.split(":")[0])
    mm = int(s.split(":")[1])
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        raise ValueError(f"Invalid time '{hhmm}'.")
    return hh, mm


def _ensure_local_ts(master: pd.DataFrame, tz: str) -> pd.Series:
    if "ts" in master.columns:
        ts = pd.to_datetime(master["ts"], errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            ts_local = ts.dt.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
        else:
            ts_local = ts.dt.tz_convert(tz)
        return ts_local

    if "ts_key" not in master.columns:
        raise ValueError("Master braucht mindestens 'ts' oder 'ts_key'.")
    ts_key = pd.to_datetime(master["ts_key"], errors="coerce")
    return ts_key.dt.tz_localize("UTC").dt.tz_convert(tz)


def _gate_closure_cutoff_ts_key(delivery_date_local: pd.Timestamp, close_h: int, close_m: int, tz: str) -> pd.Timestamp:
    d = pd.Timestamp(delivery_date_local).tz_localize(None).normalize()
    cutoff_local_naive = pd.Timestamp(f"{(d - pd.Timedelta(days=1)).date()} {close_h:02d}:{close_m:02d}")
    cutoff_local = cutoff_local_naive.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
    return cutoff_local.tz_convert("UTC").tz_localize(None)


def _rolling_quantile_at_cutoff(
    s: pd.Series,
    cutoff: pd.Timestamp,
    window_h: int,
    q: float,
    min_periods: int,
) -> float:
    hist = s.loc[:cutoff]
    if hist.empty:
        return np.nan
    tail = hist.tail(window_h)
    if tail.notna().sum() < min_periods:
        return np.nan
    return float(tail.quantile(q))


def _find_price_col_case_insensitive(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """
    Return first column in df.columns matching any candidate (case-insensitive).
    """
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    lower_map = {str(c).lower(): str(c) for c in cols}

    for cand in candidates:
        c = str(cand)
        if c in df.columns:
            return c
        lc = c.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def _candidate_price_cols(product_key: str, price_key: str) -> list[str]:
    """
    Build robust candidate column names for SDL clearing series.
    We DO NOT change data values; only try alternative names to find the same series.
    """
    base = f"sdl_{price_key}_{product_key}_chf_per_mw_h"

    variants = [
        base,
        f"{price_key}_{product_key}_chf_per_mw_h",
        f"sdl_{price_key}_{product_key}",
        f"{price_key}_{product_key}",
        f"sdl_{price_key}_{product_key}_price",
        f"sdl_{price_key}_{product_key}_mw_h",
    ]

    prod_aliases = {
        "prl_sym": ["prl_sym", "prl", "prl_symmetry", "prl_symm"],
        "srl_up": ["srl_up", "srlu", "srl_upward", "srl_up_reg"],
        "srl_down": ["srl_down", "srld", "srl_downward", "srl_down_reg"],
    }
    aliases = prod_aliases.get(product_key, [product_key])

    more = []
    for a in aliases:
        more.extend([
            f"sdl_{price_key}_{a}_chf_per_mw_h",
            f"{price_key}_{a}_chf_per_mw_h",
            f"sdl_{price_key}_{a}",
            f"{price_key}_{a}",
        ])

    seen = set()
    out = []
    for x in variants + more:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _candidate_energy_cols(product_key: str) -> Dict[str, list[str]]:
    """
    Optional: Energy/Activation Settlement (nur SRL).
    Erwartung: stündlich und via ts_key aligned.
    - Aktivierte Energiemenge [MWh]:    sdl_e_act_true_<prod>_mwh
    - Energiepreis [CHF/MWh oder EUR/MWh]: sdl_e_price_true_<prod>_chf_per_mwh / ..._eur_per_mwh
    """
    base_act = [
        f"sdl_e_act_true_{product_key}_mwh",
        f"e_act_true_{product_key}_mwh",
        f"sdl_e_{product_key}_mwh",
        f"e_{product_key}_mwh",
    ]
    base_price = [
        f"sdl_e_price_true_{product_key}_chf_per_mwh",
        f"sdl_e_price_true_{product_key}_eur_per_mwh",
        f"e_price_true_{product_key}_chf_per_mwh",
        f"e_price_true_{product_key}_eur_per_mwh",
        f"sdl_e_price_{product_key}_chf_per_mwh",
        f"sdl_e_price_{product_key}_eur_per_mwh",
    ]
    return {"act": base_act, "price": base_price}


def _to_chf(series: pd.Series, currency: str, fx_eur_to_chf: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if str(currency).upper() == "EUR":
        return s * float(fx_eur_to_chf)
    return s


def optimize_sdl_only(
    master: pd.DataFrame,
    batt: BatteryParams,
    eco: EconomicsParams,
    settings: SDLOptimizerSettings,
    *,
    scenario_market_mode: Optional[str] = None,
    currency: Optional[CurrencyParams] = None,
) -> Dict[str, Any]:
    if scenario_market_mode is not None and scenario_market_mode != "SDL_ONLY":
        raise ValueError(f"optimize_sdl_only called with scenario_market_mode='{scenario_market_mode}' (must be SDL_ONLY).")

    if currency is None:
        currency = CurrencyParams()

    df = master.copy()
    if "ts_key" not in df.columns:
        raise ValueError("Master muss 'ts_key' enthalten (UTC-naiv).")
    df["ts_key"] = pd.to_datetime(df["ts_key"], errors="coerce")
    df = df.dropna(subset=["ts_key"]).sort_values("ts_key").reset_index(drop=True)

    tz = settings.tz
    df["_ts_local"] = _ensure_local_ts(df, tz=tz)
    df["_delivery_date_local"] = df["_ts_local"].dt.floor("D")

    p_target = float(np.clip(settings.accept_prob_target, 1e-6, 1.0 - 1e-6))
    q_bid = float(1.0 - p_target)

    window_h = int(max(1, settings.window_days * 24))
    min_periods = int(max(24, np.floor(settings.min_periods_frac * window_h)))

    cost_floor = _cost_floor_chf_per_mw_h(batt=batt, eco=eco, p_offer_mw=settings.p_offer_mw)
    cost_floor_with_risk = float(cost_floor + eco.risk_premium_chf_per_mw_h)

    price_key = settings.use_price

    prl_h, prl_m = _parse_hhmm(settings.prl_close_time_local)
    srl_h, srl_m = _parse_hhmm(settings.srl_close_time_local)

    out = pd.DataFrame(
        {
            "ts_key": df["ts_key"],
            "ts_local": df["_ts_local"],
            "delivery_date_local": df["_delivery_date_local"],
        }
    )

    # Always include 'ts' for UI compatibility (value = ts_key, no semantic change)
    out["ts"] = out["ts_key"]

    total_rev_cap = np.zeros(len(df), dtype=float)
    total_rev_energy = np.zeros(len(df), dtype=float)

    missing_price_cols: Dict[str, str] = {}
    used_price_cols: Dict[str, str] = {}

    missing_energy_cols: Dict[str, str] = {}
    used_energy_cols: Dict[str, Dict[str, str]] = {}

    for product_key in ["srl_up", "srl_down", "prl_sym"]:
        pref = product_key

        # --- Capacity price column (wie bisher; optional EUR->CHF)
        candidates = _candidate_price_cols(product_key=product_key, price_key=price_key)
        clear_col = _find_price_col_case_insensitive(df, candidates)

        # Always materialize these columns in out (even if NaN)
        out[f"{pref}_bid_chf_per_mw_h"] = np.nan
        out[f"{pref}_clear_cap_chf_per_mw_h"] = np.nan
        out[f"{pref}_accepted"] = 0
        out[f"{pref}_rev_cap_chf_h"] = 0.0
        out[f"{pref}_rev_energy_chf_h"] = 0.0
        out[f"{pref}_rev_total_chf_h"] = 0.0
        out[f"{pref}_cutoff_ts_key"] = pd.NaT

        if clear_col is None:
            missing_price_cols[pref] = f"missing (tried: {candidates[:4]}...)"
            continue

        used_price_cols[pref] = str(clear_col)

        s_clear_raw = pd.to_numeric(df[clear_col], errors="coerce")
        s_clear_cap_chf = _to_chf(s_clear_raw, settings.capacity_price_currency, currency.fx_eur_to_chf)
        out[f"{pref}_clear_cap_chf_per_mw_h"] = s_clear_cap_chf

        s_series = pd.Series(s_clear_cap_chf.values, index=df["ts_key"].values).sort_index()

        if product_key == "prl_sym":
            close_h, close_m = prl_h, prl_m
        else:
            close_h, close_m = srl_h, srl_m

        unique_days = pd.Index(df["_delivery_date_local"].dropna().unique()).sort_values()
        bid_by_day = {}
        cutoff_by_day = {}

        for dday in unique_days:
            cutoff_ts_key = _gate_closure_cutoff_ts_key(dday, close_h, close_m, tz=tz)
            cutoff_by_day[dday] = cutoff_ts_key

            bid_q = _rolling_quantile_at_cutoff(
                s=s_series,
                cutoff=cutoff_ts_key,
                window_h=window_h,
                q=q_bid,
                min_periods=min_periods,
            )

            if np.isnan(bid_q):
                bid_final = np.nan
            else:
                bid_final = float(max(bid_q, cost_floor_with_risk))
            bid_by_day[dday] = bid_final

        bid_hat = df["_delivery_date_local"].map(bid_by_day).astype(float)
        accepted = (~pd.isna(bid_hat)) & (~pd.isna(s_clear_cap_chf)) & (bid_hat <= s_clear_cap_chf)

        # Capacity revenue (wie bisher): CHF/h
        rev_cap = accepted.astype(float) * float(settings.p_offer_mw) * s_clear_cap_chf

        out[f"{pref}_bid_chf_per_mw_h"] = bid_hat
        out[f"{pref}_accepted"] = accepted.astype(int)
        out[f"{pref}_rev_cap_chf_h"] = rev_cap
        out[f"{pref}_cutoff_ts_key"] = df["_delivery_date_local"].map(cutoff_by_day).values

        total_rev_cap += np.nan_to_num(rev_cap.values, nan=0.0)

        # --- Energy settlement (nur SRL)
        rev_energy = pd.Series(0.0, index=df.index)

        if product_key in ("srl_up", "srl_down"):
            cand = _candidate_energy_cols(product_key=product_key)
            act_col = _find_price_col_case_insensitive(df, cand["act"])
            eprice_col = _find_price_col_case_insensitive(df, cand["price"])

            # Energiepreis: nur sinnvoll, wenn wir einen Preis haben (sonst 0)
            e_price_chf = None
            if eprice_col is not None:
                e_price_raw = pd.to_numeric(df[eprice_col], errors="coerce")
                e_price_chf = _to_chf(e_price_raw, settings.energy_price_currency, currency.fx_eur_to_chf).fillna(0.0)

            if act_col is not None and e_price_chf is not None:
                # Best Case: echte Aktivierungsenergie vorhanden
                used_energy_cols[pref] = {"act_col": str(act_col), "price_col": str(eprice_col)}
                e_act_mwh = pd.to_numeric(df[act_col], errors="coerce").fillna(0.0)
                rev_energy = accepted.astype(float) * e_act_mwh * e_price_chf
            else:
                # Fallback: Erwartungswert über alpha (du wirst nicht immer abgerufen)
                # E_exp = alpha * p_offer_mw * dt   [MWh pro Schritt]
                alpha = float(settings.alpha_srl_up if product_key == "srl_up" else settings.alpha_srl_down)
                alpha = float(np.clip(alpha, 0.0, 1.0))

                if e_price_chf is None:
                    # Ohne Energiepreis-Zeitreihe kann man keinen Energy-Umsatz ansetzen -> bleibt 0
                    missing_energy_cols[pref] = f"missing (act_col={act_col}, price_col={eprice_col})"
                else:
                    e_exp_mwh = alpha * float(settings.p_offer_mw) * 1.0  # dt=1h in SDL-Modus (stündlich)
                    rev_energy = accepted.astype(float) * float(e_exp_mwh) * e_price_chf
                    missing_energy_cols[pref] = f"fallback_alpha_used (alpha={alpha:.3f}, act_col={act_col}, price_col={eprice_col})"

        out[f"{pref}_rev_energy_chf_h"] = rev_energy
        out[f"{pref}_rev_total_chf_h"] = out[f"{pref}_rev_cap_chf_h"] + out[f"{pref}_rev_energy_chf_h"]
        total_rev_energy += np.nan_to_num(rev_energy.values, nan=0.0)

    out["sdl_total_rev_cap_chf_h"] = total_rev_cap
    out["sdl_total_rev_energy_chf_h"] = total_rev_energy
    out["sdl_total_rev_chf_h"] = total_rev_cap + total_rev_energy

    out["sdl_accept_prob_target"] = p_target
    out["sdl_bid_quantile_used"] = q_bid
    out["sdl_window_days"] = int(settings.window_days)
    out["sdl_p_offer_mw"] = float(settings.p_offer_mw)
    out["sdl_use_price"] = str(settings.use_price)

    out["sdl_capacity_price_currency"] = str(settings.capacity_price_currency)
    out["sdl_energy_price_currency"] = str(settings.energy_price_currency)
    out["sdl_fx_eur_to_chf"] = float(currency.fx_eur_to_chf)

    out["sdl_alpha_srl_up"] = float(settings.alpha_srl_up)
    out["sdl_alpha_srl_down"] = float(settings.alpha_srl_down)

    out["sdl_cost_floor_chf_per_mw_h"] = float(cost_floor)
    out["sdl_cost_floor_plus_risk_chf_per_mw_h"] = float(cost_floor_with_risk)
    out["sdl_prl_close_local"] = settings.prl_close_time_local
    out["sdl_srl_close_local"] = settings.srl_close_time_local

    out["sdl_used_price_cols"] = str(used_price_cols)
    out["sdl_missing_price_cols"] = str(missing_price_cols)
    out["sdl_used_energy_cols"] = str(used_energy_cols)
    out["sdl_missing_energy_cols"] = str(missing_energy_cols)

    kpis: Dict[str, Any] = {
        "sdl_total_revenue_chf": float(np.nansum(out["sdl_total_rev_chf_h"].values)),
        "sdl_total_revenue_cap_chf": float(np.nansum(out["sdl_total_rev_cap_chf_h"].values)),
        "sdl_total_revenue_energy_chf": float(np.nansum(out["sdl_total_rev_energy_chf_h"].values)),
        "sdl_mean_revenue_chf_per_h": float(np.nanmean(out["sdl_total_rev_chf_h"].values)),
        "sdl_accept_prob_target": float(p_target),
        "sdl_bid_quantile_used": float(q_bid),
        "sdl_window_days": int(settings.window_days),
        "sdl_p_offer_mw": float(settings.p_offer_mw),
        "sdl_use_price": str(settings.use_price),
        "sdl_capacity_price_currency": str(settings.capacity_price_currency),
        "sdl_energy_price_currency": str(settings.energy_price_currency),
        "sdl_fx_eur_to_chf": float(currency.fx_eur_to_chf),
        "sdl_alpha_srl_up": float(settings.alpha_srl_up),
        "sdl_alpha_srl_down": float(settings.alpha_srl_down),
        "sdl_cost_floor_chf_per_mw_h": float(cost_floor),
        "sdl_cost_floor_plus_risk_chf_per_mw_h": float(cost_floor_with_risk),
        "sdl_prl_close_local": settings.prl_close_time_local,
        "sdl_srl_close_local": settings.srl_close_time_local,
        "sdl_used_price_cols": used_price_cols,
        "sdl_missing_price_cols": missing_price_cols,
        "sdl_used_energy_cols": used_energy_cols,
        "sdl_missing_energy_cols": missing_energy_cols,
    }

    for pref in ["srl_up", "srl_down", "prl_sym"]:
        kpis[f"{pref}_accept_rate"] = float(np.nanmean(out[f"{pref}_accepted"].values)) if f"{pref}_accepted" in out.columns else np.nan
        kpis[f"{pref}_revenue_cap_chf"] = float(np.nansum(out[f"{pref}_rev_cap_chf_h"].values)) if f"{pref}_rev_cap_chf_h" in out.columns else 0.0
        kpis[f"{pref}_revenue_energy_chf"] = float(np.nansum(out[f"{pref}_rev_energy_chf_h"].values)) if f"{pref}_rev_energy_chf_h" in out.columns else 0.0
        kpis[f"{pref}_revenue_total_chf"] = float(np.nansum(out[f"{pref}_rev_total_chf_h"].values)) if f"{pref}_rev_total_chf_h" in out.columns else 0.0

    revenue_breakdown = [
        {"market": "SDL", "product": "SRL_UP_CAP", "revenue_chf": kpis.get("srl_up_revenue_cap_chf", 0.0)},
        {"market": "SDL", "product": "SRL_UP_ENERGY", "revenue_chf": kpis.get("srl_up_revenue_energy_chf", 0.0)},
        {"market": "SDL", "product": "SRL_DOWN_CAP", "revenue_chf": kpis.get("srl_down_revenue_cap_chf", 0.0)},
        {"market": "SDL", "product": "SRL_DOWN_ENERGY", "revenue_chf": kpis.get("srl_down_revenue_energy_chf", 0.0)},
        {"market": "SDL", "product": "PRL_SYM_CAP", "revenue_chf": kpis.get("prl_sym_revenue_cap_chf", 0.0)},
        {"market": "SDL", "product": "TOTAL", "revenue_chf": kpis["sdl_total_revenue_chf"]},
    ]

    ts_sorted = out.sort_values("sdl_total_rev_chf_h", ascending=False)
    top_rows = ts_sorted.head(48)[["ts_key", "sdl_total_rev_chf_h"]].to_dict("records")
    worst_rows = ts_sorted.tail(48)[["ts_key", "sdl_total_rev_chf_h"]].to_dict("records")

    return {
        "kpis": kpis,
        "revenue_breakdown": revenue_breakdown,
        "top_days": top_rows,
        "worst_days": worst_rows,
        "timeseries": out,
    }