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
    terminal_soc_target_kwh: Optional[float] = None,
    terminal_soc_penalty_chf_per_kwh: float = 0.0,
    enforce_non_negative_id_value: bool = True,
    spread_deadband_chf_per_mwh: float = 0.0,
    enforce_hourly_non_negative_id_value: bool = True,
) -> pd.DataFrame:
    """
    Delta optimization vs Day-Ahead baseline:
      p_total = p_da_base + dp
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

    p_base = df["p_base"].values.astype(float)
    p_base_dis = np.clip(p_base, 0.0, None)
    p_base_ch = np.clip(-p_base, 0.0, None)

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

    for t in range(T):
        prob += (p_base_ch[t] + dp_ch[t]) <= batt.p_ch_max_kw
        prob += (p_base_dis[t] + dp_dis[t]) <= batt.p_dis_max_kw

        if not bool(df.loc[t, "open"]):
            prob += dp_ch[t] == 0
            prob += dp_dis[t] == 0
            continue

        if enforce_hourly_non_negative_id_value:
            spread = float(df.loc[t, "spread_fc"])
            if abs(spread) <= deadband:
                prob += dp_ch[t] == 0
                prob += dp_dis[t] == 0
            elif spread > deadband:
                prob += dp_ch[t] == 0
            else:
                prob += dp_dis[t] == 0

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

    if enforce_non_negative_id_value:
        prob += pulp.lpSum(id_value_terms) >= 0.0

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
# SDL-only (PRL/SRL) – calendar-exact gate closures + scenario-manager floor
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

    # Legacy / fallback
    p_offer_mw: float = 1.0

    # Neu: produktspezifische Angebotsleistungen
    p_offer_prl_mw: Optional[float] = None
    p_offer_srl_up_mw: Optional[float] = None
    p_offer_srl_down_mw: Optional[float] = None

    prl_close_time_local: str = "08:00"    # D-1
    srl_close_time_local: str = "14:30"    # D-1
    min_periods_frac: float = 0.25

    min_bid_chf_per_mw: float = 0.0

    capacity_price_currency: Literal["CHF", "EUR"] = "CHF"
    energy_price_currency: Literal["CHF", "EUR"] = "CHF"

    alpha_srl_up: float = 0.06
    alpha_srl_down: float = 0.04

    reserve_duration_minutes: float = 5.0
    soc_buffer_kwh: float = 0.0
    partial_offer_allowed: bool = False


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


def _resolve_soc_reference_series(
    df: pd.DataFrame,
    batt: BatteryParams,
    soc_ref_kwh: Optional[pd.Series] = None,
) -> pd.Series:
    e_min = float(batt.soc_min * batt.e_nom_kwh)
    e_max = float(batt.soc_max * batt.e_nom_kwh)
    default_soc = float(np.clip(batt.soc0 * batt.e_nom_kwh, e_min, e_max))

    if soc_ref_kwh is None:
        return pd.Series(default_soc, index=df.index, dtype=float)

    try:
        s = pd.Series(soc_ref_kwh).copy()
    except Exception:
        return pd.Series(default_soc, index=df.index, dtype=float)

    if len(s) == len(df):
        out = pd.to_numeric(s.reset_index(drop=True), errors="coerce")
        out = out.fillna(default_soc).clip(lower=e_min, upper=e_max)
        out.index = df.index
        return out.astype(float)

    try:
        idx_dt = pd.to_datetime(pd.Index(s.index), errors="coerce")
        tmp = pd.DataFrame(
            {
                "ts_key": idx_dt,
                "_soc_ref_kwh": pd.to_numeric(s.values, errors="coerce"),
            }
        ).dropna(subset=["ts_key"])
        if not tmp.empty:
            tmp["ts_key"] = pd.to_datetime(tmp["ts_key"], errors="coerce")
            tmp = tmp.drop_duplicates(subset=["ts_key"], keep="last").sort_values("ts_key")
            merged = df[["ts_key"]].merge(tmp, on="ts_key", how="left")["_soc_ref_kwh"]
            merged = merged.fillna(default_soc).clip(lower=e_min, upper=e_max)
            merged.index = df.index
            return merged.astype(float)
    except Exception:
        pass

    return pd.Series(default_soc, index=df.index, dtype=float)


def _feasible_sdl_offer_kw(
    *,
    soc_kwh: float,
    batt: BatteryParams,
    product_key: str,
    reserve_duration_h: float,
    soc_buffer_kwh: float = 0.0,
) -> float:
    tau_h = float(max(1e-9, reserve_duration_h))

    e_min = float(batt.soc_min * batt.e_nom_kwh) + float(max(0.0, soc_buffer_kwh))
    e_max = float(batt.soc_max * batt.e_nom_kwh) - float(max(0.0, soc_buffer_kwh))
    e = float(soc_kwh)

    e = float(np.clip(e, e_min, e_max)) if e_min <= e_max else float(
        np.clip(e, batt.soc_min * batt.e_nom_kwh, batt.soc_max * batt.e_nom_kwh)
    )

    p_up_feasible_kw = min(
        float(batt.p_dis_max_kw),
        max(0.0, (e - e_min) * float(batt.eta_dis) / tau_h),
    )

    p_down_feasible_kw = min(
        float(batt.p_ch_max_kw),
        max(0.0, (e_max - e) / max(float(batt.eta_ch), 1e-9) / tau_h),
    )

    if product_key == "prl_sym":
        return float(max(0.0, min(p_up_feasible_kw, p_down_feasible_kw)))
    if product_key == "srl_up":
        return float(max(0.0, p_up_feasible_kw))
    if product_key == "srl_down":
        return float(max(0.0, p_down_feasible_kw))
    return 0.0


def _resolve_sdl_product_offer_mw(settings: SDLOptimizerSettings, product_key: str) -> float:
    legacy = float(max(0.0, settings.p_offer_mw))

    if product_key == "prl_sym":
        value = settings.p_offer_prl_mw if settings.p_offer_prl_mw is not None else legacy
    elif product_key == "srl_up":
        value = settings.p_offer_srl_up_mw if settings.p_offer_srl_up_mw is not None else legacy
    elif product_key == "srl_down":
        value = settings.p_offer_srl_down_mw if settings.p_offer_srl_down_mw is not None else legacy
    else:
        value = legacy

    try:
        return float(max(0.0, float(value)))
    except Exception:
        return legacy


def _pick_best_realized_sdl_product_for_hour(
    row: pd.Series,
    batt: BatteryParams,
    settings: SDLOptimizerSettings,
    soc_kwh: float,
) -> Dict[str, Any]:
    label_map = {
        "prl_sym": "SDL: PRL (sym)",
        "srl_up": "SDL: SRL UP",
        "srl_down": "SDL: SRL DOWN",
    }

    reserve_duration_h = float(max(1e-9, settings.reserve_duration_minutes / 60.0))
    allow_partial = bool(settings.partial_offer_allowed)

    best = {
        "chosen_pref": None,
        "chosen_label": "NONE",
        "realized_offer_mw": 0.0,
        "rev_cap_realized_chf_h": 0.0,
        "rev_energy_realized_chf_h": 0.0,
        "rev_total_realized_chf_h": 0.0,
        "feasible_offer_mw": 0.0,
    }

    for pref in ["prl_sym", "srl_up", "srl_down"]:
        accepted = int(pd.to_numeric(row.get(f"{pref}_accepted", 0), errors="coerce") or 0) == 1
        if not accepted:
            continue

        feasible_offer_kw = _feasible_sdl_offer_kw(
            soc_kwh=float(soc_kwh),
            batt=batt,
            product_key=pref,
            reserve_duration_h=reserve_duration_h,
            soc_buffer_kwh=float(settings.soc_buffer_kwh),
        )
        feasible_offer_mw = float(max(0.0, feasible_offer_kw / 1000.0))

        nominal_offer_mw = _resolve_sdl_product_offer_mw(settings, pref)
        ref_effective_offer_mw = float(pd.to_numeric(row.get(f"{pref}_effective_offer_mw", np.nan), errors="coerce"))
        if not np.isfinite(ref_effective_offer_mw) or ref_effective_offer_mw <= 0.0:
            ref_effective_offer_mw = nominal_offer_mw

        if ref_effective_offer_mw <= 1e-9:
            realized_offer_mw = 0.0
        else:
            if allow_partial:
                realized_offer_mw = min(ref_effective_offer_mw, feasible_offer_mw)
            else:
                realized_offer_mw = ref_effective_offer_mw if feasible_offer_mw + 1e-9 >= ref_effective_offer_mw else 0.0

        ref_rev_cap = float(pd.to_numeric(row.get(f"{pref}_rev_cap_chf_h", 0.0), errors="coerce") or 0.0)
        ref_rev_energy = float(pd.to_numeric(row.get(f"{pref}_rev_energy_chf_h", 0.0), errors="coerce") or 0.0)
        ref_rev_total = float(pd.to_numeric(row.get(f"{pref}_rev_total_chf_h", ref_rev_cap + ref_rev_energy), errors="coerce") or 0.0)

        scale = realized_offer_mw / ref_effective_offer_mw if ref_effective_offer_mw > 1e-9 else 0.0
        rev_cap_realized = ref_rev_cap * scale
        rev_energy_realized = ref_rev_energy * scale
        rev_total_realized = ref_rev_total * scale

        if rev_total_realized > float(best["rev_total_realized_chf_h"]):
            best = {
                "chosen_pref": pref,
                "chosen_label": label_map[pref],
                "realized_offer_mw": float(realized_offer_mw),
                "rev_cap_realized_chf_h": float(rev_cap_realized),
                "rev_energy_realized_chf_h": float(rev_energy_realized),
                "rev_total_realized_chf_h": float(rev_total_realized),
                "feasible_offer_mw": float(feasible_offer_mw),
            }

    return best


def add_realized_sdl_only_dispatch(
    timeseries: pd.DataFrame,
    batt: BatteryParams,
    settings: SDLOptimizerSettings,
) -> pd.DataFrame:
    """
    Minimal sauberer realisierter SDL-only-Dispatch:

    - PRL: nur Reserve, kein Netto-Dispatch
    - SRL UP: Entlade-Dispatch mit alpha_srl_up
    - SRL DOWN: Lade-Dispatch mit alpha_srl_down
    - pro Stunde genau EIN SDL-Produkt
    - SOC wird stündlich fortgeschrieben
    - Leistungs- und SOC-Grenzen werden hart eingehalten
    - realisierte SDL-Erlöse werden konsistent auf genau ein Produkt pro Stunde begrenzt
    """
    out = timeseries.copy().reset_index(drop=True)

    e_min = float(batt.soc_min * batt.e_nom_kwh)
    e_max = float(batt.soc_max * batt.e_nom_kwh)
    dt = 1.0

    out["sdl_total_rev_cap_all_products_chf_h"] = pd.to_numeric(out.get("sdl_total_rev_cap_chf_h", 0.0), errors="coerce").fillna(0.0)
    out["sdl_total_rev_energy_all_products_chf_h"] = pd.to_numeric(out.get("sdl_total_rev_energy_chf_h", 0.0), errors="coerce").fillna(0.0)
    out["sdl_total_rev_all_products_chf_h"] = pd.to_numeric(out.get("sdl_total_rev_chf_h", 0.0), errors="coerce").fillna(0.0)

    for pref in ["prl_sym", "srl_up", "srl_down"]:
        out[f"{pref}_rev_cap_realized_chf_h"] = 0.0
        out[f"{pref}_rev_energy_realized_chf_h"] = 0.0
        out[f"{pref}_rev_total_realized_chf_h"] = 0.0
        out[f"{pref}_chosen"] = 0

    out["sdl_product_chosen"] = None
    out["sdl_product_label"] = "NONE"
    out["sdl_active_hour"] = 0

    out["sdl_realized_offer_mw"] = 0.0
    out["sdl_feasible_offer_mw_realized"] = 0.0

    out["sdl_p_charge_kw"] = 0.0
    out["sdl_p_discharge_kw"] = 0.0
    out["sdl_p_net_kw"] = 0.0

    out["sdl_reserved_charge_kw"] = 0.0
    out["sdl_reserved_discharge_kw"] = 0.0

    out["sdl_dispatch_curtailed_charge_kw"] = 0.0
    out["sdl_dispatch_curtailed_discharge_kw"] = 0.0

    out["market_state"] = "SDL_NONE"
    out["market_state_detail"] = "NONE"

    soc = np.zeros(len(out) + 1, dtype=float)

    soc0_candidates = [
        pd.to_numeric(pd.Series(out["sdl_soc_ref_kwh"]), errors="coerce").iloc[0] if "sdl_soc_ref_kwh" in out.columns and len(out) else np.nan,
        batt.soc0 * batt.e_nom_kwh,
    ]
    soc0 = next((float(x) for x in soc0_candidates if pd.notna(x)), batt.soc0 * batt.e_nom_kwh)
    soc[0] = float(np.clip(soc0, e_min, e_max))

    for t in range(len(out)):
        row = out.loc[t]

        picked = _pick_best_realized_sdl_product_for_hour(
            row=row,
            batt=batt,
            settings=settings,
            soc_kwh=float(soc[t]),
        )

        chosen_pref = picked["chosen_pref"]
        chosen_label = picked["chosen_label"]
        realized_offer_mw = float(picked["realized_offer_mw"])
        feasible_offer_mw = float(picked["feasible_offer_mw"])
        rev_cap_realized = float(picked["rev_cap_realized_chf_h"])
        rev_energy_realized = float(picked["rev_energy_realized_chf_h"])
        rev_total_realized = float(picked["rev_total_realized_chf_h"])

        p_ch_req_kw = 0.0
        p_dis_req_kw = 0.0
        reserved_ch_kw = 0.0
        reserved_dis_kw = 0.0

        if chosen_pref is not None and realized_offer_mw > 0.0 and rev_total_realized > 0.0:
            offer_kw = float(realized_offer_mw * 1000.0)

            if chosen_pref == "prl_sym":
                reserved_ch_kw = offer_kw
                reserved_dis_kw = offer_kw
                p_ch_req_kw = 0.0
                p_dis_req_kw = 0.0
            elif chosen_pref == "srl_up":
                reserved_ch_kw = 0.0
                reserved_dis_kw = offer_kw
                p_ch_req_kw = 0.0
                p_dis_req_kw = offer_kw * float(settings.alpha_srl_up)
            elif chosen_pref == "srl_down":
                reserved_ch_kw = offer_kw
                reserved_dis_kw = 0.0
                p_ch_req_kw = offer_kw * float(settings.alpha_srl_down)
                p_dis_req_kw = 0.0

        p_ch_req_kw = min(max(0.0, p_ch_req_kw), float(batt.p_ch_max_kw))
        p_dis_req_kw = min(max(0.0, p_dis_req_kw), float(batt.p_dis_max_kw))

        max_ch_soc_kw = max(
            0.0,
            (e_max - soc[t]) / max(float(batt.eta_ch), 1e-9) / dt
        )
        max_dis_soc_kw = max(
            0.0,
            (soc[t] - e_min) * float(batt.eta_dis) / dt
        )

        p_ch_act_kw = min(p_ch_req_kw, max_ch_soc_kw)
        p_dis_act_kw = min(p_dis_req_kw, max_dis_soc_kw)

        curtailed_ch_kw = max(0.0, p_ch_req_kw - p_ch_act_kw)
        curtailed_dis_kw = max(0.0, p_dis_req_kw - p_dis_act_kw)

        soc[t + 1] = soc[t] + float(batt.eta_ch) * p_ch_act_kw * dt - (1.0 / max(float(batt.eta_dis), 1e-9)) * p_dis_act_kw * dt
        soc[t + 1] = float(np.clip(soc[t + 1], e_min, e_max))

        if chosen_pref is not None and realized_offer_mw > 0.0 and rev_total_realized > 0.0:
            out.loc[t, f"{chosen_pref}_rev_cap_realized_chf_h"] = rev_cap_realized
            out.loc[t, f"{chosen_pref}_rev_energy_realized_chf_h"] = rev_energy_realized
            out.loc[t, f"{chosen_pref}_rev_total_realized_chf_h"] = rev_total_realized
            out.loc[t, f"{chosen_pref}_chosen"] = 1
            out.loc[t, "sdl_active_hour"] = 1
            out.loc[t, "market_state"] = "SDL_ONLY"
            out.loc[t, "market_state_detail"] = chosen_label
        else:
            out.loc[t, "sdl_active_hour"] = 0
            out.loc[t, "market_state"] = "SDL_NONE"
            out.loc[t, "market_state_detail"] = "NONE"

        out.loc[t, "sdl_product_chosen"] = chosen_pref
        out.loc[t, "sdl_product_label"] = chosen_label
        out.loc[t, "sdl_realized_offer_mw"] = realized_offer_mw
        out.loc[t, "sdl_feasible_offer_mw_realized"] = feasible_offer_mw

        out.loc[t, "sdl_p_charge_kw"] = p_ch_act_kw
        out.loc[t, "sdl_p_discharge_kw"] = p_dis_act_kw
        out.loc[t, "sdl_p_net_kw"] = p_dis_act_kw - p_ch_act_kw

        out.loc[t, "sdl_reserved_charge_kw"] = reserved_ch_kw
        out.loc[t, "sdl_reserved_discharge_kw"] = reserved_dis_kw

        out.loc[t, "sdl_dispatch_curtailed_charge_kw"] = curtailed_ch_kw
        out.loc[t, "sdl_dispatch_curtailed_discharge_kw"] = curtailed_dis_kw

    out["sdl_soc_kwh"] = soc[:-1]
    out["sdl_soc_kwh_end"] = soc[1:]
    out["sdl_soc_pct"] = (out["sdl_soc_kwh"] / max(1e-9, float(batt.e_nom_kwh))) * 100.0

    out["sdl_total_rev_cap_chf_h"] = (
        out["prl_sym_rev_cap_realized_chf_h"]
        + out["srl_up_rev_cap_realized_chf_h"]
        + out["srl_down_rev_cap_realized_chf_h"]
    ).astype(float)

    out["sdl_total_rev_energy_chf_h"] = (
        out["prl_sym_rev_energy_realized_chf_h"]
        + out["srl_up_rev_energy_realized_chf_h"]
        + out["srl_down_rev_energy_realized_chf_h"]
    ).astype(float)

    out["sdl_total_rev_chf_h"] = (
        out["prl_sym_rev_total_realized_chf_h"]
        + out["srl_up_rev_total_realized_chf_h"]
        + out["srl_down_rev_total_realized_chf_h"]
    ).astype(float)

    # Kompatibilitäts-Spalten fürs Dashboard
    out["p_charge_kw"] = out["sdl_p_charge_kw"]
    out["p_discharge_kw"] = out["sdl_p_discharge_kw"]
    out["p_net_kw"] = out["sdl_p_net_kw"]
    out["soc_kwh"] = out["sdl_soc_kwh"]
    out["soc_kwh_end"] = out["sdl_soc_kwh_end"]

    return out


def optimize_sdl_only(
    master: pd.DataFrame,
    batt: BatteryParams,
    eco: EconomicsParams,
    settings: SDLOptimizerSettings,
    *,
    scenario_market_mode: Optional[str] = None,
    currency: Optional[CurrencyParams] = None,
    soc_ref_kwh: Optional[pd.Series] = None,
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

    soc_ref_series = _resolve_soc_reference_series(df=df, batt=batt, soc_ref_kwh=soc_ref_kwh)

    p_target = float(np.clip(settings.accept_prob_target, 1e-6, 1.0 - 1e-6))
    q_bid = float(1.0 - p_target)

    window_h = int(max(1, settings.window_days * 24))
    min_periods = int(max(24, np.floor(settings.min_periods_frac * window_h)))

    cost_floor = _cost_floor_chf_per_mw_h(
        batt=batt,
        eco=eco,
        p_offer_mw=max(
            _resolve_sdl_product_offer_mw(settings, "prl_sym"),
            _resolve_sdl_product_offer_mw(settings, "srl_up"),
            _resolve_sdl_product_offer_mw(settings, "srl_down"),
        ),
    )
    cost_floor_with_risk = float(cost_floor + eco.risk_premium_chf_per_mw_h)

    scenario_min_bid = float(max(0.0, settings.min_bid_chf_per_mw))

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

    out["ts"] = out["ts_key"]
    out["sdl_soc_ref_kwh"] = soc_ref_series.astype(float).values
    out["sdl_soc_ref_pct"] = (soc_ref_series.astype(float).values / float(max(1e-9, batt.e_nom_kwh))) * 100.0
    out["sdl_reserve_duration_min"] = float(settings.reserve_duration_minutes)
    out["sdl_soc_buffer_kwh"] = float(settings.soc_buffer_kwh)
    out["sdl_partial_offer_allowed"] = bool(settings.partial_offer_allowed)

    out["sdl_p_offer_mw"] = float(settings.p_offer_mw)
    out["sdl_p_offer_prl_mw"] = float(_resolve_sdl_product_offer_mw(settings, "prl_sym"))
    out["sdl_p_offer_srl_up_mw"] = float(_resolve_sdl_product_offer_mw(settings, "srl_up"))
    out["sdl_p_offer_srl_down_mw"] = float(_resolve_sdl_product_offer_mw(settings, "srl_down"))

    total_rev_cap = np.zeros(len(df), dtype=float)
    total_rev_energy = np.zeros(len(df), dtype=float)

    missing_price_cols: Dict[str, str] = {}
    used_price_cols: Dict[str, str] = {}

    missing_energy_cols: Dict[str, str] = {}
    used_energy_cols: Dict[str, Dict[str, str]] = {}

    reserve_duration_h = float(max(1e-9, settings.reserve_duration_minutes / 60.0))

    for product_key in ["srl_up", "srl_down", "prl_sym"]:
        pref = product_key
        nominal_offer_mw = _resolve_sdl_product_offer_mw(settings, product_key)
        nominal_offer_kw = nominal_offer_mw * 1000.0

        candidates = _candidate_price_cols(product_key=product_key, price_key=price_key)
        clear_col = _find_price_col_case_insensitive(df, candidates)

        out[f"{pref}_nominal_offer_mw"] = float(nominal_offer_mw)
        out[f"{pref}_bid_chf_per_mw_h"] = np.nan
        out[f"{pref}_clear_cap_chf_per_mw_h"] = np.nan
        out[f"{pref}_settlement_cap_price_chf_per_mw_h"] = np.nan
        out[f"{pref}_settlement_mechanism"] = (
            "pay_as_cleared" if product_key == "prl_sym" else "pay_as_bid"
        )

        out[f"{pref}_accepted"] = 0
        out[f"{pref}_rev_cap_chf_h"] = 0.0
        out[f"{pref}_rev_energy_chf_h"] = 0.0
        out[f"{pref}_rev_total_chf_h"] = 0.0
        out[f"{pref}_cutoff_ts_key"] = pd.NaT

        out[f"{pref}_feasible_offer_mw"] = 0.0
        out[f"{pref}_effective_offer_mw"] = 0.0
        out[f"{pref}_tech_ok"] = 0

        feasible_offer_kw = soc_ref_series.apply(
            lambda x: _feasible_sdl_offer_kw(
                soc_kwh=float(x),
                batt=batt,
                product_key=product_key,
                reserve_duration_h=reserve_duration_h,
                soc_buffer_kwh=settings.soc_buffer_kwh,
            )
        )
        feasible_offer_mw = feasible_offer_kw / 1000.0
        out[f"{pref}_feasible_offer_mw"] = feasible_offer_mw.astype(float).values

        if settings.partial_offer_allowed:
            effective_offer_mw = np.minimum(nominal_offer_mw, feasible_offer_mw.astype(float).values)
            tech_ok = pd.Series(effective_offer_mw > 1e-9, index=df.index)
        else:
            tech_ok = feasible_offer_kw >= (nominal_offer_kw - 1e-9)
            effective_offer_mw = np.where(tech_ok.values, nominal_offer_mw, 0.0)

        effective_offer_mw_series = pd.to_numeric(pd.Series(effective_offer_mw), errors="coerce").fillna(0.0)

        out[f"{pref}_effective_offer_mw"] = effective_offer_mw_series.values
        out[f"{pref}_tech_ok"] = tech_ok.astype(int).values

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
                bid_final = float(max(bid_q, scenario_min_bid))

            bid_by_day[dday] = bid_final

        bid_hat = df["_delivery_date_local"].map(bid_by_day).astype(float)
        bid_hat_safe = bid_hat.fillna(0.0)

        price_accepted = (~pd.isna(bid_hat)) & (~pd.isna(s_clear_cap_chf)) & (bid_hat <= s_clear_cap_chf)
        accepted = price_accepted & tech_ok & soc_ref_series.notna()

        # Settlement:
        # - PRL = pay-as-cleared
        # - SRL = pay-as-bid
        if product_key == "prl_sym":
            settlement_cap_price = s_clear_cap_chf.fillna(0.0)
        else:
            settlement_cap_price = bid_hat_safe

        rev_cap = accepted.astype(float) * effective_offer_mw_series * settlement_cap_price

        out[f"{pref}_bid_chf_per_mw_h"] = bid_hat
        out[f"{pref}_settlement_cap_price_chf_per_mw_h"] = settlement_cap_price
        out[f"{pref}_accepted"] = accepted.astype(int)
        out[f"{pref}_rev_cap_chf_h"] = rev_cap
        out[f"{pref}_cutoff_ts_key"] = df["_delivery_date_local"].map(cutoff_by_day).values

        total_rev_cap += np.nan_to_num(rev_cap.values, nan=0.0)

        rev_energy = pd.Series(0.0, index=df.index)

        if product_key in ("srl_up", "srl_down"):
            cand = _candidate_energy_cols(product_key=product_key)
            act_col = _find_price_col_case_insensitive(df, cand["act"])
            eprice_col = _find_price_col_case_insensitive(df, cand["price"])

            e_price_chf = None
            if eprice_col is not None:
                e_price_raw = pd.to_numeric(df[eprice_col], errors="coerce")
                e_price_chf = _to_chf(e_price_raw, settings.energy_price_currency, currency.fx_eur_to_chf).fillna(0.0)

            if act_col is not None and e_price_chf is not None:
                used_energy_cols[pref] = {"act_col": str(act_col), "price_col": str(eprice_col)}
                e_act_mwh = pd.to_numeric(df[act_col], errors="coerce").fillna(0.0)
                rev_energy = accepted.astype(float) * e_act_mwh * e_price_chf
            else:
                alpha = float(settings.alpha_srl_up if product_key == "srl_up" else settings.alpha_srl_down)
                alpha = float(np.clip(alpha, 0.0, 1.0))

                if e_price_chf is None:
                    missing_energy_cols[pref] = f"missing (act_col={act_col}, price_col={eprice_col})"
                else:
                    e_exp_mwh = alpha * effective_offer_mw_series * 1.0
                    rev_energy = accepted.astype(float) * e_exp_mwh * e_price_chf
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
    out["sdl_use_price"] = str(settings.use_price)

    out["sdl_capacity_price_currency"] = str(settings.capacity_price_currency)
    out["sdl_energy_price_currency"] = str(settings.energy_price_currency)
    out["sdl_fx_eur_to_chf"] = float(currency.fx_eur_to_chf)

    out["sdl_alpha_srl_up"] = float(settings.alpha_srl_up)
    out["sdl_alpha_srl_down"] = float(settings.alpha_srl_down)

    out["sdl_cost_floor_chf_per_mw_h"] = float(cost_floor)
    out["sdl_cost_floor_plus_risk_chf_per_mw_h"] = float(cost_floor_with_risk)

    out["sdl_scenario_min_bid_chf_per_mw"] = float(scenario_min_bid)
    out["sdl_active_bid_floor_source"] = "scenario_manager"

    out["sdl_prl_close_local"] = settings.prl_close_time_local
    out["sdl_srl_close_local"] = settings.srl_close_time_local

    out["sdl_used_price_cols"] = str(used_price_cols)
    out["sdl_missing_price_cols"] = str(missing_price_cols)
    out["sdl_used_energy_cols"] = str(used_energy_cols)
    out["sdl_missing_energy_cols"] = str(missing_energy_cols)

    out = add_realized_sdl_only_dispatch(
        timeseries=out,
        batt=batt,
        settings=settings,
    )

    kpis: Dict[str, Any] = {
        "sdl_total_revenue_chf": float(np.nansum(out["sdl_total_rev_chf_h"].values)),
        "sdl_total_revenue_cap_chf": float(np.nansum(out["sdl_total_rev_cap_chf_h"].values)),
        "sdl_total_revenue_energy_chf": float(np.nansum(out["sdl_total_rev_energy_chf_h"].values)),
        "sdl_total_revenue_all_products_chf": float(np.nansum(out["sdl_total_rev_all_products_chf_h"].values)),
        "sdl_mean_revenue_chf_per_h": float(np.nanmean(out["sdl_total_rev_chf_h"].values)),
        "sdl_accept_prob_target": float(p_target),
        "sdl_bid_quantile_used": float(q_bid),
        "sdl_window_days": int(settings.window_days),
        "sdl_p_offer_mw": float(settings.p_offer_mw),
        "sdl_p_offer_prl_mw": float(_resolve_sdl_product_offer_mw(settings, "prl_sym")),
        "sdl_p_offer_srl_up_mw": float(_resolve_sdl_product_offer_mw(settings, "srl_up")),
        "sdl_p_offer_srl_down_mw": float(_resolve_sdl_product_offer_mw(settings, "srl_down")),
        "sdl_use_price": str(settings.use_price),
        "sdl_capacity_price_currency": str(settings.capacity_price_currency),
        "sdl_energy_price_currency": str(settings.energy_price_currency),
        "sdl_fx_eur_to_chf": float(currency.fx_eur_to_chf),
        "sdl_alpha_srl_up": float(settings.alpha_srl_up),
        "sdl_alpha_srl_down": float(settings.alpha_srl_down),
        "sdl_cost_floor_chf_per_mw_h": float(cost_floor),
        "sdl_cost_floor_plus_risk_chf_per_mw_h": float(cost_floor_with_risk),
        "sdl_scenario_min_bid_chf_per_mw": float(scenario_min_bid),
        "sdl_active_bid_floor_source": "scenario_manager",
        "sdl_prl_close_local": settings.prl_close_time_local,
        "sdl_srl_close_local": settings.srl_close_time_local,
        "sdl_used_price_cols": used_price_cols,
        "sdl_missing_price_cols": missing_price_cols,
        "sdl_used_energy_cols": used_energy_cols,
        "sdl_missing_energy_cols": missing_energy_cols,
        "sdl_reserve_duration_min": float(settings.reserve_duration_minutes),
        "sdl_soc_buffer_kwh": float(settings.soc_buffer_kwh),
        "sdl_partial_offer_allowed": bool(settings.partial_offer_allowed),
        "sdl_realized_total_charge_mwh": float(np.nansum(out["sdl_p_charge_kw"].to_numpy(dtype=float)) / 1000.0),
        "sdl_realized_total_discharge_mwh": float(np.nansum(out["sdl_p_discharge_kw"].to_numpy(dtype=float)) / 1000.0),
        "sdl_realized_avg_charge_kw": float(np.nanmean(out["sdl_p_charge_kw"].to_numpy(dtype=float))) if len(out) else np.nan,
        "sdl_realized_avg_discharge_kw": float(np.nanmean(out["sdl_p_discharge_kw"].to_numpy(dtype=float))) if len(out) else np.nan,
        "sdl_realized_active_hours": int(np.nansum(out["sdl_active_hour"].to_numpy(dtype=int))) if len(out) else 0,
    }

    for pref in ["srl_up", "srl_down", "prl_sym"]:
        kpis[f"{pref}_accept_rate"] = float(np.nanmean(out[f"{pref}_accepted"].values)) if f"{pref}_accepted" in out.columns else np.nan
        kpis[f"{pref}_revenue_cap_chf"] = float(np.nansum(out[f"{pref}_rev_cap_realized_chf_h"].values)) if f"{pref}_rev_cap_realized_chf_h" in out.columns else 0.0
        kpis[f"{pref}_revenue_energy_chf"] = float(np.nansum(out[f"{pref}_rev_energy_realized_chf_h"].values)) if f"{pref}_rev_energy_realized_chf_h" in out.columns else 0.0
        kpis[f"{pref}_revenue_total_chf"] = float(np.nansum(out[f"{pref}_rev_total_realized_chf_h"].values)) if f"{pref}_rev_total_realized_chf_h" in out.columns else 0.0
        kpis[f"{pref}_mean_feasible_offer_mw"] = float(np.nanmean(out[f"{pref}_feasible_offer_mw"].values)) if f"{pref}_feasible_offer_mw" in out.columns else 0.0
        kpis[f"{pref}_mean_effective_offer_mw"] = float(np.nanmean(out[f"{pref}_effective_offer_mw"].values)) if f"{pref}_effective_offer_mw" in out.columns else 0.0
        kpis[f"{pref}_chosen_hours"] = int(np.nansum(out[f"{pref}_chosen"].values)) if f"{pref}_chosen" in out.columns else 0

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