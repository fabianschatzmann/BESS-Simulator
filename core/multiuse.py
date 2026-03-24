# core/multiuse.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd


# =============================================================================
# Battery constraints for realized Multiuse path
# =============================================================================
@dataclass
class BatteryConstraintParams:
    e_nom_kwh: float
    p_ch_max_kw: float
    p_dis_max_kw: float
    eta_ch: float
    eta_dis: float
    soc_min: float   # fraction 0..1
    soc_max: float   # fraction 0..1
    soc0: float      # fraction 0..1


@dataclass
class MultiuseSettings:
    """
    Multiuse Layer.

    Stufe 1 (SDL-first / Residual-DA-ID):
    - SDL wird stündlich als Primärmarkt behandelt.
    - Danach darf der vorhandene DA/ID-Plan nur noch auf der verbleibenden
      Leistungs- und SOC-Freiheit realisiert werden.
    - SDL- und residuale DA/ID-Erlöse werden additiv zusammengeführt.

    Wichtiger Hinweis zu Gate-Closures:
    - Die eigentliche Marktlogik zu Gate-Closure-Zeitpunkten muss upstream beim
      Erzeugen von results_da_id und sdl_timeseries umgesetzt werden.
    - Dieses File übernimmt die Cutoff-Timestamps nur als Metadaten und nutzt
      sie für Reporting / Nachvollziehbarkeit.
    """
    require_sdl_acceptance: bool = True

    use_dynamic_margin: bool = True
    use_opportunity_cost: bool = True

    lookahead_h: int = 6

    weight_spread_vol: float = 1.0
    weight_soc_edge: float = 8.0
    weight_id_forecast_uncertainty: float = 0.5
    weight_sdl_activation: float = 1.0
    weight_degradation: float = 1.0

    soc_edge_band_pct: float = 15.0
    degradation_chf_per_kwh_throughput: float = 0.0
    lookahead_discount: float = 0.90

    perfect_forecast_upper_bound_mode: bool = True

    block_hours: int = 4
    tz_local: str = "Europe/Zurich"

    # Technische Realisierungslogik
    enforce_realized_soc: bool = True
    reserve_duration_minutes: float = 5.0
    soc_buffer_kwh: float = 0.0
    allow_partial_sdl_offer: bool = False
    assume_sdl_soc_neutral: bool = True

    # Fallback-Wirkungsgrade falls batt nicht explizit übergeben wird
    fallback_eta_ch: float = 0.95
    fallback_eta_dis: float = 0.95


# =============================================================================
# Basic helpers
# =============================================================================
def _ensure_ts_key(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "ts_key" not in out.columns:
        raise ValueError("DataFrame muss ts_key enthalten (UTC-naiv).")
    out["ts_key"] = pd.to_datetime(out["ts_key"], errors="coerce").dt.floor("H")
    out = out.dropna(subset=["ts_key"]).sort_values("ts_key").reset_index(drop=True)
    return out


def _pick_first_col(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Keine der Spalten gefunden: {candidates}")
    return None


def _as_float_series(df: pd.DataFrame, col: Optional[str], default: float = 0.0) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _as_int_series(df: pd.DataFrame, col: Optional[str], default: int = 0) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=int)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)


def _safe_rolling_std_forward(x: pd.Series, window: int) -> pd.Series:
    """
    Vorwärtsgerichtete Rolling-Std:
    Für Stunde t wird die Std über t..t+window-1 berechnet.
    """
    arr = pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(arr)
    out = np.zeros(n, dtype=float)
    w = max(1, int(window))

    for i in range(n):
        seg = arr[i:min(n, i + w)]
        if len(seg) <= 1:
            out[i] = 0.0
        else:
            out[i] = float(np.nanstd(seg))

    return pd.Series(out, index=x.index, dtype=float)


def _forward_discounted_positive_sum(x: pd.Series, horizon: int, discount: float) -> pd.Series:
    """
    Lookahead-Summe positiver DA/ID-Werte:
    opp_cost(t) = sum_{k=0..H-1} discount^k * max(x[t+k], 0)
    """
    arr = pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(arr)
    h = max(1, int(horizon))
    d = float(discount)
    weights = np.array([d**k for k in range(h)], dtype=float)

    out = np.zeros(n, dtype=float)
    for i in range(n):
        seg = np.maximum(arr[i:min(n, i + h)], 0.0)
        out[i] = float(np.sum(seg * weights[:len(seg)]))

    return pd.Series(out, index=x.index, dtype=float)


def _infer_soc_pct(out: pd.DataFrame, soc_kwh: pd.Series) -> pd.Series:
    """
    Nutzt vorhandene soc_pct/soc, falls vorhanden.
    Sonst heuristische Normierung von soc_kwh auf 0..100.
    """
    if "soc_pct" in out.columns:
        return pd.to_numeric(out["soc_pct"], errors="coerce").clip(lower=0.0, upper=100.0)

    if "soc" in out.columns:
        s = pd.to_numeric(out["soc"], errors="coerce")
        if float(s.dropna().max()) > 1.5:
            return s.clip(lower=0.0, upper=100.0)
        return (s * 100.0).clip(lower=0.0, upper=100.0)

    s = pd.to_numeric(soc_kwh, errors="coerce")
    smin = float(s.min()) if s.notna().any() else np.nan
    smax = float(s.max()) if s.notna().any() else np.nan
    if np.isfinite(smin) and np.isfinite(smax) and smax > smin:
        return ((s - smin) / (smax - smin) * 100.0).clip(lower=0.0, upper=100.0)

    return pd.Series(np.nan, index=s.index, dtype=float)


def _compute_soc_edge_penalty(soc_pct: pd.Series, band_pct: float) -> pd.Series:
    """
    0 in der Mitte, ansteigend nahe soc_min/soc_max.
    band_pct z.B. 15 => Penalty in den Randbereichen 0..15% und 85..100%.
    """
    s = pd.to_numeric(soc_pct, errors="coerce")
    band = max(1e-6, float(band_pct))

    dist_low = s
    dist_high = 100.0 - s
    dist_edge = pd.concat([dist_low, dist_high], axis=1).min(axis=1)

    penalty = ((band - dist_edge) / band).clip(lower=0.0)
    return penalty.fillna(0.0).astype(float)


def _compute_dynamic_margin_components(
    out: pd.DataFrame,
    settings: MultiuseSettings,
) -> pd.DataFrame:
    """
    Baut die einzelnen Komponenten der dynamischen Marge.
    In SDL-first ist dies nur noch Diagnose / Reporting.
    """
    spread_fc = (
        pd.to_numeric(out.get("price_id_fc", 0.0), errors="coerce").fillna(0.0)
        - pd.to_numeric(out.get("price_da_fc", 0.0), errors="coerce").fillna(0.0)
    ).astype(float)

    spread_vol = _safe_rolling_std_forward(spread_fc, settings.lookahead_h)

    if "price_id" in out.columns and "price_id_fc" in out.columns:
        id_abs_err = (
            pd.to_numeric(out["price_id"], errors="coerce").fillna(0.0)
            - pd.to_numeric(out["price_id_fc"], errors="coerce").fillna(0.0)
        ).abs()
        id_fc_unc = id_abs_err.rolling(window=max(2, settings.lookahead_h), min_periods=1).mean().astype(float)
    else:
        id_fc_unc = spread_fc.abs().rolling(window=max(2, settings.lookahead_h), min_periods=1).mean().astype(float)

    soc_pct = _infer_soc_pct(out, pd.to_numeric(out.get("soc_kwh", np.nan), errors="coerce"))
    soc_edge_penalty = _compute_soc_edge_penalty(soc_pct, settings.soc_edge_band_pct)

    sdl_any_accepted = pd.to_numeric(out.get("sdl_any_accepted", 0), errors="coerce").fillna(0).astype(int)
    sdl_activation_adj = pd.Series(np.where(sdl_any_accepted == 1, 0.0, 1.0), index=out.index, dtype=float)

    p_bess = pd.to_numeric(out.get("p_bess_kw", 0.0), errors="coerce").fillna(0.0).abs()
    degradation_cost = p_bess * float(settings.degradation_chf_per_kwh_throughput)

    comp = pd.DataFrame(
        {
            "spread_vol_chf_h": (spread_vol * float(settings.weight_spread_vol)).astype(float),
            "soc_edge_penalty_chf_h": (soc_edge_penalty * float(settings.weight_soc_edge)).astype(float),
            "id_fc_uncertainty_chf_h": (id_fc_unc * float(settings.weight_id_forecast_uncertainty)).astype(float),
            "sdl_activation_adj_chf_h": (sdl_activation_adj * float(settings.weight_sdl_activation)).astype(float),
            "degradation_cost_chf_h": (degradation_cost * float(settings.weight_degradation)).astype(float),
        },
        index=out.index,
    )

    if settings.use_dynamic_margin:
        comp["dynamic_margin_chf_h"] = (
            comp["spread_vol_chf_h"]
            + comp["soc_edge_penalty_chf_h"]
            + comp["id_fc_uncertainty_chf_h"]
            + comp["sdl_activation_adj_chf_h"]
            + comp["degradation_cost_chf_h"]
        ).astype(float)
    else:
        comp["dynamic_margin_chf_h"] = 0.0

    return comp


def _compute_opportunity_cost(
    out: pd.DataFrame,
    settings: MultiuseSettings,
) -> pd.Series:
    """
    Mehrstundenwert der freien Batterie für DA/ID.
    In SDL-first nur noch Diagnose / Reporting.
    """
    rev_energy = pd.to_numeric(out.get("rev_energy_chf_h", 0.0), errors="coerce").fillna(0.0).astype(float)

    opp = _forward_discounted_positive_sum(
        rev_energy,
        horizon=settings.lookahead_h,
        discount=settings.lookahead_discount,
    )

    if not settings.use_opportunity_cost:
        opp[:] = 0.0

    return opp.astype(float)


def _add_4h_block_keys(out: pd.DataFrame, tz_local: str, block_hours: int) -> pd.DataFrame:
    """
    Erzeugt lokale Lieferblöcke:
    00-04, 04-08, 08-12, 12-16, 16-20, 20-24
    """
    res = out.copy()

    ts_utc = pd.to_datetime(res["ts_key"], errors="coerce", utc=True)
    ts_local = ts_utc.dt.tz_convert(tz_local)

    block_h = int(block_hours)
    if block_h <= 0 or 24 % block_h != 0:
        raise ValueError("block_hours muss ein Teiler von 24 sein, z.B. 4.")

    local_hour = ts_local.dt.hour
    block_start_hour = (local_hour // block_h) * block_h

    delivery_date_local = ts_local.dt.floor("D")
    block_start_local = delivery_date_local + pd.to_timedelta(block_start_hour, unit="h")
    block_end_local = block_start_local + pd.to_timedelta(block_h, unit="h")

    res["ts_local"] = ts_local
    res["delivery_date_local"] = delivery_date_local.dt.tz_localize(None)
    res["delivery_block_start_local"] = block_start_local.dt.tz_localize(None)
    res["delivery_block_end_local"] = block_end_local.dt.tz_localize(None)
    res["delivery_block_hour_local"] = block_start_hour.astype(int)
    res["delivery_block_label"] = (
        block_start_hour.astype(str).str.zfill(2)
        + ":00-"
        + (block_start_hour + block_h).astype(str).str.zfill(2)
        + ":00"
    )

    return res


def _detect_perfect_forecast_mode(out: pd.DataFrame) -> bool:
    """
    Erkennt heuristisch, ob DA und ID im aktuellen DataFrame als Perfect Forecast vorliegen.
    """
    needed = {"price_da", "price_da_fc", "price_id", "price_id_fc"}
    if not needed.issubset(set(out.columns)):
        return False

    a = pd.to_numeric(out["price_da"], errors="coerce")
    b = pd.to_numeric(out["price_da_fc"], errors="coerce")
    c = pd.to_numeric(out["price_id"], errors="coerce")
    d = pd.to_numeric(out["price_id_fc"], errors="coerce")

    mask_da = a.notna() & b.notna()
    mask_id = c.notna() & d.notna()
    if mask_da.sum() == 0 or mask_id.sum() == 0:
        return False

    is_pf_da = bool(np.allclose(a[mask_da].to_numpy(dtype=float), b[mask_da].to_numpy(dtype=float), atol=1e-9, rtol=0.0))
    is_pf_id = bool(np.allclose(c[mask_id].to_numpy(dtype=float), d[mask_id].to_numpy(dtype=float), atol=1e-9, rtol=0.0))

    return is_pf_da and is_pf_id


# =============================================================================
# Battery constraint helpers
# =============================================================================
def _to_fraction_soc(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s.astype(float)
    if float(s.dropna().max()) > 1.5:
        return (s / 100.0).clip(lower=0.0, upper=1.0)
    return s.clip(lower=0.0, upper=1.0)


def _infer_battery_constraints_from_results(
    r: pd.DataFrame,
    settings: MultiuseSettings,
) -> Optional[BatteryConstraintParams]:
    """
    Best-effort-Fallback, falls batt nicht explizit übergeben wird.
    Für robuste Ergebnisse sollte batt explizit an build_multiuse_priority_sdl übergeben werden.
    """
    if r is None or r.empty:
        return None

    soc_col = _pick_first_col(r, ["soc_kwh", "soc_kwh_end", "soc_kwh_t"])
    soc_kwh = _as_float_series(r, soc_col, np.nan)

    soc_pct_col = _pick_first_col(r, ["soc_pct", "soc"], required=False)
    soc_frac_explicit = None
    if soc_pct_col is not None:
        soc_frac_explicit = _to_fraction_soc(r[soc_pct_col])

    # E_nom inferieren, wenn möglich aus soc_kwh und soc_frac
    e_nom_kwh = np.nan
    if soc_frac_explicit is not None:
        mask = soc_kwh.notna() & soc_frac_explicit.notna() & (soc_frac_explicit > 1e-6)
        if mask.any():
            e_nom_kwh = float(np.nanmedian((soc_kwh[mask] / soc_frac_explicit[mask]).to_numpy(dtype=float)))

    if not np.isfinite(e_nom_kwh) or e_nom_kwh <= 0:
        if soc_kwh.notna().any():
            e_nom_kwh = float(np.nanmax(soc_kwh.to_numpy(dtype=float)))
        else:
            return None

    if not np.isfinite(e_nom_kwh) or e_nom_kwh <= 0:
        return None

    p_ch_col = _pick_first_col(r, ["p_charge_kw", "p_charge", "charge_kw"])
    p_dis_col = _pick_first_col(r, ["p_discharge_kw", "p_discharge", "discharge_kw"])
    p_bess_col = _pick_first_col(r, ["p_bess_kw", "p_bess", "p_net_kw", "p_total_kw", "p_kw"])

    p_ch = _as_float_series(r, p_ch_col, 0.0).clip(lower=0.0)
    p_dis = _as_float_series(r, p_dis_col, 0.0).clip(lower=0.0)
    p_net = _as_float_series(r, p_bess_col, 0.0)

    p_ch_max_kw = float(np.nanmax(p_ch.to_numpy(dtype=float))) if len(p_ch) else 0.0
    p_dis_max_kw = float(np.nanmax(p_dis.to_numpy(dtype=float))) if len(p_dis) else 0.0

    if p_ch_max_kw <= 0:
        p_ch_max_kw = float(np.nanmax(np.clip(-p_net.to_numpy(dtype=float), 0.0, None))) if len(p_net) else 0.0
    if p_dis_max_kw <= 0:
        p_dis_max_kw = float(np.nanmax(np.clip(p_net.to_numpy(dtype=float), 0.0, None))) if len(p_net) else 0.0

    if p_ch_max_kw <= 0 and p_dis_max_kw <= 0:
        return None

    if soc_frac_explicit is not None and soc_frac_explicit.notna().any():
        soc_min = float(np.nanmin(soc_frac_explicit.to_numpy(dtype=float)))
        soc_max = float(np.nanmax(soc_frac_explicit.to_numpy(dtype=float)))
        soc0 = float(soc_frac_explicit.iloc[0]) if pd.notna(soc_frac_explicit.iloc[0]) else 0.5
    else:
        soc_min = 0.0
        soc_max = 1.0
        if soc_kwh.notna().any() and e_nom_kwh > 0:
            soc0 = float(np.clip(float(soc_kwh.iloc[0]) / e_nom_kwh, 0.0, 1.0)) if pd.notna(soc_kwh.iloc[0]) else 0.5
        else:
            soc0 = 0.5

    soc_min = float(np.clip(soc_min, 0.0, 1.0))
    soc_max = float(np.clip(soc_max, soc_min, 1.0))
    soc0 = float(np.clip(soc0, soc_min, soc_max))

    return BatteryConstraintParams(
        e_nom_kwh=float(e_nom_kwh),
        p_ch_max_kw=float(max(0.0, p_ch_max_kw)),
        p_dis_max_kw=float(max(0.0, p_dis_max_kw)),
        eta_ch=float(settings.fallback_eta_ch),
        eta_dis=float(settings.fallback_eta_dis),
        soc_min=soc_min,
        soc_max=soc_max,
        soc0=soc0,
    )


def _coerce_battery_constraints(
    batt: Optional[Any],
    r: pd.DataFrame,
    settings: MultiuseSettings,
) -> Optional[BatteryConstraintParams]:
    if batt is None:
        return _infer_battery_constraints_from_results(r, settings)

    if isinstance(batt, BatteryConstraintParams):
        return batt

    if isinstance(batt, dict):
        try:
            out = BatteryConstraintParams(
                e_nom_kwh=float(batt["e_nom_kwh"]),
                p_ch_max_kw=float(batt["p_ch_max_kw"]),
                p_dis_max_kw=float(batt["p_dis_max_kw"]),
                eta_ch=float(batt.get("eta_ch", settings.fallback_eta_ch)),
                eta_dis=float(batt.get("eta_dis", settings.fallback_eta_dis)),
                soc_min=float(batt["soc_min"]),
                soc_max=float(batt["soc_max"]),
                soc0=float(batt["soc0"]),
            )
            out.soc_min = float(np.clip(out.soc_min, 0.0, 1.0))
            out.soc_max = float(np.clip(out.soc_max, out.soc_min, 1.0))
            out.soc0 = float(np.clip(out.soc0, out.soc_min, out.soc_max))
            return out
        except Exception:
            return _infer_battery_constraints_from_results(r, settings)

    return _infer_battery_constraints_from_results(r, settings)


def _clip_soc_kwh(soc_kwh: float, batt: BatteryConstraintParams) -> float:
    e_min = float(batt.soc_min * batt.e_nom_kwh)
    e_max = float(batt.soc_max * batt.e_nom_kwh)
    return float(np.clip(float(soc_kwh), e_min, e_max))


def _split_plan_power(
    row: pd.Series,
) -> Tuple[float, float]:
    """
    Liefert geplante Lade-/Entladeleistung [kW] positiv.
    Falls keine expliziten Spalten vorhanden sind, wird p_bess_kw verwendet:
      p_bess > 0 => Entladen
      p_bess < 0 => Laden
    """
    p_ch = float(pd.to_numeric(row.get("p_charge_plan_kw", np.nan), errors="coerce"))
    p_dis = float(pd.to_numeric(row.get("p_discharge_plan_kw", np.nan), errors="coerce"))

    if np.isfinite(p_ch) and np.isfinite(p_dis):
        return max(0.0, p_ch), max(0.0, p_dis)

    p_net = float(pd.to_numeric(row.get("p_bess_kw", 0.0), errors="coerce"))
    if p_net >= 0.0:
        return 0.0, p_net
    return -p_net, 0.0


def _feasible_offer_mw_from_soc(
    *,
    soc_kwh: float,
    batt: BatteryConstraintParams,
    product_key: str,
    reserve_duration_minutes: float,
    soc_buffer_kwh: float = 0.0,
) -> float:
    """
    Maximal technisch zulässige SDL-Angebotsleistung [MW] aus aktuellem SOC.

    PRL sym:
      benötigt Up- und Down-Headroom
    SRL UP:
      benötigt Entlade-Headroom
    SRL DOWN:
      benötigt Lade-Headroom
    """
    tau_h = max(1e-9, float(reserve_duration_minutes) / 60.0)

    e_min = float(batt.soc_min * batt.e_nom_kwh) + float(max(0.0, soc_buffer_kwh))
    e_max = float(batt.soc_max * batt.e_nom_kwh) - float(max(0.0, soc_buffer_kwh))

    if e_max < e_min:
        e_min = float(batt.soc_min * batt.e_nom_kwh)
        e_max = float(batt.soc_max * batt.e_nom_kwh)

    e = float(np.clip(float(soc_kwh), e_min, e_max))

    p_up_feasible_kw = min(
        float(batt.p_dis_max_kw),
        max(0.0, (e - e_min) * float(batt.eta_dis) / tau_h),
    )

    p_down_feasible_kw = min(
        float(batt.p_ch_max_kw),
        max(0.0, (e_max - e) / max(float(batt.eta_ch), 1e-9) / tau_h),
    )

    if product_key == "prl_sym":
        return float(max(0.0, min(p_up_feasible_kw, p_down_feasible_kw)) / 1000.0)
    if product_key == "srl_up":
        return float(max(0.0, p_up_feasible_kw) / 1000.0)
    if product_key == "srl_down":
        return float(max(0.0, p_down_feasible_kw) / 1000.0)
    return 0.0


def _nominal_offer_mw_for_product(row: pd.Series, product_key: str) -> float:
    """
    Produktspezifischer Fallback für nominale Angebotsleistung [MW].
    Reihenfolge:
    1) sdl_p_offer_<produkt>_mw
    2) legacy sdl_p_offer_mw
    """
    legacy = float(pd.to_numeric(row.get("sdl_p_offer_mw", 0.0), errors="coerce") or 0.0)

    if product_key == "prl_sym":
        val = row.get("sdl_p_offer_prl_mw", legacy)
    elif product_key == "srl_up":
        val = row.get("sdl_p_offer_srl_up_mw", legacy)
    elif product_key == "srl_down":
        val = row.get("sdl_p_offer_srl_down_mw", legacy)
    else:
        val = legacy

    try:
        return float(max(0.0, float(pd.to_numeric(val, errors="coerce"))))
    except Exception:
        return legacy


def _derive_sdl_power_reservation_kw(
    chosen_pref: Optional[str],
    effective_offer_mw_realized: float,
) -> Tuple[float, float]:
    """
    Liefert reservierte SDL-Leistung in kW:
    - p_ch_reserved_kw
    - p_dis_reserved_kw
    """
    p_kw = max(0.0, float(effective_offer_mw_realized) * 1000.0)

    if chosen_pref == "prl_sym":
        return p_kw, p_kw
    if chosen_pref == "srl_up":
        return 0.0, p_kw
    if chosen_pref == "srl_down":
        return p_kw, 0.0
    return 0.0, 0.0


def _derive_sdl_operating_window(
    *,
    batt: BatteryConstraintParams,
    settings: MultiuseSettings,
    p_ch_reserved_kw: float,
    p_dis_reserved_kw: float,
) -> Tuple[float, float]:
    """
    Berechnet den operativen SOC-Korridor [kWh], der nach SDL-Reservierung
    für DA/ID noch genutzt werden darf.
    """
    tau_h = max(1e-9, float(settings.reserve_duration_minutes) / 60.0)

    e_phys_min = float(batt.soc_min * batt.e_nom_kwh)
    e_phys_max = float(batt.soc_max * batt.e_nom_kwh)

    # Für Entlade-Reserve muss Energie im Speicher gehalten werden.
    e_up_reserve_kwh = float(p_dis_reserved_kw) * tau_h / max(float(batt.eta_dis), 1e-9)

    # Für Lade-Reserve muss Headroom nach oben freigehalten werden.
    e_down_reserve_kwh = float(p_ch_reserved_kw) * tau_h * float(batt.eta_ch)

    e_oper_min = e_phys_min + e_up_reserve_kwh + float(settings.soc_buffer_kwh)
    e_oper_max = e_phys_max - e_down_reserve_kwh - float(settings.soc_buffer_kwh)

    if e_oper_max < e_oper_min:
        e_oper_min = e_phys_min
        e_oper_max = e_phys_max

    e_oper_min = float(np.clip(e_oper_min, e_phys_min, e_phys_max))
    e_oper_max = float(np.clip(e_oper_max, e_oper_min, e_phys_max))

    return e_oper_min, e_oper_max


# =============================================================================
# Realized path simulation helpers
# =============================================================================
def _simulate_daid_block_realized(
    block_hours: pd.DataFrame,
    batt: BatteryConstraintParams,
    soc0_kwh: float,
    dt_h: float = 1.0,
) -> Tuple[pd.DataFrame, float]:
    """
    Realisiert den geplanten DA/ID-Dispatch physikalisch auf dem aktuellen Multiuse-SOC.
    Dispatch und Erlöse werden proportional gekappt, falls der SOC nicht ausreicht.
    """
    e_min = float(batt.soc_min * batt.e_nom_kwh)
    e_max = float(batt.soc_max * batt.e_nom_kwh)

    soc = float(np.clip(soc0_kwh, e_min, e_max))
    rows = []

    for _, row in block_hours.iterrows():
        p_ch_plan_kw, p_dis_plan_kw = _split_plan_power(row)

        p_ch_plan_kw = float(min(max(0.0, p_ch_plan_kw), batt.p_ch_max_kw))
        p_dis_plan_kw = float(min(max(0.0, p_dis_plan_kw), batt.p_dis_max_kw))

        max_ch_soc_kw = max(0.0, (e_max - soc) / max(float(batt.eta_ch), 1e-9) / float(dt_h))
        max_dis_soc_kw = max(0.0, (soc - e_min) * float(batt.eta_dis) / float(dt_h))

        p_ch_act_kw = min(p_ch_plan_kw, batt.p_ch_max_kw, max_ch_soc_kw)
        p_dis_act_kw = min(p_dis_plan_kw, batt.p_dis_max_kw, max_dis_soc_kw)

        p_plan_through_kw = p_ch_plan_kw + p_dis_plan_kw
        p_act_through_kw = p_ch_act_kw + p_dis_act_kw

        p_plan_net_kw = p_dis_plan_kw - p_ch_plan_kw
        p_act_net_kw = p_dis_act_kw - p_ch_act_kw

        if abs(p_plan_net_kw) > 1e-9 and np.sign(p_plan_net_kw) == np.sign(p_act_net_kw):
            scale = float(np.clip(p_act_net_kw / p_plan_net_kw, 0.0, 1.0))
        elif p_plan_through_kw > 1e-9:
            scale = float(np.clip(p_act_through_kw / p_plan_through_kw, 0.0, 1.0))
        else:
            scale = 0.0

        rev_da_plan = float(pd.to_numeric(row.get("rev_da_chf_h", 0.0), errors="coerce"))
        rev_id_plan = float(pd.to_numeric(row.get("rev_id_inc_chf_h", 0.0), errors="coerce"))

        rev_da_act = rev_da_plan * scale
        rev_id_act = rev_id_plan * scale
        rev_daid_act = rev_da_act + rev_id_act

        soc_end = soc + float(batt.eta_ch) * p_ch_act_kw * float(dt_h) - (1.0 / max(float(batt.eta_dis), 1e-9)) * p_dis_act_kw * float(dt_h)
        soc_end = float(np.clip(soc_end, e_min, e_max))

        rows.append(
            {
                "ts_key": row["ts_key"],
                "soc_multiuse_kwh": soc,
                "soc_multiuse_end_kwh": soc_end,
                "soc_multiuse_pct": (soc / max(1e-9, batt.e_nom_kwh)) * 100.0,
                "p_charge_multiuse_kw": p_ch_act_kw,
                "p_discharge_multiuse_kw": p_dis_act_kw,
                "p_bess_multiuse_kw": p_dis_act_kw - p_ch_act_kw,
                "rev_da_multiuse_chf_h": rev_da_act,
                "rev_id_multiuse_chf_h": rev_id_act,
                "rev_daid_multiuse_chf_h": rev_daid_act,
                "daid_dispatch_scale": scale,
                "market_state_detail_realized": "DA_ID",
            }
        )

        soc = soc_end

    return pd.DataFrame(rows), soc


def _pick_best_feasible_sdl_product_for_hour(
    row: pd.Series,
    batt: BatteryConstraintParams,
    settings: MultiuseSettings,
    soc_kwh: float,
) -> Dict[str, Any]:
    """
    Wählt pro Stunde genau EIN SDL-Produkt:
    das beste technisch machbare und marktseitig akzeptierte Produkt.

    Dadurch wird vermieden, dass PRL/SRL gleichzeitig additiv gezählt werden.
    """
    label_map = {
        "prl_sym": "SDL: PRL (sym)",
        "srl_up": "SDL: SRL UP",
        "srl_down": "SDL: SRL DOWN",
    }

    allow_partial = bool(row.get("sdl_partial_offer_allowed", settings.allow_partial_sdl_offer))
    best = {
        "chosen_pref": None,
        "chosen_label": "SDL (technisch nicht möglich)",
        "rev_sdl_multiuse_chf_h": 0.0,
        "effective_offer_mw_realized": 0.0,
        "feasible_any_accepted": 0,
        "feasible_offer_mw_chosen": 0.0,
    }

    for pref in ["prl_sym", "srl_up", "srl_down"]:
        accepted = int(pd.to_numeric(row.get(f"{pref}_accepted", 0), errors="coerce") or 0) == 1
        if not accepted:
            continue

        feasible_offer_mw = _feasible_offer_mw_from_soc(
            soc_kwh=float(soc_kwh),
            batt=batt,
            product_key=pref,
            reserve_duration_minutes=settings.reserve_duration_minutes,
            soc_buffer_kwh=settings.soc_buffer_kwh,
        )

        ref_effective_offer_mw = float(pd.to_numeric(row.get(f"{pref}_effective_offer_mw", np.nan), errors="coerce"))
        if not np.isfinite(ref_effective_offer_mw) or ref_effective_offer_mw <= 0:
            ref_effective_offer_mw = _nominal_offer_mw_for_product(row, pref)

        ref_rev_total = float(
            pd.to_numeric(
                row.get(
                    f"{pref}_rev_total_chf_h",
                    float(pd.to_numeric(row.get(f"{pref}_rev_cap_chf_h", 0.0), errors="coerce"))
                    + float(pd.to_numeric(row.get(f"{pref}_rev_energy_chf_h", 0.0), errors="coerce")),
                ),
                errors="coerce",
            )
            or 0.0
        )

        if ref_effective_offer_mw <= 1e-9:
            realized_offer_mw = 0.0
        else:
            if allow_partial:
                realized_offer_mw = min(ref_effective_offer_mw, feasible_offer_mw)
            else:
                realized_offer_mw = ref_effective_offer_mw if feasible_offer_mw + 1e-9 >= ref_effective_offer_mw else 0.0

        scale = realized_offer_mw / ref_effective_offer_mw if ref_effective_offer_mw > 1e-9 else 0.0
        rev_realized = ref_rev_total * scale

        if rev_realized > 0.0:
            best["feasible_any_accepted"] = 1

        if rev_realized > float(best["rev_sdl_multiuse_chf_h"]):
            best = {
                "chosen_pref": pref,
                "chosen_label": label_map[pref],
                "rev_sdl_multiuse_chf_h": float(rev_realized),
                "effective_offer_mw_realized": float(realized_offer_mw),
                "feasible_any_accepted": 1 if rev_realized > 0.0 else int(best["feasible_any_accepted"]),
                "feasible_offer_mw_chosen": float(feasible_offer_mw),
            }

    return best


def _simulate_sdl_block_realized(
    block_hours: pd.DataFrame,
    batt: BatteryConstraintParams,
    settings: MultiuseSettings,
    soc0_kwh: float,
) -> Tuple[pd.DataFrame, float]:
    """
    Realisiert SDL blockweise auf aktuellem SOC.
    Standardannahme: SDL hält den SOC im Multiuse-Layer konstant
    (konservative Verfügbarkeitsprüfung, kein zusätzlicher Energiepfad).
    """
    soc = _clip_soc_kwh(float(soc0_kwh), batt)
    rows = []

    for _, row in block_hours.iterrows():
        picked = _pick_best_feasible_sdl_product_for_hour(
            row=row,
            batt=batt,
            settings=settings,
            soc_kwh=soc,
        )

        soc_end = soc
        rows.append(
            {
                "ts_key": row["ts_key"],
                "soc_multiuse_kwh": soc,
                "soc_multiuse_end_kwh": soc_end,
                "soc_multiuse_pct": (soc / max(1e-9, batt.e_nom_kwh)) * 100.0,
                "p_charge_multiuse_kw": 0.0,
                "p_discharge_multiuse_kw": 0.0,
                "p_bess_multiuse_kw": 0.0,
                "rev_da_multiuse_chf_h": 0.0,
                "rev_id_multiuse_chf_h": 0.0,
                "rev_daid_multiuse_chf_h": 0.0,
                "rev_sdl_multiuse_chf_h": float(picked["rev_sdl_multiuse_chf_h"]),
                "sdl_product_chosen": picked["chosen_pref"],
                "sdl_product_label": picked["chosen_label"],
                "sdl_feasible_any_accepted": int(picked["feasible_any_accepted"]),
                "sdl_effective_offer_mw_realized": float(picked["effective_offer_mw_realized"]),
                "sdl_feasible_offer_mw_chosen": float(picked["feasible_offer_mw_chosen"]),
                "market_state_detail_realized": picked["chosen_label"],
            }
        )
        soc = soc_end

    return pd.DataFrame(rows), soc


def _simulate_sdl_primary_block_realized(
    block_hours: pd.DataFrame,
    batt: BatteryConstraintParams,
    settings: MultiuseSettings,
    soc0_kwh: float,
    dt_h: float = 1.0,
) -> Tuple[pd.DataFrame, float]:
    """
    Stufe 1: SDL-first / residualer DA-ID-Dispatch.

    Pro Stunde:
    1) Bestes technisch machbares und akzeptiertes SDL-Produkt wählen.
    2) SDL-Leistung reservieren.
    3) Operatives SOC-Fenster für SDL-Reserve festlegen.
    4) Vorhandenen DA/ID-Plan nur noch im verbleibenden Restfenster realisieren.
    5) SDL- und residuale DA/ID-Erlöse additiv zusammenführen.

    Gate-Closures werden hier nicht neu optimiert. Dieses File geht davon aus,
    dass die zugelieferten results_da_id und sdl_timeseries bereits die gültigen
    Marktstände nach Gate-Closure repräsentieren.
    """
    e_phys_min = float(batt.soc_min * batt.e_nom_kwh)
    e_phys_max = float(batt.soc_max * batt.e_nom_kwh)

    soc = float(np.clip(soc0_kwh, e_phys_min, e_phys_max))
    rows = []

    for _, row in block_hours.iterrows():
        picked = _pick_best_feasible_sdl_product_for_hour(
            row=row,
            batt=batt,
            settings=settings,
            soc_kwh=soc,
        )

        p_ch_reserved_kw, p_dis_reserved_kw = _derive_sdl_power_reservation_kw(
            picked["chosen_pref"],
            picked["effective_offer_mw_realized"],
        )

        p_ch_residual_kw = max(0.0, float(batt.p_ch_max_kw) - float(p_ch_reserved_kw))
        p_dis_residual_kw = max(0.0, float(batt.p_dis_max_kw) - float(p_dis_reserved_kw))

        e_oper_min, e_oper_max = _derive_sdl_operating_window(
            batt=batt,
            settings=settings,
            p_ch_reserved_kw=p_ch_reserved_kw,
            p_dis_reserved_kw=p_dis_reserved_kw,
        )

        # Numerische Robustheit: wenn current SOC knapp ausserhalb des operativen
        # Fensters liegt, auf dieses Fenster clippen. Das ist nur ein technischer
        # Schutz gegen Rundungsfehler / inkonsistente Upstream-Daten.
        soc = float(np.clip(soc, e_oper_min, e_oper_max))

        p_ch_plan_kw, p_dis_plan_kw = _split_plan_power(row)

        p_ch_plan_kw = float(min(max(0.0, p_ch_plan_kw), p_ch_residual_kw))
        p_dis_plan_kw = float(min(max(0.0, p_dis_plan_kw), p_dis_residual_kw))

        max_ch_soc_kw = max(0.0, (e_oper_max - soc) / max(float(batt.eta_ch), 1e-9) / float(dt_h))
        max_dis_soc_kw = max(0.0, (soc - e_oper_min) * float(batt.eta_dis) / float(dt_h))

        p_ch_act_kw = min(p_ch_plan_kw, max_ch_soc_kw)
        p_dis_act_kw = min(p_dis_plan_kw, max_dis_soc_kw)

        p_plan_through_kw = p_ch_plan_kw + p_dis_plan_kw
        p_act_through_kw = p_ch_act_kw + p_dis_act_kw

        p_plan_net_kw = p_dis_plan_kw - p_ch_plan_kw
        p_act_net_kw = p_dis_act_kw - p_ch_act_kw

        if abs(p_plan_net_kw) > 1e-9 and np.sign(p_plan_net_kw) == np.sign(p_act_net_kw):
            scale = float(np.clip(p_act_net_kw / p_plan_net_kw, 0.0, 1.0))
        elif p_plan_through_kw > 1e-9:
            scale = float(np.clip(p_act_through_kw / p_plan_through_kw, 0.0, 1.0))
        else:
            scale = 0.0

        rev_da_plan = float(pd.to_numeric(row.get("rev_da_chf_h", 0.0), errors="coerce"))
        rev_id_plan = float(pd.to_numeric(row.get("rev_id_inc_chf_h", 0.0), errors="coerce"))
        rev_sdl_act = float(picked["rev_sdl_multiuse_chf_h"])

        rev_da_act = rev_da_plan * scale
        rev_id_act = rev_id_plan * scale
        rev_daid_act = rev_da_act + rev_id_act
        rev_multiuse_act = rev_sdl_act + rev_daid_act

        # Stufe 1: SDL wird konservativ als SOC-neutral modelliert.
        # Die SOC-Fortschreibung erfolgt daher nur mit dem residualen DA/ID-Dispatch.
        soc_end = soc + float(batt.eta_ch) * p_ch_act_kw * float(dt_h) - (1.0 / max(float(batt.eta_dis), 1e-9)) * p_dis_act_kw * float(dt_h)
        soc_end = float(np.clip(soc_end, e_oper_min, e_oper_max))
        soc_end = float(np.clip(soc_end, e_phys_min, e_phys_max))

        sdl_active = bool((picked["chosen_pref"] is not None) and (float(picked["effective_offer_mw_realized"]) > 0.0) and (rev_sdl_act > 0.0))
        daid_active = bool(abs(p_dis_act_kw - p_ch_act_kw) > 1e-9 or rev_daid_act > 0.0)

        if sdl_active and daid_active:
            market_state = "SDL_PLUS_DA_ID"
        elif sdl_active:
            market_state = "SDL_ONLY"
        else:
            market_state = "DA_ID_ONLY"

        rows.append(
            {
                "ts_key": row["ts_key"],
                "soc_multiuse_kwh": soc,
                "soc_multiuse_end_kwh": soc_end,
                "soc_multiuse_pct": (soc / max(1e-9, batt.e_nom_kwh)) * 100.0,
                "soc_oper_min_kwh": e_oper_min,
                "soc_oper_max_kwh": e_oper_max,
                "p_sdl_charge_reserved_kw": float(p_ch_reserved_kw),
                "p_sdl_discharge_reserved_kw": float(p_dis_reserved_kw),
                "p_charge_residual_kw": float(p_ch_residual_kw),
                "p_discharge_residual_kw": float(p_dis_residual_kw),
                "p_charge_multiuse_kw": float(p_ch_act_kw),
                "p_discharge_multiuse_kw": float(p_dis_act_kw),
                "p_bess_multiuse_kw": float(p_dis_act_kw - p_ch_act_kw),
                "rev_da_multiuse_chf_h": float(rev_da_act),
                "rev_id_multiuse_chf_h": float(rev_id_act),
                "rev_daid_multiuse_chf_h": float(rev_daid_act),
                "rev_sdl_multiuse_chf_h": float(rev_sdl_act),
                "rev_multiuse_chf_h": float(rev_multiuse_act),
                "daid_dispatch_scale": float(scale),
                "sdl_product_chosen": picked["chosen_pref"],
                "sdl_product_label": picked["chosen_label"],
                "sdl_feasible_any_accepted": int(picked["feasible_any_accepted"]),
                "sdl_effective_offer_mw_realized": float(picked["effective_offer_mw_realized"]),
                "sdl_feasible_offer_mw_chosen": float(picked["feasible_offer_mw_chosen"]),
                "sdl_active_hour": int(sdl_active),
                "market_state_realized": market_state,
                "market_state_detail_realized": picked["chosen_label"] if sdl_active else "DA_ID_ONLY",
            }
        )

        soc = soc_end

    return pd.DataFrame(rows), soc


# =============================================================================
# Main Multiuse builder
# =============================================================================
def build_multiuse_priority_sdl(
    *,
    results_da_id: pd.DataFrame,
    sdl_timeseries: pd.DataFrame,
    settings: MultiuseSettings,
    batt: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Baut eine Multiuse-Zeitreihe (stündlich, ts_key) mit Market-State pro Stunde.

    Stufe 1 (neu):
    - SDL-first / residualer DA-ID-Dispatch auf realisiertem SOC.
    - SDL wird als Primärmarkt reserviert.
    - DA/ID darf nur die verbleibende Leistung und das verbleibende operative
      SOC-Fenster nutzen.
    - Gate-Closures werden hier nicht neu optimiert, sondern nur als Metadaten
      aus sdl_timeseries übernommen.

    Fallback:
    - Falls keine Batterierestriktionen verfügbar sind oder SOC-Enforcement
      deaktiviert ist, wird aus Sicherheitsgründen die bisherige alte
      Blockwahl-Logik (SDL vs. DA/ID) verwendet.
    """
    r = _ensure_ts_key(results_da_id)
    s = _ensure_ts_key(sdl_timeseries)

    # -------------------------------------------------------------------------
    # Battery constraints
    # -------------------------------------------------------------------------
    batt_resolved = _coerce_battery_constraints(batt=batt, r=r, settings=settings)

    # -------------------------------------------------------------------------
    # DA/ID hourly inputs
    # -------------------------------------------------------------------------
    rev_da_col = _pick_first_col(r, ["rev_da_chf_h", "rev_da_chf", "rev_da_chf_per_h"])
    rev_id_col = _pick_first_col(r, ["rev_id_inc_chf_h", "rev_id_inc_chf", "rev_id_chf_h"])

    p_bess_col = _pick_first_col(r, ["p_bess_kw", "p_bess", "p_net_kw", "p_total_kw", "p_kw"])
    p_charge_col = _pick_first_col(r, ["p_charge_kw", "p_charge", "charge_kw", "p_charge_plan_kw"], required=False)
    p_discharge_col = _pick_first_col(r, ["p_discharge_kw", "p_discharge", "discharge_kw", "p_discharge_plan_kw"], required=False)

    soc_col = _pick_first_col(r, ["soc_kwh", "soc_kwh_end", "soc_kwh_t"], required=False)

    rev_da = _as_float_series(r, rev_da_col, 0.0)
    rev_id = _as_float_series(r, rev_id_col, 0.0)
    rev_energy = (rev_da + rev_id).astype(float)

    p_bess = _as_float_series(r, p_bess_col, 0.0)
    p_charge = _as_float_series(r, p_charge_col, np.nan)
    p_discharge = _as_float_series(r, p_discharge_col, np.nan)
    soc_kwh = _as_float_series(r, soc_col, np.nan)

    price_da_fc = _as_float_series(r, _pick_first_col(r, ["price_da_fc", "price_fc"]), 0.0)
    price_id_fc = _as_float_series(r, _pick_first_col(r, ["price_id_fc"]), 0.0)
    price_da = _as_float_series(r, _pick_first_col(r, ["price_da"]), 0.0)
    price_id = _as_float_series(r, _pick_first_col(r, ["price_id"]), 0.0)

    soc_pct_col = _pick_first_col(r, ["soc_pct", "soc"], required=False)
    soc_pct = _as_float_series(r, soc_pct_col, np.nan)

    # -------------------------------------------------------------------------
    # SDL hourly inputs
    # -------------------------------------------------------------------------
    rev_sdl_total_col = _pick_first_col(s, ["sdl_total_rev_chf_h", "sdl_total_rev_chf_per_h", "sdl_total_rev_chf"])
    rev_sdl_total = _as_float_series(s, rev_sdl_total_col, 0.0)

    acc_cols = [c for c in ["srl_up_accepted", "srl_down_accepted", "prl_sym_accepted"] if c in s.columns]
    if not acc_cols:
        sdl_any_accepted = pd.Series(1, index=s.index, dtype=int)
    else:
        acc_any = np.zeros(len(s), dtype=int)
        for c in acc_cols:
            acc_any = np.maximum(acc_any, _as_int_series(s, c, 0).to_numpy(dtype=int))
        sdl_any_accepted = pd.Series(acc_any, index=s.index, dtype=int)

    cutoff_cols = [c for c in ["srl_up_cutoff_ts_key", "srl_down_cutoff_ts_key", "prl_sym_cutoff_ts_key"] if c in s.columns]

    # -------------------------------------------------------------------------
    # Merge base frame
    # -------------------------------------------------------------------------
    out = pd.DataFrame({"ts_key": r["ts_key"]})
    out = out.merge(s[["ts_key"]].assign(_has_sdl=1), on="ts_key", how="left")
    out["_has_sdl"] = out["_has_sdl"].fillna(0).astype(int)

    tmp_r = pd.DataFrame(
        {
            "ts_key": r["ts_key"],
            "rev_da_chf_h": rev_da.to_numpy(dtype=float),
            "rev_id_inc_chf_h": rev_id.to_numpy(dtype=float),
            "rev_energy_chf_h": rev_energy.to_numpy(dtype=float),
            "p_bess_kw": p_bess.to_numpy(dtype=float),
            "p_charge_plan_kw": p_charge.to_numpy(dtype=float),
            "p_discharge_plan_kw": p_discharge.to_numpy(dtype=float),
            "soc_kwh": soc_kwh.to_numpy(dtype=float),
            "soc_pct": soc_pct.to_numpy(dtype=float),
            "price_da": price_da.to_numpy(dtype=float),
            "price_id": price_id.to_numpy(dtype=float),
            "price_da_fc": price_da_fc.to_numpy(dtype=float),
            "price_id_fc": price_id_fc.to_numpy(dtype=float),
        }
    )

    tmp_s = pd.DataFrame(
        {
            "ts_key": s["ts_key"],
            "rev_sdl_total_chf_h": rev_sdl_total.to_numpy(dtype=float),
            "sdl_any_accepted": sdl_any_accepted.to_numpy(dtype=int),
            "sdl_p_offer_mw": _as_float_series(s, _pick_first_col(s, ["sdl_p_offer_mw"]), 0.0).to_numpy(dtype=float),
            "sdl_p_offer_prl_mw": _as_float_series(s, _pick_first_col(s, ["sdl_p_offer_prl_mw"]), 0.0).to_numpy(dtype=float),
            "sdl_p_offer_srl_up_mw": _as_float_series(s, _pick_first_col(s, ["sdl_p_offer_srl_up_mw"]), 0.0).to_numpy(dtype=float),
            "sdl_p_offer_srl_down_mw": _as_float_series(s, _pick_first_col(s, ["sdl_p_offer_srl_down_mw"]), 0.0).to_numpy(dtype=float),
            "sdl_partial_offer_allowed": _as_int_series(
                s,
                _pick_first_col(s, ["sdl_partial_offer_allowed"]),
                int(settings.allow_partial_sdl_offer),
            ).astype(int).to_numpy(dtype=int),
        }
    )

    # Produktbezogene SDL-Spalten für realisierte Auswahl
    for pref in ["prl_sym", "srl_up", "srl_down"]:
        for col in [
            f"{pref}_accepted",
            f"{pref}_rev_total_chf_h",
            f"{pref}_rev_cap_chf_h",
            f"{pref}_rev_energy_chf_h",
            f"{pref}_effective_offer_mw",
            f"{pref}_feasible_offer_mw",
            f"{pref}_tech_ok",
        ]:
            if col in s.columns:
                tmp_s[col] = s[col].values
            else:
                if col.endswith("_accepted") or col.endswith("_tech_ok"):
                    tmp_s[col] = 0
                else:
                    tmp_s[col] = np.nan

    if cutoff_cols:
        cut = None
        for c in cutoff_cols:
            cc = pd.to_datetime(s[c], errors="coerce")
            if cut is None:
                cut = cc.copy()
            else:
                a = cut.to_numpy(dtype="datetime64[ns]")
                b = cc.to_numpy(dtype="datetime64[ns]")
                cut = pd.Series(np.minimum(a, b), index=s.index)
        tmp_s["decision_cutoff_ts_key"] = pd.to_datetime(cut, errors="coerce")
    else:
        tmp_s["decision_cutoff_ts_key"] = pd.NaT

    out = out.merge(tmp_r, on="ts_key", how="left")
    out = out.merge(tmp_s, on="ts_key", how="left")

    out["rev_da_chf_h"] = out["rev_da_chf_h"].fillna(0.0).astype(float)
    out["rev_id_inc_chf_h"] = out["rev_id_inc_chf_h"].fillna(0.0).astype(float)
    out["rev_energy_chf_h"] = out["rev_energy_chf_h"].fillna(0.0).astype(float)
    out["rev_sdl_total_chf_h"] = out["rev_sdl_total_chf_h"].fillna(0.0).astype(float)
    out["sdl_any_accepted"] = out["sdl_any_accepted"].fillna(0).astype(int)
    out["sdl_p_offer_mw"] = out["sdl_p_offer_mw"].fillna(0.0).astype(float)
    out["sdl_p_offer_prl_mw"] = out["sdl_p_offer_prl_mw"].fillna(out["sdl_p_offer_mw"]).astype(float)
    out["sdl_p_offer_srl_up_mw"] = out["sdl_p_offer_srl_up_mw"].fillna(out["sdl_p_offer_mw"]).astype(float)
    out["sdl_p_offer_srl_down_mw"] = out["sdl_p_offer_srl_down_mw"].fillna(out["sdl_p_offer_mw"]).astype(float)
    out["sdl_partial_offer_allowed"] = out["sdl_partial_offer_allowed"].fillna(int(settings.allow_partial_sdl_offer)).astype(int)

    # -------------------------------------------------------------------------
    # Blockbildung
    # -------------------------------------------------------------------------
    out = _add_4h_block_keys(out, tz_local=settings.tz_local, block_hours=settings.block_hours)

    # -------------------------------------------------------------------------
    # Diagnosegrössen / Reporting
    # -------------------------------------------------------------------------
    comps = _compute_dynamic_margin_components(out, settings)
    for c in comps.columns:
        out[c] = comps[c].to_numpy()

    out["opp_cost_chf_h"] = _compute_opportunity_cost(out, settings).to_numpy(dtype=float)

    out["sdl_score_chf_h"] = (
        out["rev_sdl_total_chf_h"]
        - out["sdl_activation_adj_chf_h"]
    ).astype(float)

    out["daid_score_chf_h"] = (
        out["rev_energy_chf_h"]
        + out["opp_cost_chf_h"]
    ).astype(float)

    out["decision_gap_chf_h"] = (
        out["sdl_score_chf_h"]
        - out["daid_score_chf_h"]
        - out["dynamic_margin_chf_h"]
    ).astype(float)

    # -------------------------------------------------------------------------
    # Blockaggregation (Baseline-Info)
    # -------------------------------------------------------------------------
    block_key = "delivery_block_start_local"
    grp = out.groupby(block_key, dropna=False)

    block_df = grp.agg(
        delivery_date_local=("delivery_date_local", "first"),
        delivery_block_end_local=("delivery_block_end_local", "first"),
        delivery_block_hour_local=("delivery_block_hour_local", "first"),
        delivery_block_label=("delivery_block_label", "first"),
        rev_sdl_block_chf=("rev_sdl_total_chf_h", "sum"),
        rev_daid_block_chf=("rev_energy_chf_h", "sum"),
        sdl_score_block_chf=("sdl_score_chf_h", "sum"),
        daid_score_block_chf=("daid_score_chf_h", "sum"),
        dynamic_margin_block_chf=("dynamic_margin_chf_h", "sum"),
        opp_cost_block_chf=("opp_cost_chf_h", "sum"),
        sdl_any_accepted_block=("sdl_any_accepted", "max"),
    ).reset_index()

    if "decision_cutoff_ts_key" in out.columns:
        cut_block = grp["decision_cutoff_ts_key"].min().reset_index()
        block_df = block_df.merge(cut_block, on=block_key, how="left")
    else:
        block_df["decision_cutoff_ts_key"] = pd.NaT

    perfect_forecast_detected = _detect_perfect_forecast_mode(out)

    # -------------------------------------------------------------------------
    # Default columns for final realized path
    # -------------------------------------------------------------------------
    out["choose_sdl_block"] = False
    out["choose_sdl"] = False
    out["market_state"] = "DA_ID_ONLY"
    out["market_state_detail"] = "DA_ID_ONLY"

    out["soc_kwh_multiuse"] = np.nan
    out["soc_kwh_end_multiuse"] = np.nan
    out["soc_pct_multiuse"] = np.nan

    out["soc_oper_min_kwh"] = np.nan
    out["soc_oper_max_kwh"] = np.nan

    out["p_sdl_charge_reserved_kw"] = 0.0
    out["p_sdl_discharge_reserved_kw"] = 0.0
    out["p_charge_residual_kw"] = 0.0
    out["p_discharge_residual_kw"] = 0.0

    out["p_charge_multiuse_kw"] = 0.0
    out["p_discharge_multiuse_kw"] = 0.0
    out["p_bess_multiuse_kw"] = 0.0

    out["rev_da_multiuse_chf_h"] = 0.0
    out["rev_id_multiuse_chf_h"] = 0.0
    out["rev_daid_multiuse_chf_h"] = 0.0
    out["rev_sdl_multiuse_chf_h"] = 0.0
    out["rev_multiuse_chf_h"] = 0.0

    out["daid_dispatch_scale"] = np.nan
    out["sdl_product_chosen"] = None
    out["sdl_product_label"] = None
    out["sdl_feasible_any_accepted"] = 0
    out["sdl_effective_offer_mw_realized"] = 0.0
    out["sdl_feasible_offer_mw_chosen"] = 0.0

    # -------------------------------------------------------------------------
    # SDL-first realized block path on current SOC
    # -------------------------------------------------------------------------
    if settings.enforce_realized_soc and batt_resolved is not None:
        e_min = float(batt_resolved.soc_min * batt_resolved.e_nom_kwh)
        e_max = float(batt_resolved.soc_max * batt_resolved.e_nom_kwh)

        # Start-SOC bevorzugt aus DA/ID-Resultaten, sonst batt.soc0.
        if len(out) and pd.notna(out["soc_kwh"].iloc[0]):
            soc_cur_kwh = float(np.clip(float(out["soc_kwh"].iloc[0]), e_min, e_max))
        else:
            soc_cur_kwh = float(np.clip(float(batt_resolved.soc0) * float(batt_resolved.e_nom_kwh), e_min, e_max))

        block_choice_map = {}
        block_gap_map = {}
        block_margin_map = {}
        block_opp_map = {}
        block_mode_map = {}
        block_acc_map = {}
        block_cutoff_map = {}
        block_rev_sdl_realized_map = {}
        block_rev_daid_realized_map = {}
        block_rev_multiuse_realized_map = {}
        block_soc_start_map = {}
        block_soc_end_map = {}
        block_avg_p_sdl_ch_map = {}
        block_avg_p_sdl_dis_map = {}
        block_avg_p_res_ch_map = {}
        block_avg_p_res_dis_map = {}

        for _, brow in block_df.sort_values(block_key).iterrows():
            bstart = brow[block_key]
            idx = out.index[out[block_key] == bstart]
            if len(idx) == 0:
                continue

            bh = out.loc[idx].copy().sort_values("ts_key").reset_index(drop=False)
            original_idx = bh["index"].to_numpy(dtype=int)

            soc_block_start_kwh = float(soc_cur_kwh)
            sim, soc_end = _simulate_sdl_primary_block_realized(
                block_hours=bh,
                batt=batt_resolved,
                settings=settings,
                soc0_kwh=soc_cur_kwh,
                dt_h=1.0,
            )

            out.loc[original_idx, "soc_kwh_multiuse"] = sim["soc_multiuse_kwh"].to_numpy(dtype=float)
            out.loc[original_idx, "soc_kwh_end_multiuse"] = sim["soc_multiuse_end_kwh"].to_numpy(dtype=float)
            out.loc[original_idx, "soc_pct_multiuse"] = sim["soc_multiuse_pct"].to_numpy(dtype=float)

            out.loc[original_idx, "soc_oper_min_kwh"] = sim["soc_oper_min_kwh"].to_numpy(dtype=float)
            out.loc[original_idx, "soc_oper_max_kwh"] = sim["soc_oper_max_kwh"].to_numpy(dtype=float)

            out.loc[original_idx, "p_sdl_charge_reserved_kw"] = sim["p_sdl_charge_reserved_kw"].to_numpy(dtype=float)
            out.loc[original_idx, "p_sdl_discharge_reserved_kw"] = sim["p_sdl_discharge_reserved_kw"].to_numpy(dtype=float)
            out.loc[original_idx, "p_charge_residual_kw"] = sim["p_charge_residual_kw"].to_numpy(dtype=float)
            out.loc[original_idx, "p_discharge_residual_kw"] = sim["p_discharge_residual_kw"].to_numpy(dtype=float)

            out.loc[original_idx, "p_charge_multiuse_kw"] = sim["p_charge_multiuse_kw"].to_numpy(dtype=float)
            out.loc[original_idx, "p_discharge_multiuse_kw"] = sim["p_discharge_multiuse_kw"].to_numpy(dtype=float)
            out.loc[original_idx, "p_bess_multiuse_kw"] = sim["p_bess_multiuse_kw"].to_numpy(dtype=float)

            out.loc[original_idx, "rev_da_multiuse_chf_h"] = sim["rev_da_multiuse_chf_h"].to_numpy(dtype=float)
            out.loc[original_idx, "rev_id_multiuse_chf_h"] = sim["rev_id_multiuse_chf_h"].to_numpy(dtype=float)
            out.loc[original_idx, "rev_daid_multiuse_chf_h"] = sim["rev_daid_multiuse_chf_h"].to_numpy(dtype=float)
            out.loc[original_idx, "rev_sdl_multiuse_chf_h"] = sim["rev_sdl_multiuse_chf_h"].to_numpy(dtype=float)
            out.loc[original_idx, "rev_multiuse_chf_h"] = sim["rev_multiuse_chf_h"].to_numpy(dtype=float)

            out.loc[original_idx, "daid_dispatch_scale"] = sim["daid_dispatch_scale"].to_numpy(dtype=float)
            out.loc[original_idx, "sdl_product_chosen"] = sim["sdl_product_chosen"].to_numpy()
            out.loc[original_idx, "sdl_product_label"] = sim["sdl_product_label"].to_numpy()
            out.loc[original_idx, "sdl_feasible_any_accepted"] = sim["sdl_feasible_any_accepted"].to_numpy(dtype=int)
            out.loc[original_idx, "sdl_effective_offer_mw_realized"] = sim["sdl_effective_offer_mw_realized"].to_numpy(dtype=float)
            out.loc[original_idx, "sdl_feasible_offer_mw_chosen"] = sim["sdl_feasible_offer_mw_chosen"].to_numpy(dtype=float)

            out.loc[original_idx, "market_state"] = sim["market_state_realized"].to_numpy()
            out.loc[original_idx, "market_state_detail"] = sim["market_state_detail_realized"].to_numpy()
            out.loc[original_idx, "choose_sdl"] = sim["sdl_active_hour"].to_numpy(dtype=int).astype(bool)

            rev_daid_block_realized = float(np.nansum(sim["rev_daid_multiuse_chf_h"].to_numpy(dtype=float)))
            rev_sdl_block_realized = float(np.nansum(sim["rev_sdl_multiuse_chf_h"].to_numpy(dtype=float)))
            rev_multiuse_block_realized = float(np.nansum(sim["rev_multiuse_chf_h"].to_numpy(dtype=float)))

            sdl_any_accepted_feasible_block = int(np.nanmax(sim["sdl_feasible_any_accepted"].to_numpy(dtype=int))) if len(sim) else 0
            dynamic_margin_block = float(np.nansum(bh["dynamic_margin_chf_h"].to_numpy(dtype=float)))
            opp_cost_block = float(np.nansum(bh["opp_cost_chf_h"].to_numpy(dtype=float)))

            sdl_active_block = bool(np.nanmax(sim["sdl_active_hour"].to_numpy(dtype=int)) > 0) if len(sim) else False
            diagnostic_gap_block = float(rev_sdl_block_realized - rev_daid_block_realized)

            block_choice_map[bstart] = bool(sdl_active_block)
            block_gap_map[bstart] = diagnostic_gap_block
            block_margin_map[bstart] = dynamic_margin_block
            block_opp_map[bstart] = opp_cost_block
            block_mode_map[bstart] = "sdl_first_residual_dispatch"
            block_acc_map[bstart] = int(sdl_any_accepted_feasible_block)
            block_cutoff_map[bstart] = brow["decision_cutoff_ts_key"]

            block_rev_sdl_realized_map[bstart] = rev_sdl_block_realized
            block_rev_daid_realized_map[bstart] = rev_daid_block_realized
            block_rev_multiuse_realized_map[bstart] = rev_multiuse_block_realized
            block_soc_start_map[bstart] = soc_block_start_kwh
            block_soc_end_map[bstart] = float(soc_end)
            block_avg_p_sdl_ch_map[bstart] = float(np.nanmean(sim["p_sdl_charge_reserved_kw"].to_numpy(dtype=float))) if len(sim) else 0.0
            block_avg_p_sdl_dis_map[bstart] = float(np.nanmean(sim["p_sdl_discharge_reserved_kw"].to_numpy(dtype=float))) if len(sim) else 0.0
            block_avg_p_res_ch_map[bstart] = float(np.nanmean(sim["p_charge_residual_kw"].to_numpy(dtype=float))) if len(sim) else 0.0
            block_avg_p_res_dis_map[bstart] = float(np.nanmean(sim["p_discharge_residual_kw"].to_numpy(dtype=float))) if len(sim) else 0.0

            soc_cur_kwh = float(soc_end)

        out["choose_sdl_block"] = out[block_key].map(block_choice_map).fillna(False).astype(bool)
        out["decision_gap_block_chf"] = pd.to_numeric(out[block_key].map(block_gap_map), errors="coerce")
        out["dynamic_margin_block_chf"] = pd.to_numeric(out[block_key].map(block_margin_map), errors="coerce")
        out["opp_cost_block_chf"] = pd.to_numeric(out[block_key].map(block_opp_map), errors="coerce")
        out["decision_mode"] = out[block_key].map(block_mode_map)
        out["sdl_any_accepted_block"] = out[block_key].map(block_acc_map).fillna(0).astype(int)
        out["decision_cutoff_block_ts_key"] = pd.to_datetime(out[block_key].map(block_cutoff_map), errors="coerce")
        out["rev_sdl_block_realized_chf"] = pd.to_numeric(out[block_key].map(block_rev_sdl_realized_map), errors="coerce")
        out["rev_daid_block_realized_chf"] = pd.to_numeric(out[block_key].map(block_rev_daid_realized_map), errors="coerce")
        out["rev_multiuse_block_realized_chf"] = pd.to_numeric(out[block_key].map(block_rev_multiuse_realized_map), errors="coerce")
        out["soc_block_start_kwh"] = pd.to_numeric(out[block_key].map(block_soc_start_map), errors="coerce")
        out["soc_block_end_kwh"] = pd.to_numeric(out[block_key].map(block_soc_end_map), errors="coerce")
        out["avg_p_sdl_charge_reserved_block_kw"] = pd.to_numeric(out[block_key].map(block_avg_p_sdl_ch_map), errors="coerce")
        out["avg_p_sdl_discharge_reserved_block_kw"] = pd.to_numeric(out[block_key].map(block_avg_p_sdl_dis_map), errors="coerce")
        out["avg_p_charge_residual_block_kw"] = pd.to_numeric(out[block_key].map(block_avg_p_res_ch_map), errors="coerce")
        out["avg_p_discharge_residual_block_kw"] = pd.to_numeric(out[block_key].map(block_avg_p_res_dis_map), errors="coerce")

        block_df["choose_sdl_block"] = block_df[block_key].map(block_choice_map).fillna(False).astype(bool)
        block_df["decision_gap_block_chf"] = pd.to_numeric(block_df[block_key].map(block_gap_map), errors="coerce")
        block_df["dynamic_margin_block_chf"] = pd.to_numeric(block_df[block_key].map(block_margin_map), errors="coerce")
        block_df["opp_cost_block_chf"] = pd.to_numeric(block_df[block_key].map(block_opp_map), errors="coerce")
        block_df["decision_mode"] = block_df[block_key].map(block_mode_map)
        block_df["sdl_any_accepted_block"] = block_df[block_key].map(block_acc_map).fillna(0).astype(int)
        block_df["rev_sdl_block_realized_chf"] = pd.to_numeric(block_df[block_key].map(block_rev_sdl_realized_map), errors="coerce")
        block_df["rev_daid_block_realized_chf"] = pd.to_numeric(block_df[block_key].map(block_rev_daid_realized_map), errors="coerce")
        block_df["rev_multiuse_block_realized_chf"] = pd.to_numeric(block_df[block_key].map(block_rev_multiuse_realized_map), errors="coerce")
        block_df["soc_block_start_kwh"] = pd.to_numeric(block_df[block_key].map(block_soc_start_map), errors="coerce")
        block_df["soc_block_end_kwh"] = pd.to_numeric(block_df[block_key].map(block_soc_end_map), errors="coerce")
        block_df["avg_p_sdl_charge_reserved_block_kw"] = pd.to_numeric(block_df[block_key].map(block_avg_p_sdl_ch_map), errors="coerce")
        block_df["avg_p_sdl_discharge_reserved_block_kw"] = pd.to_numeric(block_df[block_key].map(block_avg_p_sdl_dis_map), errors="coerce")
        block_df["avg_p_charge_residual_block_kw"] = pd.to_numeric(block_df[block_key].map(block_avg_p_res_ch_map), errors="coerce")
        block_df["avg_p_discharge_residual_block_kw"] = pd.to_numeric(block_df[block_key].map(block_avg_p_res_dis_map), errors="coerce")

    else:
        # ---------------------------------------------------------------------
        # Fallback: alte Blockentscheidung ohne realisierte SOC-Fortschreibung
        # ---------------------------------------------------------------------
        if settings.perfect_forecast_upper_bound_mode and perfect_forecast_detected:
            block_df["decision_gap_block_chf"] = (
                block_df["rev_sdl_block_chf"] - block_df["rev_daid_block_chf"]
            ).astype(float)

            choose_sdl_block = block_df["decision_gap_block_chf"] >= 0.0
            if settings.require_sdl_acceptance:
                choose_sdl_block = choose_sdl_block & (block_df["sdl_any_accepted_block"] == 1)

            block_df["decision_mode"] = "perfect_forecast_upper_bound_blockwise"
        else:
            block_df["decision_gap_block_chf"] = (
                block_df["sdl_score_block_chf"]
                - block_df["daid_score_block_chf"]
                - block_df["dynamic_margin_block_chf"]
            ).astype(float)

            choose_sdl_block = block_df["decision_gap_block_chf"] >= 0.0
            if settings.require_sdl_acceptance:
                choose_sdl_block = choose_sdl_block & (block_df["sdl_any_accepted_block"] == 1)

            block_df["decision_mode"] = "heuristic_blockwise"

        block_df["choose_sdl_block"] = choose_sdl_block.astype(bool)

        block_choice_map = block_df.set_index(block_key)["choose_sdl_block"].to_dict()
        block_gap_map = block_df.set_index(block_key)["decision_gap_block_chf"].to_dict()
        block_margin_map = block_df.set_index(block_key)["dynamic_margin_block_chf"].to_dict()
        block_opp_map = block_df.set_index(block_key)["opp_cost_block_chf"].to_dict()
        block_mode_map = block_df.set_index(block_key)["decision_mode"].to_dict()
        block_acc_map = block_df.set_index(block_key)["sdl_any_accepted_block"].to_dict()
        block_cutoff_map = block_df.set_index(block_key)["decision_cutoff_ts_key"].to_dict()

        out["choose_sdl_block"] = out[block_key].map(block_choice_map).fillna(False).astype(bool)
        out["choose_sdl"] = out["choose_sdl_block"].astype(bool)
        out["decision_gap_block_chf"] = pd.to_numeric(out[block_key].map(block_gap_map), errors="coerce")
        out["dynamic_margin_block_chf"] = pd.to_numeric(out[block_key].map(block_margin_map), errors="coerce")
        out["opp_cost_block_chf"] = pd.to_numeric(out[block_key].map(block_opp_map), errors="coerce")
        out["decision_mode"] = out[block_key].map(block_mode_map)
        out["sdl_any_accepted_block"] = out[block_key].map(block_acc_map).fillna(0).astype(int)
        out["decision_cutoff_block_ts_key"] = pd.to_datetime(out[block_key].map(block_cutoff_map), errors="coerce")

        out["market_state"] = np.where(out["choose_sdl"], "SDL_ONLY", "DA_ID_ONLY")
        out["market_state_detail"] = np.where(out["choose_sdl"], "SDL_ONLY", "DA_ID_ONLY")
        out["p_bess_multiuse_kw"] = np.where(out["choose_sdl"], 0.0, out["p_bess_kw"].fillna(0.0).to_numpy()).astype(float)
        out["rev_sdl_multiuse_chf_h"] = np.where(out["choose_sdl"], out["rev_sdl_total_chf_h"].to_numpy(), 0.0).astype(float)
        out["rev_daid_multiuse_chf_h"] = np.where(out["choose_sdl"], 0.0, out["rev_energy_chf_h"].to_numpy()).astype(float)
        out["rev_multiuse_chf_h"] = np.where(out["choose_sdl"], out["rev_sdl_total_chf_h"].to_numpy(), out["rev_energy_chf_h"].to_numpy()).astype(float)

        if len(out) and pd.notna(out["soc_kwh"].iloc[0]):
            out["soc_kwh_multiuse"] = out["soc_kwh"].to_numpy(dtype=float)
            out["soc_pct_multiuse"] = _infer_soc_pct(out, out["soc_kwh"]).to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # Common outputs / benchmarks
    # -------------------------------------------------------------------------
    out["rev_additiv_chf_h"] = (out["rev_sdl_total_chf_h"] + out["rev_energy_chf_h"]).astype(float)
    out["decision_locked_by"] = f"{int(settings.block_hours)}h_delivery_block_gate_closure_metadata"

    # Für Kompatibilität: konservative Multiuse-Dispatch-Spalten
    out["p_bess_multiuse_kw"] = pd.to_numeric(out["p_bess_multiuse_kw"], errors="coerce").fillna(0.0)
    out["p_charge_multiuse_kw"] = pd.to_numeric(out["p_charge_multiuse_kw"], errors="coerce").fillna(0.0)
    out["p_discharge_multiuse_kw"] = pd.to_numeric(out["p_discharge_multiuse_kw"], errors="coerce").fillna(0.0)
    out["rev_multiuse_chf_h"] = pd.to_numeric(out["rev_multiuse_chf_h"], errors="coerce").fillna(0.0)

    multiuse_total = float(np.nansum(out["rev_multiuse_chf_h"].to_numpy(dtype=float)))
    total_rev_sdl = float(np.nansum(out["rev_sdl_multiuse_chf_h"].to_numpy(dtype=float)))
    total_rev_daid = float(np.nansum(out["rev_daid_multiuse_chf_h"].to_numpy(dtype=float)))

    kpis = {
        "multiuse_total_rev_chf": multiuse_total,
        "multiuse_total_rev_sdl_chf": total_rev_sdl,
        "multiuse_total_rev_daid_residual_chf": total_rev_daid,
        "multiuse_total_rev_additiv_chf": float(np.nansum(out["rev_additiv_chf_h"].to_numpy(dtype=float))),
        "multiuse_share_hours_sdl": float(np.mean(out["choose_sdl"].astype(int).to_numpy())) if len(out) else np.nan,
        "multiuse_share_blocks_sdl": float(np.mean(block_df["choose_sdl_block"].astype(int).to_numpy())) if len(block_df) else np.nan,
        "multiuse_use_dynamic_margin": bool(settings.use_dynamic_margin),
        "multiuse_use_opportunity_cost": bool(settings.use_opportunity_cost),
        "multiuse_lookahead_h": int(settings.lookahead_h),
        "multiuse_require_sdl_acceptance": bool(settings.require_sdl_acceptance),
        "multiuse_block_hours": int(settings.block_hours),
        "multiuse_perfect_forecast_detected": bool(perfect_forecast_detected),
        "multiuse_avg_dynamic_margin_chf_h": float(np.nanmean(out["dynamic_margin_chf_h"].to_numpy(dtype=float))) if len(out) else np.nan,
        "multiuse_avg_opp_cost_chf_h": float(np.nanmean(out["opp_cost_chf_h"].to_numpy(dtype=float))) if len(out) else np.nan,
        "multiuse_avg_dynamic_margin_block_chf": float(np.nanmean(block_df["dynamic_margin_block_chf"].to_numpy(dtype=float))) if len(block_df) else np.nan,
        "multiuse_avg_opp_cost_block_chf": float(np.nanmean(block_df["opp_cost_block_chf"].to_numpy(dtype=float))) if len(block_df) else np.nan,
        "multiuse_pf_sdl_reference_total_chf": np.nan,
        "multiuse_soc_enforcement_active": bool(settings.enforce_realized_soc and batt_resolved is not None),
        "multiuse_batt_constraints_available": bool(batt_resolved is not None),
        "multiuse_reserve_duration_min": float(settings.reserve_duration_minutes),
        "multiuse_allow_partial_sdl_offer": bool(settings.allow_partial_sdl_offer),
        "multiuse_sdl_first_active": bool(settings.enforce_realized_soc and batt_resolved is not None),
        "multiuse_avg_sdl_reserved_charge_kw": float(np.nanmean(out["p_sdl_charge_reserved_kw"].to_numpy(dtype=float))) if len(out) else np.nan,
        "multiuse_avg_sdl_reserved_discharge_kw": float(np.nanmean(out["p_sdl_discharge_reserved_kw"].to_numpy(dtype=float))) if len(out) else np.nan,
        "multiuse_avg_residual_charge_kw": float(np.nanmean(out["p_charge_residual_kw"].to_numpy(dtype=float))) if len(out) else np.nan,
        "multiuse_avg_residual_discharge_kw": float(np.nanmean(out["p_discharge_residual_kw"].to_numpy(dtype=float))) if len(out) else np.nan,
    }

    if batt_resolved is not None:
        kpis.update(
            {
                "multiuse_batt_e_nom_kwh": float(batt_resolved.e_nom_kwh),
                "multiuse_batt_p_ch_max_kw": float(batt_resolved.p_ch_max_kw),
                "multiuse_batt_p_dis_max_kw": float(batt_resolved.p_dis_max_kw),
                "multiuse_batt_soc_min": float(batt_resolved.soc_min),
                "multiuse_batt_soc_max": float(batt_resolved.soc_max),
                "multiuse_batt_soc0": float(batt_resolved.soc0),
            }
        )

    return out, kpis