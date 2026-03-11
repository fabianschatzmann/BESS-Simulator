# core/multiuse.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd


@dataclass
class MultiuseSettings:
    """
    Multiuse Decision Layer:
    - Entscheidung zwischen SDL und DA/ID pro 4h-Lieferblock
    - Block-Locking statt Tages-Locking
    - dynamische Marge + Opportunitätskosten
    - optionaler Perfect-Forecast-Upper-Bound-Modus

    require_sdl_acceptance:
        Wenn True, darf SDL nur gewählt werden, wenn accepted==1 vorliegt.

    use_dynamic_margin:
        Aktiviert zeitvariable Marge aus Spread-Volatilität, SoC-Randnähe,
        ID-Forecast-Unsicherheit, SDL-Aktivierungsannahmen und Degradation.

    use_opportunity_cost:
        Aktiviert Lookahead-basierte Opportunitätskosten des freien DA/ID-Einsatzes.

    lookahead_h:
        Horizont für Opportunitätskosten bzw. Volatilitätsbetrachtung.

    perfect_forecast_upper_bound_mode:
        Wenn True und DA+ID Perfect Forecast erkannt wird
        (price_da_fc == price_da und price_id_fc == price_id),
        dann wird blockweise direkt max(SDL, DA_ID) gewählt.
        Zusätzlich wird ein PF-Sicherheitsnetz angewendet:
        Multiuse-PF darf nicht kleiner sein als der zulässige SDL-Referenzpfad
        unter denselben Block-/Acceptance-Regeln.

    block_hours:
        Lieferblocklänge in Stunden. Für deinen Fall: 4h.

    tz_local:
        Lokale Zeitzone für die Blockbildung.
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
    Keine fixe Basis-Marge mehr.
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
    Zunächst als abgezinste Vorwärtssumme positiver DA/ID-Erlöse.
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
    Erzeugt lokale 4h-Lieferblöcke:
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


def _apply_pf_upper_bound_safety_net(
    out: pd.DataFrame,
    block_df: pd.DataFrame,
    block_key: str,
    settings: MultiuseSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    PF-Sicherheitsnetz:
    Wenn Perfect Forecast erkannt ist, darf Multiuse nicht kleiner sein als
    ein zulässiger SDL-Referenzpfad unter denselben Block-/Acceptance-Regeln.

    Vorgehen:
    - compute mixed_total from current block choices
    - compute sdl_ref_total as 'alle zulässigen Blöcke -> SDL, sonst DA/ID'
    - wenn sdl_ref_total > mixed_total:
        überschreibe choose_sdl_block mit dem SDL-Referenzpfad
    """
    if block_df.empty:
        return out, block_df

    allowed_sdl_block = pd.Series(True, index=block_df.index)
    if settings.require_sdl_acceptance:
        allowed_sdl_block = block_df["sdl_any_accepted_block"].fillna(0).astype(int) == 1

    mixed_total = float(
        np.nansum(
            np.where(
                block_df["choose_sdl_block"].astype(bool).to_numpy(),
                block_df["rev_sdl_block_chf"].to_numpy(dtype=float),
                block_df["rev_daid_block_chf"].to_numpy(dtype=float),
            )
        )
    )

    sdl_ref_choose = allowed_sdl_block.astype(bool)
    sdl_ref_total = float(
        np.nansum(
            np.where(
                sdl_ref_choose.to_numpy(),
                block_df["rev_sdl_block_chf"].to_numpy(dtype=float),
                block_df["rev_daid_block_chf"].to_numpy(dtype=float),
            )
        )
    )

    block_df["pf_upper_bound_mixed_total_chf"] = mixed_total
    block_df["pf_upper_bound_sdl_reference_total_chf"] = sdl_ref_total

    if sdl_ref_total > mixed_total + 1e-9:
        block_df["choose_sdl_block"] = sdl_ref_choose.astype(bool)
        block_df["decision_mode"] = "perfect_forecast_upper_bound_vs_sdl_reference"

        # optional: gap mit Referenzlogik überschreiben
        block_df["decision_gap_block_chf"] = (
            block_df["rev_sdl_block_chf"] - block_df["rev_daid_block_chf"]
        ).astype(float)

    return out, block_df


def build_multiuse_priority_sdl(
    *,
    results_da_id: pd.DataFrame,
    sdl_timeseries: pd.DataFrame,
    settings: MultiuseSettings,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Baut eine Multiuse-Zeitreihe (stündlich, ts_key) mit Market-State pro Stunde.

    Neue Logik:
    - Entscheidung pro 4h-Lieferblock (lokal, Europe/Zurich)
    - nicht mehr pro Liefertag
    - optionaler PF-Upper-Bound-Modus
    """
    r = _ensure_ts_key(results_da_id)
    s = _ensure_ts_key(sdl_timeseries)

    # ---- DA/ID Revenue Kandidaten (stündlich)
    rev_da_col = _pick_first_col(r, ["rev_da_chf_h", "rev_da_chf", "rev_da_chf_per_h"])
    rev_id_col = _pick_first_col(r, ["rev_id_inc_chf_h", "rev_id_inc_chf", "rev_id_chf_h"])
    p_bess_col = _pick_first_col(r, ["p_bess_kw", "p_bess", "p_net_kw", "p_total_kw", "p_kw"])
    soc_col = _pick_first_col(r, ["soc_kwh", "soc_kwh_end", "soc_kwh_t"], required=False)

    rev_da = _as_float_series(r, rev_da_col, 0.0)
    rev_id = _as_float_series(r, rev_id_col, 0.0)
    rev_energy = (rev_da + rev_id).astype(float)

    p_bess = _as_float_series(r, p_bess_col, 0.0)
    soc_kwh = _as_float_series(r, soc_col, np.nan)

    price_da_fc = _as_float_series(r, _pick_first_col(r, ["price_da_fc"]), 0.0)
    price_id_fc = _as_float_series(r, _pick_first_col(r, ["price_id_fc"]), 0.0)
    price_da = _as_float_series(r, _pick_first_col(r, ["price_da"]), 0.0)
    price_id = _as_float_series(r, _pick_first_col(r, ["price_id"]), 0.0)

    soc_pct_col = _pick_first_col(r, ["soc_pct", "soc"], required=False)
    soc_pct = _as_float_series(r, soc_pct_col, np.nan)

    # ---- SDL Revenue Kandidaten (stündlich)
    rev_sdl_total_col = _pick_first_col(s, ["sdl_total_rev_chf_h", "sdl_total_rev_chf_per_h", "sdl_total_rev_chf"])
    rev_sdl = _as_float_series(s, rev_sdl_total_col, 0.0)

    acc_cols = [c for c in ["srl_up_accepted", "srl_down_accepted", "prl_sym_accepted"] if c in s.columns]

    if not acc_cols:
        sdl_any_accepted = pd.Series(1, index=s.index, dtype=int)
    else:
        acc_any = np.zeros(len(s), dtype=int)
        for c in acc_cols:
            acc_any = np.maximum(acc_any, _as_int_series(s, c, 0).to_numpy(dtype=int))
        sdl_any_accepted = pd.Series(acc_any, index=s.index, dtype=int)

    cutoff_cols = [c for c in ["srl_up_cutoff_ts_key", "srl_down_cutoff_ts_key", "prl_sym_cutoff_ts_key"] if c in s.columns]

    # Basis-Frame über ts_key
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
            "rev_sdl_chf_h": rev_sdl.to_numpy(dtype=float),
            "sdl_any_accepted": sdl_any_accepted.to_numpy(dtype=int),
        }
    )

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
    out["rev_sdl_chf_h"] = out["rev_sdl_chf_h"].fillna(0.0).astype(float)
    out["sdl_any_accepted"] = out["sdl_any_accepted"].fillna(0).astype(int)

    # ------------------------------------------------------------
    # 4h-Blockbildung lokal
    # ------------------------------------------------------------
    out = _add_4h_block_keys(out, tz_local=settings.tz_local, block_hours=settings.block_hours)

    # ------------------------------------------------------------
    # Dynamische Marge + Opportunitätskosten (stündlich)
    # ------------------------------------------------------------
    comps = _compute_dynamic_margin_components(out, settings)
    for c in comps.columns:
        out[c] = comps[c].to_numpy()

    out["opp_cost_chf_h"] = _compute_opportunity_cost(out, settings).to_numpy(dtype=float)

    out["sdl_score_chf_h"] = (
        out["rev_sdl_chf_h"]
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

    # ------------------------------------------------------------
    # Blockaggregation statt Tagesaggregation
    # ------------------------------------------------------------
    block_key = "delivery_block_start_local"
    grp = out.groupby(block_key, dropna=False)

    block_df = grp.agg(
        delivery_date_local=("delivery_date_local", "first"),
        delivery_block_end_local=("delivery_block_end_local", "first"),
        delivery_block_hour_local=("delivery_block_hour_local", "first"),
        delivery_block_label=("delivery_block_label", "first"),
        rev_sdl_block_chf=("rev_sdl_chf_h", "sum"),
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

    if settings.perfect_forecast_upper_bound_mode and perfect_forecast_detected:
        # PF-Upper-Bound: blockweise max(SDL, DA_ID)
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

    # ------------------------------------------------------------
    # PF-Sicherheitsnetz: Multiuse-PF darf nicht < zulässiger SDL-Referenz sein
    # ------------------------------------------------------------
    if settings.perfect_forecast_upper_bound_mode and perfect_forecast_detected:
        out, block_df = _apply_pf_upper_bound_safety_net(
            out=out,
            block_df=block_df,
            block_key=block_key,
            settings=settings,
        )

    # Zurück auf Stunden mappen
    block_choice_map = block_df.set_index(block_key)["choose_sdl_block"].to_dict()
    block_gap_map = block_df.set_index(block_key)["decision_gap_block_chf"].to_dict()
    block_margin_map = block_df.set_index(block_key)["dynamic_margin_block_chf"].to_dict()
    block_opp_map = block_df.set_index(block_key)["opp_cost_block_chf"].to_dict()
    block_mode_map = block_df.set_index(block_key)["decision_mode"].to_dict()
    block_acc_map = block_df.set_index(block_key)["sdl_any_accepted_block"].to_dict()
    block_cutoff_map = block_df.set_index(block_key)["decision_cutoff_ts_key"].to_dict()

    out["choose_sdl_block"] = out[block_key].map(block_choice_map).fillna(False).astype(bool)
    out["decision_gap_block_chf"] = pd.to_numeric(out[block_key].map(block_gap_map), errors="coerce")
    out["dynamic_margin_block_chf"] = pd.to_numeric(out[block_key].map(block_margin_map), errors="coerce")
    out["opp_cost_block_chf"] = pd.to_numeric(out[block_key].map(block_opp_map), errors="coerce")
    out["decision_mode"] = out[block_key].map(block_mode_map)
    out["sdl_any_accepted_block"] = out[block_key].map(block_acc_map).fillna(0).astype(int)
    out["decision_cutoff_block_ts_key"] = pd.to_datetime(out[block_key].map(block_cutoff_map), errors="coerce")

    out["choose_sdl"] = out["choose_sdl_block"].astype(bool)

    # Market state fürs Dashboard
    out["market_state"] = np.where(out["choose_sdl"], "SDL", "DA_ID")

    # Multiuse konservativ: wenn SDL gewählt, DA/ID blockieren
    out["p_bess_multiuse_kw"] = np.where(
        out["choose_sdl"],
        0.0,
        out["p_bess_kw"].fillna(0.0).to_numpy()
    ).astype(float)

    out["rev_multiuse_chf_h"] = np.where(
        out["choose_sdl"],
        out["rev_sdl_chf_h"].to_numpy(),
        out["rev_energy_chf_h"].to_numpy()
    ).astype(float)

    # Additiver Benchmark
    out["rev_additiv_chf_h"] = (out["rev_sdl_chf_h"] + out["rev_energy_chf_h"]).astype(float)

    # Kompatibilität / Erklärung
    out["decision_locked_by"] = f"{int(settings.block_hours)}h_delivery_block_gate_closure"

    multiuse_total = float(np.nansum(out["rev_multiuse_chf_h"].to_numpy()))

    # Zulässiger SDL-Referenzpfad unter denselben Block-/Acceptance-Regeln
    sdl_ref_total = float(
        np.nansum(
            np.where(
                block_df["choose_sdl_block"].astype(bool).to_numpy(),
                block_df["rev_sdl_block_chf"].to_numpy(dtype=float),
                block_df["rev_daid_block_chf"].to_numpy(dtype=float),
            )
        )
    ) if (settings.perfect_forecast_upper_bound_mode and perfect_forecast_detected) else np.nan

    kpis = {
        "multiuse_total_rev_chf": multiuse_total,
        "multiuse_total_rev_additiv_chf": float(np.nansum(out["rev_additiv_chf_h"].to_numpy())),
        "multiuse_share_hours_sdl": float(np.mean(out["choose_sdl"].astype(int).to_numpy())) if len(out) else np.nan,
        "multiuse_share_blocks_sdl": float(np.mean(block_df["choose_sdl_block"].astype(int).to_numpy())) if len(block_df) else np.nan,
        "multiuse_use_dynamic_margin": bool(settings.use_dynamic_margin),
        "multiuse_use_opportunity_cost": bool(settings.use_opportunity_cost),
        "multiuse_lookahead_h": int(settings.lookahead_h),
        "multiuse_require_sdl_acceptance": bool(settings.require_sdl_acceptance),
        "multiuse_block_hours": int(settings.block_hours),
        "multiuse_perfect_forecast_detected": bool(perfect_forecast_detected),
        "multiuse_avg_dynamic_margin_chf_h": float(np.nanmean(out["dynamic_margin_chf_h"].to_numpy())) if len(out) else np.nan,
        "multiuse_avg_opp_cost_chf_h": float(np.nanmean(out["opp_cost_chf_h"].to_numpy())) if len(out) else np.nan,
        "multiuse_avg_dynamic_margin_block_chf": float(np.nanmean(block_df["dynamic_margin_block_chf"].to_numpy())) if len(block_df) else np.nan,
        "multiuse_avg_opp_cost_block_chf": float(np.nanmean(block_df["opp_cost_block_chf"].to_numpy())) if len(block_df) else np.nan,
        "multiuse_pf_sdl_reference_total_chf": sdl_ref_total,
    }

    return out, kpis