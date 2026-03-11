# Simulate.py
import time
import numpy as np
import pandas as pd

from core.schemas import ScenarioParams


def run_backtest(params: ScenarioParams, data: dict) -> dict:
    """
    Dummy-Simulator (noch ohne Pyomo/MPC):
    - erzeugt plausible Zeitreihen (SOC, p_batt, p_grid) ohne echte Optimierung
    - liefert KPIs + Revenue breakdown

    -> später ersetzt du die Dispatch-Logik durch deinen Optimierer.
    """
    t0 = time.time()
    df = data["timeseries"].copy()

    dt_h = params.optimizer["timestep_minutes"] / 60.0
    run_days = params.run_days
    steps_per_day = data["meta"]["dt_steps_per_day"]
    n = run_days * steps_per_day
    df = df.iloc[:n].copy()

    # Battery params
    E = params.battery["E_kWh"]
    Pch = params.battery["P_ch_max_kW"]
    Pdis = params.battery["P_dis_max_kW"]
    soc_min = params.battery["soc_min"]
    soc_max = params.battery["soc_max"]

    soc = np.zeros(n)
    soc[0] = np.clip(params.battery["soc_init"], soc_min, soc_max)

    price = df["price_chf_mwh"].to_numpy()
    pv = df["pv_kw"].to_numpy()
    load = df["load_kw"].to_numpy()

    # Simple heuristic (placeholder)
    # sign convention: p_batt > 0 discharge, p_batt < 0 charge
    p_batt = np.zeros(n)
    for i in range(1, n):
        surplus = pv[i] - load[i]

        window = price[max(0, i - steps_per_day) : i + 1]
        p_low = np.quantile(window, 0.25)
        p_high = np.quantile(window, 0.75)

        action = 0.0
        if params.markets["self_consumption"] and surplus > 10 and soc[i - 1] < soc_max:
            action = -min(Pch, surplus)
        elif params.markets["day_ahead"] and price[i] < p_low and soc[i - 1] < soc_max:
            action = -0.5 * Pch
        elif params.markets["day_ahead"] and price[i] > p_high and soc[i - 1] > soc_min:
            action = 0.5 * Pdis

        # enforce SOC bounds (simplified)
        soc_next = soc[i - 1] + (-action) * dt_h / E  # charging increases SOC (action negative)
        soc_next = np.clip(soc_next, soc_min, soc_max)
        delta = soc_next - soc[i - 1]
        action = -(delta * E / dt_h)

        p_batt[i] = np.clip(action, -Pch, Pdis)
        soc[i] = soc_next

    # Grid power: import positive, export negative
    p_grid = load - pv - p_batt

    df["soc"] = soc
    df["p_batt_kw"] = p_batt
    df["p_grid_kw"] = p_grid

    # -------------------------
    # Revenues (vereinfachte Modelle)
    # -------------------------

    # Day-Ahead arbitrage revenue
    energy_mwh = p_batt * dt_h / 1000.0
    rev_da = float(np.sum(price * energy_mwh))  # CHF

    # Import cost model
    markup_chf_per_kWh = params.tariffs["import_markup_rp_kWh"] / 100.0  # CHF/kWh
    import_price_chf_kWh = (price / 1000.0) + markup_chf_per_kWh

    grid_import_kWh = np.clip(p_grid, 0, None) * dt_h
    cost_import = float(np.sum(import_price_chf_kWh * grid_import_kWh))

    # Baseline without battery
    p_grid0 = load - pv
    grid_import0_kWh = np.clip(p_grid0, 0, None) * dt_h
    cost_import0 = float(np.sum(import_price_chf_kWh * grid_import0_kWh))

    # Export remuneration (feed-in) for self-consumption modeling (simplified)
    feed_in_chf_per_kWh = params.tariffs.get("export_feed_in_rp_kWh", 6.0) / 100.0
    grid_export_kWh = np.clip(-p_grid, 0, None) * dt_h
    revenue_export = float(np.sum(feed_in_chf_per_kWh * grid_export_kWh))

    grid_export0_kWh = np.clip(-(p_grid0), 0, None) * dt_h
    revenue_export0 = float(np.sum(feed_in_chf_per_kWh * grid_export0_kWh))

    # Self-consumption value = (baseline import cost - new import cost) + (new export - baseline export)
    # (je nach Tariflogik könnt ihr das später verfeinern)
    rev_self = (cost_import0 - cost_import) + (revenue_export - revenue_export0)

    # Peak shaving
    peak0 = float(np.max(p_grid0))
    peak1 = float(np.max(p_grid))
    peak_red = peak0 - peak1

    dc = params.peak_shaving.get("demand_charge_chf_per_kw_month", 0.0)
    months = max(1.0, params.run_days / 30.0)
    rev_peak = max(0.0, peak_red) * dc * months

    revenue_total = 0.0
    breakdown = []

    if params.markets["day_ahead"]:
        breakdown.append({"market": "Day-Ahead", "revenue_chf": rev_da})
        revenue_total += rev_da
    if params.markets["self_consumption"]:
        breakdown.append({"market": "Self-consumption", "revenue_chf": float(rev_self)})
        revenue_total += float(rev_self)
    if params.markets["peak_shaving"]:
        breakdown.append({"market": "Peak shaving", "revenue_chf": float(rev_peak)})
        revenue_total += float(rev_peak)

    # KPIs
    throughput_kWh = float(np.sum(np.abs(p_batt)) * dt_h)
    cycles = throughput_kWh / (2 * params.battery["E_kWh"] + 1e-9)

    runtime_s = time.time() - t0

    kpis = {
        "annual_revenue_chf": float(revenue_total),
        "cycles_per_year": float(cycles * (365 / max(1, params.run_days))),
        "peak_reduction_kw": float(peak_red),
        "effective_rte": float(params.battery["eta_ch"] * params.battery["eta_dis"]),
        "constraint_violations": 0,
        "runtime_s": float(runtime_s),
    }

    # Top/Worst days by daily DA revenue proxy
    df_daily = (
        df.groupby("day_index")
        .apply(lambda g: float(np.sum(g["price_chf_mwh"].to_numpy() * (g["p_batt_kw"].to_numpy() * dt_h / 1000.0))))
        .reset_index(name="daily_rev_chf")
    )
    top = df_daily.sort_values("daily_rev_chf", ascending=False).head(10).to_dict("records")
    worst = df_daily.sort_values("daily_rev_chf", ascending=True).head(10).to_dict("records")

    return {
        "kpis": kpis,
        "revenue_breakdown": breakdown,
        "top_days": top,
        "worst_days": worst,
        "timeseries": df,
    }
