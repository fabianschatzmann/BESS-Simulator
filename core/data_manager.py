# data_manager.py
import io
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_data_cached():
    """
    Default-Daten (synthetisch):
    - price_chf_mwh
    - pv_kw
    - load_kw
    - t_index, day_index, step_in_day

    Später ersetzt du das durch deine echten CSV/Parquet-Loader.
    """
    n_days = 365
    dt_steps_per_day = 96  # 15-min
    n = n_days * dt_steps_per_day

    rng = np.random.default_rng(42)

    # Synthetic price [CHF/MWh]
    base = 80 + 30 * np.sin(np.linspace(0, 10 * np.pi, n))
    noise = rng.normal(0, 10, n)
    price = (base + noise).clip(0)

    # Synthetic PV [kW]
    day_phase = np.tile(np.linspace(0, 2 * np.pi, dt_steps_per_day, endpoint=False), n_days)
    pv = (np.sin(day_phase - np.pi / 2).clip(0)) * 200  # peak 200 kW

    # Synthetic load [kW]
    load = 150 + 30 * np.sin(day_phase) + rng.normal(0, 5, n)
    load = load.clip(50)

    df = pd.DataFrame(
        {
            "price_chf_mwh": price,
            "pv_kw": pv,
            "load_kw": load,
            "t_index": np.arange(n),
            "day_index": np.repeat(np.arange(n_days), dt_steps_per_day),
            "step_in_day": np.tile(np.arange(dt_steps_per_day), n_days),
        }
    )
    return {"timeseries": df, "meta": {"dt_steps_per_day": dt_steps_per_day, "n_days": n_days}}


def parse_load_profile_upload(uploaded_file):
    """
    Liest CSV oder XLSX ein und liefert:
    - df_preview (DataFrame)
    - info dict: columns, n_rows, guess_load_col, raw_df
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        content = uploaded_file.getvalue()
        # Robust: erst ; dann , probieren
        try:
            df = pd.read_csv(io.BytesIO(content), sep=";")
            if df.shape[1] == 1:
                df = pd.read_csv(io.BytesIO(content), sep=",")
        except Exception:
            df = pd.read_csv(io.BytesIO(content), sep=",")
    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Nur CSV oder XLSX werden unterstützt.")

    cols = list(df.columns)

    guess = None
    for c in cols:
        cl = str(c).lower()
        if ("load" in cl) or ("last" in cl) or ("verbrauch" in cl) or ("power" in cl) or ("kw" in cl):
            guess = c
            break

    info = {
        "n_rows": int(len(df)),
        "columns": cols,
        "guess_load_col": guess,
        "raw_df": df,
    }
    return df, info


def apply_uploaded_load_if_present(data: dict, uploaded_load_info: dict, timestep_minutes: int) -> dict:
    """
    Wenn Upload vorhanden: überschreibt data["timeseries"]["load_kw"].
    Erwartung in dieser Version: Upload-Spalte ist Leistung in kW pro Zeitschritt.
    """
    if not uploaded_load_info:
        return data

    df_up = uploaded_load_info["raw_df"].copy()

    load_col = uploaded_load_info.get("selected_load_col") or uploaded_load_info.get("guess_load_col")
    if not load_col or load_col not in df_up.columns:
        raise ValueError("Keine Load-Spalte gewählt/gefunden. Bitte im GUI auswählen.")

    series = df_up[load_col].astype(float).to_numpy()

    df = data["timeseries"].copy()
    n = len(df)

    if len(series) < n:
        raise ValueError(f"Upload hat zu wenig Zeilen ({len(series)}), benötigt mindestens {n} (für 365 Tage à 15 min).")

    series = series[:n]
    df["load_kw"] = series

    data2 = {**data, "timeseries": df}
    return data2
