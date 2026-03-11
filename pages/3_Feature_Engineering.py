# pages/4_Feature_Engineering.py
import os
import io
import sys
import pandas as pd
import streamlit as st

# Projekt-Root auf sys.path setzen (wichtig bei der Ordnerstruktur)
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.scenario_store import load_parquet, save_parquet, load_config
from core.feature_engineering import (
    build_feature_frame_multi,
    add_market_block_keys,
    coverage_summary,
)

st.set_page_config(page_title="Feature Engineering", layout="wide")
st.title("Feature Engineering")
st.caption(
    "Erstellt automatisch eine leak-free Feature-Tabelle aus dem Master. "
    "Targets und Standard-Features werden je nach Marktmodus automatisch gewählt."
)

# ----------------------------
# Szenario-Config laden (SSOT)
# ----------------------------
sname = st.session_state.get("scenario_name", "BaseCase_2025")
cfg = st.session_state.get("scenario_config") or load_config(sname) or {}
st.session_state["scenario_config"] = cfg

market_mode = str(cfg.get("market_mode", "CUSTOM") or "CUSTOM")

# ----------------------------
# Master laden: session_state ODER scenario parquet fallback ODER root fallback
# ----------------------------
def load_master_fallback():
    # 1) session_state aus Import Wizard
    m = st.session_state.get("master_2025")
    if isinstance(m, pd.DataFrame) and not m.empty:
        return m, "session_state.master_2025 (Import-Assistent)"

    # 2) session_state master (Legacy-Key)
    m = st.session_state.get("master")
    if isinstance(m, pd.DataFrame) and not m.empty:
        return m, "session_state.master"

    # 3) Szenario-Store
    m = load_parquet(sname, "master")
    if isinstance(m, pd.DataFrame) and not m.empty:
        return m, f"scenario_store: data/scenarios/{sname}/master.parquet"

    # 4) Root-Fallback
    cand = os.path.join(PROJECT_ROOT, "data", "master_2025_hourly.parquet")
    if os.path.exists(cand):
        return pd.read_parquet(cand), f"project_root: {cand}"

    return None, None


def infer_targets_from_market_mode(master_df: pd.DataFrame, mode: str) -> list[str]:
    mode = str(mode or "").upper()

    if mode == "DA_ONLY":
        targets = ["price_da"]
    elif mode in ("DA_PLUS_ID", "DA_ID_SDL_MULTIUSE"):
        targets = ["price_da", "price_id"]
    elif mode == "SDL_ONLY":
        targets = []
    else:
        # Fallback robust halten
        targets = []
        if "price_da" in master_df.columns:
            targets.append("price_da")
        if "price_id" in master_df.columns:
            targets.append("price_id")

    targets = [c for c in targets if c in master_df.columns]
    return targets


def infer_feature_cols(master_df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    preferred_order = [
        "load_fc_da", "load_fc_id",
        "pv_fc_da", "pv_fc_id",
        "wind_fc_da", "wind_fc_id",
        "load_fc", "pv_fc", "wind_fc",
    ]

    feature_cols = [c for c in preferred_order if c in master_df.columns]

    # robust: keine Targets, keine Istwert-Spalten
    feature_cols = [c for c in feature_cols if c not in set(target_cols)]
    feature_cols = [c for c in feature_cols if not str(c).endswith("_act")]

    return feature_cols


def describe_market_mode(mode: str) -> str:
    mode = str(mode or "").upper()
    if mode == "DA_ONLY":
        return "Nur Day-Ahead: Target = price_da"
    if mode == "DA_PLUS_ID":
        return "Day-Ahead + Intraday: Targets = price_da, price_id"
    if mode == "DA_ID_SDL_MULTIUSE":
        return "Multiuse: Feature Engineering für DA+ID, SDL kommt später im Runner hinzu"
    if mode == "SDL_ONLY":
        return "SDL_ONLY: Für reine SDL-Berechnung werden aktuell keine Preis-Features für DA/ID benötigt"
    return f"Benutzerdefiniert / Fallback: Marktmodus = {mode}"


master, master_src = load_master_fallback()
if master is None:
    st.error(
        "Kein Master gefunden. Bitte zuerst im Datenimport-Assistenten einen Master erstellen "
        "und ins Szenario speichern (data/scenarios/<scenario>/master.parquet)."
    )
    st.stop()

st.success(f"Master geladen ({master_src}). Zeilen: {len(master)} | Spalten: {len(master.columns)}")

with st.expander("Master-Vorschau (Head)", expanded=False):
    st.dataframe(master.head(200), use_container_width=True, height=520)

st.divider()

# ----------------------------
# Perfect Forecast (Upper Bound) – UI speichert nur pf_settings.parquet
# ----------------------------
st.subheader("Perfect Forecast (Upper Bound, optional)")
st.caption(
    "Diese Option ist bewusst NICHT leak-free und dient als Upper Bound (theoretisches Maximum) "
    "unter den aktuellen Constraints/Gate-Closure-Regeln. "
    "Wenn deaktiviert, ändert sich am bestehenden ML-/Forecast-Prozess nichts."
)

cPF1, cPF2, cPF3, cPF4 = st.columns(4)
with cPF1:
    pf_da = st.toggle(
        "Perfect Forecast: Day-Ahead",
        value=False,
        key="pf_da_toggle",
        help="Setzt später im Runner (bei aktivem PF) price_da_fc := price_da. Dient als theoretischer Upper Bound.",
    )
with cPF2:
    pf_id = st.toggle(
        "Perfect Forecast: Intraday",
        value=False,
        key="pf_id_toggle",
        help="Setzt später im Runner (bei aktivem PF) price_id_fc := price_id. ID bleibt inkrementell vs. DA.",
    )
with cPF3:
    pf_da_h = st.slider(
        "PF-Horizont Day-Ahead [h] (optional)",
        1, 168, 24, 1,
        help="Wird aktuell nur gespeichert (für spätere Varianten mit Cutoff/Horizont-Logik).",
        key="pf_da_h_slider",
    )
with cPF4:
    pf_id_h = st.slider(
        "PF-Horizont Intraday [h] (optional)",
        1, 168, 24, 1,
        help="Wird aktuell nur gespeichert (für spätere Varianten mit Cutoff/Horizont-Logik).",
        key="pf_id_h_slider",
    )

pf_settings = pd.DataFrame([{
    "perfect_forecast_da": int(bool(pf_da)),
    "perfect_forecast_id": int(bool(pf_id)),
    "pf_horizon_da_h": int(pf_da_h),
    "pf_horizon_id_h": int(pf_id_h),
}])
save_parquet(sname, "pf_settings", pf_settings)
st.info(f"PF-Einstellungen gespeichert: data/scenarios/{sname}/pf_settings.parquet")

st.divider()

# ----------------------------
# Automatische Standard-Logik
# ----------------------------
st.subheader("Standard-Feature-Set (automatisch)")

targets_default = infer_targets_from_market_mode(master, market_mode)
feature_cols_default = infer_feature_cols(master, targets_default)
ts_col = "ts" if "ts" in master.columns else None

add_calendar = True
add_price_history = True
exclude_act_cols = True
lags = [1, 24, 168]
rolls = [24, 168]
drop_missing_targets = True
drop_missing_features = True

if ts_col is None:
    st.error("Master enthält keine Spalte 'ts'. Bitte Datenimport prüfen.")
    st.stop()

if market_mode == "SDL_ONLY":
    st.warning(
        "Das Szenario steht auf SDL_ONLY. Für SDL werden in der Regel keine DA/ID-Feature-Tabellen benötigt. "
        "Diese Seite ist primär für DA_ONLY, DA_PLUS_ID und DA_ID_SDL_MULTIUSE relevant."
    )

if not targets_default and market_mode != "SDL_ONLY":
    st.error(
        "Für den aktuellen Marktmodus konnten keine gültigen Target-Spalten aus dem Master abgeleitet werden. "
        "Bitte Master und market_mode prüfen."
    )
    st.stop()

info1, info2 = st.columns(2)
with info1:
    st.markdown(f"**Szenario:** `{sname}`")
    st.markdown(f"**Marktmodus:** `{market_mode}`")
    st.markdown(f"**Interpretation:** {describe_market_mode(market_mode)}")

with info2:
    st.markdown(f"**Zeitspalte:** `{ts_col}`")
    st.markdown(f"**Targets:** `{', '.join(targets_default) if targets_default else 'keine'}`")
    st.markdown(f"**Exogene Standard-Features:** `{', '.join(feature_cols_default) if feature_cols_default else 'keine'}`")

st.caption(
    "Fest hinterlegte Standards: Kalenderfeatures aktiv, Preis-Historie aktiv, "
    "Lags = 1h / 24h / 168h, Rolling = 24h / 168h, *_act-Spalten ausgeschlossen, "
    "fehlende Targets/Features werden entfernt."
)

# ----------------------------
# Optionaler Debug-Bereich
# ----------------------------
with st.expander("Erweiterte Debug-/Analyseoptionen (optional)", expanded=False):
    st.markdown("**Technische Markt-/Gate-Closure-Keys**")
    st.caption(
        "Diese Keys sind nur für spätere Analysen/Debug gedacht. "
        "Sie bilden NICHT direkt die reale Handelslogik der Märkte ab. "
        "Die operative Gate-Closure-Logik für Intraday wird im Runner umgesetzt."
    )

    colM1, colM2, colM3, colM4 = st.columns(4)

    with colM1:
        market_profile = st.selectbox(
            "Technisches Marktprofil",
            ["DA+ID (technischer Tagesblock)", "Nur Day-Ahead (technischer Tagesblock)", "Nur Intraday (technischer Stundenblock)", "Benutzerdefiniert"],
            index=0,
            help=(
                "Nur technisches Keying für Analysen/Debug. "
                "Nicht mit den realen Marktprodukten oder Gate-Closures verwechseln."
            ),
        )

    with colM2:
        if market_profile in ("DA+ID (technischer Tagesblock)", "Nur Day-Ahead (technischer Tagesblock)"):
            block_hours_default = 24
        elif market_profile == "Nur Intraday (technischer Stundenblock)":
            block_hours_default = 1
        else:
            block_hours_default = 24

        block_hours = st.number_input(
            "Technische Gruppierung [h]",
            min_value=1,
            max_value=168,
            value=int(block_hours_default),
            step=1,
            help="Nur für Analyse-/Debug-Keys. 24h bedeutet hier ein technischer Tagesblock, nicht ein realer Angebotsblock.",
        )

    with colM3:
        if market_profile in ("DA+ID (technischer Tagesblock)", "Nur Day-Ahead (technischer Tagesblock)"):
            gco_default = 12.0
        elif market_profile == "Nur Intraday (technischer Stundenblock)":
            gco_default = 1.0
        else:
            gco_default = 12.0

        gate_closure_offset_h = st.number_input(
            "Technischer Gate-Closure-Offset [h]",
            min_value=0.0,
            max_value=72.0,
            value=float(gco_default),
            step=0.5,
            help=(
                "Nur technischer Hilfswert für zusätzliche Analyse-Keys. "
                "Die echte Dispatch-/Tradability-Logik wird dadurch nicht gesteuert."
            ),
        )

    with colM4:
        add_block_keys = st.checkbox(
            "Zusätzliche Analyse-Keys schreiben",
            value=False,
            help="Schreibt optionale technische Block-/Gate-Closure-Keys in die Feature-Tabelle.",
        )

    st.caption(
        "Empfehlung: Für den normalen Betrieb deaktiviert lassen. "
        "Diese Keys sind nicht nötig, damit Forecast und Dispatch korrekt funktionieren."
    )

st.divider()

# ----------------------------
# Persistenz / Export
# ----------------------------
st.subheader("Persistenz & Export")

colX, colY = st.columns([1, 1])
with colX:
    out_path_parquet = st.text_input(
        "Optionaler zusätzlicher Exportpfad (Parquet)",
        value="data/features_2025.parquet",
        help="Optionaler zusätzlicher Exportpfad (projekt-root relativ).",
    )

with colY:
    st.markdown("**Speicherziel im Szenario (Standard):**")
    st.code(f"data/scenarios/{sname}/features.parquet")

build_btn = st.button(
    "🧱 Features erstellen",
    type="primary",
    help="Erstellt features.parquet leak-free und speichert ins Szenario.",
)

# ----------------------------
# Build
# ----------------------------
if build_btn:
    try:
        if market_mode == "SDL_ONLY":
            st.warning(
                "Du erstellst gerade Features trotz SDL_ONLY. Das ist erlaubt, "
                "aber für reine SDL-Läufe meist nicht notwendig."
            )

        feats = build_feature_frame_multi(
            master=master,
            ts_col=ts_col,
            target_cols=list(targets_default),
            feature_cols=list(feature_cols_default),
            add_calendar=bool(add_calendar),
            add_price_history=bool(add_price_history),
            price_history_col_map=None,  # Default: jedes Target nutzt seine eigene Historie
            lags=list(lags),
            roll_windows=list(rolls),
            drop_missing_targets=bool(drop_missing_targets),
            drop_missing_features=bool(drop_missing_features),
        )

        # optionaler Debug-Key-Block
        if "add_block_keys" in locals() and add_block_keys:
            feats = add_market_block_keys(
                feats,
                ts_col="ts",
                market=str(market_profile),
                block_hours=int(block_hours),
                gate_closure_offset_hours=float(gate_closure_offset_h),
            )

        # In Session unter den Keys, die Runner erwartet
        st.session_state["features"] = feats
        st.session_state["features_2025"] = feats  # Legacy-Key beibehalten

        # Persistenz im Szenario (SSOT für Runner)
        save_parquet(sname, "features", feats)

        st.success(f"Feature-Tabelle erstellt & gespeichert: {len(feats)} Zeilen | {len(feats.columns)} Spalten")
        st.info(f"Gespeichert nach: data/scenarios/{sname}/features.parquet")

        st.markdown("### Coverage Summary")
        st.dataframe(coverage_summary(feats), use_container_width=True, height=320)

        st.markdown("### Feature-Vorschau (Head)")
        st.dataframe(feats.head(300), use_container_width=True, height=560)

        st.divider()
        st.subheader("Export (optional)")

        # Optionaler Exportpfad (project-root relativ)
        if st.button("💾 Parquet am Exportpfad speichern"):
            abs_path = os.path.join(PROJECT_ROOT, out_path_parquet)
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            feats.to_parquet(abs_path, index=False)
            st.success(f"Export gespeichert: {abs_path}")

        # Download parquet
        buf = io.BytesIO()
        feats.to_parquet(buf, index=False)
        st.download_button(
            "⬇️ features.parquet herunterladen (FULL)",
            data=buf.getvalue(),
            file_name="features.parquet",
            mime="application/octet-stream",
            key="dl_features_parquet_full",
            help="Lädt die vollständige Feature-Tabelle als Parquet herunter.",
        )

        # Download CSV
        csv_bytes = feats.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ features.csv herunterladen (FULL)",
            data=csv_bytes,
            file_name="features.csv",
            mime="text/csv",
            key="dl_features_csv_full",
            help="Lädt die vollständige Feature-Tabelle als CSV herunter.",
        )

    except Exception as e:
        st.error(f"Erstellen fehlgeschlagen: {e}")

# Bestehende Features anzeigen, falls vorhanden
if "features" in st.session_state and not build_btn:
    st.info("Features sind bereits im Session State vorhanden (features).")
    feats = st.session_state["features"]
    st.write(f"Zeilen: {len(feats)} | Spalten: {len(feats.columns)}")
    st.dataframe(feats.head(300), use_container_width=True, height=560)