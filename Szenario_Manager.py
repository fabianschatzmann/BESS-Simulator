# Szenario_Manager.py
# -*- coding: utf-8 -*-
import streamlit as st

from core.scenario_store import load_config, save_config, load_parquet

st.set_page_config(page_title="Szenario-Manager", layout="wide")
st.title("Szenario-Manager")
st.caption(
    "Single Source of Truth: Batterieparameter, Marktauswahl, Modell- und Dispatchparameter "
    "werden hier definiert und im Szenario gespeichert."
)


def _safe_index(options: list[str], value: str, default: str) -> int:
    """Return options.index(value) if present else options.index(default) else 0."""
    v = str(value) if value is not None else ""
    if v in options:
        return options.index(v)
    if default in options:
        return options.index(default)
    return 0


# ----------------------------
# Szenario-Auswahl
# ----------------------------
colA, colB = st.columns([2, 1])

with colA:
    scenario_name = st.text_input(
        "Szenario-Name",
        value=st.session_state.get("scenario_name", "BaseCase_2025"),
        help=(
            "Name/ID deines Szenarios. Unter diesem Namen werden alle Dateien gespeichert, z.B.:\n"
            "data/scenarios/<Szenario-Name>/master.parquet, features.parquet, dispatch.parquet usw.\n\n"
            "Tipp: Verwende sprechende Namen wie 'BaseCase_2025' oder 'DA_ID_SDL_2025_v1'."
        ),
    )
    st.session_state["scenario_name"] = scenario_name

with colB:
    load_btn = st.button(
        "Szenario laden",
        help="Lädt die gespeicherte Konfiguration dieses Szenarios (falls vorhanden).",
    )
    new_btn = st.button(
        "Neues Standard-Szenario",
        help="Setzt die Konfiguration zurück (leeres Szenario).",
    )

# Load existing config (or initialize)
if "scenario_config" not in st.session_state:
    cfg = load_config(scenario_name) or {}
    st.session_state["scenario_config"] = cfg

if load_btn:
    cfg = load_config(scenario_name) or {}
    st.session_state["scenario_config"] = cfg
    st.success("Szenario-Konfiguration geladen.")

if new_btn:
    st.session_state["scenario_config"] = {}
    st.success("Neues Standard-Szenario initialisiert (leer).")

cfg = st.session_state["scenario_config"]

# ----------------------------
# Ensure config structure
# ----------------------------
cfg.setdefault("battery", {})
cfg.setdefault("economics", {})
cfg.setdefault("markets", {})
cfg.setdefault("model", {})
cfg.setdefault("dispatch_policy", {})
cfg.setdefault("market_params", {})
cfg.setdefault("sdl", {})
cfg.setdefault("multiuse", {})
cfg.setdefault("ui", {})

batt = cfg["battery"]
eco = cfg["economics"]
markets = cfg["markets"]
model = cfg["model"]
dp = cfg["dispatch_policy"]
mp = cfg["market_params"]
sdl = cfg["sdl"]
mu = cfg["multiuse"]
ui = cfg["ui"]

# ----------------------------
# Workflow-Status
# ----------------------------
st.subheader("Workflow-Status")

sname = scenario_name
has_master = st.session_state.get("master") is not None or load_parquet(sname, "master") is not None
has_features = st.session_state.get("features") is not None or load_parquet(sname, "features") is not None

has_pred_legacy = st.session_state.get("pred_prices") is not None or load_parquet(sname, "pred_prices") is not None
has_pred_da = st.session_state.get("pred_prices_da") is not None or load_parquet(sname, "pred_prices_da") is not None
has_pred_id = st.session_state.get("pred_prices_id") is not None or load_parquet(sname, "pred_prices_id") is not None
has_pred = bool(has_pred_legacy or has_pred_da or has_pred_id)

has_dispatch = st.session_state.get("dispatch") is not None or load_parquet(sname, "dispatch") is not None
has_results = st.session_state.get("results") is not None or load_parquet(sname, "results_timeseries") is not None

steps = [
    ("Datenimport (Master)", has_master),
    ("Feature Engineering", has_features),
    ("Prognosen (pred_prices*)", has_pred),
    ("Dispatch / Optimierung", has_dispatch),
    ("Dashboard (Resultate)", has_results),
]
done = sum(int(ok) for _, ok in steps)
st.progress(done / len(steps), text=f"{done}/{len(steps)} Schritte erledigt")
for label, ok in steps:
    st.write(("✅ " if ok else "⬜ ") + label)

st.markdown("---")

# ----------------------------
# UI-Einstellungen
# ----------------------------
st.subheader("UI-Einstellungen")
ui["debug_mode"] = st.checkbox(
    "Debug-Modus anzeigen",
    value=bool(ui.get("debug_mode", False)),
    help=(
        "Wenn aktiv, werden zusätzliche Diagnose-Informationen angezeigt (z.B. Spaltenchecks, Merges, "
        "Warnungen zu fehlenden Daten). Für normale Nutzer meist nicht nötig."
    ),
)
st.info("Hinweis: Die Debug-Seite ist nur nutzbar, wenn der Debug-Modus aktiviert ist.")

st.markdown("---")

# ----------------------------
# Batterieparameter
# ----------------------------
st.subheader("Batterieparameter")

c1, c2, c3 = st.columns(3)
with c1:
    batt["e_nom_kwh"] = st.number_input(
        "Nennenergie E_nom [kWh]",
        value=float(batt.get("e_nom_kwh", 1000.0)),
        step=10.0,
        help=(
            "Energieinhalt des Batteriespeichers (nutzbare Nennkapazität) in kWh.\n\n"
            "Beispiel: Ein 1 MWh-Speicher entspricht 1000 kWh.\n"
            "Diese Größe bestimmt zusammen mit SOC_min/SOC_max, wie viel Energie tatsächlich verfügbar ist."
        ),
    )
    batt["p_ch_max_kw"] = st.number_input(
        "Max. Ladeleistung P_charge_max [kW]",
        value=float(batt.get("p_ch_max_kw", 500.0)),
        step=10.0,
        help=(
            "Maximale elektrische Ladeleistung in kW.\n\n"
            "Beispiel: 500 kW bedeutet, dass die Batterie pro Stunde maximal 500 kWh aufnehmen kann (bei 1h Zeitschritt), "
            "abzüglich Wirkungsgrad.\n"
            "Diese Grenze limitiert, wie schnell geladen werden kann."
        ),
    )
    batt["p_dis_max_kw"] = st.number_input(
        "Max. Entladeleistung P_discharge_max [kW]",
        value=float(batt.get("p_dis_max_kw", 500.0)),
        step=10.0,
        help=(
            "Maximale elektrische Entladeleistung in kW.\n\n"
            "Beispiel: 500 kW bedeutet, dass die Batterie pro Stunde maximal 500 kWh abgeben kann (bei 1h Zeitschritt), "
            "abzüglich Wirkungsgrad.\n"
            "Diese Grenze limitiert, wie schnell entladen werden kann."
        ),
    )

with c2:
    batt["eta_ch"] = st.number_input(
        "Wirkungsgrad Laden η_charge [-]",
        value=float(batt.get("eta_ch", 0.95)),
        min_value=0.5,
        max_value=1.0,
        step=0.01,
        help=(
            "Wirkungsgrad beim Laden (0–1). Beispiel: 0.95 bedeutet 95%.\n\n"
            "Wenn 100 kWh geladen werden, kommen effektiv 95 kWh im Speicher an.\n"
            "Dieser Parameter beeinflusst SOC-Entwicklung und Wirtschaftlichkeit."
        ),
    )
    batt["eta_dis"] = st.number_input(
        "Wirkungsgrad Entladen η_discharge [-]",
        value=float(batt.get("eta_dis", 0.95)),
        min_value=0.5,
        max_value=1.0,
        step=0.01,
        help=(
            "Wirkungsgrad beim Entladen (0–1). Beispiel: 0.95 bedeutet 95%.\n\n"
            "Um 95 kWh ins Netz abzugeben, muss der Speicher ~100 kWh entnehmen.\n"
            "Dieser Parameter beeinflusst SOC-Entwicklung und Wirtschaftlichkeit."
        ),
    )
    batt["soc0"] = st.number_input(
        "Start-Ladezustand SOC0 [-]",
        value=float(batt.get("soc0", 0.5)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help=(
            "Startzustand der Batterie als Anteil (0–1) der Nennenergie.\n\n"
            "Beispiel: 0.5 bedeutet Start bei 50%.\n"
            "Wichtig für Simulationen: Ein anderer Start-SOC kann die ersten Tage/Wochen beeinflussen."
        ),
    )

with c3:
    batt["soc_min"] = st.number_input(
        "Minimaler SOC SOC_min [-]",
        value=float(batt.get("soc_min", 0.05)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help=(
            "Untergrenze für den Ladezustand (0–1).\n\n"
            "Beispiel: 0.05 bedeutet: Batterie darf nicht unter 5% fallen.\n"
            "Das schützt die Batterie und stellt eine Reserve sicher."
        ),
    )
    batt["soc_max"] = st.number_input(
        "Maximaler SOC SOC_max [-]",
        value=float(batt.get("soc_max", 0.95)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help=(
            "Obergrenze für den Ladezustand (0–1).\n\n"
            "Beispiel: 0.95 bedeutet: Batterie darf nicht über 95% geladen werden.\n"
            "Das schützt die Batterie und lässt Raum für Regelung/Fehler."
        ),
    )

st.markdown("---")

# ----------------------------
# Wirtschaftlichkeit
# ----------------------------
st.subheader("Wirtschaftlichkeit")

st.caption(
    "Diese Werte werden für die Projektkennzahlen (NPV/Payback) verwendet. "
    "Zusätzlich werden einige davon im SDL-Teil genutzt, um einen Kosten-Floor für Gebote abzuleiten."
)

eco.setdefault("total_project_cost_chf", 0.0)
eco.setdefault("project_lifetime_years", 20)
eco.setdefault("opex_chf_per_year", 0.0)

e0, e1, e2 = st.columns(3)
with e0:
    eco["total_project_cost_chf"] = st.number_input(
        "Gesamtprojektkosten (CAPEX total) [CHF]",
        value=float(eco.get("total_project_cost_chf", 0.0) or 0.0),
        min_value=0.0,
        step=10_000.0,
        format="%.2f",
        help=(
            "Einmalige Investitionskosten zu Beginn des Projekts (t=0). Dazu zählen z.B.:\n"
            "Batterie, Container, Wechselrichter, Trafostation, Installation, Planung, Netzanschluss.\n\n"
            "Wird für NPV/Payback genutzt (als initialer Cashflow -CAPEX)."
        ),
    )

with e1:
    default_opex = 0.03 * float(eco.get("total_project_cost_chf", 0.0) or 0.0)
    if eco.get("opex_chf_per_year", None) in (None, 0.0) and default_opex > 0:
        eco["opex_chf_per_year"] = float(default_opex)

    eco["opex_chf_per_year"] = st.number_input(
        "Betriebskosten (OPEX) [CHF/Jahr]",
        value=float(eco.get("opex_chf_per_year", default_opex) or 0.0),
        min_value=0.0,
        step=1_000.0,
        format="%.2f",
        help=(
            "Jährliche Betriebskosten. Typische Bestandteile:\n"
            "- Wartung/Service\n"
            "- Versicherung\n"
            "- Monitoring/IT\n"
            "- Standortmiete (falls relevant)\n\n"
            "Standardannahme: ca. 3% der Gesamtprojektkosten pro Jahr.\n"
            "Diese Kosten werden vom Jahresertrag abgezogen, bevor NPV/Payback berechnet werden."
        ),
    )

with e2:
    eco["project_lifetime_years"] = st.number_input(
        "Projektlaufzeit [Jahre]",
        value=int(eco.get("project_lifetime_years", 20) or 20),
        min_value=1,
        step=1,
        help=(
            "Wie lange das Projekt wirtschaftlich betrachtet wird.\n\n"
            "NPV-Rechnung diskontiert die jährlichen Netto-Cashflows über diese Laufzeit (t=1..N).\n"
            "Hinweis: Diese Laufzeit kann (muss aber nicht) identisch mit der technischen Lebensdauer der Batterie sein."
        ),
    )

st.markdown("### SDL: Kostenbasis für Gebote (Cost-Floor)")

st.caption(
    "Diese Parameter werden im SDL-only Optimizer verwendet, um einen Mindestpreis zu bestimmen "
    "(annuitisierte CAPEX + fixe O&M + optionaler Risikoaufschlag)."
)

e1, e2, e3 = st.columns(3)
with e1:
    eco["capex_chf_per_kw_power"] = st.number_input(
        "CAPEX Leistung [CHF/kW]",
        value=float(eco.get("capex_chf_per_kw_power", 400.0)),
        min_value=0.0,
        step=10.0,
        help=(
            "Investitionskosten, die hauptsächlich von der Leistung abhängen (kW), z.B. Wechselrichter/Trafokette.\n"
            "Wird genutzt, um einen Kosten-Floor in CHF/MW/h zu berechnen."
        ),
    )
    eco["capex_chf_per_kwh_energy"] = st.number_input(
        "CAPEX Energie [CHF/kWh] (optional)",
        value=float(eco.get("capex_chf_per_kwh_energy", 200.0)),
        min_value=0.0,
        step=10.0,
        help=(
            "Investitionskosten, die hauptsächlich von der Energie abhängen (kWh), z.B. Batteriezellen.\n"
            "Optional: wenn du die Kosten nur über CHF/kW abbilden willst, kannst du hier 0 setzen."
        ),
    )

with e2:
    eco["fixed_om_chf_per_kw_year"] = st.number_input(
        "Fixe O&M [CHF/kW/Jahr]",
        value=float(eco.get("fixed_om_chf_per_kw_year", 10.0)),
        min_value=0.0,
        step=1.0,
        help=(
            "Fixe jährliche Kosten, die proportional zur angebotenen Leistung sind.\n"
            "Beispiele: Wartung der Leistungselektronik, Service-Verträge.\n"
            "Wird für den SDL-Kosten-Floor verwendet."
        ),
    )
    eco["asset_life_years"] = st.number_input(
        "Technische Lebensdauer (für Cost-Floor) [Jahre]",
        value=int(eco.get("asset_life_years", 15)),
        min_value=1,
        step=1,
        help=(
            "Diese Lebensdauer wird für die Annuität (CRF) beim SDL-Kosten-Floor genutzt.\n"
            "Sie ist technisch motiviert (wie lange hält die Anlage), und kann von der Projektlaufzeit abweichen."
        ),
    )

with e3:
    eco["wacc"] = st.number_input(
        "Diskontsatz / WACC [-]",
        value=float(eco.get("wacc", 0.06)),
        min_value=0.0,
        max_value=0.5,
        step=0.005,
        format="%.3f",
        help=(
            "Diskontsatz für die NPV-Rechnung (Kapital-/Risikokosten).\n\n"
            "Beispiel: 0.06 = 6%.\n"
            "Je höher der WACC, desto stärker werden zukünftige Cashflows abgezinst → NPV wird kleiner."
        ),
    )
    eco["risk_premium_chf_per_mw_h"] = st.number_input(
        "Risiko-/Marge [CHF/MW/h] (optional)",
        value=float(eco.get("risk_premium_chf_per_mw_h", 0.0)),
        min_value=0.0,
        step=0.1,
        help=(
            "Zusätzlicher Aufschlag im SDL-Gebot als Sicherheitsmarge.\n"
            "Kann genutzt werden, um Risiko/Unsicherheit oder gewünschte Gewinnmarge abzubilden.\n"
            "Erhöht den Mindest-Bid (Cost-Floor + Risiko)."
        ),
    )

st.markdown("---")

# ----------------------------
# Marktauswahl + market_mode ableiten
# ----------------------------
st.subheader("Marktauswahl")

markets["day_ahead"] = st.checkbox(
    "Day-Ahead (DA)",
    value=bool(markets.get("day_ahead", True)),
    help="Aktiviert Day-Ahead-Arbitrage (Energiehandel am Vortag).",
)
markets["intraday"] = st.checkbox(
    "Intraday (Intraday Continuous)",
    value=bool(markets.get("intraday", False)),
    help="Aktiviert Intraday Continuous als inkrementellen Mehrwert gegenüber Day-Ahead.",
)
markets["regelenergie"] = st.checkbox(
    "Regelenergie (PRL/SRL)",
    value=bool(markets.get("regelenergie", False)),
    help="Aktiviert SDL/Regelenergie (Bidding + Merit Order Zuschlag).",
)

da = bool(markets.get("day_ahead", False))
re_ = bool(markets.get("regelenergie", False))
id_ = bool(markets.get("intraday", False))

if da and (not re_) and (not id_):
    cfg["market_mode"] = "DA_ONLY"
elif id_ and (not da) and (not re_):
    cfg["market_mode"] = "ID_ONLY"
elif da and id_ and (not re_):
    cfg["market_mode"] = "DA_PLUS_ID"
elif re_ and (not da) and (not id_):
    cfg["market_mode"] = "SDL_ONLY"
elif da and re_ and (not id_):
    cfg["market_mode"] = "DA_PLUS_SDL"
elif da and id_ and re_:
    cfg["market_mode"] = "DA_ID_SDL_MULTIUSE"
else:
    cfg["market_mode"] = "CUSTOM"

st.info(f"Aktueller Marktmodus: **{cfg['market_mode']}**")

st.markdown("---")

# ----------------------------
# Multiuse (Alle Märkte) Settings
# ----------------------------
st.subheader("Multiuse (Alle Märkte) – Dynamische Marge + Opportunitätskosten")

mu.setdefault("strategy_profile", "neutral")
mu.setdefault("require_sdl_acceptance", True)
mu.setdefault("use_dynamic_margin", True)
mu.setdefault("use_opportunity_cost", True)
mu.setdefault("lookahead_h", 6)
mu.setdefault("weight_spread_vol", 1.0)
mu.setdefault("weight_soc_edge", 8.0)
mu.setdefault("weight_id_forecast_uncertainty", 0.5)
mu.setdefault("weight_sdl_activation", 1.0)
mu.setdefault("weight_degradation", 1.0)
mu.setdefault("soc_edge_band_pct", 15.0)
mu.setdefault("degradation_chf_per_kwh_throughput", float(dp.get("cycle_penalty_chf_per_kwh", 0.0) or 0.0))
mu.setdefault("lookahead_discount", 0.90)

profile_options = ["konservativ", "neutral", "aggressiv"]
profile = st.selectbox(
    "Handelsstil / Entscheidungsprofil",
    options=profile_options,
    index=_safe_index(profile_options, str(mu.get("strategy_profile", "neutral")), "neutral"),
    help=(
        "Vereinfachte Voreinstellung für die Multiuse-Entscheidungslogik.\n\n"
        "konservativ: DA/ID-Flexibilität wird höher bewertet, SDL wird zurückhaltender gewählt.\n"
        "neutral: ausgewogene Bewertung.\n"
        "aggressiv: SDL wird eher gewählt, DA/ID-Opportunitätskosten werden geringer gewichtet."
    ),
)
mu["strategy_profile"] = profile

if profile == "konservativ":
    prof_defaults = {
        "weight_spread_vol": 1.3,
        "weight_soc_edge": 10.0,
        "weight_id_forecast_uncertainty": 0.8,
        "weight_sdl_activation": 1.2,
        "weight_degradation": 1.2,
        "lookahead_discount": 0.92,
    }
elif profile == "aggressiv":
    prof_defaults = {
        "weight_spread_vol": 0.8,
        "weight_soc_edge": 5.0,
        "weight_id_forecast_uncertainty": 0.3,
        "weight_sdl_activation": 0.7,
        "weight_degradation": 0.8,
        "lookahead_discount": 0.85,
    }
else:
    prof_defaults = {
        "weight_spread_vol": 1.0,
        "weight_soc_edge": 8.0,
        "weight_id_forecast_uncertainty": 0.5,
        "weight_sdl_activation": 1.0,
        "weight_degradation": 1.0,
        "lookahead_discount": 0.90,
    }

st.caption(
    "Es gibt keine fixe manuelle Entscheidungsmarge mehr. "
    "Die Entscheidung wird aus dynamischer Marge und Opportunitätskosten abgeleitet."
)

mu["require_sdl_acceptance"] = st.checkbox(
    "SDL nur wählen, wenn Zuschlag (accepted==1) vorliegt",
    value=bool(mu.get("require_sdl_acceptance", True)),
    help=(
        "Wenn aktiv, wird SDL in der Multiuse-Spur nur dann gewählt, wenn im SDL-Timeseries "
        "mindestens ein Produkt (PRL oder SRL) akzeptiert ist.\n\n"
        "Hinweis: Das ist weiterhin eine pragmatische Phase-1/1.5 Annahme (ex-post Outcome)."
    ),
)

m1, m2, m3 = st.columns(3)
with m1:
    mu["use_dynamic_margin"] = st.checkbox(
        "Dynamische Marge aktivieren",
        value=bool(mu.get("use_dynamic_margin", True)),
        help=(
            "Wenn aktiv, wird die Mindestmarge zugunsten DA/ID pro Stunde dynamisch aus mehreren Treibern abgeleitet:\n"
            "- erwartete Spread-Volatilität\n"
            "- SoC-Randnähe\n"
            "- Intraday-Prognoseunsicherheit\n"
            "- SDL-Aktivierungsannahmen\n"
            "- Degradation"
        ),
    )
with m2:
    mu["use_opportunity_cost"] = st.checkbox(
        "Opportunitätskosten aktivieren",
        value=bool(mu.get("use_opportunity_cost", True)),
        help=(
            "Wenn aktiv, wird nicht nur der Erlös der aktuellen Stunde verglichen, sondern auch der Mehrstundenwert "
            "der freien Batterie für DA/ID über einen Lookahead-Horizont."
        ),
    )
with m3:
    mu["lookahead_h"] = st.number_input(
        "Lookahead-Horizont [h]",
        value=int(mu.get("lookahead_h", 6) or 6),
        min_value=1,
        max_value=48,
        step=1,
        help=(
            "Anzahl Stunden, über die Opportunitätskosten und Teile der dynamischen Marge betrachtet werden.\n\n"
            "Kleiner Horizont: reaktiver.\n"
            "Grösserer Horizont: vorausschauender, aber tendenziell konservativer."
        ),
    )

with st.expander("Erweiterte Multiuse-Gewichtung", expanded=False):
    st.caption(
        "Diese Parameter gewichten die einzelnen Komponenten der Multiuse-Entscheidung. "
        "Für die meisten Fälle reicht die Profilwahl oben."
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        mu["weight_spread_vol"] = st.number_input(
            "Gewicht Spread-Volatilität [-]",
            value=float(mu.get("weight_spread_vol", prof_defaults["weight_spread_vol"])),
            min_value=0.0,
            step=0.1,
            format="%.2f",
            help=(
                "Je höher dieser Wert, desto stärker erhöht volatile DA/ID-Spread-Phasen die dynamische Marge "
                "zugunsten freier Batterie-Flexibilität."
            ),
        )
        mu["weight_soc_edge"] = st.number_input(
            "Gewicht SoC-Randnähe [-]",
            value=float(mu.get("weight_soc_edge", prof_defaults["weight_soc_edge"])),
            min_value=0.0,
            step=0.5,
            format="%.2f",
            help=(
                "Je höher dieser Wert, desto stärker wird eine Batterie nahe SOC_min oder SOC_max "
                "zugunsten DA/ID-Flexibilität bewertet."
            ),
        )

    with g2:
        mu["weight_id_forecast_uncertainty"] = st.number_input(
            "Gewicht Intraday-Prognoseunsicherheit [-]",
            value=float(mu.get("weight_id_forecast_uncertainty", prof_defaults["weight_id_forecast_uncertainty"])),
            min_value=0.0,
            step=0.1,
            format="%.2f",
            help=(
                "Je höher dieser Wert, desto stärker erhöht Unsicherheit im Intraday-Forecast "
                "die dynamische Marge."
            ),
        )
        mu["weight_sdl_activation"] = st.number_input(
            "Gewicht SDL-Aktivierung / Reservierungsrisiko [-]",
            value=float(mu.get("weight_sdl_activation", prof_defaults["weight_sdl_activation"])),
            min_value=0.0,
            step=0.1,
            format="%.2f",
            help=(
                "Gewichtet den Einfluss von Aktivierungs- bzw. Reservierungsannahmen im SDL-Pfad."
            ),
        )
        mu["weight_degradation"] = st.number_input(
            "Gewicht Degradation [-]",
            value=float(mu.get("weight_degradation", prof_defaults["weight_degradation"])),
            min_value=0.0,
            step=0.1,
            format="%.2f",
            help=(
                "Gewichtet den Durchsatz-/Degradationseinfluss in der dynamischen Marge."
            ),
        )

    with g3:
        mu["soc_edge_band_pct"] = st.number_input(
            "SoC-Randband [%]",
            value=float(mu.get("soc_edge_band_pct", 15.0)),
            min_value=1.0,
            max_value=50.0,
            step=1.0,
            format="%.1f",
            help=(
                "Bandbreite um die SoC-Grenzen, in der Randnähe zusätzlich penalisiert wird.\n"
                "Beispiel: 15 bedeutet erhöhte Bewertung in den Bereichen 0–15% und 85–100%."
            ),
        )
        mu["degradation_chf_per_kwh_throughput"] = st.number_input(
            "Degradation [CHF/kWh Durchsatz]",
            value=float(
                mu.get(
                    "degradation_chf_per_kwh_throughput",
                    float(dp.get("cycle_penalty_chf_per_kwh", 0.0) or 0.0),
                )
            ),
            min_value=0.0,
            step=0.0001,
            format="%.4f",
            help=(
                "Zusätzliche Kostenannahme für Batteriedurchsatz in der Multiuse-Entscheidung.\n"
                "Kann identisch zur Dispatch-Degradationsstrafe gewählt werden."
            ),
        )
        mu["lookahead_discount"] = st.number_input(
            "Lookahead-Diskontfaktor [-]",
            value=float(mu.get("lookahead_discount", prof_defaults["lookahead_discount"])),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            help=(
                "Gewichtung zukünftiger Stunden in den Opportunitätskosten.\n"
                "1.00 = alle Stunden gleich.\n"
                "Kleinere Werte gewichten nahe Stunden stärker."
            ),
        )

st.info(
    "Im Multiuse-Dashboard werden später stündliche Diagnosegrössen gespeichert, z.B. "
    "'dynamic_margin_chf_h', 'opp_cost_chf_h', 'sdl_score_chf_h', 'daid_score_chf_h' und 'decision_gap_chf_h'."
)

st.markdown("---")

# ----------------------------
# Regelenergie (SDL) Settings
# ----------------------------
st.subheader("Regelenergie (SDL) – Bidding, Gate Closing & Aktivierungsannahmen")

sdl.setdefault("accept_prob_target", 0.70)
sdl.setdefault("window_days", 28)
sdl.setdefault("use_price", "p_clear_true")
sdl.setdefault("p_offer_mw", 1.0)
sdl.setdefault("gate_closure", {})
sdl["gate_closure"].setdefault("PRL", {"type": "D-1", "time": "08:00"})
sdl["gate_closure"].setdefault("SRL", {"type": "D-1", "time": "14:30"})

sdl.setdefault("activation", {})
sdl["activation"].setdefault("alpha_srl_up", 0.06)
sdl["activation"].setdefault("alpha_srl_down", 0.04)

g1, g2, g3 = st.columns(3)
with g1:
    sdl["accept_prob_target"] = st.number_input(
        "Ziel-Zuschlagswahrscheinlichkeit p [-]",
        value=float(sdl.get("accept_prob_target", 0.70)),
        min_value=0.01,
        max_value=0.99,
        step=0.01,
        format="%.2f",
        help=(
            "Wie oft dein Gebot im Mittel einen Zuschlag erhalten soll.\n\n"
            "Beispiel: 0.70 bedeutet: du zielst darauf ab, in ~70% der Fälle akzeptiert zu werden.\n"
            "Technisch: Wir wählen einen Bid so, dass er ungefähr dem (1-p)-Quantil der historischen Clearing-Preise entspricht.\n"
            "Höheres p → tendenziell niedrigere Gebote → häufiger Zuschlag, aber ggf. geringerer Erlös pro Stunde."
        ),
    )

with g2:
    sdl["window_days"] = st.number_input(
        "Rolling Window [Tage]",
        value=int(sdl.get("window_days", 28)),
        min_value=7,
        step=1,
        help=(
            "Wie viele vergangene Tage für die Bid-Berechnung herangezogen werden.\n\n"
            "Beispiel: 28 Tage bedeutet: Bid basiert auf den letzten 28 Tagen Clearing-Historie.\n"
            "Kürzeres Fenster reagiert schneller auf Marktänderungen, ist aber volatil.\n"
            "Längeres Fenster ist stabiler, reagiert aber langsamer."
        ),
    )
    sdl["use_price"] = st.selectbox(
        "Clearing-Serie (Outcome)",
        options=["p_clear_true", "p_vwa_true"],
        index=_safe_index(["p_clear_true", "p_vwa_true"], sdl.get("use_price", "p_clear_true"), "p_clear_true"),
        help=(
            "Welche 'Outcome'-Preisserie für die Zuschlagsentscheidung verwendet wird.\n\n"
            "p_clear_true: der (approximierte) Clearing-Preis pro Stunde.\n"
            "p_vwa_true: volumen-gewichteter Durchschnittspreis.\n\n"
            "Empfehlung: p_clear_true, weil es näher an Merit-Order-Clearing liegt."
        ),
    )

with g3:
    sdl["p_offer_mw"] = st.number_input(
        "Angebotsleistung [MW] (Baseline konstant)",
        value=float(sdl.get("p_offer_mw", 1.0)),
        min_value=0.0,
        step=0.1,
        format="%.2f",
        help=(
            "Wie viel Leistung du in der Regelenergie-Auktion anbietest.\n\n"
            "Beispiel: 1.0 MW bedeutet: bei Zuschlag verdienst du pro Stunde ungefähr:\n"
            "Erlös ≈ Preis [CHF/MW/h] × 1.0 MW.\n\n"
            "Hinweis: Im SDL-only Mode wird noch nicht geprüft, ob die Batterie diese Leistung technisch immer halten kann. "
            "Das kommt später mit integrierten Nebenbedingungen."
        ),
    )

st.markdown("### SRL: Aktivierungsannahmen (Erwartungswert)")

a1, a2 = st.columns(2)
with a1:
    sdl["activation"]["alpha_srl_up"] = st.number_input(
        "Aktivierungsgrad SRL+ α_up [-]",
        value=float(sdl.get("activation", {}).get("alpha_srl_up", 0.06)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        help=(
            "Erwartungswert-Annahme: Wie stark SRL+ im Mittel aktiviert wird, wenn du einen Zuschlag hast.\n\n"
            "Interpretation im stündlichen Modell:\n"
            "E_exp [MWh/h] = α_up × P_offer [MW] × 1h\n\n"
            "Beispiel: α_up = 0.06 und P_offer = 1 MW → erwartete Aktivierung ≈ 0.06 MWh pro Stunde.\n\n"
            "Diese Annahme steuert den erwarteten Energie-Umsatz (Arbeitspreis), zusätzlich zur Vorhaltevergütung."
        ),
    )

with a2:
    sdl["activation"]["alpha_srl_down"] = st.number_input(
        "Aktivierungsgrad SRL− α_down [-]",
        value=float(sdl.get("activation", {}).get("alpha_srl_down", 0.04)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        help=(
            "Erwartungswert-Annahme: Wie stark SRL− im Mittel aktiviert wird, wenn du einen Zuschlag hast.\n\n"
            "Interpretation im stündlichen Modell:\n"
            "E_exp [MWh/h] = α_down × P_offer [MW] × 1h\n\n"
            "Beispiel: α_down = 0.04 und P_offer = 1 MW → erwartete Aktivierung ≈ 0.04 MWh pro Stunde.\n\n"
            "Hinweis: SRL+ und SRL− können unterschiedliche Aktivierungsprofile haben – deshalb getrennte Parameter."
        ),
    )

st.markdown("---")

# ----------------------------
# Prognosemodell (DA + ID)
# ----------------------------
st.subheader("Prognosemodell (DA + ID)")

MODEL_OPTS = ["ridge", "lasso", "elasticnet", "random_forest", "gbrt"]
FORECAST_MODE_OPTS = ["fit_once", "rolling"]

c1, c2, c3 = st.columns(3)
with c1:
    model["model_name"] = st.selectbox(
        "Modelltyp",
        options=MODEL_OPTS,
        index=_safe_index(MODEL_OPTS, model.get("model_name", "ridge"), "ridge"),
        help=(
            "Auswahl des Prognosemodells.\n\n"
            "ridge/lasso/elasticnet: lineare Modelle (schnell, robust).\n"
            "random_forest/gbrt: nichtlineare Modelle (können komplexere Zusammenhänge lernen, benötigen aber oft mehr Daten).\n"
            "Die Auswahl beeinflusst nur die Preisprognose – nicht die Optimierungslogik."
        ),
    )
with c2:
    model["forecast_mode"] = st.selectbox(
        "Forecast-Modus (global)",
        options=FORECAST_MODE_OPTS,
        index=_safe_index(FORECAST_MODE_OPTS, model.get("forecast_mode", "rolling"), "rolling"),
        help=(
            "fit_once: Modell wird einmal trainiert und dann über das ganze Jahr angewendet (Benchmark, kann Leakage begünstigen).\n"
            "rolling: Modell wird fortlaufend neu trainiert (leak-free, realistischer für Simulationen).\n\n"
            "Empfehlung: rolling."
        ),
    )
with c3:
    mp["forecast_target_col"] = st.text_input(
        "Day-Ahead Target-Spalte (z.B. price_da)",
        value=str(mp.get("forecast_target_col", "price_da")),
        help=(
            "Spaltenname im Master/Features, der als Zielvariable für die Day-Ahead Preisprognose verwendet wird.\n\n"
            "In der Regel: price_da."
        ),
    )

st.markdown("### Intraday Continuous – Forecast Settings")
id1, id2, id3, id4 = st.columns(4)
with id1:
    mp["forecast_target_col_id"] = st.text_input(
        "Intraday Target-Spalte (z.B. price_id)",
        value=str(mp.get("forecast_target_col_id", "price_id")),
        help=(
            "Spaltenname im Master/Features, der als Zielvariable für die Intraday Preisprognose verwendet wird.\n\n"
            "In der Regel: price_id."
        ),
    )
with id2:
    mp["id_retrain_every_hours"] = st.number_input(
        "Intraday Retrain alle x Stunden",
        value=int(mp.get("id_retrain_every_hours", 24)),
        min_value=1,
        step=1,
        help=(
            "Wie oft das Intraday-Prognosemodell neu trainiert wird.\n\n"
            "Beispiel: 24 bedeutet 1× pro Tag.\n"
            "Kürzere Intervalle können genauer sein, sind aber langsamer."
        ),
    )
with id3:
    mp["id_horizon_hours"] = st.number_input(
        "Intraday MPC/Forecast Horizont [h]",
        value=int(mp.get("id_horizon_hours", 24)),
        min_value=1,
        max_value=72,
        step=1,
        help=(
            "Wie weit in die Zukunft Intraday geplant/optimiert wird.\n\n"
            "Beispiel: 24 bedeutet: Intraday-Delta wird über die nächsten 24 Stunden optimiert.\n"
            "Größere Horizonte sind vorausschauender, erhöhen aber Rechenzeit."
        ),
    )
with id4:
    mp["id_min_train_rows"] = st.number_input(
        "Intraday Min. Trainingszeilen",
        value=int(mp.get("id_min_train_rows", 200)),
        min_value=50,
        step=50,
        help=(
            "Mindestanzahl Datenpunkte, bevor das Intraday-Modell trainiert werden darf.\n\n"
            "Zu wenige Daten → Modell instabil.\n"
            "Empfehlung: mindestens einige Tage bis Wochen an stündlichen Daten."
        ),
    )

if model["forecast_mode"] == "rolling":
    c4, c5 = st.columns(2)
    with c4:
        model["train_days_min"] = st.number_input(
            "Min. Trainingstage (Day-Ahead rolling)",
            value=int(model.get("train_days_min", 60)),
            min_value=7,
            step=1,
            help=(
                "Wie viele vergangene Tage mindestens zum Training des Day-Ahead Modells verwendet werden.\n\n"
                "Mehr Tage → stabiler, aber evtl. weniger adaptiv.\n"
                "Weniger Tage → adaptiver, aber möglicherweise noisiger."
            ),
        )
    with c5:
        model["retrain_every_days"] = st.number_input(
            "Day-Ahead Retrain alle x Tage",
            value=int(model.get("retrain_every_days", 1)),
            min_value=1,
            step=1,
            help=(
                "Wie oft das Day-Ahead Modell neu trainiert wird.\n\n"
                "1 = täglich neu trainieren (realistischer, aber rechenintensiver)."
            ),
        )

st.markdown("---")

# ----------------------------
# Dispatch / Optimierung
# ----------------------------
st.subheader("Dispatch / Optimierung (MILP)")

dp["cycle_penalty_chf_per_kwh"] = st.number_input(
    "Durchsatz-/Degradationsstrafe [CHF pro kWh Durchsatz] (optional)",
    value=float(dp.get("cycle_penalty_chf_per_kwh", 0.0)),
    min_value=0.0,
    step=0.0001,
    format="%.4f",
    help=(
        "Optionaler Strafterm, der häufiges Laden/Entladen (Batteriedurchsatz) wirtschaftlich 'teurer' macht.\n\n"
        "Wenn >0, wird der Optimierer weniger aggressiv handeln, um Zyklen zu reduzieren.\n"
        "Wenn 0, wird nur nach Marktpreisen optimiert."
    ),
)

st.markdown("---")

# ----------------------------
# Speichern
# ----------------------------
save_btn = st.button(
    "Szenario speichern",
    type="primary",
    help="Speichert alle Einstellungen dieses Szenarios dauerhaft in data/scenarios/<scenario>/config.json (oder äquivalent).",
)
if save_btn:
    if mu.get("degradation_chf_per_kwh_throughput", None) in (None, 0.0):
        mu["degradation_chf_per_kwh_throughput"] = float(dp.get("cycle_penalty_chf_per_kwh", 0.0) or 0.0)

    st.session_state["scenario_config"] = cfg
    save_config(scenario_name, cfg)
    st.success("Szenario-Konfiguration gespeichert.")