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


def _calc_min_bid_chf_per_mw(
    total_project_cost_chf: float,
    opex_chf_per_year: float,
    project_lifetime_years: float,
    p_ch_max_kw: float,
    p_dis_max_kw: float,
    estimated_cycles_per_year: float,
) -> float:
    """
    Vereinfachter projektweiter Mindestpreis für SDL-Pay-as-Bid in CHF/MW.

    Herleitung:
    - Es werden bewusst nur die laufenden Betriebskosten (OPEX) berücksichtigt.
    - CAPEX wird NICHT berücksichtigt, da die Investition bereits getätigt wurde.
    - Relevante SDL-Leistung = min(Ladeleistung, Entladeleistung)
    - Gesamte OPEX über Projektlaufzeit = OPEX * Laufzeit
    - Gesamte Zyklenzahl = Zyklen/Jahr * Laufzeit

    Hinweis:
    - Die Laufzeit kürzt sich mathematisch heraus, bleibt aber zur Rückwärtskompatibilität
      in der Funktion enthalten.
    """
    _ = float(total_project_cost_chf or 0.0)  # bewusst ungenutzt; Interface bleibt stabil

    total_opex = float(opex_chf_per_year or 0.0) * float(project_lifetime_years or 0.0)

    p_sdl_kw = min(float(p_ch_max_kw or 0.0), float(p_dis_max_kw or 0.0))
    p_sdl_mw = p_sdl_kw / 1000.0

    total_cycles = float(estimated_cycles_per_year or 0.0) * float(project_lifetime_years or 0.0)

    if p_sdl_mw <= 0.0 or total_cycles <= 0.0:
        return 0.0

    return total_opex / (p_sdl_mw * total_cycles)


def _optimizer_order_text(market_mode: str) -> str:
    mapping = {
        "DA_ONLY": [
            "1. Day-Ahead Preisprognose",
            "2. Day-Ahead MILP-Optimierung",
            "3. Dispatch / Resultate",
        ],
        "ID_ONLY": [
            "1. Intraday Preisprognose",
            "2. Intraday Delta-Optimierung",
            "3. Dispatch / Resultate",
        ],
        "DA_PLUS_ID": [
            "1. Day-Ahead Preisprognose",
            "2. Day-Ahead MILP als Baseline",
            "3. Intraday Preisprognose",
            "4. Intraday Delta-Optimierung relativ zur DA-Baseline",
            "5. Dispatch / Resultate",
        ],
        "SDL_ONLY": [
            "1. Historische SDL-Clearing-Serien einlesen",
            "2. Rolling Bid-/Forecast-Berechnung",
            "3. Pay-as-Bid-Mindestpreis als Untergrenze anwenden",
            "4. Zuschlag bestimmen",
            "5. SDL-Erlöse berechnen",
        ],
        "DA_PLUS_SDL": [
            "1. Day-Ahead Preisprognose",
            "2. Day-Ahead MILP als Energiemarkt-Baseline",
            "3. SDL-Bid-Berechnung",
            "4. Pay-as-Bid-Mindestpreis als Untergrenze anwenden",
            "5. Zuschlag / Erlöse berechnen",
        ],
        "DA_ID_SDL_MULTIUSE": [
            "1. Day-Ahead Preisprognose und DA-Baseline",
            "2. Intraday Preisprognose und Delta-Optimierung",
            "3. SDL-Bid-Berechnung",
            "4. SDL nur berücksichtigen, wenn Zuschlag vorliegt",
            "5. SDL-first: Reserveleistung festlegen und operatives SOC-Fenster bestimmen",
            "6. Residualen DA/ID-Dispatch realisieren / Resultate",
        ],
        "CUSTOM": [
            "Die gewählte Marktkombination entspricht keinem Standardpfad.",
            "Bitte Logik und Resultate besonders prüfen.",
        ],
    }
    lines = mapping.get(market_mode, mapping["CUSTOM"])
    return "\n".join([f"- {line}" for line in lines])


def _auto_prl_offer_mw(batt: dict) -> float:
    """PRL ist symmetrisch -> kleinere der beiden Leistungen."""
    p_ch = max(float(batt.get("p_ch_max_kw", 0.0) or 0.0), 0.0)
    p_dis = max(float(batt.get("p_dis_max_kw", 0.0) or 0.0), 0.0)
    return min(p_ch, p_dis) / 1000.0


def _auto_srl_up_offer_mw(batt: dict) -> float:
    """SRL+ entspricht Entlade-Richtung."""
    p_dis = max(float(batt.get("p_dis_max_kw", 0.0) or 0.0), 0.0)
    return p_dis / 1000.0


def _auto_srl_down_offer_mw(batt: dict) -> float:
    """SRL- entspricht Lade-Richtung."""
    p_ch = max(float(batt.get("p_ch_max_kw", 0.0) or 0.0), 0.0)
    return p_ch / 1000.0


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

cfg["ui"]["debug_mode"] = True
st.session_state["debug_mode"] = True

batt = cfg["battery"]
eco = cfg["economics"]
markets = cfg["markets"]
model = cfg["model"]
dp = cfg["dispatch_policy"]
mp = cfg["market_params"]
sdl = cfg["sdl"]
mu = cfg["multiuse"]

# Feste Defaults / standardmässig aktiv
mu["use_dynamic_margin"] = True
mu["require_sdl_acceptance"] = True
mu["use_opportunity_cost"] = True
mu["lookahead_h"] = 24
sdl["use_price"] = "p_clear_true"

# Defaults für manuelle Mindestpreis-Überschreibung
sdl.setdefault("manual_min_bid_enabled", False)
sdl.setdefault("manual_min_bid_chf_per_mw", None)
sdl.setdefault("derived_min_bid_chf_per_mw", 0.0)

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
# Batterieparameter
# ----------------------------
st.subheader("Batterieparameter")

batt.setdefault("degradation_pct_per_year", 2.3)

c1, c2, c3 = st.columns(3)
with c1:
    batt["e_nom_kwh"] = st.number_input(
        "Nennenergie E_nom [kWh]",
        value=float(batt.get("e_nom_kwh", 1000.0)),
        step=10.0,
        help=(
            "Energieinhalt des Batteriespeichers in kWh.\n\n"
            "Beispiel: Ein 5 MWh-Speicher entspricht 5000 kWh."
        ),
    )
    batt["p_ch_max_kw"] = st.number_input(
        "Max. Ladeleistung P_charge_max [kW]",
        value=float(batt.get("p_ch_max_kw", 500.0)),
        step=10.0,
        help="Maximale elektrische Ladeleistung in kW.",
    )
    batt["p_dis_max_kw"] = st.number_input(
        "Max. Entladeleistung P_discharge_max [kW]",
        value=float(batt.get("p_dis_max_kw", 500.0)),
        step=10.0,
        help="Maximale elektrische Entladeleistung in kW.",
    )

with c2:
    batt["eta_ch"] = st.number_input(
        "Wirkungsgrad Laden η_charge [-]",
        value=float(batt.get("eta_ch", 0.95)),
        min_value=0.5,
        max_value=1.0,
        step=0.01,
        help="Wirkungsgrad beim Laden (0–1).",
    )
    batt["eta_dis"] = st.number_input(
        "Wirkungsgrad Entladen η_discharge [-]",
        value=float(batt.get("eta_dis", 0.95)),
        min_value=0.5,
        max_value=1.0,
        step=0.01,
        help="Wirkungsgrad beim Entladen (0–1).",
    )
    batt["soc0"] = st.number_input(
        "Start-Ladezustand SOC0 [-]",
        value=float(batt.get("soc0", 0.5)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="Startzustand der Batterie als Anteil der Nennenergie.",
    )

with c3:
    batt["soc_min"] = st.number_input(
        "Minimaler SOC SOC_min [-]",
        value=float(batt.get("soc_min", 0.05)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="Untergrenze für den Ladezustand.",
    )
    batt["soc_max"] = st.number_input(
        "Maximaler SOC SOC_max [-]",
        value=float(batt.get("soc_max", 0.95)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="Obergrenze für den Ladezustand.",
    )
    batt["degradation_pct_per_year"] = st.number_input(
        "Degradation [%/Jahr]",
        value=float(batt.get("degradation_pct_per_year", 2.3)),
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        format="%.1f",
        help=(
            "Jährliche Kapazitätsdegradation der Batterie.\n\n"
            "Beispiel: 2.3 bedeutet, dass die nutzbare Kapazität pro Jahr um 2.3% sinkt.\n"
            "Dieser Wert soll im Dashboard bei der NPV-Berechnung berücksichtigt werden."
        ),
    )

st.markdown("---")

# ----------------------------
# Wirtschaftlichkeit
# ----------------------------
st.subheader("Wirtschaftlichkeit")

st.caption(
    "Diese Werte werden für die Projektkennzahlen (NPV/Payback) verwendet. "
    "Zusätzlich wird daraus ein konstanter SDL-Mindestpreis für Pay-as-Bid-Gebote abgeleitet."
)

eco.setdefault("total_project_cost_chf", 0.0)
eco.setdefault("project_lifetime_years", 20)
eco.setdefault("opex_chf_per_year", 0.0)
eco.setdefault("estimated_cycles_per_year", 250.0)

e0, e1, e2, e3 = st.columns(4)
with e0:
    eco["total_project_cost_chf"] = st.number_input(
        "Gesamtprojektkosten (CAPEX total) [CHF]",
        value=float(eco.get("total_project_cost_chf", 0.0) or 0.0),
        min_value=0.0,
        step=10_000.0,
        format="%.2f",
        help="Einmalige Investitionskosten zu Beginn des Projekts.",
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
        help="Jährliche Betriebskosten.",
    )

with e2:
    eco["project_lifetime_years"] = st.number_input(
        "Projektlaufzeit [Jahre]",
        value=int(eco.get("project_lifetime_years", 20) or 20),
        min_value=1,
        step=1,
        help="Wirtschaftlich betrachtete Projektlaufzeit.",
    )

with e3:
    eco["estimated_cycles_per_year"] = st.number_input(
        "Geschätzte Zyklen pro Jahr [-]",
        value=float(eco.get("estimated_cycles_per_year", 250.0) or 250.0),
        min_value=0.0,
        step=10.0,
        format="%.1f",
        help="Geschätzte äquivalente Vollzyklen pro Jahr.",
    )

st.markdown("### SDL: Mindestpreis für Pay-as-Bid")

derived_min_bid_chf_per_mw = _calc_min_bid_chf_per_mw(
    total_project_cost_chf=float(eco.get("total_project_cost_chf", 0.0) or 0.0),
    opex_chf_per_year=float(eco.get("opex_chf_per_year", 0.0) or 0.0),
    project_lifetime_years=float(eco.get("project_lifetime_years", 0.0) or 0.0),
    p_ch_max_kw=float(batt.get("p_ch_max_kw", 0.0) or 0.0),
    p_dis_max_kw=float(batt.get("p_dis_max_kw", 0.0) or 0.0),
    estimated_cycles_per_year=float(eco.get("estimated_cycles_per_year", 0.0) or 0.0),
)

sdl["derived_min_bid_chf_per_mw"] = float(derived_min_bid_chf_per_mw)

manual_min_bid_default = sdl.get("manual_min_bid_chf_per_mw", None)
if manual_min_bid_default is None:
    current_min_bid = sdl.get("min_bid_chf_per_mw", None)
    manual_min_bid_default = current_min_bid if current_min_bid is not None else derived_min_bid_chf_per_mw
manual_min_bid_default = float(manual_min_bid_default)

m0, m1 = st.columns(2)
with m0:
    st.metric(
        "Abgeleiteter SDL-Mindestpreis [CHF/MW]",
        f"{derived_min_bid_chf_per_mw:,.2f}",
    )

    sdl["manual_min_bid_enabled"] = st.checkbox(
        "Abgeleiteten SDL-Mindestpreis manuell überschreiben",
        value=bool(sdl.get("manual_min_bid_enabled", False)),
        help="Wenn aktiviert, wird der manuell eingegebene Mindestpreis statt des abgeleiteten Werts im SDL-Pay-as-Bid-Pfad verwendet.",
    )

    sdl["manual_min_bid_chf_per_mw"] = st.number_input(
        "Manueller SDL-Mindestpreis [CHF/MW]",
        value=float(manual_min_bid_default),
        min_value=0.0,
        step=0.1,
        format="%.2f",
        disabled=not bool(sdl.get("manual_min_bid_enabled", False)),
        help="Dieser Wert wird nur verwendet, wenn die manuelle Überschreibung aktiviert ist.",
    )

active_min_bid_chf_per_mw = (
    float(sdl.get("manual_min_bid_chf_per_mw", manual_min_bid_default))
    if bool(sdl.get("manual_min_bid_enabled", False))
    else float(derived_min_bid_chf_per_mw)
)

sdl["min_bid_chf_per_mw"] = float(active_min_bid_chf_per_mw)
sdl["apply_min_bid_only_for_pay_as_bid"] = True

with m1:
    if bool(sdl.get("manual_min_bid_enabled", False)):
        st.info(
            f"Manuelle Überschreibung aktiv. Im SDL-Pay-as-Bid-Pfad wird mit {active_min_bid_chf_per_mw:,.2f} CHF/MW gerechnet."
        )
    else:
        st.info(
            "Der Wert wird im SDL-Pay-as-Bid-Pfad als projektweiter Mindestpreis verwendet."
        )

st.markdown("---")

# ----------------------------
# Marktauswahl + market_mode ableiten
# ----------------------------
st.subheader("Marktauswahl")

markets["day_ahead"] = st.checkbox(
    "Day-Ahead (DA)",
    value=bool(markets.get("day_ahead", True)),
    help="Aktiviert Day-Ahead-Arbitrage.",
)
markets["intraday"] = st.checkbox(
    "Intraday (Intraday Continuous)",
    value=bool(markets.get("intraday", False)),
    help="Aktiviert Intraday Continuous.",
)
markets["regelenergie"] = st.checkbox(
    "Regelenergie (SDL)",
    value=bool(markets.get("regelenergie", False)),
    help="Aktiviert Systemdienstleistungen / Regelenergie. Darunter fallen hier PRL (symmetrisch) und SRL (asymmetrisch).",
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
st.markdown("**Reihenfolge des Optimierers / Rechenpfads:**")
st.markdown(_optimizer_order_text(cfg["market_mode"]))

st.markdown("---")

# ----------------------------
# Multiuse (Alle Märkte)
# ----------------------------
st.subheader("Multiuse (Alle Märkte)")

mu.setdefault("strategy_profile", "neutral")
mu.setdefault("require_sdl_acceptance", True)
mu.setdefault("use_dynamic_margin", True)
mu.setdefault("use_opportunity_cost", True)
mu.setdefault("lookahead_h", 24)
mu.setdefault("weight_spread_vol", 1.0)
mu.setdefault("weight_soc_edge", 8.0)
mu.setdefault("weight_id_forecast_uncertainty", 0.5)
mu.setdefault("weight_sdl_activation", 1.0)
mu.setdefault("weight_degradation", 1.0)
mu.setdefault("soc_edge_band_pct", 15.0)
mu.setdefault("degradation_chf_per_kwh_throughput", float(dp.get("cycle_penalty_chf_per_kwh", 0.0) or 0.0))
mu.setdefault("lookahead_discount", 0.90)

# Fest im Hintergrund aktiv
mu["use_dynamic_margin"] = True
mu["require_sdl_acceptance"] = True
mu["use_opportunity_cost"] = True
mu["lookahead_h"] = 24

profile_options = ["konservativ", "neutral", "aggressiv"]
profile = st.selectbox(
    "Handelsstil / Entscheidungsprofil",
    options=profile_options,
    index=_safe_index(profile_options, str(mu.get("strategy_profile", "neutral")), "neutral"),
    help=(
        "Vereinfachte Voreinstellung für die Multiuse-Entscheidungslogik.\n\n"
        "konservativ: DA/ID-Flexibilität wird höher bewertet.\n"
        "neutral: ausgewogene Bewertung.\n"
        "aggressiv: SDL wird eher gewählt."
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
    "Dynamische Marge, Opportunitätskosten, die Bedingung 'SDL nur bei Zuschlag' "
    "sowie ein Lookahead-Horizont von 24 h sind standardmässig immer aktiv. "
    "Hier werden nur noch Profil und Gewichtungen gepflegt."
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
        )
        mu["weight_soc_edge"] = st.number_input(
            "Gewicht SoC-Randnähe [-]",
            value=float(mu.get("weight_soc_edge", prof_defaults["weight_soc_edge"])),
            min_value=0.0,
            step=0.5,
            format="%.2f",
        )

    with g2:
        mu["weight_id_forecast_uncertainty"] = st.number_input(
            "Gewicht Intraday-Prognoseunsicherheit [-]",
            value=float(mu.get("weight_id_forecast_uncertainty", prof_defaults["weight_id_forecast_uncertainty"])),
            min_value=0.0,
            step=0.1,
            format="%.2f",
        )
        mu["weight_sdl_activation"] = st.number_input(
            "Gewicht SDL-Aktivierung / Reservierungsrisiko [-]",
            value=float(mu.get("weight_sdl_activation", prof_defaults["weight_sdl_activation"])),
            min_value=0.0,
            step=0.1,
            format="%.2f",
        )
        mu["weight_degradation"] = st.number_input(
            "Gewicht Degradation [-]",
            value=float(mu.get("weight_degradation", prof_defaults["weight_degradation"])),
            min_value=0.0,
            step=0.1,
            format="%.2f",
        )

    with g3:
        mu["soc_edge_band_pct"] = st.number_input(
            "SoC-Randband [%]",
            value=float(mu.get("soc_edge_band_pct", 15.0)),
            min_value=1.0,
            max_value=50.0,
            step=1.0,
            format="%.1f",
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
        )
        mu["lookahead_discount"] = st.number_input(
            "Lookahead-Diskontfaktor [-]",
            value=float(mu.get("lookahead_discount", prof_defaults["lookahead_discount"])),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
        )

st.info(
    "Im Multiuse-Dashboard werden später stündliche Diagnosegrössen gespeichert, z.B. "
    "'dynamic_margin_chf_h', 'opp_cost_chf_h', 'sdl_score_chf_h', 'daid_score_chf_h' und 'decision_gap_chf_h'."
)

st.markdown("---")

# ----------------------------
# Regelenergie (SDL)
# ----------------------------
st.subheader("Regelenergie (SDL): Aktivierungsmassnahmen")

sdl.setdefault("accept_prob_target", 0.70)
sdl.setdefault("window_days", 28)
sdl["use_price"] = "p_clear_true"

# neue / klarere Leistungsfelder
legacy_offer_mw = float(sdl.get("p_offer_mw", 0.0) or 0.0)
sdl.setdefault("auto_offer_from_battery_prl", True)
sdl.setdefault("auto_offer_from_battery_srl", True)
sdl.setdefault("p_offer_prl_mw", legacy_offer_mw if legacy_offer_mw > 0 else _auto_prl_offer_mw(batt))
sdl.setdefault("p_offer_srl_up_mw", _auto_srl_up_offer_mw(batt))
sdl.setdefault("p_offer_srl_down_mw", _auto_srl_down_offer_mw(batt))

sdl.setdefault("gate_closure", {})
sdl["gate_closure"].setdefault("PRL", {"type": "D-1", "time": "08:00"})
sdl["gate_closure"].setdefault("SRL", {"type": "D-1", "time": "14:30"})

sdl.setdefault("activation", {})
sdl["activation"].setdefault("alpha_srl_up", 0.06)
sdl["activation"].setdefault("alpha_srl_down", 0.04)

st.caption(
    "PRL wird hier als symmetrisches Produkt behandelt, SRL als asymmetrisches Produkt. "
    "Damit ist PRL durch die kleinere der beiden Batterie-Leistungen begrenzt, "
    "während SRL+ und SRL− getrennt aus Entlade- bzw. Ladeleistung abgeleitet werden können."
)

st.markdown("#### Gemeinsame Bid-/Forecast-Parameter")
st.caption(
    "Die Ziel-Zuschlagswahrscheinlichkeit ist aktuell ein globaler Zielwert und gilt im Modell "
    "für den jeweiligen SDL-Bid-Pfad, also nicht separat für PRL und SRL."
)

g1, g2 = st.columns(2)
with g1:
    sdl["accept_prob_target"] = st.number_input(
        "Ziel-Zuschlagswahrscheinlichkeit p [-] (global für SDL)",
        value=float(sdl.get("accept_prob_target", 0.70)),
        min_value=0.01,
        max_value=0.99,
        step=0.01,
        format="%.2f",
        help="Globaler Zielwert für die mittlere Zuschlagswahrscheinlichkeit im SDL-Bid-Pfad.",
    )

with g2:
    sdl["window_days"] = st.number_input(
        "Rolling Window für Bid-/Forecast-Berechnung [Tage]",
        value=int(sdl.get("window_days", 28)),
        min_value=7,
        step=1,
        help="Wie viele vergangene Tage für die Bid- bzw. Forecast-Berechnung im SDL-Pfad verwendet werden.",
    )

st.markdown("#### PRL (symmetrische Reserve)")
st.caption(
    "PRL muss symmetrisch angeboten werden. "
    "Die automatische Angebotsleistung entspricht deshalb min(P_charge_max, P_discharge_max)."
)

prl_auto_offer_mw = _auto_prl_offer_mw(batt)
sdl["auto_offer_from_battery_prl"] = st.checkbox(
    "PRL-Angebotsleistung automatisch aus Batterieparametern ableiten",
    value=bool(sdl.get("auto_offer_from_battery_prl", True)),
    help="Wenn aktiv, wird PRL automatisch auf die kleinere der beiden Batterie-Leistungen gesetzt.",
)

if sdl["auto_offer_from_battery_prl"]:
    sdl["p_offer_prl_mw"] = float(prl_auto_offer_mw)
    st.number_input(
        "PRL-Angebotsleistung [MW]",
        value=float(sdl["p_offer_prl_mw"]),
        min_value=0.0,
        step=0.1,
        format="%.2f",
        disabled=True,
        help="Automatisch aus min(P_charge_max, P_discharge_max) berechnet.",
        key="prl_offer_auto_display",
    )
else:
    sdl["p_offer_prl_mw"] = st.number_input(
        "PRL-Angebotsleistung [MW]",
        value=float(sdl.get("p_offer_prl_mw", prl_auto_offer_mw)),
        min_value=0.0,
        step=0.1,
        format="%.2f",
        help="Manuell definierte symmetrische Angebotsleistung für PRL.",
        key="prl_offer_manual_input",
    )

st.markdown("#### SRL (asymmetrische Reserve)")
st.caption(
    "Für SRL können Up- und Down-Richtung getrennt betrachtet werden: "
    "SRL+ entspricht der Entladeleistung, SRL− der Ladeleistung."
)

srl_auto_up_mw = _auto_srl_up_offer_mw(batt)
srl_auto_down_mw = _auto_srl_down_offer_mw(batt)

sdl["auto_offer_from_battery_srl"] = st.checkbox(
    "SRL-Angebotsleistungen automatisch aus Batterieparametern ableiten",
    value=bool(sdl.get("auto_offer_from_battery_srl", True)),
    help="Wenn aktiv, wird SRL+ = P_discharge_max und SRL− = P_charge_max gesetzt.",
)

s1, s2 = st.columns(2)
if sdl["auto_offer_from_battery_srl"]:
    sdl["p_offer_srl_up_mw"] = float(srl_auto_up_mw)
    sdl["p_offer_srl_down_mw"] = float(srl_auto_down_mw)

    with s1:
        st.number_input(
            "SRL+ Angebotsleistung [MW]",
            value=float(sdl["p_offer_srl_up_mw"]),
            min_value=0.0,
            step=0.1,
            format="%.2f",
            disabled=True,
            help="Automatisch aus P_discharge_max berechnet.",
            key="srl_up_offer_auto_display",
        )
    with s2:
        st.number_input(
            "SRL− Angebotsleistung [MW]",
            value=float(sdl["p_offer_srl_down_mw"]),
            min_value=0.0,
            step=0.1,
            format="%.2f",
            disabled=True,
            help="Automatisch aus P_charge_max berechnet.",
            key="srl_down_offer_auto_display",
        )
else:
    with s1:
        sdl["p_offer_srl_up_mw"] = st.number_input(
            "SRL+ Angebotsleistung [MW]",
            value=float(sdl.get("p_offer_srl_up_mw", srl_auto_up_mw)),
            min_value=0.0,
            step=0.1,
            format="%.2f",
            help="Manuell definierte Angebotsleistung für SRL+ / Up / Entlade-Richtung.",
            key="srl_up_offer_manual_input",
        )
    with s2:
        sdl["p_offer_srl_down_mw"] = st.number_input(
            "SRL− Angebotsleistung [MW]",
            value=float(sdl.get("p_offer_srl_down_mw", srl_auto_down_mw)),
            min_value=0.0,
            step=0.1,
            format="%.2f",
            help="Manuell definierte Angebotsleistung für SRL− / Down / Lade-Richtung.",
            key="srl_down_offer_manual_input",
        )

st.markdown("#### SRL: Aktivierungsannahmen (Erwartungswert)")
a1, a2 = st.columns(2)
with a1:
    sdl["activation"]["alpha_srl_up"] = st.number_input(
        "SRL+ Aktivierungsgrad α_up [-]",
        value=float(sdl.get("activation", {}).get("alpha_srl_up", 0.06)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        help="Erwartungswert der Aktivierung in positiver / entladender Richtung.",
    )

with a2:
    sdl["activation"]["alpha_srl_down"] = st.number_input(
        "SRL− Aktivierungsgrad α_down [-]",
        value=float(sdl.get("activation", {}).get("alpha_srl_down", 0.04)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        help="Erwartungswert der Aktivierung in negativer / ladender Richtung.",
    )

# Rückwärtskompatibilität:
# p_offer_mw bleibt bestehen und wird auf die symmetrische PRL-Leistung gesetzt.
sdl["p_offer_mw"] = float(sdl.get("p_offer_prl_mw", prl_auto_offer_mw))

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
            "ridge/lasso/elasticnet: lineare Modelle.\n"
            "random_forest/gbrt: nichtlineare Modelle."
        ),
    )
with c2:
    model["forecast_mode"] = st.selectbox(
        "Forecast-Modus (global)",
        options=FORECAST_MODE_OPTS,
        index=_safe_index(FORECAST_MODE_OPTS, model.get("forecast_mode", "rolling"), "rolling"),
        help=(
            "fit_once: einmal trainieren.\n"
            "rolling: fortlaufend neu trainieren.\n\n"
            "Empfehlung: rolling."
        ),
    )
with c3:
    mp["forecast_target_col"] = st.text_input(
        "Day-Ahead Target-Spalte (z.B. price_da)",
        value=str(mp.get("forecast_target_col", "price_da")),
        help="Zielspalte für die Day-Ahead Preisprognose.",
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
            "Beispiel: 24 bedeutet 1× pro Tag."
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
            "Beispiel: 24 bedeutet: Intraday-Delta wird über die nächsten 24 Stunden optimiert."
        ),
    )
with id4:
    mp["id_min_train_rows"] = st.number_input(
        "Intraday Min. Trainingszeilen",
        value=int(mp.get("id_min_train_rows", 200)),
        min_value=50,
        step=50,
        help=(
            "Mindestanzahl Datenpunkte, bevor das Intraday-Modell trainiert werden darf."
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
            help="Mindestanzahl vergangener Tage für das DA-Training.",
        )
    with c5:
        model["retrain_every_days"] = st.number_input(
            "Day-Ahead Retrain alle x Tage",
            value=int(model.get("retrain_every_days", 1)),
            min_value=1,
            step=1,
            help="Wie oft das Day-Ahead Modell neu trainiert wird.",
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
    help="Optionaler Strafterm für Batteriedurchsatz.",
)

st.markdown("---")

# ----------------------------
# Speichern
# ----------------------------
save_btn = st.button(
    "Szenario speichern",
    type="primary",
    help="Speichert alle Einstellungen dieses Szenarios dauerhaft.",
)
if save_btn:
    if mu.get("degradation_chf_per_kwh_throughput", None) in (None, 0.0):
        mu["degradation_chf_per_kwh_throughput"] = float(dp.get("cycle_penalty_chf_per_kwh", 0.0) or 0.0)

    cfg.setdefault("ui", {})
    cfg["ui"]["debug_mode"] = True

    mu["use_dynamic_margin"] = True
    mu["require_sdl_acceptance"] = True
    mu["use_opportunity_cost"] = True
    mu["lookahead_h"] = 24
    sdl["use_price"] = "p_clear_true"

    # Auto-Angebotsleistungen konsistent setzen
    if bool(sdl.get("auto_offer_from_battery_prl", True)):
        sdl["p_offer_prl_mw"] = float(_auto_prl_offer_mw(batt))

    if bool(sdl.get("auto_offer_from_battery_srl", True)):
        sdl["p_offer_srl_up_mw"] = float(_auto_srl_up_offer_mw(batt))
        sdl["p_offer_srl_down_mw"] = float(_auto_srl_down_offer_mw(batt))

    # Rückwärtskompatibilität
    sdl["p_offer_mw"] = float(sdl.get("p_offer_prl_mw", _auto_prl_offer_mw(batt)))

    derived_min_bid_saved = float(
        _calc_min_bid_chf_per_mw(
            total_project_cost_chf=float(eco.get("total_project_cost_chf", 0.0) or 0.0),
            opex_chf_per_year=float(eco.get("opex_chf_per_year", 0.0) or 0.0),
            project_lifetime_years=float(eco.get("project_lifetime_years", 0.0) or 0.0),
            p_ch_max_kw=float(batt.get("p_ch_max_kw", 0.0) or 0.0),
            p_dis_max_kw=float(batt.get("p_dis_max_kw", 0.0) or 0.0),
            estimated_cycles_per_year=float(eco.get("estimated_cycles_per_year", 0.0) or 0.0),
        )
    )

    sdl["derived_min_bid_chf_per_mw"] = float(derived_min_bid_saved)

    manual_override_enabled = bool(sdl.get("manual_min_bid_enabled", False))
    manual_min_bid_saved = sdl.get("manual_min_bid_chf_per_mw", None)
    if manual_min_bid_saved is None:
        manual_min_bid_saved = derived_min_bid_saved
    manual_min_bid_saved = float(manual_min_bid_saved)

    if manual_override_enabled:
        sdl["min_bid_chf_per_mw"] = float(manual_min_bid_saved)
    else:
        sdl["min_bid_chf_per_mw"] = float(derived_min_bid_saved)

    sdl["apply_min_bid_only_for_pay_as_bid"] = True

    st.session_state["scenario_config"] = cfg
    save_config(scenario_name, cfg)
    st.success("Szenario-Konfiguration gespeichert.")