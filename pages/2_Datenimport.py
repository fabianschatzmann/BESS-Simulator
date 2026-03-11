# pages/2_Datenimport.py
# -*- coding: utf-8 -*-
import os
import io
import re
import difflib
import numpy as np
import pandas as pd
import streamlit as st

from core.scenario_store import save_parquet  # writes data/scenarios/<scenario>/<key>.parquet


# =============================================================================
# Robuste Defaults (verhindert Scope-Probleme bei Streamlit-Reruns)
# =============================================================================
if "tz" not in st.session_state:
    st.session_state["tz"] = "Europe/Zurich"
if "scenario_name" not in st.session_state:
    st.session_state["scenario_name"] = "BaseCase_2025"

tz = st.session_state["tz"]
sname = st.session_state["scenario_name"]


# =============================================================================
# Seite
# =============================================================================
st.set_page_config(page_title="Datenimport-Assistent", layout="wide")
st.title("Datenimport-Assistent")
st.caption(
    "Master 2025 (stündlich) laden/erstellen + Intraday Continuous Preis + Swissgrid SDL. "
    "Merge erfolgt ausschließlich über ts_key (UTC-naiv)."
)

# =============================================================================
# Cache-Kontrollen
# =============================================================================
CACHE_VERSION = "2026-01-03_06"

if st.button("🧹 Cache leeren (st.cache_data)", key="btn_clear_cache"):
    st.cache_data.clear()
    st.success("Cache geleert. Bitte SDL-Vorschau + Master-Build erneut ausführen.")


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("Globale Einstellungen")
    st.session_state["tz"] = st.selectbox(
        "Zeitzone (Anzeige/Annahme bei tz-naiven Zeitstempeln)",
        ["Europe/Zurich", "UTC", "Europe/Berlin"],
        index=["Europe/Zurich", "UTC", "Europe/Berlin"].index(st.session_state["tz"]),
        help=(
            "Wichtig: Der Merge erfolgt immer über ts_key (UTC-naiv). "
            "Diese Einstellung beeinflusst nur die Annahme/Interpretation von tz-naiven Zeitstempeln "
            "und die Anzeige."
        ),
    )
    tz = st.session_state["tz"]
    st.caption("Hinweis: Merge über ts_key (UTC-naiv). Anzeige kann tz-aware sein.")

    st.markdown("---")
    st.header("Szenario")
    st.session_state["scenario_name"] = st.text_input(
        "Szenarioname",
        value=st.session_state["scenario_name"],
        key="scenario_name_input",
        help=(
            "Name des Szenarios. Dateien werden unter "
            "data/scenarios/<scenario>/ gespeichert (z.B. master.parquet, sdl_prices.parquet)."
        ),
    )
    sname = st.session_state["scenario_name"]

    save_to_scenario = st.checkbox(
        "Master zusätzlich in data/scenarios/<scenario>/master.parquet speichern",
        value=True,
        key="save_to_scenario_master",
        help=(
            "Empfohlen: Aktivieren, damit Dispatch/Runner/Dashboard immer den gleichen Master "
            "pro Szenario findet (SSOT-Persistenz pro Scenario)."
        ),
    )
    st.caption("Empfohlen: aktivieren, damit Runner/Dashboard immer den gleichen Master findet.")


# =============================================================================
# Generische Helfer
# =============================================================================
def read_any_table(uploaded_file, sheet_name=None, header=0, sep_guess=True):
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        if sheet_name in (None, "", "__FIRST__"):
            df = pd.read_excel(uploaded_file, header=header)
        else:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header)
        return df

    if name.endswith(".csv") or name.endswith(".txt") or name.endswith(".tsv"):
        raw = uploaded_file.getvalue()
        if sep_guess:
            for sep in [";", "\t", ",", "|"]:
                try:
                    df = pd.read_csv(io.BytesIO(raw), sep=sep, header=header)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
            return pd.read_csv(io.BytesIO(raw), sep=",", header=header)
        return pd.read_csv(io.BytesIO(raw), sep=",", header=header)

    raise ValueError("Nur .xlsx/.xls/.csv/.txt/.tsv unterstützt.")


def coerce_numeric(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    if s.dtype == object:
        s2 = (
            s.astype(str)
            .str.replace("\u00A0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace("'", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        return pd.to_numeric(s2, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def _canon_colname(x: str) -> str:
    s = str(x).strip().replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).lower()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return s


def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    orig = list(out.columns)
    new_cols = []
    for c in orig:
        cc = _canon_colname(c)
        if cc in ["datum (mesz)", "datum (mez)", "datum", "date"]:
            new_cols.append("datum")
        elif cc in ["zeit", "time"]:
            new_cols.append("zeit")
        elif cc in ["stunde", "hour", "stunden", "hh"]:
            new_cols.append("stunde")
        elif cc in ["timestamp", "datetime", "time stamp", "ts"]:
            new_cols.append("timestamp")
        else:
            new_cols.append(str(c).strip())
    out.columns = new_cols
    return out


def resolve_col(df: pd.DataFrame, selected: str) -> str | None:
    """
    Robustes Matching eines ausgewählten Spaltennamens auf df.columns.

    Unterstützt:
      - Zeilenumbrüche in Excel-Headern (\n)
      - NBSP
      - Mehrfachspaces / Groß-Kleinschreibung
      - kleine Header-Differenzen (Monatsfiles)

    Strategie:
      1) exakter Match
      2) kanonischer exakter Match
      3) Substring-Match
      4) Fuzzy Similarity (SequenceMatcher) mit Threshold
    """
    if not selected or selected == "(none)":
        return None

    # 1) exakt
    if selected in df.columns:
        return selected

    def canon(s: str) -> str:
        s = str(s).replace("\u00A0", " ")
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    sel = canon(selected)

    cols = list(df.columns)
    canon_cols = [canon(c) for c in cols]
    canon_map = {cc: c for cc, c in zip(canon_cols, cols)}

    # 2) kanonisch exakt
    if sel in canon_map:
        return canon_map[sel]

    # 3) substring
    candidates = []
    for cc, orig in zip(canon_cols, cols):
        if sel and (sel in cc or cc in sel):
            score = min(len(sel), len(cc)) / max(len(sel), len(cc))
            candidates.append((score, orig))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # 4) fuzzy
    best = (0.0, None)
    for cc, orig in zip(canon_cols, cols):
        r = difflib.SequenceMatcher(None, sel, cc).ratio()
        if r > best[0]:
            best = (r, orig)

    if best[0] >= 0.80:
        return best[1]

    return None


def ensure_ts_key(master: pd.DataFrame, tz_assume: str = "Europe/Zurich", sdl_ts_key: pd.Series | None = None) -> pd.DataFrame:
    """
    Erzeugt/normalisiert ts_key (UTC-naiv) für Joins.

    Wenn ts tz-aware ist -> nach UTC konvertieren und tz-naiv machen.
    Wenn ts tz-naiv ist:
      - zwei Kandidaten:
          A) Annahme lokale TZ -> UTC-naiv
          B) Annahme bereits UTC-naiv
      - wenn sdl_ts_key übergeben: wähle Kandidat mit mehr Overlap
      - sonst: Default A (lokale TZ)
    """
    m = master.copy()

    if "ts_key" in m.columns and m["ts_key"].notna().any():
        m["ts_key"] = pd.to_datetime(m["ts_key"], errors="coerce").dt.floor("H")
        return m

    if "ts" not in m.columns:
        raise ValueError("Tabelle hat keine 'ts'-Spalte → ts_key kann nicht berechnet werden.")

    ts = pd.to_datetime(m["ts"], errors="coerce")

    # tz-aware -> UTC-naiv
    if getattr(ts.dt, "tz", None) is not None:
        m["ts_key"] = ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("H")
        return m

    # Kandidat B: bereits UTC-naiv
    ts_key_utc = ts.dt.floor("H")

    # Kandidat A: lokal -> UTC-naiv
    try:
        ts_local = ts.dt.tz_localize(tz_assume, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        ts_local = ts.dt.tz_localize(tz_assume, ambiguous=False, nonexistent="shift_forward")
    ts_key_local = ts_local.dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("H")

    if sdl_ts_key is None:
        m["ts_key"] = ts_key_local
        return m

    sdl_k = pd.to_datetime(sdl_ts_key, errors="coerce").dropna().dt.floor("H")
    if sdl_k.empty:
        m["ts_key"] = ts_key_local
        return m

    sdl_set = set(sdl_k.values)
    overlap_local = int(np.sum(ts_key_local.notna() & ts_key_local.isin(sdl_set)))
    overlap_utc = int(np.sum(ts_key_utc.notna() & ts_key_utc.isin(sdl_set)))

    m["ts_key"] = ts_key_local if overlap_local >= overlap_utc else ts_key_utc
    return m


# =============================================================================
# Swissgrid SDL FAST loader + parse-by-block + expand (wie bisher belassen)
# =============================================================================
SDL_NEEDED_COLS = [
    "Ausschreibung",
    "Beschreibung",
    "Angebotenes Volumen",
    "Zugesprochenes Volumen",
    "Leistungspreis",
    "Preis",
    "Land",
    "Angebotspreis",
]

_RE_SDL_TIME = re.compile(r"(\d{2}:\d{2})\s*(?:bis|to|-)\s*(\d{2}:\d{2})", re.IGNORECASE)
_RE_SDL_DIR = re.compile(r"\b(UP|DOWN)\b", re.IGNORECASE)
_RE_SDL_PROD = re.compile(r"^(PRL|SRL)_", re.IGNORECASE)

_RE_SDL_DATE_YYMMDD = re.compile(r"_(\d{2})_(\d{2})_(\d{2})$")  # *_YY_MM_DD
_RE_SDL_WEEK_KW = re.compile(r"_KW(\d{2})$", re.IGNORECASE)     # *_KWxx
_RE_SDL_YEAR_YY = re.compile(r"^(?:PRL|SRL)_(\d{2})_", re.IGNORECASE)

_RE_DATE_DDMMYYYY = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")
_RE_DATE_RANGE = re.compile(r"(\d{2}\.\d{2}\.\d{4})\s*(?:bis|to|-)\s*(\d{2}\.\d{2}\.\d{4})", re.IGNORECASE)

_RE_SDL_DIR_UP = re.compile(r"(\bUP\b|\bUPWARD\b|\bHOCH\b|\bRAUF\b|\bPOS\b|\bPOSITIVE\b|\+|↑)", re.IGNORECASE)
_RE_SDL_DIR_DOWN = re.compile(r"(\bDOWN\b|\bDOWNWARD\b|\bRUNTER\b|\bTIEF\b|\bNEG\b|\bNEGATIVE\b|[-−]|↓)", re.IGNORECASE)

DELIM_CANDIDATES = ["\t", ";", ",", "|"]


def _detect_encoding(raw_bytes: bytes) -> str:
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            raw_bytes.decode(enc)
            return enc
        except Exception:
            continue
    return "latin1"


def _looks_like_xlsx(raw_bytes: bytes, fname: str) -> bool:
    if fname.lower().endswith(".xlsx") or fname.lower().endswith(".xls"):
        return True
    return raw_bytes[:2] == b"PK"


def _split_lines(text: str):
    return text.splitlines()


def _find_header_line_idx(lines):
    for i, ln in enumerate(lines[:300]):
        if re.search(r"\bAusschreibung\b", ln, flags=re.IGNORECASE):
            return i
    return None


def _score_delimiter(header_line: str, delim: str) -> int:
    if not header_line:
        return -1
    parts = header_line.split(delim)
    n = len(parts)
    keys = {"Ausschreibung", "Beschreibung", "Zugesprochenes Volumen", "Leistungspreis", "Preis", "Land"}
    hit = sum(1 for p in parts if p.strip() in keys)
    return n + 5 * hit


def _detect_delimiter(lines, header_idx):
    header_line = lines[header_idx] if header_idx is not None and header_idx < len(lines) else (lines[0] if lines else "")
    best = None
    best_score = -1
    for d in DELIM_CANDIDATES:
        sc = _score_delimiter(header_line, d)
        if sc > best_score:
            best_score = sc
            best = d
    return best or "\t"


def load_swissgrid_sdl_fast(uploaded_file, sep_choice: str | None = None):
    raw = uploaded_file.getvalue()
    fname = getattr(uploaded_file, "name", "uploaded")
    meta = {"filename": fname, "filetype": None, "encoding": None, "delimiter": None, "header_row": None, "sheet": None}

    if _looks_like_xlsx(raw, fname):
        meta["filetype"] = "xlsx"
        bio = io.BytesIO(raw)
        xls = pd.ExcelFile(bio)

        chosen_sheet = xls.sheet_names[0]
        header_row = 0

        for sh in xls.sheet_names:
            df_try = pd.read_excel(xls, sheet_name=sh, header=None, nrows=2000, dtype=str)
            mask = df_try.apply(lambda col: col.astype(str).str.contains("Ausschreibung", case=False, na=False))
            if mask.any().any():
                chosen_sheet = sh
                break

        meta["sheet"] = chosen_sheet

        df_head = pd.read_excel(xls, sheet_name=chosen_sheet, header=None, nrows=5000, dtype=str)
        for i in range(len(df_head)):
            row = df_head.iloc[i].astype(str).tolist()
            if any(re.search(r"\bAusschreibung\b", cell, flags=re.IGNORECASE) for cell in row):
                header_row = i
                break
        meta["header_row"] = int(header_row)

        df = pd.read_excel(xls, sheet_name=chosen_sheet, header=header_row, dtype=str)
        df = df[[c for c in SDL_NEEDED_COLS if c in df.columns]].copy()
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        return df.reset_index(drop=True), meta

    meta["filetype"] = "csv"
    enc = _detect_encoding(raw)
    meta["encoding"] = enc
    text = raw.decode(enc, errors="replace")

    lines = _split_lines(text)
    header_idx = _find_header_line_idx(lines)
    if header_idx is None:
        header_idx = 0
        for i, ln in enumerate(lines[:200]):
            if ln.strip() and any(ln.count(d) >= 3 for d in DELIM_CANDIDATES):
                header_idx = i
                break
    meta["header_row"] = int(header_idx)

    delim = sep_choice if sep_choice is not None else _detect_delimiter(lines, header_idx)
    meta["delimiter"] = delim

    body = "\n".join(lines[header_idx:]).encode(enc, errors="replace")
    bio = io.BytesIO(body)

    def _usecols(col):
        return col in SDL_NEEDED_COLS

    try:
        df = pd.read_csv(bio, sep=delim, engine="c", dtype=str, usecols=_usecols, low_memory=False)
    except Exception:
        bio.seek(0)
        df = pd.read_csv(bio, sep=delim, engine="python", dtype=str, usecols=_usecols, on_bad_lines="skip")

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df.reset_index(drop=True), meta


def _kw_to_date_range(ausschreibung: str):
    if not isinstance(ausschreibung, str):
        return pd.NaT, pd.NaT
    s = ausschreibung.strip()
    mkw = _RE_SDL_WEEK_KW.search(s)
    myy = _RE_SDL_YEAR_YY.search(s)
    if not mkw or not myy:
        return pd.NaT, pd.NaT
    week = int(mkw.group(1))
    year = 2000 + int(myy.group(1))
    try:
        start = pd.Timestamp.fromisocalendar(year, week, 1).normalize()
        end = pd.Timestamp.fromisocalendar(year, week, 7).normalize()
        return start, end
    except Exception:
        return pd.NaT, pd.NaT


def parse_sdl_to_blocks_agg(df: pd.DataFrame, filter_ch: bool, products: list[str]):
    d = df.copy()

    for c in SDL_NEEDED_COLS:
        if c not in d.columns:
            d[c] = np.nan

    d["Ausschreibung"] = d["Ausschreibung"].astype(str).str.strip()
    d["Beschreibung"] = d["Beschreibung"].astype(str).str.strip()
    d["Land"] = d["Land"].astype(str).str.strip()

    d["product"] = d["Ausschreibung"].str.extract(_RE_SDL_PROD, expand=False).str.upper()
    if products:
        d = d[d["product"].isin(products)].copy()
    else:
        d = d[d["product"].isin(["PRL", "SRL"])].copy()

    if filter_ch:
        d = d[d["Land"] == "CH"].copy()

    d["v"] = coerce_numeric(d["Zugesprochenes Volumen"])
    d = d[d["v"].fillna(0) > 0].copy()

    m = d["Ausschreibung"].str.extract(_RE_SDL_DATE_YYMMDD)
    yy = pd.to_numeric(m[0], errors="coerce")
    mm = pd.to_numeric(m[1], errors="coerce")
    dd = pd.to_numeric(m[2], errors="coerce")
    date_from = pd.to_datetime(dict(year=(yy + 2000), month=mm, day=dd), errors="coerce")
    date_to = date_from.copy()

    miss = date_from.isna()
    if miss.any():
        rng = d.loc[miss, "Beschreibung"].astype(str).str.extract(_RE_DATE_RANGE)
        df0 = pd.to_datetime(rng[0], format="%d.%m.%Y", errors="coerce")
        df1 = pd.to_datetime(rng[1], format="%d.%m.%Y", errors="coerce")
        date_from.loc[miss] = df0
        date_to.loc[miss] = df1

    miss2 = date_from.isna()
    if miss2.any():
        one = d.loc[miss2, "Beschreibung"].astype(str).str.extract(_RE_DATE_DDMMYYYY)
        df0 = pd.to_datetime(dict(year=one[2], month=one[1], day=one[0]), errors="coerce")
        date_from.loc[miss2] = df0
        date_to.loc[miss2] = df0

    miss3 = date_from.isna()
    if miss3.any():
        kw_start, kw_end = zip(*d.loc[miss3, "Ausschreibung"].map(_kw_to_date_range))
        kw_start = pd.Series(kw_start, index=d.index[miss3])
        kw_end = pd.Series(kw_end, index=d.index[miss3])
        date_from.loc[miss3] = kw_start
        date_to.loc[miss3] = kw_end

    d["delivery_start"] = pd.to_datetime(date_from, errors="coerce").dt.normalize()
    d["delivery_end"] = pd.to_datetime(date_to, errors="coerce").dt.normalize()
    d = d[d["delivery_start"].notna() & d["delivery_end"].notna()].copy()

    # Zeiten: SRL weekly ohne Zeit -> Default 00:00-24:00
    t = d["Beschreibung"].str.extract(_RE_SDL_TIME)
    d["block_start"] = t[0]
    d["block_end"] = t[1]

    missing_time = d["block_start"].isna() | d["block_end"].isna()
    srl_week_no_time = (
        missing_time
        & d["Ausschreibung"].astype(str).str.contains(_RE_SDL_WEEK_KW, na=False)
        & (d["product"] == "SRL")
    )
    d.loc[srl_week_no_time, "block_start"] = "00:00"
    d.loc[srl_week_no_time, "block_end"] = "24:00"
    d = d[d["block_start"].notna() & d["block_end"].notna()].copy()

    # Richtung robust
    desc = (d["Ausschreibung"].astype(str) + " " + d["Beschreibung"].astype(str))
    desc_norm = desc.str.replace("−", "-", regex=False)

    d["direction"] = desc_norm.str.extract(_RE_SDL_DIR, expand=False).str.upper()
    need = d["direction"].isna() | (d["direction"] == "")
    is_down = need & desc_norm.str.contains(_RE_SDL_DIR_DOWN, na=False)
    d.loc[is_down, "direction"] = "DOWN"
    need2 = d["direction"].isna() | (d["direction"] == "")
    is_up = need2 & desc_norm.str.contains(_RE_SDL_DIR_UP, na=False)
    d.loc[is_up, "direction"] = "UP"

    d.loc[(d["product"] == "PRL"), "direction"] = "SYM"

    d = d[
        ((d["product"] == "SRL") & (d["direction"].isin(["UP", "DOWN"]))) |
        ((d["product"] == "PRL") & (d["direction"] == "SYM"))
    ].copy()

    # block_hours
    bs = d["block_start"].astype(str)
    be = d["block_end"].astype(str)

    bs_h = pd.to_numeric(bs.str.slice(0, 2), errors="coerce")
    bs_m = pd.to_numeric(bs.str.slice(3, 5), errors="coerce")
    be_h = pd.to_numeric(be.str.slice(0, 2), errors="coerce")
    be_m = pd.to_numeric(be.str.slice(3, 5), errors="coerce")

    be_is_24 = be == "24:00"
    bs_is_24 = bs == "24:00"
    be_h = be_h.mask(be_is_24, 24)
    be_m = be_m.mask(be_is_24, 0)
    bs_h_dur = bs_h.mask(bs_is_24, 24)
    bs_m_dur = bs_m.mask(bs_is_24, 0)

    start_min = (60 * bs_h_dur + bs_m_dur)
    end_min = (60 * be_h + be_m)
    end_min = end_min.where(end_min > start_min, end_min + 24 * 60)
    d["block_hours"] = ((end_min - start_min) / 60.0).astype(int)
    d = d[d["block_hours"].notna() & (d["block_hours"] > 0)].copy()

    # Preise
    p = coerce_numeric(d["Preis"])
    cap = coerce_numeric(d["Leistungspreis"])
    d["p_hour"] = p
    missp = d["p_hour"].isna() | (d["p_hour"] == 0)
    can = missp & cap.notna() & d["block_hours"].notna() & (d["block_hours"] > 0)
    d.loc[can, "p_hour"] = cap.loc[can] / d.loc[can, "block_hours"]

    keys = ["product", "direction", "delivery_start", "delivery_end", "block_start", "block_end", "block_hours"]
    blocks = (
        d.groupby(keys, dropna=False)
        .apply(
            lambda x: pd.Series(
                {
                    "p_clear_true": float(np.nanmax(x["p_hour"].values)),
                    "p_vwa_true": float(np.nansum(x["p_hour"].values * x["v"].values) / max(np.nansum(x["v"].values), 1e-9)),
                    "v_awarded_mw_total": float(np.nansum(x["v"].values)),
                    "n_bids": int(len(x)),
                }
            )
        )
        .reset_index()
    )

    for dropc in ["level_0", "index"]:
        if dropc in blocks.columns:
            blocks = blocks.drop(columns=[dropc])

    return blocks.reset_index(drop=True)


def _safe_localize_scalar(ts_naive: pd.Timestamp, tz_: str) -> pd.Timestamp:
    for amb in (True, False, "NaT"):
        try:
            out = ts_naive.tz_localize(tz_, ambiguous=amb, nonexistent="shift_forward")
            if amb == "NaT" and pd.isna(out):
                continue
            return out
        except Exception:
            continue
    return ts_naive.tz_localize("UTC")


def _make_start_end_naive(dday: pd.Timestamp, t0: str, t1: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    dday = pd.Timestamp(dday).normalize()
    t0 = str(t0)
    t1 = str(t1)

    # 24:00 kann nicht als Stunde geparst werden -> Tag verschieben
    if t0 == "24:00":
        start_naive = pd.Timestamp(f"{dday.date()} 00:00") + pd.Timedelta(days=1)
    else:
        start_naive = pd.Timestamp(f"{dday.date()} {t0}")

    if t1 == "24:00":
        end_naive = pd.Timestamp(f"{dday.date()} 00:00") + pd.Timedelta(days=1)
    else:
        end_naive = pd.Timestamp(f"{dday.date()} {t1}")

    if end_naive <= start_naive:
        end_naive = end_naive + pd.Timedelta(days=1)

    return start_naive, end_naive


def build_hourly_from_blocks_agg(blocks: pd.DataFrame, tz_: str) -> pd.DataFrame:
    if blocks is None or blocks.empty:
        return pd.DataFrame()

    rows = []
    for _, r in blocks.iterrows():
        ds = pd.Timestamp(r["delivery_start"]).normalize()
        de = pd.Timestamp(r["delivery_end"]).normalize()

        for dday in pd.date_range(ds, de, freq="D"):
            start_naive, end_naive = _make_start_end_naive(dday, r["block_start"], r["block_end"])
            start_local = _safe_localize_scalar(start_naive, tz_)
            end_local = _safe_localize_scalar(end_naive, tz_)

            start_utc = start_local.tz_convert("UTC")
            end_utc = end_local.tz_convert("UTC")

            ts_key = pd.date_range(start=start_utc, end=end_utc, freq="1H", inclusive="left").tz_localize(None)
            for ts in ts_key:
                rows.append(
                    {
                        "ts_key": ts,
                        "product": r["product"],
                        "direction": r["direction"],
                        "p_clear_true": r["p_clear_true"],
                        "p_vwa_true": r["p_vwa_true"],
                    }
                )

    hourly = pd.DataFrame(rows)
    if hourly.empty:
        return hourly

    hourly["ts_key"] = pd.to_datetime(hourly["ts_key"], errors="coerce").dt.floor("H")
    hourly["prod_dir"] = hourly["product"].str.lower() + "_" + hourly["direction"].str.lower()

    piv = hourly.pivot_table(
        index="ts_key",
        columns="prod_dir",
        values=["p_clear_true", "p_vwa_true"],
        aggfunc={"p_clear_true": "max", "p_vwa_true": "mean"},
    )

    piv.columns = [f"{metric}_{prod_dir}_chf_per_mw_h" for (metric, prod_dir) in piv.columns]
    piv = piv.reset_index()
    piv = piv.rename(columns={c: f"sdl_{c}" for c in piv.columns if c != "ts_key"})
    return piv


@st.cache_data(show_spinner=False)
def cached_load_sdl_fast(cache_version: str, file_bytes: bytes, filename: str, sep_choice: str | None):
    class _F:
        def __init__(self, b, n):
            self._b = b
            self.name = n

        def getvalue(self):
            return self._b

    f = _F(file_bytes, filename)
    return load_swissgrid_sdl_fast(f, sep_choice=sep_choice)


@st.cache_data(show_spinner=False)
def cached_blocks_agg(cache_version: str, sdl_df: pd.DataFrame, filter_ch: bool, products_tuple: tuple[str, ...]):
    return parse_sdl_to_blocks_agg(sdl_df, filter_ch=filter_ch, products=list(products_tuple))


@st.cache_data(show_spinner=False)
def cached_hourly_from_blocks(cache_version: str, blocks: pd.DataFrame, tz_: str):
    return build_hourly_from_blocks_agg(blocks, tz_=tz_)


# =============================================================================
# Tabs
# =============================================================================
tab_load, tab_build = st.tabs(["A) Bestehenden Master laden", "B) Master 2025 erstellen"])


# =============================================================================
# TAB A) Bestehenden Master laden
# =============================================================================
with tab_load:
    st.subheader("Bestehenden Master laden (Parquet/CSV)")

    col1, col2 = st.columns([1, 1])
    with col1:
        up = st.file_uploader(
            "Master-Datei hochladen (.parquet oder .csv)",
            type=["parquet", "csv"],
            accept_multiple_files=False,
            key="upl_master_existing",
            help="Lädt eine vorhandene Master-Datei in die Session. Es wird automatisch ts_key erzeugt/normalisiert.",
        )
    with col2:
        local_path = st.text_input(
            "…oder lokalen Pfad laden (optional)",
            value=os.path.join("data", "master_2025_hourly.parquet"),
            key="master_path_existing",
            help="Optional: Pfad zu einer lokalen Master-Datei (Parquet oder CSV).",
        )
        load_path_btn = st.button("📂 Von Pfad laden", key="btn_load_master_path", help="Lädt die Datei vom angegebenen Pfad in die Session.")

    def load_from_uploaded(uploaded):
        name = uploaded.name.lower()
        raw = uploaded.getvalue()
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(raw))
        return pd.read_csv(io.BytesIO(raw))

    if up is not None:
        try:
            loaded = load_from_uploaded(up)
            if "timestamp" in loaded.columns and "ts" not in loaded.columns:
                loaded = loaded.rename(columns={"timestamp": "ts"})
            loaded = ensure_ts_key(loaded, tz_assume=tz, sdl_ts_key=None)
            st.session_state["master_2025"] = loaded
            st.success(f"Aus Upload geladen: {up.name} | Zeilen={len(loaded)} Spalten={len(loaded.columns)}")
        except Exception as e:
            st.error(f"Laden fehlgeschlagen: {e}")

    if load_path_btn:
        try:
            cand = local_path
            if not os.path.isabs(cand):
                cand = os.path.join(os.getcwd(), cand)
            if not os.path.exists(cand):
                raise FileNotFoundError(f"Datei nicht gefunden: {cand}")
            if cand.lower().endswith(".parquet"):
                df = pd.read_parquet(cand)
            elif cand.lower().endswith(".csv"):
                df = pd.read_csv(cand)
            else:
                raise ValueError("Für den Pfad-Import werden nur .parquet oder .csv unterstützt.")
            if "timestamp" in df.columns and "ts" not in df.columns:
                df = df.rename(columns={"timestamp": "ts"})
            df = ensure_ts_key(df, tz_assume=tz, sdl_ts_key=None)
            st.session_state["master_2025"] = df
            st.success(f"Von Pfad geladen: {cand} | Zeilen={len(df)} Spalten={len(df.columns)}")
        except Exception as e:
            st.error(f"Laden vom Pfad fehlgeschlagen: {e}")

    master = st.session_state.get("master_2025")
    if master is not None:
        st.markdown("### Aktueller Master in der Session (Vorschau: 50 Zeilen)")
        st.write(f"Zeilen: {len(master)} | Spalten: {len(master.columns)}")
        st.dataframe(master.head(50), use_container_width=True, height=360)
    else:
        st.info("Noch kein Master im Session State. Bitte Datei hochladen oder von Pfad laden.")


# =============================================================================
# TAB B) Master 2025 erstellen
# =============================================================================
with tab_build:
    st.subheader("1) Quellen hochladen & konfigurieren")

    if "wiz" not in st.session_state:
        st.session_state["wiz"] = {}
    W = st.session_state["wiz"]

    DATASET_SPECS = [
        {"key": "price", "title": "Day-Ahead Preis", "values": ["price_da"]},
        {"key": "intraday", "title": "Intraday Continuous Preis (stündlich)", "values": ["price_id"]},
        {"key": "load", "title": "Last", "values": ["load_act", "load_fc_da", "load_fc_id"]},
        {"key": "pv", "title": "PV", "values": ["pv_act", "pv_fc_da", "pv_fc_id"]},
        {"key": "wind", "title": "Wind", "values": ["wind_act", "wind_fc_da", "wind_fc_id"]},
    ]

    def dataset_block(spec):
        key = spec["key"]
        title = spec["title"]
        value_cols = spec["values"]

        st.markdown(f"### {title}")
        colL, colR = st.columns([1.1, 0.9])

        with colL:
            files = st.file_uploader(
                f"{title}: Dateien hochladen (CSV/XLSX) – mehrere Monatsdateien möglich",
                type=["xlsx", "xls", "csv", "txt", "tsv"],
                accept_multiple_files=True,
                key=f"upl_{key}",
                help=(
                    "Du kannst mehrere Dateien (z.B. Monatsdateien) hochladen. "
                    "Die Dateien werden zusammengeführt, sortiert und auf stündlich normalisiert (falls nötig)."
                ),
            )

            sheet_mode = st.selectbox(
                "Excel-Sheet",
                ["__FIRST__", "Sheetname angeben"],
                index=0,
                key=f"sheetmode_{key}",
                help="Bei CSV ignoriert. Bei Excel: entweder erstes Sheet oder ein expliziter Sheetname.",
            )
            sheet_name = None
            if sheet_mode == "Sheetname angeben":
                sheet_name = st.text_input(
                    "Sheetname",
                    value="",
                    key=f"sheet_{key}",
                    help="Name des Excel-Sheets (exakt).",
                )

            header_row = st.number_input(
                "Header-Zeile (0 = erste Zeile)",
                min_value=0,
                value=0,
                step=1,
                key=f"hdr_{key}",
                help="Zeilennummer (0-basiert), in der die Spaltennamen stehen.",
            )

            cols = []
            if files:
                try:
                    df0 = read_any_table(files[0], sheet_name=sheet_name, header=header_row)
                    df0 = unify_columns(df0)
                    st.caption("Vorschau erste Datei (Spalten vereinheitlicht):")
                    st.dataframe(df0.head(30), use_container_width=True, height=320)
                    cols = list(df0.columns)
                except Exception as e:
                    st.error(f"Erste Datei kann nicht gelesen werden: {e}")

        with colR:
            st.markdown("**Zeitstempel-Zuordnung**")
            ts_mode = st.selectbox(
                "Wie ist die Zeit repräsentiert?",
                ["Zeitstempel-Spalte", "Datum + Stunde"],
                index=0,
                key=f"tsmode_{key}",
                help=(
                    "Wähle 'Zeitstempel-Spalte', wenn eine Spalte Datum+Zeit enthält. "
                    "Wähle 'Datum + Stunde', wenn Datum und Stunde getrennt vorliegen."
                ),
            )

            ts_col = date_col = hour_col = None
            if cols:
                if ts_mode == "Zeitstempel-Spalte":
                    default_ts_idx = cols.index("timestamp") if "timestamp" in cols else 0
                    ts_col = st.selectbox(
                        "Zeitstempel-Spalte",
                        cols,
                        index=default_ts_idx,
                        key=f"tscol_{key}",
                        help="Spalte mit Datum/Uhrzeit (wird via pandas.to_datetime geparst).",
                    )
                else:
                    default_date_idx = cols.index("datum") if "datum" in cols else 0
                    default_hour_idx = cols.index("stunde") if "stunde" in cols else min(1, len(cols) - 1)
                    date_col = st.selectbox(
                        "Datums-Spalte",
                        cols,
                        index=default_date_idx,
                        key=f"datecol_{key}",
                        help="Spalte mit Datum (z.B. 01.01.2025).",
                    )
                    hour_col = st.selectbox(
                        "Stunden-Spalte",
                        cols,
                        index=default_hour_idx,
                        key=f"hourcol_{key}",
                        help="Spalte mit Stunde (0–23). Wird zu Datum addiert.",
                    )
            else:
                st.info("Bitte Dateien hochladen, damit Spalten ausgewählt werden können.")

            st.markdown("**Wert-Spalten**")
            chosen = {}
            if cols:
                if len(value_cols) == 1:
                    target = value_cols[0]
                    default_idx = cols.index(target) if target in cols else 0
                    chosen[target] = st.selectbox(
                        "Wert-Spalte",
                        cols,
                        index=default_idx,
                        key=f"val_{key}",
                        help="Spalte mit dem Zielwert (z.B. price_da oder price_id).",
                    )
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        chosen[value_cols[0]] = st.selectbox(
                            "Ist-Wert",
                            ["(none)"] + cols,
                            index=0,
                            key=f"act_{key}",
                            help="Optional: Ist-Wert (z.B. pv_act).",
                        )
                    with c2:
                        chosen[value_cols[1]] = st.selectbox(
                            "Forecast Day-Ahead",
                            ["(none)"] + cols,
                            index=0,
                            key=f"fcda_{key}",
                            help="Optional: Day-Ahead Prognose (z.B. pv_fc_da).",
                        )
                    with c3:
                        chosen[value_cols[2]] = st.selectbox(
                            "Forecast Intraday",
                            ["(none)"] + cols,
                            index=0,
                            key=f"fcid_{key}",
                            help="Optional: Intraday Prognose (z.B. pv_fc_id).",
                        )
            else:
                st.info("Bitte Dateien hochladen, damit Wert-Spalten ausgewählt werden können.")

            already_hourly = st.checkbox(
                "Daten sind bereits stündlich",
                value=True,
                key=f"hourly_{key}",
                help=(
                    "Wenn deaktiviert, werden Daten per Resampling auf 1H gebracht. "
                    "Aggregation wird unten gewählt."
                ),
            )
            agg = st.selectbox(
                "Aggregation bei Resampling",
                ["mean", "sum"],
                index=0,
                key=f"agg_{key}",
                help="Nur relevant, wenn 'Daten sind bereits stündlich' deaktiviert ist.",
            )

        W[key] = {
            "files": files,
            "sheet_name": sheet_name if sheet_mode == "Sheetname angeben" else "__FIRST__",
            "header_row": int(header_row),
            "ts_mode": ts_mode,
            "ts_col": ts_col,
            "date_col": date_col,
            "hour_col": hour_col,
            "value_map": chosen,
            "already_hourly": bool(already_hourly),
            "agg": agg,
        }

    for spec in DATASET_SPECS:
        with st.expander(f"{spec['title']}", expanded=(spec["key"] in ("price", "intraday", "pv", "wind", "load"))):
            dataset_block(spec)

    # -------------------------------------------------------------------------
    # SDL Import UI (BELASSEN) — nur Labels/Tooltips deutsch (keine Logikänderung)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("1b) Swissgrid SDL (PRL + SRL) – Schneller Import (Vorschau + sdl_prices)")

    sdl_up = st.file_uploader(
        "Swissgrid SDL Resultate hochladen (CSV/TSV/XLSX)",
        type=["csv", "txt", "tsv", "xlsx", "xls"],
        accept_multiple_files=False,
        key="upl_sdl",
        help="Importiert die Swissgrid Offer-/Resultate-Liste und baut daraus stündliche SDL-Clearing-Zeitreihen.",
    )

    sdl_sep_label = st.selectbox(
        "Trennzeichen (Swissgrid SDL)",
        ["AUTO (empfohlen)", "\\t (Tab)", ";", ","],
        index=0,
        key="sdl_sep",
        help="Bei CSV/TSV: AUTO erkennt das Trennzeichen anhand der Headerzeile.",
    )
    sdl_sep = {"\\t (Tab)": "\t", ";": ";", ",": ","}.get(sdl_sep_label, None)

    cA, cB, cC, cD = st.columns([0.25, 0.25, 0.25, 0.25])
    with cA:
        sdl_filter_ch = st.checkbox("Land = CH filtern", value=True, key="sdl_filter_ch", help="Nur Einträge mit Land=CH berücksichtigen.")
    with cB:
        sdl_products = st.multiselect(
            "Produkte importieren",
            options=["PRL", "SRL"],
            default=["PRL", "SRL"],
            key="sdl_products",
            help="Welche Produkte aus der SDL-Liste verarbeitet werden sollen.",
        )
    with cC:
        merge_sdl_into_master = st.checkbox(
            "SDL in Master mergen",
            value=True,
            key="merge_sdl_into_master",
            help="Fügt sdl_* Spalten dem Master via ts_key hinzu (additiv, DA/ID unverändert).",
        )
    with cD:
        save_sdl_prices = st.checkbox(
            "sdl_prices.parquet speichern",
            value=True,
            key="save_sdl_prices",
            help="Speichert die stündlichen SDL-Zeitreihen als eigenes Artefakt im Szenario-Ordner.",
        )

    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        preview_btn = st.button("📥 SDL einlesen (Vorschau + Blöcke)", key="btn_sdl_preview", help="Liest die Datei und zeigt Vorschau + Block-Aggregation.")
    with c2:
        build_sdl_hourly_btn = st.button("🧱 SDL stündlich bauen (sdl_prices)", key="btn_sdl_build_hourly", help="Erzeugt stündliche Zeitreihen aus den Blöcken.")

    with st.expander("SDL: Vorschau / Blöcke / Stündlich", expanded=False):
        if preview_btn:
            if sdl_up is None:
                st.warning("Bitte zuerst eine SDL-Datei hochladen.")
            else:
                try:
                    file_bytes = sdl_up.getvalue()
                    with st.status("Lese SDL (FAST + Cache)…", expanded=False):
                        sdl_df, meta = cached_load_sdl_fast(CACHE_VERSION, file_bytes, sdl_up.name, sdl_sep)

                    st.write("SDL Meta:", meta)
                    st.write("Spalten:", list(sdl_df.columns))
                    st.write("Shape:", sdl_df.shape)
                    st.dataframe(sdl_df.head(50), use_container_width=True, height=300)

                    with st.status("Parse & Block-Aggregation (Cache)…", expanded=False):
                        blocks = cached_blocks_agg(CACHE_VERSION, sdl_df, filter_ch=sdl_filter_ch, products_tuple=tuple(sdl_products))

                    st.session_state["sdl_meta"] = meta
                    st.session_state["sdl_blocks_agg"] = blocks

                    st.success(f"Aggregierte Blöcke: {len(blocks):,}")
                    st.dataframe(blocks.head(50), use_container_width=True, height=300)

                    if not blocks.empty:
                        st.write(
                            {
                                "min_delivery_start": str(blocks["delivery_start"].min()),
                                "max_delivery_end": str(blocks["delivery_end"].max()),
                                "produkte": sorted(blocks["product"].dropna().unique().tolist()),
                                "richtungen": sorted(blocks["direction"].dropna().unique().tolist()),
                            }
                        )

                except Exception as e:
                    st.error(f"SDL-Parsing fehlgeschlagen: {e}")

        if build_sdl_hourly_btn:
            blocks = st.session_state.get("sdl_blocks_agg")
            if not isinstance(blocks, pd.DataFrame) or blocks.empty:
                st.warning("Keine SDL-Blöcke vorhanden. Bitte zuerst 'SDL einlesen (Vorschau + Blöcke)' ausführen.")
            else:
                try:
                    with st.status("Expandiere SDL-Blöcke → stündlich…", expanded=False):
                        sdl_hourly = cached_hourly_from_blocks(CACHE_VERSION, blocks, tz_=tz)

                    if sdl_hourly is None or sdl_hourly.empty:
                        st.warning("SDL stündlich ist leer. Prüfe SDL-Datei (Zeiträume/Spalten).")
                    else:
                        sdl_hourly["ts_key"] = pd.to_datetime(sdl_hourly["ts_key"], errors="coerce").dt.floor("H")
                        sdl_hourly["ts"] = sdl_hourly["ts_key"]
                        sdl_hourly = sdl_hourly.sort_values("ts_key").reset_index(drop=True)

                        st.session_state["sdl_hourly"] = sdl_hourly
                        st.success(f"SDL stündlich gebaut: Zeilen={len(sdl_hourly):,} Spalten={len(sdl_hourly.columns):,}")
                        st.dataframe(sdl_hourly.head(80), use_container_width=True, height=320)

                        if save_sdl_prices:
                            save_parquet(sname, "sdl_prices", sdl_hourly)
                            st.success(f"sdl_prices gespeichert: data/scenarios/{sname}/sdl_prices.parquet")

                except Exception as e:
                    st.error(f"SDL stündlich bauen fehlgeschlagen: {e}")

    # -------------------------------------------------------------------------
    # Master erstellen
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("2) Master 2025 erstellen")

    output_path = st.text_input(
        "Speicherpfad (Parquet)",
        value=os.path.join("data", "master_2025_hourly.parquet"),
        key="output_path_master",
        help="Zielpfad für die erstellte Master-Datei (wird lokal als Parquet gespeichert).",
    )
    build_btn = st.button(
        "🧱 Master 2025 erstellen",
        type="primary",
        key="btn_build_master",
        help="Erstellt den stündlichen Master, erzeugt ts_key und merged Intraday (price_id) via ts_key.",
    )

    def process_dataset(key: str) -> pd.DataFrame | None:
        cfg = W.get(key, {})
        files = cfg.get("files") or []
        if not files:
            return None

        vmap = cfg.get("value_map", {})
        ts_mode = cfg.get("ts_mode", "Zeitstempel-Spalte")
        ts_col = cfg.get("ts_col")
        date_col = cfg.get("date_col")
        hour_col = cfg.get("hour_col")
        sheet_name = cfg.get("sheet_name", "__FIRST__")
        header_row = int(cfg.get("header_row", 0))
        already_hourly = bool(cfg.get("already_hourly", True))
        agg = cfg.get("agg", "mean")

        parts = []
        for f in files:
            df = read_any_table(f, sheet_name=None if sheet_name == "__FIRST__" else sheet_name, header=header_row)
            df = unify_columns(df)

            if ts_mode == "Zeitstempel-Spalte":
                if ts_col is None:
                    raise ValueError(f"[{key}] Zeitstempel-Spalte wurde nicht ausgewählt.")
                ts_col_real = resolve_col(df, ts_col)
                if ts_col_real is None:
                    raise ValueError(f"[{key}] Zeitstempel-Spalte nicht in dieser Datei gefunden: '{ts_col}'")
                ts = pd.to_datetime(df[ts_col_real], errors="coerce", dayfirst=True)
            else:
                date_col_real = resolve_col(df, date_col) if date_col else None
                hour_col_real = resolve_col(df, hour_col) if hour_col else None
                if date_col_real is None or hour_col_real is None:
                    raise ValueError(f"[{key}] Datum/Stunde-Spalten nicht in dieser Datei gefunden.")
                d0 = pd.to_datetime(df[date_col_real], errors="coerce", dayfirst=True)
                h0 = pd.to_numeric(df[hour_col_real], errors="coerce")
                ts = d0 + pd.to_timedelta(h0.fillna(0).astype(int), unit="h")

            df_out = pd.DataFrame({"ts": ts})

            if key == "price":
                src_real = resolve_col(df, vmap.get("price_da"))
                if src_real is not None:
                    df_out["price_da"] = coerce_numeric(df[src_real])

            elif key == "intraday":
                src_real = resolve_col(df, vmap.get("price_id"))
                if src_real is not None:
                    df_out["price_id"] = coerce_numeric(df[src_real])

            elif key in ("load", "pv", "wind"):
                for out_col in [f"{key}_act", f"{key}_fc_da", f"{key}_fc_id"]:
                    src_real = resolve_col(df, vmap.get(out_col))
                    if src_real is not None:
                        df_out[out_col] = coerce_numeric(df[src_real])

            df_out = df_out.dropna(subset=["ts"]).sort_values("ts")
            parts.append(df_out)

        if not parts:
            return None

        all_df = pd.concat(parts, axis=0, ignore_index=True)
        all_df = all_df.dropna(subset=["ts"]).sort_values("ts")

        if not already_hourly:
            all_df = all_df.set_index("ts").resample("1H").agg(agg).reset_index()

        # Robust de-dup: letzter NICHT-null Wert pro Spalte
        all_df = all_df.sort_values("ts")

        def _last_non_null(s: pd.Series):
            s2 = s.dropna()
            return s2.iloc[-1] if not s2.empty else np.nan

        all_df = (
            all_df.groupby("ts", as_index=False)
            .agg({c: _last_non_null for c in all_df.columns if c != "ts"})
        )

        return all_df

    def _infer_year_for_full_timeline(df_price: pd.DataFrame) -> int:
        ts_ = pd.to_datetime(df_price["ts"], errors="coerce").dropna()
        return int(ts_.min().year) if not ts_.empty else 2025

    if build_btn:
        try:
            with st.status("Baue Master…", expanded=False):
                df_price = process_dataset("price")
                if df_price is None or "price_da" not in df_price.columns:
                    raise ValueError("Preis-Dataset fehlt oder price_da nicht verfügbar. (Master braucht price_da).")

                df_price = df_price.sort_values("ts").drop_duplicates(subset=["ts"])

                if 8700 <= len(df_price) <= 8800:
                    base = df_price[["ts", "price_da"]].copy()
                else:
                    year = _infer_year_for_full_timeline(df_price)
                    ts_full = pd.date_range(
                        f"{year}-01-01 00:00:00",
                        f"{year+1}-01-01 00:00:00",
                        freq="1H",
                        inclusive="left",
                    )
                    base = pd.DataFrame({"ts": ts_full})
                    base = base.merge(df_price[["ts", "price_da"]], on="ts", how="left")

                for k in ["load", "pv", "wind"]:
                    dfk = process_dataset(k)
                    if dfk is not None:
                        base = base.merge(dfk, on="ts", how="left", suffixes=("", "_x"))

                # ts_key für base (unverändert)
                base = ensure_ts_key(base, tz_assume=tz, sdl_ts_key=None)

                # Intraday Merge (unverändert)
                df_id = process_dataset("intraday")
                if df_id is not None and "price_id" in df_id.columns:
                    df_id = ensure_ts_key(df_id, tz_assume=tz, sdl_ts_key=base["ts_key"])
                    df_id["ts_key"] = pd.to_datetime(df_id["ts_key"], errors="coerce").dt.floor("H")
                    df_id = df_id.sort_values("ts_key").groupby("ts_key", as_index=False)["price_id"].mean()

                    if "price_id" in base.columns:
                        base = base.drop(columns=["price_id"])
                    base = base.merge(df_id[["ts_key", "price_id"]], on="ts_key", how="left")

                    filled = int(base["price_id"].notna().sum())
                    st.success(f"Intraday gemerged: price_id gefüllt {filled:,}/{len(base):,} ({100.0*filled/len(base):.1f}%)")
                else:
                    st.info("Keine Intraday-Preisdatei hochgeladen → Master ohne price_id.")

                # SDL Merge (wie bisher: addiert nur sdl_* Spalten; DA/ID-Spalten bleiben unverändert)
                sdl_hourly = st.session_state.get("sdl_hourly")
                if merge_sdl_into_master and isinstance(sdl_hourly, pd.DataFrame) and not sdl_hourly.empty:
                    sdl_hourly = sdl_hourly.copy()
                    sdl_hourly["ts_key"] = pd.to_datetime(sdl_hourly["ts_key"], errors="coerce").dt.floor("H")
                    sdl_hourly = sdl_hourly.dropna(subset=["ts_key"]).sort_values("ts_key")
                    keep_cols = ["ts_key"] + [c for c in sdl_hourly.columns if str(c).startswith("sdl_")]
                    keep_cols = [c for c in keep_cols if c in sdl_hourly.columns]
                    sdl_hourly = sdl_hourly[keep_cols].copy()

                    base = base.merge(sdl_hourly, on="ts_key", how="left")
                    st.success(f"SDL in Master gemerged: {len(keep_cols)-1} Spalten hinzugefügt.")
                else:
                    st.info("Keine SDL stündlich Daten zum Mergen (oder Merge deaktiviert).")

                st.session_state["master_2025"] = base

                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                base.to_parquet(output_path, index=False)

                if save_to_scenario:
                    save_parquet(st.session_state["scenario_name"], "master", base)

                st.success(
                    f"Master gespeichert: {output_path}"
                    + (f" + data/scenarios/{st.session_state['scenario_name']}/master.parquet" if save_to_scenario else "")
                )

        except Exception as e:
            st.error(f"Erstellen fehlgeschlagen: {e}")

    master2 = st.session_state.get("master_2025")
    if master2 is not None:
        st.markdown("### Master-Vorschau (50 Zeilen)")
        st.write(f"Zeilen: {len(master2)} | Spalten: {len(master2.columns)}")
        st.dataframe(master2.head(50), use_container_width=True, height=420)