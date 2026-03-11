# swissgrid_sdl_imports.py
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:50:57 2026

@author: fabia
"""

# core/swissgrid_sdl_import.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd


_COLS = [
    "Ausschreibung", "Beschreibung", "Angebotenes Volumen", "Einheit",
    "Zugesprochenes Volumen", "Einheit", "Leistungspreis", "Einheit",
    "Kosten", "Einheit", "Preis", "Einheit", "Land",
    "Angebotspreis", "Einheit", "Teilbarkeit"
]

_RE_TIME = re.compile(r"(\d{2}:\d{2})\s*bis\s*(\d{2}:\d{2})")
_RE_DIR = re.compile(r"\b(UP|DOWN)\b", re.IGNORECASE)
_RE_PROD = re.compile(r"^(PRL|SRL|TRL)_", re.IGNORECASE)
_RE_DATE = re.compile(r"_(\d{2})_(\d{2})_(\d{2})$")   # ..._YY_MM_DD
_RE_WEEK = re.compile(r"_KW(\d{2})$", re.IGNORECASE)  # ..._KW01


def _to_float(x) -> float:
    # Swissgrid CSV may contain commas; robust parse
    if x is None:
        return float("nan")
    s = str(x).strip().replace("'", "")
    if s == "":
        return float("nan")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _parse_product(ausschreibung: str) -> Optional[str]:
    if ausschreibung is None:
        return None
    m = _RE_PROD.search(str(ausschreibung).strip())
    return m.group(1).upper() if m else None


def _parse_delivery_date(ausschreibung: str) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
    """
    Returns (delivery_date, delivery_tag_type) where type is 'date' or 'week' or None.
    For *_YY_MM_DD we map to 20YY-MM-DD.
    For *_KWxx we return None date (week-based, handled separately if needed).
    """
    if ausschreibung is None:
        return None, None
    s = str(ausschreibung).strip()

    m = _RE_DATE.search(s)
    if m:
        yy, mm, dd = map(int, m.groups())
        # assume 2000-2099
        return pd.Timestamp(year=2000 + yy, month=mm, day=dd), "date"

    m = _RE_WEEK.search(s)
    if m:
        return None, "week"

    return None, None


def _parse_direction(desc: str) -> Optional[str]:
    if desc is None:
        return None
    m = _RE_DIR.search(str(desc))
    return m.group(1).upper() if m else None


def _parse_block_times(desc: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Parse 'HH:MM bis HH:MM' from description.
    Returns (start_str, end_str, block_hours). If none, returns (None, None, None).
    """
    if desc is None:
        return None, None, None
    m = _RE_TIME.search(str(desc))
    if not m:
        return None, None, None
    t0, t1 = m.group(1), m.group(2)
    # block hours (treat 24:00 as 00:00 + 24)
    h0 = int(t0[:2]); m0 = int(t0[3:])
    h1 = int(t1[:2]); m1 = int(t1[3:])
    minutes0 = 60*h0 + m0
    minutes1 = 60*h1 + m1
    if t1 == "24:00":
        minutes1 = 24*60
    if minutes1 <= minutes0:
        # allow wrap, but for Swissgrid blocks it should not wrap
        minutes1 += 24*60
    dur_h = int((minutes1 - minutes0) / 60)
    return t0, t1, dur_h


def load_and_normalize_swissgrid_csv(
    file_or_buffer,
    delimiter: str = "\t",
) -> pd.DataFrame:
    """
    Reads Swissgrid SDL results (big) and returns normalized row-level records.
    Keeps the same granularity as CSV (bid rows).
    """
    df = pd.read_csv(
        file_or_buffer,
        sep=delimiter,
        dtype=str,
        na_filter=False,
        encoding_errors="ignore",
    )

    # Keep only known columns if extra appear
    keep = [c for c in _COLS if c in df.columns]
    df = df[keep].copy()

    # Normalize core fields
    df["product"] = df["Ausschreibung"].apply(_parse_product)
    ddate, dtype = zip(*df["Ausschreibung"].apply(_parse_delivery_date))
    df["delivery_date"] = pd.to_datetime(list(ddate), errors="coerce")
    df["delivery_tag_type"] = list(dtype)

    df["direction"] = df["Beschreibung"].apply(_parse_direction)
    bt = df["Beschreibung"].apply(_parse_block_times)
    df["block_start"], df["block_end"], df["block_hours"] = zip(*bt)

    # Numerics
    df["vol_offered_mw"] = df["Angebotenes Volumen"].apply(_to_float)
    df["vol_awarded_mw"] = df["Zugesprochenes Volumen"].apply(_to_float)
    df["capacity_price_per_mw_block"] = df["Leistungspreis"].apply(_to_float)  # CHF/MW or EUR/MW (block)
    df["costs"] = df["Kosten"].apply(_to_float)

    # Prefer 'Preis' (CHF/MWh*) which equals CHF/(MW*h) in your samples
    df["price_chf_per_mw_h"] = df["Preis"].apply(_to_float)
    df["bid_price_chf_per_mw_h"] = df["Angebotspreis"].apply(_to_float)

    # Unit fields (optional)
    df["currency_cap"] = df.loc[:, df.columns.str.contains("Einheit")].astype(str).agg(" | ".join, axis=1)

    # Country
    if "Land" in df.columns:
        df["country"] = df["Land"].astype(str).str.strip()
    else:
        df["country"] = None

    # Basic filter: awarded > 0 rows only (keeps accepted bids)
    df = df[df["vol_awarded_mw"].fillna(0) > 0].copy()

    # If price is missing, derive price_per_mw_h from capacity price / block_hours (if possible)
    mask_missing = df["price_chf_per_mw_h"].isna() | (df["price_chf_per_mw_h"] == 0)
    can_derive = mask_missing & df["capacity_price_per_mw_block"].notna() & df["block_hours"].notna() & (df["block_hours"] > 0)
    df.loc[can_derive, "price_chf_per_mw_h"] = df.loc[can_derive, "capacity_price_per_mw_block"] / df.loc[can_derive, "block_hours"]

    return df.reset_index(drop=True)


def build_block_clearing_from_raw(
    raw: pd.DataFrame,
    product: str = "SRL",
    country: str = "CH",
) -> pd.DataFrame:
    """
    Aggregates bid rows into block-level prices:
      - p_clear_true: max(price) of accepted bids (marginal accepted proxy)
      - p_vwa_true  : volume-weighted average
    Returns one row per (delivery_date, direction, block_start, block_end).
    """
    df = raw.copy()

    df = df[df["product"] == product].copy()
    if country is not None:
        df = df[df["country"] == country].copy()

    # Only blocks with explicit times (we handle KW products later if needed)
    df = df[df["delivery_tag_type"] == "date"].copy()
    df = df[df["block_hours"].notna() & (df["block_hours"] > 0)].copy()

    key = ["delivery_date", "direction", "block_start", "block_end", "block_hours"]

    # Use price_chf_per_mw_h as the standardized capacity price per MW and hour
    df["p"] = pd.to_numeric(df["price_chf_per_mw_h"], errors="coerce")
    df["v"] = pd.to_numeric(df["vol_awarded_mw"], errors="coerce")

    g = df.groupby(key, dropna=False)

    out = g.apply(lambda x: pd.Series({
        "p_clear_true": x["p"].max(),
        "p_vwa_true": (x["p"] * x["v"]).sum() / max(x["v"].sum(), 1e-9),
        "v_awarded_mw_total": x["v"].sum(),
        "n_bids": len(x),
    })).reset_index()

    return out


def explode_blocks_to_hourly(
    blocks: pd.DataFrame,
    tz: Optional[str] = None,
) -> pd.DataFrame:
    """
    Expands block-level records into hourly records with a naive UTC key later.
    Output columns:
      ts_local (naive), delivery_date, direction, p_clear_true, p_vwa_true
    """
    rows = []
    for _, r in blocks.iterrows():
        d = pd.Timestamp(r["delivery_date"])
        t0 = r["block_start"]; t1 = r["block_end"]

        # build local naive timestamps (we later merge using ts_key from master)
        start = pd.Timestamp(f"{d.date()} {t0}")
        if t1 == "24:00":
            end = pd.Timestamp(f"{d.date()} 00:00") + pd.Timedelta(hours=24)
        else:
            end = pd.Timestamp(f"{d.date()} {t1}")

        # hourly starts
        rng = pd.date_range(start=start, end=end, freq="1H", inclusive="left")
        for ts in rng:
            rows.append({
                "ts_local": ts,  # still naive; merge via ts_key later
                "delivery_date": d.normalize(),
                "direction": r["direction"],
                "p_clear_true": float(r["p_clear_true"]),
                "p_vwa_true": float(r["p_vwa_true"]),
                "block_start": t0,
                "block_end": t1,
                "block_hours": int(r["block_hours"]),
            })

    return pd.DataFrame(rows)
