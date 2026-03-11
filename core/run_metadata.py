# core/run_metadata.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


RUN_INFO_FILENAME = "run_info.parquet"


@dataclass(frozen=True)
class RunInfo:
    """
    Minimal scenario run metadata to make the UI unambiguous about active forecast modes.

    Stored as a single-row parquet under:
        data/scenarios/<scenario>/run_info.parquet
    """

    run_ts_utc: pd.Timestamp  # tz-naive UTC timestamp
    mode_da: str              # "perfect" | "ml"
    mode_id: str              # "perfect" | "ml"
    pf_horizon_da_h: int
    pf_horizon_id_h: int

    # Optional fields for future-proofing (keep None if not used)
    model_tag_da: Optional[str] = None
    model_tag_id: Optional[str] = None
    notes: Optional[str] = None


def _utc_now_naive() -> pd.Timestamp:
    # Ensure UTC, tz-naive for consistent parquet storage
    return pd.Timestamp.utcnow().tz_localize(None)


def build_run_info(
    *,
    perfect_forecast_da: bool,
    perfect_forecast_id: bool,
    pf_horizon_da_h: int,
    pf_horizon_id_h: int,
    model_tag_da: Optional[str] = None,
    model_tag_id: Optional[str] = None,
    notes: Optional[str] = None,
    run_ts_utc: Optional[pd.Timestamp] = None,
) -> RunInfo:
    mode_da = "perfect" if perfect_forecast_da else "ml"
    mode_id = "perfect" if perfect_forecast_id else "ml"

    if run_ts_utc is None:
        run_ts_utc = _utc_now_naive()
    else:
        # Normalize to tz-naive UTC
        if getattr(run_ts_utc, "tzinfo", None) is not None:
            run_ts_utc = run_ts_utc.tz_convert("UTC").tz_localize(None)

    return RunInfo(
        run_ts_utc=run_ts_utc,
        mode_da=mode_da,
        mode_id=mode_id,
        pf_horizon_da_h=int(pf_horizon_da_h),
        pf_horizon_id_h=int(pf_horizon_id_h),
        model_tag_da=model_tag_da,
        model_tag_id=model_tag_id,
        notes=notes,
    )


def save_run_info(scenario_dir: str | Path, run_info: RunInfo) -> Path:
    scenario_dir = Path(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    out_path = scenario_dir / RUN_INFO_FILENAME

    row: Dict[str, Any] = asdict(run_info)
    df = pd.DataFrame([row])

    # Ensure stable dtypes
    df["run_ts_utc"] = pd.to_datetime(df["run_ts_utc"])
    df["mode_da"] = df["mode_da"].astype("string")
    df["mode_id"] = df["mode_id"].astype("string")
    df["pf_horizon_da_h"] = df["pf_horizon_da_h"].astype("int64")
    df["pf_horizon_id_h"] = df["pf_horizon_id_h"].astype("int64")
    if "model_tag_da" in df.columns:
        df["model_tag_da"] = df["model_tag_da"].astype("string")
    if "model_tag_id" in df.columns:
        df["model_tag_id"] = df["model_tag_id"].astype("string")
    if "notes" in df.columns:
        df["notes"] = df["notes"].astype("string")

    df.to_parquet(out_path, index=False)
    return out_path


def load_run_info(scenario_dir: str | Path) -> Optional[RunInfo]:
    scenario_dir = Path(scenario_dir)
    path = scenario_dir / RUN_INFO_FILENAME
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if df.empty:
        return None

    row = df.iloc[0].to_dict()

    # Parse timestamp safely
    run_ts = pd.to_datetime(row.get("run_ts_utc", None))
    if pd.isna(run_ts):
        run_ts = _utc_now_naive()
    else:
        # Keep tz-naive
        if getattr(run_ts, "tzinfo", None) is not None:
            run_ts = run_ts.tz_convert("UTC").tz_localize(None)

    return RunInfo(
        run_ts_utc=run_ts,
        mode_da=str(row.get("mode_da", "ml")),
        mode_id=str(row.get("mode_id", "ml")),
        pf_horizon_da_h=int(row.get("pf_horizon_da_h", 0)),
        pf_horizon_id_h=int(row.get("pf_horizon_id_h", 0)),
        model_tag_da=(None if pd.isna(row.get("model_tag_da", None)) else str(row.get("model_tag_da"))),
        model_tag_id=(None if pd.isna(row.get("model_tag_id", None)) else str(row.get("model_tag_id"))),
        notes=(None if pd.isna(row.get("notes", None)) else str(row.get("notes"))),
    )