# core/scenario_store.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd

DEFAULT_ROOT = os.path.join("data", "scenarios")


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", " "))
    return name.replace(" ", "_") or "Scenario"


def scenario_dir(scenario_name: str, root: str = DEFAULT_ROOT) -> str:
    return os.path.join(root, _safe_name(scenario_name))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_scenarios(root: str = DEFAULT_ROOT) -> list[str]:
    if not os.path.exists(root):
        return []
    out = []
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            out.append(d)
    return sorted(out)


def save_config(scenario_name: str, config: Dict[str, Any], root: str = DEFAULT_ROOT) -> str:
    d = scenario_dir(scenario_name, root)
    ensure_dir(d)
    p = os.path.join(d, "config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return p


def load_config(scenario_name: str, root: str = DEFAULT_ROOT) -> Dict[str, Any]:
    p = os.path.join(scenario_dir(scenario_name, root), "config.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_parquet(scenario_name: str, key: str, df: pd.DataFrame, root: str = DEFAULT_ROOT) -> str:
    d = scenario_dir(scenario_name, root)
    ensure_dir(d)
    p = os.path.join(d, f"{key}.parquet")
    df.to_parquet(p, index=False)
    return p


def load_parquet(scenario_name: str, key: str, root: str = DEFAULT_ROOT) -> Optional[pd.DataFrame]:
    p = os.path.join(scenario_dir(scenario_name, root), f"{key}.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p)


def exists_parquet(scenario_name: str, key: str, root: str = DEFAULT_ROOT) -> bool:
    p = os.path.join(scenario_dir(scenario_name, root), f"{key}.parquet")
    return os.path.exists(p)
