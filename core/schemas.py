# schemas.py
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class ScenarioParams:
    markets: Dict[str, bool]
    priority_mode: str
    battery: Dict[str, Any]
    grid: Dict[str, Any]
    tariffs: Dict[str, Any]
    peak_shaving: Dict[str, Any]
    optimizer: Dict[str, Any]
    run_days: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "markets": self.markets,
            "priority_mode": self.priority_mode,
            "battery": self.battery,
            "grid": self.grid,
            "tariffs": self.tariffs,
            "peak_shaving": self.peak_shaving,
            "optimizer": self.optimizer,
            "run_days": self.run_days,
        }
