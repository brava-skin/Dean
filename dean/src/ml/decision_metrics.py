from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict


@dataclass
class DecisionCounts:
    ml_assisted: int = 0
    rule_only: int = 0


class DecisionMetricsStore:
    """Thread-safe counter for ML-assisted vs rule-only decisions."""

    def __init__(self) -> None:
        self._counts = DecisionCounts()
        self._lock = Lock()

    def record(self, used_ml: bool) -> None:
        with self._lock:
            if used_ml:
                self._counts.ml_assisted += 1
            else:
                self._counts.rule_only += 1

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "ml_assisted": self._counts.ml_assisted,
                "rule_only": self._counts.rule_only,
            }

    def reset(self) -> None:
        with self._lock:
            self._counts = DecisionCounts()


decision_metrics = DecisionMetricsStore()

