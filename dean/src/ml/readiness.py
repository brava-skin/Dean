"""
Legacy compatibility wrappers for the readiness gate.

The canonical implementation now lives in `health.readiness_gate`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from health.readiness_gate import (
    HealthState,
    HealthStatus,
    ReadinessIssue,
    evaluate_health_state,
    notify_health_state,
)


def evaluate_readiness(
    supabase_client,
    ml_pipeline: Optional[Any],
    table_monitor: Optional[Any] = None,
    freshness_hours: int = 12,
) -> Tuple[bool, List[ReadinessIssue]]:
    state = evaluate_health_state(
        supabase_client,
        ml_pipeline,
        table_monitor,
        freshness_hours=freshness_hours,
    )
    return state.degraded, list(state.issues)


def notify_degraded_mode(issues: List[ReadinessIssue]) -> None:
    if not issues:
        return
    state = HealthState(
        status=HealthStatus.DEGRADED,
        degraded=True,
        issues=issues,
        message="ML readiness issues detected",
    )
    notify_health_state(state)


__all__ = [
    "ReadinessIssue",
    "evaluate_readiness",
    "notify_degraded_mode",
]

