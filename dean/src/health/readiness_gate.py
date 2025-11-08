from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from integrations.slack import notify

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    OK = "OK"
    WARN = "WARN"
    DEGRADED = "DEGRADED"


@dataclass
class ReadinessIssue:
    check: str
    problem: str
    fix: str


@dataclass
class HealthState:
    status: HealthStatus
    degraded: bool
    issues: List[ReadinessIssue] = field(default_factory=list)
    message: str = ""
    recommendations: List[str] = field(default_factory=list)
    counters: Dict[str, Any] = field(default_factory=dict)

    def banner(self) -> str:
        emoji = {
            HealthStatus.OK: "✅",
            HealthStatus.WARN: "⚠️",
            HealthStatus.DEGRADED: "🛑",
        }[self.status]
        header = f"{emoji} {self.status} • {self.message or 'Systems nominal'}"
        if not self.issues:
            return header
        lines = [header]
        for issue in self.issues:
            lines.append(f"• {issue.check}: {issue.problem} → {issue.fix}")
        return "\n".join(lines)


def _query_latest_prediction(supabase_client) -> Optional[datetime]:
    response = (
        supabase_client.table("ml_predictions")
        .select("created_at")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    if not rows:
        return None
    raw = rows[0].get("created_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:  # pragma: no cover - guard
        return None


def _active_model_exists(supabase_client) -> bool:
    response = (
        supabase_client.table("ml_models")
        .select("id")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    return bool(rows)


def evaluate_health_state(
    supabase_client,
    ml_pipeline: Optional[Any],
    table_monitor: Optional[Any] = None,
    freshness_hours: int = 12,
) -> HealthState:
    issues: List[ReadinessIssue] = []
    recommendations: List[str] = []

    if supabase_client is None:
        issues.append(
            ReadinessIssue(
                "Supabase Connection",
                "Supabase client unavailable.",
                "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to enable ML features.",
            )
        )
        return HealthState(
            status=HealthStatus.DEGRADED,
            degraded=True,
            issues=issues,
            message="Supabase connection missing",
            recommendations=recommendations,
        )

    # Model availability
    try:
        if not _active_model_exists(supabase_client):
            issues.append(
                ReadinessIssue(
                    "Model Availability",
                    "No active ML model found.",
                    "Retrain models (`python -m dean.src.ml.ml_pipeline --train`).",
                )
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Model availability check failed")
        issues.append(
            ReadinessIssue(
                "Model Availability",
                f"Failed to query ml_models ({exc}).",
                "Verify Supabase connectivity and rerun training.",
            )
        )

    # Prediction freshness
    try:
        latest_prediction = _query_latest_prediction(supabase_client)
        if not latest_prediction:
            issues.append(
                ReadinessIssue(
                    "Prediction Freshness",
                    "No predictions recorded.",
                    "Run the ML pipeline to generate predictions.",
                )
            )
        else:
            if datetime.now(timezone.utc) - latest_prediction > timedelta(hours=freshness_hours):
                issues.append(
                    ReadinessIssue(
                        "Prediction Freshness",
                        f"Last prediction is older than {freshness_hours}h.",
                        "Trigger a prediction refresh by retraining models.",
                    )
                )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction freshness check failed")
        issues.append(
            ReadinessIssue(
                "Prediction Freshness",
                f"Failed to query predictions ({exc}).",
                "Verify Supabase connectivity and rerun training.",
            )
        )

    # Table monitoring
    if table_monitor:
        try:
            insights = table_monitor.get_all_table_insights()
            problematic = [
                name
                for name, health in insights.tables.items()
                if not health.is_healthy
            ]
            if problematic:
                issues.append(
                    ReadinessIssue(
                        "Table Health",
                        f"Problematic tables: {', '.join(problematic[:5])}",
                        "Run table monitor repairs or inspect Supabase logs.",
                    )
                )
            if insights.recommendations:
                recommendations.extend(insights.recommendations)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Table monitoring check failed")
            issues.append(
                ReadinessIssue(
                    "Table Health",
                    f"Failed to compute table health ({exc}).",
                    "Check Supabase connectivity and rerun monitoring.",
                )
            )
    else:
        issues.append(
            ReadinessIssue(
                "Table Health",
                "Table monitor unavailable.",
                "Instantiate table monitoring to validate Supabase writes.",
            )
        )

    # ML pipeline fallback readiness
    if not ml_pipeline or not getattr(ml_pipeline, "decision_engine", None):
        issues.append(
            ReadinessIssue(
                "Fallback Scorer",
                "ML decision engine missing.",
                "Ensure ML pipeline initialised with a decision engine.",
            )
        )

    # Performance metrics availability
    try:
        response = (
            supabase_client.table("performance_metrics")
            .select("id")
            .limit(1)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        if not rows:
            issues.append(
                ReadinessIssue(
                    "Performance Metrics",
                    "No performance metrics captured.",
                    "Verify `_sync_performance_metrics_records` is running.",
                )
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Performance metrics check failed")
        issues.append(
            ReadinessIssue(
                "Performance Metrics",
                f"Failed to query performance metrics ({exc}).",
                "Inspect Supabase write permissions for `performance_metrics`.",
            )
        )

    if issues:
        return HealthState(
            status=HealthStatus.DEGRADED,
            degraded=True,
            issues=issues,
            message="ML readiness issues detected",
            recommendations=recommendations,
        )

    return HealthState(
        status=HealthStatus.OK,
        degraded=False,
        issues=[],
        message="ML systems ready",
        recommendations=recommendations,
    )


def notify_health_state(state: HealthState) -> None:
    banner = state.banner()
    if not banner:
        return
    if state.status is HealthStatus.OK:
        logger.info(banner)
        return
    notify(banner)


def get_reconciled_counters(
    account_snapshot: Dict[str, Any],
    stage_result: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    reconciled_account = dict(account_snapshot)
    stage_counts: Dict[str, Any] = {}

    if stage_result:
        active = stage_result.get("active_count")
        target = stage_result.get("target_count")
        hydrated = stage_result.get("hydrated_active_count")
        created = len(stage_result.get("created_ads", [])) if stage_result.get("created_ads") else 0
        killed = len(stage_result.get("kills", [])) if stage_result.get("kills") else 0
        promoted = len(stage_result.get("promotions", [])) if stage_result.get("promotions") else 0
        caps_enforced = bool(target and active == target)

        stage_counts.update(
            {
                "active": active or 0,
                "hydrated": hydrated or 0,
                "created": created,
                "kills": killed,
                "promotions": promoted,
                "caps_enforced": caps_enforced,
            }
        )

        reconciled_account["active_ads"] = active if active is not None else reconciled_account.get("active_ads")
        reconciled_account["target_ads"] = target
        reconciled_account["caps_enforced"] = caps_enforced

    return reconciled_account, stage_counts


__all__ = [
    "HealthStatus",
    "ReadinessIssue",
    "HealthState",
    "evaluate_health_state",
    "notify_health_state",
    "get_reconciled_counters",
]

