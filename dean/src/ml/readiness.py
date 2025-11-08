from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from integrations.slack import notify


@dataclass
class ReadinessIssue:
    check: str
    problem: str
    fix: str


def _fmt_issue(issue: ReadinessIssue) -> str:
    return f"{issue.check}: {issue.problem} → {issue.fix}"


def evaluate_readiness(
    supabase_client,
    ml_pipeline: Optional[Any],
    table_monitor: Optional[Any] = None,
    freshness_hours: int = 12,
) -> Tuple[bool, List[ReadinessIssue]]:
    """
    Evaluate readiness signals before running automation.

    Returns (degraded_mode_required, issues).
    """
    issues: List[ReadinessIssue] = []
    now = datetime.now(timezone.utc)

    # 1. Model availability
    try:
        response = (
            supabase_client.table("ml_models")
            .select("id, is_active, trained_at")
            .eq("is_active", True)
            .order("trained_at", desc=True)
            .limit(1)
            .execute()
        )
        models = getattr(response, "data", None) or []
        if not models:
            issues.append(
                ReadinessIssue(
                    "Model Availability",
                    "No active ML model found.",
                    "Retrain models (`python -m dean.src.ml.ml_pipeline --train`).",
                )
            )
    except Exception as exc:
        issues.append(
            ReadinessIssue(
                "Model Availability",
                f"Failed to query ml_models ({exc}).",
                "Verify Supabase connectivity and rerun training.",
            )
        )

    # 2. Prediction freshness
    try:
        response = (
            supabase_client.table("ml_predictions")
            .select("created_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = getattr(response, "data", None) or []
        if not rows:
            issues.append(
                ReadinessIssue(
                    "Prediction Freshness",
                    "No predictions recorded.",
                    "Run the ML pipeline to generate predictions.",
                )
            )
        else:
            latest_created = rows[0].get("created_at")
            if latest_created:
                try:
                    ts = datetime.fromisoformat(str(latest_created).replace("Z", "+00:00"))
                    if (now - ts) > timedelta(hours=freshness_hours):
                        issues.append(
                            ReadinessIssue(
                                "Prediction Freshness",
                                f"Last prediction is older than {freshness_hours}h.",
                                "Trigger a prediction refresh by retraining models.",
                            )
                        )
                except Exception:
                    issues.append(
                        ReadinessIssue(
                            "Prediction Freshness",
                            "Unable to parse prediction timestamp.",
                            "Inspect `ml_predictions` table for malformed timestamps.",
                        )
                    )
    except Exception as exc:
        issues.append(
            ReadinessIssue(
                "Prediction Freshness",
                f"Failed to query predictions ({exc}).",
                "Verify Supabase connectivity and rerun training.",
            )
        )

    # 3. Table write health via TableMonitor (if available)
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
                        "Run `python -m dean.src.analytics.table_monitoring --repair` or investigate Supabase logs.",
                    )
                )
        except Exception as exc:
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

    # 4. Fallback scorer availability
    if not ml_pipeline or not getattr(ml_pipeline, "decision_engine", None):
        issues.append(
            ReadinessIssue(
                "Fallback Scorer",
                "ML decision engine missing.",
                "Ensure ML pipeline initialised with a decision engine.",
            )
        )

    # 5. Missing metrics (performance_metrics coverage)
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
                    "Verify stage sync `_sync_performance_metrics_records` is running.",
                )
            )
    except Exception as exc:
        issues.append(
            ReadinessIssue(
                "Performance Metrics",
                f"Failed to query performance metrics ({exc}).",
                "Inspect Supabase write permissions for `performance_metrics`.",
            )
        )

    return (len(issues) > 0, issues)


def notify_degraded_mode(issues: List[ReadinessIssue]) -> None:
    if not issues:
        return
    lines = ["⚠️ *Readiness Gate: Degraded Mode Enabled*"]
    for issue in issues:
        lines.append(f"• {_fmt_issue(issue)}")
    notify("\n".join(lines))

