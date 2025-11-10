"""
ASC+ Campaign Stage Handler
Manages a single Advantage+ Shopping Campaign with 5 creatives
"""

from __future__ import annotations

import logging
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Iterable
import math

import pandas as pd
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def _asc_log(level: int, message: str, *args: Any) -> None:
    logger.log(level, f"[ASC] {message}", *args)

from integrations.slack import notify, alert_kill, alert_error
from integrations import fmt_eur, fmt_int
from integrations.meta_client import MetaClient
from infrastructure.utils import (
    getenv_f, getenv_i, getenv_b, cfg, cfg_or_env_f, cfg_or_env_i, cfg_or_env_b, cfg_or_env_list,
    safe_f, today_str, daily_key, ad_day_flag_key, now_minute_key, clean_text_token, prettify_ad_name, Timekit
)
from creative.image_generator import create_image_generator, ImageCreativeGenerator
from config.constants import (
    ASC_PLUS_BUDGET_MIN,
    ASC_PLUS_BUDGET_MAX,
    ASC_PLUS_MIN_BUDGET_PER_CREATIVE,
    MAX_STAGE_DURATION_HOURS,
    DB_NUMERIC_MAX,
    CREATIVE_PERFORMANCE_STAGE_VALUE,
)
from infrastructure.data_validation import (
    validate_all_timestamps,
    validate_supabase_data,
    ValidationError,
    validate_and_sanitize_data,
)
from config.validate import validate_asc_plus_config

# Import advanced ML systems
try:
    from ml.advanced_system import create_advanced_ml_system
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    logger.warning("Advanced ML system not available")

# Import new optimization systems
try:
    from ml.budget_scaling import create_budget_scaling_engine, ScalingStrategy
    from ml.creative_refresh import create_creative_refresh_manager
    from infrastructure.optimization import create_resource_optimizer
    OPTIMIZATION_SYSTEMS_AVAILABLE = True
except ImportError:
    OPTIMIZATION_SYSTEMS_AVAILABLE = False
    logger.warning("Optimization systems not available")

UTC = timezone.utc
LOCAL_TZ = ZoneInfo(os.getenv("ACCOUNT_TZ", os.getenv("ACCOUNT_TIMEZONE", "Europe/Amsterdam")))
ACCOUNT_CURRENCY = os.getenv("ACCOUNT_CURRENCY", "EUR")
HYDRATION_SECONDS = int(os.getenv("ASC_PLUS_HYDRATION_SECONDS", "180"))


_CREATIVE_PERFORMANCE_STAGE_DISABLED = False


def _roas(row: Dict[str, Any]) -> float:
    """Extract ROAS from purchase_roas array."""
    roas_list = row.get("purchase_roas") or []
    try:
        if roas_list:
            return float(roas_list[0].get("value", 0)) or 0.0
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    return 0.0


def _purchase_and_atc_counts(row: Dict[str, Any]) -> Tuple[int, int]:
    acts = row.get("actions") or []
    purch = 0
    atc = 0
    for a in acts:
        t = a.get("action_type")
        v = safe_f(a.get("value"), 0.0)
        if t == "purchase":
            purch += int(v)
        elif t == "add_to_cart":
            atc += int(v)
    return purch, atc


def _fraction_or_none(value: Any, *, allow_percent: bool = True) -> Optional[float]:
    """Convert Meta metric to a fraction within [0, 1], tolerating percentage inputs."""
    if value in (None, "", float("inf"), float("-inf")):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        if cleaned == "":
            return None
        value = cleaned
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    if numeric < 0:
        return 0.0
    if numeric > 1.0:
        if allow_percent and numeric <= 100.0:
            numeric /= 100.0
        else:
            return min(numeric, 1.0)
    return max(0.0, min(numeric, 1.0))


def _float_or_none(value: Any) -> Optional[float]:
    """Convert value to float unless it's empty/invalid, mirroring Supabase sanitize logic."""
    if value in (None, "", float("inf"), float("-inf")):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        if cleaned == "":
            return None
        value = cleaned
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _cpa(row: Dict[str, Any]) -> float:
    spend = safe_f(row.get("spend"))
    purch, _ = _purchase_and_atc_counts(row)
    return (spend / purch) if purch > 0 else float('inf')


def _cpm(row: Dict[str, Any]) -> float:
    spend = safe_f(row.get("spend"))
    imps = safe_f(row.get("impressions"))
    return (spend / imps * 1000) if imps > 0 else 0.0


def _link_clicks(row: Dict[str, Any]) -> float:
    """Prefer Meta inline/link clicks over generic click totals."""
    if isinstance(row, dict):
        if row.get("inline_link_clicks") is not None:
            return safe_f(row.get("inline_link_clicks"))
        if row.get("link_clicks") is not None:
            return safe_f(row.get("link_clicks"))
    return safe_f(row.get("clicks"))


def _ctr(row: Dict[str, Any]) -> float:
    imps = safe_f(row.get("impressions"))
    clicks = _link_clicks(row)
    return (clicks / imps) if imps > 0 else 0.0


def _meets_minimums(row: Dict[str, Any], min_impressions: int, min_clicks: int, min_spend: float) -> bool:
    return (
        safe_f(row.get("spend")) >= min_spend
        and safe_f(row.get("impressions")) >= min_impressions
        and _link_clicks(row) >= min_clicks
    )


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _compute_hours_live(created_at: Optional[datetime], now: Optional[datetime] = None) -> float:
    if not created_at:
        return 0.0
    reference = now or datetime.now(timezone.utc)
    if created_at > reference:
        return 0.0
    return max(0.0, (reference - created_at).total_seconds() / 3600.0)


def _should_keep_creative(keep_rules: List[Dict[str, Any]], ctr: float, cpm: float) -> bool:
    for rule in keep_rules or []:
        rule_type = rule.get("type")
        if rule_type == "ctr_gte":
            if ctr >= rule.get("ctr_gte", 0):
                return True
        elif rule_type == "ctr_cpm_combo":
            if ctr >= rule.get("ctr_gte", 0) and cpm <= rule.get("cpm_lte", float("inf")):
                return True
    return False


def _evaluate_lock_rules(
    lock_rules: List[Dict[str, Any]],
    ctr: float,
    cpm: float,
    now: datetime,
    existing_lock_info: Optional[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any], bool]:
    lock_info = dict(existing_lock_info or {})
    changed = False

    locked_until = _parse_iso_datetime(lock_info.get("locked_until") if isinstance(lock_info, dict) else None)
    candidate_since = _parse_iso_datetime(lock_info.get("high_ctr_met_since") if isinstance(lock_info, dict) else None)

    is_locked = False
    if locked_until and now < locked_until:
        is_locked = True
    elif locked_until and now >= locked_until:
        lock_info.pop("locked_until", None)
        locked_until = None
        changed = True

    condition_met = False
    for rule in lock_rules or []:
        if rule.get("type") != "ctr_cpm_lock":
            continue
        if ctr >= rule.get("ctr_gte", 0) and cpm <= rule.get("cpm_lte", float("inf")):
            condition_met = True
            evaluation_hours = float(rule.get("evaluation_hours", 48))
            lock_hours = float(rule.get("lock_hours", 72))
            if not candidate_since:
                candidate_since = now
                lock_info["high_ctr_met_since"] = now.isoformat()
                changed = True
            elif now - candidate_since >= timedelta(hours=evaluation_hours):
                locked_until = now + timedelta(hours=lock_hours)
                lock_info["locked_until"] = locked_until.isoformat()
                lock_info.pop("high_ctr_met_since", None)
                changed = True
                is_locked = True
            break

    if not condition_met and not is_locked and candidate_since:
        lock_info.pop("high_ctr_met_since", None)
        candidate_since = None
        changed = True

    return is_locked, lock_info, changed


def _has_minimum_delivery(
    ad_id: str,
    ad: Dict[str, Any],
    insight: Dict[str, Any],
    minimums: Dict[str, Any],
    creation_lookup: Dict[str, datetime],
    now: datetime,
) -> Tuple[bool, float, float, float, float]:
    min_spend = float(minimums.get("min_spend_eur", minimums.get("min_spend", 0)) or 0)
    min_impressions = int(minimums.get("min_impressions", 0) or 0)
    min_link_clicks = int(minimums.get("min_link_clicks", minimums.get("min_clicks", 0) or 0) or 0)
    min_hours = float(minimums.get("min_delivery_hours", 0) or 0)

    spend = safe_f(insight.get("spend"))
    impressions = safe_f(insight.get("impressions"))
    link_clicks = _link_clicks(insight)

    created_at = creation_lookup.get(ad_id)
    if not created_at:
        fallback = (
            _parse_created_time(ad.get("created_time"))
            or _parse_created_time(insight.get("created_time"))
            or _parse_created_time(insight.get("date_start"))
        )
        if fallback:
            creation_lookup[ad_id] = fallback
            created_at = fallback

    hours_live = _compute_hours_live(created_at, now)

    if spend < min_spend or impressions < min_impressions or link_clicks < min_link_clicks:
        return False, spend, impressions, link_clicks, hours_live
    if min_hours and hours_live < min_hours:
        return False, spend, impressions, link_clicks, hours_live
    return True, spend, impressions, link_clicks, hours_live


def _load_lock_metadata(
    supabase_client: Any,
    ad_ids: List[str],
    stage: str = "asc_plus",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    metadata_map: Dict[str, Any] = {}
    lock_state: Dict[str, Dict[str, Any]] = {}

    if not supabase_client or not ad_ids:
        return metadata_map, lock_state

    try:
        response = (
            supabase_client.table("ad_lifecycle")
            .select("ad_id, metadata")
            .in_("ad_id", ad_ids)
            .eq("stage", stage)
            .execute()
        )
        for row in getattr(response, "data", []) or []:
            ad_id = str(row.get("ad_id") or "")
            if not ad_id:
                continue
            metadata = row.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata_map[ad_id] = metadata
            lock_info = metadata.get("lock")
            if isinstance(lock_info, dict):
                lock_state[ad_id] = lock_info
    except Exception as exc:
        logger.debug(f"Failed to load lock metadata: {exc}")

    return metadata_map, lock_state


def _active_count(ads_list: List[Dict[str, Any]]) -> int:
    return sum(1 for a in ads_list if str(a.get("status", "")).upper() == "ACTIVE")


def _safe_float(value: Any, default: float = 0.0, precision: int = 4) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return default
        return round(number, precision)
    except (TypeError, ValueError):
        return default


def _clamp(value: Optional[float], minimum: float, maximum: float) -> Optional[float]:
    """Clamp numeric values to a safe range, returning None when value is invalid."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return max(minimum, min(maximum, number))


def _parse_created_time(value: Any) -> Optional[datetime]:
    """Convert Meta created_time values to timezone-aware datetime."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        elif text.endswith("+0000") or text.endswith("-0000"):
            text = text[:-5] + text[-5:-2] + ":" + text[-2:]
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            try:
                return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")
            except ValueError:
                return None
    return None


def _normalize_lifecycle_status(raw_status: Optional[str]) -> str:
    """Map Meta statuses to lifecycle table allowed values."""
    status_map = {
        "ACTIVE": "active",
        "PAUSED": "paused",
        "ARCHIVED": "completed",
        "DELETED": "completed",
        "WITH_ISSUES": "failed",
        "DISAPPROVED": "failed",
        "PENDING_REVIEW": "active",
        "IN_PROCESS": "active",
    }
    if not raw_status:
        return "active"
    key = str(raw_status).strip().upper()
    return status_map.get(key, "active")


def _parse_metadata(metadata: Any) -> Dict[str, Any]:
    """Ensure metadata is a dictionary."""
    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(metadata, str):
        try:
            loaded = json.loads(metadata)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return {"raw": metadata}
    return {}


def _build_stage_performance(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract key performance signals for lifecycle logging."""
    if not metrics:
        return None
    try:
        spend = safe_f(metrics.get("spend"))
        impressions = safe_f(metrics.get("impressions"))
        clicks = safe_f(metrics.get("clicks"))
        purchases = safe_f(metrics.get("purchases"))
        if spend == 0 and impressions == 0 and clicks == 0 and purchases == 0:
            return None
        return {
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "purchases": purchases,
            "ctr": safe_f(metrics.get("ctr")),
            "cpa": safe_f(metrics.get("cpa")) if metrics.get("cpa") not in (None, float("inf")) else None,
            "roas": safe_f(metrics.get("roas")),
            "cpm": safe_f(metrics.get("cpm")),
            "add_to_cart": safe_f(metrics.get("add_to_cart")),
            "frequency": safe_f(metrics.get("frequency")),
        }
    except Exception:
        return None


def _determine_transition_reason(status: str, stage_performance: Optional[Dict[str, Any]]) -> str:
    """Provide a human-readable reason stored alongside lifecycle data."""
    if status == "paused":
        return "paused_for_guardrail_review"
    if status == "failed":
        return "failed_policy_or_delivery"
    if not stage_performance:
        return "collecting_initial_performance"
    roas = safe_f(stage_performance.get("roas"))
    cpa = stage_performance.get("cpa")
    if roas >= 1.0 and (cpa is None or cpa <= 30):
        return "performing_above_threshold"
    if roas == 0 and safe_f(stage_performance.get("spend")) > 0:
        return "spend_without_results"
    return "monitoring_performance"


def _calculate_creative_performance(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Derive averaged performance metrics for a creative from current insights."""
    if not metrics:
        return {"avg_ctr": 0.0, "avg_cpa": 0.0, "avg_roas": 0.0}

    impressions = safe_f(metrics.get("impressions"))
    clicks = safe_f(metrics.get("clicks"))
    spend = safe_f(metrics.get("spend"))
    purchases = safe_f(metrics.get("purchases"))
    roas = safe_f(metrics.get("roas"))

    avg_ctr = _safe_float(clicks / impressions) if impressions > 0 else _safe_float(metrics.get("ctr"))
    avg_cpa = _safe_float(spend / purchases) if purchases > 0 else _safe_float(metrics.get("cpa"))
    avg_roas = roas if roas > 0 else (_safe_float(metrics.get("revenue", 0)) / spend if spend > 0 else 0.0)

    return {"avg_ctr": avg_ctr, "avg_cpa": avg_cpa, "avg_roas": _safe_float(avg_roas)}


def _calculate_performance_score(perf: Dict[str, float]) -> float:
    """Heuristic performance score between 0-1."""
    ctr_fraction = perf.get("avg_ctr") or 0.0
    ctr_pct = ctr_fraction * 100.0
    cpa = perf.get("avg_cpa") or 0.0
    roas = perf.get("avg_roas") or 0.0

    score = 0.0
    if ctr_pct >= 1.0:
        score += 0.3
    else:
        score += max(0.0, min(ctr_pct / 5.0, 0.3))
    if roas >= 1.0:
        score += min(roas / 10.0, 0.3)
    if cpa > 0:
        score += max(0.0, min((50 - cpa) / 50, 0.2))

    return _safe_float(max(0.0, min(score, 1.0)))


def _calculate_fatigue_index(perf: Dict[str, float]) -> float:
    """Simple fatigue indicator (higher = more fatigued)."""
    ctr_pct = (perf.get("avg_ctr") or 0.0) * 100.0
    cpa = perf.get("avg_cpa") or 0.0
    roas = perf.get("avg_roas") or 0.0

    fatigue = 0.0
    if ctr_pct < 1.0:
        fatigue += 0.3
    if cpa > 30:
        fatigue += 0.3
    if roas < 1.0:
        fatigue += 0.4
    return _safe_float(min(fatigue, 1.0))


def _sync_ad_creation_records(client: MetaClient, ads: List[Dict[str, Any]], stage: str = "asc_plus") -> Dict[str, datetime]:
    """Ensure every ad has an ad_creation_times record and return creation timestamps."""
    creation_map: Dict[str, datetime] = {}
    if not ads:
        return creation_map
    try:
        from infrastructure.supabase_storage import get_validated_supabase_client, SupabaseStorage
    except ImportError:
        logger.debug("Supabase storage unavailable; skipping ad creation sync")
        return creation_map

    supabase_client = get_validated_supabase_client(enable_validation=True)
    if not supabase_client:
        return creation_map

    ad_ids = [str(ad.get("id")) for ad in ads if ad.get("id")]
    if not ad_ids:
        return creation_map

    existing_records: Dict[str, Dict[str, Any]] = {}
    try:
        response = supabase_client.table("ad_creation_times").select(
            "ad_id, created_at_iso, stage, lifecycle_id"
        ).in_("ad_id", ad_ids).execute()
        data = getattr(response, "data", None) or []
        existing_records = {
            str(row.get("ad_id")): row for row in data if row and row.get("ad_id")
        }
        for ad_id, row in existing_records.items():
            created_at = _parse_created_time(row.get("created_at_iso"))
            if created_at:
                creation_map[ad_id] = created_at
    except Exception as exc:
        logger.debug(f"Unable to load existing ad_creation_times records: {exc}")
        existing_records = {}

    storage = SupabaseStorage(supabase_client)

    for ad in ads:
        ad_id = str(ad.get("id") or "")
        if not ad_id:
            continue

        existing = existing_records.get(ad_id)
        lifecycle_id = (
            ad.get("lifecycle_id")
            or (existing.get("lifecycle_id") if existing else "")
            or f"lifecycle_{ad_id}"
        )

        needs_sync = False
        if not existing:
            needs_sync = True
        else:
            if not existing.get("created_at_iso"):
                needs_sync = True
            if not existing.get("lifecycle_id"):
                needs_sync = True
            if existing.get("stage") != stage:
                needs_sync = True

        created_at = _parse_created_time(ad.get("created_time")) if needs_sync else _parse_created_time(existing.get("created_at_iso")) if existing else None

        if needs_sync:
            if not created_at and existing and existing.get("created_at_iso"):
                created_at = _parse_created_time(existing.get("created_at_iso"))

            if not created_at:
                try:
                    details = client._graph_get_object(f"{ad_id}", params={"fields": "created_time"})
                    if isinstance(details, dict):
                        created_at = _parse_created_time(details.get("created_time"))
                except Exception as exc:
                    logger.debug(f"Failed to fetch created_time for ad {ad_id}: {exc}")

            try:
                storage.record_ad_creation(
                    ad_id=ad_id,
                    lifecycle_id=lifecycle_id,
                    stage=stage,
                    created_at=created_at,
                )
            except Exception as exc:
                logger.debug(f"Failed to upsert ad creation record for {ad_id}: {exc}")

        if ad_id not in creation_map:
            try:
                stored = storage.get_ad_creation_time(ad_id)
            except Exception:
                stored = None
            if stored:
                creation_map[ad_id] = stored
            elif created_at:
                creation_map[ad_id] = created_at

    return creation_map


def _sync_ad_lifecycle_records(
    client: MetaClient,
    ads: List[Dict[str, Any]],
    metrics_map: Dict[str, Dict[str, Any]],
    stage: str = "asc_plus",
    campaign_id: Optional[str] = None,
    adset_id: Optional[str] = None,
) -> None:
    """Ensure lifecycle table captures complete state for each ad every tick."""
    if not ads:
        return
    try:
        from infrastructure.supabase_storage import get_validated_supabase_client, SupabaseStorage
    except ImportError:
        logger.debug("Supabase storage unavailable; skipping lifecycle sync")
        return

    supabase_client = get_validated_supabase_client(enable_validation=True)
    if not supabase_client:
        return

    ad_ids = [str(ad.get("id")) for ad in ads if ad.get("id")]
    if not ad_ids:
        return

    existing_records: Dict[str, Dict[str, Any]] = {}
    try:
        response = (
            supabase_client.table("ad_lifecycle")
            .select(
                "id, ad_id, creative_id, campaign_id, adset_id, stage, status, lifecycle_id, metadata, "
                "stage_duration_hours, previous_stage, stage_performance, transition_reason, created_at"
            )
            .in_("ad_id", ad_ids)
            .eq("stage", stage)
            .execute()
        )
        data = getattr(response, "data", None) or []
        existing_records = {
            str(row.get("ad_id")): row for row in data if row and row.get("ad_id")
        }
    except Exception as exc:
        logger.debug(f"Unable to load existing ad_lifecycle records: {exc}")
        existing_records = {}

    storage = SupabaseStorage(supabase_client)
    now = datetime.now(timezone.utc)

    for ad in ads:
        ad_id = str(ad.get("id") or "")
        if not ad_id:
            continue

        existing = existing_records.get(ad_id) or {}
        lifecycle_id = (
            ad.get("lifecycle_id")
            or existing.get("lifecycle_id")
            or f"lifecycle_{ad_id}"
        )

        creative_id = (
            ad.get("creative_id")
            or existing.get("creative_id")
            or ""
        )
        campaign_value = (
            ad.get("campaign_id")
            or campaign_id
            or existing.get("campaign_id")
            or ""
        )
        adset_value = (
            ad.get("adset_id")
            or adset_id
            or existing.get("adset_id")
            or ""
        )

        raw_status = (
            ad.get("effective_status")
            or ad.get("status")
            or existing.get("status")
        )
        status = _normalize_lifecycle_status(raw_status)

        metadata_existing = _parse_metadata(existing.get("metadata"))
        metadata_from_ad = _parse_metadata(ad.get("metadata"))
        metadata_existing.update(metadata_from_ad)
        metadata_existing.update(
            {
                "source": "asc_plus_tick",
                "synced_at": now.isoformat(),
                "ad_name": ad.get("name"),
                "raw_status": raw_status,
                "effective_status": ad.get("effective_status"),
                "created_time": ad.get("created_time"),
            }
        )

        created_at = (
            _parse_created_time(existing.get("created_at"))
            or _parse_created_time(ad.get("created_time"))
            or storage.get_ad_creation_time(ad_id)
            or now
        )

        stage_duration = max((now - created_at).total_seconds() / 3600.0, 0.0)
        stage_duration = min(stage_duration, float(MAX_STAGE_DURATION_HOURS))
        stage_duration = min(stage_duration, float(DB_NUMERIC_MAX))

        existing_previous = existing.get("previous_stage")
        previous_stage = existing_previous or "created"

        metrics = metrics_map.get(ad_id)
        stage_performance = _build_stage_performance(metrics)

        transition_reason = _determine_transition_reason(status, stage_performance)

        lifecycle_record: Dict[str, Any] = {
            "ad_id": ad_id,
            "creative_id": creative_id,
            "campaign_id": campaign_value,
            "adset_id": adset_value,
            "stage": stage,
            "status": status,
            "lifecycle_id": lifecycle_id,
            "metadata": metadata_existing,
            "stage_duration_hours": round(stage_duration, 2),
            "previous_stage": previous_stage,
            "stage_performance": stage_performance,
            "transition_reason": transition_reason,
            "created_at": created_at.isoformat(),
            "updated_at": now.isoformat(),
        }

        lifecycle_record = validate_all_timestamps(lifecycle_record)

        try:
            supabase_client.table("ad_lifecycle").upsert(
                lifecycle_record,
                on_conflict="ad_id,stage",
            ).execute()
        except Exception as exc:
            logger.debug(f"Failed to upsert lifecycle record for {ad_id}: {exc}")


def _collect_storage_metadata(
    supabase_client: Any,
    creative_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    ids = [cid for cid in set(creative_ids) if cid]
    if not ids:
        return {}
    try:
        response = (
            supabase_client.table("creative_storage")
            .select("creative_id, file_size_bytes, storage_url, metadata")
            .in_("creative_id", ids)
            .execute()
        )
        data = getattr(response, "data", None) or []
        return {row.get("creative_id"): row for row in data if row.get("creative_id")}
    except Exception as exc:
        logger.debug(f"Unable to load creative storage metadata: {exc}")
        return {}


def _sync_creative_intelligence_records(
    client: MetaClient,
    ads: List[Dict[str, Any]],
    metrics_map: Dict[str, Dict[str, Any]],
    stage: str = "asc_plus",
) -> Dict[str, Dict[str, Any]]:
    if not ads:
        return {}

    try:
        from infrastructure.supabase_storage import get_validated_supabase_client
    except ImportError:
        logger.debug("Supabase storage unavailable; skipping creative intelligence sync")
        return {}

    supabase_client = get_validated_supabase_client(enable_validation=True)
    if not supabase_client:
        return {}

    ad_ids = [str(ad.get("id")) for ad in ads if ad.get("id")]
    if not ad_ids:
        return {}

    try:
        existing_resp = (
            supabase_client.table("creative_intelligence")
            .select("*")
            .in_("ad_id", ad_ids)
            .execute()
        )
        existing_rows = getattr(existing_resp, "data", None) or []
    except Exception as exc:
        logger.debug(f"Unable to load creative intelligence records: {exc}")
        existing_rows = []

    existing_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    storage_ids: set[str] = set()
    for row in existing_rows:
        creative_id = str(row.get("creative_id") or "")
        ad_id = str(row.get("ad_id") or "")
        if creative_id and ad_id:
            existing_map[(creative_id, ad_id)] = row
            metadata = _parse_metadata(row.get("metadata"))
            storage_id = metadata.get("storage_creative_id")
            if storage_id:
                storage_ids.add(storage_id)
            storage_ids.add(creative_id)

    # Include creative IDs from ads to fetch storage metadata
    for ad in ads:
        creative_id = ad.get("creative_id")
        if creative_id:
            storage_ids.add(str(creative_id))

    storage_map = _collect_storage_metadata(supabase_client, storage_ids)
    now_iso = datetime.now(timezone.utc).isoformat()
    sanitized_records: List[Dict[str, Any]] = []

    for ad in ads:
        ad_id = str(ad.get("id") or "")
        creative_id = str(ad.get("creative_id") or "")
        if not ad_id or not creative_id:
            continue

        existing = existing_map.get((creative_id, ad_id), {})
        metadata = _parse_metadata(existing.get("metadata"))
        metrics = metrics_map.get(ad_id)
        performance = _calculate_creative_performance(metrics)
        avg_ctr = performance["avg_ctr"] or _safe_float(existing.get("avg_ctr"))
        avg_cpa = performance["avg_cpa"] or _safe_float(existing.get("avg_cpa"))
        avg_roas = performance["avg_roas"] or _safe_float(existing.get("avg_roas"))

        storage_key = metadata.get("storage_creative_id") or creative_id
        storage_info = storage_map.get(storage_key)
        if storage_info:
            file_size_bytes = storage_info.get("file_size_bytes")
            file_size_mb = (
                round(float(file_size_bytes) / (1024 * 1024), 4)
                if isinstance(file_size_bytes, (int, float))
                else None
            )
            if file_size_mb is not None:
                metadata.setdefault("storage_metadata", {})
                metadata["storage_metadata"]["file_size_bytes"] = file_size_bytes
            storage_url = storage_info.get("storage_url")
        else:
            file_size_mb = None
            storage_url = existing.get("supabase_storage_url")

        metadata["source"] = "asc_plus_sync"
        metadata.setdefault("ad_id", ad_id)
        metadata.setdefault("creative_id", creative_id)
        if "ad_name" not in metadata and ad.get("name"):
            metadata["ad_name"] = ad.get("name")

        created_at = existing.get("created_at") or now_iso
        lifecycle_id = f"lifecycle_{ad_id}"
        avg_ctr = _clamp(avg_ctr, -9.9999, 9.9999) or 0.0
        avg_cpa = _clamp(avg_cpa, -9.9999, 9.9999) or 0.0
        avg_roas = _clamp(avg_roas, -9.9999, 9.9999) or 0.0

        performance_score = _clamp(
            _calculate_performance_score(
            {"avg_ctr": avg_ctr, "avg_cpa": avg_cpa, "avg_roas": avg_roas}
            ),
            0.0,
            1.0,
        ) or 0.0
        fatigue_index = _clamp(
            _calculate_fatigue_index(
            {"avg_ctr": avg_ctr, "avg_cpa": avg_cpa, "avg_roas": avg_roas}
            ),
            0.0,
            1.0,
        ) or 0.0

        description = existing.get("description") or metadata.get("ad_copy", {}).get("description") or ""
        headline = existing.get("headline") or metadata.get("ad_copy", {}).get("headline") or ad.get("name") or ""
        primary_text = existing.get("primary_text") or metadata.get("ad_copy", {}).get("primary_text") or ""
        similarity_vector = existing.get("similarity_vector")

        if isinstance(similarity_vector, str):
            try:
                similarity_vector = json.loads(similarity_vector)
            except json.JSONDecodeError:
                similarity_vector = None

        if isinstance(similarity_vector, list) and similarity_vector and len(similarity_vector) != 384:
            logger.warning(
                "Discarding similarity vector for creative %s (%s): expected 384 dimensions, got %s",
                creative_id,
                ad_id,
                len(similarity_vector),
            )
            similarity_vector = None

        try:
            performance_rank = int(existing.get("performance_rank", 1))
        except (TypeError, ValueError):
            performance_rank = 1
        if performance_rank < 1:
            performance_rank = 1

        record: Dict[str, Any] = {
            "creative_id": creative_id,
            "ad_id": ad_id,
            "creative_type": existing.get("creative_type") or "image",
            "aspect_ratio": "1:1",
            "file_size_mb": file_size_mb if file_size_mb is not None else _safe_float(existing.get("file_size_mb")),
            "resolution": existing.get("resolution") or "1080x1080",
            "color_palette": existing.get("color_palette") or metadata.get("color_palette") or "[]",
            "text_overlay": existing.get("text_overlay") if existing.get("text_overlay") is not None else True,
            "avg_ctr": avg_ctr,
            "avg_cpa": avg_cpa,
            "avg_roas": avg_roas,
            "performance_rank": performance_rank,
            "performance_score": performance_score,
            "fatigue_index": fatigue_index,
            "similarity_vector": similarity_vector,
            "description": description,
            "headline": headline,
            "primary_text": primary_text,
            "lifecycle_id": lifecycle_id,
            "stage": stage,
            "metadata": metadata,
            "supabase_storage_url": storage_url,
            "image_prompt": existing.get("image_prompt"),
            "text_overlay_content": existing.get("text_overlay_content"),
            "created_at": created_at,
            "updated_at": now_iso,
        }

        record = validate_all_timestamps(record)

        validation = validate_supabase_data("creative_intelligence", record, strict_mode=True)
        if not validation.is_valid:
            logger.warning(
                "Skipping creative_intelligence sync for creative %s (%s): %s",
                creative_id,
                ad_id,
                "; ".join(validation.errors),
            )
            continue

        sanitized_records.append(validation.sanitized_data)

    if not sanitized_records:
        return storage_map

    try:
        supabase_client.upsert(
            "creative_intelligence",
            sanitized_records,
            on_conflict="creative_id,ad_id",
        )
    except ValidationError as exc:
        logger.error("Creative intelligence upsert blocked: %s", exc)
    except Exception as exc:
        logger.debug(f"Failed to upsert creative intelligence records: {exc}")

    return storage_map

def _sync_creative_storage_records(
    storage_map: Dict[str, Dict[str, Any]],
    ads: List[Dict[str, Any]],
    stage: str,
) -> None:
    if not storage_map or not ads:
        return

    try:
        from infrastructure.supabase_storage import get_validated_supabase_client
    except ImportError:
        logger.debug("Supabase storage unavailable; skipping creative storage sync")
        return

    supabase_client = get_validated_supabase_client(enable_validation=True)
    if not supabase_client:
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    updates: List[Dict[str, Any]] = []

    allowed_statuses = {"ACTIVE", "ELIGIBLE"}
    archived_statuses = {"ARCHIVED"}

    for ad in ads:
        ad_id = str(ad.get("id") or "")
        if not ad_id:
            continue

        creative = ad.get("creative") or {}
        creative_id = ad.get("creative_id") or creative.get("id") or creative.get("creative_id")
        if not creative_id:
            continue
        creative_id = str(creative_id)

        status_value = str(ad.get("effective_status") or ad.get("status", "")).upper()
        if status_value in allowed_statuses:
            storage_status = "active"
        elif status_value in archived_statuses:
            storage_status = "archived"
        else:
            storage_status = "paused"

        storage_row = storage_map.get(creative_id)
        if not storage_row:
            continue

        metadata = _parse_metadata(storage_row.get("metadata"))
        metadata["stage"] = stage
        metadata["ad_id"] = ad_id
        metadata["last_synced"] = now_iso

        usage_count = storage_row.get("usage_count")
        try:
            usage_count = int(usage_count) if usage_count is not None else 0
        except (TypeError, ValueError):
            usage_count = 0
        usage_count = max(usage_count, 0) + 1

        update_record = {
            "creative_id": creative_id,
            "ad_id": ad_id,
            "status": storage_status,
            "usage_count": usage_count,
            "last_used_at": now_iso,
            "metadata": metadata,
            "updated_at": now_iso,
        }

        updates.append(update_record)

    if not updates:
        return

    try:
        supabase_client.table("creative_storage").upsert(
            updates,
            on_conflict="creative_id",
        ).execute()
    except Exception as exc:
        logger.debug(f"Failed to upsert creative storage records: {exc}")


def _sync_performance_metrics_records(
    metrics_map: Dict[str, Dict[str, Any]],
    stage: str,
    date_label: str,
) -> None:
    if not metrics_map:
        return

    try:
        from infrastructure.supabase_storage import get_validated_supabase_client, SupabaseStorage
    except ImportError:
        logger.debug("Supabase storage unavailable; skipping performance metrics sync")
        return

    supabase_client = get_validated_supabase_client(enable_validation=True)
    if not supabase_client:
        return

    storage = SupabaseStorage(supabase_client)
    now = datetime.now(timezone.utc)

    upserts: List[Dict[str, Any]] = []

    for ad_id, metrics in metrics_map.items():
        if not ad_id:
            continue

        lifecycle_id = metrics.get("lifecycle_id") or f"lifecycle_{ad_id}"
        spend = safe_f(metrics.get("spend"))
        impressions = int(safe_f(metrics.get("impressions")))
        clicks = int(safe_f(metrics.get("clicks")))
        purchases = int(safe_f(metrics.get("purchases")))
        add_to_cart = int(safe_f(metrics.get("add_to_cart")))
        initiate_checkout = int(safe_f(metrics.get("initiate_checkout") or metrics.get("ic")))
        revenue = safe_f(metrics.get("revenue"))

        ctr = _fraction_or_none(metrics.get("ctr"))
        cpc = _float_or_none(metrics.get("cpc"))
        cpm = _float_or_none(metrics.get("cpm"))
        roas = _float_or_none(metrics.get("roas"))
        cpa = _float_or_none(metrics.get("cpa"))
        frequency = _float_or_none(metrics.get("frequency"))
        dwell_time = _float_or_none(metrics.get("dwell_time"))

        atc_rate = _fraction_or_none(add_to_cart / impressions, allow_percent=False) if impressions > 0 else None
        ic_rate = _fraction_or_none(initiate_checkout / impressions, allow_percent=False) if impressions > 0 else None
        purchase_rate = _fraction_or_none(purchases / impressions, allow_percent=False) if impressions > 0 else None
        atc_to_ic_rate = _fraction_or_none(initiate_checkout / add_to_cart, allow_percent=False) if add_to_cart > 0 else None
        ic_to_purchase_rate = _fraction_or_none(purchases / initiate_checkout, allow_percent=False) if initiate_checkout > 0 else None

        try:
            creation_time = storage.get_ad_creation_time(ad_id)
        except Exception:
            creation_time = None

        age_days = None
        stage_duration_hours = None
        if creation_time:
            delta = now - creation_time
            if delta.total_seconds() >= 0:
                age_days = delta.total_seconds() / 86400.0
                stage_duration_hours = delta.total_seconds() / 3600.0
        if stage_duration_hours is not None:
            stage_duration_hours = min(stage_duration_hours, float(DB_NUMERIC_MAX))

        record = {
            "ad_id": ad_id,
            "lifecycle_id": lifecycle_id,
            "stage": stage,
            "window_type": "1d",
            "date_start": date_label,
            "date_end": date_label,
            "impressions": impressions,
            "clicks": clicks,
            "spend": spend,
            "purchases": purchases,
            "add_to_cart": add_to_cart,
            "initiate_checkout": initiate_checkout,
            "ctr": ctr,
            "cpc": cpc,
            "cpm": cpm,
            "roas": roas,
            "cpa": cpa,
            "dwell_time": dwell_time,
            "frequency": frequency,
            "atc_rate": atc_rate,
            "ic_rate": ic_rate,
            "purchase_rate": purchase_rate,
            "atc_to_ic_rate": atc_to_ic_rate,
            "ic_to_purchase_rate": ic_to_purchase_rate,
            "performance_quality_score": None,
            "fatigue_index": None,
            "stability_score": None,
            "momentum_score": None,
            "age_days": age_days,
            "stage_duration_hours": stage_duration_hours,
            "revenue": revenue,
        }

        upserts.append(record)

    if not upserts:
        return

    try:
        supabase_client.table("performance_metrics").upsert(
            upserts,
            on_conflict="ad_id,stage,window_type,date_start",
        ).execute()
    except ValidationError as exc:
        logger.error("Performance metrics upsert blocked: %s", exc)
    except Exception as exc:
        logger.debug(f"Failed to upsert performance metrics records: {exc}")


def _sync_historical_data_records(
    metrics_map: Dict[str, Dict[str, Any]],
    stage: str,
) -> None:
    if not metrics_map:
        return

    try:
        from infrastructure.supabase_storage import get_validated_supabase_client, SupabaseStorage
    except ImportError:
        logger.debug("Supabase storage unavailable; skipping historical data sync")
        return

    supabase_client = get_validated_supabase_client(enable_validation=True)
    if not supabase_client:
        return

    storage = SupabaseStorage(supabase_client)

    for ad_id, metrics in metrics_map.items():
        if not ad_id:
            continue

        lifecycle_id = metrics.get("lifecycle_id") or f"lifecycle_{ad_id}"
        spend = safe_f(metrics.get("spend"))
        impressions = safe_f(metrics.get("impressions"))
        clicks = safe_f(metrics.get("clicks"))
        purchases = safe_f(metrics.get("purchases"))
        add_to_cart = safe_f(metrics.get("add_to_cart"))
        roas = safe_f(metrics.get("roas"))
        ctr = safe_f(metrics.get("ctr"))
        cpa_value = metrics.get("cpa")
        cpa = (
            safe_f(cpa_value)
            if cpa_value not in (None, "", float("inf"), float("-inf"))
            else None
        )

        try:
            storage.store_historical_data(ad_id, lifecycle_id, stage, "spend", spend)
            storage.store_historical_data(ad_id, lifecycle_id, stage, "impressions", impressions)
            storage.store_historical_data(ad_id, lifecycle_id, stage, "clicks", clicks)
            storage.store_historical_data(ad_id, lifecycle_id, stage, "purchases", purchases)
            storage.store_historical_data(ad_id, lifecycle_id, stage, "add_to_cart", add_to_cart)
        except Exception as exc:
            logger.debug(f"Failed to store base historical data for {ad_id}: {exc}")

        try:
            if impressions > 0:
                derived_ctr = ctr if ctr is not None else (clicks / impressions)
                storage.store_historical_data(ad_id, lifecycle_id, stage, "ctr", derived_ctr)
        except Exception as exc:
            logger.debug(f"Failed to store CTR historical data for {ad_id}: {exc}")

        try:
            if spend > 0 and roas > 0:
                storage.store_historical_data(ad_id, lifecycle_id, stage, "roas", roas)
        except Exception as exc:
            logger.debug(f"Failed to store ROAS historical data for {ad_id}: {exc}")

        try:
            if purchases > 0 and spend > 0:
                derived_cpa = cpa if cpa is not None else (spend / purchases)
                storage.store_historical_data(ad_id, lifecycle_id, stage, "cpa", derived_cpa)
        except Exception as exc:
            logger.debug(f"Failed to store CPA historical data for {ad_id}: {exc}")


def _sync_creative_performance_records(
    ads: List[Dict[str, Any]],
    metrics_map: Dict[str, Dict[str, Any]],
    stage: str,
    date_label: str,
) -> None:
    global _CREATIVE_PERFORMANCE_STAGE_DISABLED

    if not ads or not metrics_map:
        return

    try:
        from infrastructure.supabase_storage import get_validated_supabase_client
    except ImportError:
        logger.debug("Supabase storage unavailable; skipping creative performance sync")
        return

    supabase_client = get_validated_supabase_client(enable_validation=True)
    if not supabase_client:
        return

    if _CREATIVE_PERFORMANCE_STAGE_DISABLED:
        return

    creative_lookup: Dict[str, str] = {}
    for ad in ads:
        ad_id = str(ad.get("id") or "")
        if not ad_id:
            continue
        creative = ad.get("creative") or {}
        creative_id = ad.get("creative_id") or creative.get("id") or creative.get("creative_id")
        if creative_id:
            creative_lookup[ad_id] = str(creative_id)

    upsert_records: List[Dict[str, Any]] = []
    for ad_id, metrics in metrics_map.items():
        creative_id = creative_lookup.get(ad_id)
        if not creative_id:
            continue

        impressions = safe_f(metrics.get("impressions"))
        clicks = safe_f(metrics.get("clicks"))
        spend = safe_f(metrics.get("spend"))
        purchases = safe_f(metrics.get("purchases"))
        add_to_cart = safe_f(metrics.get("add_to_cart"))
        initiate_checkout = safe_f(metrics.get("initiate_checkout") or metrics.get("ic"))
        ctr = _safe_float(metrics.get("ctr"))
        if impressions > 0 and clicks > 0:
            ctr = _safe_float(clicks / impressions)
        elif impressions <= 0:
            ctr = None

        cpc = _safe_float(metrics.get("cpc"), precision=2) if metrics.get("cpc") is not None else (
            _safe_float(spend / clicks, precision=2) if clicks > 0 else None
        )
        cpm = _safe_float(metrics.get("cpm"), precision=2) if metrics.get("cpm") is not None else (
            _safe_float(spend * 1000.0 / impressions, precision=2) if impressions > 0 else None
        )
        roas = _safe_float(metrics.get("roas")) if metrics.get("roas") is not None else None
        if spend > 0 and metrics.get("revenue"):
            roas = _safe_float(metrics.get("revenue") / spend)
        cpa = _safe_float(metrics.get("cpa")) if metrics.get("cpa") is not None else (
            _safe_float(spend / purchases) if purchases > 0 else None
        )

        engagement_rate = _safe_float(clicks / impressions) if impressions > 0 else None
        conversion_rate = _safe_float(purchases / clicks) if clicks > 0 else None
        conversions = int(purchases)
        lifecycle_id = f"lifecycle_{ad_id}"

        record = {
            "creative_id": creative_id,
            "ad_id": ad_id,
            "stage": CREATIVE_PERFORMANCE_STAGE_VALUE,
            "date_start": date_label,
            "date_end": date_label,
            "impressions": int(impressions) if impressions >= 0 else 0,
            "clicks": int(clicks) if clicks >= 0 else 0,
            "spend": _safe_float(spend, precision=2),
            "purchases": int(purchases) if purchases >= 0 else 0,
            "add_to_cart": int(add_to_cart) if add_to_cart >= 0 else 0,
            "initiate_checkout": int(initiate_checkout) if initiate_checkout >= 0 else 0,
            "ctr": ctr,
            "cpc": cpc,
            "cpm": cpm,
            "roas": roas,
            "cpa": cpa,
            "engagement_rate": engagement_rate,
            "conversion_rate": conversion_rate,
            "conversions": conversions if conversions >= 0 else 0,
            "lifecycle_id": lifecycle_id,
            "performance_score": None,
        }

        try:
            sanitized_record = validate_and_sanitize_data("creative_performance", record)
        except ValidationError as val_exc:
            logger.error(
                "Creative performance record failed validation for %s/%s/%s: %s",
                creative_id,
                ad_id,
                date_label,
                val_exc,
            )
            continue

        upsert_records.append(sanitized_record)

    if not upsert_records:
        return

    try:
        supabase_client.upsert(
            "creative_performance",
            upsert_records,
            on_conflict="creative_id,ad_id,date_start",
        )
    except Exception as exc:
        error_str = str(exc)
        if "creative_performance_stage_check" in error_str:
            if not _CREATIVE_PERFORMANCE_STAGE_DISABLED:
                logger.warning(
                    "Supabase creative_performance stage check rejected records. "
                    "Skipping future creative performance syncs until the constraint is fixed in Supabase. "
                    "Error: %s",
                    error_str,
                )
            _CREATIVE_PERFORMANCE_STAGE_DISABLED = True
            return
        if "42P10" in error_str:
            logger.debug(
                "creative_performance table missing composite constraint; falling back to delete+insert"
            )
            base_client = (
                supabase_client.client
                if hasattr(supabase_client, "client")
                else supabase_client
            )
            for record in upsert_records:
                try:
                    base_client.table("creative_performance").delete().match(
                        {
                            "creative_id": record["creative_id"],
                            "ad_id": record["ad_id"],
                            "date_start": record["date_start"],
                        }
                    ).execute()
                except Exception as delete_exc:
                    logger.debug(
                        "Failed to delete existing creative_performance row for %s/%s/%s: %s",
                        record["creative_id"],
                        record["ad_id"],
                        record["date_start"],
                        delete_exc,
                    )

            try:
                if hasattr(supabase_client, "insert"):
                    supabase_client.insert("creative_performance", upsert_records)
                else:
                    base_client.table("creative_performance").insert(upsert_records).execute()
            except Exception as insert_exc:
                insert_str = str(insert_exc)
                if "creative_performance_stage_check" in insert_str:
                    if not _CREATIVE_PERFORMANCE_STAGE_DISABLED:
                        logger.warning(
                            "Supabase creative_performance stage check rejected records during fallback insert. "
                            "Skipping future creative performance syncs until the constraint is fixed in Supabase. "
                            "Error: %s",
                            insert_str,
                        )
                    _CREATIVE_PERFORMANCE_STAGE_DISABLED = True
                    return
                logger.error(
                    "Fallback insert failed for creative_performance records: %s",
                    insert_exc,
                )
            else:
                _CREATIVE_PERFORMANCE_STAGE_DISABLED = False
        else:
            logger.debug(f"Failed to upsert creative performance records: {exc}")
    else:
        _CREATIVE_PERFORMANCE_STAGE_DISABLED = False


def _guardrail_kill(metrics: Dict[str, Any]) -> Tuple[bool, str]:
    impressions = safe_f(metrics.get("impressions"))
    clicks = safe_f(metrics.get("clicks"))
    spend = safe_f(metrics.get("spend"))
    ctr_fraction = safe_f(metrics.get("ctr"))
    ctr_pct = ctr_fraction * 100.0
    atc = safe_f(metrics.get("add_to_cart"))
    purchases = safe_f(metrics.get("purchases"))
    roas = safe_f(metrics.get("roas"))
    cpm = safe_f(metrics.get("cpm"))
    stage_hours = safe_f(metrics.get("stage_duration_hours"))

    MIN_AGE_DAYS_FOR_KILL = 3.0
    ad_age_days = safe_f(metrics.get("ad_age_days"))
    derived_age_days = max(ad_age_days, (stage_hours / 24.0) if stage_hours else 0.0)

    if derived_age_days < MIN_AGE_DAYS_FOR_KILL:
        return False, ""

    reasons: List[str] = []

    if spend >= 12 and purchases == 0 and ctr_pct < 0.6:
        reasons.append(f"Spend €{spend:.2f} with CTR {ctr_pct:.2f}% and no conversions")
    if impressions >= 2500 and ctr_pct < 0.45 and clicks < 20:
        reasons.append(f"CTR {ctr_pct:.2f}% after {int(impressions)} impressions")
    if spend >= 8 and roas < 0.5 and purchases == 0:
        reasons.append(f"ROAS {roas:.2f} after spend €{spend:.2f}")
    if spend >= 6 and cpm > 90:
        reasons.append(f"CPM €{cpm:.2f} after spend €{spend:.2f}")
    if spend >= 5 and atc == 0 and purchases == 0 and ctr_pct < 0.7:
        reasons.append("No intent events despite initial spend")

    if reasons:
        return True, reasons[0]
    return False, ""


def _guardrail_promote(metrics: Dict[str, Any]) -> Tuple[bool, str]:
    ctr_fraction = safe_f(metrics.get("ctr"))
    ctr_pct = ctr_fraction * 100.0
    cpc = safe_f(metrics.get("cpc"))
    purchases = safe_f(metrics.get("purchases"))
    roas = safe_f(metrics.get("roas"))

    if purchases >= 1 and roas >= 1.2:
        return True, f"ROAS {roas:.2f} with {int(purchases)} purchases"
    if ctr_pct >= 2.0 and 0 < cpc < 1.20:
        return True, f"High CTR {ctr_pct:.2f}% with CPC €{cpc:.2f}"
    return False, ""


def _select_ads_for_cap(
    active_ads: List[Dict[str, Any]],
    metrics_map: Dict[str, Dict[str, Any]],
    excluded_ids: List[str],
    limit: int,
) -> List[Tuple[float, str, Dict[str, Any]]]:
    excluded = set(excluded_ids)
    candidates: List[Tuple[float, str, Dict[str, Any]]] = []
    for ad in active_ads:
        ad_id = str(ad.get("id") or "")
        if not ad_id or ad_id in excluded:
            continue
        metrics = metrics_map.get(ad_id, {})
        score = _ad_health_score(metrics)
        candidates.append((score, ad_id, metrics))
    candidates.sort(key=lambda x: x[0])  # lowest score first
    return candidates[:limit]


def _build_ad_metrics(
    row: Dict[str, Any],
    stage: str,
    date_label: str,
    creation_time: Optional[datetime] = None,
    now_ts: Optional[datetime] = None,
) -> Dict[str, Any]:
    ad_id = row.get("ad_id")
    spend_val = safe_f(row.get("spend"))

    def _first_non_null(*values: Any) -> Any:
        for value in values:
            if value not in (None, ""):
                return value
        return None

    impressions_source = _first_non_null(row.get("impressions"))
    impressions_val = safe_f(impressions_source)

    clicks_source = _first_non_null(
        row.get("inline_link_clicks"),
        row.get("link_clicks"),
        row.get("clicks"),
    )
    clicks_val = safe_f(clicks_source)

    actions = row.get("actions") or []
    add_to_cart = 0
    initiate_checkout = 0
    purchases = 0
    for action in actions:
        action_type = action.get("action_type")
        value = safe_f(action.get("value"))
        if action_type == "add_to_cart":
            add_to_cart = int(value)
        elif action_type == "initiate_checkout":
            initiate_checkout = int(value)
        elif action_type == "purchase":
            purchases = int(value)

    revenue = 0.0
    for action_value in row.get("action_values") or []:
        if action_value.get("action_type") == "purchase":
            revenue += safe_f(action_value.get("value"))

    purchase_roas_list = row.get("purchase_roas") or []
    roas: Optional[float]
    if purchase_roas_list:
        roas = safe_f(purchase_roas_list[0].get("value"))
    elif spend_val > 0:
        roas = revenue / spend_val if revenue > 0 else 0.0
    else:
        roas = None
    if roas is not None and spend_val <= 0:
        roas = None

    ctr_row = _fraction_or_none(row.get("inline_link_click_ctr")) or _fraction_or_none(
        row.get("ctr")
    )
    ctr = ctr_row if ctr_row is not None else ((clicks_val / impressions_val) if impressions_val > 0 else None)

    raw_cpc = _float_or_none(row.get("cpc"))
    if raw_cpc is None:
        raw_cpc = _float_or_none(row.get("cost_per_inline_link_click"))
    if raw_cpc is None and clicks_val > 0:
        raw_cpc = spend_val / clicks_val if clicks_val > 0 else None
    cpc = round(raw_cpc, 2) if raw_cpc is not None else None

    raw_cpm = _float_or_none(row.get("cpm"))
    if raw_cpm is None and impressions_val > 0:
        raw_cpm = spend_val * 1000.0 / impressions_val
    cpm = round(raw_cpm, 2) if raw_cpm is not None else None

    cpa = (spend_val / purchases) if purchases > 0 else None

    reach_val = safe_f(row.get("reach")) if row.get("reach") is not None else None
    frequency = _float_or_none(row.get("frequency"))
    if frequency is None:
        if reach_val and reach_val > 0:
            frequency = impressions_val / reach_val
        else:
            frequency = None

    now_ts = now_ts or datetime.now(timezone.utc)
    ad_creation = creation_time or _parse_created_time(row.get("created_time"))
    age_seconds: Optional[float] = None
    age_days: Optional[float] = None
    hydrated = True
    if ad_creation:
        delta = now_ts - ad_creation
        if delta.total_seconds() >= 0:
            age_seconds = delta.total_seconds()
            age_days = age_seconds / 86400.0
            hydrated = age_seconds >= HYDRATION_SECONDS
        else:
            hydrated = False
    else:
        raw_age = row.get("ad_age_days")
        try:
            age_days = float(raw_age) if raw_age is not None else None
        except (TypeError, ValueError):
            age_days = None
        if age_days is not None:
            age_seconds = age_days * 86400.0
            hydrated = age_seconds >= HYDRATION_SECONDS

    dwell_time = row.get("dwell_time")
    try:
        dwell_time = float(dwell_time) if dwell_time not in (None, "") else None
    except (TypeError, ValueError):
        dwell_time = None

    metrics = {
        "ad_id": ad_id,
        "lifecycle_id": f"lifecycle_{ad_id}",
        "stage": stage,
        "status": "active",
        "spend": spend_val,
        "impressions": impressions_val,
        "clicks": clicks_val,
        "ctr": ctr,
        "cpc": cpc,
        "cpm": cpm,
        "purchases": purchases,
        "add_to_cart": add_to_cart,
        "atc": add_to_cart,
        "ic": initiate_checkout,
        "initiate_checkout": initiate_checkout,
        "roas": roas,
        "cpa": cpa,
        "revenue": revenue,
        "frequency": frequency,
        "dwell_time": dwell_time,
        "date_start": date_label,
        "date_end": date_label,
        "campaign_name": row.get("campaign_name", ""),
        "campaign_id": row.get("campaign_id"),
        "adset_name": row.get("adset_name", ""),
        "adset_id": row.get("adset_id"),
        "has_recent_activity": bool(spend_val or impressions_val or clicks_val),
        "metadata": {"source": "meta_insights"},
        "hydrated": hydrated,
        "ad_age_days": age_days,
        "age_seconds": age_seconds,
    }

    return metrics


def _summarize_metrics(metrics_map: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    totals = {
        "spend": 0.0,
        "impressions": 0.0,
        "clicks": 0.0,
        "add_to_cart": 0.0,
        "purchases": 0.0,
    }
    for metrics in metrics_map.values():
        if not metrics.get("hydrated", True):
            continue
        totals["spend"] += safe_f(metrics.get("spend"))
        totals["impressions"] += safe_f(metrics.get("impressions"))
        totals["clicks"] += safe_f(metrics.get("clicks"))
        totals["add_to_cart"] += safe_f(metrics.get("add_to_cart"))
        totals["purchases"] += safe_f(metrics.get("purchases"))
    return totals


def _ad_health_score(metrics: Dict[str, Any]) -> float:
    """Compute a rough performance score (higher is better)."""
    roas = safe_f(metrics.get("roas"))
    purchases = safe_f(metrics.get("purchases"))
    ctr = safe_f(metrics.get("ctr"))
    cpm = safe_f(metrics.get("cpm"))
    spend = safe_f(metrics.get("spend"))
    add_to_cart = safe_f(metrics.get("add_to_cart"))

    score = 0.0
    score += roas * 3.0
    score += purchases * 5.0
    score += ctr * 0.4
    score += add_to_cart * 0.25

    score -= cpm / 40.0
    score -= spend / 8.0
    return score


def _short_id(value: Any) -> str:
    text = str(value or "")
    if len(text) <= 8:
        return text
    return f"...{text[-6:]}"


def _evaluate_health(final_active: int, target: int, metrics_map: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    if final_active < target:
        deficit = target - final_active
        return "WARNING", f"Active creatives below minimum. Generating {deficit} replacements."
    if len(metrics_map) < target:
        return "WARNING", "Data below thresholds. Waiting for additional insights."
    if not any(metrics.get("has_recent_activity") for metrics in metrics_map.values()):
        return "WARNING", "Insights contain no recent spend or impressions. Monitoring next tick."
    return "HEALTHY", ""


def _emit_health_notification(
    status: str,
    message: str,
    totals: Dict[str, float],
    active_count: int,
    target_count: int,
) -> str:
    tick_time = datetime.now(LOCAL_TZ).strftime("%H:%M")
    summary_lines = [
        f"ASC+ {tick_time} CET · Health {status}",
        (
            f"Active {active_count}/{target_count} | "
            f"Spend {fmt_eur(totals.get('spend', 0.0))} | "
            f"IMP {fmt_int(totals.get('impressions', 0.0))} | "
            f"Clicks {fmt_int(totals.get('clicks', 0.0))} | "
            f"ATC {fmt_int(totals.get('add_to_cart', 0.0))} | "
            f"PUR {fmt_int(totals.get('purchases', 0.0))}"
        ),
    ]
    if message:
        summary_lines.append(f"Next: {message}")

    summary = "\n".join(summary_lines)
    _asc_log(logging.INFO, "Health summary | %s", " | ".join(summary_lines))
    notify(summary)
    return summary


def _generate_creatives_for_deficit(
    deficit: int,
    client: MetaClient,
    settings: Dict[str, Any],
    campaign_id: str,
    adset_id: str,
    ml_system: Optional[Any],
    base_active_count: int,
) -> List[Dict[str, Any]]:
    if deficit <= 0:
        return []

    created_ads: List[Dict[str, Any]] = []
    supabase_client = None
    storage_manager = None
    try:
        from infrastructure.supabase_storage import get_validated_supabase_client
        from infrastructure.creative_storage import create_creative_storage_manager

        supabase_client = get_validated_supabase_client()
        if supabase_client:
            storage_manager = create_creative_storage_manager(supabase_client)
    except Exception as exc:
        _asc_log(logging.DEBUG, "Creative storage not available: %s", exc)

    image_generator = create_image_generator(
        flux_api_key=os.getenv("FLUX_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        ml_system=ml_system,
    )

    product_config = cfg(settings, "asc_plus.product") or {}
    product_info = {
        "name": product_config.get("name", "Brava Product"),
        "description": product_config.get("description", "Premium skincare"),
        "features": product_config.get("features", ["Premium quality", "Daily routine", "Men's grooming"]),
        "brand_tone": product_config.get("brand_tone", "calm confidence"),
        "target_audience": product_config.get("target_audience", "Men aged 18-54"),
    }

    created_count = 0

    # Use queued creatives first
    while deficit > 0 and storage_manager:
        queued_creative = storage_manager.get_queued_creative()
        if not queued_creative:
            break
        creative_data = {
            "supabase_storage_url": queued_creative.get("storage_url"),
            "storage_creative_id": queued_creative.get("creative_id"),
            "ad_copy": (queued_creative.get("metadata") or {}).get("ad_copy", {}),
            "text_overlay": (queued_creative.get("metadata") or {}).get("text_overlay", ""),
            "image_prompt": (queued_creative.get("metadata") or {}).get("image_prompt", ""),
            "scenario_description": (queued_creative.get("metadata") or {}).get("scenario_description", ""),
        }
        creative_id, ad_id, success = _create_creative_and_ad(
            client=client,
            image_generator=image_generator,
            creative_data=creative_data,
            adset_id=adset_id,
            active_count=base_active_count,
            created_count=created_count,
            existing_creative_ids=set(),
            ml_system=ml_system,
            campaign_id=campaign_id,
        )
        if success and ad_id:
            created_ads.append(
                {
                    "ad_id": ad_id,
                    "creative_id": creative_id,
                    "source": "queued",
                    "storage_creative_id": creative_data.get("storage_creative_id"),
                }
            )
            created_count += 1
            deficit -= 1
            if storage_manager:
                storage_manager.mark_creative_active(creative_data.get("storage_creative_id"), ad_id)
        else:
            break

    # Generate new creatives for remaining deficit
    attempt_index = 0
    while deficit > 0:
        creative_payload = generate_new_creative(image_generator, product_info, attempt_index)
        attempt_index += 1
        if not creative_payload:
            break
        creative_id, ad_id, success = _create_creative_and_ad(
            client=client,
            image_generator=image_generator,
            creative_data=creative_payload,
            adset_id=adset_id,
            active_count=base_active_count,
            created_count=created_count,
            existing_creative_ids=set(),
            ml_system=ml_system,
            campaign_id=campaign_id,
        )
        if success and ad_id:
            created_ads.append(
                {
                    "ad_id": ad_id,
                    "creative_id": creative_id,
                    "source": "generated",
                    "attempt": attempt_index,
                }
            )
            created_count += 1
            deficit -= 1
        else:
            break

    return created_ads


def _pause_ad(client: MetaClient, ad_id: str, reason: Optional[str] = None) -> bool:
    from infrastructure.error_handling import retry_with_backoff

    @retry_with_backoff(max_retries=3)
    def _pause(ad_identifier: str):
        client._graph_post(f"{ad_identifier}", {"status": "PAUSED"})

    try:
        _pause(ad_id)
        if reason:
            _asc_log(logging.INFO, "Paused ad %s per guardrail: %s", ad_id, reason)
        else:
            _asc_log(logging.INFO, "Paused ad %s per guardrail", ad_id)
        return True
    except Exception as exc:
        _asc_log(logging.ERROR, "Failed to pause ad %s: %s", ad_id, exc)
        return False


def _record_lifecycle_event(ad_id: str, status: str, reason: str) -> None:
    try:
        from infrastructure.supabase_storage import get_validated_supabase_client
        client = get_validated_supabase_client()
        if not client:
            return
        lifecycle_status = _normalize_lifecycle_status(status)
        payload = {
            "ad_id": ad_id,
            "stage": "asc_plus",
            "status": lifecycle_status,
            "metadata": {"reason": reason, "raw_status": status},
            "lifecycle_id": f"lifecycle_{ad_id}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        client.upsert("ad_lifecycle", payload, on_conflict="ad_id,stage")
    except Exception as exc:
        _asc_log(logging.DEBUG, "Lifecycle logging failed for %s: %s", ad_id, exc)


def _promote_ad(client: MetaClient, ad_id: str, adset_id: str) -> Optional[str]:
    try:
        ad_details = client._graph_get_object(f"{ad_id}", params={"fields": "name,creative{id}"})
        creative = ad_details.get("creative") or {}
        creative_id = creative.get("id") or creative.get("creative_id")
        if not creative_id:
            _asc_log(logging.WARNING, "Cannot promote ad %s: missing creative id", ad_id)
            return None
        base_name = ad_details.get("name", f"{ad_id}")
        timestamp = datetime.now(LOCAL_TZ).strftime("%H%M")
        new_name = f"{base_name} • promote {timestamp}"
        result = client.promote_ad_with_continuity(ad_id, adset_id, new_name, str(creative_id), status="ACTIVE")
        return result.get("id") if isinstance(result, dict) else None
    except Exception as exc:
        _asc_log(logging.ERROR, "Failed to promote ad %s: %s", ad_id, exc)
        return None


def _create_creative_and_ad(
    client: MetaClient,
    image_generator: ImageCreativeGenerator,
    creative_data: Dict[str, Any],
    adset_id: str,
    active_count: int,
    created_count: int,
    existing_creative_ids: set,
    ml_system: Optional[Any] = None,
    campaign_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Create a creative and ad in Meta from creative data.
    Uses Supabase Storage for image hosting.
    
    Args:
        client: Meta client instance
        image_generator: Image generator instance (for validation)
        creative_data: Dictionary with creative data (image_path, ad_copy, supabase_storage_url, etc.)
        adset_id: Ad set ID to create ad in
        active_count: Current count of active ads
        created_count: Current count of created ads in this batch
        existing_creative_ids: Set of already created creative IDs to prevent duplicates
        ml_system: Optional ML system for tracking
    
    Returns:
        Tuple of (creative_id, ad_id, success)
    """
    # Input validation
    if not client or not adset_id or not creative_data:
        logger.error("Invalid input: client, adset_id, and creative_data are required")
        return None, None, False
    
    # Use storage_creative_id from creative_data if available, otherwise generate from image hash
    storage_creative_id = creative_data.get("storage_creative_id")
    if not storage_creative_id:
        import hashlib
        image_path = creative_data.get("image_path")
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
            storage_creative_id = f"creative_{image_hash[:12]}"
        else:
            import time
            storage_creative_id = f"creative_{int(time.time())}"
    
    try:
        # Create descriptive creative name for Ads Manager
        # Include headline snippet and sequence for easy identification
        ad_copy_dict = creative_data.get("ad_copy") or {}
        if not isinstance(ad_copy_dict, dict):
            ad_copy_dict = {}
        
        headline = ad_copy_dict.get("headline", "")
        if headline:
            # Extract first 2-3 words from headline (max 25 chars)
            headline_words = headline.split()[:3]
            headline_snippet = " ".join(headline_words)
            if len(headline_snippet) > 25:
                headline_snippet = headline_snippet[:22] + "..."
            # Clean for filename safety
            headline_snippet = headline_snippet.replace(":", "").replace("/", "").replace("\\", "").strip()
        else:
            headline_snippet = "Creative"
        
        # Sequence number
        seq_num = active_count + created_count + 1
        
        # Build descriptive creative name: [ASC+] HeadlineSnippet - #N
        creative_name = f"[ASC+] {headline_snippet} - #{seq_num}"
        
        # Ensure it's not too long (keep under 80 chars for creatives)
        if len(creative_name) > 80:
            creative_name = f"[ASC+] {headline_snippet[:15]} - #{seq_num}"
        
        # Use Supabase Storage URL if available, otherwise fallback to image_path
        supabase_storage_url = creative_data.get("supabase_storage_url")
        image_path = creative_data.get("image_path")
        
        # Ensure we have either supabase_storage_url or image_path
        if not supabase_storage_url and not image_path:
            logger.error("Creative data must have either supabase_storage_url or image_path")
            return None, None, False
        
        # Ensure ad_copy is a dict (not None) - already set above, but ensure it's valid
        if not isinstance(ad_copy_dict, dict):
            ad_copy_dict = {}
        
        # Validate page_id
        page_id = os.getenv("FB_PAGE_ID")
        if not page_id:
            logger.error("FB_PAGE_ID environment variable is required")
            return None, None, False
        
        # Resolve Instagram actor ID (env override -> automatic lookup)
        env_instagram_actor_id = os.getenv("IG_ACTOR_ID")
        instagram_actor_id = env_instagram_actor_id or client.get_instagram_actor_id(page_id)
        if not instagram_actor_id:
            logger.info(
                "Instagram actor ID unavailable; creative will not run on Instagram placements unless the page is linked."
            )
        
        logger.info(
            "Creating Meta creative: name='%s', page_id='%s', instagram_actor_id=%s, has_supabase_url=%s, has_image_path=%s",
            creative_name,
            page_id,
            bool(instagram_actor_id),
            bool(supabase_storage_url),
            bool(image_path),
        )
        resolved_instagram_actor_id = instagram_actor_id
        try:
            # Clean primary text - remove "Brava Product" and em dashes
            primary_text = ad_copy_dict.get("primary_text", "")
            if primary_text:
                import re
                primary_text = primary_text.replace("Brava Product", "").replace("—", ",").replace("–", ",").strip()
                primary_text = re.sub(r'\s+', ' ', primary_text).strip()
                # Ensure it's not too long
                if len(primary_text) > 150:
                    primary_text = primary_text[:147] + "..."
            
            # Create single image creative with catalog products
            # Note: For ASC+ campaigns, the catalog is configured at the ad set level
            # Meta will automatically show product cards from the catalog below the single image
            # The creative is a single image with primary text, headline, and "Shop now" CTA
            
            creative = client.create_image_creative(
                page_id=page_id,
                name=creative_name,
                supabase_storage_url=supabase_storage_url,  # Use Supabase Storage URL
                image_path=image_path if not supabase_storage_url else None,
                primary_text=primary_text,
                headline=ad_copy_dict.get("headline", ""),
                description=ad_copy_dict.get("description", ""),
                call_to_action="SHOP_NOW",  # "Shop now" CTA
                instagram_actor_id=instagram_actor_id,  # Add Instagram account
                creative_id=storage_creative_id,  # Pass storage creative_id for tracking
            )
            logger.info(f"Meta API create_image_creative response: {creative}")
        except Exception as e:
            error_str = str(e)
            if instagram_actor_id and "instagram_actor_id" in error_str.lower():
                logger.warning(
                    "Meta creative creation failed due to instagram_actor_id (%s). Retrying without Instagram placements.",
                    error_str,
                )
                resolved_instagram_actor_id = None
                try:
                    creative = client.create_image_creative(
                        page_id=page_id,
                        name=creative_name,
                        supabase_storage_url=supabase_storage_url,
                        image_path=image_path if not supabase_storage_url else None,
                        primary_text=primary_text,
                        headline=ad_copy_dict.get("headline", ""),
                        description=ad_copy_dict.get("description", ""),
                        call_to_action="SHOP_NOW",
                        instagram_actor_id=None,
                        use_env_instagram_actor=False,
                        creative_id=storage_creative_id,
                    )
                    logger.info(
                        "Meta API create_image_creative succeeded without instagram_actor_id. Creative will run on Facebook placements only."
                    )
                except Exception as retry_exc:
                    logger.error(
                        "Meta API create_image_creative retry without instagram_actor_id failed: %s",
                        retry_exc,
                        exc_info=True,
                    )
                    return None, None, False
            else:
                logger.error(f"Meta API create_image_creative failed: {e}", exc_info=True)
                return None, None, False
        
        meta_creative_id = creative.get("id")
        if not meta_creative_id:
            logger.error(f"Failed to get creative ID from Meta response. Response: {creative}")
            return None, None, False
        
        # Check for duplicate creative (use Meta's creative ID)
        if str(meta_creative_id) in existing_creative_ids:
            logger.debug(f"Skipping duplicate creative: {meta_creative_id}")
            return str(meta_creative_id), None, False
        
        # Create descriptive ad name for Ads Manager
        # Include headline snippet, date, and sequence for easy identification
        headline = ad_copy_dict.get("headline", "")
        if headline:
            # Extract first 3-4 words from headline (max 30 chars)
            headline_words = headline.split()[:4]
            headline_snippet = " ".join(headline_words)
            if len(headline_snippet) > 30:
                headline_snippet = headline_snippet[:27] + "..."
            # Clean for filename safety
            headline_snippet = headline_snippet.replace(":", "").replace("/", "").replace("\\", "").strip()
        else:
            headline_snippet = "Creative"
        
        # Get date in YYMMDD format for easy sorting
        from datetime import datetime
        date_str = datetime.now().strftime("%y%m%d")
        
        # Sequence number
        seq_num = active_count + created_count + 1
        
        # Build descriptive ad name: [ASC+] HeadlineSnippet - YYMMDD - #N
        ad_name = f"[ASC+] {headline_snippet} - {date_str} - #{seq_num}"
        
        # Ensure it's not too long (Meta has limits, keep under 100 chars)
        if len(ad_name) > 100:
            ad_name = f"[ASC+] {headline_snippet[:20]} - {date_str} - #{seq_num}"
        
        logger.info(
            "Creating ad with name='%s', adset_id='%s', creative_id='%s', instagram_actor_id=%s",
            ad_name,
            adset_id,
            meta_creative_id,
            bool(resolved_instagram_actor_id),
        )
        try:
            # Ensure creative_id is a string (Meta API requires string format)
            creative_id_str = str(meta_creative_id).strip()
            if not creative_id_str or not creative_id_str.isdigit():
                logger.error(f"Invalid creative_id format: {meta_creative_id} (expected numeric string)")
                return None, None, False
            
            ad = client.create_ad(
                adset_id=adset_id,
                name=ad_name,
                creative_id=creative_id_str,  # Use Meta's creative ID (ensure string format)
                status="ACTIVE",
                instagram_actor_id=resolved_instagram_actor_id,  # Add Instagram at ad level when available
                use_env_instagram_actor=bool(resolved_instagram_actor_id),
                tracking_specs=None,  # Will use default pixel-based tracking if pixel ID is set
            )
            logger.info(f"Meta API create_ad response: {ad}")
        except ValueError as ve:
            logger.error(f"Validation error creating ad: {ve}")
            return None, None, False
        except RuntimeError as re:
            error_str = str(re)
            if "500" in error_str or "unknown error" in error_str.lower():
                logger.warning(f"Meta API 500 error (may be transient): {re}")
                # Don't fail completely on 500 errors - might be transient
                return None, None, False
            else:
                logger.error(f"Meta API error creating ad: {re}")
                return None, None, False
        except Exception as e:
            logger.error(f"Meta API create_ad failed: {e}", exc_info=True)
            return str(meta_creative_id), None, False
        
        ad_id = ad.get("id")
        if ad_id:
            logger.info(f"Successfully created ad with ad_id={ad_id}")
            existing_creative_ids.add(str(meta_creative_id))
            
            # Update creative_intelligence with Supabase Storage URL and Meta creative ID
            try:
                from infrastructure.supabase_storage import get_validated_supabase_client, SupabaseStorage
                supabase_client = get_validated_supabase_client()
                if supabase_client:
                    # Record ad creation time (populates ad_creation_times table)
                    try:
                        storage = SupabaseStorage(supabase_client)
                        lifecycle_id = f"lifecycle_{ad_id}"
                        storage.record_ad_creation(ad_id, lifecycle_id, "asc_plus")
                        logger.debug(f"✅ Recorded ad creation time for {ad_id}")
                    except Exception as e:
                        logger.debug(f"Failed to record ad creation time: {e}")
                    
                    # Store initial historical data (populates historical_data table)
                    try:
                        storage = SupabaseStorage(supabase_client)
                        lifecycle_id = f"lifecycle_{ad_id}"
                        # Store initial metrics as historical data
                        storage.store_historical_data(ad_id, lifecycle_id, "asc_plus", "spend", 0.0)
                        storage.store_historical_data(ad_id, lifecycle_id, "asc_plus", "impressions", 0.0)
                        storage.store_historical_data(ad_id, lifecycle_id, "asc_plus", "clicks", 0.0)
                        storage.store_historical_data(ad_id, lifecycle_id, "asc_plus", "purchases", 0.0)
                        logger.debug(f"✅ Stored initial historical data for {ad_id}")
                    except Exception as e:
                        logger.debug(f"Failed to store initial historical data: {e}")
                    creative_intel_data = {
                        "creative_id": str(meta_creative_id),  # Use Meta's creative ID
                        "ad_id": ad_id,
                        "creative_type": "image",
                        "metadata": {
                            "ad_copy": creative_data.get("ad_copy"),
                            "flux_request_id": creative_data.get("flux_request_id"),
                            "storage_creative_id": storage_creative_id,  # Store our internal ID
                            "scenario_description": creative_data.get("scenario_description"),  # Store scenario for ML learning
                        },
                    }
                    # Add optional fields if available
                    if creative_data.get("supabase_storage_url"):
                        creative_intel_data["supabase_storage_url"] = creative_data.get("supabase_storage_url")
                    if creative_data.get("image_prompt"):
                        creative_intel_data["image_prompt"] = creative_data.get("image_prompt")
                    if creative_data.get("text_overlay"):
                        creative_intel_data["text_overlay_content"] = creative_data.get("text_overlay")
                    
                    # Ensure performance metrics are calculated
                    try:
                        from infrastructure.data_optimizer import CreativeIntelligenceOptimizer
                        optimizer = CreativeIntelligenceOptimizer(supabase_client)
                        metrics = optimizer.calculate_performance_metrics(
                            str(meta_creative_id),
                            ad_id,
                        )
                        # Add calculated metrics to creative_intel_data
                        creative_intel_data['avg_ctr'] = metrics.get('avg_ctr', 0.0)
                        creative_intel_data['avg_cpa'] = metrics.get('avg_cpa', 0.0)
                        creative_intel_data['avg_roas'] = metrics.get('avg_roas', 0.0)
                        creative_intel_data['performance_score'] = metrics.get('performance_score', 0.0)
                        creative_intel_data['fatigue_index'] = metrics.get('fatigue_index', 0.0)
                    except ImportError:
                        pass  # Optimizer not available
                    except Exception as opt_error:
                        logger.debug(f"Failed to calculate metrics: {opt_error}")
                    
                    supabase_client.table("creative_intelligence").upsert(
                        creative_intel_data,
                        on_conflict="creative_id,ad_id"
                    ).execute()
                    
                    # Schedule async update for accurate metrics
                    try:
                        from infrastructure.data_optimizer import CreativeIntelligenceOptimizer
                        optimizer = CreativeIntelligenceOptimizer(supabase_client)
                        optimizer.update_creative_performance(str(meta_creative_id), ad_id)
                    except Exception:
                        pass  # Non-critical
            except Exception as e:
                logger.warning(f"Failed to update creative_intelligence with storage URL: {e}")
            
            # Track in ML system
            if ml_system:
                try:
                    ml_system.record_creative_creation(
                        ad_id=ad_id,
                        creative_data=creative_data,
                        performance_data={},
                    )
                except (AttributeError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to track creative in ML system: {e}")
            
            # Store initial performance data (populates performance_metrics and ad_lifecycle tables)
            try:
                from infrastructure.supabase_storage import get_validated_supabase_client
                supabase_client = get_validated_supabase_client()
                if supabase_client:
                    # Get campaign_id if not provided
                    final_campaign_id = campaign_id or ''
                    if not final_campaign_id:
                        try:
                            # Try to get campaign_id from adset
                            adset_info = client._graph_get_object(f"{adset_id}", params={"fields": "campaign_id"})
                            if adset_info:
                                final_campaign_id = adset_info.get('campaign_id', '')
                        except Exception:
                            pass
                    
                    # Create initial ad_data with zero values for new ad
                    initial_ad_data = {
                        'ad_id': ad_id,
                        'creative_id': str(meta_creative_id),
                        'campaign_id': final_campaign_id,
                        'adset_id': adset_id,
                        'lifecycle_id': f'lifecycle_{ad_id}',
                        'spend': 0.0,
                        'impressions': 0,
                        'clicks': 0,
                        'purchases': 0,
                        'atc': 0,
                        'ic': 0,
                        'ctr': 0.0,
                        'cpc': 0.0,
                        'cpm': 0.0,
                        'roas': 0.0,
                        'cpa': None,
                        'status': 'active',
                    }
                    # Import and call store_performance_data_in_supabase
                    import sys
                    # Add src directory to path to import from main (use os from module level, not local import)
                    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if src_dir not in sys.path:
                        sys.path.insert(0, src_dir)
                    from main import store_performance_data_in_supabase
                    store_performance_data_in_supabase(supabase_client, initial_ad_data, "asc_plus", ml_system)
                    logger.debug(f"✅ Stored initial performance data for new ad {ad_id}")
            except Exception as e:
                logger.debug(f"Failed to store initial performance data (non-critical): {e}")
            
            return str(meta_creative_id), ad_id, True
        else:
            logger.warning(f"Failed to get ad ID from Meta response")
            return str(meta_creative_id), None, False
            
    except Exception as e:
        logger.error(f"Error creating creative and ad: {e}", exc_info=True)
        return None, None, False


def ensure_asc_plus_campaign(
    client: MetaClient,
    settings: Dict[str, Any],
    store: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Ensure ASC+ campaign and adset exist.
    Returns (campaign_id, adset_id) or (None, None) on failure.
    """
    try:
        # Check if campaign already exists
        campaign_id = cfg(settings, "ids.asc_plus_campaign_id") or ""
        adset_id = cfg(settings, "ids.asc_plus_adset_id") or ""
        
        if campaign_id and adset_id:
            # Verify they still exist
            try:
                # Try to get campaign info
                client._graph_get_object(f"{campaign_id}", params={"fields": "id,name,status"})
                return campaign_id, adset_id
            except Exception:
                # Campaign doesn't exist, create new
                pass
        
        # Create ASC+ campaign
        campaign_name = "[ASC+] Brava - US Men"
        campaign = client.ensure_campaign(
            name=campaign_name,
            objective="SALES",
            buying_type="AUCTION",
        )
        campaign_id = campaign.get("id")
        
        if not campaign_id:
            notify("❌ Failed to create ASC+ campaign")
            return None, None
        
        placements_cfg = cfg(settings, "placements.asc_plus") or {}
        default_placements = cfg(settings, "placements.default.publisher_platforms") or ["facebook", "instagram"]
        if placements_cfg.get("advantage_plus_placements"):
            adset_placements: Optional[List[str]] = None
        else:
            adset_placements = placements_cfg.get("publisher_platforms") or default_placements
        
        # Create ASC+ adset with Advantage+ placements
        asc_config = cfg(settings, "asc_plus") or {}
        daily_budget = cfg_or_env_f(asc_config, "daily_budget_eur", "ASC_PLUS_BUDGET", 50.0)
        
        # Validate budget
        if daily_budget < ASC_PLUS_BUDGET_MIN:
            notify(f"⚠️ ASC+ budget too low: €{daily_budget:.2f}. Minimum is €{ASC_PLUS_BUDGET_MIN:.2f}")
            daily_budget = ASC_PLUS_BUDGET_MIN
        elif daily_budget > ASC_PLUS_BUDGET_MAX:
            notify(f"⚠️ ASC+ budget too high: €{daily_budget:.2f}. Capping at €{ASC_PLUS_BUDGET_MAX:.2f}")
            daily_budget = ASC_PLUS_BUDGET_MAX
        
        # Verify budget matches target active ads
        target_ads = cfg(settings, "asc_plus.target_active_ads") or 10
        budget_per_creative = daily_budget / target_ads if target_ads > 0 else daily_budget
        min_budget_per_creative = cfg_or_env_f(asc_config, "min_budget_per_creative_eur", None, ASC_PLUS_MIN_BUDGET_PER_CREATIVE)
        
        if budget_per_creative < min_budget_per_creative:
            notify(f"⚠️ Budget per creative (€{budget_per_creative:.2f}) below minimum (€{min_budget_per_creative:.2f})")
            notify(f"   Consider increasing daily budget or reducing target active ads")
        
        # Targeting: US, Men, 18-54
        targeting = {
            "age_min": 18,
            "age_max": 54,
            "genders": [1],  # Men
            "geo_locations": {"countries": ["US"]},
        }
        
        # Create adset with Advantage+ placements
        adset_name = "[ASC+] US Men"
        adset = client.ensure_adset(
            campaign_id=campaign_id,
            name=adset_name,
            daily_budget=daily_budget,
            optimization_goal="OFFSITE_CONVERSIONS",
            billing_event="IMPRESSIONS",
            bid_strategy="LOWEST_COST_WITHOUT_CAP",
            targeting=targeting,
            placements=list(adset_placements) if adset_placements else None,
            status="PAUSED",
        )
        
        adset_id = adset.get("id")
        if not adset_id:
            notify("❌ Failed to create ASC+ adset")
            return None, None
        
        # Verify budget was set correctly and ensure it's at least base budget
        base_budget = 50.0
        try:
            adset_budget = client.get_adset_budget(adset_id)
            if adset_budget:
                # If existing adset has budget below base, restore to base
                if adset_budget < base_budget:
                    logger.info(f"Restoring adset budget from €{adset_budget:.2f} to base €{base_budget:.2f}")
                    try:
                        client.update_adset_budget(
                            adset_id=adset_id,
                            daily_budget=base_budget,
                            current_budget=adset_budget,
                        )
                        notify(f"✅ Restored adset budget to base: €{base_budget:.2f}/day")
                        adset_budget = base_budget
                    except Exception as e:
                        logger.warning(f"Failed to restore base budget: {e}")
                elif abs(adset_budget - daily_budget) > 0.01:
                    # Budget mismatch but above base - log but don't change (might be scaled)
                    logger.debug(f"Budget mismatch: requested €{daily_budget:.2f}, got €{adset_budget:.2f} (may be scaled)")
        except Exception:
            pass
        
        final_budget = adset_budget if 'adset_budget' in locals() and adset_budget else daily_budget
        notify(f"✅ ASC+ campaign ready: {campaign_id}, adset: {adset_id}, budget: €{final_budget:.2f}/day")
        return campaign_id, adset_id
        
    except Exception as e:
        alert_error(f"Error ensuring ASC+ campaign: {e}")
        return None, None


def generate_new_creative(
    image_generator: ImageCreativeGenerator,
    product_info: Dict[str, Any],
    creative_index: int,
) -> Optional[Dict[str, Any]]:
    """Generate a new static image creative."""
    try:
        creative_data = image_generator.generate_creative(
            product_info=product_info,
            creative_style="Luxury, premium, sophisticated",
        )
        
        if not creative_data:
            notify(f"❌ Failed to generate creative #{creative_index}")
            return None
        
        return creative_data
        
    except Exception as e:
        notify(f"❌ Error generating creative: {e}")
        return None


def _legacy_run_asc_plus_tick(
    client: MetaClient,
    settings: Dict[str, Any],
    rules: Dict[str, Any],
    store: Any,
    ml_system: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run one tick of ASC+ campaign management.
    Ensures 5 creatives are always live.
    Uses advanced ML system for optimization.
    """
    try:
        # Health check
        from infrastructure.health_check import health_checker
        health_status = health_checker.get_overall_health()
        if health_status.value == "unhealthy":
            logger.warning("System health check failed")
        
        # Self-healing
        if ADVANCED_ML_AVAILABLE and ml_system:
            try:
                from ml.auto_optimization import create_self_healing_system
                from infrastructure.health_check import get_health_status
                
                healing_system = create_self_healing_system()
                health_data = get_health_status()
                issues = healing_system.detect_issues(health_data)
                
                for issue in issues:
                    healing_system.attempt_recovery(issue)
            except Exception as e:
                logger.error(f"Self-healing error: {e}")
        # Ensure campaign exists
        campaign_id, adset_id = ensure_asc_plus_campaign(client, settings, store)
        if not campaign_id or not adset_id:
            return {"ok": False, "error": "Failed to ensure ASC+ campaign"}
        
        # Get current ads using improved counting method (with campaign-level verification)
        try:
            # Use the new count_active_ads_in_adset method with campaign_id for more accurate counting
            active_count = client.count_active_ads_in_adset(adset_id, campaign_id=campaign_id)
            # Also get the full list for processing
            ads = client.list_ads_in_adset(adset_id)
            active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
            logger.info(f"📊 Active ads count: {active_count} (from improved method), {len(active_ads)} (from direct list)")
        except Exception as e:
            logger.warning(f"Failed to use improved ad counting, falling back to basic method: {e}")
            ads = client.list_ads_in_adset(adset_id)
            active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
            active_count = len(active_ads)
        
        target_count = cfg(settings, "asc_plus.target_active_ads") or 10
        
        # Get insights for current ads (with caching)
        from infrastructure.caching import cache_manager
        cache_key = f"ad_insights_{adset_id}_{datetime.now().strftime('%Y%m%d%H')}"
        
        insights = cache_manager.get(cache_key, namespace="insights")
        if not insights:
            insights = client.get_ad_insights(
                level="ad",
                time_range={"since": (datetime.now(LOCAL_TZ) - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), "until": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")},
            )
            # Convert dict_values/dict_keys to list for caching (prevents pickle errors)
            if insights:
                if isinstance(insights, dict):
                    insights = list(insights.values()) if insights else []
                elif hasattr(insights, '__iter__') and not isinstance(insights, (list, tuple, str)):
                    insights = list(insights)
            # Cache for 1 hour
            cache_manager.set(cache_key, insights, ttl_seconds=3600, namespace="insights")
        
        # Process ads for kill decisions
        asc_rules = cfg(rules, "asc_plus") or {}
        kill_rules = asc_rules.get("kill", [])
        keep_rules = asc_rules.get("keep", [])
        lock_rules = asc_rules.get("locks", [])
        minimums_cfg = asc_rules.get("minimums", {})
        now_utc = datetime.now(timezone.utc)

        creation_lookup: Dict[str, datetime] = {}
        metadata_map: Dict[str, Any] = {}
        lock_state: Dict[str, Dict[str, Any]] = {}
        metadata_updates: Dict[str, Dict[str, Any]] = {}
        supabase_lock_client: Optional[Any] = None

        try:
            from infrastructure.supabase_storage import get_validated_supabase_client, SupabaseStorage

            supabase_lock_client = get_validated_supabase_client(enable_validation=True)
            if supabase_lock_client:
                ad_ids_for_lookup = [str(ad.get("id") or "") for ad in active_ads if ad.get("id")]
                metadata_map, lock_state = _load_lock_metadata(supabase_lock_client, ad_ids_for_lookup)
                storage_for_minimums = SupabaseStorage(supabase_lock_client)
                for lookup_ad_id in ad_ids_for_lookup:
                    created_at_val = storage_for_minimums.get_ad_creation_time(lookup_ad_id)
                    if created_at_val:
                        creation_lookup[lookup_ad_id] = created_at_val
        except ImportError:
            supabase_lock_client = None
        except Exception as exc:
            logger.debug(f"Failed to initialize Supabase helpers for kill evaluation: {exc}")
            supabase_lock_client = supabase_lock_client or None
        
        killed_count = 0
        for ad in active_ads:
            ad_id_raw = ad.get("id")
            ad_id = str(ad_id_raw) if ad_id_raw else ""
            if not ad_id:
                continue

            ad_insight = next((i for i in insights if str(i.get("ad_id")) == ad_id), None)
            if not ad_insight:
                continue
            
            has_minimums, spend, impressions, link_clicks, hours_live = _has_minimum_delivery(
                ad_id,
                ad,
                ad_insight,
                minimums_cfg,
                creation_lookup,
                now_utc,
            )
            if not has_minimums:
                logger.debug(
                    "Skipping kill evaluation for %s: insufficient delivery (spend=€%.2f, impressions=%.0f, link_clicks=%.0f, hours=%.1f)",
                    _short_id(ad_id),
                    spend,
                    impressions,
                    link_clicks,
                    hours_live,
                )
                continue
            
            ctr = _ctr(ad_insight)
            cpa = _cpa(ad_insight)
            roas = _roas(ad_insight)
            cpm = _cpm(ad_insight)
            purch, atc = _purchase_and_atc_counts(ad_insight)
            
            # Evaluate lock rules first (may skip kill checks)
            lock_info_existing = lock_state.get(ad_id, {})
            is_locked = False
            if lock_rules:
                lock_result, lock_info, lock_changed = _evaluate_lock_rules(
                    lock_rules,
                    ctr,
                    cpm,
                    now_utc,
                    lock_info_existing,
                )
                is_locked = lock_result
                if lock_changed:
                    meta = metadata_map.get(ad_id)
                    if not isinstance(meta, dict):
                        meta = {}
                    if lock_info:
                        meta["lock"] = {k: v for k, v in lock_info.items() if v is not None}
                        lock_state[ad_id] = lock_info
                    else:
                        meta.pop("lock", None)
                        lock_state.pop(ad_id, None)
                    metadata_map[ad_id] = meta
                    metadata_updates[ad_id] = meta
                else:
                    if lock_info:
                        lock_state[ad_id] = lock_info
            if is_locked:
                logger.debug(
                    "Creative %s is lock-protected; skipping performance kill checks.",
                    _short_id(ad_id),
                )
                continue

            # Apply keep rules (short-circuit kill evaluation)
            if _should_keep_creative(keep_rules, ctr, cpm):
                logger.debug(
                    "Creative %s meets keep conditions (CTR=%.2f%%, CPM=€%.2f); skipping kill rules.",
                    _short_id(ad_id),
                    ctr * 100,
                    cpm,
                )
                continue
            
            should_kill = False
            kill_reason = ""
            
            for rule in kill_rules:
                rule_type = rule.get("type")
                
                if rule_type == "ctr_cpm_spend_combo":
                    if (
                        spend >= rule.get("spend_gte", 0)
                        and ctr < rule.get("ctr_lt", 0)
                        and cpm > rule.get("cpm_gt", float("inf"))
                    ):
                        should_kill = True
                        kill_reason = (
                            f"CTR {ctr*100:.2f}% with CPM €{cpm:.2f} after €{spend:.2f} spend"
                        )
                        break
                
                elif rule_type == "ctr_spend_floor":
                    if spend >= rule.get("spend_gte", 0) and ctr < rule.get("ctr_lt", 0):
                        should_kill = True
                        kill_reason = (
                            f"CTR {ctr*100:.2f}% below floor after €{spend:.2f} spend"
                        )
                        break
                
                elif rule_type == "cpm_spend_ctr_combo":
                    if (
                        spend >= rule.get("spend_gte", 0)
                        and cpm > rule.get("cpm_gt", float("inf"))
                        and ctr < rule.get("ctr_lt", 0)
                    ):
                        should_kill = True
                        kill_reason = (
                            f"CPM €{cpm:.2f} too high with CTR {ctr*100:.2f}% after €{spend:.2f} spend"
                        )
                        break

                elif rule_type == "zero_performance_quick_kill":
                    if spend >= rule.get("spend_gte", 0) and ctr < rule.get("ctr_lt", 0):
                        should_kill = True
                        kill_reason = "Zero performance"
                        break
                
                elif rule_type == "cpm_above":
                    if spend >= rule.get("spend_gte", 0) and cpm > rule.get("cpm_above", float("inf")):
                        should_kill = True
                        kill_reason = f"CPM too high: €{cpm:.2f}"
                        break
                
                elif rule_type == "ctr_below":
                    if spend >= rule.get("spend_gte", 0) and ctr < rule.get("ctr_lt", 0):
                        should_kill = True
                        kill_reason = f"CTR too low: {ctr*100:.2f}%"
                        break
                
                elif rule_type == "spend_no_purchase":
                    if spend >= rule.get("spend_gte", 0) and purch == 0:
                        should_kill = True
                        kill_reason = f"No purchases after €{spend:.2f} spend"
                        break
                
                elif rule_type == "cpa_gte":
                    if purch > 0 and cpa >= rule.get("cpa_gte", float('inf')):
                        should_kill = True
                        kill_reason = f"CPA too high: €{cpa:.2f}"
                        break
                
                elif rule_type == "roas_below":
                    if spend >= rule.get("spend_gte", 0) and roas < rule.get("roas_lt", 0):
                        should_kill = True
                        kill_reason = f"ROAS too low: {roas:.2f}"
                        break
            
            if should_kill:
                try:
                    # Auto-pause underperformer (with retry)
                    from infrastructure.error_handling import retry_with_backoff
                    
                    @retry_with_backoff(max_retries=3)
                    def pause_ad(ad_id: str):
                        try:
                            client._graph_post(f"{ad_id}", {"status": "PAUSED"})
                        except Exception:
                            from facebook_business.adobjects.ad import Ad
                            Ad(ad_id).api_update(params={"status": "PAUSED"})
                    
                    pause_ad(ad_id)
                    killed_count += 1
                    active_count -= 1
                    
                    # Prepare metrics for alert
                    alert_metrics = {
                        "spend": spend,
                        "impressions": safe_f(ad_insight.get("impressions")),
                        "clicks": safe_f(ad_insight.get("clicks")),
                        "ctr": ctr,
                        "cpa": cpa,
                        "roas": roas,
                        "cpm": cpm,
                        "purchases": purch,
                    }
                    
                    alert_kill(
                        stage="ASC+",
                        entity_name=ad.get("name", "Unknown"),
                        reason=kill_reason,
                        metrics=alert_metrics,
                    )
                    
                    # Track kill in ML system
                    if ml_system:
                        try:
                            ml_system.record_creative_kill(
                                ad_id=ad_id,
                                reason=kill_reason,
                                performance_data=ad_insight,
                            )
                        except (AttributeError, ValueError, TypeError) as e:
                            logger.warning(f"Failed to track creative kill in ML system: {e}")
                    
                    # Mark creative as killed in storage
                    try:
                        from infrastructure.creative_storage import create_creative_storage_manager
                        from infrastructure.supabase_storage import get_validated_supabase_client
                        
                        supabase_client = get_validated_supabase_client()
                        if supabase_client:
                            storage_manager = create_creative_storage_manager(supabase_client)
                            if storage_manager:
                                # Get creative_id from ad data or creative object
                                creative_id = None
                                if isinstance(ad.get("creative"), dict):
                                    creative_id = ad.get("creative", {}).get("id")
                                elif isinstance(ad.get("creative"), str):
                                    creative_id = ad.get("creative")
                                else:
                                    creative_id = ad.get("creative_id") or ad_insight.get("creative_id")
                                
                                if creative_id:
                                    # Try to find storage_creative_id from creative_intelligence
                                    try:
                                        ci_result = supabase_client.table("creative_intelligence").select(
                                            "metadata"
                                        ).eq("creative_id", str(creative_id)).execute()
                                        if ci_result.data and len(ci_result.data) > 0:
                                            metadata = ci_result.data[0].get("metadata", {})
                                            storage_creative_id = metadata.get("storage_creative_id")
                                            if storage_creative_id:
                                                storage_manager.mark_creative_killed(storage_creative_id)
                                            else:
                                                # Fallback: use Meta creative_id
                                                storage_manager.mark_creative_killed(str(creative_id))
                                    except Exception as e2:
                                        logger.warning(f"Failed to lookup storage_creative_id: {e2}")
                                        # Fallback: try with Meta creative_id
                                        storage_manager.mark_creative_killed(str(creative_id))
                    except Exception as e:
                        logger.warning(f"Failed to mark creative as killed in storage: {e}")
                            
                except (KeyError, ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Failed to kill ad {ad_id}: {e}", exc_info=True)
                    notify(f"⚠️ Failed to kill ad {ad_id}: {e}")
        
        if supabase_lock_client and metadata_updates:
            for meta_ad_id, meta in metadata_updates.items():
                if not isinstance(meta, dict):
                    continue
                try:
                    supabase_lock_client.table("ad_lifecycle").update({"metadata": meta}).eq("ad_id", meta_ad_id).eq("stage", "asc_plus").execute()
                except Exception as exc:
                    logger.debug(f"Failed to persist lock metadata for {meta_ad_id}: {exc}")
        
        # Generate new creatives if needed - SMART: Only generate 1 at a time when needed
        # Refresh active count after kills using improved counting method (with campaign-level verification)
        try:
            # Use the new count_active_ads_in_adset method with campaign_id for more accurate counting
            active_count = client.count_active_ads_in_adset(adset_id, campaign_id=campaign_id)
            # Also get the full list for processing
            ads = client.list_ads_in_adset(adset_id)
            active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
            logger.info(f"📊 Active ads count after kills: {active_count} (from improved method), {len(active_ads)} (from direct list)")
        except Exception as e:
            logger.warning(f"Failed to use improved ad counting, falling back to basic method: {e}")
            ads = client.list_ads_in_adset(adset_id)
            active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
            active_count = len(active_ads)
        needed_count = max(0, target_count - active_count)
        
        # HARD STOP: If we already have the target count, do NOT generate anything
        if active_count >= target_count:
            logger.info(f"✅ Already have {active_count} active creatives (target: {target_count}), STOPPING - no generation needed")
            notify(f"✅ Target reached: {active_count}/{target_count} active creatives - NO GENERATION")
            return {
                "campaign_id": campaign_id,
                "adset_id": adset_id,
                "active_count": active_count,
                "target_count": target_count,
                "created_count": 0,
                "killed_count": killed_count,
            }
        
        # SMART GENERATION: Check queue first, only generate if needed
        # This prevents overusage of Flux and ChatGPT
        if needed_count > 0:
            # STEP 1: Check for queued creatives first
            from infrastructure.creative_storage import create_creative_storage_manager
            from infrastructure.supabase_storage import get_validated_supabase_client
            
            supabase_client = get_validated_supabase_client()
            storage_manager = None
            queued_creative = None
            
            if supabase_client:
                try:
                    storage_manager = create_creative_storage_manager(supabase_client)
                    if storage_manager:
                        queued_creative = storage_manager.get_queued_creative()
                except Exception as e:
                    logger.warning(f"Failed to check creative queue: {e}")
            
            # Initialize image generator early (needed for both queued and new creatives)
            from creative.image_generator import create_image_generator
            import os
            image_generator = create_image_generator(
                flux_api_key=os.getenv("FLUX_API_KEY"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                ml_system=ml_system,
            )
            
            # If we found a queued creative, use it first (but continue if we need more)
            created_count = 0  # Track total ads created this tick
            if queued_creative:
                logger.info(f"✅ Using queued creative: {queued_creative.get('creative_id')}")
                notify(f"📦 Using queued creative (need {needed_count} more, have {active_count} active)")
                
                # Get the creative data from storage
                storage_url = queued_creative.get("storage_url")
                storage_creative_id = queued_creative.get("creative_id")
                metadata = queued_creative.get("metadata", {})
                
                # Reconstruct creative_data from metadata
                creative_data = {
                    "supabase_storage_url": storage_url,
                    "creative_id": storage_creative_id,
                    "ad_copy": metadata.get("ad_copy", {}),
                    "text_overlay": metadata.get("text_overlay", ""),
                    "image_prompt": metadata.get("image_prompt", ""),
                    "scenario_description": metadata.get("scenario_description", ""),
                }
                
                # Create ad with queued creative
                creative_id, ad_id, success = _create_creative_and_ad(
                    client=client,
                    image_generator=image_generator,  # Now always available
                    creative_data=creative_data,
                    adset_id=adset_id,
                    active_count=active_count,
                    created_count=0,
                    existing_creative_ids=set(),
                    ml_system=ml_system,
                    campaign_id=campaign_id,
                )
                
                if success and creative_id and ad_id:
                    # Mark creative as active
                    if storage_manager:
                        storage_manager.mark_creative_active(storage_creative_id, ad_id)
                    
                    logger.info(f"✅ Successfully used queued creative {storage_creative_id} for ad {ad_id}")
                    notify(f"✅ Created ad {ad_id} using queued creative")
                    
                    # Update active count after creating ad
                    try:
                        active_count = client.count_active_ads_in_adset(adset_id, campaign_id=campaign_id)
                        logger.info(f"📊 Active ads after queued creative: {active_count}/{target_count}")
                    except Exception as e:
                        logger.debug(f"Failed to refresh active count: {e}")
                        active_count += 1  # Fallback: increment by 1
                    
                    # Check if we've reached target - if so, we're done
                    if active_count >= target_count:
                        logger.info(f"✅ Reached target: {active_count}/{target_count} active creatives after queued creative")
                        return {
                            "campaign_id": campaign_id,
                            "adset_id": adset_id,
                            "active_count": active_count,
                            "target_count": target_count,
                            "created_count": 1,
                            "killed_count": killed_count,
                        }
                    else:
                        # Still need more ads - continue to generation logic below
                        needed_count = max(0, target_count - active_count)
                        logger.info(f"📊 Still need {needed_count} more ads (have {active_count}, target {target_count}) - continuing to generation")
                        created_count = 1  # Track that we created 1 from queue
                else:
                    logger.warning(f"Failed to create ad with queued creative, will generate new one")
                    created_count = 0  # No ads created yet
                    # Continue to generation below
            
            # STEP 2: No queued creative available - generate exactly 1
            if not queued_creative:
                notify(f"📸 Generating EXACTLY 1 new creative (need {needed_count} more to reach {target_count}, currently {active_count} active)")
            
            # Initialize template library for faster creative generation
            template_library = None
            try:
                from creative.advanced_creative import create_template_library
                template_library = create_template_library()
                logger.info("✅ Creative template library initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize template library: {e}")
            
            # Initialize advanced ML system if available
            advanced_ml = None
            if ADVANCED_ML_AVAILABLE and ml_system:
                try:
                    advanced_ml = create_advanced_ml_system(
                        supabase_client=ml_system.supabase_client if hasattr(ml_system, 'supabase_client') else None,
                        image_generator=image_generator,
                        ml_system=ml_system,
                    )
                    notify("🚀 Advanced ML system activated")
                except Exception as e:
                    notify(f"⚠️ Failed to initialize advanced ML: {e}")
                    advanced_ml = None
            
            # Load product info from settings
            product_config = cfg(settings, "asc_plus.product") or {}
            product_info = {
                "name": product_config.get("name", "Brava Product"),
                "description": product_config.get("description", "Luxury product for men"),
                "features": product_config.get("features", ["Premium quality", "Luxury design", "Sophisticated"]),
                "brand_tone": product_config.get("brand_tone", "calm confidence"),
                "target_audience": product_config.get("target_audience", "Men aged 18-54"),
            }
            
            # Initialize counters (created_count may already be set if we used a queued creative)
            if 'created_count' not in locals():
                created_count = 0
            failed_count = 0
            failed_reasons = []
            max_attempts = needed_count * 3  # Allow up to 3x attempts in case of failures
            attempt_count = 0
            existing_creative_ids = set()  # Track created creative IDs to prevent duplicates
            skip_standard_generation = False  # Flag to skip standard generation if we already generated 1
            
            # Use advanced ML pipeline if available
            if advanced_ml and advanced_ml.creative_pipeline:
                notify("🎯 Using smart ML-driven creative generation (1 creative at a time)")
                try:
                    # SMART: Only generate 1 creative at a time when needed
                    # Use ML insights from killed creatives to inform generation
                    remaining_needed = target_count - active_count
                    
                    if remaining_needed <= 0:
                        logger.info(f"✅ Already have {active_count} active creatives (target: {target_count}), no generation needed")
                    else:
                        # SMART: Generate exactly 1 creative using ML insights
                        # The ML system will use insights from killed creatives to generate the best possible creative
                        logger.info(f"📸 Generating 1 smart creative using ML insights (need {remaining_needed} more, have {active_count} active)")
                        
                        # Generate exactly 1 creative - the pipeline will use ML insights to make it optimal
                        generated_creatives = advanced_ml.generate_optimized_creatives(
                            product_info,
                            target_count=1,  # Always generate exactly 1
                        )
                        
                        # Process the single generated creative
                        if generated_creatives and len(generated_creatives) > 0:
                            creative_data = generated_creatives[0]  # Only process the first (and only) creative
                            logger.info(f"Processing smart ML-driven creative: has supabase_storage_url={bool(creative_data.get('supabase_storage_url'))}, has image_path={bool(creative_data.get('image_path'))}, has ad_copy={bool(creative_data.get('ad_copy'))}")
                            
                            if not creative_data:
                                logger.warning(f"Creative generation returned None")
                                failed_count += 1
                                failed_reasons.append("Creative generation returned None")
                            else:
                                # Analyze creative with advanced ML (optional - already optimized)
                                if advanced_ml:
                                    try:
                                        analysis = advanced_ml.analyze_creative_performance(
                                            creative_data,
                                            {},
                                        )
                                        creative_data["ml_analysis"] = analysis
                                        
                                        # Quality check
                                        if advanced_ml.quality_checker:
                                            quality = advanced_ml.quality_checker.check_quality(creative_data)
                                            creative_data["quality_check"] = quality
                                            
                                            if not quality.get("passed_checks"):
                                                logger.warning(f"Creative failed quality checks: {quality}")
                                                failed_count += 1
                                                failed_reasons.append("Creative failed quality check")
                                            else:
                                                # Create creative and ad
                                                creative_id, ad_id, success = _create_creative_and_ad(
                                                    client=client,
                                                    image_generator=image_generator,
                                                    creative_data=creative_data,
                                                    adset_id=adset_id,
                                                    active_count=active_count,
                                                    created_count=created_count,
                                                    existing_creative_ids=existing_creative_ids,
                                                    ml_system=ml_system,
                                                    campaign_id=campaign_id,
                                                )
                                                
                                                if success and creative_id and ad_id:
                                                    created_count += 1
                                                    existing_creative_ids.add(creative_id)
                                                    logger.info(f"✅ Successfully created smart creative {creative_id} and ad {ad_id}")
                                                    
                                                    # Look up the recently created creative in storage and mark as active
                                                    if storage_manager:
                                                        try:
                                                            # Get creative_id from creative_data
                                                            storage_creative_id = creative_data.get("creative_id")
                                                            if not storage_creative_id:
                                                                # Try to find it by looking up recently created
                                                                recent_creative = storage_manager.get_recently_created_creative(minutes_back=5)
                                                                if recent_creative:
                                                                    storage_creative_id = recent_creative.get("creative_id")
                                                            
                                                            if storage_creative_id:
                                                                storage_manager.mark_creative_active(storage_creative_id, ad_id)
                                                                logger.info(f"✅ Marked creative {storage_creative_id} as active")
                                                        except Exception as e:
                                                            logger.warning(f"Failed to mark creative as active: {e}")
                                                    
                                                    # Refresh active count and check if we need more
                                                    ads = client.list_ads_in_adset(adset_id)
                                                    active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
                                                    active_count = len(active_ads)
                                                    
                                                    if active_count >= target_count:
                                                        logger.info(f"✅ Reached target: {active_count}/{target_count} active creatives - stopping")
                                                        skip_standard_generation = True
                                                    else:
                                                        logger.info(f"✅ Created 1 ad, but still need {target_count - active_count} more (active: {active_count}, target: {target_count})")
                                                        skip_standard_generation = False  # Allow standard generation to continue
                                                else:
                                                    failed_count += 1
                                                    failed_reasons.append(f"Failed to create (creative_id={creative_id}, ad_id={ad_id})")
                                    except Exception as e:
                                        logger.warning(f"Error analyzing creative: {e}, creating anyway")
                                        # Create anyway if analysis fails
                                        creative_id, ad_id, success = _create_creative_and_ad(
                                            client=client,
                                            image_generator=image_generator,
                                            creative_data=creative_data,
                                            adset_id=adset_id,
                                            active_count=active_count,
                                            created_count=created_count,
                                            existing_creative_ids=existing_creative_ids,
                                            ml_system=ml_system,
                                            campaign_id=campaign_id,
                                        )
                                        
                                        if success and creative_id and ad_id:
                                            created_count += 1
                                            existing_creative_ids.add(creative_id)
                                            
                                            # Look up the recently created creative in storage and mark as active
                                            if storage_manager:
                                                try:
                                                    storage_creative_id = creative_data.get("creative_id")
                                                    if not storage_creative_id:
                                                        recent_creative = storage_manager.get_recently_created_creative(minutes_back=5)
                                                        if recent_creative:
                                                            storage_creative_id = recent_creative.get("creative_id")
                                                    
                                                    if storage_creative_id:
                                                        storage_manager.mark_creative_active(storage_creative_id, ad_id)
                                                        logger.info(f"✅ Marked creative {storage_creative_id} as active")
                                                except Exception as e:
                                                    logger.warning(f"Failed to mark creative as active: {e}")
                                            
                                            # HARD STOP: We created 1 ad, STOP immediately
                                            skip_standard_generation = True
                                            logger.info(f"🛑 HARD STOP: Created 1 ad - stopping all further generation")
                                        else:
                                            failed_count += 1
                                            failed_reasons.append(f"Failed to create after analysis error")
                                else:
                                    # No advanced ML - create directly
                                    creative_id, ad_id, success = _create_creative_and_ad(
                                        client=client,
                                        image_generator=image_generator,
                                        creative_data=creative_data,
                                        adset_id=adset_id,
                                        active_count=active_count,
                                        created_count=created_count,
                                        existing_creative_ids=existing_creative_ids,
                                        ml_system=ml_system,
                                        campaign_id=campaign_id,
                                    )
                                    
                                    if success and creative_id and ad_id:
                                        created_count += 1
                                        existing_creative_ids.add(creative_id)
                                        
                                        # Look up the recently created creative in storage and mark as active
                                        if storage_manager:
                                            try:
                                                storage_creative_id = creative_data.get("creative_id")
                                                if not storage_creative_id:
                                                    recent_creative = storage_manager.get_recently_created_creative(minutes_back=5)
                                                    if recent_creative:
                                                        storage_creative_id = recent_creative.get("creative_id")
                                                
                                                if storage_creative_id:
                                                    storage_manager.mark_creative_active(storage_creative_id, ad_id)
                                                    logger.info(f"✅ Marked creative {storage_creative_id} as active")
                                            except Exception as e:
                                                logger.warning(f"Failed to mark creative as active: {e}")
                                    else:
                                        failed_count += 1
                                        failed_reasons.append(f"Failed to create")
                        else:
                            logger.warning(f"No creative returned from pipeline")
                            failed_count += 1
                            failed_reasons.append("Pipeline returned empty list")
                    
                    # Final check - refresh active count
                    ads = client.list_ads_in_adset(adset_id)
                    active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
                    active_count = len(active_ads)
                    
                    # Check if we've reached the target - if so, stop
                    if active_count >= target_count:
                        logger.info(f"✅ Reached target: {active_count}/{target_count} active creatives - STOPPING")
                        skip_standard_generation = True
                        advanced_ml = None
                    elif created_count >= 1 and active_count < target_count:
                        # Generated 1, but still need more - allow standard generation to continue
                        logger.info(f"✅ Generated 1 creative via advanced ML, but still need {target_count - active_count} more (active: {active_count}, target: {target_count})")
                        skip_standard_generation = False  # Allow standard generation to fill the gap
                        advanced_ml = None  # Don't use advanced ML again this tick
                    else:
                        # Fallback to standard generation ONLY if we haven't generated anything yet
                        if active_count < target_count and created_count == 0:
                            logger.warning(f"Advanced pipeline didn't create any creatives, falling back to standard generation")
                            advanced_ml = None
                        skip_standard_generation = False
                except Exception as e:
                    logger.error(f"Advanced pipeline failed with exception, using standard generation: {e}", exc_info=True)
                    advanced_ml = None
            
            # Standard generation (fallback) - Continue generating until target is reached
            if not skip_standard_generation and (not advanced_ml or not advanced_ml.creative_pipeline):
                # Refresh active_count before standard generation
                ads = client.list_ads_in_adset(adset_id)
                active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
                active_count = len(active_ads)
                
                # Only generate if we still need more
                if active_count >= target_count:
                    logger.info(f"✅ Target reached: {active_count}/{target_count} active creatives - skipping standard generation")
                else:
                    # Get ML insights from killed creatives to inform generation
                    ml_insights = None
                    if ml_system and hasattr(ml_system, 'get_creative_insights'):
                        try:
                            ml_insights = ml_system.get_creative_insights()
                            logger.info("✅ Using ML insights from killed creatives to inform new generation")
                        except (AttributeError, ValueError, TypeError) as e:
                            logger.debug(f"Failed to get ML insights: {e}")
                
                    # Generate creatives until target is reached (allow enough attempts to reach target)
                    remaining_needed = target_count - active_count
                    max_attempts = min(remaining_needed * 2, 10)  # Allow up to 10 attempts or 2x needed count (whichever is less)
                    attempts = 0
                    
                    logger.info(f"🎯 Starting generation loop: need {remaining_needed} more ads (current: {active_count}, target: {target_count}, max attempts: {max_attempts})")
                    
                    while active_count < target_count and attempts < max_attempts:
                        attempts += 1
                        remaining_needed = target_count - active_count
                        logger.info(f"📸 Standard generation: Generating creative {attempts}/{max_attempts} (need {remaining_needed} more, have {active_count} active)")
                        
                        try:
                            # Use template library if available for faster generation
                            template_to_use = None
                            if template_library:
                                try:
                                    top_templates = template_library.get_top_templates(top_k=3)
                                    if top_templates:
                                        # Use top performing template
                                        template_to_use = top_templates[0]
                                        logger.info(f"📋 Using template: {template_to_use.name} (score: {template_to_use.performance_score:.2f})")
                                except Exception as e:
                                    logger.debug(f"Template selection failed: {e}")
                            
                            # Generate exactly 1 creative with ML insights and template
                            creative_data = image_generator.generate_creative(
                                product_info,
                                creative_style=f"smart_creative_{active_count + created_count + 1}",
                            )
                            
                            # Apply template structure if available
                            if template_to_use and creative_data:
                                try:
                                    template_structure = template_to_use.structure
                                    # Apply template structure to creative data
                                    if "ad_copy" in template_structure:
                                        creative_data["ad_copy"] = {**creative_data.get("ad_copy", {}), **template_structure["ad_copy"]}
                                    if "image_prompt" in template_structure:
                                        creative_data["image_prompt"] = template_structure["image_prompt"]
                                    logger.info(f"✅ Applied template structure to creative")
                                except Exception as e:
                                    logger.warning(f"Failed to apply template: {e}")
                            
                            # Performance forecasting using ML system
                            if ml_system and creative_data and hasattr(ml_system, 'predict_creative_performance'):
                                try:
                                    from ml.creative_dna import CreativeDNAAnalyzer
                                    from infrastructure.supabase_storage import get_validated_supabase_client
                                    
                                    supabase_client = get_validated_supabase_client()
                                    if supabase_client:
                                        dna_analyzer = CreativeDNAAnalyzer(supabase_client=supabase_client)
                                        
                                        # Create DNA for new creative with enhanced metadata
                                        enhanced_metadata = {
                                            "format": creative_data.get("format", "static_image"),
                                            "style": creative_data.get("style", ""),
                                            "message_type": creative_data.get("message_type", ""),
                                            "target_motivation": creative_data.get("target_motivation", ""),
                                            "forecasted_roas": creative_data.get("forecasted_roas"),
                                            "forecasted_ctr": creative_data.get("forecasted_ctr"),
                                            "forecast_confidence": creative_data.get("forecast_confidence"),
                                        }
                                        
                                        dna = dna_analyzer.create_creative_dna(
                                            creative_id=creative_data.get("creative_id", "temp"),
                                            ad_id="temp",
                                            image_prompt=creative_data.get("image_prompt", ""),
                                            text_overlay=creative_data.get("text_overlay", ""),
                                            ad_copy=creative_data.get("ad_copy", {}),
                                            performance_data=None,
                                            enhanced_metadata=enhanced_metadata,
                                        )
                                        
                                        # Find similar high-performing creatives
                                        similar_creatives = dna_analyzer.find_similar_creatives(
                                            creative_id=dna.creative_id,
                                            top_k=5,
                                            min_similarity=0.6,
                                        )
                                        
                                        if similar_creatives:
                                            # Forecast performance based on similar creatives
                                            avg_roas = sum(c[0].roas for c in similar_creatives) / len(similar_creatives)
                                            avg_ctr = sum(c[0].ctr for c in similar_creatives) / len(similar_creatives)
                                            
                                            creative_data["forecasted_roas"] = avg_roas
                                            creative_data["forecasted_ctr"] = avg_ctr
                                            creative_data["forecast_confidence"] = sum(c[1] for c in similar_creatives) / len(similar_creatives)
                                            
                                            logger.info(f"🔮 Performance forecast: ROAS={avg_roas:.2f}, CTR={avg_ctr:.2%}, confidence={creative_data['forecast_confidence']:.2%}")
                                except Exception as e:
                                    logger.debug(f"Performance forecasting failed: {e}")
                            
                            if not creative_data:
                                failed_count += 1
                                failed_reasons.append(f"Standard generation attempt {attempts}: Generation returned None")
                                continue
                            
                            # Create creative in Meta
                            creative_id, ad_id, success = _create_creative_and_ad(
                                client=client,
                                image_generator=image_generator,
                                creative_data=creative_data,
                                adset_id=adset_id,
                                active_count=active_count,
                                created_count=created_count,
                                existing_creative_ids=existing_creative_ids,
                                ml_system=ml_system,
                                campaign_id=campaign_id,
                            )
                            
                            if success and ad_id:
                                created_count += 1
                                existing_creative_ids.add(str(creative_id))
                                logger.info(f"✅ Successfully created smart creative {creative_id} and ad {ad_id}")
                                
                                # Look up the recently created creative in storage and mark as active
                                if storage_manager:
                                    try:
                                        storage_creative_id = creative_data.get("creative_id")
                                        if not storage_creative_id:
                                            recent_creative = storage_manager.get_recently_created_creative(minutes_back=5)
                                            if recent_creative:
                                                storage_creative_id = recent_creative.get("creative_id")
                                        
                                        if storage_creative_id:
                                            storage_manager.mark_creative_active(storage_creative_id, ad_id)
                                            logger.info(f"✅ Marked creative {storage_creative_id} as active")
                                    except Exception as e:
                                        logger.warning(f"Failed to mark creative as active: {e}")
                                
                                # Refresh active count after creation using improved method
                                try:
                                    active_count = client.count_active_ads_in_adset(adset_id, campaign_id=campaign_id)
                                    logger.info(f"📊 Active ads after creation: {active_count}/{target_count}")
                                except Exception as e:
                                    logger.debug(f"Failed to refresh active count, using direct list: {e}")
                                    ads = client.list_ads_in_adset(adset_id)
                                    active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
                                    active_count = len(active_ads)
                                
                                logger.info(f"✅ Created creative {attempts}/{max_attempts} - now {active_count}/{target_count} active")
                                
                                # Check if we've reached target - if so, break out of loop
                                if active_count >= target_count:
                                    logger.info(f"✅ Reached target: {active_count}/{target_count} active creatives - stopping generation")
                                    break  # Stop generating
                                else:
                                    remaining_needed = target_count - active_count
                                    logger.info(f"📊 Still need {remaining_needed} more ads - continuing generation")
                            else:
                                failed_count += 1
                                if creative_id:
                                    failed_reasons.append(f"Standard generation attempt {attempts}: Duplicate creative ID")
                                else:
                                    failed_reasons.append(f"Standard generation attempt {attempts}: Failed to create")
                                    # Track failure in ML system
                                    if ml_system and hasattr(ml_system, 'record_creative_generation_failure'):
                                        try:
                                            ml_system.record_creative_generation_failure(
                                                reason="Failed to create creative and ad in Meta",
                                                product_info=product_info,
                                            )
                                        except (AttributeError, ValueError, TypeError):
                                            pass
                        except Exception as e:
                            logger.error(f"Standard generation attempt {attempts} failed: {e}", exc_info=True)
                            failed_count += 1
                            failed_reasons.append(f"Standard generation attempt {attempts}: {str(e)[:50]}")
            
            # Final check of active count
            ads = client.list_ads_in_adset(adset_id)
            active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
            final_active_count = len(active_ads)
            
            if created_count > 0:
                notify(f"✅ Created {created_count} new creatives for ASC+ campaign (now {final_active_count}/{target_count} active)")
            if failed_count > 0:
                notify(f"⚠️ Failed to create {failed_count} creatives. Reasons: {', '.join(failed_reasons[:3])}")
            
            if final_active_count < target_count:
                notify(f"⚠️ Still need {target_count - final_active_count} more active creatives (currently {final_active_count}/{target_count})")
            elif final_active_count >= target_count:
                notify(f"✅ Target reached: {final_active_count} active creatives")
            
            # Optimize ML tables data - ensure all performance metrics are correct
            try:
                from infrastructure.data_optimizer import create_ml_data_optimizer
                from infrastructure.supabase_storage import get_validated_supabase_client
                
                supabase_client = get_validated_supabase_client()
                if supabase_client:
                    optimizer = create_ml_data_optimizer(supabase_client)
                    # Run optimization (non-blocking, will update metrics in background)
                    try:
                        optimizer.optimize_all_tables(stage='asc_plus', force_recalculate=False)
                        logger.info("✅ ML tables optimized")
                    except Exception as opt_error:
                        logger.debug(f"Table optimization error (non-critical): {opt_error}")
            except ImportError:
                pass  # Optimizer not available
            except Exception as e:
                logger.debug(f"ML table optimization failed (non-critical): {e}")
            
            # Cleanup unused and killed creatives from storage
            try:
                from infrastructure.creative_storage import create_creative_storage_manager
                from infrastructure.supabase_storage import get_validated_supabase_client
                
                supabase_client = get_validated_supabase_client()
                if supabase_client:
                    storage_manager = create_creative_storage_manager(supabase_client)
                    if storage_manager:
                        # Cleanup unused creatives (30 days default)
                        unused_deleted = storage_manager.cleanup_unused_creatives()
                        # Cleanup killed creatives (7 days default)
                        killed_deleted = storage_manager.cleanup_killed_creatives()
                        if unused_deleted > 0 or killed_deleted > 0:
                            logger.info(f"🧹 Cleaned up {unused_deleted} unused and {killed_deleted} killed creatives")
            except Exception as e:
                logger.warning(f"Failed to cleanup creatives from storage: {e}")
            
            # Budget scaling - check if budget should be adjusted
            if OPTIMIZATION_SYSTEMS_AVAILABLE:
                try:
                    from infrastructure.supabase_storage import get_validated_supabase_client
                    supabase_client = get_validated_supabase_client()
                    
                    if supabase_client:
                        # Get performance data for budget scaling
                        performance_data = []
                        for ad_insight in insights:
                            if ad_insight.get("ad_id"):
                                performance_data.append({
                                    "ad_id": ad_insight.get("ad_id"),
                                    "spend": safe_f(ad_insight.get("spend")),
                                    "revenue": safe_f(ad_insight.get("purchase_roas", [{}])[0].get("value", 0)) if ad_insight.get("purchase_roas") else 0,
                                    "purchases": _purchase_and_atc_counts(ad_insight)[0],
                                    "roas": _roas(ad_insight),
                                    "cpa": _cpa(ad_insight),
                                    "date": datetime.now().date().isoformat(),
                                })
                        
                        if len(performance_data) >= 3:
                            # Get scaling configuration
                            scaling_config = cfg(settings, "asc_plus.scaling") or {}
                            scaling_enabled = scaling_config.get("enabled", True)
                            min_confidence = scaling_config.get("min_confidence", 0.75)
                            min_roas_for_scale = scaling_config.get("min_roas_for_scale", 1.5)
                            min_purchases_for_scale = scaling_config.get("min_purchases_for_scale", 5)
                            scale_up_threshold_pct = scaling_config.get("scale_up_threshold_pct", 20)
                            
                            if not scaling_enabled:
                                logger.debug("Budget scaling disabled in config")
                            else:
                                budget_engine = create_budget_scaling_engine()
                                current_budget = cfg_or_env_f(cfg(settings, "asc_plus") or {}, "daily_budget_eur", "ASC_PLUS_BUDGET", 50.0)
                                # Ensure current_budget is a float (cfg_or_env_f might return string)
                                try:
                                    current_budget = float(current_budget) if current_budget is not None else 50.0
                                except (ValueError, TypeError):
                                    current_budget = 50.0
                                
                                # Calculate aggregate performance metrics
                                total_spend = sum(p.get("spend", 0) for p in performance_data)
                                total_revenue = sum(p.get("revenue", 0) for p in performance_data)
                                total_purchases = sum(p.get("purchases", 0) for p in performance_data)
                                avg_roas = total_revenue / total_spend if total_spend > 0 else 0.0
                                
                                # Check if ready to scale (only scale up from €50 base budget)
                                base_budget = 50.0
                                ready_to_scale = (
                                    current_budget <= base_budget and  # Only scale if at or below base budget
                                    avg_roas >= min_roas_for_scale and  # Minimum ROAS threshold
                                    total_purchases >= min_purchases_for_scale and  # Minimum purchases
                                    len(performance_data) >= 3  # Minimum data points
                                )
                                
                                if not ready_to_scale and current_budget <= base_budget:
                                    logger.debug(f"Not ready to scale: ROAS={avg_roas:.2f} (need {min_roas_for_scale}), purchases={total_purchases} (need {min_purchases_for_scale})")
                                else:
                                    decision = budget_engine.get_budget_recommendation(
                                        campaign_id=campaign_id,
                                        performance_data=performance_data,
                                        current_budget=current_budget,
                                        strategy=ScalingStrategy.ADAPTIVE,
                                        max_budget=ASC_PLUS_BUDGET_MAX,
                                        min_budget=base_budget,  # Don't go below base budget
                                    )
                                    
                                    # Ensure recommended_budget is also a float (handle both dataclass and dict responses)
                                    try:
                                        if hasattr(decision, 'recommended_budget'):
                                            recommended_budget = float(decision.recommended_budget) if decision.recommended_budget is not None else current_budget
                                        elif isinstance(decision, dict):
                                            recommended_budget = float(decision.get('recommended_budget', current_budget))
                                        else:
                                            recommended_budget = current_budget
                                    except (ValueError, TypeError, AttributeError) as e:
                                        logger.warning(f"Failed to convert recommended_budget to float: {e}, using current_budget")
                                        recommended_budget = current_budget
                                    
                                    # Ensure both are floats before comparison
                                    try:
                                        current_budget_float = float(current_budget)
                                        recommended_budget_float = float(recommended_budget)
                                        
                                        # Only scale up if:
                                        # 1. Confidence is high enough
                                        # 2. Recommended budget is higher than current (scale up only)
                                        # 3. Change is significant (>threshold%)
                                        # 4. At base budget or ready to scale
                                        budget_change_pct = ((recommended_budget_float - current_budget_float) / current_budget_float) * 100
                                        
                                        should_scale = (
                                            decision.confidence >= min_confidence and
                                            recommended_budget_float > current_budget_float and  # Only scale up
                                            budget_change_pct >= scale_up_threshold_pct and  # Significant change
                                            ready_to_scale  # Performance criteria met
                                        )
                                        
                                        if should_scale:
                                            logger.info(f"✅ Ready to scale: €{current_budget_float:.2f} -> €{recommended_budget_float:.2f} ({budget_change_pct:+.1f}%)")
                                            reason = getattr(decision, 'reason', 'performance-based') if hasattr(decision, 'reason') else 'performance-based'
                                            notify(f"🚀 Budget scaling: €{current_budget_float:.2f} -> €{recommended_budget_float:.2f} ({reason}, ROAS: {avg_roas:.2f}, confidence: {decision.confidence:.1%})")
                                            
                                            # Actually update the budget in Meta
                                            try:
                                                client.update_adset_budget(
                                                    adset_id=adset_id,
                                                    daily_budget=recommended_budget_float,
                                                    current_budget=current_budget_float,
                                                )
                                                logger.info(f"✅ Updated adset budget to €{recommended_budget_float:.2f}/day")
                                                notify(f"✅ Budget updated: €{recommended_budget_float:.2f}/day")
                                            except Exception as e:
                                                logger.warning(f"Failed to update budget: {e}")
                                        elif current_budget_float < base_budget:
                                            # If somehow below base budget, restore to base
                                            logger.info(f"Restoring budget to base: €{current_budget_float:.2f} -> €{base_budget:.2f}")
                                            try:
                                                client.update_adset_budget(
                                                    adset_id=adset_id,
                                                    daily_budget=base_budget,
                                                    current_budget=current_budget_float,
                                                )
                                            except Exception as e:
                                                logger.warning(f"Failed to restore base budget: {e}")
                                        else:
                                            logger.debug(f"Keeping budget at €{current_budget_float:.2f} (not ready to scale: ROAS={avg_roas:.2f}, purchases={total_purchases}, confidence={decision.confidence:.2%})")
                                    except (ValueError, TypeError):
                                        logger.warning(f"Budget values not numeric: current={current_budget}, recommended={recommended_budget}")
                except Exception as e:
                    logger.warning(f"Budget scaling error: {e}")
            
            # Smart creative refresh - check for creatives that need refresh
            # Enhanced with predictive fatigue detection and automated refresh execution
            if OPTIMIZATION_SYSTEMS_AVAILABLE:
                try:
                    refresh_manager = create_creative_refresh_manager()
                    
                    # Get historical performance from Supabase for better fatigue prediction
                    from infrastructure.supabase_storage import get_validated_supabase_client
                    supabase_client = get_validated_supabase_client()
                    historical_data = {}
                    
                    if supabase_client:
                        try:
                            # Fetch historical performance for each ad
                            for ad in active_ads:
                                ad_id = ad.get("id")
                                if ad_id:
                                    # Get last 7 days of performance data
                                    result = supabase_client.table("performance_metrics").select(
                                        "spend, impressions, clicks, purchases, roas, ctr, cpa, date_start"
                                    ).eq("ad_id", ad_id).order("date_start", desc=True).limit(7).execute()
                                    
                                    if result.data:
                                        historical_data[ad_id] = result.data
                        except Exception as e:
                            logger.warning(f"Failed to fetch historical performance: {e}")
                    
                    creatives_for_refresh = [
                        {
                            "creative_id": ad.get("creative", {}).get("id") if isinstance(ad.get("creative"), dict) else ad.get("id"),
                            "ad_id": ad.get("id"),
                            "created_at": ad.get("created_time"),
                            "performance": next((i for i in insights if i.get("ad_id") == ad.get("id")), {}),
                            "historical_performance": historical_data.get(ad.get("id"), []),  # Populated from Supabase
                        }
                        for ad in active_ads
                    ]
                    
                    refresh_schedule = refresh_manager.plan_refresh_schedule(
                        creatives=creatives_for_refresh,
                        target_count=target_count,
                    )
                    
                    immediate_refresh_count = refresh_schedule.get("immediate_refresh", 0)
                    staggered_refresh_count = refresh_schedule.get("staggered_refresh", 0)
                    
                    if immediate_refresh_count > 0:
                        logger.info(f"🔄 Creative refresh needed: {immediate_refresh_count} immediate, {staggered_refresh_count} scheduled")
                        notify(f"🔄 Refreshing {immediate_refresh_count} creatives (fatigue detected)")
                        
                        # Execute immediate refreshes
                        for refresh_item in refresh_schedule.get("refresh_needed", [])[:immediate_refresh_count]:
                            creative_id = refresh_item.get("creative_id")
                            ad_id = refresh_item.get("ad_id")
                            reason = refresh_item.get("reason", "fatigue")
                            
                            # Pause the old ad
                            try:
                                client._graph_post(f"{ad_id}", {"status": "PAUSED"})
                                logger.info(f"✅ Paused ad {ad_id} for refresh: {reason}")
                                active_count -= 1
                                killed_count += 1
                            except Exception as e:
                                logger.warning(f"Failed to pause ad {ad_id} for refresh: {e}")
                    
                    # Check for scheduled refreshes due now
                    due_refreshes = refresh_manager.get_scheduled_refreshes_due()
                    if due_refreshes:
                        logger.info(f"🔄 Executing {len(due_refreshes)} scheduled creative refreshes")
                        notify(f"🔄 Executing {len(due_refreshes)} scheduled refreshes")
                        
                        # Execute scheduled refreshes
                        for creative_id in due_refreshes:
                            # Find the ad for this creative
                            for ad in active_ads:
                                ad_creative_id = ad.get("creative", {}).get("id") if isinstance(ad.get("creative"), dict) else ad.get("id")
                                if str(ad_creative_id) == str(creative_id):
                                    try:
                                        client._graph_post(f"{ad.get('id')}", {"status": "PAUSED"})
                                        logger.info(f"✅ Paused ad {ad.get('id')} for scheduled refresh")
                                        active_count -= 1
                                        killed_count += 1
                                        refresh_manager.clear_scheduled_refresh(creative_id)
                                    except Exception as e:
                                        logger.warning(f"Failed to pause ad {ad.get('id')} for scheduled refresh: {e}")
                                    break
                    
                    # Refresh buffer: Pre-generate creatives for upcoming refreshes
                    if refresh_schedule.get("new_creatives_needed", 0) > 0:
                        logger.info(f"📦 Pre-generating {refresh_schedule['new_creatives_needed']} creatives for refresh buffer")
                        
                        # Check if we should pre-generate creatives for the queue
                        if storage_manager and storage_manager.should_pre_generate_creatives(target_count=target_count, buffer_size=3):
                            logger.info(f"📦 Queue buffer low, will pre-generate creatives in background")
                            notify(f"📦 Pre-generating creatives for queue buffer")
                except Exception as e:
                    logger.warning(f"Creative refresh error: {e}")
            
            # Pre-generate creatives for queue if buffer is low
            if storage_manager:
                try:
                    queued_count = storage_manager.get_queued_creative_count()
                    if storage_manager.should_pre_generate_creatives(target_count=target_count, buffer_size=3):
                        logger.info(f"📦 Queue buffer low ({queued_count} queued), pre-generating creatives")
                        # This will be handled in the next tick or background process
                        # For now, just log that we need more queued creatives
                except Exception as e:
                    logger.debug(f"Queue pre-generation check failed: {e}")
            
            # Resource optimization - optimize memory if needed
            if OPTIMIZATION_SYSTEMS_AVAILABLE:
                try:
                    resource_optimizer = create_resource_optimizer()
                    if resource_optimizer.memory_optimizer and resource_optimizer.memory_optimizer.should_optimize_memory():
                        from infrastructure.caching import cache_manager
                        results = resource_optimizer.optimize_all(cache_manager=cache_manager)
                        if results.get("memory", {}).get("freed_mb", 0) > 50:
                            logger.info(f"Memory optimized: freed {results['memory']['freed_mb']:.1f} MB")
                except Exception as e:
                    logger.warning(f"Resource optimization error: {e}")
            
            # Auto-optimization discovery
            if advanced_ml and advanced_ml.auto_optimizer:
                try:
                    creatives_with_perf = [
                        {
                            "ad_id": ad.get("id"),
                            "creative_id": ad.get("creative", {}).get("id") if isinstance(ad.get("creative"), dict) else None,
                            "performance": next(
                                (i for i in insights if i.get("ad_id") == ad.get("id")),
                                {}
                            ),
                        }
                        for ad in active_ads
                    ]
                    
                    opportunities = advanced_ml.auto_optimizer.discover_opportunities(
                        creatives_with_perf,
                        {},
                    )
                    
                    prioritized = advanced_ml.auto_optimizer.prioritize_opportunities(opportunities)
                    
                    # Execute top opportunities
                    for opp in prioritized[:3]:  # Top 3
                        if opp.confidence >= 0.7:
                            advanced_ml.auto_optimizer.execute_optimization(opp, client)
                except Exception as e:
                    logger.error(f"Auto-optimization error: {e}")
        
        return {
            "ok": True,
            "campaign_id": campaign_id,
            "adset_id": adset_id,
            "active_count": final_active_count if 'final_active_count' in locals() else active_count,
            "target_count": target_count,
            "killed_count": killed_count,
            "created_count": created_count,
            "health_status": health_status.value if 'health_status' in locals() else "unknown",
        }
        
    except Exception as e:
        alert_error(f"Error in ASC+ tick: {e}")
        return {"ok": False, "error": str(e)}


__all__ = ["run_asc_plus_tick", "ensure_asc_plus_campaign"]




def run_asc_plus_tick(
    client: MetaClient,
    settings: Dict[str, Any],
    rules: Dict[str, Any],
    store: Any,
    ml_system: Optional[Any] = None,
) -> Dict[str, Any]:
    """Modern ASC+ tick controller with guardrails and creative floor enforcement."""
    validate_asc_plus_config(settings)
    timekit = Timekit()
    target_count = cfg(settings, "asc_plus.target_active_ads") or 10
    max_active = cfg(settings, "asc_plus.max_active_ads") or target_count

    result: Dict[str, Any] = {
        "ok": False,
        "campaign_id": None,
        "adset_id": None,
        "active_count": 0,
        "target_count": target_count,
        "max_active_count": max_active,
        "kills": [],
        "promotions": [],
        "created_ads": [],
        "ad_metrics": {},
        "health": "WARNING",
        "health_message": "",
    }

    _asc_log(logging.INFO, "Starting ASC+ tick")
    campaign_id, adset_id = ensure_asc_plus_campaign(client, settings, store)
    if not campaign_id or not adset_id:
        empty_totals = {"spend": 0.0, "impressions": 0.0, "clicks": 0.0, "add_to_cart": 0.0, "purchases": 0.0}
        result["health_summary"] = _emit_health_notification(
            "WARNING",
            "Failed to ensure campaign/adset",
            empty_totals,
            active_count=0,
            target_count=result["target_count"],
        )
        return result

    result["campaign_id"] = campaign_id
    result["adset_id"] = adset_id

    ads_list = client.list_ads_in_adset(adset_id)

    # Step 1: fetch insights for today + trailing window
    insight_rows = client.get_recent_ad_insights(adset_id=adset_id, campaign_id=campaign_id)
    account_today = timekit.today_ymd_account()
    metrics_map: Dict[str, Dict[str, Any]] = {}
    now_utc = datetime.now(timezone.utc)
    creation_times = _sync_ad_creation_records(client, ads_list, stage="asc_plus")
    for row in insight_rows:
        ad_id = row.get("ad_id")
        if not ad_id:
            continue
        metrics_map[ad_id] = _build_ad_metrics(
            row,
            "asc_plus",
            account_today,
            creation_time=creation_times.get(str(ad_id)) if creation_times else None,
            now_ts=now_utc,
        )

    totals = _summarize_metrics(metrics_map)

    try:
        active_count = client.count_active_ads_in_adset(adset_id, campaign_id=campaign_id)
    except Exception as exc:
        _asc_log(logging.WARNING, "Active ad consensus failed: %s", exc)
        active_count = 0
    result["active_count"] = active_count

    _sync_ad_creation_records(client, ads_list)
    _sync_ad_lifecycle_records(
        client,
        ads_list,
        metrics_map,
        stage="asc_plus",
        campaign_id=campaign_id,
        adset_id=adset_id,
    )
    _sync_performance_metrics_records(
        metrics_map,
        stage="asc_plus",
        date_label=account_today,
    )
    storage_map = _sync_creative_intelligence_records(
        client,
        ads_list,
        metrics_map,
        stage="asc_plus",
    )
    _sync_creative_performance_records(
        ads_list,
        metrics_map,
        stage="asc_plus",
        date_label=account_today,
    )
    _sync_creative_storage_records(
        storage_map,
        ads_list,
        stage="asc_plus",
    )
    _sync_historical_data_records(
        metrics_map,
        stage="asc_plus",
    )
    allowed_statuses = {"ACTIVE", "ELIGIBLE"}
    active_ads = [
        ad
        for ad in ads_list
        if str(ad.get("effective_status") or ad.get("status", "")).upper() in allowed_statuses
    ]

    hydrated_active_ads = []
    for ad in active_ads:
        ad_id = str(ad.get("id") or "")
        if not ad_id:
            continue
        metrics = metrics_map.get(ad_id)
        if metrics and metrics.get("hydrated", True):
            hydrated_active_ads.append(ad)
    hydrated_active_count = len(hydrated_active_ads)
    result["hydrated_active_count"] = hydrated_active_count

    # Step 3: apply guardrail kill/promote rules
    killed_ads: List[Tuple[str, str]] = []
    promoted_ads: List[Tuple[str, str, Optional[str]]] = []
    killed_ids = set()
    for ad in active_ads:
        ad_id = ad.get("id")
        if not ad_id:
            continue
        metrics = metrics_map.get(ad_id) or _build_ad_metrics(
            {"ad_id": ad_id, "spend": 0, "impressions": 0, "clicks": 0},
            "asc_plus",
            account_today,
            creation_time=creation_times.get(str(ad_id)) if creation_times else None,
            now_ts=now_utc,
        )
        kill, kill_reason = _guardrail_kill(metrics)
        if kill:
            if _pause_ad(client, ad_id, kill_reason):
                killed_ads.append((ad_id, kill_reason))
                killed_ids.add(ad_id)
                active_count = max(0, active_count - 1)
                if metrics.get("hydrated", True):
                    hydrated_active_count = max(0, hydrated_active_count - 1)
                _record_lifecycle_event(ad_id, "PAUSED", kill_reason)
            continue

    # Refresh active count after guardrails
    try:
        active_count = client.count_active_ads_in_adset(adset_id, campaign_id=campaign_id)
    except Exception:
        pass
    if killed_ids:
        hydrated_active_ads = [
            ad for ad in hydrated_active_ads if str(ad.get("id")) not in killed_ids
        ]
        hydrated_active_count = len(hydrated_active_ads)

    planning_active_count = active_count

    # Step 4: enforce max active ceiling by trimming lowest performers
    excess = max(0, planning_active_count - max_active)
    if excess > 0:
        cap_pool = hydrated_active_ads if hydrated_active_ads else active_ads
        cap_candidates = _select_ads_for_cap(cap_pool, metrics_map, list(killed_ids), excess * 2)
        trimmed: List[Tuple[str, str]] = []
        for _, candidate_ad_id, candidate_metrics in cap_candidates:
            if len(trimmed) >= excess:
                break
            if candidate_ad_id in killed_ids:
                continue
            reason = f"Capped to maintain {max_active} live ads (score { _ad_health_score(candidate_metrics):.2f})"
            if _pause_ad(client, candidate_ad_id, reason):
                trimmed.append((candidate_ad_id, reason))
                killed_ids.add(candidate_ad_id)
                active_count = max(0, active_count - 1)
                planning_active_count = max(0, planning_active_count - 1)
                if candidate_metrics.get("hydrated", True):
                    hydrated_active_count = max(0, hydrated_active_count - 1)
                _record_lifecycle_event(candidate_ad_id, "PAUSED", reason)
        if trimmed:
            killed_ads.extend(trimmed)
            trimmed_ids = {ad_id for ad_id, _ in trimmed}
            hydrated_active_ads = [
                ad for ad in hydrated_active_ads if str(ad.get("id")) not in trimmed_ids
            ]
            hydrated_active_count = len(hydrated_active_ads)
        try:
            active_count = client.count_active_ads_in_adset(adset_id, campaign_id=campaign_id)
        except Exception:
            pass
        planning_active_count = active_count

    effective_target = min(target_count, max_active)
    available_slots = max(0, max_active - min(active_count, max_active))
    desired_slots = max(0, effective_target - planning_active_count)
    deficit = min(desired_slots, available_slots)
    created_records = _generate_creatives_for_deficit(deficit, client, settings, campaign_id, adset_id, ml_system, active_count)
    if created_records:
        try:
            active_count = client.count_active_ads_in_adset(adset_id, campaign_id=campaign_id)
        except Exception:
            active_count += len(created_records)
        for record in created_records:
            new_ad_id = record.get("ad_id")
            if not new_ad_id:
                continue
            metrics_map[new_ad_id] = _build_ad_metrics(
                {"ad_id": new_ad_id, "spend": 0, "impressions": 0, "clicks": 0},
                "asc_plus",
                account_today,
                creation_time=creation_times.get(str(new_ad_id)) if creation_times else None,
                now_ts=now_utc,
            )
    else:
        created_records = []

    result["active_count"] = active_count
    result["caps_enforced"] = bool(result["active_count"] == result["target_count"])
    result["kills"] = killed_ads
    result["promotions"] = promoted_ads
    result["created_ads"] = [record["ad_id"] for record in created_records if record.get("ad_id")]
    result["created_details"] = created_records
    result["ad_metrics"] = metrics_map

    # Slack actions summary
    action_lines: List[str] = []
    if killed_ads:
        action_lines.append(f"🛑 Paused {len(killed_ads)} ads:")
        for ad_id, reason in killed_ads[:10]:
            action_lines.append(f"   • {_short_id(ad_id)} – {reason}")
        if len(killed_ads) > 10:
            action_lines.append(f"   • … {len(killed_ads) - 10} more")
    if promoted_ads:
        action_lines.append(f"🚀 Promoted {len(promoted_ads)} ads:")
        for src_id, dst_id, reason in promoted_ads[:5]:
            action_lines.append(f"   • {_short_id(src_id)} → {_short_id(dst_id)} – {reason or 'Promotion rule'}")
        if len(promoted_ads) > 5:
            action_lines.append(f"   • … {len(promoted_ads) - 5} more")
    if created_records:
        action_lines.append(f"✨ Created {len(created_records)} ads (target {effective_target}):")
        for record in created_records[:5]:
            action_lines.append(f"   • {_short_id(record.get('ad_id'))} – via {record.get('source')}")
        if len(created_records) > 5:
            action_lines.append(f"   • … {len(created_records) - 5} more")
    if action_lines:
        action_lines.append(f"📊 Active now: {active_count}/{effective_target} (max {max_active})")
        notify("ASC+ actions update\n" + "\n".join(action_lines))

    health_status, health_message = _evaluate_health(active_count, effective_target, metrics_map)
    result["health"] = health_status
    result["health_message"] = health_message

    result["hydrated_active_count"] = hydrated_active_count
    health_summary = _emit_health_notification(
        health_status,
        health_message,
        totals,
        active_count,
        result["target_count"],
    )

    result["ok"] = True
    result["health_summary"] = health_summary
    return result
