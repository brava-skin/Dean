from __future__ import annotations

import logging
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
import math

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def _asc_log(level: int, message: str, *args: Any) -> None:
    logger.log(level, f"[ASC] {message}", *args)

from integrations.slack import notify, alert_error
from integrations import fmt_eur, fmt_int, fmt_pct
from integrations.meta_client import MetaClient
from infrastructure.utils import (
    cfg, cfg_or_env_f,
    safe_f, today_str, Timekit
)
from creative.image_generator import create_image_generator, ImageCreativeGenerator
from config import (
    ASC_PLUS_BUDGET_MIN,
    ASC_PLUS_BUDGET_MAX,
    ASC_PLUS_MIN_BUDGET_PER_CREATIVE,
    CREATIVE_PERFORMANCE_STAGE_VALUE,
    validate_asc_plus_config
)
from infrastructure.data_validation import (
    validate_all_timestamps,
    validate_supabase_data,
    ValidationError,
    validate_and_sanitize_data,
)

UTC = timezone.utc
LOCAL_TZ = ZoneInfo(os.getenv("ACCOUNT_TZ", os.getenv("ACCOUNT_TIMEZONE", "Europe/Amsterdam")))
ACCOUNT_CURRENCY = os.getenv("ACCOUNT_CURRENCY", "EUR")
HYDRATION_SECONDS = int(os.getenv("ASC_PLUS_HYDRATION_SECONDS", "180"))


_CREATIVE_PERFORMANCE_STAGE_DISABLED = False


def _roas(row: Dict[str, Any]) -> float:
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
        if t == "omni_purchase":
            purch += int(v)
        elif t == "purchase":
            if purch == 0:
                purch += int(v)
        elif t == "omni_add_to_cart":
            atc += int(v)
        elif t == "add_to_cart":
            if atc == 0:
                atc += int(v)
    return purch, atc


def _fraction_or_none(value: Any, *, allow_percent: bool = True) -> Optional[float]:
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


def _calculate_creative_performance(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
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
        response = supabase_client.table("ads").select(
            "ad_id, created_at, lifecycle_id"
        ).in_("ad_id", ad_ids).execute()
        data = getattr(response, "data", None) or []
        existing_records = {
            str(row.get("ad_id")): row for row in data if row and row.get("ad_id")
        }
        for ad_id, row in existing_records.items():
            created_at = _parse_created_time(row.get("created_at"))
            if created_at:
                creation_map[ad_id] = created_at
    except Exception as exc:
        logger.debug(f"Unable to load existing ads records: {exc}")
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
    rules: Optional[Dict[str, Any]] = None,
) -> None:
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
            supabase_client.table("ads")
            .select(
                "ad_id, creative_id, campaign_id, adset_id, status, lifecycle_id, metadata, "
                "kill_reason, created_at, killed_at"
            )
            .in_("ad_id", ad_ids)
            .execute()
        )
        data = getattr(response, "data", None) or []
        existing_records = {
            str(row.get("ad_id")): row for row in data if row and row.get("ad_id")
        }
    except Exception as exc:
        logger.debug(f"Unable to load existing ads records: {exc}")
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
        if status in ["ACTIVE", "PAUSED"]:
            status = status.lower()
        elif status in ["DELETED", "ARCHIVED", "DISAPPROVED"]:
            status = "killed"

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

        kill_reason = existing.get("kill_reason")
        if (status == "killed" or status == "paused") and not kill_reason:
            metrics = metrics_map.get(ad_id)
            if metrics:
                rules = rules or {}
                asc_rules = rules.get("asc_plus_atc", {})
                kill_rules = asc_rules.get("kill", [])
                
                ctr = safe_f(metrics.get("ctr", 0))
                cpc = safe_f(metrics.get("cpc", 0))
                cpm = safe_f(metrics.get("cpm", 0))
                atc = safe_f(metrics.get("add_to_cart", 0))
                
                for rule in kill_rules:
                    rule_type = rule.get("type", "")
                    if rule_type == "ctr_spend_floor" and ctr < safe_f(rule.get("ctr_lt", 0.008)):
                        kill_reason = "low_ctr"
                        break
                    elif rule_type == "cpc_spend_combo" and cpc > safe_f(rule.get("cpc_gt", 1.90)):
                        kill_reason = "high_cpc"
                        break
                    elif rule_type == "cpm_ctr_combo" and cpm > safe_f(rule.get("cpm_gt", 80)):
                        kill_reason = "high_cpm"
                        break
                    elif rule_type == "atc_efficiency_fail" and atc < safe_f(rule.get("atc_lt", 1)):
                        kill_reason = "low_atc"
                        break
                
                if not kill_reason:
                    kill_reason = "performance_threshold"
            else:
                kill_reason = "manual_or_meta_status"

        killed_at = None
        if status == "killed" and not existing.get("killed_at"):
            killed_at = now.isoformat()
        elif existing.get("killed_at"):
            killed_at = existing.get("killed_at")

        ads_record: Dict[str, Any] = {
            "ad_id": ad_id,
            "creative_id": creative_id,
            "campaign_id": campaign_value,
            "adset_id": adset_value,
            "status": status,
            "lifecycle_id": lifecycle_id,
            "metadata": metadata_existing,
            "kill_reason": kill_reason,
            "created_at": created_at.isoformat(),
            "killed_at": killed_at,
            "updated_at": now.isoformat(),
        }

        ads_record = validate_all_timestamps(ads_record)

        try:
            supabase_client.table("ads").upsert(
                ads_record,
                on_conflict="ad_id",
            ).execute()
        except Exception as exc:
            logger.debug(f"Failed to upsert ads record for {ad_id}: {exc}")


def _collect_storage_metadata(
    supabase_client: Any,
    creative_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    ids = [cid for cid in set(creative_ids) if cid]
    if not ids:
        return {}
    try:
        response = (
            supabase_client.table("ads")
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
            supabase_client.table("ads")
            .select("ad_id, creative_id, headline, primary_text, description, image_prompt, performance_score, fatigue_index, metadata")
            .in_("ad_id", ad_ids)
            .execute()
        )
        existing_rows = getattr(existing_resp, "data", None) or []
    except Exception as exc:
        logger.debug(f"Unable to load ads records: {exc}")
        existing_rows = []

    existing_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    storage_ids: set[str] = set()
    ads_needing_creative_content: List[str] = []
    
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
            
            if not row.get("headline") and not row.get("primary_text"):
                ads_needing_creative_content.append(ad_id)

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
            if storage_url and "placeholder" in storage_url.lower():
                storage_url = None
        else:
            file_size_mb = None
            storage_url = existing.get("storage_url") or existing.get("supabase_storage_url")
            if storage_url and "placeholder" in storage_url.lower():
                storage_url = None

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
        headline = existing.get("headline") or metadata.get("ad_copy", {}).get("headline") or ""
        if not headline and ad.get("name"):
            ad_name = ad.get("name", "")
            if ad_name.startswith("[ASC+]"):
                headline = ad_name.replace("[ASC+]", "").strip().split(" - ")[0] or ""
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

        file_size_bytes_val = None
        if storage_info and storage_info.get("file_size_bytes"):
            file_size_bytes_val = int(storage_info.get("file_size_bytes"))
        elif existing.get("file_size_bytes"):
            file_size_bytes_val = int(existing.get("file_size_bytes"))
        
        storage_path_val = None
        if storage_info and storage_info.get("storage_path"):
            storage_path_val = storage_info.get("storage_path")
        elif existing.get("storage_path"):
            storage_path_val = existing.get("storage_path")
        elif storage_url:
            storage_path_val = storage_url.split("/")[-1] if "/" in storage_url else f"creative_{creative_id}.jpg"
        
        if not storage_url:
            storage_url = existing.get("storage_url") or ""
        
        record: Dict[str, Any] = {
            "creative_id": creative_id,
            "ad_id": ad_id,
            "creative_type": existing.get("creative_type") or "image",
            "aspect_ratio": "1:1",
            "file_size_mb": file_size_mb if file_size_mb is not None else _safe_float(existing.get("file_size_mb")),
            "file_size_bytes": file_size_bytes_val if file_size_bytes_val is not None else (existing.get("file_size_bytes") or 0),
            "storage_path": storage_path_val or existing.get("storage_path") or f"creative_{creative_id}.jpg",
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
            "storage_url": storage_url or "",
            "image_prompt": existing.get("image_prompt"),
            "text_overlay_content": existing.get("text_overlay_content"),
            "created_at": created_at,
            "updated_at": now_iso,
        }

        record = validate_all_timestamps(record)

        ads_update: Dict[str, Any] = {
            "ad_id": ad_id,
            "creative_id": creative_id,
            "headline": headline,
            "primary_text": primary_text,
            "description": description,
            "image_prompt": existing.get("image_prompt"),
            "performance_score": performance_score,
            "fatigue_index": fatigue_index,
            "metadata": metadata,
            "updated_at": now_iso,
        }

        validation = validate_supabase_data("ads", ads_update, strict_mode=True)
        if not validation.is_valid:
            logger.warning(
                "Skipping ads sync for creative %s (%s): %s",
                creative_id,
                ad_id,
                "; ".join(validation.errors),
            )
            continue

        sanitized_records.append(validation.sanitized_data)

    if not sanitized_records:
        return storage_map

    try:
        supabase_client.table("ads").upsert(
            sanitized_records,
            on_conflict="ad_id",
        ).execute()
    except ValidationError as exc:
        logger.error("Ads upsert blocked: %s", exc)
    except Exception as exc:
        logger.debug(f"Failed to upsert ads records: {exc}")
    
    if ads_needing_creative_content and client:
        _fetch_and_update_creative_content(client, supabase_client, ads_needing_creative_content)

    return storage_map


def _fetch_and_update_creative_content(
    client: MetaClient,
    supabase_client: Any,
    ad_ids: List[str],
) -> None:
    if not ad_ids or not client or not supabase_client:
        return
    
    try:
        from facebook_business.adobjects.ad import Ad
        from facebook_business.adobjects.adcreative import AdCreative
        
        USE_SDK = False
        try:
            from facebook_business.api import FacebookAdsApi
            USE_SDK = True
        except ImportError:
            pass
        
        if not USE_SDK:
            return
        
        client._init_sdk_if_needed()
        
        updates: List[Dict[str, Any]] = []
        
        for ad_id in ad_ids[:20]:
            try:
                ad_obj = Ad(ad_id)
                ad_data = ad_obj.api_get(fields=["id", "creative", "name"])
                
                creative_id = ad_data.get("creative", {}).get("id") if isinstance(ad_data.get("creative"), dict) else None
                if not creative_id:
                    creative_id = ad_data.get("creative")
                
                if not creative_id:
                    continue
                
                creative_obj = AdCreative(str(creative_id))
                creative_data = creative_obj.api_get(fields=["object_story_spec", "name", "body", "title", "link_url", "image_url"])
                
                headline = ""
                primary_text = ""
                description = ""
                
                oss = creative_data.get("object_story_spec", {})
                if oss:
                    link_data = oss.get("link_data", {})
                    if link_data:
                        headline = link_data.get("name", "") or link_data.get("headline", "")
                        primary_text = link_data.get("message", "") or link_data.get("description", "")
                        description = link_data.get("description", "") or link_data.get("caption", "")
                
                if not headline:
                    headline = creative_data.get("title", "") or creative_data.get("name", "")
                if not primary_text:
                    primary_text = creative_data.get("body", "")
                
                if headline or primary_text or description:
                    updates.append({
                        "ad_id": ad_id,
                        "headline": headline[:200] if headline else "",
                        "primary_text": primary_text[:1000] if primary_text else "",
                        "description": description[:500] if description else "",
                    })
                    logger.debug(f"Fetched creative content for ad {ad_id}: headline={headline[:50] if headline else 'N/A'}...")
                
            except Exception as e:
                logger.debug(f"Failed to fetch creative content for ad {ad_id}: {e}")
                continue
        
        if updates:
            try:
                supabase_client.table("ads").upsert(
                    updates,
                    on_conflict="ad_id",
                ).execute()
                logger.info(f"Updated creative content for {len(updates)} ads")
            except Exception as e:
                logger.warning(f"Failed to update creative content in ads table: {e}")
    
    except ImportError:
        logger.debug("Facebook SDK not available for fetching creative content")
    except Exception as e:
        logger.debug(f"Error fetching creative content: {e}")


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
            "ad_id": ad_id,
            "creative_id": creative_id,
            "status": storage_status,
            "metadata": metadata,
            "updated_at": now_iso,
        }

        updates.append(update_record)

    if not updates:
        return

    try:
        supabase_client.table("ads").upsert(
            updates,
            on_conflict="ad_id",
        ).execute()
    except Exception as exc:
        logger.debug(f"Failed to upsert ads records: {exc}")


def _sync_performance_metrics_records(
    metrics_map: Dict[str, Dict[str, Any]],
    stage: str,
    date_label: str,
    active_ad_ids: Optional[List[str]] = None,
) -> None:
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
    
    processed_ad_ids = set()

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
        
        if impressions > 0 and clicks > impressions:
            _asc_log(
                logging.WARNING,
                "Data quality issue for ad %s: %d clicks from %d impressions (impossible). Capping clicks to impressions.",
                ad_id,
                clicks,
                impressions,
            )
            clicks = impressions

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

        record = {
            "ad_id": ad_id,
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
            "atc_rate": atc_rate,
            "purchase_rate": purchase_rate,
        }

        upserts.append(record)
        processed_ad_ids.add(ad_id)
    
    if active_ad_ids:
        for ad_id in active_ad_ids:
            ad_id_str = str(ad_id)
            if ad_id_str in processed_ad_ids or not ad_id_str:
                continue
            
            try:
                existing_check = supabase_client.table("ads").select("ad_id").eq("ad_id", ad_id_str).limit(1).execute()
                if not existing_check.data:
                    continue
            except Exception:
                continue
            
            zero_record = {
                "ad_id": ad_id_str,
                "window_type": "1d",
                "date_start": date_label,
                "date_end": date_label,
                "impressions": 0,
                "clicks": 0,
                "spend": 0.0,
                "purchases": 0,
                "add_to_cart": 0,
                "initiate_checkout": 0,
                "ctr": None,
                "cpc": None,
                "cpm": None,
                "roas": None,
                "cpa": None,
                "atc_rate": None,
                "purchase_rate": None,
            }
            upserts.append(zero_record)

    if not upserts:
        return

    try:
        supabase_client.table("performance_metrics").upsert(
            upserts,
            on_conflict="ad_id,window_type,date_start",
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


def _guardrail_kill(metrics: Dict[str, Any], rules: Optional[Dict[str, Any]] = None, rule_engine: Optional[Any] = None) -> Tuple[bool, str]:
    rules = rules or {}
    asc_rules = rules.get("asc_plus_atc", {})
    engine_rules = asc_rules.get("engine", {})
    fairness_rules = engine_rules.get("fairness", {})
    kill_rules = asc_rules.get("kill", [])
    minimums = asc_rules.get("minimums", {})
    
    min_spend_before_kill = safe_f(fairness_rules.get("min_spend_before_kill_eur", 55))
    min_runtime_hours = safe_f(fairness_rules.get("min_runtime_hours", 48))
    min_impressions = safe_f(minimums.get("min_impressions", 2500))
    
    spend = safe_f(metrics.get("spend"))
    stage_hours = safe_f(metrics.get("stage_duration_hours"))
    ad_age_days = safe_f(metrics.get("ad_age_days"))
    derived_age_days = max(ad_age_days, (stage_hours / 24.0) if stage_hours else 0.0)
    derived_runtime_hours = derived_age_days * 24.0

    if spend < min_spend_before_kill or derived_runtime_hours < min_runtime_hours:
        return False, ""

    if rule_engine and kill_rules:
        try:
            from analytics.metrics import metrics_from_row, MetricsConfig
            metrics_cfg = MetricsConfig(
                prefer_roas_field=False,
                account_currency="EUR",
                product_currency="EUR",
            )
            m = metrics_from_row(metrics, cfg=metrics_cfg)
            row = {"ad_id": metrics.get("ad_id", "")}
            
            for rule in kill_rules:
                ok, reason = rule_engine._eval(rule, m, row)
                if ok:
                    return True, reason
        except Exception as exc:
            _asc_log(logging.WARNING, "Failed to evaluate kill rules: %s", exc)

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
    candidates.sort(key=lambda x: x[0])
    return candidates[:limit]


def _enforce_hard_cap(
    client: MetaClient,
    campaign_id: str,
    adset_id: str,
    metrics_map: Dict[str, Dict[str, Any]],
    creation_lookup: Dict[str, datetime],
    max_active: int,
    killed_ids: Set[str],
    allowed_statuses: Set[str],
) -> Tuple[int, List[Dict[str, Any]], int, List[Tuple[str, str]], Set[str]]:
    killed_log: List[Tuple[str, str]] = []
    safety_counter = 0

    def _active_ads_snapshot() -> List[Dict[str, Any]]:
        try:
            refreshed_ads = client.list_ads_in_adset(adset_id)
        except Exception:
            refreshed_ads = []
        return [
            ad
            for ad in refreshed_ads
            if str(ad.get("effective_status") or ad.get("status", "")).upper() in allowed_statuses
        ]

    active_ads = _active_ads_snapshot()
    try:
        active_count = client.count_active_ads_in_adset(adset_id, campaign_id=campaign_id)
    except Exception:
        active_count = len(active_ads)

    while active_count > max_active and safety_counter < 5:
        overflow = active_count - max_active
        cap_candidates = _select_ads_for_cap(
            active_ads,
            metrics_map,
            list(killed_ids),
            max(overflow * 3, overflow + 2),
        )
        if not cap_candidates:
            break
        paused_this_round = False
        for _, candidate_ad_id, candidate_metrics in cap_candidates:
            if active_count <= max_active:
                break
            if candidate_ad_id in killed_ids:
                continue
            if not candidate_metrics:
                candidate_metrics = _build_ad_metrics(
                    {"ad_id": candidate_ad_id, "spend": 0, "impressions": 0, "clicks": 0},
                    "asc_plus",
                    today_str(),
                    creation_time=creation_lookup.get(str(candidate_ad_id)),
                    now_ts=datetime.now(timezone.utc),
                )
                metrics_map[candidate_ad_id] = candidate_metrics
            reason = (
                f"Hard cap enforced to limit active ads to {max_active} "
                f"(score {_ad_health_score(candidate_metrics):.2f})"
            )
            if _pause_ad(client, candidate_ad_id, reason):
                killed_ids.add(candidate_ad_id)
                killed_log.append((candidate_ad_id, reason))
                paused_this_round = True
        if not paused_this_round:
            break
        safety_counter += 1
        active_ads = _active_ads_snapshot()
        try:
            active_count = client.count_active_ads_in_adset(adset_id, campaign_id=campaign_id)
        except Exception:
            active_count = len(active_ads)

    hydrated_candidates = []
    hydrated_count = 0
    for ad in active_ads:
        ad_id = str(ad.get("id") or "")
        metrics = metrics_map.get(ad_id)
        if metrics and metrics.get("hydrated", True):
            hydrated_candidates.append(ad)
            hydrated_count += 1

    return active_count, hydrated_candidates, hydrated_count, killed_log, killed_ids


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
        if action_type == "omni_add_to_cart":
            add_to_cart += int(value)
        elif action_type == "add_to_cart":
            if add_to_cart == 0:
                add_to_cart = int(value)
        elif action_type == "omni_initiate_checkout":
            initiate_checkout += int(value)
        elif action_type == "initiate_checkout":
            if initiate_checkout == 0:
                initiate_checkout = int(value)
        elif action_type == "omni_purchase":
            purchases += int(value)
        elif action_type == "purchase":
            if purchases == 0:
                purchases += int(value)

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
    if raw_cpc is None and clicks_val > 0:
        raw_cpc = spend_val / clicks_val if clicks_val > 0 else None
    if raw_cpc is None:
        raw_cpc = _float_or_none(row.get("cost_per_inline_link_click"))
    cpc = round(raw_cpc, 2) if raw_cpc is not None else None

    raw_cpm = _float_or_none(row.get("cpm"))
    if raw_cpm is None and impressions_val > 0:
        raw_cpm = spend_val * 1000.0 / impressions_val
    cpm = round(raw_cpm, 2) if raw_cpm is not None else None

    cpa = (spend_val / purchases) if purchases > 0 else None
    
    cost_per_atc = (spend_val / add_to_cart) if add_to_cart > 0 else None

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
        "cost_per_atc": cost_per_atc,
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


def _summarize_today_metrics(
    client: MetaClient,
    adset_id: str,
    campaign_id: str,
    account_today: str,
    active_ad_ids_set: Set[str],
) -> Dict[str, float]:
    totals = {
        "spend": 0.0,
        "impressions": 0.0,
        "clicks": 0.0,
        "add_to_cart": 0.0,
        "initiate_checkout": 0.0,
        "purchases": 0.0,
    }
    try:
        filters = []
        if adset_id:
            filters.append({"field": "adset.id", "operator": "EQUAL", "value": adset_id})
        if campaign_id:
            filters.append({"field": "campaign.id", "operator": "EQUAL", "value": campaign_id})
        
        today_rows = client.get_ad_insights(
            level="ad",
            date_preset="today",
            filtering=filters,
            fields=["ad_id", "spend", "impressions", "clicks", "actions"],
            limit=500,
        ) or []
        
        for row in today_rows:
            ad_id = str(row.get("ad_id", ""))
            if ad_id not in active_ad_ids_set:
                continue
            
            spend_val = safe_f(row.get("spend"))
            impressions_val = safe_f(row.get("impressions"))
            clicks_val = safe_f(row.get("clicks"))
            
            actions = row.get("actions") or []
            add_to_cart = 0
            initiate_checkout = 0
            purchases = 0
            for action in actions:
                action_type = action.get("action_type")
                value = safe_f(action.get("value"))
                if action_type == "omni_add_to_cart":
                    add_to_cart += int(value)
                elif action_type == "add_to_cart":
                    if add_to_cart == 0:
                        add_to_cart = int(value)
                elif action_type == "omni_initiate_checkout":
                    initiate_checkout += int(value)
                elif action_type == "initiate_checkout":
                    if initiate_checkout == 0:
                        initiate_checkout = int(value)
                elif action_type == "omni_purchase":
                    purchases += int(value)
                elif action_type == "purchase":
                    if purchases == 0:
                        purchases += int(value)
            
            totals["spend"] += spend_val
            totals["impressions"] += impressions_val
            totals["clicks"] += clicks_val
            totals["add_to_cart"] += add_to_cart
            totals["initiate_checkout"] += initiate_checkout
            totals["purchases"] += purchases
    except Exception as exc:
        _asc_log(logging.WARNING, "Failed to fetch today's metrics for health summary: %s", exc)
    
    return totals


def _ad_health_score(metrics: Dict[str, Any]) -> float:
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
    
    spend = totals.get('spend', 0.0)
    atc = totals.get('add_to_cart', 0.0)
    cost_per_atc = (spend / atc) if atc > 0 else None
    
    purchases = totals.get('purchases', 0.0)
    cpa = (spend / purchases) if purchases > 0 else None
    
    impressions = totals.get('impressions', 0.0)
    clicks = totals.get('clicks', 0.0)
    ctr = (clicks / impressions * 100) if impressions > 0 else None
    cpc = (spend / clicks) if clicks > 0 else None
    cpm = (spend / impressions * 1000) if impressions > 0 else None
    
    summary_lines = [
        f"ASC+ {tick_time} CET · Health {status}",
        (
            f"Active {active_count}/{target_count} | "
            f"Spend {fmt_eur(spend)} | "
            f"IMP {fmt_int(impressions)} | "
            f"Clicks {fmt_int(clicks)} | "
            f"ATC {fmt_int(atc)} | "
            f"PUR {fmt_int(purchases)}"
        ),
        (
            f"CTR {fmt_pct(ctr / 100) if ctr is not None else '-'} | "
            f"CPC {fmt_eur(cpc)} | "
            f"CPM {fmt_eur(cpm)} | "
            f"CPA {fmt_eur(cpa)} | "
            f"Cost/ATC {fmt_eur(cost_per_atc)}"
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
        from infrastructure.supabase_storage import get_validated_supabase_client, create_creative_storage_manager

        supabase_client = get_validated_supabase_client()
        if supabase_client:
            storage_manager = create_creative_storage_manager(supabase_client)
    except Exception as exc:
        _asc_log(logging.DEBUG, "Creative storage not available: %s", exc)

    image_generator = create_image_generator(
        flux_api_key=os.getenv("FLUX_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        ml_system=None,
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
            "status": lifecycle_status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        if lifecycle_status == "killed":
            payload["kill_reason"] = reason
            payload["killed_at"] = datetime.now(timezone.utc).isoformat()
        elif lifecycle_status == "paused" and reason:
            payload["kill_reason"] = reason
        
        client.table("ads").upsert(payload, on_conflict="ad_id").execute()
    except Exception as exc:
        _asc_log(logging.DEBUG, "Lifecycle logging failed for %s: %s", ad_id, exc)


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
    if not client or not adset_id or not creative_data:
        logger.error("Invalid input: client, adset_id, and creative_data are required")
        return None, None, False
    
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
        ad_copy_dict = creative_data.get("ad_copy") or {}
        if not isinstance(ad_copy_dict, dict):
            ad_copy_dict = {}
        
        headline = ad_copy_dict.get("headline", "")
        if headline:
            headline_words = headline.split()[:3]
            headline_snippet = " ".join(headline_words)
            if len(headline_snippet) > 25:
                headline_snippet = headline_snippet[:22] + "..."
            headline_snippet = headline_snippet.replace(":", "").replace("/", "").replace("\\", "").strip()
        else:
            headline_snippet = "Creative"
        
        seq_num = active_count + created_count + 1
        
        creative_name = f"[ASC+] {headline_snippet} - #{seq_num}"
        
        if len(creative_name) > 80:
            creative_name = f"[ASC+] {headline_snippet[:15]} - #{seq_num}"
        
        supabase_storage_url = creative_data.get("supabase_storage_url")
        image_path = creative_data.get("image_path")
        
        if not supabase_storage_url and not image_path:
            logger.error("Creative data must have either supabase_storage_url or image_path")
            return None, None, False
        
            ad_copy_dict = {}
        
        page_id = os.getenv("FB_PAGE_ID")
        if not page_id:
            logger.error("FB_PAGE_ID environment variable is required")
            return None, None, False
        
        env_instagram_actor_id = os.getenv("IG_ACTOR_ID")
        instagram_actor_id = env_instagram_actor_id or client.get_instagram_actor_id(page_id)
        
        if instagram_actor_id:
            try:
                int(instagram_actor_id)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid Instagram actor ID format: {instagram_actor_id}. Will skip Instagram placements."
                )
                instagram_actor_id = None
        
        if not instagram_actor_id:
            logger.debug(
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
            primary_text = ad_copy_dict.get("primary_text", "")
            if primary_text:
                import re
                primary_text = primary_text.replace("Brava Product", "").replace("—", ",").replace("–", ",").strip()
                primary_text = re.sub(r'\s+', ' ', primary_text).strip()
                if len(primary_text) > 150:
                    primary_text = primary_text[:147] + "..."
            
            creative = client.create_image_creative(
                page_id=page_id,
                name=creative_name,
                supabase_storage_url=supabase_storage_url,
                image_path=image_path if not supabase_storage_url else None,
                primary_text=primary_text,
                headline=ad_copy_dict.get("headline", ""),
                description=ad_copy_dict.get("description", ""),
                call_to_action="SHOP_NOW",
                instagram_actor_id=instagram_actor_id,
                creative_id=storage_creative_id,
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
        
        if str(meta_creative_id) in existing_creative_ids:
            logger.debug(f"Skipping duplicate creative: {meta_creative_id}")
            return str(meta_creative_id), None, False
        
        headline = ad_copy_dict.get("headline", "")
        if headline:
            headline_words = headline.split()[:4]
            headline_snippet = " ".join(headline_words)
            if len(headline_snippet) > 30:
                headline_snippet = headline_snippet[:27] + "..."
            headline_snippet = headline_snippet.replace(":", "").replace("/", "").replace("\\", "").strip()
        else:
            headline_snippet = "Creative"
        
        from datetime import datetime
        date_str = datetime.now().strftime("%y%m%d")
        
        seq_num = active_count + created_count + 1
        
        ad_name = f"[ASC+] {headline_snippet} - {date_str} - #{seq_num}"
        
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
            creative_id_str = str(meta_creative_id).strip()
            if not creative_id_str or not creative_id_str.isdigit():
                logger.error(f"Invalid creative_id format: {meta_creative_id} (expected numeric string)")
                return None, None, False
            
            ad = client.create_ad(
                adset_id=adset_id,
                name=ad_name,
                creative_id=creative_id_str,
                status="ACTIVE",
                instagram_actor_id=resolved_instagram_actor_id,
                use_env_instagram_actor=bool(resolved_instagram_actor_id),
                tracking_specs=None,
            )
            logger.info(f"Meta API create_ad response: {ad}")
        except ValueError as ve:
            logger.error(f"Validation error creating ad: {ve}")
            return None, None, False
        except RuntimeError as re:
            error_str = str(re)
            if "500" in error_str or "unknown error" in error_str.lower():
                logger.warning(f"Meta API 500 error (may be transient): {re}")
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
            
            try:
                from infrastructure.supabase_storage import get_validated_supabase_client, SupabaseStorage
                supabase_client = get_validated_supabase_client()
                if supabase_client:
                    try:
                        storage = SupabaseStorage(supabase_client)
                        lifecycle_id = f"lifecycle_{ad_id}"
                        storage.record_ad_creation(ad_id, lifecycle_id, "asc_plus")
                        logger.debug(f"✅ Recorded ad creation time for {ad_id}")
                    except Exception as e:
                        logger.debug(f"Failed to record ad creation time: {e}")
                    
                    try:
                        storage = SupabaseStorage(supabase_client)
                        lifecycle_id = f"lifecycle_{ad_id}"
                        storage.store_historical_data(ad_id, lifecycle_id, "asc_plus", "spend", 0.0)
                        storage.store_historical_data(ad_id, lifecycle_id, "asc_plus", "impressions", 0.0)
                        storage.store_historical_data(ad_id, lifecycle_id, "asc_plus", "clicks", 0.0)
                        storage.store_historical_data(ad_id, lifecycle_id, "asc_plus", "purchases", 0.0)
                        logger.debug(f"✅ Stored initial historical data for {ad_id}")
                    except Exception as e:
                        logger.debug(f"Failed to store initial historical data: {e}")
                    ad_copy_dict = creative_data.get("ad_copy") or {}
                    if not isinstance(ad_copy_dict, dict):
                        ad_copy_dict = {}
                    
                    now_iso = datetime.now(timezone.utc).isoformat()
                    supabase_storage_url = creative_data.get("supabase_storage_url", "")
                    if supabase_storage_url:
                        storage_path = creative_data.get("storage_path", supabase_storage_url.split("/")[-1] if "/" in supabase_storage_url else f"creative_{meta_creative_id}.jpg")
                    else:
                        storage_path = creative_data.get("storage_path", f"creative_{meta_creative_id}.jpg")
                    file_size_bytes = creative_data.get("file_size_bytes", 0)
                    if not file_size_bytes:
                        try:
                            image_path = creative_data.get("image_path")
                            if image_path and os.path.exists(image_path):
                                file_size_bytes = os.path.getsize(image_path)
                        except Exception:
                            pass
                    
                    creative_intel_data = {
                        "creative_id": str(meta_creative_id),
                        "ad_id": ad_id,
                        "status": "active",
                        "storage_url": supabase_storage_url or "",
                        "storage_path": storage_path,
                        "file_size_bytes": int(file_size_bytes) if file_size_bytes else 0,
                        "file_type": "image/jpeg",
                        "headline": ad_copy_dict.get("headline", ""),
                        "primary_text": ad_copy_dict.get("primary_text", ""),
                        "description": ad_copy_dict.get("description", ""),
                        "metadata": {
                            "ad_copy": creative_data.get("ad_copy"),
                            "flux_request_id": creative_data.get("flux_request_id"),
                            "storage_creative_id": storage_creative_id,
                            "scenario_description": creative_data.get("scenario_description"),
                        },
                        "created_at": now_iso,
                        "updated_at": now_iso,
                    }
                    if creative_data.get("image_prompt"):
                        creative_intel_data["image_prompt"] = creative_data.get("image_prompt")
                    if creative_data.get("text_overlay"):
                        creative_intel_data["text_overlay_content"] = creative_data.get("text_overlay")
                    
                    try:
                        from infrastructure.data_validation import CreativeIntelligenceOptimizer
                        optimizer = CreativeIntelligenceOptimizer(supabase_client)
                        metrics = optimizer.calculate_performance_metrics(
                            str(meta_creative_id),
                            ad_id,
                        )
                        creative_intel_data['performance_score'] = metrics.get('performance_score', 0.0)
                        creative_intel_data['fatigue_index'] = metrics.get('fatigue_index', 0.0)
                    except ImportError:
                        pass
                    except Exception as opt_error:
                        logger.debug(f"Failed to calculate metrics: {opt_error}")
                    
                    try:
                        from infrastructure.data_validation import validate_supabase_data
                        validation = validate_supabase_data("ads", creative_intel_data, strict_mode=True)
                        if not validation.is_valid:
                            logger.warning(
                                "Validation failed for new ad %s: %s",
                                ad_id,
                                "; ".join(validation.errors),
                            )
                        else:
                            supabase_client.table("ads").upsert(
                                validation.sanitized_data if validation.sanitized_data else creative_intel_data,
                                on_conflict="ad_id"
                            ).execute()
                            logger.debug(f"✅ Successfully upserted new ad {ad_id} to Supabase")
                    except Exception as e:
                        logger.warning(f"Failed to upsert new ad {ad_id} to Supabase: {e}")
                    
                    try:
                        from infrastructure.data_validation import CreativeIntelligenceOptimizer
                        optimizer = CreativeIntelligenceOptimizer(supabase_client)
                        optimizer.update_creative_performance(str(meta_creative_id), ad_id)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Failed to update ads table with storage URL: {e}")
            
            try:
                from infrastructure.supabase_storage import get_validated_supabase_client
                supabase_client = get_validated_supabase_client()
                if supabase_client:
                    final_campaign_id = campaign_id or ''
                    if not final_campaign_id:
                        try:
                            adset_info = client._graph_get_object(f"{adset_id}", params={"fields": "campaign_id"})
                            if adset_info:
                                final_campaign_id = adset_info.get('campaign_id', '')
                        except Exception:
                            pass
                    
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
                    import sys
                    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if src_dir not in sys.path:
                        sys.path.insert(0, src_dir)
                    from main import store_performance_data_in_supabase
                    store_performance_data_in_supabase(supabase_client, initial_ad_data, "asc_plus", None)
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
    try:
        campaign_id = cfg(settings, "ids.asc_plus_campaign_id") or ""
        adset_id = cfg(settings, "ids.asc_plus_adset_id") or ""
        
        if campaign_id and adset_id:
            try:
                client._graph_get_object(f"{campaign_id}", params={"fields": "id,name,status"})
                return campaign_id, adset_id
            except Exception:
                pass
        
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
        
        asc_config = cfg(settings, "asc_plus") or {}
        daily_budget = cfg_or_env_f(asc_config, "daily_budget_eur", "ASC_PLUS_BUDGET", 50.0)
        
        if daily_budget < ASC_PLUS_BUDGET_MIN:
            notify(f"⚠️ ASC+ budget too low: €{daily_budget:.2f}. Minimum is €{ASC_PLUS_BUDGET_MIN:.2f}")
            daily_budget = ASC_PLUS_BUDGET_MIN
        elif daily_budget > ASC_PLUS_BUDGET_MAX:
            notify(f"⚠️ ASC+ budget too high: €{daily_budget:.2f}. Capping at €{ASC_PLUS_BUDGET_MAX:.2f}")
            daily_budget = ASC_PLUS_BUDGET_MAX
        
        target_ads = cfg(settings, "asc_plus.target_active_ads") or 10
        budget_per_creative = daily_budget / target_ads if target_ads > 0 else daily_budget
        min_budget_per_creative = cfg_or_env_f(asc_config, "min_budget_per_creative_eur", None, ASC_PLUS_MIN_BUDGET_PER_CREATIVE)
        
        if budget_per_creative < min_budget_per_creative:
            notify(f"⚠️ Budget per creative (€{budget_per_creative:.2f}) below minimum (€{min_budget_per_creative:.2f})")
            notify(f"   Consider increasing daily budget or reducing target active ads")
        
        targeting = {
            "age_min": 18,
            "age_max": 54,
            "genders": [1],
            "geo_locations": {"countries": ["US"]},
        }
        
        adset_name = "[ASC+] US Men ATC"
        optimization_goal = cfg(asc_config, "optimization_goal") or "ADD_TO_CART"
        adset = client.ensure_adset(
            campaign_id=campaign_id,
            name=adset_name,
            daily_budget=daily_budget,
            optimization_goal=optimization_goal,
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
        
        fixed_budget = 100.0
        try:
            adset_budget = client.get_adset_budget(adset_id)
            if adset_budget:
                if abs(adset_budget - fixed_budget) > 0.01:
                    logger.info(f"Enforcing fixed budget: €{adset_budget:.2f} -> €{fixed_budget:.2f}/day")
                    try:
                        client.update_adset_budget(
                            adset_id=adset_id,
                            daily_budget=fixed_budget,
                            current_budget=adset_budget,
                        )
                        notify(f"✅ Fixed budget enforced: €{fixed_budget:.2f}/day")
                        adset_budget = fixed_budget
                    except Exception as e:
                        logger.warning(f"Failed to enforce fixed budget: {e}")
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


def run_asc_plus_tick(
    client: MetaClient,
    settings: Dict[str, Any],
    rules: Dict[str, Any],
    store: Any,
    ml_system: Optional[Any] = None,
) -> Dict[str, Any]:
    validate_asc_plus_config(settings)
    from rules.rules import AdvancedRuleEngine as RuleEngine
    rule_engine = RuleEngine(rules, store)
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
        rules=rules,
    )
    allowed_statuses = {"ACTIVE", "ELIGIBLE"}
    active_ad_ids = [
        str(ad.get("id"))
        for ad in ads_list
        if str(ad.get("effective_status") or ad.get("status", "")).upper() in allowed_statuses
    ]
    _sync_performance_metrics_records(
        metrics_map,
        stage="asc_plus",
        date_label=account_today,
        active_ad_ids=active_ad_ids,
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

    killed_ads: List[Tuple[str, str]] = []
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
        kill, kill_reason = _guardrail_kill(metrics, rules, rule_engine)
        if kill:
            if _pause_ad(client, ad_id, kill_reason):
                killed_ads.append((ad_id, kill_reason))
                killed_ids.add(ad_id)
                active_count = max(0, active_count - 1)
                if metrics.get("hydrated", True):
                    hydrated_active_count = max(0, hydrated_active_count - 1)
                _record_lifecycle_event(ad_id, "PAUSED", kill_reason)
            continue

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

    (
        active_count,
        hydrated_active_ads,
        hydrated_active_count,
        forced_kills,
        killed_ids,
    ) = _enforce_hard_cap(
        client,
        campaign_id,
        adset_id,
        metrics_map,
        creation_times or {},
        max_active,
        killed_ids,
        allowed_statuses,
    )
    if forced_kills:
        killed_ads.extend(forced_kills)
    planning_active_count = min(active_count, max_active)

    effective_target = min(target_count, max_active)
    available_slots = max(0, max_active - min(active_count, max_active))
    desired_slots = max(0, effective_target - planning_active_count)
    deficit = min(desired_slots, available_slots)
    created_records = _generate_creatives_for_deficit(deficit, client, settings, campaign_id, adset_id, None, active_count)
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
    result["created_ads"] = [record["ad_id"] for record in created_records if record.get("ad_id")]
    result["created_details"] = created_records
    result["ad_metrics"] = metrics_map

    action_lines: List[str] = []
    if killed_ads:
        action_lines.append(f"🛑 Paused {len(killed_ads)} ads:")
        for ad_id, reason in killed_ads[:10]:
            action_lines.append(f"   • {_short_id(ad_id)} – {reason}")
        if len(killed_ads) > 10:
            action_lines.append(f"   • … {len(killed_ads) - 10} more")
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
    
    final_active_ad_ids = {str(ad.get("id")) for ad in hydrated_active_ads if ad.get("id")}
    today_totals = _summarize_today_metrics(client, adset_id, campaign_id, account_today, final_active_ad_ids)
    health_summary = _emit_health_notification(
        health_status,
        health_message,
        today_totals,
        active_count,
        result["target_count"],
    )

    result["ok"] = True
    result["health_summary"] = health_summary
    return result
