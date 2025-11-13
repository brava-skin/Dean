#!/usr/bin/env python3
"""
Database Maintenance Utilities
Combined utilities for Meta insights ingestion and failed insert repair.

Usage:
    # Ingest Meta insights
    python -m dean.src.scripts.database_maintenance ingest --start-date 2024-10-01 --end-date 2024-10-05
    
    # Repair failed inserts
    python -m dean.src.scripts.database_maintenance repair --start 2025-10-01T00:00:00Z --end 2025-10-02T00:00:00Z
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None  # type: ignore

from analytics.metrics import Metrics, metrics_from_row
from integrations.meta_client import AccountAuth, ClientConfig, MetaClient
from integrations.slack import notify
from infrastructure.supabase_storage import get_validated_supabase_client

logger = logging.getLogger(__name__)

# =====================================================
# INSIGHTS INGESTION
# =====================================================

def _load_settings(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_meta_client(settings: Dict[str, Any], *, dry_run: bool) -> MetaClient:
    auth = AccountAuth(
        account_id=os.getenv("FB_AD_ACCOUNT_ID", ""),
        access_token=os.getenv("META_ACCESS_TOKEN", ""),
        app_id=os.getenv("META_APP_ID", ""),
        app_secret=os.getenv("META_APP_SECRET", ""),
        api_version=os.getenv("META_API_VERSION"),
    )
    if not auth.account_id or not auth.access_token:
        raise RuntimeError("Missing FB_AD_ACCOUNT_ID or META_ACCESS_TOKEN environment variables.")

    cfg = ClientConfig(
        timezone=settings.get("account", {}).get("timezone", ClientConfig.timezone),
        attribution_click_days=settings.get("meta", {}).get("attribution", {}).get("click_days", 7),
        attribution_view_days=settings.get("meta", {}).get("attribution", {}).get("view_days", 1),
    )
    return MetaClient(auth, cfg=cfg, dry_run=dry_run)


def _daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=1)


def _to_float(value: Optional[str]) -> Optional[float]:
    if value in (None, "", [], {}, ()):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _row_has_required_values(row: Dict[str, Any]) -> bool:
    impressions = _to_float(row.get("impressions"))
    spend = _to_float(row.get("spend"))
    return impressions is not None and spend is not None


def _extract_counts(metrics: Metrics) -> Tuple[int, int, int, float, float, float]:
    return (
        int(round(metrics.impressions or 0)),
        int(round(metrics.clicks or 0)),
        int(round(metrics.purchases or 0)),
        float(metrics.add_to_cart or 0.0),
        float(metrics.initiate_checkout or 0.0),
        float(metrics.revenue or 0.0),
    )


def _fetch_insights_for_day(
    client: MetaClient,
    day: date,
    *,
    adset_id: Optional[str],
    debounce: float,
) -> List[Dict[str, Any]]:
    if getattr(client, "dry_run", False):
        logger.info("Meta client in dry-run mode; skipping fetch for %s", day.isoformat())
        return []

    start_iso = day.strftime("%Y-%m-%d")
    end_iso = (day + timedelta(days=1)).strftime("%Y-%m-%d")

    filtering = []
    if adset_id:
        filtering.append({"field": "adset.id", "operator": "IN", "value": [adset_id]})

    rows: List[Dict[str, Any]] = []
    seen_cursors: set[str] = set()
    after: Optional[str] = None

    while True:
        params = {
            "level": "ad",
            "time_range": {"since": start_iso, "until": end_iso},
            "filtering": filtering,
            "limit": 500,
            "fields": list(client.cfg.fields_default),
            "action_attribution_windows": [
                f"{client.cfg.attribution_click_days}d_click",
                f"{client.cfg.attribution_view_days}d_view",
            ],
            "paginate": False,
        }
        if after:
            params["after"] = after

        cursor = client._graph_get_object(  # pylint: disable=protected-access
            f"{client.ad_account_id_act}/insights",
            params=params,
        )
        data = cursor.get("data") if isinstance(cursor, dict) else []
        if not isinstance(data, list):
            data = []

        rows.extend(data)

        paging = cursor.get("paging") if isinstance(cursor, dict) else None
        next_after = paging.get("cursors", {}).get("after") if paging else None
        if not next_after or next_after in seen_cursors:
            break

        seen_cursors.add(next_after)
        after = next_after
        if debounce > 0:
            time.sleep(debounce)

    return rows


def _build_performance_record(
    ad_id: str,
    day: date,
    stage: str,
    metrics: Metrics,
    raw_row: Dict[str, Any],
) -> Dict[str, Any]:
    impressions, clicks, purchases, add_to_cart, initiate_checkout, revenue = _extract_counts(metrics)
    spend = float(metrics.spend or 0.0)

    ctr = metrics.ctr if metrics.ctr is not None else (clicks / impressions if impressions > 0 else None)
    cpc = metrics.cpc if metrics.cpc is not None else (spend / clicks if clicks > 0 else None)
    cpm = metrics.cpm if metrics.cpm is not None else (spend * 1000.0 / impressions if impressions > 0 else None)
    cpa = metrics.cpa if metrics.cpa is not None else (spend / purchases if purchases > 0 else None)
    atc_rate = add_to_cart / impressions if impressions > 0 else None
    ic_rate = initiate_checkout / impressions if impressions > 0 else None
    purchase_rate = purchases / impressions if impressions > 0 else None

    return {
        "ad_id": ad_id,
        "lifecycle_id": raw_row.get("lifecycle_id") or f"lifecycle_{ad_id}",
        "stage": stage,
        "window_type": "1d",
        "date_start": day.strftime("%Y-%m-%d"),
        "date_end": day.strftime("%Y-%m-%d"),
        "impressions": impressions,
        "clicks": clicks,
        "spend": spend,
        "purchases": purchases,
        "add_to_cart": int(round(add_to_cart)),
        "initiate_checkout": int(round(initiate_checkout)),
        "ctr": ctr,
        "cpc": cpc,
        "cpm": cpm,
        "roas": metrics.roas,
        "cpa": cpa,
        "atc_rate": atc_rate,
        "ic_rate": ic_rate,
        "purchase_rate": purchase_rate,
        "revenue": revenue,
    }


def ingest_day(
    client: MetaClient,
    supabase,
    day: date,
    *,
    stage: str,
    adset_id: Optional[str],
    debounce: float,
    dry_run: bool,
) -> Tuple[int, int]:
    rows = _fetch_insights_for_day(client, day, adset_id=adset_id, debounce=debounce)
    logger.info("Fetched %d raw rows for %s", len(rows), day.isoformat())

    processed: List[Dict[str, Any]] = []
    for row in rows:
        ad_id = str(row.get("ad_id") or "").strip()
        if not ad_id:
            continue
        if not _row_has_required_values(row):
            continue

        metrics = metrics_from_row(row)
        if isinstance(metrics, tuple):
            metrics = metrics[0]
        record = _build_performance_record(ad_id, day, stage, metrics, row)
        processed.append(record)

    if not processed:
        logger.info("No valid rows to upsert for %s", day.isoformat())
        return 0, len(rows)

    if dry_run:
        logger.info("Dry-run mode: skipping Supabase upsert for %d records", len(processed))
        return len(processed), len(rows)

    supabase.table("performance_metrics").upsert(
        processed,
        on_conflict="ad_id,stage,window_type,date_start",
    ).execute()
    logger.info("Upserted %d records for %s", len(processed), day.isoformat())
    return len(processed), len(rows)


def cmd_ingest(args) -> None:
    """Command: Ingest Meta ad insights into Supabase."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid --start-date: {exc}") from exc

    if args.end_date:
        try:
            end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise SystemExit(f"Invalid --end-date: {exc}") from exc
    else:
        end = start + timedelta(days=1)

    if end <= start:
        raise SystemExit("--end-date must be after --start-date")

    settings = _load_settings(args.settings)
    adset_id = args.adset_id or settings.get("ids", {}).get("asc_plus_adset_id")

    client = _build_meta_client(settings, dry_run=args.dry_run)
    supabase = None
    if not args.dry_run:
        supabase = get_validated_supabase_client(enable_validation=True)
        if not supabase:
            raise SystemExit("Failed to initialize Supabase client.")

    total_rows = 0
    total_upserts = 0
    for day in _daterange(start, end):
        ingested, fetched = ingest_day(
            client,
            supabase,
            day,
            stage=args.stage,
            adset_id=adset_id,
            debounce=args.debounce,
            dry_run=args.dry_run,
        )
        total_rows += fetched
        total_upserts += ingested

    logger.info(
        "Ingestion complete: %d rows processed across %d days (%d inserted).",
        total_rows,
        (end - start).days,
        total_upserts,
    )


# =====================================================
# INSERT REPAIR
# =====================================================

DEFAULT_ERROR_TABLE = os.getenv("SUPABASE_INSERT_ERROR_TABLE", "insert_failures")
DEFAULT_BATCH_SIZE = int(os.getenv("INSERT_REPAIR_BATCH_SIZE", "50"))
MAX_RETRIES = int(os.getenv("INSERT_REPAIR_MAX_RETRIES", "4"))
BACKOFF_BASE = float(os.getenv("INSERT_REPAIR_BACKOFF_BASE", "1.0"))


def _build_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not (create_client and url and key):
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def _get_validated_client():
    try:
        return get_validated_supabase_client(enable_validation=True)
    except Exception:
        return None


def _parse_payload(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _iso_to_datetime(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value)


def replay_failed_inserts(
    start_iso: str,
    end_iso: str,
    error_table: str = DEFAULT_ERROR_TABLE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[int, int, int]:
    supabase = _build_supabase_client()
    if not supabase:
        raise RuntimeError("Supabase client unavailable (check SUPABASE_URL and keys).")

    validated_client = _get_validated_client() or supabase

    start_dt = _iso_to_datetime(start_iso)
    end_dt = _iso_to_datetime(end_iso)

    fetch = (
        supabase.table(error_table)
        .select("*")
        .gte("created_at", start_dt.isoformat())
        .lte("created_at", end_dt.isoformat())
        .order("created_at")
        .execute()
    )

    rows = getattr(fetch, "data", None) or []
    if not rows:
        notify("✅ Insert repair: no failed inserts found in the provided window.")
        return 0, 0, 0

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        table = row.get("table_name")
        if not table:
            continue
        payload = _parse_payload(row.get("payload"))
        if not payload:
            continue
        payload["_repair_origin"] = "insert_replay"
        grouped[table].append({"id": row.get("id"), "payload": payload})

    repaired = 0
    skipped = 0
    total = sum(len(entries) for entries in grouped.values())
    start_time = time.monotonic()

    for table_name, entries in grouped.items():
        idx = 0
        while idx < len(entries):
            chunk = entries[idx : idx + batch_size]
            payloads = [entry["payload"] for entry in chunk]
            ids = [entry["id"] for entry in chunk if entry.get("id") is not None]
            attempt = 0
            backoff = BACKOFF_BASE
            while attempt < MAX_RETRIES:
                try:
                    validated_client.table(table_name).insert(payloads).execute()
                    repaired += len(payloads)
                    if ids:
                        supabase.table(error_table).update(
                            {"status": "repaired", "repaired_at": datetime.now(timezone.utc).isoformat()}
                        ).in_("id", ids).execute()
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt >= MAX_RETRIES:
                        skipped += len(payloads)
                        if ids:
                            supabase.table(error_table).update(
                                {
                                    "status": "skipped",
                                    "repair_note": f"Retries exhausted: {exc}",
                                }
                            ).in_("id", ids).execute()
                    else:
                        time.sleep(backoff)
                        backoff *= 2
            idx += batch_size

    elapsed = time.monotonic() - start_time
    summary = (
        f"🛠️ Insert repair complete\n"
        f"• Window: {start_iso} → {end_iso}\n"
        f"• Rows repaired: {repaired}/{total}\n"
        f"• Rows skipped: {skipped}\n"
        f"• Duration: {elapsed:.1f}s"
    )
    notify(summary)
    return repaired, skipped, total


def cmd_repair(args) -> None:
    """Command: Replay failed Supabase inserts."""
    repaired, skipped, total = replay_failed_inserts(
        args.start,
        args.end,
        error_table=args.table,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "window": {"start": args.start, "end": args.end},
                "repaired": repaired,
                "skipped": skipped,
                "attempted": total,
            }
        )
    )


# =====================================================
# MAIN ENTRY POINT
# =====================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Database maintenance utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest Meta ad insights into Supabase")
    ingest_parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings YAML.")
    ingest_parser.add_argument("--start-date", required=True, help="Start date (inclusive) in YYYY-MM-DD (UTC midnight).")
    ingest_parser.add_argument("--end-date", help="End date (exclusive) in YYYY-MM-DD (UTC midnight). Defaults to start-date + 1 day.")
    ingest_parser.add_argument("--stage", default="asc_plus", help="Stage label stored in Supabase (default: asc_plus).")
    ingest_parser.add_argument("--adset-id", help="Optional Meta adset id filter. Falls back to settings.ids.asc_plus_adset_id.")
    ingest_parser.add_argument("--debounce", type=float, default=0.5, help="Seconds to sleep between paginated requests (default 0.5).")
    ingest_parser.add_argument("--dry-run", action="store_true", help="Print actions without writing to Supabase.")
    
    # Repair subcommand
    repair_parser = subparsers.add_parser("repair", help="Replay failed Supabase inserts")
    repair_parser.add_argument("--start", required=True, help="ISO timestamp (inclusive)")
    repair_parser.add_argument("--end", required=True, help="ISO timestamp (inclusive)")
    repair_parser.add_argument("--table", default=DEFAULT_ERROR_TABLE, help="Error log table name")
    repair_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per insert")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "repair":
        cmd_repair(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
