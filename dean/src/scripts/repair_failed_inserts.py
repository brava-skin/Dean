"""
One-time repair routine to replay failed Supabase inserts.

Usage:
    python -m dean.src.scripts.repair_failed_inserts --start 2025-10-01T00:00:00Z --end 2025-10-02T00:00:00Z
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

try:
    from supabase import create_client  # type: ignore
except Exception:  # pragma: no cover - supabase optional in some envs
    create_client = None  # type: ignore

from integrations.slack import notify

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
        from infrastructure.supabase_storage import get_validated_supabase_client

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


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay failed Supabase inserts.")
    parser.add_argument("--start", required=True, help="ISO timestamp (inclusive)")
    parser.add_argument("--end", required=True, help="ISO timestamp (inclusive)")
    parser.add_argument("--table", default=DEFAULT_ERROR_TABLE, help="Error log table name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per insert")
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    main()

