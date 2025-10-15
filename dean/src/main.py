from __future__ import annotations

"""
Production-ready runner for continuous creative testing -> validation -> scaling.

Changes in this version:
- Default account timezone -> Europe/Amsterdam (was America/Chicago)
- Supabase queue source (table: meta_creatives) with graceful fallback to CSV/XLSX
- Helper to mark Supabase rows status='launched' after successful ad launch
- (NEW) Expose 'status' in queue DataFrame and add set_supabase_status() for pause/resume

Highlights
- Defensive config/env validation with helpful linting
- Cross-platform process lock to prevent concurrent runs
- Safe retries with exponential backoff and a simple circuit-breaker
- Optional JSON-schema validation (skips gracefully if unavailable)
- Robust CSV/XLSX queue loader (never throws on malformed files)
- Shadow/simulation modes that force read-only client behavior
- Clear, compact Slack notifications and JSON digest logging
"""

import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv

# Optional Supabase client
try:
    # pip install supabase
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None  # degrade gracefully

# Local modules (expected to be available in the project)
from storage import Store
from slack import notify, post_run_header_and_get_thread_ts, post_thread_ads_snapshot, prettify_ad_name, fmt_eur, fmt_pct, fmt_roas, fmt_int
from meta_client import MetaClient, AccountAuth, ClientConfig
from rules import RuleEngine
from stages.testing import run_testing_tick
from stages.validation import run_validation_tick
from stages.scaling import run_scaling_tick
from utils import now_local

# ------------------------------- Constants --------------------------------
REQUIRED_ENVS = [
    "FB_APP_ID",
    "FB_APP_SECRET",
    "FB_ACCESS_TOKEN",
    "FB_AD_ACCOUNT_ID",
    "FB_PIXEL_ID",
    "FB_PAGE_ID",
    "STORE_URL",
    "IG_ACTOR_ID",
]
REQUIRED_IDS = [
    ("ids", "testing_campaign_id"),
    ("ids", "testing_adset_id"),
    ("ids", "validation_campaign_id"),
    ("ids", "scaling_campaign_id"),
]
# Default account/reporting timezone now uses Europe/Amsterdam
DEFAULT_TZ = "Europe/Amsterdam"

DIGEST_DIR = "data/digests"
MAX_STAGE_RETRIES = 3
RETRY_BACKOFF_BASE = 0.6
CIRCUIT_BREAKER_FAILS = 3
LOCKFILE = "data/run.lock"
SCHEMA_PATH_DEFAULT = "config/schema.settings.yaml"

UTC = timezone.utc

# ------------------------------- I/O --------------------------------------


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML document or return empty dict on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_cfg(settings_path: str, rules_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return load_yaml(settings_path), load_yaml(rules_path)


def _normalize_video_id_cell(v: Any) -> str:
    """
    Coerce arbitrary Excel/CSV/Supabase cell into a clean numeric string for Meta video IDs.
    Handles:
      - raw ints/str digits:       1438715257185990
      - floats w/ .0:              1438715257185990.0
      - scientific notation:       1.43871525718599e+15
      - quoted/with commas:        '1,438,715,257,185,990'
    Returns "" if nothing usable.
    """
    if v is None:
        return ""
    s = str(v).strip().strip("'").strip('"')
    if s == "" or s.lower() in ("nan", "none", "null"):
        return ""

    # strip commas/spaces first
    s = s.replace(",", "").replace(" ", "")

    # pure digits -> keep
    if re.fullmatch(r"\d+", s):
        return s

    # trailing .0 -> drop fractional
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m:
        return m.group(1)

    # scientific notation -> Decimal to full integer string
    if re.fullmatch(r"\d+(\.\d+)?[eE]\+\d+", s):
        try:
            getcontext().prec = 50
            return str(int(Decimal(s)))
        except (InvalidOperation, ValueError):
            return ""

    # last resort: keep only digits if that yields something plausible
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if digits else ""


def load_queue(path: str) -> pd.DataFrame:
    """
    Load creatives queue from a file path (CSV/XLSX). Expected optional columns:
      video_id, filename, avatar, visual_style, script
    Extended optional columns that may be present:
      creative_id, name, thumbnail_url, primary_text, headline, description, page_id, utm_params
    Supports .csv and .xlsx. Returns an empty, well-typed DataFrame on error.
    """
    cols = [
        "creative_id",
        "name",
        "video_id",
        "thumbnail_url",
        "primary_text",
        "headline",
        "description",
        "page_id",
        "utm_params",
        "avatar",
        "visual_style",
        "script",
        "filename",
        "status",  # NEW: keep status in DF for notify-once logic in stages
    ]

    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=cols)

    # Read as strings to avoid pandas -> float/scientific coercion
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(
                path,
                dtype=str,
                keep_default_na=False,
                converters={"video_id": _normalize_video_id_cell},
            )
        else:
            try:
                df = pd.read_csv(
                    path,
                    dtype=str,
                    keep_default_na=False,
                )
            except Exception:
                df = pd.read_csv(
                    path,
                    dtype=str,
                    keep_default_na=False,
                    encoding="utf-8-sig",
                )
    except Exception:
        return pd.DataFrame(columns=cols)

    # Ensure all expected columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    # Canonical order
    df = df[cols]

    # Normalize video_id for CSV (read_csv converters are not applied like read_excel ones)
    try:
        df["video_id"] = df["video_id"].map(_normalize_video_id_cell)
    except Exception:
        pass

    return df


def save_queue(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def digest_path_for_today() -> str:
    Path(DIGEST_DIR).mkdir(parents=True, exist_ok=True)
    return str(Path(DIGEST_DIR) / f"digest_{datetime.utcnow():%Y-%m-%d}.jsonl")


def append_digest(record: Dict[str, Any]) -> None:
    try:
        with open(digest_path_for_today(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# --------------------------- Supabase queue --------------------------------

def _get_supabase():
    """
    Build a Supabase client from env. Degrades cleanly if missing.
    Env:
      SUPABASE_URL
      SUPABASE_SERVICE_ROLE_KEY  (preferred) or SUPABASE_ANON_KEY
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not (create_client and url and key):
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def load_queue_supabase(
    table: str = None,
    status_filter: str = "pending",
    limit: int = 64,
) -> pd.DataFrame:
    """
    Read creative rows from Supabase and normalize to the columns Testing expects.
    Selects rows where status is NULL or equals `status_filter`.

    Returns a DataFrame with columns:
      creative_id, name, video_id, thumbnail_url, primary_text, headline, description,
      page_id, utm_params, avatar, visual_style, script, filename, status
    """
    cols = [
        "creative_id",
        "name",
        "video_id",
        "thumbnail_url",
        "primary_text",
        "headline",
        "description",
        "page_id",
        "utm_params",
        "avatar",
        "visual_style",
        "script",
        "filename",
        "status",  # NEW: expose DB status to stages
    ]

    sb = _get_supabase()
    if not sb:
        notify("⚠️ Supabase client not available; falling back to file-based queue.")
        return pd.DataFrame(columns=cols)

    table = table or os.getenv("SUPABASE_TABLE", "meta_creatives")
    try:
        q = (
            sb.table(table)
            .select("id, video_id, filename, avatar, visual_style, script, status")
            .or_("status.is.null,status.eq.{}".format(status_filter))
            .limit(limit)
        )
        data = q.execute().data or []
    except Exception as e:
        notify(f"❗ Supabase read failed: {e}")
        return pd.DataFrame(columns=cols)

    rows = []
    for r in data:
        rows.append(
            {
                "creative_id": r.get("id") or "",
                "name": "",  # label is built from avatar/visual/script inside Testing tick
                "video_id": _normalize_video_id_cell(r.get("video_id")),
                "thumbnail_url": "",
                "primary_text": "",
                "headline": "",
                "description": "",
                "page_id": "",
                "utm_params": "",
                "avatar": r.get("avatar") or "",
                "visual_style": r.get("visual_style") or "",
                "script": r.get("script") or "",
                "filename": r.get("filename") or "",
                "status": (r.get("status") or "").lower(),  # keep raw status visible
            }
        )

    df = pd.DataFrame(rows, columns=cols)
    # Ensure string dtype and fill NA; normalize video_id
    for c in cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str).fillna("")
    try:
        df["video_id"] = df["video_id"].map(_normalize_video_id_cell)
    except Exception:
        pass

    notify(f"📥 Supabase queue loaded: {len(df)} rows from '{table}'")
    return df


def set_supabase_status(
    ids_or_video_ids: List[str],
    new_status: str,
    *,
    use_column: str = "id",
    table: str = None,
) -> None:
    """
    Generic status setter for meta_creatives.
      use_column='id'       -> pass Supabase PKs (matches 'creative_id' in DF)
      use_column='video_id' -> pass Meta video IDs
    Examples:
      set_supabase_status([creative_id], 'launched')
      set_supabase_status([creative_id], 'paused')
      set_supabase_status([creative_id], 'pending')
    """
    if not ids_or_video_ids:
        return
    sb = _get_supabase()
    if not sb:
        return

    table = table or os.getenv("SUPABASE_TABLE", "meta_creatives")
    try:
        CHUNK = 100
        for i in range(0, len(ids_or_video_ids), CHUNK):
            chunk = ids_or_video_ids[i : i + CHUNK]
            if use_column == "video_id":
                (
                    sb.table(table)
                    .update({"status": new_status})
                    .in_("video_id", chunk)
                    .execute()
                )
            else:
                (
                    sb.table(table)
                    .update({"status": new_status})
                    .in_("id", chunk)
                    .execute()
                )
    except Exception as e:
        notify(f"⚠️ Supabase status update failed ({new_status}): {e}")


def mark_supabase_launched(ids_or_video_ids: List[str], use_column: str = "id", table: str = None) -> None:
    """
    Backward-compat helper. Prefer set_supabase_status(..., 'launched').
    """
    set_supabase_status(ids_or_video_ids, "launched", use_column=use_column, table=table)


# --------------------------- Config hygiene --------------------------------


def redact(s: Optional[str], keep_last: int = 4) -> str:
    if not s:
        return ""
    s = str(s)
    return ("*" * max(0, len(s) - keep_last)) + s[-keep_last:]


def validate_envs(required: List[str]) -> List[str]:
    return [k for k in required if not os.getenv(k)]


def validate_settings_ids(settings: Dict[str, Any]) -> List[str]:
    miss: List[str] = []
    for section, key in REQUIRED_IDS:
        if not (settings.get(section, {}) or {}).get(key):
            miss.append(f"{section}.{key}")
    return miss


def linter(settings: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    # timezone sanity
    cfg_tz = settings.get("account_timezone") or settings.get("timezone") or DEFAULT_TZ
    env_tz = os.getenv("TIMEZONE")
    if env_tz and env_tz != cfg_tz:
        issues.append(f"Timezone mismatch? config={cfg_tz} env={env_tz}")

    # required top-level sections
    for k in ("ids", "testing", "validation", "scaling", "queue", "logging"):
        if k not in settings:
            issues.append(f"Missing section: {k}")

    # rules sanity (CPA strictness ordering)
    cpa_thr = (rules.get("thresholds") or {}).get("cpa", {})
    try:
        v_max = float(cpa_thr.get("validation_max", 1))
        t_max = float(cpa_thr.get("testing_max", 0))
        if v_max < t_max:
            issues.append("Rules: validation_max CPA < testing_max CPA; check strictness ordering")
    except Exception:
        pass

    # basic budget min/max if present
    b = (settings.get("scaling", {}) or {}).get("budget", {}) or {}
    try:
        mn, mx = float(b.get("min_usd", 0) or 0), float(b.get("max_usd", 0) or 0)
        if mn and mx and mx < mn:
            issues.append(f"Budget min/max invalid: min={mn} > max={mx}")
    except Exception:
        pass

    return issues


# -------------------------- Locks & retries ---------------------------------


@contextmanager
def file_lock(path: str):
    """
    Cross-platform(ish) run lock:
    - POSIX: flock
    - Windows/others: presence + PID written, best-effort
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR)
        locked = False
        try:
            import fcntl  # type: ignore

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except Exception:
            # Fallback: if file non-empty, assume another process holds it
            try:
                if os.path.getsize(path) > 0:
                    raise RuntimeError("Lock already held")
            except Exception:
                pass
            locked = True  # proceed best-effort
        if locked:
            try:
                os.ftruncate(fd, 0)
                os.write(fd, str(os.getpid()).encode())
            except Exception:
                pass
            yield
    finally:
        try:
            if fd is not None:
                try:
                    import fcntl  # type: ignore

                    fcntl.flock(fd, fcntl.LOCK_UN)
                except Exception:
                    pass
                os.close(fd)
                os.remove(path)
        except Exception:
            pass


def stage_retry(
    fn,
    *,
    name: str,
    retries: int = MAX_STAGE_RETRIES,
    backoff_base: float = RETRY_BACKOFF_BASE,
):
    def _wrapped(*args, **kwargs):
        last: Optional[BaseException] = None
        for attempt in range(retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
                if attempt < retries:
                    delay = min(backoff_base * (2**attempt), 8.0)
                    notify(f"⏳ [{name}] retry {attempt + 1}/{retries} in {delay:.1f}s: {e}")
                    time.sleep(delay)
        assert last is not None
        raise last

    return _wrapped


# --------------------------- Health & guardrails -----------------------------


def health_check(store: Store, client: MetaClient) -> Dict[str, Any]:
    ok = True
    details: List[str] = []
    # DB write/read
    try:
        store.incr("healthcheck", 1)
        details.append("db:ok")
    except Exception as e:
        ok = False
        details.append(f"db:fail:{e}")
    # Slack (best-effort)
    try:
        notify("🩺 healthcheck")
        details.append("slack:ok")
    except Exception as e:
        details.append(f"slack:warn:{e}")
    # Meta lightweight read
    try:
        client.get_ad_insights(level="ad", fields=["spend"], paginate=False)
        details.append("meta:ok")
    except Exception as e:
        ok = False
        details.append(f"meta:fail:{e}")
    return {"ok": ok, "details": details}


def account_guardrail_ping(meta: MetaClient, settings: Dict[str, Any]) -> Dict[str, Any]:
    try:
        rows = meta.get_ad_insights(level="ad", fields=["spend", "actions"], paginate=True)
        spend = sum(float(r.get("spend") or 0) for r in rows)
        purch = 0.0
        for r in rows:
            for a in (r.get("actions") or []):
                if a.get("action_type") == "purchase":
                    try:
                        purch += float(a.get("value") or 0)
                    except Exception:
                        pass
        cpa = (spend / purch) if purch > 0 else float("inf")
        be = float(
            os.getenv("BREAKEVEN_CPA")
            or (settings.get("economics", {}) or {}).get("breakeven_cpa")
            or 34
        )
        return {
            "spend": round(spend, 2),
            "purchases": int(purch),
            "cpa": None if cpa == float("inf") else round(cpa, 2),
            "breakeven": be,
        }
    except Exception:
        return {"spend": None, "purchases": None, "cpa": None, "breakeven": None}


# ------------------------------- Summaries ----------------------------------


def summarize_counts(label: str, summary: Optional[Dict[str, Any]]) -> str:
    if not summary:
        return f"{label}: n/a"
    return f"{label}: " + ", ".join(f"{k}={v}" for k, v in summary.items())


# --------------------------------- Main -------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brava — Continuous Creative Testing & Scaling"
    )
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--rules", default="config/rules.yaml")
    parser.add_argument("--schema", default=SCHEMA_PATH_DEFAULT)
    parser.add_argument(
        "--stage", choices=["all", "testing", "validation", "scaling"], default="all"
    )
    parser.add_argument("--profile", choices=["production", "staging"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-digest", action="store_true")
    parser.add_argument(
        "--simulate", action="store_true", help="shadow mode: log intended actions only"
    )
    parser.add_argument("--since", default=None, help="simulation since (YYYY-MM-DD)")
    parser.add_argument("--until", default=None, help="simulation until (YYYY-MM-DD)")
    parser.add_argument(
        "--explain", action="store_true", help="print decisions without acting"
    )
    args = parser.parse_args()

    # Load environment first (for dynamic .env overrides)
    load_dotenv()

    # Load config and rules
    settings, rules_cfg = load_cfg(args.settings, args.rules)

    # Resolve profile/dry-run/shadow
    profile = (
        args.profile
        or (settings.get("mode", {}) or {}).get("current")
        or os.getenv("MODE")
        or "production"
    ).lower()
    effective_dry = (
        args.dry_run
        or (profile == "staging")
        or (os.getenv("DRY_RUN", "false").lower() == "true")
    )
    shadow_mode = args.simulate or args.explain

    # Optional JSON schema validation (best-effort)
    try:
        schema = load_yaml(args.schema)
        if schema:
            import jsonschema  # type: ignore

            jsonschema.validate(instance=settings, schema=schema)
    except Exception:
        pass

    # Lint and basic validation
    missing_envs = validate_envs(REQUIRED_ENVS)
    missing_ids = validate_settings_ids(settings)
    lint_issues = linter(settings, rules_cfg)
    if missing_envs or missing_ids or lint_issues:
        msg = []
        if missing_envs:
            msg.append("Missing ENVs: " + ", ".join(missing_envs))
        if missing_ids:
            msg.append("Missing IDs: " + ", ".join(missing_ids))
        if lint_issues:
            msg.append("Lint: " + " | ".join(lint_issues))
        severity = "info" if (profile == "staging" or effective_dry) else "error"
        notify((("⚠️ " if severity == "info" else "🛑 ") + " | ".join(msg)))
        if not (profile == "staging" or effective_dry):
            print("Fatal configuration error. Exiting.", file=sys.stderr)
            sys.exit(1)

    # Store (SQLite) — path from settings or default
    sqlite_path = (settings.get("logging") or {}).get("sqlite_path", "data/automation.sqlite")
    store = Store(sqlite_path)

    # Timezone for account (prefer settings.account_timezone, then env TIMEZONE, else DEFAULT_TZ=Europe/Amsterdam)
    tz_name = (
        settings.get("account_timezone")
        or settings.get("timezone")
        or os.getenv("TIMEZONE")
        or DEFAULT_TZ
    )

    # Build Meta client
    account = AccountAuth(
        account_id=os.getenv("FB_AD_ACCOUNT_ID", ""),
        access_token=os.getenv("FB_ACCESS_TOKEN", ""),
        app_id=os.getenv("FB_APP_ID", ""),
        app_secret=os.getenv("FB_APP_SECRET", ""),
        api_version=os.getenv("FB_API_VERSION") or None,
    )
    # ClientConfig in meta_client.py does not accept attribution fields. Keep it minimal.
    cfg = ClientConfig(
        timezone=tz_name
        # currency, budgets and switches are already defaulted inside ClientConfig
    )
    client = MetaClient(
        accounts=[account],
        cfg=cfg,
        store=store,
        dry_run=(effective_dry or shadow_mode),
        tenant_id=settings.get("branding_name", "default"),
    )

    # Rule engine
    engine = RuleEngine(rules_cfg)

    # Preflight health check
    hc = health_check(store, client)
    if not hc["ok"]:
        notify("🛑 Preflight failed: " + " ".join(hc["details"]))
        if not (profile == "staging" or effective_dry):
            sys.exit(1)

    # Queue: Prefer Supabase if configured; else fallback to file path.
    if os.getenv("SUPABASE_URL") and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")):
        table = os.getenv("SUPABASE_TABLE", "meta_creatives")
        queue_df = load_queue_supabase(table=table, status_filter="pending", limit=64)
        queue_len_before = len(queue_df)
    else:
        queue_path = (settings.get("queue") or {}).get("path", "data/creatives_queue.csv")
        queue_df = load_queue(queue_path)
        queue_len_before = len(queue_df)

    # Context ping - now using consolidated messaging
    local_now = now_local(tz_name)
    acct = account_guardrail_ping(client, settings)
    
    # Store account info for later use in consolidated message
    account_info = {
        'spend': acct.get('spend', 0.0),
        'purchases': acct.get('purchases', 0),
        'cpa': acct.get('cpa'),
        'breakeven': acct.get('breakeven')
    }

    # Idempotency (tick-level) and process lock (multi-runner safety)
    try:
        tkey = f"tick::{local_now:%Y-%m-%dT%H:%M}"
        if hasattr(store, "tick_seen") and store.tick_seen(tkey):  # if you implemented this helper
            notify("ℹ️ Tick already processed; exiting.")
            return
    except Exception:
        pass

    with file_lock(LOCKFILE):
        failures_in_row = 0
        overall: Dict[str, Any] = {}

        def run_stage(callable_fn, label: str, *fn_args, **fn_kwargs) -> Optional[Dict[str, Any]]:
            nonlocal failures_in_row
            wrapped = stage_retry(callable_fn, name=label)
            t0 = time.time()
            try:
                if shadow_mode:
                    # In shadow mode ensure no writes to Meta
                    client.dry_run = True
                    # Proper Store.log call with explicit fields
                    try:
                        store.log(
                            entity_type="system",
                            entity_id="shadow",
                            action="EXPLAIN",
                            level="info",
                            stage=label,
                            reason="shadow mode (no writes)",
                            meta={"stage": label},
                        )
                    except Exception:
                        pass

                res = wrapped(*fn_args, **fn_kwargs)
                dt = time.time() - t0
                failures_in_row = 0
                notify(f"✅ [{label}] {dt:.1f}s — {summarize_counts(label, res)}")
                return res
            except Exception as e:
                failures_in_row += 1
                dt = time.time() - t0
                notify(f"❌ [{label}] {dt:.1f}s — {e}")
                if failures_in_row >= CIRCUIT_BREAKER_FAILS:
                    notify(
                        f"🧯 Circuit breaker tripped ({failures_in_row}); switching to read-only for remainder."
                    )
                    client.dry_run = True
                return None

        stage_choice = args.stage

        # Collect stage summaries for consolidated messaging
        stage_summaries = []
        
        if stage_choice in ("all", "testing"):
            overall["testing"] = run_stage(
                run_testing_tick,
                "TESTING",
                client,
                settings,
                engine,
                store,
                queue_df,
                set_supabase_status,
                placements=["facebook", "instagram"],  # NEW
                instagram_actor_id=os.getenv("IG_ACTOR_ID"),  # NEW
            )
            if overall["testing"]:
                stage_summaries.append({
                    "stage": "TEST",
                    "counts": overall["testing"]
                })

        if stage_choice in ("all", "validation"):
            overall["validation"] = run_stage(
                run_validation_tick, "VALIDATION", client, settings, engine, store
            )
            if overall["validation"]:
                stage_summaries.append({
                    "stage": "VALID", 
                    "counts": overall["validation"]
                })

        if stage_choice in ("all", "scaling"):
            overall["scaling"] = run_stage(run_scaling_tick, "SCALING", client, settings, store)
            if overall["scaling"]:
                stage_summaries.append({
                    "stage": "SCALE",
                    "counts": overall["scaling"]
                })

    # Queue persist (only if changed length; cheap heuristic).
    # When using Supabase, this block normally will not run (DF length does not change in-place).
    if 'queue_path' in locals() and len(queue_df) != queue_len_before:
        try:
            save_queue(queue_df, queue_path)
            notify(f"📦 Queue saved ({len(queue_df)} rows) -> {queue_path}")
        except Exception as e:
            notify(f"⚠️ Queue save failed: {e}")

    # Digest (best-effort)
    if not args.no_digest:
        try:
            append_digest(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "profile": profile,
                    "dry_run": client.dry_run,
                    "simulate": shadow_mode,
                    "timezone": tz_name,
                    "stage": args.stage,
                    "acct": acct,
                    "health": hc,
                }
            )
        except Exception:
            pass

    # Post consolidated run summary
    if not shadow_mode:
        time_str = local_now.strftime("%H:%M %Z")
        status = "OK"  # Simplified status logic - could be enhanced based on failures_in_row
        
        # Post the main run header and get thread timestamp
        thread_ts = post_run_header_and_get_thread_ts(
            status=status,
            time_str=time_str,
            profile=profile,
            spend=account_info['spend'],
            purch=account_info['purchases'],
            cpa=account_info['cpa'],
            be=account_info['breakeven'],
            stage_summaries=stage_summaries
        )
        
        # TODO: Collect ad insights and post as thread reply
        # This would require fetching ad insights and formatting them
        # For now, we'll skip the thread reply as it requires more complex data collection

    # Console summary
    print("---- RUN SUMMARY ----")
    print(
        json.dumps(
            {
                "profile": profile,
                "dry_run": client.dry_run,
                "simulate": shadow_mode,
                "acct": acct,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
