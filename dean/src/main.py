from __future__ import annotations

"""
DEAN SELF-LEARNING META ADS AUTOMATION SYSTEM
Next-Generation ML-Enhanced Main Runner

This is the completely overhauled main runner that integrates:
- Advanced ML intelligence with XGBoost prediction engines
- Cross-stage transfer learning and temporal modeling
- Adaptive rules engine with dynamic threshold adjustment
- Advanced performance tracking and fatigue detection
- Comprehensive Supabase backend for ML data storage
- Predictive reporting and transparency system

The system continuously learns from campaign data across Testing → Validation → Scaling,
identifies signals that predict purchases, and dynamically adjusts all rules to
keep CPA consistently below €27.50 while scaling safely and profitably.
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

# ML Intelligence System (NEW) - Conditional imports
try:
    from ml.ml_intelligence import MLIntelligenceSystem, MLConfig, create_ml_system
    from rules import IntelligentRuleEngine, RuleConfig, create_intelligent_rule_engine
    from analytics import PerformanceTrackingSystem, create_performance_tracking_system
    from ml.ml_reporting import MLReportingSystem, create_ml_reporting_system
    from ml.ml_enhancements import (
        create_model_validator, create_data_progress_tracker, create_anomaly_detector,
        create_time_series_forecaster, create_creative_similarity_analyzer, create_causal_impact_analyzer
    )
    from ml.ml_decision_engine import create_ml_decision_engine
    from ml.ml_pipeline import create_ml_pipeline, MLPipelineConfig
    from ml.ml_monitoring import create_ml_dashboard, get_ml_learning_summary, send_ml_learning_report
    from ml.ml_advanced_features import (
        create_ql_agent, create_lstm_predictor, create_auto_feature_engineer,
        create_bayesian_optimizer, create_portfolio_optimizer, create_seasonality_detector, create_shap_explainer
    )
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML system not available: {e}")
    print("   System will run in standard mode")
    ML_AVAILABLE = False
    # Create dummy classes for compatibility
    class MLIntelligenceSystem: pass
    class MLConfig: pass
    class IntelligentRuleEngine: pass
    class RuleConfig: pass
    class PerformanceTrackingSystem: pass
    class MLReportingSystem: pass
    def create_ml_system(*args, **kwargs): return None
    def create_intelligent_rule_engine(*args, **kwargs): return None
    def create_performance_tracking_system(*args, **kwargs): return None
    def create_ml_reporting_system(*args, **kwargs): return None
    def create_model_validator(*args, **kwargs): return None
    def create_data_progress_tracker(*args, **kwargs): return None
    def create_anomaly_detector(*args, **kwargs): return None
    def create_ml_pipeline(*args, **kwargs): return None
    def create_ml_dashboard(*args, **kwargs): return None
    def get_ml_learning_summary(*args, **kwargs): return {}
    def send_ml_learning_report(*args, **kwargs): return None
    def create_ql_agent(*args, **kwargs): return None
    def create_lstm_predictor(*args, **kwargs): return None
    def create_auto_feature_engineer(*args, **kwargs): return None
    def create_bayesian_optimizer(*args, **kwargs): return None
    def create_portfolio_optimizer(*args, **kwargs): return None
    def create_seasonality_detector(*args, **kwargs): return None
    def create_shap_explainer(*args, **kwargs): return None
    def create_causal_impact_analyzer(*args, **kwargs): return None
    def create_ml_decision_engine(*args, **kwargs): return None

# Legacy modules (updated for ML integration)
from infrastructure import Store
from integrations import notify, post_run_header_and_get_thread_ts, post_thread_ads_snapshot, prettify_ad_name, fmt_eur, fmt_pct, fmt_roas, fmt_int
from integrations import MetaClient, AccountAuth, ClientConfig
from rules.rules import AdvancedRuleEngine as RuleEngine
from stages.testing import run_testing_tick
from stages.validation import run_validation_tick
from stages.scaling import run_scaling_tick
from infrastructure import now_local, getenv_f, getenv_i, getenv_b, cfg, cfg_or_env_f, cfg_or_env_i, cfg_or_env_b, cfg_or_env_list, safe_f, today_str, daily_key, ad_day_flag_key, now_minute_key, clean_text_token, prettify_ad_name
from infrastructure import start_background_scheduler, stop_background_scheduler, get_scheduler

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

def store_performance_data_in_supabase(supabase_client, ad_data: Dict[str, Any], stage: str) -> None:
    """Store performance data in Supabase for ML system."""
    if not supabase_client:
        print("⚠️ No Supabase client available")
        return
    
    try:
        
        # Test Supabase connection first
        try:
            test_result = supabase_client.table('performance_metrics').select('*').limit(1).execute()
        except Exception as e:
            notify(f"❌ Supabase connection failed: {e}")
            return
        
        # Store in performance_metrics table
        performance_data = {
            'ad_id': ad_data.get('ad_id', ''),
            'lifecycle_id': ad_data.get('lifecycle_id', ''),
            'stage': stage,
            'window_type': '1d',
            'date_start': ad_data.get('date_start', ''),
            'date_end': ad_data.get('date_end', ''),
            'spend': float(ad_data.get('spend', 0)),
            'impressions': int(ad_data.get('impressions', 0)),
            'clicks': int(ad_data.get('clicks', 0)),
            'ctr': float(ad_data.get('ctr', 0)),
            'cpc': float(ad_data.get('cpc', 0)),
            'cpm': float(ad_data.get('cpp', 0)),
            'purchases': int(ad_data.get('purchases', 0)),
            'add_to_cart': int(ad_data.get('atc', 0)),
            'initiate_checkout': int(ad_data.get('ic', 0)),
            'roas': float(ad_data.get('roas', 0)),
            'cpa': float(ad_data.get('cpa', 0)) if ad_data.get('cpa') is not None else 0,
            # Note: created_at and updated_at are handled by database defaults
            # We don't need to insert them explicitly
        }
        
        # Insert performance data (use upsert to handle duplicates)
        result = supabase_client.table('performance_metrics').upsert(
            performance_data,
            on_conflict='ad_id,window_type,date_start'
        ).execute()
        notify(f"✅ Performance data inserted: {result}")
        
        # Store in ad_lifecycle table
        lifecycle_data = {
            'ad_id': ad_data.get('ad_id', ''),
            'creative_id': ad_data.get('creative_id', ''),
            'campaign_id': ad_data.get('campaign_id', ''),
            'adset_id': ad_data.get('adset_id', ''),
            'stage': stage,
            'status': ad_data.get('status', 'active'),
            'lifecycle_id': ad_data.get('lifecycle_id', ''),
            'metadata': ad_data.get('metadata', {})
            # Note: created_at and updated_at are handled by database defaults
        }
        
        # Insert lifecycle data (upsert to avoid duplicates)
        result = supabase_client.table('ad_lifecycle').upsert(
            lifecycle_data,
            on_conflict='ad_id,stage'
        ).execute()
        notify(f"✅ Lifecycle data inserted: {result}")
        
    except Exception as e:
        notify(f"❌ Failed to store performance data in Supabase: {e}")
        import traceback
        traceback.print_exc()

def store_ml_insights_in_supabase(supabase_client, ad_id: str, insights: Dict[str, Any]) -> None:
    """Store ML insights in Supabase."""
    if not supabase_client:
        return
    
    try:
        # Store creative intelligence data
        creative_data = {
            'creative_id': insights.get('creative_id', f'creative_{ad_id}'),
            'ad_id': ad_id,
            'creative_type': insights.get('creative_type', 'unknown'),
            'performance_score': float(insights.get('performance_score', 0)),
            'fatigue_index': float(insights.get('fatigue_index', 0)),
            'similarity_vector': insights.get('similarity_vector', []),
            'metadata': insights.get('metadata', {})
            # Note: created_at and updated_at are handled by database defaults
        }
        
        supabase_client.table('creative_intelligence').upsert(creative_data).execute()
        
    except Exception as e:
        print(f"⚠️ Failed to store ML insights in Supabase: {e}")


def store_timeseries_data_in_supabase(supabase_client, ad_id: str, ad_data: Dict[str, Any], stage: str) -> None:
    """Store hourly time-series data in Supabase for temporal modeling (NEW)."""
    if not supabase_client:
        return
    
    try:
        # Store current metrics as time-series data point
        from datetime import datetime
        
        metrics_to_track = ['ctr', 'cpa', 'roas', 'spend', 'purchases', 'cpc', 'cpm']
        
        for metric_name in metrics_to_track:
            metric_value = ad_data.get(metric_name)
            if metric_value is not None:
                timeseries_data = {
                    'ad_id': ad_id,
                    'lifecycle_id': ad_data.get('lifecycle_id', ''),
                    'stage': stage,
                    'metric_name': metric_name,
                    'metric_value': float(metric_value),
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'impressions': ad_data.get('impressions', 0),
                        'clicks': ad_data.get('clicks', 0)
                    }
                }
                
                supabase_client.table('time_series_data').insert(timeseries_data).execute()
        
        notify(f"📊 Time-series data stored for {ad_id}: {len(metrics_to_track)} metrics")
        
    except Exception as e:
        print(f"⚠️ Failed to store time-series data: {e}")


def store_creative_data_in_supabase(supabase_client, meta_client, ad_id: str, stage: str) -> None:
    """Fetch and store creative intelligence data from Meta API (NEW)."""
    if not supabase_client or not meta_client:
        return
    
    try:
        # Fetch creative data from Meta API
        try:
            ad = meta_client.api.call(
                'GET',
                (ad_id,),
                params={'fields': 'creative,name'}
            )
            
            creative_id = ad.get('creative', {}).get('id') if isinstance(ad.get('creative'), dict) else ad.get('creative')
            
            if not creative_id:
                return
            
            # Fetch creative details
            creative = meta_client.api.call(
                'GET',
                (creative_id,),
                params={'fields': 'title,body,image_url,video_id,object_type'}
            )
            
            # Store in creative_intelligence
            creative_data = {
                'creative_id': str(creative_id),
                'ad_id': ad_id,
                'creative_type': creative.get('object_type', 'unknown'),
                'performance_score': 0.5,  # Will be updated by ML
                'fatigue_index': 0.0,  # Will be calculated by ML
                'similarity_vector': [],  # Will be calculated by sentence-transformers
                'metadata': {
                    'title': creative.get('title', ''),
                    'body': creative.get('body', ''),
                    'stage': stage,
                    'ad_name': ad.get('name', '')
                }
            }
            
            supabase_client.table('creative_intelligence').upsert(creative_data, on_conflict='creative_id').execute()
            notify(f"🎨 Creative data stored for {ad_id}")
            
        except Exception as e:
            # Silently fail - creative data is optional
            pass
        
    except Exception as e:
        print(f"⚠️ Failed to store creative data: {e}")


def setup_supabase_table():
    """
    Helper function to create the required Supabase table schema.
    Run this once to set up your Supabase table with the correct columns.
    """
    sb = _get_supabase()
    if not sb:
        notify("❌ Supabase client not available. Check your SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.")
        return False
    
    table = os.getenv("SUPABASE_TABLE", "meta_creatives")
    
    # SQL to create the table with required columns
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id SERIAL PRIMARY KEY,
        video_id TEXT,
        filename TEXT,
        avatar TEXT,
        visual_style TEXT,
        script TEXT,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """
    
    try:
        # Note: This requires SQL execution permissions in Supabase
        # You might need to run this manually in your Supabase SQL editor
        notify(f"📋 To set up your Supabase table, run this SQL in your Supabase SQL editor:")
        notify(f"```sql")
        notify(f"{create_table_sql}")
        notify(f"```")
        notify(f"📝 Table name: {table}")
        notify(f"🔑 Required columns: id, video_id, filename, avatar, visual_style, script, status")
        return True
    except Exception as e:
        notify(f"❗ Failed to create table: {e}")
        return False


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
        # Build select query with required columns
        select_columns = ["id", "video_id", "filename", "avatar", "visual_style", "script", "status"]
        select_str = ", ".join(select_columns)
        
        q = (
            sb.table(table)
            .select(select_str)
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

    # Queue loaded silently
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
    cfg_tz = settings.get("account", {}).get("timezone") or settings.get("account_timezone") or settings.get("timezone") or DEFAULT_TZ
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
    Cross-platform run lock:
    - POSIX (macOS/Linux): flock
    - Windows: presence + PID written, best-effort
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
        # Silent health check - no notification
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


def check_ad_account_health(client: MetaClient, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive ad account health check with alerting for critical issues.
    Uses configuration settings to determine alert thresholds and types.
    """
    # Check if account health monitoring is enabled
    account_health_config = settings.get("account_health", {})
    if not account_health_config.get("enabled", True):
        return {"ok": True, "disabled": True}
    
    try:
        health_result = client.check_account_health()
        
        if not health_result["ok"]:
            # Critical issues detected
            critical_issues = health_result.get("critical_issues", [])
            account_id = client.ad_account_id_act
            
            # Send critical alerts
            from integrations import alert_ad_account_health_critical, alert_payment_issue, alert_account_balance_low
            
            alert_ad_account_health_critical(account_id, critical_issues)
            
            # Check for specific payment issues
            health_details = health_result.get("health_details", {})
            if health_details.get("payment_status") == "failed":
                payment_details = health_details.get("funding_source", {})
                alert_payment_issue(account_id, "Payment Failed", str(payment_details))
            
            # Check for low balance using configured thresholds
            balance = health_details.get("balance")
            if balance is not None:
                currency = settings.get("account", {}).get("currency") or settings.get("economics", {}).get("currency", "EUR")
                critical_threshold = account_health_config.get("thresholds", {}).get("balance_critical_eur", 0.0)
                if balance <= critical_threshold:
                    alert_account_balance_low(account_id, balance, currency)
            
            return {"ok": False, "critical_issues": critical_issues, "health_details": health_details}
        
        else:
            # Check for warnings using configured thresholds
            warnings = health_result.get("warnings", [])
            health_details = health_result.get("health_details", {})
            account_id = client.ad_account_id_act
            currency = settings.get("economics", {}).get("currency", "EUR")
            
            # Check spend cap warnings
            spent = health_details.get("amount_spent")
            cap = health_details.get("spend_cap")
            if spent is not None and cap is not None and cap > 0:
                percentage = (spent / cap) * 100
                warning_threshold = account_health_config.get("thresholds", {}).get("spend_cap_warning_pct", 80)
                if percentage >= warning_threshold:
                    from integrations import alert_spend_cap_approaching
                    alert_spend_cap_approaching(account_id, spent, cap, currency)
            
            # Check balance warnings - alert when approaching auto-charge threshold
            balance = health_details.get("balance")
            if balance is not None:
                # Try to get auto-charge threshold from Meta's billing API first
                auto_charge_threshold = health_details.get("auto_charge_threshold")
                
                if auto_charge_threshold is None:
                    # Use dynamic threshold tracking system
                    from infrastructure import Store
                    # Use the same SQLite path as the main system
                    sqlite_path = settings.get("logging", {}).get("sqlite", {}).get("path", "dean/data/state.sqlite")
                    store = Store(sqlite_path)
                    
                    # Get current tracked threshold from storage
                    current_threshold = store.get_state("auto_charge_threshold_eur")
                    if current_threshold is None:
                        # Initialize with configured threshold
                        current_threshold = account_health_config.get("thresholds", {}).get("auto_charge_threshold_eur", 75.0)
                        store.set_state("auto_charge_threshold_eur", current_threshold)
                    
                    auto_charge_threshold = current_threshold
                    
                    # Check if balance has hit the current threshold (indicating a charge occurred)
                    if balance >= auto_charge_threshold:
                        # Increase threshold by €5 for next charge
                        new_threshold = auto_charge_threshold + 5.0
                        store.set_state("auto_charge_threshold_eur", new_threshold)
                        auto_charge_threshold = new_threshold
                        
                        # Send notification about threshold update
                        from integrations import alert_threshold_updated
                        alert_threshold_updated(account_id, new_threshold, currency)
                
                warning_buffer = account_health_config.get("thresholds", {}).get("balance_warning_buffer_eur", 10.0)
                warning_threshold = auto_charge_threshold - warning_buffer
                
                # Alert when balance is HIGH (approaching auto-charge threshold)
                if balance >= warning_threshold:
                    from integrations import alert_account_balance_low
                    alert_account_balance_low(account_id, balance, currency, auto_charge_threshold)
            
            # Send general warnings if any
            if warnings:
                from integrations import alert_ad_account_health_warning
                alert_ad_account_health_warning(account_id, warnings)
            
            return {"ok": True, "warnings": warnings, "health_details": health_details}
    
    except Exception as e:
        # If health check fails, it's a critical issue
        account_id = getattr(client, 'ad_account_id_act', 'unknown')
        from integrations import alert_ad_account_health_critical
        alert_ad_account_health_critical(account_id, [f"Health check failed: {str(e)}"])
        return {"ok": False, "error": str(e)}


def account_guardrail_ping(meta: MetaClient, settings: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Get today-only comprehensive metrics
        from datetime import datetime
        import zoneinfo
        
        # Use account timezone for today's data
        tz_name = settings.get("account", {}).get("timezone", "Europe/Amsterdam")
        local_tz = zoneinfo.ZoneInfo(tz_name)
        now = datetime.now(local_tz)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        rows = meta.get_ad_insights(
            level="ad", 
            fields=["spend", "actions", "impressions", "clicks", "ctr", "cpc", "cpp"], 
            time_range={
                "since": midnight.strftime("%Y-%m-%d"),
                "until": now.strftime("%Y-%m-%d")
            },
            paginate=True
        )
        
        spend = sum(float(r.get("spend") or 0) for r in rows)
        impressions = sum(int(r.get("impressions") or 0) for r in rows)
        clicks = sum(int(r.get("clicks") or 0) for r in rows)
        purch = 0.0
        atc = 0.0  # Add to cart
        ic = 0.0   # Initiate checkout
        
        for r in rows:
            for a in (r.get("actions") or []):
                action_type = a.get("action_type")
                try:
                    value = float(a.get("value") or 0)
                    if action_type == "purchase":
                        purch += value
                    elif action_type == "add_to_cart":
                        atc += value
                    elif action_type == "initiate_checkout":
                        ic += value
                except Exception:
                    pass
        
        # Calculate metrics
        cpa = (spend / purch) if purch > 0 else float("inf")
        ctr = (clicks / impressions * 100) if impressions > 0 else 0
        cpc = (spend / clicks) if clicks > 0 else 0
        
        be = float(
            os.getenv("BREAKEVEN_CPA")
            or (settings.get("economics", {}) or {}).get("breakeven_cpa")
            or 27.51
        )
        
        return {
            "spend": round(spend, 2),
            "purchases": int(purch),
            "cpa": None if cpa == float("inf") else round(cpa, 2),
            "breakeven": be,
            "impressions": impressions,
            "clicks": clicks,
            "ctr": round(ctr, 2),
            "cpc": round(cpc, 2),
            "atc": int(atc),
            "ic": int(ic),
        }
    except Exception:
        return {
            "spend": None, "purchases": None, "cpa": None, "breakeven": None,
            "impressions": None, "clicks": None, "ctr": None, "cpc": None,
            "atc": None, "ic": None
        }


# ------------------------------- Summaries ----------------------------------


def summarize_counts(label: str, summary: Optional[Dict[str, Any]]) -> str:
    if not summary:
        return f"{label}: n/a"
    return f"{label}: " + ", ".join(f"{k}={v}" for k, v in summary.items())


# --------------------------------- Main -------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brava - Continuous Creative Testing & Scaling"
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
    parser.add_argument("--continuous-mode", action="store_true", help="Continuous operation mode (DigitalOcean)")
    parser.add_argument(
        "--simulate", action="store_true", help="shadow mode: log intended actions only"
    )
    parser.add_argument("--since", default=None, help="simulation since (YYYY-MM-DD)")
    parser.add_argument("--until", default=None, help="simulation until (YYYY-MM-DD)")
    parser.add_argument(
        "--explain", action="store_true", help="print decisions without acting"
    )
    parser.add_argument(
        "--background", action="store_true", help="run in background mode with automated scheduling"
    )
    parser.add_argument(
        "--ml-mode", action="store_true", default=True, help="enable ML-enhanced mode (requires Supabase)"
    )
    parser.add_argument(
        "--no-ml", action="store_true", help="disable ML mode and use legacy system"
    )
    args = parser.parse_args()

    # Load environment first (for dynamic .env overrides)
    load_dotenv()

    # Load config and rules
    settings, rules_cfg = load_cfg(args.settings, args.rules)
    
    # Load production config if profile is production
    production_cfg = {}
    if args.profile == "production":
        production_config_path = "config/production.yaml"
        if os.path.exists(production_config_path):
            production_cfg = load_yaml(production_config_path)
            # Merge production config into settings (production config takes precedence)
            if production_cfg:
                settings.update(production_cfg)
    
    # Determine ML mode (default enabled, can be disabled with --no-ml)
    # Check production config first, then args
    ml_mode_enabled = production_cfg.get("ml_system", {}).get("enabled", True) if production_cfg else True
    if args.no_ml:
        ml_mode_enabled = False
    ml_mode_enabled = ml_mode_enabled and args.ml_mode
    
    # Check for ML mode
    if ml_mode_enabled:
        if not ML_AVAILABLE:
            notify("❌ ML mode requires ML packages to be installed")
            notify("   Install OpenMP runtime: brew install libomp")
            notify("   Then reinstall XGBoost: pip install xgboost")
            notify("   Falling back to legacy system...")
            ml_mode_enabled = False
        elif not (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY")):
            notify("❌ ML mode requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")
            notify("   Falling back to legacy system...")
            ml_mode_enabled = False
        else:
            notify("🤖 ML-Enhanced mode enabled - using advanced intelligence system")
    
    if not ml_mode_enabled:
        notify("📊 Legacy mode enabled - using standard automation system")
    
    # Continuous mode setup (24/7 DigitalOcean optimization)
    if args.continuous_mode:
        notify("🔄 24/7 Continuous mode enabled - optimized for DigitalOcean deployment")
        notify("📊 Maximum UI protection and ML data feeding active")
        notify("🛡️ Single concurrent request for UI protection")
        
        # Adjust rate limiting for 24/7 continuous operation
        os.environ["META_REQUEST_DELAY"] = "2.0"  # Very slow requests
        os.environ["META_PEAK_HOURS_DELAY"] = "3.0"  # Even slower during peak hours
        os.environ["META_NIGHT_HOURS_DELAY"] = "1.5"  # Slightly faster at night
        os.environ["META_MAX_CONCURRENT_INSIGHTS"] = "1"  # SINGLE concurrent request
        os.environ["META_RETRY_MAX"] = "12"  # More retries
        os.environ["META_BACKOFF_BASE"] = "2.0"  # Stronger exponential backoff
        os.environ["META_USAGE_THRESHOLD"] = "0.6"  # Very conservative
        os.environ["META_EMERGENCY_THRESHOLD"] = "0.8"  # Emergency stop
        os.environ["META_UI_PROTECTION_MODE"] = "true"  # Maximum UI protection
    
    # Merge rules configuration into settings so stages can access it
    if rules_cfg:
        settings.update(rules_cfg)

    # Resolve profile/dry-run/shadow
    profile = (
        args.profile
        or production_cfg.get("profile", {}).get("name")
        or settings.get("mode", {}).get("current")
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

    # Store (SQLite) - path from settings
    sqlite_path = settings.get("logging", {}).get("sqlite", {}).get("path", "dean/data/state.sqlite")
    store = Store(sqlite_path)

    # Timezone for account
    tz_name = (
        settings.get("account", {}).get("timezone")
        or settings.get("account_timezone")
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
    
    # ML System initialization (if ML mode enabled)
    ml_system = None
    rule_engine_ml = None
    performance_tracker = None
    reporting_system = None
    ml_decision_engine = None
    model_validator = None
    data_progress_tracker = None
    anomaly_detector = None
    time_series_forecaster = None
    creative_similarity_analyzer = None
    causal_impact_analyzer = None
    
    if ml_mode_enabled:
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            
            # Initialize core ML system
            ml_system = create_ml_system(supabase_url, supabase_key)
            
            # Initialize intelligent rule engine
            rule_engine_ml = create_intelligent_rule_engine(supabase_url, supabase_key, ml_system)
            
            # Initialize performance tracker
            performance_tracker = create_performance_tracking_system(supabase_url, supabase_key)
            
            # Initialize reporting system
            reporting_system = create_ml_reporting_system(supabase_url, supabase_key)
            
            # Initialize ML enhancements
            ml_decision_engine = create_ml_decision_engine(ml_system, engine, confidence_threshold=0.7)
            model_validator = create_model_validator(supabase_url, supabase_key)
            data_progress_tracker = create_data_progress_tracker(supabase_url, supabase_key)
            anomaly_detector = create_anomaly_detector(supabase_url, supabase_key)
            time_series_forecaster = create_time_series_forecaster(supabase_url, supabase_key)
            creative_similarity_analyzer = create_creative_similarity_analyzer(supabase_url, supabase_key)
            causal_impact_analyzer = create_causal_impact_analyzer(supabase_url, supabase_key)
            
            # Initialize unified ML pipeline (NEW)
            ml_pipeline_config = MLPipelineConfig(
                enable_ml_decisions=True,
                enable_anomaly_detection=True,
                enable_time_series=True,
                enable_creative_similarity=True,
                enable_model_validation=True
            )
            ml_pipeline = create_ml_pipeline(
                ml_system=ml_system,
                rule_engine=rule_engine_ml,
                decision_engine=ml_decision_engine,
                performance_tracker=performance_tracker,
                reporting_system=reporting_system,
                model_validator=model_validator,
                data_tracker=data_progress_tracker,
                anomaly_detector=anomaly_detector,
                ts_forecaster=time_series_forecaster,
                similarity_analyzer=creative_similarity_analyzer,
                causal_analyzer=causal_impact_analyzer,
                config=ml_pipeline_config
            )
            
            # Initialize ML dashboard for monitoring (NEW)
            ml_dashboard = create_ml_dashboard(supabase_url, supabase_key)
            
            # Initialize advanced ML features (NEW) - Optional features
            try:
                from ml.ml_advanced_features import (
                    create_rl_agent, create_portfolio_optimizer, 
                    create_seasonality_analyzer, create_lr_scheduler
                )
                rl_agent = create_rl_agent(learning_rate=0.1, exploration_rate=0.1)
                portfolio_optimizer = create_portfolio_optimizer()
                seasonality_analyzer = create_seasonality_analyzer()
                lr_scheduler = create_lr_scheduler(initial_lr=0.1, schedule_type='cosine')
            except ImportError:
                # Advanced features not available, continue without them
                rl_agent = None
                portfolio_optimizer = None
                seasonality_analyzer = None
                lr_scheduler = None
            
            # Show ML readiness for each stage
            if data_progress_tracker:
                for stage_name in ['testing', 'validation', 'scaling']:
                    readiness = data_progress_tracker.get_ml_readiness(stage_name)
                    notify(f"📊 {stage_name.title()}: {readiness.get('message', 'Unknown')}")
            
            # Show ML system health
            if ml_dashboard:
                health = ml_dashboard.get_health_metrics(hours_back=24)
                notify(f"🤖 ML Health: {health.avg_model_accuracy:.1%} accuracy, {health.predictions_made_24h} predictions/24h")
            
            notify("✅ ML system initialized successfully (20 enhancements active)")
            
        except Exception as e:
            notify(f"❌ Failed to initialize ML system: {e}")
            notify("   Falling back to legacy system...")
            ml_mode_enabled = False

    # Preflight health check
    hc = health_check(store, client)
    if not hc["ok"]:
        notify("🛑 Preflight failed: " + " ".join(hc["details"]))
        if not (profile == "staging" or effective_dry):
            sys.exit(1)
    
    # Ad account health check with alerting
    account_health = check_ad_account_health(client, settings)
    if not account_health["ok"]:
        notify("🚨 Ad account health issues detected - check alerts for details")
        # Don't exit on account health issues, but log them
        if account_health.get("critical_issues"):
            notify(f"Critical issues: {', '.join(account_health['critical_issues'])}")

    # Queue: Prefer Supabase if configured; else fallback to file path.
    if os.getenv("SUPABASE_URL") and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")):
        table = os.getenv("SUPABASE_TABLE", "meta_creatives")
        queue_df = load_queue_supabase(table=table, status_filter="pending", limit=64)
        queue_len_before = len(queue_df)
        notify(f"📊 Queue loaded from Supabase: {len(queue_df)} creatives available")
    else:
        queue_path = (settings.get("queue") or {}).get("path", "data/creatives_queue.csv")
        queue_df = load_queue(queue_path)
        queue_len_before = len(queue_df)
        notify(f"📊 Queue loaded from file: {len(queue_df)} creatives available")

    # Context ping - now using consolidated messaging
    local_now = now_local(tz_name)
    acct = account_guardrail_ping(client, settings)
    
    # Store account info for later use in consolidated message
    account_info = {
        'spend': acct.get('spend', 0.0),
        'purchases': acct.get('purchases', 0),
        'cpa': acct.get('cpa'),
        'breakeven': acct.get('breakeven'),
        'impressions': acct.get('impressions', 0),
        'clicks': acct.get('clicks', 0),
        'ctr': acct.get('ctr'),
        'cpc': acct.get('cpc'),
        'atc': acct.get('atc', 0),
        'ic': acct.get('ic', 0),
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
                # Stage success notifications removed - now handled in consolidated message
                return res
            except Exception as e:
                failures_in_row += 1
                dt = time.time() - t0
                notify(f"❌ [{label}] {dt:.1f}s - {e}")
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
            # Run standard testing first to collect data
            overall["testing"] = run_stage(
                run_testing_tick,
                "TESTING",
                client,
                settings,
                engine,
                store,
                queue_df,
                set_supabase_status,
                placements=["facebook", "instagram"],
                instagram_actor_id=os.getenv("IG_ACTOR_ID"),
                ml_pipeline=ml_pipeline if ml_mode_enabled else None,  # NEW: Pass ML pipeline
            )
            
            # After data collection, run ML analysis if enabled
            if ml_mode_enabled and ml_system and overall["testing"]:
                try:
                    # Store performance data in Supabase first
                    supabase_client = _get_supabase()
                    if supabase_client:
                        notify(f"📊 Testing data collected: {len(overall['testing'])} items")
                        for ad_id, ad_data in overall["testing"].items():
                            if isinstance(ad_data, dict) and 'spend' in ad_data:
                                store_performance_data_in_supabase(supabase_client, ad_data, "testing")
                        notify("📊 Performance data stored in Supabase for ML system")
                    
                    # Now run ML analysis on the stored data
                    testing_analysis = ml_system.analyze_ad_intelligence("testing_stage", "testing")
                    
                    # Add ML insights to results
                    overall["testing"]["ml_insights"] = testing_analysis
                    overall["testing"]["intelligence_score"] = testing_analysis.get("intelligence_score", 0)
                    
                    # Store ML insights
                    if supabase_client and testing_analysis:
                        for ad_id in overall["testing"].keys():
                            if isinstance(ad_id, str):
                                store_ml_insights_in_supabase(supabase_client, ad_id, testing_analysis)
                                
                    # ML analysis completed - status will be reported later
                    
                except Exception as e:
                    notify(f"⚠️ ML analysis failed: {e}")
            # Store performance data in Supabase for ML system (standard path)
            if overall["testing"] and not (ml_mode_enabled and ml_system):
                supabase_client = _get_supabase()
                if supabase_client:
                    try:
                        # Store testing stage performance data
                        for ad_id, ad_data in overall["testing"].items():
                            if isinstance(ad_data, dict) and 'spend' in ad_data:
                                store_performance_data_in_supabase(supabase_client, ad_data, "testing")
                                # NEW: Store time-series and creative data
                                store_timeseries_data_in_supabase(supabase_client, ad_id, ad_data, "testing")
                                store_creative_data_in_supabase(supabase_client, client, ad_id, "testing")
                                
                        notify("📊 Performance data + time-series + creative data stored in Supabase")
                    except Exception as e:
                        notify(f"⚠️ Failed to store data in Supabase: {e}")
            
            if overall["testing"]:
                stage_summaries.append({
                    "stage": "TEST",
                    "counts": overall["testing"]
                })

        if stage_choice in ("all", "validation"):
            overall["validation"] = run_stage(
                run_validation_tick, "VALIDATION", client, settings, engine, store, ml_pipeline=ml_pipeline if ml_mode_enabled else None  # NEW: Pass ML pipeline
            )
            if overall["validation"]:
                # Store validation stage performance data in Supabase
                supabase_client = _get_supabase()
                if supabase_client:
                    try:
                        for ad_id, ad_data in overall["validation"].items():
                            if isinstance(ad_data, dict) and 'spend' in ad_data:
                                store_performance_data_in_supabase(supabase_client, ad_data, "validation")
                                # NEW: Store time-series and creative data
                                store_timeseries_data_in_supabase(supabase_client, ad_id, ad_data, "validation")
                                store_creative_data_in_supabase(supabase_client, client, ad_id, "validation")
                        notify("📊 Validation data + time-series + creative data stored in Supabase")
                    except Exception as e:
                        notify(f"⚠️ Failed to store validation data in Supabase: {e}")
                        
                stage_summaries.append({
                    "stage": "VALID", 
                    "counts": overall["validation"]
                })

        if stage_choice in ("all", "scaling"):
            overall["scaling"] = run_stage(run_scaling_tick, "SCALING", client, settings, store)
            if overall["scaling"]:
                # Store scaling stage performance data in Supabase
                supabase_client = _get_supabase()
                if supabase_client:
                    try:
                        for ad_id, ad_data in overall["scaling"].items():
                            if isinstance(ad_data, dict) and 'spend' in ad_data:
                                store_performance_data_in_supabase(supabase_client, ad_data, "scaling")
                                # NEW: Store time-series and creative data
                                store_timeseries_data_in_supabase(supabase_client, ad_id, ad_data, "scaling")
                                store_creative_data_in_supabase(supabase_client, client, ad_id, "scaling")
                        notify("📊 Scaling data + time-series + creative data stored in Supabase")
                    except Exception as e:
                        notify(f"⚠️ Failed to store scaling data in Supabase: {e}")
                        
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
    
    # Run model validation if needed (NEW - weekly validation)
    if ml_mode_enabled and ml_pipeline:
        try:
            validation_results = ml_pipeline.validate_models_if_needed()
            if validation_results:
                notify("🔬 Model validation completed")
                # Alert if any model has low accuracy
                for model_name, metrics in validation_results.items():
                    accuracy = metrics.get('accuracy', 0)
                    if accuracy < 0.6:
                        notify(f"⚠️ Model {model_name} accuracy: {accuracy:.1%} - retraining recommended")
        except Exception as e:
            notify(f"⚠️ Model validation error: {e}")
    
    # Generate ML dashboard summary (NEW)
    if ml_mode_enabled and ml_dashboard:
        try:
            summary = ml_dashboard.generate_summary_report()
            notify(summary)
        except Exception as e:
            notify(f"⚠️ ML dashboard error: {e}")

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
            stage_summaries=stage_summaries,
            impressions=account_info['impressions'],
            clicks=account_info['clicks'],
            ctr=account_info['ctr'],
            cpc=account_info['cpc'],
            atc=account_info['atc'],
            ic=account_info['ic']
        )
        
        # Collect ad insights and post as thread reply
        try:
            if stage_choice in ("all", "testing"):
                # Get today's insights with local timezone
                from datetime import datetime, timezone
                import zoneinfo
                
                # Define attribution windows for consistency
                attr_windows = ["7d_click", "1d_view"]
                
                local_tz = zoneinfo.ZoneInfo(tz_name)
                now = datetime.now(local_tz)
                midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Get today's data for ACTIVE ads only
                rows_today = client.get_ad_insights(
                    level="ad",
                    filtering=[
                        {"field": "adset.id", "operator": "IN", "value": [settings.get("ids", {}).get("testing_adset_id")]},
                        {"field": "ad.status", "operator": "IN", "value": ["ACTIVE"]}
                    ],
                    fields=["ad_id", "ad_name", "spend", "actions"],
                    time_range={
                        "since": midnight.strftime("%Y-%m-%d"),
                        "until": now.strftime("%Y-%m-%d")
                    },
                    action_attribution_windows=list(attr_windows),
                    paginate=True
                )
                
                # Get lifetime data
                rows_lifetime = client.get_ad_insights(
                    level="ad",
                    filtering=[{"field": "adset.id", "operator": "IN", "value": [settings.get("ids", {}).get("testing_adset_id")]}],
                    fields=["ad_id", "spend", "actions"],
                    time_range={
                        "since": "2024-01-01",  # Far back enough to capture all lifetime
                        "until": now.strftime("%Y-%m-%d")
                    },
                    action_attribution_windows=list(attr_windows),
                    paginate=True
                )
                
                # Build snapshot using helper
                from integrations import build_ads_snapshot
                ad_lines = build_ads_snapshot(rows_today or [], rows_lifetime or [], tz_name)
                
                if ad_lines:
                    post_thread_ads_snapshot(thread_ts, ad_lines)
        except Exception:
            pass

    # ML Model Training (if ML mode enabled and data is available)
    if ml_mode_enabled and ml_system:
        try:
            # Small delay to ensure data is fully committed to Supabase
            time.sleep(2)
            
            # Train testing stage models directly
            perf_success = ml_system.predictor.train_model('performance_predictor', 'testing', 'cpa')
            roas_success = ml_system.predictor.train_model('roas_predictor', 'testing', 'roas')
            purchase_success = ml_system.predictor.train_model('purchase_probability', 'testing', 'purchases')
            
            training_success = perf_success or roas_success or purchase_success
            
            if training_success:
                # Get learning summary
                try:
                    from ml.ml_monitoring import get_ml_learning_summary
                    summary = get_ml_learning_summary(supabase_client)
                    if summary["recent_training"]:
                        training_info = ", ".join([f"{t['type']}({t['stage']})" for t in summary["recent_training"]])
                        notify(f"🧠 ML Learning: Trained {training_info} | Models: {summary['active_models']} | Data: {summary['data_points_24h']}")
                    else:
                        notify("🧠 ML models trained successfully")
                except:
                    notify("🧠 ML models trained successfully")
            else:
                notify("⚠️ ML model training failed")
        except Exception as e:
            notify(f"❌ ML training error: {e}")

    # ML Learning Report (enhanced diagnostics)
    if ml_mode_enabled and supabase_client:
        try:
            from ml.ml_monitoring import send_ml_learning_report
            send_ml_learning_report(supabase_client, notify)
        except Exception as e:
            notify(f"❌ ML status error: {e}")

    # Console summary (logs only, not Slack)
    from datetime import datetime
    import pytz
    amsterdam_tz = pytz.timezone('Europe/Amsterdam')
    amsterdam_time = datetime.now(amsterdam_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    print("---- RUN SUMMARY ----")
    print(f"Time: {amsterdam_time}")
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

    # Execution completed (no need for redundant notification)
    
    # Background mode handling (optional)
    if args.background:
        notify("🤖 Starting background scheduler mode")
        start_background_scheduler(settings, rules_cfg, store)
        
        try:
            # Keep the process running
            import signal
            def signal_handler(sig, frame):
                notify("🛑 Background scheduler stopping...")
                stop_background_scheduler()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Keep running
            while True:
                time.sleep(60)
                
        except KeyboardInterrupt:
            notify("🛑 Background scheduler stopped by user")
            stop_background_scheduler()
        except Exception as e:
            notify(f"❌ Background scheduler error: {e}")
            stop_background_scheduler()
            sys.exit(1)


if __name__ == "__main__":
    main()
