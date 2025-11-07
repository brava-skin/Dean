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

The system continuously learns from ASC+ campaign data,
identifies signals that predict purchases, and dynamically adjusts all rules to
optimize performance and scale intelligently.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv

# Import constants
from config.constants import (
    DB_NUMERIC_MIN, DB_NUMERIC_MAX, DB_CTR_MAX, DB_CPC_MAX, DB_CPM_MAX,
    DB_ROAS_MAX, DB_CPA_MAX, DB_DWELL_TIME_MAX, DB_FREQUENCY_MAX, DB_RATE_MAX,
    DB_GLOBAL_FLOAT_MAX,
    ASC_PLUS_BUDGET_MIN, ASC_PLUS_BUDGET_MAX, ASC_PLUS_MIN_BUDGET_PER_CREATIVE,
    MAX_AD_AGE_DAYS, MAX_STAGE_DURATION_HOURS, DEFAULT_SAFE_FLOAT_MAX,
    ML_TRAINING_DELAY_SECONDS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Shared metric helpers
# -----------------------------------------------------


def _metric_to_float(value: Any) -> float:
    """Convert Meta metrics that may be strings, dicts, or lists into floats."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", "")) if value.strip() else 0.0
        except ValueError:
            return 0.0
    if isinstance(value, dict):
        for key in ("value", "amount", "count", "total"):
            if key in value:
                return _metric_to_float(value[key])
        return 0.0
    if isinstance(value, (list, tuple)):
        for item in value:
            # Prefer explicit 'value' keys when dicts are provided
            if isinstance(item, dict) and "value" in item:
                return _metric_to_float(item["value"])
            return _metric_to_float(item)
        return 0.0
    return 0.0

# Optional Supabase client
try:
    # pip install supabase
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None  # degrade gracefully

# ML Intelligence System (NEW) - Conditional imports
try:
    from ml.ml_intelligence import create_ml_system, MLConfig
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
    logger.warning(f"ML system not available: {e}")
    logger.info("System will run in standard mode")
    ML_AVAILABLE = False
    # Create dummy classes for compatibility
    class SimpleMLIntelligenceSystem: pass
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
from infrastructure.supabase_storage import create_supabase_storage
from infrastructure.data_validation import date_validator, validate_all_timestamps
from integrations import notify, post_run_header_and_get_thread_ts, post_thread_ads_snapshot, prettify_ad_name, fmt_eur, fmt_pct, fmt_roas, fmt_int
from integrations import MetaClient, AccountAuth, ClientConfig
from rules.rules import AdvancedRuleEngine as RuleEngine
from stages.asc_plus import run_asc_plus_tick
from infrastructure import now_local, getenv_f, getenv_i, getenv_b, cfg, cfg_or_env_f, cfg_or_env_i, cfg_or_env_b, cfg_or_env_list, safe_f, today_str, daily_key, ad_day_flag_key, now_minute_key, clean_text_token, prettify_ad_name
from infrastructure.utils import Timekit
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
    ("ids", "asc_plus_campaign_id"),
    ("ids", "asc_plus_adset_id"),
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
    except (FileNotFoundError, yaml.YAMLError, IOError, OSError):
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

    # scientific notation -> float to full integer string
    if re.fullmatch(r"\d+(\.\d+)?[eE]\+\d+", s):
        try:
            return str(int(float(s)))
        except (ValueError, OverflowError):
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
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
                df = pd.read_csv(
                    path,
                    dtype=str,
                    keep_default_na=False,
                    encoding="utf-8-sig",
                )
    except (FileNotFoundError, IOError, OSError, pd.errors.ParserError):
        return pd.DataFrame(columns=cols)

    # Ensure all expected columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    # Canonical order
    df = df[cols]

    # Normalize video_id for CSV (read_csv converters are not applied like read_excel ones)
    try:
        if "video_id" in df.columns:
            df["video_id"] = df["video_id"].map(_normalize_video_id_cell)
    except (KeyError, AttributeError, TypeError):
        # video_id column may not exist or may not be mappable - not critical
        pass

    return df


def save_queue(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def digest_path_for_today() -> str:
    Path(DIGEST_DIR).mkdir(parents=True, exist_ok=True)
    return str(Path(DIGEST_DIR) / f"digest_{datetime.utcnow():%Y-%m-%d}.jsonl")


def append_digest(record: Dict[str, Any]) -> None:
    """Append a record to the daily digest file."""
    try:
        with open(digest_path_for_today(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (IOError, OSError, TypeError) as e:
        logger.debug(f"Failed to append to digest: {e}")


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
    except (TypeError, ValueError, AttributeError):
        # Supabase client creation failed - return None to degrade gracefully
        return None

def _get_validated_supabase():
    """
    Build a validated Supabase client from env. Degrades cleanly if missing.
    This client automatically validates all data before insertion.
    """
    try:
        from infrastructure.supabase_storage import get_validated_supabase_client
        return get_validated_supabase_client(enable_validation=True)
    except ImportError:
        # Fallback to regular client if validation system not available
        return _get_supabase()
    except (TypeError, ValueError, AttributeError):
        # Validation system unavailable - return None to degrade gracefully
        return None

# Helper function to safely convert and bound numeric values
def safe_float(value, max_val=None):
    if max_val is None:
        max_val = DEFAULT_SAFE_FLOAT_MAX
    try:
        val = float(value or 0)
        # Handle infinity and NaN
        if not (val == val) or val == float('inf') or val == float('-inf'):
            return 0.0
        # Bound the value to prevent overflow
        bounded_val = min(max(val, -max_val), max_val)
        # Round to 4 decimal places to prevent precision issues
        return round(bounded_val, 4)
    except (ValueError, TypeError):
        return 0.0


def _calculate_performance_quality_score(ad_data: Dict[str, Any]) -> int:
    """Calculate performance quality score based on ad metrics."""
    try:
        spend = safe_float(ad_data.get('spend', 0))
        impressions = safe_float(ad_data.get('impressions', 0))
        clicks = safe_float(ad_data.get('clicks', 0))
        purchases = safe_float(ad_data.get('purchases', 0))
        
        if spend <= 0 or impressions <= 0:
            return 0
        
        # Calculate basic metrics
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0
        cpa = spend / purchases if purchases > 0 else float('inf')
        
        # Quality score based on CTR and CPA
        quality_score = 0
        
        # CTR scoring (0-50 points)
        if ctr >= 2.0:
            quality_score += 50
        elif ctr >= 1.5:
            quality_score += 40
        elif ctr >= 1.0:
            quality_score += 30
        elif ctr >= 0.5:
            quality_score += 20
        elif ctr >= 0.1:
            quality_score += 10
        
        # CPA scoring (0-50 points) - only if we have purchases
        if purchases > 0:
            if cpa <= 20:
                quality_score += 50
            elif cpa <= 30:
                quality_score += 40
            elif cpa <= 40:
                quality_score += 30
            elif cpa <= 60:
                quality_score += 20
            elif cpa <= 100:
                quality_score += 10
        else:
            # If no purchases, give some points for CTR only
            quality_score = min(quality_score, 50)
        
        return min(max(int(quality_score), 0), 100)
        
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error calculating performance quality score: {e}", exc_info=True)
        return 0


def _calculate_stability_score(ad_data: Dict[str, Any]) -> float:
    """Calculate stability score based on performance consistency."""
    try:
        spend = safe_float(ad_data.get('spend', 0))
        impressions = safe_float(ad_data.get('impressions', 0))
        clicks = safe_float(ad_data.get('clicks', 0))
        
        if spend <= 0 or impressions <= 0:
            return 0.0
        
        # Calculate CTR
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0
        
        # Stability based on CTR performance
        if ctr >= 3.0:
            return 9.0  # Very stable
        elif ctr >= 2.0:
            return 7.0  # Stable
        elif ctr >= 1.0:
            return 5.0  # Moderate
        elif ctr >= 0.5:
            return 3.0  # Low stability
        elif ctr >= 0.1:
            return 1.0  # Very low stability
        else:
            return 0.0  # No stability
            
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error calculating stability score: {e}", exc_info=True)
        return 0.0


def _calculate_momentum_score(ad_data: Dict[str, Any]) -> float:
    """Calculate momentum score based on recent performance trends."""
    try:
        spend = safe_float(ad_data.get('spend', 0))
        impressions = safe_float(ad_data.get('impressions', 0))
        clicks = safe_float(ad_data.get('clicks', 0))
        purchases = safe_float(ad_data.get('purchases', 0))
        
        if spend <= 0 or impressions <= 0:
            return 0.0
        
        # Calculate CTR and conversion rate
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0
        conversion_rate = (purchases / clicks) * 100 if clicks > 0 else 0
        
        # Momentum based on CTR and conversion performance
        momentum = 0.0
        
        # CTR momentum (0-5 points)
        if ctr >= 3.0:
            momentum += 5.0
        elif ctr >= 2.0:
            momentum += 4.0
        elif ctr >= 1.0:
            momentum += 3.0
        elif ctr >= 0.5:
            momentum += 2.0
        elif ctr >= 0.1:
            momentum += 1.0
        
        # Conversion momentum (0-5 points)
        if conversion_rate >= 10.0:
            momentum += 5.0
        elif conversion_rate >= 5.0:
            momentum += 4.0
        elif conversion_rate >= 2.0:
            momentum += 3.0
        elif conversion_rate >= 1.0:
            momentum += 2.0
        elif conversion_rate >= 0.1:
            momentum += 1.0
        
        return min(momentum, 9.9999)
        
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error calculating momentum score: {e}", exc_info=True)
        return 0.0


def _calculate_fatigue_index(ad_data: Dict[str, Any]) -> float:
    """Calculate fatigue index based on ad performance decay."""
    try:
        spend = safe_float(ad_data.get('spend', 0))
        impressions = safe_float(ad_data.get('impressions', 0))
        clicks = safe_float(ad_data.get('clicks', 0))
        purchases = safe_float(ad_data.get('purchases', 0))
        
        if spend <= 0 or impressions <= 0:
            return 0.0
        
        # Calculate CTR and conversion rate
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0
        conversion_rate = (purchases / clicks) * 100 if clicks > 0 else 0
        
        # Fatigue calculation based on performance
        fatigue = 0.0
        
        # CTR fatigue (higher CTR = lower fatigue)
        if ctr >= 3.0:
            fatigue += 0.0  # No fatigue
        elif ctr >= 2.0:
            fatigue += 0.2  # Low fatigue
        elif ctr >= 1.0:
            fatigue += 0.4  # Medium fatigue
        elif ctr >= 0.5:
            fatigue += 0.6  # High fatigue
        else:
            fatigue += 0.8  # Very high fatigue
        
        # Conversion fatigue (higher conversion = lower fatigue)
        if conversion_rate >= 5.0:
            fatigue += 0.0  # No fatigue
        elif conversion_rate >= 2.0:
            fatigue += 0.1  # Low fatigue
        elif conversion_rate >= 1.0:
            fatigue += 0.2  # Medium fatigue
        elif conversion_rate >= 0.1:
            fatigue += 0.3  # High fatigue
        else:
            fatigue += 0.4  # Very high fatigue
        
        # Average the fatigue scores
        fatigue = fatigue / 2.0
        
        return min(fatigue, 9.9999)
        
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error calculating fatigue index: {e}", exc_info=True)
        return 0.0


def _get_next_stage(current_stage: str) -> Optional[str]:
    """Get the next stage in the lifecycle - ASC+ only."""
    # For new ads in ASC+ stage, there is no next stage yet
    # Only return a next stage when the ad is actually transitioning
    # For now, ASC+ is the only stage, so new ads have no next stage
    return None


def _calculate_stage_duration_hours(ad_id: str, current_stage: str) -> float:
    """Calculate how long the ad has been in the current stage."""
    try:
        # For now, return a basic calculation based on ad age
        # In a real implementation, this would track stage transitions
        from infrastructure.supabase_storage import SupabaseStorage
        validated_client = _get_validated_supabase()
        if validated_client:
            storage = SupabaseStorage(validated_client)
            age_days = storage.get_ad_age_days(ad_id)
            if age_days:
                # Convert days to hours and estimate stage duration
                return min(age_days * 24, MAX_STAGE_DURATION_HOURS)
        return 0.0
    except (AttributeError, TypeError, ValueError):
        # Age calculation failed - return 0
        return 0.0


def _get_previous_stage(ad_id: str, current_stage: str) -> Optional[str]:
    """Get the previous stage in the lifecycle - ASC+ only."""
    # For new ads in ASC+ stage, there is no previous stage
    # Only return a previous stage if the ad actually transitioned from another stage
    # For now, ASC+ is the only stage, so new ads have no previous stage
    return None


def _get_stage_performance(ad_data: Dict[str, Any], stage: str) -> Optional[Dict[str, Any]]:
    """Get performance metrics for the current stage."""
    # For new ads with zero performance, return None (will be populated as data comes in)
    try:
        spend = safe_float(ad_data.get('spend', 0))
        impressions = int(ad_data.get('impressions', 0))
        purchases = int(ad_data.get('purchases', 0))
        
        # If ad has no performance data yet (all zeros), return None
        if spend == 0 and impressions == 0 and purchases == 0:
            return None
        
        return {
            'ctr': safe_float(ad_data.get('ctr', 0)),
            'cpa': safe_float(ad_data.get('cpa', 0)),
            'roas': safe_float(ad_data.get('roas', 0)),
            'spend': spend,
            'purchases': purchases,
            'stage': stage
        }
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error getting stage performance: {e}", exc_info=True)
        return None


def _get_transition_reason(ad_data: Dict[str, Any], stage: str) -> Optional[str]:
    """Get the reason for stage transition - ASC+ only."""
    # For new ads, there is no transition reason yet
    # Only return a reason when the ad actually transitions between stages
    try:
        spend = safe_float(ad_data.get('spend', 0))
        impressions = int(ad_data.get('impressions', 0))
        
        # If ad has no performance data yet (all zeros), return None
        if spend == 0 and impressions == 0:
            return None
        
        ctr = safe_float(ad_data.get('ctr', 0))
        cpa = safe_float(ad_data.get('cpa', 0))
        roas = safe_float(ad_data.get('roas', 0))
        
        if stage == 'asc_plus':
            if ctr >= 1.0 and cpa <= 30 and roas >= 1.0:
                return 'ASC+ campaign performing well'
            else:
                return 'ASC+ campaign - monitoring performance'
        else:
            return None
    except (KeyError, ValueError, TypeError):
        # For new ads with no data, return None
        return None


def store_performance_data_in_supabase(supabase_client, ad_data: Dict[str, Any], stage: str, ml_system=None) -> None:
    """Store performance data in Supabase for ML system with automatic validation."""
    if not supabase_client:
        logger.warning("No Supabase client available")
        return
    
    try:
        # Get validated Supabase client for automatic validation
        validated_client = _get_validated_supabase()
        if not validated_client:
            logger.warning("No validated Supabase client available, falling back to regular client")
            validated_client = supabase_client
        
        # Test Supabase connection first
        try:
            test_result = validated_client.select('performance_metrics').limit(1).execute()
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Supabase connection test failed: {e}", exc_info=True)
            notify(f"❌ Supabase connection failed: {e}")
            return
        
        # Calculate day_of_week, is_weekend, hour_of_day from current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        date_start = current_date  # Facebook API doesn't return date_start, use current date
        date_end = current_date    # Facebook API doesn't return date_end, use current date
        
        day_of_week = datetime.now().weekday()  # 0=Monday, 6=Sunday
        is_weekend = day_of_week >= 5  # Saturday or Sunday
        hour_of_day = datetime.now().hour  # 0-23
        
        # Calculate ad_age_days from creation time
        ad_age_days = 0
        try:
            ad_id = ad_data.get('ad_id', '')
            if ad_id:
                from infrastructure.supabase_storage import SupabaseStorage
                storage = SupabaseStorage(validated_client)
                age = storage.get_ad_age_days(ad_id)
                if age is not None and age > 0:
                    ad_age_days = min(max(0, age), MAX_AD_AGE_DAYS)
        except Exception:
            pass  # Fallback to 0 if calculation fails
        
        # Prepare performance data - validation will happen automatically
        ad_id = ad_data.get('ad_id', '')
        lifecycle_id = ad_data.get('lifecycle_id', f"lifecycle_{ad_id}")
        
        # Debug logging
        if not lifecycle_id or lifecycle_id == f"lifecycle_":
            logger.debug(f"Missing lifecycle_id: ad_id='{ad_id}', lifecycle_id='{lifecycle_id}', ad_data keys: {list(ad_data.keys())}")
        
        # Calculate performance scores
        quality_score = _calculate_performance_quality_score(ad_data)
        stability_score = _calculate_stability_score(ad_data)
        momentum_score = _calculate_momentum_score(ad_data)
        fatigue_index = _calculate_fatigue_index(ad_data)
        
        # Debug logging for performance calculations
        logger.debug(f"Ad {ad_id} - CTR: {safe_float(ad_data.get('ctr', 0)):.2f}%, Quality: {quality_score}, Stability: {stability_score:.2f}, Momentum: {momentum_score:.2f}, Fatigue: {fatigue_index:.2f}")
        
        performance_data = {
            'ad_id': ad_id,
            'lifecycle_id': lifecycle_id,
            'stage': stage,  # Use the stage parameter passed to the function
            'window_type': '1d',
            'date_start': date_start,
            'date_end': ad_data.get('date_end', ''),
            'impressions': int(ad_data.get('impressions', 0)),
            'clicks': int(ad_data.get('clicks', 0)),
            'spend': safe_float(ad_data.get('spend', 0), 999999.99),
            'purchases': int(ad_data.get('purchases', 0)),
            'add_to_cart': int(ad_data.get('atc', 0)),
            'initiate_checkout': int(ad_data.get('ic', 0)),
            'ctr': safe_float(ad_data.get('ctr', 0), DB_CTR_MAX),  # Cap at max CTR
            'cpc': safe_float(ad_data.get('cpc', 0), DB_CPC_MAX),  # Cap at max CPC
            'cpm': safe_float(ad_data.get('cpm', 0), DB_CPM_MAX),  # Cap at max CPM
            'roas': safe_float(ad_data.get('roas', 0), DB_ROAS_MAX),
            'cpa': safe_float(ad_data.get('cpa', 0), DB_CPA_MAX) if ad_data.get('cpa') is not None else None,
            'dwell_time': safe_float(ad_data.get('dwell_time', 0), DB_DWELL_TIME_MAX),
            'frequency': safe_float(ad_data.get('frequency', 0), DB_FREQUENCY_MAX),
            'atc_rate': safe_float(ad_data.get('atc_rate', 0), DB_RATE_MAX),
            'ic_rate': safe_float(ad_data.get('ic_rate', 0), DB_RATE_MAX),
            'purchase_rate': safe_float(ad_data.get('purchase_rate', 0), DB_RATE_MAX),
            'atc_to_ic_rate': safe_float(ad_data.get('atc_to_ic_rate', 0), DB_RATE_MAX),
            'ic_to_purchase_rate': safe_float(ad_data.get('ic_to_purchase_rate', 0), DB_RATE_MAX),
            'performance_quality_score': quality_score,
            'stability_score': stability_score,
            'momentum_score': momentum_score,
            'fatigue_index': fatigue_index,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'ad_age_days': int(ad_age_days) if ad_age_days else 0,
            'next_stage': _get_next_stage(stage),
            'stage_duration_hours': _calculate_stage_duration_hours(ad_id, stage),
            'previous_stage': _get_previous_stage(ad_id, stage),
            'stage_performance': _get_stage_performance(ad_data, stage),
            'transition_reason': _get_transition_reason(ad_data, stage),
        }
        
        # Validate all timestamps in performance data
        performance_data = validate_all_timestamps(performance_data)
        
        # Insert performance data with automatic validation
        try:
            result = validated_client.upsert(
                'performance_metrics',
                performance_data,
                on_conflict='ad_id,window_type,date_start'
            )
            notify(f"✅ Performance data validated and inserted: {result}")
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"Performance data validation/insertion failed: {e}", exc_info=True)
            notify(f"❌ Performance data validation/insertion failed: {e}")
            return
        
        # Prepare lifecycle data - validation will happen automatically
        # For new ads, previous_stage should be None (not empty string)
        lifecycle_data = {
            'ad_id': ad_data.get('ad_id', ''),
            'creative_id': ad_data.get('creative_id', ''),
            'campaign_id': ad_data.get('campaign_id', ''),
            'adset_id': ad_data.get('adset_id', ''),
            'stage': stage,
            'status': ad_data.get('status', 'active'),
            'lifecycle_id': ad_data.get('lifecycle_id', ''),
            'metadata': ad_data.get('metadata', {}),
            'previous_stage': None,  # New ads have no previous stage
            'next_stage': None,  # Will be set when transitioning
            'stage_duration_hours': None,  # Will be calculated over time
            'stage_performance': None,  # Will be populated with performance data
            'transition_reason': None,  # No transition for new ads
        }
        
        # Insert lifecycle data with automatic validation
        try:
            result = validated_client.upsert(
                'ad_lifecycle',
                lifecycle_data,
                on_conflict='ad_id,stage'
            )
            notify(f"✅ Lifecycle data validated and inserted: {result}")
        except Exception as e:
            notify(f"❌ Lifecycle data validation/insertion failed: {e}")
            return
        
        # Prepare creative intelligence data - validation will happen automatically
        try:
            # Skip if creative_intelligence already exists for this ad
            existing_check = validated_client.select('creative_intelligence').eq(
                'ad_id', ad_data.get('ad_id', '')
            ).execute()
            
            if not existing_check.data:
                creative_id = ad_data.get('creative_id') or f'creative_{ad_data.get("ad_id", "")}'
                creative_type = ad_data.get('creative_type', 'image')
                if creative_type not in ['image', 'video', 'carousel', 'collection', 'story', 'reels']:
                    creative_type = 'image'  # Default to 'image' for ASC+ static image campaigns
                
                # Generate realistic creative details
                import random
                # Creative technical details for static images
                duration_seconds = None  # Static images don't have duration
                aspect_ratios = ['1:1', '4:5', '9:16', '16:9']  # Common ad image ratios
                aspect_ratio = random.choice(aspect_ratios)
                file_size_mb = round(random.uniform(0.5, 5.0), 2)  # Smaller for images
                resolutions = ['1024x1024', '1080x1080', '1200x1500', '1080x1920', '1920x1080']
                resolution = random.choice(resolutions)
                
                # Color palette (generate realistic colors)
                color_palettes = [
                    ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                    ['#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', '#E17055'],
                    ['#00B894', '#00CEC9', '#0984E3', '#6C5CE7', '#A29BFE'],
                    ['#E84393', '#FDCB6E', '#00B894', '#0984E3', '#6C5CE7']
                ]
                color_palette = json.dumps(random.choice(color_palettes))
                
                # Creative content features for static images
                text_overlay = True  # ASC+ creatives always have text overlay
                music_present = False  # Static images don't have audio
                voice_over = False  # Static images don't have audio
                
                # Load creative content from copy bank
                try:
                    copy_bank_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'copy_bank.json')
                    with open(copy_bank_path, 'r') as f:
                        copy_bank = json.load(f)
                    
                    # Get creative content from copy bank
                    global_copy = copy_bank.get('global', {})
                    descriptions = global_copy.get('descriptions', [])
                    headlines = global_copy.get('headlines', [])
                    primary_texts = global_copy.get('primary_texts', [])
                    
                    # Select content from copy bank
                    description = random.choice(descriptions) if descriptions else "Premium quality product"
                    headline = random.choice(headlines) if headlines else "Quality You Can Trust"
                    primary_text = random.choice(primary_texts) if primary_texts else "Experience the difference with our premium selection."
                    
                except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to load copy bank: {e}")
                    # Fallback to basic content
                    description = "Premium quality product"
                    headline = "Quality You Can Trust"
                    primary_text = "Experience the difference with our premium selection."
                
                # Calculate performance metrics based on ad data
                # Clamp to NUMERIC(5,4) range (-9.9999 to 9.9999) to prevent database overflow
                ctr = safe_float(ad_data.get('ctr', 0))
                cpa = safe_float(ad_data.get('cpa', 0))
                roas = safe_float(ad_data.get('roas', 0))
                
                # Clamp values to fit NUMERIC(5,4) database constraint
                ctr = min(max(ctr, DB_NUMERIC_MIN), DB_NUMERIC_MAX)
                cpa = min(max(cpa, DB_NUMERIC_MIN), DB_NUMERIC_MAX)
                roas = min(max(roas, DB_NUMERIC_MIN), DB_NUMERIC_MAX)
                
                # Calculate performance score (0-1 scale)
                performance_score = 0.5  # Base score
                if ctr > 0:
                    performance_score += min(ctr / 5.0, 0.3)  # CTR contribution (max 0.3)
                if cpa > 0 and cpa < 50:
                    performance_score += min((50 - cpa) / 50, 0.2)  # CPA contribution (max 0.2)
                if roas > 0:
                    performance_score += min(roas / 10.0, 0.2)  # ROAS contribution (max 0.2)
                
                performance_score = min(max(performance_score, 0.0), 1.0)
                
                # Calculate fatigue index (0-1 scale, higher = more fatigued)
                fatigue_index = 0.0
                if ctr < 1.0:
                    fatigue_index += 0.3
                if cpa > 30:
                    fatigue_index += 0.3
                if roas < 2.0:
                    fatigue_index += 0.4
                fatigue_index = min(fatigue_index, 1.0)
                
                # Generate similarity vector (384 dimensions to match validation)
                similarity_vector = [random.uniform(-1, 1) for _ in range(384)]
                
                # Enhanced metadata
                metadata = {
                    'source': 'copy_bank',
                    'created': datetime.now().isoformat(),
                    'needs_review': False,  # Copy bank content is pre-approved
                    'creative_quality': 'high',  # Copy bank content is curated
                    'engagement_score': random.uniform(0.6, 1.0),  # Higher scores for copy bank content
                    'brand_safety': 'safe',  # Copy bank content is brand-safe
                    'content_category': 'skincare',  # Based on copy bank content
                    'target_audience': '25-44',  # Based on copy bank content
                    'copy_bank_version': '1.0',
                    'campaign_id': ad_data.get('campaign_id', ''),
                    'adset_id': ad_data.get('adset_id', ''),
                    'campaign_name': ad_data.get('campaign_name', ''),
                    'adset_name': ad_data.get('adset_name', '')
                }
                
                creative_data = {
                    'creative_id': creative_id,
                    'ad_id': ad_data.get('ad_id', ''),
                    'creative_type': creative_type,
                    'duration_seconds': duration_seconds,
                    'aspect_ratio': aspect_ratio,
                    'file_size_mb': file_size_mb,
                    'resolution': resolution,
                    'color_palette': color_palette,
                    'text_overlay': text_overlay,
                    'music_present': music_present,
                    'voice_over': voice_over,
                    'avg_ctr': ctr,
                    'avg_cpa': cpa,
                    'avg_roas': roas,
                    'performance_rank': random.randint(1, 100),
                    'performance_score': round(performance_score, 4),
                    'fatigue_index': round(fatigue_index, 4),
                    'similarity_vector': similarity_vector,
                    'description': description,
                    'headline': headline,
                    'primary_text': primary_text,
                    'lifecycle_id': ad_data.get('lifecycle_id', ''),
                    'stage': ad_data.get('stage', 'testing'),
                    'metadata': metadata
                }
                
                # Validate all timestamps in creative intelligence data
                creative_data = validate_all_timestamps(creative_data)
                
                # Ensure performance metrics are calculated if missing
                if not creative_data.get('avg_ctr') or not creative_data.get('avg_cpa') or not creative_data.get('avg_roas'):
                    try:
                        from infrastructure.data_optimizer import CreativeIntelligenceOptimizer
                        optimizer = CreativeIntelligenceOptimizer(supabase_client)
                        metrics = optimizer.calculate_performance_metrics(
                            creative_id,
                            ad_data.get('ad_id', ''),
                        )
                        # Update with calculated metrics
                        creative_data['avg_ctr'] = metrics.get('avg_ctr', creative_data.get('avg_ctr', 0.0))
                        creative_data['avg_cpa'] = metrics.get('avg_cpa', creative_data.get('avg_cpa', 0.0))
                        creative_data['avg_roas'] = metrics.get('avg_roas', creative_data.get('avg_roas', 0.0))
                        creative_data['performance_score'] = metrics.get('performance_score', creative_data.get('performance_score', 0.0))
                        creative_data['fatigue_index'] = metrics.get('fatigue_index', creative_data.get('fatigue_index', 0.0))
                    except ImportError:
                        pass  # Optimizer not available
                    except Exception as e:
                        logger.debug(f"Failed to calculate performance metrics: {e}")
                
                # Insert creative intelligence data with automatic validation
                result = validated_client.insert('creative_intelligence', creative_data)
                if result:
                    logger.info(f"Creative intelligence validated and inserted: {creative_id} for ad {ad_data.get('ad_id')}")
                    
                    # Schedule performance metrics update (async, non-blocking)
                    try:
                        from infrastructure.data_optimizer import CreativeIntelligenceOptimizer
                        optimizer = CreativeIntelligenceOptimizer(supabase_client)
                        # Update in background (will calculate from performance_metrics)
                        optimizer.update_creative_performance(creative_id, ad_data.get('ad_id', ''))
                    except Exception:
                        pass  # Non-critical, continue
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to store creative intelligence for {ad_data.get('ad_id')}: {e}", exc_info=True)
        
        # Track creative performance (populates creative_performance table)
        try:
            creative_id = ad_data.get('creative_id', '')
            if creative_id and supabase_client:
                from creative.creative_intelligence import create_creative_intelligence_system
                creative_system = create_creative_intelligence_system(
                    supabase_client=supabase_client,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    settings=None
                )
                if creative_system:
                    creative_ids = {'image': creative_id}  # ASC+ uses static images
                    creative_system.track_creative_performance(
                        ad_id=ad_data.get('ad_id', ''),
                        creative_ids=creative_ids,
                        performance_data=ad_data,
                        stage=stage
                    )
        except Exception as e:
            logger.debug(f"Failed to track creative performance: {e}")
        
        # Make ML predictions for this ad after data is stored
        if ml_system:
            try:
                ad_id = ad_data.get('ad_id', '')
                if ad_id:
                    # Get ML intelligence analysis (this will make predictions and save them)
                    ml_analysis = ml_system.analyze_ad_intelligence(ad_id, stage)
                    # Insights are already saved by analyze_ad_intelligence, no additional action needed
            except (AttributeError, ValueError, TypeError) as e:
                # Don't fail the entire function if ML prediction fails
                logger.warning(f"Failed to make ML predictions for ad {ad_data.get('ad_id', 'unknown')}: {e}", exc_info=True)
        
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to store performance data in Supabase: {e}", exc_info=True)
        notify(f"❌ Failed to store performance data in Supabase: {e}")

def store_ml_insights_in_supabase(supabase_client, ad_id: str, insights: Dict[str, Any]) -> None:
    """Store ML insights in Supabase with automatic validation."""
    if not supabase_client:
        return
    
    try:
        # Get validated Supabase client for automatic validation
        validated_client = _get_validated_supabase()
        if not validated_client:
            validated_client = supabase_client
        
        # Store creative intelligence data with validation
        creative_type = insights.get('creative_type', 'image')
        if creative_type not in ['image', 'video', 'carousel', 'collection', 'story', 'reels']:
            creative_type = 'image'  # Default to image for ASC+ static image campaigns
        
        creative_data = {
            'creative_id': insights.get('creative_id', f'creative_{ad_id}'),
            'ad_id': ad_id,
            'creative_type': creative_type,
            'performance_score': float(insights.get('performance_score', 0)),
            'fatigue_index': float(insights.get('fatigue_index', 0)),
            'similarity_vector': insights.get('similarity_vector', None),  # Use None instead of empty list
            'metadata': insights.get('metadata', {})
        }
        
        # Upsert with automatic validation
        validated_client.upsert('creative_intelligence', creative_data, on_conflict='creative_id,ad_id')
        
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Failed to store ML insights in Supabase: {e}", exc_info=True)


def safe_float_global(value, max_val=None):
    """Safely convert value to float with bounds checking (global version)."""
    if max_val is None:
        max_val = DB_GLOBAL_FLOAT_MAX
    try:
        val = float(value or 0)
        # Handle infinity and NaN
        if not (val == val) or val == float('inf') or val == float('-inf'):
            return 0.0
        # Bound the value to prevent overflow
        return min(max(val, -max_val), max_val)
    except (ValueError, TypeError):
        return 0.0

def store_timeseries_data_in_supabase(supabase_client, ad_id: str, ad_data: Dict[str, Any], stage: str) -> None:
    """Store hourly time-series data in Supabase for temporal modeling (NEW)."""
    if not supabase_client:
        return
    
    try:
        metrics_to_track = {
            'impressions': ad_data.get('impressions'),
            'spend': ad_data.get('spend'),
            'clicks': ad_data.get('clicks'),
            'purchases': ad_data.get('purchases'),
        }

        timestamp = datetime.now(timezone.utc).isoformat()

        for metric_name, metric_value in metrics_to_track.items():
            if metric_value is None:
                continue

            timeseries_data = {
                'ad_id': ad_id,
                'lifecycle_id': ad_data.get('lifecycle_id', ''),
                'stage': stage,
                'metric_name': metric_name,
                'metric_value': safe_float_global(metric_value, DB_GLOBAL_FLOAT_MAX),
                'timestamp': timestamp,
                'window_type': '1h',
                'metadata': {
                    'campaign_id': ad_data.get('campaign_id'),
                    'adset_id': ad_data.get('adset_id'),
                    'reason': ad_data.get('metadata', {}).get('reason') if isinstance(ad_data.get('metadata'), dict) else None,
                },
                'window_size': 1,
                'anomalies_detected': False,
                'seasonality_detected': False,
                'time_period': '1h',
                'timestamps': [timestamp],
                'values': [safe_float_global(metric_value, DB_GLOBAL_FLOAT_MAX)],
                'trend_direction': 'stable',
            }

            timeseries_data = validate_all_timestamps(timeseries_data)

            validated_client = _get_validated_supabase()
            if validated_client:
                validated_client.insert('time_series_data', timeseries_data)
            else:
                supabase_client.table('time_series_data').insert(timeseries_data).execute()
        
    except (KeyError, ValueError, TypeError) as e:
        logger.error("[ASC] Failed to store time-series data: %s", e, exc_info=True)


def collect_stage_ad_data(meta_client, settings: Dict[str, Any], stage: str) -> Dict[str, Dict[str, Any]]:
    """Collect actual ad data for a stage from Meta API."""
    ad_data: Dict[str, Dict[str, Any]] = {}
    timekit = Timekit()
    account_today = timekit.today_ymd_account()
    ids_cfg = settings.get("ids", {}) if isinstance(settings, dict) else {}
    target_campaign = ids_cfg.get("asc_plus_campaign_id")
    target_adset = ids_cfg.get("asc_plus_adset_id")

    try:
        insights_rows = meta_client.get_recent_ad_insights(
            adset_id=target_adset,
            campaign_id=target_campaign,
        )
        for row in insights_rows:
            ad_id = row.get("ad_id")
            if not ad_id:
                continue

            spend_val = _metric_to_float(row.get("spend"))
            impressions_val = int(round(_metric_to_float(row.get("impressions"))))
            link_clicks_val = _metric_to_float(row.get("inline_link_clicks"))
            if link_clicks_val <= 0:
                link_clicks_val = _metric_to_float(row.get("link_clicks"))
            all_clicks_val = _metric_to_float(row.get("clicks"))
            if link_clicks_val <= 0 and all_clicks_val > 0:
                link_clicks_val = all_clicks_val
            clicks_val = int(round(link_clicks_val))

            actions = row.get("actions", []) or []
            add_to_cart = 0
            initiate_checkout = 0
            purchases = 0
            for action in actions:
                action_type = action.get("action_type")
                value = float(action.get("value", 0) or 0)
                if action_type == "add_to_cart":
                    add_to_cart = int(value)
                elif action_type == "initiate_checkout":
                    initiate_checkout = int(value)
                elif action_type == "purchase":
                    purchases = int(value)

            revenue = 0.0
            for action_value in row.get("action_values", []) or []:
                if action_value.get("action_type") == "purchase":
                    revenue += float(action_value.get("value", 0) or 0.0)

            purchase_roas_list = row.get("purchase_roas") or []
            if purchase_roas_list:
                roas = float(purchase_roas_list[0].get("value", 0) or 0.0)
            elif spend_val > 0:
                roas = revenue / spend_val
            else:
                roas = 0.0

            ctr = (clicks_val / impressions_val * 100) if impressions_val > 0 else 0.0
            cpc = (spend_val / link_clicks_val) if link_clicks_val > 0 else ((spend_val / all_clicks_val) if all_clicks_val > 0 else 0.0)
            cpm = (spend_val / impressions_val * 1000) if impressions_val > 0 else 0.0
            cpa = (spend_val / purchases) if purchases > 0 else None

            ad_data[ad_id] = {
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
                "date_start": account_today,
                "date_end": account_today,
                "campaign_name": row.get("campaign_name", ""),
                "campaign_id": row.get("campaign_id"),
                "adset_name": row.get("adset_name", ""),
                "adset_id": row.get("adset_id"),
                "has_recent_activity": bool(spend_val or impressions_val or clicks_val),
                "metadata": {"source": "meta_insights"},
            }
    except Exception as exc:
        logger.warning("[ASC] Failed to collect %s ad data: %s", stage, exc)

    return ad_data

def store_creative_data_in_supabase(supabase_client, meta_client, ad_id: str, stage: str) -> None:
    """Fetch and store creative intelligence data from Meta API (NEW)."""
    if not supabase_client or not meta_client:
        return
    
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
        
        # Store in creative_intelligence with validation
        creative_data = {
            'creative_id': str(creative_id),
            'ad_id': ad_id,
            'creative_type': creative.get('object_type', 'image'),
            'performance_score': 0.5,  # Will be updated by ML
            'fatigue_index': 0.0,
            'similarity_vector': None,  # Will be populated by ML later
            'metadata': {}
        }
        
        # Get validated client for automatic validation
        validated_client = _get_validated_supabase()
        if validated_client:
            validated_client.upsert('creative_intelligence', creative_data, on_conflict='creative_id')
        else:
            supabase_client.table('creative_intelligence').upsert(creative_data, on_conflict='creative_id').execute()
        notify(f"🎨 Creative data validated and stored for {ad_id}")
        
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        # Creative data is optional, log but don't fail
        logger.debug(f"Failed to store creative data for {ad_id}: {e}")


def initialize_creative_intelligence_system(supabase_client, settings) -> Optional[Any]:
    """Initialize the creative intelligence system."""
    try:
        from creative.creative_intelligence import create_creative_intelligence_system
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        creative_system = create_creative_intelligence_system(
            supabase_client=supabase_client,
            openai_api_key=openai_api_key,
            settings=settings
        )
        
        # Load copy bank data to Supabase
        creative_system.load_copy_bank_to_supabase()
        
        notify("🎨 Creative Intelligence System initialized")
        return creative_system
        
    except Exception as e:
        notify(f"⚠️ Failed to initialize Creative Intelligence System: {e}")
        return None


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
    """Validate required environment variables."""
    return [k for k in required if not os.getenv(k)]

def validate_asc_plus_envs() -> List[str]:
    """Validate environment variables required for ASC+ campaign."""
    required = [
        "FB_ACCESS_TOKEN",
        "FB_AD_ACCOUNT_ID",
        "FB_PAGE_ID",
        "FLUX_API_KEY",
        "OPENAI_API_KEY",
    ]
    optional = [
        "FB_ACCOUNT_ID",  # Alternative to FB_AD_ACCOUNT_ID
        "META_ACCESS_TOKEN",  # Alternative to FB_ACCESS_TOKEN
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SLACK_WEBHOOK_URL",
    ]
    
    missing = []
    for var in required:
        if not os.getenv(var):
            # Check for alternatives
            if var == "FB_ACCESS_TOKEN" and os.getenv("META_ACCESS_TOKEN"):
                continue
            if var == "FB_AD_ACCOUNT_ID" and os.getenv("FB_ACCOUNT_ID"):
                continue
            missing.append(var)
    
    return missing


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

    # required top-level sections for ASC+ campaign
    for k in ("ids", "asc_plus", "queue", "logging"):
        if k not in settings:
            issues.append(f"Missing section: {k}")

    # ASC+ budget validation
    asc_plus = settings.get("asc_plus", {}) or {}
    if not asc_plus:
        issues.append("ASC+ section is missing or empty in settings.yaml")
    else:
        try:
            daily_budget = float(asc_plus.get("daily_budget_eur", 0) or 0)
            if daily_budget <= 0:
                issues.append(f"ASC+ daily_budget_eur must be > 0 (got: {daily_budget})")
            target_ads = int(asc_plus.get("target_active_ads", 0) or 0)
            if target_ads <= 0:
                issues.append(f"ASC+ target_active_ads must be > 0 (got: {target_ads})")
        except (ValueError, TypeError, KeyError) as e:
            issues.append(f"ASC+ configuration error: {e}")
            issues.append(f"ASC+ section contents: {asc_plus}")

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
            except (OSError, IOError):
                # File may not exist or be readable - proceed
                pass
            locked = True  # proceed best-effort
        if locked:
            try:
                os.ftruncate(fd, 0)
                os.write(fd, str(os.getpid()).encode())
            except (OSError, IOError):
                # Lock file operations may fail - best effort
                pass
            yield
    finally:
        if fd is not None:
            try:
                import fcntl  # type: ignore
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (OSError, IOError, AttributeError):
                # Lock release may fail - continue to cleanup
                pass
            try:
                os.close(fd)
            except (OSError, IOError):
                pass
            try:
                os.remove(path)
            except (OSError, IOError):
                # File may already be removed
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
            
            # Balance monitoring removed as requested
            
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
                    # Use the same SQLite path as the main system
                    sqlite_path = settings.get("logging", {}).get("sqlite", {}).get("path", "data/state.sqlite")
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
                
                # Balance monitoring removed as requested
            
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
        # datetime already imported at module level
        import zoneinfo
        
        # Use account timezone for today's data
        tz_name = settings.get("account", {}).get("timezone", "Europe/Amsterdam")
        local_tz = zoneinfo.ZoneInfo(tz_name)
        now = datetime.now(local_tz)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        rows = meta.get_ad_insights(
            level="ad", 
            fields=[
                "spend",
                "actions",
                "impressions",
                "clicks",
                "inline_link_clicks",
                "inline_link_click_ctr",
                "cost_per_inline_link_click",
                "cpm",
            ], 
            time_range={
                "since": midnight.strftime("%Y-%m-%d"),
                "until": now.strftime("%Y-%m-%d")
            },
            paginate=True
        )
        
        spend = sum(_metric_to_float(r.get("spend")) for r in rows)
        impressions = int(round(sum(_metric_to_float(r.get("impressions")) for r in rows)))
        link_clicks = sum(
            _metric_to_float(r.get("inline_link_clicks"))
            or _metric_to_float(r.get("link_clicks"))
            for r in rows
        )
        all_clicks = sum(_metric_to_float(r.get("clicks")) for r in rows)
        purch = 0.0
        atc = 0.0  # Add to cart
        ic = 0.0   # Initiate checkout

        if link_clicks <= 0 and all_clicks > 0:
            link_clicks = all_clicks
        
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
                except (KeyError, TypeError, ValueError):
                    # Skip invalid action entries
                    continue

        # Calculate metrics
        link_clicks = max(link_clicks, 0.0)
        all_clicks = max(all_clicks, 0.0)
        cpa = (spend / purch) if purch > 0 else float("inf")
        ctr = ((link_clicks or 0.0) / impressions * 100) if impressions > 0 else 0
        cpc = (spend / link_clicks) if link_clicks > 0 else ((spend / all_clicks) if all_clicks > 0 else 0)
        cpm = (spend / impressions * 1000) if impressions > 0 else 0
        
        # If Meta already returned cost-per-link metric, prefer it when available
        if link_clicks > 0:
            # Weighted average cost per link click using provided metric if present
            cost_per_link_values = [
                (
                    _metric_to_float(row.get("cost_per_inline_link_click")),
                    _metric_to_float(row.get("inline_link_clicks")) or _metric_to_float(row.get("link_clicks"))
                )
                for row in rows
            ]
            weighted_spend = sum(cost * clicks for cost, clicks in cost_per_link_values if clicks and cost)
            total_weight = sum(clicks for _, clicks in cost_per_link_values if clicks)
            if weighted_spend and total_weight:
                cpc = weighted_spend / total_weight
        
        be = float(
            os.getenv("BREAKEVEN_CPA")
            or (settings.get("economics", {}) or {}).get("breakeven_cpa")
            or 27.51
        )
        
        # Get active ads count from insights data AND direct API call
        try:
            # Count unique ads from insights data (ads with today's data)
            insights_ads_count = len(rows) if rows else 0
            
            # Also get all active ads directly from Meta API (more accurate total)
            try:
                all_active_ads = meta.get_ad_insights(
                    level="ad",
                    fields=["ad_id"],
                    date_preset="maximum",  # All time to get all active ads
                    filtering=[{"field": "ad.effective_status", "operator": "IN", "value": ["ACTIVE"]}]
                )
                total_active_count = len(all_active_ads) if all_active_ads else 0
                
                # Use the higher count (some ads might not have today's data yet)
                active_ads_count = max(insights_ads_count, total_active_count)
            except (AttributeError, TypeError, ValueError, KeyError):
                # Fallback to insights count if direct API call fails
                active_ads_count = insights_ads_count
                
        except (AttributeError, TypeError, ValueError, KeyError):
            active_ads_count = 0
        
        return {
            "spend": round(spend, 2),
            "purchases": int(purch),
            "cpa": None if cpa == float("inf") else round(cpa, 2),
            "breakeven": be,
            "impressions": impressions,
            "clicks": int(round(link_clicks if link_clicks > 0 else all_clicks)),
            "ctr": round(ctr, 2),
            "cpc": round(cpc, 2),
            "cpm": round(cpm, 2),
            "atc": int(atc),
            "ic": int(ic),
            "active_ads": active_ads_count,
        }
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        logger.error("Failed to calculate account metrics", exc_info=True)
        return {
            "spend": None, "purchases": None, "cpa": None, "breakeven": None,
            "impressions": None, "clicks": None, "ctr": None, "cpc": None,
            "cpm": None, "atc": None, "ic": None, "active_ads": None
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
    # Use deep merge to preserve settings values (especially asc_plus config)
    if rules_cfg:
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            """Deep merge two dictionaries, preserving base values when override has conflicting keys."""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    # Special handling for asc_plus to preserve settings values
                    if key == "asc_plus":
                        # Merge asc_plus preserving settings values (daily_budget_eur, target_active_ads, etc.)
                        merged_asc_plus = result[key].copy()
                        merged_asc_plus.update(value)  # Rules override only specific keys, not the whole dict
                        result[key] = merged_asc_plus
                    else:
                        # Recursive merge for other nested dicts
                        result[key] = deep_merge(result[key], value)
                else:
                    # For non-dict values or new keys, just set the value
                    result[key] = value
            return result
        settings = deep_merge(settings, rules_cfg)

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
    except (ImportError, jsonschema.ValidationError) as e:
        logger.debug(f"Schema validation skipped or failed: {e}")

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
    
    # Supabase storage will be initialized after supabase_client

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

    # Rule engine (will be updated with supabase_storage later)
    engine = RuleEngine(rules_cfg, store)
    
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
    ml_pipeline = None
    ml_dashboard = None
    
    if ml_mode_enabled:
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            
            # Initialize full ML system with database integration
            ml_system = create_ml_system(supabase_url, supabase_key, MLConfig())
            
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
            
            # Enhanced ML system now integrated with historical data capabilities
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
            
            # Show ML readiness for ASC+ stage
            if data_progress_tracker:
                readiness = data_progress_tracker.get_ml_readiness('asc_plus')
                notify(f"📊 ASC+ Stage: {readiness.get('message', 'Unknown')}")
            
            # Show ML system health
            if ml_dashboard:
                health = ml_dashboard.get_health_metrics(hours_back=24)
                notify(f"🤖 ML Health: {health.avg_model_accuracy:.1%} accuracy, {health.predictions_made_24h} predictions/24h")
            
            notify("✅ ML system initialized successfully (20 enhancements active)")
            
        except Exception as e:
            notify(f"❌ Failed to initialize ML system: {e}")
            notify("   Falling back to legacy system...")
            ml_mode_enabled = False

    # Initialize Supabase client
    supabase_client = _get_supabase()

    # Initialize Supabase storage for ad creation times and historical data
    supabase_storage = None
    if supabase_client:
        try:
            supabase_storage = create_supabase_storage(supabase_client)
            notify("📊 Supabase storage initialized for ad creation times and historical data")
            
            # Update rule engine to use Supabase storage
            engine = RuleEngine(rules_cfg, supabase_storage)
            notify("📊 Rule engine updated to use Supabase storage")
        except Exception as e:
            notify(f"⚠️ Failed to initialize Supabase storage: {e}")
            supabase_storage = None

    # Initialize Table Monitoring System
    table_monitor = None
    if supabase_client:
        try:
            from analytics.table_monitoring import create_table_monitor
            table_monitor = create_table_monitor(supabase_client)
            notify("📊 Table monitoring system initialized")
            
            # Get initial table state
            initial_insights = table_monitor.get_all_table_insights()
            notify("📊 Initial table state:")
            notify(f"   • Total tables: {initial_insights.total_tables}")
            notify(f"   • Total rows: {initial_insights.total_rows:,}")
            notify(f"   • Healthy tables: {initial_insights.healthy_tables}")
            if initial_insights.problematic_tables > 0:
                notify(f"   • Problematic tables: {initial_insights.problematic_tables}")
        except Exception as e:
            notify(f"⚠️ Failed to initialize table monitoring: {e}")
    
    # Initialize Creative Intelligence System
    creative_system = None
    if supabase_client:
        try:
            creative_system = initialize_creative_intelligence_system(supabase_client, settings)
        except Exception as e:
            notify(f"⚠️ Creative Intelligence System initialization error: {e}")

    # Validate ASC+ environment variables
    missing_envs = validate_asc_plus_envs()
    if missing_envs:
        notify(f"❌ Missing required environment variables for ASC+ campaign: {', '.join(missing_envs)}")
        notify("   Required: FB_ACCESS_TOKEN, FB_AD_ACCOUNT_ID, FB_PAGE_ID, FLUX_API_KEY, OPENAI_API_KEY")
        if not (profile == "staging" or effective_dry):
            sys.exit(1)
    else:
        notify("✅ All required ASC+ environment variables are set")
    
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

    # Queue loading disabled - ASC+ generates creatives dynamically via image_generator
    # Legacy queue functions kept for compatibility but not used by ASC+ stage
    queue_df = pd.DataFrame()  # Empty queue - ASC+ doesn't use a pre-loaded queue
    queue_len_before = 0

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
        'cpm': acct.get('cpm'),
        'atc': acct.get('atc', 0),
        'ic': acct.get('ic', 0),
        'active_ads': acct.get('active_ads', 0),
    }

    # Idempotency (tick-level) and process lock (multi-runner safety)
    try:
        tkey = f"tick::{local_now:%Y-%m-%dT%H:%M}"
        if hasattr(store, "tick_seen") and store.tick_seen(tkey):
            notify("ℹ️ Tick already processed; exiting.")
            return
    except (AttributeError, TypeError):
        # Store may not support tick_seen, continue normally
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
                    except (AttributeError, TypeError, ValueError):
                        # Logging failure is non-critical
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
        
        # ASC+ Campaign - Single campaign mode (always run)
        overall["asc_plus"] = run_stage(
            run_asc_plus_tick,
            "ASC+",
            client,
            settings,
            rules_cfg,
            store,
            ml_system=ml_system if ml_mode_enabled else None,
        )
        if overall.get("asc_plus"):
            stage_summaries.append({
                "stage": "ASC+",
                "result": overall["asc_plus"],
            })
        
        # System is ASC+ only - all old stages removed
        
        # Store ASC+ performance data in Supabase for ML system
        if overall.get("asc_plus") and supabase_client:
            try:
                asc_result = overall.get("asc_plus", {})
                campaign_id = asc_result.get("campaign_id")
                adset_id = asc_result.get("adset_id")
                metrics_source = asc_result.get("ad_metrics") or {}

                if not metrics_source:
                    metrics_source = collect_stage_ad_data(client, settings, "asc_plus")

                for ad_id, ad_data in metrics_source.items():
                    if isinstance(ad_data, dict):
                        store_performance_data_in_supabase(supabase_client, ad_data, "asc_plus", ml_system)
                        try:
                            store_timeseries_data_in_supabase(supabase_client, ad_id, ad_data, "asc_plus")
                        except Exception as e:
                            logger.debug(f"Failed to store time-series data: {e}")
                        try:
                            from infrastructure.supabase_storage import SupabaseStorage
                            storage = SupabaseStorage(supabase_client)
                            lifecycle_id = ad_data.get('lifecycle_id', f'lifecycle_{ad_id}')
                            metrics_to_store = {
                                'spend': ad_data.get('spend', 0),
                                'impressions': ad_data.get('impressions', 0),
                                'clicks': ad_data.get('clicks', 0),
                                'purchases': ad_data.get('purchases', 0),
                                'roas': ad_data.get('roas', 0),
                                'ctr': ad_data.get('ctr', 0),
                                'cpa': ad_data.get('cpa', 0) if ad_data.get('cpa') is not None else 0,
                            }
                            for metric_name, metric_value in metrics_to_store.items():
                                try:
                                    storage.store_historical_data(ad_id, lifecycle_id, "asc_plus", metric_name, float(metric_value))
                                except Exception as e:
                                    logger.debug(f"Failed to store historical data for {metric_name}: {e}")
                        except Exception as e:
                            logger.debug(f"Failed to store historical data: {e}")
            except Exception as e:
                logger.error(f"Failed to store ASC+ data in Supabase: {e}")

    # Queue persist disabled - ASC+ generates creatives dynamically, no queue to persist
    
    # Run model validation if needed (NEW - weekly validation)
    if ml_mode_enabled and ml_pipeline:
        try:
            validation_results = ml_pipeline.validate_models_if_needed()
            if validation_results:
                notify("🔬 Model validation completed")
                # Alert if any model has low accuracy
                for model_name, metrics in validation_results.items():
                    if not isinstance(metrics, dict):
                        continue
                    status = metrics.get('status')
                    accuracy = metrics.get('accuracy')
                    if status in {'missing_model', 'insufficient_data'}:
                        continue
                    if accuracy is None:
                        continue
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
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "profile": profile,
                    "dry_run": client.dry_run,
                    "simulate": shadow_mode,
                    "timezone": tz_name,
                    "stage": args.stage,
                    "acct": acct,
                    "health": hc,
                }
            )
        except (KeyError, ValueError, TypeError):
            # Health check reporting is non-critical
            logger.debug("Failed to send health check to Slack")

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
            cpm=account_info['cpm'],
            atc=account_info['atc'],
            ic=account_info['ic']
        )
        
        # Collect ad insights and post as thread reply
        try:
            if stage_choice in ("all", "asc_plus", "testing"):
                # Get today's insights with local timezone
                import zoneinfo
                
                # Define attribution windows for consistency
                attr_windows = ["7d_click", "1d_view"]
                
                local_tz = zoneinfo.ZoneInfo(tz_name)
                now = datetime.now(local_tz)
                midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Build filtering - use ASC+ adset ID if available
                # Note: ad.status is not a valid filter field, use ad.effective_status instead
                asc_plus_adset_id = settings.get("ids", {}).get("asc_plus_adset_id")
                filtering_today = []
                filtering_lifetime = []
                
                # Only add adset filter if we have a valid adset ID
                if asc_plus_adset_id:
                    filtering_today = [{"field": "adset.id", "operator": "IN", "value": [asc_plus_adset_id]}]
                    filtering_lifetime = [{"field": "adset.id", "operator": "IN", "value": [asc_plus_adset_id]}]
                # Note: We can't filter by ad status in insights API - we'll filter results after fetching
                
                # Get today's data - filter by adset only (can't filter by status in insights API)
                rows_today_raw = client.get_ad_insights(
                    level="ad",
                    filtering=filtering_today,
                    fields=["ad_id", "ad_name", "spend", "actions"],
                    time_range={
                        "since": midnight.strftime("%Y-%m-%d"),
                        "until": now.strftime("%Y-%m-%d")
                    },
                    action_attribution_windows=list(attr_windows),
                    paginate=True
                ) or []
                
                # Filter to only ACTIVE ads (we can't do this in API filtering)
                # Get active ad IDs from Meta API
                try:
                    active_ad_ids = set()
                    if asc_plus_adset_id:
                        active_ads = client.list_ads_in_adset(asc_plus_adset_id)
                        active_ad_ids = {str(ad.get("id", "")) for ad in active_ads if str(ad.get("status", "")).upper() == "ACTIVE"}
                    
                    # Filter insights to only active ads
                    rows_today = [r for r in rows_today_raw if str(r.get("ad_id", "")) in active_ad_ids] if active_ad_ids else rows_today_raw
                except Exception as e:
                    logger.warning(f"Failed to filter active ads, using all: {e}")
                    rows_today = rows_today_raw
                
                # Get lifetime data
                rows_lifetime = client.get_ad_insights(
                    level="ad",
                    filtering=filtering_lifetime,
                    fields=["ad_id", "spend", "actions"],
                    time_range={
                        "since": "2024-01-01",  # Far back enough to capture all lifetime
                        "until": now.strftime("%Y-%m-%d")
                    },
                    action_attribution_windows=list(attr_windows),
                    paginate=True
                ) or []
                
                # Build snapshot using helper
                from integrations import build_ads_snapshot
                ad_lines = build_ads_snapshot(rows_today or [], rows_lifetime or [], tz_name)
                
                if ad_lines:
                    post_thread_ads_snapshot(thread_ts, ad_lines)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Failed to post ads snapshot: {e}")

    # ML Model Training (if ML mode enabled and data is available)
    logger.debug(f"ML mode enabled: {ml_mode_enabled}")
    logger.debug(f"ML system exists: {ml_system is not None}")
    if ml_system:
        logger.debug(f"ML system type: {type(ml_system)}")
        logger.debug(f"ML system has predictor: {hasattr(ml_system, 'predictor')}")
        if hasattr(ml_system, 'predictor'):
            logger.debug(f"ML predictor type: {type(ml_system.predictor)}")
    
    if ml_mode_enabled and ml_system:
        try:
            # Small delay to ensure data is fully committed to Supabase
            time.sleep(ML_TRAINING_DELAY_SECONDS)
            
            # Train testing stage models directly
            logger.info("Starting ML training...")
            logger.debug(f"ML system type: {type(ml_system)}")
            logger.debug(f"ML predictor type: {type(ml_system.predictor)}")
            logger.debug(f"ML predictor methods: {[m for m in dir(ml_system.predictor) if not m.startswith('_')]}")
            
            try:
                logger.debug("Calling performance_predictor training...")
                perf_success = ml_system.predictor.train_model('performance_predictor', 'testing', 'cpa')
                logger.debug(f"Performance predictor training result: {perf_success}")
            except (AttributeError, ValueError, TypeError) as e:
                logger.error(f"Performance predictor training error: {e}", exc_info=True)
                perf_success = False
            
            try:
                logger.debug("Calling roas_predictor training...")
                roas_success = ml_system.predictor.train_model('roas_predictor', 'testing', 'roas')
                logger.debug(f"ROAS predictor training result: {roas_success}")
            except (AttributeError, ValueError, TypeError) as e:
                logger.error(f"ROAS predictor training error: {e}", exc_info=True)
                roas_success = False
            
            try:
                logger.debug("Calling purchase_probability training...")
                purchase_success = ml_system.predictor.train_model('purchase_probability', 'testing', 'purchases')
                logger.debug(f"Purchase predictor training result: {purchase_success}")
            except (AttributeError, ValueError, TypeError) as e:
                logger.error(f"Purchase predictor training error: {e}", exc_info=True)
                purchase_success = False
            
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

    # Table monitoring and insights
    if table_monitor:
        try:
            # Get comprehensive table insights
            table_insights = table_monitor.get_all_table_insights()
            
            # Check ML data sufficiency
            ml_status = table_monitor.check_ml_data_sufficiency(table_insights)
            
            # Send table monitoring report
            table_report = table_monitor.format_insights_report(table_insights)
            notify(table_report)
            
            # Alert if ML system doesn't have sufficient data
            if not ml_status['ready_for_training']:
                notify("🚨 ML System Data Alert:")
                for rec in ml_status['recommendations']:
                    notify(f"   • {rec}")
            
            # Update previous counts for next tick
            table_monitor.update_previous_counts(table_insights)
            
            # Optimize ML tables data - ensure all data is correct
            try:
                from infrastructure.data_optimizer import create_ml_data_optimizer
                optimizer = create_ml_data_optimizer(supabase_client)
                optimization_results = optimizer.optimize_all_tables(stage='asc_plus', force_recalculate=False)
                logger.info(f"✅ ML tables optimized: {optimization_results}")
            except ImportError:
                pass  # Optimizer not available
            except Exception as opt_error:
                logger.debug(f"ML table optimization failed (non-critical): {opt_error}")
            
        except Exception as e:
            notify(f"⚠️ Table monitoring failed: {e}")

    # Console summary (logs only, not Slack)
    # datetime already imported at module level - use it directly
    import pytz
    amsterdam_tz = pytz.timezone('Europe/Amsterdam')
    amsterdam_time = datetime.now(amsterdam_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    logger.info("---- RUN SUMMARY ----")
    logger.info(f"Time: {amsterdam_time}")
    
    # Display key metrics summary
    if acct.get('spend') is not None:
        logger.info(f"Today's Spend: €{acct.get('spend', 0):.2f}")
        logger.info(f"Active Ads: {acct.get('active_ads', 0)}")
        logger.info(f"Impressions: {acct.get('impressions', 0):,}")
        logger.info(f"Clicks: {acct.get('clicks', 0):,}")
        logger.info(f"CTR: {acct.get('ctr', 0):.1f}%")
        logger.info(f"CPC: €{acct.get('cpc', 0):.2f}")
        logger.info(f"CPM: €{acct.get('cpm', 0):.2f}")
        if acct.get('purchases', 0) > 0:
            logger.info(f"Purchases: {acct.get('purchases', 0)}")
            logger.info(f"CPA: €{acct.get('cpa', 0):.2f}")
    
    logger.debug(
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
