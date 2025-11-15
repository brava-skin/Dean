from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

from io import TextIOWrapper

import pandas as pd
import yaml
from dotenv import load_dotenv

from config import (
    DB_CPC_MAX, DB_CPM_MAX, DB_ROAS_MAX, DB_CPA_MAX,
    MAX_AD_AGE_DAYS, DEFAULT_SAFE_FLOAT_MAX,
    validate_asc_plus_config
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def configure_logging_filters() -> None:
    noise_levels = {
        "httpx": logging.WARNING,
        "urllib3": logging.WARNING,
        "matplotlib": logging.ERROR,
        "matplotlib.font_manager": logging.ERROR,
        "sentence_transformers": logging.WARNING,
        "integrations.meta_client": logging.WARNING,
        "supabase": logging.WARNING,
        "infrastructure.supabase_storage": logging.WARNING,
        "creative.creative_intelligence": logging.WARNING,
    }
    for name, level in noise_levels.items():
        logging.getLogger(name).setLevel(level)
    try:
        from integrations import slack as slack_module
        slack_module.LOG_STDOUT = False
    except Exception:
        logger.debug("Slack logger could not be reconfigured", exc_info=True)


configure_logging_filters()


def _install_stdout_noise_filter() -> None:
    class _NoiseFilter(io.TextIOBase):
        def __init__(self, wrapped: TextIOWrapper):
            self._wrapped = wrapped
            self._buffer = ""
            self._suppressed_prefixes = (
                "📋 Campaign Configuration",
                "- Campaign Type:", "- Budget:", "- Target:", "- Audience:", "- Creative Type:",
                "📁 Checking configuration files",
                "total ", "drwx", "-rw-",
                "production.yaml", "rules.yaml", "settings.yaml",
                "# ── ASC+ Campaign Configuration",
                "asc_plus:", "  asc_plus:", "  daily_budget_eur:",
                "  keep_ads_live:", "  max_active_ads:", "  target_active_ads:",
                "  optimization_goal:", "# Fixed", "# ATC optimization",
            )

        def write(self, data: str) -> int:
            self._buffer += data
            written = 0
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if not self._should_suppress(line):
                    self._wrapped.write(line + "\n")
                    written += len(line) + 1
            return written

        def flush(self) -> None:
            if self._buffer and not self._should_suppress(self._buffer):
                self._wrapped.write(self._buffer)
            self._buffer = ""
            self._wrapped.flush()

        def _should_suppress(self, line: str) -> bool:
            stripped = line.strip()
            if not stripped:
                return False
            return any(stripped.startswith(prefix) for prefix in self._suppressed_prefixes)

    if not isinstance(sys.stdout, _NoiseFilter):
        sys.stdout = _NoiseFilter(sys.stdout)
    if not isinstance(sys.stderr, _NoiseFilter):
        sys.stderr = _NoiseFilter(sys.stderr)


_install_stdout_noise_filter()


def log_config_files(paths: Sequence[str]) -> None:
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            logger.debug("Config file %s (missing)", path)
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError:
            logger.debug("Config file %s (size unknown)", path)
        else:
            logger.debug("Config file %s (%d bytes)", path, size_bytes)


def _metric_to_float(value: Any) -> float:
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
            if isinstance(item, dict) and "value" in item:
                return _metric_to_float(item["value"])
            return _metric_to_float(item)
        return 0.0
    return 0.0

try:
    from supabase import create_client
except Exception:
    create_client = None

def get_reconciled_counters(account_snapshot, stage_result=None):
    return account_snapshot, stage_result or {}

from infrastructure import Store
from infrastructure.supabase_storage import create_supabase_storage
from infrastructure.data_validation import validate_all_timestamps
from integrations import notify, post_run_header_and_get_thread_ts, post_thread_ads_snapshot
from integrations import MetaClient, AccountAuth, ClientConfig
from rules.rules import AdvancedRuleEngine as RuleEngine
from stages.asc_plus import run_asc_plus_tick
from infrastructure import now_local
from infrastructure.utils import Timekit
from infrastructure import start_background_scheduler, stop_background_scheduler

REQUIRED_ENVS = [
    "FB_APP_ID", "FB_APP_SECRET", "FB_ACCESS_TOKEN", "FB_AD_ACCOUNT_ID",
    "FB_PIXEL_ID", "FB_PAGE_ID", "STORE_URL", "IG_ACTOR_ID",
]
REQUIRED_IDS = [
    ("ids", "asc_plus_campaign_id"),
    ("ids", "asc_plus_adset_id"),
]
DEFAULT_TZ = "Europe/Amsterdam"
MAX_STAGE_RETRIES = 3
RETRY_BACKOFF_BASE = 0.6
CIRCUIT_BREAKER_FAILS = 3
LOCKFILE = "data/run.lock"
SCHEMA_PATH_DEFAULT = "config/schema.settings.yaml"


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError, IOError, OSError):
        return {}


def load_cfg(settings_path: str, rules_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return load_yaml(settings_path), load_yaml(rules_path)


def _normalize_video_id_cell(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().strip("'").strip('"')
    if s == "" or s.lower() in ("nan", "none", "null"):
        return ""
    s = s.replace(",", "").replace(" ", "")
    if re.fullmatch(r"\d+", s):
        return s
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d+(\.\d+)?[eE]\+\d+", s):
        try:
            return str(int(float(s)))
        except (ValueError, OverflowError):
            return ""
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if digits else ""


def load_queue(path: str) -> pd.DataFrame:
    cols = [
        "creative_id", "name", "video_id", "thumbnail_url", "primary_text",
        "headline", "description", "page_id", "utm_params", "avatar",
        "visual_style", "script", "filename", "status",
    ]
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(
                path, dtype=str, keep_default_na=False,
                converters={"video_id": _normalize_video_id_cell},
            )
        else:
            try:
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
                df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (FileNotFoundError, IOError, OSError, pd.errors.ParserError):
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    try:
        if "video_id" in df.columns:
            df["video_id"] = df["video_id"].map(_normalize_video_id_cell)
    except (KeyError, AttributeError, TypeError):
        pass
    return df


def _get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not (create_client and url and key):
        return None
    try:
        return create_client(url, key)
    except (TypeError, ValueError, AttributeError):
        return None

def _get_validated_supabase():
    try:
        from infrastructure.supabase_storage import get_validated_supabase_client
        return get_validated_supabase_client(enable_validation=True)
    except ImportError:
        return _get_supabase()
    except (TypeError, ValueError, AttributeError):
        return None

def safe_float(value, max_val=None):
    if max_val is None:
        max_val = DEFAULT_SAFE_FLOAT_MAX
    try:
        val = float(value or 0)
        if not (val == val) or val == float('inf') or val == float('-inf'):
            return 0.0
        bounded_val = min(max(val, -max_val), max_val)
        return round(bounded_val, 4)
    except (ValueError, TypeError):
        return 0.0


def _calculate_performance_quality_score(ad_data: Dict[str, Any]) -> int:
    try:
        spend = safe_float(ad_data.get('spend', 0))
        impressions = safe_float(ad_data.get('impressions', 0))
        clicks = safe_float(ad_data.get('clicks', 0))
        purchases = safe_float(ad_data.get('purchases', 0))
        if spend <= 0 or impressions <= 0:
            return 0
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0
        cpa = spend / purchases if purchases > 0 else float('inf')
        quality_score = 0
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
            quality_score = min(quality_score, 50)
        return min(max(int(quality_score), 0), 100)
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error calculating performance quality score: {e}", exc_info=True)
        return 0


def _calculate_stability_score(ad_data: Dict[str, Any]) -> float:
    try:
        spend = safe_float(ad_data.get('spend', 0))
        impressions = safe_float(ad_data.get('impressions', 0))
        clicks = safe_float(ad_data.get('clicks', 0))
        if spend <= 0 or impressions <= 0:
            return 0.0
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0
        if ctr >= 3.0:
            return 9.0
        elif ctr >= 2.0:
            return 7.0
        elif ctr >= 1.0:
            return 5.0
        elif ctr >= 0.5:
            return 3.0
        elif ctr >= 0.1:
            return 1.0
        else:
            return 0.0
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error calculating stability score: {e}", exc_info=True)
        return 0.0


def _calculate_momentum_score(ad_data: Dict[str, Any]) -> float:
    try:
        spend = safe_float(ad_data.get('spend', 0))
        impressions = safe_float(ad_data.get('impressions', 0))
        clicks = safe_float(ad_data.get('clicks', 0))
        purchases = safe_float(ad_data.get('purchases', 0))
        if spend <= 0 or impressions <= 0:
            return 0.0
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0
        conversion_rate = (purchases / clicks) * 100 if clicks > 0 else 0
        momentum = 0.0
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
    try:
        spend = safe_float(ad_data.get('spend', 0))
        impressions = safe_float(ad_data.get('impressions', 0))
        clicks = safe_float(ad_data.get('clicks', 0))
        purchases = safe_float(ad_data.get('purchases', 0))
        if spend <= 0 or impressions <= 0:
            return 0.0
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0
        conversion_rate = (purchases / clicks) * 100 if clicks > 0 else 0
        fatigue = 0.0
        if ctr >= 3.0:
            fatigue += 0.0
        elif ctr >= 2.0:
            fatigue += 0.2
        elif ctr >= 1.0:
            fatigue += 0.4
        elif ctr >= 0.5:
            fatigue += 0.6
        else:
            fatigue += 0.8
        if conversion_rate >= 5.0:
            fatigue += 0.0
        elif conversion_rate >= 2.0:
            fatigue += 0.1
        elif conversion_rate >= 1.0:
            fatigue += 0.2
        elif conversion_rate >= 0.1:
            fatigue += 0.3
        else:
            fatigue += 0.4
        return min(fatigue / 2.0, 9.9999)
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error calculating fatigue index: {e}", exc_info=True)
        return 0.0


def store_performance_data_in_supabase(supabase_client, ad_data: Dict[str, Any], stage: str) -> None:
    if not supabase_client:
        logger.warning("No Supabase client available")
        return
    try:
        validated_client = _get_validated_supabase()
        if not validated_client:
            logger.warning("No validated Supabase client available, falling back to regular client")
            validated_client = supabase_client
        try:
            test_result = validated_client.select('performance_metrics').limit(1).execute()
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Supabase connection test failed: {e}", exc_info=True)
            notify(f"❌ Supabase connection failed: {e}")
            return
        current_date = datetime.now().strftime('%Y-%m-%d')
        date_start = current_date
        date_end = current_date
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
            pass
        ad_id = ad_data.get('ad_id', '')
        lifecycle_id = ad_data.get('lifecycle_id', f"lifecycle_{ad_id}")
        if not lifecycle_id or lifecycle_id == f"lifecycle_":
            logger.debug(f"Missing lifecycle_id: ad_id='{ad_id}', lifecycle_id='{lifecycle_id}', ad_data keys: {list(ad_data.keys())}")
        quality_score = _calculate_performance_quality_score(ad_data)
        stability_score = _calculate_stability_score(ad_data)
        momentum_score = _calculate_momentum_score(ad_data)
        fatigue_index = _calculate_fatigue_index(ad_data)
        logger.debug(f"Ad {ad_id} - CTR: {safe_float(ad_data.get('ctr', 0)):.2f}%, Quality: {quality_score}, Stability: {stability_score:.2f}, Momentum: {momentum_score:.2f}, Fatigue: {fatigue_index:.2f}")
        ctr_val = ad_data.get('ctr')
        cpc_val = ad_data.get('cpc')
        cpm_val = ad_data.get('cpm')
        roas_val = ad_data.get('roas')
        cpa_val = ad_data.get('cpa')
        impressions = int(ad_data.get('impressions', 0))
        add_to_cart = int(ad_data.get('atc', 0))
        purchases = int(ad_data.get('purchases', 0))
        atc_rate_val = ad_data.get('atc_rate')
        if atc_rate_val is None and impressions > 0:
            atc_rate_val = add_to_cart / impressions
        purchase_rate_val = ad_data.get('purchase_rate')
        if purchase_rate_val is None and impressions > 0:
            purchase_rate_val = purchases / impressions

        def clamp_fraction(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            return max(0.0, min(1.0, safe_float(value, 1.0)))

        creative_id = ad_data.get('creative_id') or ''
        campaign_id = ad_data.get('campaign_id') or ''
        adset_id = ad_data.get('adset_id') or ''
        normalized_performance_score = min(max(quality_score / 100.0, 0.0), 1.0) if quality_score is not None else None
        if creative_id and ad_id:
            try:
                existing = validated_client.client.table('ads').select('ad_id').eq('ad_id', ad_id).limit(1).execute()
                if existing.data:
                    ads_update = {
                        'ad_id': ad_id,
                        'performance_score': normalized_performance_score,
                        'fatigue_index': min(max(fatigue_index, 0.0), 1.0) if fatigue_index is not None else None,
                        'updated_at': datetime.now(timezone.utc).isoformat(),
                    }
                    if creative_id:
                        ads_update['creative_id'] = creative_id
                    if campaign_id:
                        ads_update['campaign_id'] = campaign_id
                    if adset_id:
                        ads_update['adset_id'] = adset_id
                    validated_client.client.table('ads').update(ads_update).eq('ad_id', ad_id).execute()
            except Exception as e:
                logger.debug(f"Failed to update existing ad in ads table: {e}")
        ad_exists = False
        if ad_id:
            try:
                existing = validated_client.client.table('ads').select('ad_id').eq('ad_id', ad_id).limit(1).execute()
                ad_exists = bool(existing.data)
            except Exception as e:
                logger.debug(f"Failed to check if ad exists: {e}")
                ad_exists = False
        if not ad_exists:
            logger.debug(f"Skipping performance_metrics insert for ad {ad_id} - ad does not exist in ads table")
            return
        performance_data = {
            'ad_id': ad_id,
            'window_type': '1d',
            'date_start': date_start,
            'date_end': ad_data.get('date_end', date_start),
            'impressions': impressions,
            'clicks': int(ad_data.get('clicks', 0)),
            'spend': safe_float(ad_data.get('spend', 0), 999999.99),
            'purchases': purchases,
            'add_to_cart': add_to_cart,
            'initiate_checkout': int(ad_data.get('ic', 0)),
            'ctr': clamp_fraction(ctr_val),
            'cpc': safe_float(cpc_val, DB_CPC_MAX) if cpc_val is not None else None,
            'cpm': safe_float(cpm_val, DB_CPM_MAX) if cpm_val is not None else None,
            'roas': safe_float(roas_val, DB_ROAS_MAX) if roas_val is not None else None,
            'cpa': safe_float(cpa_val, DB_CPA_MAX) if cpa_val is not None else None,
            'atc_rate': clamp_fraction(atc_rate_val),
            'purchase_rate': clamp_fraction(purchase_rate_val),
        }
        performance_data = validate_all_timestamps(performance_data)
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
                    creative_ids = {'image': creative_id}
                    creative_system.track_creative_performance(
                        ad_id=ad_data.get('ad_id', ''),
                        creative_ids=creative_ids,
                        performance_data=ad_data,
                        stage=stage
                    )
        except Exception as e:
            logger.debug(f"Failed to track creative performance: {e}")
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to store performance data in Supabase: {e}", exc_info=True)
        notify(f"❌ Failed to store performance data in Supabase: {e}")


def collect_stage_ad_data(meta_client, settings: Dict[str, Any], stage: str) -> Dict[str, Dict[str, Any]]:
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
                if action_type == "omni_add_to_cart":
                    add_to_cart += int(value)
                elif action_type == "add_to_cart":
                    if add_to_cart == 0:
                        add_to_cart = int(value)
                elif action_type == "initiate_checkout":
                    initiate_checkout += int(value)
                elif action_type == "purchase":
                    purchases += int(value)
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

def initialize_creative_intelligence_system(supabase_client, settings) -> Optional[Any]:
    try:
        from creative.creative_intelligence import create_creative_intelligence_system
        openai_api_key = os.getenv("OPENAI_API_KEY")
        creative_system = create_creative_intelligence_system(
            supabase_client=supabase_client,
            openai_api_key=openai_api_key,
            settings=settings
        )
        notify("🎨 Creative Intelligence System initialized")
        return creative_system
    except Exception as e:
        notify(f"⚠️ Failed to initialize Creative Intelligence System: {e}")
        return None


def load_queue_supabase(
    table: str = None,
    status_filter: str = "pending",
    limit: int = 64,
) -> pd.DataFrame:
    cols = [
        "creative_id", "name", "video_id", "thumbnail_url", "primary_text",
        "headline", "description", "page_id", "utm_params", "avatar",
        "visual_style", "script", "filename", "status",
    ]
    sb = _get_supabase()
    if not sb:
        notify("⚠️ Supabase client not available; falling back to file-based queue.")
        return pd.DataFrame(columns=cols)
    table = table or os.getenv("SUPABASE_TABLE", "meta_creatives")
    try:
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
        rows.append({
            "creative_id": r.get("id") or "",
            "name": "",
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
            "status": (r.get("status") or "").lower(),
        })
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str).fillna("")
    try:
        df["video_id"] = df["video_id"].map(_normalize_video_id_cell)
    except Exception:
        pass
    return df


def set_supabase_status(
    ids_or_video_ids: List[str],
    new_status: str,
    *,
    use_column: str = "id",
    table: str = None,
) -> None:
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


def validate_envs(required: List[str]) -> List[str]:
    return [k for k in required if not os.getenv(k)]

def validate_asc_plus_envs() -> List[str]:
    required = [
        "FB_ACCESS_TOKEN", "FB_AD_ACCOUNT_ID", "FB_PAGE_ID",
        "FLUX_API_KEY", "OPENAI_API_KEY",
    ]
    missing = []
    for var in required:
        if not os.getenv(var):
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
    cfg_tz = settings.get("account", {}).get("timezone") or settings.get("account_timezone") or settings.get("timezone") or DEFAULT_TZ
    env_tz = os.getenv("TIMEZONE")
    if env_tz and env_tz != cfg_tz:
        issues.append(f"Timezone mismatch? config={cfg_tz} env={env_tz}")
    for k in ("ids", "asc_plus", "logging"):
        if k not in settings:
            issues.append(f"Missing section: {k}")
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


@contextmanager
def file_lock(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR)
        locked = False
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except Exception:
            try:
                if os.path.getsize(path) > 0:
                    raise RuntimeError("Lock already held")
            except (OSError, IOError):
                pass
            locked = True
        if locked:
            try:
                os.ftruncate(fd, 0)
                os.write(fd, str(os.getpid()).encode())
            except (OSError, IOError):
                pass
            yield
    finally:
        if fd is not None:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (OSError, IOError, AttributeError):
                pass
            try:
                os.close(fd)
            except (OSError, IOError):
                pass
            try:
                os.remove(path)
            except (OSError, IOError):
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


def health_check(store: Store, client: MetaClient) -> Dict[str, Any]:
    ok = True
    details: List[str] = []
    try:
        store.incr("healthcheck", 1)
        details.append("db:ok")
    except Exception as e:
        ok = False
        details.append(f"db:fail:{e}")
    try:
        details.append("slack:ok")
    except Exception as e:
        details.append(f"slack:warn:{e}")
    try:
        client.get_ad_insights(level="ad", fields=["spend"], paginate=False)
        details.append("meta:ok")
    except Exception as e:
        ok = False
        details.append(f"meta:fail:{e}")
    return {"ok": ok, "details": details}


def check_ad_account_health(client: MetaClient, settings: Dict[str, Any]) -> Dict[str, Any]:
    account_health_config = settings.get("account_health", {})
    if not account_health_config.get("enabled", True):
        return {"ok": True, "disabled": True}
    try:
        health_result = client.check_account_health()
        if not health_result["ok"]:
            critical_issues = health_result.get("critical_issues", [])
            account_id = client.ad_account_id_act
            from integrations import alert_ad_account_health_critical
            alert_ad_account_health_critical(account_id, critical_issues)
            health_details = health_result.get("health_details", {})
            return {"ok": False, "critical_issues": critical_issues, "health_details": health_details}
        else:
            warnings = health_result.get("warnings", [])
            health_details = health_result.get("health_details", {})
            account_id = client.ad_account_id_act
            currency = settings.get("economics", {}).get("currency", "EUR")
            spent = health_details.get("amount_spent")
            cap = health_details.get("spend_cap")
            if spent is not None and cap is not None and cap > 0:
                percentage = (spent / cap) * 100
                warning_threshold = account_health_config.get("thresholds", {}).get("spend_cap_warning_pct", 80)
                if percentage >= warning_threshold:
                    from integrations import alert_spend_cap_approaching
                    alert_spend_cap_approaching(account_id, spent, cap, currency)
            if warnings:
                from integrations import alert_ad_account_health_warning
                alert_ad_account_health_warning(account_id, warnings)
            return {"ok": True, "warnings": warnings, "health_details": health_details}
    except Exception as e:
        account_id = getattr(client, 'ad_account_id_act', 'unknown')
        from integrations import alert_ad_account_health_critical
        alert_ad_account_health_critical(account_id, [f"Health check failed: {str(e)}"])
        return {"ok": False, "error": str(e)}


def account_guardrail_ping(meta: MetaClient, settings: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import zoneinfo
        tz_name = settings.get("account", {}).get("timezone", "Europe/Amsterdam")
        local_tz = zoneinfo.ZoneInfo(tz_name)
        now = datetime.now(local_tz)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        rows = meta.get_ad_insights(
            level="ad",
            fields=["spend", "actions", "impressions", "clicks", "inline_link_clicks",
                    "inline_link_click_ctr", "cost_per_inline_link_click", "cpc", "cpm"],
            time_range={
                "since": midnight.strftime("%Y-%m-%d"),
                "until": now.strftime("%Y-%m-%d")
            },
            paginate=True
        )
        spend = sum(_metric_to_float(r.get("spend")) for r in rows)
        impressions = int(round(sum(_metric_to_float(r.get("impressions")) for r in rows)))
        link_clicks = sum(
            _metric_to_float(r.get("inline_link_clicks")) or _metric_to_float(r.get("link_clicks"))
            for r in rows
        )
        all_clicks = sum(_metric_to_float(r.get("clicks")) for r in rows)
        purch = 0.0
        atc = 0.0
        ic = 0.0
        if link_clicks <= 0 and all_clicks > 0:
            link_clicks = all_clicks
        for r in rows:
            for a in (r.get("actions") or []):
                action_type = a.get("action_type")
                try:
                    value = float(a.get("value") or 0)
                    if action_type == "purchase":
                        purch += value
                    elif action_type == "omni_add_to_cart":
                        atc += value
                    elif action_type == "add_to_cart":
                        atc += value
                    elif action_type == "initiate_checkout":
                        ic += value
                except (KeyError, TypeError, ValueError):
                    continue
        link_clicks = max(link_clicks, 0.0)
        all_clicks = max(all_clicks, 0.0)
        cpa = (spend / purch) if purch > 0 else None
        ctr = (link_clicks / impressions) if impressions > 0 else None
        cpc = None
        cpc_values = [_metric_to_float(row.get("cpc")) for row in rows if row.get("cpc")]
        if cpc_values:
            cpc_weights = [all_clicks / len(rows) for _ in cpc_values]
            weighted_cpc = sum(c * w for c, w in zip(cpc_values, cpc_weights) if c)
            total_weight = sum(w for w in cpc_weights)
            if weighted_cpc and total_weight:
                cpc = weighted_cpc / total_weight
        if cpc is None and all_clicks > 0:
            cpc = spend / all_clicks
        cpm = (spend * 1000 / impressions) if impressions > 0 else None
        cost_per_atc = (spend / atc) if atc > 0 else None
        be = float(
            os.getenv("BREAKEVEN_CPA")
            or (settings.get("economics", {}) or {}).get("breakeven_cpa")
            or 27.51
        )
        try:
            insights_ads_count = len(rows) if rows else 0
            try:
                all_active_ads = meta.get_ad_insights(
                    level="ad",
                    fields=["ad_id"],
                    date_preset="maximum",
                    filtering=[{"field": "ad.effective_status", "operator": "IN", "value": ["ACTIVE"]}]
                )
                total_active_count = len(all_active_ads) if all_active_ads else 0
                active_ads_count = max(insights_ads_count, total_active_count)
            except (AttributeError, TypeError, ValueError, KeyError):
                active_ads_count = insights_ads_count
        except (AttributeError, TypeError, ValueError, KeyError):
            active_ads_count = 0
        return {
            "spend": round(spend, 2),
            "purchases": int(purch),
            "cpa": round(cpa, 2) if cpa is not None else None,
            "breakeven": be,
            "impressions": impressions,
            "clicks": int(round(link_clicks if link_clicks > 0 else all_clicks)),
            "ctr": round(ctr, 4) if ctr is not None else None,
            "cpc": round(cpc, 2) if cpc is not None else None,
            "cpm": round(cpm, 2) if cpm is not None else None,
            "atc": int(atc),
            "ic": int(ic),
            "cost_per_atc": round(cost_per_atc, 2) if cost_per_atc is not None else None,
            "active_ads": active_ads_count,
        }
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        logger.error("Failed to calculate account metrics", exc_info=True)
        return {
            "spend": None, "purchases": None, "cpa": None, "breakeven": None,
            "impressions": None, "clicks": None, "ctr": None, "cpc": None,
            "cpm": None, "atc": None, "ic": None, "cost_per_atc": None, "active_ads": None
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Brava - Continuous Creative Testing & Scaling")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--rules", default="config/rules.yaml")
    parser.add_argument("--schema", default=SCHEMA_PATH_DEFAULT)
    parser.add_argument("--stage", choices=["all", "asc_plus"], default="all")
    parser.add_argument("--profile", choices=["production", "staging"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continuous-mode", action="store_true", help="Continuous operation mode (DigitalOcean)")
    parser.add_argument("--simulate", action="store_true", help="shadow mode: log intended actions only")
    parser.add_argument("--since", default=None, help="simulation since (YYYY-MM-DD)")
    parser.add_argument("--until", default=None, help="simulation until (YYYY-MM-DD)")
    parser.add_argument("--explain", action="store_true", help="print decisions without acting")
    parser.add_argument("--background", action="store_true", help="run in background mode with automated scheduling")
    args = parser.parse_args()

    load_dotenv()
    settings, rules_cfg = load_cfg(args.settings, args.rules)
    config_paths = [args.settings, args.rules]
    log_config_files(config_paths)

    if args.continuous_mode:
        notify("🔄 24/7 Continuous mode enabled - optimized for DigitalOcean deployment")
        notify("📊 Maximum UI protection and ML data feeding active")
        notify("🛡️ Single concurrent request for UI protection")
        os.environ["META_REQUEST_DELAY"] = "2.0"
        os.environ["META_PEAK_HOURS_DELAY"] = "3.0"
        os.environ["META_NIGHT_HOURS_DELAY"] = "1.5"
        os.environ["META_MAX_CONCURRENT_INSIGHTS"] = "1"
        os.environ["META_RETRY_MAX"] = "12"
        os.environ["META_BACKOFF_BASE"] = "2.0"
        os.environ["META_USAGE_THRESHOLD"] = "0.6"
        os.environ["META_EMERGENCY_THRESHOLD"] = "0.8"
        os.environ["META_UI_PROTECTION_MODE"] = "true"

    if rules_cfg:
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()
            for key, value in override.items():
                if key == "asc_plus_atc" and isinstance(value, dict):
                    if "asc_plus" not in result:
                        result["asc_plus"] = {}
                    if isinstance(result["asc_plus"], dict):
                        result["asc_plus"].update(value)
                    continue
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    if key == "asc_plus":
                        merged_asc_plus = result[key].copy()
                        merged_asc_plus.update(value)
                        result[key] = merged_asc_plus
                    else:
                        result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        settings = deep_merge(settings, rules_cfg)

    try:
        validate_asc_plus_config(settings)
    except ValueError as exc:
        notify(f"❌ ASC+ configuration invalid: {exc}")
        print("Fatal configuration error. Exiting.", file=sys.stderr)
        sys.exit(1)

    profile = (
        args.profile
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

    try:
        schema = load_yaml(args.schema)
        if schema:
            import jsonschema
            jsonschema.validate(instance=settings, schema=schema)
    except (ImportError, jsonschema.ValidationError) as e:
        logger.debug(f"Schema validation skipped or failed: {e}")

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

    sqlite_path = settings.get("logging", {}).get("sqlite", {}).get("path", "dean/data/state.sqlite")
    store = Store(sqlite_path)

    tz_name = (
        settings.get("account", {}).get("timezone")
        or settings.get("account_timezone")
        or os.getenv("TIMEZONE")
        or DEFAULT_TZ
    )

    account = AccountAuth(
        account_id=os.getenv("FB_AD_ACCOUNT_ID", ""),
        access_token=os.getenv("FB_ACCESS_TOKEN", ""),
        app_id=os.getenv("FB_APP_ID", ""),
        app_secret=os.getenv("FB_APP_SECRET", ""),
        api_version=os.getenv("FB_API_VERSION") or None,
    )
    client_cfg = ClientConfig(timezone=tz_name)
    client = MetaClient(
        accounts=[account],
        cfg=client_cfg,
        store=store,
        dry_run=(effective_dry or shadow_mode),
        tenant_id=settings.get("branding_name", "default"),
    )

    engine = RuleEngine(rules_cfg, store)
    supabase_client = _get_supabase()

    supabase_storage = None
    if supabase_client:
        try:
            supabase_storage = create_supabase_storage(supabase_client)
            notify("📊 Supabase storage initialized for ad creation times and historical data")
            engine = RuleEngine(rules_cfg, supabase_storage)
            notify("📊 Rule engine updated to use Supabase storage")
        except Exception as e:
            notify(f"⚠️ Failed to initialize Supabase storage: {e}")
            supabase_storage = None

    creative_system = None
    if supabase_client:
        try:
            creative_system = initialize_creative_intelligence_system(supabase_client, settings)
        except Exception as e:
            notify(f"⚠️ Creative Intelligence System initialization error: {e}")

    missing_envs = validate_asc_plus_envs()
    if missing_envs:
        notify(f"❌ Missing required environment variables for ASC+ campaign: {', '.join(missing_envs)}")
        notify("   Required: FB_ACCESS_TOKEN, FB_AD_ACCOUNT_ID, FB_PAGE_ID, FLUX_API_KEY, OPENAI_API_KEY")
        if not (profile == "staging" or effective_dry):
            sys.exit(1)
    else:
        notify("✅ All required ASC+ environment variables are set")

    hc = health_check(store, client)
    if not hc["ok"]:
        notify("🛑 Preflight failed: " + " ".join(hc["details"]))
        if not (profile == "staging" or effective_dry):
            sys.exit(1)

    account_health = check_ad_account_health(client, settings)
    if not account_health["ok"]:
        notify("🚨 Ad account health issues detected - check alerts for details")
        if account_health.get("critical_issues"):
            notify(f"Critical issues: {', '.join(account_health['critical_issues'])}")

    local_now = now_local(tz_name)
    acct = account_guardrail_ping(client, settings)

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

    try:
        tkey = f"tick::{local_now:%Y-%m-%dT%H:%M}"
        if hasattr(store, "tick_seen") and store.tick_seen(tkey):
            notify("ℹ️ Tick already processed; exiting.")
            return
    except (AttributeError, TypeError):
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
                    client.dry_run = True
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
                        pass
                res = wrapped(*fn_args, **fn_kwargs)
                dt = time.time() - t0
                failures_in_row = 0
                return res
            except Exception as e:
                failures_in_row += 1
                dt = time.time() - t0
                notify(f"❌ [{label}] {dt:.1f}s - {e}")
                if failures_in_row >= CIRCUIT_BREAKER_FAILS:
                    notify(f"🧯 Circuit breaker tripped ({failures_in_row}); switching to read-only for remainder.")
                    client.dry_run = True
                return None

        stage_choice = args.stage
        stage_summaries = []

        overall["asc_plus"] = run_stage(
            run_asc_plus_tick,
            "ASC+",
            client,
            settings,
            rules_cfg,
            store,
        )
        asc_result = overall.get("asc_plus")

        reconciled_counts: Dict[str, Any] = {}
        account_info, reconciled_counts = get_reconciled_counters(account_info, asc_result)
        if asc_result:
            stage_summaries.append({
                "stage": "ASC+",
                "result": asc_result,
                "counts": reconciled_counts,
            })

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
                        store_performance_data_in_supabase(supabase_client, ad_data, "asc_plus")
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

    if not shadow_mode:
        time_str = local_now.strftime("%H:%M %Z")
        degraded_mode = failures_in_row > 0 or (overall.get("asc_plus", {}).get("health") == "WARNING")
        status = "DEGRADED" if degraded_mode else "OK"
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
            ic=account_info['ic'],
            cost_per_atc=account_info.get('cost_per_atc')
        )
        try:
            if stage_choice in ("all", "asc_plus"):
                import zoneinfo
                attr_windows = ["7d_click", "1d_view"]
                local_tz = zoneinfo.ZoneInfo(tz_name)
                now = datetime.now(local_tz)
                midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                asc_plus_adset_id = settings.get("ids", {}).get("asc_plus_adset_id")
                filtering_today = []
                filtering_lifetime = []
                if asc_plus_adset_id:
                    filtering_today = [{"field": "adset.id", "operator": "IN", "value": [asc_plus_adset_id]}]
                    filtering_lifetime = [{"field": "adset.id", "operator": "IN", "value": [asc_plus_adset_id]}]
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
                try:
                    active_ad_ids = set()
                    if asc_plus_adset_id:
                        active_ads = client.list_ads_in_adset(asc_plus_adset_id)
                        active_ad_ids = {str(ad.get("id", "")) for ad in active_ads if str(ad.get("status", "")).upper() == "ACTIVE"}
                    rows_today = [r for r in rows_today_raw if str(r.get("ad_id", "")) in active_ad_ids] if active_ad_ids else rows_today_raw
                except Exception as e:
                    logger.warning(f"Failed to filter active ads, using all: {e}")
                    rows_today = rows_today_raw
                rows_lifetime = client.get_ad_insights(
                    level="ad",
                    filtering=filtering_lifetime,
                    fields=["ad_id", "spend", "actions"],
                    time_range={
                        "since": "2024-01-01",
                        "until": now.strftime("%Y-%m-%d")
                    },
                    action_attribution_windows=list(attr_windows),
                    paginate=True
                ) or []
                from integrations import build_ads_snapshot
                ad_lines = build_ads_snapshot(rows_today or [], rows_lifetime or [], tz_name)
                if ad_lines:
                    post_thread_ads_snapshot(thread_ts, ad_lines)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Failed to post ads snapshot: {e}")

    import pytz
    amsterdam_tz = pytz.timezone('Europe/Amsterdam')
    amsterdam_time = datetime.now(amsterdam_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info("---- RUN SUMMARY ----")
    logger.info(f"Time: {amsterdam_time}")
    if acct.get('spend') is not None:
        logger.info(f"Today's Spend: €{acct.get('spend', 0):.2f}")
        logger.info(f"Active Ads: {acct.get('active_ads', 0)}")
        logger.info(f"Impressions: {acct.get('impressions', 0):,}")
        logger.info(f"Clicks: {acct.get('clicks', 0):,}")
        ctr_display = acct.get('ctr', 0) or 0.0
        logger.info(f"CTR: {ctr_display * 100:.1f}%")
        cpc_value = acct.get('cpc')
        cpm_value = acct.get('cpm')
        logger.info(f"CPC: €{(cpc_value if cpc_value is not None else 0):.2f}")
        logger.info(f"CPM: €{(cpm_value if cpm_value is not None else 0):.2f}")
        if acct.get('purchases', 0) > 0:
            logger.info(f"Purchases: {acct.get('purchases', 0)}")
            cpa_value = acct.get('cpa')
            logger.info(f"CPA: €{(cpa_value if cpa_value is not None else 0):.2f}")
    logger.debug(json.dumps({"profile": profile, "dry_run": client.dry_run, "simulate": shadow_mode, "acct": acct}, indent=2))

    if args.background:
        notify("🤖 Starting background scheduler mode")
        start_background_scheduler(settings, rules_cfg, store)
        try:
            import signal
            def signal_handler(sig, frame):
                notify("🛑 Background scheduler stopping...")
                stop_background_scheduler()
                sys.exit(0)
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
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
