"""
DEAN SELF-LEARNING META ADS AUTOMATION SYSTEM
Advanced ML-Enhanced Main Runner

This is the next-generation main runner that integrates the complete ML intelligence system
with Supabase backend, adaptive rules, and advanced performance tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv

# ML Intelligence System
from ml_intelligence import MLIntelligenceSystem, MLConfig, create_ml_system
from adaptive_rules import IntelligentRuleEngine, RuleConfig, create_intelligent_rule_engine
from performance_tracking import PerformanceTrackingSystem, create_performance_tracking_system

# Legacy modules (updated for ML integration)
from storage import Store
from slack import (
    notify, post_run_header_and_get_thread_ts, post_thread_ads_snapshot,
    prettify_ad_name, fmt_eur, fmt_pct, fmt_roas, fmt_int,
    alert_kill, alert_promote, alert_scale, alert_fatigue,
    alert_data_quality, alert_error, alert_queue_empty, alert_new_launch
)
from meta_client import MetaClient, AccountAuth, ClientConfig
from rules import RuleEngine
from stages.testing import run_testing_tick
from stages.validation import run_validation_tick
from stages.scaling import run_scaling_tick
from utils import (
    now_local, getenv_f, getenv_i, getenv_b, cfg, cfg_or_env_f, 
    cfg_or_env_i, cfg_or_env_b, cfg_or_env_list, safe_f, today_str,
    daily_key, ad_day_flag_key, now_minute_key, clean_text_token, prettify_ad_name
)
from scheduler import start_background_scheduler, stop_background_scheduler, get_scheduler

# =====================================================
# CONFIGURATION AND CONSTANTS
# =====================================================

REQUIRED_ENVS = [
    "FB_APP_ID", "FB_APP_SECRET", "FB_ACCESS_TOKEN", "FB_AD_ACCOUNT_ID",
    "FB_PIXEL_ID", "FB_PAGE_ID", "STORE_URL", "IG_ACTOR_ID",
    "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"  # ML system requirements
]

REQUIRED_IDS = [
    ("ids", "testing_campaign_id"), ("ids", "testing_adset_id"),
    ("ids", "validation_campaign_id"), ("ids", "scaling_campaign_id")
]

DEFAULT_TZ = "Europe/Amsterdam"
DIGEST_DIR = "data/digests"
MAX_STAGE_RETRIES = 3
RETRY_BACKOFF_BASE = 0.6
CIRCUIT_BREAKER_FAILS = 3
LOCKFILE = "data/run.lock"
SCHEMA_PATH_DEFAULT = "config/schema.settings.yaml"

UTC = timezone.utc

# =====================================================
# ML SYSTEM INITIALIZATION
# =====================================================

class MLEnhancedDean:
    """Main ML-enhanced Dean system orchestrator."""
    
    def __init__(self, settings: Dict[str, Any], rules: Dict[str, Any], store: Store):
        self.settings = settings
        self.rules = rules
        self.store = store
        
        # Initialize ML system
        self.ml_config = MLConfig(
            retrain_frequency_hours=24,
            prediction_horizon_hours=24,
            confidence_threshold=0.7,
            learning_rate=0.1,
            adaptation_rate=0.05
        )
        
        self.rule_config = RuleConfig(
            target_cpa=27.50,
            target_roas=2.0,
            target_ctr=0.008,
            learning_rate=0.1,
            max_adjustment_pct=0.2
        )
        
        # Initialize ML components
        self.ml_system = None
        self.rule_engine = None
        self.performance_tracker = None
        
        # Legacy components
        self.meta_client = None
        self.legacy_rule_engine = None
        
        self.logger = logging.getLogger(f"{__name__}.MLEnhancedDean")
    
    def initialize(self) -> bool:
        """Initialize all ML and legacy components."""
        try:
            self.logger.info("Initializing ML-enhanced Dean system...")
            
            # Initialize Supabase clients
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            
            if not supabase_url or not supabase_key:
                self.logger.error("Supabase credentials not found")
                return False
            
            # Initialize ML system
            self.ml_system = create_ml_system(supabase_url, supabase_key, self.ml_config)
            self.rule_engine = create_intelligent_rule_engine(
                supabase_url, supabase_key, self.ml_system, self.rule_config
            )
            self.performance_tracker = create_performance_tracking_system(
                supabase_url, supabase_key
            )
            
            # Initialize legacy Meta client
            account = AccountAuth(
                account_id=os.getenv("FB_AD_ACCOUNT_ID", ""),
                access_token=os.getenv("FB_ACCESS_TOKEN", ""),
                app_id=os.getenv("FB_APP_ID", ""),
                app_secret=os.getenv("FB_APP_SECRET", ""),
                api_version=os.getenv("FB_API_VERSION") or None,
            )
            
            tz_name = (
                self.settings.get("account_timezone")
                or self.settings.get("timezone")
                or os.getenv("TIMEZONE")
                or DEFAULT_TZ
            )
            
            cfg = ClientConfig(timezone=tz_name)
            self.meta_client = MetaClient(
                accounts=[account],
                cfg=cfg,
                store=self.store,
                dry_run=False,
                tenant_id=self.settings.get("branding_name", "default"),
            )
            
            # Initialize legacy rule engine for backward compatibility
            self.legacy_rule_engine = RuleEngine(self.rules)
            
            self.logger.info("ML-enhanced Dean system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing ML system: {e}")
            return False
    
    def run_ml_enhanced_tick(self, stage: str, queue_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run ML-enhanced tick for a specific stage."""
        try:
            self.logger.info(f"Running ML-enhanced {stage} tick...")
            
            # Get current performance data
            performance_data = self._get_stage_performance_data(stage)
            
            # Run ML analysis
            ml_insights = self._analyze_stage_with_ml(stage, performance_data)
            
            # Run legacy stage logic with ML enhancements
            if stage == "testing":
                result = self._run_ml_enhanced_testing(queue_df, ml_insights)
            elif stage == "validation":
                result = self._run_ml_enhanced_validation(ml_insights)
            elif stage == "scaling":
                result = self._run_ml_enhanced_scaling(ml_insights)
            else:
                result = {"error": f"Unknown stage: {stage}"}
            
            # Update adaptive rules based on performance
            self._adapt_stage_rules(stage, performance_data, ml_insights)
            
            # Save ML insights
            self._save_ml_insights(stage, ml_insights, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running ML-enhanced {stage} tick: {e}")
            return {"error": str(e)}
    
    def _get_stage_performance_data(self, stage: str) -> Dict[str, Any]:
        """Get current performance data for a stage."""
        try:
            # Get ad insights for the stage
            rows = self.meta_client.get_ad_insights(
                level="ad",
                fields=["ad_id", "ad_name", "spend", "impressions", "clicks", "actions", "action_values"],
                time_range={"since": today_ymd_account(), "until": today_ymd_account()},
                paginate=True
            )
            
            # Filter by stage
            stage_ads = [row for row in rows if f"[{stage.upper()}]" in row.get("ad_name", "")]
            
            if not stage_ads:
                return {}
            
            # Calculate stage metrics
            total_spend = sum(float(r.get("spend", 0)) for r in stage_ads)
            total_impressions = sum(int(r.get("impressions", 0)) for r in stage_ads)
            total_clicks = sum(int(r.get("clicks", 0)) for r in stage_ads)
            total_purchases = 0
            total_revenue = 0
            
            for row in stage_ads:
                for action in (row.get("actions") or []):
                    if action.get("action_type") == "purchase":
                        total_purchases += int(action.get("value", 0))
                
                for value in (row.get("action_values") or []):
                    if value.get("action_type") == "purchase":
                        total_revenue += float(value.get("value", 0))
            
            # Calculate derived metrics
            ctr = (total_clicks / total_impressions) if total_impressions > 0 else 0
            cpa = (total_spend / total_purchases) if total_purchases > 0 else 0
            roas = (total_revenue / total_spend) if total_spend > 0 else 0
            
            return {
                "stage": stage,
                "total_ads": len(stage_ads),
                "total_spend": total_spend,
                "total_impressions": total_impressions,
                "total_clicks": total_clicks,
                "total_purchases": total_purchases,
                "total_revenue": total_revenue,
                "avg_ctr": ctr,
                "avg_cpa": cpa,
                "avg_roas": roas,
                "ads": stage_ads
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stage performance data: {e}")
            return {}
    
    def _analyze_stage_with_ml(self, stage: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stage performance using ML intelligence."""
        try:
            if not performance_data or not performance_data.get("ads"):
                return {}
            
            # Analyze each ad with ML
            ml_insights = {
                "stage": stage,
                "ad_analyses": {},
                "stage_insights": {},
                "recommendations": []
            }
            
            for ad_data in performance_data["ads"]:
                ad_id = ad_data.get("ad_id")
                if not ad_id:
                    continue
                
                # Get ML analysis for this ad
                ad_analysis = self.ml_system.analyze_ad_intelligence(ad_id, stage)
                if ad_analysis:
                    ml_insights["ad_analyses"][ad_id] = ad_analysis
                    
                    # Extract key insights
                    intelligence_score = ad_analysis.get("intelligence_score", 0)
                    fatigue_analysis = ad_analysis.get("fatigue_analysis", {})
                    predictions = ad_analysis.get("predictions")
                    
                    # Generate recommendations
                    if intelligence_score < 0.3:
                        ml_insights["recommendations"].append(f"Ad {ad_id}: Low intelligence score ({intelligence_score:.2f})")
                    
                    if fatigue_analysis.get("fatigue_score", 0) > 0.7:
                        ml_insights["recommendations"].append(f"Ad {ad_id}: High fatigue detected")
                    
                    if predictions and predictions.get("predicted_value", 0) > 50:  # High predicted CPA
                        ml_insights["recommendations"].append(f"Ad {ad_id}: ML predicts high CPA")
            
            # Stage-level insights
            ml_insights["stage_insights"] = {
                "total_ads_analyzed": len(ml_insights["ad_analyses"]),
                "avg_intelligence_score": np.mean([
                    analysis.get("intelligence_score", 0) 
                    for analysis in ml_insights["ad_analyses"].values()
                ]) if ml_insights["ad_analyses"] else 0,
                "high_fatigue_ads": sum(
                    1 for analysis in ml_insights["ad_analyses"].values()
                    if analysis.get("fatigue_analysis", {}).get("fatigue_score", 0) > 0.7
                ),
                "ml_recommendations_count": len(ml_insights["recommendations"])
            }
            
            return ml_insights
            
        except Exception as e:
            self.logger.error(f"Error analyzing stage with ML: {e}")
            return {}
    
    def _run_ml_enhanced_testing(self, queue_df: Optional[pd.DataFrame], 
                                ml_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML-enhanced testing stage."""
        try:
            # Run legacy testing with ML enhancements
            result = run_testing_tick(
                self.meta_client,
                self.settings,
                self.legacy_rule_engine,
                self.store,
                queue_df or pd.DataFrame(),
                set_supabase_status=None,  # Will be handled by ML system
                placements=["facebook", "instagram"],
                instagram_actor_id=os.getenv("IG_ACTOR_ID"),
            )
            
            # Enhance result with ML insights
            if result and ml_insights:
                result["ml_insights"] = ml_insights
                result["intelligence_score"] = ml_insights.get("stage_insights", {}).get("avg_intelligence_score", 0)
                result["ml_recommendations"] = ml_insights.get("recommendations", [])
            
            return result or {}
            
        except Exception as e:
            self.logger.error(f"Error running ML-enhanced testing: {e}")
            return {"error": str(e)}
    
    def _run_ml_enhanced_validation(self, ml_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML-enhanced validation stage."""
        try:
            # Run legacy validation with ML enhancements
            result = run_validation_tick(
                self.meta_client,
                self.settings,
                self.legacy_rule_engine,
                self.store,
            )
            
            # Enhance result with ML insights
            if result and ml_insights:
                result["ml_insights"] = ml_insights
                result["intelligence_score"] = ml_insights.get("stage_insights", {}).get("avg_intelligence_score", 0)
                result["ml_recommendations"] = ml_insights.get("recommendations", [])
            
            return result or {}
            
        except Exception as e:
            self.logger.error(f"Error running ML-enhanced validation: {e}")
            return {"error": str(e)}
    
    def _run_ml_enhanced_scaling(self, ml_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML-enhanced scaling stage."""
        try:
            # Run legacy scaling with ML enhancements
            result = run_scaling_tick(
                self.meta_client,
                self.settings,
                self.store,
            )
            
            # Enhance result with ML insights
            if result and ml_insights:
                result["ml_insights"] = ml_insights
                result["intelligence_score"] = ml_insights.get("stage_insights", {}).get("avg_intelligence_score", 0)
                result["ml_recommendations"] = ml_insights.get("recommendations", [])
            
            return result or {}
            
        except Exception as e:
            self.logger.error(f"Error running ML-enhanced scaling: {e}")
            return {"error": str(e)}
    
    def _adapt_stage_rules(self, stage: str, performance_data: Dict[str, Any], 
                          ml_insights: Dict[str, Any]) -> None:
        """Adapt stage rules based on performance and ML insights."""
        try:
            # Use intelligent rule engine to adapt rules
            if self.rule_engine:
                success = self.rule_engine.adapt_stage_rules(stage)
                if success:
                    self.logger.info(f"Successfully adapted rules for {stage} stage")
                else:
                    self.logger.warning(f"Failed to adapt rules for {stage} stage")
            
        except Exception as e:
            self.logger.error(f"Error adapting stage rules: {e}")
    
    def _save_ml_insights(self, stage: str, ml_insights: Dict[str, Any], 
                         result: Dict[str, Any]) -> None:
        """Save ML insights to database."""
        try:
            # This would save insights to Supabase for future learning
            # Implementation depends on specific database schema
            pass
            
        except Exception as e:
            self.logger.error(f"Error saving ML insights: {e}")

# =====================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =====================================================

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML document or return empty dict on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def load_cfg(settings_path: str, rules_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return load_yaml(settings_path), load_yaml(rules_path)

def load_queue_supabase(table: str = None, status_filter: str = "pending", limit: int = 64) -> pd.DataFrame:
    """Load queue from Supabase (legacy compatibility)."""
    try:
        from supabase import create_client
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            return pd.DataFrame()
        
        client = create_client(supabase_url, supabase_key)
        table = table or os.getenv("SUPABASE_TABLE", "meta_creatives")
        
        response = client.table(table).select("*").or_(
            f"status.is.null,status.eq.{status_filter}"
        ).limit(limit).execute()
        
        if not response.data:
            return pd.DataFrame()
        
        # Convert to expected format
        rows = []
        for r in response.data:
            rows.append({
                "creative_id": str(r.get("id", "")),
                "name": "",
                "video_id": str(r.get("video_id", "")),
                "thumbnail_url": "",
                "primary_text": "",
                "headline": "",
                "description": "",
                "page_id": "",
                "utm_params": "",
                "avatar": str(r.get("avatar", "")),
                "visual_style": str(r.get("visual_style", "")),
                "script": str(r.get("script", "")),
                "filename": str(r.get("filename", "")),
                "status": str(r.get("status", "")).lower(),
            })
        
        return pd.DataFrame(rows)
        
    except Exception as e:
        notify(f"Error loading Supabase queue: {e}")
        return pd.DataFrame()

def set_supabase_status(ids_or_video_ids: List[str], new_status: str, 
                       use_column: str = "id", table: str = None) -> None:
    """Set Supabase status (legacy compatibility)."""
    try:
        from supabase import create_client
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key or not ids_or_video_ids:
            return
        
        client = create_client(supabase_url, supabase_key)
        table = table or os.getenv("SUPABASE_TABLE", "meta_creatives")
        
        if use_column == "video_id":
            client.table(table).update({"status": new_status}).in_("video_id", ids_or_video_ids).execute()
        else:
            client.table(table).update({"status": new_status}).in_("id", ids_or_video_ids).execute()
            
    except Exception as e:
        notify(f"Error updating Supabase status: {e}")

# =====================================================
# MAIN EXECUTION FUNCTIONS
# =====================================================

def run_ml_enhanced_dean(settings: Dict[str, Any], rules: Dict[str, Any], 
                        store: Store, background: bool = False) -> Dict[str, Any]:
    """Run the ML-enhanced Dean system."""
    try:
        # Initialize ML-enhanced Dean
        dean = MLEnhancedDean(settings, rules, store)
        
        if not dean.initialize():
            return {"error": "Failed to initialize ML system"}
        
        # Load queue
        if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            queue_df = load_queue_supabase()
        else:
            queue_df = pd.DataFrame()
        
        # Run ML-enhanced stages
        results = {}
        
        # Testing stage
        testing_result = dean.run_ml_enhanced_tick("testing", queue_df)
        results["testing"] = testing_result
        
        # Validation stage
        validation_result = dean.run_ml_enhanced_tick("validation")
        results["validation"] = validation_result
        
        # Scaling stage
        scaling_result = dean.run_ml_enhanced_tick("scaling")
        results["scaling"] = scaling_result
        
        # Generate ML-enhanced summary
        summary = generate_ml_enhanced_summary(results)
        
        # Start background scheduler if requested
        if background:
            from scheduler import start_background_scheduler
            start_background_scheduler(settings, rules, store)
        
        return {
            "status": "success",
            "results": results,
            "summary": summary,
            "ml_insights": {
                "total_ads_analyzed": sum(
                    len(result.get("ml_insights", {}).get("ad_analyses", {}))
                    for result in results.values()
                ),
                "avg_intelligence_score": np.mean([
                    result.get("intelligence_score", 0)
                    for result in results.values()
                ]),
                "ml_recommendations": [
                    rec for result in results.values()
                    for rec in result.get("ml_recommendations", [])
                ]
            }
        }
        
    except Exception as e:
        notify(f"Error running ML-enhanced Dean: {e}")
        return {"error": str(e)}

def generate_ml_enhanced_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ML-enhanced summary."""
    try:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "ml_insights": {},
            "recommendations": []
        }
        
        for stage, result in results.items():
            if isinstance(result, dict) and "error" not in result:
                summary["stages"][stage] = {
                    "status": "success",
                    "actions": result.get("kills", 0) + result.get("promotions", 0) + result.get("launched", 0),
                    "intelligence_score": result.get("intelligence_score", 0),
                    "ml_recommendations": result.get("ml_recommendations", [])
                }
                
                # Collect ML recommendations
                summary["recommendations"].extend(result.get("ml_recommendations", []))
            else:
                summary["stages"][stage] = {
                    "status": "error",
                    "error": result.get("error", "Unknown error")
                }
        
        # Calculate overall ML insights
        intelligence_scores = [
            stage_data.get("intelligence_score", 0)
            for stage_data in summary["stages"].values()
            if isinstance(stage_data, dict)
        ]
        
        summary["ml_insights"] = {
            "avg_intelligence_score": np.mean(intelligence_scores) if intelligence_scores else 0,
            "total_recommendations": len(summary["recommendations"]),
            "high_intelligence_stages": sum(1 for score in intelligence_scores if score > 0.7)
        }
        
        return summary
        
    except Exception as e:
        notify(f"Error generating ML-enhanced summary: {e}")
        return {"error": str(e)}

# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Dean ML-Enhanced Meta Ads Automation")
    parser.add_argument("--background", action="store_true", help="Run in background mode")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings file path")
    parser.add_argument("--rules", default="config/rules.yaml", help="Rules file path")
    parser.add_argument("--store", default="data/dean.db", help="Store database path")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Load configuration
    settings, rules = load_cfg(args.settings, args.rules)
    
    # Initialize store
    store = Store(args.store)
    
    try:
        # Run ML-enhanced Dean
        result = run_ml_enhanced_dean(settings, rules, store, args.background)
        
        if "error" in result:
            notify(f"❌ ML-Enhanced Dean failed: {result['error']}")
            sys.exit(1)
        else:
            notify(f"✅ ML-Enhanced Dean completed successfully")
            notify(f"📊 ML Insights: {result.get('ml_insights', {})}")
            
            if args.background:
                notify("🤖 Running in background mode - ML system will continue learning")
                # Keep running for background mode
                while True:
                    time.sleep(3600)  # Check every hour
                    
    except KeyboardInterrupt:
        notify("🛑 ML-Enhanced Dean stopped by user")
    except Exception as e:
        notify(f"❌ ML-Enhanced Dean error: {e}")
        sys.exit(1)
    finally:
        store.close()

if __name__ == "__main__":
    main()
