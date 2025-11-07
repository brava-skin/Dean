"""
Background scheduler for automated monitoring and alerts.
Handles hourly ticks, periodic summaries, and critical event alerts.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import threading
import schedule
from pathlib import Path

from .storage import Store
from integrations.slack import (
    notify, alert_kill, alert_promote, alert_scale, alert_fatigue, 
    alert_data_quality, alert_error, alert_queue_empty, alert_new_launch,
    alert_system_health, alert_budget_alert, post_digest, client as slack_client
)
# Note: MetaClient imported lazily to avoid circular imports
from .utils import now_local, getenv_f, getenv_i, getenv_b, cfg, cfg_or_env_f, cfg_or_env_i, cfg_or_env_b, cfg_or_env_list, safe_f, today_str, daily_key, ad_day_flag_key, now_minute_key, clean_text_token, prettify_ad_name
from stages.asc_plus import run_asc_plus_tick
from rules.rules import AdvancedRuleEngine as RuleEngine


class BackgroundScheduler:
    """
    Background scheduler that runs monitoring tasks at specified intervals.
    - Hourly ticks for testing/validation/scaling
    - 3-hour summaries of metrics and active ads
    - Daily morning summaries
    - Critical event alerts
    """
    
    def __init__(self, settings: Dict[str, Any], rules: Dict[str, Any], store: Store):
        self.settings = settings
        self.rules = rules
        self.store = store
        self.running = False
        self.thread = None
        
        # Initialize Meta client
        from integrations.meta_client import AccountAuth
        account = AccountAuth(
            account_id=os.getenv("FB_AD_ACCOUNT_ID", ""),
            access_token=os.getenv("FB_ACCESS_TOKEN", ""),
            app_id=os.getenv("FB_APP_ID", ""),
            app_secret=os.getenv("FB_APP_SECRET", ""),
            api_version=os.getenv("FB_API_VERSION") or None,
        )
        
        tz_name = (
            settings.get("account", {}).get("timezone")
            or settings.get("account_timezone")
            or settings.get("timezone")
            or os.getenv("TIMEZONE")
            or "Europe/Amsterdam"
        )
        
        from integrations.meta_client import ClientConfig, MetaClient
        cfg = ClientConfig(timezone=tz_name)
        self.client = MetaClient(
            accounts=[account],
            cfg=cfg,
            store=store,
            dry_run=False,  # Background runs should be live
            tenant_id=settings.get("branding_name", "default"),
        )
        
        self.engine = RuleEngine(rules)

        scheduler_cfg = settings.get("scheduler", {})
        self.tick_interval_minutes = int(scheduler_cfg.get("asc_plus_tick_minutes", 60))
        self.summary_interval_minutes = int(scheduler_cfg.get("summary_minutes", 180))
        self.daily_digest_time = scheduler_cfg.get("daily_digest_time_local", "08:00")
        self.stage_run_order = scheduler_cfg.get("run_order", ["asc_plus"])
        
        # Track last run times to avoid duplicate processing
        self.last_hourly_tick = None
        self.last_3h_summary = None
        self.last_daily_summary = None
        
        # Alert state tracking
        self.alert_cooldowns = {}
        
    def start(self):
        """Start the background scheduler."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        cadence_label = (
            f"every {self.tick_interval_minutes} minutes"
            if self.tick_interval_minutes < 60 or self.tick_interval_minutes % 60 != 0
            else f"every {self.tick_interval_minutes // 60} hour(s)"
        )
        notify(f"🤖 Background scheduler started - monitoring {cadence_label}")
        
    def stop(self):
        """Stop the background scheduler."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        notify("🛑 Background scheduler stopped")
        
    def _run_scheduler(self):
        """Main scheduler loop."""
        # Schedule tasks based on configuration
        if self.tick_interval_minutes <= 0:
            self.tick_interval_minutes = 60

        if self.tick_interval_minutes % 60 == 0:
            hours = max(1, self.tick_interval_minutes // 60)
            if hours == 1:
                schedule.every().hour.at(":00").do(self._run_hourly_tick)
            else:
                schedule.every(hours).hours.do(self._run_hourly_tick)
        else:
            schedule.every(self.tick_interval_minutes).minutes.do(self._run_hourly_tick)

        if self.summary_interval_minutes <= 0:
            self.summary_interval_minutes = 180

        if self.summary_interval_minutes % 60 == 0:
            summary_hours = max(1, self.summary_interval_minutes // 60)
            schedule.every(summary_hours).hours.do(self._run_3h_summary)
        else:
            schedule.every(self.summary_interval_minutes).minutes.do(self._run_3h_summary)

        summary_time = self.daily_digest_time if self.daily_digest_time else "08:00"
        schedule.every().day.at(summary_time).do(self._run_daily_summary)
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                notify(f"❌ Scheduler error: {e}")
                time.sleep(60)  # Wait a minute before retrying
                
    def _run_hourly_tick(self):
        """Run the main automation tick every hour."""
        try:
            current_time = now_local(self.settings.get("account", {}).get("timezone") or self.settings.get("account_timezone", "Europe/Amsterdam"))
            
            # Check if we already ran this hour
            hour_key = current_time.strftime("%Y-%m-%d-%H")
            if self.last_hourly_tick == hour_key:
                return
                
            self.last_hourly_tick = hour_key
            
            notify(
                f"🔄 Automation tick ({self.tick_interval_minutes}m cadence) starting at {current_time.strftime('%H:%M')}"
            )
            
            # Load queue
            if os.getenv("SUPABASE_URL") and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")):
                from main import load_queue_supabase, set_supabase_status
                table = os.getenv("SUPABASE_TABLE", "meta_creatives")
                queue_df = load_queue_supabase(table=table, status_filter="pending", limit=64)
            else:
                from main import load_queue
                queue_path = (self.settings.get("queue") or {}).get("path", "data/creatives_queue.csv")
                queue_df = load_queue(queue_path)
            
            # Run ASC+ stage
            stage_summaries = []
            
            for stage_name in self.stage_run_order:
                stage_key = stage_name.lower()
                if stage_key == "asc_plus":
                    asc_plus_result = self._run_stage_safely(
                        run_asc_plus_tick,
                        "ASC+",
                        self.client,
                        self.settings,
                        self.store,
                    )
                    if asc_plus_result:
                        stage_summaries.append({"stage": "ASC+", "counts": asc_plus_result})
                else:
                    notify(f"⚠️ Unknown stage '{stage_name}' configured for scheduler run order")
            
            # Check for critical alerts
            self._check_critical_alerts()
            
            notify(
                f"✅ Automation tick complete - processed {len(stage_summaries)} stage(s)"
            )
            
        except Exception as e:
            alert_error(f"Hourly tick failed: {e}")
            
    def _run_stage_safely(self, stage_func, stage_name: str, *args, **kwargs):
        """Run a stage function with error handling."""
        try:
            return stage_func(*args, **kwargs)
        except Exception as e:
            alert_error(f"{stage_name} stage failed: {e}")
            return None
            
    def _run_3h_summary(self):
        """Run 3-hour summary of metrics and active ads."""
        try:
            current_time = now_local(self.settings.get("account", {}).get("timezone") or self.settings.get("account_timezone", "Europe/Amsterdam"))
            
            # Check if we already ran this 3-hour window
            hour_window = current_time.hour // 3
            window_key = f"{current_time.strftime('%Y-%m-%d')}-{hour_window}"
            if self.last_3h_summary == window_key:
                return
                
            self.last_3h_summary = window_key
            
            notify("📊 3-hour summary starting")
            
            # Get account metrics
            account_metrics = self._get_account_metrics()
            
            # Get active ads summary
            active_ads = self._get_active_ads_summary()
            
            # Format summary message
            summary_lines = [
                f"📈 Account: spend €{account_metrics.get('spend', 0):.2f}, purchases {account_metrics.get('purchases', 0)}",
                f"🎯 CPA: {account_metrics.get('cpa', 'N/A')}, ROAS: {account_metrics.get('roas', 'N/A')}",
                f"📱 Active ads: {active_ads.get('total', 0)} total, {active_ads.get('testing', 0)} testing, {active_ads.get('validation', 0)} validation, {active_ads.get('scaling', 0)} scaling"
            ]
            
            if active_ads.get('top_performers'):
                summary_lines.append("🏆 Top performers:")
                for ad in active_ads['top_performers'][:3]:
                    summary_lines.append(f"  • {ad['name']}: €{ad['spend']:.2f} spend, {ad['purchases']} purchases")
            
            # Send summary
            notify("\n".join(summary_lines))
            
        except Exception as e:
            alert_error(f"3-hour summary failed: {e}")
            
    def _run_daily_summary(self):
        """Run daily morning summary of previous day."""
        try:
            current_time = now_local(self.settings.get("account", {}).get("timezone") or self.settings.get("account_timezone", "Europe/Amsterdam"))
            
            # Check if we already ran today
            today_key = current_time.strftime("%Y-%m-%d")
            if self.last_daily_summary == today_key:
                return
                
            self.last_daily_summary = today_key
            
            notify("🌅 Daily summary starting")
            
            # Get yesterday's metrics
            yesterday = current_time - timedelta(days=1)
            daily_metrics = self._get_daily_metrics(yesterday)
            
            # Get stage statistics
            stage_stats = self._get_daily_stage_stats(yesterday)
            
            # Format daily summary
            summary_lines = [
                f"📅 Yesterday's Performance ({yesterday.strftime('%Y-%m-%d')})",
                f"💰 Total spend: €{daily_metrics.get('spend', 0):.2f}",
                f"🛒 Purchases: {daily_metrics.get('purchases', 0)}",
                f"📊 CPA: {daily_metrics.get('cpa', 'N/A')}",
                f"📈 ROAS: {daily_metrics.get('roas', 'N/A')}",
                "",
                "📋 Stage Activity:"
            ]
            
            for stage, stats in stage_stats.items():
                summary_lines.append(f"  {stage}: {stats}")
            
            # Send daily summary
            notify("\n".join(summary_lines))
            
        except Exception as e:
            alert_error(f"Daily summary failed: {e}")
            
    def _get_account_metrics(self) -> Dict[str, Any]:
        """Get current account metrics."""
        try:
            rows = self.client.get_ad_insights(
                level="ad",
                fields=["spend", "actions", "action_values"],
                time_range={
                    "since": (now_local(self.settings.get("account", {}).get("timezone") or self.settings.get("account_timezone", "Europe/Amsterdam")) - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "until": now_local(self.settings.get("account", {}).get("timezone") or self.settings.get("account_timezone", "Europe/Amsterdam")).strftime("%Y-%m-%d")
                },
                paginate=True
            )
            
            total_spend = sum(float(r.get("spend", 0)) for r in rows)
            total_purchases = 0
            total_revenue = 0
            
            for row in rows:
                for action in (row.get("actions") or []):
                    if action.get("action_type") == "purchase":
                        total_purchases += int(action.get("value", 0))
                        
                for value in (row.get("action_values") or []):
                    if value.get("action_type") == "purchase":
                        total_revenue += float(value.get("value", 0))
            
            cpa = total_spend / total_purchases if total_purchases > 0 else None
            roas = total_revenue / total_spend if total_spend > 0 else None
            
            return {
                "spend": total_spend,
                "purchases": total_purchases,
                "revenue": total_revenue,
                "cpa": cpa,
                "roas": roas
            }
        except Exception as e:
            notify(f"⚠️ Failed to get account metrics: {e}")
            return {}
            
    def _get_active_ads_summary(self) -> Dict[str, Any]:
        """Get summary of active ads by stage."""
        try:
            rows = self.client.get_ad_insights(
                level="ad",
                fields=["ad_id", "ad_name", "spend", "actions"],
                time_range={
                    "since": (now_local(self.settings.get("account", {}).get("timezone") or self.settings.get("account_timezone", "Europe/Amsterdam")) - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "until": now_local(self.settings.get("account", {}).get("timezone") or self.settings.get("account_timezone", "Europe/Amsterdam")).strftime("%Y-%m-%d")
                },
                paginate=True
            )
            
            testing_ads = []
            validation_ads = []
            scaling_ads = []
            
            for row in rows:
                ad_name = row.get("ad_name", "")
                spend = float(row.get("spend", 0))
                purchases = 0
                
                for action in (row.get("actions") or []):
                    if action.get("action_type") == "purchase":
                        purchases += int(action.get("value", 0))
                
                ad_info = {
                    "name": ad_name,
                    "spend": spend,
                    "purchases": purchases
                }
                
                if "[TEST]" in ad_name:
                    testing_ads.append(ad_info)
                elif "[VALID]" in ad_name:
                    validation_ads.append(ad_info)
                elif "[SCALE]" in ad_name:
                    scaling_ads.append(ad_info)
            
            # Sort by spend descending
            testing_ads.sort(key=lambda x: x["spend"], reverse=True)
            validation_ads.sort(key=lambda x: x["spend"], reverse=True)
            scaling_ads.sort(key=lambda x: x["spend"], reverse=True)
            
            return {
                "total": len(rows),
                "testing": len(testing_ads),
                "validation": len(validation_ads),
                "scaling": len(scaling_ads),
                "top_performers": (testing_ads + validation_ads + scaling_ads)[:5]
            }
        except Exception as e:
            notify(f"⚠️ Failed to get active ads summary: {e}")
            return {"total": 0, "testing": 0, "validation": 0, "scaling": 0, "top_performers": []}
            
    def _get_daily_metrics(self, date: datetime) -> Dict[str, Any]:
        """Get metrics for a specific date."""
        try:
            date_str = date.strftime("%Y-%m-%d")
            rows = self.client.get_ad_insights(
                level="ad",
                fields=["spend", "actions", "action_values"],
                time_range={"since": date_str, "until": date_str},
                paginate=True
            )
            
            total_spend = sum(float(r.get("spend", 0)) for r in rows)
            total_purchases = 0
            total_revenue = 0
            
            for row in rows:
                for action in (row.get("actions") or []):
                    if action.get("action_type") == "purchase":
                        total_purchases += int(action.get("value", 0))
                        
                for value in (row.get("action_values") or []):
                    if value.get("action_type") == "purchase":
                        total_revenue += float(value.get("value", 0))
            
            cpa = total_spend / total_purchases if total_purchases > 0 else None
            roas = total_revenue / total_spend if total_spend > 0 else None
            
            return {
                "spend": total_spend,
                "purchases": total_purchases,
                "revenue": total_revenue,
                "cpa": cpa,
                "roas": roas
            }
        except Exception as e:
            notify(f"⚠️ Failed to get daily metrics for {date.strftime('%Y-%m-%d')}: {e}")
            return {}
            
    def _get_daily_stage_stats(self, date: datetime) -> Dict[str, str]:
        """Get daily statistics by stage."""
        try:
            date_str = date.strftime("%Y-%m-%d")
            rows = self.client.get_ad_insights(
                level="ad",
                fields=["ad_name", "spend", "actions"],
                time_range={"since": date_str, "until": date_str},
                paginate=True
            )
            
            testing_stats = {"ads": 0, "spend": 0, "purchases": 0}
            validation_stats = {"ads": 0, "spend": 0, "purchases": 0}
            scaling_stats = {"ads": 0, "spend": 0, "purchases": 0}
            
            for row in rows:
                ad_name = row.get("ad_name", "")
                spend = float(row.get("spend", 0))
                purchases = 0
                
                for action in (row.get("actions") or []):
                    if action.get("action_type") == "purchase":
                        purchases += int(action.get("value", 0))
                
                if "[TEST]" in ad_name:
                    testing_stats["ads"] += 1
                    testing_stats["spend"] += spend
                    testing_stats["purchases"] += purchases
                elif "[VALID]" in ad_name:
                    validation_stats["ads"] += 1
                    validation_stats["spend"] += spend
                    validation_stats["purchases"] += purchases
                elif "[SCALE]" in ad_name:
                    scaling_stats["ads"] += 1
                    scaling_stats["spend"] += spend
                    scaling_stats["purchases"] += purchases
            
            return {
                "Testing": f"{testing_stats['ads']} ads, €{testing_stats['spend']:.2f} spend, {testing_stats['purchases']} purchases",
                "Validation": f"{validation_stats['ads']} ads, €{validation_stats['spend']:.2f} spend, {validation_stats['purchases']} purchases",
                "Scaling": f"{scaling_stats['ads']} ads, €{scaling_stats['spend']:.2f} spend, {scaling_stats['purchases']} purchases"
            }
        except Exception as e:
            notify(f"⚠️ Failed to get daily stage stats: {e}")
            return {}
            
    def _check_critical_alerts(self):
        """Check for critical events that need immediate alerts."""
        try:
            # Check for empty queue
            self._check_queue_empty()
            
            # Check for system health
            self._check_system_health()
            
        except Exception as e:
            alert_error(f"Critical alerts check failed: {e}")
            
    def _check_queue_empty(self):
        """Alert if creative queue is empty."""
        try:
            if os.getenv("SUPABASE_URL"):
                from main import load_queue_supabase
                table = os.getenv("SUPABASE_TABLE", "meta_creatives")
                queue_df = load_queue_supabase(table=table, status_filter="pending", limit=64)
            else:
                from main import load_queue
                queue_path = (self.settings.get("queue") or {}).get("path", "data/creatives_queue.csv")
                queue_df = load_queue(queue_path)
            
            if len(queue_df) == 0:
                # Check cooldown to avoid spam
                cooldown_key = "queue_empty_alert"
                if self._check_alert_cooldown(cooldown_key, hours=6):
                    alert_queue_empty()
                    self._set_alert_cooldown(cooldown_key)
                    
        except Exception as e:
            notify(f"⚠️ Failed to check queue status: {e}")
            
    def _check_system_health(self):
        """Check system health and alert on issues."""
        try:
            # Check if we can connect to Meta API
            try:
                self.client.get_ad_insights(level="ad", fields=["spend"], paginate=False)
            except Exception as e:
                if self._check_alert_cooldown("meta_api_error", hours=2):
                    alert_system_health(f"Meta API connection failed: {e}")
                    self._set_alert_cooldown("meta_api_error")
                    
            # Check database connectivity
            try:
                self.store.incr("healthcheck", 1)
            except Exception as e:
                if self._check_alert_cooldown("db_error", hours=1):
                    alert_system_health(f"Database connection failed: {e}")
                    self._set_alert_cooldown("db_error")
                    
        except Exception as e:
            notify(f"⚠️ Failed to check system health: {e}")
            
    def _check_alert_cooldown(self, alert_key: str, hours: int = 1) -> bool:
        """Check if enough time has passed since last alert."""
        if alert_key not in self.alert_cooldowns:
            return True
            
        last_alert = self.alert_cooldowns[alert_key]
        time_since = time.time() - last_alert
        return time_since >= (hours * 3600)
        
    def _set_alert_cooldown(self, alert_key: str):
        """Set the cooldown timestamp for an alert."""
        self.alert_cooldowns[alert_key] = time.time()


# Global scheduler instance
_scheduler: Optional[BackgroundScheduler] = None


def start_background_scheduler(settings: Dict[str, Any], rules: Dict[str, Any], store: Store):
    """Start the background scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(settings, rules, store)
    _scheduler.start()


def stop_background_scheduler():
    """Stop the background scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None


def get_scheduler() -> Optional[BackgroundScheduler]:
    """Get the current scheduler instance."""
    return _scheduler
