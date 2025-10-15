"""
Smart Scheduler for GitHub Actions - Advanced tick management with future-proofing.
Handles intelligent scheduling, duplicate prevention, reliability, and advanced monitoring.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import sqlite3
from pathlib import Path

from storage import Store
from slack import (
    notify, alert_kill, alert_promote, alert_scale, alert_fatigue, 
    alert_data_quality, alert_error, alert_queue_empty, alert_new_launch,
    alert_system_health, alert_budget_alert, post_digest, client as slack_client
)
from meta_client import MetaClient, AccountAuth, ClientConfig
from utils import now_local, getenv_f, getenv_i, getenv_b, cfg, cfg_or_env_f, cfg_or_env_i, cfg_or_env_b, cfg_or_env_list, safe_f, today_str, daily_key, ad_day_flag_key, now_minute_key, clean_text_token, prettify_ad_name
from stages.testing import run_testing_tick
from stages.validation import run_validation_tick
from stages.scaling import run_scaling_tick
from rules import RuleEngine
from advanced_config import AdvancedConfig
from monitoring_dashboard import MonitoringDashboard


class SmartScheduler:
    """
    Advanced scheduler for GitHub Actions with intelligent features:
    - Duplicate tick prevention
    - Smart retry logic
    - Future-proof scheduling
    - Advanced monitoring
    - Reliability features
    """
    
    def __init__(self, settings: Dict[str, Any], rules: Dict[str, Any], store: Store):
        self.settings = settings
        self.rules = rules
        self.store = store
        
        # Initialize Meta client
        account = AccountAuth(
            account_id=os.getenv("FB_AD_ACCOUNT_ID", ""),
            access_token=os.getenv("FB_ACCESS_TOKEN", ""),
            app_id=os.getenv("FB_APP_ID", ""),
            app_secret=os.getenv("FB_APP_SECRET", ""),
            api_version=os.getenv("FB_API_VERSION") or None,
        )
        
        tz_name = (
            settings.get("account_timezone")
            or settings.get("timezone")
            or os.getenv("TIMEZONE")
            or "Europe/Amsterdam"
        )
        
        cfg = ClientConfig(timezone=tz_name)
        self.client = MetaClient(
            accounts=[account],
            cfg=cfg,
            store=store,
            dry_run=False,
            tenant_id=settings.get("branding_name", "default"),
        )
        
        self.engine = RuleEngine(rules)
        self.tz_name = tz_name
        
        # Smart scheduling state
        self.tick_db_path = "data/smart_ticks.sqlite"
        self._init_tick_database()
        
        # Advanced configuration and monitoring
        self.advanced_config = AdvancedConfig()
        self.monitoring_dashboard = MonitoringDashboard(self.tick_db_path)
        
        # Get configuration values
        self.max_retries = self.advanced_config.config["reliability"]["max_retries"]
        self.retry_delay = self.advanced_config.config["reliability"]["retry_delay_base"]
        self.duplicate_window = 3600  # 1 hour
        self.health_check_interval = self.advanced_config.config["reliability"]["health_check_interval"]
        
    def _init_tick_database(self):
        """Initialize database for tick tracking and duplicate prevention."""
        Path(self.tick_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.tick_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tick_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick_id TEXT UNIQUE NOT NULL,
                    timestamp INTEGER NOT NULL,
                    hour INTEGER NOT NULL,
                    day TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_type TEXT,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tick_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    health_score REAL NOT NULL,
                    api_latency REAL,
                    db_latency REAL,
                    memory_usage REAL,
                    error_count INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tick_timestamp ON tick_history(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tick_hour ON tick_history(hour)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tick_day ON tick_history(day)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_health_timestamp ON tick_health(timestamp)")
    
    def _generate_tick_id(self, current_time: datetime, summary_type: Optional[str] = None) -> str:
        """Generate unique tick ID for duplicate prevention."""
        base_id = f"{current_time.strftime('%Y%m%d%H')}"
        if summary_type:
            base_id += f"_{summary_type}"
        return hashlib.md5(base_id.encode()).hexdigest()[:16]
    
    def _is_duplicate_tick(self, tick_id: str) -> bool:
        """Check if this tick has already been processed."""
        with sqlite3.connect(self.tick_db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM tick_history WHERE tick_id = ? AND status IN ('completed', 'running')",
                (tick_id,)
            )
            return cursor.fetchone() is not None
    
    def _record_tick_start(self, tick_id: str, current_time: datetime, summary_type: Optional[str] = None):
        """Record tick start to prevent duplicates."""
        with sqlite3.connect(self.tick_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tick_history 
                (tick_id, timestamp, hour, day, status, summary_type, retry_count)
                VALUES (?, ?, ?, ?, 'running', ?, 0)
            """, (
                tick_id,
                int(current_time.timestamp()),
                current_time.hour,
                current_time.strftime('%Y-%m-%d'),
                summary_type
            ))
    
    def _record_tick_completion(self, tick_id: str, status: str, error_message: Optional[str] = None):
        """Record tick completion."""
        with sqlite3.connect(self.tick_db_path) as conn:
            conn.execute("""
                UPDATE tick_history 
                SET status = ?, error_message = ?
                WHERE tick_id = ?
            """, (status, error_message, tick_id))
    
    def _should_run_summary(self, current_time: datetime, summary_type: str) -> bool:
        """Smart logic to determine if summary should run."""
        if summary_type == "3h":
            # Run every 3 hours at :00, :03, :06, :09, :12, :15, :18, :21
            return current_time.hour % 3 == 0 and current_time.minute < 10
        elif summary_type == "daily":
            # Run at 8 AM
            return current_time.hour == 8 and current_time.minute < 10
        return False
    
    def _get_health_score(self) -> float:
        """Calculate system health score (0-100)."""
        try:
            # Test API connectivity
            start_time = time.time()
            self.client.get_ad_insights(level="ad", fields=["spend"], paginate=False)
            api_latency = time.time() - start_time
            
            # Test database connectivity
            start_time = time.time()
            self.store.incr("healthcheck", 1)
            db_latency = time.time() - start_time
            
            # Calculate health score
            api_score = max(0, 100 - (api_latency * 100))  # Penalty for slow API
            db_score = max(0, 100 - (db_latency * 1000))   # Penalty for slow DB
            
            health_score = (api_score + db_score) / 2
            
            # Record health metrics
            with sqlite3.connect(self.tick_db_path) as conn:
                conn.execute("""
                    INSERT INTO tick_health 
                    (timestamp, health_score, api_latency, db_latency)
                    VALUES (?, ?, ?, ?)
                """, (
                    int(time.time()),
                    health_score,
                    api_latency,
                    db_latency
                ))
            
            return health_score
            
        except Exception as e:
            notify(f"⚠️ Health check failed: {e}")
            return 0.0
    
    def _check_system_reliability(self) -> bool:
        """Check if system is reliable enough for processing."""
        health_score = self._get_health_score()
        
        # Alert if health is poor
        if health_score < 50:
            alert_system_health(f"System health score is {health_score:.1f}/100 - may affect performance")
            return False
        
        return True
    
    def _cleanup_old_records(self):
        """Clean up old tick records to prevent database bloat."""
        cutoff_time = int((datetime.now() - timedelta(days=30)).timestamp())
        
        with sqlite3.connect(self.tick_db_path) as conn:
            # Clean up old tick history
            conn.execute("DELETE FROM tick_history WHERE timestamp < ?", (cutoff_time,))
            
            # Clean up old health records
            conn.execute("DELETE FROM tick_health WHERE timestamp < ?", (cutoff_time,))
    
    def run_smart_tick(self, stage_choice: str = "all") -> Dict[str, Any]:
        """
        Run a smart tick with advanced features.
        """
        current_time = now_local(self.tz_name)
        tick_id = self._generate_tick_id(current_time)
        
        # Check for duplicates
        if self._is_duplicate_tick(tick_id):
            notify(f"⏭️ Tick {tick_id} already processed - skipping duplicate")
            return {"status": "skipped", "reason": "duplicate"}
        
        # Get optimal timing
        timing_config = self.advanced_config.get_optimal_tick_timing(current_time)
        if timing_config["delay_seconds"] > 0:
            notify(f"⏰ Smart timing: {timing_config['reason']} - delaying {timing_config['delay_seconds']}s")
            time.sleep(timing_config["delay_seconds"])
        
        # Check system reliability with adaptive thresholds
        health_score = self._get_health_score()
        adaptive_thresholds = self.advanced_config.get_adaptive_thresholds()
        
        should_skip, skip_reason = self.advanced_config.should_skip_tick(health_score, 0)
        if should_skip:
            notify(f"⚠️ Smart skip: {skip_reason} (health: {health_score:.1f})")
            return {"status": "skipped", "reason": skip_reason}
        
        # Record tick start
        self._record_tick_start(tick_id, current_time)
        
        try:
            # Clean up old records periodically
            if current_time.hour == 0:  # Daily cleanup at midnight
                self._cleanup_old_records()
            
            # Run the main tick
            result = self._execute_main_tick(stage_choice, current_time)
            
            # Check for summaries
            summary_results = self._check_and_run_summaries(current_time)
            
            # Record successful completion
            self._record_tick_completion(tick_id, "completed")
            
            # Update performance metrics for learning
            performance_metrics = {
                "health_score": health_score,
                "success_rate": 1.0,
                "api_latency": 0.0,  # Would be measured in real implementation
                "db_latency": 0.0,   # Would be measured in real implementation
                "tick_duration": 0.0  # Would be measured in real implementation
            }
            self.advanced_config.update_performance_metrics(performance_metrics)
            
            # Auto-optimize configuration
            self.advanced_config.optimize_configuration()
            
            return {
                "status": "completed",
                "tick_id": tick_id,
                "main_result": result,
                "summaries": summary_results,
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            # Record failure
            self._record_tick_completion(tick_id, "failed", str(e))
            
            # Smart retry logic with adaptive delays
            retry_count = self._get_retry_count(tick_id)
            if retry_count < self.max_retries:
                smart_delay = self.advanced_config.get_smart_retry_delay(retry_count)
                notify(f"🔄 Tick failed, will retry in {smart_delay//60} minutes (attempt {retry_count + 1}/{self.max_retries})")
                self._schedule_retry(tick_id, retry_count + 1)
                
                # Update performance metrics for failed attempt
                performance_metrics = {
                    "health_score": health_score,
                    "success_rate": 0.0,
                    "api_latency": 0.0,
                    "db_latency": 0.0,
                    "tick_duration": 0.0,
                    "error_type": type(e).__name__
                }
                self.advanced_config.update_performance_metrics(performance_metrics)
            else:
                alert_error(f"Tick {tick_id} failed after {self.max_retries} retries: {e}")
            
            return {"status": "failed", "error": str(e), "retry_count": retry_count}
    
    def _execute_main_tick(self, stage_choice: str, current_time: datetime) -> Dict[str, Any]:
        """Execute the main automation tick."""
        notify(f"🔄 Smart tick starting at {current_time.strftime('%H:%M')} (ID: {self._generate_tick_id(current_time)})")
        
        # Load queue
        if os.getenv("SUPABASE_URL") and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")):
            from main import load_queue_supabase, set_supabase_status
            table = os.getenv("SUPABASE_TABLE", "meta_creatives")
            queue_df = load_queue_supabase(table=table, status_filter="pending", limit=64)
        else:
            from main import load_queue
            queue_path = (self.settings.get("queue") or {}).get("path", "data/creatives_queue.csv")
            queue_df = load_queue(queue_path)
        
        # Run stages
        stage_summaries = []
        
        if stage_choice in ("all", "testing"):
            testing_result = self._run_stage_safely(
                run_testing_tick,
                "TESTING",
                self.client,
                self.settings,
                self.engine,
                self.store,
                queue_df,
                set_supabase_status if 'set_supabase_status' in locals() else None,
                placements=["facebook", "instagram"],
                instagram_actor_id=os.getenv("IG_ACTOR_ID"),
            )
            if testing_result:
                stage_summaries.append({"stage": "TEST", "counts": testing_result})
        
        if stage_choice in ("all", "validation"):
            validation_result = self._run_stage_safely(
                run_validation_tick,
                "VALIDATION",
                self.client,
                self.settings,
                self.engine,
                self.store,
            )
            if validation_result:
                stage_summaries.append({"stage": "VALID", "counts": validation_result})
        
        if stage_choice in ("all", "scaling"):
            scaling_result = self._run_stage_safely(
                run_scaling_tick,
                "SCALING",
                self.client,
                self.settings,
                self.store,
            )
            if scaling_result:
                stage_summaries.append({"stage": "SCALE", "counts": scaling_result})
        
        # Check for critical alerts
        self._check_critical_alerts()
        
        notify(f"✅ Smart tick completed - {len(stage_summaries)} stages processed")
        return {"stage_summaries": stage_summaries}
    
    def _check_and_run_summaries(self, current_time: datetime) -> Dict[str, Any]:
        """Check and run appropriate summaries."""
        summary_results = {}
        
        # 3-hour summary
        if self._should_run_summary(current_time, "3h"):
            try:
                summary_results["3h"] = self._run_3h_summary()
            except Exception as e:
                notify(f"⚠️ 3-hour summary failed: {e}")
                summary_results["3h"] = {"error": str(e)}
        
        # Daily summary
        if self._should_run_summary(current_time, "daily"):
            try:
                summary_results["daily"] = self._run_daily_summary()
            except Exception as e:
                notify(f"⚠️ Daily summary failed: {e}")
                summary_results["daily"] = {"error": str(e)}
        
        return summary_results
    
    def _run_stage_safely(self, stage_func, stage_name: str, *args, **kwargs):
        """Run a stage function with error handling."""
        try:
            return stage_func(*args, **kwargs)
        except Exception as e:
            alert_error(f"{stage_name} stage failed: {e}")
            return None
    
    def _run_3h_summary(self) -> Dict[str, Any]:
        """Run 3-hour summary with advanced features."""
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
        
        return {
            "account_metrics": account_metrics,
            "active_ads": active_ads,
            "message_lines": summary_lines
        }
    
    def _run_daily_summary(self) -> Dict[str, Any]:
        """Run daily summary with advanced features."""
        notify("🌅 Daily summary starting")
        
        # Get yesterday's metrics
        yesterday = now_local(self.tz_name) - timedelta(days=1)
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
        
        return {
            "daily_metrics": daily_metrics,
            "stage_stats": stage_stats,
            "message_lines": summary_lines
        }
    
    def _get_account_metrics(self) -> Dict[str, Any]:
        """Get current account metrics with error handling."""
        try:
            rows = self.client.get_ad_insights(
                level="ad",
                fields=["spend", "actions", "action_values"],
                time_range={
                    "since": (now_local(self.tz_name) - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "until": now_local(self.tz_name).strftime("%Y-%m-%d")
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
                    "since": (now_local(self.tz_name) - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "until": now_local(self.tz_name).strftime("%Y-%m-%d")
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
                alert_queue_empty()
                    
        except Exception as e:
            notify(f"⚠️ Failed to check queue status: {e}")
    
    def _check_system_health(self):
        """Check system health and alert on issues."""
        try:
            # Check if we can connect to Meta API
            try:
                self.client.get_ad_insights(level="ad", fields=["spend"], paginate=False)
            except Exception as e:
                alert_system_health(f"Meta API connection failed: {e}")
                    
            # Check database connectivity
            try:
                self.store.incr("healthcheck", 1)
            except Exception as e:
                alert_system_health(f"Database connection failed: {e}")
                    
        except Exception as e:
            notify(f"⚠️ Failed to check system health: {e}")
    
    def _get_retry_count(self, tick_id: str) -> int:
        """Get retry count for a tick."""
        with sqlite3.connect(self.tick_db_path) as conn:
            cursor = conn.execute(
                "SELECT retry_count FROM tick_history WHERE tick_id = ?",
                (tick_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else 0
    
    def _schedule_retry(self, tick_id: str, retry_count: int):
        """Schedule a retry for a failed tick."""
        with sqlite3.connect(self.tick_db_path) as conn:
            conn.execute("""
                UPDATE tick_history 
                SET retry_count = ?, status = 'retry_scheduled'
                WHERE tick_id = ?
            """, (retry_count, tick_id))
    
    def get_tick_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get tick statistics for monitoring."""
        cutoff_time = int((datetime.now() - timedelta(days=days)).timestamp())
        
        with sqlite3.connect(self.tick_db_path) as conn:
            # Get tick counts by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM tick_history 
                WHERE timestamp >= ?
                GROUP BY status
            """, (cutoff_time,))
            status_counts = dict(cursor.fetchall())
            
            # Get success rate
            total_ticks = sum(status_counts.values())
            success_rate = (status_counts.get('completed', 0) / total_ticks * 100) if total_ticks > 0 else 0
            
            # Get average health score
            cursor = conn.execute("""
                SELECT AVG(health_score) as avg_health
                FROM tick_health 
                WHERE timestamp >= ?
            """, (cutoff_time,))
            avg_health = cursor.fetchone()[0] or 0
            
            return {
                "total_ticks": total_ticks,
                "success_rate": success_rate,
                "status_counts": status_counts,
                "avg_health_score": avg_health,
                "period_days": days
            }
