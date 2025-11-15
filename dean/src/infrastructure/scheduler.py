import os
import time
from datetime import timedelta
from typing import Any, Dict, Optional
import threading
import schedule

from .storage import Store
from integrations.slack import (
    notify, alert_error, alert_queue_empty,
    alert_system_health
)
from .utils import now_local
from stages.asc_plus import run_asc_plus_tick
from rules.rules import AdvancedRuleEngine as RuleEngine


class BackgroundScheduler:
    def __init__(self, settings: Dict[str, Any], rules: Dict[str, Any], store: Store):
        self.settings = settings
        self.rules = rules
        self.store = store
        self.running = False
        self.thread = None
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
            dry_run=False,
            tenant_id=settings.get("branding_name", "default"),
        )
        self.engine = RuleEngine(rules)
        scheduler_cfg = settings.get("scheduler", {})
        self.tick_interval_minutes = int(scheduler_cfg.get("asc_plus_tick_minutes", 60))
        self.stage_run_order = scheduler_cfg.get("run_order", ["asc_plus"])
        self.last_hourly_tick = None
        self.alert_cooldowns = {}
        
    def start(self):
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
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        notify("🛑 Background scheduler stopped")
        
    def _run_scheduler(self):
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
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(30)
            except Exception as e:
                notify(f"❌ Scheduler error: {e}")
                time.sleep(60)
                
    def _run_hourly_tick(self):
        try:
            current_time = now_local(self.settings.get("account", {}).get("timezone") or self.settings.get("account_timezone", "Europe/Amsterdam"))
            hour_key = current_time.strftime("%Y-%m-%d-%H")
            if self.last_hourly_tick == hour_key:
                return
            self.last_hourly_tick = hour_key
            notify(
                f"🔄 Automation tick ({self.tick_interval_minutes}m cadence) starting at {current_time.strftime('%H:%M')}"
            )
            if os.getenv("SUPABASE_URL") and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")):
                from main import load_queue_supabase
                table = os.getenv("SUPABASE_TABLE", "meta_creatives")
                queue_df = load_queue_supabase(table=table, status_filter="pending", limit=64)
            else:
                from main import load_queue
                queue_path = (self.settings.get("queue") or {}).get("path", "data/creatives_queue.csv")
                queue_df = load_queue(queue_path)
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
            self._check_critical_alerts()
            notify(
                f"✅ Automation tick complete - processed {len(stage_summaries)} stage(s)"
            )
        except Exception as e:
            alert_error(f"Hourly tick failed: {e}")
            
    def _run_stage_safely(self, stage_func, stage_name: str, *args, **kwargs):
        try:
            return stage_func(*args, **kwargs)
        except Exception as e:
            alert_error(f"{stage_name} stage failed: {e}")
            return None
    def _check_critical_alerts(self):
        try:
            self._check_queue_empty()
            self._check_system_health()
        except Exception as e:
            alert_error(f"Critical alerts check failed: {e}")
            
    def _check_queue_empty(self):
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
                cooldown_key = "queue_empty_alert"
                if self._check_alert_cooldown(cooldown_key, hours=6):
                    alert_queue_empty()
                    self._set_alert_cooldown(cooldown_key)
        except Exception as e:
            notify(f"⚠️ Failed to check queue status: {e}")
            
    def _check_system_health(self):
        try:
            try:
                self.client.get_ad_insights(level="ad", fields=["spend"], paginate=False)
            except Exception as e:
                if self._check_alert_cooldown("meta_api_error", hours=2):
                    alert_system_health(f"Meta API connection failed: {e}")
                    self._set_alert_cooldown("meta_api_error")
            try:
                self.store.incr("healthcheck", 1)
            except Exception as e:
                if self._check_alert_cooldown("db_error", hours=1):
                    alert_system_health(f"Database connection failed: {e}")
                    self._set_alert_cooldown("db_error")
        except Exception as e:
            notify(f"⚠️ Failed to check system health: {e}")
            
    def _check_alert_cooldown(self, alert_key: str, hours: int = 1) -> bool:
        if alert_key not in self.alert_cooldowns:
            return True
        last_alert = self.alert_cooldowns[alert_key]
        time_since = time.time() - last_alert
        return time_since >= (hours * 3600)
        
    def _set_alert_cooldown(self, alert_key: str):
        self.alert_cooldowns[alert_key] = time.time()


_scheduler: Optional[BackgroundScheduler] = None


def start_background_scheduler(settings: Dict[str, Any], rules: Dict[str, Any], store: Store):
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(settings, rules, store)
    _scheduler.start()


def stop_background_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None


def get_scheduler() -> Optional[BackgroundScheduler]:
    return _scheduler
