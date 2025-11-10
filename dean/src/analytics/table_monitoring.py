"""
DEAN TABLE MONITORING SYSTEM
Comprehensive monitoring of Supabase table health and data collection

This module provides:
- Real-time table row count tracking
- Comparison with previous ticks
- Data collection health monitoring
- ML system data validation
- Automated alerts for empty or problematic tables
"""

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from supabase import Client

from integrations.slack import notify


@dataclass
class TableHealth:
    """Health status of a Supabase table."""
    table_name: str
    current_rows: int
    previous_rows: int
    new_rows: int
    growth_rate: float
    is_healthy: bool
    last_updated: datetime
    issues: List[str] = field(default_factory=list)
    last_insert_at: Optional[datetime] = None
    recent_activity: bool = False


@dataclass
class TableInsights:
    """Comprehensive insights about all tables."""
    timestamp: datetime
    total_tables: int
    healthy_tables: int
    problematic_tables: int
    total_rows: int
    new_rows_since_last_tick: int
    tables: Dict[str, TableHealth]
    ml_readiness: Dict[str, bool]
    recommendations: List[str]


class TableMonitor:
    """Comprehensive table monitoring system."""
    
    def __init__(self, supabase_client: Optional[Client] = None):
        self.supabase_client = supabase_client
        self.previous_counts: Dict[str, int] = {}
        self.monitoring_enabled = supabase_client is not None
        self.state_path = Path(".cache/table_monitor_state.json")
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # In readonly environments we simply skip persistence
            pass
        self.last_snapshot_at: Optional[datetime] = None
        self._load_previous_counts()
        
        # Critical tables for ML system (based on actual usage in codebase)
        self.critical_tables = [
            'performance_metrics',
            'ad_lifecycle', 
            'time_series_data',
            'ml_predictions',
            'learning_events',
            'creative_intelligence',
            'historical_data',
            'ad_creation_times',
            'ml_models'
        ]
        
        # ML training requirements
        self.ml_table_requirements = {
            'performance_metrics': {'min_rows': 5, 'description': 'Ad performance metrics'},
            'ad_lifecycle': {'min_rows': 5, 'description': 'Ad lifecycle tracking'},
            'time_series_data': {'min_rows': 5, 'description': 'Time-series performance data'},
            'creative_intelligence': {'min_rows': 5, 'description': 'Creative assets and performance'},
            'historical_data': {'min_rows': 10, 'description': 'Historical metric tracking'},
            'ad_creation_times': {'min_rows': 5, 'description': 'Ad creation timestamps'},
            'ml_models': {'min_rows': 1, 'description': 'ML model storage'},
            'ml_predictions': {'min_rows': 1, 'description': 'ML predictions storage'},
            'learning_events': {'min_rows': 1, 'description': 'ML learning events'}
        }

        # Timestamp fields used to measure recent activity
        self.table_timestamp_fields: Dict[str, str] = {
            'ml_models': 'trained_at',
            'ml_predictions': 'created_at',
        }
        self.table_activity_fields: Dict[str, List[str]] = {
            'performance_metrics': ['updated_at', 'created_at'],
            'ad_lifecycle': ['updated_at', 'created_at'],
            'time_series_data': ['created_at', 'updated_at'],
            'ml_predictions': ['created_at', 'updated_at'],
            'learning_events': ['created_at', 'updated_at'],
            'creative_intelligence': ['created_at', 'updated_at'],
            'historical_data': ['created_at', 'updated_at'],
            'ad_creation_times': ['created_at', 'updated_at'],
            'ml_models': ['created_at', 'updated_at', 'trained_at'],
        }
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get current row count for a table."""
        if not self.supabase_client:
            return 0
        
        try:
            response = self.supabase_client.table(table_name).select('id', count='exact').execute()
            count = response.count
            # Ensure we return a valid integer
            if isinstance(count, (int, float)) and count >= 0:
                return int(count)
            else:
                return 0
        except Exception as e:
            # Don't print error directly as it corrupts the report formatting
            return 0

    def _get_last_insert_time(self, table_name: str) -> Optional[datetime]:
        """Fetch the most recent insert timestamp for a table."""
        if not self.supabase_client:
            return None

        timestamp_field = self.table_timestamp_fields.get(table_name)
        if not timestamp_field:
            return None

        try:
            response = (
                self.supabase_client.table(table_name)
                .select(timestamp_field)
                .order(timestamp_field, desc=True)
                .limit(1)
                .execute()
            )
            if not response or not getattr(response, 'data', None):
                return None

            raw_value = response.data[0].get(timestamp_field)
            if not raw_value:
                return None

            if isinstance(raw_value, datetime):
                ts = raw_value
            elif isinstance(raw_value, str):
                ts = datetime.fromisoformat(raw_value.replace('Z', '+00:00'))
            else:
                return None

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
        except Exception:
            return None

    def _count_recent_rows(self, table_name: str, since: Optional[datetime]) -> int:
        """Count rows touched since the last snapshot (created or updated)."""
        if not self.supabase_client or not since:
            return 0
        activity_fields = self.table_activity_fields.get(table_name, [])
        for timestamp_field in activity_fields:
            try:
                response = (
                    self.supabase_client.table(table_name)
                    .select('id', count='exact')
                    .gt(timestamp_field, since.isoformat())
                    .execute()
                )
                recent = int(getattr(response, 'count', 0) or 0)
                if recent > 0:
                    return recent
            except Exception:
                continue
        return 0
    
    def analyze_table_health(self, table_name: str) -> TableHealth:
        """Analyze health of a specific table."""
        current_rows = self.get_table_row_count(table_name)
        previous_rows = self.previous_counts.get(table_name, 0)
        new_rows = current_rows - previous_rows
        growth_rate = (new_rows / max(previous_rows, 1)) * 100 if previous_rows > 0 else 0
        
        # Determine if table is healthy
        issues = []
        is_healthy = True
        last_insert_at = self._get_last_insert_time(table_name)
        now = datetime.now(timezone.utc)
        recent_activity = False
        
        if current_rows == 0:
            issues.append("Table is empty")
            is_healthy = False
        elif new_rows == 0 and current_rows > 0:
            issues.append("No new data since last tick")
        elif new_rows < 0:
            issues.append("Row count decreased (possible data loss)")
            is_healthy = False

        if new_rows == 0 and current_rows >= 0:
            recent_inserts = self._count_recent_rows(table_name, self.last_snapshot_at)
            if recent_inserts > 0:
                new_rows = recent_inserts
                growth_rate = (new_rows / max(previous_rows, 1)) * 100 if previous_rows > 0 else 0
                issues = [issue for issue in issues if issue != "No new data since last tick"]
                recent_activity = True

        if table_name in self.table_timestamp_fields:
            if last_insert_at:
                recent_activity = (now - last_insert_at) <= timedelta(hours=24)
            else:
                recent_activity = False

            if current_rows > 0 and not recent_activity:
                issues.append("No rows inserted in the last 24h")
                is_healthy = False
        
        # Check ML requirements
        if table_name in self.ml_table_requirements:
            min_required = self.ml_table_requirements[table_name]['min_rows']
            if current_rows < min_required:
                issues.append(f"Below ML minimum ({current_rows}/{min_required} rows)")
                is_healthy = False
        
        return TableHealth(
            table_name=table_name,
            current_rows=current_rows,
            previous_rows=previous_rows,
            new_rows=new_rows,
            growth_rate=growth_rate,
            is_healthy=is_healthy,
            last_updated=datetime.now(timezone.utc),
            issues=issues,
            last_insert_at=last_insert_at,
            recent_activity=recent_activity,
        )
    
    def get_all_table_insights(self) -> TableInsights:
        """Get comprehensive insights for all tables."""
        if not self.monitoring_enabled:
            return TableInsights(
                timestamp=datetime.now(timezone.utc),
                total_tables=0,
                healthy_tables=0,
                problematic_tables=0,
                total_rows=0,
                new_rows_since_last_tick=0,
                tables={},
                ml_readiness={},
                recommendations=["Monitoring disabled - no Supabase client"]
            )
        
        # Get all table names
        all_tables = self.critical_tables.copy()
        
        # Analyze each table
        tables = {}
        healthy_count = 0
        problematic_count = 0
        total_rows = 0
        new_rows_total = 0
        
        for table_name in all_tables:
            health = self.analyze_table_health(table_name)
            tables[table_name] = health
            
            if health.is_healthy:
                healthy_count += 1
            else:
                problematic_count += 1
            
            total_rows += health.current_rows
            new_rows_total += health.new_rows
        
        # Check ML readiness
        ml_readiness = {}
        for table_name, requirements in self.ml_table_requirements.items():
            if table_name in tables:
                health = tables[table_name]
                ml_readiness[table_name] = (
                    health.current_rows >= requirements['min_rows'] and 
                    health.is_healthy
                )
            else:
                ml_readiness[table_name] = False
        
        # Generate recommendations
        recommendations = []
        if problematic_count > 0:
            recommendations.append(f"⚠️ {problematic_count} tables have issues")
        
        empty_tables = [name for name, health in tables.items() if health.current_rows == 0]
        if empty_tables:
            recommendations.append(f"🚨 Empty tables: {', '.join(empty_tables)}")
        
        stagnant_tables = [name for name, health in tables.items() if health.new_rows == 0 and health.current_rows > 0]
        if stagnant_tables:
            recommendations.append(f"📊 No new data: {', '.join(stagnant_tables)}")
        
        ml_ready_tables = sum(1 for ready in ml_readiness.values() if ready)
        ml_total_tables = len(self.ml_table_requirements)
        if ml_ready_tables < ml_total_tables:
            recommendations.append(f"🤖 ML readiness: {ml_ready_tables}/{ml_total_tables} tables ready")
        
        return TableInsights(
            timestamp=datetime.now(timezone.utc),
            total_tables=len(all_tables),
            healthy_tables=healthy_count,
            problematic_tables=problematic_count,
            total_rows=total_rows,
            new_rows_since_last_tick=new_rows_total,
            tables=tables,
            ml_readiness=ml_readiness,
            recommendations=recommendations
        )
    
    def update_previous_counts(self, insights: TableInsights) -> None:
        """Update previous counts for next comparison."""
        for table_name, health in insights.tables.items():
            self.previous_counts[table_name] = health.current_rows
        self.last_snapshot_at = datetime.now(timezone.utc)
        self._persist_previous_counts()
    
    def format_insights_report(self, insights: TableInsights) -> str:
        """Format insights as a readable report."""
        report = []
        report.append("📊 **TABLE MONITORING REPORT**")
        report.append(f"   Generated: {insights.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
        report.append("")
        
        # Summary
        report.append("📈 **Summary**")
        report.append(f"   • Total Tables: {insights.total_tables}")
        report.append(f"   • Healthy Tables: {insights.healthy_tables}")
        report.append(f"   • Problematic Tables: {insights.problematic_tables}")
        report.append(f"   • Total Rows: {insights.total_rows:,}")
        report.append(f"   • New Rows (this tick): {insights.new_rows_since_last_tick:,}")
        report.append("")
        
        # Table details
        report.append("📋 **Table Details**")
        for table_name, health in insights.tables.items():
            status_emoji = "[OK]" if health.is_healthy else "[ERROR]"
            # Ensure current_rows is a valid number
            try:
                current_rows_str = f"{health.current_rows:,}" if isinstance(health.current_rows, (int, float)) else str(health.current_rows)
            except:
                current_rows_str = "0"
            
            report.append(f"   {status_emoji} **{table_name}**: {current_rows_str} rows")
            
            if health.new_rows > 0:
                try:
                    new_rows_str = f"{health.new_rows:,}" if isinstance(health.new_rows, (int, float)) else str(health.new_rows)
                except:
                    new_rows_str = "0"
                report.append(f"      📈 +{new_rows_str} new rows")
            elif health.new_rows == 0 and health.current_rows > 0:
                report.append(f"      📊 No new data")
            
            if health.issues:
                for issue in health.issues:
                    report.append(f"      ⚠️ {issue}")
        
        report.append("")
        
        # ML Readiness
        report.append("🤖 **ML System Readiness**")
        ml_ready = sum(1 for ready in insights.ml_readiness.values() if ready)
        ml_total = len(insights.ml_readiness)
        report.append(f"   • ML Ready Tables: {ml_ready}/{ml_total}")
        
        for table_name, is_ready in insights.ml_readiness.items():
            status = "[OK] Ready" if is_ready else "[ERROR] Not Ready"
            requirements = self.ml_table_requirements.get(table_name, {})
            min_rows = requirements.get('min_rows', 0)
            table_health = insights.tables.get(
                table_name,
                TableHealth(
                    table_name="",
                    current_rows=0,
                    previous_rows=0,
                    new_rows=0,
                    growth_rate=0.0,
                    is_healthy=False,
                    last_updated=datetime.now(timezone.utc),
                ),
            )
            current_rows = table_health.current_rows
            
            # Ensure current_rows is a valid number for display
            try:
                current_rows_display = f"{current_rows:,}" if isinstance(current_rows, (int, float)) and current_rows >= 0 else "0"
            except:
                current_rows_display = "0"
                
            report.append(f"   • {table_name}: {status} ({current_rows_display}/{min_rows} rows)")
        
        report.append("")
        
        # Recommendations
        if insights.recommendations:
            report.append("💡 **Recommendations**")
            for rec in insights.recommendations:
                report.append(f"   • {rec}")
        
        return "\n".join(report)
    
    def check_ml_data_sufficiency(self, insights: TableInsights) -> Dict[str, Any]:
        """Check if there's sufficient data for ML training."""
        ml_status = {
            'ready_for_training': True,
            'missing_tables': [],
            'insufficient_data': [],
            'recommendations': []
        }
        
        for table_name, requirements in self.ml_table_requirements.items():
            if table_name not in insights.tables:
                ml_status['missing_tables'].append(table_name)
                ml_status['ready_for_training'] = False
                continue
            
            health = insights.tables[table_name]
            min_required = requirements['min_rows']
            
            if health.current_rows < min_required:
                ml_status['insufficient_data'].append({
                    'table': table_name,
                    'current': health.current_rows,
                    'required': min_required,
                    'description': requirements['description']
                })
                ml_status['ready_for_training'] = False
        
        # Generate recommendations
        if ml_status['missing_tables']:
            ml_status['recommendations'].append(f"Missing tables: {', '.join(ml_status['missing_tables'])}")
        
        if ml_status['insufficient_data']:
            for item in ml_status['insufficient_data']:
                ml_status['recommendations'].append(
                    f"{item['table']}: {item['current']}/{item['required']} rows "
                    f"({item['description']})"
                )
        
        if ml_status['ready_for_training']:
            ml_status['recommendations'].append("✅ All ML tables have sufficient data for training")
        
        return ml_status

    def _load_previous_counts(self) -> None:
        """Load persisted table counts so deltas survive process restarts."""
        if not hasattr(self, "state_path") or not isinstance(self.state_path, Path):
            return
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text())
            counts = raw.get("counts", {})
            loaded: Dict[str, int] = {}
            for key, value in counts.items():
                try:
                    loaded[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
            if loaded:
                self.previous_counts = loaded
            timestamp_raw = raw.get("timestamp")
            if timestamp_raw:
                try:
                    parsed = datetime.fromisoformat(timestamp_raw)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    self.last_snapshot_at = parsed
                except Exception:
                    self.last_snapshot_at = None
        except Exception:
            # If the state file is corrupted we ignore it
            self.previous_counts = {}
            self.last_snapshot_at = None

    def _persist_previous_counts(self) -> None:
        """Persist current table counts so the next tick has a baseline."""
        if not hasattr(self, "state_path") or not isinstance(self.state_path, Path):
            return
        snapshot_time = self.last_snapshot_at or datetime.now(timezone.utc)
        payload = {
            "timestamp": snapshot_time.isoformat(),
            "counts": self.previous_counts,
        }
        try:
            self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except Exception:
            # Persistence failures are non-fatal
            pass

    def alert_ml_tables(self, insights: TableInsights) -> None:
        """Emit Slack alerts when ML tables lack fresh data."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for table_name in ['ml_models', 'ml_predictions']:
            health = insights.tables.get(table_name)
            if not health:
                continue

            if health.current_rows == 0:
                cause = "table is empty"
                suggestion = (
                    "rerun the ML training pipeline to repopulate `ml_models` "
                    "and regenerate predictions."
                    if table_name == 'ml_models'
                    else "trigger the ML pipeline to regenerate predictions."
                )
                notify(
                    f"🚨 `{table_name}` has no rows as of {timestamp}. "
                    f"Cause: {cause}. Suggested fix: {suggestion}"
                )
            elif table_name in self.table_timestamp_fields and not health.recent_activity:
                last_seen = health.last_insert_at.isoformat() if health.last_insert_at else "unknown"
                if table_name == 'ml_models':
                    suggestion = "rerun ML training to register a fresh active model."
                else:
                    suggestion = "invoke ML predictions to refresh `ml_predictions`."
                notify(
                    f"⚠️ `{table_name}` has no rows in the last 24h (last insert {last_seen}). "
                    f"Timestamp: {timestamp}. Suggested fix: {suggestion}"
                )


def create_table_monitor(supabase_client: Optional[Client] = None) -> TableMonitor:
    """Create and initialize a table monitor."""
    return TableMonitor(supabase_client)
