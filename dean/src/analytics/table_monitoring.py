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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
from supabase import Client


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
        
        # Critical tables for ML system
        self.critical_tables = [
            'performance_data',
            'ad_lifecycle', 
            'time_series_data',
            'ml_predictions',
            'learning_events',
            'creative_library',
            'historical_data',
            'ad_creation_times'
        ]
        
        # ML training requirements
        self.ml_table_requirements = {
            'performance_data': {'min_rows': 10, 'description': 'Ad performance metrics'},
            'ad_lifecycle': {'min_rows': 5, 'description': 'Ad lifecycle tracking'},
            'time_series_data': {'min_rows': 20, 'description': 'Time-series performance data'},
            'creative_library': {'min_rows': 10, 'description': 'Creative assets and performance'},
            'historical_data': {'min_rows': 50, 'description': 'Historical metric tracking'},
            'ad_creation_times': {'min_rows': 5, 'description': 'Ad creation timestamps'}
        }
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get current row count for a table."""
        if not self.supabase_client:
            return 0
        
        try:
            response = self.supabase_client.table(table_name).select('id', count='exact').execute()
            return response.count or 0
        except Exception as e:
            print(f"⚠️ Failed to get row count for {table_name}: {e}")
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
        
        if current_rows == 0:
            issues.append("Table is empty")
            is_healthy = False
        elif new_rows == 0 and current_rows > 0:
            issues.append("No new data since last tick")
        elif new_rows < 0:
            issues.append("Row count decreased (possible data loss)")
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
            issues=issues
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
            status_emoji = "✅" if health.is_healthy else "❌"
            report.append(f"   {status_emoji} **{table_name}**: {health.current_rows:,} rows")
            
            if health.new_rows > 0:
                report.append(f"      📈 +{health.new_rows:,} new rows")
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
            status = "✅ Ready" if is_ready else "❌ Not Ready"
            requirements = self.ml_table_requirements.get(table_name, {})
            min_rows = requirements.get('min_rows', 0)
            current_rows = insights.tables.get(table_name, TableHealth("", 0, 0, 0, 0, False, datetime.now())).current_rows
            report.append(f"   • {table_name}: {status} ({current_rows}/{min_rows} rows)")
        
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


def create_table_monitor(supabase_client: Optional[Client] = None) -> TableMonitor:
    """Create and initialize a table monitor."""
    return TableMonitor(supabase_client)
