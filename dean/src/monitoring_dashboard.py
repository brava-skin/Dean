"""
Advanced Monitoring Dashboard for Smart Scheduler.
Provides insights, statistics, and health monitoring.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

from slack import notify, client as slack_client


class MonitoringDashboard:
    """
    Advanced monitoring dashboard for the smart scheduler.
    Provides insights, statistics, and health monitoring.
    """
    
    def __init__(self, tick_db_path: str = "data/smart_ticks.sqlite"):
        self.tick_db_path = tick_db_path
    
    def get_system_health_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        cutoff_time = int((datetime.now() - timedelta(days=days)).timestamp())
        
        with sqlite3.connect(self.tick_db_path) as conn:
            # Get tick statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_ticks,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_ticks,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_ticks,
                    SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) as skipped_ticks,
                    AVG(CASE WHEN status = 'completed' THEN 1.0 ELSE 0.0 END) * 100 as success_rate
                FROM tick_history 
                WHERE timestamp >= ?
            """, (cutoff_time,))
            
            tick_stats = cursor.fetchone()
            total_ticks, successful, failed, skipped, success_rate = tick_stats
            
            # Get health metrics
            cursor = conn.execute("""
                SELECT 
                    AVG(health_score) as avg_health,
                    AVG(api_latency) as avg_api_latency,
                    AVG(db_latency) as avg_db_latency,
                    MAX(health_score) as max_health,
                    MIN(health_score) as min_health
                FROM tick_health 
                WHERE timestamp >= ?
            """, (cutoff_time,))
            
            health_stats = cursor.fetchone()
            avg_health, avg_api_latency, avg_db_latency, max_health, min_health = health_stats
            
            # Get recent errors
            cursor = conn.execute("""
                SELECT error_message, COUNT(*) as count
                FROM tick_history 
                WHERE timestamp >= ? AND error_message IS NOT NULL
                GROUP BY error_message
                ORDER BY count DESC
                LIMIT 5
            """, (cutoff_time,))
            
            recent_errors = cursor.fetchall()
            
            # Get summary execution stats
            cursor = conn.execute("""
                SELECT 
                    summary_type,
                    COUNT(*) as execution_count,
                    AVG(CASE WHEN status = 'completed' THEN 1.0 ELSE 0.0 END) * 100 as success_rate
                FROM tick_history 
                WHERE timestamp >= ? AND summary_type IS NOT NULL
                GROUP BY summary_type
            """, (cutoff_time,))
            
            summary_stats = cursor.fetchall()
            
            return {
                "period_days": days,
                "tick_statistics": {
                    "total_ticks": total_ticks or 0,
                    "successful_ticks": successful or 0,
                    "failed_ticks": failed or 0,
                    "skipped_ticks": skipped or 0,
                    "success_rate": success_rate or 0
                },
                "health_metrics": {
                    "average_health_score": avg_health or 0,
                    "max_health_score": max_health or 0,
                    "min_health_score": min_health or 0,
                    "average_api_latency": avg_api_latency or 0,
                    "average_db_latency": avg_db_latency or 0
                },
                "recent_errors": [
                    {"error": error, "count": count} 
                    for error, count in recent_errors
                ],
                "summary_execution": [
                    {"type": summary_type, "executions": count, "success_rate": rate}
                    for summary_type, count, rate in summary_stats
                ]
            }
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get performance trends over time."""
        cutoff_time = int((datetime.now() - timedelta(days=days)).timestamp())
        
        with sqlite3.connect(self.tick_db_path) as conn:
            # Get daily performance
            cursor = conn.execute("""
                SELECT 
                    date,
                    COUNT(*) as total_ticks,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                    AVG(health_score) as avg_health
                FROM (
                    SELECT 
                        date(timestamp, 'unixepoch') as date,
                        status,
                        (SELECT AVG(health_score) FROM tick_health 
                         WHERE date(timestamp, 'unixepoch') = date(tick_history.timestamp, 'unixepoch')) as health_score
                    FROM tick_history 
                    WHERE timestamp >= ?
                )
                GROUP BY date
                ORDER BY date
            """, (cutoff_time,))
            
            daily_performance = cursor.fetchall()
            
            # Get hourly distribution
            cursor = conn.execute("""
                SELECT 
                    hour,
                    COUNT(*) as tick_count,
                    AVG(CASE WHEN status = 'completed' THEN 1.0 ELSE 0.0 END) * 100 as success_rate
                FROM tick_history 
                WHERE timestamp >= ?
                GROUP BY hour
                ORDER BY hour
            """, (cutoff_time,))
            
            hourly_distribution = cursor.fetchall()
            
            return {
                "daily_performance": [
                    {
                        "date": date,
                        "total_ticks": total,
                        "successful_ticks": successful,
                        "success_rate": (successful / total * 100) if total > 0 else 0,
                        "avg_health_score": avg_health or 0
                    }
                    for date, total, successful, avg_health in daily_performance
                ],
                "hourly_distribution": [
                    {
                        "hour": hour,
                        "tick_count": count,
                        "success_rate": success_rate or 0
                    }
                    for hour, count, success_rate in hourly_distribution
                ]
            }
    
    def get_alert_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of alerts and notifications."""
        # This would integrate with your alert system
        # For now, return a placeholder structure
        return {
            "period_days": days,
            "alert_types": {
                "queue_empty": 0,
                "system_health": 0,
                "ad_kills": 0,
                "promotions": 0,
                "scaling": 0,
                "fatigue": 0
            },
            "summary_notifications": {
                "3h_summaries": 0,
                "daily_summaries": 0
            },
            "alert_frequency": {
                "high_priority": 0,
                "medium_priority": 0,
                "low_priority": 0
            }
        }
    
    def generate_health_report_message(self, days: int = 7) -> str:
        """Generate a human-readable health report message."""
        health_report = self.get_system_health_report(days)
        trends = self.get_performance_trends(days)
        
        # Calculate health status
        avg_health = health_report["health_metrics"]["average_health_score"]
        success_rate = health_report["tick_statistics"]["success_rate"]
        
        if avg_health >= 80 and success_rate >= 95:
            status_emoji = "🟢"
            status_text = "Excellent"
        elif avg_health >= 60 and success_rate >= 85:
            status_emoji = "🟡"
            status_text = "Good"
        else:
            status_emoji = "🔴"
            status_text = "Needs Attention"
        
        # Build message
        lines = [
            f"📊 System Health Report ({days} days)",
            f"{status_emoji} Overall Status: {status_text}",
            "",
            "📈 Performance:",
            f"  • Success Rate: {success_rate:.1f}%",
            f"  • Total Ticks: {health_report['tick_statistics']['total_ticks']}",
            f"  • Successful: {health_report['tick_statistics']['successful_ticks']}",
            f"  • Failed: {health_report['tick_statistics']['failed_ticks']}",
            f"  • Skipped: {health_report['tick_statistics']['skipped_ticks']}",
            "",
            "🔧 Health Metrics:",
            f"  • Health Score: {avg_health:.1f}/100",
            f"  • API Latency: {health_report['health_metrics']['average_api_latency']:.3f}s",
            f"  • DB Latency: {health_report['health_metrics']['average_db_latency']:.3f}s",
        ]
        
        # Add recent errors if any
        if health_report["recent_errors"]:
            lines.extend([
                "",
                "⚠️ Recent Errors:"
            ])
            for error_info in health_report["recent_errors"][:3]:
                lines.append(f"  • {error_info['error'][:50]}... ({error_info['count']} times)")
        
        # Add summary execution stats
        if health_report["summary_execution"]:
            lines.extend([
                "",
                "📋 Summary Execution:"
            ])
            for summary_info in health_report["summary_execution"]:
                lines.append(f"  • {summary_info['type']}: {summary_info['executions']} runs, {summary_info['success_rate']:.1f}% success")
        
        return "\n".join(lines)
    
    def send_health_report(self, days: int = 7):
        """Send health report to Slack."""
        try:
            message = self.generate_health_report_message(days)
            notify(message)
        except Exception as e:
            notify(f"⚠️ Failed to generate health report: {e}")
    
    def get_system_recommendations(self, days: int = 7) -> List[str]:
        """Get system improvement recommendations."""
        health_report = self.get_system_health_report(days)
        recommendations = []
        
        # Check success rate
        success_rate = health_report["tick_statistics"]["success_rate"]
        if success_rate < 90:
            recommendations.append("🔧 Low success rate - check error logs and system health")
        
        # Check health score
        avg_health = health_report["health_metrics"]["average_health_score"]
        if avg_health < 70:
            recommendations.append("🏥 System health is poor - consider scaling resources")
        
        # Check API latency
        api_latency = health_report["health_metrics"]["average_api_latency"]
        if api_latency > 5.0:
            recommendations.append("🐌 High API latency - check Meta API status and network")
        
        # Check DB latency
        db_latency = health_report["health_metrics"]["average_db_latency"]
        if db_latency > 1.0:
            recommendations.append("💾 High database latency - check database performance")
        
        # Check for frequent errors
        if health_report["recent_errors"]:
            recommendations.append("🚨 Frequent errors detected - review error patterns")
        
        # Check summary execution
        summary_stats = health_report["summary_execution"]
        for summary_info in summary_stats:
            if summary_info["success_rate"] < 80:
                recommendations.append(f"📊 {summary_info['type']} summaries failing - check summary logic")
        
        return recommendations
    
    def export_metrics(self, days: int = 7, format: str = "json") -> str:
        """Export metrics in specified format."""
        health_report = self.get_system_health_report(days)
        trends = self.get_performance_trends(days)
        recommendations = self.get_system_recommendations(days)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "period_days": days,
            "health_report": health_report,
            "performance_trends": trends,
            "recommendations": recommendations
        }
        
        if format == "json":
            return json.dumps(export_data, indent=2)
        elif format == "csv":
            # Convert to CSV format (simplified)
            lines = ["metric,value"]
            lines.append(f"total_ticks,{health_report['tick_statistics']['total_ticks']}")
            lines.append(f"success_rate,{health_report['tick_statistics']['success_rate']}")
            lines.append(f"avg_health_score,{health_report['health_metrics']['average_health_score']}")
            return "\n".join(lines)
        else:
            return str(export_data)
