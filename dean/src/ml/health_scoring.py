"""
Health Scoring System
Comprehensive health metrics for system, campaigns, and creatives
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class HealthScorer:
    """Calculates health scores for various components."""
    
    def __init__(self):
        self.weights = {
            "roas": 0.4,
            "ctr": 0.3,
            "cpa": 0.2,
            "stability": 0.1,
        }
    
    def calculate_creative_health(
        self,
        performance_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calculate health score for a creative."""
        roas = performance_data.get("roas", 0.0)
        ctr = performance_data.get("ctr", 0.0)
        cpa = performance_data.get("cpa", 0.0)
        spend = performance_data.get("spend", 0.0)
        impressions = performance_data.get("impressions", 0)
        
        # Normalize metrics to 0-1 scale
        roas_score = min(roas / 3.0, 1.0)  # ROAS of 3.0 = perfect
        ctr_score = min(ctr / 0.02, 1.0)   # CTR of 2% = perfect
        cpa_score = 1.0 - min(cpa / 50.0, 1.0) if cpa > 0 else 0.5  # CPA of 0 = perfect
        
        # Stability score
        stability_score = 1.0
        if historical_data and len(historical_data) >= 3:
            recent_values = [p.get("roas", 0.0) for p in historical_data[-3:]]
            if len(recent_values) >= 2:
                variance = statistics.variance(recent_values) if len(recent_values) > 1 else 0.0
                stability_score = 1.0 / (1.0 + variance)
        
        # Composite health score
        health_score = (
            roas_score * self.weights["roas"] +
            ctr_score * self.weights["ctr"] +
            cpa_score * self.weights["cpa"] +
            stability_score * self.weights["stability"]
        )
        
        # Determine health status
        if health_score >= 0.8:
            status = "excellent"
        elif health_score >= 0.6:
            status = "good"
        elif health_score >= 0.4:
            status = "fair"
        elif health_score >= 0.2:
            status = "poor"
        else:
            status = "critical"
        
        return {
            "health_score": float(health_score),
            "status": status,
            "components": {
                "roas_score": float(roas_score),
                "ctr_score": float(ctr_score),
                "cpa_score": float(cpa_score),
                "stability_score": float(stability_score),
            },
            "data_quality": "good" if impressions >= 200 and spend >= 20 else "insufficient",
        }
    
    def calculate_campaign_health(
        self,
        creatives: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate health score for entire campaign."""
        if not creatives:
            return {
                "health_score": 0.0,
                "status": "no_data",
            }
        
        # Calculate health for each creative
        creative_healths = []
        for creative in creatives:
            perf = creative.get("performance", {})
            health = self.calculate_creative_health(perf)
            creative_healths.append(health["health_score"])
        
        # Average health
        avg_health = statistics.mean(creative_healths) if creative_healths else 0.0
        
        # Status
        if avg_health >= 0.8:
            status = "excellent"
        elif avg_health >= 0.6:
            status = "good"
        elif avg_health >= 0.4:
            status = "fair"
        else:
            status = "needs_attention"
        
        # Count by status
        status_counts = {}
        for creative in creatives:
            perf = creative.get("performance", {})
            health = self.calculate_creative_health(perf)
            status = health["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "health_score": float(avg_health),
            "status": status,
            "creative_count": len(creatives),
            "healthy_creatives": sum(1 for h in creative_healths if h >= 0.6),
            "status_breakdown": status_counts,
        }
    
    def calculate_system_health(
        self,
        campaign_healths: List[Dict[str, Any]],
        system_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate overall system health."""
        if not campaign_healths:
            return {
                "health_score": 0.0,
                "status": "no_data",
            }
        
        # Average campaign health
        campaign_scores = [ch.get("health_score", 0.0) for ch in campaign_healths]
        avg_campaign_health = statistics.mean(campaign_scores) if campaign_scores else 0.0
        
        # System metrics
        api_health = system_metrics.get("api_health", 1.0)
        data_quality = system_metrics.get("data_quality", 1.0)
        
        # Composite score
        system_score = (
            avg_campaign_health * 0.7 +
            api_health * 0.2 +
            data_quality * 0.1
        )
        
        # Status
        if system_score >= 0.8:
            status = "excellent"
        elif system_score >= 0.6:
            status = "good"
        elif system_score >= 0.4:
            status = "fair"
        else:
            status = "needs_attention"
        
        return {
            "health_score": float(system_score),
            "status": status,
            "campaign_health": float(avg_campaign_health),
            "api_health": float(api_health),
            "data_quality": float(data_quality),
        }
    
    def predict_health_trend(
        self,
        health_history: List[float],
        horizon: int = 24,
    ) -> Dict[str, Any]:
        """Predict health trend."""
        if len(health_history) < 3:
            return {
                "trend": "insufficient_data",
                "predicted_health": health_history[-1] if health_history else 0.5,
            }
        
        # Simple trend calculation
        recent = health_history[-3:]
        older = health_history[:-3] if len(health_history) > 3 else health_history[:2]
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        if recent_avg > older_avg * 1.05:
            trend = "improving"
        elif recent_avg < older_avg * 0.95:
            trend = "declining"
        else:
            trend = "stable"
        
        # Predict future
        if len(health_history) >= 2:
            slope = (recent_avg - older_avg) / len(health_history)
            predicted = recent_avg + (slope * horizon)
            predicted = max(0.0, min(1.0, predicted))
        else:
            predicted = recent_avg
        
        return {
            "trend": trend,
            "predicted_health": float(predicted),
            "current_health": float(recent_avg),
        }


def create_health_scorer() -> HealthScorer:
    """Create a health scorer."""
    return HealthScorer()


__all__ = ["HealthScorer", "create_health_scorer"]

