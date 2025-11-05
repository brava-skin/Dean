"""
Intelligent Creative Refresh Strategy
Proactive creative replacement and refresh management
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CreativeRefreshManager:
    """Manages intelligent creative refresh with predictive fatigue detection."""
    
    def __init__(
        self,
        fatigue_threshold: float = 0.25,  # 25% performance drop
        min_age_hours: int = 12,  # Minimum age before refresh
        refresh_buffer: int = 2,  # Number of new creatives to have ready
        predictive_fatigue_enabled: bool = True,  # Enable predictive fatigue detection
        staggered_refresh_enabled: bool = True,  # Enable staggered refresh scheduling
    ):
        self.fatigue_threshold = fatigue_threshold
        self.min_age_hours = min_age_hours
        self.refresh_buffer = refresh_buffer
        self.predictive_fatigue_enabled = predictive_fatigue_enabled
        self.staggered_refresh_enabled = staggered_refresh_enabled
        self.refresh_schedule: Dict[str, datetime] = {}  # Track scheduled refreshes
    
    def should_refresh_creative(
        self,
        creative: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Determine if a creative should be refreshed."""
        performance = creative.get("performance", {})
        created_at = creative.get("created_at")
        
        if not created_at:
            return {
                "should_refresh": False,
                "reason": "no_creation_date",
            }
        
        # Parse creation date
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        
        age_hours = (datetime.now() - created_at).total_seconds() / 3600.0
        
        # Too new to refresh
        if age_hours < self.min_age_hours:
            return {
                "should_refresh": False,
                "reason": "too_new",
                "age_hours": age_hours,
            }
        
        # Check for fatigue
        if len(historical_performance) >= 3:
            recent = historical_performance[-3:]
            older = historical_performance[:-3] if len(historical_performance) > 3 else historical_performance[:2]
            
            recent_ctr = sum(p.get("ctr", 0) for p in recent) / len(recent)
            older_ctr = sum(p.get("ctr", 0) for p in older) / len(older) if older else recent_ctr
            
            if older_ctr > 0:
                drop_pct = (older_ctr - recent_ctr) / older_ctr
                
                if drop_pct >= self.fatigue_threshold:
                    return {
                        "should_refresh": True,
                        "reason": "fatigue_detected",
                        "fatigue_pct": drop_pct,
                        "age_hours": age_hours,
                    }
        
        # Check performance thresholds
        roas = performance.get("roas", 0.0)
        ctr = performance.get("ctr", 0.0)
        spend = performance.get("spend", 0.0)
        
        # Low performance with sufficient spend
        if spend >= 40.0 and (roas < 0.8 or ctr < 0.005):
            return {
                "should_refresh": True,
                "reason": "low_performance",
                "roas": roas,
                "ctr": ctr,
                "age_hours": age_hours,
            }
        
        # Predictive fatigue detection
        if self.predictive_fatigue_enabled and len(historical_performance) >= 5:
            fatigue_prediction = self._predict_fatigue(historical_performance)
            if fatigue_prediction.get("will_fatigue_soon", False):
                return {
                    "should_refresh": True,
                    "reason": "predicted_fatigue",
                    "predicted_fatigue_in_hours": fatigue_prediction.get("hours_until_fatigue"),
                    "confidence": fatigue_prediction.get("confidence", 0.0),
                    "age_hours": age_hours,
                }
        
        return {
            "should_refresh": False,
            "reason": "performing_well",
            "age_hours": age_hours,
        }
    
    def _predict_fatigue(self, historical_performance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict if creative will fatigue soon based on performance trends."""
        if len(historical_performance) < 5:
            return {"will_fatigue_soon": False, "confidence": 0.0}
        
        # Extract metrics
        ctr_values = [p.get("ctr", 0) for p in historical_performance]
        roas_values = [p.get("roas", 0) for p in historical_performance]
        
        # Calculate trend (simple linear regression)
        n = len(ctr_values)
        x = list(range(n))
        
        # CTR trend
        x_mean = sum(x) / n
        y_mean = sum(ctr_values) / n
        
        numerator = sum((x[i] - x_mean) * (ctr_values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            ctr_slope = 0
        else:
            ctr_slope = numerator / denominator
        
        # ROAS trend
        roas_mean = sum(roas_values) / n
        roas_numerator = sum((x[i] - x_mean) * (roas_values[i] - roas_mean) for i in range(n))
        roas_slope = roas_numerator / denominator if denominator > 0 else 0
        
        # Predict fatigue if both metrics declining
        if ctr_slope < -0.001 and roas_slope < -0.1:
            # Estimate hours until fatigue threshold
            current_ctr = ctr_values[-1]
            if current_ctr > 0 and ctr_slope < 0:
                hours_until_fatigue = abs(current_ctr * 0.25 / (ctr_slope * 24)) if ctr_slope != 0 else 999
                confidence = min(0.9, abs(ctr_slope) * 1000)  # Higher slope = higher confidence
                
                if hours_until_fatigue < 24:  # Will fatigue within 24 hours
                    return {
                        "will_fatigue_soon": True,
                        "hours_until_fatigue": hours_until_fatigue,
                        "confidence": confidence,
                        "ctr_slope": ctr_slope,
                        "roas_slope": roas_slope,
                    }
        
        return {"will_fatigue_soon": False, "confidence": 0.0}
    
    def plan_refresh_schedule(
        self,
        creatives: List[Dict[str, Any]],
        target_count: int = 5,
    ) -> Dict[str, Any]:
        """Plan refresh schedule for creatives with staggered scheduling."""
        refresh_needed = []
        refresh_soon = []
        scheduled_refreshes = []
        
        for creative in creatives:
            creative_id = creative.get("creative_id") or creative.get("ad_id")
            performance = creative.get("performance", {})
            historical = creative.get("historical_performance", [])
            
            refresh_decision = self.should_refresh_creative(creative, historical)
            
            if refresh_decision["should_refresh"]:
                priority = "high" if refresh_decision.get("fatigue_pct", 0) > 0.3 or refresh_decision.get("reason") == "predicted_fatigue" else "medium"
                
                # Staggered refresh scheduling
                if self.staggered_refresh_enabled and len(refresh_needed) > 0:
                    # Schedule refresh with delay
                    delay_hours = len(refresh_needed) * 2  # Stagger by 2 hours each
                    scheduled_time = datetime.now() + timedelta(hours=delay_hours)
                    self.refresh_schedule[creative_id] = scheduled_time
                    
                    scheduled_refreshes.append({
                        "creative_id": creative_id,
                        "reason": refresh_decision["reason"],
                        "priority": priority,
                        "scheduled_for": scheduled_time.isoformat(),
                        "delay_hours": delay_hours,
                    })
                else:
                    # Immediate refresh
                    refresh_needed.append({
                        "creative_id": creative_id,
                        "reason": refresh_decision["reason"],
                        "priority": priority,
                    })
            elif refresh_decision.get("age_hours", 0) > 48:
                # Old but still performing - refresh soon
                refresh_soon.append({
                    "creative_id": creative_id,
                    "reason": "preventive_refresh",
                    "priority": "low",
                })
        
        # Calculate how many new creatives needed
        current_count = len(creatives)
        needed_count = max(0, target_count - current_count)
        
        # Add refresh buffer
        refresh_count = len(refresh_needed) + len(scheduled_refreshes) + self.refresh_buffer
        total_needed = needed_count + refresh_count
        
        return {
            "refresh_needed": refresh_needed,
            "scheduled_refreshes": scheduled_refreshes,
            "refresh_soon": refresh_soon,
            "new_creatives_needed": total_needed,
            "immediate_refresh": len(refresh_needed),
            "staggered_refresh": len(scheduled_refreshes),
        }
    
    def get_scheduled_refreshes_due(self) -> List[str]:
        """Get list of creative IDs that are due for refresh."""
        now = datetime.now()
        due_creatives = []
        
        for creative_id, scheduled_time in self.refresh_schedule.items():
            if now >= scheduled_time:
                due_creatives.append(creative_id)
        
        return due_creatives
    
    def clear_scheduled_refresh(self, creative_id: str):
        """Clear scheduled refresh for a creative."""
        if creative_id in self.refresh_schedule:
            del self.refresh_schedule[creative_id]
    
    def maintain_diversity(
        self,
        existing_creatives: List[Dict[str, Any]],
        new_creative: Dict[str, Any],
        min_similarity: float = 0.7,
    ) -> bool:
        """Check if new creative maintains diversity."""
        # Simplified diversity check
        # In production, would use Creative DNA similarity
        
        new_prompt = new_creative.get("image_prompt", "")
        new_text = new_creative.get("text_overlay", "")
        
        for existing in existing_creatives:
            existing_prompt = existing.get("image_prompt", "")
            existing_text = existing.get("text_overlay", "")
            
            # Simple text similarity check
            prompt_overlap = len(set(new_prompt.lower().split()) & set(existing_prompt.lower().split()))
            text_match = new_text == existing_text
            
            if prompt_overlap > 5 or text_match:
                return False  # Too similar
        
        return True  # Diverse enough


def create_creative_refresh_manager() -> CreativeRefreshManager:
    """Create a creative refresh manager."""
    return CreativeRefreshManager()


__all__ = ["CreativeRefreshManager", "create_creative_refresh_manager"]

