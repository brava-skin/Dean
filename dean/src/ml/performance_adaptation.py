"""
Real-Time Performance Adaptation
Dynamically adjusts rules and thresholds based on performance
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class AdaptiveThresholds:
    """Adaptive threshold management."""
    
    def __init__(self):
        self.base_thresholds = {
            "ctr_floor": 0.008,
            "cpa_max": 40.0,
            "roas_min": 1.0,
            "cpm_max": 120.0,
        }
        self.adaptation_history: List[Dict[str, Any]] = []
    
    def adapt_thresholds(
        self,
        recent_performance: List[Dict[str, Any]],
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Adapt thresholds based on recent performance."""
        if not recent_performance:
            return self.base_thresholds.copy()
        
        # Calculate recent averages
        ctrs = [p.get("ctr", 0) for p in recent_performance if p.get("ctr", 0) > 0]
        cpas = [p.get("cpa", 0) for p in recent_performance if p.get("cpa", 0) > 0 and p.get("cpa", 0) < 1000]
        roas_list = [p.get("roas", 0) for p in recent_performance if p.get("roas", 0) > 0]
        cpms = [p.get("cpm", 0) for p in recent_performance if p.get("cpm", 0) > 0]
        
        adapted = self.base_thresholds.copy()
        
        # Adapt CTR floor based on recent performance
        if ctrs:
            avg_ctr = statistics.mean(ctrs)
            # If average CTR is higher, raise the floor
            if avg_ctr > self.base_thresholds["ctr_floor"] * 1.2:
                adapted["ctr_floor"] = avg_ctr * 0.8
            elif avg_ctr < self.base_thresholds["ctr_floor"] * 0.8:
                adapted["ctr_floor"] = max(avg_ctr * 0.9, 0.005)
        
        # Adapt CPA max based on recent performance
        if cpas:
            avg_cpa = statistics.mean(cpas)
            # If average CPA is lower, tighten the threshold
            if avg_cpa < self.base_thresholds["cpa_max"] * 0.8:
                adapted["cpa_max"] = avg_cpa * 1.2
            elif avg_cpa > self.base_thresholds["cpa_max"] * 1.2:
                adapted["cpa_max"] = min(avg_cpa * 1.1, 60.0)
        
        # Adapt ROAS min based on recent performance
        if roas_list:
            avg_roas = statistics.mean(roas_list)
            # If average ROAS is higher, raise the minimum
            if avg_roas > self.base_thresholds["roas_min"] * 1.2:
                adapted["roas_min"] = avg_roas * 0.9
            elif avg_roas < self.base_thresholds["roas_min"] * 0.8:
                adapted["roas_min"] = max(avg_roas * 0.95, 0.8)
        
        # Adapt CPM max based on market conditions
        if cpms:
            avg_cpm = statistics.mean(cpms)
            # If average CPM is higher, raise the threshold
            if avg_cpm > self.base_thresholds["cpm_max"] * 1.2:
                adapted["cpm_max"] = avg_cpm * 1.1
            elif avg_cpm < self.base_thresholds["cpm_max"] * 0.8:
                adapted["cpm_max"] = max(avg_cpm * 1.2, 100.0)
        
        # Record adaptation
        self.adaptation_history.append({
            "timestamp": datetime.now(),
            "thresholds": adapted.copy(),
            "recent_performance_count": len(recent_performance),
        })
        
        return adapted
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds."""
        if self.adaptation_history:
            return self.adaptation_history[-1]["thresholds"]
        return self.base_thresholds.copy()


class MarketConditionDetector:
    """Detects market conditions."""
    
    def detect_conditions(
        self,
        recent_performance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect current market conditions."""
        if not recent_performance:
            return {
                "condition": "normal",
                "competition_level": "medium",
            }
        
        # Calculate metrics
        cpms = [p.get("cpm", 0) for p in recent_performance if p.get("cpm", 0) > 0]
        ctrs = [p.get("ctr", 0) for p in recent_performance if p.get("ctr", 0) > 0]
        
        if not cpms or not ctrs:
            return {
                "condition": "normal",
                "competition_level": "medium",
            }
        
        avg_cpm = statistics.mean(cpms)
        avg_ctr = statistics.mean(ctrs)
        
        # Detect competition level
        if avg_cpm > 150:
            competition = "high"
        elif avg_cpm < 80:
            competition = "low"
        else:
            competition = "medium"
        
        # Detect market condition
        if avg_ctr < 0.005 and avg_cpm > 120:
            condition = "challenging"
        elif avg_ctr > 0.015 and avg_cpm < 90:
            condition = "favorable"
        else:
            condition = "normal"
        
        return {
            "condition": condition,
            "competition_level": competition,
            "avg_cpm": avg_cpm,
            "avg_ctr": avg_ctr,
        }


class DynamicRuleAdjuster:
    """Dynamically adjusts rules based on performance."""
    
    def __init__(self):
        self.adaptive_thresholds = AdaptiveThresholds()
        self.market_detector = MarketConditionDetector()
    
    def adjust_rules(
        self,
        current_rules: Dict[str, Any],
        recent_performance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Adjust rules based on performance."""
        # Detect market conditions
        market_conditions = self.market_detector.detect_conditions(recent_performance)
        
        # Adapt thresholds
        adapted_thresholds = self.adaptive_thresholds.adapt_thresholds(
            recent_performance,
            market_conditions,
        )
        
        # Adjust rules
        adjusted_rules = current_rules.copy()
        
        # Update kill rules with adapted thresholds
        if "asc_plus" in adjusted_rules and "kill" in adjusted_rules["asc_plus"]:
            kill_rules = adjusted_rules["asc_plus"]["kill"]
            
            # Update threshold-based rules
            for rule in kill_rules:
                rule_type = rule.get("type")
                
                if rule_type == "ctr_below":
                    rule["ctr_lt"] = adapted_thresholds["ctr_floor"]
                
                elif rule_type == "cpa_gte":
                    rule["cpa_gte"] = adapted_thresholds["cpa_max"]
                
                elif rule_type == "roas_below":
                    rule["roas_lt"] = adapted_thresholds["roas_min"]
                
                elif rule_type == "cpm_above":
                    rule["cpm_above"] = adapted_thresholds["cpm_max"]
        
        return {
            "rules": adjusted_rules,
            "thresholds": adapted_thresholds,
            "market_conditions": market_conditions,
        }


def create_performance_adaptation() -> DynamicRuleAdjuster:
    """Create a performance adaptation system."""
    return DynamicRuleAdjuster()


__all__ = [
    "DynamicRuleAdjuster",
    "AdaptiveThresholds",
    "MarketConditionDetector",
    "create_performance_adaptation",
]

