"""
Dynamic Budget Allocation System
Real-time budget optimization across creatives
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class BudgetOptimizer:
    """Optimizes budget allocation across creatives."""
    
    def __init__(self):
        self.min_budget_per_creative = 5.0
        self.max_budget_per_creative = 50.0
    
    def allocate_budget(
        self,
        total_budget: float,
        creatives: List[Dict[str, Any]],
        allocation_strategy: str = "performance_based",
    ) -> Dict[str, float]:
        """Allocate budget across creatives."""
        if not creatives:
            return {}
        
        allocations = {}
        
        if allocation_strategy == "equal":
            # Equal allocation
            budget_per_creative = total_budget / len(creatives)
            for creative in creatives:
                creative_id = creative.get("creative_id") or creative.get("ad_id", "")
                allocations[creative_id] = budget_per_creative
        
        elif allocation_strategy == "performance_based":
            # Performance-based allocation
            allocations = self._performance_based_allocation(total_budget, creatives)
        
        elif allocation_strategy == "risk_adjusted":
            # Risk-adjusted allocation
            allocations = self._risk_adjusted_allocation(total_budget, creatives)
        
        # Ensure minimums and maximums
        for creative_id in allocations:
            allocations[creative_id] = max(
                self.min_budget_per_creative,
                min(allocations[creative_id], self.max_budget_per_creative)
            )
        
        # Normalize to total budget
        total_allocated = sum(allocations.values())
        if total_allocated > 0:
            scale = total_budget / total_allocated
            for creative_id in allocations:
                allocations[creative_id] *= scale
        
        return allocations
    
    def _performance_based_allocation(
        self,
        total_budget: float,
        creatives: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Allocate based on performance scores."""
        # Calculate performance scores
        scores = {}
        for creative in creatives:
            creative_id = creative.get("creative_id") or creative.get("ad_id", "")
            performance = creative.get("performance", {})
            
            roas = performance.get("roas", 0.0)
            ctr = performance.get("ctr", 0.0)
            purchases = performance.get("purchases", 0)
            spend = performance.get("spend", 0.0)
            
            # Composite score
            if spend > 0:
                score = (roas * 0.5) + (ctr * 100 * 0.3) + (purchases / spend * 10 * 0.2)
            else:
                # New creative - use default score
                score = 1.0
            
            scores[creative_id] = max(score, 0.1)  # Minimum score
        
        # Allocate proportionally
        total_score = sum(scores.values())
        allocations = {}
        
        for creative_id, score in scores.items():
            if total_score > 0:
                allocations[creative_id] = (score / total_score) * total_budget
            else:
                allocations[creative_id] = total_budget / len(creatives)
        
        return allocations
    
    def _risk_adjusted_allocation(
        self,
        total_budget: float,
        creatives: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Allocate with risk adjustment."""
        # Calculate risk-adjusted returns
        risk_scores = {}
        
        for creative in creatives:
            creative_id = creative.get("creative_id") or creative.get("ad_id", "")
            performance = creative.get("performance", {})
            
            roas = performance.get("roas", 0.0)
            ctr = performance.get("ctr", 0.0)
            impressions = performance.get("impressions", 0)
            
            # Risk = variance in performance (simplified)
            # Lower impressions = higher risk
            risk = 1.0 / (1.0 + impressions / 1000.0)
            
            # Risk-adjusted return
            risk_adjusted_return = roas * (1.0 - risk * 0.3)
            risk_scores[creative_id] = max(risk_adjusted_return, 0.1)
        
        # Allocate based on risk-adjusted scores
        total_score = sum(risk_scores.values())
        allocations = {}
        
        for creative_id, score in risk_scores.items():
            if total_score > 0:
                allocations[creative_id] = (score / total_score) * total_budget
            else:
                allocations[creative_id] = total_budget / len(creatives)
        
        return allocations
    
    def recommend_budget_rebalancing(
        self,
        current_allocations: Dict[str, float],
        creatives: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Recommend budget rebalancing."""
        total_budget = sum(current_allocations.values())
        
        # Calculate optimal allocation
        optimal_allocations = self.allocate_budget(
            total_budget,
            creatives,
            allocation_strategy="performance_based",
        )
        
        # Calculate differences
        recommendations = {}
        for creative_id in optimal_allocations:
            current = current_allocations.get(creative_id, 0.0)
            optimal = optimal_allocations[creative_id]
            diff = optimal - current
            
            if abs(diff) > 1.0:  # Only recommend if difference > €1
                recommendations[creative_id] = {
                    "current": current,
                    "recommended": optimal,
                    "difference": diff,
                    "action": "increase" if diff > 0 else "decrease",
                }
        
        return {
            "recommendations": recommendations,
            "total_budget": total_budget,
        }


def create_budget_optimizer() -> BudgetOptimizer:
    """Create a budget optimizer."""
    return BudgetOptimizer()


__all__ = ["BudgetOptimizer", "create_budget_optimizer"]

