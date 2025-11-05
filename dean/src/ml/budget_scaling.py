"""
Intelligent Budget Scaling System
Auto-scaling, performance-based increases, risk-adjusted scaling, and budget recommendations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Budget scaling strategies."""
    CONSERVATIVE = "conservative"  # Slow, safe scaling
    MODERATE = "moderate"  # Balanced scaling
    AGGRESSIVE = "aggressive"  # Fast scaling
    ADAPTIVE = "adaptive"  # ML-driven scaling


@dataclass
class ScalingDecision:
    """Represents a budget scaling decision."""
    recommended_budget: float
    current_budget: float
    adjustment_percentage: float
    confidence: float
    risk_level: str
    reason: str
    strategy: ScalingStrategy
    estimated_roas: Optional[float] = None
    estimated_purchases: Optional[int] = None
    max_budget: Optional[float] = None
    min_budget: Optional[float] = None


@dataclass
class CampaignPerformance:
    """Campaign performance metrics."""
    campaign_id: str
    current_budget: float
    spend: float
    roas: float
    cpa: float
    purchases: int
    impressions: int
    clicks: int
    ctr: float
    date_range_days: int
    trend: str  # "improving", "declining", "stable"
    volatility: float  # 0-1, higher = more volatile
    confidence_score: float  # 0-1, data quality confidence


class BudgetScalingEngine:
    """Intelligent budget scaling engine."""
    
    def __init__(
        self,
        max_budget_increase_pct: float = 0.5,  # Max 50% increase at once
        min_budget_decrease_pct: float = 0.2,  # Max 20% decrease at once
        min_roas_for_scaling: float = 2.0,  # Minimum ROAS to scale
        min_data_points: int = 3,  # Minimum days of data
        risk_adjustment: bool = True,
    ):
        self.max_budget_increase_pct = max_budget_increase_pct
        self.min_budget_decrease_pct = min_budget_decrease_pct
        self.min_roas_for_scaling = min_roas_for_scaling
        self.min_data_points = min_data_points
        self.risk_adjustment = risk_adjustment
    
    def analyze_campaign_performance(
        self,
        campaign_id: str,
        performance_data: List[Dict[str, Any]],
        current_budget: float,
    ) -> CampaignPerformance:
        """Analyze campaign performance and calculate metrics."""
        if not performance_data or len(performance_data) < self.min_data_points:
            return CampaignPerformance(
                campaign_id=campaign_id,
                current_budget=current_budget,
                spend=0.0,
                roas=0.0,
                cpa=0.0,
                purchases=0,
                impressions=0,
                clicks=0,
                ctr=0.0,
                date_range_days=len(performance_data) if performance_data else 0,
                trend="insufficient_data",
                volatility=1.0,
                confidence_score=0.0,
            )
        
        # Calculate aggregates
        total_spend = sum(p.get("spend", 0) for p in performance_data)
        total_revenue = sum(p.get("revenue", 0) for p in performance_data)
        total_purchases = sum(p.get("purchases", 0) for p in performance_data)
        total_impressions = sum(p.get("impressions", 0) for p in performance_data)
        total_clicks = sum(p.get("clicks", 0) for p in performance_data)
        
        # Calculate metrics
        roas = total_revenue / total_spend if total_spend > 0 else 0.0
        cpa = total_spend / total_purchases if total_purchases > 0 else 0.0
        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0.0
        
        # Calculate trend
        roas_values = [p.get("roas", 0) for p in performance_data if p.get("roas")]
        if len(roas_values) >= 2:
            recent_avg = sum(roas_values[-3:]) / min(3, len(roas_values))
            older_avg = sum(roas_values[:-3]) / max(1, len(roas_values) - 3) if len(roas_values) > 3 else recent_avg
            if recent_avg > older_avg * 1.1:
                trend = "improving"
            elif recent_avg < older_avg * 0.9:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Calculate volatility (standard deviation of ROAS)
        if len(roas_values) > 1:
            mean_roas = sum(roas_values) / len(roas_values)
            variance = sum((x - mean_roas) ** 2 for x in roas_values) / len(roas_values)
            std_dev = variance ** 0.5
            volatility = min(1.0, std_dev / mean_roas if mean_roas > 0 else 1.0)
        else:
            volatility = 1.0
        
        # Calculate confidence score
        data_points = len(performance_data)
        days_span = (performance_data[-1].get("date", datetime.now()) - performance_data[0].get("date", datetime.now())).days + 1
        confidence_score = min(1.0, (data_points / 7) * (days_span / 7))  # Full confidence at 7 days with daily data
        
        return CampaignPerformance(
            campaign_id=campaign_id,
            current_budget=current_budget,
            spend=total_spend,
            roas=roas,
            cpa=cpa,
            purchases=total_purchases,
            impressions=total_impressions,
            clicks=total_clicks,
            ctr=ctr,
            date_range_days=days_span,
            trend=trend,
            volatility=volatility,
            confidence_score=confidence_score,
        )
    
    def calculate_scaling_decision(
        self,
        performance: CampaignPerformance,
        strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE,
        max_budget: Optional[float] = None,
        min_budget: Optional[float] = None,
    ) -> ScalingDecision:
        """Calculate budget scaling decision based on performance."""
        # Base scaling factors by strategy
        strategy_factors = {
            ScalingStrategy.CONSERVATIVE: 0.1,  # 10% max increase
            ScalingStrategy.MODERATE: 0.25,  # 25% max increase
            ScalingStrategy.AGGRESSIVE: 0.5,  # 50% max increase
            ScalingStrategy.ADAPTIVE: 0.3,  # 30% default, adjusted by ML
        }
        
        base_factor = strategy_factors.get(strategy, 0.25)
        
        # Check if we have enough data
        if performance.confidence_score < 0.5:
            return ScalingDecision(
                recommended_budget=performance.current_budget,
                current_budget=performance.current_budget,
                adjustment_percentage=0.0,
                confidence=1.0 - performance.confidence_score,
                risk_level="high",
                reason="insufficient_data",
                strategy=strategy,
            )
        
        # Check minimum ROAS threshold
        if performance.roas < self.min_roas_for_scaling:
            return ScalingDecision(
                recommended_budget=performance.current_budget * (1 - self.min_budget_decrease_pct),
                current_budget=performance.current_budget,
                adjustment_percentage=-self.min_budget_decrease_pct,
                confidence=0.8,
                risk_level="low",
                reason=f"low_roas_{performance.roas:.2f}",
                strategy=strategy,
            )
        
        # Calculate adjustment based on performance
        adjustment_pct = 0.0
        reason = "no_change"
        risk_level = "low"
        
        # ROAS-based scaling
        if performance.roas >= 3.0:
            # Excellent performance - scale aggressively
            adjustment_pct = base_factor * 1.5
            reason = "excellent_roas"
        elif performance.roas >= 2.5:
            # Good performance - scale moderately
            adjustment_pct = base_factor
            reason = "good_roas"
        elif performance.roas >= 2.0:
            # Decent performance - scale conservatively
            adjustment_pct = base_factor * 0.5
            reason = "decent_roas"
        else:
            # Below threshold - no scaling
            adjustment_pct = 0.0
            reason = "below_threshold"
        
        # Adjust based on trend
        if performance.trend == "improving":
            adjustment_pct *= 1.2  # 20% boost for improving trends
            reason += "_improving"
        elif performance.trend == "declining":
            adjustment_pct *= 0.5  # 50% reduction for declining trends
            reason += "_declining"
            risk_level = "medium"
        
        # Risk adjustment
        if self.risk_adjustment:
            # Reduce scaling for high volatility
            volatility_adjustment = 1.0 - (performance.volatility * 0.5)
            adjustment_pct *= max(0.3, volatility_adjustment)
            
            # Adjust risk level based on volatility
            if performance.volatility > 0.5:
                risk_level = "high"
            elif performance.volatility > 0.3:
                risk_level = "medium"
        
        # Cap adjustments
        adjustment_pct = min(adjustment_pct, self.max_budget_increase_pct)
        adjustment_pct = max(adjustment_pct, -self.min_budget_decrease_pct)
        
        # Calculate recommended budget
        recommended_budget = performance.current_budget * (1 + adjustment_pct)
        
        # Apply min/max constraints
        if max_budget:
            recommended_budget = min(recommended_budget, max_budget)
        if min_budget:
            recommended_budget = max(recommended_budget, min_budget)
        
        # Calculate confidence
        confidence = performance.confidence_score * (1.0 - performance.volatility * 0.3)
        
        # Estimate future performance
        estimated_roas = performance.roas * (0.95 if adjustment_pct > 0 else 1.05)  # Slight degradation with scaling
        estimated_purchases = int((recommended_budget * estimated_roas) / (performance.cpa if performance.cpa > 0 else 50))
        
        return ScalingDecision(
            recommended_budget=recommended_budget,
            current_budget=performance.current_budget,
            adjustment_percentage=adjustment_pct,
            confidence=confidence,
            risk_level=risk_level,
            reason=reason,
            strategy=strategy,
            estimated_roas=estimated_roas,
            estimated_purchases=estimated_purchases,
            max_budget=max_budget,
            min_budget=min_budget,
        )
    
    def get_budget_recommendation(
        self,
        campaign_id: str,
        performance_data: List[Dict[str, Any]],
        current_budget: float,
        strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE,
        max_budget: Optional[float] = None,
        min_budget: Optional[float] = None,
    ) -> ScalingDecision:
        """Get budget recommendation for a campaign."""
        # Analyze performance
        performance = self.analyze_campaign_performance(
            campaign_id=campaign_id,
            performance_data=performance_data,
            current_budget=current_budget,
        )
        
        # Calculate scaling decision
        decision = self.calculate_scaling_decision(
            performance=performance,
            strategy=strategy,
            max_budget=max_budget,
            min_budget=min_budget,
        )
        
        return decision


def create_budget_scaling_engine(**kwargs) -> BudgetScalingEngine:
    """Factory function to create BudgetScalingEngine."""
    return BudgetScalingEngine(**kwargs)

