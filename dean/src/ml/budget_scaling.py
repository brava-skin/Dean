"""
ADVANCED Intelligent Budget Scaling System
ML-based predictive scaling, multi-objective optimization, risk-adjusted scaling with Monte Carlo simulation
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Optional advanced imports
try:
    from scipy import optimize, stats
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Budget scaling strategies."""
    CONSERVATIVE = "conservative"  # Slow, safe scaling
    MODERATE = "moderate"  # Balanced scaling
    AGGRESSIVE = "aggressive"  # Fast scaling
    ADAPTIVE = "adaptive"  # ML-driven scaling
    ML_PREDICTIVE = "ml_predictive"  # Advanced ML-based predictive scaling


@dataclass
class ScalingDecision:
    """Represents a budget scaling decision with advanced metrics."""
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
    # Advanced metrics
    risk_adjusted_return: Optional[float] = None
    monte_carlo_confidence: Optional[float] = None
    pareto_optimal: bool = False
    expected_value: Optional[float] = None
    value_at_risk: Optional[float] = None
    sharpe_ratio: Optional[float] = None


@dataclass
class CampaignPerformance:
    """Campaign performance metrics with advanced analytics."""
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
    # Advanced metrics
    momentum: float = 0.0  # Rate of change
    acceleration: float = 0.0  # Rate of change of momentum
    stability_index: float = 0.0  # Performance stability
    efficiency_score: float = 0.0  # Budget efficiency
    market_share: float = 0.0  # Estimated market share
    saturation_level: float = 0.0  # Audience saturation


class MLPredictiveScaler:
    """ML-based predictive budget scaling."""
    
    def __init__(self):
        self.roas_model = None
        self.purchase_model = None
        self.scaler = StandardScaler() if SCIPY_AVAILABLE else None
        self.trained = False
    
    def train_models(self, historical_data: List[Dict[str, Any]]):
        """Train ML models on historical scaling data."""
        if not SCIPY_AVAILABLE or len(historical_data) < 10:
            return False
        
        try:
            # Prepare features
            X = []
            y_roas = []
            y_purchases = []
            
            for data in historical_data:
                features = [
                    data.get("budget", 0),
                    data.get("spend", 0),
                    data.get("impressions", 0),
                    data.get("clicks", 0),
                    data.get("days_active", 0),
                    data.get("previous_roas", 0),
                    data.get("trend", 0),
                    data.get("volatility", 0),
                ]
                X.append(features)
                y_roas.append(data.get("roas", 0))
                y_purchases.append(data.get("purchases", 0))
            
            X = np.array(X)
            y_roas = np.array(y_roas)
            y_purchases = np.array(y_purchases)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train ROAS model
            self.roas_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.roas_model.fit(X_scaled, y_roas)
            
            # Train purchase model
            self.purchase_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.purchase_model.fit(X_scaled, y_purchases)
            
            self.trained = True
            logger.info("ML predictive scaler trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train ML scaler: {e}")
            return False
    
    def predict_performance(self, budget: float, features: Dict[str, Any]) -> Dict[str, float]:
        """Predict performance for a given budget."""
        if not self.trained:
            return {"roas": 0.0, "purchases": 0}
        
        try:
            feature_vector = np.array([[
                budget,
                features.get("spend", 0),
                features.get("impressions", 0),
                features.get("clicks", 0),
                features.get("days_active", 0),
                features.get("previous_roas", 0),
                features.get("trend", 0),
                features.get("volatility", 0),
            ]])
            
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            predicted_roas = self.roas_model.predict(feature_vector_scaled)[0]
            predicted_purchases = max(0, int(self.purchase_model.predict(feature_vector_scaled)[0]))
            
            return {
                "roas": float(predicted_roas),
                "purchases": predicted_purchases,
            }
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return {"roas": 0.0, "purchases": 0}


class MonteCarloRiskAnalyzer:
    """Monte Carlo simulation for risk assessment."""
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
    
    def analyze_risk(
        self,
        current_budget: float,
        recommended_budget: float,
        historical_roas: List[float],
        historical_volatility: float,
    ) -> Dict[str, float]:
        """Run Monte Carlo simulation to assess risk."""
        if not historical_roas or len(historical_roas) < 3:
            return {
                "value_at_risk_95": 0.0,
                "expected_value": 0.0,
                "sharpe_ratio": 0.0,
                "confidence": 0.0,
            }
        
        try:
            mean_roas = np.mean(historical_roas)
            std_roas = np.std(historical_roas) if len(historical_roas) > 1 else mean_roas * 0.2
            
            # Simulate future ROAS
            simulated_roas = np.random.normal(mean_roas, std_roas, self.n_simulations)
            simulated_roas = np.maximum(simulated_roas, 0)  # No negative ROAS
            
            # Calculate returns for recommended budget
            simulated_returns = recommended_budget * simulated_roas
            simulated_current_returns = current_budget * simulated_roas
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(simulated_returns, 5)
            
            # Expected value
            expected_value = np.mean(simulated_returns)
            
            # Sharpe ratio (risk-adjusted return)
            returns_diff = simulated_returns - simulated_current_returns
            sharpe = np.mean(returns_diff) / (np.std(returns_diff) + 1e-6) if np.std(returns_diff) > 0 else 0
            
            # Confidence based on positive outcomes
            positive_outcomes = np.sum(simulated_returns > simulated_current_returns)
            confidence = positive_outcomes / self.n_simulations
            
            return {
                "value_at_risk_95": float(var_95),
                "expected_value": float(expected_value),
                "sharpe_ratio": float(sharpe),
                "confidence": float(confidence),
                "expected_improvement": float(expected_value - np.mean(simulated_current_returns)),
            }
        except Exception as e:
            logger.error(f"Monte Carlo analysis failed: {e}")
            return {
                "value_at_risk_95": 0.0,
                "expected_value": 0.0,
                "sharpe_ratio": 0.0,
                "confidence": 0.0,
            }


class MultiObjectiveOptimizer:
    """Multi-objective optimization for budget allocation."""
    
    def optimize(
        self,
        current_budget: float,
        performance: CampaignPerformance,
        objectives: List[str] = ["roas", "purchases", "efficiency"],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Optimize budget across multiple objectives."""
        if not SCIPY_AVAILABLE:
            return {"optimal_budget": current_budget, "score": 0.0}
        
        if weights is None:
            weights = {"roas": 0.4, "purchases": 0.4, "efficiency": 0.2}
        
        try:
            # Define objective function
            def objective(budget):
                # Normalize budget
                budget = max(10.0, min(budget[0], current_budget * 3))
                
                # Estimate performance at this budget
                budget_ratio = budget / current_budget if current_budget > 0 else 1.0
                
                # Diminishing returns model
                roas_estimate = performance.roas * (1 - 0.1 * (budget_ratio - 1))
                purchases_estimate = performance.purchases * budget_ratio * 0.9
                efficiency = roas_estimate / budget_ratio
                
                # Weighted score (negative for minimization)
                score = -(
                    weights["roas"] * roas_estimate +
                    weights["purchases"] * (purchases_estimate / 100) +
                    weights["efficiency"] * efficiency
                )
                
                return score
            
            # Optimize
            result = optimize.minimize_scalar(
                objective,
                bounds=(current_budget * 0.5, current_budget * 2.0),
                method='bounded',
            )
            
            optimal_budget = result.x
            optimal_score = -result.fun
            
            return {
                "optimal_budget": float(optimal_budget),
                "score": float(optimal_score),
                "pareto_optimal": True,
            }
        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}")
            return {"optimal_budget": current_budget, "score": 0.0}


class BudgetScalingEngine:
    """ADVANCED Intelligent budget scaling engine with ML, Monte Carlo, and multi-objective optimization."""
    
    def __init__(
        self,
        max_budget_increase_pct: float = 0.5,
        min_budget_decrease_pct: float = 0.2,
        min_roas_for_scaling: float = 2.0,
        min_data_points: int = 3,
        risk_adjustment: bool = True,
        use_ml_predictions: bool = True,
        use_monte_carlo: bool = True,
        use_multi_objective: bool = True,
    ):
        self.max_budget_increase_pct = max_budget_increase_pct
        self.min_budget_decrease_pct = min_budget_decrease_pct
        self.min_roas_for_scaling = min_roas_for_scaling
        self.min_data_points = min_data_points
        self.risk_adjustment = risk_adjustment
        self.use_ml_predictions = use_ml_predictions
        self.use_monte_carlo = use_monte_carlo
        self.use_multi_objective = use_multi_objective
        
        # Advanced components
        self.ml_scaler = MLPredictiveScaler() if use_ml_predictions else None
        self.monte_carlo = MonteCarloRiskAnalyzer() if use_monte_carlo else None
        self.multi_objective = MultiObjectiveOptimizer() if use_multi_objective else None
    
    def analyze_campaign_performance(
        self,
        campaign_id: str,
        performance_data: List[Dict[str, Any]],
        current_budget: float,
    ) -> CampaignPerformance:
        """Analyze campaign performance with advanced metrics."""
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
        
        # Advanced trend analysis
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
            
            # Calculate momentum (rate of change)
            if len(roas_values) >= 3:
                momentum = (roas_values[-1] - roas_values[-2]) / max(roas_values[-2], 0.01)
                acceleration = (roas_values[-1] - 2*roas_values[-2] + roas_values[-3]) / max(roas_values[-2], 0.01)
            else:
                momentum = 0.0
                acceleration = 0.0
        else:
            trend = "insufficient_data"
            momentum = 0.0
            acceleration = 0.0
        
        # Calculate volatility
        if len(roas_values) > 1:
            mean_roas = np.mean(roas_values)
            std_dev = np.std(roas_values)
            volatility = min(1.0, std_dev / mean_roas if mean_roas > 0 else 1.0)
            
            # Stability index (inverse of volatility)
            stability_index = max(0.0, 1.0 - volatility)
        else:
            volatility = 1.0
            stability_index = 0.0
        
        # Calculate confidence score
        data_points = len(performance_data)
        days_span = (performance_data[-1].get("date", datetime.now()) - performance_data[0].get("date", datetime.now())).days + 1
        confidence_score = min(1.0, (data_points / 7) * (days_span / 7))
        
        # Efficiency score (ROAS per dollar spent)
        efficiency_score = roas / current_budget if current_budget > 0 else 0.0
        
        # Saturation level (based on impressions and reach)
        saturation_level = min(1.0, total_impressions / 1000000.0)  # Simplified
        
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
            momentum=momentum,
            acceleration=acceleration,
            stability_index=stability_index,
            efficiency_score=efficiency_score,
            saturation_level=saturation_level,
        )
    
    def calculate_scaling_decision(
        self,
        performance: CampaignPerformance,
        strategy: ScalingStrategy = ScalingStrategy.ML_PREDICTIVE,
        max_budget: Optional[float] = None,
        min_budget: Optional[float] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None,
    ) -> ScalingDecision:
        """Calculate budget scaling decision with advanced ML and optimization."""
        # Train ML models if needed
        if self.ml_scaler and historical_data and not self.ml_scaler.trained:
            self.ml_scaler.train_models(historical_data)
        
        # Base scaling factors
        strategy_factors = {
            ScalingStrategy.CONSERVATIVE: 0.1,
            ScalingStrategy.MODERATE: 0.25,
            ScalingStrategy.AGGRESSIVE: 0.5,
            ScalingStrategy.ADAPTIVE: 0.3,
            ScalingStrategy.ML_PREDICTIVE: 0.3,
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
        
        # ML-based prediction
        ml_prediction = None
        if self.ml_scaler and self.ml_scaler.trained:
            test_budgets = [
                performance.current_budget * (1 + base_factor),
                performance.current_budget * (1 + base_factor * 1.5),
            ]
            best_budget = performance.current_budget
            best_score = 0.0
            
            for test_budget in test_budgets:
                features = {
                    "spend": performance.spend,
                    "impressions": performance.impressions,
                    "clicks": performance.clicks,
                    "days_active": performance.date_range_days,
                    "previous_roas": performance.roas,
                    "trend": 1.0 if performance.trend == "improving" else -1.0,
                    "volatility": performance.volatility,
                }
                pred = self.ml_scaler.predict_performance(test_budget, features)
                score = pred["roas"] * pred["purchases"]
                if score > best_score:
                    best_score = score
                    best_budget = test_budget
                    ml_prediction = pred
            
            if ml_prediction:
                adjustment_pct = (best_budget - performance.current_budget) / performance.current_budget
                adjustment_pct = min(adjustment_pct, self.max_budget_increase_pct)
                recommended_budget = performance.current_budget * (1 + adjustment_pct)
            else:
                recommended_budget = performance.current_budget * (1 + base_factor)
                adjustment_pct = base_factor
        else:
            # Fallback to rule-based
            if performance.roas >= 3.0:
                adjustment_pct = base_factor * 1.5
            elif performance.roas >= 2.5:
                adjustment_pct = base_factor
            elif performance.roas >= 2.0:
                adjustment_pct = base_factor * 0.5
            else:
                adjustment_pct = 0.0
            
            # Adjust for trend and momentum
            if performance.trend == "improving":
                adjustment_pct *= (1.2 + performance.momentum * 0.1)
            elif performance.trend == "declining":
                adjustment_pct *= 0.5
            
            recommended_budget = performance.current_budget * (1 + adjustment_pct)
        
        # Multi-objective optimization
        if self.multi_objective:
            opt_result = self.multi_objective.optimize(
                performance.current_budget,
                performance,
            )
            if opt_result.get("score", 0) > 0:
                # Blend ML recommendation with multi-objective result
                recommended_budget = 0.7 * recommended_budget + 0.3 * opt_result["optimal_budget"]
        
        # Apply constraints
        adjustment_pct = (recommended_budget - performance.current_budget) / performance.current_budget
        adjustment_pct = min(adjustment_pct, self.max_budget_increase_pct)
        adjustment_pct = max(adjustment_pct, -self.min_budget_decrease_pct)
        recommended_budget = performance.current_budget * (1 + adjustment_pct)
        
        if max_budget:
            recommended_budget = min(recommended_budget, max_budget)
        if min_budget:
            recommended_budget = max(recommended_budget, min_budget)
        
        # Monte Carlo risk analysis
        risk_metrics = {}
        if self.monte_carlo and len(performance_data) >= 3:
            historical_roas = [p.get("roas", 0) for p in performance_data if p.get("roas")]
            risk_metrics = self.monte_carlo.analyze_risk(
                performance.current_budget,
                recommended_budget,
                historical_roas,
                performance.volatility,
            )
        
        # Calculate confidence
        confidence = performance.confidence_score * (1.0 - performance.volatility * 0.3)
        if risk_metrics:
            confidence = (confidence + risk_metrics.get("confidence", 0)) / 2
        
        # Determine risk level
        if risk_metrics:
            var_95 = risk_metrics.get("value_at_risk_95", 0)
            if var_95 < performance.current_budget * 0.5:
                risk_level = "high"
            elif var_95 < performance.current_budget * 0.8:
                risk_level = "medium"
            else:
                risk_level = "low"
        else:
            risk_level = "medium" if performance.volatility > 0.3 else "low"
        
        # Estimate future performance
        if ml_prediction:
            estimated_roas = ml_prediction.get("roas", performance.roas * 0.95)
            estimated_purchases = ml_prediction.get("purchases", 0)
        else:
            estimated_roas = performance.roas * (0.95 if adjustment_pct > 0 else 1.05)
            estimated_purchases = int((recommended_budget * estimated_roas) / (performance.cpa if performance.cpa > 0 else 50))
        
        reason = f"ml_optimized_{strategy.value}" if ml_prediction else f"rule_based_{performance.trend}"
        
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
            risk_adjusted_return=risk_metrics.get("expected_improvement"),
            monte_carlo_confidence=risk_metrics.get("confidence"),
            pareto_optimal=self.multi_objective is not None,
            expected_value=risk_metrics.get("expected_value"),
            value_at_risk=risk_metrics.get("value_at_risk_95"),
            sharpe_ratio=risk_metrics.get("sharpe_ratio"),
        )
    
    def get_budget_recommendation(
        self,
        campaign_id: str,
        performance_data: List[Dict[str, Any]],
        current_budget: float,
        strategy: ScalingStrategy = ScalingStrategy.ML_PREDICTIVE,
        max_budget: Optional[float] = None,
        min_budget: Optional[float] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None,
    ) -> ScalingDecision:
        """Get advanced budget recommendation for a campaign."""
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
            historical_data=historical_data,
        )
        
        return decision


def create_budget_scaling_engine(**kwargs) -> BudgetScalingEngine:
    """Factory function to create advanced BudgetScalingEngine."""
    return BudgetScalingEngine(**kwargs)


__all__ = [
    "BudgetScalingEngine",
    "ScalingStrategy",
    "ScalingDecision",
    "CampaignPerformance",
    "create_budget_scaling_engine",
]
