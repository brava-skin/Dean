"""
ADVANCED Real-Time Performance Adaptation
Bayesian optimization for thresholds, market regime detection, advanced trend analysis
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import statistics
from dataclasses import dataclass

# Advanced imports
try:
    from scipy import stats, optimize
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.acquisition import gaussian_ei
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Market regime classification."""
    regime_type: str  # "bull", "bear", "volatile", "stable", "trending"
    confidence: float
    characteristics: Dict[str, float]
    expected_duration_days: int
    recommended_strategy: str


@dataclass
class TrendAnalysis:
    """Advanced trend analysis results."""
    trend_direction: str  # "up", "down", "sideways"
    trend_strength: float  # 0-1
    momentum: float
    acceleration: float
    volatility: float
    seasonality_detected: bool
    regime: MarketRegime
    forecast_horizon_days: int
    forecast_values: List[float]
    confidence: float


class BayesianThresholdOptimizer:
    """Bayesian optimization for threshold tuning."""
    
    def __init__(self):
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_thresholds: Dict[str, float] = {}
    
    def optimize_threshold(
        self,
        threshold_name: str,
        performance_data: List[Dict[str, Any]],
        objective_metric: str = "roas",
        bounds: Tuple[float, float] = (0.0, 100.0),
    ) -> float:
        """Optimize a single threshold using Bayesian optimization."""
        if not BAYESOPT_AVAILABLE or len(performance_data) < 10:
            # Fallback to simple optimization
            return self._simple_optimize(threshold_name, performance_data, objective_metric, bounds)
        
        try:
            # Define objective function
            def objective(threshold_value):
                # Simulate performance with this threshold
                score = self._evaluate_threshold(
                    threshold_name,
                    threshold_value,
                    performance_data,
                    objective_metric,
                )
                return -score  # Negative for minimization
            
            # Bayesian optimization
            result = gp_minimize(
                func=objective,
                dimensions=[Real(bounds[0], bounds[1])],
                n_calls=20,
                random_state=42,
                acq_func='EI',  # Expected Improvement
            )
            
            optimal_threshold = result.x[0]
            
            self.optimization_history.append({
                "threshold_name": threshold_name,
                "optimal_value": optimal_threshold,
                "score": -result.fun,
                "timestamp": datetime.now(),
            })
            
            self.best_thresholds[threshold_name] = optimal_threshold
            
            logger.info(f"Bayesian optimization for {threshold_name}: {optimal_threshold:.2f}")
            return optimal_threshold
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return self._simple_optimize(threshold_name, performance_data, objective_metric, bounds)
    
    def _evaluate_threshold(
        self,
        threshold_name: str,
        threshold_value: float,
        performance_data: List[Dict[str, Any]],
        objective_metric: str,
    ) -> float:
        """Evaluate threshold performance."""
        # Simulate decisions with this threshold
        scores = []
        
        for data in performance_data:
            metric_value = data.get(threshold_name.replace("_threshold", ""), 0)
            
            # Decision based on threshold
            if threshold_name.startswith("ctr"):
                decision = "good" if metric_value >= threshold_value else "bad"
            elif threshold_name.startswith("cpa") or threshold_name.startswith("cpm"):
                decision = "good" if metric_value <= threshold_value else "bad"
            else:
                decision = "good" if metric_value >= threshold_value else "bad"
            
            # Score based on objective
            if decision == "good":
                score = data.get(objective_metric, 0)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _simple_optimize(
        self,
        threshold_name: str,
        performance_data: List[Dict[str, Any]],
        objective_metric: str,
        bounds: Tuple[float, float],
    ) -> float:
        """Simple grid search optimization."""
        best_value = (bounds[0] + bounds[1]) / 2
        best_score = 0.0
        
        # Test multiple values
        test_values = np.linspace(bounds[0], bounds[1], 20)
        
        for value in test_values:
            score = self._evaluate_threshold(
                threshold_name,
                value,
                performance_data,
                objective_metric,
            )
            if score > best_score:
                best_score = score
                best_value = value
        
        return best_value


class MarketRegimeDetector:
    """Advanced market regime detection using clustering and statistical analysis."""
    
    def __init__(self):
        self.regime_history: List[MarketRegime] = []
        self.scaler = StandardScaler() if SCIPY_AVAILABLE else None
    
    def detect_regime(
        self,
        recent_performance: List[Dict[str, Any]],
        lookback_days: int = 30,
    ) -> MarketRegime:
        """Detect current market regime."""
        if not recent_performance or len(recent_performance) < 7:
            return MarketRegime(
                regime_type="stable",
                confidence=0.5,
                characteristics={},
                expected_duration_days=7,
                recommended_strategy="conservative",
            )
        
        # Extract features
        cpms = [p.get("cpm", 0) for p in recent_performance if p.get("cpm", 0) > 0]
        ctrs = [p.get("ctr", 0) for p in recent_performance if p.get("ctr", 0) > 0]
        roas_list = [p.get("roas", 0) for p in recent_performance if p.get("roas", 0) > 0]
        
        if not cpms or not ctrs or not roas_list:
            return MarketRegime(
                regime_type="stable",
                confidence=0.5,
                characteristics={},
                expected_duration_days=7,
                recommended_strategy="conservative",
            )
        
        # Calculate statistics
        avg_cpm = np.mean(cpms)
        avg_ctr = np.mean(ctrs)
        avg_roas = np.mean(roas_list)
        
        cpm_std = np.std(cpms)
        ctr_std = np.std(ctrs)
        roas_std = np.std(roas_list)
        
        # Calculate trends
        cpm_trend = self._calculate_trend(cpms)
        ctr_trend = self._calculate_trend(ctrs)
        roas_trend = self._calculate_trend(roas_list)
        
        # Classify regime
        volatility = (cpm_std / avg_cpm + ctr_std / avg_ctr + roas_std / avg_roas) / 3
        
        if volatility > 0.3:
            regime_type = "volatile"
            recommended_strategy = "conservative"
            confidence = 0.7
        elif avg_roas > 2.5 and ctr_trend > 0.1:
            regime_type = "bull"
            recommended_strategy = "aggressive"
            confidence = 0.8
        elif avg_roas < 1.5 and ctr_trend < -0.1:
            regime_type = "bear"
            recommended_strategy = "defensive"
            confidence = 0.75
        elif abs(cpm_trend) < 0.05 and abs(ctr_trend) < 0.05:
            regime_type = "stable"
            recommended_strategy = "moderate"
            confidence = 0.85
        else:
            regime_type = "trending"
            recommended_strategy = "adaptive"
            confidence = 0.7
        
        # Estimate duration based on historical patterns
        expected_duration = self._estimate_regime_duration(regime_type)
        
        characteristics = {
            "avg_cpm": avg_cpm,
            "avg_ctr": avg_ctr,
            "avg_roas": avg_roas,
            "volatility": volatility,
            "cpm_trend": cpm_trend,
            "ctr_trend": ctr_trend,
            "roas_trend": roas_trend,
        }
        
        regime = MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            characteristics=characteristics,
            expected_duration_days=expected_duration,
            recommended_strategy=recommended_strategy,
        )
        
        self.regime_history.append(regime)
        return regime
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        
        # Linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(values)
        
        numerator = np.sum((x - x_mean) * (values - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope / y_mean if y_mean > 0 else 0.0  # Normalized slope
    
    def _estimate_regime_duration(self, regime_type: str) -> int:
        """Estimate expected duration of regime."""
        # Based on historical patterns
        duration_map = {
            "bull": 14,
            "bear": 10,
            "volatile": 5,
            "stable": 21,
            "trending": 7,
        }
        return duration_map.get(regime_type, 7)


class AdvancedTrendAnalyzer:
    """Advanced trend analysis with seasonality detection and forecasting."""
    
    def __init__(self):
        self.trend_history: List[TrendAnalysis] = []
    
    def analyze_trend(
        self,
        performance_data: List[Dict[str, Any]],
        metric: str = "roas",
        forecast_horizon: int = 7,
    ) -> TrendAnalysis:
        """Perform advanced trend analysis."""
        if not performance_data or len(performance_data) < 7:
            return TrendAnalysis(
                trend_direction="sideways",
                trend_strength=0.0,
                momentum=0.0,
                acceleration=0.0,
                volatility=0.0,
                seasonality_detected=False,
                regime=MarketRegime("stable", 0.5, {}, 7, "conservative"),
                forecast_horizon_days=forecast_horizon,
                forecast_values=[],
                confidence=0.0,
            )
        
        # Extract metric values
        values = [p.get(metric, 0) for p in performance_data if p.get(metric) is not None]
        dates = [p.get("date", datetime.now()) for p in performance_data]
        
        if len(values) < 7:
            return TrendAnalysis(
                trend_direction="sideways",
                trend_strength=0.0,
                momentum=0.0,
                acceleration=0.0,
                volatility=0.0,
                seasonality_detected=False,
                regime=MarketRegime("stable", 0.5, {}, 7, "conservative"),
                forecast_horizon_days=forecast_horizon,
                forecast_values=[],
                confidence=0.0,
            )
        
        # Calculate basic statistics
        mean_value = np.mean(values)
        std_value = np.std(values)
        volatility = std_value / mean_value if mean_value > 0 else 0.0
        
        # Calculate trend direction and strength
        n = len(values)
        x = np.arange(n)
        
        # Linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(values)
        
        numerator = np.sum((x - x_mean) * (values - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        slope = numerator / denominator if denominator > 0 else 0.0
        trend_direction = "up" if slope > 0 else "down" if slope < 0 else "sideways"
        
        # Trend strength (R-squared)
        y_pred = y_mean + slope * (x - x_mean)
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - y_mean) ** 2)
        trend_strength = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Momentum (rate of change)
        if len(values) >= 2:
            momentum = (values[-1] - values[-2]) / max(values[-2], 0.01)
        else:
            momentum = 0.0
        
        # Acceleration (change in momentum)
        if len(values) >= 3:
            prev_momentum = (values[-2] - values[-3]) / max(values[-3], 0.01)
            acceleration = momentum - prev_momentum
        else:
            acceleration = 0.0
        
        # Seasonality detection
        seasonality_detected = False
        if STATSMODELS_AVAILABLE and len(values) >= 14:
            try:
                # Try to detect weekly seasonality
                if len(values) >= 14:
                    # Simple autocorrelation test
                    autocorr = np.corrcoef(values[:-7], values[7:])[0, 1]
                    seasonality_detected = abs(autocorr) > 0.5
            except Exception:
                pass
        
        # Forecast future values
        forecast_values = []
        if len(values) >= 3:
            # Simple linear extrapolation
            for i in range(forecast_horizon):
                forecast = values[-1] + slope * (i + 1)
                forecast_values.append(max(0, forecast))
        
        # Detect market regime
        regime_detector = MarketRegimeDetector()
        regime = regime_detector.detect_regime(performance_data)
        
        # Calculate confidence
        confidence = min(1.0, len(values) / 30.0) * (1.0 - volatility * 0.5)
        
        trend_analysis = TrendAnalysis(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            momentum=momentum,
            acceleration=acceleration,
            volatility=volatility,
            seasonality_detected=seasonality_detected,
            regime=regime,
            forecast_horizon_days=forecast_horizon,
            forecast_values=forecast_values,
            confidence=confidence,
        )
        
        self.trend_history.append(trend_analysis)
        return trend_analysis


class AdaptiveThresholds:
    """ADVANCED Adaptive threshold management with Bayesian optimization."""
    
    def __init__(self, use_bayesian_optimization: bool = True):
        self.base_thresholds = {
            "ctr_floor": 0.008,
            "cpa_max": 40.0,
            "roas_min": 1.0,
            "cpm_max": 120.0,
        }
        self.adaptation_history: List[Dict[str, Any]] = []
        self.use_bayesian = use_bayesian_optimization and BAYESOPT_AVAILABLE
        
        if self.use_bayesian:
            self.bayesian_optimizer = BayesianThresholdOptimizer()
        else:
            self.bayesian_optimizer = None
    
    def adapt_thresholds(
        self,
        recent_performance: List[Dict[str, Any]],
        market_conditions: Optional[Dict[str, Any]] = None,
        optimize: bool = True,
    ) -> Dict[str, float]:
        """Adapt thresholds with Bayesian optimization."""
        if not recent_performance:
            return self.base_thresholds.copy()
        
        adapted = self.base_thresholds.copy()
        
        # Use Bayesian optimization if available
        if optimize and self.bayesian_optimizer and len(recent_performance) >= 10:
            # Optimize CTR threshold
            adapted["ctr_floor"] = self.bayesian_optimizer.optimize_threshold(
                "ctr_threshold",
                recent_performance,
                objective_metric="roas",
                bounds=(0.005, 0.02),
            )
            
            # Optimize CPA threshold
            adapted["cpa_max"] = self.bayesian_optimizer.optimize_threshold(
                "cpa_threshold",
                recent_performance,
                objective_metric="roas",
                bounds=(20.0, 60.0),
            )
            
            # Optimize ROAS threshold
            adapted["roas_min"] = self.bayesian_optimizer.optimize_threshold(
                "roas_threshold",
                recent_performance,
                objective_metric="purchases",
                bounds=(0.8, 2.0),
            )
        else:
            # Fallback to rule-based adaptation
            adapted = self._rule_based_adaptation(recent_performance, market_conditions)
        
        # Record adaptation
        self.adaptation_history.append({
            "timestamp": datetime.now(),
            "thresholds": adapted.copy(),
            "recent_performance_count": len(recent_performance),
            "optimization_method": "bayesian" if optimize and self.use_bayesian else "rule_based",
        })
        
        return adapted
    
    def _rule_based_adaptation(
        self,
        recent_performance: List[Dict[str, Any]],
        market_conditions: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Rule-based threshold adaptation (fallback)."""
        adapted = self.base_thresholds.copy()
        
        # Calculate recent averages
        ctrs = [p.get("ctr", 0) for p in recent_performance if p.get("ctr", 0) > 0]
        cpas = [p.get("cpa", 0) for p in recent_performance if p.get("cpa", 0) > 0 and p.get("cpa", 0) < 1000]
        roas_list = [p.get("roas", 0) for p in recent_performance if p.get("roas", 0) > 0]
        cpms = [p.get("cpm", 0) for p in recent_performance if p.get("cpm", 0) > 0]
        
        # Adapt CTR floor
        if ctrs:
            avg_ctr = statistics.mean(ctrs)
            if avg_ctr > self.base_thresholds["ctr_floor"] * 1.2:
                adapted["ctr_floor"] = avg_ctr * 0.8
            elif avg_ctr < self.base_thresholds["ctr_floor"] * 0.8:
                adapted["ctr_floor"] = max(avg_ctr * 0.9, 0.005)
        
        # Adapt CPA max
        if cpas:
            avg_cpa = statistics.mean(cpas)
            if avg_cpa < self.base_thresholds["cpa_max"] * 0.8:
                adapted["cpa_max"] = avg_cpa * 1.2
            elif avg_cpa > self.base_thresholds["cpa_max"] * 1.2:
                adapted["cpa_max"] = min(avg_cpa * 1.1, 60.0)
        
        # Adapt ROAS min
        if roas_list:
            avg_roas = statistics.mean(roas_list)
            if avg_roas > self.base_thresholds["roas_min"] * 1.2:
                adapted["roas_min"] = avg_roas * 0.9
            elif avg_roas < self.base_thresholds["roas_min"] * 0.8:
                adapted["roas_min"] = max(avg_roas * 0.95, 0.8)
        
        # Adapt CPM max
        if cpms:
            avg_cpm = statistics.mean(cpms)
            if avg_cpm > self.base_thresholds["cpm_max"] * 1.2:
                adapted["cpm_max"] = avg_cpm * 1.1
            elif avg_cpm < self.base_thresholds["cpm_max"] * 0.8:
                adapted["cpm_max"] = max(avg_cpm * 1.2, 100.0)
        
        return adapted
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds."""
        if self.adaptation_history:
            return self.adaptation_history[-1]["thresholds"]
        return self.base_thresholds.copy()


class MarketConditionDetector:
    """ADVANCED Market condition detection with regime analysis."""
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.trend_analyzer = AdvancedTrendAnalyzer()
    
    def detect_conditions(
        self,
        recent_performance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect current market conditions with advanced analysis."""
        if not recent_performance:
            return {
                "condition": "normal",
                "competition_level": "medium",
                "regime": "stable",
            }
        
        # Detect regime
        regime = self.regime_detector.detect_regime(recent_performance)
        
        # Analyze trends
        trend_analysis = self.trend_analyzer.analyze_trend(recent_performance, metric="roas")
        
        # Calculate metrics
        cpms = [p.get("cpm", 0) for p in recent_performance if p.get("cpm", 0) > 0]
        ctrs = [p.get("ctr", 0) for p in recent_performance if p.get("ctr", 0) > 0]
        
        if not cpms or not ctrs:
            return {
                "condition": "normal",
                "competition_level": "medium",
                "regime": regime.regime_type,
                "regime_confidence": regime.confidence,
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
            "regime": regime.regime_type,
            "regime_confidence": regime.confidence,
            "recommended_strategy": regime.recommended_strategy,
            "trend_direction": trend_analysis.trend_direction,
            "trend_strength": trend_analysis.trend_strength,
            "momentum": trend_analysis.momentum,
            "volatility": trend_analysis.volatility,
        }


class DynamicRuleAdjuster:
    """ADVANCED Dynamically adjusts rules with Bayesian optimization and regime awareness."""
    
    def __init__(self, use_bayesian: bool = True):
        self.adaptive_thresholds = AdaptiveThresholds(use_bayesian_optimization=use_bayesian)
        self.market_detector = MarketConditionDetector()
        self.trend_analyzer = AdvancedTrendAnalyzer()
    
    def adjust_rules(
        self,
        current_rules: Dict[str, Any],
        recent_performance: List[Dict[str, Any]],
        optimize: bool = True,
    ) -> Dict[str, Any]:
        """Adjust rules with advanced analysis."""
        # Detect market conditions and regime
        market_conditions = self.market_detector.detect_conditions(recent_performance)
        
        # Analyze trends
        trend_analysis = self.trend_analyzer.analyze_trend(recent_performance, metric="roas")
        
        # Adapt thresholds with Bayesian optimization
        adapted_thresholds = self.adaptive_thresholds.adapt_thresholds(
            recent_performance,
            market_conditions,
            optimize=optimize,
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
            "trend_analysis": {
                "direction": trend_analysis.trend_direction,
                "strength": trend_analysis.trend_strength,
                "momentum": trend_analysis.momentum,
                "regime": trend_analysis.regime.regime_type,
            },
        }


def create_performance_adaptation(**kwargs) -> DynamicRuleAdjuster:
    """Create advanced performance adaptation system."""
    return DynamicRuleAdjuster(**kwargs)


__all__ = [
    "DynamicRuleAdjuster",
    "AdaptiveThresholds",
    "MarketConditionDetector",
    "MarketRegimeDetector",
    "AdvancedTrendAnalyzer",
    "BayesianThresholdOptimizer",
    "create_performance_adaptation",
]
