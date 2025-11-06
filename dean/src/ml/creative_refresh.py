"""
ADVANCED Intelligent Creative Refresh Strategy
Predictive fatigue models, optimal refresh timing with reinforcement learning, creative lifecycle management
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

# Advanced imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from ml.ml_advanced_features import QLearningAgent, RLState, RLAction
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    QLearningAgent = None
    RLState = None
    RLAction = None

try:
    from scipy import stats, optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FatiguePrediction:
    """Fatigue prediction with confidence intervals."""
    will_fatigue: bool
    hours_until_fatigue: float
    confidence: float
    fatigue_severity: float  # 0-1, how severe the fatigue will be
    predicted_performance_drop: float  # Expected performance drop percentage
    optimal_refresh_time: datetime
    confidence_interval_lower: float
    confidence_interval_upper: float


@dataclass
class CreativeLifecycle:
    """Creative lifecycle stage tracking."""
    creative_id: str
    stage: str  # "launch", "growth", "peak", "decline", "fatigue"
    stage_start: datetime
    expected_duration_hours: float
    performance_trajectory: List[float]
    next_stage_prediction: str
    refresh_recommendation: str  # "hold", "refresh_soon", "refresh_now", "pause"


class MLFatiguePredictor:
    """ML-based fatigue prediction using ensemble models."""
    
    def __init__(self):
        self.ctr_model = None
        self.roas_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.trained = False
    
    def train_models(self, historical_data: List[Dict[str, Any]]):
        """Train ML models on historical fatigue patterns."""
        if not SKLEARN_AVAILABLE or len(historical_data) < 20:
            return False
        
        try:
            # Prepare features and targets
            X = []
            y_ctr = []
            y_roas = []
            
            for i in range(len(historical_data) - 1):
                current = historical_data[i]
                next_data = historical_data[i + 1]
                
                # Features: current performance metrics
                features = [
                    current.get("ctr", 0),
                    current.get("roas", 0),
                    current.get("cpa", 0) / 10,  # Normalize
                    current.get("spend", 0) / 100,  # Normalize
                    current.get("impressions", 0) / 10000,  # Normalize
                    current.get("clicks", 0) / 100,  # Normalize
                    current.get("age_hours", 0) / 24,  # Normalize to days
                    current.get("frequency", 0),
                ]
                
                # Add trend features
                if i >= 2:
                    prev_ctr = historical_data[i-1].get("ctr", 0)
                    prev_roas = historical_data[i-1].get("roas", 0)
                    features.append(current.get("ctr", 0) - prev_ctr)  # CTR momentum
                    features.append(current.get("roas", 0) - prev_roas)  # ROAS momentum
                else:
                    features.extend([0, 0])
                
                X.append(features)
                
                # Targets: next period performance
                y_ctr.append(next_data.get("ctr", 0))
                y_roas.append(next_data.get("roas", 0))
            
            if len(X) < 10:
                return False
            
            X = np.array(X)
            y_ctr = np.array(y_ctr)
            y_roas = np.array(y_roas)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train CTR model
            self.ctr_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            )
            self.ctr_model.fit(X_scaled, y_ctr)
            
            # Train ROAS model
            self.roas_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
            )
            self.roas_model.fit(X_scaled, y_roas)
            
            self.trained = True
            logger.info("ML fatigue predictor trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train ML fatigue predictor: {e}")
            return False
    
    def predict_fatigue(
        self,
        current_performance: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
        lookahead_hours: int = 24,
    ) -> FatiguePrediction:
        """Predict fatigue using ML models."""
        if not self.trained or len(historical_performance) < 3:
            # Fallback to simple prediction
            return self._simple_fatigue_prediction(historical_performance)
        
        try:
            # Prepare features
            features = [
                current_performance.get("ctr", 0),
                current_performance.get("roas", 0),
                current_performance.get("cpa", 0) / 10,
                current_performance.get("spend", 0) / 100,
                current_performance.get("impressions", 0) / 10000,
                current_performance.get("clicks", 0) / 100,
                current_performance.get("age_hours", 0) / 24,
                current_performance.get("frequency", 0),
            ]
            
            # Add trend features
            if len(historical_performance) >= 2:
                prev = historical_performance[-2]
                features.append(current_performance.get("ctr", 0) - prev.get("ctr", 0))
                features.append(current_performance.get("roas", 0) - prev.get("roas", 0))
            else:
                features.extend([0, 0])
            
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            # Predict future CTR and ROAS
            predicted_ctr = self.ctr_model.predict(X_scaled)[0]
            predicted_roas = self.roas_model.predict(X_scaled)[0]
            
            # Get prediction intervals (using model variance)
            if hasattr(self.ctr_model, 'estimators_'):
                ctr_predictions = [est.predict(X_scaled)[0] for est in self.ctr_model.estimators_]
                ctr_std = np.std(ctr_predictions)
                ctr_lower = predicted_ctr - 1.96 * ctr_std
                ctr_upper = predicted_ctr + 1.96 * ctr_std
            else:
                ctr_lower = predicted_ctr * 0.8
                ctr_upper = predicted_ctr * 1.2
            
            if hasattr(self.roas_model, 'estimators_'):
                roas_predictions = [est.predict(X_scaled)[0] for est in self.roas_model.estimators_]
                roas_std = np.std(roas_predictions)
                roas_lower = predicted_roas - 1.96 * roas_std
                roas_upper = predicted_roas + 1.96 * roas_std
            else:
                roas_lower = predicted_roas * 0.8
                roas_upper = predicted_roas * 1.2
            
            # Calculate performance drop
            current_ctr = current_performance.get("ctr", 0)
            current_roas = current_performance.get("roas", 0)
            
            ctr_drop = (current_ctr - predicted_ctr) / current_ctr if current_ctr > 0 else 0
            roas_drop = (current_roas - predicted_roas) / current_roas if current_roas > 0 else 0
            
            avg_drop = (ctr_drop + roas_drop) / 2
            
            # Determine if fatigue will occur
            will_fatigue = avg_drop > 0.15  # 15% drop threshold
            
            # Estimate hours until fatigue
            if will_fatigue and avg_drop > 0:
                # Linear extrapolation
                hours_until_fatigue = lookahead_hours * (0.15 / avg_drop)
            else:
                hours_until_fatigue = 999
            
            # Calculate confidence
            confidence = min(0.95, len(historical_performance) / 20.0)
            
            # Fatigue severity
            fatigue_severity = min(1.0, avg_drop / 0.5)  # Normalize to 0-1
            
            # Optimal refresh time (before fatigue becomes severe)
            optimal_refresh_time = datetime.now() + timedelta(hours=hours_until_fatigue * 0.7)
            
            return FatiguePrediction(
                will_fatigue=will_fatigue,
                hours_until_fatigue=hours_until_fatigue,
                confidence=confidence,
                fatigue_severity=fatigue_severity,
                predicted_performance_drop=avg_drop,
                optimal_refresh_time=optimal_refresh_time,
                confidence_interval_lower=min(ctr_lower, roas_lower),
                confidence_interval_upper=max(ctr_upper, roas_upper),
            )
            
        except Exception as e:
            logger.error(f"ML fatigue prediction failed: {e}")
            return self._simple_fatigue_prediction(historical_performance)
    
    def _simple_fatigue_prediction(
        self,
        historical_performance: List[Dict[str, Any]],
    ) -> FatiguePrediction:
        """Simple fatigue prediction (fallback)."""
        if len(historical_performance) < 3:
            return FatiguePrediction(
                will_fatigue=False,
                hours_until_fatigue=999,
                confidence=0.0,
                fatigue_severity=0.0,
                predicted_performance_drop=0.0,
                optimal_refresh_time=datetime.now() + timedelta(days=7),
                confidence_interval_lower=0.0,
                confidence_interval_upper=0.0,
            )
        
        # Simple trend analysis
        ctr_values = [p.get("ctr", 0) for p in historical_performance]
        roas_values = [p.get("roas", 0) for p in historical_performance]
        
        if len(ctr_values) >= 2:
            ctr_slope = (ctr_values[-1] - ctr_values[0]) / len(ctr_values)
            roas_slope = (roas_values[-1] - roas_values[0]) / len(roas_values)
            
            will_fatigue = ctr_slope < -0.001 and roas_slope < -0.1
            hours_until_fatigue = 24 if will_fatigue else 999
            confidence = 0.6
        else:
            will_fatigue = False
            hours_until_fatigue = 999
            confidence = 0.0
        
        return FatiguePrediction(
            will_fatigue=will_fatigue,
            hours_until_fatigue=hours_until_fatigue,
            confidence=confidence,
            fatigue_severity=0.3 if will_fatigue else 0.0,
            predicted_performance_drop=0.2 if will_fatigue else 0.0,
            optimal_refresh_time=datetime.now() + timedelta(hours=hours_until_fatigue * 0.7),
            confidence_interval_lower=0.0,
            confidence_interval_upper=0.0,
        )


class RLRefreshTimingOptimizer:
    """Reinforcement learning for optimal refresh timing."""
    
    def __init__(self):
        self.rl_agent = None
        if RL_AVAILABLE:
            self.rl_agent = QLearningAgent(
                learning_rate=0.1,
                discount_factor=0.95,
                exploration_rate=0.1,
            )
    
    def get_optimal_refresh_time(
        self,
        creative: Dict[str, Any],
        fatigue_prediction: FatiguePrediction,
    ) -> datetime:
        """Determine optimal refresh time using RL."""
        if not self.rl_agent:
            # Fallback to prediction-based timing
            return fatigue_prediction.optimal_refresh_time
        
        try:
            # Convert to RL state
            performance = creative.get("performance", {})
            state = RLState(
                cpa=performance.get("cpa", 0),
                roas=performance.get("roas", 0),
                ctr=performance.get("ctr", 0),
                spend=performance.get("spend", 0),
                days_active=creative.get("age_hours", 0) / 24,
                stage="asc_plus",
            )
            
            # Get RL action
            rl_action = self.rl_agent.get_action(state, explore=False)
            
            # Adjust timing based on RL recommendation
            if rl_action.action_type == "refresh":
                # RL recommends refresh - use prediction time
                return fatigue_prediction.optimal_refresh_time
            elif rl_action.action_type == "hold":
                # RL recommends holding - delay refresh
                return fatigue_prediction.optimal_refresh_time + timedelta(hours=12)
            else:
                # Default to prediction
                return fatigue_prediction.optimal_refresh_time
                
        except Exception as e:
            logger.error(f"RL refresh timing failed: {e}")
            return fatigue_prediction.optimal_refresh_time
    
    def learn_from_refresh_outcome(
        self,
        creative: Dict[str, Any],
        refresh_time: datetime,
        outcome: Dict[str, Any],
    ):
        """Learn from refresh outcome to improve timing."""
        if not self.rl_agent:
            return
        
        try:
            # Calculate reward based on outcome
            reward = 0.0
            if outcome.get("improved_performance", False):
                reward = 1.0
            elif outcome.get("maintained_performance", False):
                reward = 0.5
            else:
                reward = -0.5
            
            # Update RL agent
            performance = creative.get("performance", {})
            state = RLState(
                cpa=performance.get("cpa", 0),
                roas=performance.get("roas", 0),
                ctr=performance.get("ctr", 0),
                spend=performance.get("spend", 0),
                days_active=creative.get("age_hours", 0) / 24,
                stage="asc_plus",
            )
            
            next_state = RLState(
                cpa=outcome.get("cpa", performance.get("cpa", 0)),
                roas=outcome.get("roas", performance.get("roas", 0)),
                ctr=outcome.get("ctr", performance.get("ctr", 0)),
                spend=outcome.get("spend", performance.get("spend", 0)),
                days_active=(creative.get("age_hours", 0) / 24) + 1,
                stage="asc_plus",
            )
            
            self.rl_agent.update_q_value(state, "refresh", reward, next_state)
            
        except Exception as e:
            logger.error(f"RL learning failed: {e}")


class CreativeLifecycleManager:
    """Manages creative lifecycle stages and transitions."""
    
    def __init__(self):
        self.lifecycles: Dict[str, CreativeLifecycle] = {}
    
    def update_lifecycle(
        self,
        creative_id: str,
        performance: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
    ) -> CreativeLifecycle:
        """Update and predict creative lifecycle stage."""
        # Determine current stage
        age_hours = performance.get("age_hours", 0)
        roas = performance.get("roas", 0)
        ctr = performance.get("ctr", 0)
        spend = performance.get("spend", 0)
        
        # Calculate performance trajectory
        trajectory = [p.get("roas", 0) for p in historical_performance[-7:]] if len(historical_performance) >= 7 else [roas]
        
        # Determine stage
        if age_hours < 24:
            stage = "launch"
            expected_duration = 24 - age_hours
        elif roas > 2.0 and ctr > 0.01 and spend < 50:
            stage = "growth"
            expected_duration = 48
        elif roas > 2.5 and spend > 50:
            stage = "peak"
            expected_duration = 72
        elif roas < 1.5 or ctr < 0.005:
            stage = "decline"
            expected_duration = 24
        elif len(trajectory) >= 3 and trajectory[-1] < trajectory[0] * 0.8:
            stage = "fatigue"
            expected_duration = 12
        else:
            stage = "growth"
            expected_duration = 48
        
        # Predict next stage
        if stage == "launch":
            next_stage = "growth" if roas > 1.5 else "decline"
        elif stage == "growth":
            next_stage = "peak" if roas > 2.5 else "decline"
        elif stage == "peak":
            next_stage = "decline"
        elif stage == "decline":
            next_stage = "fatigue"
        else:
            next_stage = "fatigue"
        
        # Refresh recommendation
        if stage == "fatigue":
            refresh_rec = "refresh_now"
        elif stage == "decline" and age_hours > 48:
            refresh_rec = "refresh_soon"
        elif stage == "peak" and age_hours > 72:
            refresh_rec = "refresh_soon"
        else:
            refresh_rec = "hold"
        
        lifecycle = CreativeLifecycle(
            creative_id=creative_id,
            stage=stage,
            stage_start=datetime.now() - timedelta(hours=age_hours),
            expected_duration_hours=expected_duration,
            performance_trajectory=trajectory,
            next_stage_prediction=next_stage,
            refresh_recommendation=refresh_rec,
        )
        
        self.lifecycles[creative_id] = lifecycle
        return lifecycle


class CreativeRefreshManager:
    """ADVANCED Manages intelligent creative refresh with ML and RL."""
    
    def __init__(
        self,
        fatigue_threshold: float = 0.25,
        min_age_hours: int = 12,
        refresh_buffer: int = 2,
        predictive_fatigue_enabled: bool = True,
        staggered_refresh_enabled: bool = True,
        use_ml_predictions: bool = True,
        use_rl_timing: bool = True,
    ):
        self.fatigue_threshold = fatigue_threshold
        self.min_age_hours = min_age_hours
        self.refresh_buffer = refresh_buffer
        self.predictive_fatigue_enabled = predictive_fatigue_enabled
        self.staggered_refresh_enabled = staggered_refresh_enabled
        self.use_ml_predictions = use_ml_predictions and SKLEARN_AVAILABLE
        self.use_rl_timing = use_rl_timing and RL_AVAILABLE
        
        # Advanced components
        self.ml_predictor = MLFatiguePredictor() if self.use_ml_predictions else None
        self.rl_optimizer = RLRefreshTimingOptimizer() if self.use_rl_timing else None
        self.lifecycle_manager = CreativeLifecycleManager()
        
        self.refresh_schedule: Dict[str, datetime] = {}
        self.refresh_history: List[Dict[str, Any]] = []
    
    def train_fatigue_models(self, historical_data: List[Dict[str, Any]]):
        """Train ML fatigue prediction models."""
        if self.ml_predictor:
            return self.ml_predictor.train_models(historical_data)
        return False
    
    def should_refresh_creative(
        self,
        creative: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Determine if a creative should be refreshed with advanced analysis."""
        performance = creative.get("performance", {})
        created_at = creative.get("created_at")
        creative_id = creative.get("creative_id") or creative.get("ad_id")
        
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
        
        # Update lifecycle
        performance["age_hours"] = age_hours
        lifecycle = self.lifecycle_manager.update_lifecycle(
            creative_id,
            performance,
            historical_performance,
        )
        
        # Check lifecycle recommendation
        if lifecycle.refresh_recommendation == "refresh_now":
            return {
                "should_refresh": True,
                "reason": f"lifecycle_{lifecycle.stage}",
                "age_hours": age_hours,
                "lifecycle_stage": lifecycle.stage,
            }
        
        # ML-based fatigue prediction
        if self.predictive_fatigue_enabled and len(historical_performance) >= 5:
            if self.ml_predictor and self.ml_predictor.trained:
                fatigue_pred = self.ml_predictor.predict_fatigue(
                    performance,
                    historical_performance,
                )
            else:
                fatigue_pred = self.ml_predictor._simple_fatigue_prediction(historical_performance) if self.ml_predictor else None
            
            if fatigue_pred and fatigue_pred.will_fatigue:
                # Use RL to optimize timing
                if self.rl_optimizer:
                    optimal_time = self.rl_optimizer.get_optimal_refresh_time(creative, fatigue_pred)
                    hours_until_optimal = (optimal_time - datetime.now()).total_seconds() / 3600.0
                    
                    if hours_until_optimal <= 6:  # Within 6 hours
                        return {
                            "should_refresh": True,
                            "reason": "predicted_fatigue_optimal_timing",
                            "predicted_fatigue_in_hours": fatigue_pred.hours_until_fatigue,
                            "confidence": fatigue_pred.confidence,
                            "fatigue_severity": fatigue_pred.fatigue_severity,
                            "age_hours": age_hours,
                            "optimal_refresh_time": optimal_time.isoformat(),
                        }
                else:
                    if fatigue_pred.hours_until_fatigue < 24:
                        return {
                            "should_refresh": True,
                            "reason": "predicted_fatigue",
                            "predicted_fatigue_in_hours": fatigue_pred.hours_until_fatigue,
                            "confidence": fatigue_pred.confidence,
                            "age_hours": age_hours,
                        }
        
        # Traditional fatigue detection
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
        
        # Performance thresholds
        roas = performance.get("roas", 0.0)
        ctr = performance.get("ctr", 0.0)
        spend = performance.get("spend", 0.0)
        
        if spend >= 40.0 and (roas < 0.8 or ctr < 0.005):
            return {
                "should_refresh": True,
                "reason": "low_performance",
                "roas": roas,
                "ctr": ctr,
                "age_hours": age_hours,
            }
        
        return {
            "should_refresh": False,
            "reason": "performing_well",
            "age_hours": age_hours,
            "lifecycle_stage": lifecycle.stage,
        }
    
    def plan_refresh_schedule(
        self,
        creatives: List[Dict[str, Any]],
        target_count: int = 10,
    ) -> Dict[str, Any]:
        """Plan refresh schedule with advanced optimization."""
        refresh_needed = []
        refresh_soon = []
        scheduled_refreshes = []
        
        for creative in creatives:
            creative_id = creative.get("creative_id") or creative.get("ad_id")
            performance = creative.get("performance", {})
            historical = creative.get("historical_performance", [])
            
            refresh_decision = self.should_refresh_creative(creative, historical)
            
            if refresh_decision["should_refresh"]:
                priority = "high" if refresh_decision.get("fatigue_severity", 0) > 0.5 else "medium"
                
                # Get optimal refresh time
                optimal_time = datetime.now()
                if "optimal_refresh_time" in refresh_decision:
                    optimal_time = datetime.fromisoformat(refresh_decision["optimal_refresh_time"])
                
                # Staggered refresh scheduling
                if self.staggered_refresh_enabled and len(refresh_needed) > 0:
                    delay_hours = len(refresh_needed) * 2
                    scheduled_time = max(optimal_time, datetime.now() + timedelta(hours=delay_hours))
                    self.refresh_schedule[creative_id] = scheduled_time
                    
                    scheduled_refreshes.append({
                        "creative_id": creative_id,
                        "reason": refresh_decision["reason"],
                        "priority": priority,
                        "scheduled_for": scheduled_time.isoformat(),
                        "delay_hours": delay_hours,
                        "optimal_time": optimal_time.isoformat(),
                    })
                else:
                    refresh_needed.append({
                        "creative_id": creative_id,
                        "reason": refresh_decision["reason"],
                        "priority": priority,
                        "optimal_time": optimal_time.isoformat(),
                    })
            elif refresh_decision.get("age_hours", 0) > 48:
                refresh_soon.append({
                    "creative_id": creative_id,
                    "reason": "preventive_refresh",
                    "priority": "low",
                })
        
        current_count = len(creatives)
        needed_count = max(0, target_count - current_count)
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
    
    def record_refresh_outcome(
        self,
        creative_id: str,
        outcome: Dict[str, Any],
    ):
        """Record refresh outcome for RL learning."""
        if self.rl_optimizer:
            creative = {"creative_id": creative_id, "performance": outcome}
            self.rl_optimizer.learn_from_refresh_outcome(
                creative,
                datetime.now(),
                outcome,
            )
        
        self.refresh_history.append({
            "creative_id": creative_id,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
        })
    
    def maintain_diversity(
        self,
        existing_creatives: List[Dict[str, Any]],
        new_creative: Dict[str, Any],
        min_similarity: float = 0.7,
    ) -> bool:
        """Check if new creative maintains diversity."""
        new_prompt = new_creative.get("image_prompt", "")
        new_text = new_creative.get("text_overlay", "")
        
        for existing in existing_creatives:
            existing_prompt = existing.get("image_prompt", "")
            existing_text = existing.get("text_overlay", "")
            
            prompt_overlap = len(set(new_prompt.lower().split()) & set(existing_prompt.lower().split()))
            text_match = new_text == existing_text
            
            if prompt_overlap > 5 or text_match:
                return False
        
        return True


def create_creative_refresh_manager(**kwargs) -> CreativeRefreshManager:
    """Create advanced creative refresh manager."""
    return CreativeRefreshManager(**kwargs)


__all__ = [
    "CreativeRefreshManager",
    "MLFatiguePredictor",
    "RLRefreshTimingOptimizer",
    "CreativeLifecycleManager",
    "FatiguePrediction",
    "CreativeLifecycle",
    "create_creative_refresh_manager",
]
