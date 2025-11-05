"""
Predictive Performance Modeling
Predicts creative performance before and shortly after launch
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using simple prediction models")


class PredictivePerformanceModel:
    """Predicts creative performance using ML models."""
    
    def __init__(self, supabase_client=None):
        self.supabase_client = supabase_client
        self.models = {}
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
    
    def _extract_features(
        self,
        creative_data: Dict[str, Any],
        performance_data: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Extract features from creative and performance data."""
        features = []
        
        # Creative features
        image_prompt = creative_data.get("image_prompt", "")
        text_overlay = creative_data.get("text_overlay", "")
        ad_copy = creative_data.get("ad_copy", {})
        
        # Text length features
        features.append(len(image_prompt))
        features.append(len(text_overlay))
        features.append(len(ad_copy.get("headline", "")))
        features.append(len(ad_copy.get("primary_text", "")))
        
        # Keyword features (simplified)
        keywords = ["premium", "luxury", "sophisticated", "calm", "confidence", "discipline"]
        for keyword in keywords:
            count = (image_prompt + text_overlay).lower().count(keyword)
            features.append(count)
        
        # Performance features (if available)
        if performance_data:
            features.append(performance_data.get("impressions", 0))
            features.append(performance_data.get("clicks", 0))
            features.append(performance_data.get("spend", 0.0))
            features.append(performance_data.get("ctr", 0.0))
        else:
            features.extend([0, 0, 0.0, 0.0])
        
        # Time features
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        features.append(hour)
        features.append(day_of_week)
        
        return np.array(features, dtype=np.float32)
    
    def train(
        self,
        training_data: List[Dict[str, Any]],
        target_metric: str = "roas",
    ):
        """Train the predictive model."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using simple prediction")
            self.is_trained = True
            return
        
        if not training_data:
            logger.warning("No training data provided")
            return
        
        # Extract features and targets
        X = []
        y = []
        
        for item in training_data:
            creative_data = item.get("creative_data", {})
            performance_data = item.get("performance_data", {})
            
            features = self._extract_features(creative_data, performance_data)
            X.append(features)
            
            target = performance_data.get(target_metric, 0.0)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 10:
            logger.warning(f"Not enough training data: {len(X)} samples")
            return
        
        # Scale features
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Train multiple models
        try:
            # Gradient Boosting (best for non-linear relationships)
            self.models["gradient_boosting"] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            self.models["gradient_boosting"].fit(X, y)
            
            # Random Forest (good for feature importance)
            self.models["random_forest"] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )
            self.models["random_forest"].fit(X, y)
            
            # Ridge Regression (good baseline)
            self.models["ridge"] = Ridge(alpha=1.0)
            self.models["ridge"].fit(X, y)
            
            self.is_trained = True
            logger.info(f"✅ Trained predictive models on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            self.is_trained = False
    
    def predict(
        self,
        creative_data: Dict[str, Any],
        performance_data: Optional[Dict[str, Any]] = None,
        target_metric: str = "roas",
    ) -> Dict[str, Any]:
        """Predict performance for a creative."""
        if not self.is_trained:
            # Return default prediction
            return {
                "predicted_value": 1.0,
                "confidence": 0.0,
                "model": "untrained",
            }
        
        # Extract features
        features = self._extract_features(creative_data, performance_data)
        features = features.reshape(1, -1)
        
        # Scale if scaler is available
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            try:
                pred = model.predict(features)[0]
                predictions[model_name] = float(pred)
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
        
        if not predictions:
            return {
                "predicted_value": 1.0,
                "confidence": 0.0,
                "model": "error",
            }
        
        # Ensemble prediction (average)
        avg_prediction = np.mean(list(predictions.values()))
        
        # Calculate confidence (inverse of variance)
        variance = np.var(list(predictions.values()))
        confidence = 1.0 / (1.0 + variance)
        
        return {
            "predicted_value": float(avg_prediction),
            "confidence": float(min(confidence, 1.0)),
            "model": "ensemble",
            "individual_predictions": predictions,
        }
    
    def predict_early_performance(
        self,
        creative_data: Dict[str, Any],
        early_performance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Predict final performance based on early signals (first 2 hours)."""
        # Use early performance to predict final performance
        impressions = early_performance.get("impressions", 0)
        clicks = early_performance.get("clicks", 0)
        spend = early_performance.get("spend", 0.0)
        
        if impressions == 0:
            return {
                "predicted_roas": 1.0,
                "predicted_ctr": 0.01,
                "confidence": 0.0,
            }
        
        # Early CTR
        early_ctr = clicks / impressions if impressions > 0 else 0.0
        
        # Simple heuristics for early prediction
        # If CTR is high early, likely to maintain
        # If CTR is very low early, likely to stay low
        
        predicted_ctr = early_ctr * 0.9  # Slight decay assumption
        
        # Predict ROAS based on early CTR (correlation)
        # High CTR often correlates with good ROAS
        predicted_roas = 1.0 + (predicted_ctr - 0.01) * 50  # Rough correlation
        
        # Confidence based on data volume
        confidence = min(impressions / 1000.0, 1.0)  # More impressions = more confidence
        
        return {
            "predicted_roas": float(max(predicted_roas, 0.1)),
            "predicted_ctr": float(predicted_ctr),
            "confidence": float(confidence),
            "early_ctr": float(early_ctr),
        }


class EarlySignalDetector:
    """Detects early signals for creative performance."""
    
    def __init__(self):
        self.signal_thresholds = {
            "high_performer": {
                "ctr": 0.015,
                "cpm": 80.0,
                "spend": 5.0,
            },
            "low_performer": {
                "ctr": 0.003,
                "cpm": 150.0,
                "spend": 10.0,
            },
        }
    
    def detect_signal(
        self,
        performance_data: Dict[str, Any],
        min_spend: float = 5.0,
    ) -> Dict[str, Any]:
        """Detect early performance signal."""
        spend = performance_data.get("spend", 0.0)
        impressions = performance_data.get("impressions", 0)
        clicks = performance_data.get("clicks", 0)
        
        if spend < min_spend:
            return {
                "signal": "insufficient_data",
                "confidence": 0.0,
            }
        
        ctr = clicks / impressions if impressions > 0 else 0.0
        cpm = (spend / impressions * 1000) if impressions > 0 else 0.0
        
        # Check for high performer signals
        high_thresholds = self.signal_thresholds["high_performer"]
        if (
            ctr >= high_thresholds["ctr"] and
            cpm <= high_thresholds["cpm"] and
            spend >= high_thresholds["spend"]
        ):
            return {
                "signal": "high_performer",
                "confidence": 0.7,
                "ctr": ctr,
                "cpm": cpm,
            }
        
        # Check for low performer signals
        low_thresholds = self.signal_thresholds["low_performer"]
        if (
            ctr <= low_thresholds["ctr"] and
            cpm >= low_thresholds["cpm"] and
            spend >= low_thresholds["spend"]
        ):
            return {
                "signal": "low_performer",
                "confidence": 0.8,
                "ctr": ctr,
                "cpm": cpm,
            }
        
        return {
            "signal": "neutral",
            "confidence": 0.5,
            "ctr": ctr,
            "cpm": cpm,
        }


def create_predictive_model(supabase_client=None) -> PredictivePerformanceModel:
    """Create a predictive performance model."""
    return PredictivePerformanceModel(supabase_client=supabase_client)


def create_early_signal_detector() -> EarlySignalDetector:
    """Create an early signal detector."""
    return EarlySignalDetector()


__all__ = [
    "PredictivePerformanceModel",
    "EarlySignalDetector",
    "create_predictive_model",
    "create_early_signal_detector",
]

