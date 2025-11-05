"""
Model Training & Management Pipeline
Automated retraining, versioning, monitoring
"""

from __future__ import annotations

import logging
import pickle
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


from dataclasses import dataclass

@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    model_type: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    training_data_size: int
    features: List[str]
    model_path: Optional[str] = None


class ModelRegistry:
    """Model registry for versioning."""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, List[ModelVersion]] = {}
    
    def register_model(
        self,
        model_type: str,
        model: Any,
        performance_metrics: Dict[str, float],
        training_data_size: int,
        features: List[str],
    ) -> str:
        """Register a new model version."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_filename = f"{model_type}_{version}.pkl"
        model_path = self.registry_path / model_filename
        joblib.dump(model, model_path)
        
        model_version = ModelVersion(
            version=version,
            model_type=model_type,
            created_at=datetime.now(),
            performance_metrics=performance_metrics,
            training_data_size=training_data_size,
            features=features,
            model_path=str(model_path),
        )
        
        if model_type not in self.models:
            self.models[model_type] = []
        
        self.models[model_type].append(model_version)
        
        logger.info(f"Registered model {model_type} version {version}")
        return version
    
    def get_latest_model(self, model_type: str) -> Optional[ModelVersion]:
        """Get latest model version."""
        if model_type not in self.models or not self.models[model_type]:
            return None
        
        versions = self.models[model_type]
        return max(versions, key=lambda v: v.created_at)
    
    def load_model(self, model_type: str, version: Optional[str] = None) -> Optional[Any]:
        """Load model from registry."""
        if version:
            # Load specific version
            for mv in self.models.get(model_type, []):
                if mv.version == version:
                    return joblib.load(mv.model_path)
        else:
            # Load latest
            latest = self.get_latest_model(model_type)
            if latest:
                return joblib.load(latest.model_path)
        
        return None


class ModelTrainingPipeline:
    """Automated model training pipeline."""
    
    def __init__(
        self,
        registry: ModelRegistry,
        supabase_client=None,
    ):
        self.registry = registry
        self.supabase_client = supabase_client
        self.training_history: List[Dict[str, Any]] = []
    
    def train_model(
        self,
        model_type: str,
        training_data: List[Dict[str, Any]],
        target_metric: str = "roas",
    ) -> Optional[str]:
        """Train a model and register it."""
        if not training_data or len(training_data) < 10:
            logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return None
        
        try:
            from ml.predictive_modeling import PredictivePerformanceModel
            
            # Create and train model
            model = PredictivePerformanceModel(self.supabase_client)
            model.train(training_data, target_metric=target_metric)
            
            if not model.is_trained:
                logger.error("Model training failed")
                return None
            
            # Evaluate model (simplified)
            performance_metrics = {
                "training_samples": len(training_data),
                "model_type": model_type,
            }
            
            # Extract features
            features = []
            if training_data:
                sample = training_data[0]
                creative_data = sample.get("creative_data", {})
                # Extract feature names (simplified)
                features = list(creative_data.keys())
            
            # Register model
            version = self.registry.register_model(
                model_type=model_type,
                model=model,
                performance_metrics=performance_metrics,
                training_data_size=len(training_data),
                features=features,
            )
            
            self.training_history.append({
                "model_type": model_type,
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "training_samples": len(training_data),
            })
            
            logger.info(f"Model {model_type} trained and registered as version {version}")
            return version
        
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return None
    
    def should_retrain(
        self,
        model_type: str,
        new_data_count: int,
        performance_drop: float = 0.1,
    ) -> bool:
        """Determine if model should be retrained."""
        latest = self.registry.get_latest_model(model_type)
        if not latest:
            return True  # No model exists
        
        # Check if enough new data
        if new_data_count < latest.training_data_size * 0.2:
            return False
        
        # Check performance degradation
        # This would compare current performance to historical
        # Simplified: always retrain if enough new data
        return True


class ModelMonitor:
    """Monitor model performance."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def track_prediction(
        self,
        model_type: str,
        version: str,
        prediction: float,
        actual: float,
    ):
        """Track prediction accuracy."""
        if model_type not in self.performance_history:
            self.performance_history[model_type] = []
        
        error = abs(prediction - actual)
        mape = error / actual if actual > 0 else 0.0
        
        self.performance_history[model_type].append({
            "version": version,
            "prediction": prediction,
            "actual": actual,
            "error": error,
            "mape": mape,
            "timestamp": datetime.now().isoformat(),
        })
    
    def get_model_performance(
        self,
        model_type: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get performance metrics for model."""
        if model_type not in self.performance_history:
            return {}
        
        history = self.performance_history[model_type]
        
        if version:
            history = [h for h in history if h["version"] == version]
        
        if not history:
            return {}
        
        errors = [h["error"] for h in history]
        mapes = [h["mape"] for h in history]
        
        return {
            "mean_error": sum(errors) / len(errors),
            "mean_mape": sum(mapes) / len(mapes),
            "sample_count": len(history),
        }
    
    def detect_drift(
        self,
        model_type: str,
        threshold: float = 0.2,
    ) -> bool:
        """Detect model performance drift."""
        performance = self.get_model_performance(model_type)
        mape = performance.get("mean_mape", 0.0)
        
        return mape > threshold


def create_model_registry(registry_path: str = "models/registry") -> ModelRegistry:
    """Create model registry."""
    return ModelRegistry(registry_path)


def create_training_pipeline(
    registry: ModelRegistry,
    supabase_client=None,
) -> ModelTrainingPipeline:
    """Create training pipeline."""
    return ModelTrainingPipeline(registry, supabase_client)


def create_model_monitor(registry: ModelRegistry) -> ModelMonitor:
    """Create model monitor."""
    return ModelMonitor(registry)


__all__ = [
    "ModelRegistry",
    "ModelVersion",
    "ModelTrainingPipeline",
    "ModelMonitor",
    "create_model_registry",
    "create_training_pipeline",
    "create_model_monitor",
]

