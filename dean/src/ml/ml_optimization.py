"""
ML System Optimization Module
Comprehensive optimizations for training, inference, and data handling
"""

from __future__ import annotations

import logging
import time
import hashlib
import pickle
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Advanced imports
try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.ensemble import StackingRegressor
    from joblib import Parallel, delayed
    SKLEARN_ADVANCED_AVAILABLE = True
except ImportError:
    SKLEARN_ADVANCED_AVAILABLE = False

try:
    from infrastructure.caching import cache_manager
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    cache_manager = None

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking."""
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mae: float
    rmse: float
    r2_score: float
    training_time_seconds: float
    inference_time_ms: float
    last_trained: datetime
    prediction_count: int
    drift_score: float = 0.0


class FeatureCache:
    """Caches engineered features to avoid recomputation."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self.max_size = max_size
        self.ttl_hours = ttl_hours
    
    def _generate_key(self, df_hash: str, feature_config: Dict[str, Any]) -> str:
        """Generate cache key."""
        config_str = str(sorted(feature_config.items()))
        return hashlib.md5(f"{df_hash}_{config_str}".encode()).hexdigest()
    
    def get(self, df: pd.DataFrame, feature_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Get cached features."""
        if not CACHING_AVAILABLE:
            return None
        
        try:
            df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
            key = self._generate_key(df_hash, feature_config)
            
            # Check in-memory cache
            if key in self.cache:
                cached_df, cached_time = self.cache[key]
                if (datetime.now() - cached_time).total_seconds() < self.ttl_hours * 3600:
                    return cached_df.copy()
                else:
                    del self.cache[key]
            
            # Check Redis cache
            if cache_manager:
                cached_data = cache_manager.get(f"features_{key}")
                if cached_data:
                    cached_df = pickle.loads(cached_data)
                    self.cache[key] = (cached_df, datetime.now())
                    return cached_df.copy()
            
            return None
        except Exception as e:
            logger.debug(f"Feature cache get failed: {e}")
            return None
    
    def set(self, df: pd.DataFrame, feature_config: Dict[str, Any], engineered_df: pd.DataFrame):
        """Cache engineered features."""
        if not CACHING_AVAILABLE:
            return
        
        try:
            df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
            key = self._generate_key(df_hash, feature_config)
            
            # Store in-memory
            if len(self.cache) >= self.max_size:
                # Remove oldest
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (engineered_df.copy(), datetime.now())
            
            # Store in Redis
            if cache_manager:
                cache_manager.set(
                    f"features_{key}",
                    pickle.dumps(engineered_df),
                    ttl_seconds=self.ttl_hours * 3600,
                )
        except Exception as e:
            logger.debug(f"Feature cache set failed: {e}")


class PredictionCache:
    """Caches predictions to avoid redundant model inference."""
    
    def __init__(self, ttl_minutes: int = 30):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl_minutes = ttl_minutes
    
    def _generate_key(self, model_type: str, stage: str, features: Dict[str, float]) -> str:
        """Generate cache key from features."""
        feature_str = "_".join(f"{k}:{v:.4f}" for k, v in sorted(features.items()))
        return hashlib.md5(f"{model_type}_{stage}_{feature_str}".encode()).hexdigest()
    
    def get(self, model_type: str, stage: str, features: Dict[str, float]) -> Optional[Any]:
        """Get cached prediction."""
        if not CACHING_AVAILABLE:
            return None
        
        try:
            key = self._generate_key(model_type, stage, features)
            
            # Check in-memory
            if key in self.cache:
                prediction, cached_time = self.cache[key]
                if (datetime.now() - cached_time).total_seconds() < self.ttl_minutes * 60:
                    return prediction
                else:
                    del self.cache[key]
            
            # Check Redis
            if cache_manager:
                cached_data = cache_manager.get(f"prediction_{key}")
                if cached_data:
                    prediction = pickle.loads(cached_data)
                    self.cache[key] = (prediction, datetime.now())
                    return prediction
            
            return None
        except Exception as e:
            logger.debug(f"Prediction cache get failed: {e}")
            return None
    
    def set(self, model_type: str, stage: str, features: Dict[str, float], prediction: Any):
        """Cache prediction."""
        if not CACHING_AVAILABLE:
            return
        
        try:
            key = self._generate_key(model_type, stage, features)
            
            # Store in-memory
            self.cache[key] = (prediction, datetime.now())
            
            # Store in Redis
            if cache_manager:
                cache_manager.set(
                    f"prediction_{key}",
                    pickle.dumps(prediction),
                    ttl_seconds=self.ttl_minutes * 60,
                )
        except Exception as e:
            logger.debug(f"Prediction cache set failed: {e}")


class OptimizedModelTrainer:
    """Optimized model training with early stopping and hyperparameter tuning."""
    
    def __init__(self):
        self.performance_metrics: Dict[str, ModelPerformanceMetrics] = {}
    
    def train_with_early_stopping(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 0.001,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train model with early stopping."""
        try:
            best_score = float('inf')
            best_model = None
            patience_counter = 0
            training_history = []
            
            # For XGBoost, use built-in early stopping
            if hasattr(model, 'fit') and hasattr(model, 'set_params'):
                # XGBoost early stopping
                if 'XGBRegressor' in str(type(model)):
                    model.set_params(
                        early_stopping_rounds=patience,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )
                    model.fit(X_train, y_train)
                    best_model = model
                    best_score = model.best_score if hasattr(model, 'best_score') else 0.0
                else:
                    # For other models, manual early stopping
                    for epoch in range(max_epochs):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = np.mean((y_val - y_pred) ** 2)  # MSE
                        
                        training_history.append({
                            'epoch': epoch,
                            'val_score': score,
                        })
                        
                        if score < best_score - min_delta:
                            best_score = score
                            best_model = pickle.loads(pickle.dumps(model))  # Deep copy
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                break
                    
                    if best_model is None:
                        best_model = model
            else:
                # Fallback: regular training
                model.fit(X_train, y_train)
                best_model = model
            
            return best_model, {
                'best_score': best_score,
                'epochs_trained': len(training_history),
                'history': training_history,
            }
        except Exception as e:
            logger.error(f"Early stopping training failed: {e}")
            # Fallback to regular training
            model.fit(X_train, y_train)
            return model, {}
    
    def optimize_hyperparameters(
        self,
        model_class,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        n_iter: int = 20,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optimize hyperparameters using randomized search."""
        if not SKLEARN_ADVANCED_AVAILABLE:
            return model_class(), {}
        
        try:
            # Use RandomizedSearchCV for faster optimization
            search = RandomizedSearchCV(
                model_class(),
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                random_state=42,
                verbose=0,
            )
            
            search.fit(X_train, y_train)
            
            return search.best_estimator_, {
                'best_params': search.best_params_,
                'best_score': -search.best_score_,
                'cv_results': search.cv_results_,
            }
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {e}")
            return model_class(), {}


class BatchPredictor:
    """Optimized batch prediction for multiple ads."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def predict_batch(
        self,
        model,
        scaler,
        features_list: List[Dict[str, float]],
        feature_cols: List[str],
    ) -> List[float]:
        """Predict for multiple ads in batch."""
        try:
            # Convert features to matrix
            feature_matrix = []
            for features in features_list:
                feature_vector = [features.get(col, 0.0) for col in feature_cols]
                feature_matrix.append(feature_vector)
            
            X = np.array(feature_matrix)
            
            # Scale
            if scaler:
                X = scaler.transform(X)
            
            # Batch predict
            predictions = model.predict(X)
            
            return predictions.tolist()
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [0.0] * len(features_list)


class EnsembleOptimizer:
    """Optimized ensemble with stacking and dynamic weighting."""
    
    def __init__(self):
        self.ensemble_weights: Dict[str, float] = {}
    
    def create_stacked_ensemble(
        self,
        base_models: List[Tuple[str, Any]],
        meta_model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
    ) -> Any:
        """Create stacked ensemble model."""
        if not SKLEARN_ADVANCED_AVAILABLE:
            # Fallback: simple averaging
            return base_models[0][1] if base_models else None
        
        try:
            # Create stacking regressor
            estimators = [(name, model) for name, model in base_models]
            stacked = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_model,
                cv=cv,
                n_jobs=-1,
            )
            
            stacked.fit(X_train, y_train)
            return stacked
        except Exception as e:
            logger.warning(f"Stacking failed: {e}, using simple ensemble")
            return base_models[0][1] if base_models else None
    
    def calculate_dynamic_weights(
        self,
        models: Dict[str, Any],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate dynamic weights based on validation performance."""
        weights = {}
        errors = {}
        
        # Calculate error for each model
        for name, model in models.items():
            try:
                y_pred = model.predict(X_val)
                mae = np.mean(np.abs(y_val - y_pred))
                errors[name] = mae
            except Exception:
                errors[name] = float('inf')
        
        # Convert errors to weights (inverse, normalized)
        if errors:
            min_error = min(errors.values())
            if min_error > 0:
                # Weight inversely proportional to error
                total_weight = sum(1.0 / max(e, min_error * 0.1) for e in errors.values())
                for name, error in errors.items():
                    weights[name] = (1.0 / max(error, min_error * 0.1)) / total_weight
            else:
                # Equal weights if all perfect
                equal_weight = 1.0 / len(errors)
                weights = {name: equal_weight for name in errors.keys()}
        
        self.ensemble_weights = weights
        return weights
    
    def weighted_ensemble_predict(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Make weighted ensemble prediction."""
        if not models:
            return np.zeros(X.shape[0])
        
        weights = weights or self.ensemble_weights or {name: 1.0/len(models) for name in models.keys()}
        
        predictions = []
        for name, model in models.items():
            try:
                pred = model.predict(X)
                weight = weights.get(name, 0.0)
                predictions.append(pred * weight)
            except Exception:
                continue
        
        if not predictions:
            return np.zeros(X.shape[0])
        
        # Sum weighted predictions
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred


class ModelDriftDetector:
    """Detects model performance drift."""
    
    def __init__(self, drift_threshold: float = 0.15):
        self.drift_threshold = drift_threshold
        self.baseline_metrics: Dict[str, float] = {}
    
    def detect_drift(
        self,
        model_id: str,
        current_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, float]:
        """Detect if model performance has drifted."""
        baseline = baseline_metrics or self.baseline_metrics.get(model_id, {})
        
        if not baseline:
            # No baseline, assume no drift
            return False, 0.0
        
        # Calculate drift score
        drift_scores = []
        
        for metric in ['mae', 'rmse', 'r2_score']:
            if metric in baseline and metric in current_metrics:
                baseline_val = baseline[metric]
                current_val = current_metrics[metric]
                
                if baseline_val > 0:
                    if metric == 'r2_score':
                        # For R², lower is worse
                        drift = (baseline_val - current_val) / abs(baseline_val)
                    else:
                        # For MAE/RMSE, higher is worse
                        drift = (current_val - baseline_val) / baseline_val
                    
                    drift_scores.append(abs(drift))
        
        avg_drift = np.mean(drift_scores) if drift_scores else 0.0
        has_drift = avg_drift > self.drift_threshold
        
        return has_drift, avg_drift
    
    def update_baseline(self, model_id: str, metrics: Dict[str, float]):
        """Update baseline metrics."""
        self.baseline_metrics[model_id] = metrics.copy()


class DataLoaderOptimizer:
    """Optimized data loading with batching and caching."""
    
    def __init__(self):
        self.query_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self.cache_ttl_hours = 1
    
    def load_data_batch(
        self,
        supabase_client,
        ad_ids: List[str],
        stages: Optional[List[str]] = None,
        days_back: int = 30,
    ) -> pd.DataFrame:
        """Load data in optimized batches."""
        try:
            # Check cache
            cache_key = f"{hash(tuple(sorted(ad_ids)))}_{stages}_{days_back}"
            if cache_key in self.query_cache:
                cached_df, cached_time = self.query_cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < self.cache_ttl_hours * 3600:
                    return cached_df.copy()
            
            # Load in batches if many ad_ids
            if len(ad_ids) > 100:
                all_dfs = []
                for i in range(0, len(ad_ids), 100):
                    batch_ids = ad_ids[i:i+100]
                    df = supabase_client.get_performance_data(
                        ad_ids=batch_ids,
                        stages=stages,
                        days_back=days_back,
                    )
                    if not df.empty:
                        all_dfs.append(df)
                
                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    self.query_cache[cache_key] = (combined_df, datetime.now())
                    return combined_df
            else:
                df = supabase_client.get_performance_data(
                    ad_ids=ad_ids,
                    stages=stages,
                    days_back=days_back,
                )
                if not df.empty:
                    self.query_cache[cache_key] = (df, datetime.now())
                    return df
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Batch data loading failed: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear query cache."""
        self.query_cache.clear()


class ParallelFeatureEngineer:
    """Parallel feature engineering for faster processing."""
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs
    
    def engineer_features_parallel(
        self,
        df: pd.DataFrame,
        feature_config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Engineer features in parallel."""
        try:
            if SKLEARN_ADVANCED_AVAILABLE and self.n_jobs != 1:
                # Parallel processing for rolling features
                def compute_rolling(group):
                    return group.rolling(window=7).mean()
                
                # Group by ad_id and compute in parallel
                if 'ad_id' in df.columns:
                    grouped = df.groupby('ad_id')
                    results = Parallel(n_jobs=self.n_jobs)(
                        delayed(compute_rolling)(group) for _, group in grouped
                    )
                    # Combine results
                    df_result = pd.concat(results, ignore_index=True)
                    return df_result
                else:
                    return df
            else:
                # Sequential processing
                return df
        except Exception as e:
            logger.warning(f"Parallel feature engineering failed: {e}")
            return df


class ModelPerformanceTracker:
    """Tracks model performance over time."""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[ModelPerformanceMetrics]] = {}
    
    def track_performance(
        self,
        model_id: str,
        metrics: ModelPerformanceMetrics,
    ):
        """Track model performance."""
        if model_id not in self.metrics_history:
            self.metrics_history[model_id] = []
        
        self.metrics_history[model_id].append(metrics)
        
        # Keep only last 100 entries
        if len(self.metrics_history[model_id]) > 100:
            self.metrics_history[model_id] = self.metrics_history[model_id][-100:]
    
    def get_performance_trend(
        self,
        model_id: str,
        metric: str = 'accuracy',
    ) -> Dict[str, Any]:
        """Get performance trend for a model."""
        if model_id not in self.metrics_history:
            return {'trend': 'unknown', 'change': 0.0}
        
        history = self.metrics_history[model_id]
        if len(history) < 2:
            return {'trend': 'insufficient_data', 'change': 0.0}
        
        # Get metric values
        values = [getattr(m, metric, 0.0) for m in history]
        
        # Calculate trend
        recent_avg = np.mean(values[-5:]) if len(values) >= 5 else values[-1]
        older_avg = np.mean(values[:-5]) if len(values) >= 5 else values[0]
        
        change = recent_avg - older_avg
        change_pct = (change / older_avg * 100) if older_avg > 0 else 0.0
        
        if change_pct > 5:
            trend = 'improving'
        elif change_pct < -5:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change': change,
            'change_pct': change_pct,
            'current': recent_avg,
            'baseline': older_avg,
        }


# Global optimizers
_feature_cache = FeatureCache()
_prediction_cache = PredictionCache()
_model_trainer = OptimizedModelTrainer()
_batch_predictor = BatchPredictor()
_ensemble_optimizer = EnsembleOptimizer()
_drift_detector = ModelDriftDetector()
_data_loader = DataLoaderOptimizer()
_performance_tracker = ModelPerformanceTracker()


def get_feature_cache() -> FeatureCache:
    """Get global feature cache."""
    return _feature_cache


def get_prediction_cache() -> PredictionCache:
    """Get global prediction cache."""
    return _prediction_cache


def get_optimized_trainer() -> OptimizedModelTrainer:
    """Get optimized model trainer."""
    return _model_trainer


def get_batch_predictor() -> BatchPredictor:
    """Get batch predictor."""
    return _batch_predictor


def get_ensemble_optimizer() -> EnsembleOptimizer:
    """Get ensemble optimizer."""
    return _ensemble_optimizer


def get_drift_detector() -> ModelDriftDetector:
    """Get drift detector."""
    return _drift_detector


def get_data_loader() -> DataLoaderOptimizer:
    """Get optimized data loader."""
    return _data_loader


def get_performance_tracker() -> ModelPerformanceTracker:
    """Get performance tracker."""
    return _performance_tracker


__all__ = [
    "FeatureCache",
    "PredictionCache",
    "OptimizedModelTrainer",
    "BatchPredictor",
    "EnsembleOptimizer",
    "ModelDriftDetector",
    "DataLoaderOptimizer",
    "ParallelFeatureEngineer",
    "ModelPerformanceTracker",
    "ModelPerformanceMetrics",
    "get_feature_cache",
    "get_prediction_cache",
    "get_optimized_trainer",
    "get_batch_predictor",
    "get_ensemble_optimizer",
    "get_drift_detector",
    "get_data_loader",
    "get_performance_tracker",
]

