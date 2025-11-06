"""
ADVANCED Anomaly Detection System
Isolation forest, autoencoders, multivariate anomaly detection, contextual anomaly detection
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

# Advanced imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Anomaly detection result with advanced metrics."""
    is_anomaly: bool
    anomaly_score: float  # 0-1, higher = more anomalous
    confidence: float
    anomaly_type: str  # "outlier", "contextual", "multivariate", "temporal"
    features_contributing: Dict[str, float]  # Feature contribution to anomaly
    severity: str  # "low", "medium", "high", "critical"
    recommended_action: str
    context: Dict[str, Any]


class IsolationForestDetector:
    """Isolation Forest for anomaly detection."""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = None
        self.trained = False
    
    def train(self, data: np.ndarray):
        """Train isolation forest model."""
        if not SKLEARN_AVAILABLE:
            return False
        
        try:
            # Scale data
            self.scaler = RobustScaler()  # Robust to outliers
            data_scaled = self.scaler.fit_transform(data)
            
            # Train model
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(data_scaled)
            self.trained = True
            logger.info("Isolation Forest trained successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to train Isolation Forest: {e}")
            return False
    
    def detect(self, data_point: np.ndarray) -> Tuple[bool, float]:
        """Detect if data point is anomalous."""
        if not self.trained:
            return False, 0.0
        
        try:
            data_scaled = self.scaler.transform([data_point])
            prediction = self.model.predict([data_point])[0]
            score = self.model.score_samples([data_point])[0]
            
            # Convert to anomaly score (0-1, higher = more anomalous)
            # Isolation Forest returns negative scores for anomalies
            anomaly_score = 1.0 / (1.0 + np.exp(score))  # Sigmoid transformation
            
            is_anomaly = prediction == -1
            return is_anomaly, float(anomaly_score)
        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {e}")
            return False, 0.0


class AutoencoderDetector:
    """Autoencoder for anomaly detection using reconstruction error."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = None
        self.threshold = None
        self.trained = False
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - Autoencoder disabled")
            return
        
        # Build autoencoder
        self.model = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
        )
    
    def train(self, data: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train autoencoder model."""
        if not TORCH_AVAILABLE or self.model is None:
            return False
        
        try:
            # Scale data
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            data_scaled = self.scaler.fit_transform(data)
            
            # Convert to tensors
            X = torch.FloatTensor(data_scaled)
            
            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(epochs):
                # Forward pass
                reconstructed = self.model(X)
                loss = criterion(reconstructed, X)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Calculate threshold (95th percentile of reconstruction errors)
            with torch.no_grad():
                reconstructed = self.model(X)
                errors = torch.mean((X - reconstructed) ** 2, dim=1).numpy()
                self.threshold = np.percentile(errors, 95)
            
            self.trained = True
            logger.info(f"Autoencoder trained successfully (threshold: {self.threshold:.4f})")
            return True
        except Exception as e:
            logger.error(f"Failed to train Autoencoder: {e}")
            return False
    
    def detect(self, data_point: np.ndarray) -> Tuple[bool, float]:
        """Detect if data point is anomalous."""
        if not self.trained:
            return False, 0.0
        
        try:
            data_scaled = self.scaler.transform([data_point])
            X = torch.FloatTensor(data_scaled)
            
            with torch.no_grad():
                reconstructed = self.model(X)
                error = torch.mean((X - reconstructed) ** 2).item()
            
            is_anomaly = error > self.threshold
            anomaly_score = min(1.0, error / (self.threshold * 2))  # Normalize to 0-1
            
            return is_anomaly, float(anomaly_score)
        except Exception as e:
            logger.error(f"Autoencoder detection failed: {e}")
            return False, 0.0


class MultivariateAnomalyDetector:
    """Multivariate anomaly detection using PCA and clustering."""
    
    def __init__(self):
        self.pca = None
        self.scaler = None
        self.clusterer = None
        self.trained = False
    
    def train(self, data: np.ndarray, n_components: int = 3):
        """Train multivariate detector."""
        if not SKLEARN_AVAILABLE:
            return False
        
        try:
            # Scale data
            self.scaler = StandardScaler()
            data_scaled = self.scaler.fit_transform(data)
            
            # PCA for dimensionality reduction
            self.pca = PCA(n_components=min(n_components, data.shape[1]))
            data_pca = self.pca.fit_transform(data_scaled)
            
            # DBSCAN for outlier detection
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)
            clusters = self.clusterer.fit_predict(data_pca)
            
            self.trained = True
            logger.info(f"Multivariate detector trained (outliers: {np.sum(clusters == -1)})")
            return True
        except Exception as e:
            logger.error(f"Failed to train multivariate detector: {e}")
            return False
    
    def detect(self, data_point: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """Detect multivariate anomaly."""
        if not self.trained:
            return False, 0.0, {}
        
        try:
            data_scaled = self.scaler.transform([data_point])
            data_pca = self.pca.transform(data_scaled)
            
            # Check if point is in outlier cluster
            cluster = self.clusterer.fit_predict(data_pca)[0]
            is_anomaly = cluster == -1
            
            # Calculate distance to nearest cluster
            if cluster == -1:
                # Find distance to nearest non-outlier cluster
                core_samples = self.clusterer.core_sample_indices_
                if len(core_samples) > 0:
                    distances = np.linalg.norm(
                        data_pca - self.clusterer.components_[core_samples],
                        axis=1
                    )
                    min_distance = np.min(distances)
                    anomaly_score = min(1.0, min_distance / 2.0)  # Normalize
                else:
                    anomaly_score = 0.8
            else:
                anomaly_score = 0.0
            
            # Feature contribution (using PCA components)
            feature_contributions = {}
            if self.pca:
                contributions = np.abs(self.pca.components_ @ data_scaled[0])
                for i, contrib in enumerate(contributions):
                    feature_contributions[f"feature_{i}"] = float(contrib)
            
            return is_anomaly, float(anomaly_score), feature_contributions
        except Exception as e:
            logger.error(f"Multivariate detection failed: {e}")
            return False, 0.0, {}


class ContextualAnomalyDetector:
    """Contextual anomaly detection considering time and context."""
    
    def __init__(self):
        self.context_history: List[Dict[str, Any]] = []
    
    def detect(
        self,
        current_performance: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Tuple[bool, float]:
        """Detect contextual anomalies."""
        if not historical_performance:
            return False, 0.0
        
        # Extract context
        time_of_day = context.get("hour", 12)
        day_of_week = context.get("day_of_week", 0)
        market_condition = context.get("market_condition", "normal")
        
        # Get historical performance for similar context
        similar_context_data = [
            p for p in historical_performance
            if abs(p.get("hour", 12) - time_of_day) < 2
            and p.get("day_of_week", 0) == day_of_week
            and p.get("market_condition", "normal") == market_condition
        ]
        
        if len(similar_context_data) < 3:
            # Not enough context data
            return False, 0.0
        
        # Compare current to contextual baseline
        current_roas = current_performance.get("roas", 0)
        current_ctr = current_performance.get("ctr", 0)
        
        contextual_roas = np.mean([p.get("roas", 0) for p in similar_context_data])
        contextual_ctr = np.mean([p.get("ctr", 0) for p in similar_context_data])
        
        roas_deviation = abs(current_roas - contextual_roas) / max(contextual_roas, 0.01)
        ctr_deviation = abs(current_ctr - contextual_ctr) / max(contextual_ctr, 0.01)
        
        avg_deviation = (roas_deviation + ctr_deviation) / 2
        
        is_anomaly = avg_deviation > 0.3  # 30% deviation threshold
        anomaly_score = min(1.0, avg_deviation)
        
        return is_anomaly, float(anomaly_score)


class AnomalyDetector:
    """ADVANCED Anomaly detector with multiple methods."""
    
    def __init__(
        self,
        use_isolation_forest: bool = True,
        use_autoencoder: bool = True,
        use_multivariate: bool = True,
        use_contextual: bool = True,
    ):
        self.use_isolation_forest = use_isolation_forest and SKLEARN_AVAILABLE
        self.use_autoencoder = use_autoencoder and TORCH_AVAILABLE
        self.use_multivariate = use_multivariate and SKLEARN_AVAILABLE
        self.use_contextual = use_contextual
        
        # Initialize detectors
        self.isolation_forest = IsolationForestDetector() if self.use_isolation_forest else None
        self.autoencoder = None  # Will be initialized when training
        self.multivariate = MultivariateAnomalyDetector() if self.use_multivariate else None
        self.contextual = ContextualAnomalyDetector() if self.use_contextual else None
        
        # Fallback z-score threshold
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
    
    def train_models(self, historical_data: List[Dict[str, Any]]):
        """Train all anomaly detection models."""
        if not historical_data or len(historical_data) < 20:
            return False
        
        try:
            # Prepare feature matrix
            features = []
            for data in historical_data:
                feature_vector = [
                    data.get("roas", 0),
                    data.get("ctr", 0) * 100,
                    data.get("cpa", 0) / 10,
                    data.get("cpm", 0) / 10,
                    data.get("spend", 0) / 100,
                    data.get("impressions", 0) / 10000,
                    data.get("clicks", 0) / 100,
                ]
                features.append(feature_vector)
            
            X = np.array(features)
            
            # Train isolation forest
            if self.isolation_forest:
                self.isolation_forest.train(X)
            
            # Train autoencoder
            if self.use_autoencoder:
                input_dim = X.shape[1]
                self.autoencoder = AutoencoderDetector(input_dim=input_dim, encoding_dim=4)
                self.autoencoder.train(X, epochs=50)
            
            # Train multivariate detector
            if self.multivariate:
                self.multivariate.train(X)
            
            logger.info("All anomaly detection models trained successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to train anomaly detection models: {e}")
            return False
    
    def detect_anomalies(
        self,
        data_points: List[Dict[str, Any]],
        metric: str = "roas",
    ) -> List[AnomalyResult]:
        """Detect anomalies in data points."""
        if not data_points or len(data_points) < 3:
            return []
        
        results = []
        
        for point in data_points:
            result = self.detect_anomaly(point, {})
            results.append(result)
        
        return results
    
    def detect_anomaly(
        self,
        current_performance: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> AnomalyResult:
        """Detect if current performance is anomalous."""
        context = context or {}
        
        # Prepare feature vector
        feature_vector = np.array([[
            current_performance.get("roas", 0),
            current_performance.get("ctr", 0) * 100,
            current_performance.get("cpa", 0) / 10,
            current_performance.get("cpm", 0) / 10,
            current_performance.get("spend", 0) / 100,
            current_performance.get("impressions", 0) / 10000,
            current_performance.get("clicks", 0) / 100,
        ]])
        
        # Ensemble detection
        detections = []
        scores = []
        feature_contributions = {}
        
        # Isolation Forest
        if self.isolation_forest and self.isolation_forest.trained:
            is_anomaly, score = self.isolation_forest.detect(feature_vector[0])
            detections.append(is_anomaly)
            scores.append(score)
        
        # Autoencoder
        if self.autoencoder and self.autoencoder.trained:
            is_anomaly, score = self.autoencoder.detect(feature_vector[0])
            detections.append(is_anomaly)
            scores.append(score)
        
        # Multivariate
        if self.multivariate and self.multivariate.trained:
            is_anomaly, score, contributions = self.multivariate.detect(feature_vector[0])
            detections.append(is_anomaly)
            scores.append(score)
            feature_contributions.update(contributions)
        
        # Contextual
        if self.contextual and historical_performance:
            is_anomaly, score = self.contextual.detect(
                current_performance,
                historical_performance,
                context,
            )
            detections.append(is_anomaly)
            scores.append(score)
        
        # Fallback to z-score if no models trained
        if not detections and historical_performance:
            z_score_result = self._z_score_detection(current_performance, historical_performance)
            detections.append(z_score_result["is_anomaly"])
            scores.append(z_score_result["score"])
        
        # Ensemble decision
        is_anomaly = sum(detections) >= len(detections) * 0.5  # Majority vote
        avg_score = np.mean(scores) if scores else 0.0
        
        # Determine anomaly type
        if self.contextual and len([d for d in detections if d]) > 0:
            anomaly_type = "contextual"
        elif self.multivariate and len([d for d in detections if d]) > 0:
            anomaly_type = "multivariate"
        else:
            anomaly_type = "outlier"
        
        # Determine severity
        if avg_score > 0.8:
            severity = "critical"
            recommended_action = "pause_immediately"
        elif avg_score > 0.6:
            severity = "high"
            recommended_action = "investigate_and_pause"
        elif avg_score > 0.4:
            severity = "medium"
            recommended_action = "monitor_closely"
        else:
            severity = "low"
            recommended_action = "continue_monitoring"
        
        confidence = min(1.0, len(detections) / 4.0) * avg_score
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=avg_score,
            confidence=confidence,
            anomaly_type=anomaly_type,
            features_contributing=feature_contributions,
            severity=severity,
            recommended_action=recommended_action,
            context=context,
        )
    
    def _z_score_detection(
        self,
        current_performance: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fallback z-score detection."""
        import statistics
        
        current_roas = current_performance.get("roas", 0.0)
        current_ctr = current_performance.get("ctr", 0.0)
        
        historical_roas = [p.get("roas", 0.0) for p in historical_performance if p.get("roas") is not None]
        historical_ctr = [p.get("ctr", 0.0) for p in historical_performance if p.get("ctr") is not None]
        
        max_z_score = 0.0
        is_anomaly = False
        
        if historical_roas:
            mean_roas = statistics.mean(historical_roas)
            std_roas = statistics.stdev(historical_roas) if len(historical_roas) > 1 else 0.0
            
            if std_roas > 0:
                z_score = abs((current_roas - mean_roas) / std_roas)
                max_z_score = max(max_z_score, z_score)
                if z_score > self.z_score_threshold:
                    is_anomaly = True
        
        if historical_ctr:
            mean_ctr = statistics.mean(historical_ctr)
            std_ctr = statistics.stdev(historical_ctr) if len(historical_ctr) > 1 else 0.0
            
            if std_ctr > 0:
                z_score = abs((current_ctr - mean_ctr) / std_ctr)
                max_z_score = max(max_z_score, z_score)
                if z_score > self.z_score_threshold:
                    is_anomaly = True
        
        score = min(1.0, max_z_score / self.z_score_threshold)
        
        return {
            "is_anomaly": is_anomaly,
            "score": score,
        }
    
    def detect_performance_anomaly(
        self,
        current_performance: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> AnomalyResult:
        """Detect performance anomaly (alias for detect_anomaly)."""
        return self.detect_anomaly(current_performance, historical_performance, context)
    
    def detect_spike(
        self,
        recent_data: List[Dict[str, Any]],
        metric: str = "roas",
    ) -> Dict[str, Any]:
        """Detect performance spikes."""
        if len(recent_data) < 3:
            return {
                "spike_detected": False,
                "type": None,
            }
        
        values = [
            float(p.get(metric, 0.0))
            for p in recent_data
            if p.get(metric) is not None
        ]
        
        if len(values) < 3:
            return {
                "spike_detected": False,
                "type": None,
            }
        
        import statistics
        
        recent = values[-1]
        previous = values[-2] if len(values) > 1 else values[0]
        baseline = statistics.mean(values[:-1]) if len(values) > 1 else values[0]
        
        # Positive spike
        if recent > previous * 1.5 and recent > baseline * 1.3:
            return {
                "spike_detected": True,
                "type": "positive",
                "increase_pct": float((recent - previous) / previous * 100) if previous > 0 else 0.0,
            }
        
        # Negative spike
        if recent < previous * 0.5 and recent < baseline * 0.7:
            return {
                "spike_detected": True,
                "type": "negative",
                "decrease_pct": float((previous - recent) / previous * 100) if previous > 0 else 0.0,
            }
        
        return {
            "spike_detected": False,
            "type": None,
        }


def create_anomaly_detector(**kwargs) -> AnomalyDetector:
    """Create advanced anomaly detector."""
    return AnomalyDetector(**kwargs)


__all__ = [
    "AnomalyDetector",
    "AnomalyResult",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "MultivariateAnomalyDetector",
    "ContextualAnomalyDetector",
    "create_anomaly_detector",
]
