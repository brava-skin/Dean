"""
Anomaly Detection System
Detects unusual patterns in performance
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import statistics

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalies in performance data."""
    
    def __init__(self):
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
    
    def detect_anomalies(
        self,
        data_points: List[Dict[str, Any]],
        metric: str = "roas",
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in data."""
        if not data_points or len(data_points) < 3:
            return []
        
        values = [
            float(point.get(metric, 0.0))
            for point in data_points
            if point.get(metric) is not None
        ]
        
        if len(values) < 3:
            return []
        
        # Calculate statistics
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Z-score method
        anomalies = []
        for i, (point, value) in enumerate(zip(data_points, values)):
            if std > 0:
                z_score = abs((value - mean) / std)
                
                if z_score > self.z_score_threshold:
                    anomalies.append({
                        "index": i,
                        "value": value,
                        "z_score": z_score,
                        "metric": metric,
                        "anomaly_type": "outlier",
                        "data_point": point,
                    })
        
        return anomalies
    
    def detect_performance_anomaly(
        self,
        current_performance: Dict[str, Any],
        historical_performance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect if current performance is anomalous."""
        if not historical_performance:
            return {
                "is_anomaly": False,
                "confidence": 0.0,
            }
        
        # Compare current to historical
        current_roas = current_performance.get("roas", 0.0)
        current_ctr = current_performance.get("ctr", 0.0)
        
        historical_roas = [
            p.get("roas", 0.0) for p in historical_performance
            if p.get("roas") is not None
        ]
        historical_ctr = [
            p.get("ctr", 0.0) for p in historical_performance
            if p.get("ctr") is not None
        ]
        
        anomalies = []
        
        # Check ROAS
        if historical_roas:
            mean_roas = statistics.mean(historical_roas)
            std_roas = statistics.stdev(historical_roas) if len(historical_roas) > 1 else 0.0
            
            if std_roas > 0:
                z_score = abs((current_roas - mean_roas) / std_roas)
                if z_score > self.z_score_threshold:
                    anomalies.append({
                        "metric": "roas",
                        "z_score": z_score,
                        "current": current_roas,
                        "mean": mean_roas,
                    })
        
        # Check CTR
        if historical_ctr:
            mean_ctr = statistics.mean(historical_ctr)
            std_ctr = statistics.stdev(historical_ctr) if len(historical_ctr) > 1 else 0.0
            
            if std_ctr > 0:
                z_score = abs((current_ctr - mean_ctr) / std_ctr)
                if z_score > self.z_score_threshold:
                    anomalies.append({
                        "metric": "ctr",
                        "z_score": z_score,
                        "current": current_ctr,
                        "mean": mean_ctr,
                    })
        
        if anomalies:
            max_z_score = max(a["z_score"] for a in anomalies)
            confidence = min(max_z_score / self.z_score_threshold, 1.0)
            
            return {
                "is_anomaly": True,
                "confidence": float(confidence),
                "anomalies": anomalies,
            }
        
        return {
            "is_anomaly": False,
            "confidence": 0.0,
        }
    
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
        
        # Check for sudden increase
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


def create_anomaly_detector() -> AnomalyDetector:
    """Create an anomaly detector."""
    return AnomalyDetector()


__all__ = ["AnomalyDetector", "create_anomaly_detector"]

