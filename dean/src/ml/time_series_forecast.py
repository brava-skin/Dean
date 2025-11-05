"""
Time Series Forecasting
Predicts creative performance over time
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TimeSeriesForecaster:
    """Forecasts time series performance."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
    
    def forecast_creative_performance(
        self,
        historical_data: List[Dict[str, Any]],
        horizon_hours: int = 24,
        metric: str = "roas",
    ) -> Dict[str, Any]:
        """Forecast creative performance."""
        if not historical_data or len(historical_data) < 3:
            return {
                "forecast": None,
                "confidence": 0.0,
                "trend": "insufficient_data",
            }
        
        # Extract time series
        timestamps = []
        values = []
        
        for point in historical_data:
            ts = point.get("timestamp")
            val = point.get(metric, 0.0)
            
            if ts and val is not None:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                timestamps.append(ts)
                values.append(float(val))
        
        if len(values) < 3:
            return {
                "forecast": None,
                "confidence": 0.0,
                "trend": "insufficient_data",
            }
        
        # Simple linear regression forecast
        if SKLEARN_AVAILABLE:
            try:
                # Convert timestamps to numeric (hours since first)
                first_ts = timestamps[0]
                X = np.array([
                    (ts - first_ts).total_seconds() / 3600.0
                    for ts in timestamps
                ]).reshape(-1, 1)
                y = np.array(values)
                
                # Fit model
                model = LinearRegression()
                model.fit(X, y)
                
                # Forecast
                last_ts = timestamps[-1]
                forecast_time = last_ts + timedelta(hours=horizon_hours)
                forecast_X = np.array([[
                    (forecast_time - first_ts).total_seconds() / 3600.0
                ]])
                forecast_value = model.predict(forecast_X)[0]
                
                # Calculate trend
                slope = model.coef_[0]
                if slope > 0.01:
                    trend = "increasing"
                elif slope < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                # Calculate confidence (based on R^2)
                score = model.score(X, y)
                confidence = max(0.0, min(1.0, score))
                
                return {
                    "forecast": float(forecast_value),
                    "confidence": float(confidence),
                    "trend": trend,
                    "slope": float(slope),
                }
            except Exception as e:
                logger.error(f"Error in forecasting: {e}")
        
        # Fallback: simple moving average
        recent_values = values[-min(5, len(values)):]
        forecast_value = np.mean(recent_values)
        
        # Calculate trend
        if len(values) >= 2:
            recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
            older_avg = np.mean(values[:-3]) if len(values) > 3 else values[0]
            
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "forecast": float(forecast_value),
            "confidence": 0.5,
            "trend": trend,
        }
    
    def detect_fatigue(
        self,
        historical_data: List[Dict[str, Any]],
        metric: str = "ctr",
    ) -> Dict[str, Any]:
        """Detect creative fatigue."""
        if not historical_data or len(historical_data) < 5:
            return {
                "fatigue_detected": False,
                "confidence": 0.0,
            }
        
        values = [
            float(point.get(metric, 0.0))
            for point in historical_data
            if point.get(metric) is not None
        ]
        
        if len(values) < 5:
            return {
                "fatigue_detected": False,
                "confidence": 0.0,
            }
        
        # Calculate trend
        recent = values[-3:]
        older = values[:-3] if len(values) > 3 else values[:2]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        # Fatigue: significant drop in performance
        drop_pct = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0.0
        
        fatigue_detected = drop_pct > 0.25  # 25% drop indicates fatigue
        confidence = min(drop_pct, 1.0)
        
        return {
            "fatigue_detected": fatigue_detected,
            "confidence": float(confidence),
            "drop_percentage": float(drop_pct),
            "recent_avg": float(recent_avg),
            "older_avg": float(older_avg),
        }


def create_time_series_forecaster() -> TimeSeriesForecaster:
    """Create a time series forecaster."""
    return TimeSeriesForecaster()


__all__ = ["TimeSeriesForecaster", "create_time_series_forecaster"]

