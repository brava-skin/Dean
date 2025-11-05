"""
DEAN ML SYSTEM MONITORING
Combined ML status and dashboard functionality

This module provides:
- ML system status monitoring
- Health metrics and diagnostics
- Performance dashboard
- Learning progress tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
from supabase import Client

from infrastructure.utils import now_utc

logger = logging.getLogger(__name__)

# =====================================================
# ML STATUS MONITORING
# =====================================================

def get_amsterdam_time() -> datetime:
    """Get current time in Amsterdam timezone."""
    amsterdam_tz = pytz.timezone('Europe/Amsterdam')
    return datetime.now(amsterdam_tz)

def get_ml_learning_summary(supabase_client: Client) -> Dict[str, Any]:
    """Get detailed ML learning summary for diagnostics."""
    try:
        amsterdam_now = get_amsterdam_time()
        last_24h = (amsterdam_now - timedelta(hours=24)).isoformat()
        
        # Get model status
        models_response = supabase_client.table('ml_models').select('*').eq('is_active', True).execute()
        active_models = models_response.data if models_response.data else []
        
        # Get recent predictions
        predictions_response = supabase_client.table('ml_predictions').select('*').gte('created_at', last_24h).execute()
        recent_predictions = predictions_response.data if predictions_response.data else []
        
        # Get learning events
        learning_response = supabase_client.table('learning_events').select('*').gte('created_at', last_24h).execute()
        learning_events = learning_response.data if learning_response.data else []
        
        # Get performance data for training
        perf_response = supabase_client.table('performance_metrics').select('*').gte('created_at', last_24h).execute()
        recent_data = perf_response.data if perf_response.data else []
        
        # Analyze learning progress
        model_types = set()
        stages_with_models = set()
        for model in active_models:
            model_types.add(model.get('model_type', 'unknown'))
            stages_with_models.add(model.get('stage', 'unknown'))
        
        # Check for recent training
        recent_training = []
        for model in active_models:
            trained_at = model.get('trained_at')
            if trained_at:
                try:
                    trained_time = datetime.fromisoformat(trained_at.replace('Z', '+00:00'))
                    if (amsterdam_now - trained_time).total_seconds() < 86400:  # Last 24h
                        recent_training.append({
                            'type': model.get('model_type'),
                            'stage': model.get('stage'),
                            'trained_at': trained_time.strftime('%H:%M')
                        })
                except:
                    pass
        
        # Analyze learning events
        event_types = {}
        for event in learning_events:
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Determine ML status
        if len(recent_training) > 0:
            status = "LEARNING"
        elif len(active_models) > 0:
            status = "READY"
        else:
            status = "INITIALIZING"
        
        return {
            "status": status,
            "amsterdam_time": amsterdam_now.strftime('%Y-%m-%d %H:%M:%S %Z'),
            "active_models": len(active_models),
            "model_types": list(model_types),
            "stages_ready": list(stages_with_models),
            "recent_training": recent_training,
            "predictions_24h": len(recent_predictions),
            "learning_events_24h": len(learning_events),
            "event_types": event_types,
            "data_points_24h": len(recent_data),
            "diagnostics": {
                "models_available": len(active_models) > 0,
                "recent_activity": len(learning_events) > 0 or len(recent_training) > 0,
                "data_flowing": len(recent_data) > 0,
                "predictions_made": len(recent_predictions) > 0
            }
        }
        
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "amsterdam_time": get_amsterdam_time().strftime('%Y-%m-%d %H:%M:%S %Z'),
            "active_models": 0,
            "model_types": [],
            "stages_ready": [],
            "recent_training": [],
            "predictions_24h": 0,
            "learning_events_24h": 0,
            "event_types": {},
            "data_points_24h": 0,
            "diagnostics": {
                "models_available": False,
                "recent_activity": False,
                "data_flowing": False,
                "predictions_made": False
            }
        }

def send_ml_learning_report(supabase_client: Client, notify_func) -> None:
    """Send detailed ML learning report for diagnostics."""
    summary = get_ml_learning_summary(supabase_client)
    
    if summary["status"] == "ERROR":
        notify_func(f"🤖 ML ERROR: {summary.get('error', 'Unknown error')} | Time: {summary['amsterdam_time']}")
        return
    
    # Create diagnostic message
    status_emoji = {
        "LEARNING": "🧠",
        "READY": "✅", 
        "INITIALIZING": "⏳",
        "ERROR": "❌"
    }
    
    emoji = status_emoji.get(summary["status"], "🤖")
    
    # Build status message
    message_parts = [f"{emoji} ML {summary['status']}"]
    
    if summary["recent_training"]:
        training_info = ", ".join([f"{t['type']}({t['stage']})@{t['trained_at']}" for t in summary["recent_training"]])
        message_parts.append(f"Trained: {training_info}")
    
    if summary["active_models"] > 0:
        message_parts.append(f"Models: {summary['active_models']}")
    
    if summary["predictions_24h"] > 0:
        message_parts.append(f"Predictions: {summary['predictions_24h']}")
    
    if summary["learning_events_24h"] > 0:
        message_parts.append(f"Learning: {summary['learning_events_24h']}")
    
    if summary["data_points_24h"] > 0:
        message_parts.append(f"Data: {summary['data_points_24h']}")
    
    # Add diagnostic info
    diag = summary["diagnostics"]
    if not diag["data_flowing"]:
        message_parts.append("⚠️ No data flow")
    if not diag["models_available"]:
        message_parts.append("⚠️ No models")
    
    message_parts.append(f"Time: {summary['amsterdam_time']}")
    
    notify_func(" | ".join(message_parts))

# =====================================================
# ML DASHBOARD
# =====================================================

@dataclass
class MLHealthMetrics:
    """Snapshot of ML system health."""
    timestamp: datetime
    
    # Model performance
    avg_model_accuracy: float
    avg_prediction_confidence: float
    models_trained_24h: int
    predictions_made_24h: int
    
    # Decision influence
    ml_influence_avg: float
    ml_overrides_count: int
    ml_agreement_rate: float
    
    # Data quality
    anomalies_detected_24h: int
    data_quality_score: float
    
    # System health
    error_rate: float
    avg_execution_time_ms: float

class MLDashboard:
    """Monitoring dashboard for ML system."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        from supabase import create_client
        self.client: Client = create_client(supabase_url, supabase_key)
        self.logger = logging.getLogger(f"{__name__}.MLDashboard")
    
    def get_health_metrics(self, hours_back: int = 24) -> MLHealthMetrics:
        """Get current ML system health metrics."""
        try:
            start_time = (now_utc() - timedelta(hours=hours_back)).isoformat()
            
            # Get model metrics
            model_metrics = self._get_model_metrics(start_time)
            
            # Get prediction metrics
            prediction_metrics = self._get_prediction_metrics(start_time)
            
            # Get decision metrics
            decision_metrics = self._get_decision_metrics(start_time)
            
            # Get anomaly metrics
            anomaly_metrics = self._get_anomaly_metrics(start_time)
            
            return MLHealthMetrics(
                timestamp=now_utc(),
                avg_model_accuracy=model_metrics.get('avg_accuracy', 0.0),
                avg_prediction_confidence=prediction_metrics.get('avg_confidence', 0.0),
                models_trained_24h=model_metrics.get('count', 0),
                predictions_made_24h=prediction_metrics.get('count', 0),
                ml_influence_avg=decision_metrics.get('avg_influence', 0.0),
                ml_overrides_count=decision_metrics.get('overrides', 0),
                ml_agreement_rate=decision_metrics.get('agreement_rate', 0.0),
                anomalies_detected_24h=anomaly_metrics.get('count', 0),
                data_quality_score=anomaly_metrics.get('quality_score', 1.0),
                error_rate=prediction_metrics.get('error_rate', 0.0),
                avg_execution_time_ms=prediction_metrics.get('avg_time_ms', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting health metrics: {e}")
            return MLHealthMetrics(
                timestamp=now_utc(),
                avg_model_accuracy=0.0,
                avg_prediction_confidence=0.0,
                models_trained_24h=0,
                predictions_made_24h=0,
                ml_influence_avg=0.0,
                ml_overrides_count=0,
                ml_agreement_rate=0.0,
                anomalies_detected_24h=0,
                data_quality_score=0.0,
                error_rate=1.0,
                avg_execution_time_ms=0.0
            )
    
    def _get_model_metrics(self, since: str) -> Dict[str, Any]:
        """Get model training metrics."""
        try:
            # Try to get model metrics - performance_metrics may not exist in schema
            try:
                response = self.client.table('ml_models').select(
                    'id, trained_at'
                ).gte('trained_at', since).execute()
            except Exception as e:
                # If column doesn't exist, try without it
                self.logger.warning(f"Could not query performance_metrics column: {e}")
                response = self.client.table('ml_models').select(
                    'id, trained_at'
                ).gte('trained_at', since).execute()
            
            if not response.data:
                return {'avg_accuracy': 0.0, 'count': 0}
            
            # Extract accuracy from model data if available
            # Note: performance_metrics column may not exist in schema
            accuracies = []
            for m in response.data:
                # Try to get performance_metrics if column exists
                try:
                    perf_response = self.client.table('ml_models').select('performance_metrics').eq('id', m['id']).execute()
                    if perf_response.data and perf_response.data[0].get('performance_metrics'):
                        perf_metrics = perf_response.data[0]['performance_metrics']
                        if isinstance(perf_metrics, dict):
                            # Use test_r2 or cv_score as accuracy proxy
                            accuracy = perf_metrics.get('test_r2') or perf_metrics.get('cv_score', 0)
                            if accuracy and not (np.isnan(accuracy) or np.isinf(accuracy)):
                                accuracies.append(float(accuracy))
                except Exception:
                    # Column doesn't exist or is not accessible, skip
                    pass
            
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            # Ensure accuracy is never negative (R2 can be negative for poor models)
            avg_accuracy = max(0.0, avg_accuracy)
            
            return {
                'avg_accuracy': avg_accuracy,
                'count': len(response.data)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model metrics: {e}")
            return {'avg_accuracy': 0.0, 'count': 0}
    
    def _get_prediction_metrics(self, since: str) -> Dict[str, Any]:
        """Get prediction metrics."""
        try:
            response = self.client.table('ml_predictions').select(
                'id, confidence_score, created_at'
            ).gte('created_at', since).execute()
            
            if not response.data:
                return {'avg_confidence': 0.0, 'count': 0, 'error_rate': 0.0, 'avg_time_ms': 0.0}
            
            confidences = [p.get('confidence_score', 0) for p in response.data]
            
            return {
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'count': len(response.data),
                'error_rate': 0.0,  # Would need error tracking
                'avg_time_ms': 0.0  # Would need timing data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting prediction metrics: {e}")
            return {'avg_confidence': 0.0, 'count': 0, 'error_rate': 1.0, 'avg_time_ms': 0.0}
    
    def _get_decision_metrics(self, since: str) -> Dict[str, Any]:
        """Get decision influence metrics."""
        try:
            # This would need a new table tracking ML decisions vs rule decisions
            # For now, return placeholders
            return {
                'avg_influence': 0.5,
                'overrides': 0,
                'agreement_rate': 0.8
            }
            
        except Exception as e:
            self.logger.error(f"Error getting decision metrics: {e}")
            return {'avg_influence': 0.0, 'overrides': 0, 'agreement_rate': 0.0}
    
    def _get_anomaly_metrics(self, since: str) -> Dict[str, Any]:
        """Get anomaly detection metrics."""
        try:
            # Check performance_metrics for anomalies flagged
            response = self.client.table('performance_metrics').select(
                'id, created_at'
            ).gte('created_at', since).execute()
            
            total_records = len(response.data) if response.data else 1
            
            # Would need anomaly tracking in DB
            return {
                'count': 0,
                'quality_score': 1.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting anomaly metrics: {e}")
            return {'count': 0, 'quality_score': 0.0}
    
    def generate_summary_report(self) -> str:
        """Generate human-readable summary of ML system."""
        metrics = self.get_health_metrics()
        
        report = [
            "🤖 **ML SYSTEM HEALTH REPORT**",
            f"   Generated: {metrics.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "📊 **Model Performance (24h)**",
            f"   • Average Accuracy: {metrics.avg_model_accuracy:.1%}",
            f"   • Models Trained: {metrics.models_trained_24h}",
            f"   • Predictions Made: {metrics.predictions_made_24h}",
            f"   • Avg Confidence: {metrics.avg_prediction_confidence:.1%}",
            "",
            "🎯 **Decision Influence (24h)**",
            f"   • ML Influence: {metrics.ml_influence_avg:.1%}",
            f"   • ML Overrides: {metrics.ml_overrides_count}",
            f"   • Agreement Rate: {metrics.ml_agreement_rate:.1%}",
            "",
            "⚠️ **Data Quality (24h)**",
            f"   • Anomalies Detected: {metrics.anomalies_detected_24h}",
            f"   • Quality Score: {metrics.data_quality_score:.1%}",
            "",
            "⚡ **System Performance**",
            f"   • Error Rate: {metrics.error_rate:.1%}",
            f"   • Avg Execution: {metrics.avg_execution_time_ms:.0f}ms",
        ]
        
        # Add health status
        if metrics.avg_model_accuracy > 0.7 and metrics.error_rate < 0.1:
            report.append("")
            report.append("✅ **Status: HEALTHY**")
        elif metrics.avg_model_accuracy > 0.5:
            report.append("")
            report.append("⚠️ **Status: DEGRADED**")
        else:
            report.append("")
            report.append("❌ **Status: UNHEALTHY**")
        
        return "\n".join(report)
    
    def get_model_accuracy_trend(self, days_back: int = 30) -> pd.DataFrame:
        """Get model accuracy over time."""
        try:
            start_time = (now_utc() - timedelta(days=days_back)).isoformat()
            
            response = self.client.table('ml_models').select(
                'model_type, stage, accuracy_score, trained_at'
            ).gte('trained_at', start_time).order('trained_at').execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df['trained_at'] = pd.to_datetime(df['trained_at'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting accuracy trend: {e}")
            return pd.DataFrame()
    
    def get_prediction_volume_trend(self, days_back: int = 30) -> pd.DataFrame:
        """Get prediction volume over time."""
        try:
            start_time = (now_utc() - timedelta(days=days_back)).isoformat()
            
            response = self.client.table('ml_predictions').select(
                'stage, created_at'
            ).gte('created_at', start_time).execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['date'] = df['created_at'].dt.date
            
            # Count by date and stage
            trend = df.groupby(['date', 'stage']).size().reset_index(name='predictions')
            
            return trend
            
        except Exception as e:
            self.logger.error(f"Error getting prediction trend: {e}")
            return pd.DataFrame()

# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_ml_dashboard(supabase_url: str, supabase_key: str) -> MLDashboard:
    """Create ML dashboard."""
    return MLDashboard(supabase_url, supabase_key)

def get_ml_system_health(dashboard: MLDashboard) -> MLHealthMetrics:
    """Get current ML system health."""
    return dashboard.get_health_metrics()

def generate_ml_summary(dashboard: MLDashboard) -> str:
    """Generate ML system summary report."""
    return dashboard.generate_summary_report()
