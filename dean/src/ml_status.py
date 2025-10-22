"""
DEAN ML SYSTEM STATUS
Enhanced ML monitoring with learning insights
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from supabase import Client
import pytz

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
