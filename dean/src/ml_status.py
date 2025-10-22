"""
DEAN ML SYSTEM STATUS
Simplified ML status reporting for production use
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from supabase import Client

def get_ml_status(supabase_client: Client) -> Dict[str, Any]:
    """Get concise ML system status."""
    try:
        # Get model performance
        models_response = supabase_client.table('ml_models').select('*').eq('is_active', True).execute()
        active_models = len(models_response.data) if models_response.data else 0
        
        # Get predictions count (last 24h)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        predictions_response = supabase_client.table('ml_predictions').select('*').gte('created_at', yesterday).execute()
        predictions_count = len(predictions_response.data) if predictions_response.data else 0
        
        # Get learning events count
        learning_response = supabase_client.table('learning_events').select('*').gte('created_at', yesterday).execute()
        learning_events = len(learning_response.data) if learning_response.data else 0
        
        # Calculate ML health
        ml_health = "HEALTHY" if active_models > 0 else "LEARNING"
        if predictions_count > 0:
            ml_health = "ACTIVE"
        
        return {
            "status": ml_health,
            "active_models": active_models,
            "predictions_24h": predictions_count,
            "learning_events_24h": learning_events,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "active_models": 0,
            "predictions_24h": 0,
            "learning_events_24h": 0,
            "last_updated": datetime.now().isoformat()
        }

def send_ml_status_report(supabase_client: Client, notify_func) -> None:
    """Send concise ML status report."""
    status = get_ml_status(supabase_client)
    
    if status["status"] == "ERROR":
        notify_func(f"🤖 ML Status: {status['status']} - {status.get('error', 'Unknown error')}")
        return
    
    # Create concise status message
    if status["status"] == "ACTIVE":
        message = f"🤖 ML Status: {status['status']} - {status['active_models']} models, {status['predictions_24h']} predictions"
    elif status["status"] == "HEALTHY":
        message = f"🤖 ML Status: {status['status']} - {status['active_models']} models ready"
    else:
        message = f"🤖 ML Status: {status['status']} - {status['learning_events_24h']} learning events"
    
    notify_func(message)
