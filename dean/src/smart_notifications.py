"""
Smart Slack Notification Manager
Reduces spam while maintaining rich content
"""

import yaml
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pytz
from pathlib import Path

class SmartNotificationManager:
    def __init__(self, config_path: str = "config/slack_notifications.yaml"):
        """Initialize smart notification manager."""
        self.config_path = config_path
        self.config = self._load_config()
        self.message_history = []
        self.last_notifications = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load notification configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration if file doesn't exist
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'notifications': {
                'daily_summary': {'enabled': True, 'frequency': 'once_per_day', 'time': '09:00'},
                'ml_learning': {'enabled': True, 'frequency': 'every_6_hours'},
                'performance_alerts': {'enabled': True, 'frequency': 'immediate'},
                'data_collection': {'enabled': True, 'frequency': 'every_2_hours'}
            },
            'content_filters': {
                'skip_messages': ['data_stored', 'ml_analysis_completed', 'performance_data_inserted'],
                'keep_messages': ['ml_health_report', 'performance_summary', 'critical_alerts']
            },
            'aggregation': {
                'group_similar': True,
                'max_messages_per_hour': 3,
                'batch_window': '5_minutes'
            }
        }
    
    def should_send_notification(self, message_type: str, content: str) -> bool:
        """Determine if notification should be sent based on smart rules."""
        now = datetime.now(pytz.timezone('Europe/Amsterdam'))
        
        # Check if message type should be skipped
        if message_type in self.config.get('content_filters', {}).get('skip_messages', []):
            return False
        
        # Check frequency limits
        if not self._check_frequency_limits(message_type, now):
            return False
        
        # Check quiet hours
        if self._is_quiet_hours(now):
            return self._is_important_message(message_type)
        
        # Check aggregation limits
        if not self._check_aggregation_limits(now):
            return False
        
        return True
    
    def _check_frequency_limits(self, message_type: str, now: datetime) -> bool:
        """Check if message type respects frequency limits."""
        notifications = self.config.get('notifications', {})
        
        if message_type not in notifications:
            return True
        
        config = notifications[message_type]
        frequency = config.get('frequency', 'immediate')
        
        if frequency == 'immediate':
            return True
        elif frequency == 'once_per_day':
            return self._check_daily_limit(message_type, now)
        elif frequency == 'every_6_hours':
            return self._check_hourly_limit(message_type, now, 6)
        elif frequency == 'every_2_hours':
            return self._check_hourly_limit(message_type, now, 2)
        
        return True
    
    def _check_daily_limit(self, message_type: str, now: datetime) -> bool:
        """Check if daily limit is respected."""
        today = now.date()
        last_sent = self.last_notifications.get(f"{message_type}_daily")
        
        if last_sent is None or last_sent.date() < today:
            self.last_notifications[f"{message_type}_daily"] = now
            return True
        
        return False
    
    def _check_hourly_limit(self, message_type: str, now: datetime, hours: int) -> bool:
        """Check if hourly limit is respected."""
        last_sent = self.last_notifications.get(f"{message_type}_hourly")
        
        if last_sent is None or (now - last_sent).total_seconds() >= hours * 3600:
            self.last_notifications[f"{message_type}_hourly"] = now
            return True
        
        return False
    
    def _is_quiet_hours(self, now: datetime) -> bool:
        """Check if current time is in quiet hours."""
        time_controls = self.config.get('time_controls', {})
        quiet_hours = time_controls.get('quiet_hours', {})
        
        if not quiet_hours.get('enabled', False):
            return False
        
        current_hour = now.hour
        start_hour = int(quiet_hours.get('start', '22:00').split(':')[0])
        end_hour = int(quiet_hours.get('end', '08:00').split(':')[0])
        
        if start_hour > end_hour:  # Overnight quiet hours
            return current_hour >= start_hour or current_hour < end_hour
        else:  # Same day quiet hours
            return start_hour <= current_hour < end_hour
    
    def _is_important_message(self, message_type: str) -> bool:
        """Check if message is important enough for quiet hours."""
        important_types = ['critical_alerts', 'ml_health_report', 'performance_summary']
        return message_type in important_types
    
    def _check_aggregation_limits(self, now: datetime) -> bool:
        """Check aggregation limits."""
        aggregation = self.config.get('aggregation', {})
        max_per_hour = aggregation.get('max_messages_per_hour', 3)
        
        # Count messages in last hour
        one_hour_ago = now - timedelta(hours=1)
        recent_messages = [msg for msg in self.message_history if msg['timestamp'] > one_hour_ago]
        
        return len(recent_messages) < max_per_hour
    
    def format_smart_message(self, message_type: str, content: str, data: Dict[str, Any] = None) -> str:
        """Format message with smart templates."""
        templates = self.config.get('templates', {})
        
        if message_type == 'daily_summary' and 'daily_summary' in templates:
            return self._format_daily_summary(data or {})
        elif message_type == 'ml_learning' and 'ml_learning' in templates:
            return self._format_ml_learning(data or {})
        elif message_type == 'critical_alert' and 'critical_alert' in templates:
            return self._format_critical_alert(data or {})
        
        # Default formatting
        return self._format_default_message(message_type, content, data)
    
    def _format_daily_summary(self, data: Dict[str, Any]) -> str:
        """Format daily summary message."""
        template = self.config.get('templates', {}).get('daily_summary', '')
        if not template:
            return self._format_default_message('daily_summary', 'Daily summary', data)
        
        # Fill template with data
        formatted = template.format(
            date=datetime.now(pytz.timezone('Europe/Amsterdam')).strftime('%Y-%m-%d'),
            total_ads=data.get('total_ads', 'N/A'),
            total_spend=data.get('total_spend', 'N/A'),
            total_purchases=data.get('total_purchases', 'N/A'),
            avg_cpa=data.get('avg_cpa', 'N/A'),
            avg_roas=data.get('avg_roas', 'N/A'),
            ml_status=data.get('ml_status', 'N/A'),
            active_models=data.get('active_models', 'N/A'),
            predictions_24h=data.get('predictions_24h', 'N/A'),
            learning_events_24h=data.get('learning_events_24h', 'N/A'),
            key_insights=data.get('key_insights', 'System running smoothly'),
            uptime=data.get('uptime', 'N/A'),
            error_rate=data.get('error_rate', 'N/A'),
            data_points_24h=data.get('data_points_24h', 'N/A')
        )
        
        return formatted
    
    def _format_ml_learning(self, data: Dict[str, Any]) -> str:
        """Format ML learning message."""
        template = self.config.get('templates', {}).get('ml_learning', '')
        if not template:
            return self._format_default_message('ml_learning', 'ML learning update', data)
        
        formatted = template.format(
            time=datetime.now(pytz.timezone('Europe/Amsterdam')).strftime('%H:%M'),
            ml_status=data.get('ml_status', 'N/A'),
            models_trained=data.get('models_trained', 'N/A'),
            recent_training=data.get('recent_training', 'N/A'),
            data_points_24h=data.get('data_points_24h', 'N/A'),
            predictions_24h=data.get('predictions_24h', 'N/A'),
            avg_accuracy=data.get('avg_accuracy', 'N/A'),
            avg_confidence=data.get('avg_confidence', 'N/A')
        )
        
        return formatted
    
    def _format_critical_alert(self, data: Dict[str, Any]) -> str:
        """Format critical alert message."""
        template = self.config.get('templates', {}).get('critical_alert', '')
        if not template:
            return self._format_default_message('critical_alert', 'Critical alert', data)
        
        formatted = template.format(
            time=datetime.now(pytz.timezone('Europe/Amsterdam')).strftime('%H:%M'),
            alert_message=data.get('alert_message', 'Unknown alert'),
            action_required=data.get('action_required', 'Monitor system')
        )
        
        return formatted
    
    def _format_default_message(self, message_type: str, content: str, data: Dict[str, Any] = None) -> str:
        """Format default message."""
        timestamp = datetime.now(pytz.timezone('Europe/Amsterdam')).strftime('%H:%M')
        
        if message_type == 'ml_health_report':
            return f"🤖 **ML Health** ({timestamp}): {content}"
        elif message_type == 'performance_summary':
            return f"📊 **Performance** ({timestamp}): {content}"
        elif message_type == 'critical_alert':
            return f"🚨 **Alert** ({timestamp}): {content}"
        else:
            return f"ℹ️ **{message_type.title()}** ({timestamp}): {content}"
    
    def log_message(self, message_type: str, content: str, sent: bool = False):
        """Log message for tracking."""
        self.message_history.append({
            'timestamp': datetime.now(pytz.timezone('Europe/Amsterdam')),
            'type': message_type,
            'content': content,
            'sent': sent
        })
        
        # Keep only last 100 messages
        if len(self.message_history) > 100:
            self.message_history = self.message_history[-100:]

# Global instance
smart_notifications = SmartNotificationManager()
