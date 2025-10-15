"""
Advanced Configuration System for Smart Scheduler.
Provides intelligent configuration management and optimization.
"""

import os
import json
import yaml
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from slack import notify


class AdvancedConfig:
    """
    Advanced configuration system with intelligent features:
    - Auto-optimization based on performance
    - Dynamic scheduling adjustments
    - Smart retry logic
    - Performance-based tuning
    """
    
    def __init__(self, config_path: str = "config/advanced_settings.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.performance_history = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load advanced configuration with defaults."""
        default_config = {
            "scheduling": {
                "smart_timing": True,
                "adaptive_delays": True,
                "peak_hour_avoidance": True,
                "timezone_optimization": True
            },
            "reliability": {
                "max_retries": 3,
                "retry_delay_base": 300,  # 5 minutes
                "retry_delay_multiplier": 2.0,
                "circuit_breaker_threshold": 5,
                "health_check_interval": 1800  # 30 minutes
            },
            "performance": {
                "health_score_threshold": 70,
                "api_latency_threshold": 10.0,
                "db_latency_threshold": 2.0,
                "memory_usage_threshold": 80.0
            },
            "monitoring": {
                "detailed_logging": True,
                "performance_tracking": True,
                "alert_cooldowns": {
                    "queue_empty": 21600,  # 6 hours
                    "system_health": 7200,  # 2 hours
                    "api_errors": 3600,  # 1 hour
                    "db_errors": 1800  # 30 minutes
                }
            },
            "optimization": {
                "auto_tune_retries": True,
                "auto_adjust_delays": True,
                "performance_learning": True,
                "adaptive_thresholds": True
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                # Merge with defaults
                self._deep_merge(default_config, user_config)
            except Exception as e:
                notify(f"⚠️ Failed to load advanced config: {e}, using defaults")
        
        return default_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Deep merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get_smart_retry_delay(self, retry_count: int, base_delay: Optional[int] = None) -> int:
        """Calculate smart retry delay based on performance history."""
        if base_delay is None:
            base_delay = self.config["reliability"]["retry_delay_base"]
        
        multiplier = self.config["reliability"]["retry_delay_multiplier"]
        delay = int(base_delay * (multiplier ** retry_count))
        
        # Apply performance-based adjustments
        if self.config["optimization"]["auto_adjust_delays"]:
            recent_performance = self._get_recent_performance()
            if recent_performance < 0.8:  # Poor performance
                delay = int(delay * 1.5)  # Increase delay
            elif recent_performance > 0.95:  # Excellent performance
                delay = int(delay * 0.8)  # Decrease delay
        
        return min(delay, 3600)  # Cap at 1 hour
    
    def get_optimal_tick_timing(self, current_time: datetime) -> Dict[str, Any]:
        """Get optimal timing for tick execution."""
        timing_config = {
            "should_run": True,
            "delay_seconds": 0,
            "reason": "normal_execution"
        }
        
        if not self.config["scheduling"]["smart_timing"]:
            return timing_config
        
        # Peak hour avoidance
        if self.config["scheduling"]["peak_hour_avoidance"]:
            hour = current_time.hour
            if hour in [9, 10, 11, 14, 15, 16]:  # Peak business hours
                timing_config["delay_seconds"] = 300  # 5 minute delay
                timing_config["reason"] = "peak_hour_delay"
        
        # Timezone optimization
        if self.config["scheduling"]["timezone_optimization"]:
            # Avoid running during likely sleep hours in target timezone
            if 2 <= current_time.hour <= 6:
                timing_config["delay_seconds"] = 600  # 10 minute delay
                timing_config["reason"] = "timezone_optimization"
        
        return timing_config
    
    def should_skip_tick(self, health_score: float, recent_errors: int) -> Tuple[bool, str]:
        """Determine if tick should be skipped based on system health."""
        threshold = self.config["performance"]["health_score_threshold"]
        
        if health_score < threshold:
            return True, f"health_score_below_threshold_{threshold}"
        
        if recent_errors > self.config["reliability"]["circuit_breaker_threshold"]:
            return True, "circuit_breaker_triggered"
        
        return False, "normal_execution"
    
    def get_alert_cooldown(self, alert_type: str) -> int:
        """Get cooldown period for specific alert type."""
        cooldowns = self.config["monitoring"]["alert_cooldowns"]
        return cooldowns.get(alert_type, 3600)  # Default 1 hour
    
    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics for learning."""
        if not self.config["optimization"]["performance_learning"]:
            return
        
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # Keep only last 100 entries
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def _get_recent_performance(self, hours: int = 24) -> float:
        """Get recent performance score (0-1)."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_entries = [
            entry for entry in self.performance_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        if not recent_entries:
            return 1.0  # Default to good performance
        
        # Calculate performance score based on metrics
        total_score = 0
        for entry in recent_entries:
            metrics = entry["metrics"]
            score = 1.0
            
            # Adjust based on health score
            if "health_score" in metrics:
                score *= (metrics["health_score"] / 100)
            
            # Adjust based on success rate
            if "success_rate" in metrics:
                score *= (metrics["success_rate"] / 100)
            
            # Adjust based on latency
            if "api_latency" in metrics:
                if metrics["api_latency"] > 5.0:
                    score *= 0.8
                elif metrics["api_latency"] > 10.0:
                    score *= 0.6
            
            total_score += score
        
        return total_score / len(recent_entries)
    
    def get_adaptive_thresholds(self) -> Dict[str, float]:
        """Get adaptive thresholds based on performance history."""
        if not self.config["optimization"]["adaptive_thresholds"]:
            return {
                "health_score": self.config["performance"]["health_score_threshold"],
                "api_latency": self.config["performance"]["api_latency_threshold"],
                "db_latency": self.config["performance"]["db_latency_threshold"]
            }
        
        recent_performance = self._get_recent_performance()
        
        # Adjust thresholds based on performance
        base_health = self.config["performance"]["health_score_threshold"]
        base_api = self.config["performance"]["api_latency_threshold"]
        base_db = self.config["performance"]["db_latency_threshold"]
        
        if recent_performance < 0.7:  # Poor performance
            return {
                "health_score": base_health * 0.8,  # Lower threshold
                "api_latency": base_api * 1.2,  # Higher tolerance
                "db_latency": base_db * 1.2
            }
        elif recent_performance > 0.9:  # Excellent performance
            return {
                "health_score": base_health * 1.1,  # Higher threshold
                "api_latency": base_api * 0.8,  # Lower tolerance
                "db_latency": base_db * 0.8
            }
        else:
            return {
                "health_score": base_health,
                "api_latency": base_api,
                "db_latency": base_db
            }
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            notify(f"⚠️ Failed to save advanced config: {e}")
    
    def get_configuration_summary(self) -> str:
        """Get human-readable configuration summary."""
        lines = [
            "🔧 Advanced Configuration Summary",
            "",
            "📅 Scheduling:",
            f"  • Smart Timing: {'✅' if self.config['scheduling']['smart_timing'] else '❌'}",
            f"  • Peak Hour Avoidance: {'✅' if self.config['scheduling']['peak_hour_avoidance'] else '❌'}",
            f"  • Timezone Optimization: {'✅' if self.config['scheduling']['timezone_optimization'] else '❌'}",
            "",
            "🛡️ Reliability:",
            f"  • Max Retries: {self.config['reliability']['max_retries']}",
            f"  • Base Retry Delay: {self.config['reliability']['retry_delay_base']}s",
            f"  • Circuit Breaker: {self.config['reliability']['circuit_breaker_threshold']} failures",
            "",
            "📊 Performance:",
            f"  • Health Threshold: {self.config['performance']['health_score_threshold']}/100",
            f"  • API Latency Limit: {self.config['performance']['api_latency_threshold']}s",
            f"  • DB Latency Limit: {self.config['performance']['db_latency_threshold']}s",
            "",
            "🧠 Optimization:",
            f"  • Auto-tune Retries: {'✅' if self.config['optimization']['auto_tune_retries'] else '❌'}",
            f"  • Performance Learning: {'✅' if self.config['optimization']['performance_learning'] else '❌'}",
            f"  • Adaptive Thresholds: {'✅' if self.config['optimization']['adaptive_thresholds'] else '❌'}"
        ]
        
        return "\n".join(lines)
    
    def optimize_configuration(self):
        """Automatically optimize configuration based on performance."""
        if not self.config["optimization"]["performance_learning"]:
            return
        
        recent_performance = self._get_recent_performance()
        
        # Optimize retry settings
        if self.config["optimization"]["auto_tune_retries"]:
            if recent_performance < 0.8:
                # Increase retry count for poor performance
                self.config["reliability"]["max_retries"] = min(
                    self.config["reliability"]["max_retries"] + 1, 5
                )
            elif recent_performance > 0.95:
                # Decrease retry count for excellent performance
                self.config["reliability"]["max_retries"] = max(
                    self.config["reliability"]["max_retries"] - 1, 1
                )
        
        # Optimize thresholds
        if self.config["optimization"]["adaptive_thresholds"]:
            if recent_performance < 0.7:
                # Lower thresholds for poor performance
                self.config["performance"]["health_score_threshold"] = max(
                    self.config["performance"]["health_score_threshold"] - 5, 50
                )
            elif recent_performance > 0.9:
                # Raise thresholds for excellent performance
                self.config["performance"]["health_score_threshold"] = min(
                    self.config["performance"]["health_score_threshold"] + 5, 95
                )
        
        # Save optimized configuration
        self.save_config()
        notify("🔧 Configuration automatically optimized based on performance")
