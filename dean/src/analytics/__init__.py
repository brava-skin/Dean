"""
DEAN ANALYTICS SYSTEM
Performance tracking and metrics analysis

This package contains:
- performance_tracking: Performance monitoring and fatigue detection
- metrics: Metrics collection and analysis
"""

from .performance_tracking import PerformanceTrackingSystem, create_performance_tracking_system
from .metrics import MetricsConfig, Metrics, metrics_from_row, aggregate_rows, tripwire_threshold_account

__all__ = [
    'PerformanceTrackingSystem', 'create_performance_tracking_system',
    'MetricsConfig', 'Metrics', 'metrics_from_row', 'aggregate_rows', 'tripwire_threshold_account'
]
