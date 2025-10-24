"""
DEAN INTEGRATIONS SYSTEM
External service integrations

This package contains:
- meta_client: Meta API client
- slack: Slack notifications and alerts
"""

from .meta_client import MetaClient, ClientConfig, AccountAuth

# Import slack functions explicitly to avoid wildcard import issues
from .slack import (
    notify, post_run_header_and_get_thread_ts, post_thread_ads_snapshot, 
    prettify_ad_name, fmt_eur, fmt_pct, fmt_roas, fmt_int,
    build_ads_snapshot,
    alert_kill, alert_promote, alert_scale, alert_fatigue, alert_data_quality,
    alert_error, alert_insights_warning, alert_queue_empty, alert_new_launch,
    alert_system_health, alert_ad_account_health_critical, alert_ad_account_health_warning,
    alert_payment_issue, alert_account_balance_low, alert_threshold_updated,
    alert_spend_cap_approaching, alert_budget_alert
)

__all__ = [
    'MetaClient', 'ClientConfig', 'AccountAuth',
    # Slack functions
    'notify', 'post_run_header_and_get_thread_ts', 'post_thread_ads_snapshot', 
    'prettify_ad_name', 'fmt_eur', 'fmt_pct', 'fmt_roas', 'fmt_int',
    'build_ads_snapshot',
    # Alert functions
    'alert_kill', 'alert_promote', 'alert_scale', 'alert_fatigue', 'alert_data_quality',
    'alert_error', 'alert_insights_warning', 'alert_queue_empty', 'alert_new_launch',
    'alert_system_health', 'alert_ad_account_health_critical', 'alert_ad_account_health_warning',
    'alert_payment_issue', 'alert_account_balance_low', 'alert_threshold_updated',
    'alert_spend_cap_approaching', 'alert_budget_alert'
]
