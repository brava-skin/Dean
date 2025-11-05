"""
DEAN INFRASTRUCTURE SYSTEM
Core infrastructure and utilities

This package contains:
- storage: Data storage and persistence
- scheduler: Background task scheduling
- utils: Utility functions
- data_validation: Data validation system
- validated_supabase: Validated Supabase client
"""

from .storage import Store
from .scheduler import BackgroundScheduler, start_background_scheduler, stop_background_scheduler, get_scheduler
from .utils import (
    getenv_f, getenv_i, getenv_b, cfg, cfg_or_env_f, cfg_or_env_i, cfg_or_env_b, cfg_or_env_list,
    safe_f, today_str, daily_key, ad_day_flag_key, now_minute_key, clean_text_token, prettify_ad_name,
    now_utc, now_local, today_ymd_account, yesterday_ymd_account, Clock, RealClock, FixedClock
)

__all__ = [
    'Store', 'BackgroundScheduler',
    'start_background_scheduler', 'stop_background_scheduler', 'get_scheduler',
    'getenv_f', 'getenv_i', 'getenv_b', 'cfg', 'cfg_or_env_f', 'cfg_or_env_i', 'cfg_or_env_b', 'cfg_or_env_list',
    'safe_f', 'today_str', 'daily_key', 'ad_day_flag_key', 'now_minute_key', 'clean_text_token', 'prettify_ad_name',
    'now_utc', 'now_local', 'today_ymd_account', 'yesterday_ymd_account', 'Clock', 'RealClock', 'FixedClock'
]
