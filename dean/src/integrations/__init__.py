"""
DEAN INTEGRATIONS SYSTEM
External service integrations

This package contains:
- meta_client: Meta API client
- slack: Slack notifications and alerts
"""

from .meta_client import MetaClient, ClientConfig, AccountAuth
from .slack import *

__all__ = [
    'MetaClient', 'ClientConfig', 'AccountAuth'
]
