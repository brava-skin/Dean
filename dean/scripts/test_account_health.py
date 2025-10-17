#!/usr/bin/env python3
"""
Test script for ad account health monitoring functionality.
This script demonstrates how the new account health checks work.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meta_client import MetaClient, AccountAuth, ClientConfig
from main import check_ad_account_health
import yaml

def test_account_health():
    """Test the account health monitoring functionality."""
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "rules.yaml"
    with open(config_path, 'r') as f:
        settings = yaml.safe_load(f)
    
    # Create a mock client for testing (dry run mode)
    account_auth = AccountAuth(
        account_id=os.getenv("META_ACCOUNT_ID", "act_123456789"),
        access_token=os.getenv("META_ACCESS_TOKEN", "test_token"),
        app_id=os.getenv("META_APP_ID", "test_app"),
        app_secret=os.getenv("META_APP_SECRET", "test_secret")
    )
    
    client = MetaClient(
        accounts=[account_auth],
        dry_run=True,  # Use dry run mode for testing
        tenant_id="test"
    )
    
    print("🔍 Testing Ad Account Health Monitoring")
    print("=" * 50)
    
    # Test 1: Basic health check
    print("\n1. Testing basic account health check...")
    health_result = client.check_account_health()
    print(f"   Health check result: {health_result['ok']}")
    print(f"   Health details: {health_result.get('health_details', {})}")
    
    # Test 2: Integrated health check with alerting
    print("\n2. Testing integrated health check with alerting...")
    account_health = check_ad_account_health(client, settings)
    print(f"   Account health result: {account_health['ok']}")
    if account_health.get('critical_issues'):
        print(f"   Critical issues: {account_health['critical_issues']}")
    if account_health.get('warnings'):
        print(f"   Warnings: {account_health['warnings']}")
    
    # Test 3: Configuration validation
    print("\n3. Testing configuration...")
    account_health_config = settings.get("account_health", {})
    print(f"   Account health monitoring enabled: {account_health_config.get('enabled', True)}")
    print(f"   Alert thresholds: {account_health_config.get('thresholds', {})}")
    
    print("\n✅ Account health monitoring test completed!")
    print("\n📋 What this monitoring provides:")
    print("   • Payment method status checks")
    print("   • Account balance monitoring")
    print("   • Spend cap warnings")
    print("   • Business verification status")
    print("   • Account status validation")
    print("   • Critical issue alerting via Slack")
    
    print("\n🚨 Alert Types:")
    print("   • Critical: Account disabled, payment failures")
    print("   • Warning: Low balance, approaching spend cap")
    print("   • Info: Business verification issues")
    
    return True

if __name__ == "__main__":
    try:
        test_account_health()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
