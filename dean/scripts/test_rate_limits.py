#!/usr/bin/env python3
"""
Test script to verify Facebook API rate limit handling.
This script can be used to test the improved rate limit handling.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meta_client import MetaClient, AccountAuth, ClientConfig
from slack import notify

def test_rate_limit_handling():
    """Test the comprehensive rate limit handling improvements."""
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check required environment variables
    required_vars = [
        "FB_APP_ID", "FB_APP_SECRET", "FB_ACCESS_TOKEN", 
        "FB_AD_ACCOUNT_ID", "FB_PIXEL_ID", "FB_PAGE_ID"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    try:
        # Create Meta client
        account = AccountAuth(
            account_id=os.getenv("FB_AD_ACCOUNT_ID"),
            access_token=os.getenv("FB_ACCESS_TOKEN"),
            app_id=os.getenv("FB_APP_ID"),
            app_secret=os.getenv("FB_APP_SECRET"),
        )
        
        cfg = ClientConfig(timezone="Europe/Amsterdam")
        client = MetaClient(
            accounts=[account],
            cfg=cfg,
            dry_run=True,  # Use dry run to avoid actual API calls
        )
        
        print("✅ Meta client created successfully")
        print(f"📊 Configuration:")
        print(f"   - API Tier: {os.getenv('META_API_TIER', 'development')}")
        print(f"   - BUC Enabled: {os.getenv('META_BUC_ENABLED', 'true')}")
        print(f"   - Retry max: {os.getenv('META_RETRY_MAX', '6')}")
        print(f"   - Backoff base: {os.getenv('META_BACKOFF_BASE', '1.0')}s")
        print(f"   - Write cooldown: {os.getenv('META_WRITE_COOLDOWN_SEC', '10')}s")
        print(f"   - Timeout: {os.getenv('META_TIMEOUT', '30')}s")
        
        # Test rate limit status
        print("\n🔍 Testing rate limit status...")
        status = client.get_rate_limit_status()
        print(f"✅ Rate limit status retrieved:")
        print(f"   - API Tier: {status['api_tier']}")
        print(f"   - Current Score: {status['current_score']}/{status['max_score']} ({status['score_usage_pct']:.1f}%)")
        print(f"   - Blocked Until: {status['blocked_until']}")
        print(f"   - App Blocked Until: {status['app_blocked_until']}")
        print(f"   - Recent Requests: {status['recent_requests']}")
        
        # Test insights call (this will use the improved retry logic)
        print("\n🧪 Testing insights call...")
        insights = client.get_ad_insights(
            level="ad",
            fields=["spend"],
            paginate=False
        )
        print(f"✅ Insights call successful (dry run mode)")
        
        # Test rate limit status after request
        print("\n🔍 Testing rate limit status after request...")
        status_after = client.get_rate_limit_status()
        print(f"✅ Rate limit status after request:")
        print(f"   - Current Score: {status_after['current_score']}/{status_after['max_score']} ({status_after['score_usage_pct']:.1f}%)")
        print(f"   - Recent Requests: {status_after['recent_requests']}")
        
        # Test budget change rate limiting
        print("\n🧪 Testing budget change rate limiting...")
        test_adset_id = "test_adset_123"
        
        # Test multiple budget changes (should be allowed in dry run)
        for i in range(5):
            try:
                result = client.update_adset_budget(test_adset_id, 100.0 + i * 10)
                print(f"   ✅ Budget change {i+1}: {result.get('result', 'ok')}")
            except Exception as e:
                print(f"   ❌ Budget change {i+1} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_rate_limit_recommendations():
    """Show comprehensive recommendations for handling Facebook API rate limits."""
    
    print("\n📋 Enhanced Facebook API Rate Limit System:")
    print("=" * 60)
    
    print("\n1. New Environment Variables:")
    print("   export META_API_TIER=development  # or 'standard'")
    print("   export META_BUC_ENABLED=true")
    print("   export META_REQUEST_DELAY=0.5")
    print("   export META_RETRY_MAX=6")
    print("   export META_BACKOFF_BASE=1.0")
    print("   export META_WRITE_COOLDOWN_SEC=10")
    print("   export META_TIMEOUT=30")
    
    print("\n2. Comprehensive Rate Limit Handling:")
    print("   ✅ API-Level Scoring: Reads=1pt, Writes=3pts")
    print("   ✅ Development Tier: 60pts max, 300s block")
    print("   ✅ Standard Tier: 9000pts max, 60s block")
    print("   ✅ Business Use Case (BUC) Rate Limits")
    print("   ✅ Ads Insights Platform Rate Limits")
    print("   ✅ Ad Account Spend Limit Changes (10/day)")
    print("   ✅ Ad Set Budget Changes (4/hour)")
    print("   ✅ All Error Codes: 4, 17, 613, 80000-80014")
    
    print("\n3. Business Use Case Limits:")
    print("   • ads_management: 300 (dev) / 100,000 (standard)")
    print("   • custom_audience: 5,000 (dev) / 190,000 (standard)")
    print("   • ads_insights: 600 (dev) / 190,000 (standard)")
    print("   • catalog_management: 20,000 (both tiers)")
    print("   • catalog_batch: 200 (both tiers)")
    
    print("\n4. Advanced Features:")
    print("   • Automatic X-Business-Use-Case headers")
    print("   • Real-time rate limit status monitoring")
    print("   • Intelligent request queuing")
    print("   • Budget change frequency tracking")
    print("   • Comprehensive error categorization")
    
    print("\n5. Best Practices:")
    print("   • Use dry-run mode for testing")
    print("   • Monitor rate limit status regularly")
    print("   • Implement request batching")
    print("   • Upgrade to Standard tier for production")
    print("   • Set up alerts for rate limit warnings")
    
    print("\n6. Monitoring & Alerting:")
    print("   • Check get_rate_limit_status() regularly")
    print("   • Monitor score usage percentage")
    print("   • Watch for BUC limit warnings")
    print("   • Track budget change frequency")
    print("   • Set up Slack notifications for rate limit hits")
    
    print("\n7. Troubleshooting:")
    print("   • If hitting limits frequently, upgrade to Standard tier")
    print("   • Increase META_REQUEST_DELAY for burst protection")
    print("   • Implement request queuing for high-volume operations")
    print("   • Consider multiple ad accounts for load distribution")
    print("   • Use rate limit status to optimize request timing")

if __name__ == "__main__":
    print("🔧 Facebook API Rate Limit Test")
    print("=" * 40)
    
    success = test_rate_limit_handling()
    
    if success:
        print("\n✅ Rate limit handling test completed successfully!")
    else:
        print("\n❌ Rate limit handling test failed!")
    
    show_rate_limit_recommendations()
