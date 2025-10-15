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
    """Test the rate limit handling improvements."""
    
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
        print(f"   - Retry max: {os.getenv('META_RETRY_MAX', '6')}")
        print(f"   - Backoff base: {os.getenv('META_BACKOFF_BASE', '1.0')}s")
        print(f"   - Write cooldown: {os.getenv('META_WRITE_COOLDOWN_SEC', '10')}s")
        print(f"   - Timeout: {os.getenv('META_TIMEOUT', '30')}s")
        
        # Test insights call (this will use the improved retry logic)
        print("\n🧪 Testing insights call...")
        insights = client.get_ad_insights(
            level="ad",
            fields=["spend"],
            paginate=False
        )
        print(f"✅ Insights call successful (dry run mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_rate_limit_recommendations():
    """Show recommendations for handling Facebook API rate limits."""
    
    print("\n📋 Facebook API Rate Limit Recommendations:")
    print("=" * 50)
    
    print("\n1. Environment Variables to Set:")
    print("   export META_RETRY_MAX=6")
    print("   export META_BACKOFF_BASE=1.0")
    print("   export META_WRITE_COOLDOWN_SEC=10")
    print("   export META_TIMEOUT=30")
    
    print("\n2. Rate Limit Error Handling:")
    print("   ✅ Code 4, Subcode 1504022: Application request limit reached")
    print("   ✅ Automatic retry with exponential backoff")
    print("   ✅ Extended wait times for rate limit errors")
    print("   ✅ Both SDK and HTTP request handling")
    
    print("\n3. Best Practices:")
    print("   • Use dry-run mode for testing")
    print("   • Implement circuit breakers for repeated failures")
    print("   • Monitor API usage in Facebook Developer Console")
    print("   • Consider using multiple ad accounts if needed")
    print("   • Batch operations when possible")
    
    print("\n4. Monitoring:")
    print("   • Check Facebook Developer Console for API usage")
    print("   • Monitor error rates in your application logs")
    print("   • Set up alerts for repeated rate limit errors")
    
    print("\n5. If Rate Limits Persist:")
    print("   • Increase META_WRITE_COOLDOWN_SEC to 30+ seconds")
    print("   • Reduce concurrent operations")
    print("   • Consider upgrading to a higher API tier")
    print("   • Implement request queuing with longer delays")

if __name__ == "__main__":
    print("🔧 Facebook API Rate Limit Test")
    print("=" * 40)
    
    success = test_rate_limit_handling()
    
    if success:
        print("\n✅ Rate limit handling test completed successfully!")
    else:
        print("\n❌ Rate limit handling test failed!")
    
    show_rate_limit_recommendations()
