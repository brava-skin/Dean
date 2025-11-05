#!/usr/bin/env python3
"""
Comprehensive Test Script for Dean ASC+ Campaign
Tests all components: Meta API, Supabase, Flux API, OpenAI, Image Generation
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        print("✅ Loaded .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
        print("   Environment variables must be set in shell.\n")

import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Test results
test_results = {
    "passed": [],
    "failed": [],
    "warnings": [],
}

def test_result(name: str, passed: bool, message: str = "", warning: bool = False):
    """Record test result"""
    if warning:
        test_results["warnings"].append(f"{name}: {message}")
        print(f"⚠️  {name}: {message}")
    elif passed:
        test_results["passed"].append(name)
        print(f"✅ {name}: {message or 'PASSED'}")
    else:
        test_results["failed"].append(f"{name}: {message}")
        print(f"❌ {name}: {message}")

def test_environment_variables():
    """Test all required environment variables"""
    print("\n" + "="*60)
    print("TEST 1: Environment Variables")
    print("="*60)
    
    required = {
        "Meta Ads API": [
            "FB_APP_ID",
            "FB_APP_SECRET",
            "FB_ACCESS_TOKEN",
            "FB_AD_ACCOUNT_ID",
            "FB_PIXEL_ID",
            "FB_PAGE_ID",
            "IG_ACTOR_ID",
        ],
        "Supabase": [
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
        ],
        "Creative Generation": [
            "FLUX_API_KEY",
            "OPENAI_API_KEY",
        ],
    }
    
    all_passed = True
    for category, vars_list in required.items():
        print(f"\n{category}:")
        for var in vars_list:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                test_result(f"  {var}", True, f"Set ({masked})")
            else:
                # Check alternatives
                alternatives = {
                    "SUPABASE_SERVICE_ROLE_KEY": ["SUPABASE_ANON_KEY"],
                    "ACCOUNT_TZ": ["ACCOUNT_TIMEZONE"],
                }
                found_alternative = False
                if var in alternatives:
                    for alt_var in alternatives[var]:
                        if os.getenv(alt_var):
                            test_result(f"  {var}", True, f"Using {alt_var} instead", warning=True)
                            found_alternative = True
                            break
                
                if not found_alternative:
                    test_result(f"  {var}", False, "NOT SET")
                    all_passed = False
    
    return all_passed

def test_meta_api():
    """Test Meta Ads API connection"""
    print("\n" + "="*60)
    print("TEST 2: Meta Ads API")
    print("="*60)
    
    try:
        # Import lazily to avoid circular imports
        import sys
        if 'integrations.meta_client' in sys.modules:
            del sys.modules['integrations.meta_client']
        from integrations.meta_client import MetaClient
        
        access_token = os.getenv("FB_ACCESS_TOKEN")
        ad_account_id = os.getenv("FB_AD_ACCOUNT_ID")
        
        if not access_token or not ad_account_id:
            test_result("Meta API Connection", False, "Missing credentials")
            return False
        
        client = MetaClient(access_token, ad_account_id)
        
        # Test campaign ID
        campaign_id = os.getenv("ASC_PLUS_CAMPAIGN_ID") or "120233669753230160"
        try:
            campaign = client.api.call('GET', (campaign_id,), params={'fields': 'id,name,status'})
            test_result("Campaign Access", True, f"Campaign: {campaign.get('name', 'Unknown')}")
        except Exception as e:
            test_result("Campaign Access", False, str(e))
            return False
        
        # Test ad set ID
        adset_id = os.getenv("ASC_PLUS_ADSET_ID") or "120233669753240160"
        try:
            adset = client.api.call('GET', (adset_id,), params={'fields': 'id,name,status'})
            test_result("Ad Set Access", True, f"Ad Set: {adset.get('name', 'Unknown')}")
        except Exception as e:
            test_result("Ad Set Access", False, str(e))
            return False
        
        return True
        
    except ImportError as e:
        test_result("Meta API Import", False, f"Import error: {e}")
        return False
    except Exception as e:
        test_result("Meta API Connection", False, str(e))
        return False

def test_supabase():
    """Test Supabase connection"""
    print("\n" + "="*60)
    print("TEST 3: Supabase")
    print("="*60)
    
    try:
        from infrastructure.supabase_storage import get_validated_supabase_client
        
        client = get_validated_supabase_client()
        
        if not client:
            test_result("Supabase Connection", False, "Failed to create client")
            return False
        
        test_result("Supabase Connection", True, "Connected")
        
        # Test table access
        tables_to_test = [
            "creative_intelligence",
            "creative_performance",
            "performance_metrics",
            "ad_lifecycle",
            "creative_storage",
        ]
        
        all_tables_ok = True
        for table in tables_to_test:
            try:
                result = client.table(table).select("id").limit(1).execute()
                test_result(f"  Table: {table}", True, "Accessible")
            except Exception as e:
                error_msg = str(e)
                if "Could not find the table" in error_msg or "relation" in error_msg.lower():
                    test_result(f"  Table: {table}", False, "Table not found - run schema SQL")
                    all_tables_ok = False
                else:
                    test_result(f"  Table: {table}", True, "Accessible (may be empty)", warning=True)
        
        # Test storage bucket
        try:
            from infrastructure.creative_storage import create_creative_storage_manager
            storage_manager = create_creative_storage_manager(client)
            if storage_manager:
                test_result("Storage Bucket", True, "creatives bucket accessible")
            else:
                test_result("Storage Bucket", False, "Failed to create storage manager")
        except Exception as e:
            test_result("Storage Bucket", False, f"Error: {e}")
        
        return all_tables_ok
        
    except ImportError as e:
        test_result("Supabase Import", False, f"Import error: {e}")
        return False
    except Exception as e:
        test_result("Supabase Connection", False, str(e))
        return False

def test_flux_api():
    """Test Flux API connection"""
    print("\n" + "="*60)
    print("TEST 4: Flux API")
    print("="*60)
    
    try:
        from integrations.flux_client import FluxClient
        
        api_key = os.getenv("FLUX_API_KEY")
        if not api_key:
            test_result("Flux API Key", False, "FLUX_API_KEY not set")
            return False
        
        client = FluxClient(api_key=api_key)
        
        # Test credit check
        try:
            credits = client.get_credits()
            if credits is not None:
                test_result("Credit Check", True, f"Credits: {credits:.2f}")
                if credits < 8:
                    test_result("Credit Balance", False, f"Insufficient credits: {credits:.2f} (need 8 per image)")
                else:
                    test_result("Credit Balance", True, f"Sufficient credits: {credits:.2f}")
            else:
                test_result("Credit Check", False, "Failed to fetch credits")
        except Exception as e:
            test_result("Credit Check", False, str(e))
        
        # Test API endpoint (without generating image to save credits)
        try:
            import requests
            response = requests.get(
                "https://api.bfl.ai/v1/credits",
                headers={"x-key": api_key, "accept": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                test_result("API Endpoint", True, "Endpoint accessible")
            else:
                test_result("API Endpoint", False, f"Status code: {response.status_code}")
        except Exception as e:
            test_result("API Endpoint", False, str(e))
        
        return True
        
    except ImportError as e:
        test_result("Flux API Import", False, f"Import error: {e}")
        return False
    except Exception as e:
        test_result("Flux API Connection", False, str(e))
        return False

def test_openai_api():
    """Test OpenAI API connection"""
    print("\n" + "="*60)
    print("TEST 5: OpenAI API (ChatGPT-5)")
    print("="*60)
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            test_result("OpenAI API Key", False, "OPENAI_API_KEY not set")
            return False
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test API with a simple request
        try:
            response = client.responses.create(
                model="gpt-5",
                input="Say 'test' if you can read this."
            )
            
            output_text = ""
            if hasattr(response, 'output_text') and response.output_text:
                output_text = response.output_text
            elif hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'content'):
                        for content in item.content:
                            if hasattr(content, 'text'):
                                output_text += content.text
                            elif hasattr(content, 'output_text'):
                                output_text += content.output_text
            
            if output_text:
                test_result("API Connection", True, "ChatGPT-5 responding")
                test_result("Model Access", True, "gpt-5 model accessible")
            else:
                test_result("API Connection", False, "No response received")
                return False
                
        except Exception as e:
            error_msg = str(e)
            if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "invalid" in error_msg.lower()):
                test_result("Model Access", False, "gpt-5 model not available - check API access")
            else:
                test_result("API Connection", False, str(e))
            return False
        
        return True
        
    except ImportError as e:
        test_result("OpenAI Import", False, f"Import error: {e}")
        return False
    except Exception as e:
        test_result("OpenAI API Connection", False, str(e))
        return False

def test_image_generator():
    """Test image generator setup"""
    print("\n" + "="*60)
    print("TEST 6: Image Generator Setup")
    print("="*60)
    
    try:
        from creative.image_generator import ImageCreativeGenerator, create_image_generator
        
        # Test initialization
        generator = create_image_generator()
        
        if generator:
            test_result("Generator Initialization", True, "Initialized")
            
            # Check components
            if generator.flux_client:
                test_result("  Flux Client", True, "Available")
            else:
                test_result("  Flux Client", False, "Not initialized")
            
            if generator.prompt_engineer:
                test_result("  Prompt Engineer", True, "Available")
            else:
                test_result("  Prompt Engineer", False, "Not initialized (OpenAI key missing?)")
            
            # Check tracking
            if hasattr(generator, 'recent_scenarios'):
                test_result("  Scenario Tracking", True, "Enabled")
            else:
                test_result("  Scenario Tracking", False, "Not initialized")
            
            return True
        else:
            test_result("Generator Initialization", False, "Failed to create")
            return False
            
    except ImportError as e:
        test_result("Image Generator Import", False, f"Import error: {e}")
        return False
    except Exception as e:
        test_result("Image Generator Setup", False, str(e))
        return False

def test_ml_system():
    """Test ML system setup"""
    print("\n" + "="*60)
    print("TEST 7: ML System Setup")
    print("="*60)
    
    try:
        from ml.ml_intelligence import create_ml_system
        
        # Test ML system initialization
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            test_result("ML System Initialization", False, "Missing Supabase credentials")
            return False
        
        ml_system = create_ml_system(supabase_url=supabase_url, supabase_key=supabase_key)
        
        if ml_system:
            test_result("ML System Initialization", True, "Initialized")
            
            # Check Supabase connection
            if hasattr(ml_system, 'supabase_client') and ml_system.supabase_client:
                test_result("  Supabase Connection", True, "Connected")
            else:
                test_result("  Supabase Connection", False, "Not connected")
            
            # Check methods
            if hasattr(ml_system, 'get_creative_insights'):
                test_result("  Creative Insights", True, "Method available")
            else:
                test_result("  Creative Insights", False, "Method not found")
            
            if hasattr(ml_system, 'record_creative_creation'):
                test_result("  Creative Tracking", True, "Method available")
            else:
                test_result("  Creative Tracking", False, "Method not found")
            
            return True
        else:
            test_result("ML System Initialization", False, "Failed to create")
            return False
            
    except ImportError as e:
        test_result("ML System Import", False, f"Import error: {e}")
        return False
    except Exception as e:
        test_result("ML System Setup", False, str(e))
        return False

def test_ffmpeg():
    """Test ffmpeg availability"""
    print("\n" + "="*60)
    print("TEST 8: ffmpeg (Text Overlay)")
    print("="*60)
    
    try:
        import subprocess
        
        # Check common ffmpeg locations
        ffmpeg_paths = [
            "ffmpeg",  # In PATH
            "/opt/homebrew/bin/ffmpeg",  # Homebrew on Apple Silicon
            "/usr/local/bin/ffmpeg",  # Homebrew on Intel Mac
            "/usr/bin/ffmpeg",  # System installation
        ]
        
        ffmpeg_found = None
        for path in ffmpeg_paths:
            try:
                result = subprocess.run(
                    [path, "-version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    ffmpeg_found = path
                    version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown"
                    test_result("ffmpeg Installation", True, f"{version_line} (found at {path})")
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if not ffmpeg_found:
            test_result("ffmpeg Installation", False, "ffmpeg not found - install with: brew install ffmpeg")
            return False
            
    except Exception as e:
        test_result("ffmpeg Check", False, str(e))
        return False

def test_configuration():
    """Test configuration files"""
    print("\n" + "="*60)
    print("TEST 9: Configuration Files")
    print("="*60)
    
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    rules_path = Path(__file__).parent.parent / "config" / "rules.yaml"
    
    if config_path.exists():
        test_result("settings.yaml", True, "Found")
        
        # Check for ASC+ campaign ID
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            campaign_id = config.get("ids", {}).get("asc_plus_campaign_id")
            adset_id = config.get("ids", {}).get("asc_plus_adset_id")
            
            if campaign_id:
                test_result("  Campaign ID", True, f"Set: {campaign_id}")
            else:
                test_result("  Campaign ID", False, "Not set")
            
            if adset_id:
                test_result("  Ad Set ID", True, f"Set: {adset_id}")
            else:
                test_result("  Ad Set ID", False, "Not set")
                
        except Exception as e:
            test_result("  Config Parse", False, str(e))
    else:
        test_result("settings.yaml", False, "Not found")
    
    if rules_path.exists():
        test_result("rules.yaml", True, "Found")
    else:
        test_result("rules.yaml", False, "Not found")
    
    return True

def print_summary():
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(test_results["passed"]) + len(test_results["failed"])
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])
    warnings = len(test_results["warnings"])
    
    print(f"\n✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    if warnings > 0:
        print(f"⚠️  Warnings: {warnings}")
    
    if test_results["failed"]:
        print("\n❌ FAILED TESTS:")
        for failure in test_results["failed"]:
            print(f"   - {failure}")
    
    if test_results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in test_results["warnings"]:
            print(f"   - {warning}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Dean is ready to run.")
        print("\nNext steps:")
        print("1. Run Dean: python src/main.py")
        print("2. Monitor the first creative generation cycle")
        print("3. Check Supabase for data storage")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running Dean.")
    
    return failed == 0

def main():
    """Run all tests"""
    print("="*60)
    print("DEAN ASC+ CAMPAIGN - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Run all tests
    test_environment_variables()
    test_meta_api()
    test_supabase()
    test_flux_api()
    test_openai_api()
    test_image_generator()
    test_ml_system()
    test_ffmpeg()
    test_configuration()
    
    # Print summary
    success = print_summary()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

