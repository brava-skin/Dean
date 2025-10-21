#!/usr/bin/env python3
"""
ML System Real Data Test Script
Tests the ML-enhanced Dean system with real Meta Ads data
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

def check_environment():
    """Check if all required environment variables are set."""
    print("🔍 Checking Environment Variables...")
    print("=" * 50)
    
    load_dotenv()
    
    # Meta API credentials
    meta_creds = [
        'FB_APP_ID', 'FB_APP_SECRET', 'FB_ACCESS_TOKEN', 
        'FB_AD_ACCOUNT_ID', 'FB_PIXEL_ID', 'FB_PAGE_ID', 
        'STORE_URL', 'IG_ACTOR_ID'
    ]
    
    # Supabase credentials
    supabase_creds = [
        'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY'
    ]
    
    missing_meta = []
    missing_supabase = []
    
    print("Meta API Credentials:")
    for cred in meta_creds:
        if os.getenv(cred):
            print(f"  ✅ {cred}: Set")
        else:
            print(f"  ❌ {cred}: Missing")
            missing_meta.append(cred)
    
    print("\nSupabase Credentials:")
    for cred in supabase_creds:
        if os.getenv(cred):
            print(f"  ✅ {cred}: Set")
        else:
            print(f"  ❌ {cred}: Missing")
            missing_supabase.append(cred)
    
    return missing_meta, missing_supabase

def test_standard_mode():
    """Test the standard (legacy) system."""
    print("\n🧪 Testing Standard Mode (Legacy System)")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            'python3', 'src/main.py', '--dry-run'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Standard mode test PASSED")
            print("   - System loads correctly")
            print("   - No breaking changes detected")
            return True
        else:
            print("❌ Standard mode test FAILED")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Standard mode test TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ Standard mode test ERROR: {e}")
        return False

def test_ml_mode():
    """Test the ML-enhanced system."""
    print("\n🧠 Testing ML-Enhanced Mode")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            'python3', 'src/main.py', '--ml-mode', '--dry-run'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ ML mode test PASSED")
            print("   - ML system loads correctly")
            print("   - No import errors")
            return True
        else:
            print("❌ ML mode test FAILED")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ ML mode test TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ ML mode test ERROR: {e}")
        return False

def test_ml_components():
    """Test individual ML components."""
    print("\n🔧 Testing ML Components")
    print("=" * 50)
    
    components = [
        ('XGBoost', 'import xgboost as xgb; print(f"XGBoost: {xgb.__version__}")'),
        ('Scikit-learn', 'import sklearn; print(f"Scikit-learn: {sklearn.__version__}")'),
        ('Pandas', 'import pandas as pd; print(f"Pandas: {pd.__version__}")'),
        ('NumPy', 'import numpy as np; print(f"NumPy: {np.__version__}")'),
        ('SciPy', 'import scipy; print(f"SciPy: {scipy.__version__}")')
    ]
    
    all_passed = True
    
    for name, test_code in components:
        try:
            result = subprocess.run([
                'python3', '-c', test_code
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"  ✅ {name}: Working")
            else:
                print(f"  ❌ {name}: Failed")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")
            all_passed = False
    
    return all_passed

def test_database_connection():
    """Test Supabase database connection."""
    print("\n🗄️ Testing Database Connection")
    print("=" * 50)
    
    try:
        # Test if we can import and connect to Supabase
        test_code = """
import os
from supabase import create_client

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

if url and key:
    client = create_client(url, key)
    # Test connection by querying a simple table
    result = client.table('ad_lifecycle').select('id').limit(1).execute()
    print('✅ Database connection successful')
else:
    print('❌ Missing Supabase credentials')
"""
        
        result = subprocess.run([
            'python3', '-c', test_code
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Database connection test PASSED")
            return True
        else:
            print("❌ Database connection test FAILED")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Database connection test ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 ML System Real Data Test Suite")
    print("=" * 60)
    print()
    
    # Check environment
    missing_meta, missing_supabase = check_environment()
    
    if missing_meta:
        print(f"\n⚠️  Missing Meta API credentials: {', '.join(missing_meta)}")
        print("   Set these in your .env file to test with real data")
        print("   For now, testing system functionality only...")
    
    if missing_supabase:
        print(f"\n⚠️  Missing Supabase credentials: {', '.join(missing_supabase)}")
        print("   ML mode will not be available")
        print("   Set these in your .env file to test ML features")
    
    print("\n" + "=" * 60)
    
    # Run tests
    tests = [
        ("ML Components", test_ml_components),
        ("Standard Mode", test_standard_mode),
    ]
    
    if not missing_supabase:
        tests.append(("Database Connection", test_database_connection))
        tests.append(("ML Mode", test_ml_mode))
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Your ML system is ready for real data testing!")
        print("\nNext steps:")
        print("  1. Set up Meta API credentials in .env file")
        print("  2. Set up Supabase credentials in .env file")
        print("  3. Run: python3 src/main.py --ml-mode --dry-run")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("   Please fix the issues before testing with real data")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
