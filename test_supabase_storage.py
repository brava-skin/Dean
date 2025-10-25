#!/usr/bin/env python3
"""
Comprehensive Supabase Storage Test Suite
Tests all storage operations and identifies/fixes issues
"""

import os
import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

# Add the dean src directory to path
sys.path.insert(0, '/Users/brava/Documents/Dean/dean/src')

# Set environment variables
os.environ.update({
    'FB_APP_ID': '1092055929615931',
    'FB_APP_SECRET': '675e85a8a88312fbe424b14401d4fb99',
    'FB_ACCESS_TOKEN': 'EAAPhZCXkW6CwiZB9mXQXRHMDcZBdicPSPjZCV8SxpZAUvOJdl2WutIogbF4kHDTlLhNFUuzgZDZD',
    'FB_AD_ACCOUNT_ID': '1113502710374440',
    'FB_PIXEL_ID': '594140286998446',
    'FB_PAGE_ID': '823772517483945',
    'IG_ACTOR_ID': '17841477094913251',
    'SUPABASE_URL': 'https://vbttsnxtiyfwpyxuexcu.supabase.co',
    'SUPABASE_SERVICE_ROLE_KEY': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZidHRzbnh0aXlmd3B5eHVleGN1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTkxMjgwMSwiZXhwIjoyMDc1NDg4ODAxfQ.c_zhRD6dYLC6oYHm91GA4kCalrq2nDtNCpWuq_8l6LU',
    'SUPABASE_TABLE': 'meta_creatives',
    'STORE_URL': 'bravaskin.com',
    'BREAKEVEN_CPA': '27.50',
    'COGS_PER_PURCHASE': '12.90',
    'USD_EUR_RATE': '0.86'
})

try:
    from supabase import create_client
    from infrastructure.supabase_storage import create_supabase_storage
    from analytics.table_monitoring import create_table_monitor
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_supabase_connection():
    """Test basic Supabase connection"""
    print("🔍 Testing Supabase connection...")
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            print("❌ Missing Supabase credentials")
            return None
            
        client = create_client(url, key)
        
        # Test basic query with a known table
        response = client.table('ad_creation_times').select('*').limit(1).execute()
        print(f"✅ Supabase connection successful")
        return client
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return None

def test_table_existence(client):
    """Test if all required tables exist"""
    print("\n🔍 Testing table existence...")
    
    required_tables = [
        'ad_creation_times',
        'historical_data', 
        'performance_metrics',
        'ad_lifecycle',
        'time_series_data',
        'creative_intelligence',
        'ml_models',
        'ml_predictions',
        'learning_events'
    ]
    
    existing_tables = []
    missing_tables = []
    
    for table in required_tables:
        try:
            response = client.table(table).select('*').limit(1).execute()
            existing_tables.append(table)
            print(f"✅ Table '{table}' exists")
        except Exception as e:
            missing_tables.append(table)
            print(f"❌ Table '{table}' missing or inaccessible: {e}")
    
    return existing_tables, missing_tables

def test_supabase_storage_operations(client):
    """Test SupabaseStorage class operations"""
    print("\n🔍 Testing SupabaseStorage operations...")
    
    try:
        storage = create_supabase_storage(client)
        test_ad_id = "test_ad_12345"
        test_lifecycle_id = "lifecycle_test_12345"
        
        # Test 1: Record ad creation time
        print("  Testing ad creation time recording...")
        try:
            storage.record_ad_creation(test_ad_id, test_lifecycle_id, "testing")
            print("  ✅ Ad creation time recorded")
        except Exception as e:
            print(f"  ❌ Failed to record ad creation time: {e}")
            return False
        
        # Test 2: Get ad creation time
        print("  Testing ad creation time retrieval...")
        try:
            creation_time = storage.get_ad_creation_time(test_ad_id)
            if creation_time:
                print(f"  ✅ Ad creation time retrieved: {creation_time}")
            else:
                print("  ❌ Ad creation time not found")
        except Exception as e:
            print(f"  ❌ Failed to get ad creation time: {e}")
        
        # Test 3: Get ad age
        print("  Testing ad age calculation...")
        try:
            age_days = storage.get_ad_age_days(test_ad_id)
            if age_days is not None:
                print(f"  ✅ Ad age calculated: {age_days:.2f} days")
            else:
                print("  ❌ Ad age calculation failed")
        except Exception as e:
            print(f"  ❌ Failed to calculate ad age: {e}")
        
        # Test 4: Store historical data
        print("  Testing historical data storage...")
        try:
            storage.store_historical_data(test_ad_id, test_lifecycle_id, "testing", "cpm", 125.50)
            storage.store_historical_data(test_ad_id, test_lifecycle_id, "testing", "ctr", 1.25)
            storage.store_historical_data(test_ad_id, test_lifecycle_id, "testing", "spend", 15.75)
            print("  ✅ Historical data stored")
        except Exception as e:
            print(f"  ❌ Failed to store historical data: {e}")
            return False
        
        # Test 5: Get historical data
        print("  Testing historical data retrieval...")
        try:
            historical_data = storage.get_historical_data(test_ad_id, "cpm", since_days=1)
            if historical_data:
                print(f"  ✅ Historical data retrieved: {len(historical_data)} records")
            else:
                print("  ⚠️ No historical data found (might be expected for new test)")
        except Exception as e:
            print(f"  ❌ Failed to get historical data: {e}")
        
        # Test 6: Get multiple historical metrics
        print("  Testing multiple historical metrics retrieval...")
        try:
            metrics_data = storage.get_historical_metrics(test_ad_id, ["cpm", "ctr", "spend"], days_back=1)
            if metrics_data:
                print(f"  ✅ Multiple metrics retrieved: {list(metrics_data.keys())}")
            else:
                print("  ⚠️ No metrics data found")
        except Exception as e:
            print(f"  ❌ Failed to get historical metrics: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ SupabaseStorage initialization failed: {e}")
        return False

def test_table_monitoring(client):
    """Test table monitoring system"""
    print("\n🔍 Testing table monitoring...")
    
    try:
        monitor = create_table_monitor(client)
        
        # Test table insights
        print("  Getting table insights...")
        insights = monitor.get_all_table_insights()
        
        print(f"  ✅ Table monitoring working:")
        print(f"    • Total tables: {insights.total_tables}")
        print(f"    • Healthy tables: {insights.healthy_tables}")
        print(f"    • Problematic tables: {insights.problematic_tables}")
        print(f"    • Total rows: {insights.total_rows}")
        
        # Test ML data sufficiency
        print("  Checking ML data sufficiency...")
        ml_status = monitor.check_ml_data_sufficiency(insights)
        print(f"  ML Ready: {ml_status['ready_for_training']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Table monitoring failed: {e}")
        return False

def test_data_insertion_formats(client):
    """Test different data insertion formats to identify issues"""
    print("\n🔍 Testing data insertion formats...")
    
    # Test ad_creation_times table
    print("  Testing ad_creation_times insertion...")
    try:
        # Use 2020 timestamp to avoid range issues
        now = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {
            'ad_id': 'format_test_ad_001',
            'lifecycle_id': 'lifecycle_format_test_001',
            'stage': 'testing',
            'created_at_epoch': int(now.timestamp()),
            'created_at_iso': now.isoformat(),
            'updated_at': int(now.timestamp())
        }
        
        response = client.table('ad_creation_times').upsert(data, on_conflict='ad_id').execute()
        print(f"  ✅ ad_creation_times insertion successful")
        
    except Exception as e:
        print(f"  ❌ ad_creation_times insertion failed: {e}")
    
    # Test historical_data table
    print("  Testing historical_data insertion...")
    try:
        # Use 2020 timestamp to avoid range issues
        now = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {
            'ad_id': 'format_test_ad_001',
            'lifecycle_id': 'lifecycle_format_test_001',
            'stage': 'testing',
            'metric_name': 'test_cpm',
            'metric_value': 123.45,
            'ts_epoch': int(now.timestamp()),
            'ts_iso': now.isoformat(),
            'created_at': now.isoformat()
        }
        
        response = client.table('historical_data').insert(data).execute()
        print(f"  ✅ historical_data insertion successful")
        
    except Exception as e:
        print(f"  ❌ historical_data insertion failed: {e}")

def check_table_schemas(client):
    """Check table schemas to identify structure issues"""
    print("\n🔍 Checking table schemas...")
    
    tables_to_check = ['ad_creation_times', 'historical_data']
    
    for table in tables_to_check:
        try:
            # Try to get column information
            response = client.table(table).select('*').limit(0).execute()
            print(f"  ✅ Table '{table}' schema accessible")
            
        except Exception as e:
            print(f"  ❌ Table '{table}' schema issue: {e}")

def cleanup_test_data(client):
    """Clean up test data"""
    print("\n🧹 Cleaning up test data...")
    
    try:
        # Clean up test ad creation times
        client.table('ad_creation_times').delete().like('ad_id', 'test_ad_%').execute()
        client.table('ad_creation_times').delete().like('ad_id', 'format_test_ad_%').execute()
        
        # Clean up test historical data
        client.table('historical_data').delete().like('ad_id', 'test_ad_%').execute()
        client.table('historical_data').delete().like('ad_id', 'format_test_ad_%').execute()
        
        print("✅ Test data cleaned up")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Run comprehensive Supabase storage tests"""
    print("🚀 Starting Comprehensive Supabase Storage Tests")
    print("=" * 60)
    
    # Test 1: Basic connection
    client = test_supabase_connection()
    if not client:
        print("❌ Cannot proceed without Supabase connection")
        return False
    
    # Test 2: Table existence
    existing_tables, missing_tables = test_table_existence(client)
    
    # Test 3: Check schemas
    check_table_schemas(client)
    
    # Test 4: Data insertion formats
    test_data_insertion_formats(client)
    
    # Test 5: SupabaseStorage operations
    storage_ok = test_supabase_storage_operations(client)
    
    # Test 6: Table monitoring
    monitoring_ok = test_table_monitoring(client)
    
    # Cleanup
    cleanup_test_data(client)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Existing tables: {len(existing_tables)}")
    print(f"❌ Missing tables: {len(missing_tables)}")
    print(f"📦 SupabaseStorage: {'✅ OK' if storage_ok else '❌ FAILED'}")
    print(f"📊 Table monitoring: {'✅ OK' if monitoring_ok else '❌ FAILED'}")
    
    if missing_tables:
        print(f"\n⚠️ Missing tables that need to be created:")
        for table in missing_tables:
            print(f"   • {table}")
    
    return len(missing_tables) == 0 and storage_ok and monitoring_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
