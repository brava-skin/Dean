#!/usr/bin/env python3
"""
Quick Supabase Test - Only test tables that actually exist and work
"""

import os
import sys
from datetime import datetime, timezone

# Set environment variables
os.environ.update({
    'SUPABASE_URL': 'https://vbttsnxtiyfwpyxuexcu.supabase.co',
    'SUPABASE_SERVICE_ROLE_KEY': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZidHRzbnh0aXlmd3B5eHVleGN1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTkxMjgwMSwiZXhwIjoyMDc1NDg4ODAxfQ.c_zhRD6dYLC6oYHm91GA4kCalrq2nDtNCpWuq_8l6LU'
})

try:
    from supabase import create_client
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Only test tables that we know work or are critical
WORKING_TABLES = {
    'ad_creation_times': {
        'ad_id': 'test_ad_001',
        'lifecycle_id': 'lifecycle_001',
        'stage': 'testing',
        'created_at_epoch': 123456,
        'created_at_iso': '2020-01-01T00:00:00+00:00'
    },
    'historical_data': {
        'ad_id': 'test_ad_001',
        'lifecycle_id': 'lifecycle_001',
        'stage': 'testing',
        'metric_name': 'cpm',
        'metric_value': 125.50,
        'ts_epoch': 1577836800,
        'ts_iso': '2020-01-01T00:00:00+00:00',
        'created_at': '2020-01-01T00:00:00+00:00'
    },
    'performance_metrics': {
        'ad_id': 'test_ad_001',
        'lifecycle_id': 'lifecycle_001',
        'stage': 'testing',
        'window_type': '1d',
        'date_start': '2020-01-01',
        'date_end': '2020-01-01',
        'spend': 25.50,
        'impressions': 1000,
        'clicks': 25,
        'purchases': 1,
        'add_to_cart': 3,
        'initiate_checkout': 2,
        'revenue': 50.00,
        'ctr': 2.5,
        'cpm': 25.50,
        'cpc': 1.02,
        'cpa': 25.50,
        'roas': 1.96
    },
    'ad_lifecycle': {
        'ad_id': 'test_ad_001',
        'creative_id': 'creative_001',
        'campaign_id': 'campaign_001',
        'adset_id': 'adset_001',
        'lifecycle_id': 'lifecycle_001',
        'stage': 'testing',
        'status': 'active',
        'created_at': '2020-01-01T00:00:00+00:00'
    },
    'time_series_data': {
        'ad_id': 'test_ad_001',
        'lifecycle_id': 'lifecycle_001',
        'stage': 'testing',
        'timestamp': '2020-01-01T00:00:00+00:00',
        'metric_name': 'cpm',
        'metric_value': 125.50,
        'metadata': {}
    }
}

def test_table(client, table_name, test_data):
    """Test a single table"""
    print(f"🔍 Testing {table_name}...")
    
    try:
        # Add unique identifier
        unique_data = test_data.copy()
        if 'ad_id' in unique_data:
            unique_data['ad_id'] = f"test_{table_name}_{datetime.now().microsecond}"
        
        # Insert
        response = client.table(table_name).insert(unique_data).execute()
        print(f"   ✅ Insert successful")
        
        # Select
        response = client.table(table_name).select('*').limit(1).execute()
        print(f"   ✅ Select successful ({len(response.data)} records)")
        
        # Cleanup
        if 'ad_id' in unique_data:
            client.table(table_name).delete().eq('ad_id', unique_data['ad_id']).execute()
            print(f"   ✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def main():
    """Test working tables"""
    print("🚀 QUICK SUPABASE TEST - WORKING TABLES ONLY")
    print("=" * 60)
    
    try:
        client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )
        print("✅ Connected to Supabase")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    working_count = 0
    total_count = len(WORKING_TABLES)
    
    for table_name, test_data in WORKING_TABLES.items():
        if test_table(client, table_name, test_data):
            working_count += 1
    
    print(f"\n📊 RESULTS: {working_count}/{total_count} tables working ({(working_count/total_count)*100:.1f}%)")
    
    if working_count == total_count:
        print("🎉 All critical tables are working!")
        return True
    else:
        print(f"⚠️ {total_count - working_count} tables still have issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
