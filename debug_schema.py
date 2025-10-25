#!/usr/bin/env python3
"""
Debug Supabase table schemas to understand the timestamp issue
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

def debug_ad_creation_times_schema():
    """Debug the ad_creation_times table schema"""
    print("🔍 Debugging ad_creation_times table schema...")
    
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        client = create_client(url, key)
        
        # Try different timestamp formats
        now = datetime.now(timezone.utc)
        
        test_cases = [
            {
                'name': 'ISO string only',
                'data': {
                    'ad_id': 'debug_test_001',
                    'lifecycle_id': 'lifecycle_debug_001',
                    'stage': 'testing',
                    'created_at_iso': now.isoformat(),
                    'updated_at': int(now.timestamp())
                }
            },
            {
                'name': 'Epoch only',
                'data': {
                    'ad_id': 'debug_test_002',
                    'lifecycle_id': 'lifecycle_debug_002',
                    'stage': 'testing',
                    'created_at_epoch': int(now.timestamp()),
                    'updated_at': int(now.timestamp())
                }
            },
            {
                'name': 'Both formats',
                'data': {
                    'ad_id': 'debug_test_003',
                    'lifecycle_id': 'lifecycle_debug_003',
                    'stage': 'testing',
                    'created_at_epoch': int(now.timestamp()),
                    'created_at_iso': now.isoformat(),
                    'updated_at': int(now.timestamp())
                }
            },
            {
                'name': 'Minimal data',
                'data': {
                    'ad_id': 'debug_test_004',
                    'stage': 'testing'
                }
            }
        ]
        
        for test_case in test_cases:
            print(f"\n  Testing: {test_case['name']}")
            try:
                response = client.table('ad_creation_times').upsert(test_case['data'], on_conflict='ad_id').execute()
                print(f"  ✅ Success: {test_case['name']}")
            except Exception as e:
                print(f"  ❌ Failed: {test_case['name']} - {e}")
        
        # Try to read existing data to understand the schema
        print(f"\n  Reading existing data...")
        try:
            response = client.table('ad_creation_times').select('*').limit(5).execute()
            if response.data:
                print(f"  ✅ Found {len(response.data)} existing records")
                for i, record in enumerate(response.data[:2]):
                    print(f"    Record {i+1}: {record}")
            else:
                print(f"  ℹ️ No existing records found")
        except Exception as e:
            print(f"  ❌ Failed to read existing data: {e}")
        
        # Cleanup
        print(f"\n  Cleaning up test data...")
        try:
            client.table('ad_creation_times').delete().like('ad_id', 'debug_test_%').execute()
            print(f"  ✅ Cleanup successful")
        except Exception as e:
            print(f"  ⚠️ Cleanup warning: {e}")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    debug_ad_creation_times_schema()
