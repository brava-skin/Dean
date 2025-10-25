#!/usr/bin/env python3
"""
Discover actual Supabase table schemas
"""

import os
import sys

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

def discover_table_schema(client, table_name):
    """Discover the actual schema of a table by examining existing data"""
    print(f"\n🔍 Discovering schema for: {table_name}")
    
    try:
        # Get a sample record to see the actual columns
        response = client.table(table_name).select('*').limit(1).execute()
        
        if response.data and len(response.data) > 0:
            record = response.data[0]
            print(f"   ✅ Sample record found with columns:")
            for key, value in record.items():
                value_type = type(value).__name__
                print(f"      • {key}: {value_type} = {value}")
        else:
            print(f"   ℹ️ No data found, trying to insert minimal data to discover required columns...")
            
            # Try inserting minimal data to see what's required
            try:
                minimal_data = {'test_field': 'test_value'}
                client.table(table_name).insert(minimal_data).execute()
                print(f"   ✅ Minimal insert successful")
            except Exception as e:
                print(f"   📋 Required columns from error: {e}")
                
                # Try to extract column names from error message
                if "null value in column" in str(e):
                    # Extract required column name
                    import re
                    match = re.search(r'null value in column "([^"]+)"', str(e))
                    if match:
                        required_col = match.group(1)
                        print(f"      → Required column: {required_col}")
        
    except Exception as e:
        print(f"   ❌ Failed to discover schema: {e}")

def main():
    """Discover schemas for all problematic tables"""
    print("🔍 SUPABASE SCHEMA DISCOVERY")
    print("=" * 50)
    
    try:
        client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )
        print("✅ Connected to Supabase")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # Tables that need schema discovery
    problematic_tables = [
        'time_series_data',
        'creative_intelligence', 
        'ml_models',
        'ml_predictions',
        'learning_events',
        'creative_library',
        'creative_performance',
        'ai_generated_creatives',
        'adaptive_rules',
        'fatigue_analysis',
        'system_health'
    ]
    
    for table in problematic_tables:
        discover_table_schema(client, table)

if __name__ == "__main__":
    main()
