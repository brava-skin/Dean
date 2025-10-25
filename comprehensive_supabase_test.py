#!/usr/bin/env python3
"""
Comprehensive Supabase Test Suite - ALL TABLES
Tests every Supabase table used in the codebase
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
    'SUPABASE_URL': 'https://vbttsnxtiyfwpyxuexcu.supabase.co',
    'SUPABASE_SERVICE_ROLE_KEY': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZidHRzbnh0aXlmd3B5eHVleGN1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTkxMjgwMSwiZXhwIjoyMDc1NDg4ODAxfQ.c_zhRD6dYLC6oYHm91GA4kCalrq2nDtNCpWuq_8l6LU'
})

try:
    from supabase import create_client
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# All Supabase tables identified in the codebase
SUPABASE_TABLES = {
    # Core data tables
    'ad_creation_times': {
        'description': 'Ad creation timestamps for time-based rules',
        'test_data': {
            'ad_id': 'test_ad_001',
            'lifecycle_id': 'lifecycle_001',
            'stage': 'testing',
            'created_at_epoch': 123456,  # Use small integer
            'created_at_iso': '2020-01-01T00:00:00+00:00'
        }
    },
    'historical_data': {
        'description': 'Historical performance metrics',
        'test_data': {
            'ad_id': 'test_ad_001',
            'lifecycle_id': 'lifecycle_001',
            'stage': 'testing',
            'metric_name': 'cpm',
            'metric_value': 125.50,
            'ts_epoch': 1577836800,
            'ts_iso': '2020-01-01T00:00:00+00:00',
            'created_at': '2020-01-01T00:00:00+00:00'
        }
    },
    'performance_metrics': {
        'description': 'Main performance data storage',
        'test_data': {
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
        }
    },
    'ad_lifecycle': {
        'description': 'Ad lifecycle tracking',
        'test_data': {
            'ad_id': 'test_ad_001',
            'creative_id': 'creative_001',
            'campaign_id': 'campaign_001',
            'adset_id': 'adset_001',
            'lifecycle_id': 'lifecycle_001',
            'stage': 'testing',
            'status': 'active',
            'created_at': '2020-01-01T00:00:00+00:00'
        }
    },
    'time_series_data': {
        'description': 'Time-series performance data',
        'test_data': {
            'ad_id': 'test_ad_001',
            'lifecycle_id': 'lifecycle_001',
            'stage': 'testing',
            'timestamp': '2020-01-01T00:00:00+00:00',
            'metric_name': 'cpm',
            'metric_value': 125.50,
            'metadata': {}
            # Removed 'window_size' as it doesn't exist in actual schema
        }
    },
    'creative_intelligence': {
        'description': 'Creative assets and performance',
        'test_data': {
            'creative_id': 'creative_001',
            'ad_id': 'test_ad_001',
            'creative_type': 'video'
            # Removed all non-existent columns based on actual schema
        }
    },
    'ml_models': {
        'description': 'ML model storage',
        'test_data': {
            'model_name': 'test_model',
            'model_type': 'performance_predictor',  # Use valid model type from schema
            'stage': 'testing',
            'version': 1,
            'model_data': '74657374',  # hex encoded 'test'
            'model_metadata': {'test': 'data'},
            'features_used': ['cpm', 'ctr'],
            'performance_metrics': {'accuracy': 0.85},
            'is_active': True,
            'trained_at': '2020-01-01T00:00:00+00:00'
        }
    },
    'ml_predictions': {
        'description': 'ML predictions storage',
        'test_data': {
            'ad_id': 'test_ad_001',
            'prediction_type': 'performance',
            'prediction_value': 0.75,
            'created_at': '2020-01-01T00:00:00+00:00'
            # Removed 'model_name' and 'confidence' as they don't exist in actual schema
        }
    },
    'learning_events': {
        'description': 'ML learning events',
        'test_data': {
            'event_type': 'model_training',
            'stage': 'testing',
            'created_at': '2020-01-01T00:00:00+00:00'
            # Removed 'model_name' and 'event_data' as they don't exist in actual schema
        }
    },
    'creative_library': {
        'description': 'Creative assets library',
        'test_data': {
            'creative_id': 'creative_001',
            'creative_type': 'video',
            'content': 'Test content',  # Based on actual schema
            'category': 'global',
            'performance_score': 85,
            'usage_count': 0,
            'created_by': 'system'
        }
    },
    'creative_performance': {
        'description': 'Creative performance tracking',
        'test_data': {
            'creative_id': 'creative_001',
            'ad_id': 'test_ad_001',
            'performance_score': 85,
            'ctr': 2.5,
            'cpm': 25.50,
            'updated_at': '2020-01-01T00:00:00+00:00'
            # Removed 'conversions' as it doesn't exist in actual schema
        }
    },
    'ai_generated_creatives': {
        'description': 'AI-generated creative content',
        'test_data': {
            'creative_id': 'ai_creative_001',
            'generated_by': 'gpt-5',
            'prompt_used': 'Test prompt',
            'primary_text': 'AI generated text',
            'headline': 'AI headline',
            'similarity_score': 0.85
            # Removed 'description' as it doesn't exist in actual schema
        }
    },
    'adaptive_rules': {
        'description': 'Adaptive rule configurations',
        'test_data': {
            'rule_name': 'test_rule',
            'rule_type': 'threshold',  # Based on actual schema
            'stage': 'testing',
            'current_value': 100.0,  # Based on actual schema
            'confidence_weight': 1.0,
            'learning_rate': 0.1,
            'is_active': True
        }
    },
    'rule_decisions': {
        'description': 'Rule decision logging',
        'test_data': {
            'ad_id': 'test_ad_001',
            'rule_name': 'test_rule',
            'decision': 'kill',
            'confidence': 0.95,
            'metadata': {'reason': 'high_cpm'},
            'created_at': '2020-01-01T00:00:00+00:00'
        }
    },
    'fatigue_analysis': {
        'description': 'Creative fatigue analysis',
        'test_data': {
            'ad_id': 'test_ad_001',
            'fatigue_score': 0.75,
            'trend_direction': 'declining',
            'created_at': '2020-01-01T00:00:00+00:00'
            # Removed 'days_active' as it doesn't exist in actual schema
        }
    },
    'performance_decay': {
        'description': 'Performance decay tracking',
        'test_data': {
            'ad_id': 'test_ad_001',
            'decay_rate': 0.15,
            'original_performance': 0.85,
            'current_performance': 0.70,
            'created_at': '2020-01-01T00:00:00+00:00'
        }
    },
    'model_validations': {
        'description': 'ML model validation results',
        'test_data': {
            'model_name': 'test_model',
            'validation_type': 'cross_validation',
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'created_at': '2020-01-01T00:00:00+00:00'
        }
    },
    'system_health': {
        'description': 'System health monitoring',
        'test_data': {
            'stage': 'testing',  # Based on actual schema
            'health_score': 85,
            'stability_score': 0.8,
            'confidence_score': 0.7,
            'efficiency_score': 0.75,
            'metrics': {'accuracy': 0.85, 'latency': 150}
        }
    }
}

def get_supabase_client():
    """Get Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)

def test_table_operations(client, table_name: str, table_info: Dict):
    """Test all operations for a specific table"""
    print(f"\n🔍 Testing table: {table_name}")
    print(f"   Description: {table_info['description']}")
    
    results = {
        'exists': False,
        'insert': False,
        'select': False,
        'update': False,
        'delete': False,
        'errors': []
    }
    
    try:
        # Test 1: Check if table exists
        try:
            response = client.table(table_name).select('*').limit(1).execute()
            results['exists'] = True
            print(f"   ✅ Table exists")
        except Exception as e:
            results['errors'].append(f"Table existence check failed: {e}")
            print(f"   ❌ Table doesn't exist or inaccessible: {e}")
            return results
        
        # Test 2: Insert test data
        try:
            test_data = table_info['test_data'].copy()
            # Add unique identifier to avoid conflicts
            if 'ad_id' in test_data:
                test_data['ad_id'] = f"test_{table_name}_{datetime.now().microsecond}"
            elif 'creative_id' in test_data:
                test_data['creative_id'] = f"test_{table_name}_{datetime.now().microsecond}"
            elif 'model_name' in test_data:
                test_data['model_name'] = f"test_{table_name}_{datetime.now().microsecond}"
            
            response = client.table(table_name).insert(test_data).execute()
            results['insert'] = True
            print(f"   ✅ Insert successful")
            
            # Store the inserted ID for cleanup
            inserted_id = None
            if response.data and len(response.data) > 0:
                inserted_data = response.data[0]
                inserted_id = (
                    inserted_data.get('id') or 
                    inserted_data.get('ad_id') or 
                    inserted_data.get('creative_id') or
                    inserted_data.get('model_name')
                )
            
        except Exception as e:
            results['errors'].append(f"Insert failed: {e}")
            print(f"   ❌ Insert failed: {e}")
        
        # Test 3: Select data
        try:
            response = client.table(table_name).select('*').limit(5).execute()
            results['select'] = True
            print(f"   ✅ Select successful ({len(response.data)} records)")
        except Exception as e:
            results['errors'].append(f"Select failed: {e}")
            print(f"   ❌ Select failed: {e}")
        
        # Test 4: Update data (if we have an ID)
        if results['insert'] and inserted_id:
            try:
                update_data = {'updated_at': '2020-01-02T00:00:00+00:00'}
                if 'ad_id' in test_data:
                    response = client.table(table_name).update(update_data).eq('ad_id', inserted_id).execute()
                elif 'creative_id' in test_data:
                    response = client.table(table_name).update(update_data).eq('creative_id', inserted_id).execute()
                elif 'model_name' in test_data:
                    response = client.table(table_name).update(update_data).eq('model_name', inserted_id).execute()
                else:
                    response = client.table(table_name).update(update_data).eq('id', inserted_id).execute()
                
                results['update'] = True
                print(f"   ✅ Update successful")
            except Exception as e:
                results['errors'].append(f"Update failed: {e}")
                print(f"   ❌ Update failed: {e}")
        
        # Test 5: Delete test data (cleanup)
        if results['insert'] and inserted_id:
            try:
                if 'ad_id' in test_data:
                    response = client.table(table_name).delete().eq('ad_id', inserted_id).execute()
                elif 'creative_id' in test_data:
                    response = client.table(table_name).delete().eq('creative_id', inserted_id).execute()
                elif 'model_name' in test_data:
                    response = client.table(table_name).delete().eq('model_name', inserted_id).execute()
                else:
                    response = client.table(table_name).delete().eq('id', inserted_id).execute()
                
                results['delete'] = True
                print(f"   ✅ Delete successful (cleanup)")
            except Exception as e:
                results['errors'].append(f"Delete failed: {e}")
                print(f"   ⚠️ Delete failed (cleanup): {e}")
        
    except Exception as e:
        results['errors'].append(f"General error: {e}")
        print(f"   ❌ General error: {e}")
    
    return results

def main():
    """Run comprehensive Supabase table tests"""
    print("🚀 COMPREHENSIVE SUPABASE TABLE TESTING")
    print("=" * 80)
    print(f"Testing {len(SUPABASE_TABLES)} tables identified in the codebase...")
    
    try:
        client = get_supabase_client()
        print("✅ Supabase connection established")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return False
    
    # Test all tables
    results = {}
    total_tables = len(SUPABASE_TABLES)
    working_tables = 0
    
    for table_name, table_info in SUPABASE_TABLES.items():
        table_results = test_table_operations(client, table_name, table_info)
        results[table_name] = table_results
        
        # Count as working if basic operations succeed
        if table_results['exists'] and table_results['insert'] and table_results['select']:
            working_tables += 1
    
    # Summary Report
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    print(f"📈 Overall Status: {working_tables}/{total_tables} tables fully functional")
    print(f"📊 Success Rate: {(working_tables/total_tables)*100:.1f}%")
    
    print(f"\n✅ WORKING TABLES ({working_tables}):")
    for table_name, table_results in results.items():
        if table_results['exists'] and table_results['insert'] and table_results['select']:
            ops = []
            if table_results['insert']: ops.append("INSERT")
            if table_results['select']: ops.append("SELECT") 
            if table_results['update']: ops.append("UPDATE")
            if table_results['delete']: ops.append("DELETE")
            print(f"   • {table_name}: {', '.join(ops)}")
    
    failing_tables = total_tables - working_tables
    if failing_tables > 0:
        print(f"\n❌ FAILING TABLES ({failing_tables}):")
        for table_name, table_results in results.items():
            if not (table_results['exists'] and table_results['insert'] and table_results['select']):
                print(f"   • {table_name}:")
                for error in table_results['errors']:
                    print(f"     - {error}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if working_tables == total_tables:
        print("   🎉 All tables are working perfectly! No action needed.")
    else:
        print("   🔧 Fix the failing tables by:")
        print("      1. Checking table schemas in Supabase dashboard")
        print("      2. Ensuring all required columns exist")
        print("      3. Adjusting data types and constraints")
        print("      4. Adding missing tables if needed")
    
    return working_tables == total_tables

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
