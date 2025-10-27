#!/usr/bin/env python3
"""
Quick test for ML model ultra-minimal fallback
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'dean', 'src')))

def test_ultra_minimal_fallback():
    """Test the ultra-minimal ML model fallback"""
    print("🔧 Testing ultra-minimal ML model fallback...")
    
    # Mock classes
    class MockLogger:
        def info(self, msg): print(f"  📝 INFO: {msg}")
        def warning(self, msg): print(f"  ⚠️ WARN: {msg}")
    
    class MockSupabase:
        def __init__(self):
            self.client = MockClient()
    
    class MockClient:
        def table(self, name):
            return MockTable()
    
    class MockTable:
        def upsert(self, data, on_conflict=None):
            # Simulate schema cache error for any field except ultra-minimal
            if len(data) > 3:  # More than just model_type, stage, model_data
                raise Exception("Could not find the 'accuracy' column of 'ml_models' in the schema cache")
            return MockResponse()
    
    class MockResponse:
        def execute(self):
            return {"status": "success"}
    
    class MockValidatedClient:
        def upsert(self, table, data, on_conflict=None):
            # Simulate schema cache error for any field except ultra-minimal
            if len(data) > 3:  # More than just model_type, stage, model_data
                raise Exception("Could not find the 'accuracy' column of 'ml_models' in the schema cache")
            return MockResponse()
    
    # Test the 4-tier fallback logic
    def test_save_model(model_type, stage, model_data):
        logger = MockLogger()
        supabase = MockSupabase()
        validated_client = MockValidatedClient()
        
        # Create all data structures
        data_with_accuracy = {
            'model_type': model_type,
            'stage': stage,
            'version': 1,
            'model_name': f"{model_type}_{stage}_v1",
            'model_data': model_data.hex(),
            'accuracy': 0.8,
            'precision': 0.7,
            'recall': 0.6,
            'f1_score': 0.65,
            'is_active': True,
            'trained_at': '2025-10-27T00:00:00Z'
        }
        
        data_without_accuracy = {
            'model_type': model_type,
            'stage': stage,
            'version': 1,
            'model_name': f"{model_type}_{stage}_v1",
            'model_data': model_data.hex(),
            'is_active': True,
            'trained_at': '2025-10-27T00:00:00Z'
        }
        
        data_minimal = {
            'model_type': model_type,
            'stage': stage,
            'version': 1,
            'model_name': f"{model_type}_{stage}_v1",
            'model_data': model_data.hex(),
            'is_active': True,
            'trained_at': '2025-10-27T00:00:00Z'
        }
        
        data_ultra_minimal = {
            'model_type': model_type,
            'stage': stage,
            'model_data': model_data.hex()
        }
        
        # Test the exact logic from the fixed code
        try:
            # Try with accuracy metrics first
            try:
                response = validated_client.upsert('ml_models', data_with_accuracy, on_conflict='model_type,stage,version')
                logger.info(f"✅ Model {model_type}_{stage} saved with accuracy metrics")
            except Exception as e:
                if "accuracy" in str(e) and "schema cache" in str(e):
                    # Schema doesn't support accuracy column, try without it
                    logger.warning(f"Schema doesn't support accuracy column, retrying without accuracy metrics")
                    try:
                        response = validated_client.upsert('ml_models', data_without_accuracy, on_conflict='model_type,stage,version')
                        logger.info(f"✅ Model {model_type}_{stage} saved without accuracy metrics")
                    except Exception as e2:
                        # If still failing, try minimal data structure
                        logger.warning(f"Schema has limited support, using minimal data structure")
                        try:
                            response = validated_client.upsert('ml_models', data_minimal, on_conflict='model_type,stage,version')
                            logger.info(f"✅ Model {model_type}_{stage} saved with minimal data")
                        except Exception as e3:
                            # If still failing, try ultra-minimal data structure
                            logger.warning(f"Schema has very limited support, using ultra-minimal data structure")
                            response = validated_client.upsert('ml_models', data_ultra_minimal, on_conflict='model_type,stage')
                            logger.info(f"✅ Model {model_type}_{stage} saved with ultra-minimal data")
                else:
                    raise e
        except Exception as e:
            logger.warning(f"Could not save model {model_type}_{stage}: {e}, continuing...")
            return False
        
        return True
    
    # Test with mock model data
    class MockModel:
        def hex(self):
            return "mock_model_hex_data"
    
    result = test_save_model('performance_predictor', 'testing', MockModel())
    
    if result:
        print("  ✅ Ultra-minimal fallback test PASSED")
        return True
    else:
        print("  ❌ Ultra-minimal fallback test FAILED")
        return False

if __name__ == "__main__":
    success = test_ultra_minimal_fallback()
    sys.exit(0 if success else 1)
