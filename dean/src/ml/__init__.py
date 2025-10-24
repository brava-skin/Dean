"""
DEAN ML SYSTEM
Machine Learning components for Meta Ads automation

This package contains all ML-related modules:
- ml_intelligence: Core ML system
- ml_pipeline: ML orchestration
- ml_decision_engine: ML decision logic
- ml_monitoring: Status and dashboard
- ml_advanced_features: Advanced ML capabilities
- ml_enhancements: Core enhancements
- ml_reporting: ML reporting system
"""

# Import main classes for easy access
# Note: Direct imports to avoid circular import issues

__all__ = [
    # Core ML system
    'MLIntelligenceSystem', 'MLConfig', 'create_ml_system',
    
    # ML pipeline
    'create_ml_pipeline', 'MLPipelineConfig',
    
    # Decision engine
    'MLDecisionEngine', 'create_ml_decision_engine',
    
    # Monitoring
    'create_ml_dashboard', 'get_ml_learning_summary', 'send_ml_learning_report',
    
    # Advanced features
    'create_ql_agent', 'create_lstm_predictor', 'create_auto_feature_engineer',
    'create_bayesian_optimizer', 'create_portfolio_optimizer', 'create_seasonality_detector', 'create_shap_explainer',
    
    # Enhancements
    'create_model_validator', 'create_data_progress_tracker', 'create_anomaly_detector',
    'create_time_series_forecaster', 'create_creative_similarity_analyzer', 'create_causal_impact_analyzer',
    
    # Reporting
    'MLReportingSystem', 'create_ml_reporting_system'
]
