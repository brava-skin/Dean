"""
DEAN RULES SYSTEM
Business rules and adaptive decision logic

This package contains:
- rules: Core business rules and thresholds
- adaptive_rules: ML-enhanced adaptive rules
"""

# Note: Direct imports to avoid circular import issues

# Import the actual classes
from .adaptive_rules import IntelligentRuleEngine, RuleConfig, create_intelligent_rule_engine

__all__ = [
    'IntelligentRuleEngine', 'RuleConfig', 'create_intelligent_rule_engine'
]
