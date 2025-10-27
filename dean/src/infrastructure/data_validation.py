"""
Comprehensive Data Validation System for Supabase
================================================
This module provides field-level validation for all Supabase table operations.
Prevents invalid data from being stored by validating every field before insertion.
"""

import re
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)

class ValidationSeverity(Enum):
    """Validation severity levels."""
    ERROR = "error"      # Block insertion
    WARNING = "warning"  # Log warning but allow insertion
    INFO = "info"        # Log info but allow insertion

@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]
    sanitized_data: Dict[str, Any]

class FieldValidator:
    """Base class for field validators."""
    
    def __init__(self, field_name: str, required: bool = False, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.field_name = field_name
        self.required = required
        self.severity = severity
    
    def validate(self, value: Any, data: Dict[str, Any]) -> List[str]:
        """Validate a field value. Return list of error messages."""
        errors = []
        
        # Check required fields
        if self.required and (value is None or value == ''):
            errors.append(f"Field '{self.field_name}' is required")
            return errors
        
        # Skip validation for None/empty values if not required
        if not self.required and (value is None or value == ''):
            return errors
        
        # Perform field-specific validation
        field_errors = self._validate_field(value, data)
        errors.extend(field_errors)
        
        return errors
    
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        """Override in subclasses for field-specific validation."""
        return []
    
    def sanitize(self, value: Any) -> Any:
        """Sanitize/clean the field value. Override in subclasses."""
        return value

class StringValidator(FieldValidator):
    """Validator for string fields."""
    
    def __init__(self, field_name: str, max_length: int = None, min_length: int = None, 
                 pattern: str = None, allowed_values: List[str] = None, **kwargs):
        super().__init__(field_name, **kwargs)
        self.max_length = max_length
        self.min_length = min_length
        self.pattern = pattern
        self.allowed_values = allowed_values
    
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        errors = []
        
        # Convert to string
        str_value = str(value) if value is not None else ""
        
        # Check length constraints
        if self.min_length and len(str_value) < self.min_length:
            errors.append(f"Field '{self.field_name}' must be at least {self.min_length} characters")
        
        if self.max_length and len(str_value) > self.max_length:
            errors.append(f"Field '{self.field_name}' must be at most {self.max_length} characters")
        
        # Check pattern
        if self.pattern and str_value:
            if not re.match(self.pattern, str_value):
                errors.append(f"Field '{self.field_name}' does not match required pattern")
        
        # Check allowed values
        if self.allowed_values and str_value not in self.allowed_values:
            errors.append(f"Field '{self.field_name}' must be one of: {', '.join(self.allowed_values)}")
        
        return errors
    
    def sanitize(self, value: Any) -> str:
        """Sanitize string value."""
        if value is None:
            return ""
        return str(value).strip()

class IntegerValidator(FieldValidator):
    """Validator for integer fields."""
    
    def __init__(self, field_name: str, min_value: int = None, max_value: int = None, **kwargs):
        super().__init__(field_name, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
    
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        errors = []
        
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            errors.append(f"Field '{self.field_name}' must be a valid integer")
            return errors
        
        # Check range constraints
        if self.min_value is not None and int_value < self.min_value:
            errors.append(f"Field '{self.field_name}' must be at least {self.min_value}")
        
        if self.max_value is not None and int_value > self.max_value:
            errors.append(f"Field '{self.field_name}' must be at most {self.max_value}")
        
        return errors
    
    def sanitize(self, value: Any) -> Optional[int]:
        """Sanitize integer value."""
        if value is None or value == '':
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

class FloatValidator(FieldValidator):
    """Validator for float fields."""
    
    def __init__(self, field_name: str, min_value: float = None, max_value: float = None, 
                 allow_inf: bool = False, allow_nan: bool = False, **kwargs):
        super().__init__(field_name, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.allow_inf = allow_inf
        self.allow_nan = allow_nan
    
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        errors = []
        
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            errors.append(f"Field '{self.field_name}' must be a valid number")
            return errors
        
        # Check for infinity
        if not self.allow_inf and (float_value == float('inf') or float_value == float('-inf')):
            errors.append(f"Field '{self.field_name}' cannot be infinity")
        
        # Check for NaN
        if not self.allow_nan and float_value != float_value:  # NaN check
            errors.append(f"Field '{self.field_name}' cannot be NaN")
        
        # Check range constraints
        if self.min_value is not None and float_value < self.min_value:
            errors.append(f"Field '{self.field_name}' must be at least {self.min_value}")
        
        if self.max_value is not None and float_value > self.max_value:
            errors.append(f"Field '{self.field_name}' must be at most {self.max_value}")
        
        return errors
    
    def sanitize(self, value: Any) -> Optional[float]:
        """Sanitize float value."""
        if value is None or value == '':
            return None
        try:
            float_value = float(value)
            # Handle infinity and NaN
            if not self.allow_inf and (float_value == float('inf') or float_value == float('-inf')):
                return 0.0
            if not self.allow_nan and float_value != float_value:  # NaN check
                return 0.0
            return float_value
        except (ValueError, TypeError):
            return None

class BooleanValidator(FieldValidator):
    """Validator for boolean fields."""
    
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        errors = []
        
        # Accept various boolean representations
        if value not in [True, False, 1, 0, "true", "false", "1", "0", "yes", "no"]:
            errors.append(f"Field '{self.field_name}' must be a boolean value")
        
        return errors
    
    def sanitize(self, value: Any) -> Optional[bool]:
        """Sanitize boolean value."""
        if value is None or value == '':
            return None
        
        if isinstance(value, bool):
            return value
        
        str_value = str(value).lower()
        if str_value in ['true', '1', 'yes']:
            return True
        elif str_value in ['false', '0', 'no']:
            return False
        
        return None

class DateValidator(FieldValidator):
    """Validator for date fields with flexible format support."""
    
    def __init__(self, field_name: str, date_format: str = "%Y-%m-%d", **kwargs):
        super().__init__(field_name, **kwargs)
        self.date_format = date_format
        # Common date formats to try
        self.common_formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S.%f",
        ]
    
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        errors = []
        
        if isinstance(value, str) and value:
            # Try the specified format first
            try:
                datetime.strptime(value, self.date_format)
                return errors
            except ValueError:
                pass
            
            # Try common formats
            for fmt in self.common_formats:
                try:
                    datetime.strptime(value, fmt)
                    return errors
                except ValueError:
                    continue
            
            errors.append(f"Field '{self.field_name}' must be a valid date in format {self.date_format}")
        
        return errors
    
    def sanitize(self, value: Any) -> Optional[str]:
        """Sanitize date value."""
        if value is None or value == '':
            return None
        
        if isinstance(value, str):
            # Try the specified format first
            try:
                datetime.strptime(value, self.date_format)
                return value
            except ValueError:
                pass
            
            # Try common formats and convert to the specified format
            for fmt in self.common_formats:
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt.strftime(self.date_format)
                except ValueError:
                    continue
            
            return None
        
        return None

class JSONValidator(FieldValidator):
    """Validator for JSON fields."""
    
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        errors = []
        
        if value is not None and value != '':
            if isinstance(value, str):
                try:
                    json.loads(value)
                except json.JSONDecodeError:
                    errors.append(f"Field '{self.field_name}' must be valid JSON")
            elif not isinstance(value, (dict, list)):
                errors.append(f"Field '{self.field_name}' must be a valid JSON object or array")
        
        return errors
    
    def sanitize(self, value: Any) -> Optional[Dict[str, Any]]:
        """Sanitize JSON value."""
        if value is None or value == '':
            return None
        
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        
        if isinstance(value, (dict, list)):
            return value
        
        return None

class CustomValidator(FieldValidator):
    """Validator for custom validation logic."""
    
    def __init__(self, field_name: str, validation_func: Callable[[Any, Dict[str, Any]], List[str]], 
                 sanitize_func: Callable[[Any], Any] = None, **kwargs):
        super().__init__(field_name, **kwargs)
        self.validation_func = validation_func
        self.sanitize_func = sanitize_func or (lambda x: x)
    
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        try:
            return self.validation_func(value, data)
        except Exception as e:
            return [f"Custom validation error for '{self.field_name}': {str(e)}"]
    
    def sanitize(self, value: Any) -> Any:
        """Sanitize using custom function."""
        try:
            return self.sanitize_func(value)
        except Exception:
            return value

class TableValidator:
    """Validator for entire Supabase tables."""
    
    def __init__(self, table_name: str, validators: Dict[str, FieldValidator]):
        self.table_name = table_name
        self.validators = validators
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate all fields in the data."""
        errors = []
        warnings = []
        info = []
        sanitized_data = {}
        
        # Validate each field
        for field_name, validator in self.validators.items():
            value = data.get(field_name)
            
            # Validate field
            field_errors = validator.validate(value, data)
            
            # Categorize by severity
            for error in field_errors:
                if validator.severity == ValidationSeverity.ERROR:
                    errors.append(f"{field_name}: {error}")
                elif validator.severity == ValidationSeverity.WARNING:
                    warnings.append(f"{field_name}: {error}")
                else:
                    info.append(f"{field_name}: {error}")
            
            # Sanitize field
            sanitized_data[field_name] = validator.sanitize(value)
        
        # Check for missing required fields
        for field_name, validator in self.validators.items():
            if validator.required and field_name not in data:
                errors.append(f"Required field '{field_name}' is missing")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            sanitized_data=sanitized_data
        )

class SupabaseDataValidator:
    """Main validator class for all Supabase operations."""
    
    def __init__(self):
        self.table_validators = self._create_table_validators()
    
    def _create_table_validators(self) -> Dict[str, TableValidator]:
        """Create validators for all Supabase tables."""
        return {
            'performance_metrics': TableValidator('performance_metrics', {
                'ad_id': StringValidator('ad_id', required=True, max_length=100),
                'window_type': StringValidator('window_type', required=True, 
                                             allowed_values=['1d', '7d', '30d']),
                'date_start': DateValidator('date_start', required=True),
                'date_end': DateValidator('date_end', required=True),
                'impressions': IntegerValidator('impressions', min_value=0),
                'clicks': IntegerValidator('clicks', min_value=0),
                'spend': FloatValidator('spend', min_value=0, max_value=999999.99),
                'purchases': IntegerValidator('purchases', min_value=0),
                'add_to_cart': IntegerValidator('add_to_cart', min_value=0),
                'initiate_checkout': IntegerValidator('initiate_checkout', min_value=0),
                'ctr': FloatValidator('ctr', min_value=0, max_value=100),
                'cpc': FloatValidator('cpc', min_value=0, max_value=999.99),
                'cpm': FloatValidator('cpm', min_value=0, max_value=9999.99),
                'roas': FloatValidator('roas', min_value=0, max_value=999.99),
                'cpa': FloatValidator('cpa', min_value=0, max_value=999.99),
                'dwell_time': FloatValidator('dwell_time', min_value=0, max_value=999999.99),
                'frequency': FloatValidator('frequency', min_value=0, max_value=999.99),
                'atc_rate': FloatValidator('atc_rate', min_value=0, max_value=9.9999),
                'ic_rate': FloatValidator('ic_rate', min_value=0, max_value=9.9999),
                'purchase_rate': FloatValidator('purchase_rate', min_value=0, max_value=9.9999),
                'atc_to_ic_rate': FloatValidator('atc_to_ic_rate', min_value=0, max_value=9.9999),
                'ic_to_purchase_rate': FloatValidator('ic_to_purchase_rate', min_value=0, max_value=9.9999),
                'performance_quality_score': IntegerValidator('performance_quality_score', min_value=0, max_value=100),
                'stability_score': FloatValidator('stability_score', min_value=0, max_value=9.9999),
                'momentum_score': FloatValidator('momentum_score', min_value=0, max_value=9.9999),
                'fatigue_index': FloatValidator('fatigue_index', min_value=0, max_value=9.9999),
                'hour_of_day': IntegerValidator('hour_of_day', min_value=0, max_value=23),
                'day_of_week': IntegerValidator('day_of_week', min_value=0, max_value=6),
                'is_weekend': BooleanValidator('is_weekend'),
                'ad_age_days': IntegerValidator('ad_age_days', min_value=0),
            }),
            
            'ad_lifecycle': TableValidator('ad_lifecycle', {
                'ad_id': StringValidator('ad_id', required=True, max_length=100),
                'creative_id': StringValidator('creative_id', max_length=100),
                'campaign_id': StringValidator('campaign_id', max_length=100),
                'adset_id': StringValidator('adset_id', max_length=100),
                'stage': StringValidator('stage', required=True, 
                                       allowed_values=['testing', 'validation', 'scaling', 'completed']),
                'status': StringValidator('status', required=True, 
                                        allowed_values=['active', 'paused', 'completed', 'failed']),
                'lifecycle_id': StringValidator('lifecycle_id', max_length=100),
                'metadata': JSONValidator('metadata'),
                'stage_duration_hours': FloatValidator('stage_duration_hours', min_value=0),
                'previous_stage': StringValidator('previous_stage', 
                                                 allowed_values=['created', 'testing', 'validation', 'scaling']),
                'stage_performance': JSONValidator('stage_performance'),
                'transition_reason': StringValidator('transition_reason', max_length=200),
            }),
            
            'creative_intelligence': TableValidator('creative_intelligence', {
                'creative_id': StringValidator('creative_id', required=True, max_length=100),
                'ad_id': StringValidator('ad_id', required=True, max_length=100),
                'creative_type': StringValidator('creative_type', required=True,
                                                allowed_values=['image', 'video', 'carousel', 'collection', 'story', 'reels']),
                'duration_seconds': IntegerValidator('duration_seconds', min_value=0, max_value=3600),
                'aspect_ratio': StringValidator('aspect_ratio', pattern=r'^\d+:\d+$'),
                'file_size_mb': FloatValidator('file_size_mb', min_value=0, max_value=1000),
                'resolution': StringValidator('resolution', pattern=r'^\d+x\d+$'),
                'color_palette': JSONValidator('color_palette'),
                'text_overlay': BooleanValidator('text_overlay'),
                'music_present': BooleanValidator('music_present'),
                'voice_over': BooleanValidator('voice_over'),
                'avg_ctr': FloatValidator('avg_ctr', min_value=0, max_value=100),
                'avg_cpa': FloatValidator('avg_cpa', min_value=0, max_value=999.99),
                'avg_roas': FloatValidator('avg_roas', min_value=0, max_value=999.99),
                'performance_rank': IntegerValidator('performance_rank', min_value=1),
                'similarity_vector': CustomValidator('similarity_vector', 
                                                   self._validate_similarity_vector,
                                                   self._sanitize_similarity_vector),
                'metadata': JSONValidator('metadata'),
                'fatigue_index': FloatValidator('fatigue_index', min_value=0, max_value=1),
                'description': StringValidator('description', max_length=500),
                'headline': StringValidator('headline', max_length=200),
                'lifecycle_id': StringValidator('lifecycle_id', max_length=100),
                'stage': StringValidator('stage', 
                                       allowed_values=['testing', 'validation', 'scaling', 'completed']),
                'primary_text': StringValidator('primary_text', max_length=1000),
                'performance_score': FloatValidator('performance_score', min_value=0, max_value=1),
            }),
            
            'creative_library': TableValidator('creative_library', {
                'creative_id': StringValidator('creative_id', required=True, max_length=100),
                'creative_type': StringValidator('creative_type', required=True,
                                                allowed_values=['headline', 'description', 'primary_text', 'video', 'image', 'carousel']),
                'content': StringValidator('content', required=True, max_length=2000),
                'category': StringValidator('category', max_length=100),
                'tags': CustomValidator('tags', self._validate_tags_array, self._sanitize_tags_array),
                'performance_score': FloatValidator('performance_score', min_value=0, max_value=1),
                'usage_count': IntegerValidator('usage_count', min_value=0),
                'created_by': StringValidator('created_by', max_length=100),
                'metadata': JSONValidator('metadata'),
                'description': StringValidator('description', max_length=500),
                'primary_text': StringValidator('primary_text', max_length=1000),
                'headline': StringValidator('headline', max_length=200),
            }),
            
            'creative_performance': TableValidator('creative_performance', {
                'creative_id': StringValidator('creative_id', required=True, max_length=100),
                'ad_id': StringValidator('ad_id', required=True, max_length=100),
                'stage': StringValidator('stage', required=True,
                                       allowed_values=['testing', 'validation', 'scaling', 'completed']),
                'date_start': DateValidator('date_start', required=True),
                'date_end': DateValidator('date_end', required=True),
                'impressions': IntegerValidator('impressions', min_value=0),
                'clicks': IntegerValidator('clicks', min_value=0),
                'spend': FloatValidator('spend', min_value=0, max_value=999999.99),
                'purchases': IntegerValidator('purchases', min_value=0),
                'add_to_cart': IntegerValidator('add_to_cart', min_value=0),
                'initiate_checkout': IntegerValidator('initiate_checkout', min_value=0),
                'ctr': FloatValidator('ctr', min_value=0, max_value=100),
                'cpc': FloatValidator('cpc', min_value=0, max_value=999.99),
                'cpm': FloatValidator('cpm', min_value=0, max_value=9999.99),
                'roas': FloatValidator('roas', min_value=0, max_value=999.99),
                'cpa': FloatValidator('cpa', min_value=0, max_value=999.99),
                'engagement_rate': FloatValidator('engagement_rate', min_value=0, max_value=100),
                'conversion_rate': FloatValidator('conversion_rate', min_value=0, max_value=100),
                'conversions': IntegerValidator('conversions', min_value=0),
                'lifecycle_id': StringValidator('lifecycle_id', max_length=100),
                'performance_score': FloatValidator('performance_score', min_value=0, max_value=1),
            }),
            
            'ml_models': TableValidator('ml_models', {
                'model_type': StringValidator('model_type', required=True, max_length=100),
                'stage': StringValidator('stage', required=True,
                                       allowed_values=['testing', 'validation', 'scaling']),
                'version': IntegerValidator('version', required=True, min_value=1),
                'model_data': CustomValidator('model_data', 
                                             self._validate_model_data,
                                             self._sanitize_model_data),
                'precision': FloatValidator('precision', min_value=0, max_value=1),
                'recall': FloatValidator('recall', min_value=0, max_value=1),
                'f1_score': FloatValidator('f1_score', min_value=0, max_value=1),
                'is_active': BooleanValidator('is_active'),
                'trained_at': DateValidator('trained_at', date_format="%Y-%m-%dT%H:%M:%S"),
                'metadata': JSONValidator('metadata'),
            }),
            
            'historical_data': TableValidator('historical_data', {
                'ad_id': StringValidator('ad_id', required=True, max_length=100),
                'lifecycle_id': StringValidator('lifecycle_id', max_length=100),
                'stage': StringValidator('stage', required=True,
                                       allowed_values=['testing', 'validation', 'scaling']),
                'metric_name': StringValidator('metric_name', required=True, max_length=100),
                'metric_value': FloatValidator('metric_value', required=True),
                'ts_iso': DateValidator('ts_iso', date_format="%Y-%m-%dT%H:%M:%S"),
                'ts_epoch': IntegerValidator('ts_epoch', min_value=0),
            }),
            
            'time_series_data': TableValidator('time_series_data', {
                'ad_id': StringValidator('ad_id', required=True, max_length=100),
                'lifecycle_id': StringValidator('lifecycle_id', max_length=100),
                'stage': StringValidator('stage', required=True,
                                       allowed_values=['testing', 'validation', 'scaling']),
                'metric_name': StringValidator('metric_name', required=True, max_length=100),
                'metric_value': FloatValidator('metric_value', required=True),
                'timestamp': DateValidator('timestamp', date_format="%Y-%m-%dT%H:%M:%S"),
            }),
            
            'ad_creation_times': TableValidator('ad_creation_times', {
                'ad_id': StringValidator('ad_id', required=True, max_length=100),
                'stage': StringValidator('stage', required=True,
                                       allowed_values=['testing', 'validation', 'scaling']),
                'created_at_iso': DateValidator('created_at_iso', date_format="%Y-%m-%dT%H:%M:%S"),
                'created_at_epoch': IntegerValidator('created_at_epoch', min_value=0),
            }),
            
            'ml_predictions': TableValidator('ml_predictions', {
                'ad_id': StringValidator('ad_id', required=True, max_length=100),
                'lifecycle_id': StringValidator('lifecycle_id', max_length=100),
                'model_id': StringValidator('model_id', max_length=100),
                'stage': StringValidator('stage', required=True,
                                       allowed_values=['testing', 'validation', 'scaling']),
                'prediction_type': StringValidator('prediction_type', max_length=100),
                'prediction_value': FloatValidator('prediction_value'),
                'created_at': DateValidator('created_at', date_format="%Y-%m-%dT%H:%M:%S"),
            }),
            
            'learning_events': TableValidator('learning_events', {
                'event_type': StringValidator('event_type', required=True, max_length=100),
                'stage': StringValidator('stage', required=True,
                                       allowed_values=['testing', 'validation', 'scaling']),
                'created_at': DateValidator('created_at', date_format="%Y-%m-%dT%H:%M:%S"),
            }),
        }
    
    def _validate_similarity_vector(self, value: Any, data: Dict[str, Any]) -> List[str]:
        """Validate similarity vector field."""
        errors = []
        
        if value is None or value == '':
            return errors
        
        # Check if it's a list of floats
        if isinstance(value, list):
            if len(value) == 0:
                errors.append("Similarity vector cannot be empty")
            elif not all(isinstance(x, (int, float)) for x in value):
                errors.append("Similarity vector must contain only numbers")
            elif len(value) != 384:  # Standard embedding size
                errors.append(f"Similarity vector must have exactly 384 dimensions, got {len(value)}")
        else:
            errors.append("Similarity vector must be a list of numbers")
        
        return errors
    
    def _sanitize_similarity_vector(self, value: Any) -> Optional[List[float]]:
        """Sanitize similarity vector."""
        if value is None or value == '':
            return None
        
        if isinstance(value, list):
            try:
                return [float(x) for x in value if isinstance(x, (int, float))]
            except (ValueError, TypeError):
                return None
        
        return None
    
    def _validate_tags_array(self, value: Any, data: Dict[str, Any]) -> List[str]:
        """Validate tags array field."""
        errors = []
        
        if value is None or value == '':
            return errors
        
        if isinstance(value, list):
            if not all(isinstance(x, str) for x in value):
                errors.append("Tags must be a list of strings")
            elif len(value) > 10:
                errors.append("Tags cannot exceed 10 items")
        else:
            errors.append("Tags must be a list")
        
        return errors
    
    def _sanitize_tags_array(self, value: Any) -> Optional[List[str]]:
        """Sanitize tags array."""
        if value is None or value == '':
            return None
        
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        
        return None
    
    def _validate_model_data(self, value: Any, data: Dict[str, Any]) -> List[str]:
        """Validate ML model data."""
        errors = []
        
        if value is None or value == '':
            return errors
        
        if isinstance(value, str):
            # Check if it's valid hex
            try:
                bytes.fromhex(value)
                if len(value) < 1000:  # Minimum reasonable model size
                    errors.append("Model data appears too small")
            except ValueError:
                errors.append("Model data must be valid hex string")
        else:
            errors.append("Model data must be a hex string")
        
        return errors
    
    def _sanitize_model_data(self, value: Any) -> Optional[str]:
        """Sanitize model data."""
        if value is None or value == '':
            return None
        
        if isinstance(value, str):
            try:
                # Validate hex and return
                bytes.fromhex(value)
                return value
            except ValueError:
                return None
        
        return None
    
    def validate_table_data(self, table_name: str, data: Dict[str, Any]) -> ValidationResult:
        """Validate data for a specific table."""
        if table_name not in self.table_validators:
            raise ValueError(f"No validator found for table '{table_name}'")
        
        validator = self.table_validators[table_name]
        return validator.validate(data)
    
    def validate_batch_data(self, table_name: str, data_list: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate a batch of data for a specific table."""
        results = []
        
        for i, data in enumerate(data_list):
            try:
                result = self.validate_table_data(table_name, data)
                results.append(result)
            except Exception as e:
                # Create error result for this item
                error_result = ValidationResult(
                    is_valid=False,
                    errors=[f"Validation error for item {i}: {str(e)}"],
                    warnings=[],
                    info=[],
                    sanitized_data={}
                )
                results.append(error_result)
        
        return results

# Global validator instance
data_validator = SupabaseDataValidator()

def validate_supabase_data(table_name: str, data: Dict[str, Any], 
                          strict_mode: bool = True) -> ValidationResult:
    """
    Validate data before inserting into Supabase.
    
    Args:
        table_name: Name of the Supabase table
        data: Data dictionary to validate
        strict_mode: If True, errors block insertion. If False, warnings are logged but insertion continues.
    
    Returns:
        ValidationResult with validation status and sanitized data
    """
    try:
        result = data_validator.validate_table_data(table_name, data)
        
        # Log validation results
        if result.errors:
            logger.error(f"Validation errors for {table_name}: {result.errors}")
        
        if result.warnings:
            logger.warning(f"Validation warnings for {table_name}: {result.warnings}")
        
        if result.info:
            logger.info(f"Validation info for {table_name}: {result.info}")
        
        # In strict mode, errors prevent insertion
        if strict_mode and result.errors:
            logger.error(f"Data validation failed for {table_name}. Insertion blocked.")
            return result
        
        return result
        
    except Exception as e:
        logger.error(f"Validation error for {table_name}: {e}")
        return ValidationResult(
            is_valid=False,
            errors=[f"Validation system error: {str(e)}"],
            warnings=[],
            info=[],
            sanitized_data={}
        )

def validate_and_sanitize_data(table_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize data for Supabase insertion.
    Returns sanitized data ready for insertion.
    """
    result = validate_supabase_data(table_name, data, strict_mode=True)
    
    if not result.is_valid:
        raise ValidationError(f"Data validation failed: {', '.join(result.errors)}")
    
    return result.sanitized_data
