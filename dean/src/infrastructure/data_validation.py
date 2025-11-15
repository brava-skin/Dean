from __future__ import annotations

import re
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from config import CREATIVE_PERFORMANCE_STAGE_VALUE

logger = logging.getLogger(__name__)
STRICT_MODE = os.getenv("STRICT_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}

class ValidationError(Exception):
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]
    sanitized_data: Dict[str, Any]

class FieldValidator:
    def __init__(self, field_name: str, required: bool = False, severity: ValidationSeverity = ValidationSeverity.ERROR, default: Any = None):
        self.field_name = field_name
        self.required = required
        self.severity = severity
        self.default = default
    
    def validate(self, value: Any, data: Dict[str, Any]) -> List[str]:
        errors = []
        if self.required and (value is None or value == ''):
            errors.append(f"Field '{self.field_name}' is required")
            return errors
        if not self.required and (value is None or value == ''):
            return errors
        field_errors = self._validate_field(value, data)
        errors.extend(field_errors)
        return errors
    
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        return []
    
    def sanitize(self, value: Any) -> Any:
        if (value is None or value == '') and self.default is not None:
            return self.default
        if value == '' and not self.required:
            return None
        return value

class StringValidator(FieldValidator):
    def __init__(self, field_name: str, max_length: int = None, min_length: int = None, 
                 pattern: str = None, allowed_values: List[str] = None, **kwargs):
        super().__init__(field_name, **kwargs)
        self.max_length = max_length
        self.min_length = min_length
        self.pattern = pattern
        self.allowed_values = allowed_values
    
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        errors = []
        str_value = str(value) if value is not None else ""
        if self.min_length and len(str_value) < self.min_length:
            errors.append(f"Field '{self.field_name}' must be at least {self.min_length} characters")
        if self.max_length and len(str_value) > self.max_length:
            errors.append(f"Field '{self.field_name}' must be at most {self.max_length} characters")
        if self.pattern and str_value:
            if not re.match(self.pattern, str_value):
                errors.append(f"Field '{self.field_name}' does not match required pattern")
        if self.allowed_values and str_value not in self.allowed_values:
            errors.append(f"Field '{self.field_name}' must be one of: {', '.join(self.allowed_values)}")
        return errors
    
    def sanitize(self, value: Any) -> Optional[str]:
        if value is None or value == '':
            if self.default is not None:
                return str(self.default)
            if not self.required:
                return None
            return ""
        return str(value).strip()

class IntegerValidator(FieldValidator):
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
        if self.min_value is not None and int_value < self.min_value:
            errors.append(f"Field '{self.field_name}' must be at least {self.min_value}")
        if self.max_value is not None and int_value > self.max_value:
            errors.append(f"Field '{self.field_name}' must be at most {self.max_value}")
        return errors
    
    def sanitize(self, value: Any) -> Optional[int]:
        if value is None or value == '':
            if self.default is not None:
                return int(self.default)
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            if self.default is not None:
                return int(self.default)
            return None

class FloatValidator(FieldValidator):
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
        if not self.allow_inf and (float_value == float('inf') or float_value == float('-inf')):
            errors.append(f"Field '{self.field_name}' cannot be infinity")
        if not self.allow_nan and float_value != float_value:
            errors.append(f"Field '{self.field_name}' cannot be NaN")
        if self.min_value is not None and float_value < self.min_value:
            errors.append(f"Field '{self.field_name}' must be at least {self.min_value}")
        if self.max_value is not None and float_value > self.max_value:
            errors.append(f"Field '{self.field_name}' must be at most {self.max_value}")
        return errors
    
    def sanitize(self, value: Any) -> Optional[float]:
        if value is None or value == '':
            if self.default is not None:
                return float(self.default)
            return None
        try:
            float_value = float(value)
            if not self.allow_inf and (float_value == float('inf') or float_value == float('-inf')):
                return 0.0
            if not self.allow_nan and float_value != float_value:
                return 0.0
            return float_value
        except (ValueError, TypeError):
            return None

class BooleanValidator(FieldValidator):
    def _validate_field(self, value: Any, data: Dict[str, Any]) -> List[str]:
        errors = []
        if value not in [True, False, 1, 0, "true", "false", "1", "0", "yes", "no"]:
            errors.append(f"Field '{self.field_name}' must be a boolean value")
        return errors
    
    def sanitize(self, value: Any) -> Optional[bool]:
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
    def __init__(self, field_name: str, date_format: str = "%Y-%m-%d", **kwargs):
        super().__init__(field_name, **kwargs)
        self.date_format = date_format
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
            try:
                datetime.strptime(value, self.date_format)
                return errors
            except ValueError:
                pass
            for fmt in self.common_formats:
                try:
                    datetime.strptime(value, fmt)
                    return errors
                except ValueError:
                    continue
            errors.append(f"Field '{self.field_name}' must be a valid date in format {self.date_format}")
        return errors
    
    def sanitize(self, value: Any) -> Optional[str]:
        if value is None or value == '':
            return None
        if isinstance(value, str):
            try:
                datetime.strptime(value, self.date_format)
                return value
            except ValueError:
                pass
            for fmt in self.common_formats:
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt.strftime(self.date_format)
                except ValueError:
                    continue
            return None
        return None

class JSONValidator(FieldValidator):
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
        try:
            return self.sanitize_func(value)
        except Exception:
            return value

class UnitEnforcer:
    FRACTION_FIELDS = {
        "ctr",
        "unique_ctr",
        "atc_rate",
        "ic_rate",
        "purchase_rate",
        "atc_to_ic_rate",
        "ic_to_purchase_rate",
        "engagement_rate",
        "conversion_rate",
        "avg_ctr",
    }

    NON_NEGATIVE_FIELDS = {
        "cpc",
        "cpm",
        "cpa",
        "roas",
        "spend",
    }

    EPS = 1e-9

    def enforce(self, table_name: str, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []

        for field in self.FRACTION_FIELDS:
            if field not in data:
                continue
            value = data.get(field)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"{field}: expected numeric fraction, got {value!r}")
                continue
            if numeric < -self.EPS or numeric > 1 + self.EPS:
                errors.append(f"{field}: fraction {numeric:.6f} out of expected [0,1] range")

        for field in self.NON_NEGATIVE_FIELDS:
            if field not in data:
                continue
            value = data.get(field)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"{field}: expected numeric value, got {value!r}")
                continue
            if numeric < -self.EPS:
                errors.append(f"{field}: negative value {numeric:.6f} is not allowed")

        return errors

class TableValidator:
    def __init__(self, table_name: str, validators: Dict[str, FieldValidator]):
        self.table_name = table_name
        self.validators = validators
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []
        info = []
        sanitized_data = {}
        for field_name, validator in self.validators.items():
            value = data.get(field_name)
            if value is None and validator.default is not None:
                value = validator.default
            field_errors = validator.validate(value, data)
            for error in field_errors:
                if validator.severity == ValidationSeverity.ERROR:
                    errors.append(f"{field_name}: {error}")
                elif validator.severity == ValidationSeverity.WARNING:
                    warnings.append(f"{field_name}: {error}")
                else:
                    info.append(f"{field_name}: {error}")
            sanitized_value = validator.sanitize(value)
            if sanitized_value is not None or field_name in data:
                sanitized_data[field_name] = sanitized_value
        for field_name, validator in self.validators.items():
            if validator.required and field_name not in data and validator.default is None:
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
    def __init__(self):
        self.table_validators = self._create_table_validators()
        self.unit_enforcer = UnitEnforcer()
    
    def _create_table_validators(self) -> Dict[str, TableValidator]:
        return {
            'ads': TableValidator('ads', {
                'ad_id': StringValidator('ad_id', required=True, max_length=100),
                'creative_id': StringValidator('creative_id', required=True, max_length=100),
                'campaign_id': StringValidator('campaign_id', max_length=100),
                'adset_id': StringValidator('adset_id', max_length=100),
                'status': StringValidator('status', required=True,
                                        allowed_values=['active', 'paused', 'killed'], default='active'),
                'kill_reason': StringValidator('kill_reason', max_length=500),
                'created_at': DateValidator('created_at', date_format="%Y-%m-%dT%H:%M:%S"),
                'killed_at': DateValidator('killed_at', date_format="%Y-%m-%dT%H:%M:%S"),
                'storage_url': StringValidator('storage_url', required=True, max_length=500),
                'storage_path': StringValidator('storage_path', required=True, max_length=500),
                'file_size_bytes': IntegerValidator('file_size_bytes', required=True, min_value=0),
                'file_type': StringValidator('file_type', required=True, max_length=50, default='image/jpeg'),
                'headline': StringValidator('headline', max_length=200),
                'primary_text': StringValidator('primary_text', max_length=1000),
                'description': StringValidator('description', max_length=500),
                'image_prompt': StringValidator('image_prompt', max_length=2000),
                'performance_score': FloatValidator('performance_score', min_value=0, max_value=1),
                'fatigue_index': FloatValidator('fatigue_index', min_value=0, max_value=1),
                'metadata': JSONValidator('metadata', required=True),
                'updated_at': DateValidator('updated_at', date_format="%Y-%m-%dT%H:%M:%S"),
            }),
            
            'performance_metrics': TableValidator('performance_metrics', {
                'ad_id': StringValidator('ad_id', required=True, max_length=100),
                'date_start': DateValidator('date_start', required=True),
                'date_end': DateValidator('date_end', required=True),
                'window_type': StringValidator('window_type', required=True, 
                                             allowed_values=['1d', '7d', '30d']),
                'impressions': IntegerValidator('impressions', required=True, min_value=0, default=0),
                'clicks': IntegerValidator('clicks', required=True, min_value=0, default=0),
                'spend': FloatValidator('spend', required=True, min_value=0, max_value=999999.99, default=0),
                'purchases': IntegerValidator('purchases', required=True, min_value=0, default=0),
                'add_to_cart': IntegerValidator('add_to_cart', required=True, min_value=0, default=0),
                'initiate_checkout': IntegerValidator('initiate_checkout', required=True, min_value=0, default=0),
                'ctr': FloatValidator('ctr', min_value=0, max_value=1),
                'cpc': FloatValidator('cpc', min_value=0, max_value=999.99),
                'cpm': FloatValidator('cpm', min_value=0, max_value=9999.99),
                'roas': FloatValidator('roas', min_value=0, max_value=999.99),
                'cpa': FloatValidator('cpa', min_value=0, max_value=999.99),
                'atc_rate': FloatValidator('atc_rate', min_value=0, max_value=1),
                'purchase_rate': FloatValidator('purchase_rate', min_value=0, max_value=1),
                'created_at': DateValidator('created_at', date_format="%Y-%m-%dT%H:%M:%S"),
                'updated_at': DateValidator('updated_at', date_format="%Y-%m-%dT%H:%M:%S"),
            }),
            
            'creative_library': TableValidator('creative_library', {
                'creative_id': StringValidator('creative_id', required=True, max_length=100),
                'creative_type': StringValidator('creative_type', required=True,
                                                allowed_values=['headline', 'description', 'primary_text']),
                'content': StringValidator('content', required=True, max_length=2000),
                'category': StringValidator('category', max_length=100),
                'tags': CustomValidator('tags', self._validate_tags_array, self._sanitize_tags_array),
                'performance_score': FloatValidator('performance_score', min_value=0, max_value=1),
                'usage_count': IntegerValidator('usage_count', min_value=0, default=0),
                'created_by': StringValidator('created_by', max_length=100, default='system'),
                'metadata': JSONValidator('metadata', required=True),
                'created_at': DateValidator('created_at', date_format="%Y-%m-%dT%H:%M:%S"),
                'updated_at': DateValidator('updated_at', date_format="%Y-%m-%dT%H:%M:%S"),
            }),
            
            'insert_failures': TableValidator('insert_failures', {
                'table_name': StringValidator('table_name', required=True, max_length=100),
                'payload': JSONValidator('payload', required=True),
                'error_message': StringValidator('error_message', required=True, max_length=2000),
                'created_at': DateValidator('created_at', date_format="%Y-%m-%dT%H:%M:%S"),
                'retry_count': IntegerValidator('retry_count', min_value=0, default=0),
                'last_retry_at': DateValidator('last_retry_at', date_format="%Y-%m-%dT%H:%M:%S"),
            }),
        }
    
    def _validate_tags_array(self, value: Any, data: Dict[str, Any]) -> List[str]:
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
        if value is None or value == '':
            return None
        
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        
        return None
    
    def validate_table_data(self, table_name: str, data: Dict[str, Any]) -> ValidationResult:
        if table_name not in self.table_validators:
            raise ValueError(f"No validator found for table '{table_name}'")
        
        validator = self.table_validators[table_name]
        result = validator.validate(data)

        if table_name == 'creative_performance':
            stage_value = result.sanitized_data.get('stage')
            if stage_value == 'asc_plus':
                result.sanitized_data['stage'] = CREATIVE_PERFORMANCE_STAGE_VALUE

        if STRICT_MODE:
            unit_errors = self.unit_enforcer.enforce(table_name, result.sanitized_data)
            if unit_errors:
                result.errors.extend(unit_errors)
                result.is_valid = False
        
        return result
    
    def validate_batch_data(self, table_name: str, data_list: List[Dict[str, Any]]) -> List[ValidationResult]:
        results = []
        for i, data in enumerate(data_list):
            try:
                result = self.validate_table_data(table_name, data)
                results.append(result)
            except Exception as e:
                error_result = ValidationResult(
                    is_valid=False,
                    errors=[f"Validation error for item {i}: {str(e)}"],
                    warnings=[],
                    info=[],
                    sanitized_data={}
                )
                results.append(error_result)
        
        return results

data_validator = SupabaseDataValidator()

def validate_supabase_data(table_name: str, data: Dict[str, Any], 
                          strict_mode: bool = True) -> ValidationResult:
    try:
        result = data_validator.validate_table_data(table_name, data)
        if result.errors:
            logger.error(f"Validation errors for {table_name}: {result.errors}")
        if result.warnings:
            logger.warning(f"Validation warnings for {table_name}: {result.warnings}")
        if result.info:
            logger.info(f"Validation info for {table_name}: {result.info}")
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
    result = validate_supabase_data(table_name, data, strict_mode=True)
    
    if not result.is_valid:
        raise ValidationError(f"Data validation failed: {', '.join(result.errors)}")
    
    return result.sanitized_data


class DateValidationUtility:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DateValidationUtility")
        self.min_valid_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.max_future_offset = timedelta(hours=1)
        
    def get_current_timestamp(self) -> datetime:
        return datetime.now(timezone.utc)
    
    def validate_and_correct_date(self, 
                                date_value: Any, 
                                field_name: str = "timestamp",
                                allow_future: bool = False,
                                max_future_hours: int = 1) -> datetime:
        now = self.get_current_timestamp()
        
        if date_value is None or date_value == "":
            self.logger.debug(f"Empty {field_name}, using current timestamp")
            return now
        
        if isinstance(date_value, str):
            try:
                if ('*' in date_value or '***' in date_value or 
                    len(date_value) < 8 or
                    date_value.count('-') < 2 or
                    not any(c.isdigit() for c in date_value)):
                    self.logger.warning(f"Malformed date {field_name} '{date_value}', using current timestamp")
                    return now
                    
                if 'T' in date_value:
                    parsed_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                else:
                    try:
                        parsed_date = datetime.strptime(date_value, '%Y-%m-%d')
                    except ValueError:
                        parsed_date = datetime.strptime(date_value, '%Y-%m-%d %H:%M:%S%z')
            except (ValueError, TypeError):
                self.logger.warning(f"Failed to parse {field_name} '{date_value}', using current timestamp")
                return now
                
        elif isinstance(date_value, (int, float)):
            try:
                if date_value > 1e10:
                    parsed_date = datetime.fromtimestamp(date_value / 1000, tz=timezone.utc)
                else:
                    parsed_date = datetime.fromtimestamp(date_value, tz=timezone.utc)
            except (ValueError, OSError):
                self.logger.warning(f"Invalid epoch timestamp {field_name} '{date_value}', using current timestamp")
                return now
                
        elif isinstance(date_value, datetime):
            parsed_date = date_value
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        else:
            self.logger.warning(f"Unsupported date type {field_name} '{type(date_value)}', using current timestamp")
            return now
        
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if self.min_valid_date.tzinfo is None:
            self.min_valid_date = self.min_valid_date.replace(tzinfo=timezone.utc)
        
        try:
            _ = parsed_date < now
        except TypeError as tz_error:
            self.logger.warning(f"Timezone comparison error for {field_name}: {tz_error}, using current timestamp")
            return now
            
        if parsed_date < self.min_valid_date:
            self.logger.warning(f"{field_name} too old ({parsed_date}), using current timestamp")
            return now
            
        if not allow_future and parsed_date > now + self.max_future_offset:
            self.logger.warning(f"{field_name} in future ({parsed_date}), using current timestamp")
            return now
            
        if allow_future and parsed_date > now + timedelta(hours=max_future_hours):
            self.logger.warning(f"{field_name} too far in future ({parsed_date}), using current timestamp")
            return now
        
        self.logger.debug(f"{field_name} validated: {parsed_date}")
        return parsed_date
    
    def validate_data_timestamps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        timestamp_fields = [
            'created_at', 'updated_at', 'timestamp', 'expires_at',
            'date_start', 'date_end', 'trained_at', 'recorded_at'
        ]
        
        for field in timestamp_fields:
            if field in data and data[field] is not None:
                try:
                    if field == 'expires_at':
                        horizon_hours = data.get('prediction_horizon_hours', 24)
                        try:
                            horizon_hours = max(1, int(float(horizon_hours)))
                        except (TypeError, ValueError):
                            horizon_hours = 24
                        validated_date = self.validate_and_correct_date(
                            data[field],
                            field,
                            allow_future=True,
                            max_future_hours=horizon_hours + 1,
                        )
                    else:
                        validated_date = self.validate_and_correct_date(
                            data[field],
                            field
                        )
                    data[field] = validated_date.isoformat()
                except Exception as e:
                    self.logger.error(f"Error validating {field}: {e}")
                    data[field] = self.get_current_timestamp().isoformat()
        
        return data


date_validator = DateValidationUtility()

def get_validated_timestamp(date_value: Any = None, field_name: str = "timestamp") -> datetime:
    return date_validator.validate_and_correct_date(date_value, field_name)

def add_current_timestamps(data: Dict[str, Any]) -> Dict[str, Any]:
    now = date_validator.get_current_timestamp()
    if 'created_at' not in data or data['created_at'] is None:
        data['created_at'] = now.isoformat()
    if 'updated_at' not in data or data['updated_at'] is None:
        data['updated_at'] = now.isoformat()
    if 'timestamp' not in data or data['timestamp'] is None:
        data['timestamp'] = now.isoformat()
    return data

def validate_all_timestamps(data: Dict[str, Any]) -> Dict[str, Any]:
    return date_validator.validate_data_timestamps(data)




class CreativeIntelligenceOptimizer:
    def __init__(self, supabase_client: Any) -> None:
        self.client = supabase_client
    
    def calculate_performance_metrics(
        self,
        creative_id: str,
        ad_id: str,
        days_back: int = 30,
    ) -> Dict[str, float]:
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            response = self.client.table('performance_metrics').select(
                'ctr, cpa, roas, spend, purchases, impressions, clicks'
            ).eq('ad_id', ad_id).gte('date_start', start_date).execute()
            
            if not response.data:
                try:
                    creative_response = self.client.table('ads').select(
                        'ad_id'
                    ).eq('creative_id', creative_id).execute()
                    
                    if creative_response.data:
                        ad_ids = [row.get('ad_id') for row in creative_response.data if row.get('ad_id')]
                        if ad_ids:
                            response = self.client.table('performance_metrics').select(
                                'ctr, cpa, roas, spend, purchases, impressions, clicks'
                            ).in_('ad_id', ad_ids).gte('date_start', start_date).execute()
                except Exception:
                    pass
            
            if not response.data:
                return {
                    'avg_ctr': 0.0,
                    'avg_cpa': 0.0,
                    'avg_roas': 0.0,
                    'performance_rank': 999,
                    'performance_score': 0.0,
                    'fatigue_index': 0.0,
                }
            
            df = pd.DataFrame(response.data)
            
            if 'spend' in df.columns and df['spend'].sum() > 0:
                total_spend = df['spend'].sum()
                avg_ctr = (df['ctr'] * df['spend']).sum() / total_spend if total_spend > 0 else df['ctr'].mean()
                avg_cpa = (df['cpa'] * df['spend']).sum() / total_spend if total_spend > 0 else df['cpa'].mean()
                avg_roas = (df['roas'] * df['spend']).sum() / total_spend if total_spend > 0 else df['roas'].mean()
            else:
                avg_ctr = df['ctr'].mean() if 'ctr' in df.columns else 0.0
                avg_cpa = df['cpa'].mean() if 'cpa' in df.columns else 0.0
                avg_roas = df['roas'].mean() if 'roas' in df.columns else 0.0
            
            performance_score = self._calculate_performance_score(avg_ctr, avg_cpa, avg_roas)
            fatigue_index = self._calculate_fatigue_index(df)
            performance_rank = 999
            
            return {
                'avg_ctr': float(avg_ctr) if not np.isnan(avg_ctr) else 0.0,
                'avg_cpa': float(avg_cpa) if not np.isnan(avg_cpa) else 0.0,
                'avg_roas': float(avg_roas) if not np.isnan(avg_roas) else 0.0,
                'performance_rank': performance_rank,
                'performance_score': float(performance_score),
                'fatigue_index': float(fatigue_index),
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics for {creative_id}: {e}")
            return {
                'avg_ctr': 0.0,
                'avg_cpa': 0.0,
                'avg_roas': 0.0,
                'performance_rank': 999,
                'performance_score': 0.0,
                'fatigue_index': 0.0,
            }
    
    def _calculate_performance_score(
        self,
        avg_ctr: float,
        avg_cpa: float,
        avg_roas: float,
    ) -> float:
        ctr_score = min(1.0, avg_ctr / 0.02)
        cpa_score = min(1.0, max(0.0, (50.0 - avg_cpa) / 50.0))
        roas_score = min(1.0, avg_roas / 3.0)
        
        performance_score = (
            ctr_score * 0.3 +
            cpa_score * 0.3 +
            roas_score * 0.4
        )
        
        return max(0.0, min(1.0, performance_score))
    
    def _calculate_fatigue_index(self, df: pd.DataFrame) -> float:
        if len(df) < 3:
            return 0.0
        
        if 'roas' in df.columns and len(df) >= 3:
            recent_roas = df['roas'].tail(3).mean()
            older_roas = df['roas'].head(len(df) - 3).mean() if len(df) > 3 else recent_roas
            
            if older_roas > 0:
                decline_pct = (older_roas - recent_roas) / older_roas
                fatigue_index = max(0.0, min(1.0, decline_pct))
                return fatigue_index
        
        return 0.0
    
    def update_creative_performance(
        self,
        creative_id: str,
        ad_id: str,
        force_recalculate: bool = False,
    ) -> bool:
        try:
            if not force_recalculate:
                existing = self.client.table('ads').select(
                    'performance_score, fatigue_index, updated_at'
                ).eq('ad_id', ad_id).execute()
                
                if existing.data:
                    last_update = existing.data[0].get('updated_at')
                    if last_update:
                        try:
                            last_update_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                            hours_since_update = (datetime.now(timezone.utc) - last_update_dt).total_seconds() / 3600
                            if hours_since_update < 6:
                                return True
                        except Exception:
                            pass
            
            metrics = self.calculate_performance_metrics(creative_id, ad_id)
            
            update_data = {
                'performance_score': metrics['performance_score'],
                'fatigue_index': metrics['fatigue_index'],
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            
            self.client.table('ads').update(update_data).eq(
                'ad_id', ad_id
            ).execute()
            
            logger.info(f"✅ Updated performance metrics for ad {ad_id} (creative {creative_id})")
            return True
        except Exception as e:
            logger.error(f"Error updating creative performance for {creative_id}: {e}")
            return False
    
    def calculate_performance_ranks(self, stage: str = 'asc_plus') -> bool:
        logger.debug("Performance ranks calculation skipped - not applicable with consolidated schema")
        return True
    
    def backfill_missing_metrics(self, stage: str = 'asc_plus') -> Dict[str, int]:
        try:
            response = self.client.table('ads').select(
                'ad_id, creative_id, performance_score, fatigue_index'
            ).limit(1000).execute()
            
            if not response.data:
                return {'updated': 0, 'skipped': 0, 'errors': 0}
            
            updated = 0
            skipped = 0
            errors = 0
            
            for creative in response.data:
                creative_id = creative.get('creative_id')
                ad_id = creative.get('ad_id')
                
                if not creative_id or not ad_id:
                    skipped += 1
                    continue
                
                avg_ctr = creative.get('avg_ctr')
                avg_cpa = creative.get('avg_cpa')
                avg_roas = creative.get('avg_roas')
                
                needs_update = (
                    avg_ctr is None or 
                    avg_cpa is None or 
                    avg_roas is None or
                    (avg_ctr == 0.0 and avg_cpa == 0.0 and avg_roas == 0.0)
                )
                
                if needs_update:
                    if self.update_creative_performance(creative_id, ad_id, force_recalculate=True):
                        updated += 1
                    else:
                        errors += 1
                else:
                    skipped += 1
            
            logger.info(f"✅ Backfilled metrics: {updated} updated, {skipped} skipped, {errors} errors")
            return {'updated': updated, 'skipped': skipped, 'errors': errors}
        except Exception as e:
            logger.error(f"Error backfilling metrics: {e}")
            return {'updated': 0, 'skipped': 0, 'errors': 1}


__all__ = [
    "ValidationError",
    "ValidationResult",
    "validate_supabase_data",
    "validate_and_sanitize_data",
    "DateValidationUtility",
    "date_validator",
    "get_validated_timestamp",
    "add_current_timestamps",
    "validate_all_timestamps",
    "CreativeIntelligenceOptimizer",
]
