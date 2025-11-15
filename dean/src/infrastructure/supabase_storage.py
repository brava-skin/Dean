"""
Supabase-based storage for ad creation times and historical data.
This replaces SQLite storage to ensure ML system has access to all data.
Includes validated operations and helper functions.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
import logging
import random
import os
import re
from supabase import create_client, Client

from .data_validation import (
    validate_supabase_data,
    validate_and_sanitize_data,
    ValidationError,
    ValidationResult,
)

# Import date validation from data_validation
from .data_validation import date_validator, validate_all_timestamps

logger = logging.getLogger(__name__)

# =====================================================
# ERROR LOGGING HELPERS
# =====================================================

def _sample_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    keys_of_interest = [
        "id",
        "ad_id",
        "creative_id",
        "lifecycle_id",
        "stage",
        "window_type",
        "date_start",
        "date_end",
        "model_type",
        "model_name",
    ]

    sample: Dict[str, Any] = {}
    for key in keys_of_interest:
        if key in payload:
            sample[key] = payload[key]

    for metric_key in ("ctr", "cpc", "cpm", "roas"):
        if metric_key in payload:
            sample[metric_key] = payload[metric_key]

    return sample or None


def _log_supabase_error(operation: str, table: str, error: Any, payload: Optional[Dict[str, Any]] = None) -> None:
    code: Optional[str] = None
    details: Optional[str] = None

    if isinstance(error, dict):
        code = error.get("code")
        details = error.get("details") or error.get("hint")
    elif hasattr(error, "args"):
        for arg in error.args:
            if isinstance(arg, dict):
                code = code or arg.get("code")
                details = details or arg.get("details") or arg.get("hint")

    message = str(error)
    suggestions: List[str] = []

    column_match = re.search(r"Could not find the '([^']+)' column", message)
    if column_match:
        missing_column = column_match.group(1)
        suggestions.append(
            f"Supabase schema cache is missing column '{missing_column}' on {table}. Apply the latest migrations or run `supabase db refresh schema`/`supabase db reset --linked`."
        )

    if "Field 'ctr' must be at most 1" in message:
        ctr_value = None
        if payload and isinstance(payload, dict):
            ctr_value = payload.get("ctr")
        if ctr_value is not None:
            suggestions.append(f"CTR must be a fraction <= 1. Offending value: {ctr_value}")
        suggestions.append("Ensure metrics store CTR as clicks/impressions (fraction).")

    if code == "PGRST204" and not suggestions:
        suggestions.append(
            "Supabase schema cache may be stale. Trigger a schema refresh or remove unused fields from the insert payload."
        )

    sample = _sample_payload(payload)

    logger.error(
        "SUPABASE ERROR [%s.%s] code=%s message=%s details=%s suggestions=%s sample=%s",
        table,
        operation,
        code or "unknown",
        message,
        details or "n/a",
        "; ".join(suggestions) or "n/a",
        sample,
    )


# =====================================================
# HELPER FUNCTIONS
# =====================================================

class SupabaseStorage:
    """Supabase storage for ad creation times and historical data."""
    
    def __init__(self, supabase_client: Any):
        self.client = supabase_client
    
    def record_ad_creation(self, ad_id: str, lifecycle_id: str, stage: str, 
                          created_at: Optional[datetime] = None) -> None:
        """Record when an ad was created - updates ads table with created_at."""
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        
        try:
            # Ensure timestamps are within valid range (not future dates)
            now = datetime.now(timezone.utc)
            if created_at > now:
                created_at = now
            
            # Ensure created_at is not too far in the past (not before 2020)
            min_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            if created_at < min_date:
                created_at = now - timedelta(days=random.randint(1, 30))
            
            # Update ads table with created_at timestamp
            # Only update if ad exists (don't create new ad record here)
            data = {
                'created_at': created_at.isoformat(),
                'updated_at': now.isoformat(),
            }
            
            # Validate all timestamps
            data = validate_all_timestamps(data)
            
            # Update existing ad record (if it exists)
            try:
                self.client.table('ads').update(data).eq('ad_id', ad_id).execute()
                logger.debug(f"Updated ad creation time for {ad_id} in ads table")
            except Exception as e:
                # Ad might not exist yet, that's okay - it will be created when ad is created
                logger.debug(f"Ad {ad_id} not found in ads table yet (will be created later): {e}")
            
        except Exception as e:
            logger.error(f"Failed to record ad creation time for {ad_id}: {e}")
            # Don't raise - this is non-critical
    
    def get_ad_creation_time(self, ad_id: str) -> Optional[datetime]:
        """Get when an ad was created from ads table."""
        try:
            # Get from ads table
            response = self.client.table('ads').select('created_at').eq(
                'ad_id', ad_id
            ).execute()
            
            if response.data and response.data[0].get('created_at'):
                created_at_str = response.data[0]['created_at']
                try:
                    return datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                except:
                    pass
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get ad creation time for {ad_id}: {e}")
            return None
    
    def cleanup_invalid_creation_times(self) -> None:
        """Clean up invalid ad creation times with bad timestamps - now uses ads table."""
        try:
            # Get all ads with potentially invalid timestamps
            all_ads = self.client.table('ads').select('ad_id, created_at').execute()
            
            now = datetime.now(timezone.utc)
            min_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            
            for record in (all_ads.data or []):
                ad_id = record.get('ad_id')
                created_at_str = record.get('created_at')
                
                if not created_at_str:
                    continue
                
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                except:
                    continue
                
                # Check if timestamp is invalid (too old or future)
                if created_at < min_date or created_at > now:
                    # Update with current timestamp
                    self.record_ad_creation(
                        ad_id=ad_id,
                        lifecycle_id=record.get('lifecycle_id', ''),
                        stage=record.get('stage', 'asc_plus')
                    )
                    logger.info(f"Fixed invalid creation time for ad {ad_id}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup invalid creation times: {e}")
    
    def get_ad_age_days(self, ad_id: str) -> Optional[float]:
        """Get ad age in days."""
        creation_time = self.get_ad_creation_time(ad_id)
        if not creation_time:
            return None
        
        now = datetime.now(timezone.utc)
        age = now - creation_time
        return age.total_seconds() / 86400  # Convert to days
    
    def store_historical_data(self, ad_id: str, lifecycle_id: str, stage: str, 
                             metric_name: str, metric_value: float, 
                             timestamp: Optional[datetime] = None) -> None:
        """Store historical metric data - now uses performance_metrics table."""
        # Historical data is now tracked via performance_metrics table
        # This method is kept for backward compatibility but does nothing
        # Individual metric tracking is handled by performance_metrics aggregation
        logger.debug(f"Historical data storage for {ad_id}:{metric_name} - now handled by performance_metrics table")
        pass
    
    def get_historical_data(self, ad_id: str, metric_name: str, 
                           since_days: int = 7) -> List[Dict[str, Any]]:
        """Get historical metric data from performance_metrics table."""
        try:
            # Get performance metrics for the ad within the date range
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).date()
            
            # Map metric names to performance_metrics columns
            metric_column_map = {
                'ctr': 'ctr',
                'cpc': 'cpc',
                'cpm': 'cpm',
                'roas': 'roas',
                'cpa': 'cpa',
                'atc_rate': 'atc_rate',
                'purchase_rate': 'purchase_rate',
                'spend': 'spend',
                'impressions': 'impressions',
                'clicks': 'clicks',
                'purchases': 'purchases',
                'add_to_cart': 'add_to_cart',
            }
            
            column_name = metric_column_map.get(metric_name, 'impressions')
            
            response = self.client.table('performance_metrics').select(
                'date_start', 'date_end', column_name
            ).eq('ad_id', ad_id).gte('date_start', cutoff_date.isoformat()).order(
                'date_start', desc=False
            ).execute()
            
            # Convert to historical data format
            result = []
            for row in (response.data or []):
                value = row.get(column_name, 0)
                if value is not None:
                    date_start = row.get('date_start', '')
                    try:
                        ts_epoch = int(datetime.fromisoformat(date_start.replace('Z', '+00:00')).timestamp())
                    except:
                        ts_epoch = 0
                    result.append({
                        'metric_value': float(value),
                        'ts_iso': date_start,
                        'ts_epoch': ts_epoch
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {ad_id}: {e}")
            return []
    
    def get_historical_metrics(self, ad_id: str, metric_names: List[str], 
                              days_back: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Get comprehensive historical metrics from performance_metrics table."""
        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).date()
            
            # Map metric names to performance_metrics columns
            metric_column_map = {
                'ctr': 'ctr', 'cpc': 'cpc', 'cpm': 'cpm', 'roas': 'roas', 'cpa': 'cpa',
                'atc_rate': 'atc_rate', 'purchase_rate': 'purchase_rate',
                'spend': 'spend', 'impressions': 'impressions', 'clicks': 'clicks',
                'purchases': 'purchases', 'add_to_cart': 'add_to_cart',
            }
            
            # Get all performance metrics for the ad
            response = self.client.table('performance_metrics').select('*').eq(
                'ad_id', ad_id
            ).gte('date_start', cutoff_date.isoformat()).order('date_start', desc=False).execute()
            
            # Group by metric name
            result = {name: [] for name in metric_names}
            for row in (response.data or []):
                for metric_name in metric_names:
                    column_name = metric_column_map.get(metric_name)
                    if column_name and column_name in row and row[column_name] is not None:
                        date_start = row.get('date_start', '')
                        try:
                            ts_epoch = int(datetime.fromisoformat(date_start.replace('Z', '+00:00')).timestamp())
                        except:
                            ts_epoch = 0
                        result[metric_name].append({
                            'metric_value': float(row[column_name]),
                            'ts_iso': date_start,
                            'ts_epoch': ts_epoch
                        })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get historical metrics for {ad_id}: {e}")
            return {name: [] for name in metric_names}
    
    def cleanup_old_historical_data(self, days_to_keep: int = 90) -> int:
        """Clean up old performance metrics to prevent database bloat."""
        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).date()
            
            response = self.client.table('performance_metrics').delete().lt(
                'date_start', cutoff_date.isoformat()
            ).execute()
            
            logger.info(f"Cleaned up performance metrics older than {days_to_keep} days")
            return 0  # Can't get exact count from Supabase delete
            
        except Exception as e:
            logger.error(f"Failed to cleanup old performance metrics: {e}")
            return 0


def create_supabase_storage(supabase_client: Any) -> SupabaseStorage:
    """Factory function to create SupabaseStorage instance."""
    return SupabaseStorage(supabase_client)


# =====================================================
# VALIDATED SUPABASE CLIENT
# =====================================================

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


def _normalize_fraction_fields(data: Optional[Dict[str, Any]]) -> None:
    if not isinstance(data, dict):
        return

    for field in FRACTION_FIELDS:
        if field not in data or data[field] is None:
            continue
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            continue

        if value < 0:
            data[field] = 0.0
        elif value > 1:
            if value <= 100:
                data[field] = min(1.0, round(value / 100.0, 6))
            else:
                data[field] = 1.0


class ValidatedSupabaseClient:
    """
    Wrapper around Supabase client that automatically validates data before operations.
    """
    
    def __init__(self, url: str, key: str, enable_validation: bool = True):
        """
        Initialize validated Supabase client.
        
        Args:
            url: Supabase URL
            key: Supabase API key
            enable_validation: Whether to enable data validation (default: True)
        """
        self.client = create_client(url, key)
        self.enable_validation = enable_validation
        
        if enable_validation:
            logger.info("✅ Data validation enabled for all Supabase operations")
        else:
            logger.warning("⚠️ Data validation disabled - invalid data may be stored")
    
    def insert(self, table: str, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
               validate: bool = None) -> Any:
        """Insert data into Supabase table with validation."""
        should_validate = validate if validate is not None else self.enable_validation
        
        if should_validate:
            if isinstance(data, list):
                return self._insert_batch_validated(table, data)
            else:
                return self._insert_single_validated(table, data)
        else:
            return self.client.table(table).insert(data).execute()
    
    def upsert(self, table: str, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
               on_conflict: str = None, validate: bool = None) -> Any:
        """Upsert data into Supabase table with validation."""
        should_validate = validate if validate is not None else self.enable_validation
        
        if should_validate:
            if isinstance(data, list):
                return self._upsert_batch_validated(table, data, on_conflict)
            else:
                return self._upsert_single_validated(table, data, on_conflict)
        else:
            if on_conflict:
                query = self.client.table(table).upsert(data, on_conflict=on_conflict)
            else:
                query = self.client.table(table).upsert(data)
            return query.execute()
    
    def update(self, table: str, data: Dict[str, Any], 
               validate: bool = None, **kwargs) -> Any:
        """Update data in Supabase table with validation."""
        should_validate = validate if validate is not None else self.enable_validation
        
        if should_validate:
            sanitized_data = self._validate_and_sanitize(table, data)
            query = self.client.table(table).update(sanitized_data)
            
            # Apply query filters
            for key, value in kwargs.items():
                if key == 'eq' and 'value' in kwargs:
                    query = query.eq(value, kwargs['value'])
                elif key == 'eq2' and 'value2' in kwargs:
                    query = query.eq(value, kwargs['value2'])
                elif key == 'eq3' and 'value3' in kwargs:
                    query = query.eq(value, kwargs['value3'])
                elif key.startswith('value'):
                    continue
                elif hasattr(query, key):
                    query = getattr(query, key)(value)
            
            return query.execute()
        else:
            query = self.client.table(table).update(data)
            for key, value in kwargs.items():
                if key == 'eq' and 'value' in kwargs:
                    query = query.eq(value, kwargs['value'])
                elif key == 'eq2' and 'value2' in kwargs:
                    query = query.eq(value, kwargs['value2'])
                elif key == 'eq3' and 'value3' in kwargs:
                    query = query.eq(value, kwargs['value3'])
                elif key.startswith('value'):
                    continue
                elif hasattr(query, key):
                    query = getattr(query, key)(value)
            
            return query.execute()
    
    def _insert_single_validated(self, table: str, data: Dict[str, Any]) -> Any:
        """Insert single validated record."""
        _normalize_fraction_fields(data)
        try:
            sanitized_data = validate_and_sanitize_data(table, data)
            logger.debug(f"✅ Data validated for {table} insert")
            return self.client.table(table).insert(sanitized_data).execute()
        except ValidationError as e:
            _log_supabase_error("validation", table, e, data)
            raise e
        except Exception as e:
            _log_supabase_error("insert", table, e, data)
            raise e
    
    def _insert_batch_validated(self, table: str, data_list: List[Dict[str, Any]]) -> Any:
        """Insert batch of validated records."""
        validated_records = []
        failed_records = []
        
        for i, data in enumerate(data_list):
            _normalize_fraction_fields(data)
            try:
                sanitized_data = validate_and_sanitize_data(table, data)
                validated_records.append(sanitized_data)
            except ValidationError as e:
                logger.error(f"❌ Validation failed for {table} insert item {i}: {e}")
                failed_records.append((i, str(e)))
        
        if failed_records:
            logger.error(f"❌ {len(failed_records)} records failed validation for {table}")
            if len(validated_records) == 0:
                raise ValidationError(f"All records failed validation for {table}")
        
        if validated_records:
            logger.info(f"✅ Inserting {len(validated_records)} validated records into {table}")
            try:
                return self.client.table(table).insert(validated_records).execute()
            except Exception as e:
                sample = validated_records[0] if validated_records else None
                _log_supabase_error("insert_batch", table, e, sample)
                raise e
        else:
            return None
    
    def _upsert_single_validated(self, table: str, data: Dict[str, Any], on_conflict: str = None) -> Any:
        """Upsert single validated record."""
        _normalize_fraction_fields(data)
        try:
            sanitized_data = validate_and_sanitize_data(table, data)
            logger.debug(f"✅ Data validated for {table} upsert")
            
            if on_conflict:
                query = self.client.table(table).upsert(sanitized_data, on_conflict=on_conflict)
            else:
                query = self.client.table(table).upsert(sanitized_data)
            
            return query.execute()
        except ValidationError as e:
            _log_supabase_error("validation", table, e, data)
            raise e
        except Exception as e:
            _log_supabase_error("upsert", table, e, data)
            raise e
    
    def _upsert_batch_validated(self, table: str, data_list: List[Dict[str, Any]], on_conflict: str = None) -> Any:
        """Upsert batch of validated records."""
        validated_records = []
        failed_records = []
        
        for i, data in enumerate(data_list):
            _normalize_fraction_fields(data)
            try:
                sanitized_data = validate_and_sanitize_data(table, data)
                validated_records.append(sanitized_data)
            except ValidationError as e:
                logger.error(f"❌ Validation failed for {table} upsert item {i}: {e}")
                failed_records.append((i, str(e)))
        
        if failed_records:
            logger.error(f"❌ {len(failed_records)} records failed validation for {table}")
            if len(validated_records) == 0:
                raise ValidationError(f"All records failed validation for {table}")
        
        if validated_records:
            logger.info(f"✅ Upserting {len(validated_records)} validated records into {table}")

            try:
                if on_conflict:
                    query = self.client.table(table).upsert(validated_records, on_conflict=on_conflict)
                else:
                    query = self.client.table(table).upsert(validated_records)

                return query.execute()
            except Exception as e:
                sample = validated_records[0] if validated_records else None
                _log_supabase_error("upsert_batch", table, e, sample)
                raise e
        else:
            return None
    
    def _validate_and_sanitize(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize data for a table."""
        return validate_and_sanitize_data(table, data)
    
    def validate_data(self, table: str, data: Dict[str, Any]) -> ValidationResult:
        """Validate data without inserting it."""
        return validate_supabase_data(table, data, strict_mode=False)
    
    def validate_batch_data(self, table: str, data_list: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate batch of data without inserting it."""
        results = []
        for data in data_list:
            result = validate_supabase_data(table, data, strict_mode=False)
            results.append(result)
        return results
    
    def select(self, table: str, *args, **kwargs):
        """Delegate select operations to underlying client."""
        return self.client.table(table).select(*args, **kwargs)
    
    def delete(self, table: str, **kwargs):
        """Delegate delete operations to underlying client."""
        query = self.client.table(table).delete()
        for key, value in kwargs.items():
            if hasattr(query, key):
                query = getattr(query, key)(value)
        return query.execute()
    
    def rpc(self, function_name: str, params: Dict[str, Any] = None):
        """Delegate RPC operations to underlying client."""
        return self.client.rpc(function_name, params)
    
    def table(self, table_name: str):
        """Get a table reference for direct operations."""
        return self.client.table(table_name)
    
    def __getattr__(self, name):
        """Delegate unknown methods to the underlying Supabase client."""
        return getattr(self.client, name)


def create_validated_supabase_client(url: str, key: str, enable_validation: bool = True) -> ValidatedSupabaseClient:
    """Create a validated Supabase client."""
    return ValidatedSupabaseClient(url, key, enable_validation)


def get_validated_supabase_client(enable_validation: bool = True) -> Optional[ValidatedSupabaseClient]:
    """Get a validated Supabase client using environment variables."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    if not (url and key):
        logger.error("❌ Supabase credentials not found. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        return None
    
    try:
        return create_validated_supabase_client(url, key, enable_validation)
    except Exception as e:
        logger.error(f"❌ Failed to create validated Supabase client: {e}")
        return None
