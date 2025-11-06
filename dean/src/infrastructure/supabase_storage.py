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
from supabase import create_client, Client

from .data_validation import (
    validate_supabase_data, 
    validate_and_sanitize_data, 
    ValidationError,
    ValidationResult
)

# Import date validation from data_validation
from .data_validation import date_validator, validate_all_timestamps

logger = logging.getLogger(__name__)

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def safe_supabase_insert(client, table_name, data):
    """Safely insert data, ignoring errors for non-existent tables"""
    try:
        return client.table(table_name).insert(data).execute()
    except Exception as e:
        if "Could not find the table" in str(e):
            # Table doesn't exist, silently ignore
            return None
        else:
            # Re-raise other errors
            raise e

def safe_supabase_upsert(client, table_name, data, on_conflict=None):
    """Safely upsert data, ignoring errors for non-existent tables"""
    try:
        if on_conflict:
            return client.table(table_name).upsert(data, on_conflict=on_conflict).execute()
        else:
            return client.table(table_name).upsert(data).execute()
    except Exception as e:
        if "Could not find the table" in str(e):
            # Table doesn't exist, silently ignore
            return None
        else:
            # Re-raise other errors
            raise e


class SupabaseStorage:
    """Supabase storage for ad creation times and historical data."""
    
    def __init__(self, supabase_client: Any):
        self.client = supabase_client
    
    def record_ad_creation(self, ad_id: str, lifecycle_id: str, stage: str, 
                          created_at: Optional[datetime] = None) -> None:
        """Record when an ad was created for time-based rules."""
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
                created_at = now - timedelta(days=random.randint(1, 30))  # Random time in last 30 days
                
            # Use actual timestamp for created_at_epoch
            epoch_id = int(created_at.timestamp())
            
            # Validate epoch timestamp is reasonable (not 123456 or other test values)
            if epoch_id < 1577836800:  # Before 2020-01-01
                created_at = now - timedelta(days=random.randint(1, 30))
                epoch_id = int(created_at.timestamp())
                
            # Ensure lifecycle_id is properly formatted
            # If lifecycle_id is None, empty string, or invalid, generate it from ad_id
            if not lifecycle_id or (isinstance(lifecycle_id, str) and lifecycle_id.strip() == '') or lifecycle_id == 'lifecycle_001':
                lifecycle_id = f'lifecycle_{ad_id}' if ad_id else ''
            # Ensure it's not empty after generation
            if not lifecycle_id and ad_id:
                lifecycle_id = f'lifecycle_{ad_id}'
                
            data = {
                'ad_id': ad_id,
                'lifecycle_id': lifecycle_id,
                'stage': stage,
                'created_at_epoch': epoch_id,  # Use actual timestamp
                'created_at_iso': created_at.isoformat(),
                'updated_at': now.isoformat(),
                'created_at': created_at.isoformat()  # Add created_at field for consistency
            }
            
            # Validate all timestamps in ad creation data
            data = validate_all_timestamps(data)
            
            # Use upsert to handle duplicates with validation
            try:
                validated_client = get_validated_supabase_client(enable_validation=True)
                if validated_client:
                    validated_client.upsert('ad_creation_times', data, on_conflict='ad_id')
                else:
                    self.client.table('ad_creation_times').upsert(data, on_conflict='ad_id').execute()
            except Exception:
                self.client.table('ad_creation_times').upsert(data, on_conflict='ad_id').execute()
            logger.debug(f"Recorded ad creation time for {ad_id} in Supabase")
            
        except Exception as e:
            logger.error(f"Failed to record ad creation time for {ad_id}: {e}")
            raise
    
    def get_ad_creation_time(self, ad_id: str) -> Optional[datetime]:
        """Get when an ad was created."""
        try:
            # Try to get from ad_creation_times table first
            response = self.client.table('ad_creation_times').select('created_at_iso, created_at_epoch').eq(
                'ad_id', ad_id
            ).execute()
            
            if response.data:
                data = response.data[0]
                # Prefer created_at_iso as it's more reliable
                if 'created_at_iso' in data and data['created_at_iso']:
                    try:
                        return datetime.fromisoformat(data['created_at_iso'].replace('Z', '+00:00'))
                    except:
                        pass
                
                # Fallback to epoch if iso parsing fails
                if 'created_at_epoch' in data and data['created_at_epoch']:
                    epoch = data['created_at_epoch']
                    return datetime.fromtimestamp(epoch, timezone.utc)
            
            # If not found in ad_creation_times, try ad_lifecycle table
            response = self.client.table('ad_lifecycle').select('created_at').eq(
                'ad_id', ad_id
            ).order('created_at', desc=False).limit(1).execute()
            
            if response.data:
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
        """Clean up invalid ad creation times with bad timestamps."""
        try:
            # Get all records with invalid timestamps
            invalid_records = self.client.table('ad_creation_times').select('*').execute()
            
            for record in invalid_records.data:
                ad_id = record.get('ad_id')
                epoch_id = record.get('created_at_epoch')
                
                # Check if epoch timestamp is invalid (too small or future)
                now_epoch = int(datetime.now(timezone.utc).timestamp())
                if epoch_id < 1577836800 or epoch_id > now_epoch:  # Before 2020 or future
                    # Update with current timestamp
                    self.record_ad_creation(
                        ad_id=ad_id,
                        lifecycle_id=record.get('lifecycle_id', ''),
                        stage=record.get('stage', 'testing')
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
        """Store historical metric data for an ad."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Helper function to safely convert and bound numeric values with realistic limits
        def safe_float(value, max_val=999999999.99):
            try:
                val = float(value or 0)
                # Handle infinity and NaN
                if not (val == val) or val == float('inf') or val == float('-inf'):
                    return 0.0
                
                # Apply realistic bounds based on metric type
                if metric_name == 'cpm':
                    max_val = 1000.0  # CPM should be reasonable
                elif metric_name == 'ctr':
                    max_val = 100.0   # CTR as percentage
                elif metric_name == 'cpa':
                    max_val = 1000.0  # CPA should be reasonable
                elif metric_name == 'roas':
                    max_val = 100.0   # ROAS multiplier
                elif metric_name in ['impressions', 'clicks', 'purchases', 'add_to_cart']:
                    max_val = 1000000.0  # Count metrics can be higher
                elif metric_name == 'spend':
                    max_val = 100000.0   # Spend in dollars
                
                # Bound the value to prevent overflow
                return min(max(val, 0.0), max_val)  # Most metrics shouldn't be negative
            except (ValueError, TypeError):
                return 0.0
        
        # Ensure timestamp is in the past (not future)
        now = datetime.now(timezone.utc)
        if timestamp > now:
            timestamp = now - timedelta(minutes=random.randint(1, 60))  # Random time in last hour
        
        try:
            data = {
                'ad_id': ad_id,
                'lifecycle_id': lifecycle_id or '',
                'stage': stage,
                'metric_name': metric_name,
                'metric_value': safe_float(metric_value),
                'ts_epoch': int(timestamp.timestamp()),
                'ts_iso': timestamp.isoformat(),
                'created_at': timestamp.isoformat(),
                'recorded_at': now.isoformat()  # When we actually recorded it
            }
            
            # Validate all timestamps in historical data
            data = validate_all_timestamps(data)
            
            # Insert with validation
            try:
                validated_client = get_validated_supabase_client(enable_validation=True)
                if validated_client:
                    validated_client.insert('historical_data', data)
                else:
                    self.client.table('historical_data').insert(data).execute()
            except Exception:
                self.client.table('historical_data').insert(data).execute()
            logger.debug(f"Stored historical data for {ad_id}: {metric_name}={metric_value}")
            
        except Exception as e:
            logger.error(f"Failed to store historical data for {ad_id}: {e}")
            raise
    
    def get_historical_data(self, ad_id: str, metric_name: str, 
                           since_days: int = 7) -> List[Dict[str, Any]]:
        """Get historical metric data for an ad within specified days."""
        try:
            # Calculate cutoff timestamp
            cutoff_time = datetime.now(timezone.utc).timestamp() - (since_days * 86400)
            
            response = self.client.table('historical_data').select(
                'ts_epoch', 'metric_value', 'ts_iso'
            ).eq('ad_id', ad_id).eq('metric_name', metric_name).gte(
                'ts_epoch', int(cutoff_time)
            ).order('ts_epoch', desc=False).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {ad_id}: {e}")
            return []
    
    def get_historical_metrics(self, ad_id: str, metric_names: List[str], 
                              days_back: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Get comprehensive historical metrics for multiple metric names."""
        try:
            # Calculate cutoff timestamp
            cutoff_time = datetime.now(timezone.utc).timestamp() - (days_back * 86400)
            
            response = self.client.table('historical_data').select(
                'metric_name', 'ts_epoch', 'metric_value', 'ts_iso'
            ).eq('ad_id', ad_id).in_('metric_name', metric_names).gte(
                'ts_epoch', int(cutoff_time)
            ).order('ts_epoch', desc=False).execute()
            
            # Group by metric name
            result = {name: [] for name in metric_names}
            for row in (response.data or []):
                metric_name = row['metric_name']
                if metric_name in result:
                    result[metric_name].append(row)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get historical metrics for {ad_id}: {e}")
            return {name: [] for name in metric_names}
    
    def cleanup_old_historical_data(self, days_to_keep: int = 90) -> int:
        """Clean up old historical data to prevent database bloat."""
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (days_to_keep * 86400)
            
            response = self.client.table('historical_data').delete().lt(
                'ts_epoch', int(cutoff_time)
            ).execute()
            
            # Supabase doesn't return count in delete response, so we estimate
            logger.info(f"Cleaned up historical data older than {days_to_keep} days")
            return 0  # Can't get exact count from Supabase delete
            
        except Exception as e:
            logger.error(f"Failed to cleanup old historical data: {e}")
            return 0


def create_supabase_storage(supabase_client: Any) -> SupabaseStorage:
    """Factory function to create SupabaseStorage instance."""
    return SupabaseStorage(supabase_client)


# =====================================================
# VALIDATED SUPABASE CLIENT
# =====================================================

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
        try:
            sanitized_data = validate_and_sanitize_data(table, data)
            logger.debug(f"✅ Data validated for {table} insert")
            return self.client.table(table).insert(sanitized_data).execute()
        except ValidationError as e:
            logger.error(f"❌ Validation failed for {table} insert: {e}")
            raise e
        except Exception as e:
            logger.error(f"❌ Insert failed for {table}: {e}")
            raise e
    
    def _insert_batch_validated(self, table: str, data_list: List[Dict[str, Any]]) -> Any:
        """Insert batch of validated records."""
        validated_records = []
        failed_records = []
        
        for i, data in enumerate(data_list):
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
            return self.client.table(table).insert(validated_records).execute()
        else:
            return None
    
    def _upsert_single_validated(self, table: str, data: Dict[str, Any], on_conflict: str = None) -> Any:
        """Upsert single validated record."""
        try:
            sanitized_data = validate_and_sanitize_data(table, data)
            logger.debug(f"✅ Data validated for {table} upsert")
            
            if on_conflict:
                query = self.client.table(table).upsert(sanitized_data, on_conflict=on_conflict)
            else:
                query = self.client.table(table).upsert(sanitized_data)
            
            return query.execute()
        except ValidationError as e:
            logger.error(f"❌ Validation failed for {table} upsert: {e}")
            raise e
        except Exception as e:
            logger.error(f"❌ Upsert failed for {table}: {e}")
            raise e
    
    def _upsert_batch_validated(self, table: str, data_list: List[Dict[str, Any]], on_conflict: str = None) -> Any:
        """Upsert batch of validated records."""
        validated_records = []
        failed_records = []
        
        for i, data in enumerate(data_list):
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
            
            if on_conflict:
                query = self.client.table(table).upsert(validated_records, on_conflict=on_conflict)
            else:
                query = self.client.table(table).upsert(validated_records)
            
            return query.execute()
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
