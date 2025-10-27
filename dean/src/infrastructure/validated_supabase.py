"""
Supabase Operations with Data Validation
========================================
This module provides validated Supabase operations that automatically validate
all data before insertion to prevent invalid data from being stored.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from supabase import create_client, Client

from .data_validation import (
    validate_supabase_data, 
    validate_and_sanitize_data, 
    ValidationError,
    ValidationResult
)

logger = logging.getLogger(__name__)

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
        """
        Insert data into Supabase table with validation.
        
        Args:
            table: Table name
            data: Data to insert (single record or list of records)
            validate: Override validation setting for this operation
        
        Returns:
            Supabase response
        """
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
        """
        Upsert data into Supabase table with validation.
        
        Args:
            table: Table name
            data: Data to upsert (single record or list of records)
            on_conflict: Conflict resolution strategy
            validate: Override validation setting for this operation
        
        Returns:
            Supabase response
        """
        should_validate = validate if validate is not None else self.enable_validation
        
        if should_validate:
            if isinstance(data, list):
                return self._upsert_batch_validated(table, data, on_conflict)
            else:
                return self._upsert_single_validated(table, data, on_conflict)
        else:
            query = self.client.table(table).upsert(data)
            if on_conflict:
                # Note: Supabase doesn't have on_conflict method, so we ignore it
                logger.warning(f"on_conflict parameter '{on_conflict}' ignored - Supabase upsert handles conflicts automatically")
            return query.execute()
    
    def update(self, table: str, data: Dict[str, Any], 
               validate: bool = None, **kwargs) -> Any:
        """
        Update data in Supabase table with validation.
        
        Args:
            table: Table name
            data: Data to update
            validate: Override validation setting for this operation
            **kwargs: Additional query parameters (eq, neq, etc.)
        
        Returns:
            Supabase response
        """
        should_validate = validate if validate is not None else self.enable_validation
        
        if should_validate:
            sanitized_data = self._validate_and_sanitize(table, data)
            query = self.client.table(table).update(sanitized_data)
            
            # Apply query filters
            for key, value in kwargs.items():
                if key == 'eq' and 'value' in kwargs:
                    # Handle eq with value parameter: eq='model_type', value=model_type
                    query = query.eq(value, kwargs['value'])
                elif key == 'eq2' and 'value2' in kwargs:
                    # Handle eq2 with value2 parameter
                    query = query.eq(value, kwargs['value2'])
                elif key == 'eq3' and 'value3' in kwargs:
                    # Handle eq3 with value3 parameter
                    query = query.eq(value, kwargs['value3'])
                elif key.startswith('value'):
                    # Skip value parameters, they're handled above
                    continue
                elif hasattr(query, key):
                    query = getattr(query, key)(value)
            
            return query.execute()
        else:
            query = self.client.table(table).update(data)
            
            # Apply query filters
            for key, value in kwargs.items():
                if key == 'eq' and 'value' in kwargs:
                    # Handle eq with value parameter: eq='model_type', value=model_type
                    query = query.eq(value, kwargs['value'])
                elif key == 'eq2' and 'value2' in kwargs:
                    # Handle eq2 with value2 parameter
                    query = query.eq(value, kwargs['value2'])
                elif key == 'eq3' and 'value3' in kwargs:
                    # Handle eq3 with value3 parameter
                    query = query.eq(value, kwargs['value3'])
                elif key.startswith('value'):
                    # Skip value parameters, they're handled above
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
            # Optionally raise error or continue with valid records
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
            
            query = self.client.table(table).upsert(sanitized_data)
            if on_conflict:
                # Note: Supabase doesn't have on_conflict method, so we ignore it
                logger.warning(f"on_conflict parameter '{on_conflict}' ignored - Supabase upsert handles conflicts automatically")
            
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
            # Optionally raise error or continue with valid records
            if len(validated_records) == 0:
                raise ValidationError(f"All records failed validation for {table}")
        
        if validated_records:
            logger.info(f"✅ Upserting {len(validated_records)} validated records into {table}")
            
            query = self.client.table(table).upsert(validated_records)
            if on_conflict:
                # Note: Supabase doesn't have on_conflict method, so we ignore it
                logger.warning(f"on_conflict parameter '{on_conflict}' ignored - Supabase upsert handles conflicts automatically")
            
            return query.execute()
        else:
            return None
    
    def _validate_and_sanitize(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize data for a table."""
        return validate_and_sanitize_data(table, data)
    
    def validate_data(self, table: str, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data without inserting it.
        
        Args:
            table: Table name
            data: Data to validate
        
        Returns:
            ValidationResult with validation status
        """
        return validate_supabase_data(table, data, strict_mode=False)
    
    def validate_batch_data(self, table: str, data_list: List[Dict[str, Any]]) -> List[ValidationResult]:
        """
        Validate batch of data without inserting it.
        
        Args:
            table: Table name
            data_list: List of data to validate
        
        Returns:
            List of ValidationResult objects
        """
        results = []
        for data in data_list:
            result = validate_supabase_data(table, data, strict_mode=False)
            results.append(result)
        return results
    
    # Delegate other operations to the underlying client
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
        """
        Get a table reference for direct operations.
        Returns a ValidatedTableWrapper that provides validated operations.
        """
        return ValidatedTableWrapper(self, table_name)
    
    # Delegate other operations to the underlying client
    def __getattr__(self, name):
        """Delegate unknown methods to the underlying Supabase client."""
        return getattr(self.client, name)

class ValidatedTableWrapper:
    """
    Wrapper for table operations that provides validated methods.
    """
    
    def __init__(self, validated_client: ValidatedSupabaseClient, table_name: str):
        self.validated_client = validated_client
        self.table_name = table_name
        self.client = validated_client.client
    
    def insert(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
               validate: bool = None) -> Any:
        """Insert data with validation."""
        return self.validated_client.insert(self.table_name, data, validate)
    
    def upsert(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
               on_conflict: str = None, validate: bool = None) -> Any:
        """Upsert data with validation."""
        return self.validated_client.upsert(self.table_name, data, on_conflict, validate)
    
    def update(self, data: Dict[str, Any], validate: bool = None, **kwargs) -> Any:
        """Update data with validation."""
        return self.validated_client.update(self.table_name, data, validate, **kwargs)
    
    def select(self, columns: str = "*", **kwargs):
        """Select data - delegate to underlying client."""
        return self.client.table(self.table_name).select(columns, **kwargs)
    
    def delete(self, **kwargs):
        """Delete data - delegate to underlying client."""
        return self.client.table(self.table_name).delete(**kwargs)
    
    # Delegate other operations to the underlying table
    def __getattr__(self, name):
        """Delegate unknown methods to the underlying Supabase table."""
        return getattr(self.client.table(self.table_name), name)

def create_validated_supabase_client(url: str, key: str, enable_validation: bool = True) -> ValidatedSupabaseClient:
    """
    Create a validated Supabase client.
    
    Args:
        url: Supabase URL
        key: Supabase API key
        enable_validation: Whether to enable data validation
    
    Returns:
        ValidatedSupabaseClient instance
    """
    return ValidatedSupabaseClient(url, key, enable_validation)

# Convenience function for easy integration
def get_validated_supabase_client(enable_validation: bool = True) -> Optional[ValidatedSupabaseClient]:
    """
    Get a validated Supabase client using environment variables.
    
    Args:
        enable_validation: Whether to enable data validation
    
    Returns:
        ValidatedSupabaseClient instance or None if credentials not found
    """
    import os
    
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
