"""
Centralized date validation and correction utility for Supabase data storage.
Ensures all timestamps are current and consistent across the system.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DateValidator:
    """Centralized date validation and correction utility."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DateValidator")
        self.min_valid_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.max_future_offset = timedelta(hours=1)  # Allow 1 hour in future for timezone differences
        
    def get_current_timestamp(self) -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)
    
    def validate_and_correct_date(self, 
                                date_value: Any, 
                                field_name: str = "timestamp",
                                allow_future: bool = False,
                                max_future_hours: int = 1) -> datetime:
        """
        Validate and correct a date value to ensure it's current and valid.
        
        Args:
            date_value: The date value to validate (can be datetime, string, int, or None)
            field_name: Name of the field for logging purposes
            allow_future: Whether to allow future dates
            max_future_hours: Maximum hours in the future allowed
            
        Returns:
            Corrected datetime object
        """
        now = self.get_current_timestamp()
        
        # Handle None or empty values
        if date_value is None or date_value == "":
            self.logger.debug(f"Empty {field_name}, using current timestamp")
            return now
        
        # Handle different input types
        if isinstance(date_value, str):
            try:
                # Check for malformed dates with asterisks or other invalid characters
                if '*' in date_value or '***' in date_value or len(date_value) < 10:
                    self.logger.warning(f"Malformed date {field_name} '{date_value}', using current timestamp")
                    return now
                    
                # Try parsing ISO format
                if 'T' in date_value:
                    parsed_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                else:
                    # Try parsing other common formats
                    parsed_date = datetime.strptime(date_value, '%Y-%m-%d %H:%M:%S%z')
            except (ValueError, TypeError):
                self.logger.warning(f"Failed to parse {field_name} '{date_value}', using current timestamp")
                return now
                
        elif isinstance(date_value, (int, float)):
            try:
                # Handle epoch timestamps
                if date_value > 1e10:  # Milliseconds
                    parsed_date = datetime.fromtimestamp(date_value / 1000, tz=timezone.utc)
                else:  # Seconds
                    parsed_date = datetime.fromtimestamp(date_value, tz=timezone.utc)
            except (ValueError, OSError):
                self.logger.warning(f"Invalid epoch timestamp {field_name} '{date_value}', using current timestamp")
                return now
                
        elif isinstance(date_value, datetime):
            parsed_date = date_value
            # Ensure timezone awareness
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        else:
            self.logger.warning(f"Unsupported date type {field_name} '{type(date_value)}', using current timestamp")
            return now
        
        # Ensure both dates are timezone-aware for comparison
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if self.min_valid_date.tzinfo is None:
            self.min_valid_date = self.min_valid_date.replace(tzinfo=timezone.utc)
            
        # Validate date range
        if parsed_date < self.min_valid_date:
            self.logger.warning(f"{field_name} too old ({parsed_date}), using current timestamp")
            return now
            
        if not allow_future and parsed_date > now + self.max_future_offset:
            self.logger.warning(f"{field_name} in future ({parsed_date}), using current timestamp")
            return now
            
        # Check future date limit if allowed
        if allow_future and parsed_date > now + timedelta(hours=max_future_hours):
            self.logger.warning(f"{field_name} too far in future ({parsed_date}), using current timestamp")
            return now
        
        self.logger.debug(f"{field_name} validated: {parsed_date}")
        return parsed_date
    
    def validate_timestamp_pair(self, 
                              created_at: Any, 
                              updated_at: Any = None,
                              field_prefix: str = "") -> tuple[datetime, datetime]:
        """
        Validate a pair of timestamps ensuring created_at <= updated_at.
        
        Args:
            created_at: Creation timestamp
            updated_at: Update timestamp (optional)
            field_prefix: Prefix for field names in logging
            
        Returns:
            Tuple of (validated_created_at, validated_updated_at)
        """
        now = self.get_current_timestamp()
        
        # Validate created_at
        validated_created_at = self.validate_and_correct_date(
            created_at, 
            f"{field_prefix}created_at"
        )
        
        # Validate updated_at
        if updated_at is not None:
            validated_updated_at = self.validate_and_correct_date(
                updated_at, 
                f"{field_prefix}updated_at"
            )
        else:
            validated_updated_at = now
        
        # Ensure created_at <= updated_at
        if validated_created_at > validated_updated_at:
            self.logger.warning(f"created_at ({validated_created_at}) > updated_at ({validated_updated_at}), correcting")
            validated_updated_at = validated_created_at + timedelta(seconds=1)
        
        return validated_created_at, validated_updated_at
    
    def add_timestamps_to_data(self, 
                             data: Dict[str, Any], 
                             created_at: Any = None,
                             updated_at: Any = None,
                             expires_at: Any = None) -> Dict[str, Any]:
        """
        Add validated timestamps to a data dictionary.
        
        Args:
            data: Data dictionary to add timestamps to
            created_at: Creation timestamp (optional)
            updated_at: Update timestamp (optional)
            expires_at: Expiration timestamp (optional)
            
        Returns:
            Data dictionary with validated timestamps
        """
        now = self.get_current_timestamp()
        
        # Add created_at
        if 'created_at' not in data or data['created_at'] is None:
            data['created_at'] = self.validate_and_correct_date(
                created_at, 
                "created_at"
            ).isoformat()
        
        # Add updated_at
        if 'updated_at' not in data or data['updated_at'] is None:
            data['updated_at'] = self.validate_and_correct_date(
                updated_at, 
                "updated_at"
            ).isoformat()
        
        # Add expires_at if provided
        if expires_at is not None:
            data['expires_at'] = self.validate_and_correct_date(
                expires_at, 
                "expires_at",
                allow_future=True,
                max_future_hours=24*365  # Allow up to 1 year in future
            ).isoformat()
        
        # Add timestamp for compatibility
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = now.isoformat()
        
        return data
    
    def validate_data_timestamps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all timestamp fields in a data dictionary.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            Data dictionary with validated timestamps
        """
        timestamp_fields = [
            'created_at', 'updated_at', 'timestamp', 'expires_at',
            'date_start', 'date_end', 'trained_at', 'recorded_at'
        ]
        
        for field in timestamp_fields:
            if field in data and data[field] is not None:
                try:
                    validated_date = self.validate_and_correct_date(
                        data[field], 
                        field
                    )
                    data[field] = validated_date.isoformat()
                except Exception as e:
                    self.logger.error(f"Error validating {field}: {e}")
                    # Use current timestamp as fallback
                    data[field] = self.get_current_timestamp().isoformat()
        
        return data
    
    def log_date_corrections(self, 
                           original_data: Dict[str, Any], 
                           corrected_data: Dict[str, Any],
                           table_name: str) -> None:
        """
        Log any date corrections made to data.
        
        Args:
            original_data: Original data before correction
            corrected_data: Data after correction
            table_name: Name of the table being updated
        """
        timestamp_fields = ['created_at', 'updated_at', 'timestamp', 'expires_at']
        
        corrections = []
        for field in timestamp_fields:
            if field in original_data and field in corrected_data:
                if original_data[field] != corrected_data[field]:
                    corrections.append(f"{field}: {original_data[field]} -> {corrected_data[field]}")
        
        if corrections:
            self.logger.info(f"Date corrections for {table_name}: {', '.join(corrections)}")

# Global instance for easy access
date_validator = DateValidator()

def get_validated_timestamp(date_value: Any = None, field_name: str = "timestamp") -> datetime:
    """Convenience function to get a validated timestamp."""
    return date_validator.validate_and_correct_date(date_value, field_name)

def add_current_timestamps(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to add current timestamps to data."""
    return date_validator.add_timestamps_to_data(data)

def validate_all_timestamps(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to validate all timestamps in data."""
    return date_validator.validate_data_timestamps(data)
