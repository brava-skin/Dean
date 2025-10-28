"""
Supabase-based storage for ad creation times and historical data.
This replaces SQLite storage to ensure ML system has access to all data.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import logging
import random

logger = logging.getLogger(__name__)


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
                
            # Use actual timestamp for created_at_epoch
            epoch_id = int(created_at.timestamp())
                
            data = {
                'ad_id': ad_id,
                'lifecycle_id': lifecycle_id or '',
                'stage': stage,
                'created_at_epoch': epoch_id,  # Use actual timestamp
                'created_at_iso': created_at.isoformat()
            }
            
            # Use upsert to handle duplicates with validation
            try:
                from infrastructure.validated_supabase import get_validated_supabase_client
                validated_client = get_validated_supabase_client(enable_validation=True)
                if validated_client:
                    validated_client.upsert('ad_creation_times', data, on_conflict='ad_id')
                else:
                    self.client.table('ad_creation_times').upsert(data, on_conflict='ad_id').execute()
            except ImportError:
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
            
            # Insert with validation
            try:
                from infrastructure.validated_supabase import get_validated_supabase_client
                validated_client = get_validated_supabase_client(enable_validation=True)
                if validated_client:
                    validated_client.insert('historical_data', data)
                else:
                    self.client.table('historical_data').insert(data).execute()
            except ImportError:
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
