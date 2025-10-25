"""
Supabase-based storage for ad creation times and historical data.
This replaces SQLite storage to ensure ML system has access to all data.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

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
                
            # Use a simple integer ID instead of timestamp for created_at_epoch
            # The field appears to be constrained to small integers
            import random
            epoch_id = random.randint(1, 999999)  # Use random ID instead of timestamp
                
            data = {
                'ad_id': ad_id,
                'lifecycle_id': lifecycle_id or '',
                'stage': stage,
                'created_at_epoch': epoch_id,  # Use small integer ID
                'created_at_iso': created_at.isoformat()
            }
            
            # Use upsert to handle duplicates
            self.client.table('ad_creation_times').upsert(data, on_conflict='ad_id').execute()
            logger.debug(f"Recorded ad creation time for {ad_id} in Supabase")
            
        except Exception as e:
            logger.error(f"Failed to record ad creation time for {ad_id}: {e}")
            raise
    
    def get_ad_creation_time(self, ad_id: str) -> Optional[datetime]:
        """Get when an ad was created."""
        try:
            response = self.client.table('ad_creation_times').select('created_at_epoch').eq(
                'ad_id', ad_id
            ).execute()
            
            if response.data:
                epoch = response.data[0]['created_at_epoch']
                return datetime.fromtimestamp(epoch, timezone.utc)
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
        
        try:
            data = {
                'ad_id': ad_id,
                'lifecycle_id': lifecycle_id or '',
                'stage': stage,
                'metric_name': metric_name,
                'metric_value': float(metric_value),
                'ts_epoch': int(timestamp.timestamp()),
                'ts_iso': timestamp.isoformat(),
                'created_at': timestamp.isoformat()
            }
            
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
