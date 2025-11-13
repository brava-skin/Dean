"""
Creative Storage Management for Supabase
Handles uploading, tracking, and cleanup of creative images in Supabase Storage
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib

from integrations.slack import notify

logger = logging.getLogger(__name__)

# Storage configuration
CREATIVE_STORAGE_BUCKET = os.getenv("CREATIVE_STORAGE_BUCKET", "creatives")
CREATIVE_UNUSED_DAYS = int(os.getenv("CREATIVE_UNUSED_DAYS", "30"))  # Days before unused creatives are deleted
CREATIVE_KILLED_DAYS = int(os.getenv("CREATIVE_KILLED_DAYS", "7"))  # Days before killed creatives are deleted


class CreativeStorageManager:
    """Manages creative images in Supabase Storage."""
    
    def __init__(self, supabase_client):
        """
        Initialize creative storage manager.
        
        Args:
            supabase_client: Supabase client instance
        """
        self.client = supabase_client
        self.bucket_name = CREATIVE_STORAGE_BUCKET
        
    def upload_creative(
        self,
        creative_id: str,
        image_path: str,
        ad_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Upload a creative image to Supabase Storage.
        
        Args:
            creative_id: Unique creative identifier
            image_path: Local path to the image file
            ad_id: Optional ad ID associated with this creative
            metadata: Optional metadata to store
            
        Returns:
            Public URL of uploaded image or None on failure
        """
        try:
            if not self.client:
                logger.error("Supabase client not available")
                return None
            
            # Read image file
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            file_size = image_path_obj.stat().st_size
            file_ext = image_path_obj.suffix.lower() or ".png"
            
            # Determine MIME type
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }
            file_type = mime_types.get(file_ext, "image/png")
            
            # Generate storage path: {creative_id}{ext} (stored in bucket root)
            storage_path = f"{creative_id}{file_ext}"
            
            # Upload to Supabase Storage
            with open(image_path, "rb") as f:
                file_data = f.read()
            
            # Upload file with upsert option
            try:
                # Try to upload with upsert
                response = self.client.storage.from_(self.bucket_name).upload(
                    storage_path,
                    file_data,
                    file_options={"content-type": file_type, "upsert": "true"},
                )
            except Exception as e:
                # If upsert fails, try regular upload (might fail if exists)
                logger.warning(f"Upsert upload failed, trying regular upload: {e}")
                response = self.client.storage.from_(self.bucket_name).upload(
                    storage_path,
                    file_data,
                    file_options={"content-type": file_type},
                )
            
            if not response:
                logger.error(f"Failed to upload creative {creative_id} to storage")
                return None
            
            # Get public URL
            try:
                public_url_response = self.client.storage.from_(self.bucket_name).get_public_url(storage_path)
                # Handle both string and dict responses
                if isinstance(public_url_response, dict):
                    public_url = public_url_response.get("publicUrl") or public_url_response.get("url")
                else:
                    public_url = public_url_response
            except Exception as e:
                # Fallback: construct URL manually
                supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
                if supabase_url:
                    public_url = f"{supabase_url}/storage/v1/object/public/{self.bucket_name}/{storage_path}"
                else:
                    logger.error("Cannot construct public URL: SUPABASE_URL not set")
                    return None
            
            if not public_url:
                logger.error(f"Failed to get public URL for creative {creative_id}")
                return None
            
            # Store metadata in ads table (consolidated from creative_storage)
            # Note: ad_id is required for ads table, so we only store if ad_id exists
            # If ad_id doesn't exist yet, the ad record will be created when the ad is created
            if ad_id:
                enhanced_metadata = metadata or {}
                
                # Ensure enhanced metadata fields are included
                if "format" not in enhanced_metadata:
                    enhanced_metadata["format"] = "static_image"
                if "style" not in enhanced_metadata:
                    enhanced_metadata["style"] = ""
                if "message_type" not in enhanced_metadata:
                    enhanced_metadata["message_type"] = ""
                if "target_motivation" not in enhanced_metadata:
                    enhanced_metadata["target_motivation"] = ""
                
                # Update ads table with storage info
                # If ad doesn't exist yet, it will be created when ad is created
                storage_data = {
                    "creative_id": creative_id,
                    "storage_path": storage_path,
                    "storage_url": public_url,
                    "file_size_bytes": file_size,
                    "file_type": file_type,
                    "status": "active",  # Creative is active when uploaded (ad will be created)
                    "metadata": enhanced_metadata,
                }
                
                try:
                    # Try to update existing ad record
                    self.client.table("ads").update(storage_data).eq("ad_id", ad_id).execute()
                    logger.debug(f"Updated ad {ad_id} with storage info for creative {creative_id}")
                except Exception as e:
                    # Ad might not exist yet - that's okay, it will be created when ad is created
                    logger.debug(f"Ad {ad_id} not found yet (will be created later): {e}")
            
            logger.info(f"✅ Uploaded creative {creative_id} to Supabase Storage")
            return public_url
                
        except Exception as e:
            logger.error(f"Error uploading creative {creative_id}: {e}", exc_info=True)
            notify(f"❌ Failed to upload creative {creative_id} to storage: {e}")
            return None
    
    def mark_creative_killed(self, creative_id: str) -> bool:
        """
        Mark a creative as killed - updates ads table.
        
        Args:
            creative_id: Creative ID to mark as killed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Update status in ads table for all ads using this creative
            now = datetime.now(timezone.utc).isoformat()
            self.client.table("ads").update({
                "status": "killed",
                "killed_at": now,
                "updated_at": now,
            }).eq("creative_id", creative_id).execute()
            
            logger.info(f"✅ Marked ads with creative {creative_id} as killed")
            return True
            
        except Exception as e:
            logger.error(f"Error marking creative {creative_id} as killed: {e}")
            return False
    
    def get_queued_creative(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recently created queued creative - not applicable with new schema.
        Creatives are now created when ads are created, so there's no queue.
        
        Returns:
            None (queue concept removed with consolidated schema)
        """
        # With the new consolidated schema, creatives are created when ads are created
        # There's no separate queue - ads are created directly
        return None
    
    def get_queued_creative_count(self) -> int:
        """
        Get count of queued creatives - not applicable with new schema.
        
        Returns:
            0 (queue concept removed)
        """
        return 0
    
    def should_pre_generate_creatives(self, target_count: int = 10, buffer_size: int = 3) -> bool:
        """
        Determine if we should pre-generate creatives for the queue.
        
        Args:
            target_count: Target number of active creatives
            buffer_size: Number of queued creatives to maintain as buffer
            
        Returns:
            True if we should pre-generate, False otherwise
        """
        queued_count = self.get_queued_creative_count()
        return queued_count < buffer_size
    
    def get_recently_created_creative(self, minutes_back: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get the most recently created ad/creative (within last N minutes).
        
        Args:
            minutes_back: How many minutes back to look (default 5)
            
        Returns:
            Ad data dict with creative info or None if none found
        """
        try:
            if not self.client:
                return None
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes_back)
            
            # Get most recently created ad
            result = self.client.table("ads").select(
                "ad_id, creative_id, storage_url, storage_path, metadata, status, created_at"
            ).gte("created_at", cutoff_time.isoformat()).order("created_at", desc=True).limit(1).execute()
            
            if result.data and len(result.data) > 0:
                ad = result.data[0]
                # Return in format expected by callers
                return {
                    "creative_id": ad.get("creative_id"),
                    "storage_url": ad.get("storage_url"),
                    "storage_path": ad.get("storage_path"),
                    "metadata": ad.get("metadata", {}),
                    "status": ad.get("status"),
                    "created_at": ad.get("created_at"),
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recently created creative: {e}")
            return None
    
    def mark_creative_active(self, creative_id: str, ad_id: Optional[str] = None) -> bool:
        """
        Mark a creative/ad as active (used in campaign).
        
        Args:
            creative_id: Creative ID to mark as active
            ad_id: Ad ID associated with this creative (required)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client or not ad_id:
                return False
            
            # Update ads table - mark ad as active
            now = datetime.now(timezone.utc).isoformat()
            self.client.table("ads").update({
                "status": "active",
                "updated_at": now,
            }).eq("ad_id", ad_id).eq("creative_id", creative_id).execute()
            
            logger.info(f"✅ Marked ad {ad_id} (creative {creative_id}) as active")
            return True
            
        except Exception as e:
            logger.error(f"Error marking creative {creative_id} as active: {e}")
            return False
    
    def update_usage(self, creative_id: str) -> bool:
        """
        Update usage for a creative - updates ads table.
        
        Args:
            creative_id: Creative ID to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Update all ads using this creative
            now = datetime.now(timezone.utc).isoformat()
            self.client.table("ads").update({
                "status": "active",  # Ensure active status
                "updated_at": now,
            }).eq("creative_id", creative_id).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating usage for creative {creative_id}: {e}")
            return False
    
    def cleanup_unused_creatives(self) -> int:
        """
        Delete unused creatives from storage after they've been unused for CREATIVE_UNUSED_DAYS.
        
        Returns:
            Number of creatives deleted
        """
        try:
            if not self.client:
                return 0
            
            # With new schema, ads are either active or killed
            # No unused state to clean up
            return 0
            
        except Exception as e:
            logger.error(f"Error cleaning up unused creatives: {e}", exc_info=True)
            return 0
    
    def cleanup_killed_creatives(self) -> int:
        """
        Delete killed creatives from storage after CREATIVE_KILLED_DAYS.
        
        Returns:
            Number of creatives deleted
        """
        try:
            if not self.client:
                return 0
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=CREATIVE_KILLED_DAYS)
            
            # Find killed ads older than cutoff
            result = self.client.table("ads").select(
                "ad_id, creative_id, storage_path"
            ).eq("status", "killed").lt(
                "killed_at", cutoff_date.isoformat()
            ).execute()
            
            deleted_count = 0
            
            for record in (result.data or []):
                creative_id = record.get("creative_id")
                storage_path = record.get("storage_path")
                
                try:
                    # Delete from storage (remove expects a list)
                    try:
                        if storage_path:
                            self.client.storage.from_(self.bucket_name).remove([storage_path])
                    except Exception as storage_error:
                        logger.debug(f"Storage file not found (may already be deleted): {storage_error}")
                    
                    # Delete ad record from database
                    self.client.table("ads").delete().eq(
                        "ad_id", record.get("ad_id")
                    ).execute()
                    
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted killed ad/creative {creative_id} from storage")
                    
                except Exception as e:
                    logger.error(f"Error deleting killed creative {creative_id}: {e}")
            
            if deleted_count > 0:
                notify(f"🧹 Cleaned up {deleted_count} killed creatives from storage")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up killed creatives: {e}", exc_info=True)
            return 0
    
    def mark_creative_unused(self, creative_id: str) -> bool:
        """
        Mark a creative as unused - not applicable with new schema.
        Ads are either active or killed.
        
        Args:
            creative_id: Creative ID (not used)
            
        Returns:
            False (unused state removed)
        """
        # With new schema, ads are either active or killed
        # No unused state
        return False
    
    def get_creative_url(self, creative_id: str) -> Optional[str]:
        """
        Get the public URL for a creative from ads table.
        
        Args:
            creative_id: Creative ID
            
        Returns:
            Public URL or None if not found
        """
        try:
            if not self.client:
                return None
            
            result = self.client.table("ads").select(
                "storage_url, storage_path"
            ).eq("creative_id", creative_id).limit(1).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0].get("storage_url")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting creative URL for {creative_id}: {e}")
            return None


def create_creative_storage_manager(supabase_client) -> Optional[CreativeStorageManager]:
    """Create a creative storage manager instance."""
    if not supabase_client:
        return None
    return CreativeStorageManager(supabase_client)

