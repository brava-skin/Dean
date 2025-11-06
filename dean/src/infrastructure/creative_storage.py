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
            
            # Store metadata in creative_storage table
            # Status: "queue" when first uploaded, "active" when used in campaign, "killed" when ad is killed
            # Enhanced metadata includes format, style, message_type, target_motivation, forecasts
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
            
            storage_data = {
                "creative_id": creative_id,
                "ad_id": ad_id,
                "storage_path": storage_path,
                "storage_url": public_url,
                "file_size_bytes": file_size,
                "file_type": file_type,
                "status": "queue",  # Start as "queue" - will be "active" when used in campaign
                "last_used_at": datetime.now(timezone.utc).isoformat(),
                "usage_count": 0,  # Will be incremented when used
                "metadata": enhanced_metadata,  # Enhanced metadata with format, style, message_type, etc.
            }
            
            try:
                self.client.table("creative_storage").upsert(
                    storage_data,
                    on_conflict="creative_id"
                ).execute()
                logger.info(f"✅ Uploaded creative {creative_id} to Supabase Storage")
                return public_url
            except Exception as e:
                logger.error(f"Failed to record creative storage metadata: {e}")
                # Still return URL even if metadata insert fails
                return public_url
                
        except Exception as e:
            logger.error(f"Error uploading creative {creative_id}: {e}", exc_info=True)
            notify(f"❌ Failed to upload creative {creative_id} to storage: {e}")
            return None
    
    def mark_creative_killed(self, creative_id: str) -> bool:
        """
        Mark a creative as killed in storage.
        
        Args:
            creative_id: Creative ID to mark as killed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Update status in creative_storage table
            self.client.table("creative_storage").update({
                "status": "killed",
                "killed_at": datetime.now(timezone.utc).isoformat(),
            }).eq("creative_id", creative_id).execute()
            
            logger.info(f"✅ Marked creative {creative_id} as killed")
            return True
            
        except Exception as e:
            logger.error(f"Error marking creative {creative_id} as killed: {e}")
            return False
    
    def get_queued_creative(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recently created queued creative (FIFO - oldest first).
        
        Returns:
            Creative data dict with storage_url, creative_id, metadata, etc. or None if none available
        """
        try:
            if not self.client:
                return None
            
            # Get the oldest queued creative (FIFO)
            result = self.client.table("creative_storage").select(
                "creative_id, storage_url, storage_path, metadata, created_at"
            ).eq("status", "queue").order("created_at", desc=False).limit(1).execute()
            
            if result.data and len(result.data) > 0:
                creative = result.data[0]
                logger.info(f"✅ Found queued creative: {creative.get('creative_id')}")
                return creative
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting queued creative: {e}")
            return None
    
    def get_queued_creative_count(self) -> int:
        """
        Get count of queued creatives available for use.
        
        Returns:
            Number of queued creatives
        """
        try:
            if not self.client:
                return 0
            
            result = self.client.table("creative_storage").select(
                "creative_id", count="exact"
            ).eq("status", "queue").execute()
            
            return result.count if hasattr(result, 'count') else len(result.data or [])
            
        except Exception as e:
            logger.error(f"Error getting queued creative count: {e}")
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
        Get the most recently created creative (within last N minutes).
        Used to find a creative that was just generated.
        
        Args:
            minutes_back: How many minutes back to look (default 5)
            
        Returns:
            Creative data dict or None if none found
        """
        try:
            if not self.client:
                return None
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes_back)
            
            # Get most recently created creative (any status, but prefer queue)
            result = self.client.table("creative_storage").select(
                "creative_id, storage_url, storage_path, metadata, status, created_at"
            ).gte("created_at", cutoff_time.isoformat()).order("created_at", desc=True).limit(1).execute()
            
            if result.data and len(result.data) > 0:
                creative = result.data[0]
                logger.info(f"✅ Found recently created creative: {creative.get('creative_id')} (status: {creative.get('status')})")
                return creative
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recently created creative: {e}")
            return None
    
    def mark_creative_active(self, creative_id: str, ad_id: Optional[str] = None) -> bool:
        """
        Mark a creative as active (used in campaign).
        
        Args:
            creative_id: Creative ID to mark as active
            ad_id: Optional ad ID associated with this creative
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                return False
            
            update_data = {
                "status": "active",
                "last_used_at": datetime.now(timezone.utc).isoformat(),
            }
            
            if ad_id:
                update_data["ad_id"] = ad_id
            
            # Increment usage count
            result = self.client.table("creative_storage").select("usage_count").eq(
                "creative_id", creative_id
            ).execute()
            
            if result.data:
                current_count = result.data[0].get("usage_count", 0) or 0
                update_data["usage_count"] = current_count + 1
            
            self.client.table("creative_storage").update(update_data).eq(
                "creative_id", creative_id
            ).execute()
            
            logger.info(f"✅ Marked creative {creative_id} as active")
            return True
            
        except Exception as e:
            logger.error(f"Error marking creative {creative_id} as active: {e}")
            return False
    
    def update_usage(self, creative_id: str) -> bool:
        """
        Update last_used_at timestamp for a creative.
        
        Args:
            creative_id: Creative ID to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Get current usage count
            result = self.client.table("creative_storage").select("usage_count").eq(
                "creative_id", creative_id
            ).execute()
            
            current_count = 0
            if result.data and len(result.data) > 0:
                current_count = result.data[0].get("usage_count", 0)
            
            # Update usage
            self.client.table("creative_storage").update({
                "last_used_at": datetime.now(timezone.utc).isoformat(),
                "usage_count": current_count + 1,
                "status": "active",  # Reactivate if it was marked unused
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
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=CREATIVE_UNUSED_DAYS)
            
            # Find unused creatives
            result = self.client.table("creative_storage").select(
                "creative_id, storage_path, status"
            ).eq("status", "unused").lt(
                "last_used_at", cutoff_date.isoformat()
            ).execute()
            
            deleted_count = 0
            
            for record in (result.data or []):
                creative_id = record.get("creative_id")
                storage_path = record.get("storage_path")
                
                try:
                    # Delete from storage (remove expects a list)
                    try:
                        self.client.storage.from_(self.bucket_name).remove([storage_path])
                    except Exception as storage_error:
                        # If file doesn't exist in storage, continue with DB cleanup
                        logger.debug(f"Storage file not found (may already be deleted): {storage_error}")
                    
                    # Delete from database
                    self.client.table("creative_storage").delete().eq(
                        "creative_id", creative_id
                    ).execute()
                    
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted unused creative {creative_id} from storage")
                    
                except Exception as e:
                    logger.error(f"Error deleting creative {creative_id}: {e}")
            
            if deleted_count > 0:
                notify(f"🧹 Cleaned up {deleted_count} unused creatives from storage")
            
            return deleted_count
            
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
            
            # Find killed creatives older than cutoff
            result = self.client.table("creative_storage").select(
                "creative_id, storage_path"
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
                        self.client.storage.from_(self.bucket_name).remove([storage_path])
                    except Exception as storage_error:
                        # If file doesn't exist in storage, continue with DB cleanup
                        logger.debug(f"Storage file not found (may already be deleted): {storage_error}")
                    
                    # Delete from database
                    self.client.table("creative_storage").delete().eq(
                        "creative_id", creative_id
                    ).execute()
                    
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted killed creative {creative_id} from storage")
                    
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
        Mark a creative as unused (when ad is no longer active but not killed).
        
        Args:
            creative_id: Creative ID to mark as unused
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Update status to unused
            self.client.table("creative_storage").update({
                "status": "unused",
            }).eq("creative_id", creative_id).execute()
            
            logger.debug(f"Marked creative {creative_id} as unused")
            return True
            
        except Exception as e:
            logger.error(f"Error marking creative {creative_id} as unused: {e}")
            return False
    
    def get_creative_url(self, creative_id: str) -> Optional[str]:
        """
        Get the public URL for a creative.
        
        Args:
            creative_id: Creative ID
            
        Returns:
            Public URL or None if not found
        """
        try:
            if not self.client:
                return None
            
            result = self.client.table("creative_storage").select(
                "storage_url, storage_path"
            ).eq("creative_id", creative_id).execute()
            
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

