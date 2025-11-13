#!/usr/bin/env python3
"""
Backfill script to add recently created ads to the new consolidated ads table.
Fetches ad data from Meta API and inserts into Supabase ads table.
"""

import os
import sys
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from infrastructure.supabase_storage import get_validated_supabase_client
from infrastructure.data_validation import validate_supabase_data, validate_all_timestamps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ad IDs from the test run
AD_IDS = [
    "120234165084250160",
    "120234165242300160",
    "120234165585930160",
    "120234166843640160",
    "120234167133540160",
    "120234167275700160",
    "120234167397640160",
    "120234167480760160",
    "120234167700050160",
    "120234167961050160",
]

# Campaign and Adset IDs from config
CAMPAIGN_ID = "120234137971970160"
ADSET_ID = "120234137971980160"


def fetch_ad_from_meta(access_token: str, ad_id: str) -> Optional[Dict[str, Any]]:
    """Fetch ad data from Meta API."""
    try:
        # Fetch ad with all necessary fields
        fields = [
            "id",
            "name",
            "status",
            "effective_status",
            "creative",
            "campaign_id",
            "adset_id",
            "created_time",
            "updated_time",
        ]
        
        url = f"https://graph.facebook.com/v23.0/{ad_id}"
        params = {
            "fields": ",".join(fields),
            "access_token": access_token,
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
                logger.error(f"Meta API error for ad {ad_id}: {error_msg}")
            except:
                logger.error(f"Meta API error for ad {ad_id}: {response.status_code} - {response.text[:200]}")
            return None
        
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch ad {ad_id} from Meta: {e}")
        return None


def extract_creative_id(ad: Dict[str, Any]) -> Optional[str]:
    """Extract creative ID from ad data."""
    creative = ad.get("creative")
    if isinstance(creative, dict):
        return str(creative.get("id", ""))
    elif isinstance(creative, str):
        return creative
    return None


def normalize_status(raw_status: Optional[str]) -> str:
    """Normalize Meta status to ads table status."""
    if not raw_status:
        return "active"
    
    status_map = {
        "ACTIVE": "active",
        "PAUSED": "paused",
        "ARCHIVED": "killed",
        "DELETED": "killed",
        "DISAPPROVED": "killed",
        "WITH_ISSUES": "killed",
    }
    
    key = str(raw_status).strip().upper()
    return status_map.get(key, "active")


def build_ads_record(ad: Dict[str, Any], creative_id: Optional[str]) -> Dict[str, Any]:
    """Build ads table record from Meta ad data."""
    ad_id = str(ad.get("id", ""))
    now = datetime.now(timezone.utc).isoformat()
    
    # Parse created time
    created_at = now
    if ad.get("created_time"):
        try:
            created_at = datetime.fromisoformat(ad.get("created_time").replace("Z", "+00:00")).isoformat()
        except:
            pass
    
    # Get status
    status = normalize_status(ad.get("effective_status") or ad.get("status"))
    
    # Build metadata
    metadata = {
        "source": "backfill",
        "ad_name": ad.get("name", ""),
        "raw_status": ad.get("status"),
        "effective_status": ad.get("effective_status"),
        "created_time": ad.get("created_time"),
        "updated_time": ad.get("updated_time"),
    }
    
    # Build ads record
    # Note: storage fields are optional - will be populated when creative is uploaded
    record = {
        "ad_id": ad_id,
        "creative_id": creative_id or "",
        "campaign_id": str(ad.get("campaign_id") or CAMPAIGN_ID),
        "adset_id": str(ad.get("adset_id") or ADSET_ID),
        "status": status,
        "metadata": metadata,
        "created_at": created_at,
        "updated_at": now,
        # Storage fields - use placeholder values (will be populated by next pipeline run)
        "storage_url": f"https://placeholder.backfill/{ad_id}",
        "storage_path": f"backfill/{ad_id}",
        "file_size_bytes": 0,
        "file_type": "image/jpeg",  # Default file type
    }
    
    # Add killed_at if status is killed
    if status == "killed":
        record["killed_at"] = now
    
    return record


def backfill_ads(ad_ids: List[str]) -> Dict[str, Any]:
    """Backfill ads from Meta to Supabase."""
    results = {
        "success": [],
        "failed": [],
        "skipped": [],
    }
    
    # Get access token
    access_token = os.getenv("FB_ACCESS_TOKEN")
    if not access_token:
        logger.error("FB_ACCESS_TOKEN not set")
        return results
    
    supabase_client = get_validated_supabase_client(enable_validation=True)
    if not supabase_client:
        logger.error("Failed to initialize Supabase client")
        return results
    
    for ad_id in ad_ids:
        logger.info(f"Processing ad {ad_id}...")
        
        # Fetch ad from Meta
        ad = fetch_ad_from_meta(access_token, ad_id)
        if not ad:
            results["failed"].append(ad_id)
            continue
        
        # Check if ad already exists
        try:
            existing = supabase_client.table("ads").select("ad_id").eq("ad_id", ad_id).execute()
            if existing.data:
                logger.info(f"Ad {ad_id} already exists, skipping")
                results["skipped"].append(ad_id)
                continue
        except Exception as e:
            logger.warning(f"Error checking existing ad {ad_id}: {e}")
        
        # Extract creative ID
        creative_id = extract_creative_id(ad)
        if not creative_id:
            logger.warning(f"No creative ID found for ad {ad_id}")
        
        # Build record
        record = build_ads_record(ad, creative_id)
        
        # Validate timestamps
        record = validate_all_timestamps(record)
        
        # For backfill, we'll insert directly since we know the structure is correct
        # The storage fields use placeholder values and will be updated by the next pipeline run
        
        # Insert into Supabase (bypass validation for backfill)
        try:
            supabase_client.table("ads").insert(record).execute()
            logger.info(f"✅ Successfully backfilled ad {ad_id}")
            results["success"].append(ad_id)
        except Exception as e:
            logger.error(f"Failed to insert ad {ad_id}: {e}")
            results["failed"].append(ad_id)
    
    return results


def main():
    """Main entry point."""
    logger.info("Starting ad backfill...")
    logger.info(f"Processing {len(AD_IDS)} ads")
    
    results = backfill_ads(AD_IDS)
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("Backfill Results:")
    logger.info(f"  ✅ Success: {len(results['success'])}")
    logger.info(f"  ❌ Failed: {len(results['failed'])}")
    logger.info(f"  ⏭️  Skipped: {len(results['skipped'])}")
    logger.info("=" * 50)
    
    if results["success"]:
        logger.info(f"\nSuccessfully backfilled ads: {', '.join(results['success'])}")
    
    if results["failed"]:
        logger.error(f"\nFailed to backfill ads: {', '.join(results['failed'])}")
    
    if results["skipped"]:
        logger.info(f"\nSkipped (already exist): {', '.join(results['skipped'])}")


if __name__ == "__main__":
    main()

