#!/usr/bin/env python3
"""
Quick script to add a single creative to the ASC+ campaign.
Uses the most recent creative from Supabase Storage.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

# Import directly to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location("meta_client", Path(__file__).parent.parent / "src" / "integrations" / "meta_client.py")
meta_client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta_client_module)
MetaClient = meta_client_module.MetaClient

from infrastructure.creative_storage import create_creative_storage_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get campaign and adset IDs from environment or config
    campaign_id = os.getenv("ASC_PLUS_CAMPAIGN_ID", "120233669753230160")
    adset_id = os.getenv("ASC_PLUS_ADSET_ID", "120233669753240160")
    page_id = os.getenv("FB_PAGE_ID")
    
    if not page_id:
        logger.error("FB_PAGE_ID environment variable is required")
        return
    
    # Initialize clients
    client = MetaClient()
    storage_manager = create_creative_storage_manager()
    
    # Get the most recent creative from storage
    logger.info("Fetching most recent creative from Supabase Storage...")
    recent_creatives = storage_manager.list_creatives(limit=1)
    
    if not recent_creatives:
        logger.error("No creatives found in Supabase Storage")
        return
    
    creative_data = recent_creatives[0]
    supabase_url = creative_data.get("storage_url")
    creative_id = creative_data.get("creative_id")
    
    logger.info(f"Using creative: {creative_id}")
    logger.info(f"Storage URL: {supabase_url}")
    
    if not supabase_url:
        logger.error("Creative has no storage URL")
        return
    
    # Get ad copy from creative metadata or use defaults
    metadata = creative_data.get("metadata", {})
    ad_copy = metadata.get("ad_copy", {})
    
    if not ad_copy:
        # Use default calm confidence ad copy
        ad_copy = {
            "headline": "For men who value discipline.",
            "primary_text": "Precision skincare designed to elevate daily standards.",
            "description": ""
        }
    
    # Create Meta creative
    logger.info("Creating Meta creative...")
    try:
        creative_name = f"ASC Plus Creative {creative_id[:8]}"
        meta_creative = client.create_image_creative(
            page_id=page_id,
            name=creative_name,
            supabase_storage_url=supabase_url,
            primary_text=ad_copy.get("primary_text", ""),
            headline=ad_copy.get("headline", ""),
            description=ad_copy.get("description", ""),
        )
        
        if not meta_creative:
            logger.error("Failed to create Meta creative")
            return
        
        meta_creative_id = meta_creative.get("id")
        logger.info(f"✅ Created Meta creative: {meta_creative_id}")
        
        # Create ad
        logger.info("Creating ad in campaign...")
        ad = client.create_ad(
            adset_id=adset_id,
            name=f"ASC Plus Ad {creative_id[:8]}",
            creative_id=meta_creative_id,
            status="ACTIVE",
        )
        
        if not ad:
            logger.error("Failed to create ad")
            return
        
        ad_id = ad.get("id")
        logger.info(f"✅ Created ad: {ad_id}")
        logger.info(f"✅ Successfully added creative to campaign!")
        logger.info(f"   Campaign ID: {campaign_id}")
        logger.info(f"   Ad Set ID: {adset_id}")
        logger.info(f"   Ad ID: {ad_id}")
        logger.info(f"   Creative ID: {meta_creative_id}")
        
    except Exception as e:
        logger.error(f"Error creating creative/ad: {e}", exc_info=True)

if __name__ == "__main__":
    main()

