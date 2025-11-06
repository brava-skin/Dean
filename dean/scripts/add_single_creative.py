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

# Import directly without circular dependencies
import importlib.util

# Load meta_client
spec = importlib.util.spec_from_file_location("meta_client", Path(__file__).parent.parent / "src" / "integrations" / "meta_client.py")
meta_client_module = importlib.util.module_from_spec(spec)
sys.modules['integrations.meta_client'] = meta_client_module
spec.loader.exec_module(meta_client_module)
MetaClient = meta_client_module.MetaClient

# Load supabase storage
spec2 = importlib.util.spec_from_file_location("supabase_storage", Path(__file__).parent.parent / "src" / "infrastructure" / "supabase_storage.py")
supabase_storage_module = importlib.util.module_from_spec(spec2)
sys.modules['infrastructure.supabase_storage'] = supabase_storage_module
spec2.loader.exec_module(supabase_storage_module)
get_validated_supabase_client = supabase_storage_module.get_validated_supabase_client

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get campaign and adset IDs from environment or config
    adset_id = os.getenv("ASC_PLUS_ADSET_ID", "120233669753240160")
    page_id = os.getenv("FB_PAGE_ID")
    
    if not page_id:
        logger.error("FB_PAGE_ID environment variable is required")
        return
    
    # Initialize clients
    client = MetaClient()
    supabase_client = get_validated_supabase_client()
    
    # Get the most recent creative from storage
    logger.info("Fetching most recent creative from Supabase Storage...")
    
    try:
        response = supabase_client.table("creative_storage").select("*").order("created_at", desc=True).limit(1).execute()
        if not response.data:
            logger.error("No creatives found in Supabase Storage")
            return
        
        creative_data = response.data[0]
        supabase_url = creative_data.get("storage_url")
        creative_id = creative_data.get("creative_id")
        metadata = creative_data.get("metadata", {})
        
        logger.info(f"Using creative: {creative_id}")
        logger.info(f"Storage URL: {supabase_url}")
        
        if not supabase_url:
            logger.error("Creative has no storage URL")
            return
        
        # Get ad copy from metadata
        ad_copy = metadata.get("ad_copy", {
            "headline": "For men who value discipline.",
            "primary_text": "Precision skincare designed to elevate daily standards.",
            "description": ""
        })
        
        # Create Meta creative
        logger.info("Creating Meta creative...")
        creative_name = f"ASC Plus Creative {creative_id[:8]}"
        
        try:
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
            ad_name = f"ASC Plus Ad {creative_id[:8]}"
            ad = client.create_ad(
                adset_id=adset_id,
                name=ad_name,
                creative_id=meta_creative_id,
                status="ACTIVE",
            )
            
            if not ad:
                logger.error("Failed to create ad")
                return
            
            ad_id = ad.get("id")
            logger.info(f"✅ Created ad: {ad_id}")
            logger.info(f"✅ Successfully added creative to campaign!")
            logger.info(f"   Ad Set ID: {adset_id}")
            logger.info(f"   Ad ID: {ad_id}")
            logger.info(f"   Creative ID: {meta_creative_id}")
            
        except Exception as e:
            logger.error(f"Error creating creative/ad: {e}", exc_info=True)
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
