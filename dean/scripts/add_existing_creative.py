#!/usr/bin/env python3
"""
Add an existing creative from Supabase Storage to the ASC+ campaign.
Does NOT generate a new creative - uses the most recent one from storage.
"""
import os
import sys
import json
import requests
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

# Set the new access token
FB_ACCESS_TOKEN = "EAAPhOBXCdjsBPyazNHw9YQvEK2E9o94IKfkXhZCpgobgnHxCSDC2U0aaYqZCTXKtra8J1RWsSZCzVyxSZCK0MPss2wnHWR7L5Xq1Y0zNIAaWzLZBD1bzrO1ChlveGEklvAteZAZCbUZAUZAtD8hLPuUaoGWsVgVZCPktXD8rbcSg62PqbJVJgcE6ICCgsy9cDbBRWbPeaTdKp3PBSc91m1vpMWHkPlzI0gqibjrdyHW9vC9uVKHuiLdKEpLU0LQGqrLLNwkmBDYiuN7XTZAQEWiJmgKUSKQKnk18DR8gSs0pQZDZD"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_supabase_client():
    """Get Supabase client using existing infrastructure"""
    # Import after path is set
    from infrastructure.supabase_storage import get_validated_supabase_client
    client = get_validated_supabase_client()
    if not client:
        raise ValueError("Failed to get Supabase client. Check SUPABASE_URL and SUPABASE_KEY environment variables.")
    return client

def upload_image_to_meta(image_url, ad_account_id):
    """Upload image from URL to Meta's Ad Image library and return hash"""
    account_id = ad_account_id.replace("act_", "")
    
    # First, download the image
    import tempfile
    response = requests.get(image_url)
    response.raise_for_status()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name
    
    try:
        # Upload to Meta Ad Image library
        upload_url = f"https://graph.facebook.com/v21.0/act_{account_id}/adimages"
        with open(tmp_path, "rb") as img_file:
            files = {"file": img_file}
            data = {
                "access_token": FB_ACCESS_TOKEN
            }
            upload_response = requests.post(upload_url, files=files, data=data)
            if upload_response.status_code != 200:
                try:
                    error_data = upload_response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    logger.error(f"Image upload error {upload_response.status_code}: {error_msg}")
                    logger.error(f"Full error: {json.dumps(error_data, indent=2)}")
                except:
                    logger.error(f"Response text: {upload_response.text}")
                raise requests.exceptions.HTTPError(f"Image upload failed: {upload_response.status_code}")
            
            result = upload_response.json()
            logger.info(f"Upload response: {json.dumps(result, indent=2)}")
            
            # Extract hash from response
            # Response format: {"images": {"<hash>": {"hash": "<hash>", "url": "..."}}}
            images = result.get("images", {})
            if images:
                # Get first hash - the key is the hash
                image_hash = list(images.keys())[0]
                # Also verify the hash value inside
                hash_data = images[image_hash]
                actual_hash = hash_data.get("hash", image_hash)
                logger.info(f"✅ Uploaded image to Meta, hash: {actual_hash}")
                return actual_hash
            else:
                raise ValueError(f"No image hash in upload response: {result}")
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

def create_meta_creative(page_id, name, image_url, headline, primary_text, description="", ad_account_id=""):
    """Create a Meta creative using Graph API - uploads image first"""
    # First, upload image to Meta's Ad Image library
    logger.info("Uploading image to Meta Ad Image library...")
    image_hash = upload_image_to_meta(image_url, ad_account_id)
    
    # Remove 'act_' prefix if present
    account_id = ad_account_id.replace("act_", "")
    url = f"https://graph.facebook.com/v21.0/act_{account_id}/adcreatives"
    
    store_url = os.getenv("SHOPIFY_STORE_URL", "https://brava-skin.com")
    
    # Build link_data according to Meta API format - use image_hash, not image_url
    # Note: link_description is NOT supported, use name for headline instead
    link_data = {
        "image_hash": image_hash,  # Use hash from uploaded image
        "link": store_url,
        "message": primary_text,  # This is the primary_text
        "name": headline[:100] if headline else "",  # Use "name" field for headline
    }
    
    if description:
        link_data["description"] = description[:150]
    
    link_data["call_to_action"] = {
        "type": "SHOP_NOW",
        "value": {
            "link": store_url
        }
    }
    
    # Build object_story_spec
    object_story_spec = {
        "page_id": page_id,
        "link_data": link_data
    }
    
    # Use POST body - Meta expects object_story_spec as a dict, not JSON string
    data = {
        "name": name,
        "object_story_spec": json.dumps(object_story_spec),
        "access_token": FB_ACCESS_TOKEN
    }
    
    response = requests.post(url, data=data)
    if response.status_code != 200:
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "Unknown error")
            error_code = error_data.get("error", {}).get("code", "Unknown")
            logger.error(f"Meta API error {response.status_code}: [{error_code}] {error_msg}")
            logger.error(f"Full error: {json.dumps(error_data, indent=2)}")
        except:
            logger.error(f"Response text: {response.text}")
        raise requests.exceptions.HTTPError(f"{response.status_code} Client Error: {error_msg if 'error_msg' in locals() else 'Unknown'}")
    return response.json()

def create_meta_ad(adset_id, name, creative_id, ad_account_id):
    """Create a Meta ad using Graph API"""
    # Remove 'act_' prefix if present
    account_id = ad_account_id.replace("act_", "")
    url = f"https://graph.facebook.com/v21.0/act_{account_id}/ads"
    
    params = {
        "access_token": FB_ACCESS_TOKEN,
        "name": name,
        "adset_id": adset_id,
        "creative": json.dumps({"creative_id": creative_id}),
        "status": "ACTIVE"
    }
    
    response = requests.post(url, params=params)
    response.raise_for_status()
    return response.json()

def main():
    # Get IDs
    adset_id = os.getenv("ASC_PLUS_ADSET_ID", "120233669753240160")
    page_id = os.getenv("FB_PAGE_ID")
    
    if not page_id:
        logger.error("FB_PAGE_ID environment variable is required")
        return
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    # Get the most recent creative from storage
    logger.info("Fetching most recent creative from Supabase Storage...")
    
    try:
        response = supabase.table("creative_storage").select("*").order("created_at", desc=True).limit(1).execute()
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
        
        headline = ad_copy.get("headline", "For men who value discipline.")
        primary_text = ad_copy.get("primary_text", "Precision skincare designed to elevate daily standards.")
        description = ad_copy.get("description", "")
        
        logger.info(f"Ad copy: headline='{headline}', primary_text='{primary_text[:50]}...'")
        
        # Get ad account ID first
        ad_account_id = os.getenv("FB_AD_ACCOUNT_ID", "")
        if not ad_account_id:
            # Try to get from /me/adaccounts
            try:
                me_url = "https://graph.facebook.com/v21.0/me/adaccounts"
                response = requests.get(me_url, params={"access_token": FB_ACCESS_TOKEN})
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data") and len(data["data"]) > 0:
                        ad_account_id = data["data"][0].get("id", "")
                        logger.info(f"Found ad account ID: {ad_account_id}")
            except Exception as e:
                logger.warning(f"Could not auto-detect ad account ID: {e}")
        
        if not ad_account_id:
            raise ValueError("FB_AD_ACCOUNT_ID must be set or detectable from token")
        
        # Create Meta creative
        logger.info("Creating Meta creative...")
        creative_name = f"[ASC+] Creative {creative_id[:8]}"
        
        try:
            meta_creative = create_meta_creative(
                page_id=page_id,
                name=creative_name,
                image_url=supabase_url,
                headline=headline,
                primary_text=primary_text,
                description=description,
                ad_account_id=ad_account_id
            )
            
            meta_creative_id = meta_creative.get("id")
            if not meta_creative_id:
                logger.error(f"Failed to get creative ID from Meta response: {meta_creative}")
                return
            
            logger.info(f"✅ Created Meta creative: {meta_creative_id}")
            
            # Create ad
            logger.info("Creating ad in campaign...")
            ad_name = f"[ASC+] Ad {creative_id[:8]}"
            ad = create_meta_ad(
                adset_id=adset_id,
                name=ad_name,
                creative_id=meta_creative_id,
                ad_account_id=ad_account_id
            )
            
            ad_id = ad.get("id")
            if not ad_id:
                logger.error(f"Failed to get ad ID from Meta response: {ad}")
                return
            
            logger.info(f"✅ Created ad: {ad_id}")
            logger.info("=" * 60)
            logger.info("✅ SUCCESS! Creative added to campaign")
            logger.info(f"   Ad Set ID: {adset_id}")
            logger.info(f"   Ad ID: {ad_id}")
            logger.info(f"   Creative ID: {meta_creative_id}")
            logger.info(f"   Storage Creative ID: {creative_id}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error creating creative/ad: {e}", exc_info=True)
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
