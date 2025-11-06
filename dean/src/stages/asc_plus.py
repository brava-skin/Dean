"""
ASC+ Campaign Stage Handler
Manages a single Advantage+ Shopping Campaign with 5 creatives
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

from integrations.slack import notify, alert_kill, alert_error
from integrations.meta_client import MetaClient
from infrastructure.utils import (
    getenv_f, getenv_i, getenv_b, cfg, cfg_or_env_f, cfg_or_env_i, cfg_or_env_b, cfg_or_env_list,
    safe_f, today_str, daily_key, ad_day_flag_key, now_minute_key, clean_text_token, prettify_ad_name
)
from creative.image_generator import create_image_generator, ImageCreativeGenerator
from config.constants import (
    ASC_PLUS_BUDGET_MIN, ASC_PLUS_BUDGET_MAX, ASC_PLUS_MIN_BUDGET_PER_CREATIVE
)

# Import advanced ML systems
try:
    from ml.advanced_system import create_advanced_ml_system
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    logger.warning("Advanced ML system not available")

# Import new optimization systems
try:
    from ml.budget_scaling import create_budget_scaling_engine, ScalingStrategy
    from ml.creative_refresh import create_creative_refresh_manager
    from infrastructure.optimization import create_resource_optimizer
    OPTIMIZATION_SYSTEMS_AVAILABLE = True
except ImportError:
    OPTIMIZATION_SYSTEMS_AVAILABLE = False
    logger.warning("Optimization systems not available")

UTC = timezone.utc
LOCAL_TZ = ZoneInfo(os.getenv("ACCOUNT_TZ", os.getenv("ACCOUNT_TIMEZONE", "Europe/Amsterdam")))
ACCOUNT_CURRENCY = os.getenv("ACCOUNT_CURRENCY", "EUR")


def _ctr(row: Dict[str, Any]) -> float:
    imps = safe_f(row.get("impressions"))
    clicks = safe_f(row.get("clicks"))
    return (clicks / imps) if imps > 0 else 0.0


def _roas(row: Dict[str, Any]) -> float:
    """Extract ROAS from purchase_roas array."""
    roas_list = row.get("purchase_roas") or []
    try:
        if roas_list:
            return float(roas_list[0].get("value", 0)) or 0.0
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    return 0.0


def _purchase_and_atc_counts(row: Dict[str, Any]) -> Tuple[int, int]:
    acts = row.get("actions") or []
    purch = 0
    atc = 0
    for a in acts:
        t = a.get("action_type")
        v = safe_f(a.get("value"), 0.0)
        if t == "purchase":
            purch += int(v)
        elif t == "add_to_cart":
            atc += int(v)
    return purch, atc


def _cpa(row: Dict[str, Any]) -> float:
    spend = safe_f(row.get("spend"))
    purch, _ = _purchase_and_atc_counts(row)
    return (spend / purch) if purch > 0 else float('inf')


def _cpm(row: Dict[str, Any]) -> float:
    spend = safe_f(row.get("spend"))
    imps = safe_f(row.get("impressions"))
    return (spend / imps * 1000) if imps > 0 else 0.0


def _meets_minimums(row: Dict[str, Any], min_impressions: int, min_clicks: int, min_spend: float) -> bool:
    return (
        safe_f(row.get("spend")) >= min_spend
        and safe_f(row.get("impressions")) >= min_impressions
        and safe_f(row.get("clicks")) >= min_clicks
    )


def _active_count(ads_list: List[Dict[str, Any]]) -> int:
    return sum(1 for a in ads_list if str(a.get("status", "")).upper() == "ACTIVE")


def _create_creative_and_ad(
    client: MetaClient,
    image_generator: ImageCreativeGenerator,
    creative_data: Dict[str, Any],
    adset_id: str,
    active_count: int,
    created_count: int,
    existing_creative_ids: set,
    ml_system: Optional[Any] = None,
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Create a creative and ad in Meta from creative data.
    Uses Supabase Storage for image hosting.
    
    Args:
        client: Meta client instance
        image_generator: Image generator instance (for validation)
        creative_data: Dictionary with creative data (image_path, ad_copy, supabase_storage_url, etc.)
        adset_id: Ad set ID to create ad in
        active_count: Current count of active ads
        created_count: Current count of created ads in this batch
        existing_creative_ids: Set of already created creative IDs to prevent duplicates
        ml_system: Optional ML system for tracking
    
    Returns:
        Tuple of (creative_id, ad_id, success)
    """
    # Input validation
    if not client or not adset_id or not creative_data:
        logger.error("Invalid input: client, adset_id, and creative_data are required")
        return None, None, False
    
    # Use storage_creative_id from creative_data if available, otherwise generate from image hash
    storage_creative_id = creative_data.get("storage_creative_id")
    if not storage_creative_id:
        import hashlib
        image_path = creative_data.get("image_path")
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
            storage_creative_id = f"creative_{image_hash[:12]}"
        else:
            import time
            storage_creative_id = f"creative_{int(time.time())}"
    
    try:
        # Create descriptive creative name for Ads Manager
        # Include headline snippet and sequence for easy identification
        ad_copy_dict = creative_data.get("ad_copy") or {}
        if not isinstance(ad_copy_dict, dict):
            ad_copy_dict = {}
        
        headline = ad_copy_dict.get("headline", "")
        if headline:
            # Extract first 2-3 words from headline (max 25 chars)
            headline_words = headline.split()[:3]
            headline_snippet = " ".join(headline_words)
            if len(headline_snippet) > 25:
                headline_snippet = headline_snippet[:22] + "..."
            # Clean for filename safety
            headline_snippet = headline_snippet.replace(":", "").replace("/", "").replace("\\", "").strip()
        else:
            headline_snippet = "Creative"
        
        # Sequence number
        seq_num = active_count + created_count + 1
        
        # Build descriptive creative name: [ASC+] HeadlineSnippet - #N
        creative_name = f"[ASC+] {headline_snippet} - #{seq_num}"
        
        # Ensure it's not too long (keep under 80 chars for creatives)
        if len(creative_name) > 80:
            creative_name = f"[ASC+] {headline_snippet[:15]} - #{seq_num}"
        
        # Use Supabase Storage URL if available, otherwise fallback to image_path
        supabase_storage_url = creative_data.get("supabase_storage_url")
        image_path = creative_data.get("image_path")
        
        # Ensure we have either supabase_storage_url or image_path
        if not supabase_storage_url and not image_path:
            logger.error("Creative data must have either supabase_storage_url or image_path")
            return None, None, False
        
        # Ensure ad_copy is a dict (not None) - already set above, but ensure it's valid
        if not isinstance(ad_copy_dict, dict):
            ad_copy_dict = {}
        
        # Validate page_id
        page_id = os.getenv("FB_PAGE_ID")
        if not page_id:
            logger.error("FB_PAGE_ID environment variable is required")
            return None, None, False
        
        # Get Instagram actor ID
        instagram_actor_id = os.getenv("IG_ACTOR_ID")
        
        logger.info(f"Creating Meta creative: name='{creative_name}', page_id='{page_id}', instagram_actor_id={bool(instagram_actor_id)}, has_supabase_url={bool(supabase_storage_url)}, has_image_path={bool(image_path)}")
        try:
            # Clean primary text - remove "Brava Product" and em dashes
            primary_text = ad_copy_dict.get("primary_text", "")
            if primary_text:
                import re
                primary_text = primary_text.replace("Brava Product", "").replace("—", ",").replace("–", ",").strip()
                primary_text = re.sub(r'\s+', ' ', primary_text).strip()
                # Ensure it's not too long
                if len(primary_text) > 150:
                    primary_text = primary_text[:147] + "..."
            
            # Create single image creative with catalog products
            # Note: For ASC+ campaigns, the catalog is configured at the ad set level
            # Meta will automatically show product cards from the catalog below the single image
            # The creative is a single image with primary text, headline, and "Shop now" CTA
            
            creative = client.create_image_creative(
                page_id=page_id,
                name=creative_name,
                supabase_storage_url=supabase_storage_url,  # Use Supabase Storage URL
                image_path=image_path if not supabase_storage_url else None,
                primary_text=primary_text,
                headline=ad_copy_dict.get("headline", ""),
                description=ad_copy_dict.get("description", ""),
                call_to_action="SHOP_NOW",  # "Shop now" CTA
                instagram_actor_id=instagram_actor_id,  # Add Instagram account
                creative_id=storage_creative_id,  # Pass storage creative_id for tracking
            )
            logger.info(f"Meta API create_image_creative response: {creative}")
        except Exception as e:
            logger.error(f"Meta API create_image_creative failed: {e}", exc_info=True)
            return None, None, False
        
        meta_creative_id = creative.get("id")
        if not meta_creative_id:
            logger.error(f"Failed to get creative ID from Meta response. Response: {creative}")
            return None, None, False
        
        # Check for duplicate creative (use Meta's creative ID)
        if str(meta_creative_id) in existing_creative_ids:
            logger.debug(f"Skipping duplicate creative: {meta_creative_id}")
            return str(meta_creative_id), None, False
        
        # Create descriptive ad name for Ads Manager
        # Include headline snippet, date, and sequence for easy identification
        headline = ad_copy_dict.get("headline", "")
        if headline:
            # Extract first 3-4 words from headline (max 30 chars)
            headline_words = headline.split()[:4]
            headline_snippet = " ".join(headline_words)
            if len(headline_snippet) > 30:
                headline_snippet = headline_snippet[:27] + "..."
            # Clean for filename safety
            headline_snippet = headline_snippet.replace(":", "").replace("/", "").replace("\\", "").strip()
        else:
            headline_snippet = "Creative"
        
        # Get date in YYMMDD format for easy sorting
        from datetime import datetime
        date_str = datetime.now().strftime("%y%m%d")
        
        # Sequence number
        seq_num = active_count + created_count + 1
        
        # Build descriptive ad name: [ASC+] HeadlineSnippet - YYMMDD - #N
        ad_name = f"[ASC+] {headline_snippet} - {date_str} - #{seq_num}"
        
        # Ensure it's not too long (Meta has limits, keep under 100 chars)
        if len(ad_name) > 100:
            ad_name = f"[ASC+] {headline_snippet[:20]} - {date_str} - #{seq_num}"
        
        logger.info(f"Creating ad with name='{ad_name}', adset_id='{adset_id}', creative_id='{meta_creative_id}'")
        try:
            ad = client.create_ad(
                adset_id=adset_id,
                name=ad_name,
                creative_id=meta_creative_id,  # Use Meta's creative ID
                status="ACTIVE",
            )
            logger.info(f"Meta API create_ad response: {ad}")
        except Exception as e:
            logger.error(f"Meta API create_ad failed: {e}", exc_info=True)
            return str(meta_creative_id), None, False
        
        ad_id = ad.get("id")
        if ad_id:
            logger.info(f"Successfully created ad with ad_id={ad_id}")
            existing_creative_ids.add(str(meta_creative_id))
            
            # Update creative_intelligence with Supabase Storage URL and Meta creative ID
            try:
                from infrastructure.supabase_storage import get_validated_supabase_client
                supabase_client = get_validated_supabase_client()
                if supabase_client:
                    creative_intel_data = {
                        "creative_id": str(meta_creative_id),  # Use Meta's creative ID
                        "ad_id": ad_id,
                        "creative_type": "image",
                        "metadata": {
                            "ad_copy": creative_data.get("ad_copy"),
                            "flux_request_id": creative_data.get("flux_request_id"),
                            "storage_creative_id": storage_creative_id,  # Store our internal ID
                            "scenario_description": creative_data.get("scenario_description"),  # Store scenario for ML learning
                        },
                    }
                    # Add optional fields if available
                    if creative_data.get("supabase_storage_url"):
                        creative_intel_data["supabase_storage_url"] = creative_data.get("supabase_storage_url")
                    if creative_data.get("image_prompt"):
                        creative_intel_data["image_prompt"] = creative_data.get("image_prompt")
                    if creative_data.get("text_overlay"):
                        creative_intel_data["text_overlay_content"] = creative_data.get("text_overlay")
                    
                    supabase_client.table("creative_intelligence").upsert(
                        creative_intel_data,
                        on_conflict="creative_id,ad_id"
                    ).execute()
            except Exception as e:
                logger.warning(f"Failed to update creative_intelligence with storage URL: {e}")
            
            # Track in ML system
            if ml_system:
                try:
                    ml_system.record_creative_creation(
                        ad_id=ad_id,
                        creative_data=creative_data,
                        performance_data={},
                    )
                except (AttributeError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to track creative in ML system: {e}")
            
            return str(meta_creative_id), ad_id, True
        else:
            logger.warning(f"Failed to get ad ID from Meta response")
            return str(meta_creative_id), None, False
            
    except Exception as e:
        logger.error(f"Error creating creative and ad: {e}", exc_info=True)
        return None, None, False


def ensure_asc_plus_campaign(
    client: MetaClient,
    settings: Dict[str, Any],
    store: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Ensure ASC+ campaign and adset exist.
    Returns (campaign_id, adset_id) or (None, None) on failure.
    """
    try:
        # Check if campaign already exists
        campaign_id = cfg(settings, "ids.asc_plus_campaign_id") or ""
        adset_id = cfg(settings, "ids.asc_plus_adset_id") or ""
        
        if campaign_id and adset_id:
            # Verify they still exist
            try:
                # Try to get campaign info
                client._graph_get_object(f"{campaign_id}", params={"fields": "id,name,status"})
                return campaign_id, adset_id
            except Exception:
                # Campaign doesn't exist, create new
                pass
        
        # Create ASC+ campaign
        campaign_name = "[ASC+] Brava - US Men"
        campaign = client.ensure_campaign(
            name=campaign_name,
            objective="SALES",
            buying_type="AUCTION",
        )
        campaign_id = campaign.get("id")
        
        if not campaign_id:
            notify("❌ Failed to create ASC+ campaign")
            return None, None
        
        # Create ASC+ adset with Advantage+ placements
        asc_config = cfg(settings, "asc_plus") or {}
        daily_budget = cfg_or_env_f(asc_config, "daily_budget_eur", "ASC_PLUS_BUDGET", 50.0)
        
        # Validate budget
        if daily_budget < ASC_PLUS_BUDGET_MIN:
            notify(f"⚠️ ASC+ budget too low: €{daily_budget:.2f}. Minimum is €{ASC_PLUS_BUDGET_MIN:.2f}")
            daily_budget = ASC_PLUS_BUDGET_MIN
        elif daily_budget > ASC_PLUS_BUDGET_MAX:
            notify(f"⚠️ ASC+ budget too high: €{daily_budget:.2f}. Capping at €{ASC_PLUS_BUDGET_MAX:.2f}")
            daily_budget = ASC_PLUS_BUDGET_MAX
        
        # Verify budget matches target active ads
        target_ads = cfg(settings, "asc_plus.target_active_ads") or 5
        budget_per_creative = daily_budget / target_ads if target_ads > 0 else daily_budget
        min_budget_per_creative = cfg_or_env_f(asc_config, "min_budget_per_creative_eur", None, ASC_PLUS_MIN_BUDGET_PER_CREATIVE)
        
        if budget_per_creative < min_budget_per_creative:
            notify(f"⚠️ Budget per creative (€{budget_per_creative:.2f}) below minimum (€{min_budget_per_creative:.2f})")
            notify(f"   Consider increasing daily budget or reducing target active ads")
        
        # Targeting: US, Men, 18-54
        targeting = {
            "age_min": 18,
            "age_max": 54,
            "genders": [1],  # Men
            "geo_locations": {"countries": ["US"]},
        }
        
        # Create adset with Advantage+ placements
        adset_name = "[ASC+] US Men"
        adset = client.ensure_adset(
            campaign_id=campaign_id,
            name=adset_name,
            daily_budget=daily_budget,
            optimization_goal="OFFSITE_CONVERSIONS",
            billing_event="IMPRESSIONS",
            bid_strategy="LOWEST_COST_WITHOUT_CAP",
            targeting=targeting,
            placements=["facebook", "instagram"],  # Advantage+ will be applied automatically
            status="PAUSED",
        )
        
        adset_id = adset.get("id")
        if not adset_id:
            notify("❌ Failed to create ASC+ adset")
            return None, None
        
        # Verify budget was set correctly
        try:
            adset_budget = client.get_adset_budget(adset_id)
            if adset_budget and abs(adset_budget - daily_budget) > 0.01:
                notify(f"⚠️ Budget mismatch: requested €{daily_budget:.2f}, got €{adset_budget:.2f}")
        except Exception:
            pass
        
        notify(f"✅ ASC+ campaign created: {campaign_id}, adset: {adset_id}, budget: €{daily_budget:.2f}/day")
        return campaign_id, adset_id
        
    except Exception as e:
        alert_error(f"Error ensuring ASC+ campaign: {e}")
        return None, None


def generate_new_creative(
    image_generator: ImageCreativeGenerator,
    product_info: Dict[str, Any],
    creative_index: int,
) -> Optional[Dict[str, Any]]:
    """Generate a new static image creative."""
    try:
        creative_data = image_generator.generate_creative(
            product_info=product_info,
            creative_style="Luxury, premium, sophisticated",
        )
        
        if not creative_data:
            notify(f"❌ Failed to generate creative #{creative_index}")
            return None
        
        return creative_data
        
    except Exception as e:
        notify(f"❌ Error generating creative: {e}")
        return None


def run_asc_plus_tick(
    client: MetaClient,
    settings: Dict[str, Any],
    rules: Dict[str, Any],
    store: Any,
    ml_system: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run one tick of ASC+ campaign management.
    Ensures 5 creatives are always live.
    Uses advanced ML system for optimization.
    """
    try:
        # Health check
        from infrastructure.health_check import health_checker
        health_status = health_checker.get_overall_health()
        if health_status.value == "unhealthy":
            logger.warning("System health check failed")
        
        # Self-healing
        if ADVANCED_ML_AVAILABLE and ml_system:
            try:
                from ml.auto_optimization import create_self_healing_system
                from infrastructure.health_check import get_health_status
                
                healing_system = create_self_healing_system()
                health_data = get_health_status()
                issues = healing_system.detect_issues(health_data)
                
                for issue in issues:
                    healing_system.attempt_recovery(issue)
            except Exception as e:
                logger.error(f"Self-healing error: {e}")
        # Ensure campaign exists
        campaign_id, adset_id = ensure_asc_plus_campaign(client, settings, store)
        if not campaign_id or not adset_id:
            return {"ok": False, "error": "Failed to ensure ASC+ campaign"}
        
        # Get current ads
        ads = client.list_ads_in_adset(adset_id)
        active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
        active_count = len(active_ads)
        
        target_count = cfg(settings, "asc_plus.target_active_ads") or 5
        
        # Get insights for current ads (with caching)
        from infrastructure.caching import cache_manager
        cache_key = f"ad_insights_{adset_id}_{datetime.now().strftime('%Y%m%d%H')}"
        
        insights = cache_manager.get(cache_key, namespace="insights")
        if not insights:
            insights = client.get_ad_insights(
                level="ad",
                time_range={"since": (datetime.now(LOCAL_TZ) - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), "until": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")},
            )
            # Convert dict_values/dict_keys to list for caching (prevents pickle errors)
            if insights:
                if isinstance(insights, dict):
                    insights = list(insights.values()) if insights else []
                elif hasattr(insights, '__iter__') and not isinstance(insights, (list, tuple, str)):
                    insights = list(insights)
            # Cache for 1 hour
            cache_manager.set(cache_key, insights, ttl_seconds=3600, namespace="insights")
        
        # Process ads for kill decisions
        asc_rules = cfg(rules, "asc_plus") or {}
        kill_rules = asc_rules.get("kill", [])
        
        killed_count = 0
        for ad in active_ads:
            ad_id = ad.get("id")
            ad_insight = next((i for i in insights if i.get("ad_id") == ad_id), None)
            
            if not ad_insight:
                continue
            
            # Evaluate kill rules
            should_kill = False
            kill_reason = ""
            
            spend = safe_f(ad_insight.get("spend"))
            ctr = _ctr(ad_insight)
            cpa = _cpa(ad_insight)
            roas = _roas(ad_insight)
            cpm = _cpm(ad_insight)
            purch, atc = _purchase_and_atc_counts(ad_insight)
            
            # Apply kill rules
            for rule in kill_rules:
                rule_type = rule.get("type")
                
                if rule_type == "zero_performance_quick_kill":
                    if spend >= rule.get("spend_gte", 0) and ctr < rule.get("ctr_lt", 0):
                        should_kill = True
                        kill_reason = "Zero performance"
                        break
                
                elif rule_type == "cpm_above":
                    if spend >= rule.get("spend_gte", 0) and cpm > rule.get("cpm_above", float('inf')):
                        should_kill = True
                        kill_reason = f"CPM too high: €{cpm:.2f}"
                        break
                
                elif rule_type == "ctr_below":
                    if spend >= rule.get("spend_gte", 0) and ctr < rule.get("ctr_lt", 0):
                        should_kill = True
                        kill_reason = f"CTR too low: {ctr*100:.2f}%"
                        break
                
                elif rule_type == "spend_no_purchase":
                    if spend >= rule.get("spend_gte", 0) and purch == 0:
                        should_kill = True
                        kill_reason = f"No purchases after €{spend:.2f} spend"
                        break
                
                elif rule_type == "cpa_gte":
                    if purch > 0 and cpa >= rule.get("cpa_gte", float('inf')):
                        should_kill = True
                        kill_reason = f"CPA too high: €{cpa:.2f}"
                        break
                
                elif rule_type == "roas_below":
                    if spend >= rule.get("spend_gte", 0) and roas < rule.get("roas_lt", 0):
                        should_kill = True
                        kill_reason = f"ROAS too low: {roas:.2f}"
                        break
            
            if should_kill:
                try:
                    # Auto-pause underperformer (with retry)
                    from infrastructure.error_handling import retry_with_backoff
                    
                    @retry_with_backoff(max_retries=3)
                    def pause_ad(ad_id: str):
                        try:
                            client._graph_post(f"{ad_id}", {"status": "PAUSED"})
                        except Exception:
                            from facebook_business.adobjects.ad import Ad
                            Ad(ad_id).api_update(params={"status": "PAUSED"})
                    
                    pause_ad(ad_id)
                    killed_count += 1
                    active_count -= 1
                    
                    # Prepare metrics for alert
                    alert_metrics = {
                        "spend": spend,
                        "impressions": safe_f(ad_insight.get("impressions")),
                        "clicks": safe_f(ad_insight.get("clicks")),
                        "ctr": ctr,
                        "cpa": cpa,
                        "roas": roas,
                        "cpm": cpm,
                        "purchases": purch,
                    }
                    
                    alert_kill(
                        stage="ASC+",
                        entity_name=ad.get("name", "Unknown"),
                        reason=kill_reason,
                        metrics=alert_metrics,
                    )
                    
                    # Track kill in ML system
                    if ml_system:
                        try:
                            ml_system.record_creative_kill(
                                ad_id=ad_id,
                                reason=kill_reason,
                                performance_data=ad_insight,
                            )
                        except (AttributeError, ValueError, TypeError) as e:
                            logger.warning(f"Failed to track creative kill in ML system: {e}")
                    
                    # Mark creative as killed in storage
                    try:
                        from infrastructure.creative_storage import create_creative_storage_manager
                        from infrastructure.supabase_storage import get_validated_supabase_client
                        
                        supabase_client = get_validated_supabase_client()
                        if supabase_client:
                            storage_manager = create_creative_storage_manager(supabase_client)
                            if storage_manager:
                                # Get creative_id from ad data or creative object
                                creative_id = None
                                if isinstance(ad.get("creative"), dict):
                                    creative_id = ad.get("creative", {}).get("id")
                                elif isinstance(ad.get("creative"), str):
                                    creative_id = ad.get("creative")
                                else:
                                    creative_id = ad.get("creative_id") or ad_insight.get("creative_id")
                                
                                if creative_id:
                                    # Try to find storage_creative_id from creative_intelligence
                                    try:
                                        ci_result = supabase_client.table("creative_intelligence").select(
                                            "metadata"
                                        ).eq("creative_id", str(creative_id)).execute()
                                        if ci_result.data and len(ci_result.data) > 0:
                                            metadata = ci_result.data[0].get("metadata", {})
                                            storage_creative_id = metadata.get("storage_creative_id")
                                            if storage_creative_id:
                                                storage_manager.mark_creative_killed(storage_creative_id)
                                            else:
                                                # Fallback: use Meta creative_id
                                                storage_manager.mark_creative_killed(str(creative_id))
                                    except Exception as e2:
                                        logger.warning(f"Failed to lookup storage_creative_id: {e2}")
                                        # Fallback: try with Meta creative_id
                                        storage_manager.mark_creative_killed(str(creative_id))
                    except Exception as e:
                        logger.warning(f"Failed to mark creative as killed in storage: {e}")
                            
                except (KeyError, ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Failed to kill ad {ad_id}: {e}", exc_info=True)
                    notify(f"⚠️ Failed to kill ad {ad_id}: {e}")
        
        # Generate new creatives if needed - SMART: Only generate 1 at a time when needed
        # Refresh active count after kills
        ads = client.list_ads_in_adset(adset_id)
        active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
        active_count = len(active_ads)
        needed_count = max(0, target_count - active_count)
        
        # HARD STOP: If we already have the target count, do NOT generate anything
        if active_count >= target_count:
            logger.info(f"✅ Already have {active_count} active creatives (target: {target_count}), STOPPING - no generation needed")
            notify(f"✅ Target reached: {active_count}/{target_count} active creatives - NO GENERATION")
            return {
                "campaign_id": campaign_id,
                "adset_id": adset_id,
                "active_count": active_count,
                "target_count": target_count,
                "created_count": 0,
                "killed_count": killed_count,
            }
        
        # SMART GENERATION: Check queue first, only generate if needed
        # This prevents overusage of Flux and ChatGPT
        if needed_count > 0:
            # STEP 1: Check for queued creatives first
            from infrastructure.creative_storage import create_creative_storage_manager
            from infrastructure.supabase_storage import get_validated_supabase_client
            
            supabase_client = get_validated_supabase_client()
            storage_manager = None
            queued_creative = None
            
            if supabase_client:
                try:
                    storage_manager = create_creative_storage_manager(supabase_client)
                    if storage_manager:
                        queued_creative = storage_manager.get_queued_creative()
                except Exception as e:
                    logger.warning(f"Failed to check creative queue: {e}")
            
            # If we found a queued creative, use it instead of generating
            if queued_creative:
                logger.info(f"✅ Using queued creative: {queued_creative.get('creative_id')}")
                notify(f"📦 Using queued creative (need {needed_count} more, have {active_count} active)")
                
                # Get the creative data from storage
                storage_url = queued_creative.get("storage_url")
                storage_creative_id = queued_creative.get("creative_id")
                metadata = queued_creative.get("metadata", {})
                
                # Reconstruct creative_data from metadata
                creative_data = {
                    "supabase_storage_url": storage_url,
                    "creative_id": storage_creative_id,
                    "ad_copy": metadata.get("ad_copy", {}),
                    "text_overlay": metadata.get("text_overlay", ""),
                    "image_prompt": metadata.get("image_prompt", ""),
                    "scenario_description": metadata.get("scenario_description", ""),
                }
                
                # Create ad with queued creative
                creative_id, ad_id, success = _create_creative_and_ad(
                    client=client,
                    image_generator=None,  # Not needed for queued creative
                    creative_data=creative_data,
                    adset_id=adset_id,
                    active_count=active_count,
                    created_count=0,
                    existing_creative_ids=set(),
                    ml_system=ml_system,
                )
                
                if success and creative_id and ad_id:
                    # Mark creative as active
                    if storage_manager:
                        storage_manager.mark_creative_active(storage_creative_id, ad_id)
                    
                    logger.info(f"✅ Successfully used queued creative {storage_creative_id} for ad {ad_id}")
                    notify(f"✅ Created ad {ad_id} using queued creative")
                    
                    # Return early - we used a queued creative, no generation needed
                    return {
                        "campaign_id": campaign_id,
                        "adset_id": adset_id,
                        "active_count": active_count + 1,
                        "target_count": target_count,
                        "created_count": 1,
                        "killed_count": killed_count,
                    }
                else:
                    logger.warning(f"Failed to create ad with queued creative, will generate new one")
                    # Continue to generation below
            
            # STEP 2: No queued creative available - generate exactly 1
            if not queued_creative:
                notify(f"📸 Generating EXACTLY 1 new creative (need {needed_count} more to reach {target_count}, currently {active_count} active)")
                
                # Initialize image generator with ML system
                from creative.image_generator import create_image_generator
                import os
                image_generator = create_image_generator(
                    flux_api_key=os.getenv("FLUX_API_KEY"),
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    ml_system=ml_system,
                )
            
            # Initialize advanced ML system if available
            advanced_ml = None
            if ADVANCED_ML_AVAILABLE and ml_system:
                try:
                    advanced_ml = create_advanced_ml_system(
                        supabase_client=ml_system.supabase_client if hasattr(ml_system, 'supabase_client') else None,
                        image_generator=image_generator,
                        ml_system=ml_system,
                    )
                    notify("🚀 Advanced ML system activated")
                except Exception as e:
                    notify(f"⚠️ Failed to initialize advanced ML: {e}")
                    advanced_ml = None
            
            # Load product info from settings
            product_config = cfg(settings, "asc_plus.product") or {}
            product_info = {
                "name": product_config.get("name", "Brava Product"),
                "description": product_config.get("description", "Luxury product for men"),
                "features": product_config.get("features", ["Premium quality", "Luxury design", "Sophisticated"]),
                "brand_tone": product_config.get("brand_tone", "calm confidence"),
                "target_audience": product_config.get("target_audience", "Men aged 18-54"),
            }
            
            created_count = 0
            failed_count = 0
            failed_reasons = []
            max_attempts = needed_count * 3  # Allow up to 3x attempts in case of failures
            attempt_count = 0
            existing_creative_ids = set()  # Track created creative IDs to prevent duplicates
            skip_standard_generation = False  # Flag to skip standard generation if we already generated 1
            
            # Use advanced ML pipeline if available
            if advanced_ml and advanced_ml.creative_pipeline:
                notify("🎯 Using smart ML-driven creative generation (1 creative at a time)")
                try:
                    # SMART: Only generate 1 creative at a time when needed
                    # Use ML insights from killed creatives to inform generation
                    remaining_needed = target_count - active_count
                    
                    if remaining_needed <= 0:
                        logger.info(f"✅ Already have {active_count} active creatives (target: {target_count}), no generation needed")
                    else:
                        # SMART: Generate exactly 1 creative using ML insights
                        # The ML system will use insights from killed creatives to generate the best possible creative
                        logger.info(f"📸 Generating 1 smart creative using ML insights (need {remaining_needed} more, have {active_count} active)")
                        
                        # Generate exactly 1 creative - the pipeline will use ML insights to make it optimal
                        generated_creatives = advanced_ml.generate_optimized_creatives(
                            product_info,
                            target_count=1,  # Always generate exactly 1
                        )
                        
                        # Process the single generated creative
                        if generated_creatives and len(generated_creatives) > 0:
                            creative_data = generated_creatives[0]  # Only process the first (and only) creative
                            logger.info(f"Processing smart ML-driven creative: has supabase_storage_url={bool(creative_data.get('supabase_storage_url'))}, has image_path={bool(creative_data.get('image_path'))}, has ad_copy={bool(creative_data.get('ad_copy'))}")
                            
                            if not creative_data:
                                logger.warning(f"Creative generation returned None")
                                failed_count += 1
                                failed_reasons.append("Creative generation returned None")
                            else:
                                # Analyze creative with advanced ML (optional - already optimized)
                                if advanced_ml:
                                    try:
                                        analysis = advanced_ml.analyze_creative_performance(
                                            creative_data,
                                            {},
                                        )
                                        creative_data["ml_analysis"] = analysis
                                        
                                        # Quality check
                                        if advanced_ml.quality_checker:
                                            quality = advanced_ml.quality_checker.check_quality(creative_data)
                                            creative_data["quality_check"] = quality
                                            
                                            if not quality.get("passed_checks"):
                                                logger.warning(f"Creative failed quality checks: {quality}")
                                                failed_count += 1
                                                failed_reasons.append("Creative failed quality check")
                                            else:
                                                # Create creative and ad
                                                creative_id, ad_id, success = _create_creative_and_ad(
                                                    client=client,
                                                    image_generator=image_generator,
                                                    creative_data=creative_data,
                                                    adset_id=adset_id,
                                                    active_count=active_count,
                                                    created_count=created_count,
                                                    existing_creative_ids=existing_creative_ids,
                                                    ml_system=ml_system,
                                                )
                                                
                                                if success and creative_id and ad_id:
                                                    created_count += 1
                                                    existing_creative_ids.add(creative_id)
                                                    logger.info(f"✅ Successfully created smart creative {creative_id} and ad {ad_id}")
                                                    
                                                    # Look up the recently created creative in storage and mark as active
                                                    if storage_manager:
                                                        try:
                                                            # Get creative_id from creative_data
                                                            storage_creative_id = creative_data.get("creative_id")
                                                            if not storage_creative_id:
                                                                # Try to find it by looking up recently created
                                                                recent_creative = storage_manager.get_recently_created_creative(minutes_back=5)
                                                                if recent_creative:
                                                                    storage_creative_id = recent_creative.get("creative_id")
                                                            
                                                            if storage_creative_id:
                                                                storage_manager.mark_creative_active(storage_creative_id, ad_id)
                                                                logger.info(f"✅ Marked creative {storage_creative_id} as active")
                                                        except Exception as e:
                                                            logger.warning(f"Failed to mark creative as active: {e}")
                                                    
                                                    # Refresh active count and check if we need more
                                                    ads = client.list_ads_in_adset(adset_id)
                                                    active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
                                                    active_count = len(active_ads)
                                                    
                                                    if active_count >= target_count:
                                                        logger.info(f"✅ Reached target: {active_count}/{target_count} active creatives - stopping")
                                                        skip_standard_generation = True
                                                    else:
                                                        logger.info(f"✅ Created 1 ad, but still need {target_count - active_count} more (active: {active_count}, target: {target_count})")
                                                        skip_standard_generation = False  # Allow standard generation to continue
                                                else:
                                                    failed_count += 1
                                                    failed_reasons.append(f"Failed to create (creative_id={creative_id}, ad_id={ad_id})")
                                    except Exception as e:
                                        logger.warning(f"Error analyzing creative: {e}, creating anyway")
                                        # Create anyway if analysis fails
                                        creative_id, ad_id, success = _create_creative_and_ad(
                                            client=client,
                                            image_generator=image_generator,
                                            creative_data=creative_data,
                                            adset_id=adset_id,
                                            active_count=active_count,
                                            created_count=created_count,
                                            existing_creative_ids=existing_creative_ids,
                                            ml_system=ml_system,
                                        )
                                        
                                        if success and creative_id and ad_id:
                                            created_count += 1
                                            existing_creative_ids.add(creative_id)
                                            
                                            # Look up the recently created creative in storage and mark as active
                                            if storage_manager:
                                                try:
                                                    storage_creative_id = creative_data.get("creative_id")
                                                    if not storage_creative_id:
                                                        recent_creative = storage_manager.get_recently_created_creative(minutes_back=5)
                                                        if recent_creative:
                                                            storage_creative_id = recent_creative.get("creative_id")
                                                    
                                                    if storage_creative_id:
                                                        storage_manager.mark_creative_active(storage_creative_id, ad_id)
                                                        logger.info(f"✅ Marked creative {storage_creative_id} as active")
                                                except Exception as e:
                                                    logger.warning(f"Failed to mark creative as active: {e}")
                                            
                                            # HARD STOP: We created 1 ad, STOP immediately
                                            skip_standard_generation = True
                                            logger.info(f"🛑 HARD STOP: Created 1 ad - stopping all further generation")
                                        else:
                                            failed_count += 1
                                            failed_reasons.append(f"Failed to create after analysis error")
                                else:
                                    # No advanced ML - create directly
                                    creative_id, ad_id, success = _create_creative_and_ad(
                                        client=client,
                                        image_generator=image_generator,
                                        creative_data=creative_data,
                                        adset_id=adset_id,
                                        active_count=active_count,
                                        created_count=created_count,
                                        existing_creative_ids=existing_creative_ids,
                                        ml_system=ml_system,
                                    )
                                    
                                    if success and creative_id and ad_id:
                                        created_count += 1
                                        existing_creative_ids.add(creative_id)
                                        
                                        # Look up the recently created creative in storage and mark as active
                                        if storage_manager:
                                            try:
                                                storage_creative_id = creative_data.get("creative_id")
                                                if not storage_creative_id:
                                                    recent_creative = storage_manager.get_recently_created_creative(minutes_back=5)
                                                    if recent_creative:
                                                        storage_creative_id = recent_creative.get("creative_id")
                                                
                                                if storage_creative_id:
                                                    storage_manager.mark_creative_active(storage_creative_id, ad_id)
                                                    logger.info(f"✅ Marked creative {storage_creative_id} as active")
                                            except Exception as e:
                                                logger.warning(f"Failed to mark creative as active: {e}")
                                    else:
                                        failed_count += 1
                                        failed_reasons.append(f"Failed to create")
                        else:
                            logger.warning(f"No creative returned from pipeline")
                            failed_count += 1
                            failed_reasons.append("Pipeline returned empty list")
                    
                    # Final check - refresh active count
                    ads = client.list_ads_in_adset(adset_id)
                    active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
                    active_count = len(active_ads)
                    
                    # Check if we've reached the target - if so, stop
                    # Refresh active_count to get latest status
                    ads = client.list_ads_in_adset(adset_id)
                    active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
                    active_count = len(active_ads)
                    
                    if active_count >= target_count:
                        logger.info(f"✅ Reached target: {active_count}/{target_count} active creatives - STOPPING")
                        skip_standard_generation = True
                        advanced_ml = None
                    elif created_count >= 1 and active_count < target_count:
                        # Generated 1, but still need more - allow standard generation to continue
                        logger.info(f"✅ Generated 1 creative via advanced ML, but still need {target_count - active_count} more (active: {active_count}, target: {target_count})")
                        skip_standard_generation = False  # Allow standard generation to fill the gap
                        advanced_ml = None  # Don't use advanced ML again this tick
                    else:
                        # Fallback to standard generation ONLY if we haven't generated anything yet
                        if active_count < target_count and created_count == 0:
                            logger.warning(f"Advanced pipeline didn't create any creatives, falling back to standard generation")
                            advanced_ml = None
                        skip_standard_generation = False
                except Exception as e:
                    logger.error(f"Advanced pipeline failed with exception, using standard generation: {e}", exc_info=True)
                    advanced_ml = None
            
            # Standard generation (fallback) - Continue generating until target is reached
            if not skip_standard_generation and (not advanced_ml or not advanced_ml.creative_pipeline):
                # Refresh active_count before standard generation
                ads = client.list_ads_in_adset(adset_id)
                active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
                active_count = len(active_ads)
                
                # Only generate if we still need more
                if active_count >= target_count:
                    logger.info(f"✅ Target reached: {active_count}/{target_count} active creatives - skipping standard generation")
                else:
                    # Get ML insights from killed creatives to inform generation
                    ml_insights = None
                    if ml_system and hasattr(ml_system, 'get_creative_insights'):
                        try:
                            ml_insights = ml_system.get_creative_insights()
                            logger.info("✅ Using ML insights from killed creatives to inform new generation")
                        except (AttributeError, ValueError, TypeError) as e:
                            logger.debug(f"Failed to get ML insights: {e}")
                
                # SMART: Only generate 1 creative at a time
                remaining_needed = target_count - active_count
                if remaining_needed > 0:
                    logger.info(f"📸 Standard generation: Generating EXACTLY 1 smart creative using ML insights (need {remaining_needed} more, have {active_count} active)")
                    
                    try:
                        # Generate exactly 1 creative with ML insights
                        creative_data = image_generator.generate_creative(
                            product_info,
                            creative_style=f"smart_creative_{active_count + created_count + 1}",
                        )
                        
                        if not creative_data:
                            failed_count += 1
                            failed_reasons.append("Standard generation: Generation returned None")
                        else:
                            # Create creative in Meta
                            creative_id, ad_id, success = _create_creative_and_ad(
                                client=client,
                                image_generator=image_generator,
                                creative_data=creative_data,
                                adset_id=adset_id,
                                active_count=active_count,
                                created_count=created_count,
                                existing_creative_ids=existing_creative_ids,
                                ml_system=ml_system,
                            )
                            
                            if success and ad_id:
                                created_count += 1
                                existing_creative_ids.add(str(creative_id))
                                logger.info(f"✅ Successfully created smart creative {creative_id} and ad {ad_id}")
                                
                                # Look up the recently created creative in storage and mark as active
                                if storage_manager:
                                    try:
                                        storage_creative_id = creative_data.get("creative_id")
                                        if not storage_creative_id:
                                            recent_creative = storage_manager.get_recently_created_creative(minutes_back=5)
                                            if recent_creative:
                                                storage_creative_id = recent_creative.get("creative_id")
                                        
                                        if storage_creative_id:
                                            storage_manager.mark_creative_active(storage_creative_id, ad_id)
                                            logger.info(f"✅ Marked creative {storage_creative_id} as active")
                                    except Exception as e:
                                        logger.warning(f"Failed to mark creative as active: {e}")
                                
                                # Refresh active count after creation
                                ads = client.list_ads_in_adset(adset_id)
                                active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
                                active_count = len(active_ads)
                                
                                logger.info(f"✅ Created creative - now {active_count}/{target_count} active")
                                
                                # Check if we've reached target - if so, break out of loop
                                if active_count >= target_count:
                                    logger.info(f"✅ Reached target: {active_count}/{target_count} active creatives")
                                    break  # Stop generating
                            else:
                                failed_count += 1
                                if creative_id:
                                    failed_reasons.append(f"Standard generation attempt {attempts}: Duplicate creative ID")
                                else:
                                    failed_reasons.append(f"Standard generation attempt {attempts}: Failed to create")
                                    # Track failure in ML system
                                    if ml_system and hasattr(ml_system, 'record_creative_generation_failure'):
                                        try:
                                            ml_system.record_creative_generation_failure(
                                                reason="Failed to create creative and ad in Meta",
                                                product_info=product_info,
                                            )
                                        except (AttributeError, ValueError, TypeError):
                                            pass
                    except Exception as e:
                        logger.warning(f"Failed to generate creative: {e}")
                        failed_count += 1
                        failed_reasons.append(f"Standard generation: {str(e)[:50]}")
            
            # Final check of active count
            ads = client.list_ads_in_adset(adset_id)
            active_ads = [a for a in ads if str(a.get("status", "")).upper() == "ACTIVE"]
            final_active_count = len(active_ads)
            
            if created_count > 0:
                notify(f"✅ Created {created_count} new creatives for ASC+ campaign (now {final_active_count}/{target_count} active)")
            if failed_count > 0:
                notify(f"⚠️ Failed to create {failed_count} creatives. Reasons: {', '.join(failed_reasons[:3])}")
            
            if final_active_count < target_count:
                notify(f"⚠️ Still need {target_count - final_active_count} more active creatives (currently {final_active_count}/{target_count})")
            elif final_active_count >= target_count:
                notify(f"✅ Target reached: {final_active_count} active creatives")
            
            # Cleanup unused and killed creatives from storage
            try:
                from infrastructure.creative_storage import create_creative_storage_manager
                from infrastructure.supabase_storage import get_validated_supabase_client
                
                supabase_client = get_validated_supabase_client()
                if supabase_client:
                    storage_manager = create_creative_storage_manager(supabase_client)
                    if storage_manager:
                        # Cleanup unused creatives (30 days default)
                        unused_deleted = storage_manager.cleanup_unused_creatives()
                        # Cleanup killed creatives (7 days default)
                        killed_deleted = storage_manager.cleanup_killed_creatives()
                        if unused_deleted > 0 or killed_deleted > 0:
                            logger.info(f"🧹 Cleaned up {unused_deleted} unused and {killed_deleted} killed creatives")
            except Exception as e:
                logger.warning(f"Failed to cleanup creatives from storage: {e}")
            
            # Budget scaling - check if budget should be adjusted
            if OPTIMIZATION_SYSTEMS_AVAILABLE:
                try:
                    from infrastructure.supabase_storage import get_validated_supabase_client
                    supabase_client = get_validated_supabase_client()
                    
                    if supabase_client:
                        # Get performance data for budget scaling
                        performance_data = []
                        for ad_insight in insights:
                            if ad_insight.get("ad_id"):
                                performance_data.append({
                                    "ad_id": ad_insight.get("ad_id"),
                                    "spend": safe_f(ad_insight.get("spend")),
                                    "revenue": safe_f(ad_insight.get("purchase_roas", [{}])[0].get("value", 0)) if ad_insight.get("purchase_roas") else 0,
                                    "purchases": _purchase_and_atc_counts(ad_insight)[0],
                                    "roas": _roas(ad_insight),
                                    "cpa": _cpa(ad_insight),
                                    "date": datetime.now().date().isoformat(),
                                })
                        
                        if len(performance_data) >= 3:
                            budget_engine = create_budget_scaling_engine()
                            current_budget = cfg_or_env_f(cfg(settings, "asc_plus") or {}, "daily_budget_eur", "ASC_PLUS_BUDGET", 50.0)
                            # Ensure current_budget is a float (cfg_or_env_f might return string)
                            try:
                                current_budget = float(current_budget) if current_budget is not None else 50.0
                            except (ValueError, TypeError):
                                current_budget = 50.0
                            
                            decision = budget_engine.get_budget_recommendation(
                                campaign_id=campaign_id,
                                performance_data=performance_data,
                                current_budget=current_budget,
                                strategy=ScalingStrategy.ADAPTIVE,
                                max_budget=ASC_PLUS_BUDGET_MAX,
                                min_budget=ASC_PLUS_BUDGET_MIN,
                            )
                            
                            # Ensure recommended_budget is also a float (handle both dataclass and dict responses)
                            try:
                                if hasattr(decision, 'recommended_budget'):
                                    recommended_budget = float(decision.recommended_budget) if decision.recommended_budget is not None else current_budget
                                elif isinstance(decision, dict):
                                    recommended_budget = float(decision.get('recommended_budget', current_budget))
                                else:
                                    recommended_budget = current_budget
                            except (ValueError, TypeError, AttributeError) as e:
                                logger.warning(f"Failed to convert recommended_budget to float: {e}, using current_budget")
                                recommended_budget = current_budget
                            
                            # Ensure both are floats before comparison
                            try:
                                current_budget_float = float(current_budget)
                                recommended_budget_float = float(recommended_budget)
                            except (ValueError, TypeError):
                                logger.warning(f"Budget values not numeric: current={current_budget}, recommended={recommended_budget}")
                                continue  # Skip this iteration
                            
                            if recommended_budget_float != current_budget_float and decision.confidence > 0.7:
                                budget_change_pct = ((recommended_budget_float - current_budget_float) / current_budget_float) * 100
                                if abs(budget_change_pct) > 10:  # Only adjust if >10% change
                                    logger.info(f"Budget scaling recommendation: €{current_budget_float:.2f} -> €{recommended_budget_float:.2f} ({budget_change_pct:+.1f}%)")
                                    reason = getattr(decision, 'reason', 'performance-based') if hasattr(decision, 'reason') else 'performance-based'
                                    notify(f"💡 Budget scaling: €{current_budget_float:.2f} -> €{recommended_budget_float:.2f} ({reason}, confidence: {decision.confidence:.1%})")
                except Exception as e:
                    logger.warning(f"Budget scaling error: {e}")
            
            # Smart creative refresh - check for creatives that need refresh
            if OPTIMIZATION_SYSTEMS_AVAILABLE:
                try:
                    refresh_manager = create_creative_refresh_manager()
                    creatives_for_refresh = [
                        {
                            "creative_id": ad.get("creative", {}).get("id") if isinstance(ad.get("creative"), dict) else ad.get("id"),
                            "ad_id": ad.get("id"),
                            "created_at": ad.get("created_time"),
                            "performance": next((i for i in insights if i.get("ad_id") == ad.get("id")), {}),
                            "historical_performance": [],  # Would be populated from Supabase
                        }
                        for ad in active_ads
                    ]
                    
                    refresh_schedule = refresh_manager.plan_refresh_schedule(
                        creatives=creatives_for_refresh,
                        target_count=target_count,
                    )
                    
                    if refresh_schedule.get("immediate_refresh") > 0:
                        logger.info(f"Creative refresh needed: {refresh_schedule['immediate_refresh']} immediate, {refresh_schedule.get('staggered_refresh', 0)} scheduled")
                    
                    # Check for scheduled refreshes due now
                    due_refreshes = refresh_manager.get_scheduled_refreshes_due()
                    if due_refreshes:
                        logger.info(f"Executing {len(due_refreshes)} scheduled creative refreshes")
                except Exception as e:
                    logger.warning(f"Creative refresh error: {e}")
            
            # Resource optimization - optimize memory if needed
            if OPTIMIZATION_SYSTEMS_AVAILABLE:
                try:
                    resource_optimizer = create_resource_optimizer()
                    if resource_optimizer.memory_optimizer and resource_optimizer.memory_optimizer.should_optimize_memory():
                        from infrastructure.caching import cache_manager
                        results = resource_optimizer.optimize_all(cache_manager=cache_manager)
                        if results.get("memory", {}).get("freed_mb", 0) > 50:
                            logger.info(f"Memory optimized: freed {results['memory']['freed_mb']:.1f} MB")
                except Exception as e:
                    logger.warning(f"Resource optimization error: {e}")
            
            # Auto-optimization discovery
            if advanced_ml and advanced_ml.auto_optimizer:
                try:
                    creatives_with_perf = [
                        {
                            "ad_id": ad.get("id"),
                            "creative_id": ad.get("creative", {}).get("id") if isinstance(ad.get("creative"), dict) else None,
                            "performance": next(
                                (i for i in insights if i.get("ad_id") == ad.get("id")),
                                {}
                            ),
                        }
                        for ad in active_ads
                    ]
                    
                    opportunities = advanced_ml.auto_optimizer.discover_opportunities(
                        creatives_with_perf,
                        {},
                    )
                    
                    prioritized = advanced_ml.auto_optimizer.prioritize_opportunities(opportunities)
                    
                    # Execute top opportunities
                    for opp in prioritized[:3]:  # Top 3
                        if opp.confidence >= 0.7:
                            advanced_ml.auto_optimizer.execute_optimization(opp, client)
                except Exception as e:
                    logger.error(f"Auto-optimization error: {e}")
        
        return {
            "ok": True,
            "campaign_id": campaign_id,
            "adset_id": adset_id,
            "active_count": final_active_count if 'final_active_count' in locals() else active_count,
            "target_count": target_count,
            "killed_count": killed_count,
            "created_count": created_count,
            "health_status": health_status.value if 'health_status' in locals() else "unknown",
        }
        
    except Exception as e:
        alert_error(f"Error in ASC+ tick: {e}")
        return {"ok": False, "error": str(e)}


__all__ = ["run_asc_plus_tick", "ensure_asc_plus_campaign"]

