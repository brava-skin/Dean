"""
Creative Intelligence System for Dean
Advanced creative management with performance tracking, ML analysis, and AI generation
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import logging

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Import validated Supabase client
try:
    from infrastructure.supabase_storage import get_validated_supabase_client
    VALIDATED_SUPABASE_AVAILABLE = True
except ImportError:
    VALIDATED_SUPABASE_AVAILABLE = False

from infrastructure.data_validation import validate_and_sanitize_data, ValidationError
from config import CREATIVE_PERFORMANCE_STAGE_VALUE


CREATIVE_PERFORMANCE_STAGE_DISABLED = False


class CreativeIntelligenceSystem:
    """Advanced creative intelligence system with ML and AI capabilities."""
    
    def __init__(
        self, 
        supabase_client: Optional[Any] = None, 
        openai_api_key: Optional[str] = None, 
        settings: Optional[Dict[str, Any]] = None
    ) -> None:
        self.supabase_client = supabase_client
        self.openai_api_key = openai_api_key
        self.settings = settings or {}
        self.logger = logging.getLogger(__name__)
        
        # Get configuration from settings
        self.config = self.settings.get('creative_intelligence', {})
    
    def _get_validated_client(self):
        """Get validated Supabase client for automatic data validation."""
        if VALIDATED_SUPABASE_AVAILABLE:
            try:
                return get_validated_supabase_client(enable_validation=True)
            except Exception as e:
                self.logger.warning(f"Failed to get validated client: {e}")
        return self.supabase_client
    
    # Copy bank functions removed - ASC+ uses AI-generated copy (ChatGPT-5) instead
    # The copy_bank.json file is no longer used in the creative generation process
    
    def track_creative_performance(self, ad_id: str, creative_ids: Dict[str, str], 
                                 performance_data: Dict[str, Any], stage: str) -> bool:
        """Track performance of specific creatives used in an ad."""
        try:
            global CREATIVE_PERFORMANCE_STAGE_DISABLED

            if not self.supabase_client:
                return False

            if CREATIVE_PERFORMANCE_STAGE_DISABLED:
                return True
            
            def _float(value: Any, default: float = 0.0) -> float:
                if value in (None, "", [], {}):
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            def _int(value: Any, default: int = 0) -> int:
                try:
                    return int(_float(value, float(default)))
                except (TypeError, ValueError):
                    return default
            
            # Track performance for each creative type
            for creative_type, creative_id in creative_ids.items():
                if not creative_id:
                    continue

                if CREATIVE_PERFORMANCE_STAGE_DISABLED:
                    continue
                
                performance_record = {
                    "creative_id": creative_id,
                    "ad_id": ad_id,
                    "stage": CREATIVE_PERFORMANCE_STAGE_VALUE,
                    "date_start": performance_data.get("date_start", datetime.now().strftime("%Y-%m-%d")),
                    "date_end": performance_data.get("date_end", datetime.now().strftime("%Y-%m-%d")),
                    "impressions": _int(performance_data.get("impressions")),
                    "clicks": _int(performance_data.get("clicks")),
                    "spend": _float(performance_data.get("spend")),
                    "purchases": _int(performance_data.get("purchases")),
                    "add_to_cart": _int(performance_data.get("add_to_cart")),
                    "initiate_checkout": _int(performance_data.get("initiate_checkout")),
                    "ctr": _float(performance_data.get("ctr")),
                    "cpc": _float(performance_data.get("cpc")),
                    "cpm": _float(performance_data.get("cpm")),
                    "roas": _float(performance_data.get("roas")),
                    "cpa": _float(performance_data.get("cpa")),
                    "engagement_rate": _float(performance_data.get("engagement_rate")),
                    "conversion_rate": _float(performance_data.get("conversion_rate")),
                }

                try:
                    sanitized_record = validate_and_sanitize_data("creative_performance", performance_record)
                except ValidationError as val_exc:
                    self.logger.error(
                        "Creative performance validation failed for creative_id=%s ad_id=%s date=%s: %s",
                        creative_id,
                        ad_id,
                        performance_record.get("date_start"),
                        val_exc,
                    )
                    continue
                
                # Get validated client for automatic validation
                validated_client = self._get_validated_client()
                
                try:
                    if validated_client and hasattr(validated_client, 'upsert'):
                        validated_client.upsert(
                            'creative_performance',
                            sanitized_record,
                            on_conflict='creative_id,ad_id,date_start'
                        )
                    else:
                        self.supabase_client.table('creative_performance').upsert(
                            sanitized_record,
                            on_conflict='creative_id,ad_id,date_start'
                        ).execute()
                except Exception as upsert_exc:
                    error_str = str(upsert_exc)
                    if 'creative_performance_stage_check' in error_str:
                        if not CREATIVE_PERFORMANCE_STAGE_DISABLED:
                            self.logger.warning(
                                "Supabase creative_performance stage check rejected records. "
                                "Disabling creative performance tracking until the constraint is fixed. Error: %s",
                                error_str,
                            )
                        CREATIVE_PERFORMANCE_STAGE_DISABLED = True
                        continue
                    if '42P10' in error_str:
                        self.logger.debug(
                            "creative_performance lacks composite constraint; using delete+insert fallback for creative_id=%s ad_id=%s date=%s",
                            creative_id,
                            ad_id,
                            sanitized_record.get('date_start'),
                        )
                        base_client = None
                        if validated_client and hasattr(validated_client, "client"):
                            base_client = validated_client.client
                        elif hasattr(self.supabase_client, "client"):
                            base_client = self.supabase_client.client
                        else:
                            base_client = self.supabase_client

                        try:
                            base_client.table('creative_performance').delete().match(
                                {
                                    'creative_id': creative_id,
                                    'ad_id': ad_id,
                                    'date_start': sanitized_record.get('date_start'),
                                }
                            ).execute()
                        except Exception as delete_exc:
                            self.logger.debug(
                                "Fallback delete failed for creative_performance %s/%s/%s: %s",
                                creative_id,
                                ad_id,
                                sanitized_record.get('date_start'),
                                delete_exc,
                            )
                        try:
                            if validated_client and hasattr(validated_client, 'insert'):
                                validated_client.insert('creative_performance', sanitized_record)
                            else:
                                base_client.table('creative_performance').insert(sanitized_record).execute()
                        except Exception as insert_exc:
                            insert_str = str(insert_exc)
                            if 'creative_performance_stage_check' in insert_str:
                                if not CREATIVE_PERFORMANCE_STAGE_DISABLED:
                                    self.logger.warning(
                                        "Supabase creative_performance stage check rejected records during fallback insert. "
                                        "Disabling creative performance tracking until the constraint is fixed. Error: %s",
                                        insert_str,
                                    )
                                CREATIVE_PERFORMANCE_STAGE_DISABLED = True
                                continue
                            self.logger.error(
                                "Fallback insert failed for creative_performance %s/%s/%s: %s",
                                creative_id,
                                ad_id,
                                sanitized_record.get('date_start'),
                                insert_exc,
                            )
                        else:
                            CREATIVE_PERFORMANCE_STAGE_DISABLED = False
                    else:
                        self.logger.error(f"Failed to track creative performance: {upsert_exc}")
                else:
                    CREATIVE_PERFORMANCE_STAGE_DISABLED = False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to track creative performance: {e}")
            return False


def create_creative_intelligence_system(supabase_client=None, openai_api_key=None, settings=None) -> CreativeIntelligenceSystem:
    """Create and initialize the creative intelligence system."""
    return CreativeIntelligenceSystem(supabase_client, openai_api_key, settings)
