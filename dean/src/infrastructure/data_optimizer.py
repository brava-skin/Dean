"""
Data Optimizer for Creative Tables
Ensures creative_intelligence and creative_storage tables
always have correct, complete data
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CreativeIntelligenceOptimizer:
    """Optimizes creative_intelligence table data."""
    
    def __init__(self, supabase_client: Any) -> None:
        self.client = supabase_client
    
    def calculate_performance_metrics(
        self,
        creative_id: str,
        ad_id: str,
        days_back: int = 30,
    ) -> Dict[str, float]:
        """Calculate avg_ctr, avg_cpa, avg_roas from performance_metrics."""
        try:
            # Get performance data for this creative/ad
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Try to get data by ad_id first
            response = self.client.table('performance_metrics').select(
                'ctr, cpa, roas, spend, purchases, impressions, clicks'
            ).eq('ad_id', ad_id).gte('date_start', start_date).execute()
            
            # If no data by ad_id, try to find ads with this creative_id
            if not response.data:
                # Get ad_ids that use this creative from ads table
                try:
                    creative_response = self.client.table('ads').select(
                        'ad_id'
                    ).eq('creative_id', creative_id).execute()
                    
                    if creative_response.data:
                        ad_ids = [row.get('ad_id') for row in creative_response.data if row.get('ad_id')]
                        if ad_ids:
                            response = self.client.table('performance_metrics').select(
                                'ctr, cpa, roas, spend, purchases, impressions, clicks'
                            ).in_('ad_id', ad_ids).gte('date_start', start_date).execute()
                except Exception:
                    pass  # Fall through to defaults
            
            if not response.data:
                # No performance data yet - return defaults
                return {
                    'avg_ctr': 0.0,
                    'avg_cpa': 0.0,
                    'avg_roas': 0.0,
                    'performance_rank': 999,
                    'performance_score': 0.0,
                    'fatigue_index': 0.0,
                }
            
            df = pd.DataFrame(response.data)
            
            # Calculate averages (weighted by spend if available)
            if 'spend' in df.columns and df['spend'].sum() > 0:
                # Weighted averages
                total_spend = df['spend'].sum()
                avg_ctr = (df['ctr'] * df['spend']).sum() / total_spend if total_spend > 0 else df['ctr'].mean()
                avg_cpa = (df['cpa'] * df['spend']).sum() / total_spend if total_spend > 0 else df['cpa'].mean()
                avg_roas = (df['roas'] * df['spend']).sum() / total_spend if total_spend > 0 else df['roas'].mean()
            else:
                # Simple averages
                avg_ctr = df['ctr'].mean() if 'ctr' in df.columns else 0.0
                avg_cpa = df['cpa'].mean() if 'cpa' in df.columns else 0.0
                avg_roas = df['roas'].mean() if 'roas' in df.columns else 0.0
            
            # Calculate performance score (0-1)
            performance_score = self._calculate_performance_score(avg_ctr, avg_cpa, avg_roas)
            
            # Calculate fatigue index (0-1, higher = more fatigued)
            fatigue_index = self._calculate_fatigue_index(df)
            
            # Calculate performance rank (will be updated in batch)
            performance_rank = 999  # Placeholder, will be calculated in batch
            
            return {
                'avg_ctr': float(avg_ctr) if not np.isnan(avg_ctr) else 0.0,
                'avg_cpa': float(avg_cpa) if not np.isnan(avg_cpa) else 0.0,
                'avg_roas': float(avg_roas) if not np.isnan(avg_roas) else 0.0,
                'performance_rank': performance_rank,
                'performance_score': float(performance_score),
                'fatigue_index': float(fatigue_index),
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics for {creative_id}: {e}")
            return {
                'avg_ctr': 0.0,
                'avg_cpa': 0.0,
                'avg_roas': 0.0,
                'performance_rank': 999,
                'performance_score': 0.0,
                'fatigue_index': 0.0,
            }
    
    def _calculate_performance_score(
        self,
        avg_ctr: float,
        avg_cpa: float,
        avg_roas: float,
    ) -> float:
        """Calculate performance score (0-1)."""
        # Normalize metrics
        ctr_score = min(1.0, avg_ctr / 0.02)  # 2% CTR = perfect
        cpa_score = min(1.0, max(0.0, (50.0 - avg_cpa) / 50.0))  # Lower CPA is better
        roas_score = min(1.0, avg_roas / 3.0)  # 3.0 ROAS = perfect
        
        # Weighted combination
        performance_score = (
            ctr_score * 0.3 +
            cpa_score * 0.3 +
            roas_score * 0.4
        )
        
        return max(0.0, min(1.0, performance_score))
    
    def _calculate_fatigue_index(self, df: pd.DataFrame) -> float:
        """Calculate fatigue index based on performance trend."""
        if len(df) < 3:
            return 0.0
        
        # Check if performance is declining
        if 'roas' in df.columns and len(df) >= 3:
            recent_roas = df['roas'].tail(3).mean()
            older_roas = df['roas'].head(len(df) - 3).mean() if len(df) > 3 else recent_roas
            
            if older_roas > 0:
                decline_pct = (older_roas - recent_roas) / older_roas
                fatigue_index = max(0.0, min(1.0, decline_pct))
                return fatigue_index
        
        return 0.0
    
    def update_creative_performance(
        self,
        creative_id: str,
        ad_id: str,
        force_recalculate: bool = False,
    ) -> bool:
        """Update creative performance metrics in ads table."""
        try:
            # Check if metrics need updating
            if not force_recalculate:
                existing = self.client.table('ads').select(
                    'performance_score, fatigue_index, updated_at'
                ).eq('ad_id', ad_id).execute()
                
                if existing.data:
                    last_update = existing.data[0].get('updated_at')
                    if last_update:
                        try:
                            last_update_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                            hours_since_update = (datetime.now(timezone.utc) - last_update_dt).total_seconds() / 3600
                            # Only update if older than 6 hours
                            if hours_since_update < 6:
                                return True
                        except Exception:
                            pass
            
            # Calculate metrics
            metrics = self.calculate_performance_metrics(creative_id, ad_id)
            
            # Update ads table
            update_data = {
                'performance_score': metrics['performance_score'],
                'fatigue_index': metrics['fatigue_index'],
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            
            self.client.table('ads').update(update_data).eq(
                'ad_id', ad_id
            ).execute()
            
            logger.info(f"✅ Updated performance metrics for ad {ad_id} (creative {creative_id})")
            return True
        except Exception as e:
            logger.error(f"Error updating creative performance for {creative_id}: {e}")
            return False
    
    def calculate_performance_ranks(self, stage: str = 'asc_plus') -> bool:
        """Calculate and update performance ranks for all ads - not applicable with new schema."""
        # Performance ranks are no longer stored in the consolidated schema
        # This method is kept for backward compatibility but does nothing
        logger.debug("Performance ranks calculation skipped - not applicable with consolidated schema")
        return True
    
    def backfill_missing_metrics(self, stage: str = 'asc_plus') -> Dict[str, int]:
        """Backfill missing performance metrics for all ads."""
        try:
            # Get all ads (with or without metrics)
            response = self.client.table('ads').select(
                'ad_id, creative_id, performance_score, fatigue_index'
            ).limit(1000).execute()  # Limit to avoid timeout
            
            if not response.data:
                return {'updated': 0, 'skipped': 0, 'errors': 0}
            
            updated = 0
            skipped = 0
            errors = 0
            
            for creative in response.data:
                creative_id = creative.get('creative_id')
                ad_id = creative.get('ad_id')
                
                if not creative_id or not ad_id:
                    skipped += 1
                    continue
                
                # Check if metrics are missing or need recalculation
                avg_ctr = creative.get('avg_ctr')
                avg_cpa = creative.get('avg_cpa')
                avg_roas = creative.get('avg_roas')
                
                # Always update if any metric is missing, null, or all are zero (likely missing data)
                needs_update = (
                    avg_ctr is None or 
                    avg_cpa is None or 
                    avg_roas is None or
                    (avg_ctr == 0.0 and avg_cpa == 0.0 and avg_roas == 0.0)  # All zeros = likely missing data
                )
                
                if needs_update:
                    if self.update_creative_performance(creative_id, ad_id, force_recalculate=True):
                        updated += 1
                    else:
                        errors += 1
                else:
                    skipped += 1
            
            logger.info(f"✅ Backfilled metrics: {updated} updated, {skipped} skipped, {errors} errors")
            return {'updated': updated, 'skipped': skipped, 'errors': errors}
        except Exception as e:
            logger.error(f"Error backfilling metrics: {e}")
            return {'updated': 0, 'skipped': 0, 'errors': 1}


__all__ = [
    "CreativeIntelligenceOptimizer",
]

