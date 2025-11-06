"""
Data Optimizer for ML Tables
Ensures creative_intelligence, creative_storage, and learning_events tables
always have correct, complete data for ML system
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
    
    def __init__(self, supabase_client):
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
            if not response.data or len(response.data) == 0:
                # Get ad_ids that use this creative from creative_intelligence
                try:
                    creative_response = self.client.table('creative_intelligence').select(
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
            
            if not response.data or len(response.data) == 0:
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
        """Update creative performance metrics in creative_intelligence table."""
        try:
            # Check if metrics need updating
            if not force_recalculate:
                existing = self.client.table('creative_intelligence').select(
                    'avg_ctr, avg_cpa, avg_roas, updated_at'
                ).eq('creative_id', creative_id).execute()
                
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
            
            # Update creative_intelligence table
            update_data = {
                'avg_ctr': metrics['avg_ctr'],
                'avg_cpa': metrics['avg_cpa'],
                'avg_roas': metrics['avg_roas'],
                'performance_score': metrics['performance_score'],
                'fatigue_index': metrics['fatigue_index'],
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            
            self.client.table('creative_intelligence').update(update_data).eq(
                'creative_id', creative_id
            ).execute()
            
            logger.info(f"✅ Updated performance metrics for creative {creative_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating creative performance for {creative_id}: {e}")
            return False
    
    def calculate_performance_ranks(self, stage: str = 'asc_plus') -> bool:
        """Calculate and update performance ranks for all creatives in a stage."""
        try:
            # Get all creatives with performance scores
            response = self.client.table('creative_intelligence').select(
                'creative_id, performance_score, avg_roas, avg_ctr'
            ).eq('stage', stage).execute()
            
            if not response.data:
                return False
            
            df = pd.DataFrame(response.data)
            
            # Calculate ranks based on performance score
            df['performance_rank'] = df['performance_score'].rank(ascending=False, method='min').astype(int)
            
            # Update ranks in batches
            for _, row in df.iterrows():
                self.client.table('creative_intelligence').update({
                    'performance_rank': int(row['performance_rank']),
                }).eq('creative_id', row['creative_id']).execute()
            
            logger.info(f"✅ Updated performance ranks for {len(df)} creatives")
            return True
        except Exception as e:
            logger.error(f"Error calculating performance ranks: {e}")
            return False
    
    def backfill_missing_metrics(self, stage: str = 'asc_plus') -> Dict[str, int]:
        """Backfill missing performance metrics for all creatives."""
        try:
            # Get all creatives (with or without metrics)
            response = self.client.table('creative_intelligence').select(
                'creative_id, ad_id, avg_ctr, avg_cpa, avg_roas'
            ).eq('stage', stage).limit(1000).execute()  # Limit to avoid timeout
            
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


class CreativeStorageOptimizer:
    """Optimizes creative_storage table data."""
    
    def __init__(self, supabase_client):
        self.client = supabase_client
    
    def ensure_complete_metadata(
        self,
        creative_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ensure metadata has all required fields."""
        try:
            # Get existing metadata
            response = self.client.table('creative_storage').select('metadata').eq(
                'creative_id', creative_id
            ).execute()
            
            existing_metadata = {}
            if response.data and response.data[0].get('metadata'):
                existing_metadata = response.data[0]['metadata']
            
            # Merge with provided metadata
            complete_metadata = {**existing_metadata, **(metadata or {})}
            
            # Ensure all required fields
            required_fields = {
                'format': 'static_image',
                'style': '',
                'message_type': '',
                'target_motivation': '',
            }
            
            for field, default in required_fields.items():
                if field not in complete_metadata:
                    complete_metadata[field] = default
            
            # Update if needed
            if complete_metadata != existing_metadata:
                self.client.table('creative_storage').update({
                    'metadata': complete_metadata,
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                }).eq('creative_id', creative_id).execute()
            
            return complete_metadata
        except Exception as e:
            logger.error(f"Error ensuring complete metadata for {creative_id}: {e}")
            return metadata or {}
    
    def validate_storage_data(self, creative_id: str) -> Tuple[bool, List[str]]:
        """Validate creative_storage data completeness."""
        issues = []
        
        try:
            response = self.client.table('creative_storage').select('*').eq(
                'creative_id', creative_id
            ).execute()
            
            if not response.data:
                issues.append("Creative not found in storage")
                return False, issues
            
            data = response.data[0]
            
            # Check required fields
            required_fields = ['creative_id', 'storage_path', 'storage_url', 'file_type', 'status']
            for field in required_fields:
                if not data.get(field):
                    issues.append(f"Missing required field: {field}")
            
            # Check metadata
            metadata = data.get('metadata', {})
            required_metadata = ['format', 'style', 'message_type', 'target_motivation']
            for field in required_metadata:
                if field not in metadata:
                    issues.append(f"Missing metadata field: {field}")
            
            # Check status validity
            valid_statuses = ['queue', 'active', 'killed']
            if data.get('status') not in valid_statuses:
                issues.append(f"Invalid status: {data.get('status')}")
            
            return len(issues) == 0, issues
        except Exception as e:
            logger.error(f"Error validating storage data for {creative_id}: {e}")
            issues.append(f"Validation error: {e}")
            return False, issues
    
    def fix_storage_data(self, creative_id: str) -> bool:
        """Fix any issues in creative_storage data."""
        try:
            is_valid, issues = self.validate_storage_data(creative_id)
            
            # Always ensure metadata is complete (even if validation passes)
            self.ensure_complete_metadata(creative_id)
            
            if is_valid:
                return True
            
            # Fix status issues
            response = self.client.table('creative_storage').select('status').eq(
                'creative_id', creative_id
            ).execute()
            
            if response.data:
                status = response.data[0].get('status')
                if status not in ['queue', 'active', 'killed']:
                    # Set to queue if invalid
                    self.client.table('creative_storage').update({
                        'status': 'queue',
                    }).eq('creative_id', creative_id).execute()
            
            logger.debug(f"✅ Fixed storage data for {creative_id}")
            return True
        except Exception as e:
            logger.debug(f"Error fixing storage data for {creative_id}: {e}")
            return False


class LearningEventsOptimizer:
    """Optimizes learning_events table data."""
    
    def __init__(self, supabase_client):
        self.client = supabase_client
    
    def ensure_complete_event(
        self,
        event_id: str,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ensure learning event has all required fields."""
        try:
            # Get existing event
            response = self.client.table('learning_events').select('*').eq('id', event_id).execute()
            
            existing_data = {}
            if response.data:
                existing_data = response.data[0]
            
            # Merge with provided data
            complete_data = {**existing_data, **(event_data or {})}
            
            # Ensure required fields
            required_fields = {
                'event_type': 'unknown',
                'ad_id': '',
                'lifecycle_id': '',
                'stage': 'asc_plus',
                'confidence_score': 0.5,
                'impact_score': 0.5,
                'learning_data': {},
                'event_data': {},
            }
            
            for field, default in required_fields.items():
                if field not in complete_data or complete_data[field] is None:
                    complete_data[field] = default
            
            # Validate event_type
            valid_event_types = [
                'creative_created',
                'creative_generation_failed',
                'ad_killed',
                'ad_promoted',
                'budget_scaled',
                'rule_adapted',
                'model_trained',
                'stage_transition',
            ]
            
            if complete_data.get('event_type') not in valid_event_types:
                complete_data['event_type'] = 'unknown'
            
            # Update if needed (only update fields that can be updated, not read-only ones like id, created_at)
            if complete_data != existing_data:
                update_fields = {
                    'confidence_score': complete_data.get('confidence_score', 0.5),
                    'impact_score': complete_data.get('impact_score', 0.5),
                    'learning_data': complete_data.get('learning_data', {}),
                    'event_data': complete_data.get('event_data', {}),
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                }
                # Only update if values actually changed
                if (update_fields['confidence_score'] != existing_data.get('confidence_score') or
                    update_fields['impact_score'] != existing_data.get('impact_score')):
                    self.client.table('learning_events').update(update_fields).eq('id', event_id).execute()
            
            return complete_data
        except Exception as e:
            logger.error(f"Error ensuring complete event for {event_id}: {e}")
            return event_data or {}
    
    def calculate_confidence_score(
        self,
        event_type: str,
        learning_data: Dict[str, Any],
    ) -> float:
        """Calculate confidence score for learning event."""
        try:
            base_confidence = 0.5
            
            # Adjust based on event type
            if event_type == 'creative_created':
                base_confidence = 0.7
            elif event_type == 'ad_killed':
                base_confidence = 0.8
            elif event_type == 'model_trained':
                base_confidence = 0.9
            
            # Adjust based on learning data
            if 'learning_confidence' in learning_data:
                base_confidence = learning_data['learning_confidence']
            elif 'success_metrics' in learning_data:
                base_confidence = max(base_confidence, learning_data['success_metrics'])
            
            return max(0.0, min(1.0, base_confidence))
        except Exception:
            return 0.5
    
    def calculate_impact_score(
        self,
        event_type: str,
        event_data: Dict[str, Any],
    ) -> float:
        """Calculate impact score for learning event."""
        try:
            base_impact = 0.5
            
            # Adjust based on event type
            if event_type in ['ad_killed', 'ad_promoted', 'budget_scaled']:
                base_impact = 0.7
            elif event_type == 'model_trained':
                base_impact = 0.8
            
            # Adjust based on event data
            if 'impact_score' in event_data:
                base_impact = event_data['impact_score']
            elif 'event_impact' in event_data:
                base_impact = event_data['event_impact']
            
            return max(0.0, min(1.0, base_impact))
        except Exception:
            return 0.5
    
    def backfill_event_scores(self, days_back: int = 30) -> Dict[str, int]:
        """Backfill confidence and impact scores for learning events."""
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            response = self.client.table('learning_events').select(
                'id, event_type, learning_data, event_data, confidence_score, impact_score'
            ).gte('created_at', start_date).execute()
            
            if not response.data:
                return {'updated': 0, 'skipped': 0, 'errors': 0}
            
            updated = 0
            skipped = 0
            errors = 0
            
            for event in response.data:
                event_id = event.get('id')
                event_type = event.get('event_type', 'unknown')
                learning_data = event.get('learning_data', {})
                event_data = event.get('event_data', {})
                
                # Calculate scores
                confidence = self.calculate_confidence_score(event_type, learning_data)
                impact = self.calculate_impact_score(event_type, event_data)
                
                # Check if update needed
                current_confidence = event.get('confidence_score', 0.0)
                current_impact = event.get('impact_score', 0.0)
                
                if abs(confidence - current_confidence) > 0.01 or abs(impact - current_impact) > 0.01:
                    try:
                        self.client.table('learning_events').update({
                            'confidence_score': confidence,
                            'impact_score': impact,
                            'updated_at': datetime.now(timezone.utc).isoformat(),
                        }).eq('id', event_id).execute()
                        updated += 1
                    except Exception:
                        errors += 1
                else:
                    skipped += 1
            
            logger.info(f"✅ Backfilled event scores: {updated} updated, {skipped} skipped, {errors} errors")
            return {'updated': updated, 'skipped': skipped, 'errors': errors}
        except Exception as e:
            logger.error(f"Error backfilling event scores: {e}")
            return {'updated': 0, 'skipped': 0, 'errors': 1}


class MLDataOptimizer:
    """Main optimizer for all ML-related tables."""
    
    def __init__(self, supabase_client):
        self.client = supabase_client
        self.creative_intel_optimizer = CreativeIntelligenceOptimizer(supabase_client)
        self.creative_storage_optimizer = CreativeStorageOptimizer(supabase_client)
        self.learning_events_optimizer = LearningEventsOptimizer(supabase_client)
    
    def optimize_all_tables(
        self,
        stage: str = 'asc_plus',
        force_recalculate: bool = False,
    ) -> Dict[str, Any]:
        """Optimize all ML tables."""
        results = {
            'creative_intelligence': {},
            'creative_storage': {},
            'learning_events': {},
        }
        
        try:
            # Optimize creative_intelligence
            logger.info("Optimizing creative_intelligence table...")
            backfill_results = self.creative_intel_optimizer.backfill_missing_metrics(stage)
            results['creative_intelligence'] = backfill_results
            
            # Calculate performance ranks
            self.creative_intel_optimizer.calculate_performance_ranks(stage)
            
            # Optimize learning_events
            logger.info("Optimizing learning_events table...")
            event_results = self.learning_events_optimizer.backfill_event_scores()
            results['learning_events'] = event_results
            
            # Validate and auto-fix creative_storage
            logger.info("Validating and fixing creative_storage table...")
            storage_issues = self._validate_all_storage()
            fixed_count = 0
            # Auto-fix missing metadata fields
            for issue in storage_issues:
                if 'Missing metadata field' in issue:
                    # Extract creative_id from issue string
                    creative_id = issue.split(':')[0] if ':' in issue else None
                    if creative_id:
                        try:
                            if self.creative_storage_optimizer.fix_storage_data(creative_id):
                                fixed_count += 1
                        except Exception:
                            pass
            results['creative_storage'] = {
                'validated': len(storage_issues) == 0,
                'issues_found': len(storage_issues),
                'fixed': fixed_count,
                'issues': storage_issues[:10],  # Limit to first 10
            }
            
            logger.info("✅ All ML tables optimized")
            return results
        except Exception as e:
            logger.error(f"Error optimizing ML tables: {e}")
            return results
    
    def _validate_all_storage(self) -> List[str]:
        """Validate all creative_storage entries."""
        issues = []
        try:
            response = self.client.table('creative_storage').select('creative_id').execute()
            
            for creative in response.data[:100]:  # Limit to first 100
                creative_id = creative.get('creative_id')
                if creative_id:
                    is_valid, creative_issues = self.creative_storage_optimizer.validate_storage_data(creative_id)
                    if not is_valid:
                        issues.extend([f"{creative_id}: {issue}" for issue in creative_issues])
        except Exception as e:
            logger.error(f"Error validating storage: {e}")
            issues.append(f"Validation error: {e}")
        
        return issues
    
    def ensure_creative_data_completeness(
        self,
        creative_id: str,
        ad_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Ensure creative has complete data in all tables."""
        try:
            # Update creative_intelligence performance metrics
            self.creative_intel_optimizer.update_creative_performance(creative_id, ad_id)
            
            # Ensure creative_storage metadata is complete
            self.creative_storage_optimizer.ensure_complete_metadata(creative_id, metadata)
            
            logger.info(f"✅ Ensured data completeness for creative {creative_id}")
            return True
        except Exception as e:
            logger.error(f"Error ensuring data completeness for {creative_id}: {e}")
            return False


def create_ml_data_optimizer(supabase_client) -> MLDataOptimizer:
    """Create ML data optimizer."""
    return MLDataOptimizer(supabase_client)


__all__ = [
    "MLDataOptimizer",
    "CreativeIntelligenceOptimizer",
    "CreativeStorageOptimizer",
    "LearningEventsOptimizer",
    "create_ml_data_optimizer",
]

