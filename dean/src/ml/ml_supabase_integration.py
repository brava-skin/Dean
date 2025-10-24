"""
ML System Supabase Integration

This module integrates the enhanced ML system with Supabase historical data tables,
providing seamless access to historical data for improved ML predictions and analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from supabase import Client

from infrastructure.utils import now_utc
from ml.ml_historical_enhancements import HistoricalDataAnalyzer, EnhancedMLPredictor


class SupabaseMLIntegration:
    """Integration layer between ML system and Supabase historical data."""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.historical_analyzer = HistoricalDataAnalyzer(supabase_client)
        self.enhanced_predictor = EnhancedMLPredictor(supabase_client, self.historical_analyzer)
        self.logger = logging.getLogger(f"{__name__}.SupabaseMLIntegration")
    
    def store_historical_metrics(self, ad_id: str, lifecycle_id: str, stage: str, 
                                metrics: Dict[str, float], timestamp: Optional[datetime] = None) -> bool:
        """Store historical metrics in Supabase for ML analysis."""
        try:
            ts = timestamp or now_utc()
            ts_epoch = int(ts.timestamp())
            ts_iso = ts.isoformat()
            
            # Prepare data for batch insert
            historical_data = []
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    historical_data.append({
                        'ad_id': ad_id,
                        'lifecycle_id': lifecycle_id,
                        'stage': stage,
                        'metric_name': metric_name,
                        'metric_value': float(metric_value),
                        'ts_epoch': ts_epoch,
                        'ts_iso': ts_iso
                    })
            
            # Batch insert historical data
            if historical_data:
                response = self.client.table('historical_data').insert(historical_data).execute()
                self.logger.info(f"Stored {len(historical_data)} historical metrics for ad {ad_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error storing historical metrics: {e}")
            return False
    
    def store_ad_creation_time(self, ad_id: str, lifecycle_id: str, stage: str, 
                              created_at: Optional[datetime] = None) -> bool:
        """Store ad creation time for time-based analysis."""
        try:
            ts = created_at or now_utc()
            ts_epoch = int(ts.timestamp())
            ts_iso = ts.isoformat()
            
            creation_data = {
                'ad_id': ad_id,
                'lifecycle_id': lifecycle_id,
                'stage': stage,
                'created_at_epoch': ts_epoch,
                'created_at_iso': ts_iso
            }
            
            # Upsert to handle updates
            response = self.client.table('ad_creation_times').upsert(
                creation_data, on_conflict='ad_id'
            ).execute()
            
            self.logger.info(f"Stored creation time for ad {ad_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing ad creation time: {e}")
            return False
    
    def get_enhanced_performance_data(self, ad_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Get enhanced performance data with historical context."""
        try:
            # Get current performance data
            current_response = self.client.table('performance_metrics').select('*').eq(
                'ad_id', ad_id
            ).order('created_at', desc=True).limit(1).execute()
            
            current_data = current_response.data[0] if current_response.data else {}
            
            # Get historical trends
            trends = self.historical_analyzer.analyze_performance_trends(ad_id, days_back)
            
            # Get ad age features
            age_features = self.historical_analyzer.get_ad_age_features(ad_id)
            
            # Get anomaly detection
            anomalies = self.historical_analyzer.detect_performance_anomalies(ad_id, days_back=7)
            
            return {
                'current_performance': current_data,
                'historical_trends': trends,
                'age_features': age_features,
                'anomalies': anomalies,
                'enhanced_features': {
                    **current_data,
                    **age_features,
                    'trends': trends,
                    'anomalies': anomalies
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced performance data: {e}")
            return {}
    
    def get_creative_insights(self, creative_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive creative insights using historical data."""
        try:
            # Get creative performance history
            creative_history = self.historical_analyzer.get_creative_performance_history(creative_id, days_back)
            
            # Get all ads using this creative
            ads_response = self.client.table('ad_lifecycle').select('ad_id, stage').eq(
                'creative_id', creative_id
            ).execute()
            
            ads_data = ads_response.data if ads_response.data else []
            
            # Analyze performance across stages
            stage_performance = {}
            for ad_data in ads_data:
                ad_id = ad_data['ad_id']
                stage = ad_data['stage']
                
                enhanced_data = self.get_enhanced_performance_data(ad_id, days_back)
                if enhanced_data:
                    if stage not in stage_performance:
                        stage_performance[stage] = []
                    stage_performance[stage].append(enhanced_data)
            
            return {
                'creative_id': creative_id,
                'creative_history': creative_history,
                'stage_performance': stage_performance,
                'total_ads': len(ads_data),
                'analysis_period_days': days_back
            }
            
        except Exception as e:
            self.logger.error(f"Error getting creative insights: {e}")
            return {}
    
    def predict_ad_performance(self, ad_id: str, stage: str) -> Dict[str, Any]:
        """Make enhanced performance predictions with historical context."""
        try:
            prediction = self.enhanced_predictor.predict_with_historical_context(ad_id, stage)
            
            # Store prediction in Supabase
            if prediction.get('status') == 'success':
                self._store_prediction(ad_id, stage, prediction)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting ad performance: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _store_prediction(self, ad_id: str, stage: str, prediction: Dict[str, Any]) -> None:
        """Store enhanced prediction in Supabase."""
        try:
            prediction_data = {
                'ad_id': ad_id,
                'stage': stage,
                'prediction_type': 'enhanced_historical',
                'prediction_data': prediction['prediction'],
                'confidence_score': prediction.get('confidence', 0.0),
                'features_used': prediction.get('features', {}),
                'created_at': now_utc().isoformat()
            }
            
            response = self.client.table('ml_predictions').insert(prediction_data).execute()
            self.logger.info(f"Stored enhanced prediction for ad {ad_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing prediction: {e}")
    
    def get_ml_insights_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get comprehensive ML insights summary."""
        try:
            # Get recent predictions
            start_date = (now_utc() - timedelta(days=days_back)).isoformat()
            
            predictions_response = self.client.table('ml_predictions').select('*').gte(
                'created_at', start_date
            ).execute()
            
            predictions = predictions_response.data if predictions_response.data else []
            
            # Get historical data usage
            historical_response = self.client.table('historical_data').select('*').gte(
                'ts_iso', start_date
            ).execute()
            
            historical_data = historical_response.data if historical_response.data else []
            
            # Get ad creation data
            creation_response = self.client.table('ad_creation_times').select('*').gte(
                'created_at_iso', start_date
            ).execute()
            
            creation_data = creation_response.data if creation_response.data else []
            
            # Calculate insights
            insights = {
                'predictions_made': len(predictions),
                'historical_data_points': len(historical_data),
                'ads_created': len(creation_data),
                'avg_confidence': 0.0,
                'prediction_accuracy': 0.0,  # Would need to calculate based on actual vs predicted
                'data_quality_score': self._calculate_data_quality_score(historical_data),
                'trending_metrics': self._get_trending_metrics(historical_data),
                'anomalies_detected': self._count_anomalies(historical_data)
            }
            
            # Calculate average confidence
            if predictions:
                confidences = [p.get('confidence_score', 0) for p in predictions if p.get('confidence_score')]
                insights['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting ML insights summary: {e}")
            return {}
    
    def _calculate_data_quality_score(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate data quality score based on historical data."""
        if not historical_data:
            return 0.0
        
        # Simple data quality metrics
        total_points = len(historical_data)
        
        # Check for completeness (no null values)
        complete_points = sum(1 for d in historical_data if d.get('metric_value') is not None)
        
        # Check for recency (data from last 24 hours)
        recent_threshold = (now_utc() - timedelta(hours=24)).isoformat()
        recent_points = sum(1 for d in historical_data if d.get('ts_iso', '') >= recent_threshold)
        
        # Calculate quality score
        completeness_score = complete_points / total_points if total_points > 0 else 0
        recency_score = recent_points / total_points if total_points > 0 else 0
        
        return (completeness_score + recency_score) / 2
    
    def _get_trending_metrics(self, historical_data: List[Dict[str, Any]]) -> List[str]:
        """Get trending metrics based on data volume."""
        if not historical_data:
            return []
        
        # Count metric occurrences
        metric_counts = {}
        for d in historical_data:
            metric_name = d.get('metric_name', '')
            if metric_name:
                metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
        
        # Sort by count and return top metrics
        sorted_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)
        return [metric for metric, count in sorted_metrics[:5]]
    
    def _count_anomalies(self, historical_data: List[Dict[str, Any]]) -> int:
        """Count potential anomalies in historical data."""
        if not historical_data:
            return 0
        
        # Simple anomaly detection based on extreme values
        anomaly_count = 0
        metric_groups = {}
        
        # Group by metric
        for d in historical_data:
            metric_name = d.get('metric_name', '')
            metric_value = d.get('metric_value', 0)
            
            if metric_name and metric_value is not None:
                if metric_name not in metric_groups:
                    metric_groups[metric_name] = []
                metric_groups[metric_name].append(metric_value)
        
        # Check for anomalies in each metric group
        for metric_name, values in metric_groups.items():
            if len(values) >= 3:
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                
                if std_val > 0:
                    for value in values:
                        z_score = abs(value - mean_val) / std_val
                        if z_score > 2.5:  # Anomaly threshold
                            anomaly_count += 1
        
        return anomaly_count


def create_supabase_ml_integration(supabase_client: Client) -> SupabaseMLIntegration:
    """Create Supabase ML integration."""
    return SupabaseMLIntegration(supabase_client)
