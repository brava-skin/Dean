"""
Enhanced ML System with Historical Data Integration

This module provides enhanced ML capabilities using the new Supabase historical data tables:
- historical_data: For comprehensive time-series analysis
- ad_creation_times: For age-based features and time-based predictions
- Enhanced trend analysis and pattern recognition
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from supabase import Client

from infrastructure.utils import now_utc


class HistoricalDataAnalyzer:
    """Enhanced analyzer using Supabase historical data tables."""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.logger = logging.getLogger(f"{__name__}.HistoricalDataAnalyzer")
    
    def get_historical_metrics(self, ad_id: str, metric_names: List[str], 
                             days_back: int = 30) -> pd.DataFrame:
        """Get comprehensive historical metrics for an ad."""
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            # Query historical_data table
            response = self.client.table('historical_data').select('*').eq(
                'ad_id', ad_id
            ).in_('metric_name', metric_names).gte('ts_iso', start_date).order(
                'ts_epoch', desc=False
            ).execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df['ts_iso'] = pd.to_datetime(df['ts_iso'])
            
            # Pivot to get metrics as columns
            pivot_df = df.pivot_table(
                index=['ts_iso', 'ad_id', 'stage'], 
                columns='metric_name', 
                values='metric_value', 
                aggfunc='mean'
            ).reset_index()
            
            # Fill missing values
            pivot_df = pivot_df.fillna(0)
            
            return pivot_df
            
        except Exception as e:
            self.logger.error(f"Error getting historical metrics: {e}")
            return pd.DataFrame()
    
    def get_ad_age_features(self, ad_id: str) -> Dict[str, Any]:
        """Get age-based features for an ad."""
        try:
            # Get ad creation time
            response = self.client.table('ad_creation_times').select('*').eq(
                'ad_id', ad_id
            ).execute()
            
            if not response.data:
                return {}
            
            creation_data = response.data[0]
            created_at = datetime.fromisoformat(creation_data['created_at_iso'].replace('Z', '+00:00'))
            
            # Calculate age in different units
            now = now_utc()
            age_delta = now - created_at
            
            age_features = {
                'ad_age_days': age_delta.total_seconds() / (24 * 3600),
                'ad_age_hours': age_delta.total_seconds() / 3600,
                'ad_age_weeks': age_delta.days / 7,
                'created_at_epoch': creation_data['created_at_epoch'],
                'stage': creation_data['stage']
            }
            
            return age_features
            
        except Exception as e:
            self.logger.error(f"Error getting ad age features: {e}")
            return {}
    
    def analyze_performance_trends(self, ad_id: str, days_back: int = 14) -> Dict[str, Any]:
        """Analyze comprehensive performance trends."""
        try:
            # Get historical metrics
            metric_names = ['cpm', 'ctr', 'spend', 'impressions', 'clicks', 'purchases', 'atc_rate']
            df = self.get_historical_metrics(ad_id, metric_names, days_back)
            
            if df.empty:
                return {'status': 'insufficient_data'}
            
            # Calculate trend analysis for each metric
            trends = {}
            
            for metric in metric_names:
                if metric in df.columns:
                    values = df[metric].values
                    if len(values) >= 3:
                        # Linear trend analysis
                        x = np.arange(len(values))
                        slope, _, r_value, p_value, _ = np.polyfit(x, values, 1)
                        
                        # Trend classification
                        if p_value < 0.05:  # Significant trend
                            if slope > 0:
                                trend_direction = 'improving'
                            else:
                                trend_direction = 'declining'
                        else:
                            trend_direction = 'stable'
                        
                        # Volatility analysis
                        volatility = np.std(values) / (np.mean(values) + 1e-6)
                        
                        # Recent vs older performance
                        if len(values) >= 6:
                            recent_avg = np.mean(values[-3:])
                            older_avg = np.mean(values[:-3])
                            momentum = (recent_avg - older_avg) / (older_avg + 1e-6)
                        else:
                            momentum = 0
                        
                        trends[metric] = {
                            'trend_direction': trend_direction,
                            'slope': float(slope),
                            'r_squared': float(r_value ** 2),
                            'volatility': float(volatility),
                            'momentum': float(momentum),
                            'current_value': float(values[-1]) if len(values) > 0 else 0,
                            'avg_value': float(np.mean(values))
                        }
            
            return {
                'status': 'success',
                'trends': trends,
                'data_points': len(df),
                'analysis_period_days': days_back
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def detect_performance_anomalies(self, ad_id: str, days_back: int = 7) -> Dict[str, Any]:
        """Detect performance anomalies using historical data."""
        try:
            # Get historical metrics
            metric_names = ['cpm', 'ctr', 'spend', 'impressions', 'clicks']
            df = self.get_historical_metrics(ad_id, metric_names, days_back)
            
            if df.empty or len(df) < 3:
                return {'status': 'insufficient_data'}
            
            anomalies = {}
            
            for metric in metric_names:
                if metric in df.columns:
                    values = df[metric].values
                    
                    # Statistical anomaly detection
                    mean_val = np.mean(values[:-1])  # Exclude latest value
                    std_val = np.std(values[:-1])
                    
                    if std_val > 0:
                        current_val = values[-1]
                        z_score = abs(current_val - mean_val) / std_val
                        
                        # Anomaly threshold (2 standard deviations)
                        if z_score > 2.0:
                            anomaly_type = 'spike' if current_val > mean_val else 'drop'
                            severity = 'high' if z_score > 3.0 else 'medium'
                            
                            anomalies[metric] = {
                                'type': anomaly_type,
                                'severity': severity,
                                'z_score': float(z_score),
                                'current_value': float(current_val),
                                'expected_value': float(mean_val),
                                'deviation_pct': float(abs(current_val - mean_val) / (mean_val + 1e-6) * 100)
                            }
            
            return {
                'status': 'success',
                'anomalies': anomalies,
                'analysis_period_days': days_back
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting performance anomalies: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_creative_performance_history(self, creative_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Get performance history for a creative across all ads."""
        try:
            # Get all ads using this creative
            response = self.client.table('ad_lifecycle').select('ad_id').eq(
                'creative_id', creative_id
            ).execute()
            
            if not response.data:
                return {'status': 'no_ads_found'}
            
            ad_ids = [row['ad_id'] for row in response.data]
            
            # Get historical performance for all ads
            start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            response = self.client.table('historical_data').select('*').in_(
                'ad_id', ad_ids
            ).gte('ts_iso', start_date).order('ts_epoch', desc=False).execute()
            
            if not response.data:
                return {'status': 'insufficient_data'}
            
            df = pd.DataFrame(response.data)
            df['ts_iso'] = pd.to_datetime(df['ts_iso'])
            
            # Aggregate performance by date
            daily_performance = df.groupby(['ts_iso', 'metric_name'])['metric_value'].sum().reset_index()
            pivot_df = daily_performance.pivot_table(
                index='ts_iso', 
                columns='metric_name', 
                values='metric_value', 
                aggfunc='sum'
            ).fillna(0)
            
            # Calculate creative-level metrics
            creative_metrics = {}
            for metric in pivot_df.columns:
                values = pivot_df[metric].values
                if len(values) > 0:
                    creative_metrics[metric] = {
                        'total': float(np.sum(values)),
                        'avg_daily': float(np.mean(values)),
                        'max_daily': float(np.max(values)),
                        'trend': 'improving' if len(values) >= 3 and np.polyfit(range(len(values)), values, 1)[0] > 0 else 'stable'
                    }
            
            return {
                'status': 'success',
                'creative_metrics': creative_metrics,
                'ads_count': len(ad_ids),
                'analysis_period_days': days_back,
                'data_points': len(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting creative performance history: {e}")
            return {'status': 'error', 'error': str(e)}


class EnhancedMLPredictor:
    """Enhanced ML predictor with historical data integration."""
    
    def __init__(self, supabase_client: Client, historical_analyzer: HistoricalDataAnalyzer):
        self.client = supabase_client
        self.historical_analyzer = historical_analyzer
        self.logger = logging.getLogger(f"{__name__}.EnhancedMLPredictor")
    
    def predict_with_historical_context(self, ad_id: str, stage: str) -> Dict[str, Any]:
        """Make predictions with enhanced historical context."""
        try:
            # Get current performance data
            current_data = self._get_current_performance(ad_id)
            if not current_data:
                return {'status': 'no_current_data'}
            
            # Get historical trends
            trends = self.historical_analyzer.analyze_performance_trends(ad_id, days_back=14)
            
            # Get ad age features
            age_features = self.historical_analyzer.get_ad_age_features(ad_id)
            
            # Get anomaly detection
            anomalies = self.historical_analyzer.detect_performance_anomalies(ad_id, days_back=7)
            
            # Combine all features for prediction
            enhanced_features = {
                **current_data,
                **age_features,
                'trends': trends,
                'anomalies': anomalies
            }
            
            # Make prediction (simplified - would integrate with actual ML models)
            prediction = self._make_enhanced_prediction(enhanced_features, stage)
            
            return {
                'status': 'success',
                'prediction': prediction,
                'features': enhanced_features,
                'confidence': self._calculate_confidence(trends, anomalies, age_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error making enhanced prediction: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_current_performance(self, ad_id: str) -> Dict[str, Any]:
        """Get current performance data."""
        try:
            response = self.client.table('performance_metrics').select('*').eq(
                'ad_id', ad_id
            ).order('created_at', desc=True).limit(1).execute()
            
            if response.data:
                return response.data[0]
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting current performance: {e}")
            return {}
    
    def _make_enhanced_prediction(self, features: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Make enhanced prediction using historical context."""
        # This would integrate with your existing ML models
        # For now, return a simplified prediction structure
        
        prediction = {
            'next_day_performance': {
                'predicted_spend': features.get('spend', 0) * 1.1,  # Simplified
                'predicted_cpa': features.get('cpa', 0) * 0.95,
                'predicted_roas': features.get('roas', 0) * 1.05
            },
            'confidence_score': 0.75,
            'risk_factors': []
        }
        
        # Add risk factors based on trends and anomalies
        if features.get('trends', {}).get('cpm', {}).get('trend_direction') == 'declining':
            prediction['risk_factors'].append('CPM declining trend')
        
        if features.get('anomalies', {}).get('anomalies'):
            prediction['risk_factors'].append('Performance anomalies detected')
        
        if features.get('ad_age_days', 0) > 7:
            prediction['risk_factors'].append('Ad age may affect performance')
        
        return prediction
    
    def _calculate_confidence(self, trends: Dict[str, Any], anomalies: Dict[str, Any], 
                            age_features: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on data quality."""
        confidence = 0.8  # Base confidence
        
        # Reduce confidence based on anomalies
        if anomalies.get('anomalies'):
            confidence -= 0.2
        
        # Reduce confidence for very new or very old ads
        age_days = age_features.get('ad_age_days', 0)
        if age_days < 1:
            confidence -= 0.3  # Very new ads
        elif age_days > 14:
            confidence -= 0.1  # Very old ads
        
        return max(0.1, min(0.95, confidence))


def create_historical_data_analyzer(supabase_client: Client) -> HistoricalDataAnalyzer:
    """Create historical data analyzer."""
    return HistoricalDataAnalyzer(supabase_client)


def create_enhanced_ml_predictor(supabase_client: Client, 
                               historical_analyzer: HistoricalDataAnalyzer) -> EnhancedMLPredictor:
    """Create enhanced ML predictor."""
    return EnhancedMLPredictor(supabase_client, historical_analyzer)
