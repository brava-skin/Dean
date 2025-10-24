"""
DEAN SELF-LEARNING META ADS AUTOMATION SYSTEM
Advanced Performance Tracking & Fatigue Detection

This module implements sophisticated performance decay tracking, fatigue detection,
and predictive analytics for maintaining optimal ad performance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from supabase import create_client, Client

from infrastructure.utils import now_utc, today_ymd_account, yesterday_ymd_account

logger = logging.getLogger(__name__)

# =====================================================
# PERFORMANCE TRACKING SYSTEM
# =====================================================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics with temporal context."""
    ad_id: str
    stage: str
    timestamp: datetime
    
    # Core metrics
    spend: float
    impressions: int
    clicks: int
    purchases: int
    add_to_cart: int
    initiate_checkout: int
    revenue: float
    
    # Calculated metrics
    ctr: float
    cpm: float
    cpc: float
    cpa: float
    roas: float
    aov: float
    
    # Engagement metrics
    three_sec_views: int
    video_views: int
    watch_time: float
    dwell_time: float
    frequency: float
    
    # Conversion metrics
    atc_rate: float
    ic_rate: float
    purchase_rate: float
    atc_to_ic_rate: float
    ic_to_purchase_rate: float
    
    # Quality scores
    performance_quality_score: int
    stability_score: float
    momentum_score: float
    fatigue_index: float

@dataclass
class FatigueAnalysis:
    """Container for fatigue analysis results."""
    ad_id: str
    fatigue_score: float
    fatigue_confidence: float
    fatigue_trend: str  # 'increasing', 'stable', 'decreasing'
    fatigue_velocity: float
    fatigue_acceleration: float
    half_life_days: Optional[float]
    fatigue_signals: Dict[str, Any]
    recommended_action: str
    created_at: datetime

@dataclass
class PerformanceDecay:
    """Container for performance decay analysis."""
    ad_id: str
    metric_name: str
    decay_rate: float
    decay_confidence: float
    decay_trend: str
    recovery_probability: float
    decay_signals: Dict[str, Any]
    created_at: datetime

class SupabasePerformanceClient:
    """Supabase client for performance tracking."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.logger = logging.getLogger(f"{__name__}.SupabasePerformanceClient")
    
    def get_performance_history(self, ad_id: str, days_back: int = 30) -> pd.DataFrame:
        """Get performance history for an ad."""
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            response = self.client.table('performance_metrics').select('*').eq(
                'ad_id', ad_id
            ).gte('date_start', start_date).order('date_start').execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            
            # Convert types
            numeric_columns = [
                'spend', 'impressions', 'clicks', 'purchases', 'add_to_cart',
                'initiate_checkout', 'revenue', 'ctr', 'cpm', 'cpc', 'cpa',
                'roas', 'aov', 'three_sec_views', 'video_views', 'watch_time',
                'dwell_time', 'frequency', 'atc_rate', 'ic_rate', 'purchase_rate',
                'atc_to_ic_rate', 'ic_to_purchase_rate', 'performance_quality_score',
                'stability_score', 'momentum_score', 'fatigue_index'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert dates
            df['date_start'] = pd.to_datetime(df['date_start'])
            df['date_end'] = pd.to_datetime(df['date_end'])
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching performance history for {ad_id}: {e}")
            return pd.DataFrame()
    
    def get_time_series_data(self, ad_id: str, metric_name: str, 
                           days_back: int = 30) -> pd.DataFrame:
        """Get time series data for temporal analysis."""
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            response = self.client.table('time_series_data').select('*').eq(
                'ad_id', ad_id
            ).eq('metric_name', metric_name).gte('timestamp', start_date).order('timestamp').execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')
            
            return df.sort_values('timestamp')
            
        except Exception as e:
            self.logger.error(f"Error fetching time series data: {e}")
            return pd.DataFrame()
    
    def save_fatigue_analysis(self, analysis: FatigueAnalysis) -> str:
        """Save fatigue analysis to database."""
        try:
            data = {
                'ad_id': analysis.ad_id,
                'fatigue_score': analysis.fatigue_score,
                'fatigue_confidence': analysis.fatigue_confidence,
                'fatigue_trend': analysis.fatigue_trend,
                'fatigue_velocity': analysis.fatigue_velocity,
                'fatigue_acceleration': analysis.fatigue_acceleration,
                'half_life_days': analysis.half_life_days,
                'fatigue_signals': analysis.fatigue_signals,
                'recommended_action': analysis.recommended_action,
                'created_at': analysis.created_at.isoformat()
            }
            
            response = self.client.table('fatigue_analysis').insert(data).execute()
            
            if response.data:
                analysis_id = response.data[0]['id']
                self.logger.info(f"Saved fatigue analysis {analysis_id} for ad {analysis.ad_id}")
                return analysis_id
            else:
                self.logger.error(f"Failed to save fatigue analysis for ad {analysis.ad_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving fatigue analysis: {e}")
            return None
    
    def save_performance_decay(self, decay: PerformanceDecay) -> str:
        """Save performance decay analysis to database."""
        try:
            data = {
                'ad_id': decay.ad_id,
                'metric_name': decay.metric_name,
                'decay_rate': decay.decay_rate,
                'decay_confidence': decay.decay_confidence,
                'decay_trend': decay.decay_trend,
                'recovery_probability': decay.recovery_probability,
                'decay_signals': decay.decay_signals,
                'created_at': decay.created_at.isoformat()
            }
            
            response = self.client.table('performance_decay').insert(data).execute()
            
            if response.data:
                decay_id = response.data[0]['id']
                self.logger.info(f"Saved performance decay {decay_id} for ad {decay.ad_id}")
                return decay_id
            else:
                self.logger.error(f"Failed to save performance decay for ad {decay.ad_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving performance decay: {e}")
            return None

class FatigueDetector:
    """Advanced fatigue detection using multiple algorithms."""
    
    def __init__(self, supabase_client: SupabasePerformanceClient):
        self.supabase = supabase_client
        self.logger = logging.getLogger(f"{__name__}.FatigueDetector")
    
    def detect_fatigue(self, ad_id: str, days_back: int = 14) -> FatigueAnalysis:
        """Detect fatigue using multiple detection methods."""
        try:
            # Get performance history
            df = self.supabase.get_performance_history(ad_id, days_back)
            if df.empty or len(df) < 3:
                return self._create_empty_fatigue_analysis(ad_id)
            
            # Multiple fatigue detection methods
            fatigue_signals = {}
            
            # 1. CTR fatigue
            ctr_fatigue = self._detect_ctr_fatigue(df)
            fatigue_signals['ctr_fatigue'] = ctr_fatigue
            
            # 2. CPA fatigue
            cpa_fatigue = self._detect_cpa_fatigue(df)
            fatigue_signals['cpa_fatigue'] = cpa_fatigue
            
            # 3. ROAS fatigue
            roas_fatigue = self._detect_roas_fatigue(df)
            fatigue_signals['roas_fatigue'] = roas_fatigue
            
            # 4. Engagement fatigue
            engagement_fatigue = self._detect_engagement_fatigue(df)
            fatigue_signals['engagement_fatigue'] = engagement_fatigue
            
            # 5. Volatility fatigue
            volatility_fatigue = self._detect_volatility_fatigue(df)
            fatigue_signals['volatility_fatigue'] = volatility_fatigue
            
            # 6. Momentum fatigue
            momentum_fatigue = self._detect_momentum_fatigue(df)
            fatigue_signals['momentum_fatigue'] = momentum_fatigue
            
            # Calculate overall fatigue score
            fatigue_score, fatigue_confidence = self._calculate_fatigue_score(fatigue_signals)
            
            # Determine fatigue trend
            fatigue_trend = self._determine_fatigue_trend(df, fatigue_signals)
            
            # Calculate fatigue velocity and acceleration
            fatigue_velocity = self._calculate_fatigue_velocity(df)
            fatigue_acceleration = self._calculate_fatigue_acceleration(df)
            
            # Estimate half-life
            half_life_days = self._estimate_half_life(df, fatigue_score)
            
            # Determine recommended action
            recommended_action = self._determine_fatigue_action(
                fatigue_score, fatigue_confidence, fatigue_trend, half_life_days
            )
            
            return FatigueAnalysis(
                ad_id=ad_id,
                fatigue_score=fatigue_score,
                fatigue_confidence=fatigue_confidence,
                fatigue_trend=fatigue_trend,
                fatigue_velocity=fatigue_velocity,
                fatigue_acceleration=fatigue_acceleration,
                half_life_days=half_life_days,
                fatigue_signals=fatigue_signals,
                recommended_action=recommended_action,
                created_at=now_utc()
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting fatigue for {ad_id}: {e}")
            return self._create_empty_fatigue_analysis(ad_id)
    
    def _detect_ctr_fatigue(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect CTR-based fatigue."""
        try:
            if 'ctr' not in df.columns or len(df) < 3:
                return {'fatigue': False, 'confidence': 0.0, 'reason': 'insufficient_data'}
            
            ctr_values = df['ctr'].dropna()
            if len(ctr_values) < 3:
                return {'fatigue': False, 'confidence': 0.0, 'reason': 'insufficient_data'}
            
            # Linear trend analysis
            x = np.arange(len(ctr_values))
            slope, _, r_value, p_value, _ = stats.linregress(x, ctr_values)
            
            # CTR decline threshold
            ctr_decline_threshold = -0.0001  # 0.01% decline per day
            
            fatigue = slope < ctr_decline_threshold and p_value < 0.1
            confidence = 1 - p_value if fatigue else 0.0
            
            return {
                'fatigue': fatigue,
                'confidence': confidence,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'reason': f"CTR declining at {slope:.6f} per day" if fatigue else "CTR stable"
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting CTR fatigue: {e}")
            return {'fatigue': False, 'confidence': 0.0, 'reason': f'error: {str(e)}'}
    
    def _detect_cpa_fatigue(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect CPA-based fatigue."""
        try:
            if 'cpa' not in df.columns or len(df) < 3:
                return {'fatigue': False, 'confidence': 0.0, 'reason': 'insufficient_data'}
            
            cpa_values = df['cpa'].dropna()
            if len(cpa_values) < 3:
                return {'fatigue': False, 'confidence': 0.0, 'reason': 'insufficient_data'}
            
            # Linear trend analysis
            x = np.arange(len(cpa_values))
            slope, _, r_value, p_value, _ = stats.linregress(x, cpa_values)
            
            # CPA increase threshold
            cpa_increase_threshold = 0.5  # €0.50 increase per day
            
            fatigue = slope > cpa_increase_threshold and p_value < 0.1
            confidence = 1 - p_value if fatigue else 0.0
            
            return {
                'fatigue': fatigue,
                'confidence': confidence,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'reason': f"CPA increasing at €{slope:.2f} per day" if fatigue else "CPA stable"
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting CPA fatigue: {e}")
            return {'fatigue': False, 'confidence': 0.0, 'reason': f'error: {str(e)}'}
    
    def _detect_roas_fatigue(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect ROAS-based fatigue."""
        try:
            if 'roas' not in df.columns or len(df) < 3:
                return {'fatigue': False, 'confidence': 0.0, 'reason': 'insufficient_data'}
            
            roas_values = df['roas'].dropna()
            if len(roas_values) < 3:
                return {'fatigue': False, 'confidence': 0.0, 'reason': 'insufficient_data'}
            
            # Linear trend analysis
            x = np.arange(len(roas_values))
            slope, _, r_value, p_value, _ = stats.linregress(x, roas_values)
            
            # ROAS decline threshold
            roas_decline_threshold = -0.05  # 0.05 decline per day
            
            fatigue = slope < roas_decline_threshold and p_value < 0.1
            confidence = 1 - p_value if fatigue else 0.0
            
            return {
                'fatigue': fatigue,
                'confidence': confidence,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'reason': f"ROAS declining at {slope:.3f} per day" if fatigue else "ROAS stable"
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting ROAS fatigue: {e}")
            return {'fatigue': False, 'confidence': 0.0, 'reason': f'error: {str(e)}'}
    
    def _detect_engagement_fatigue(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect engagement-based fatigue."""
        try:
            engagement_metrics = ['atc_rate', 'purchase_rate', 'three_sec_views']
            fatigue_signals = {}
            
            for metric in engagement_metrics:
                if metric in df.columns and len(df) >= 3:
                    values = df[metric].dropna()
                    if len(values) >= 3:
                        x = np.arange(len(values))
                        slope, _, r_value, p_value, _ = stats.linregress(x, values)
                        
                        # Engagement decline threshold
                        decline_threshold = -0.001  # 0.1% decline per day
                        
                        fatigue_signals[metric] = {
                            'fatigue': slope < decline_threshold and p_value < 0.1,
                            'confidence': 1 - p_value if slope < decline_threshold else 0.0,
                            'slope': slope,
                            'reason': f"{metric} declining" if slope < decline_threshold else f"{metric} stable"
                        }
            
            # Overall engagement fatigue
            if fatigue_signals:
                fatigued_metrics = sum(1 for signal in fatigue_signals.values() if signal['fatigue'])
                total_metrics = len(fatigue_signals)
                fatigue_ratio = fatigued_metrics / total_metrics
                
                return {
                    'fatigue': fatigue_ratio > 0.5,  # More than half of metrics declining
                    'confidence': fatigue_ratio,
                    'fatigue_ratio': fatigue_ratio,
                    'fatigued_metrics': fatigued_metrics,
                    'total_metrics': total_metrics,
                    'signals': fatigue_signals,
                    'reason': f"{fatigued_metrics}/{total_metrics} engagement metrics declining"
                }
            else:
                return {'fatigue': False, 'confidence': 0.0, 'reason': 'no_engagement_data'}
                
        except Exception as e:
            self.logger.error(f"Error detecting engagement fatigue: {e}")
            return {'fatigue': False, 'confidence': 0.0, 'reason': f'error: {str(e)}'}
    
    def _detect_volatility_fatigue(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect volatility-based fatigue."""
        try:
            key_metrics = ['ctr', 'cpa', 'roas']
            volatility_signals = {}
            
            for metric in key_metrics:
                if metric in df.columns and len(df) >= 7:
                    values = df[metric].dropna()
                    if len(values) >= 7:
                        # Calculate rolling volatility
                        rolling_std = values.rolling(window=3, min_periods=2).std()
                        recent_volatility = rolling_std.tail(3).mean()
                        historical_volatility = rolling_std.head(3).mean()
                        
                        # Volatility increase indicates fatigue
                        volatility_increase = (recent_volatility - historical_volatility) / (historical_volatility + 1e-6)
                        
                        volatility_signals[metric] = {
                            'volatility_increase': volatility_increase,
                            'recent_volatility': recent_volatility,
                            'historical_volatility': historical_volatility,
                            'fatigue': volatility_increase > 0.5  # 50% increase in volatility
                        }
            
            if volatility_signals:
                high_volatility_metrics = sum(1 for signal in volatility_signals.values() if signal['fatigue'])
                total_metrics = len(volatility_signals)
                
                return {
                    'fatigue': high_volatility_metrics > 0,
                    'confidence': high_volatility_metrics / total_metrics,
                    'high_volatility_metrics': high_volatility_metrics,
                    'total_metrics': total_metrics,
                    'signals': volatility_signals,
                    'reason': f"{high_volatility_metrics}/{total_metrics} metrics showing high volatility"
                }
            else:
                return {'fatigue': False, 'confidence': 0.0, 'reason': 'no_volatility_data'}
                
        except Exception as e:
            self.logger.error(f"Error detecting volatility fatigue: {e}")
            return {'fatigue': False, 'confidence': 0.0, 'reason': f'error: {str(e)}'}
    
    def _detect_momentum_fatigue(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect momentum-based fatigue."""
        try:
            if 'momentum_score' not in df.columns or len(df) < 5:
                return {'fatigue': False, 'confidence': 0.0, 'reason': 'insufficient_momentum_data'}
            
            momentum_values = df['momentum_score'].dropna()
            if len(momentum_values) < 5:
                return {'fatigue': False, 'confidence': 0.0, 'reason': 'insufficient_momentum_data'}
            
            # Analyze momentum trend
            x = np.arange(len(momentum_values))
            slope, _, r_value, p_value, _ = stats.linregress(x, momentum_values)
            
            # Negative momentum indicates fatigue
            momentum_fatigue = slope < -0.1 and p_value < 0.1
            confidence = 1 - p_value if momentum_fatigue else 0.0
            
            return {
                'fatigue': momentum_fatigue,
                'confidence': confidence,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'reason': f"Momentum declining at {slope:.3f} per day" if momentum_fatigue else "Momentum stable"
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum fatigue: {e}")
            return {'fatigue': False, 'confidence': 0.0, 'reason': f'error: {str(e)}'}
    
    def _calculate_fatigue_score(self, fatigue_signals: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate overall fatigue score and confidence."""
        try:
            fatigue_scores = []
            confidence_scores = []
            
            for signal_name, signal_data in fatigue_signals.items():
                if isinstance(signal_data, dict) and 'fatigue' in signal_data:
                    if signal_data['fatigue']:
                        fatigue_scores.append(1.0)
                        confidence_scores.append(signal_data.get('confidence', 0.5))
                    else:
                        fatigue_scores.append(0.0)
                        confidence_scores.append(signal_data.get('confidence', 0.5))
            
            if not fatigue_scores:
                return 0.0, 0.0
            
            # Weighted average based on confidence
            total_weight = sum(confidence_scores)
            if total_weight > 0:
                weighted_fatigue = sum(score * conf for score, conf in zip(fatigue_scores, confidence_scores)) / total_weight
                avg_confidence = np.mean(confidence_scores)
            else:
                weighted_fatigue = np.mean(fatigue_scores)
                avg_confidence = 0.5
            
            return float(weighted_fatigue), float(avg_confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating fatigue score: {e}")
            return 0.0, 0.0
    
    def _determine_fatigue_trend(self, df: pd.DataFrame, fatigue_signals: Dict[str, Any]) -> str:
        """Determine overall fatigue trend."""
        try:
            if 'fatigue_index' in df.columns and len(df) >= 3:
                fatigue_values = df['fatigue_index'].dropna()
                if len(fatigue_values) >= 3:
                    x = np.arange(len(fatigue_values))
                    slope, _, _, p_value, _ = stats.linregress(x, fatigue_values)
                    
                    if p_value < 0.1:  # Significant trend
                        if slope > 0.01:
                            return 'increasing'
                        elif slope < -0.01:
                            return 'decreasing'
                    
            return 'stable'
            
        except Exception as e:
            self.logger.error(f"Error determining fatigue trend: {e}")
            return 'stable'
    
    def _calculate_fatigue_velocity(self, df: pd.DataFrame) -> float:
        """Calculate fatigue velocity (rate of change)."""
        try:
            if 'fatigue_index' in df.columns and len(df) >= 2:
                fatigue_values = df['fatigue_index'].dropna()
                if len(fatigue_values) >= 2:
                    return float(fatigue_values.diff().mean())
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating fatigue velocity: {e}")
            return 0.0
    
    def _calculate_fatigue_acceleration(self, df: pd.DataFrame) -> float:
        """Calculate fatigue acceleration (rate of change of velocity)."""
        try:
            if 'fatigue_index' in df.columns and len(df) >= 3:
                fatigue_values = df['fatigue_index'].dropna()
                if len(fatigue_values) >= 3:
                    velocity = fatigue_values.diff()
                    acceleration = velocity.diff()
                    return float(acceleration.mean())
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating fatigue acceleration: {e}")
            return 0.0
    
    def _estimate_half_life(self, df: pd.DataFrame, fatigue_score: float) -> Optional[float]:
        """Estimate half-life of ad performance."""
        try:
            if fatigue_score < 0.3:  # Low fatigue
                return None
            
            # Use exponential decay model
            if 'performance_quality_score' in df.columns and len(df) >= 3:
                quality_values = df['performance_quality_score'].dropna()
                if len(quality_values) >= 3:
                    # Fit exponential decay: y = a * exp(-b * x)
                    x = np.arange(len(quality_values))
                    y = quality_values.values
                    
                    # Linear regression on log-transformed data
                    log_y = np.log(y + 1e-6)  # Add small value to avoid log(0)
                    slope, intercept, r_value, p_value, _ = stats.linregress(x, log_y)
                    
                    if p_value < 0.1 and slope < 0:  # Significant decay
                        decay_rate = -slope
                        half_life = np.log(2) / decay_rate
                        return float(half_life)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error estimating half-life: {e}")
            return None
    
    def _determine_fatigue_action(self, fatigue_score: float, confidence: float,
                                trend: str, half_life_days: Optional[float]) -> str:
        """Determine recommended action based on fatigue analysis."""
        try:
            if fatigue_score < 0.3:
                return 'continue'
            elif fatigue_score < 0.6:
                if trend == 'increasing':
                    return 'monitor_closely'
                else:
                    return 'continue'
            elif fatigue_score < 0.8:
                if half_life_days and half_life_days < 7:
                    return 'scale_down'
                else:
                    return 'monitor_closely'
            else:
                if half_life_days and half_life_days < 3:
                    return 'pause'
                else:
                    return 'scale_down'
                    
        except Exception as e:
            self.logger.error(f"Error determining fatigue action: {e}")
            return 'monitor_closely'
    
    def _create_empty_fatigue_analysis(self, ad_id: str) -> FatigueAnalysis:
        """Create empty fatigue analysis for insufficient data."""
        return FatigueAnalysis(
            ad_id=ad_id,
            fatigue_score=0.0,
            fatigue_confidence=0.0,
            fatigue_trend='stable',
            fatigue_velocity=0.0,
            fatigue_acceleration=0.0,
            half_life_days=None,
            fatigue_signals={},
            recommended_action='continue',
            created_at=now_utc()
        )

class PerformanceDecayAnalyzer:
    """Analyzer for performance decay patterns."""
    
    def __init__(self, supabase_client: SupabasePerformanceClient):
        self.supabase = supabase_client
        self.logger = logging.getLogger(f"{__name__}.PerformanceDecayAnalyzer")
    
    def analyze_decay(self, ad_id: str, metric_name: str, 
                     days_back: int = 14) -> PerformanceDecay:
        """Analyze performance decay for a specific metric."""
        try:
            # Get time series data
            df = self.supabase.get_time_series_data(ad_id, metric_name, days_back)
            if df.empty or len(df) < 3:
                return self._create_empty_decay_analysis(ad_id, metric_name)
            
            # Calculate decay rate
            decay_rate, decay_confidence = self._calculate_decay_rate(df)
            
            # Determine decay trend
            decay_trend = self._determine_decay_trend(df)
            
            # Calculate recovery probability
            recovery_probability = self._calculate_recovery_probability(df, decay_rate)
            
            # Identify decay signals
            decay_signals = self._identify_decay_signals(df)
            
            return PerformanceDecay(
                ad_id=ad_id,
                metric_name=metric_name,
                decay_rate=decay_rate,
                decay_confidence=decay_confidence,
                decay_trend=decay_trend,
                recovery_probability=recovery_probability,
                decay_signals=decay_signals,
                created_at=now_utc()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing decay for {ad_id} metric {metric_name}: {e}")
            return self._create_empty_decay_analysis(ad_id, metric_name)
    
    def _calculate_decay_rate(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate decay rate using exponential fitting."""
        try:
            values = df['metric_value'].dropna()
            if len(values) < 3:
                return 0.0, 0.0
            
            x = np.arange(len(values))
            y = values.values
            
            # Fit exponential decay: y = a * exp(-b * x)
            log_y = np.log(y + 1e-6)  # Add small value to avoid log(0)
            slope, intercept, r_value, p_value, _ = stats.linregress(x, log_y)
            
            decay_rate = -slope if slope < 0 else 0.0
            confidence = 1 - p_value if p_value < 0.1 else 0.0
            
            return float(decay_rate), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating decay rate: {e}")
            return 0.0, 0.0
    
    def _determine_decay_trend(self, df: pd.DataFrame) -> str:
        """Determine decay trend."""
        try:
            values = df['metric_value'].dropna()
            if len(values) < 3:
                return 'stable'
            
            x = np.arange(len(values))
            slope, _, _, p_value, _ = stats.linregress(x, values)
            
            if p_value < 0.1:  # Significant trend
                if slope < -0.01:
                    return 'declining'
                elif slope > 0.01:
                    return 'improving'
            
            return 'stable'
            
        except Exception as e:
            self.logger.error(f"Error determining decay trend: {e}")
            return 'stable'
    
    def _calculate_recovery_probability(self, df: pd.DataFrame, decay_rate: float) -> float:
        """Calculate probability of performance recovery."""
        try:
            if decay_rate <= 0:
                return 1.0  # No decay, high recovery probability
            
            # Analyze recent performance vs historical
            if len(df) >= 7:
                recent_performance = df.tail(3)['metric_value'].mean()
                historical_performance = df.head(3)['metric_value'].mean()
                
                if recent_performance > historical_performance:
                    return 0.8  # High recovery probability
                elif recent_performance > historical_performance * 0.8:
                    return 0.6  # Medium recovery probability
                else:
                    return 0.3  # Low recovery probability
            
            return 0.5  # Default medium probability
            
        except Exception as e:
            self.logger.error(f"Error calculating recovery probability: {e}")
            return 0.5
    
    def _identify_decay_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify specific decay signals."""
        try:
            signals = {}
            values = df['metric_value'].dropna()
            
            if len(values) >= 5:
                # Volatility increase
                rolling_std = values.rolling(window=3, min_periods=2).std()
                recent_volatility = rolling_std.tail(2).mean()
                historical_volatility = rolling_std.head(2).mean()
                
                signals['volatility_increase'] = {
                    'value': (recent_volatility - historical_volatility) / (historical_volatility + 1e-6),
                    'significant': recent_volatility > historical_volatility * 1.5
                }
                
                # Performance drop
                recent_avg = values.tail(3).mean()
                historical_avg = values.head(3).mean()
                
                signals['performance_drop'] = {
                    'value': (recent_avg - historical_avg) / (historical_avg + 1e-6),
                    'significant': recent_avg < historical_avg * 0.8
                }
                
                # Trend consistency
                x = np.arange(len(values))
                slope, _, r_value, _, _ = stats.linregress(x, values)
                
                signals['trend_consistency'] = {
                    'r_squared': r_value ** 2,
                    'slope': slope,
                    'consistent': r_value ** 2 > 0.5
                }
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error identifying decay signals: {e}")
            return {}
    
    def _create_empty_decay_analysis(self, ad_id: str, metric_name: str) -> PerformanceDecay:
        """Create empty decay analysis for insufficient data."""
        return PerformanceDecay(
            ad_id=ad_id,
            metric_name=metric_name,
            decay_rate=0.0,
            decay_confidence=0.0,
            decay_trend='stable',
            recovery_probability=0.5,
            decay_signals={},
            created_at=now_utc()
        )

class PerformanceTrackingSystem:
    """Main performance tracking system."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase = SupabasePerformanceClient(supabase_url, supabase_key)
        self.fatigue_detector = FatigueDetector(self.supabase)
        self.decay_analyzer = PerformanceDecayAnalyzer(self.supabase)
        self.logger = logging.getLogger(f"{__name__}.PerformanceTrackingSystem")
    
    def analyze_ad_performance(self, ad_id: str, days_back: int = 14) -> Dict[str, Any]:
        """Comprehensive performance analysis for an ad."""
        try:
            # Fatigue analysis
            fatigue_analysis = self.fatigue_detector.detect_fatigue(ad_id, days_back)
            
            # Decay analysis for key metrics
            key_metrics = ['ctr', 'cpa', 'roas', 'performance_quality_score']
            decay_analyses = {}
            
            for metric in key_metrics:
                decay_analysis = self.decay_analyzer.analyze_decay(ad_id, metric, days_back)
                decay_analyses[metric] = decay_analysis
            
            # Save analyses to database
            self.supabase.save_fatigue_analysis(fatigue_analysis)
            for decay_analysis in decay_analyses.values():
                self.supabase.save_performance_decay(decay_analysis)
            
            return {
                'fatigue_analysis': fatigue_analysis,
                'decay_analyses': decay_analyses,
                'overall_health_score': self._calculate_health_score(fatigue_analysis, decay_analyses),
                'recommended_actions': self._generate_recommendations(fatigue_analysis, decay_analyses)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing ad performance for {ad_id}: {e}")
            return {}
    
    def _calculate_health_score(self, fatigue_analysis: FatigueAnalysis,
                              decay_analyses: Dict[str, PerformanceDecay]) -> float:
        """Calculate overall health score for an ad."""
        try:
            # Base score from fatigue
            fatigue_score = 1.0 - fatigue_analysis.fatigue_score
            
            # Adjust for decay
            decay_penalties = []
            for metric, decay in decay_analyses.items():
                if decay.decay_trend == 'declining':
                    decay_penalties.append(decay.decay_rate * 0.1)
            
            if decay_penalties:
                avg_decay_penalty = np.mean(decay_penalties)
                fatigue_score -= avg_decay_penalty
            
            # Recovery bonus
            recovery_scores = [decay.recovery_probability for decay in decay_analyses.values()]
            if recovery_scores:
                avg_recovery = np.mean(recovery_scores)
                fatigue_score += (avg_recovery - 0.5) * 0.2
            
            return max(0.0, min(1.0, fatigue_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 0.5
    
    def _generate_recommendations(self, fatigue_analysis: FatigueAnalysis,
                                decay_analyses: Dict[str, PerformanceDecay]) -> List[str]:
        """Generate actionable recommendations."""
        try:
            recommendations = []
            
            # Fatigue-based recommendations
            if fatigue_analysis.fatigue_score > 0.7:
                recommendations.append("High fatigue detected - consider pausing or refreshing creative")
            elif fatigue_analysis.fatigue_score > 0.5:
                recommendations.append("Moderate fatigue - monitor closely and consider scaling down")
            
            # Decay-based recommendations
            for metric, decay in decay_analyses.items():
                if decay.decay_trend == 'declining' and decay.decay_confidence > 0.7:
                    recommendations.append(f"{metric.upper()} declining - investigate and adjust targeting")
                elif decay.recovery_probability < 0.3:
                    recommendations.append(f"{metric.upper()} recovery unlikely - consider replacement")
            
            # Half-life recommendations
            if fatigue_analysis.half_life_days and fatigue_analysis.half_life_days < 7:
                recommendations.append(f"Performance half-life only {fatigue_analysis.half_life_days:.1f} days - urgent action needed")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]

# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_performance_tracking_system(supabase_url: str, supabase_key: str) -> PerformanceTrackingSystem:
    """Create performance tracking system."""
    return PerformanceTrackingSystem(supabase_url, supabase_key)

def analyze_ad_fatigue(tracking_system: PerformanceTrackingSystem, 
                      ad_id: str, days_back: int = 14) -> FatigueAnalysis:
    """Analyze ad fatigue."""
    return tracking_system.fatigue_detector.detect_fatigue(ad_id, days_back)

def analyze_ad_decay(tracking_system: PerformanceTrackingSystem,
                    ad_id: str, metric_name: str, days_back: int = 14) -> PerformanceDecay:
    """Analyze ad performance decay."""
    return tracking_system.decay_analyzer.analyze_decay(ad_id, metric_name, days_back)
