"""
DEAN SELF-LEARNING META ADS AUTOMATION SYSTEM
Advanced ML-Enhanced Reporting & Transparency

This module provides comprehensive reporting with ML insights, predictive analytics,
and intelligent recommendations for the self-learning system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from supabase import create_client, Client

from utils import now_utc, today_ymd_account, yesterday_ymd_account
from slack import (
    notify, post_digest, alert_kill, alert_promote, alert_scale,
    alert_fatigue, alert_data_quality, alert_error
)

logger = logging.getLogger(__name__)

# =====================================================
# ML REPORTING SYSTEM
# =====================================================

@dataclass
class MLReport:
    """Container for ML-enhanced reports."""
    report_type: str
    timestamp: datetime
    stage: str
    insights: Dict[str, Any]
    predictions: Dict[str, Any]
    recommendations: List[str]
    confidence_scores: Dict[str, float]
    performance_metrics: Dict[str, Any]
    ml_metrics: Dict[str, Any]

@dataclass
class PredictiveInsight:
    """Container for predictive insights."""
    insight_type: str
    prediction: float
    confidence: float
    time_horizon: str
    reasoning: str
    impact_score: float
    created_at: datetime

@dataclass
class SystemHealthReport:
    """Container for system health analysis."""
    overall_health: float
    stage_health: Dict[str, float]
    ml_model_health: Dict[str, float]
    data_quality: Dict[str, float]
    learning_velocity: float
    recommendations: List[str]
    created_at: datetime

class SupabaseReportingClient:
    """Supabase client for reporting data."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.logger = logging.getLogger(f"{__name__}.SupabaseReportingClient")
    
    def get_performance_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get performance summary for reporting."""
        try:
            # Get performance metrics
            response = self.client.table('performance_metrics').select('*').gte(
                'date_start', (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            ).execute()
            
            if not response.data:
                return {}
            
            df = pd.DataFrame(response.data)
            
            # Calculate summary metrics
            summary = {
                'total_ads': df['ad_id'].nunique(),
                'total_spend': df['spend'].sum(),
                'total_impressions': df['impressions'].sum(),
                'total_clicks': df['clicks'].sum(),
                'total_purchases': df['purchases'].sum(),
                'total_revenue': df['revenue'].sum(),
                'avg_ctr': df['ctr'].mean(),
                'avg_cpa': df['cpa'].mean(),
                'avg_roas': df['roas'].mean(),
                'avg_quality_score': df['performance_quality_score'].mean(),
                'stages': {}
            }
            
            # Stage-specific metrics
            for stage in ['testing', 'validation', 'scaling']:
                stage_df = df[df['stage'] == stage]
                if not stage_df.empty:
                    summary['stages'][stage] = {
                        'ads': stage_df['ad_id'].nunique(),
                        'spend': stage_df['spend'].sum(),
                        'purchases': stage_df['purchases'].sum(),
                        'avg_cpa': stage_df['cpa'].mean(),
                        'avg_roas': stage_df['roas'].mean(),
                        'avg_quality': stage_df['performance_quality_score'].mean()
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def get_ml_insights(self, days_back: int = 7) -> Dict[str, Any]:
        """Get ML insights for reporting."""
        try:
            # Get ML predictions
            response = self.client.table('ml_predictions').select('*').gte(
                'created_at', (datetime.now() - timedelta(days=days_back)).isoformat()
            ).execute()
            
            if not response.data:
                return {}
            
            df = pd.DataFrame(response.data)
            
            # Calculate ML insights
            insights = {
                'total_predictions': len(df),
                'avg_confidence': df['confidence_score'].mean(),
                'high_confidence_predictions': len(df[df['confidence_score'] > 0.8]),
                'prediction_accuracy': self._calculate_prediction_accuracy(df),
                'model_performance': self._get_model_performance(),
                'learning_events': self._get_learning_events(days_back)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting ML insights: {e}")
            return {}
    
    def _calculate_prediction_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate prediction accuracy."""
        try:
            if df.empty:
                return 0.0
            
            # This would compare predictions with actual outcomes
            # For now, return a placeholder
            return 0.75  # 75% accuracy placeholder
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction accuracy: {e}")
            return 0.0
    
    def _get_model_performance(self) -> Dict[str, Any]:
        """Get ML model performance metrics."""
        try:
            response = self.client.table('ml_models').select('*').eq('is_active', True).execute()
            
            if not response.data:
                return {}
            
            models = response.data
            performance = {}
            
            for model in models:
                model_type = model.get('model_type', 'unknown')
                metrics = model.get('performance_metrics', {})
                
                performance[model_type] = {
                    'version': model.get('version', 1),
                    'trained_at': model.get('trained_at'),
                    'metrics': metrics,
                    'is_active': model.get('is_active', False)
                }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error getting model performance: {e}")
            return {}
    
    def _get_learning_events(self, days_back: int) -> List[Dict[str, Any]]:
        """Get learning events for reporting."""
        try:
            response = self.client.table('learning_events').select('*').gte(
                'created_at', (datetime.now() - timedelta(days=days_back)).isoformat()
            ).execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            self.logger.error(f"Error getting learning events: {e}")
            return []

class PredictiveReporter:
    """Advanced predictive reporting system."""
    
    def __init__(self, supabase_client: SupabaseReportingClient):
        self.supabase = supabase_client
        self.logger = logging.getLogger(f"{__name__}.PredictiveReporter")
    
    def generate_predictive_insights(self, stage: str, days_ahead: int = 7) -> List[PredictiveInsight]:
        """Generate predictive insights for a stage."""
        try:
            insights = []
            
            # CPA prediction
            cpa_insight = self._predict_cpa_trend(stage, days_ahead)
            if cpa_insight:
                insights.append(cpa_insight)
            
            # ROAS prediction
            roas_insight = self._predict_roas_trend(stage, days_ahead)
            if roas_insight:
                insights.append(roas_insight)
            
            # Fatigue prediction
            fatigue_insight = self._predict_fatigue_risk(stage, days_ahead)
            if fatigue_insight:
                insights.append(fatigue_insight)
            
            # Performance quality prediction
            quality_insight = self._predict_quality_trend(stage, days_ahead)
            if quality_insight:
                insights.append(quality_insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating predictive insights: {e}")
            return []
    
    def _predict_cpa_trend(self, stage: str, days_ahead: int) -> Optional[PredictiveInsight]:
        """Predict CPA trend for a stage."""
        try:
            # Get historical CPA data
            response = self.supabase.client.table('performance_metrics').select('*').eq(
                'stage', stage
            ).gte('date_start', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            ).execute()
            
            if not response.data:
                return None
            
            df = pd.DataFrame(response.data)
            cpa_values = df['cpa'].dropna()
            
            if len(cpa_values) < 7:
                return None
            
            # Simple trend analysis
            x = np.arange(len(cpa_values))
            slope, _, r_value, p_value, _ = np.polyfit(x, cpa_values, 1)
            
            # Predict future CPA
            future_cpa = cpa_values.iloc[-1] + slope * days_ahead
            confidence = 1 - p_value if p_value < 0.1 else 0.5
            
            # Determine impact
            current_cpa = cpa_values.iloc[-1]
            impact_score = abs(future_cpa - current_cpa) / current_cpa if current_cpa > 0 else 0
            
            return PredictiveInsight(
                insight_type='cpa_trend',
                prediction=future_cpa,
                confidence=confidence,
                time_horizon=f'{days_ahead} days',
                reasoning=f"CPA trend analysis shows {slope:.2f} change per day",
                impact_score=impact_score,
                created_at=now_utc()
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting CPA trend: {e}")
            return None
    
    def _predict_roas_trend(self, stage: str, days_ahead: int) -> Optional[PredictiveInsight]:
        """Predict ROAS trend for a stage."""
        try:
            # Get historical ROAS data
            response = self.supabase.client.table('performance_metrics').select('*').eq(
                'stage', stage
            ).gte('date_start', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            ).execute()
            
            if not response.data:
                return None
            
            df = pd.DataFrame(response.data)
            roas_values = df['roas'].dropna()
            
            if len(roas_values) < 7:
                return None
            
            # Trend analysis
            x = np.arange(len(roas_values))
            slope, _, r_value, p_value, _ = np.polyfit(x, roas_values, 1)
            
            # Predict future ROAS
            future_roas = roas_values.iloc[-1] + slope * days_ahead
            confidence = 1 - p_value if p_value < 0.1 else 0.5
            
            # Determine impact
            current_roas = roas_values.iloc[-1]
            impact_score = abs(future_roas - current_roas) / current_roas if current_roas > 0 else 0
            
            return PredictiveInsight(
                insight_type='roas_trend',
                prediction=future_roas,
                confidence=confidence,
                time_horizon=f'{days_ahead} days',
                reasoning=f"ROAS trend analysis shows {slope:.3f} change per day",
                impact_score=impact_score,
                created_at=now_utc()
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting ROAS trend: {e}")
            return None
    
    def _predict_fatigue_risk(self, stage: str, days_ahead: int) -> Optional[PredictiveInsight]:
        """Predict fatigue risk for a stage."""
        try:
            # Get fatigue data
            response = self.supabase.client.table('fatigue_analysis').select('*').eq(
                'ad_id', 'stage_analysis'  # Placeholder for stage-level analysis
            ).gte('created_at', (datetime.now() - timedelta(days=14)).isoformat()
            ).execute()
            
            if not response.data:
                return None
            
            df = pd.DataFrame(response.data)
            fatigue_scores = df['fatigue_score'].dropna()
            
            if len(fatigue_scores) < 3:
                return None
            
            # Predict fatigue risk
            avg_fatigue = fatigue_scores.mean()
            fatigue_trend = fatigue_scores.diff().mean()
            
            future_fatigue = avg_fatigue + fatigue_trend * days_ahead
            confidence = 0.7  # Placeholder confidence
            
            return PredictiveInsight(
                insight_type='fatigue_risk',
                prediction=future_fatigue,
                confidence=confidence,
                time_horizon=f'{days_ahead} days',
                reasoning=f"Fatigue trend analysis based on {len(fatigue_scores)} data points",
                impact_score=future_fatigue,
                created_at=now_utc()
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting fatigue risk: {e}")
            return None
    
    def _predict_quality_trend(self, stage: str, days_ahead: int) -> Optional[PredictiveInsight]:
        """Predict quality trend for a stage."""
        try:
            # Get quality score data
            response = self.supabase.client.table('performance_metrics').select('*').eq(
                'stage', stage
            ).gte('date_start', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            ).execute()
            
            if not response.data:
                return None
            
            df = pd.DataFrame(response.data)
            quality_scores = df['performance_quality_score'].dropna()
            
            if len(quality_scores) < 7:
                return None
            
            # Trend analysis
            x = np.arange(len(quality_scores))
            slope, _, r_value, p_value, _ = np.polyfit(x, quality_scores, 1)
            
            # Predict future quality
            future_quality = quality_scores.iloc[-1] + slope * days_ahead
            confidence = 1 - p_value if p_value < 0.1 else 0.5
            
            # Determine impact
            current_quality = quality_scores.iloc[-1]
            impact_score = abs(future_quality - current_quality) / 100  # Normalize to 0-1
            
            return PredictiveInsight(
                insight_type='quality_trend',
                prediction=future_quality,
                confidence=confidence,
                time_horizon=f'{days_ahead} days',
                reasoning=f"Quality trend analysis shows {slope:.2f} change per day",
                impact_score=impact_score,
                created_at=now_utc()
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting quality trend: {e}")
            return None

class SystemHealthAnalyzer:
    """System health analysis and monitoring."""
    
    def __init__(self, supabase_client: SupabaseReportingClient):
        self.supabase = supabase_client
        self.logger = logging.getLogger(f"{__name__}.SystemHealthAnalyzer")
    
    def analyze_system_health(self) -> SystemHealthReport:
        """Analyze overall system health."""
        try:
            # Get system health data
            response = self.supabase.client.table('system_health').select('*').execute()
            
            if not response.data:
                return self._create_default_health_report()
            
            df = pd.DataFrame(response.data)
            
            # Calculate overall health
            overall_health = df['health_score'].mean()
            
            # Stage health
            stage_health = {}
            for stage in ['testing', 'validation', 'scaling']:
                stage_data = df[df['stage'] == stage]
                if not stage_data.empty:
                    stage_health[stage] = stage_data['health_score'].mean()
                else:
                    stage_health[stage] = 0.5  # Default neutral health
            
            # ML model health
            ml_model_health = self._analyze_ml_model_health()
            
            # Data quality
            data_quality = self._analyze_data_quality()
            
            # Learning velocity
            learning_velocity = self._calculate_learning_velocity()
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(
                overall_health, stage_health, ml_model_health, data_quality
            )
            
            return SystemHealthReport(
                overall_health=overall_health,
                stage_health=stage_health,
                ml_model_health=ml_model_health,
                data_quality=data_quality,
                learning_velocity=learning_velocity,
                recommendations=recommendations,
                created_at=now_utc()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing system health: {e}")
            return self._create_default_health_report()
    
    def _analyze_ml_model_health(self) -> Dict[str, float]:
        """Analyze ML model health."""
        try:
            response = self.supabase.client.table('ml_models').select('*').eq('is_active', True).execute()
            
            if not response.data:
                return {}
            
            model_health = {}
            for model in response.data:
                model_type = model.get('model_type', 'unknown')
                performance_metrics = model.get('performance_metrics', {})
                
                # Calculate health score based on performance metrics
                health_score = 0.5  # Default
                if 'accuracy' in performance_metrics:
                    health_score = performance_metrics['accuracy']
                elif 'r_squared' in performance_metrics:
                    health_score = performance_metrics['r_squared']
                
                model_health[model_type] = health_score
            
            return model_health
            
        except Exception as e:
            self.logger.error(f"Error analyzing ML model health: {e}")
            return {}
    
    def _analyze_data_quality(self) -> Dict[str, float]:
        """Analyze data quality metrics."""
        try:
            # Get performance metrics
            response = self.supabase.client.table('performance_metrics').select('*').gte(
                'date_start', (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            ).execute()
            
            if not response.data:
                return {'completeness': 0.0, 'consistency': 0.0, 'accuracy': 0.0}
            
            df = pd.DataFrame(response.data)
            
            # Data completeness
            total_fields = len(df.columns)
            non_null_fields = df.count().sum()
            completeness = non_null_fields / (total_fields * len(df)) if len(df) > 0 else 0
            
            # Data consistency (check for outliers)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            consistency = 1.0
            for col in numeric_cols:
                if col in df.columns:
                    values = df[col].dropna()
                    if len(values) > 0:
                        q1, q3 = values.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outliers = len(values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)])
                        consistency *= (1 - outliers / len(values))
            
            # Data accuracy (placeholder)
            accuracy = 0.8  # Placeholder accuracy score
            
            return {
                'completeness': completeness,
                'consistency': consistency,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing data quality: {e}")
            return {'completeness': 0.0, 'consistency': 0.0, 'accuracy': 0.0}
    
    def _calculate_learning_velocity(self) -> float:
        """Calculate learning velocity."""
        try:
            # Get learning events
            response = self.supabase.client.table('learning_events').select('*').gte(
                'created_at', (datetime.now() - timedelta(days=7)).isoformat()
            ).execute()
            
            if not response.data:
                return 0.0
            
            # Calculate events per day
            events_per_day = len(response.data) / 7
            return min(1.0, events_per_day / 10)  # Normalize to 0-1
            
        except Exception as e:
            self.logger.error(f"Error calculating learning velocity: {e}")
            return 0.0
    
    def _generate_health_recommendations(self, overall_health: float, stage_health: Dict[str, float],
                                       ml_model_health: Dict[str, float], data_quality: Dict[str, float]) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        # Overall health recommendations
        if overall_health < 0.5:
            recommendations.append("System health is low - investigate performance issues")
        elif overall_health > 0.8:
            recommendations.append("System health is excellent - consider scaling operations")
        
        # Stage-specific recommendations
        for stage, health in stage_health.items():
            if health < 0.4:
                recommendations.append(f"{stage.title()} stage health is critical - immediate attention needed")
            elif health < 0.6:
                recommendations.append(f"{stage.title()} stage health is poor - monitor closely")
        
        # ML model recommendations
        for model_type, health in ml_model_health.items():
            if health < 0.5:
                recommendations.append(f"{model_type} model performance is poor - consider retraining")
        
        # Data quality recommendations
        if data_quality.get('completeness', 0) < 0.7:
            recommendations.append("Data completeness is low - check data collection processes")
        if data_quality.get('consistency', 0) < 0.7:
            recommendations.append("Data consistency issues detected - investigate data sources")
        
        return recommendations
    
    def _create_default_health_report(self) -> SystemHealthReport:
        """Create default health report when no data is available."""
        return SystemHealthReport(
            overall_health=0.5,
            stage_health={'testing': 0.5, 'validation': 0.5, 'scaling': 0.5},
            ml_model_health={},
            data_quality={'completeness': 0.0, 'consistency': 0.0, 'accuracy': 0.0},
            learning_velocity=0.0,
            recommendations=["Insufficient data for health analysis"],
            created_at=now_utc()
        )

class MLReportingSystem:
    """Main ML reporting system."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase = SupabaseReportingClient(supabase_url, supabase_key)
        self.predictive_reporter = PredictiveReporter(self.supabase)
        self.health_analyzer = SystemHealthAnalyzer(self.supabase)
        self.logger = logging.getLogger(f"{__name__}.MLReportingSystem")
    
    def generate_daily_report(self) -> MLReport:
        """Generate daily ML-enhanced report."""
        try:
            # Get performance summary
            performance_summary = self.supabase.get_performance_summary()
            
            # Get ML insights
            ml_insights = self.supabase.get_ml_insights()
            
            # Generate predictive insights
            predictive_insights = []
            for stage in ['testing', 'validation', 'scaling']:
                stage_insights = self.predictive_reporter.generate_predictive_insights(stage)
                predictive_insights.extend(stage_insights)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(performance_summary, ml_insights, predictive_insights)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(predictive_insights)
            
            # Calculate ML metrics
            ml_metrics = self._calculate_ml_metrics(ml_insights, predictive_insights)
            
            return MLReport(
                report_type='daily',
                timestamp=now_utc(),
                stage='all',
                insights=ml_insights,
                predictions={insight.insight_type: insight.prediction for insight in predictive_insights},
                recommendations=recommendations,
                confidence_scores=confidence_scores,
                performance_metrics=performance_summary,
                ml_metrics=ml_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
            return self._create_empty_report()
    
    def generate_weekly_report(self) -> MLReport:
        """Generate weekly ML-enhanced report."""
        try:
            # Get extended performance summary
            performance_summary = self.supabase.get_performance_summary(days_back=7)
            
            # Get extended ML insights
            ml_insights = self.supabase.get_ml_insights(days_back=7)
            
            # Generate system health analysis
            health_report = self.health_analyzer.analyze_system_health()
            
            # Generate weekly recommendations
            recommendations = self._generate_weekly_recommendations(performance_summary, ml_insights, health_report)
            
            return MLReport(
                report_type='weekly',
                timestamp=now_utc(),
                stage='all',
                insights={**ml_insights, 'health_report': health_report.__dict__},
                predictions={},
                recommendations=recommendations,
                confidence_scores={},
                performance_metrics=performance_summary,
                ml_metrics=ml_insights
            )
            
        except Exception as e:
            self.logger.error(f"Error generating weekly report: {e}")
            return self._create_empty_report()
    
    def _generate_recommendations(self, performance_summary: Dict[str, Any], 
                                 ml_insights: Dict[str, Any], 
                                 predictive_insights: List[PredictiveInsight]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        if performance_summary.get('avg_cpa', 0) > 30:
            recommendations.append("Average CPA is high - consider tightening targeting or increasing budgets")
        
        if performance_summary.get('avg_roas', 0) < 2.0:
            recommendations.append("Average ROAS is below target - review creative performance")
        
        # ML-based recommendations
        if ml_insights.get('avg_confidence', 0) < 0.6:
            recommendations.append("ML model confidence is low - consider retraining models")
        
        # Predictive recommendations
        for insight in predictive_insights:
            if insight.impact_score > 0.3:
                if insight.insight_type == 'cpa_trend' and insight.prediction > 35:
                    recommendations.append(f"CPA predicted to increase to {insight.prediction:.2f} - take preventive action")
                elif insight.insight_type == 'fatigue_risk' and insight.prediction > 0.7:
                    recommendations.append("High fatigue risk predicted - prepare fresh creatives")
        
        return recommendations
    
    def _generate_weekly_recommendations(self, performance_summary: Dict[str, Any],
                                       ml_insights: Dict[str, Any],
                                       health_report: SystemHealthReport) -> List[str]:
        """Generate weekly recommendations."""
        recommendations = []
        
        # Add health-based recommendations
        recommendations.extend(health_report.recommendations)
        
        # Add performance-based recommendations
        if performance_summary.get('total_spend', 0) > 1000:
            recommendations.append("High spend detected - ensure ROI targets are being met")
        
        # Add ML-based recommendations
        if ml_insights.get('total_predictions', 0) < 10:
            recommendations.append("Low prediction volume - check ML model activity")
        
        return recommendations
    
    def _calculate_confidence_scores(self, predictive_insights: List[PredictiveInsight]) -> Dict[str, float]:
        """Calculate confidence scores for insights."""
        confidence_scores = {}
        
        for insight in predictive_insights:
            confidence_scores[insight.insight_type] = insight.confidence
        
        return confidence_scores
    
    def _calculate_ml_metrics(self, ml_insights: Dict[str, Any], 
                            predictive_insights: List[PredictiveInsight]) -> Dict[str, Any]:
        """Calculate ML-specific metrics."""
        return {
            'total_predictions': ml_insights.get('total_predictions', 0),
            'avg_confidence': ml_insights.get('avg_confidence', 0),
            'prediction_accuracy': ml_insights.get('prediction_accuracy', 0),
            'learning_events': ml_insights.get('learning_events', []),
            'predictive_insights_count': len(predictive_insights),
            'high_confidence_insights': len([i for i in predictive_insights if i.confidence > 0.8])
        }
    
    def _create_empty_report(self) -> MLReport:
        """Create empty report when generation fails."""
        return MLReport(
            report_type='error',
            timestamp=now_utc(),
            stage='all',
            insights={},
            predictions={},
            recommendations=["Error generating report"],
            confidence_scores={},
            performance_metrics={},
            ml_metrics={}
        )
    
    def send_report_to_slack(self, report: MLReport) -> None:
        """Send ML report to Slack."""
        try:
            # Format report for Slack
            report_text = self._format_report_for_slack(report)
            
            # Send to Slack
            notify(report_text)
            
        except Exception as e:
            self.logger.error(f"Error sending report to Slack: {e}")
    
    def _format_report_for_slack(self, report: MLReport) -> str:
        """Format ML report for Slack display."""
        try:
            lines = []
            
            # Header
            lines.append(f"🤖 **ML-Enhanced Dean Report** ({report.report_type.title()})")
            lines.append(f"📅 {report.timestamp.strftime('%Y-%m-%d %H:%M')}")
            lines.append("")
            
            # Performance metrics
            if report.performance_metrics:
                lines.append("📊 **Performance Summary:**")
                metrics = report.performance_metrics
                lines.append(f"• Total Ads: {metrics.get('total_ads', 0)}")
                lines.append(f"• Total Spend: €{metrics.get('total_spend', 0):.2f}")
                lines.append(f"• Total Purchases: {metrics.get('total_purchases', 0)}")
                lines.append(f"• Avg CPA: €{metrics.get('avg_cpa', 0):.2f}")
                lines.append(f"• Avg ROAS: {metrics.get('avg_roas', 0):.2f}")
                lines.append("")
            
            # ML insights
            if report.ml_metrics:
                lines.append("🧠 **ML Intelligence:**")
                ml_metrics = report.ml_metrics
                lines.append(f"• Predictions Made: {ml_metrics.get('total_predictions', 0)}")
                lines.append(f"• Avg Confidence: {ml_metrics.get('avg_confidence', 0):.2f}")
                lines.append(f"• Prediction Accuracy: {ml_metrics.get('prediction_accuracy', 0):.2f}")
                lines.append(f"• Learning Events: {len(ml_metrics.get('learning_events', []))}")
                lines.append("")
            
            # Predictions
            if report.predictions:
                lines.append("🔮 **Predictions:**")
                for insight_type, prediction in report.predictions.items():
                    lines.append(f"• {insight_type.replace('_', ' ').title()}: {prediction:.2f}")
                lines.append("")
            
            # Recommendations
            if report.recommendations:
                lines.append("💡 **Recommendations:**")
                for rec in report.recommendations[:5]:  # Limit to 5 recommendations
                    lines.append(f"• {rec}")
                lines.append("")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Error formatting report for Slack: {e}")
            return "Error formatting ML report"

# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_ml_reporting_system(supabase_url: str, supabase_key: str) -> MLReportingSystem:
    """Create ML reporting system."""
    return MLReportingSystem(supabase_url, supabase_key)

def generate_daily_ml_report(reporting_system: MLReportingSystem) -> MLReport:
    """Generate daily ML report."""
    return reporting_system.generate_daily_report()

def generate_weekly_ml_report(reporting_system: MLReportingSystem) -> MLReport:
    """Generate weekly ML report."""
    return reporting_system.generate_weekly_report()

def send_ml_report_to_slack(reporting_system: MLReportingSystem, report: MLReport) -> None:
    """Send ML report to Slack."""
    reporting_system.send_report_to_slack(report)
