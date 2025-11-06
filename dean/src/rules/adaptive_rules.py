"""
DEAN SELF-LEARNING META ADS AUTOMATION SYSTEM
Advanced Adaptive Rules Engine

This module implements the intelligent rule engine that dynamically adjusts
thresholds, budgets, and pacing based on ML insights and performance patterns.
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

from infrastructure.utils import now_utc, today_ymd_account
# Note: MLIntelligenceSystem imported lazily to avoid circular imports

logger = logging.getLogger(__name__)

# =====================================================
# ADAPTIVE RULES SYSTEM
# =====================================================

@dataclass
class RuleConfig:
    """Configuration for adaptive rules."""
    # Learning parameters
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.05
    confidence_weight: float = 0.8
    
    # Rule adjustment limits
    max_adjustment_pct: float = 0.2  # Max 20% adjustment per cycle
    min_threshold_value: float = 0.01
    max_threshold_value: float = 1000.0
    
    # Performance targets
    target_cpa: float = 27.50
    target_roas: float = 2.0
    target_ctr: float = 0.008
    
    # Stability requirements
    min_data_points: int = 7
    stability_window_days: int = 14
    confidence_threshold: float = 0.7

@dataclass
class AdaptiveRule:
    """Individual adaptive rule with learning capabilities."""
    rule_name: str
    stage: str
    rule_type: str
    current_value: float
    previous_value: float
    adjustment_reason: str
    confidence_weight: float
    learning_rate: float
    is_active: bool
    created_at: datetime
    updated_at: datetime
    performance_history: List[Dict[str, Any]]
    ml_insights: Dict[str, Any]

@dataclass
class RuleDecision:
    """Result of rule evaluation with ML insights."""
    rule_name: str
    decision: str  # 'kill', 'promote', 'scale_up', 'scale_down', 'hold'
    confidence: float
    reasoning: str
    ml_predictions: Dict[str, Any]
    threshold_adjustments: Dict[str, float]
    created_at: datetime

class SupabaseRulesClient:
    """Supabase client for adaptive rules management."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.logger = logging.getLogger(f"{__name__}.SupabaseRulesClient")
    
    def get_adaptive_rules(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch adaptive rules from database."""
        try:
            query = self.client.table('adaptive_rules').select('*')
            if stage:
                query = query.eq('stage', stage)
            
            response = query.execute()
            return response.data if response.data else []
            
        except Exception as e:
            self.logger.error(f"Error fetching adaptive rules: {e}")
            return []
    
    def update_rule(self, rule_id: str, new_value: float, 
                   adjustment_reason: str, performance_data: Dict[str, Any]) -> bool:
        """Update adaptive rule with new value and reasoning."""
        try:
            data = {
                'current_value': new_value,
                'adjustment_reason': adjustment_reason,
                'updated_at': now_utc().isoformat(),
                'performance_history': performance_data
            }
            
            response = self.client.table('adaptive_rules').update(data).eq('id', rule_id).execute()
            
            if response.data:
                self.logger.info(f"Updated rule {rule_id} to {new_value}")
                return True
            else:
                self.logger.error(f"Failed to update rule {rule_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating rule: {e}")
            return False
    
    def save_rule_decision(self, decision: RuleDecision) -> str:
        """Save rule decision to database."""
        try:
            data = {
                'rule_name': decision.rule_name,
                'decision': decision.decision,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'ml_predictions': decision.ml_predictions,
                'threshold_adjustments': decision.threshold_adjustments,
                'created_at': decision.created_at.isoformat()
            }
            
            response = self.client.table('rule_decisions').insert(data).execute()
            
            if response.data:
                decision_id = response.data[0]['id']
                self.logger.info(f"Saved rule decision {decision_id}")
                return decision_id
            else:
                self.logger.error("Failed to save rule decision")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving rule decision: {e}")
            return None

class PerformanceAnalyzer:
    """Advanced performance analysis for rule adaptation."""
    
    def __init__(self, supabase_client: SupabaseRulesClient):
        self.supabase = supabase_client
        self.logger = logging.getLogger(f"{__name__}.PerformanceAnalyzer")
    
    def analyze_stage_performance(self, stage: str, days_back: int = 14) -> Dict[str, Any]:
        """Analyze overall stage performance for rule adaptation."""
        try:
            # Get performance data
            from ml.ml_intelligence import SupabaseMLClient
            ml_client = SupabaseMLClient(
                self.supabase.client.supabase_url,
                self.supabase.client.supabase_key
            )
            
            df = ml_client.get_performance_data(stages=[stage], days_back=days_back)
            if df.empty:
                return {}
            
            # Calculate stage metrics
            stage_metrics = {
                'total_ads': df['ad_id'].nunique(),
                'avg_cpa': df['cpa'].mean(),
                'avg_roas': df['roas'].mean(),
                'avg_ctr': df['ctr'].mean(),
                'avg_quality_score': df['performance_quality_score'].mean(),
                'avg_stability': df['stability_score'].mean(),
                'avg_fatigue': df['fatigue_index'].mean(),
                'promotion_rate': self.calculate_promotion_rate(df),
                'kill_rate': self.calculate_kill_rate(df),
                'volatility': self.calculate_volatility(df)
            }
            
            # Trend analysis
            trends = self.analyze_trends(df)
            stage_metrics.update(trends)
            
            return stage_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing stage performance: {e}")
            return {}
    
    def calculate_promotion_rate(self, df: pd.DataFrame) -> float:
        """Calculate promotion rate from stage."""
        try:
            # This would need to be tracked in learning_events
            # For now, return a placeholder
            return 0.1  # 10% promotion rate
        except Exception as e:
            self.logger.error(f"Error calculating promotion rate: {e}")
            return 0.0
    
    def calculate_kill_rate(self, df: pd.DataFrame) -> float:
        """Calculate kill rate from stage."""
        try:
            # This would need to be tracked in learning_events
            # For now, return a placeholder
            return 0.2  # 20% kill rate
        except Exception as e:
            self.logger.error(f"Error calculating kill rate: {e}")
            return 0.0
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate performance volatility."""
        try:
            if len(df) < 2:
                return 0.0
            
            # Calculate coefficient of variation for key metrics
            metrics = ['cpa', 'roas', 'ctr']
            volatilities = []
            
            for metric in metrics:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 1:
                        cv = values.std() / (values.mean() + 1e-6)
                        volatilities.append(cv)
            
            return np.mean(volatilities) if volatilities else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance trends."""
        try:
            trends = {}
            
            # CPA trend
            if 'cpa' in df.columns and len(df) > 3:
                cpa_values = df['cpa'].dropna()
                if len(cpa_values) > 3:
                    x = np.arange(len(cpa_values))
                    slope, _, r_value, p_value, _ = np.polyfit(x, cpa_values, 1)
                    trends['cpa_trend'] = {
                        'direction': 'improving' if slope < 0 else 'declining',
                        'strength': abs(slope),
                        'confidence': 1 - p_value,
                        'r_squared': r_value ** 2
                    }
            
            # ROAS trend
            if 'roas' in df.columns and len(df) > 3:
                roas_values = df['roas'].dropna()
                if len(roas_values) > 3:
                    x = np.arange(len(roas_values))
                    slope, _, r_value, p_value, _ = np.polyfit(x, roas_values, 1)
                    trends['roas_trend'] = {
                        'direction': 'improving' if slope > 0 else 'declining',
                        'strength': abs(slope),
                        'confidence': 1 - p_value,
                        'r_squared': r_value ** 2
                    }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return {}

class RuleAdaptationEngine:
    """Engine for adapting rules based on performance and ML insights."""
    
    def __init__(self, config: RuleConfig, supabase_client: SupabaseRulesClient,
                 ml_system: 'MLIntelligenceSystem'):
        self.config = config
        self.supabase = supabase_client
        self.ml_system = ml_system
        self.performance_analyzer = PerformanceAnalyzer(supabase_client)
        self.logger = logging.getLogger(f"{__name__}.RuleAdaptationEngine")
    
    def adapt_rule(self, rule: Dict[str, Any], stage_performance: Dict[str, Any],
                  ml_predictions: Dict[str, Any]) -> Optional[AdaptiveRule]:
        """Adapt a rule based on performance and ML insights."""
        try:
            rule_name = rule['rule_name']
            current_value = rule['current_value']
            rule_type = rule['rule_type']
            
            # Calculate adaptation
            adaptation = self.calculate_adaptation(
                rule_name, current_value, stage_performance, ml_predictions
            )
            
            if abs(adaptation) < self.config.adaptation_threshold:
                # No significant change needed
                return None
            
            # Apply bounds
            new_value = max(
                self.config.min_threshold_value,
                min(self.config.max_threshold_value, current_value + adaptation)
            )
            
            # Create adaptive rule
            adaptive_rule = AdaptiveRule(
                rule_name=rule_name,
                stage=rule['stage'],
                rule_type=rule_type,
                current_value=new_value,
                previous_value=current_value,
                adjustment_reason=self.generate_adjustment_reason(
                    rule_name, adaptation, stage_performance, ml_predictions
                ),
                confidence_weight=self.calculate_confidence_weight(
                    stage_performance, ml_predictions
                ),
                learning_rate=self.config.learning_rate,
                is_active=True,
                created_at=now_utc(),
                updated_at=now_utc(),
                performance_history=[stage_performance],
                ml_insights=ml_predictions
            )
            
            return adaptive_rule
            
        except Exception as e:
            self.logger.error(f"Error adapting rule {rule.get('rule_name', 'unknown')}: {e}")
            return None
    
    def calculate_adaptation(self, rule_name: str, current_value: float,
                           stage_performance: Dict[str, Any],
                           ml_predictions: Dict[str, Any]) -> float:
        """Calculate how much to adjust a rule."""
        try:
            adaptation = 0.0
            
            # CPA threshold adaptation
            if 'cpa' in rule_name.lower():
                target_cpa = self.config.target_cpa
                current_cpa = stage_performance.get('avg_cpa', target_cpa)
                
                # Adjust based on performance vs target
                if current_cpa > target_cpa * 1.1:  # 10% above target
                    adaptation = current_value * 0.05  # Tighten by 5%
                elif current_cpa < target_cpa * 0.9:  # 10% below target
                    adaptation = -current_value * 0.03  # Loosen by 3%
                
                # ML prediction influence
                if 'cpa_prediction' in ml_predictions:
                    predicted_cpa = ml_predictions['cpa_prediction']
                    if predicted_cpa > target_cpa * 1.2:
                        adaptation += current_value * 0.02  # Additional tightening
            
            # ROAS threshold adaptation
            elif 'roas' in rule_name.lower():
                target_roas = self.config.target_roas
                current_roas = stage_performance.get('avg_roas', target_roas)
                
                if current_roas < target_roas * 0.8:  # 20% below target
                    adaptation = current_value * 0.1  # Tighten by 10%
                elif current_roas > target_roas * 1.5:  # 50% above target
                    adaptation = -current_value * 0.05  # Loosen by 5%
            
            # CTR threshold adaptation
            elif 'ctr' in rule_name.lower():
                target_ctr = self.config.target_ctr
                current_ctr = stage_performance.get('avg_ctr', target_ctr)
                
                if current_ctr < target_ctr * 0.7:  # 30% below target
                    adaptation = -current_value * 0.1  # Loosen by 10%
                elif current_ctr > target_ctr * 1.3:  # 30% above target
                    adaptation = current_value * 0.05  # Tighten by 5%
            
            # Quality score adaptation
            elif 'quality' in rule_name.lower():
                current_quality = stage_performance.get('avg_quality_score', 50)
                
                if current_quality < 40:  # Low quality
                    adaptation = -current_value * 0.15  # Loosen significantly
                elif current_quality > 80:  # High quality
                    adaptation = current_value * 0.1  # Tighten
            
            # Apply learning rate
            adaptation *= self.config.learning_rate
            
            # Apply confidence weighting
            confidence = self.calculate_confidence_weight(stage_performance, ml_predictions)
            adaptation *= confidence
            
            return adaptation
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptation for {rule_name}: {e}")
            return 0.0
    
    def calculate_confidence_weight(self, stage_performance: Dict[str, Any],
                                  ml_predictions: Dict[str, Any]) -> float:
        """Calculate confidence weight for rule adaptation."""
        try:
            confidence = 1.0
            
            # Data quality factor
            total_ads = stage_performance.get('total_ads', 0)
            if total_ads < self.config.min_data_points:
                confidence *= 0.5  # Low confidence with few data points
            
            # Volatility factor
            volatility = stage_performance.get('volatility', 0)
            if volatility > 0.5:  # High volatility
                confidence *= 0.7  # Reduce confidence
            
            # ML prediction confidence
            if 'confidence_score' in ml_predictions:
                ml_confidence = ml_predictions['confidence_score']
                confidence *= ml_confidence
            
            # Trend stability
            cpa_trend = stage_performance.get('cpa_trend', {})
            if cpa_trend.get('confidence', 0) > 0.8:
                confidence *= 1.1  # Boost confidence for stable trends
            
            return min(1.0, max(0.1, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence weight: {e}")
            return 0.5
    
    def generate_adjustment_reason(self, rule_name: str, adaptation: float,
                                 stage_performance: Dict[str, Any],
                                 ml_predictions: Dict[str, Any]) -> str:
        """Generate human-readable reason for rule adjustment."""
        try:
            reasons = []
            
            if abs(adaptation) < 0.01:
                return "No significant adjustment needed"
            
            direction = "tightened" if adaptation > 0 else "loosened"
            pct_change = abs(adaptation / stage_performance.get('avg_cpa', 1)) * 100
            
            # Performance-based reasons
            if 'cpa' in rule_name.lower():
                current_cpa = stage_performance.get('avg_cpa', 0)
                target_cpa = self.config.target_cpa
                
                if current_cpa > target_cpa * 1.1:
                    reasons.append(f"CPA {current_cpa:.2f} above target {target_cpa:.2f}")
                elif current_cpa < target_cpa * 0.9:
                    reasons.append(f"CPA {current_cpa:.2f} below target {target_cpa:.2f}")
            
            # ML prediction reasons
            if 'cpa_prediction' in ml_predictions:
                predicted_cpa = ml_predictions['cpa_prediction']
                if predicted_cpa > self.config.target_cpa * 1.2:
                    reasons.append(f"ML predicts high CPA {predicted_cpa:.2f}")
            
            # Volatility reasons
            volatility = stage_performance.get('volatility', 0)
            if volatility > 0.5:
                reasons.append(f"High volatility {volatility:.2f}")
            
            # Trend reasons
            cpa_trend = stage_performance.get('cpa_trend', {})
            if cpa_trend.get('direction') == 'declining':
                reasons.append("CPA trend declining")
            elif cpa_trend.get('direction') == 'improving':
                reasons.append("CPA trend improving")
            
            base_reason = f"Rule {direction} by {pct_change:.1f}%"
            if reasons:
                base_reason += f" due to: {', '.join(reasons)}"
            
            return base_reason
            
        except Exception as e:
            self.logger.error(f"Error generating adjustment reason: {e}")
            return f"Rule adjusted by {adaptation:.4f}"

class IntelligentRuleEngine:
    """Main intelligent rule engine with ML integration."""
    
    def __init__(self, supabase_url: str, supabase_key: str,
                 ml_system: 'MLIntelligenceSystem', config: Optional[RuleConfig] = None):
        self.config = config or RuleConfig()
        self.supabase = SupabaseRulesClient(supabase_url, supabase_key)
        self.ml_system = ml_system
        self.adaptation_engine = RuleAdaptationEngine(
            self.config, self.supabase, ml_system
        )
        self.logger = logging.getLogger(f"{__name__}.IntelligentRuleEngine")
    
    def evaluate_ad_decision(self, ad_id: str, stage: str, 
                           performance_data: Dict[str, Any]) -> RuleDecision:
        """Evaluate ad decision using intelligent rules."""
        try:
            # Get ML predictions
            ml_insights = self.ml_system.analyze_ad_intelligence(ad_id, stage)
            predictions = ml_insights.get('predictions', {})
            
            # Get current rules
            rules = self.supabase.get_adaptive_rules(stage)
            
            # Evaluate each rule
            decisions = []
            threshold_adjustments = {}
            
            for rule in rules:
                decision = self.evaluate_single_rule(
                    rule, performance_data, predictions
                )
                if decision:
                    decisions.append(decision)
                    threshold_adjustments[rule['rule_name']] = decision.get('new_threshold')
            
            # Determine overall decision
            overall_decision = self.determine_overall_decision(decisions, predictions)
            
            # Create rule decision
            rule_decision = RuleDecision(
                rule_name=f"ad_{ad_id}_decision",
                decision=overall_decision['action'],
                confidence=overall_decision['confidence'],
                reasoning=overall_decision['reasoning'],
                ml_predictions=predictions,
                threshold_adjustments=threshold_adjustments,
                created_at=now_utc()
            )
            
            # Save decision
            self.supabase.save_rule_decision(rule_decision)
            
            return rule_decision
            
        except Exception as e:
            self.logger.error(f"Error evaluating ad decision: {e}")
            return RuleDecision(
                rule_name=f"ad_{ad_id}_decision",
                decision='hold',
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                ml_predictions={},
                threshold_adjustments={},
                created_at=now_utc()
            )
    
    def evaluate_single_rule(self, rule: Dict[str, Any], 
                           performance_data: Dict[str, Any],
                           ml_predictions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a single rule against performance data."""
        try:
            rule_name = rule['rule_name']
            rule_type = rule['rule_type']
            current_threshold = rule['current_value']
            
            # Get relevant performance metric
            metric_value = self.get_metric_value(rule_name, performance_data)
            if metric_value is None:
                return None
            
            # Apply rule logic
            decision = self.apply_rule_logic(
                rule_name, rule_type, metric_value, current_threshold,
                performance_data, ml_predictions
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.get('rule_name', 'unknown')}: {e}")
            return None
    
    def get_metric_value(self, rule_name: str, performance_data: Dict[str, Any]) -> Optional[float]:
        """Get the relevant metric value for a rule."""
        try:
            rule_name_lower = rule_name.lower()
            
            if 'cpa' in rule_name_lower:
                return performance_data.get('cpa')
            elif 'roas' in rule_name_lower:
                return performance_data.get('roas')
            elif 'ctr' in rule_name_lower:
                return performance_data.get('ctr')
            elif 'quality' in rule_name_lower:
                return performance_data.get('performance_quality_score')
            elif 'stability' in rule_name_lower:
                return performance_data.get('stability_score')
            elif 'fatigue' in rule_name_lower:
                return performance_data.get('fatigue_index')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting metric value for {rule_name}: {e}")
            return None
    
    def apply_rule_logic(self, rule_name: str, rule_type: str, 
                        metric_value: float, threshold: float,
                        performance_data: Dict[str, Any],
                        ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rule logic to determine action."""
        try:
            # Basic threshold comparison
            if rule_type == 'threshold':
                if 'cpa' in rule_name.lower():
                    # For CPA, lower is better
                    if metric_value > threshold:
                        return {
                            'action': 'kill',
                            'confidence': 0.8,
                            'reasoning': f"CPA {metric_value:.2f} exceeds threshold {threshold:.2f}",
                            'new_threshold': threshold
                        }
                elif 'roas' in rule_name.lower():
                    # For ROAS, higher is better
                    if metric_value < threshold:
                        return {
                            'action': 'kill',
                            'confidence': 0.8,
                            'reasoning': f"ROAS {metric_value:.2f} below threshold {threshold:.2f}",
                            'new_threshold': threshold
                        }
                elif 'ctr' in rule_name.lower():
                    # For CTR, higher is better
                    if metric_value < threshold:
                        return {
                            'action': 'kill',
                            'confidence': 0.7,
                            'reasoning': f"CTR {metric_value:.4f} below threshold {threshold:.4f}",
                            'new_threshold': threshold
                        }
            
            # ML-enhanced decisions
            if ml_predictions:
                predicted_cpa = ml_predictions.get('predicted_value')
                confidence = ml_predictions.get('confidence_score', 0)
                
                if predicted_cpa and confidence > 0.7:
                    if predicted_cpa > self.config.target_cpa * 1.3:
                        return {
                            'action': 'kill',
                            'confidence': confidence,
                            'reasoning': f"ML predicts high CPA {predicted_cpa:.2f}",
                            'new_threshold': threshold
                        }
                    elif predicted_cpa < self.config.target_cpa * 0.8:
                        return {
                            'action': 'promote',
                            'confidence': confidence,
                            'reasoning': f"ML predicts low CPA {predicted_cpa:.2f}",
                            'new_threshold': threshold
                        }
            
            # Default: hold
            return {
                'action': 'hold',
                'confidence': 0.5,
                'reasoning': f"Metric {metric_value:.4f} within acceptable range",
                'new_threshold': threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error applying rule logic: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': f"Error in rule logic: {str(e)}",
                'new_threshold': threshold
            }
    
    def determine_overall_decision(self, decisions: List[Dict[str, Any]],
                                 ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall decision from individual rule decisions."""
        try:
            if not decisions:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'reasoning': 'No rule decisions available'
                }
            
            # Count actions
            action_counts = {}
            total_confidence = 0
            
            for decision in decisions:
                action = decision['action']
                confidence = decision['confidence']
                
                action_counts[action] = action_counts.get(action, 0) + 1
                total_confidence += confidence
            
            # Determine primary action
            primary_action = max(action_counts, key=action_counts.get)
            avg_confidence = total_confidence / len(decisions)
            
            # Generate reasoning
            reasoning_parts = []
            for action, count in action_counts.items():
                reasoning_parts.append(f"{count} rules suggest {action}")
            
            reasoning = f"Overall decision: {primary_action} (confidence: {avg_confidence:.2f}). " + \
                       "; ".join(reasoning_parts)
            
            # ML prediction override
            if ml_predictions and ml_predictions.get('confidence_score', 0) > 0.8:
                predicted_cpa = ml_predictions.get('predicted_value')
                if predicted_cpa and predicted_cpa > self.config.target_cpa * 1.5:
                    primary_action = 'kill'
                    reasoning += f" ML override: predicted CPA {predicted_cpa:.2f} too high"
                elif predicted_cpa and predicted_cpa < self.config.target_cpa * 0.7:
                    primary_action = 'promote'
                    reasoning += f" ML override: predicted CPA {predicted_cpa:.2f} very low"
            
            return {
                'action': primary_action,
                'confidence': avg_confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            self.logger.error(f"Error determining overall decision: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)}"
            }
    
    def adapt_stage_rules(self, stage: str) -> bool:
        """Adapt all rules for a stage based on performance."""
        try:
            # Get stage performance
            stage_performance = self.adaptation_engine.performance_analyzer.analyze_stage_performance(stage)
            if not stage_performance:
                self.logger.warning(f"No performance data for stage {stage}")
                return False
            
            # Get ML predictions for stage
            ml_predictions = {
                'stage_performance': stage_performance,
                'confidence_score': 0.8  # Placeholder
            }
            
            # Get current rules
            rules = self.supabase.get_adaptive_rules(stage)
            if not rules:
                self.logger.warning(f"No rules found for stage {stage}")
                return False
            
            # Adapt each rule
            adapted_count = 0
            for rule in rules:
                adaptive_rule = self.adaptation_engine.adapt_rule(
                    rule, stage_performance, ml_predictions
                )
                
                if adaptive_rule:
                    # Update rule in database
                    success = self.supabase.update_rule(
                        rule['id'],
                        adaptive_rule.current_value,
                        adaptive_rule.adjustment_reason,
                        adaptive_rule.performance_history[0]
                    )
                    
                    if success:
                        adapted_count += 1
                        self.logger.info(f"Adapted rule {rule['rule_name']} to {adaptive_rule.current_value}")
            
            self.logger.info(f"Adapted {adapted_count}/{len(rules)} rules for stage {stage}")
            return adapted_count > 0
            
        except Exception as e:
            self.logger.error(f"Error adapting stage rules for {stage}: {e}")
            return False

# =====================================================
# ADVANCED RULE MINING AND GENERATION
# =====================================================

class RuleMiner:
    """Mines rules from historical performance data."""
    
    def __init__(self, min_support: float = 0.3, min_confidence: float = 0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(f"{__name__}.RuleMiner")
    
    def mine_rules(
        self,
        performance_data: pd.DataFrame,
        target_metric: str = "should_kill",
    ) -> List[Dict[str, Any]]:
        """Mine rules from performance data using association rule mining."""
        if performance_data.empty or len(performance_data) < 20:
            return []
        
        try:
            rules = []
            
            # Discretize continuous features
            df_discretized = self._discretize_features(performance_data.copy())
            
            # Mine rules for kill decisions
            if target_metric == "should_kill":
                rules.extend(self._mine_kill_rules(df_discretized))
            
            # Mine rules for promotion decisions
            if target_metric == "should_promote":
                rules.extend(self._mine_promote_rules(df_discretized))
            
            # Filter by support and confidence
            filtered_rules = [
                r for r in rules
                if r.get("support", 0) >= self.min_support
                and r.get("confidence", 0) >= self.min_confidence
            ]
            
            self.logger.info(f"Mined {len(filtered_rules)} rules from {len(performance_data)} records")
            return filtered_rules
            
        except Exception as e:
            self.logger.error(f"Rule mining failed: {e}")
            return []
    
    def _discretize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discretize continuous features for rule mining."""
        # Discretize CPA
        if "cpa" in df.columns:
            df["cpa_level"] = pd.cut(
                df["cpa"],
                bins=[0, 20, 30, 40, 1000],
                labels=["low", "medium", "high", "very_high"],
            )
        
        # Discretize ROAS
        if "roas" in df.columns:
            df["roas_level"] = pd.cut(
                df["roas"],
                bins=[0, 0.8, 1.5, 2.5, 100],
                labels=["very_low", "low", "medium", "high"],
            )
        
        # Discretize CTR
        if "ctr" in df.columns:
            df["ctr_level"] = pd.cut(
                df["ctr"] * 100,  # Convert to percentage
                bins=[0, 0.5, 1.0, 2.0, 100],
                labels=["very_low", "low", "medium", "high"],
            )
        
        return df
    
    def _mine_kill_rules(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Mine rules for kill decisions."""
        rules = []
        
        # Rule: High CPA -> Kill
        if "cpa_level" in df.columns:
            high_cpa = df[df["cpa_level"].isin(["high", "very_high"])]
            if len(high_cpa) > 0:
                support = len(high_cpa) / len(df)
                # Assume high CPA leads to kill (would need actual kill data)
                confidence = 0.8  # Placeholder
                rules.append({
                    "condition": "cpa_level in ['high', 'very_high']",
                    "action": "kill",
                    "support": support,
                    "confidence": confidence,
                    "rule_type": "cpa_based",
                })
        
        # Rule: Low ROAS + Low CTR -> Kill
        if "roas_level" in df.columns and "ctr_level" in df.columns:
            low_performance = df[
                (df["roas_level"].isin(["very_low", "low"])) &
                (df["ctr_level"].isin(["very_low", "low"]))
            ]
            if len(low_performance) > 0:
                support = len(low_performance) / len(df)
                confidence = 0.85
                rules.append({
                    "condition": "roas_level in ['very_low', 'low'] AND ctr_level in ['very_low', 'low']",
                    "action": "kill",
                    "support": support,
                    "confidence": confidence,
                    "rule_type": "performance_based",
                })
        
        return rules
    
    def _mine_promote_rules(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Mine rules for promotion decisions."""
        rules = []
        
        # Rule: High ROAS + Low Spend -> Promote
        if "roas_level" in df.columns and "spend" in df.columns:
            high_roas_low_spend = df[
                (df["roas_level"] == "high") &
                (df["spend"] < df["spend"].quantile(0.5))
            ]
            if len(high_roas_low_spend) > 0:
                support = len(high_roas_low_spend) / len(df)
                confidence = 0.75
                rules.append({
                    "condition": "roas_level == 'high' AND spend < median",
                    "action": "promote",
                    "support": support,
                    "confidence": confidence,
                    "rule_type": "scaling_opportunity",
                })
        
        return rules


class DynamicRuleGenerator:
    """Dynamically generates rules based on patterns."""
    
    def __init__(self):
        self.generated_rules: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"{__name__}.DynamicRuleGenerator")
    
    def generate_rules(
        self,
        performance_patterns: Dict[str, Any],
        market_conditions: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate rules based on current patterns and market conditions."""
        rules = []
        
        # Generate rules based on market conditions
        if market_conditions.get("regime") == "volatile":
            rules.append({
                "rule_name": "volatile_market_ctr_threshold",
                "rule_type": "ctr_below",
                "threshold": 0.006,  # Lower threshold in volatile market
                "reason": "Market volatility detected - adjusting CTR threshold",
                "confidence": 0.7,
            })
        
        # Generate rules based on performance patterns
        if performance_patterns.get("avg_roas", 0) > 2.5:
            rules.append({
                "rule_name": "high_performance_roas_floor",
                "rule_type": "roas_below",
                "threshold": 2.0,  # Higher floor for high-performing campaigns
                "reason": "High average ROAS - raising minimum threshold",
                "confidence": 0.8,
            })
        
        # Generate rules based on trend
        if performance_patterns.get("trend") == "declining":
            rules.append({
                "rule_name": "declining_trend_cpa_ceiling",
                "rule_type": "cpa_gte",
                "threshold": 35.0,  # Tighter CPA in declining trend
                "reason": "Declining trend detected - tightening CPA threshold",
                "confidence": 0.75,
            })
        
        self.generated_rules.extend(rules)
        return rules


class RuleConfidenceScorer:
    """Scores rule confidence based on historical performance."""
    
    def __init__(self):
        self.rule_performance_history: Dict[str, List[bool]] = {}
        self.logger = logging.getLogger(f"{__name__}.RuleConfidenceScorer")
    
    def score_rule_confidence(
        self,
        rule: Dict[str, Any],
        historical_outcomes: List[Dict[str, Any]],
    ) -> float:
        """Calculate confidence score for a rule."""
        rule_name = rule.get("rule_name", "unknown")
        
        if not historical_outcomes:
            return 0.5  # Default confidence
        
        # Calculate accuracy
        correct_predictions = 0
        total_predictions = 0
        
        for outcome in historical_outcomes:
            rule_decision = outcome.get("rule_decision")
            actual_outcome = outcome.get("actual_outcome")
            
            if rule_decision and actual_outcome:
                if rule_decision == actual_outcome:
                    correct_predictions += 1
                total_predictions += 1
        
        if total_predictions == 0:
            return 0.5
        
        accuracy = correct_predictions / total_predictions
        
        # Adjust for sample size
        sample_size_factor = min(1.0, total_predictions / 50.0)
        
        confidence = accuracy * sample_size_factor
        
        # Store for tracking
        if rule_name not in self.rule_performance_history:
            self.rule_performance_history[rule_name] = []
        self.rule_performance_history[rule_name].append(confidence > 0.7)
        
        return confidence
    
    def get_rule_reliability(self, rule_name: str) -> float:
        """Get overall reliability of a rule."""
        if rule_name not in self.rule_performance_history:
            return 0.5
        
        history = self.rule_performance_history[rule_name]
        if not history:
            return 0.5
        
        reliability = sum(history) / len(history)
        return reliability


class RuleConflictResolver:
    """Resolves conflicts between competing rules."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RuleConflictResolver")
    
    def resolve_conflicts(
        self,
        rule_decisions: List[RuleDecision],
        confidence_scorer: Optional[RuleConfidenceScorer] = None,
    ) -> RuleDecision:
        """Resolve conflicts between multiple rule decisions."""
        if not rule_decisions:
            return RuleDecision(
                rule_name="no_rules",
                decision="hold",
                confidence=0.0,
                reasoning="No rules to evaluate",
                ml_predictions={},
                threshold_adjustments={},
                created_at=now_utc(),
            )
        
        # Group by decision
        decision_groups: Dict[str, List[RuleDecision]] = {}
        for decision in rule_decisions:
            decision_type = decision.decision
            if decision_type not in decision_groups:
                decision_groups[decision_type] = []
            decision_groups[decision_type].append(decision)
        
        # Score each decision group
        scored_groups = []
        for decision_type, decisions in decision_groups.items():
            # Average confidence
            avg_confidence = np.mean([d.confidence for d in decisions])
            
            # Weight by rule confidence if scorer available
            if confidence_scorer:
                rule_confidences = [
                    confidence_scorer.get_rule_reliability(d.rule_name)
                    for d in decisions
                ]
                weighted_confidence = np.mean([
                    d.confidence * rc
                    for d, rc in zip(decisions, rule_confidences)
                ])
                avg_confidence = weighted_confidence
            
            # Count votes
            vote_count = len(decisions)
            
            scored_groups.append({
                "decision": decision_type,
                "confidence": avg_confidence,
                "votes": vote_count,
                "decisions": decisions,
            })
        
        # Select best decision
        if not scored_groups:
            return rule_decisions[0]
        
        # Sort by confidence and votes
        scored_groups.sort(
            key=lambda x: (x["confidence"], x["votes"]),
            reverse=True,
        )
        
        best_group = scored_groups[0]
        
        # Combine reasoning
        reasoning_parts = [d.reasoning for d in best_group["decisions"]]
        combined_reasoning = f"{best_group['decision']} (confidence: {best_group['confidence']:.2f}, votes: {best_group['votes']})"
        combined_reasoning += ". " + "; ".join(reasoning_parts[:3])  # Limit to 3 reasons
        
        # Combine ML predictions
        combined_ml = {}
        for decision in best_group["decisions"]:
            combined_ml.update(decision.ml_predictions)
        
        # Combine threshold adjustments
        combined_thresholds = {}
        for decision in best_group["decisions"]:
            combined_thresholds.update(decision.threshold_adjustments)
        
        return RuleDecision(
            rule_name="resolved_conflict",
            decision=best_group["decision"],
            confidence=best_group["confidence"],
            reasoning=combined_reasoning,
            ml_predictions=combined_ml,
            threshold_adjustments=combined_thresholds,
            created_at=now_utc(),
        )
    
    def detect_conflicts(self, rule_decisions: List[RuleDecision]) -> List[Dict[str, Any]]:
        """Detect conflicts between rule decisions."""
        conflicts = []
        
        # Group by decision type
        decision_types = set(d.decision for d in rule_decisions)
        
        if len(decision_types) > 1:
            # Conflict detected
            conflict_details = {}
            for decision_type in decision_types:
                matching_decisions = [d for d in rule_decisions if d.decision == decision_type]
                conflict_details[decision_type] = {
                    "count": len(matching_decisions),
                    "avg_confidence": np.mean([d.confidence for d in matching_decisions]),
                    "rules": [d.rule_name for d in matching_decisions],
                }
            
            conflicts.append({
                "conflict_type": "decision_mismatch",
                "details": conflict_details,
                "severity": "high" if len(decision_types) >= 3 else "medium",
            })
        
        return conflicts


# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_intelligent_rule_engine(supabase_url: str, supabase_key: str,
                                 ml_system: 'MLIntelligenceSystem',
                                 config: Optional[RuleConfig] = None) -> IntelligentRuleEngine:
    """Create intelligent rule engine with ML integration."""
    return IntelligentRuleEngine(supabase_url, supabase_key, ml_system, config)

def evaluate_ad_with_ml(rule_engine: IntelligentRuleEngine, ad_id: str, stage: str,
                       performance_data: Dict[str, Any]) -> RuleDecision:
    """Evaluate ad decision using ML-enhanced rules."""
    return rule_engine.evaluate_ad_decision(ad_id, stage, performance_data)

def adapt_stage_rules(rule_engine: IntelligentRuleEngine, stage: str) -> bool:
    """Adapt all rules for a stage based on performance."""
    return rule_engine.adapt_stage_rules(stage)
