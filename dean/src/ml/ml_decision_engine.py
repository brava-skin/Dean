"""
DEAN ML DECISION ENGINE
Integrates ML predictions into actual automation decisions

This module bridges the gap between ML predictions and automation actions,
ensuring ML insights directly influence kill/promote/scale decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from infrastructure.utils import now_utc

logger = logging.getLogger(__name__)

# =====================================================
# ML-ENHANCED DECISION ENGINE
# =====================================================

@dataclass
class MLDecision:
    """ML-enhanced decision with reasoning."""
    action: str  # 'kill', 'promote', 'scale_up', 'scale_down', 'hold', 'monitor'
    confidence: float
    reasoning: str
    ml_predictions: Dict[str, Any]
    rule_based_decision: str
    final_decision: str
    ml_influence_pct: float
    created_at: datetime

class MLDecisionEngine:
    """Combines ML predictions with rule-based logic for optimal decisions."""
    
    def __init__(self, ml_system, rule_engine, confidence_threshold: float = 0.7):
        self.ml_system = ml_system
        self.rule_engine = rule_engine
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(f"{__name__}.MLDecisionEngine")
    
    def should_kill_ad(self, ad_id: str, stage: str, performance_data: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Enhanced kill decision using ML predictions."""
        try:
            # Get rule-based decision first
            from rules import Metrics
            
            metrics = Metrics(
                cpa=performance_data.get('cpa'),
                roas=performance_data.get('roas'),
                ctr=performance_data.get('ctr'),
                spend=performance_data.get('spend', 0),
                purchases=performance_data.get('purchases', 0),
                impressions=performance_data.get('impressions', 0),
                clicks=performance_data.get('clicks', 0),
                atc=performance_data.get('atc', 0),
                ic=performance_data.get('ic', 0)
            )
            
            # Rule-based decision
            if stage == 'testing':
                rule_kill, rule_reason = self.rule_engine.should_kill_testing(metrics)
            elif stage == 'validation':
                rule_kill, rule_reason = self.rule_engine.should_kill_validation(metrics)
            else:
                rule_kill = False
                rule_reason = "No rule check"
            
            # Get ML predictions
            ml_analysis = self.ml_system.analyze_ad_intelligence(ad_id, stage)
            
            if not ml_analysis or not ml_analysis.get('predictions'):
                # No ML data, use rules only
                return rule_kill, rule_reason, 0.5
            
            predictions = ml_analysis.get('predictions', {})
            confidence = predictions.get('confidence_score', 0)
            
            # ML-based decision factors
            predicted_cpa = predictions.get('predicted_value', 0)
            fatigue_score = ml_analysis.get('fatigue_analysis', {}).get('fatigue_score', 0)
            
            # Determine ML recommendation
            ml_recommends_kill = False
            ml_reasoning = ""
            
            # High confidence ML prediction
            if confidence > self.confidence_threshold:
                # Predict poor future performance
                if stage == 'testing' and predicted_cpa > 45:
                    ml_recommends_kill = True
                    ml_reasoning = f"ML predicts CPA will reach €{predicted_cpa:.2f} (confidence: {confidence:.2%})"
                elif stage == 'validation' and predicted_cpa > 55:
                    ml_recommends_kill = True
                    ml_reasoning = f"ML predicts CPA will reach €{predicted_cpa:.2f} (confidence: {confidence:.2%})"
                
                # Fatigue detected
                if fatigue_score > 0.7:
                    ml_recommends_kill = True
                    ml_reasoning += f" + High fatigue ({fatigue_score:.2%})"
            
            # Combined decision logic
            if confidence > self.confidence_threshold:
                # High confidence ML - give ML more weight
                if ml_recommends_kill and rule_kill:
                    # Both agree to kill - high confidence
                    final_decision = True
                    final_reason = f"Rules + ML agree: {rule_reason}. {ml_reasoning}"
                    ml_influence = 0.5
                elif ml_recommends_kill and not rule_kill:
                    # ML says kill but rules say keep - trust ML if very confident
                    if confidence > 0.85:
                        final_decision = True
                        final_reason = f"ML override: {ml_reasoning} (confidence: {confidence:.2%})"
                        ml_influence = 0.9
                    else:
                        final_decision = False
                        final_reason = f"Rules say keep (ML confidence only {confidence:.2%})"
                        ml_influence = 0.3
                elif not ml_recommends_kill and rule_kill:
                    # Rules say kill but ML says keep - override rules if ML confident
                    if confidence > 0.85:
                        final_decision = False
                        final_reason = f"ML override: Predicted improvement (confidence: {confidence:.2%})"
                        ml_influence = 0.9
                    else:
                        final_decision = True
                        final_reason = f"Rules: {rule_reason} (ML not confident enough to override)"
                        ml_influence = 0.2
                else:
                    # Both agree to keep
                    final_decision = False
                    final_reason = f"Rules + ML agree: Keep running"
                    ml_influence = 0.5
            else:
                # Low ML confidence - use rules primarily
                final_decision = rule_kill
                final_reason = f"Rules: {rule_reason} (ML confidence low: {confidence:.2%})"
                ml_influence = 0.1
            
            return final_decision, final_reason, ml_influence
            
        except Exception as e:
            self.logger.error(f"Error in ML-enhanced kill decision: {e}")
            # Fallback to rules
            return rule_kill, f"Rules: {rule_reason} (ML error)", 0.0
    
    def should_promote_ad(self, ad_id: str, stage: str, performance_data: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Enhanced promotion decision using ML predictions."""
        try:
            # Get rule-based decision
            from rules import Metrics
            
            metrics = Metrics(
                cpa=performance_data.get('cpa'),
                roas=performance_data.get('roas'),
                ctr=performance_data.get('ctr'),
                spend=performance_data.get('spend', 0),
                purchases=performance_data.get('purchases', 0),
                impressions=performance_data.get('impressions', 0),
                clicks=performance_data.get('clicks', 0),
                atc=performance_data.get('atc', 0),
                ic=performance_data.get('ic', 0)
            )
            
            # Rule-based decision
            if stage == 'testing':
                rule_promote, rule_reason = self.rule_engine.should_advance_from_testing(metrics)
            elif stage == 'validation':
                rule_promote, rule_reason = self.rule_engine.should_advance_from_validation(metrics)
            else:
                rule_promote = False
                rule_reason = "Not applicable"
            
            # Get ML predictions
            ml_analysis = self.ml_system.analyze_ad_intelligence(ad_id, stage)
            
            if not ml_analysis or not ml_analysis.get('predictions'):
                # No ML data, use rules only
                return rule_promote, rule_reason, 0.5
            
            predictions = ml_analysis.get('predictions', {})
            confidence = predictions.get('confidence_score', 0)
            predicted_roas = ml_analysis.get('predictions', {}).get('predicted_value', 0)
            
            # Cross-stage insights
            cross_stage = ml_analysis.get('cross_stage_insights', {})
            
            # ML recommendation
            ml_recommends_promote = False
            ml_reasoning = ""
            
            if confidence > self.confidence_threshold:
                # Predict strong future performance
                if predicted_roas > 2.0:
                    ml_recommends_promote = True
                    ml_reasoning = f"ML predicts strong ROAS: {predicted_roas:.2f} (confidence: {confidence:.2%})"
                
                # Similar to past winners
                if cross_stage.get('quality_score', 0) > 0.7:
                    ml_recommends_promote = True
                    ml_reasoning += f" + Similar to successful ads"
            
            # Combined decision
            if confidence > self.confidence_threshold:
                if ml_recommends_promote and rule_promote:
                    # Both agree - promote with high confidence
                    final_decision = True
                    final_reason = f"Rules + ML agree: {rule_reason}. {ml_reasoning}"
                    ml_influence = 0.5
                elif ml_recommends_promote and not rule_promote:
                    # ML says promote but rules don't - careful override
                    if confidence > 0.90:
                        final_decision = True
                        final_reason = f"ML override: {ml_reasoning}"
                        ml_influence = 0.9
                    else:
                        final_decision = False
                        final_reason = f"Rules say wait (ML confidence {confidence:.2%})"
                        ml_influence = 0.3
                elif not ml_recommends_promote and rule_promote:
                    # Rules say promote but ML says wait
                    if confidence > 0.85:
                        final_decision = False
                        final_reason = f"ML override: Wait for better performance"
                        ml_influence = 0.8
                    else:
                        final_decision = True
                        final_reason = f"Rules: {rule_reason} (ML not confident)"
                        ml_influence = 0.2
                else:
                    # Both agree to wait
                    final_decision = False
                    final_reason = "Rules + ML agree: Not ready yet"
                    ml_influence = 0.5
            else:
                # Low confidence - use rules
                final_decision = rule_promote
                final_reason = f"Rules: {rule_reason} (ML confidence low)"
                ml_influence = 0.1
            
            return final_decision, final_reason, ml_influence
            
        except Exception as e:
            self.logger.error(f"Error in ML-enhanced promotion decision: {e}")
            return rule_promote, f"Rules: {rule_reason} (ML error)", 0.0
    
    def recommend_budget(self, ad_id: str, stage: str, current_budget: float,
                        performance_data: Dict[str, Any]) -> Tuple[float, str, float]:
        """ML-recommended budget adjustment."""
        try:
            # Get ML predictions
            ml_analysis = self.ml_system.analyze_ad_intelligence(ad_id, stage)
            
            if not ml_analysis or not ml_analysis.get('predictions'):
                # No ML - keep current budget
                return current_budget, "No ML data available", 0.0
            
            predictions = ml_analysis.get('predictions', {})
            confidence = predictions.get('confidence_score', 0)
            predicted_roas = predictions.get('predicted_value', 0)
            
            if confidence < 0.7:
                return current_budget, "ML confidence too low", 0.0
            
            # Budget recommendation logic
            if predicted_roas > 3.0:
                # Excellent predicted ROAS - increase budget
                new_budget = current_budget * 1.5
                reason = f"ML predicts ROAS {predicted_roas:.2f} - scaling up"
                ml_influence = confidence
            elif predicted_roas > 2.0:
                # Good predicted ROAS - moderate increase
                new_budget = current_budget * 1.2
                reason = f"ML predicts ROAS {predicted_roas:.2f} - moderate increase"
                ml_influence = confidence
            elif predicted_roas < 1.0:
                # Poor predicted ROAS - decrease budget
                new_budget = current_budget * 0.7
                reason = f"ML predicts ROAS {predicted_roas:.2f} - scaling down"
                ml_influence = confidence
            else:
                # Stable predicted ROAS - maintain budget
                new_budget = current_budget
                reason = f"ML predicts stable ROAS {predicted_roas:.2f}"
                ml_influence = confidence * 0.5
            
            return new_budget, reason, ml_influence
            
        except Exception as e:
            self.logger.error(f"Error recommending budget: {e}")
            return current_budget, f"Error: {e}", 0.0

# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_ml_decision_engine(ml_system, rule_engine, confidence_threshold: float = 0.7) -> MLDecisionEngine:
    """Create ML-enhanced decision engine."""
    return MLDecisionEngine(ml_system, rule_engine, confidence_threshold)

def make_ml_enhanced_kill_decision(decision_engine: MLDecisionEngine, ad_id: str, 
                                  stage: str, performance_data: Dict[str, Any]) -> Tuple[bool, str, float]:
    """Make ML-enhanced kill decision."""
    return decision_engine.should_kill_ad(ad_id, stage, performance_data)

def make_ml_enhanced_promotion_decision(decision_engine: MLDecisionEngine, ad_id: str,
                                      stage: str, performance_data: Dict[str, Any]) -> Tuple[bool, str, float]:
    """Make ML-enhanced promotion decision."""
    return decision_engine.should_promote_ad(ad_id, stage, performance_data)

def get_ml_budget_recommendation(decision_engine: MLDecisionEngine, ad_id: str,
                                stage: str, current_budget: float,
                                performance_data: Dict[str, Any]) -> Tuple[float, str, float]:
    """Get ML budget recommendation."""
    return decision_engine.recommend_budget(ad_id, stage, current_budget, performance_data)

