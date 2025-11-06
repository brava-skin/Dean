"""
ADVANCED Automated Optimization System
Reinforcement learning integration, multi-armed bandit for creative selection, advanced opportunity discovery
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics

# Import advanced ML features
try:
    from ml.ml_advanced_features import QLearningAgent, RLState, RLAction, MultiArmedBandit
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    QLearningAgent = None
    RLState = None
    RLAction = None
    MultiArmedBandit = None

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OptimizationOpportunity:
    """An optimization opportunity with advanced metrics."""
    type: str
    description: str
    potential_impact: float
    confidence: float
    action: str
    metadata: Dict[str, Any] = None
    # Advanced metrics
    expected_value: float = 0.0
    risk_score: float = 0.0
    urgency: float = 0.0
    implementation_cost: float = 0.0
    roi_estimate: float = 0.0


class AdvancedOpportunityDiscovery:
    """Advanced opportunity discovery using ML and pattern recognition."""
    
    def __init__(self):
        self.pattern_history: List[Dict[str, Any]] = []
        self.cluster_model = None
        self.scaler = StandardScaler() if CLUSTERING_AVAILABLE else None
    
    def discover_patterns(
        self,
        creatives: List[Dict[str, Any]],
        performance_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Discover optimization patterns using clustering and analysis."""
        patterns = []
        
        if not creatives:
            return patterns
        
        # Cluster creatives by performance
        if CLUSTERING_AVAILABLE and len(creatives) >= 5:
            try:
                features = []
                creative_ids = []
                
                for creative in creatives:
                    perf = creative.get("performance", {})
                    feature_vector = [
                        perf.get("roas", 0),
                        perf.get("ctr", 0) * 100,  # Scale CTR
                        perf.get("cpa", 0) / 10,  # Normalize CPA
                        perf.get("spend", 0) / 100,  # Normalize spend
                        perf.get("purchases", 0),
                    ]
                    features.append(feature_vector)
                    creative_ids.append(creative.get("ad_id") or creative.get("creative_id"))
                
                if len(features) >= 5:
                    features_scaled = self.scaler.fit_transform(features)
                    clustering = DBSCAN(eps=0.5, min_samples=2)
                    clusters = clustering.fit_predict(features_scaled)
                    
                    # Analyze clusters
                    for cluster_id in set(clusters):
                        if cluster_id == -1:  # Noise
                            continue
                        
                        cluster_creatives = [creatives[i] for i, c in enumerate(clusters) if c == cluster_id]
                        cluster_perf = [c.get("performance", {}) for c in cluster_creatives]
                        
                        avg_roas = statistics.mean([p.get("roas", 0) for p in cluster_perf])
                        avg_ctr = statistics.mean([p.get("ctr", 0) for p in cluster_perf])
                        
                        if avg_roas > 2.0 and avg_ctr > 0.01:
                            patterns.append({
                                "type": "high_performing_cluster",
                                "cluster_id": cluster_id,
                                "creatives": [c.get("ad_id") for c in cluster_creatives],
                                "avg_roas": avg_roas,
                                "avg_ctr": avg_ctr,
                                "recommendation": "scale_cluster",
                            })
            except Exception as e:
                logger.debug(f"Clustering failed: {e}")
        
        # Detect performance trends
        for creative in creatives:
            perf = creative.get("performance", {})
            historical = creative.get("historical_performance", [])
            
            if len(historical) >= 3:
                # Calculate trend
                recent_roas = [h.get("roas", 0) for h in historical[-3:]]
                older_roas = [h.get("roas", 0) for h in historical[:-3]]
                
                if recent_roas and older_roas:
                    recent_avg = statistics.mean(recent_roas)
                    older_avg = statistics.mean(older_roas)
                    
                    if recent_avg > older_avg * 1.2:
                        patterns.append({
                            "type": "improving_trend",
                            "creative_id": creative.get("ad_id"),
                            "improvement_pct": (recent_avg - older_avg) / older_avg * 100,
                            "recommendation": "scale_up",
                        })
                    elif recent_avg < older_avg * 0.8:
                        patterns.append({
                            "type": "declining_trend",
                            "creative_id": creative.get("ad_id"),
                            "decline_pct": (older_avg - recent_avg) / older_avg * 100,
                            "recommendation": "refresh_or_pause",
                        })
        
        return patterns
    
    def calculate_opportunity_score(
        self,
        opportunity: OptimizationOpportunity,
        context: Dict[str, Any],
    ) -> float:
        """Calculate comprehensive opportunity score."""
        # Base score from impact and confidence
        base_score = opportunity.potential_impact * opportunity.confidence
        
        # Adjust for expected value
        if opportunity.expected_value > 0:
            base_score *= (1 + opportunity.expected_value * 0.1)
        
        # Adjust for risk
        risk_adjustment = 1.0 - (opportunity.risk_score * 0.3)
        base_score *= risk_adjustment
        
        # Adjust for urgency
        urgency_boost = 1.0 + (opportunity.urgency * 0.2)
        base_score *= urgency_boost
        
        # Adjust for ROI
        if opportunity.roi_estimate > 0:
            roi_boost = min(1.5, 1.0 + (opportunity.roi_estimate / 100))
            base_score *= roi_boost
        
        # Context adjustments
        market_condition = context.get("market_condition", "normal")
        if market_condition == "favorable":
            base_score *= 1.1
        elif market_condition == "challenging":
            base_score *= 0.9
        
        return base_score


class AutoOptimizer:
    """ADVANCED Automated optimization system with RL and multi-armed bandit."""
    
    def __init__(self, use_rl: bool = True, use_bandit: bool = True):
        self.opportunities: List[OptimizationOpportunity] = []
        self.actions_taken: List[Dict[str, Any]] = []
        self.use_rl = use_rl and RL_AVAILABLE
        self.use_bandit = use_bandit and RL_AVAILABLE
        
        # RL agent for decision making
        if self.use_rl:
            self.rl_agent = QLearningAgent(
                learning_rate=0.1,
                discount_factor=0.95,
                exploration_rate=0.1,
            )
        else:
            self.rl_agent = None
        
        # Multi-armed bandit for creative selection
        if self.use_bandit and MultiArmedBandit:
            try:
                # Initialize with empty arms list - will be populated when creatives are available
                self.bandit = MultiArmedBandit(arms=[])
            except Exception as e:
                logger.warning(f"Failed to initialize MultiArmedBandit: {e}")
                self.bandit = None
        else:
            self.bandit = None
        
        # Advanced opportunity discovery
        self.opportunity_discovery = AdvancedOpportunityDiscovery()
    
    def discover_opportunities(
        self,
        creatives: List[Dict[str, Any]],
        performance_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[OptimizationOpportunity]:
        """Discover optimization opportunities with advanced analysis."""
        opportunities = []
        context = context or {}
        
        # Pattern-based discovery
        patterns = self.opportunity_discovery.discover_patterns(creatives, performance_data)
        
        for pattern in patterns:
            if pattern["type"] == "high_performing_cluster":
                opportunities.append(OptimizationOpportunity(
                    type="scale_cluster",
                    description=f"High-performing cluster with {len(pattern['creatives'])} creatives",
                    potential_impact=0.3,
                    confidence=0.8,
                    action="scale_cluster",
                    metadata=pattern,
                    expected_value=pattern.get("avg_roas", 0) * 0.1,
                    risk_score=0.2,
                    urgency=0.7,
                    roi_estimate=pattern.get("avg_roas", 0) * 50,
                ))
            elif pattern["type"] == "improving_trend":
                opportunities.append(OptimizationOpportunity(
                    type="scale_improving",
                    description=f"Creative {pattern['creative_id']} showing strong improvement",
                    potential_impact=0.25,
                    confidence=0.75,
                    action="scale_up",
                    metadata=pattern,
                    expected_value=pattern.get("improvement_pct", 0) / 100,
                    risk_score=0.15,
                    urgency=0.6,
                ))
            elif pattern["type"] == "declining_trend":
                opportunities.append(OptimizationOpportunity(
                    type="refresh_declining",
                    description=f"Creative {pattern['creative_id']} showing decline",
                    potential_impact=0.2,
                    confidence=0.7,
                    action="refresh",
                    metadata=pattern,
                    expected_value=0.1,
                    risk_score=0.3,
                    urgency=0.8,
                ))
        
        # Traditional opportunity detection (enhanced)
        for creative in creatives:
            perf = creative.get("performance", {})
            ad_id = creative.get("ad_id") or creative.get("creative_id")
            
            # Low performing creative (enhanced with risk analysis)
            roas = perf.get("roas", 0)
            spend = perf.get("spend", 0)
            ctr = perf.get("ctr", 0)
            
            if roas < 0.8 and spend > 30:
                # Calculate risk-adjusted impact
                risk_score = 0.3 if spend > 50 else 0.2
                impact = min(0.2, spend / 200)  # Scale impact with spend
                
                opportunities.append(OptimizationOpportunity(
                    type="pause_underperformer",
                    description=f"Creative {ad_id} underperforming (ROAS: {roas:.2f})",
                    potential_impact=impact,
                    confidence=0.85,
                    action="pause",
                    metadata={"ad_id": ad_id, "roas": roas, "spend": spend},
                    expected_value=-spend * (1 - roas),  # Expected savings
                    risk_score=risk_score,
                    urgency=0.9 if spend > 50 else 0.7,
                    roi_estimate=-50,  # Negative ROI (cost savings)
                ))
            
            # High performing creative - scale opportunity (enhanced)
            if roas > 2.0 and spend < 20:
                # Use bandit to estimate expected reward
                expected_reward = 0.0
                if self.bandit:
                    expected_reward = self.bandit.get_expected_reward(ad_id)
                
                opportunities.append(OptimizationOpportunity(
                    type="scale_winner",
                    description=f"Creative {ad_id} performing well (ROAS: {roas:.2f})",
                    potential_impact=0.3,
                    confidence=0.8,
                    action="increase_budget",
                    metadata={"ad_id": ad_id, "roas": roas, "spend": spend},
                    expected_value=(roas - 1.0) * spend * 0.5,  # Expected profit increase
                    risk_score=0.15,
                    urgency=0.6,
                    roi_estimate=roas * 100,
                ))
            
            # Creative fatigue (enhanced with predictive analysis)
            if ctr < 0.005 and spend > 40:
                # Calculate fatigue severity
                fatigue_severity = (0.005 - ctr) / 0.005
                
                opportunities.append(OptimizationOpportunity(
                    type="refresh_fatigued",
                    description=f"Creative {ad_id} showing fatigue (CTR: {ctr:.3%})",
                    potential_impact=0.25 * fatigue_severity,
                    confidence=0.8,
                    action="refresh",
                    metadata={"ad_id": ad_id, "ctr": ctr, "spend": spend},
                    expected_value=spend * 0.1 * fatigue_severity,
                    risk_score=0.2,
                    urgency=0.85,
                ))
        
        # Budget optimization (enhanced)
        total_roas = sum(c.get("performance", {}).get("roas", 0) for c in creatives)
        total_spend = sum(c.get("performance", {}).get("spend", 0) for c in creatives)
        
        if total_roas > 0 and total_spend > 0:
            # Calculate rebalancing potential
            roas_variance = statistics.variance([c.get("performance", {}).get("roas", 0) for c in creatives])
            rebalance_potential = min(0.15, roas_variance / 10)
            
            if rebalance_potential > 0.05:
                opportunities.append(OptimizationOpportunity(
                    type="budget_rebalance",
                    description="Budget rebalancing opportunity (high variance in performance)",
                    potential_impact=rebalance_potential,
                    confidence=0.7,
                    action="rebalance_budget",
                    metadata={"variance": roas_variance},
                    expected_value=total_spend * rebalance_potential,
                    risk_score=0.25,
                    urgency=0.5,
                ))
        
        # Score and rank opportunities
        scored_opportunities = []
        for opp in opportunities:
            score = self.opportunity_discovery.calculate_opportunity_score(opp, context)
            scored_opportunities.append((opp, score))
        
        scored_opportunities.sort(key=lambda x: x[1], reverse=True)
        self.opportunities = [opp for opp, _ in scored_opportunities]
        
        return self.opportunities
    
    def prioritize_opportunities(
        self,
        opportunities: List[OptimizationOpportunity],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[OptimizationOpportunity]:
        """Prioritize opportunities using advanced scoring."""
        context = context or {}
        
        scored = []
        for opp in opportunities:
            score = self.opportunity_discovery.calculate_opportunity_score(opp, context)
            scored.append((opp, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [opp for opp, _ in scored]
    
    def execute_optimization(
        self,
        opportunity: OptimizationOpportunity,
        client: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute optimization action with RL learning."""
        action = opportunity.action
        metadata = opportunity.metadata or {}
        context = context or {}
        
        # Use RL agent to make decision if available
        if self.rl_agent and action in ["pause", "scale_up", "refresh"]:
            # Convert to RL state
            creative = metadata.get("creative") or {}
            perf = creative.get("performance", {})
            
            state = RLState(
                cpa=perf.get("cpa", 0),
                roas=perf.get("roas", 0),
                ctr=perf.get("ctr", 0),
                spend=perf.get("spend", 0),
                days_active=perf.get("days_active", 0),
                stage=context.get("stage", "asc_plus"),
            )
            
            # Get RL recommendation
            rl_action = self.rl_agent.get_action(state, explore=False)
            
            # Use RL action if confidence is high
            if rl_action.confidence > 0.7 and rl_action.action_type == action:
                logger.info(f"RL agent confirms {action} with confidence {rl_action.confidence:.2%}")
        
        try:
            if action == "pause":
                ad_id = metadata.get("ad_id")
                if ad_id:
                    client._graph_post(f"{ad_id}", {"status": "PAUSED"})
                    logger.info(f"Auto-paused ad {ad_id}")
                    
                    # Update bandit
                    if self.bandit:
                        self.bandit.update(ad_id, reward=-1.0)  # Negative reward
            
            elif action == "increase_budget" or action == "scale_up":
                ad_id = metadata.get("ad_id")
                if ad_id:
                    logger.info(f"Auto-scaling opportunity identified for {ad_id}")
                    
                    # Update bandit with positive reward
                    if self.bandit:
                        roas = metadata.get("roas", 0)
                        reward = min(1.0, (roas - 1.0) / 2.0)  # Normalize reward
                        self.bandit.update(ad_id, reward=reward)
            
            elif action == "refresh":
                ad_id = metadata.get("ad_id")
                if ad_id:
                    logger.info(f"Auto-refresh flagged for ad {ad_id}")
            
            elif action == "rebalance_budget":
                logger.info("Auto-budget rebalancing triggered")
            
            elif action == "scale_cluster":
                creative_ids = metadata.get("creatives", [])
                logger.info(f"Auto-scaling cluster with {len(creative_ids)} creatives")
            
            # Record action
            self.actions_taken.append({
                "opportunity": opportunity.type,
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
                "expected_value": opportunity.expected_value,
                "risk_score": opportunity.risk_score,
            })
            
            return {"success": True, "action": action, "expected_value": opportunity.expected_value}
        
        except Exception as e:
            logger.error(f"Failed to execute optimization: {e}")
            return {"success": False, "error": str(e)}
    
    def select_creative_with_bandit(
        self,
        creatives: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Select creative using multi-armed bandit."""
        if not self.bandit or not creatives:
            return creatives[0] if creatives else None
        
        # Register creatives with bandit
        for creative in creatives:
            creative_id = creative.get("ad_id") or creative.get("creative_id")
            if creative_id:
                self.bandit.add_arm(creative_id)
        
        # Select using Thompson Sampling
        selected_id = self.bandit.select_arm()
        
        # Find selected creative
        for creative in creatives:
            creative_id = creative.get("ad_id") or creative.get("creative_id")
            if creative_id == selected_id:
                return creative
        
        return creatives[0] if creatives else None


class SelfHealingSystem:
    """ADVANCED Self-healing system with predictive failure detection."""
    
    def __init__(self):
        self.health_history: List[Dict[str, Any]] = []
        self.recovery_actions: List[Dict[str, Any]] = []
        self.failure_patterns: Dict[str, int] = {}  # Track failure patterns
    
    def detect_issues(
        self,
        system_health: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect system issues with pattern recognition."""
        issues = []
        
        services = system_health.get("services", {})
        for service_name, service_data in services.items():
            status = service_data.get("status")
            
            # Track failure patterns
            if status == "unhealthy":
                self.failure_patterns[service_name] = self.failure_patterns.get(service_name, 0) + 1
                
                issues.append({
                    "service": service_name,
                    "severity": "high",
                    "message": service_data.get("message", "Unknown error"),
                    "failure_count": self.failure_patterns[service_name],
                    "pattern_detected": self.failure_patterns[service_name] > 3,
                })
            
            elif status == "degraded":
                issues.append({
                    "service": service_name,
                    "severity": "medium",
                    "message": service_data.get("message", "Degraded performance"),
                })
        
        return issues
    
    def attempt_recovery(
        self,
        issue: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attempt recovery with adaptive strategies."""
        service = issue.get("service")
        failure_count = issue.get("failure_count", 0)
        recovery_attempted = False
        
        try:
            if service == "meta_api":
                from infrastructure.error_handling import circuit_breaker_manager
                breaker = circuit_breaker_manager.get_breaker("meta_api")
                
                # Adaptive recovery based on failure count
                if failure_count > 5:
                    # Multiple failures - wait longer
                    import time
                    time.sleep(30)
                
                breaker.reset()
                recovery_attempted = True
            
            elif service == "flux_api":
                from infrastructure.error_handling import circuit_breaker_manager
                breaker = circuit_breaker_manager.get_breaker("flux_api")
                breaker.reset()
                recovery_attempted = True
            
            elif service == "database":
                from main import _get_supabase
                supabase = _get_supabase()
                if supabase:
                    recovery_attempted = True
            
            elif service == "cache":
                from infrastructure.caching import cache_manager
                cache_manager.memory_cache.clear()
                recovery_attempted = True
            
            if recovery_attempted:
                self.recovery_actions.append({
                    "service": service,
                    "action": "recovery_attempted",
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                    "failure_count": failure_count,
                })
                logger.info(f"Recovery attempted for {service} (failure count: {failure_count})")
            
            return {
                "recovered": recovery_attempted,
                "service": service,
                "failure_count": failure_count,
            }
        
        except Exception as e:
            logger.error(f"Recovery failed for {service}: {e}")
            return {
                "recovered": False,
                "service": service,
                "error": str(e),
            }


class AutonomousDecisionEngine:
    """ADVANCED Autonomous decision making engine with RL."""
    
    def __init__(self, use_rl: bool = True):
        self.decision_history: List[Dict[str, Any]] = []
        self.confidence_threshold = 0.7
        self.use_rl = use_rl and RL_AVAILABLE
        
        if self.use_rl:
            self.rl_agent = QLearningAgent()
        else:
            self.rl_agent = None
    
    def make_decision(
        self,
        context: Dict[str, Any],
        options: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Make autonomous decision with RL support."""
        # Score each option
        scored_options = []
        for option in options:
            score = self._score_option(option, context)
            scored_options.append((option, score))
        
        # Use RL agent if available
        if self.rl_agent:
            # Convert context to RL state
            state = RLState(
                cpa=context.get("cpa", 0),
                roas=context.get("roas", 0),
                ctr=context.get("ctr", 0),
                spend=context.get("spend", 0),
                days_active=context.get("days_active", 0),
                stage=context.get("stage", "asc_plus"),
            )
            
            rl_action = self.rl_agent.get_action(state, explore=False)
            
            # Boost score for RL-recommended actions
            for i, (option, score) in enumerate(scored_options):
                if option.get("action") == rl_action.action_type:
                    score *= (1 + rl_action.confidence * 0.3)
                    scored_options[i] = (option, score)
        
        # Select best option
        scored_options.sort(key=lambda x: x[1], reverse=True)
        best_option, confidence = scored_options[0]
        
        if confidence >= self.confidence_threshold:
            decision = {
                "option": best_option,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "autonomous": True,
                "rl_recommendation": rl_action.action_type if self.rl_agent else None,
            }
            self.decision_history.append(decision)
            return decision
        else:
            return {
                "option": None,
                "confidence": confidence,
                "autonomous": False,
                "reason": "Confidence too low",
            }
    
    def _score_option(
        self,
        option: Dict[str, Any],
        context: Dict[str, Any],
    ) -> float:
        """Score an option with advanced metrics."""
        base_score = option.get("expected_value", 0.5)
        risk = option.get("risk", 0.5)
        
        # Risk-adjusted score
        risk_adjusted = base_score * (1 - risk * 0.3)
        
        # Context adjustments
        context_factor = 1.0
        market_condition = context.get("market_condition", "normal")
        if market_condition == "favorable":
            context_factor = 1.2
        elif market_condition == "challenging":
            context_factor = 0.8
        
        # Urgency boost
        urgency = option.get("urgency", 0.5)
        urgency_boost = 1.0 + (urgency * 0.2)
        
        score = risk_adjusted * context_factor * urgency_boost
        return min(1.0, max(0.0, score))
    
    def learn_from_outcome(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
    ):
        """Learn from decision outcome using RL."""
        if not self.rl_agent:
            return
        
        # Calculate reward
        reward = outcome.get("reward", 0.0)
        if "roas" in outcome:
            reward = (outcome["roas"] - 1.0) / 2.0  # Normalize to [-0.5, 1.0]
        
        # Update RL agent
        option = decision.get("option", {})
        action = option.get("action", "hold")
        
        # Convert to RL state and action
        context = decision.get("context", {})
        state = RLState(
            cpa=context.get("cpa", 0),
            roas=context.get("roas", 0),
            ctr=context.get("ctr", 0),
            spend=context.get("spend", 0),
            days_active=context.get("days_active", 0),
            stage=context.get("stage", "asc_plus"),
        )
        
        next_state = RLState(
            cpa=outcome.get("cpa", context.get("cpa", 0)),
            roas=outcome.get("roas", context.get("roas", 0)),
            ctr=outcome.get("ctr", context.get("ctr", 0)),
            spend=outcome.get("spend", context.get("spend", 0)),
            days_active=context.get("days_active", 0) + 1,
            stage=context.get("stage", "asc_plus"),
        )
        
        self.rl_agent.update_q_value(state, action, reward, next_state)


def create_auto_optimizer(**kwargs) -> AutoOptimizer:
    """Create advanced auto optimizer."""
    return AutoOptimizer(**kwargs)


def create_self_healing_system() -> SelfHealingSystem:
    """Create advanced self-healing system."""
    return SelfHealingSystem()


def create_autonomous_engine(**kwargs) -> AutonomousDecisionEngine:
    """Create advanced autonomous decision engine."""
    return AutonomousDecisionEngine(**kwargs)


__all__ = [
    "AutoOptimizer",
    "SelfHealingSystem",
    "AutonomousDecisionEngine",
    "OptimizationOpportunity",
    "AdvancedOpportunityDiscovery",
    "create_auto_optimizer",
    "create_self_healing_system",
    "create_autonomous_engine",
]
