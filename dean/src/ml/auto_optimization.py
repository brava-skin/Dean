"""
Automated Optimization System
Self-healing, autonomous decision making, auto-discovery
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationOpportunity:
    """An optimization opportunity."""
    type: str
    description: str
    potential_impact: float
    confidence: float
    action: str
    metadata: Dict[str, Any] = None


class AutoOptimizer:
    """Automated optimization system."""
    
    def __init__(self):
        self.opportunities: List[OptimizationOpportunity] = []
        self.actions_taken: List[Dict[str, Any]] = []
    
    def discover_opportunities(
        self,
        creatives: List[Dict[str, Any]],
        performance_data: Dict[str, Any],
    ) -> List[OptimizationOpportunity]:
        """Discover optimization opportunities."""
        opportunities = []
        
        # Analyze creatives
        for creative in creatives:
            perf = creative.get("performance", {})
            
            # Low performing creative
            if perf.get("roas", 0) < 0.8 and perf.get("spend", 0) > 30:
                opportunities.append(OptimizationOpportunity(
                    type="pause_underperformer",
                    description=f"Creative {creative.get('ad_id')} underperforming",
                    potential_impact=0.15,
                    confidence=0.8,
                    action="pause",
                    metadata={"ad_id": creative.get("ad_id")},
                ))
            
            # High performing creative - scale opportunity
            if perf.get("roas", 0) > 2.0 and perf.get("spend", 0) < 20:
                opportunities.append(OptimizationOpportunity(
                    type="scale_winner",
                    description=f"Creative {creative.get('ad_id')} performing well",
                    potential_impact=0.25,
                    confidence=0.7,
                    action="increase_budget",
                    metadata={"ad_id": creative.get("ad_id")},
                ))
            
            # Creative fatigue
            if perf.get("ctr", 0) < 0.005 and perf.get("spend", 0) > 40:
                opportunities.append(OptimizationOpportunity(
                    type="refresh_fatigued",
                    description=f"Creative {creative.get('ad_id')} showing fatigue",
                    potential_impact=0.20,
                    confidence=0.75,
                    action="refresh",
                    metadata={"ad_id": creative.get("ad_id")},
                ))
        
        # Budget optimization
        total_roas = sum(c.get("performance", {}).get("roas", 0) for c in creatives)
        if total_roas > 0:
            opportunities.append(OptimizationOpportunity(
                type="budget_rebalance",
                description="Budget rebalancing opportunity",
                potential_impact=0.10,
                confidence=0.6,
                action="rebalance_budget",
            ))
        
        self.opportunities = opportunities
        return opportunities
    
    def prioritize_opportunities(
        self,
        opportunities: List[OptimizationOpportunity],
    ) -> List[OptimizationOpportunity]:
        """Prioritize opportunities by impact and confidence."""
        scored = [
            (opp, opp.potential_impact * opp.confidence)
            for opp in opportunities
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [opp for opp, _ in scored]
    
    def execute_optimization(
        self,
        opportunity: OptimizationOpportunity,
        client: Any,
    ) -> Dict[str, Any]:
        """Execute an optimization action."""
        action = opportunity.action
        metadata = opportunity.metadata or {}
        
        try:
            if action == "pause":
                ad_id = metadata.get("ad_id")
                if ad_id:
                    # Pause ad
                    client._graph_post(f"{ad_id}", {"status": "PAUSED"})
                    logger.info(f"Auto-paused ad {ad_id}")
            
            elif action == "increase_budget":
                # Budget increase would be handled by budget optimizer
                logger.info(f"Auto-scaling opportunity identified")
            
            elif action == "refresh":
                ad_id = metadata.get("ad_id")
                if ad_id:
                    # Mark for refresh
                    logger.info(f"Auto-refresh flagged for ad {ad_id}")
            
            elif action == "rebalance_budget":
                # Budget rebalancing
                logger.info("Auto-budget rebalancing triggered")
            
            self.actions_taken.append({
                "opportunity": opportunity.type,
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
            })
            
            return {"success": True, "action": action}
        
        except Exception as e:
            logger.error(f"Failed to execute optimization: {e}")
            return {"success": False, "error": str(e)}


class SelfHealingSystem:
    """Self-healing system for automatic recovery."""
    
    def __init__(self):
        self.health_history: List[Dict[str, Any]] = []
        self.recovery_actions: List[Dict[str, Any]] = []
    
    def detect_issues(
        self,
        system_health: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect system issues."""
        issues = []
        
        services = system_health.get("services", {})
        for service_name, service_data in services.items():
            status = service_data.get("status")
            
            if status == "unhealthy":
                issues.append({
                    "service": service_name,
                    "severity": "high",
                    "message": service_data.get("message", "Unknown error"),
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
        """Attempt to recover from issue."""
        service = issue.get("service")
        recovery_attempted = False
        
        try:
            if service == "meta_api":
                # Reset circuit breaker
                from infrastructure.error_handling import circuit_breaker_manager
                breaker = circuit_breaker_manager.get_breaker("meta_api")
                breaker.reset()
                recovery_attempted = True
            
            elif service == "flux_api":
                from infrastructure.error_handling import circuit_breaker_manager
                breaker = circuit_breaker_manager.get_breaker("flux_api")
                breaker.reset()
                recovery_attempted = True
            
            elif service == "database":
                # Try to reconnect
                from main import _get_supabase
                supabase = _get_supabase()
                if supabase:
                    recovery_attempted = True
            
            elif service == "cache":
                # Clear cache and retry
                from infrastructure.caching import cache_manager
                cache_manager.memory_cache.clear()
                recovery_attempted = True
            
            if recovery_attempted:
                self.recovery_actions.append({
                    "service": service,
                    "action": "recovery_attempted",
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                })
                logger.info(f"Recovery attempted for {service}")
            
            return {
                "recovered": recovery_attempted,
                "service": service,
            }
        
        except Exception as e:
            logger.error(f"Recovery failed for {service}: {e}")
            return {
                "recovered": False,
                "service": service,
                "error": str(e),
            }


class AutonomousDecisionEngine:
    """Autonomous decision making engine."""
    
    def __init__(self):
        self.decision_history: List[Dict[str, Any]] = []
        self.confidence_threshold = 0.7
    
    def make_decision(
        self,
        context: Dict[str, Any],
        options: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Make autonomous decision."""
        # Score each option
        scored_options = []
        for option in options:
            score = self._score_option(option, context)
            scored_options.append((option, score))
        
        # Select best option
        scored_options.sort(key=lambda x: x[1], reverse=True)
        best_option, confidence = scored_options[0]
        
        if confidence >= self.confidence_threshold:
            decision = {
                "option": best_option,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "autonomous": True,
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
        """Score an option based on context."""
        # Simple scoring - can be enhanced with ML
        base_score = option.get("expected_value", 0.5)
        risk = option.get("risk", 0.5)
        
        # Adjust for context
        context_factor = 1.0
        if context.get("market_condition") == "favorable":
            context_factor = 1.2
        elif context.get("market_condition") == "challenging":
            context_factor = 0.8
        
        score = base_score * context_factor * (1 - risk * 0.3)
        return min(1.0, max(0.0, score))


def create_auto_optimizer() -> AutoOptimizer:
    """Create auto optimizer."""
    return AutoOptimizer()


def create_self_healing_system() -> SelfHealingSystem:
    """Create self-healing system."""
    return SelfHealingSystem()


def create_autonomous_engine() -> AutonomousDecisionEngine:
    """Create autonomous decision engine."""
    return AutonomousDecisionEngine()


__all__ = [
    "AutoOptimizer",
    "SelfHealingSystem",
    "AutonomousDecisionEngine",
    "OptimizationOpportunity",
    "create_auto_optimizer",
    "create_self_healing_system",
    "create_autonomous_engine",
]

