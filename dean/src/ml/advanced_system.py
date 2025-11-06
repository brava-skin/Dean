"""
Advanced System Integration
Brings together all advanced ML features
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import all advanced modules
try:
    from ml.creative_dna import create_creative_dna_analyzer, CreativeDNAAnalyzer
    from ml.variant_testing import create_variant_testing_engine, VariantTestingEngine
    from ml.predictive_modeling import create_predictive_model, create_early_signal_detector
    from ml.budget_optimizer import create_budget_optimizer
    from ml.performance_adaptation import create_performance_adaptation
    from ml.prompt_optimization import create_prompt_evolution_engine
    from ml.creative_elements import create_creative_element_analyzer
    from ml.time_series_forecast import create_time_series_forecaster
    from ml.anomaly_detection import create_anomaly_detector
    from ml.health_scoring import create_health_scorer
    from ml.creative_refresh import create_creative_refresh_manager
    from ml.creative_pipeline import create_creative_pipeline
    from ml.ml_advanced_features import (
        create_bandit, create_bayesian_optimizer, create_rl_agent, create_genetic_algorithm
    )
    from ml.auto_optimization import (
        create_auto_optimizer, create_self_healing_system, create_autonomous_engine
    )
    from ml.model_training import (
        create_model_registry, create_training_pipeline, create_model_monitor
    )
    from creative.advanced_creative import (
        create_template_library, create_version_manager,
        create_quality_checker, create_style_transfer
    )
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some advanced modules not available: {e}")
    ADVANCED_MODULES_AVAILABLE = False


class AdvancedMLSystem:
    """Unified interface for all advanced ML features."""
    
    def __init__(
        self,
        supabase_client=None,
        image_generator=None,
        ml_system=None,
    ):
        self.supabase_client = supabase_client
        self.image_generator = image_generator
        self.ml_system = ml_system
        
        # Initialize all advanced systems
        self.creative_dna = create_creative_dna_analyzer(supabase_client) if ADVANCED_MODULES_AVAILABLE else None
        self.variant_engine = create_variant_testing_engine(supabase_client) if ADVANCED_MODULES_AVAILABLE else None
        self.predictive_model = create_predictive_model(supabase_client) if ADVANCED_MODULES_AVAILABLE else None
        self.early_signal = create_early_signal_detector() if ADVANCED_MODULES_AVAILABLE else None
        self.budget_optimizer = create_budget_optimizer() if ADVANCED_MODULES_AVAILABLE else None
        self.performance_adaptation = create_performance_adaptation() if ADVANCED_MODULES_AVAILABLE else None
        self.prompt_evolution = create_prompt_evolution_engine() if ADVANCED_MODULES_AVAILABLE else None
        self.element_analyzer = create_creative_element_analyzer(supabase_client) if ADVANCED_MODULES_AVAILABLE else None
        self.time_series = create_time_series_forecaster() if ADVANCED_MODULES_AVAILABLE else None
        self.anomaly_detector = create_anomaly_detector() if ADVANCED_MODULES_AVAILABLE else None
        self.health_scorer = create_health_scorer() if ADVANCED_MODULES_AVAILABLE else None
        self.refresh_manager = create_creative_refresh_manager() if ADVANCED_MODULES_AVAILABLE else None
        
        # Create creative pipeline
        self.creative_pipeline = None
        if ADVANCED_MODULES_AVAILABLE and image_generator:
            self.creative_pipeline = create_creative_pipeline(
                image_generator=image_generator,
                predictive_model=self.predictive_model,
                creative_dna=self.creative_dna,
                variant_engine=self.variant_engine,
                advanced_ml=self,  # Pass self for access to all features
            )
        
        # Advanced ML features
        self.bandit = None
        self.bayesian_optimizer = None
        self.rl_agent = None
        self.genetic_algorithm = None
        
        # Auto-optimization
        self.auto_optimizer = create_auto_optimizer() if ADVANCED_MODULES_AVAILABLE else None
        self.self_healing = create_self_healing_system() if ADVANCED_MODULES_AVAILABLE else None
        self.autonomous_engine = create_autonomous_engine() if ADVANCED_MODULES_AVAILABLE else None
        
        # Model training
        self.model_registry = create_model_registry() if ADVANCED_MODULES_AVAILABLE else None
        self.training_pipeline = create_training_pipeline(
            self.model_registry, supabase_client
        ) if ADVANCED_MODULES_AVAILABLE and self.model_registry else None
        self.model_monitor = create_model_monitor(
            self.model_registry
        ) if ADVANCED_MODULES_AVAILABLE and self.model_registry else None
        
        # Advanced creative features
        self.template_library = create_template_library() if ADVANCED_MODULES_AVAILABLE else None
        self.version_manager = create_version_manager() if ADVANCED_MODULES_AVAILABLE else None
        self.quality_checker = create_quality_checker() if ADVANCED_MODULES_AVAILABLE else None
        self.style_transfer = create_style_transfer(
            self.creative_dna
        ) if ADVANCED_MODULES_AVAILABLE else None
        
        # Prompt optimization
        try:
            from ml.prompt_optimization import create_prompt_library, create_rl_prompt_optimizer
            self.prompt_library = create_prompt_library() if ADVANCED_MODULES_AVAILABLE else None
            self.rl_prompt_optimizer = create_rl_prompt_optimizer() if ADVANCED_MODULES_AVAILABLE else None
        except ImportError:
            self.prompt_library = None
            self.rl_prompt_optimizer = None
        
        logger.info("✅ Advanced ML System initialized with all features")
    
    def generate_optimized_creatives(
        self,
        product_info: Dict[str, Any],
        target_count: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate optimized creatives using the full pipeline."""
        if self.creative_pipeline:
            # For testing: generate only 1 if target is 1, otherwise use normal logic
            if target_count == 1:
                generate_count = 1  # Test mode: only generate 1
            else:
                # Generate fewer variations to avoid timeout (max 10, but aim for target_count + 2)
                generate_count = min(target_count + 2, 10)
            logger.info(f"Generating {generate_count} creatives to filter to top {target_count}")
            return self.creative_pipeline.generate_creatives_batch(
                product_info,
                target_count=target_count,
                generate_count=generate_count,  # Reduced from 20 to avoid timeout
            )
        elif self.image_generator:
            # Fallback to simple generation
            creatives = []
            for i in range(target_count):
                creative = self.image_generator.generate_creative(
                    product_info,
                    creative_style=f"creative_{i}",
                )
                if creative:
                    creatives.append(creative)
            return creatives
        return []
    
    def analyze_creative_performance(
        self,
        creative_data: Dict[str, Any],
        performance_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Comprehensive creative performance analysis."""
        analysis = {}
        
        # Creative DNA
        if self.creative_dna:
            try:
                dna = self.creative_dna.create_creative_dna(
                    creative_data.get("creative_id", ""),
                    creative_data.get("ad_id", ""),
                    creative_data.get("image_prompt", ""),
                    creative_data.get("text_overlay", ""),
                    creative_data.get("ad_copy", {}),
                    performance_data,
                )
                analysis["dna"] = {
                    "performance_score": dna.performance_score,
                    "roas": dna.roas,
                    "ctr": dna.ctr,
                }
            except Exception as e:
                logger.error(f"Error in DNA analysis: {e}")
        
        # Element analysis
        if self.element_analyzer:
            try:
                element_analysis = self.element_analyzer.analyze_element_performance(
                    creative_data,
                    performance_data,
                )
                analysis["elements"] = element_analysis
            except Exception as e:
                logger.error(f"Error in element analysis: {e}")
        
        # Health score
        if self.health_scorer:
            try:
                health = self.health_scorer.calculate_creative_health(
                    performance_data,
                )
                analysis["health"] = health
            except Exception as e:
                logger.error(f"Error in health scoring: {e}")
        
        return analysis
    
    def get_optimization_recommendations(
        self,
        creatives: List[Dict[str, Any]],
        total_budget: float,
    ) -> Dict[str, Any]:
        """Get optimization recommendations."""
        recommendations = {}
        
        # Budget optimization
        if self.budget_optimizer:
            try:
                budget_rec = self.budget_optimizer.recommend_budget_rebalancing(
                    {c.get("ad_id", ""): 10.0 for c in creatives},  # Current allocations
                    creatives,
                )
                recommendations["budget"] = budget_rec
            except Exception as e:
                logger.error(f"Error in budget optimization: {e}")
        
        # Refresh recommendations
        if self.refresh_manager:
            try:
                refresh_plan = self.refresh_manager.plan_refresh_schedule(
                    creatives,
                    target_count=5,
                )
                recommendations["refresh"] = refresh_plan
            except Exception as e:
                logger.error(f"Error in refresh planning: {e}")
        
        return recommendations


def create_advanced_ml_system(
    supabase_client=None,
    image_generator=None,
    ml_system=None,
) -> AdvancedMLSystem:
    """Create an advanced ML system."""
    return AdvancedMLSystem(
        supabase_client=supabase_client,
        image_generator=image_generator,
        ml_system=ml_system,
    )


__all__ = ["AdvancedMLSystem", "create_advanced_ml_system"]

