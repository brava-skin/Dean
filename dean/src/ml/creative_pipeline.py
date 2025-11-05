"""
Automated Creative Generation Pipeline
Multi-stage creative generation with ML filtering
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CreativeGenerationPipeline:
    """Multi-stage creative generation pipeline."""
    
    def __init__(
        self,
        image_generator=None,
        predictive_model=None,
        creative_dna=None,
        variant_engine=None,
        advanced_ml=None,
    ):
        self.image_generator = image_generator
        self.predictive_model = predictive_model
        self.creative_dna = creative_dna
        self.variant_engine = variant_engine
        self.advanced_ml = advanced_ml
        self.generation_stages = ["generation", "filtering", "prediction", "selection"]
    
    def generate_creatives_batch(
        self,
        product_info: Dict[str, Any],
        target_count: int = 5,
        generate_count: int = 20,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of creatives through the pipeline."""
        # Stage 1: Generate variations
        logger.info(f"Stage 1: Generating {generate_count} creative variations")
        generated_creatives = []
        
        # Get advanced ML system if available
        advanced_ml = getattr(self.image_generator, 'advanced_ml', None)
        if not advanced_ml:
            # Try to get from parent
            advanced_ml = getattr(self, 'advanced_ml', None)
        
        for i in range(generate_count):
            try:
                creative = self.image_generator.generate_creative(
                    product_info,
                    creative_style=f"variation_{i}",
                    advanced_ml=advanced_ml,
                )
                
                if creative:
                    creative["generation_stage"] = "generated"
                    creative["generation_index"] = i
                    generated_creatives.append(creative)
            except Exception as e:
                logger.error(f"Error generating creative {i}: {e}")
        
        if not generated_creatives:
            logger.warning("No creatives generated in Stage 1")
            return []
        
        logger.info(f"Stage 1 complete: {len(generated_creatives)} creatives generated")
        
        # Stage 2: Pre-filter using ML predictions
        logger.info(f"Stage 2: Pre-filtering {len(generated_creatives)} creatives")
        filtered_creatives = []
        
        if self.predictive_model:
            for creative in generated_creatives:
                try:
                    prediction = self.predictive_model.predict(
                        creative,
                        target_metric="roas",
                    )
                    
                    # Filter by prediction
                    if prediction["predicted_value"] >= 1.0 and prediction["confidence"] >= 0.5:
                        creative["prediction"] = prediction
                        creative["generation_stage"] = "filtered"
                        filtered_creatives.append(creative)
                except Exception as e:
                    logger.error(f"Error predicting for creative: {e}")
                    # Include anyway if prediction fails
                    filtered_creatives.append(creative)
        else:
            filtered_creatives = generated_creatives
        
        logger.info(f"Stage 2 complete: {len(filtered_creatives)} creatives passed filter")
        
        # Stage 3: Select top performers
        logger.info(f"Stage 3: Selecting top {target_count} creatives")
        selected_creatives = []
        
        # Sort by predicted performance
        if self.predictive_model:
            filtered_creatives.sort(
                key=lambda c: c.get("prediction", {}).get("predicted_value", 0.0),
                reverse=True,
            )
        
        # Select top N
        selected_creatives = filtered_creatives[:target_count]
        
        for creative in selected_creatives:
            creative["generation_stage"] = "selected"
        
        logger.info(f"Stage 3 complete: {len(selected_creatives)} creatives selected")
        
        return selected_creatives
    
    def generate_with_variants(
        self,
        product_info: Dict[str, Any],
        base_creative: Optional[Dict[str, Any]] = None,
        variant_types: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate variants of a base creative."""
        if not self.variant_engine or not base_creative:
            return []
        
        # Generate variants
        variants = self.variant_engine.create_variants(
            base_creative,
            variant_types=variant_types or ["text_overlay", "prompt"],
            variants_per_type=3,
        )
        
        # Generate images for variants
        generated_variants = []
        for variant in variants:
            try:
                # Modify creative based on variant
                variant_creative = base_creative.copy()
                variant_creative.update(variant.variations)
                
                # Generate image
                creative = self.image_generator.generate_creative(
                    product_info,
                    creative_style=f"variant_{variant.variant_id}",
                )
                
                if creative:
                    creative["variant_id"] = variant.variant_id
                    creative["variant_type"] = variant.variant_type
                    generated_variants.append(creative)
            except Exception as e:
                logger.error(f"Error generating variant: {e}")
        
        return generated_variants
    
    def quality_score(
        self,
        creative: Dict[str, Any],
    ) -> float:
        """Calculate quality score for a creative."""
        score = 0.5  # Base score
        
        # Check if has all required elements
        if creative.get("image_path"):
            score += 0.2
        if creative.get("text_overlay"):
            score += 0.1
        if creative.get("ad_copy", {}).get("headline"):
            score += 0.1
        if creative.get("ad_copy", {}).get("primary_text"):
            score += 0.1
        
        # Prediction bonus
        if creative.get("prediction", {}).get("predicted_value", 0) > 1.5:
            score += 0.1
        
        return min(score, 1.0)


def create_creative_pipeline(
    image_generator=None,
    predictive_model=None,
    creative_dna=None,
    variant_engine=None,
    advanced_ml=None,
) -> CreativeGenerationPipeline:
    """Create a creative generation pipeline."""
    return CreativeGenerationPipeline(
        image_generator=image_generator,
        predictive_model=predictive_model,
        creative_dna=creative_dna,
        variant_engine=variant_engine,
        advanced_ml=advanced_ml,
    )


__all__ = ["CreativeGenerationPipeline", "create_creative_pipeline"]

