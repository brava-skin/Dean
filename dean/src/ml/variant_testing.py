"""
Multi-Variant A/B Testing Engine
Automated variant generation and statistical testing
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """A/B test variant."""
    variant_id: str
    creative_id: str
    variant_type: str  # "text_overlay", "prompt", "ad_copy", "combined"
    variations: Dict[str, Any]
    performance: Dict[str, float]
    impressions: int = 0
    clicks: int = 0
    purchases: int = 0
    spend: float = 0.0
    roas: float = 0.0
    ctr: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def update_performance(self, performance_data: Dict[str, Any]):
        """Update variant performance metrics."""
        self.impressions = performance_data.get("impressions", 0)
        self.clicks = performance_data.get("clicks", 0)
        self.purchases = performance_data.get("purchases", 0)
        self.spend = performance_data.get("spend", 0.0)
        self.roas = performance_data.get("roas", 0.0)
        self.ctr = performance_data.get("ctr", 0.0)
        self.performance = performance_data


class VariantGenerator:
    """Generates creative variants for A/B testing."""
    
    def __init__(self):
        self.variant_count = 0
    
    def generate_text_overlay_variants(
        self,
        base_text: str,
        count: int = 3,
    ) -> List[str]:
        """Generate text overlay variants."""
        variants = [base_text]  # Include original
        
        # Position variants
        # (We'll keep text the same but mark for different positions)
        
        # Style variants
        calm_confidence_texts = [
            "This is maintenance, not vanity.",
            "The man who cares stands out.",
            "Elevate your baseline.",
            "Look like you live with intention.",
            "Take care. Not for others. For yourself.",
            "Consistency builds presence.",
            "The face you show the world matters.",
            "Refined, not complicated.",
        ]
        
        # Add variations
        for i in range(count - 1):
            if i < len(calm_confidence_texts):
                variants.append(calm_confidence_texts[i])
            else:
                # Generate slight variations
                variants.append(f"{base_text}.")
        
        return variants[:count]
    
    def generate_prompt_variants(
        self,
        base_prompt: str,
        count: int = 3,
    ) -> List[str]:
        """Generate image prompt variants."""
        variants = [base_prompt]
        
        # Style variations
        style_modifiers = [
            "editorial, cinematic",
            "minimalist, sophisticated",
            "luxury, refined",
            "premium, timeless",
            "elegant, understated",
        ]
        
        # Lighting variations
        lighting_modifiers = [
            "natural daylight, soft shadows",
            "warm golden hour lighting",
            "studio lighting, professional",
            "dramatic contrast lighting",
            "soft diffused light",
        ]
        
        for i in range(count - 1):
            style = style_modifiers[i % len(style_modifiers)]
            lighting = lighting_modifiers[i % len(lighting_modifiers)]
            variant = f"{base_prompt}, {style}, {lighting}"
            variants.append(variant)
        
        return variants[:count]
    
    def generate_ad_copy_variants(
        self,
        base_copy: Dict[str, str],
        count: int = 3,
    ) -> List[Dict[str, str]]:
        """Generate ad copy variants."""
        variants = [base_copy]
        
        # Headline variations
        headline_options = [
            "For men who value discipline.",
            "Clean skin is the foundation of presence.",
            "Your routine communicates who you are.",
            "Look like a man who takes care of himself.",
            "Precision in every detail.",
        ]
        
        # Primary text variations
        primary_text_options = [
            "Precision skincare designed to elevate daily standards.",
            "Built for the man who demands excellence.",
            "Quality that speaks for itself.",
            "Invest in yourself. Every day.",
            "The standard, redefined.",
        ]
        
        for i in range(count - 1):
            variant = {
                "headline": headline_options[i % len(headline_options)],
                "primary_text": primary_text_options[i % len(primary_text_options)],
                "description": base_copy.get("description", ""),
            }
            variants.append(variant)
        
        return variants[:count]


class BayesianABTest:
    """Bayesian A/B testing for variants."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha  # Prior alpha
        self.beta = beta    # Prior beta
    
    def update(self, variant: Variant) -> Dict[str, float]:
        """Update Bayesian posterior for variant."""
        # Convert performance to beta distribution parameters
        # Using ROAS as success metric (normalized to 0-1)
        roas_normalized = min(variant.roas / 10.0, 1.0) if variant.roas > 0 else 0.0
        purchases = variant.purchases
        
        # Update posterior
        posterior_alpha = self.alpha + purchases * roas_normalized
        posterior_beta = self.beta + purchases * (1 - roas_normalized)
        
        # Expected value
        if posterior_alpha + posterior_beta > 0:
            expected_value = posterior_alpha / (posterior_alpha + posterior_beta)
        else:
            expected_value = 0.5
        
        # Confidence (inverse of variance)
        if posterior_alpha + posterior_beta > 1:
            variance = (posterior_alpha * posterior_beta) / (
                (posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1)
            )
            confidence = 1.0 / (1.0 + variance)
        else:
            confidence = 0.0
        
        return {
            "expected_value": expected_value,
            "confidence": confidence,
            "posterior_alpha": posterior_alpha,
            "posterior_beta": posterior_beta,
        }
    
    def compare_variants(
        self,
        variant_a: Variant,
        variant_b: Variant,
    ) -> Dict[str, Any]:
        """Compare two variants using Bayesian analysis."""
        result_a = self.update(variant_a)
        result_b = self.update(variant_b)
        
        # Probability that A beats B
        prob_a_beats_b = self._probability_a_beats_b(
            result_a["posterior_alpha"], result_a["posterior_beta"],
            result_b["posterior_alpha"], result_b["posterior_beta"],
        )
        
        return {
            "variant_a": {
                "expected_value": result_a["expected_value"],
                "confidence": result_a["confidence"],
            },
            "variant_b": {
                "expected_value": result_b["expected_value"],
                "confidence": result_b["confidence"],
            },
            "prob_a_beats_b": prob_a_beats_b,
            "winner": "A" if prob_a_beats_b > 0.5 else "B",
            "confidence": max(result_a["confidence"], result_b["confidence"]),
        }
    
    def _probability_a_beats_b(
        self,
        alpha_a: float,
        beta_a: float,
        alpha_b: float,
        beta_b: float,
    ) -> float:
        """Calculate probability that A beats B."""
        # Simplified calculation using beta distribution
        # More accurate would use Monte Carlo simulation
        if alpha_a + beta_a == 0 or alpha_b + beta_b == 0:
            return 0.5
        
        mean_a = alpha_a / (alpha_a + beta_a)
        mean_b = alpha_b / (alpha_b + beta_b)
        
        # Simple heuristic: if mean_a > mean_b, probability > 0.5
        if mean_a > mean_b:
            diff = (mean_a - mean_b) * 0.5 + 0.5
            return min(diff, 0.95)
        else:
            diff = (mean_b - mean_a) * 0.5
            return max(0.05, diff)


class VariantTestingEngine:
    """Main engine for variant testing."""
    
    def __init__(self, supabase_client=None):
        self.supabase_client = supabase_client
        self.generator = VariantGenerator()
        self.bayesian_test = BayesianABTest()
        self.active_variants: Dict[str, Variant] = {}
        self.completed_tests: List[Dict[str, Any]] = []
    
    def create_variants(
        self,
        base_creative: Dict[str, Any],
        variant_types: List[str] = None,
        variants_per_type: int = 3,
    ) -> List[Variant]:
        """Create variants for a base creative."""
        if variant_types is None:
            variant_types = ["text_overlay", "prompt", "ad_copy"]
        
        all_variants = []
        
        for variant_type in variant_types:
            if variant_type == "text_overlay":
                text_variants = self.generator.generate_text_overlay_variants(
                    base_creative.get("text_overlay", ""),
                    variants_per_type,
                )
                for i, text in enumerate(text_variants):
                    variant = Variant(
                        variant_id=f"{base_creative.get('creative_id', 'base')}_text_{i}",
                        creative_id=base_creative.get("creative_id", ""),
                        variant_type="text_overlay",
                        variations={"text_overlay": text},
                        performance={},
                    )
                    all_variants.append(variant)
            
            elif variant_type == "prompt":
                prompt_variants = self.generator.generate_prompt_variants(
                    base_creative.get("image_prompt", ""),
                    variants_per_type,
                )
                for i, prompt in enumerate(prompt_variants):
                    variant = Variant(
                        variant_id=f"{base_creative.get('creative_id', 'base')}_prompt_{i}",
                        creative_id=base_creative.get("creative_id", ""),
                        variant_type="prompt",
                        variations={"image_prompt": prompt},
                        performance={},
                    )
                    all_variants.append(variant)
            
            elif variant_type == "ad_copy":
                copy_variants = self.generator.generate_ad_copy_variants(
                    base_creative.get("ad_copy", {}),
                    variants_per_type,
                )
                for i, copy in enumerate(copy_variants):
                    variant = Variant(
                        variant_id=f"{base_creative.get('creative_id', 'base')}_copy_{i}",
                        creative_id=base_creative.get("creative_id", ""),
                        variant_type="ad_copy",
                        variations={"ad_copy": copy},
                        performance={},
                    )
                    all_variants.append(variant)
        
        return all_variants
    
    def evaluate_variant(
        self,
        variant: Variant,
        performance_data: Dict[str, Any],
        min_impressions: int = 200,
        min_spend: float = 20.0,
    ) -> Dict[str, Any]:
        """Evaluate a variant's performance."""
        variant.update_performance(performance_data)
        
        # Check if we have enough data
        if variant.impressions < min_impressions or variant.spend < min_spend:
            return {
                "status": "insufficient_data",
                "confidence": 0.0,
            }
        
        # Bayesian analysis
        bayesian_result = self.bayesian_test.update(variant)
        
        # Decision logic
        if bayesian_result["confidence"] > 0.7:
            if variant.roas > 1.5 and variant.ctr > 0.01:
                status = "winner"
            elif variant.roas < 0.8 or variant.ctr < 0.005:
                status = "loser"
            else:
                status = "neutral"
        else:
            status = "testing"
        
        return {
            "status": status,
            "confidence": bayesian_result["confidence"],
            "expected_value": bayesian_result["expected_value"],
            "variant": variant,
        }
    
    def select_winner(
        self,
        variants: List[Variant],
    ) -> Optional[Variant]:
        """Select winning variant from a test."""
        if not variants:
            return None
        
        # Evaluate all variants
        evaluated = []
        for variant in variants:
            if variant.impressions > 0:
                evaluation = self.evaluate_variant(variant, variant.performance)
                evaluated.append((variant, evaluation))
        
        if not evaluated:
            return None
        
        # Sort by expected value and confidence
        evaluated.sort(
            key=lambda x: x[1]["expected_value"] * x[1]["confidence"],
            reverse=True
        )
        
        return evaluated[0][0]


def create_variant_testing_engine(supabase_client=None) -> VariantTestingEngine:
    """Create a variant testing engine."""
    return VariantTestingEngine(supabase_client=supabase_client)


__all__ = [
    "VariantTestingEngine",
    "Variant",
    "VariantGenerator",
    "BayesianABTest",
    "create_variant_testing_engine",
]

