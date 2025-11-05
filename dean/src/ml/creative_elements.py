"""
Creative Element Decomposition
Breaks down creative performance by individual elements
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CreativeElementAnalyzer:
    """Analyzes creative performance by element."""
    
    def __init__(self, supabase_client=None):
        self.supabase_client = supabase_client
        self.element_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_roas": 0.0,
            "total_ctr": 0.0,
            "total_purchases": 0,
            "total_spend": 0.0,
        })
    
    def analyze_element_performance(
        self,
        creative_data: Dict[str, Any],
        performance_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze performance of individual elements."""
        # Extract elements
        image_prompt = creative_data.get("image_prompt", "")
        text_overlay = creative_data.get("text_overlay", "")
        ad_copy = creative_data.get("ad_copy", {})
        
        # Extract prompt components
        prompt_keywords = self._extract_prompt_keywords(image_prompt)
        
        # Track performance by element
        roas = performance_data.get("roas", 0.0)
        ctr = performance_data.get("ctr", 0.0)
        purchases = performance_data.get("purchases", 0)
        spend = performance_data.get("spend", 0.0)
        
        # Track prompt keywords
        for keyword in prompt_keywords:
            elem = self.element_performance[f"prompt:{keyword}"]
            elem["count"] += 1
            elem["total_roas"] += roas
            elem["total_ctr"] += ctr
            elem["total_purchases"] += purchases
            elem["total_spend"] += spend
        
        # Track text overlay
        if text_overlay:
            elem = self.element_performance[f"text_overlay:{text_overlay}"]
            elem["count"] += 1
            elem["total_roas"] += roas
            elem["total_ctr"] += ctr
            elem["total_purchases"] += purchases
            elem["total_spend"] += spend
        
        # Track ad copy elements
        for key, value in ad_copy.items():
            if value:
                elem = self.element_performance[f"ad_copy:{key}:{value[:50]}"]
                elem["count"] += 1
                elem["total_roas"] += roas
                elem["total_ctr"] += ctr
                elem["total_purchases"] += purchases
                elem["total_spend"] += spend
        
        return {
            "elements_tracked": len(self.element_performance),
            "prompt_keywords": prompt_keywords,
        }
    
    def _extract_prompt_keywords(self, prompt: str) -> List[str]:
        """Extract meaningful keywords from prompt."""
        # Important keywords
        important_keywords = [
            "editorial", "cinematic", "minimalist", "sophisticated",
            "luxury", "refined", "premium", "timeless", "elegant",
            "natural daylight", "soft shadows", "warm golden hour",
            "studio lighting", "professional", "dramatic contrast",
            "soft diffused light", "calm", "confidence", "discipline",
        ]
        
        prompt_lower = prompt.lower()
        found_keywords = [
            kw for kw in important_keywords
            if kw in prompt_lower
        ]
        
        return found_keywords
    
    def get_top_elements(
        self,
        element_type: Optional[str] = None,
        top_k: int = 10,
        min_count: int = 2,
    ) -> List[Dict[str, Any]]:
        """Get top performing elements."""
        scored_elements = []
        
        for element_key, data in self.element_performance.items():
            if element_type and not element_key.startswith(element_type):
                continue
            
            if data["count"] < min_count:
                continue
            
            # Calculate average performance
            avg_roas = data["total_roas"] / data["count"] if data["count"] > 0 else 0.0
            avg_ctr = data["total_ctr"] / data["count"] if data["count"] > 0 else 0.0
            
            # Composite score
            score = (avg_roas * 0.6) + (avg_ctr * 100 * 0.4)
            
            scored_elements.append({
                "element": element_key,
                "count": data["count"],
                "avg_roas": avg_roas,
                "avg_ctr": avg_ctr,
                "score": score,
            })
        
        # Sort by score
        scored_elements.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_elements[:top_k]
    
    def recommend_element_combinations(
        self,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recommend element combinations based on top performers."""
        # Get top elements of each type
        top_prompts = self.get_top_elements("prompt:", top_k=5)
        top_text_overlays = self.get_top_elements("text_overlay:", top_k=3)
        top_ad_copy = self.get_top_elements("ad_copy:", top_k=5)
        
        # Generate combinations
        combinations = []
        
        for prompt_elem in top_prompts[:3]:
            for text_elem in top_text_overlays[:2]:
                for copy_elem in top_ad_copy[:2]:
                    # Calculate expected score
                    expected_score = (
                        prompt_elem["score"] * 0.4 +
                        text_elem["score"] * 0.3 +
                        copy_elem["score"] * 0.3
                    )
                    
                    combinations.append({
                        "prompt_element": prompt_elem["element"],
                        "text_overlay_element": text_elem["element"],
                        "ad_copy_element": copy_elem["element"],
                        "expected_score": expected_score,
                    })
        
        # Sort by expected score
        combinations.sort(key=lambda x: x["expected_score"], reverse=True)
        
        return combinations[:top_k]


def create_creative_element_analyzer(supabase_client=None) -> CreativeElementAnalyzer:
    """Create a creative element analyzer."""
    return CreativeElementAnalyzer(supabase_client=supabase_client)


__all__ = ["CreativeElementAnalyzer", "create_creative_element_analyzer"]

