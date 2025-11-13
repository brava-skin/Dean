"""
Advanced Creative Intelligence
Style transfer, templates, versioning, quality checks
"""

from __future__ import annotations

import logging
import hashlib
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CreativeTemplate:
    """Creative template with performance tracking."""
    template_id: str
    name: str
    structure: Dict[str, Any]
    performance_score: float = 0.0
    usage_count: int = 0
    average_roas: float = 0.0
    average_ctr: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class CreativeTemplateLibrary:
    """Library of creative templates."""
    
    def __init__(self) -> None:
        self.templates: Dict[str, CreativeTemplate] = {}
    
    def add_template(
        self,
        name: str,
        structure: Dict[str, Any],
    ) -> str:
        """Add a new template."""
        template_id = hashlib.md5(
            f"{name}_{datetime.now()}".encode()
        ).hexdigest()[:12]
        
        template = CreativeTemplate(
            template_id=template_id,
            name=name,
            structure=structure,
        )
        
        self.templates[template_id] = template
        return template_id
    
    def get_top_templates(self, top_k: int = 5) -> List[CreativeTemplate]:
        """Get top performing templates."""
        sorted_templates = sorted(
            self.templates.values(),
            key=lambda t: t.performance_score,
            reverse=True,
        )
        return sorted_templates[:top_k]
    
    def update_performance(
        self,
        template_id: str,
        roas: float,
        ctr: float,
    ) -> None:
        """Update template performance."""
        if template_id not in self.templates:
            return
        
        template = self.templates[template_id]
        template.usage_count += 1
        
        # Update averages
        if template.usage_count == 1:
            template.average_roas = roas
            template.average_ctr = ctr
        else:
            template.average_roas = (
                (template.average_roas * (template.usage_count - 1) + roas) /
                template.usage_count
            )
            template.average_ctr = (
                (template.average_ctr * (template.usage_count - 1) + ctr) /
                template.usage_count
            )
        
        # Update performance score
        template.performance_score = (
            template.average_roas * 0.6 + template.average_ctr * 100 * 0.4
        )


class CreativeVersionManager:
    """Manages creative versions and rollback."""
    
    def __init__(self) -> None:
        self.versions: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_version(
        self,
        creative_id: str,
        creative_data: Dict[str, Any],
    ) -> str:
        """Create a new version."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if creative_id not in self.versions:
            self.versions[creative_id] = []
        
        version_data = {
            "version": version,
            "creative_data": creative_data.copy(),
            "created_at": datetime.now().isoformat(),
        }
        
        self.versions[creative_id].append(version_data)
        return version
    
    def rollback(
        self,
        creative_id: str,
        target_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Rollback to a previous version."""
        if creative_id not in self.versions:
            return None
        
        versions = self.versions[creative_id]
        
        if target_version:
            # Rollback to specific version
            for v in versions:
                if v["version"] == target_version:
                    return v["creative_data"]
        else:
            # Rollback to previous version
            if len(versions) > 1:
                return versions[-2]["creative_data"]
        
        return None


class CreativeQualityChecker:
    """Automated creative quality checks."""
    
    def __init__(self) -> None:
        self.quality_rules: List[Callable[[Dict[str, Any]], bool]] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default quality rules."""
        
        def check_image_present(creative: Dict[str, Any]) -> bool:
            return bool(creative.get("image_path") or creative.get("image_url"))
        
        def check_text_overlay(creative: Dict[str, Any]) -> bool:
            text = creative.get("text_overlay", "")
            return len(text) > 0 and len(text) <= 100
        
        def check_ad_copy(creative: Dict[str, Any]) -> bool:
            ad_copy = creative.get("ad_copy", {})
            return bool(
                ad_copy.get("headline") and
                ad_copy.get("primary_text")
            )
        
        self.quality_rules = [
            check_image_present,
            check_text_overlay,
            check_ad_copy,
        ]
    
    def check_quality(self, creative: Dict[str, Any]) -> Dict[str, Any]:
        """Check creative quality."""
        results = {}
        passed = 0
        total = len(self.quality_rules)
        
        for i, rule in enumerate(self.quality_rules):
            try:
                result = rule(creative)
                results[f"rule_{i}"] = result
                if result:
                    passed += 1
            except Exception as e:
                results[f"rule_{i}"] = False
                logger.error(f"Quality check rule {i} error: {e}")
        
        quality_score = passed / total if total > 0 else 0.0
        
        return {
            "quality_score": quality_score,
            "passed": passed,
            "total": total,
            "results": results,
            "passed_checks": quality_score >= 0.8,
        }


class StyleTransferEngine:
    """Style transfer from winning creatives."""
    
    def __init__(self, creative_dna: Optional[Any] = None) -> None:
        self.creative_dna = creative_dna
        self.winning_styles: Dict[str, Dict[str, Any]] = {}
    
    def extract_style(
        self,
        creative_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract style from creative."""
        return {
            "prompt_style": creative_data.get("image_prompt", "").split(", ")[:5],
            "text_style": creative_data.get("text_overlay", ""),
            "color_palette": creative_data.get("metadata", {}).get("colors", []),
        }
    
    def apply_style(
        self,
        base_creative: Dict[str, Any],
        style: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply style to base creative."""
        modified = base_creative.copy()
        
        # Apply prompt style
        if style.get("prompt_style"):
            original_prompt = modified.get("image_prompt", "")
            style_modifiers = ", ".join(style["prompt_style"])
            modified["image_prompt"] = f"{original_prompt}, {style_modifiers}"
        
        # Apply text style
        if style.get("text_style"):
            modified["text_overlay"] = style["text_style"]
        
        return modified


def create_template_library() -> CreativeTemplateLibrary:
    """Create template library."""
    return CreativeTemplateLibrary()


def create_version_manager() -> CreativeVersionManager:
    """Create version manager."""
    return CreativeVersionManager()


def create_quality_checker() -> CreativeQualityChecker:
    """Create quality checker."""
    return CreativeQualityChecker()


def create_style_transfer(creative_dna=None) -> StyleTransferEngine:
    """Create style transfer engine."""
    return StyleTransferEngine(creative_dna)


__all__ = [
    "CreativeTemplateLibrary",
    "CreativeVersionManager",
    "CreativeQualityChecker",
    "StyleTransferEngine",
    "create_template_library",
    "create_version_manager",
    "create_quality_checker",
    "create_style_transfer",
]

