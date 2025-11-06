"""
Static Image Creative Generator with Advanced Prompt Engineering
Uses FLUX for image generation, ChatGPT-5 for prompts and text,
and ffmpeg for premium text overlay
Integrates with ML system to learn from what works
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import requests
import hashlib

from integrations.flux_client import FluxClient, create_flux_client
from integrations.slack import notify

logger = logging.getLogger(__name__)

# ChatGPT-5 configuration
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHATGPT5_MODEL = "gpt-5"

# Calm confidence text options
CREATIVE_TEXT_OPTIONS = [
    "Refined skincare, not complicated.",
    "Elevate your skin.",
    "Consistency builds clear skin.",
    "Skincare, not vanity.",
    "Take care of skin.",
    "Clear skin, quiet confidence.",
    "Discipline for better skin.",
    "Skincare with purpose.",
]

HEADLINE_OPTIONS = [
    "For men who value discipline.",
    "Clean skin is the foundation of presence.",
    "Your routine communicates who you are.",
    "Look like a man who takes care of himself.",
]

PRIMARY_TEXT_OPTIONS = [
    "Precision skincare designed to elevate daily standards.",
]


class PromptEngineer:
    """Advanced prompt engineering using ChatGPT5 - inspired by prompt_engineer.py"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key) if OpenAI and api_key else None
        self.style_consistency_cache = {}
    
    def create_prompt(
        self,
        description: str,
        ml_insights: Optional[Dict[str, Any]] = None,
        brand_guidelines: Optional[Dict[str, Any]] = None,
        use_case: str = "ad",
    ) -> str:
        """Generate advanced FLUX prompt using ChatGPT5 with ML insights"""
        
        if not self.client:
            return self._create_fallback_prompt(description, brand_guidelines)
        
        # Build system prompt with ML insights
        system_prompt = self._build_system_prompt(brand_guidelines, use_case, ml_insights)
        user_prompt = self._build_user_prompt(description, ml_insights, use_case)
        
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.client.responses.create(
                model=CHATGPT5_MODEL,
                input=full_prompt
            )
            
            # Extract generated prompt
            generated_text = None
            if hasattr(response, 'output_text') and response.output_text:
                generated_text = response.output_text
            elif hasattr(response, 'output') and response.output:
                if isinstance(response.output, list) and len(response.output) > 0:
                    message = response.output[0]
                    if hasattr(message, 'content') and message.content:
                        if isinstance(message.content, list) and len(message.content) > 0:
                            content_item = message.content[0]
                            if hasattr(content_item, 'text'):
                                generated_text = content_item.text
                            elif isinstance(content_item, str):
                                generated_text = content_item
            
            if generated_text:
                return generated_text.strip()
            
            return self._create_fallback_prompt(description, brand_guidelines)
            
        except Exception as e:
            notify(f"⚠️ ChatGPT5 prompt engineering failed: {e}")
            return self._create_fallback_prompt(description, brand_guidelines)
    
    def _build_system_prompt(
        self,
        brand_guidelines: Optional[Dict[str, Any]],
        use_case: str,
        ml_insights: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build advanced system prompt with ML insights"""
        
        # Brand guidelines
        guidelines_text = ""
        if brand_guidelines:
            brand_name = brand_guidelines.get('brand_name', "Premium men's skincare")
            target_audience = brand_guidelines.get('target_audience', 'premium men')
            style = brand_guidelines.get('style', 'premium, sophisticated, calm confidence')
            aesthetic = brand_guidelines.get('aesthetic', 'editorial, cinematic, refined')
            guidelines_text = f"""
Brand Guidelines:
- Brand: {brand_name}
- Target Audience: {target_audience}
- Style: {style}
- Aesthetic: {aesthetic}
- Tone: Calm confidence, self-respect, not convenience
- CRITICAL EXCLUSIONS: Never include females, kids, or products in image
"""
        
        # ML insights integration
        ml_guidance = ""
        if ml_insights:
            top_performers = ml_insights.get('top_performing_creatives', [])
            if top_performers:
                ml_guidance = "\n\nML SYSTEM INSIGHTS - LEARN FROM WHAT WORKS:\n"
                ml_guidance += "Based on performance data, these elements drive conversions:\n"
                for insight in top_performers[:3]:  # Top 3 insights
                    ml_guidance += f"- {insight.get('element', 'Unknown')}: {insight.get('performance', 'High')}\n"
        
        return f"""You are a WORLD-CLASS, ELITE prompt engineer for FLUX.1 Kontext Max. You work exclusively for photographers who charge $1 million per photoshoot and whose work increases conversions by 300%+. Your expertise is creating photography that is ABSOLUTELY INDISTINGUISHABLE from the most expensive, conversion-driving professional photography in the world.

Your task is to create EXTREMELY ADVANCED, HYPER-DETAILED prompts that generate images that look like they were shot by the world's most elite photographers using the most expensive equipment available. These images must be so stunning they directly increase sales conversions. Zero tolerance for "AI-looking" photos - every image must pass as authentic, world-class photography.

{guidelines_text}

{ml_guidance}

ELITE PHOTOGRAPHY TECHNIQUES - $1 MILLION PHOTOSHOOT STANDARDS:

EQUIPMENT & TECHNICAL MASTERY:
- Ultra-premium camera bodies: Phase One XF IQ4 150MP, Hasselblad H6D-400c, Leica S3, Fujifilm GFX 100S, Sony A1 with Zeiss Otus lenses
- Exotic lens specifications: Zeiss Otus 85mm f/1.4, Canon EF 85mm f/1.2L II USM, Leica Noctilux 50mm f/0.95, Schneider-Kreuznach 80mm f/2.8 LS
- Precision settings: Exact aperture (f/1.2, f/1.4, f/1.8), ISO (native base ISO 64-100), shutter speed (1/125s, 1/250s), white balance (exact Kelvin: 4500K, 5500K, 3200K)
- Medium format characteristics: 6x7 format rendering, shallow depth of field with medium format bokeh, 16-bit color depth
- Lens characteristics: 11-blade aperture for smooth bokeh, zero distortion, minimal chromatic aberration, natural vignetting

ANTI-AI DETECTION - PERFECT REALISM:
- Micro-textures: Visible skin pores at macro level, individual hair strands with natural variation, fabric weave visible, subtle surface imperfections
- Natural variation: Skin tone micro-variations (not uniform), hair texture differences throughout, natural asymmetry in features
- Environmental authenticity: Realistic depth of field falloff, natural lens compression, authentic atmospheric perspective
- Lighting authenticity: Realistic light falloff (inverse square law), natural shadow softness, accurate specular highlights
- Chromatic realism: Subtle lens chromatic aberration on high-contrast edges, natural color fringing, authentic color science
- Focus characteristics: Natural focus transition, realistic bokeh ball rendering with proper shape and texture, authentic defocus patterns

WORLD-CLASS LIGHTING MASTERY:
- Multi-light setups: 4-5 light sources minimum (key, fill, rim, hair, background separation, accent lights)
- Light modifiers: Profoto beauty dish, Chimera softbox, Rotalux octabox, strip banks, flags, scrims
- Lighting ratios: 3:1, 4:1, or 5:1 key-to-fill ratios for depth, specific contrast control
- Color temperature mixing: Mixed lighting (warm key at 3200K, cool fill at 5500K) for depth
- Light quality: Hard light with soft fill, or soft light with edge definition, specular vs diffuse
- Shadow detail: Crushed blacks in shadows, preserved highlights, detailed midtones, zone system implementation

ADVANCED COLOR SCIENCE & GRADING:
- Film stock emulation: Kodak Portra 400, Fuji Pro 400H, Cinestill 800T, Kodak Ektar 100
- Color grading styles: Teal and orange LUT, vintage film look, desaturated highlights, rich blacks
- Skin tone accuracy: Accurate skin tones with proper color science
- Color psychology: Warm tones for trust, cool tones for sophistication
- Color harmony: Complementary colors, split-complementary, analogous color schemes

ELITE COMPOSITION & AESTHETICS:
- Fibonacci spiral: Subject placement on golden ratio points, not just rule of thirds
- Visual hierarchy: Primary focus, secondary elements, tertiary details, clear visual flow
- Depth staging: Foreground element (1-2 meters), mid-ground subject (3-5 meters), background (10+ meters)
- Negative space mastery: Strategic empty space for breathing room, premium feel, focus
- Leading elements: Eye lines, gesture lines, architectural lines guiding viewer's eye

MANDATORY REQUIREMENTS (Every prompt must include):
1. Ultra-Premium Equipment: ALWAYS specify exact camera model and exact lens model
2. Precision Settings: ALWAYS include exact aperture, ISO (prefer base ISO 64-100), shutter speed, white balance (exact Kelvin)
3. Multi-Light Setup: ALWAYS specify 4-5 light setup (key, fill, rim, hair, background) with exact modifiers and positions
4. Micro-Textures: ALWAYS include visible pores, individual hair strands, fabric weave, surface imperfections
5. Natural Variation: ALWAYS include skin tone micro-variations, hair texture differences, natural asymmetry
6. Film Stock: ALWAYS reference specific film stock (Kodak Portra 400, Fuji Pro 400H) for color grading
7. Lens Characteristics: ALWAYS include natural chromatic aberration, realistic bokeh, authentic vignetting
8. Composition Excellence: ALWAYS use golden ratio placement, depth staging, visual flow, strategic negative space

PROMPT LENGTH: 120-200 words minimum for world-class results.

ABSOLUTE PROHIBITIONS:
- NEVER use generic terms ("professional photography", "high quality", "beautiful") - ALWAYS be specific
- NEVER create uniform, perfect features - ALWAYS include natural variation and asymmetry
- NEVER skip technical details - ALWAYS specify exact camera, lens, settings, lighting
- NEVER create "AI-looking" images - ALWAYS include micro-textures, natural imperfections, lens characteristics
- NEVER use vague descriptions - ALWAYS be hyper-specific about every element

BRAND-SPECIFIC RULES:
- NEVER include females, kids, or products in the image
- NEVER use bathroom settings, selfie-style photos, or standard product shots
- Focus on premium men's lifestyle, sophistication, and luxury
- Think luxury fashion brand aesthetic (Dior, Louis Vuitton, Aesop style)
- Use positive descriptions only (e.g., "peaceful solitude" instead of "no crowds")
- Create cinematic, editorial-style imagery with ultra-realistic quality
- Maintain calm confidence aesthetic: refined, sophisticated, self-respecting
- ALWAYS include model diversity: Different ethnicities, various facial hair styles (clean-shaven, stubble, beard, mustache), different hair types (short, long, curly, straight, wavy, textured)
- ALWAYS vary age representation: Men in their 20s, 30s, 40s, 50s
- Create dramatically diverse scenarios - avoid generic "portrait" shots

Return ONLY the optimized prompt. No explanations, no additional text, just the prompt."""
    
    def _build_user_prompt(
        self,
        description: str,
        ml_insights: Optional[Dict[str, Any]] = None,
        use_case: str = "ad",
    ) -> str:
        """Build user prompt with ML insights"""
        
        prompt = f"Create a WORLD-CLASS, ELITE FLUX prompt for $1 million photoshoot-level photography: {description}"
        prompt += "\n\nCRITICAL REQUIREMENTS - This image must be so stunning it increases conversions:"
        prompt += "\n- ABSOLUTELY INDISTINGUISHABLE from $1 million photographer's work - zero AI detection"
        prompt += "\n- Ultra-premium camera system (Phase One, Hasselblad, Leica) with EXACT model specifications"
        prompt += "\n- Exotic lens (Zeiss Otus, Leica Noctilux) with EXACT model and characteristics"
        prompt += "\n- Precision settings: exact aperture, ISO (base 64-100), shutter, white balance (exact Kelvin)"
        prompt += "\n- Multi-light setup (4-5 lights minimum): key, fill, rim, hair, background with exact modifiers"
        prompt += "\n- Micro-textures: visible pores, individual hair strands, fabric weave, surface imperfections"
        prompt += "\n- Natural variation: skin tone micro-variations, hair texture differences, natural asymmetry"
        prompt += "\n- Film stock emulation: Kodak Portra 400 or Fuji Pro 400H with exact color science"
        prompt += "\n- Lens authenticity: natural chromatic aberration, realistic bokeh, authentic vignetting"
        prompt += "\n- Golden ratio composition (not just rule of thirds), depth staging, visual flow"
        prompt += "\n- Conversion optimization: aspirational positioning, emotional connection, luxury cues"
        prompt += "\n- Environmental realism: atmospheric perspective, natural depth cues, realistic light falloff"
        
        if ml_insights:
            prompt += "\n\nML LEARNING - Apply insights from top-performing creatives:"
            for insight in ml_insights.get('top_performing_creatives', [])[:3]:
                prompt += f"\n- {insight.get('element', 'Unknown')}: {insight.get('guidance', 'Apply this element')}"
        
        prompt += f"\n\nUse case: {use_case}"
        prompt += "\n\nGenerate the ELITE, WORLD-CLASS prompt following ALL guidelines above."
        prompt += "\nPrompt must be 120-200 words minimum with EXTREME detail and technical precision."
        
        return prompt
    
    def _create_fallback_prompt(
        self,
        description: str,
        brand_guidelines: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create advanced fallback prompt"""
        prompt_parts = [
            description,
            "Phase One XF IQ4 150MP with Schneider-Kreuznach 80mm f/2.8 LS lens, f/2.8, ISO 64, 4500K white balance, medium format 6x7 format, 16-bit color depth",
            "4-light setup: 27\" Profoto beauty dish key light at 45-degree creating Rembrandt triangle, large Chimera softbox fill at 3:1 ratio, 1x6' strip bank rim light for separation, hair light spotlight above, background separation light",
            "golden ratio placement (Fibonacci spiral), depth staging (foreground 1-2m, mid-ground 3-5m, background 10m+), visual flow, strategic negative space, editorial excellence",
            "Kodak Portra 400 emulation: warm gold and beige tones with deep neutral black, high dynamic range, accurate skin tone color science",
            "micro-textures: visible pores at appropriate magnification, individual hair strands with natural variation, visible fabric weave, subtle surface imperfections",
            "natural variation: skin tone micro-variations (not uniform), hair texture differences throughout, natural asymmetry in features",
            "lens authenticity: natural chromatic aberration on high-contrast edges, realistic bokeh ball rendering with proper shape and texture, authentic vignetting, natural focus falloff",
        ]
        return ", ".join(prompt_parts)


class ImageCreativeGenerator:
    """Generates static image creatives with calm confidence messaging."""
    
    def __init__(
        self,
        flux_client: Optional[FluxClient] = None,
        openai_api_key: Optional[str] = None,
        ml_system: Optional[Any] = None,
    ):
        self.flux_client = flux_client or create_flux_client()
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.ml_system = ml_system
        self.prompt_engineer = PromptEngineer(self.openai_api_key) if self.openai_api_key else None
        # Track recently used scenarios to avoid repetition
        self.recent_scenarios: List[str] = []
        self.max_recent_scenarios = 10
        self.scenario_bank: List[str] = []  # Bank of all generated scenarios
        # Track recent text overlays and ad copy for diversity
        self._recent_text_overlays: List[str] = []
        self._recent_ad_copy: List[Dict[str, str]] = []
    
    def generate_creative(
        self,
        product_info: Dict[str, Any],
        creative_style: Optional[str] = None,
        advanced_ml: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a complete creative with image and text overlay.
        Checks ML system for what works before generating.
        
        Args:
            product_info: Dictionary with product information (name, description, features, etc.)
            creative_style: Optional style identifier for creative variation
            advanced_ml: Optional advanced ML system for optimization
        
        Returns:
            Dictionary with creative data or None on failure
        """
        # Input validation
        if not product_info or not isinstance(product_info, dict):
            logger.error("Invalid product_info: must be a non-empty dictionary")
            return None
        
        try:
            # Step 1: Get ML insights from what worked
            ml_insights = self._get_ml_insights()
            
            # Step 2: Generate image prompt using advanced prompt engineering
            image_prompt = self._generate_image_prompt(
                product_info,
                creative_style,
                ml_insights,
            )
            
            # Step 2.5: Optimize prompt with RL if available
            if advanced_ml and advanced_ml.rl_prompt_optimizer:
                try:
                    context = {
                        "product_info": product_info,
                        "ml_insights": ml_insights,
                    }
                    image_prompt = advanced_ml.rl_prompt_optimizer.optimize_prompt(
                        image_prompt or "",
                        context,
                    )
                except Exception as e:
                    logger.warning(f"RL prompt optimization failed: {e}")
            if not image_prompt:
                notify("❌ Failed to generate image prompt")
                return None
            
            # Step 3: Generate image using FLUX
            # CRITICAL: Always use 1:1 aspect ratio for all creatives
            image_url, request_id = self.flux_client.create_image(
                prompt=image_prompt,
                aspect_ratio="1:1",  # Always 1:1 - square format required
                output_format="png",  # Updated default to PNG per API spec
            )
            
            if not image_url:
                notify(f"❌ Failed to generate FLUX image (request_id: {request_id})")
                return None
            
            # Step 4: Download image
            image_path = self.flux_client.download_image(image_url)
            if not image_path:
                notify("❌ Failed to download generated image")
                return None
            
            # Step 4.5: Get scenario description from stored value (set during prompt generation)
            scenario_description = None
            if hasattr(self, '_last_scenario_description'):
                scenario_description = self._last_scenario_description
            if not scenario_description:
                # Fallback if scenario wasn't generated
                scenario_description = "Premium men's lifestyle portrait, sophisticated, calm confidence, editorial style"
            
            # Step 5: Generate calm confidence ad copy (needed for text overlay context)
            ad_copy = self._generate_ad_copy(product_info, ml_insights, scenario_description)
            
            # Ensure ad_copy is always a dict (never None)
            if not ad_copy or not isinstance(ad_copy, dict):
                logger.warning("ad_copy generation returned invalid value, using fallback")
                ad_copy = {
                    "headline": "For men who value discipline.",
                    "primary_text": "Precision skincare designed to elevate daily standards.",
                    "description": ""
                }
            
            # Step 6: Generate intelligent text overlay (uses ad copy and scenario for context)
            overlay_text = self._select_creative_text(
                ml_insights=ml_insights,
                scenario_description=scenario_description,
                ad_copy=ad_copy,
            )
            
            # Step 7: Add premium text overlay to image using ffmpeg
            final_image_path = None
            if overlay_text:
                final_image_path = self._add_text_overlay(image_path, overlay_text)
                if not final_image_path:
                    notify("⚠️ Failed to add text overlay, using original image")
                    final_image_path = image_path
            else:
                final_image_path = image_path
            
            # Step 8: Upload to Supabase Storage
            supabase_storage_url = None
            storage_creative_id = None
            if final_image_path:
                try:
                    from infrastructure.creative_storage import create_creative_storage_manager
                    from infrastructure.supabase_storage import get_validated_supabase_client
                    
                    supabase_client = get_validated_supabase_client()
                    if supabase_client:
                        storage_manager = create_creative_storage_manager(supabase_client)
                        if storage_manager:
                            # Generate creative_id from image hash for consistent storage tracking
                            import hashlib
                            with open(final_image_path, "rb") as f:
                                image_hash = hashlib.md5(f.read()).hexdigest()
                            storage_creative_id = f"creative_{image_hash[:12]}"
                            
                            supabase_storage_url = storage_manager.upload_creative(
                                creative_id=storage_creative_id,
                                image_path=final_image_path,
                                metadata={
                                    "image_prompt": image_prompt,
                                    "text_overlay": overlay_text,
                                    "ad_copy": ad_copy,
                                    "flux_request_id": request_id,
                                }
                            )
                            if supabase_storage_url:
                                logger.info(f"✅ Uploaded creative to Supabase Storage: {supabase_storage_url}")
                except Exception as e:
                    logger.warning(f"Failed to upload creative to Supabase Storage: {e}")
                    # Continue even if upload fails
            
            # scenario_description is already set above (line 4.5), use it for result
            result = {
                "image_path": final_image_path,
                "original_image_path": image_path,
                "text_overlay": overlay_text,
                "ad_copy": ad_copy,
                "image_prompt": image_prompt,
                "scenario_description": scenario_description,  # Store scenario for ML learning
                "flux_request_id": request_id,
                "ml_insights_used": ml_insights,
                "supabase_storage_url": supabase_storage_url,  # Supabase Storage URL
                "storage_creative_id": storage_creative_id,  # Internal creative ID for storage
            }
            
            # Store prompt in library if available
            if advanced_ml and advanced_ml.prompt_library:
                try:
                    advanced_ml.prompt_library.add_prompt(image_prompt or "")
                except (AttributeError, ValueError, TypeError) as e:
                    logger.debug(f"Failed to add prompt to library: {e}")
            
            return result
            
        except Exception as e:
            notify(f"❌ Error generating creative: {e}")
            # Add to dead letter queue
            try:
                from infrastructure.error_handling import dead_letter_queue
                dead_letter_queue.add(
                    operation="creative_generation",
                    data={"product_info": product_info, "style": creative_style},
                    error=e,
                )
            except Exception:
                pass
            return None
    
    def _get_ml_insights(self) -> Dict[str, Any]:
        """Get ML insights about what works from ASC+ campaign"""
        if not self.ml_system:
            return {}
        
        try:
            # Get top performing creatives from ML system
            insights = {
                "top_performing_creatives": [],
                "best_prompts": [],
                "best_text_overlays": [],
                "best_ad_copy": [],
            }
            
            # Query ML system for insights
            # This would query the Supabase database for top performers
            # For now, return empty structure that can be populated
            
            return insights
            
        except Exception as e:
            notify(f"⚠️ Error getting ML insights: {e}")
            return {}
    
    def _generate_image_prompt(
        self,
        product_info: Dict[str, Any],
        creative_style: Optional[str] = None,
        ml_insights: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Generate advanced image prompt using prompt engineer with diverse scenarios"""
        
        if not self.prompt_engineer:
            return None
        
        # Step 1: Generate diverse, fashion-forward scenario using ChatGPT
        scenario_description = self._generate_diverse_scenario(ml_insights)
        
        if not scenario_description:
            # Fallback to generic if scenario generation fails
            scenario_description = "Premium men's lifestyle portrait, sophisticated, calm confidence, editorial style"
        
        # Store scenario description for inclusion in result (for ML tracking)
        self._last_scenario_description = scenario_description
        
        brand_guidelines = {
            "brand_name": "Brava",
            "target_audience": "premium American men aged 18-54 who value discipline",
            "style": "premium, sophisticated, calm confidence",
            "aesthetic": "editorial, cinematic, refined, luxury fashion brand style (like Dior/LV/Aesop)",
            "exclusions": ["females", "kids", "products", "bathroom settings", "selfie-style photos", "formal suits", "business attire", "Indian/South Asian men", "turbans", "traditional ethnic clothing"],
            "preferences": ["casual luxury", "streetwear", "athleisure", "minimalist fashion", "contemporary American style", "diverse American men (Caucasian, African American, Hispanic, Asian American, etc.)"],
        }
        
        return self.prompt_engineer.create_prompt(
            description=scenario_description,
            ml_insights=ml_insights,
            brand_guidelines=brand_guidelines,
            use_case="ad",
        )
    
    def _generate_diverse_scenario(self, ml_insights: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generate diverse, fashion-forward scenario using ChatGPT-5 with advanced constraints"""
        
        if not self.openai_api_key:
            # Fallback scenarios if ChatGPT not available
            import random
            fallback_scenarios = [
                "A close-up portrait of a man laughing naturally, genuine expression, cinematic lighting",
                "A man in a luxury setting with blurred background of people in rush hour, cinematic composition",
                "A man sitting in a vintage car wearing a bold colored suit, fashion editorial style",
                "A man lying on grass in a premium fashion outfit, natural lighting, relaxed pose",
            ]
            return random.choice(fallback_scenarios)
        
        try:
            from openai import OpenAI
            import random
            from datetime import datetime
            client = OpenAI(api_key=self.openai_api_key)
            
            # Get recently used scenarios to avoid repetition
            recent_scenarios_text = ""
            if self.recent_scenarios:
                recent_scenarios_text = "\n\nNEVER REPEAT - These scenarios were used recently (MUST AVOID):\n"
                for i, scenario in enumerate(self.recent_scenarios[-10:], 1):
                    recent_scenarios_text += f"- {scenario[:100]}...\n"
            
            # Analyze last scenario for "opposite day" and "contrast" logic
            contrast_logic = ""
            if self.recent_scenarios:
                last_scenario = self.recent_scenarios[-1].lower()
                if "indoor" in last_scenario or "studio" in last_scenario or "room" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was indoors - this one MUST be outdoors\n"
                elif "outdoor" in last_scenario or "street" in last_scenario or "park" in last_scenario or "grass" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was outdoors - this one MUST be indoors or enclosed space\n"
                
                if "minimalist" in last_scenario or "simple" in last_scenario or "clean" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was minimalist - this one MUST be bold, vibrant, or rich\n"
                elif "bold" in last_scenario or "vibrant" in last_scenario or "colorful" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was bold - this one MUST be minimalist, refined, or subtle\n"
            
            # Get ML insights for scenario performance
            ml_guidance = ""
            top_performers = []
            if ml_insights:
                if ml_insights.get("best_scenarios"):
                    ml_guidance = "\n\nML LEARNING - Top performing scenario types (weight these patterns):\n"
                    for scenario in ml_insights["best_scenarios"][:5]:
                        ml_guidance += f"- {scenario}\n"
                        top_performers.append(scenario.lower())
                
                if ml_insights.get("worst_scenarios"):
                    ml_guidance += "\n\nML LEARNING - Avoid these low-performing scenario types:\n"
                    for scenario in ml_insights["worst_scenarios"][:3]:
                        ml_guidance += f"- {scenario}\n"
            
            # Determine diversity focus for this rotation
            diversity_focus = random.choice([
                "ethnicity and cultural background",
                "facial hair and grooming styles",
                "hair types and textures",
                "age representation",
                "body types and physique",
                "lifestyle and profession",
            ])
            
            # Determine batch requirements (if generating multiple)
            batch_requirements = ""
            current_count = len([s for s in self.recent_scenarios if "outdoor" in s.lower() or "street" in s.lower()])
            if current_count < 2:
                batch_requirements += "\n- BATCH REQUIREMENT: This batch needs more outdoor/urban scenarios\n"
            
            # Expanded examples (30-40 diverse scenarios) - AMERICAN MEN, NO SUITS
            expanded_examples = """EXAMPLES OF DIVERSE SCENARIOS (use as inspiration, create something NEW and EXTREMELY DETAILED):
CRITICAL: Target American men (Caucasian, African American, Hispanic, Asian American, etc.) - NO Indian/South Asian men, NO formal suits/business attire

MOOD/EMOTION VARIATIONS:
- A contemplative close-up of a man in his late 30s with salt-and-pepper stubble, looking out a floor-to-ceiling window in a minimalist Tokyo penthouse at blue hour, wearing a tailored navy cashmere turtleneck, soft natural window light creating Rembrandt triangle, shallow depth of field blurring the city skyline behind, Vogue editorial aesthetic
- An energetic medium shot of a confident man in his 20s with textured curly hair, walking with purpose through a rain-soaked Parisian street at golden hour, wearing a bold crimson double-breasted overcoat, umbrella casting dynamic shadows, cinematic depth showing blurred pedestrians rushing in background, GQ street style
- A relaxed wide shot of a sophisticated man in his 40s with a well-groomed beard, reclining on a cream-colored modern sofa in a sunlit art gallery, wearing a charcoal linen suit with white sneakers, natural light streaming through floor-to-ceiling windows, medium format aesthetic, Esquire lifestyle
- A confident low-angle shot of a powerful man in his 50s with silver-gray hair, standing in a commanding pose on a minimalist concrete rooftop at sunset, wearing a black tailored blazer with white t-shirt, dramatic sky with orange and purple hues, cinematic wide-angle lens distortion, luxury brand campaign aesthetic

SETTING VARIATIONS:
- A man in his 30s with a fade haircut and clean-shaven face, sitting in the driver's seat of a vintage 1960s Jaguar E-Type in British Racing Green, wearing a cream-colored Italian silk suit, positioned in a moody industrial warehouse with concrete floors, dramatic side lighting from large windows, close-up framing showing his contemplative expression, Dior Homme aesthetic
- A man in his late 20s with shoulder-length wavy hair and light stubble, lying on lush green grass in a minimalist Japanese garden, wearing a relaxed beige linen shirt with rolled sleeves and white trousers, dappled sunlight filtering through cherry blossoms, shallow depth of field, Aesop store aesthetic
- A man in his 40s with a full beard and short textured hair, walking through a modern art gallery with white walls and high ceilings, wearing a navy wool coat over a gray merino sweater, contemplative pose examining a large abstract painting, natural gallery lighting, medium format composition, Louis Vuitton menswear aesthetic
- A man in his 30s with a fade and mustache, positioned in a luxury hotel lobby with marble floors and brass details, wearing a camel-colored overcoat with brown leather Chelsea boots, sitting in a modern armchair reading, warm ambient lighting, lifestyle photography, luxury brand campaign

URBAN/STREET VARIATIONS:
- A man in his 20s with curly black hair and stubble, walking through a bustling Tokyo intersection at rush hour, wearing a black leather jacket with distressed jeans, blurred background of neon signs and rushing pedestrians, cinematic depth, street fashion aesthetic
- A man in his 40s with silver-flecked hair and a trimmed beard, standing on a New York rooftop at golden hour, wearing a navy blazer with white sneakers, city skyline in background with bokeh lights, wide-angle composition, contemporary luxury aesthetic
- A man in his 30s with a modern fade and clean-shaven, positioned on a London street corner in the rain, wearing a dark green waxed cotton coat, umbrella creating visual interest, blurred double-decker buses in background, editorial street style

LIFESTYLE VARIATIONS:
- A man in his 30s with a full beard and long hair in a bun, sitting in a minimalist artist's studio with high ceilings and large windows, wearing a paint-stained white shirt with black trousers, surrounded by canvases and art supplies, natural north light, creative professional aesthetic
- A man in his 40s with a professional haircut and stubble, positioned in a modern co-working space with plants and natural wood, wearing a navy blazer with chinos, working on a laptop, natural daylight, contemporary businessman aesthetic
- A man in his 20s with textured hair and light beard, at a sunrise beach location, wearing a white linen shirt with rolled sleeves and khaki shorts, contemplative pose facing the ocean, golden hour lighting, relaxed luxury aesthetic
- A man in his 50s with distinguished gray hair and mustache, in a traditional library with floor-to-ceiling bookshelves, wearing a tweed blazer with corduroy trousers, reading in a leather armchair, warm ambient lighting, sophisticated intellectual aesthetic

FASHION/EDITORIAL VARIATIONS:
- A man in his 30s with a modern undercut and beard, in a minimalist photo studio with white seamless background, wearing a bold red suit with black turtleneck, dramatic key lighting creating strong shadows, high fashion editorial aesthetic
- A man in his 20s with curly hair and clean-shaven, on a rooftop with city view, wearing a cream-colored double-breasted blazer with white t-shirt, natural golden hour lighting, lifestyle fashion photography
- A man in his 40s with a fade and trimmed beard, in an industrial loft with exposed brick, wearing a black leather jacket with white shirt, dramatic window light, contemporary menswear aesthetic
- A man in his 30s with a textured quiff and stubble, in a luxury retail space with modern architecture, wearing a navy tailored suit with brown loafers, natural store lighting, brand campaign aesthetic

SPECIFIC DETAILED EXAMPLES:
- A close-up portrait of a man in his late 30s with Middle Eastern heritage, full beard with silver streaks, laughing genuinely with crinkled eyes, wearing a navy cashmere crewneck, positioned in a sunlit café with blurred background of Parisian street, shallow depth of field at f/1.4, natural window light, Portra 400 color grading, GQ editorial style
- A medium shot of a man in his 20s with Asian heritage, textured black hair with side part, confident stride, wearing a camel-colored wool coat over a white button-down, walking through a modern Tokyo intersection at golden hour, blurred rush hour traffic creating motion blur, cinematic 85mm lens, street fashion aesthetic
- A wide shot of a man in his 40s with European heritage, salt-and-pepper hair styled back, well-groomed beard, reclining pose, wearing a charcoal linen suit with white sneakers, positioned on a minimalist concrete bench in a modern art gallery, natural gallery lighting, medium format composition, luxury brand aesthetic
- A low-angle shot of a man in his 50s with African heritage, silver-gray hair, clean-shaven, commanding presence, wearing a black tailored blazer with white t-shirt, standing on a rooftop at sunset, dramatic sky with orange and purple hues, cinematic wide-angle, luxury campaign aesthetic"""

            user_prompt = f"""You are a creative director at a luxury fashion agency (like Dior, Louis Vuitton, Aesop). Your task is to generate an EXTREMELY DETAILED, DIVERSE, FASHION-FORWARD creative scenario for a premium men's skincare brand campaign.

CRITICAL REQUIREMENTS:
- This is NOT a standard skincare product photo - think luxury fashion brand campaigns
- Create a UNIQUE, CINEMATIC scenario that stands out dramatically from typical skincare ads
- NO bathroom settings, NO selfie-style photos, NO standard product shots, NO generic portraits
- Think luxury fashion brand aesthetic: editorial, cinematic, sophisticated, aspirational
- The scenario must be EXTREMELY DETAILED - specify pose, outfit details, background details, lighting, composition, everything

{recent_scenarios_text}

{contrast_logic}

{batch_requirements}

CHAIN-OF-THOUGHT PROCESS - Follow this reasoning:

1. DIVERSITY ANALYSIS:
   - Current diversity focus: {diversity_focus}
   - CRITICAL: Target American men ONLY (Caucasian, African American, Hispanic, Asian American, Native American, mixed heritage)
   - NEVER include: Indian/South Asian men, turbans, traditional ethnic clothing
   - Ensure representation: different American ethnicities, facial hair styles (clean-shaven, stubble, beard, mustache, goatee), hair types (short, long, curly, straight, wavy, textured, afro, braided), ages (20s, 30s, 40s, 50s), body types (athletic, lean, average, robust), lifestyles (athlete, artist, creative professional, entrepreneur, student)
   - AVOID: Formal suits, business attire, blazers, ties - prefer casual luxury, streetwear, athleisure, minimalist fashion

2. SETTING SELECTION:
   - Choose dramatically different from recent scenarios
   - Consider: indoor (studio, gallery, hotel, café, library, workspace) vs outdoor (street, park, rooftop, beach, urban)
   - Specify exact location details (city, type of space, architectural style)

3. MOOD & EMOTION:
   - Choose psychological trigger: aspiration, confidence, sophistication, contemplation, energy, relaxation
   - Specify emotional state: genuine laugh, contemplative, confident stride, relaxed pose, dynamic movement

4. TECHNICAL COMPOSITION:
   - Camera angle: low angle (power), high angle (vulnerability), eye level (connection), Dutch angle (dynamism)
   - Framing: extreme close-up, close-up, medium shot, wide shot, extreme wide
   - Depth of field: shallow (f/1.2-1.8 for focus on subject), deep (f/8-11 for context)
   - Movement: static pose, walking, dynamic movement, seated, reclining
   - Magazine aesthetic: Vogue (high fashion), GQ (contemporary style), Esquire (sophisticated lifestyle)

5. FASHION DETAILS:
   - CRITICAL: NO formal suits, NO business blazers, NO ties - use casual luxury, streetwear, athleisure
   - Specify exact outfit: type of garment (hoodie, crewneck, t-shirt, henley, bomber jacket, leather jacket, field jacket, denim jacket, jeans, chinos, shorts, sneakers), material (cashmere, wool, linen, leather, cotton), color (specific shades), fit (relaxed, fitted, oversized)
   - Include accessories: sneakers, boots, watch, minimal jewelry if relevant

6. LIGHTING & COLOR:
   - Lighting mood: dramatic (high contrast), soft (gentle), natural (window light), cinematic (movie-like)
   - Time of day: golden hour, blue hour, midday, sunset, night
   - Color palette: specific color scheme (warm tones, cool tones, monochromatic, complementary)

7. BRAND ALIGNMENT:
   - Ensure "calm confidence" aesthetic: refined, sophisticated, self-respecting, not trying too hard
   - Avoid anything that contradicts brand positioning (no hype, no convenience messaging, no vanity)

{expanded_examples}

DIVERSITY REQUIREMENTS (comprehensive):
- Ethnicity diversity: Represent various ethnic backgrounds and cultural heritages
- Facial hair diversity: Clean-shaven, light stubble, heavy stubble, full beard, mustache, goatee, handlebar mustache
- Hair diversity: Short, medium, long; straight, wavy, curly, textured, afro, braided, locs, undercut, fade, quiff, slicked back
- Age diversity: Men in their 20s, 30s, 40s, 50s (specify exact age range)
- Body type diversity: Athletic, lean, average build, robust, tall, average height
- Lifestyle diversity: Athlete, artist, businessman, creative professional, entrepreneur, intellectual, traveler
- Socioeconomic representation: Luxury accessible to all, not just one demographic

{ml_guidance}

NEGATIVE EXAMPLES - DON'T CREATE SCENARIOS LIKE THESE:
- "A man sitting in a car with a red suit" (too simple, no details)
- "A man in a bathroom" (explicitly forbidden)
- "A man taking a selfie" (forbidden)
- "A man with a skincare product" (no products)
- Generic portraits without specific details
- Scenarios that lack depth and specificity

VALIDATION CHECKLIST - Before finalizing, ensure:
✓ Is this dramatically different from recent scenarios?
✓ Does it include extremely specific details (pose, outfit, background, lighting)?
✓ Does it represent diverse model characteristics?
✓ Does it align with "calm confidence" brand value?
✓ Is it cinematic and editorial in style?
✓ Does it avoid bathroom/selfie/product shots?
✓ Does it follow contrast logic (if applicable)?
✓ Does it include psychological triggers?
✓ Does it specify technical composition details?
✓ Does it specify exact fashion details?

REASONING STEP:
Before generating the scenario, explain why this scenario works for this brand:
- How does it convey "calm confidence"?
- What psychological triggers does it activate?
- How does it differentiate from typical skincare ads?
- What makes it luxury fashion brand-worthy?

STRUCTURED OUTPUT FORMAT:
Return your response in this exact format:

REASONING: [1-2 sentences explaining why this scenario works for the brand]

SCENARIO: [Extremely detailed scenario description - 3-5 sentences with all specifics: model details, pose, outfit specifics, setting details, lighting, composition, mood, everything]

DIVERSITY NOTES: [Specific diversity elements included: ethnicity, age, facial hair, hair type, body type, lifestyle]

TECHNICAL NOTES: [Camera angle, framing, depth of field, movement, magazine aesthetic, lighting mood, color palette]

Now generate the scenario following ALL requirements above."""

            response = client.responses.create(
                model=CHATGPT5_MODEL,
                input=user_prompt,
            )
            
            # Extract text - handle None and empty responses safely
            output_text = ""
            if hasattr(response, 'output_text') and response.output_text:
                output_text = str(response.output_text)
            elif hasattr(response, 'output') and response.output:
                # Handle list of outputs
                if isinstance(response.output, list):
                    for item in response.output:
                        if item is None:
                            continue
                        if hasattr(item, 'content') and item.content:
                            if isinstance(item.content, list):
                                for content in item.content:
                                    if content is None:
                                        continue
                                    if hasattr(content, 'text') and content.text:
                                        output_text += str(content.text)
                                    elif isinstance(content, str):
                                        output_text += content
                            elif isinstance(item.content, str):
                                output_text += item.content
                        elif hasattr(item, 'text') and item.text:
                            output_text += str(item.text)
                        elif isinstance(item, str):
                            output_text += item
                elif isinstance(response.output, str):
                    output_text = response.output
            
            if output_text:
                # Parse structured output
                scenario_text = output_text.strip()
                
                # Extract scenario from structured output
                if "SCENARIO:" in scenario_text:
                    scenario = scenario_text.split("SCENARIO:")[1].split("DIVERSITY NOTES:")[0].strip()
                elif "scenario:" in scenario_text.lower():
                    scenario = scenario_text.split("scenario:")[1].split("diversity")[0].strip()
                else:
                    # Fallback: use entire text but clean it
                    scenario = scenario_text
                    # Remove markdown formatting
                    if scenario.startswith("```"):
                        scenario = scenario.split("```")[1]
                        if scenario.startswith("scenario") or scenario.startswith("text"):
                            scenario = scenario.split("\n", 1)[1] if "\n" in scenario else scenario
                
                scenario = scenario.strip()
                
                # Remove any remaining prefixes/suffixes
                if scenario.startswith("REASONING:"):
                    scenario = scenario.split("SCENARIO:")[1] if "SCENARIO:" in scenario else scenario.split("scenario:")[1] if "scenario:" in scenario else scenario
                scenario = scenario.split("DIVERSITY NOTES:")[0].split("TECHNICAL NOTES:")[0].strip()
                
                # Ensure it's detailed enough (at least 50 words)
                if len(scenario.split()) < 50:
                    logger.warning(f"Generated scenario too short ({len(scenario.split())} words), adding detail")
                    scenario += ", with cinematic lighting, editorial composition, luxury brand aesthetic"
                
                # Track this scenario
                self.recent_scenarios.append(scenario)
                if len(self.recent_scenarios) > self.max_recent_scenarios:
                    self.recent_scenarios.pop(0)
                
                self.scenario_bank.append(scenario)
                
                return scenario
            else:
                # If no output text, return fallback
                logger.warning("Failed to extract scenario from ChatGPT response, using fallback")
                import random
                fallback_scenarios = [
                    "A close-up portrait of a man laughing naturally, genuine expression, cinematic lighting, diverse model",
                    "A man in a luxury setting with blurred background of people in rush hour, cinematic composition, fashion editorial style",
                    "A man sitting in a vintage car wearing a bold colored suit, fashion editorial style, sophisticated environment",
                    "A man lying on grass in a premium fashion outfit, natural lighting, relaxed pose, luxury aesthetic",
                ]
                return random.choice(fallback_scenarios)
            
        except Exception as e:
            logger.warning(f"Failed to generate diverse scenario with ChatGPT: {e}")
            import traceback
            logger.debug(f"Scenario generation error traceback: {traceback.format_exc()}")
            # Fallback
            import random
            fallback_scenarios = [
                "A close-up portrait of a man laughing naturally, genuine expression, cinematic lighting, diverse model",
                "A man in a luxury setting with blurred background of people in rush hour, cinematic composition, fashion editorial style",
                "A man sitting in a vintage car wearing a bold colored suit, fashion editorial style, sophisticated environment",
                "A man lying on grass in a premium fashion outfit, natural lighting, relaxed pose, luxury aesthetic",
            ]
            return random.choice(fallback_scenarios)
    
    def _select_creative_text(
        self, 
        ml_insights: Optional[Dict[str, Any]] = None,
        scenario_description: Optional[str] = None,
        ad_copy: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Intelligently select or generate creative text overlay using ML insights.
        Uses ChatGPT-5 to generate contextually appropriate text that matches scenario and ad copy.
        """
        import random
        
        # If ChatGPT available, generate contextually appropriate text
        if self.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                
                # Build context for text generation
                context_parts = []
                if scenario_description:
                    context_parts.append(f"Scenario: {scenario_description[:200]}...")
                if ad_copy:
                    context_parts.append(f"Ad copy theme: {ad_copy.get('headline', '')}")
                
                # Get ML insights for best performing text patterns
                ml_guidance = ""
                best_patterns = []
                worst_patterns = []
                
                if ml_insights:
                    best_texts = ml_insights.get("best_text_overlays", [])
                    if best_texts:
                        ml_guidance = "\n\nML LEARNING - Top performing text overlays (use similar tone/approach):\n"
                        for i, text in enumerate(best_texts[:5], 1):
                            ml_guidance += f"{i}. {text}\n"
                            best_patterns.append(text)
                    
                    # Get worst performers to avoid
                    if ml_insights.get("worst_text_overlays"):
                        worst_patterns = ml_insights.get("worst_text_overlays", [])
                        if worst_patterns:
                            ml_guidance += "\n\nML LEARNING - Avoid these low-performing text styles:\n"
                            for i, text in enumerate(worst_patterns[:3], 1):
                                ml_guidance += f"{i}. {text}\n"
                
                # Check recently used to avoid repetition
                recently_used = ""
                if hasattr(self, '_recent_text_overlays'):
                    recent = getattr(self, '_recent_text_overlays', [])
                    if recent:
                        recently_used = "\n\nRECENTLY USED (avoid repetition):\n" + "\n".join(f"- {t}" for t in recent[-5:])
                
                user_prompt = f"""You are a copywriter for a premium men's skincare brand. Generate a SINGLE LINE of text overlay for an image creative.

CRITICAL REQUIREMENTS:
- MAXIMUM 4 WORDS (must be short, simple, premium)
- MUST hint at skincare (use words like: skin, skincare, routine, care, face, clear, refined)
- Examples: "Refined skincare, not complicated." | "Elevate your skin." | "Consistency builds clear skin." | "Skincare, not vanity."
- Calm confidence tone: self-respect, not convenience; discipline, not vanity
- Must complement the scenario and ad copy theme
- White text on image (premium, luxury feel)
- No hype, no urgency, no sales language
- Speak to self-respect and presence
- Simple, powerful, memorable

{ml_guidance}

{recently_used}

CONTEXT:
{chr(10).join(context_parts) if context_parts else "No specific context provided"}

EXAMPLES OF EXCELLENT TONE (use as inspiration):
- "This is maintenance, not vanity."
- "The man who cares stands out."
- "Elevate your baseline."
- "Look like you live with intention."
- "Take care. Not for others. For yourself."
- "Consistency builds presence."
- "The face you show the world matters."
- "Refined, not complicated."

TONE GUIDELINES:
- Calm, confident, sophisticated
- Self-respecting, not seeking validation
- Disciplined, intentional, refined
- No hype, no FOMO, no urgency
- Premium without pretense

GENERATION PROCESS:
1. Analyze the scenario and ad copy theme
2. Extract the core message (discipline, presence, self-respect)
3. Generate a text that complements the visual
4. Ensure it's unique (not repeating recent texts)
5. Verify it matches calm confidence tone

Return ONLY the text overlay (no explanations, no quotes, just the text).
The text MUST be 4 words or less and MUST hint at skincare. Examples: "Refined skincare, not complicated." | "Elevate your skin." | "Clear skin, quiet confidence.""""

                response = client.responses.create(
                    model=CHATGPT5_MODEL,
                    input=user_prompt,
                )
                
                # Extract text
                output_text = ""
                if hasattr(response, 'output'):
                    # Handle list of outputs safely
                    if isinstance(response.output, list):
                        for item in response.output:
                            if item is None:
                                continue
                            if hasattr(item, 'content') and item.content:
                                if isinstance(item.content, list):
                                    for content in item.content:
                                        if content is None:
                                            continue
                                        if hasattr(content, 'text') and content.text:
                                            output_text += str(content.text)
                                        elif isinstance(content, str):
                                            output_text += content
                                elif isinstance(item.content, str):
                                    output_text += item.content
                            elif hasattr(item, 'text') and item.text:
                                output_text += str(item.text)
                            elif isinstance(item, str):
                                output_text += item
                    elif isinstance(response.output, str):
                        output_text = response.output
                
                if not output_text and hasattr(response, 'output_text'):
                    output_text = response.output_text
                
                if output_text:
                    # Clean up the response
                    text = output_text.strip()
                    # Remove quotes if present
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                    if text.startswith("'") and text.endswith("'"):
                        text = text[1:-1]
                    # Remove any prefixes
                    if ":" in text and len(text.split(":")[0]) < 20:
                        text = text.split(":", 1)[1].strip()
                    
                    # Validate length - MAX 4 WORDS
                    word_count = len(text.split())
                    if 1 <= word_count <= 4:  # Max 4 words
                        # Track this text
                        if not hasattr(self, '_recent_text_overlays'):
                            self._recent_text_overlays = []
                        self._recent_text_overlays.append(text)
                        if len(self._recent_text_overlays) > 10:
                            self._recent_text_overlays.pop(0)
                        
                        return text
                    else:
                        logger.warning(f"Generated text overlay length invalid ({word_count} words), using ML-weighted selection")
            
            except Exception as e:
                logger.warning(f"Failed to generate text overlay with ChatGPT: {e}")
        
        # Fallback: ML-weighted selection from best performers
        if ml_insights and ml_insights.get("best_text_overlays"):
            best_texts = ml_insights["best_text_overlays"]
            if best_texts:
                # Filter to only texts that hint at skincare
                skincare_texts = [t for t in best_texts if any(word in t.lower() for word in ["skin", "skincare", "routine", "care", "face", "clear", "refined"])]
                if skincare_texts:
                    # Weight selection: 70% chance of best performer, 30% chance of others
                    if random.random() < 0.7 and len(skincare_texts) > 0:
                        return skincare_texts[0]  # Top performer
                    else:
                        return random.choice(skincare_texts)  # Random from best
        
        # Use calm confidence options with diversity tracking
        if not hasattr(self, '_recent_text_overlays'):
            self._recent_text_overlays = []
        
        # Avoid recently used
        available_options = [t for t in CREATIVE_TEXT_OPTIONS if t not in self._recent_text_overlays[-5:]]
        if not available_options:
            available_options = CREATIVE_TEXT_OPTIONS
        
        selected = random.choice(available_options)
        self._recent_text_overlays.append(selected)
        if len(self._recent_text_overlays) > 10:
            self._recent_text_overlays.pop(0)
        
        return selected
    
    def _generate_ad_copy(
        self,
        product_info: Dict[str, Any],
        ml_insights: Optional[Dict[str, Any]] = None,
        scenario_description: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate advanced, ML-optimized ad copy using ChatGPT-5.
        Integrates scenario context, performance data, and brand guidelines.
        """
        
        if not self.openai_api_key:
            # Fallback to default options
            import random
            return {
                "primary_text": random.choice(PRIMARY_TEXT_OPTIONS),
                "headline": random.choice(HEADLINE_OPTIONS),
                "description": "",
            }
        
        try:
            from openai import OpenAI
            import random
            client = OpenAI(api_key=self.openai_api_key)
            
            # Build comprehensive ML guidance
            ml_guidance = ""
            best_performers = []
            worst_performers = []
            performance_patterns = ""
            
            if ml_insights:
                # Best performing ad copy
                best_ad_copy = ml_insights.get("best_ad_copy", [])
                if best_ad_copy:
                    ml_guidance += "\n\nML LEARNING - Top performing ad copy (analyze patterns and apply):\n"
                    for i, copy_item in enumerate(best_ad_copy[:5], 1):
                        if isinstance(copy_item, dict):
                            headline = copy_item.get("headline", "")
                            primary = copy_item.get("primary_text", "")
                            if headline or primary:
                                ml_guidance += f"{i}. Headline: {headline}\n   Primary: {primary[:100]}...\n"
                                best_performers.append(copy_item)
                        elif isinstance(copy_item, str):
                            ml_guidance += f"{i}. {copy_item}\n"
                            best_performers.append(copy_item)
                    
                    # Extract patterns
                    if best_performers:
                        patterns = []
                        if any("discipline" in str(p).lower() for p in best_performers):
                            patterns.append("Discipline-focused messaging performs well")
                        if any("presence" in str(p).lower() for p in best_performers):
                            patterns.append("Presence-focused messaging performs well")
                        if any("foundation" in str(p).lower() for p in best_performers):
                            patterns.append("Foundation/essentials messaging performs well")
                        
                        if patterns:
                            performance_patterns = "\n\nPERFORMANCE PATTERNS DETECTED:\n" + "\n".join(f"- {p}" for p in patterns)
                
                # Worst performers to avoid
                if ml_insights.get("worst_ad_copy"):
                    worst_ad_copy = ml_insights.get("worst_ad_copy", [])
                    if worst_ad_copy:
                        ml_guidance += "\n\nML LEARNING - Avoid these low-performing copy styles:\n"
                        for i, copy_item in enumerate(worst_ad_copy[:3], 1):
                            if isinstance(copy_item, dict):
                                headline = copy_item.get("headline", "")
                                if headline:
                                    ml_guidance += f"{i}. {headline}\n"
                            elif isinstance(copy_item, str):
                                ml_guidance += f"{i}. {copy_item}\n"
                            worst_performers.append(copy_item)
                
                # Scenario performance correlation
                if ml_insights.get("best_scenarios") and scenario_description:
                    best_scenarios = ml_insights.get("best_scenarios", [])
                    # Check if current scenario matches high-performing patterns
                    scenario_lower = scenario_description.lower()
                    matching_scenarios = [s for s in best_scenarios if any(word in s.lower() for word in scenario_lower.split()[:10])]
                    if matching_scenarios:
                        ml_guidance += f"\n\nSCENARIO CONTEXT: Current scenario matches high-performing patterns.\n"
                        ml_guidance += f"Generate copy that complements this scenario type effectively.\n"
            
            # Check recently used to avoid repetition
            recently_used = ""
            if hasattr(self, '_recent_ad_copy'):
                recent = getattr(self, '_recent_ad_copy', [])
                if recent:
                    recently_used = "\n\nRECENTLY USED (ensure diversity):\n"
                    for copy_item in recent[-5:]:
                        if isinstance(copy_item, dict):
                            recently_used += f"- Headline: {copy_item.get('headline', '')}\n"
                        elif isinstance(copy_item, str):
                            recently_used += f"- {copy_item}\n"
            
            # Build scenario context
            scenario_context = ""
            if scenario_description:
                # Extract key elements from scenario for copy alignment
                scenario_keywords = []
                scenario_lower = scenario_description.lower()
                if "luxury" in scenario_lower or "premium" in scenario_lower:
                    scenario_keywords.append("luxury positioning")
                if "confident" in scenario_lower or "confident" in scenario_lower:
                    scenario_keywords.append("confidence theme")
                if "sophisticated" in scenario_lower or "refined" in scenario_lower:
                    scenario_keywords.append("sophistication")
                if "contemplative" in scenario_lower or "calm" in scenario_lower:
                    scenario_keywords.append("calm confidence")
                
                if scenario_keywords:
                    scenario_context = f"\n\nSCENARIO ALIGNMENT: The image scenario conveys: {', '.join(scenario_keywords)}.\n"
                    scenario_context += "Ensure ad copy aligns with and enhances this visual narrative.\n"
            
            user_prompt = f"""You are a world-class copywriter for luxury brands (like Dior, Louis Vuitton, Aesop). Your task is to generate premium ad copy for a Meta Ads campaign.

BRAND POSITIONING:
- Premium men's skincare brand
- Target: Men aged 18-54 in United States
- Tone: CALM CONFIDENCE - not hype, not convenience, not vanity
- Core message: Self-respect, discipline, presence, refinement

CRITICAL TONE REQUIREMENTS:
- Calm confidence: assured, not aggressive
- Self-respect: valuing oneself, not seeking validation
- Discipline: commitment to routine, not convenience-seeking
- Presence: how you show up matters, not just appearance
- Refined sophistication: quality over quantity, intentional over impulsive
- NO hype, NO urgency, NO FOMO, NO sales language
- NO convenience messaging, NO "quick fixes", NO "easy solutions"

PRODUCT CONTEXT:
- Product: {product_info.get("name", "Premium skincare")}
- Description: {product_info.get("description", "Luxury product for men")}

{scenario_context}

{ml_guidance}

{performance_patterns}

{recently_used}

COPY REQUIREMENTS:

1. HEADLINE (max 60 characters, must be exactly 60 or less):
   - Short, impactful, confident
   - Speaks to discipline, presence, or self-respect
   - Must grab attention without being pushy
   - Examples: "For men who value discipline.", "Clean skin is the foundation of presence."

2. PRIMARY TEXT (max 300 characters, must be exactly 300 or less):
   - Calm, confident narrative
   - Speaks to self-respect and intentional living
   - Connects to the scenario's mood/theme
   - Examples: "Precision skincare designed to elevate daily standards.", "Your routine communicates who you are."

3. DESCRIPTION (max 150 characters, optional but recommended):
   - Supporting text that reinforces the message
   - Additional context or benefit
   - Maintains calm confidence tone

GENERATION PROCESS:
1. Analyze ML performance data and extract winning patterns
2. Consider scenario context and align copy accordingly
3. Ensure diversity (not repeating recent copy)
4. Apply calm confidence tone consistently
5. Verify all character limits are met
6. Ensure copy complements the scenario's mood

EXCELLENT EXAMPLES (use as inspiration, not to copy):
Headline: "For men who value discipline."
Primary: "Precision skincare designed to elevate daily standards. Your routine communicates who you are."

Headline: "Clean skin is the foundation of presence."
Primary: "Look like a man who takes care of himself. This is maintenance, not vanity."

Headline: "Your routine communicates who you are."
Primary: "Precision skincare for men who understand that the face you show the world matters. Refined, not complicated."

VALIDATION CHECKLIST:
✓ Headline: 60 characters or less, confident, discipline/presence theme
✓ Primary text: 300 characters or less, calm confidence, self-respect
✓ Description: 150 characters or less (optional)
✓ Tone: Calm confidence, no hype, no urgency
✓ Alignment: Complements scenario context
✓ Diversity: Different from recent copy
✓ ML-informed: Uses patterns from best performers

Return JSON format (exact structure required):
{{
  "primary_text": "...",
  "headline": "...",
  "description": "..."
}}

Ensure all text meets character limits and maintains calm confidence tone."""
            
            response = client.responses.create(
                model=CHATGPT5_MODEL,
                input=user_prompt,
            )
            
            # Extract text - handle None and empty responses safely
            output_text = ""
            if hasattr(response, 'output_text') and response.output_text:
                output_text = str(response.output_text)
            elif hasattr(response, 'output') and response.output:
                # Handle list of outputs
                if isinstance(response.output, list):
                    for item in response.output:
                        if item is None:
                            continue
                        if hasattr(item, 'content') and item.content:
                            if isinstance(item.content, list):
                                for content in item.content:
                                    if content is None:
                                        continue
                                    if hasattr(content, 'text') and content.text:
                                        output_text += str(content.text)
                                    elif isinstance(content, str):
                                        output_text += content
                            elif isinstance(item.content, str):
                                output_text += item.content
                        elif hasattr(item, 'text') and item.text:
                            output_text += str(item.text)
                        elif isinstance(item, str):
                            output_text += item
                elif isinstance(response.output, str):
                    output_text = response.output
            
            if output_text:
                import json
                try:
                    cleaned = output_text.strip()
                    # Remove markdown code blocks
                    if cleaned.startswith("```"):
                        parts = cleaned.split("```")
                        if len(parts) > 1:
                            cleaned = parts[1]
                            if cleaned.startswith("json"):
                                cleaned = cleaned[4:]
                    cleaned = cleaned.strip()
                    
                    ad_copy = json.loads(cleaned)
                    
                    # Validate character limits
                    headline = ad_copy.get("headline", "").strip()
                    primary_text = ad_copy.get("primary_text", "").strip()
                    description = ad_copy.get("description", "").strip()
                    
                    # Enforce character limits
                    if len(headline) > 60:
                        headline = headline[:57] + "..."
                        logger.warning(f"Headline truncated to 60 chars: {headline}")
                    
                    if len(primary_text) > 300:
                        primary_text = primary_text[:297] + "..."
                        logger.warning(f"Primary text truncated to 300 chars")
                    
                    if description and len(description) > 150:
                        description = description[:147] + "..."
                        logger.warning(f"Description truncated to 150 chars")
                    
                    result = {
                        "primary_text": primary_text,
                        "headline": headline,
                        "description": description,
                    }
                    
                    # Track this ad copy for diversity
                    if not hasattr(self, '_recent_ad_copy'):
                        self._recent_ad_copy = []
                    self._recent_ad_copy.append(result)
                    if len(self._recent_ad_copy) > 10:
                        self._recent_ad_copy.pop(0)
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse ad copy JSON: {e}")
                    # Try to extract fields manually
                    headline_match = None
                    primary_match = None
                    desc_match = None
                    
                    # Try to find fields in text
                    if '"headline"' in output_text or "'headline'" in output_text:
                        import re
                        headline_match = re.search(r'(?:headline["\']?\s*:\s*["\'])([^"\']+)', output_text, re.IGNORECASE)
                    if '"primary_text"' in output_text or "'primary_text'" in output_text:
                        primary_match = re.search(r'(?:primary_text["\']?\s*:\s*["\'])([^"\']+)', output_text, re.IGNORECASE)
                    if '"description"' in output_text or "'description'" in output_text:
                        desc_match = re.search(r'(?:description["\']?\s*:\s*["\'])([^"\']+)', output_text, re.IGNORECASE)
                    
                    if headline_match or primary_match:
                        result = {
                            "primary_text": primary_match.group(1) if primary_match else "",
                            "headline": headline_match.group(1) if headline_match else "",
                            "description": desc_match.group(1) if desc_match else "",
                        }
                        
                        # Validate limits
                        if len(result["headline"]) > 60:
                            result["headline"] = result["headline"][:57] + "..."
                        if len(result["primary_text"]) > 300:
                            result["primary_text"] = result["primary_text"][:297] + "..."
                        if result["description"] and len(result["description"]) > 150:
                            result["description"] = result["description"][:147] + "..."
                        
                        # Track
                        if not hasattr(self, '_recent_ad_copy'):
                            self._recent_ad_copy = []
                        self._recent_ad_copy.append(result)
                        if len(self._recent_ad_copy) > 10:
                            self._recent_ad_copy.pop(0)
                        
                        return result
            
            # Fallback: ML-weighted selection
            if ml_insights and ml_insights.get("best_ad_copy"):
                best_copy = ml_insights.get("best_ad_copy", [])
                if best_copy:
                    # Prefer dict format, fallback to string
                    if isinstance(best_copy[0], dict):
                        selected = best_copy[0].copy()
                        # Track
                        if not hasattr(self, '_recent_ad_copy'):
                            self._recent_ad_copy = []
                        self._recent_ad_copy.append(selected)
                        if len(self._recent_ad_copy) > 10:
                            self._recent_ad_copy.pop(0)
                        return selected
                    else:
                        # String format - convert to dict
                        import random
                        return {
                            "primary_text": random.choice(PRIMARY_TEXT_OPTIONS),
                            "headline": best_copy[0] if isinstance(best_copy[0], str) else random.choice(HEADLINE_OPTIONS),
                            "description": "",
                        }
            
            # Final fallback - ensure we always return a valid dict
            import random
            try:
                return {
                    "primary_text": random.choice(PRIMARY_TEXT_OPTIONS) if PRIMARY_TEXT_OPTIONS else "Precision skincare designed to elevate daily standards.",
                    "headline": random.choice(HEADLINE_OPTIONS) if HEADLINE_OPTIONS else "For men who value discipline.",
                    "description": "",
                }
            except Exception as e:
                logger.warning(f"Error in final ad copy fallback: {e}")
                # Ultimate fallback - always return valid dict
                return {
                    "headline": "For men who value discipline.",
                    "primary_text": "Precision skincare designed to elevate daily standards.",
                    "description": ""
                }
            
        except Exception as e:
            logger.error(f"Error generating ad copy: {e}")
            notify(f"❌ Error generating ad copy: {e}")
            # Fallback
            import random
            return {
                "primary_text": random.choice(PRIMARY_TEXT_OPTIONS),
                "headline": random.choice(HEADLINE_OPTIONS),
                "description": "",
            }
    
    def _add_text_overlay(
        self,
        image_path: str,
        text: str,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add premium text overlay to image using ffmpeg.
        Poppins font (or system sans-serif fallback), proper wrapping, bottom positioning with margins.
        """
        try:
            # Check if ffmpeg is available (REQUIRED - not optional)
            # Check common installation paths (macOS Homebrew, Linux, etc.)
            ffmpeg_paths = [
                "/opt/homebrew/bin/ffmpeg",  # macOS Homebrew (Apple Silicon)
                "/usr/local/bin/ffmpeg",     # macOS Homebrew (Intel) or Linux
                "ffmpeg",                    # System PATH
                "/usr/bin/ffmpeg",           # Linux system path
            ]
            ffmpeg_cmd = None
            for path in ffmpeg_paths:
                try:
                    result = subprocess.run(
                        [path, "-version"],
                        capture_output=True,
                        check=True,
                        timeout=5,
                    )
                    ffmpeg_cmd = path
                    break
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            if not ffmpeg_cmd:
                error_msg = "❌ CRITICAL: ffmpeg is REQUIRED but not found. Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)"
                logger.error(error_msg)
                notify(error_msg)
                raise RuntimeError("ffmpeg is required for text overlay generation but is not installed or not in PATH")
            
            if output_path is None:
                input_path = Path(image_path)
                output_path = str(input_path.parent / f"{input_path.stem}_overlay{input_path.suffix}")
            
            # Properly escape text for ffmpeg (escape single quotes, backslashes, colons, etc.)
            escaped_text = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")
            
            # Poppins font paths (try multiple options for different systems)
            # Ubuntu/Linux: Poppins from Google Fonts or system
            # macOS: Poppins from Google Fonts or system
            # Windows: Poppins from Google Fonts or system
            font_paths = [
                "/usr/share/fonts/truetype/poppins/Poppins-Bold.ttf",  # Ubuntu/Linux - Poppins
                "/usr/share/fonts/truetype/Poppins-Bold.ttf",  # Ubuntu/Linux alternative path
                "/usr/local/share/fonts/Poppins-Bold.ttf",  # Linux alternative
                "~/.fonts/Poppins-Bold.ttf",  # User fonts
                "/System/Library/Fonts/Supplemental/Poppins-Bold.ttf",  # macOS
                "~/Library/Fonts/Poppins-Bold.ttf",  # macOS user fonts
                "C:/Windows/Fonts/Poppins-Bold.ttf",  # Windows
                "C:/Users/*/AppData/Local/Microsoft/Windows/Fonts/Poppins-Bold.ttf",  # Windows user fonts
                # Fallback to system sans-serif if Poppins not found
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Ubuntu/Linux fallback
                "/System/Library/Fonts/Supplemental/Helvetica-Bold.ttf",  # macOS fallback
                "C:/Windows/Fonts/arialbd.ttf",  # Windows fallback
            ]
            
            # Try to find Poppins font (preferred) or fallback to system sans-serif
            font_path = None
            for fp in font_paths:
                expanded_path = Path(fp).expanduser()  # Expand ~ to home directory
                if expanded_path.exists():
                    font_path = str(expanded_path)
                    break
            
            # Calculate appropriate font size based on text length and image dimensions
            # Get image dimensions first
            try:
                probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", image_path]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
                if probe_result.returncode == 0:
                    dimensions = probe_result.stdout.strip().split("x")
                    if len(dimensions) == 2:
                        img_width = int(dimensions[0])
                        # Font size: ~4% of image width, but min 36, max 56
                        base_fontsize = max(36, min(56, int(img_width * 0.04)))
                    else:
                        base_fontsize = 42
                else:
                    base_fontsize = 42
            except Exception:
                base_fontsize = 42
            
            # Adjust font size based on text length (shorter text = slightly larger)
            text_length = len(text)
            if text_length < 30:
                fontsize = base_fontsize + 4
            elif text_length < 50:
                fontsize = base_fontsize
            else:
                fontsize = base_fontsize - 4
            
            # Ensure text fits within image width (wrap long text)
            # Calculate max characters per line (rough estimate: ~60% of image width)
            max_chars_per_line = max(30, int(base_fontsize * 1.2))
            
            # Wrap text if needed
            if len(text) > max_chars_per_line:
                words = text.split()
                wrapped_lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_chars_per_line:
                        current_line += (" " + word if current_line else word)
                    else:
                        if current_line:
                            wrapped_lines.append(current_line)
                        current_line = word
                if current_line:
                    wrapped_lines.append(current_line)
                wrapped_text = "\\n".join(wrapped_lines)
            else:
                wrapped_text = text
            
            # Escape the wrapped text
            escaped_wrapped = wrapped_text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
            
            # Position: bottom center with generous margins
            # Bottom margin: 80px (more generous), side margins: 60px each
            bottom_margin = 80
            side_margin = 60
            
            # Build drawtext filter with premium styling
            # Use elegant shadows for depth and readability (no background box)
            drawtext_filter = (
                f"drawtext=text='{escaped_wrapped}'"
                f":fontsize={fontsize}"
                f":fontcolor=white"
                f":x=(w-text_w)/2"  # Center horizontally
                f":y=h-th-{bottom_margin}"  # Bottom with margin
                # Premium shadow for depth and readability (soft, elegant)
                f":shadowcolor=black@0.8"
                f":shadowx=2"
                f":shadowy=2"
                # No background box - clean, minimal luxury aesthetic
            )
            
            # Add font path if available
            if font_path:
                drawtext_filter += f":fontfile={font_path}"
            
            # Ensure 1:1 aspect ratio is preserved
            vf_filter = (
                f"scale=iw:iw:force_original_aspect_ratio=decrease,"
                f"pad=iw:iw:0:0:color=black@0,"
                f"{drawtext_filter}"
            )
            
            cmd = [
                ffmpeg_cmd,
                "-i", image_path,
                "-vf", vf_filter,
                "-y",
                output_path,
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0 and Path(output_path).exists():
                notify(f"✅ Premium text overlay added: {text[:50]}...")
                return output_path
            else:
                # Fallback: Try without font path (system default) with simpler styling
                logger.warning(f"First attempt failed, trying fallback: {result.stderr[:200]}")
                drawtext_fallback = (
                    f"drawtext=text='{escaped_wrapped}'"
                    f":fontsize={fontsize}"
                    f":fontcolor=white"
                    f":x=(w-text_w)/2"
                    f":y=h-th-{bottom_margin}"
                    f":shadowcolor=black@0.8"
                    f":shadowx=2"
                    f":shadowy=2"
                    # No background box - clean, minimal luxury aesthetic
                )
                
                vf_fallback = (
                    f"scale=iw:iw:force_original_aspect_ratio=decrease,"
                    f"pad=iw:iw:0:0:color=black@0,"
                    f"{drawtext_fallback}"
                )
                
                cmd_fallback = [
                    ffmpeg_cmd,
                    "-i", image_path,
                    "-vf", vf_fallback,
                    "-y",
                    output_path,
                ]
                
                result_fallback = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=30)
                if result_fallback.returncode == 0 and Path(output_path).exists():
                    notify(f"✅ Premium text overlay added (fallback): {text[:50]}...")
                    return output_path
                else:
                    logger.error(f"ffmpeg error: {result_fallback.stderr[:500]}")
                    notify(f"⚠️ Failed to add text overlay: {result_fallback.stderr[:100]}")
                    return None
                
        except Exception as e:
            logger.error(f"Error adding text overlay: {e}", exc_info=True)
            notify(f"❌ Error adding text overlay: {e}")
            return None


def create_image_generator(
    flux_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    ml_system: Optional[Any] = None,
) -> ImageCreativeGenerator:
    """Create an image creative generator instance."""
    flux_client = create_flux_client(flux_api_key) if flux_api_key else None
    return ImageCreativeGenerator(
        flux_client=flux_client,
        openai_api_key=openai_api_key,
        ml_system=ml_system,
    )


__all__ = ["ImageCreativeGenerator", "create_image_generator", "PromptEngineer"]
