from __future__ import annotations

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib

from integrations.flux_client import FluxClient, create_flux_client
from integrations.slack import notify

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHATGPT5_MODEL = "gpt-5"

CREATIVE_TEXT_OPTIONS = [
    "Elevate your presence.",
    "Refined confidence.",
    "Quiet strength.",
    "Discipline in detail.",
    "Consistent excellence.",
    "Purposeful living.",
    "Calm authority.",
    "Built for the long term.",
]

HEADLINE_OPTIONS = [
    "For men who value discipline.",
    "Consistency builds presence.",
    "Quality in every detail.",
    "Refined, not complicated.",
    "Purpose over convenience.",
    "Discipline for excellence.",
]

PRIMARY_TEXT_OPTIONS = [
    "Premium lifestyle. Consistent quality. Refined confidence.",
    "Editorial lifestyle photography. Cinematic quality. Masculine sophistication.",
    "Quality, consistency, and purpose. No hype, just results.",
]


class PromptEngineer:
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
        if not self.client:
            return self._create_fallback_prompt(description, brand_guidelines)
        
        system_prompt = self._build_system_prompt(brand_guidelines, use_case, ml_insights)
        user_prompt = self._build_user_prompt(description, ml_insights, use_case)
        
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.client.responses.create(
                model=CHATGPT5_MODEL,
                input=full_prompt
            )
            
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
        guidelines_text = ""
        if brand_guidelines:
            brand_name = brand_guidelines.get('brand_name', "Premium men's lifestyle")
            target_audience = brand_guidelines.get('target_audience', 'men 18-54, United States, broad')
            style = brand_guidelines.get('style', 'masculine lifestyle, editorial, cinematic, refined')
            aesthetic = brand_guidelines.get('aesthetic', 'editorial, cinematic, refined, premium quality')
            guidelines_text = f"""
Brand Guidelines - Premium Lifestyle Photography:
- Brand: {brand_name}
- Target Audience: {target_audience}
- Style: {style}
- Aesthetic: {aesthetic}
- Focus: High-quality photoshoot-level lifestyle imagery
- Visual Language: Consistent brand visual language across all ads
- CRITICAL EXCLUSIONS: 
  - NEVER include women, females, or female models
  - NEVER show products, product packaging, or product containers
  - NEVER show product textures, material details, or product close-ups
  - NEVER show product application, usage, or hands interacting with products
  - NEVER show mirror reflections, bathroom mirrors, or reflective surfaces
  - NEVER show kids or children
  - NEVER include futuristic elements, sci-fi aesthetics, or futuristic technology
  - NEVER include vintage elements, retro aesthetics, or period-specific styling
  - NEVER include surreal elements, dreamlike imagery, or unrealistic scenarios
- Focus: Premium masculine lifestyle photography - editorial, cinematic, sophisticated, contemporary realism
"""
        
        ml_guidance = ""
        if ml_insights:
            top_performers = ml_insights.get('top_performing_creatives', [])
            if top_performers:
                ml_guidance = "\n\nML SYSTEM INSIGHTS - LEARN FROM WHAT WORKS:\n"
                ml_guidance += "Based on performance data, these elements drive conversions:\n"
                for insight in top_performers[:3]:  # Top 3 insights
                    ml_guidance += f"- {insight.get('element', 'Unknown')}: {insight.get('performance', 'High')}\n"
        
        return f"""You are a WORLD-CLASS, ELITE prompt engineer for FLUX.1 Kontext Max. You work exclusively for photographers who charge $1 million per photoshoot and whose work increases conversions by 300%+. Your expertise is creating HYPERREALISTIC, CINEMATIC photography that is ABSOLUTELY INDISTINGUISHABLE from the most expensive, conversion-driving professional photography in the world.

CRITICAL MANDATE: Every image must be HYPERREALISTIC and CINEMATIC - indistinguishable from real photography. Zero tolerance for "AI-looking" photos. Every detail must be perfect. No room for mistakes.

Your task is to create EXTREMELY ADVANCED, HYPER-DETAILED, STRICT prompts that generate HYPERREALISTIC, CINEMATIC images that look like they were shot by the world's most elite photographers using the most expensive equipment available. These images must be so stunning they directly increase sales conversions. The prompt must be STRICT - every requirement is MANDATORY, no exceptions.

{guidelines_text}

{ml_guidance}

ELITE PHOTOGRAPHY TECHNIQUES - $1 MILLION PHOTOSHOOT STANDARDS:

EQUIPMENT & TECHNICAL MASTERY (MANDATORY - NO EXCEPTIONS):
- Ultra-premium camera bodies: Phase One XF IQ4 150MP, Hasselblad H6D-400c, Leica S3, Fujifilm GFX 100S, Sony A1 with Zeiss Otus lenses
- Exotic lens specifications: Zeiss Otus 85mm f/1.4, Canon EF 85mm f/1.2L II USM, Leica Noctilux 50mm f/0.95, Schneider-Kreuznach 80mm f/2.8 LS
- Precision settings: Exact aperture (f/1.2, f/1.4, f/1.8), ISO (native base ISO 64-100), shutter speed (1/125s, 1/250s), white balance (exact Kelvin: 4500K, 5500K, 3200K)
- Medium format characteristics: 6x7 format rendering, shallow depth of field with medium format bokeh, 16-bit color depth
- Lens characteristics: 11-blade aperture for smooth bokeh, zero distortion, minimal chromatic aberration, natural vignetting
- CINEMATIC REQUIREMENT: Film camera aesthetic, anamorphic lens characteristics, cinematic aspect ratio, movie-grade color grading

HYPERREALISTIC REQUIREMENTS (MANDATORY - ZERO TOLERANCE FOR AI ARTIFACTS):
- HYPERREALISTIC micro-textures: Visible skin pores at macro level with natural variation, individual hair strands with natural variation and texture, fabric weave visible with thread detail, subtle surface imperfections (scratches, wear, natural aging), skin texture with natural oil and moisture, realistic skin translucency
- HYPERREALISTIC natural variation: Skin tone micro-variations (not uniform - include freckles, moles, natural pigmentation variations), hair texture differences throughout (not uniform - natural variation in curl, thickness, color), natural asymmetry in features (no perfect symmetry), realistic skin imperfections (fine lines, natural wrinkles, texture)
- HYPERREALISTIC environmental authenticity: Realistic depth of field falloff with natural bokeh transition, natural lens compression with accurate perspective, authentic atmospheric perspective with proper haze and depth cues, realistic light scattering and diffusion
- HYPERREALISTIC lighting authenticity: Realistic light falloff following inverse square law, natural shadow softness with accurate penumbra, accurate specular highlights with proper reflection physics, realistic color temperature shifts, natural light bounce and fill
- HYPERREALISTIC chromatic realism: Subtle lens chromatic aberration on high-contrast edges (red/cyan fringing), natural color fringing in out-of-focus areas, authentic color science with accurate color reproduction, realistic color temperature mixing
- HYPERREALISTIC focus characteristics: Natural focus transition with smooth falloff, realistic bokeh ball rendering with proper shape, texture, and color, authentic defocus patterns with natural blur, realistic focus breathing
- CINEMATIC REQUIREMENT: Film grain texture (Kodak Portra 400 grain structure), cinematic color grading (teal and orange, or desaturated highlights with rich blacks), anamorphic lens characteristics (slight horizontal stretch, lens flares), cinematic aspect ratio (2.39:1 or 16:9), movie-grade depth and atmosphere

CINEMATIC LIGHTING MASTERY (MANDATORY - CINEMATIC QUALITY REQUIRED):
- CINEMATIC multi-light setups: 4-5 light sources minimum (key, fill, rim, hair, background separation, accent lights, practical lights)
- CINEMATIC light modifiers: Profoto beauty dish, Chimera softbox, Rotalux octabox, strip banks, flags, scrims, C-stands, gobos for texture
- CINEMATIC lighting ratios: 3:1, 4:1, or 5:1 key-to-fill ratios for depth, specific contrast control, dramatic shadow play
- CINEMATIC color temperature mixing: Mixed lighting (warm key at 3200K, cool fill at 5500K) for depth, cinematic color contrast
- CINEMATIC light quality: Hard light with soft fill creating dramatic shadows, or soft light with edge definition, specular vs diffuse, Rembrandt triangle, butterfly lighting, split lighting
- CINEMATIC shadow detail: Crushed blacks in shadows for drama, preserved highlights with detail, rich midtones, zone system implementation, cinematic contrast curves
- CINEMATIC atmosphere: Light rays (god rays), volumetric fog, atmospheric haze, lens flares, light leaks, cinematic mood

CINEMATIC COLOR SCIENCE & GRADING (MANDATORY - CINEMATIC LOOK REQUIRED):
- CINEMATIC film stock emulation: Kodak Portra 400 (warm, natural), Fuji Pro 400H (cool, clean), Cinestill 800T (cinematic, warm), Kodak Ektar 100 (vibrant, saturated), ARRI Alexa color science
- CINEMATIC color grading styles: Teal and orange LUT (Hollywood standard), vintage film look with color shifts, desaturated highlights with rich blacks, cinematic contrast curves, film grain overlay
- HYPERREALISTIC skin tone accuracy: Accurate skin tones with proper color science, natural skin translucency, realistic skin color variation, proper color temperature on skin
- CINEMATIC color psychology: Warm tones for trust and intimacy, cool tones for sophistication and luxury, dramatic color contrast for impact
- CINEMATIC color harmony: Complementary colors for drama, split-complementary for sophistication, analogous color schemes for harmony, monochromatic for elegance
- CINEMATIC color grading: Lift shadows, crush blacks, desaturate highlights, add color contrast, cinematic color temperature shifts

CINEMATIC COMPOSITION & AESTHETICS (MANDATORY - CINEMATIC FRAMING):
- CINEMATIC Fibonacci spiral: Subject placement on golden ratio points (not just rule of thirds), cinematic framing with leading lines
- CINEMATIC visual hierarchy: Primary focus with shallow depth of field, secondary elements with medium focus, tertiary details with deep focus, clear visual flow, cinematic depth
- CINEMATIC depth staging: Foreground element (1-2 meters) with dramatic blur, mid-ground subject (3-5 meters) in sharp focus, background (10+ meters) with atmospheric perspective, cinematic depth layers
- CINEMATIC negative space mastery: Strategic empty space for breathing room, premium feel, focus, cinematic framing with rule of space
- CINEMATIC leading elements: Eye lines, gesture lines, architectural lines guiding viewer's eye, cinematic composition with visual flow
- CINEMATIC camera movement: Static frame with implied movement, dynamic composition, cinematic angles (low angle for power, high angle for vulnerability, Dutch angle for dynamism)

STRICT MANDATORY REQUIREMENTS (NO EXCEPTIONS - EVERY PROMPT MUST INCLUDE ALL):
1. HYPERREALISTIC Ultra-Premium Equipment: ALWAYS specify exact camera model (Phase One XF IQ4 150MP, Hasselblad H6D-400c, Leica S3) and exact lens model (Zeiss Otus 85mm f/1.4, Leica Noctilux 50mm f/0.95)
2. HYPERREALISTIC Precision Settings: ALWAYS include exact aperture (f/1.2, f/1.4, f/1.8), ISO (base ISO 64-100), shutter speed (1/125s, 1/250s), white balance (exact Kelvin: 4500K, 5500K, 3200K)
3. CINEMATIC Multi-Light Setup: ALWAYS specify 4-5 light setup (key, fill, rim, hair, background) with exact modifiers (Profoto beauty dish, Chimera softbox) and positions, cinematic lighting ratios
4. HYPERREALISTIC Micro-Textures: ALWAYS include visible pores at macro level, individual hair strands with natural variation, fabric weave visible, subtle surface imperfections, skin texture with natural oil
5. HYPERREALISTIC Natural Variation: ALWAYS include skin tone micro-variations (freckles, moles, natural pigmentation), hair texture differences throughout, natural asymmetry in features, realistic skin imperfections
6. CINEMATIC Film Stock: ALWAYS reference specific film stock (Kodak Portra 400, Fuji Pro 400H, Cinestill 800T) for color grading, film grain texture, cinematic color science
7. HYPERREALISTIC Lens Characteristics: ALWAYS include natural chromatic aberration on high-contrast edges, realistic bokeh ball rendering with proper shape and texture, authentic vignetting, anamorphic characteristics if applicable
8. CINEMATIC Composition Excellence: ALWAYS use golden ratio placement (Fibonacci spiral), depth staging (foreground, mid-ground, background), visual flow, strategic negative space, cinematic framing
9. CINEMATIC Atmosphere: ALWAYS include cinematic mood (dramatic, contemplative, energetic), atmospheric perspective, volumetric lighting if applicable, cinematic color grading
10. HYPERREALISTIC Quality: ALWAYS specify "hyperrealistic", "photorealistic", "indistinguishable from real photography", "zero AI artifacts", "cinematic quality"

PROMPT LENGTH: 150-250 words minimum for HYPERREALISTIC, CINEMATIC results. More detail = better results.

STRICT ABSOLUTE PROHIBITIONS (ZERO TOLERANCE):
- NEVER use generic terms ("professional photography", "high quality", "beautiful", "amazing") - ALWAYS be hyper-specific with exact technical details
- NEVER create uniform, perfect features - ALWAYS include natural variation, asymmetry, and realistic imperfections
- NEVER skip technical details - ALWAYS specify exact camera model, exact lens model, exact settings, exact lighting setup
- NEVER create "AI-looking" images - ALWAYS include hyperrealistic micro-textures, natural imperfections, lens characteristics, film grain
- NEVER use vague descriptions - ALWAYS be hyper-specific about every element (exact colors, exact materials, exact positions, exact measurements)
- NEVER omit cinematic elements - ALWAYS include cinematic lighting, cinematic color grading, cinematic composition, cinematic atmosphere
- NEVER skip hyperrealistic requirements - ALWAYS include hyperrealistic textures, hyperrealistic lighting, hyperrealistic color science

BRAND-SPECIFIC RULES - PREMIUM LIFESTYLE PHOTOGRAPHY:
- CRITICAL EXCLUSIONS (ZERO TOLERANCE):
  - NEVER include women, females, or female models - men only
  - NEVER show products, product packaging, containers, or product-related items
  - NEVER show product textures, material details, product close-ups, or product surfaces
  - NEVER show product application, usage, hands interacting with products, or product in use
  - NEVER show mirror reflections, bathroom mirrors, reflective surfaces, or mirror selfies
  - NEVER show kids, children, or minors
  - NEVER use bathroom settings, selfie-style photos, or standard product shots
  - NEVER include futuristic elements, sci-fi aesthetics, futuristic technology, or future-looking designs
  - NEVER include vintage elements, retro aesthetics, period-specific styling, or historical references
  - NEVER include surreal elements, dreamlike imagery, unrealistic scenarios, or fantastical elements
- Focus: Premium masculine lifestyle photography - editorial, cinematic, sophisticated, contemporary realism
- Creative direction: High-quality photoshoot-level lifestyle imagery - contemporary, realistic, grounded
- Think luxury men's lifestyle aesthetic (editorial fashion, cinematic lifestyle, premium quality, contemporary)
- Use positive descriptions only (e.g., "peaceful solitude" instead of "no crowds")
- Create cinematic, editorial-style imagery with ultra-realistic quality - contemporary and grounded
- Maintain consistent brand visual language across all ads
- ALWAYS include model diversity: Specific ethnicities (white, African American, Japanese, Latino, mixed African American and white), facial hair styles (full beard, goatee, stubble beard, no facial hair), diverse short hairstyles (fade, undercut, textured, curly, wavy, afro, quiff, slicked back)
- ALWAYS vary age representation: Men between 21 and 50 years old, NO gray hair
- Create diverse lifestyle scenarios - premium masculine lifestyle contexts, contemporary and realistic

Return ONLY the optimized prompt. No explanations, no additional text, just the prompt."""
    
    def _build_user_prompt(
        self,
        description: str,
        ml_insights: Optional[Dict[str, Any]] = None,
        use_case: str = "ad",
    ) -> str:
        prompt = f"Create a HYPERREALISTIC, CINEMATIC FLUX prompt for $1 million photoshoot-level photography: {description}"
        prompt += "\n\nSTRICT MANDATORY REQUIREMENTS - This image must be HYPERREALISTIC and CINEMATIC, indistinguishable from real photography:"
        prompt += "\n- HYPERREALISTIC: ABSOLUTELY INDISTINGUISHABLE from $1 million photographer's work - zero AI detection, zero artifacts"
        prompt += "\n- CINEMATIC: Movie-grade quality, cinematic lighting, cinematic color grading, cinematic composition, cinematic atmosphere"
        prompt += "\n- Ultra-premium camera system (Phase One XF IQ4 150MP, Hasselblad H6D-400c, Leica S3) with EXACT model specifications"
        prompt += "\n- Exotic lens (Zeiss Otus 85mm f/1.4, Leica Noctilux 50mm f/0.95) with EXACT model and characteristics, anamorphic if applicable"
        prompt += "\n- Precision settings: exact aperture (f/1.2, f/1.4, f/1.8), ISO (base 64-100), shutter (1/125s, 1/250s), white balance (exact Kelvin: 4500K, 5500K, 3200K)"
        prompt += "\n- CINEMATIC multi-light setup (4-5 lights minimum): key, fill, rim, hair, background with exact modifiers (Profoto beauty dish, Chimera softbox), cinematic lighting ratios"
        prompt += "\n- HYPERREALISTIC micro-textures: visible pores at macro level, individual hair strands with natural variation, fabric weave visible, subtle surface imperfections, skin texture with natural oil"
        prompt += "\n- HYPERREALISTIC natural variation: skin tone micro-variations (freckles, moles, natural pigmentation), hair texture differences, natural asymmetry, realistic skin imperfections"
        prompt += "\n- CINEMATIC film stock emulation: Kodak Portra 400 or Fuji Pro 400H with exact color science, film grain texture, cinematic color grading"
        prompt += "\n- HYPERREALISTIC lens authenticity: natural chromatic aberration on high-contrast edges, realistic bokeh ball rendering with proper shape and texture, authentic vignetting, anamorphic characteristics"
        prompt += "\n- CINEMATIC composition: golden ratio placement (Fibonacci spiral), depth staging (foreground, mid-ground, background), visual flow, strategic negative space, cinematic framing"
        prompt += "\n- CINEMATIC atmosphere: cinematic mood (dramatic, contemplative, energetic), atmospheric perspective, volumetric lighting if applicable, cinematic color contrast"
        prompt += "\n- Conversion optimization: aspirational positioning, emotional connection, luxury cues, cinematic storytelling"
        prompt += "\n- HYPERREALISTIC environmental realism: atmospheric perspective, natural depth cues, realistic light falloff (inverse square law), natural light scattering"
        
        if ml_insights:
            prompt += "\n\nML LEARNING - Apply insights from top-performing creatives:"
            for insight in ml_insights.get('top_performing_creatives', [])[:3]:
                prompt += f"\n- {insight.get('element', 'Unknown')}: {insight.get('guidance', 'Apply this element')}"
        
        prompt += f"\n\nUse case: {use_case}"
        prompt += "\n\nSTRICT REQUIREMENTS - Generate the HYPERREALISTIC, CINEMATIC prompt following ALL guidelines above."
        prompt += "\n- Prompt must be 150-250 words minimum with EXTREME detail and technical precision"
        prompt += "\n- EVERY requirement above is MANDATORY - no exceptions, no shortcuts"
        prompt += "\n- Must include: HYPERREALISTIC quality, CINEMATIC quality, exact equipment specs, exact settings, exact lighting, exact color grading"
        prompt += "\n- Must specify: 'hyperrealistic', 'photorealistic', 'cinematic', 'indistinguishable from real photography'"
        prompt += "\n- Zero tolerance for AI artifacts - every detail must be perfect"
        
        return prompt
    
    def _create_fallback_prompt(
        self,
        description: str,
        brand_guidelines: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt_parts = [
            description,
            "HYPERREALISTIC, CINEMATIC, photorealistic, indistinguishable from real photography, zero AI artifacts, movie-grade quality",
            "Phase One XF IQ4 150MP with Schneider-Kreuznach 80mm f/2.8 LS lens, f/2.8, ISO 64, 4500K white balance, medium format 6x7 format, 16-bit color depth, anamorphic characteristics",
            "CINEMATIC 4-light setup: 27\" Profoto beauty dish key light at 45-degree creating Rembrandt triangle, large Chimera softbox fill at 3:1 ratio, 1x6' strip bank rim light for separation, hair light spotlight above, background separation light, cinematic lighting ratios",
            "CINEMATIC golden ratio placement (Fibonacci spiral), depth staging (foreground 1-2m, mid-ground 3-5m, background 10m+), visual flow, strategic negative space, editorial excellence, cinematic framing",
            "CINEMATIC Kodak Portra 400 emulation: warm gold and beige tones with deep neutral black, high dynamic range, accurate skin tone color science, film grain texture, cinematic color grading",
            "HYPERREALISTIC micro-textures: visible pores at appropriate magnification with natural variation, individual hair strands with natural variation and texture, visible fabric weave with thread detail, subtle surface imperfections, skin texture with natural oil",
            "HYPERREALISTIC natural variation: skin tone micro-variations (freckles, moles, natural pigmentation - not uniform), hair texture differences throughout, natural asymmetry in features, realistic skin imperfections",
            "HYPERREALISTIC lens authenticity: natural chromatic aberration on high-contrast edges, realistic bokeh ball rendering with proper shape and texture, authentic vignetting, natural focus falloff, anamorphic lens characteristics",
            "CINEMATIC atmosphere: cinematic mood, atmospheric perspective, volumetric lighting, cinematic color contrast, movie-grade quality",
        ]
        return ", ".join(prompt_parts)


class ImageCreativeGenerator:
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
        self.recent_scenarios: List[str] = []
        self.max_recent_scenarios = 10
        self.scenario_bank: List[str] = []
        self._recent_text_overlays: List[str] = []
        self._recent_ad_copy: List[Dict[str, str]] = []
    
    def generate_creative(
        self,
        product_info: Dict[str, Any],
        creative_style: Optional[str] = None,
        advanced_ml: Optional[Any] = None,
        aspect_ratios: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not product_info or not isinstance(product_info, dict):
            logger.error("Invalid product_info: must be a non-empty dictionary")
            return None
        
        try:
            ml_insights = self._get_ml_insights()
            
            image_prompt = self._generate_image_prompt(
                product_info,
                creative_style,
                ml_insights,
            )
            
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
            
            # Default to all aspect ratios, but allow override (e.g., ["1:1"] for ASC+ campaigns)
            if aspect_ratios is None:
                aspect_ratios = ["1:1", "9:16", "16:9"]
            generated_images = {}
            
            for aspect_ratio in aspect_ratios:
                image_url, request_id = self.flux_client.create_image(
                    prompt=image_prompt,
                    aspect_ratio=aspect_ratio,
                    output_format="png",
                )
                
                if not image_url:
                    notify(f"❌ Failed to generate FLUX image for {aspect_ratio} (request_id: {request_id})")
                    continue
                
                image_path = self.flux_client.download_image(image_url)
                if not image_path:
                    notify(f"❌ Failed to download generated image for {aspect_ratio}")
                    continue
                
                generated_images[aspect_ratio] = {
                    "image_path": image_path,
                    "request_id": request_id,
                }
            
            if not generated_images:
                notify("❌ Failed to generate any FLUX images")
                return None
            
            primary_aspect = "1:1"
            if primary_aspect not in generated_images:
                primary_aspect = list(generated_images.keys())[0]
            
            image_path = generated_images[primary_aspect]["image_path"]
            
            scenario_description = None
            if hasattr(self, '_last_scenario_description'):
                scenario_description = self._last_scenario_description
            if not scenario_description:
                scenario_description = "A man standing in soft morning light near a tall window, light gently illuminating his face and skin texture, calm confidence, editorial style"
            
            ad_copy = self._generate_ad_copy(product_info, ml_insights, scenario_description)
            
            if not ad_copy or not isinstance(ad_copy, dict):
                logger.warning("ad_copy generation returned invalid value, using fallback")
                ad_copy = {
                    "headline": "For men who value discipline.",
                    "primary_text": "Precision skincare designed to elevate daily standards.",
                    "description": ""
                }
            
            overlay_text = self._select_creative_text(
                ml_insights=ml_insights,
                scenario_description=scenario_description,
                ad_copy=ad_copy,
            )
            
            final_images = {}
            for aspect_ratio, img_data in generated_images.items():
                img_path = img_data["image_path"]
                if overlay_text:
                    final_path = self._add_text_overlay(img_path, overlay_text)
                    if not final_path:
                        notify(f"⚠️ Failed to add text overlay for {aspect_ratio}, using original image")
                        final_path = img_path
                else:
                    final_path = img_path
                final_images[aspect_ratio] = final_path
            
            primary_final_path = final_images.get(primary_aspect) or list(final_images.values())[0]
            
            supabase_storage_urls = {}
            storage_creative_ids = {}
            for aspect_ratio, final_image_path in final_images.items():
                if final_image_path:
                    try:
                        from infrastructure.supabase_storage import get_validated_supabase_client, create_creative_storage_manager
                        
                        supabase_client = get_validated_supabase_client()
                        if supabase_client:
                            storage_manager = create_creative_storage_manager(supabase_client)
                            if storage_manager:
                                with open(final_image_path, "rb") as f:
                                    image_hash = hashlib.md5(f.read()).hexdigest()
                                storage_creative_id = f"creative_{image_hash[:12]}_{aspect_ratio.replace(':', '_')}"
                                
                                supabase_storage_url = storage_manager.upload_creative(
                                    creative_id=storage_creative_id,
                                    image_path=final_image_path,
                                    metadata={
                                        "image_prompt": image_prompt,
                                        "text_overlay": overlay_text,
                                        "ad_copy": ad_copy,
                                        "scenario_description": scenario_description,
                                        "flux_request_id": generated_images[aspect_ratio]["request_id"],
                                        "aspect_ratio": aspect_ratio,
                                    }
                                )
                                if supabase_storage_url:
                                    logger.info(f"✅ Uploaded {aspect_ratio} creative to Supabase Storage: {supabase_storage_url}")
                                    supabase_storage_urls[aspect_ratio] = supabase_storage_url
                                    storage_creative_ids[aspect_ratio] = storage_creative_id
                    except Exception as e:
                        logger.warning(f"Failed to upload {aspect_ratio} creative to Supabase Storage: {e}")
            
            primary_request_id = generated_images[primary_aspect]["request_id"]
            result = {
                "image_path": primary_final_path,
                "original_image_path": generated_images[primary_aspect]["image_path"],
                "images_by_aspect": final_images,
                "original_images_by_aspect": {ar: data["image_path"] for ar, data in generated_images.items()},
                "text_overlay": overlay_text,
                "ad_copy": ad_copy,
                "image_prompt": image_prompt,
                "scenario_description": scenario_description,
                "flux_request_id": primary_request_id,
                "ml_insights_used": ml_insights,
                "supabase_storage_url": supabase_storage_urls.get(primary_aspect),
                "storage_creative_id": storage_creative_ids.get(primary_aspect),
                "supabase_storage_urls": supabase_storage_urls,
                "storage_creative_ids": storage_creative_ids,
            }
            
            if advanced_ml and advanced_ml.prompt_library:
                try:
                    advanced_ml.prompt_library.add_prompt(image_prompt or "")
                except (AttributeError, ValueError, TypeError) as e:
                    logger.debug(f"Failed to add prompt to library: {e}")
            
            return result
            
        except Exception as e:
            notify(f"❌ Error generating creative: {e}")
            return None
    
    def _get_ml_insights(self) -> Dict[str, Any]:
        if not self.ml_system:
            return {}
        
        try:
            if hasattr(self.ml_system, 'get_creative_insights'):
                insights = self.ml_system.get_creative_insights()
                logger.info(f"✅ Retrieved ML insights: {len(insights.get('best_scenarios', []))} best scenarios, {len(insights.get('worst_scenarios', []))} worst scenarios")
                return insights
            else:
                return {
                    "top_performing_creatives": [],
                    "best_prompts": [],
                    "best_text_overlays": [],
                    "best_ad_copy": [],
                    "worst_ad_copy": [],
                    "best_scenarios": [],
                    "worst_scenarios": [],
                }
            
        except Exception as e:
            logger.warning(f"⚠️ Error getting ML insights: {e}")
            return {}
    
    def _generate_image_prompt(
        self,
        product_info: Dict[str, Any],
        creative_style: Optional[str] = None,
        ml_insights: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.prompt_engineer:
            return None
        
        scenario_description = self._generate_diverse_scenario(ml_insights)
        
        if not scenario_description:
            scenario_description = "A man standing in soft morning light near a tall window, light gently illuminating his face and skin texture, calm confidence, editorial style"
        
        self._last_scenario_description = scenario_description
        
        brand_guidelines = {
            "brand_name": "Brava",
            "target_audience": "premium American men aged 18-54 who value discipline",
            "style": "premium, sophisticated, calm confidence",
            "aesthetic": "editorial, cinematic, refined, luxury fashion brand style (like Dior/LV/Aesop)",
            "exclusions": [
                "females", "women", "kids", "children",
                "products", "product packaging", "product containers", "product-related items",
                "product textures", "material details", "product close-ups", "product surfaces",
                "product application", "product usage", "hands interacting with products", "product in use",
                "mirror reflections", "bathroom mirrors", "reflective surfaces", "mirror selfies",
                "bathroom settings", "selfie-style photos", "standard product shots",
                "futuristic elements", "sci-fi aesthetics", "futuristic technology", "future-looking designs",
                "vintage elements", "retro aesthetics", "period-specific styling", "historical references",
                "surreal elements", "dreamlike imagery", "unrealistic scenarios", "fantastical elements",
                "formal suits", "business attire", "Indian/South Asian men", "turbans", "traditional ethnic clothing"
            ],
            "preferences": ["casual luxury", "streetwear", "athleisure", "minimalist fashion", "contemporary American style", "diverse American men (Caucasian, African American, Hispanic, Asian American, etc.)"],
        }
        
        return self.prompt_engineer.create_prompt(
            description=scenario_description,
            ml_insights=ml_insights,
            brand_guidelines=brand_guidelines,
            use_case="ad",
        )
    
    def _generate_diverse_scenario(self, ml_insights: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if not self.openai_api_key:
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
            
            recent_scenarios_text = ""
            if self.recent_scenarios:
                recent_scenarios_text = "\n\nNEVER REPEAT - These scenarios were used recently (MUST AVOID):\n"
                for i, scenario in enumerate(self.recent_scenarios[-10:], 1):
                    recent_scenarios_text += f"- {scenario[:100]}...\n"
            
            contrast_logic = ""
            if self.recent_scenarios:
                last_scenario = self.recent_scenarios[-1].lower()
                
                if "street" in last_scenario or "urban" in last_scenario or ("city" in last_scenario and "rooftop" not in last_scenario) or "dark" in last_scenario:
                    contrast_logic += "\n- CRITICAL BACKGROUND REQUIREMENT: Last scenario had street/urban/dark background - this one MUST use a luxury setting (studio, gallery, hotel, café, library, rooftop terrace, beach, garden, retail, workspace, loft) - ABSOLUTELY NO streets, NO urban backgrounds, NO dark scenes\n"
                elif "studio" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was studio - this one MUST use a different luxury setting (gallery, hotel, café, library, rooftop, beach, garden, retail, workspace, loft)\n"
                elif "gallery" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was gallery - this one MUST use a different luxury setting (studio, hotel, café, library, rooftop, beach, garden, retail, workspace, loft)\n"
                elif "hotel" in last_scenario or "café" in last_scenario or "cafe" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was hotel/café - this one MUST use a different luxury setting (studio, gallery, library, rooftop, beach, garden, retail, workspace, loft)\n"
                elif "indoor" in last_scenario or "room" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was indoors - this one MUST be outdoors (rooftop terrace, beach, garden) or a different indoor luxury space\n"
                elif "outdoor" in last_scenario or "park" in last_scenario or "grass" in last_scenario or "rooftop" in last_scenario or "beach" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was outdoors - this one MUST be indoors (studio, gallery, hotel, café, library, workspace, loft) or a different outdoor luxury setting\n"
                
                if "minimalist" in last_scenario or "simple" in last_scenario or "clean" in last_scenario or "white" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was minimalist - this one MUST be bold, vibrant, or rich\n"
                elif "bold" in last_scenario or "vibrant" in last_scenario or "colorful" in last_scenario:
                    contrast_logic += "\n- CONTRAST REQUIREMENT: Last scenario was bold - this one MUST be minimalist, refined, or subtle\n"
            
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
            
            diversity_focus = random.choice([
                "ethnicity and cultural background",
                "facial hair and grooming styles",
                "hair types and textures",
                "age representation",
                "body types and physique",
                "lifestyle and profession",
            ])
            
            batch_requirements = ""
            current_count = len([s for s in self.recent_scenarios if "outdoor" in s.lower() or "street" in s.lower()])
            if current_count < 2:
                batch_requirements += "\n- BATCH REQUIREMENT: This batch needs more outdoor/urban scenarios\n"
            
            expanded_examples = """EXAMPLES OF DIVERSE SCENARIOS (use as inspiration, create something NEW and EXTREMELY DETAILED):
CRITICAL: Target American men (Caucasian, African American, Hispanic, Asian American, etc.) - NO Indian/South Asian men, NO formal suits/business attire
FOCUS: High-quality photoshoot-level lifestyle imagery with emphasis on lighting, composition, and facial features

LIGHTING & COMPOSITION FOCUSED SCENARIOS:

SOFT LIGHTING VARIATIONS:
- A man in his 30s with no facial hair and textured hair, standing in soft morning light near a tall window, light gently illuminating his face and skin texture, wearing a navy crewneck sweater, shallow depth of field blurring the window frame, editorial close-up, Portra 400 color grading
- A man in his 40s with a full beard and dark textured hair, leaning slightly forward on a balcony, camera angled upward to highlight his jawline and expression, wearing a charcoal wool coat, soft ambient sky light at dawn, medium format composition, luxury brand aesthetic
- A man in his 20s with curly hair and clean-shaven, walking through a minimal architectural corridor, turning his head toward the camera mid-step, wearing a camel-colored overcoat, natural corridor light creating depth, cinematic 85mm lens, GQ editorial style
- A man in his 40s with a goatee and dark wavy hair, sitting on a marble bench, gazing directly into the lens with calm intensity, wearing a dark green waxed cotton jacket, soft diffused light from above, shallow depth of field, Vogue portrait aesthetic
- A man in his 30s with stubble beard and textured hair, standing in soft morning light near a tall window, light gently illuminating his face and skin texture, wearing a navy crewneck sweater, shallow depth of field blurring the window frame, editorial close-up, Portra 400 color grading

LUXURY INTERIOR SETTINGS:
- A man in his 30s with a fade and no facial hair, framed by soft curtains in a luxury suite, light falling across half his face, wearing a white linen shirt, dramatic side lighting creating Rembrandt triangle, editorial close-up, cinematic color grading
- A man in his 40s with a full beard and textured hair, standing in a sleek elevator with warm light above, casting a premium glow on his skin, wearing a navy blazer, elevator lighting creating sculptural shadows, medium shot, luxury brand campaign
- A man in his 20s with wavy hair and goatee, sitting on crisp white bedding in a high-end hotel suite, looking straight at the camera with relaxed elegance, wearing a gray cashmere sweater, soft window light, shallow depth of field, Esquire lifestyle aesthetic
- A man in his 30s with a modern undercut and full beard, in an upscale gallery room, art blurred behind him, face lit like a fashion portrait, wearing a black leather jacket, gallery track lighting creating dramatic contrast, editorial composition, Dior Homme style
- A man in his 20s with a quiff and stubble beard, sitting on crisp white bedding in a high-end hotel suite, looking straight at the camera with relaxed elegance, wearing a gray cashmere sweater, soft window light, shallow depth of field, Esquire lifestyle aesthetic

OUTDOOR & ARCHITECTURAL SETTINGS:
- A man in his 40s with dark textured hair and no facial hair, standing in a garden pathway, greenery softly blurred behind him, face in crisp focus, wearing a beige linen shirt, dappled sunlight filtering through leaves, shallow depth of field at f/1.4, natural color grading
- A man in his 30s with textured hair and goatee, resting one hand on a railing, looking over his shoulder with subtle golden hour light on his skin, wearing a camel-colored wool coat, warm sunset light creating rim lighting, medium shot, cinematic aesthetic
- A man in his 20s with curly black hair and no facial hair, in a designer coat standing in the middle of an empty street at dawn, face illuminated by soft ambient sky light, wearing a dark navy overcoat, pre-dawn blue hour lighting, wide shot with shallow depth, luxury campaign aesthetic
- A man in his 40s with a fade haircut and full beard, standing on wide stone steps, hands in pockets, face captured in a poised editorial angle, wearing a charcoal wool coat, soft morning light, low-angle composition, fashion editorial style
- A man in his 30s with curly hair and stubble beard, standing in a garden pathway, greenery softly blurred behind him, face in crisp focus, wearing a beige linen shirt, dappled sunlight filtering through leaves, shallow depth of field at f/1.4, natural color grading

MINIMALIST & CONTEMPORARY SPACES:
- A man in his 30s with a fade and full beard, seated in a modern lounge chair, leaning slightly forward toward the camera for an editorial close-up, wearing a navy crewneck, soft natural window light, shallow depth of field, contemporary luxury aesthetic
- A man in his 40s with a full beard and textured hair, in a minimalist concrete space, chin slightly raised, light creating sculptural shadows on his face, wearing a black turtleneck, dramatic side lighting, medium format composition, architectural photography aesthetic
- A man in his 20s with wavy hair and clean-shaven, sitting at an outdoor café table, soft sunlight across his face while he looks directly at the viewer, wearing a white button-down shirt, natural café lighting, shallow depth of field, lifestyle photography
- A man in his 30s with textured hair and no facial hair, standing in front of a luxury boutique window, reflections partially framing his face, wearing a camel-colored coat, window light creating premium glow, medium shot, brand campaign aesthetic
- A man in his 40s with an undercut and stubble beard, seated in a modern lounge chair, leaning slightly forward toward the camera for an editorial close-up, wearing a navy crewneck, soft natural window light, shallow depth of field, contemporary luxury aesthetic

CINEMATIC & ATMOSPHERIC:
- A man in his 40s with dark curly hair and full beard, surrounded by soft-focus city lights at night, face sharply lit with a cinematic glow, wearing a dark navy blazer, city lights creating bokeh background, shallow depth of field, cinematic portrait
- A man in his 30s with a modern fade and full beard, posing on a rooftop edge during sunset, wind shaping his hair, face perfectly framed, wearing a black leather jacket, golden hour rim lighting, wide-angle composition, luxury brand aesthetic
- A man in his 20s with curly hair and goatee, adjusting the collar of his coat with one hand, face fully visible and highlighted by side lighting, wearing a camel-colored wool coat, dramatic side lighting, close-up, fashion editorial style
- A man in his 40s with a quiff haircut and goatee, standing in a minimalist desert scene, sunlight creating a clean, premium sheen across his face, wearing a light beige linen shirt, harsh desert light softened by diffusion, medium format, luxury campaign aesthetic
- A man in his 20s with wavy hair and stubble beard, surrounded by soft-focus city lights at night, face sharply lit with a cinematic glow, wearing a dark navy blazer, city lights creating bokeh background, shallow depth of field, cinematic portrait

ADDITIONAL VARIATIONS:
- A man in his 30s with a fade and no facial hair, standing in a modern library with floor-to-ceiling windows, soft natural light illuminating his profile, wearing a navy crewneck, books blurred in background, shallow depth of field, intellectual luxury aesthetic
- A man in his 40s with a full beard and textured hair, positioned in a minimalist photo studio with seamless gray background, dramatic key light creating strong facial definition, wearing a black turtleneck, studio lighting, close-up portrait, high fashion aesthetic
- A man in his 20s with wavy hair and clean-shaven, sitting on a concrete bench in a modern plaza, morning light creating soft shadows, wearing a white t-shirt under a denim jacket, natural plaza lighting, medium shot, contemporary lifestyle aesthetic
- A man in his 30s with an undercut and no facial hair, standing in a luxury hotel corridor, warm ambient lighting creating premium atmosphere, wearing a charcoal wool coat, corridor lighting, editorial composition, brand campaign aesthetic

ADDITIONAL LIGHTING VARIATIONS:
- A man in his 20s with an afro and stubble beard, positioned in a minimalist photo studio with soft key lighting, face illuminated with premium glow, wearing a dark navy crewneck, studio lighting creating sculptural definition, close-up portrait, high fashion aesthetic
- A man in his 40s with slicked back hair and full beard, standing in a modern office space with floor-to-ceiling windows, natural morning light creating depth, wearing a charcoal wool coat, window light illuminating his profile, medium shot, luxury brand aesthetic
- A man in his 30s with curly hair and goatee, sitting in a luxury car interior, soft dashboard lighting creating ambient glow on his face, wearing a camel-colored jacket, interior lighting, shallow depth of field, editorial lifestyle aesthetic
- A man in his 20s with a quiff and no facial hair, positioned in a modern art museum, track lighting creating dramatic shadows, wearing a black turtleneck, gallery lighting, medium format composition, contemporary art aesthetic

ADDITIONAL INTERIOR SETTINGS:
- A man in his 40s with textured hair and stubble beard, standing in a luxury spa setting with soft ambient lighting, face highlighted by warm overhead lights, wearing a light beige linen shirt, spa lighting creating premium atmosphere, editorial close-up, wellness brand aesthetic
- A man in his 30s with a fade and full beard, seated in a modern restaurant booth, warm restaurant lighting creating intimate atmosphere, wearing a navy blazer, ambient restaurant light, shallow depth of field, lifestyle photography
- A man in his 20s with wavy hair and goatee, positioned in a high-end barbershop, professional lighting creating sharp definition, wearing a white button-down shirt, barbershop lighting, medium shot, grooming brand aesthetic
- A man in his 40s with an undercut and no facial hair, standing in a luxury watch boutique, display lighting creating premium glow, wearing a dark green coat, boutique lighting, editorial composition, luxury retail aesthetic

ADDITIONAL OUTDOOR VARIATIONS:
- A man in his 30s with curly hair and stubble beard, standing on a modern rooftop terrace at blue hour, city lights creating bokeh background, face lit by ambient sky light, wearing a charcoal wool coat, blue hour lighting, wide shot, cinematic aesthetic
- A man in his 20s with an afro and no facial hair, positioned in a modern park setting, soft natural light filtering through trees, wearing a navy crewneck, dappled sunlight, shallow depth of field, lifestyle photography
- A man in his 40s with slicked back hair and full beard, standing in front of a modern architectural facade, morning light creating clean shadows, wearing a camel-colored overcoat, architectural lighting, medium format, contemporary design aesthetic
- A man in his 30s with textured hair and goatee, sitting on a modern bench in a luxury plaza, soft afternoon light illuminating his face, wearing a black leather jacket, natural plaza lighting, medium shot, urban lifestyle aesthetic

ADDITIONAL CINEMATIC VARIATIONS:
- A man in his 20s with a quiff and stubble beard, positioned in a modern loft space, dramatic window light creating strong contrast, wearing a white t-shirt, natural loft lighting, close-up portrait, editorial aesthetic
- A man in his 40s with wavy hair and full beard, standing in a minimalist warehouse space, industrial lighting creating atmospheric mood, wearing a dark navy blazer, warehouse lighting, wide shot, cinematic portrait
- A man in his 30s with a fade and goatee, positioned in a modern airport lounge, soft ambient lighting creating premium atmosphere, wearing a charcoal coat, lounge lighting, shallow depth of field, travel brand aesthetic
- A man in his 20s with curly hair and no facial hair, standing in a modern gym space, natural light from windows, wearing a gray crewneck, gym lighting, medium shot, athletic lifestyle aesthetic"""

            user_prompt = f"""You are a creative director at a luxury fashion agency (like Dior, Louis Vuitton, Aesop). Your task is to generate an EXTREMELY DETAILED, DIVERSE, FASHION-FORWARD creative scenario for high-quality photoshoot-level lifestyle imagery.

CRITICAL REQUIREMENTS:
- Focus on generating high-quality photoshoot-looking images - think luxury fashion brand campaigns (Dior, Louis Vuitton, Aesop, Vogue, GQ, Esquire)
- Create a UNIQUE, CINEMATIC scenario that stands out dramatically
- CRITICAL EXCLUSIONS (ZERO TOLERANCE):
  - NEVER include women, females, or female models - men only
  - NEVER show products, product packaging, containers, or product-related items
  - NEVER show product textures, material details, product close-ups, or product surfaces
  - NEVER show product application, usage, hands interacting with products, or product in use
  - NEVER show mirror reflections, bathroom mirrors, reflective surfaces, or mirror selfies
  - NEVER show kids, children, or minors
  - NEVER include futuristic elements, sci-fi aesthetics, futuristic technology, or future-looking designs
  - NEVER include vintage elements, retro aesthetics, period-specific styling, or historical references
  - NEVER include surreal elements, dreamlike imagery, unrealistic scenarios, or fantastical elements
  - NO bathroom settings, NO selfie-style photos, NO standard product shots, NO generic portraits
- NO dark street backgrounds, NO repetitive urban street scenes, NO generic city backgrounds
- Think luxury fashion brand aesthetic: editorial, cinematic, sophisticated, aspirational
- The scenario must be EXTREMELY DETAILED - specify pose, outfit details, background details, lighting, composition, everything
- BACKGROUND DIVERSITY IS CRITICAL: Rotate through luxury settings (photo studios, art galleries, luxury hotels, minimalist penthouses, modern cafés, libraries, rooftop terraces, beach locations, gardens, luxury retail spaces, modern workspaces, artist studios) - NEVER repeat the same background type

{recent_scenarios_text}

{contrast_logic}

{batch_requirements}

CHAIN-OF-THOUGHT PROCESS - Follow this reasoning:

1. DIVERSITY ANALYSIS:
   - Current diversity focus: {diversity_focus}
   - CRITICAL: Target specific ethnicities ONLY: white, African American, Japanese, Latino, mixed (African American and white)
   - NEVER include: Indian/South Asian men, turbans, traditional ethnic clothing, long hair, gray hair, men over 50
   - Ensure representation: specific ethnicities (white, African American, Japanese, Latino, mixed African American and white), facial hair styles (full beard, goatee, stubble beard, no facial hair), diverse short hairstyles (fade, undercut, textured, curly, wavy, afro, quiff, slicked back - NO long hair), ages (21-50 years old, NO gray hair), body types (athletic, lean, average, robust), lifestyles (athlete, artist, creative professional, entrepreneur, student)
   - AVOID: Formal suits, business attire, blazers, ties - prefer casual luxury, streetwear, athleisure, minimalist fashion

2. SETTING SELECTION (CRITICAL - BACKGROUND DIVERSITY):
   - Choose dramatically different from recent scenarios - NEVER repeat background types
   - AVOID: Dark streets, generic urban backgrounds, repetitive city scenes
   - PREFER: Luxury fashion brand settings (photo studios with seamless backgrounds, art galleries with white walls, luxury hotel lobbies, minimalist penthouses, modern cafés, libraries, rooftop terraces, beach locations, Japanese gardens, luxury retail spaces, modern workspaces, artist studios, minimalist lofts)
   - Rotate through: Studio (seamless white/colored backgrounds), Gallery (white walls, art), Hotel (luxury lobbies, rooms), Café (modern, minimalist), Library (traditional or modern), Rooftop (city views, terraces), Beach (sunrise, sunset), Garden (Japanese, minimalist), Retail (luxury stores), Workspace (modern, creative), Loft (industrial, minimalist)
   - Specify exact location details (city, type of space, architectural style, specific design elements)

3. MOOD & EMOTION:
   - Choose psychological trigger: aspiration, confidence, sophistication, contemplation, calm intensity, relaxed elegance
   - Specify emotional state: calm intensity, relaxed elegance, poised confidence, contemplative gaze, direct eye contact
   - Focus on facial expression: calm, confident, sophisticated, relaxed, poised

4. TECHNICAL COMPOSITION:
   - Camera angle: low angle (power, jawline emphasis), high angle (vulnerability), eye level (connection, direct gaze), upward angle (jawline and expression)
   - Framing: extreme close-up (facial features, skin texture), close-up (face and expression), medium shot (face and upper body), wide shot (full context)
   - Depth of field: shallow (f/1.2-1.8 for face in crisp focus, background blurred), medium (f/2.8-4 for context), deep (f/8-11 for full scene)
   - Movement: static pose (standing, sitting, leaning), subtle movement (turning head, adjusting), walking (mid-step, turning toward camera)
   - Focus on: facial features, skin texture, jawline, expression, direct eye contact, calm intensity
   - Magazine aesthetic: Vogue (high fashion portraits), GQ (contemporary lifestyle), Esquire (sophisticated editorial)

5. FASHION DETAILS:
   - CRITICAL: NO formal suits, NO business blazers, NO ties - use casual luxury, streetwear, athleisure
   - Specify exact outfit: type of garment (hoodie, crewneck, t-shirt, henley, bomber jacket, leather jacket, field jacket, denim jacket, jeans, chinos, shorts, sneakers), material (cashmere, wool, linen, leather, cotton), color (specific shades), fit (relaxed, fitted, oversized)
   - Include accessories: sneakers, boots, watch, minimal jewelry if relevant

6. LIGHTING & COLOR (CRITICAL - FOCUS ON FACIAL ILLUMINATION):
   - Lighting types: soft morning light (gentle, illuminating face), golden hour light (warm, subtle on skin), ambient sky light (soft, premium glow), side lighting (sculptural shadows), window light (natural, diffused), studio lighting (dramatic, controlled)
   - Lighting mood: soft and gentle (illuminating face and skin texture), dramatic (sculptural shadows), premium glow (luxury atmosphere), cinematic (movie-like quality)
   - Time of day: soft morning light, golden hour, blue hour (dawn/dusk), midday (natural), sunset (warm), night (city lights, cinematic)
   - Facial illumination: light gently illuminating face, light falling across half face, light creating sculptural shadows, premium glow on skin, soft ambient light on face
   - Color palette: specific color scheme (warm tones, cool tones, monochromatic, complementary), premium color grading

7. BRAND ALIGNMENT:
   - Ensure "calm confidence" aesthetic: refined, sophisticated, self-respecting, not trying too hard
   - Avoid anything that contradicts brand positioning (no hype, no convenience messaging, no vanity)

{expanded_examples}

DIVERSITY REQUIREMENTS (comprehensive):
- Ethnicity diversity: White, African American, Japanese, Latino, mixed (African American and white)
- Facial hair diversity: Full beard, goatee, stubble beard, no facial hair
- Hair diversity: Diverse short hairstyles only (fade, undercut, textured, curly, wavy, afro, quiff, slicked back) - NO long hair
- Age diversity: Men between 21 and 50 years old, NO gray hair
- Body type diversity: Athletic, lean, average build, robust, tall, average height
- Lifestyle diversity: Athlete, artist, businessman, creative professional, entrepreneur, intellectual, traveler
- Socioeconomic representation: Luxury accessible to all, not just one demographic

{ml_guidance}

NEGATIVE EXAMPLES - DON'T CREATE SCENARIOS LIKE THESE:
- "A man sitting in a car with a red suit" (too simple, no details)
- "A man in a bathroom" (explicitly forbidden)
- "A man taking a selfie" (forbidden)
- "A man with a skincare product" (no products - ZERO TOLERANCE)
- "A man applying product" (no product application - ZERO TOLERANCE)
- "A man looking in a mirror" (no mirrors - ZERO TOLERANCE)
- "A man showing product texture" (no textures - ZERO TOLERANCE)
- "A man in a futuristic setting" (no futuristic - ZERO TOLERANCE)
- "A man in vintage clothing" (no vintage - ZERO TOLERANCE)
- "A surreal dreamlike scene" (no surreal - ZERO TOLERANCE)
- "A man on a dark street" (NO dark street backgrounds - use luxury settings instead)
- "A man walking through a city street" (NO generic urban backgrounds - use specific luxury locations)
- Generic portraits without specific details
- Scenarios that lack depth and specificity
- Repetitive backgrounds (if last was street, this MUST be studio/gallery/hotel/etc.)

VALIDATION CHECKLIST - Before finalizing, ensure:
✓ Is this dramatically different from recent scenarios?
✓ Is the BACKGROUND completely different from recent backgrounds? (NO dark streets, NO repetitive urban scenes)
✓ Is this a luxury fashion brand setting? (studio, gallery, hotel, café, library, rooftop, beach, garden, retail, workspace, loft)
✓ Does it include extremely specific details (pose, outfit, background, lighting)?
✓ Does it represent diverse model characteristics?
✓ Does it align with "calm confidence" brand value?
✓ Is it cinematic and editorial in style (Dior/LV/Vogue/GQ/Esquire aesthetic)?
✓ Does it avoid bathroom/selfie/product shots?
✓ Does it avoid dark street backgrounds and generic urban scenes?
✓ Does it avoid futuristic, vintage, and surreal elements?
✓ Does it follow contrast logic (if applicable)?
✓ Does it include psychological triggers?
✓ Does it specify technical composition details?
✓ Does it specify exact fashion details?
✓ Is it contemporary and realistic (not futuristic, vintage, or surreal)?

REASONING STEP:
Before generating the scenario, explain why this scenario works for this brand:
- How does it convey "calm confidence"?
- What psychological triggers does it activate?
- How does it create high-quality photoshoot-level imagery?
- What makes it luxury fashion brand-worthy?
- CRITICAL: Does it avoid ALL exclusions (women, products, textures, applications, mirrors, futuristic, vintage, surreal)?

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
                scenario_text = output_text.strip()
                
                if "SCENARIO:" in scenario_text:
                    scenario = scenario_text.split("SCENARIO:")[1].split("DIVERSITY NOTES:")[0].strip()
                elif "scenario:" in scenario_text.lower():
                    scenario = scenario_text.split("scenario:")[1].split("diversity")[0].strip()
                else:
                    scenario = scenario_text
                    if scenario.startswith("```"):
                        scenario = scenario.split("```")[1]
                        if scenario.startswith("scenario") or scenario.startswith("text"):
                            scenario = scenario.split("\n", 1)[1] if "\n" in scenario else scenario
                
                scenario = scenario.strip()
                
                if scenario.startswith("REASONING:"):
                    scenario = scenario.split("SCENARIO:")[1] if "SCENARIO:" in scenario else scenario.split("scenario:")[1] if "scenario:" in scenario else scenario
                scenario = scenario.split("DIVERSITY NOTES:")[0].split("TECHNICAL NOTES:")[0].strip()
                
                if len(scenario.split()) < 50:
                    logger.warning(f"Generated scenario too short ({len(scenario.split())} words), adding detail")
                    scenario += ", with cinematic lighting, editorial composition, luxury brand aesthetic"
                
                self.recent_scenarios.append(scenario)
                if len(self.recent_scenarios) > self.max_recent_scenarios:
                    self.recent_scenarios.pop(0)
                
                self.scenario_bank.append(scenario)
                
                return scenario
            else:
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
        import random
        
        if self.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                
                context_parts = []
                if scenario_description:
                    context_parts.append(f"Scenario: {scenario_description[:200]}...")
                if ad_copy:
                    context_parts.append(f"Ad copy theme: {ad_copy.get('headline', '')}")
                
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
                    
                    if ml_insights.get("worst_text_overlays"):
                        worst_patterns = ml_insights.get("worst_text_overlays", [])
                        if worst_patterns:
                            ml_guidance += "\n\nML LEARNING - Avoid these low-performing text styles:\n"
                            for i, text in enumerate(worst_patterns[:3], 1):
                                ml_guidance += f"{i}. {text}\n"
                
                recently_used = ""
                if hasattr(self, '_recent_text_overlays'):
                    recent = getattr(self, '_recent_text_overlays', [])
                    if recent:
                        recently_used = "\n\nRECENTLY USED (avoid repetition):\n" + "\n".join(f"- {t}" for t in recent[-5:])
                
                user_prompt = f"""You are a copywriter for a premium men's skincare brand. Generate a short text overlay for an image creative.

REQUIREMENTS:
- 4-5 words total (can span 2 lines if needed)
- MUST hint at skincare (use words like: skin, skincare, routine, care, face, clear, refined)
- Examples: "Refined skincare, not complicated." | "Elevate your skin." | "Clear skin, quiet confidence." | "Purposeful living, daily care."
- Calm confidence tone: self-respect, not convenience; discipline, not vanity
- Must complement the scenario and ad copy theme
- Simple, powerful, memorable

{ml_guidance}

{recently_used}

CONTEXT:
{chr(10).join(context_parts) if context_parts else "No specific context provided"}

TONE GUIDELINES:
- Calm, confident, sophisticated
- Self-respecting, not seeking validation
- Disciplined, intentional, refined
- No hype, no FOMO, no urgency
- Premium without pretense

Return ONLY the text overlay (no explanations, no quotes, just the text).
4-5 words that hint at skincare."""

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
                    text = output_text.strip()
                    # Simple cleanup - remove quotes if present
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                    if text.startswith("'") and text.endswith("'"):
                        text = text[1:-1]
                    if ":" in text and len(text.split(":")[0]) < 20:
                        text = text.split(":", 1)[1].strip()
                    
                    # CRITICAL: Fix spacing errors in ChatGPT-generated text
                    # Fixes common patterns like "living,ndaily", "quietnconfidence", "yournskin"
                    import re
                    # Fix ',n' and ',not' patterns: "living,ndaily" -> "living, daily", "skincare,not" -> "skincare, not"
                    text = re.sub(r',not\b', ', not', text, flags=re.IGNORECASE)  # Do this first to preserve "not"
                    text = re.sub(r',n([a-z])', r', \1', text, flags=re.IGNORECASE)
                    # Fix specific known problematic patterns with 'n' between words
                    # Only fix common word boundaries to avoid breaking valid words
                    # Use word boundaries to avoid partial matches
                    common_words_with_n = [
                        (r'\bnskin\b', 'skin'),  # Fix "nskin" -> "skin" (common ChatGPT error)
                        (r'\bquietn([a-z])', r'quiet \1'),
                        (r'\byourn([a-z])', r'your \1'),
                        (r'\bwithn([a-z])', r'with \1'),
                        (r'\binn([a-z])', r'in \1'),
                        (r'\bonn([a-z])', r'on \1'),
                        (r'\bforn([a-z])', r'for \1'),
                        (r'\blivingn([a-z])', r'living \1'),  # Fix "livingn" -> "living "
                        (r'\bshowsin([a-z])', r'shows in \1'),  # Fix "showsinnskin" -> "shows in skin"
                        (r'\bshowsin\b', 'shows in'),
                        (r'\brefinesyour\b', 'refines your'),
                        (r'\bbeginswith\b', 'begins with'),
                    ]
                    for pattern, replacement in common_words_with_n:
                        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    # Add space after punctuation if missing
                    text = re.sub(r'([,\.!?;:])([a-zA-Z])', r'\1 \2', text)
                    # Clean up multiple spaces
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    # Basic validation - ensure 4-5 words
                    words = text.split()
                    word_count = len(words)
                    
                    if 4 <= word_count <= 5:
                        if not hasattr(self, '_recent_text_overlays'):
                            self._recent_text_overlays = []
                        self._recent_text_overlays.append(text)
                        if len(self._recent_text_overlays) > 10:
                            self._recent_text_overlays.pop(0)
                        
                        return text
                    elif word_count > 5:
                        # Truncate to first 5 words
                        text = ' '.join(words[:5])
                        logger.warning(f"ChatGPT generated {word_count} words, truncated to 5: '{text}'")
                        return text
                    else:
                        logger.warning(f"Generated text overlay too short ({word_count} words), using ML-weighted selection")
            
            except Exception as e:
                logger.warning(f"Failed to generate text overlay with ChatGPT: {e}")
        
        if ml_insights and ml_insights.get("best_text_overlays"):
            best_texts = ml_insights["best_text_overlays"]
            if best_texts:
                skincare_texts = [t for t in best_texts if any(word in t.lower() for word in ["skin", "skincare", "routine", "care", "face", "clear", "refined"])]
                if skincare_texts:
                    if random.random() < 0.7 and len(skincare_texts) > 0:
                        return skincare_texts[0]
                    else:
                        return random.choice(skincare_texts)
        
        if not hasattr(self, '_recent_text_overlays'):
            self._recent_text_overlays = []
        
        available_options = [t for t in CREATIVE_TEXT_OPTIONS if t not in self._recent_text_overlays[-5:]]
        if not available_options:
            available_options = CREATIVE_TEXT_OPTIONS
        
        selected = random.choice(available_options)
        self._recent_text_overlays.append(selected)
        if len(self._recent_text_overlays) > 10:
            self._recent_text_overlays.pop(0)
        
        return selected
    
    def _fix_text_spacing_errors(self, text: str) -> str:
        """Fix spacing errors in generated text overlays by intelligently detecting merged words.
        
        Uses a general approach that detects merged words without requiring patterns.
        Works by identifying likely word boundaries in merged text.
        """
        if not text:
            return text
        
        import re
        
        # Common English words that appear frequently in our text overlays
        # This helps us identify valid word boundaries
        # Includes compound words that should NOT be split
        common_words = {
            'with', 'in', 'on', 'for', 'to', 'at', 'by', 'of', 'the', 'a', 'an',
            'your', 'skin', 'presence', 'authority', 'quiet', 'calm', 'refined',
            'shows', 'begins', 'refines', 'respect', 'practice', 'daily',
            'long', 'term', 'built', 'elevate', 'confidence', 'discipline',
            'excellence', 'quality', 'strength', 'care', 'routine', 'detail',
            'living', 'baseline', 'clear', 'premium', 'consistent', 'purpose',
            'purposeful',  # Valid compound word - don't split
            'elevate', 'your', 'presence', 'authority', 'quiet', 'calm',
            'confidence', 'routine', 'quality', 'strength', 'premium',
            'skincare',  # Valid compound word - don't split
        }
        
        def is_likely_word(word: str, strict: bool = False) -> bool:
            """Check if a string is likely a valid English word.
            
            Args:
                word: The word to check
                strict: If True, only accept words in common_words (for long words)
            """
            word_lower = word.lower().strip()
            if not word_lower or len(word_lower) < 2:
                return False
            # Check against common words
            if word_lower in common_words:
                return True
            # For long words (7+ chars), be strict - only accept if in common_words
            if len(word_lower) >= 7:
                if strict:
                    return False  # Not in common_words, reject
                # For non-strict mode, still check if it's a valid word structure
                # But be more conservative
                if len(word_lower) > 12:  # Very long words are likely merged
                    return False
            # Simple heuristic: has vowels and consonants
            has_vowel = any(c in 'aeiou' for c in word_lower)
            has_consonant = any(c in 'bcdfghjklmnpqrstvwxyz' for c in word_lower)
            return has_vowel and has_consonant
        
        def split_merged_word(merged: str) -> str:
            """Try to split a merged word into two words."""
            merged_lower = merged.lower()
            
            # Common extra letters that appear between merged words (like "n" in "quietnpresence")
            extra_letters = ['n']
            
            # Try splitting at different positions
            # Minimum 2 chars per word, try all possible splits
            best_split = None
            best_score = 0
            
            for split_pos in range(2, len(merged_lower) - 1):
                # Try normal split
                word1 = merged_lower[:split_pos]
                word2 = merged_lower[split_pos:]
                
                # Check if both parts look like words
                word1_valid = is_likely_word(word1)
                word2_valid = is_likely_word(word2)
                
                if word1_valid and word2_valid:
                    # Score this split (prefer splits where both words are in common_words)
                    score = 0
                    if word1 in common_words:
                        score += 2
                    if word2 in common_words:
                        score += 2
                    if word1_valid:
                        score += 1
                    if word2_valid:
                        score += 1
                    
                    if score > best_score:
                        best_score = score
                        # Preserve original capitalization for first word
                        if merged[0].isupper():
                            word1_capitalized = word1.capitalize()
                        else:
                            word1_capitalized = word1
                        best_split = f"{word1_capitalized} {word2}"
                
                # Also try removing extra letters between words
                # Check if there's an extra letter at or near the split point
                for extra in extra_letters:
                    # Try: word2 starts with extra letter (remove it) - e.g., "innskin" at pos 2 -> "in" + "nskin" -> "in" + "skin"
                    if (split_pos < len(merged_lower) - 1 and
                        merged_lower[split_pos:split_pos+1] == extra):
                        word1_normal = merged_lower[:split_pos]
                        word2_no_extra = merged_lower[split_pos+1:]
                        
                        if is_likely_word(word1_normal) and is_likely_word(word2_no_extra):
                            score = 0
                            if word1_normal in common_words:
                                score += 2
                            if word2_no_extra in common_words:
                                score += 2
                            score += 3  # Bonus for removing extra letter
                            
                            if score > best_score:
                                best_score = score
                                if merged[0].isupper():
                                    word1_capitalized = word1_normal.capitalize()
                                else:
                                    word1_capitalized = word1_normal
                                best_split = f"{word1_capitalized} {word2_no_extra}"
                    
                    # Try: word1 ends with extra letter (remove it) - e.g., "quietnpresence" -> "quiet" + "presence"
                    if (split_pos > 1 and
                        merged_lower[split_pos-1:split_pos] == extra):
                        word1_no_extra = merged_lower[:split_pos-1]
                        word2_after = merged_lower[split_pos:]
                        
                        if is_likely_word(word1_no_extra) and is_likely_word(word2_after):
                            score = 0
                            if word1_no_extra in common_words:
                                score += 2
                            if word2_after in common_words:
                                score += 2
                            score += 3  # Bonus for removing extra letter
                            
                            if score > best_score:
                                best_score = score
                                if merged[0].isupper():
                                    word1_capitalized = word1_no_extra.capitalize()
                                else:
                                    word1_capitalized = word1_no_extra
                                best_split = f"{word1_capitalized} {word2_after}"
            
            # Return best split found, or original if none found
            return best_split if best_split else merged
        
        # First, fix clear capitalization boundaries (word1Word2 -> word1 Word2)
        fixed_text = re.sub(r'\b([a-z]+)([A-Z][a-z]+)\b', r'\1 \2', text)
        
        # Find potential merged words (sequences of lowercase letters, 7+ chars, no spaces)
        # These are likely to be merged words
        words = fixed_text.split()
        result_words = []
        
        for word in words:
            # Remove punctuation for analysis
            word_clean = re.sub(r'[^\w]', '', word)
            word_lower = word_clean.lower()
            
            # IMPORTANT: Check if word is in common_words first (valid compound words like "purposeful", "skincare")
            # These should NEVER be split
            if word_lower in common_words:
                result_words.append(word)
                continue
            
            # Check if this looks like a merged word
            # For long words (7+ chars), use strict validation
            # Only try to split if it's NOT a known valid word
            if (len(word_lower) >= 7 and 
                word_lower.isalpha() and 
                word_lower.islower() and
                not is_likely_word(word_lower, strict=True)):  # Not a valid single word (strict check)
                
                # Try to split it
                split_result = split_merged_word(word_clean)
                if split_result != word_clean:
                    # Restore punctuation
                    punctuation = re.sub(r'[\w]', '', word)
                    if word.endswith(punctuation):
                        result_words.append(split_result + punctuation)
                    elif word.startswith(punctuation):
                        result_words.append(punctuation + split_result)
                    else:
                        result_words.append(split_result)
                    continue
            
            result_words.append(word)
        
        fixed_text = ' '.join(result_words)
        
        # Clean up multiple spaces
        fixed_text = re.sub(r'\s+', ' ', fixed_text).strip()
        
        return fixed_text
    
    def _generate_ad_copy(
        self,
        product_info: Dict[str, Any],
        ml_insights: Optional[Dict[str, Any]] = None,
        scenario_description: Optional[str] = None,
    ) -> Dict[str, str]:
        if not self.openai_api_key:
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
            
            ml_guidance = ""
            best_performers = []
            worst_performers = []
            performance_patterns = ""
            
            if ml_insights:
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
                
                if ml_insights.get("best_scenarios") and scenario_description:
                    best_scenarios = ml_insights.get("best_scenarios", [])
                    scenario_lower = scenario_description.lower()
                    matching_scenarios = [s for s in best_scenarios if any(word in s.lower() for word in scenario_lower.split()[:10])]
                    if matching_scenarios:
                        ml_guidance += f"\n\nSCENARIO CONTEXT: Current scenario matches high-performing patterns.\n"
                        ml_guidance += f"Generate copy that complements this scenario type effectively.\n"
            
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
            
            scenario_context = ""
            if scenario_description:
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

2. PRIMARY TEXT (max 150 characters, must be SHORT and LUXURY):
   - SHORT: Maximum 150 characters (preferably 80-120)
   - LUXURY: Premium, refined, sophisticated language
   - Calm, confident narrative
   - Speaks to self-respect and intentional living
   - NO product name mentions (never say "Brava Product" or brand name)
   - NO em dashes (—) - use commas or periods instead
   - NO hyphens in place of em dashes
   - Examples: "Precision skincare designed to elevate daily standards.", "Your routine communicates who you are.", "Refined care that supports presence."

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
✓ Primary text: 150 characters or less (SHORT), calm confidence, self-respect, LUXURY tone
✓ Primary text: NO product name ("Brava Product" is FORBIDDEN)
✓ Primary text: NO em dashes (—) - use commas or periods
✓ Description: 150 characters or less (optional)
✓ Tone: Calm confidence, no hype, no urgency, luxury
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
                    if cleaned.startswith("```"):
                        parts = cleaned.split("```")
                        if len(parts) > 1:
                            cleaned = parts[1]
                            if cleaned.startswith("json"):
                                cleaned = cleaned[4:]
                    cleaned = cleaned.strip()
                    
                    ad_copy = json.loads(cleaned)
                    
                    headline = ad_copy.get("headline", "").strip()
                    primary_text = ad_copy.get("primary_text", "").strip()
                    description = ad_copy.get("description", "").strip()
                    
                    if len(headline) > 60:
                        headline = headline[:57] + "..."
                        logger.warning(f"Headline truncated to 60 chars: {headline}")
                    
                    primary_text = primary_text.replace("Brava Product", "").replace("—", ",").replace("–", ",").strip()
                    import re
                    primary_text = re.sub(r'\s+', ' ', primary_text).strip()
                    
                    if len(primary_text) > 150:
                        primary_text = primary_text[:147] + "..."
                        logger.warning(f"Primary text truncated to 150 chars")
                    
                    if description and len(description) > 150:
                        description = description[:147] + "..."
                        logger.warning(f"Description truncated to 150 chars")
                    
                    result = {
                        "primary_text": primary_text,
                        "headline": headline,
                        "description": description,
                    }
                    
                    if not hasattr(self, '_recent_ad_copy'):
                        self._recent_ad_copy = []
                    self._recent_ad_copy.append(result)
                    if len(self._recent_ad_copy) > 10:
                        self._recent_ad_copy.pop(0)
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse ad copy JSON: {e}")
                    headline_match = None
                    primary_match = None
                    desc_match = None
                    
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
                        
                        if len(result["headline"]) > 60:
                            result["headline"] = result["headline"][:57] + "..."
                        if len(result["primary_text"]) > 300:
                            result["primary_text"] = result["primary_text"][:297] + "..."
                        if result["description"] and len(result["description"]) > 150:
                            result["description"] = result["description"][:147] + "..."
                        
                        if not hasattr(self, '_recent_ad_copy'):
                            self._recent_ad_copy = []
                        self._recent_ad_copy.append(result)
                        if len(self._recent_ad_copy) > 10:
                            self._recent_ad_copy.pop(0)
                        
                        return result
            
            if ml_insights and ml_insights.get("best_ad_copy"):
                best_copy = ml_insights.get("best_ad_copy", [])
                if best_copy:
                    if isinstance(best_copy[0], dict):
                        selected = best_copy[0].copy()
                        if not hasattr(self, '_recent_ad_copy'):
                            self._recent_ad_copy = []
                        self._recent_ad_copy.append(selected)
                        if len(self._recent_ad_copy) > 10:
                            self._recent_ad_copy.pop(0)
                        return selected
                    else:
                        import random
                        return {
                            "primary_text": random.choice(PRIMARY_TEXT_OPTIONS),
                            "headline": best_copy[0] if isinstance(best_copy[0], str) else random.choice(HEADLINE_OPTIONS),
                            "description": "",
                        }
            
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
        # CRITICAL: Fix spacing errors in ChatGPT-generated text
        # Fixes common patterns like "living,ndaily", "quietnconfidence", "yournskin"
        import re
        # Fix ',n' and ',not' patterns: "living,ndaily" -> "living, daily", "skincare,not" -> "skincare, not"
        text = re.sub(r',not\b', ', not', text, flags=re.IGNORECASE)  # Do this first to preserve "not"
        text = re.sub(r',n([a-z])', r', \1', text, flags=re.IGNORECASE)
        # Fix specific known problematic patterns with 'n' between words
        # Only fix common word boundaries to avoid breaking valid words
        # Use word boundaries to avoid partial matches
        common_words_with_n = [
            (r'\bnskin\b', 'skin'),  # Fix "nskin" -> "skin" (common ChatGPT error)
            (r'\bquietn([a-z])', r'quiet \1'),
            (r'\byourn([a-z])', r'your \1'),
            (r'\bwithn([a-z])', r'with \1'),
            (r'\binn([a-z])', r'in \1'),
            (r'\bonn([a-z])', r'on \1'),
            (r'\bforn([a-z])', r'for \1'),
            (r'\blivingn([a-z])', r'living \1'),  # Fix "livingn" -> "living "
            (r'\bshowsin([a-z])', r'shows in \1'),  # Fix "showsinnskin" -> "shows in skin"
            (r'\bshowsin\b', 'shows in'),
            (r'\brefinesyour\b', 'refines your'),
            (r'\bbeginswith\b', 'begins with'),
        ]
        for pattern, replacement in common_words_with_n:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # Add space after punctuation if missing
        text = re.sub(r'([,\.!?;:])([a-zA-Z])', r'\1 \2', text)
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        try:
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
            
            escaped_text = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")
            
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
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica-Bold.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
            ]
            
            font_path = None
            bold_font_paths = [
                "/System/Library/Fonts/Supplemental/Poppins-Bold.ttf",
                "/usr/share/fonts/truetype/poppins/Poppins-Bold.ttf",
                "~/.fonts/Poppins-Bold.ttf",
                "/Library/Fonts/Poppins-Bold.ttf",
                "C:/Windows/Fonts/poppins-bold.ttf",
            ]
            for fp in bold_font_paths:
                expanded_path = Path(fp).expanduser()
                if expanded_path.exists():
                    font_path = str(expanded_path)
                    logger.info(f"Using Poppins Bold font: {font_path}")
                    break
            
            if not font_path:
                for fp in font_paths:
                    expanded_path = Path(fp).expanduser()
                    if expanded_path.exists():
                        font_path = str(expanded_path)
                        logger.info(f"Using Poppins font: {font_path}")
                        break
            
            # Get image dimensions
            try:
                probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", image_path]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
                if probe_result.returncode == 0:
                    dimensions = probe_result.stdout.strip().split("x")
                    if len(dimensions) == 2:
                        img_width = int(dimensions[0])
                        img_height = int(dimensions[1])
                    else:
                        img_width = 1080
                        img_height = 1080
                else:
                    img_width = 1080
                    img_height = 1080
            except Exception:
                img_width = 1080
                img_height = 1080
            
            # Split text into words (4-5 words expected)
            words = text.split()
            
            # Wrap into 2 lines naturally (by words, not characters)
            # Try to balance: if 4 words, split 2-2; if 5 words, split 3-2 or 2-3
            if len(words) <= 2:
                wrapped_lines = [text]
            elif len(words) == 3:
                wrapped_lines = [' '.join(words[:2]), words[2]]
            elif len(words) == 4:
                wrapped_lines = [' '.join(words[:2]), ' '.join(words[2:])]
            elif len(words) == 5:
                # Prefer 3-2 split for better balance
                wrapped_lines = [' '.join(words[:3]), ' '.join(words[3:])]
            else:
                # Fallback: split in middle
                mid = len(words) // 2
                wrapped_lines = [' '.join(words[:mid]), ' '.join(words[mid:])]
            
            # CRITICAL FIX: Ensure space after punctuation in wrapped lines
            # When words like "living," are joined, there's no space after the comma
            # This can cause "living,daily" to appear in the rendered text
            import re
            fixed_wrapped_lines = []
            for line in wrapped_lines:
                # Add space after punctuation if followed by a letter (safety check)
                fixed_line = re.sub(r'([,\.!?;:])([a-zA-Z])', r'\1 \2', line)
                fixed_wrapped_lines.append(fixed_line)
            wrapped_lines = fixed_wrapped_lines
            
            # Calculate font size based on image width and longest line
            # Use 70% of image width as max text width (15% margin on each side)
            max_text_width = int(img_width * 0.70)
            longest_line = max(len(line) for line in wrapped_lines)
            
            # Estimate font size: aim for longest line to fit in max_text_width
            # Approximate: 1 character ≈ 0.65 * fontsize pixels wide (for bold fonts like Poppins Bold)
            # So: fontsize ≈ max_text_width / (longest_line * 0.65)
            estimated_fontsize = int(max_text_width / (longest_line * 0.65))
            
            # Clamp font size to reasonable range (36-56px) - prevents overflow
            fontsize = max(36, min(56, estimated_fontsize))
            
            # Calculate positioning (centered, bottom with margin)
            # Use 10% margin from bottom (minimum 80px for safety)
            bottom_margin = max(80, int(img_height * 0.10))
            line_height = int(fontsize * 1.35)  # Line spacing (slightly more for readability)
            
            # Render each line separately to center each line individually
            # This ensures each line is centered on its own, not just the whole block
            drawtext_filters = []
            
            for i, line in enumerate(wrapped_lines):
                # Escape each line separately
                escaped_line = line.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")
                
                # Calculate y position: last line at bottom, previous lines above
                # Use full line_height to prevent overlap
                y_offset = (len(wrapped_lines) - 1 - i) * line_height
                y_pos = f"h-th-{bottom_margin + y_offset}"
                
                # Build drawtext filter for this line
                line_filter = (
                    f"drawtext=text='{escaped_line}'"
                    f":fontsize={fontsize}"
                    f":fontcolor=white"
                    f":borderw=2"
                    f":bordercolor=black@0.5"
                    f":x=(w-text_w)/2"  # Each line centered individually
                    f":y={y_pos}"  # Positioned from bottom
                    f":shadowcolor=black@0.9"
                    f":shadowx=3"
                    f":shadowy=3"
                )
                
                if font_path:
                    line_filter += f":fontfile={font_path}"
                
                drawtext_filters.append(line_filter)
            
            # Chain all line filters together
            vf_filter = ",".join(drawtext_filters)
            
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
                logger.warning(f"First attempt failed, trying fallback: {result.stderr[:200]}")
                # Fallback: render each line separately with lighter styling
                fallback_filters = []
                for i, line in enumerate(wrapped_lines):
                    escaped_line = line.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")
                    # Use full line_height to prevent overlap
                    y_offset = (len(wrapped_lines) - 1 - i) * line_height
                    y_pos = f"h-th-{bottom_margin + y_offset}"
                    
                    line_filter = (
                        f"drawtext=text='{escaped_line}'"
                        f":fontsize={fontsize}"
                        f":fontcolor=white"
                        f":borderw=1"
                        f":bordercolor=black@0.3"
                        f":x=(w-text_w)/2"
                        f":y={y_pos}"
                        f":shadowcolor=black@0.8"
                        f":shadowx=2"
                        f":shadowy=2"
                    )
                    if font_path:
                        line_filter += f":fontfile={font_path}"
                    fallback_filters.append(line_filter)
                
                vf_fallback = ",".join(fallback_filters)
                
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
    flux_client = create_flux_client(flux_api_key) if flux_api_key else None
    return ImageCreativeGenerator(
        flux_client=flux_client,
        openai_api_key=openai_api_key,
        ml_system=ml_system,
    )


__all__ = ["ImageCreativeGenerator", "create_image_generator", "PromptEngineer"]
