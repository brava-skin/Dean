"""
FLUX API Client for Image Generation
Integrates with FLUX.1 Kontext [max] for static image generation
"""

from __future__ import annotations

import os
import time
import logging
import requests
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from integrations.slack import notify
from infrastructure.error_handling import (
    retry_with_backoff,
    enhanced_retry_with_backoff,
    with_circuit_breaker,
    circuit_breaker_manager,
)

# Import rate limit manager
try:
    from infrastructure.rate_limit_manager import get_rate_limit_manager, RateLimitType
    RATE_LIMIT_MANAGER_AVAILABLE = True
except ImportError:
    RATE_LIMIT_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

# FLUX API Configuration
FLUX_API_KEY = os.getenv("FLUX_API_KEY")
FLUX_API_URL = "https://api.bfl.ai/v1/flux-kontext-max"
FLUX_CREDITS_URL = "https://api.bfl.ai/v1/credits"
FLUX_MODEL = "FLUX.1 Kontext [max]"

# Note: FLUX_API_KEY validation happens lazily in FluxClient.__init__ to allow imports without env vars

# Credit pricing (1 credit = $0.01 USD)
FLUX_CREDITS_PER_IMAGE = {
    "FLUX.1 Kontext [max]": 8,  # $0.08 per image
    "FLUX.1 Kontext [pro]": 4,  # $0.04 per image
    "FLUX1.1 [pro]": 4,  # $0.04 per image
    "FLUX1.1 [pro] Ultra": 6,  # $0.06 per image
    "FLUX1.1 [pro] Raw": 6,  # $0.06 per image
    "FLUX.1 Fill [pro]": 5,  # $0.05 per image
}

# Polling configuration
POLLING_INTERVAL = 0.5  # seconds
MAX_POLLING_ATTEMPTS = 300  # Max 150 seconds (2.5 minutes)
SIGNED_URL_VALIDITY_SECONDS = 600  # 10 minutes


class FluxClient:
    """Client for FLUX.1 Kontext API image generation with credit management."""
    
    def __init__(self, api_key: Optional[str] = None, check_credits: bool = True):
        # Validate API key (lazy check - only when client is actually created)
        self.api_key = api_key or FLUX_API_KEY
        if not self.api_key:
            raise ValueError("FLUX_API_KEY environment variable is required. Set it in your .env file or pass api_key parameter.")
        
        self.base_url = FLUX_API_URL
        self.credits_url = FLUX_CREDITS_URL
        self.check_credits = check_credits
        self._cached_credits: Optional[float] = None
        self._credits_cache_time: float = 0
        from config.constants import FLUX_CREDITS_CACHE_TTL_SECONDS
        self._credits_cache_ttl: float = FLUX_CREDITS_CACHE_TTL_SECONDS
    
    def create_image(
        self,
        prompt: str,
        aspect_ratio: Optional[str] = None,
        seed: Optional[int] = None,
        prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
        output_format: str = "png",
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        input_image: Optional[str] = None,
        input_image_2: Optional[str] = None,
        input_image_3: Optional[str] = None,
        input_image_4: Optional[str] = None,
        max_retries: int = 3,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Create an image using FLUX Kontext Max API with retry logic.
        
        Args:
            prompt: Text prompt for image generation (required)
            aspect_ratio: Aspect ratio between 21:9 and 9:21 (optional, e.g., "16:9", "1:1", "9:16")
            seed: Optional seed for reproducibility (None for random)
            prompt_upsampling: Whether to perform upsampling on the prompt for more creative generation (default: False)
            safety_tolerance: Moderation level 0-6, 0 being most strict, 6 being least strict (default: 2)
            output_format: "jpeg" or "png" (default: "png")
            webhook_url: URL to receive webhook notifications (optional)
            webhook_secret: Optional secret for webhook signature verification (optional)
            input_image: Base64 encoded image or URL to use with Kontext (optional, multiref)
            input_image_2: Base64 encoded image or URL for experimental multiref (optional)
            input_image_3: Base64 encoded image or URL for experimental multiref (optional)
            input_image_4: Base64 encoded image or URL for experimental multiref (optional)
            max_retries: Maximum number of retry attempts (default: 3)
        
        Returns:
            Tuple of (image_url, request_id) or (None, None) on failure
        """
        # Check credits before making request (optimized for single image)
        if self.check_credits:
            credits_needed = FLUX_CREDITS_PER_IMAGE.get(FLUX_MODEL, 8)
            current_credits = self.get_credits()
            
            if current_credits is not None and current_credits < credits_needed:
                error_msg = f"Insufficient FLUX credits: {current_credits:.2f} available, {credits_needed} needed"
                notify(f"❌ {error_msg}")
                logger.error(error_msg)
                return None, None
        
        # Use circuit breaker with retry wrapper
        def _create_with_retry():
            last_error = None
            for attempt in range(max_retries):
                try:
                    # Create request payload
                    payload = {
                        "prompt": prompt,
                        "prompt_upsampling": prompt_upsampling,
                        "safety_tolerance": safety_tolerance,
                        "output_format": output_format,
                    }
                    
                    # Add optional parameters
                    if aspect_ratio is not None:
                        payload["aspect_ratio"] = aspect_ratio
                    if seed is not None:
                        payload["seed"] = seed
                    if webhook_url:
                        payload["webhook_url"] = webhook_url
                    if webhook_secret:
                        payload["webhook_secret"] = webhook_secret
                    if input_image:
                        payload["input_image"] = input_image
                    if input_image_2:
                        payload["input_image_2"] = input_image_2
                    if input_image_3:
                        payload["input_image_3"] = input_image_3
                    if input_image_4:
                        payload["input_image_4"] = input_image_4
                    
                    response = requests.post(
                        self.base_url,
                        headers={
                            "accept": "application/json",
                            "x-key": self.api_key,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=30,
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    request_id = result.get("id")
                    polling_url = result.get("polling_url")
                    
                    if not request_id or not polling_url:
                        notify(f"❌ FLUX API error: Missing request_id or polling_url")
                        if attempt < max_retries - 1:
                            continue
                        return None, None
                    
                    # Poll for result
                    image_url = self._poll_for_result(polling_url, request_id)
                    
                    if image_url:
                        return image_url, request_id
                    else:
                        # Polling failed, retry
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # Exponential backoff
                            notify(f"⏳ FLUX polling failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            return None, request_id
                            
                except requests.exceptions.RequestException as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        notify(f"⏳ FLUX API request failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(wait_time)
                        continue
                    else:
                        notify(f"❌ FLUX API request failed after {max_retries} attempts: {e}")
                        return None, None
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        notify(f"⏳ FLUX API error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(wait_time)
                        continue
                    else:
                        notify(f"❌ FLUX API error after {max_retries} attempts: {e}")
                        return None, None
            
            # All retries exhausted
            notify(f"❌ FLUX API failed after {max_retries} attempts. Last error: {last_error}")
            return None, None
        
        # Use circuit breaker
        try:
            breaker = circuit_breaker_manager.get_breaker("flux_api")
            return breaker.call(_create_with_retry)
        except Exception as e:
            logger.error(f"FLUX API circuit breaker error: {e}")
            return None, None
    
    def get_credits(self, use_cache: bool = True) -> Optional[float]:
        """
        Get the user's current credit balance.
        
        Args:
            use_cache: Whether to use cached credits (default: True)
        
        Returns:
            Credit balance as float, or None on failure
        """
        # Return cached value if still valid
        if use_cache and self._cached_credits is not None:
            if time.time() - self._credits_cache_time < self._credits_cache_ttl:
                return self._cached_credits
        
        try:
            response = requests.get(
                self.credits_url,
                headers={
                    "accept": "application/json",
                    "x-key": self.api_key,
                },
                timeout=10,
            )
            
            response.raise_for_status()
            result = response.json()
            
            credits = result.get("credits")
            if credits is not None:
                self._cached_credits = float(credits)
                self._credits_cache_time = time.time()
                logger.debug(f"FLUX credits: {self._cached_credits:.2f}")
                return self._cached_credits
            else:
                logger.warning("FLUX API returned credits response without 'credits' field")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get FLUX credits: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting FLUX credits: {e}")
            return None
    
    def get_credits_for_model(self, model: str = None) -> int:
        """
        Get the number of credits required for a specific model.
        
        Args:
            model: Model name (default: FLUX_MODEL)
        
        Returns:
            Credits required per image
        """
        model = model or FLUX_MODEL
        return FLUX_CREDITS_PER_IMAGE.get(model, 8)
    
    def can_afford_image(self, model: str = None, count: int = 1) -> bool:
        """
        Check if user has enough credits for image generation.
        
        Args:
            model: Model name (default: FLUX_MODEL)
            count: Number of images to generate (default: 1)
        
        Returns:
            True if user has enough credits, False otherwise
        """
        credits_needed = self.get_credits_for_model(model) * count
        current_credits = self.get_credits()
        
        if current_credits is None:
            # If we can't get credits, assume we can proceed (graceful degradation)
            logger.warning("Could not check FLUX credits, proceeding anyway")
            return True
        
        return current_credits >= credits_needed
    
    def invalidate_credits_cache(self):
        """Invalidate the cached credits value."""
        self._cached_credits = None
        self._credits_cache_time = 0
    
    def _poll_for_result(self, polling_url: str, request_id: str) -> Optional[str]:
        """
        Poll for image generation result with exponential backoff and better error handling.
        
        Args:
            polling_url: URL to poll for results
            request_id: Request ID for tracking
        
        Returns:
            Image URL if successful, None otherwise
        """
        start_time = time.time()
        attempt = 0
        consecutive_errors = 0
        max_consecutive_errors = 5  # Max errors before giving up
        base_interval = POLLING_INTERVAL
        max_interval = 5.0  # Max 5 seconds between polls
        
        while attempt < MAX_POLLING_ATTEMPTS:
            try:
                # Exponential backoff for polling interval (starts at 0.5s, increases on errors)
                current_interval = min(base_interval * (2 ** min(consecutive_errors, 3)), max_interval)
                time.sleep(current_interval)
                attempt += 1
                
                response = requests.get(
                    polling_url,
                    headers={
                        "accept": "application/json",
                        "x-key": self.api_key,
                    },
                    timeout=15,  # Increased timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                status = result.get("status", "").upper()
                
                if status == "READY":
                    sample = result.get("result", {}).get("sample")
                    if sample:
                        # Invalidate credits cache after successful generation
                        self.invalidate_credits_cache()
                        elapsed = time.time() - start_time
                        logger.info(f"✅ FLUX image generated: {request_id} (took {elapsed:.1f}s)")
                        notify(f"✅ FLUX image generated: {request_id}")
                        return sample
                    else:
                        notify(f"⚠️ FLUX result ready but no sample URL: {request_id}")
                        return None
                
                elif status in ("ERROR", "FAILED"):
                    error_msg = result.get("error", "Unknown error")
                    notify(f"❌ FLUX generation failed: {error_msg} (request_id: {request_id})")
                    return None
                
                # Reset consecutive errors on successful poll (even if not ready)
                consecutive_errors = 0
                
                # Continue polling for PENDING, PROCESSING, etc.
                elapsed = time.time() - start_time
                if attempt % 20 == 0:  # Log every 10 seconds (20 * 0.5s)
                    logger.debug(f"⏳ FLUX polling... (attempt {attempt}, {elapsed:.1f}s, status: {status})")
                
            except requests.exceptions.Timeout:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    notify(f"❌ FLUX polling: Too many consecutive timeouts ({consecutive_errors}), giving up")
                    return None
                logger.warning(f"⚠️ FLUX polling timeout (attempt {attempt}, consecutive errors: {consecutive_errors})")
                continue
            except requests.exceptions.RequestException as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    notify(f"❌ FLUX polling: Too many consecutive errors ({consecutive_errors}), giving up: {e}")
                    return None
                logger.warning(f"⚠️ FLUX polling error (attempt {attempt}, consecutive errors: {consecutive_errors}): {e}")
                continue
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    notify(f"❌ FLUX polling: Too many consecutive exceptions ({consecutive_errors}), giving up: {e}")
                    return None
                logger.warning(f"⚠️ FLUX polling exception (attempt {attempt}, consecutive errors: {consecutive_errors}): {e}")
                continue
        
        elapsed = time.time() - start_time
        notify(f"⏰ FLUX polling timeout after {MAX_POLLING_ATTEMPTS} attempts ({elapsed:.1f}s)")
        return None
    
    def download_image(self, image_url: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Download image from signed URL.
        
        Args:
            image_url: Signed URL from FLUX API
            output_path: Optional path to save image (default: temp file)
        
        Returns:
            Path to downloaded image or None on failure
        """
        try:
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            if output_path is None:
                # Create temp file with .png extension (default format)
                temp_dir = Path(tempfile.gettempdir()) / "dean_flux"
                temp_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(temp_dir / f"flux_image_{int(time.time())}.png")
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return output_path
            
        except Exception as e:
            notify(f"❌ Failed to download FLUX image: {e}")
            return None


def create_flux_client(api_key: Optional[str] = None, check_credits: bool = True) -> FluxClient:
    """
    Create a FLUX client instance.
    
    Args:
        api_key: FLUX API key (default: from FLUX_API_KEY env var)
        check_credits: Whether to check credits before generating images (default: True)
    
    Returns:
        FluxClient instance
    """
    return FluxClient(api_key=api_key, check_credits=check_credits)


__all__ = [
    "FluxClient",
    "create_flux_client",
    "FLUX_CREDITS_PER_IMAGE",
    "FLUX_MODEL",
]

