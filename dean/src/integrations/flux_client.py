from __future__ import annotations

import os
import time
import logging
import requests
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from integrations.slack import notify
from infrastructure.error_handling import circuit_breaker_manager

logger = logging.getLogger(__name__)

FLUX_API_KEY = os.getenv("FLUX_API_KEY")
FLUX_API_URL = "https://api.bfl.ai/v1/flux-kontext-max"
FLUX_CREDITS_URL = "https://api.bfl.ai/v1/credits"
FLUX_MODEL = "FLUX.1 Kontext [max]"

FLUX_CREDITS_PER_IMAGE = {
    "FLUX.1 Kontext [max]": 8,
    "FLUX.1 Kontext [pro]": 4,
    "FLUX1.1 [pro]": 4,
    "FLUX1.1 [pro] Ultra": 6,
    "FLUX1.1 [pro] Raw": 6,
    "FLUX.1 Fill [pro]": 5,
}

POLLING_INTERVAL = 0.5
MAX_POLLING_ATTEMPTS = 300


class FluxClient:
    def __init__(self, api_key: Optional[str] = None, check_credits: bool = True) -> None:
        self.api_key = api_key or FLUX_API_KEY
        if not self.api_key:
            raise ValueError("FLUX_API_KEY environment variable is required. Set it in your .env file or pass api_key parameter.")
        
        self.base_url = FLUX_API_URL
        self.credits_url = FLUX_CREDITS_URL
        self.check_credits = check_credits
        self._cached_credits: Optional[float] = None
        self._credits_cache_time: float = 0.0
        from config import FLUX_CREDITS_CACHE_TTL_SECONDS
        self._credits_cache_ttl: float = float(FLUX_CREDITS_CACHE_TTL_SECONDS)
    
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
        if self.check_credits:
            credits_needed = FLUX_CREDITS_PER_IMAGE.get(FLUX_MODEL, 8)
            current_credits = self.get_credits()
            
            if current_credits is not None and current_credits < credits_needed:
                error_msg = f"Insufficient FLUX credits: {current_credits:.2f} available, {credits_needed} needed"
                notify(f"❌ {error_msg}")
                logger.error(error_msg)
                return None, None
        
        def _create_with_retry():
            last_error = None
            for attempt in range(max_retries):
                try:
                    payload = {
                        "prompt": prompt,
                        "prompt_upsampling": prompt_upsampling,
                        "safety_tolerance": safety_tolerance,
                        "output_format": output_format,
                    }
                    
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
                    
                    image_url = self._poll_for_result(polling_url, request_id)
                    
                    if image_url:
                        return image_url, request_id
                    else:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2
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
            
            notify(f"❌ FLUX API failed after {max_retries} attempts. Last error: {last_error}")
            return None, None
        
        try:
            breaker = circuit_breaker_manager.get_breaker("flux_api")
            return breaker.call(_create_with_retry)
        except Exception as e:
            logger.error(f"FLUX API circuit breaker error: {e}")
            return None, None
    
    def get_credits(self, use_cache: bool = True) -> Optional[float]:
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
        model = model or FLUX_MODEL
        return FLUX_CREDITS_PER_IMAGE.get(model, 8)
    
    def can_afford_image(self, model: str = None, count: int = 1) -> bool:
        credits_needed = self.get_credits_for_model(model) * count
        current_credits = self.get_credits()
        
        if current_credits is None:
            logger.warning("Could not check FLUX credits, proceeding anyway")
            return True
        
        return current_credits >= credits_needed
    
    def invalidate_credits_cache(self):
        self._cached_credits = None
        self._credits_cache_time = 0
    
    def _poll_for_result(self, polling_url: str, request_id: str) -> Optional[str]:
        start_time = time.time()
        attempt = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        base_interval = POLLING_INTERVAL
        max_interval = 5.0
        
        while attempt < MAX_POLLING_ATTEMPTS:
            try:
                current_interval = min(base_interval * (2 ** min(consecutive_errors, 3)), max_interval)
                time.sleep(current_interval)
                attempt += 1
                
                response = requests.get(
                    polling_url,
                    headers={
                        "accept": "application/json",
                        "x-key": self.api_key,
                    },
                    timeout=15,
                )
                
                response.raise_for_status()
                result = response.json()
                
                status = result.get("status", "").upper()
                
                if status == "READY":
                    sample = result.get("result", {}).get("sample")
                    if sample:
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
                
                consecutive_errors = 0
                
                elapsed = time.time() - start_time
                if attempt % 20 == 0:
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
        try:
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            if output_path is None:
                temp_dir = Path(tempfile.gettempdir()) / "dean_flux"
                temp_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(temp_dir / f"flux_image_{int(time.time())}.png")
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return output_path
            
        except Exception as e:
            notify(f"❌ Failed to download FLUX image: {e}")
            return None


def create_flux_client(api_key: Optional[str] = None, check_credits: bool = True) -> FluxClient:
    return FluxClient(api_key=api_key, check_credits=check_credits)


__all__ = [
    "FluxClient",
    "create_flux_client",
    "FLUX_CREDITS_PER_IMAGE",
    "FLUX_MODEL",
]
