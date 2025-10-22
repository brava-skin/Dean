"""
Dean Continuous Rate Limiter
Optimized for DigitalOcean deployment with advanced Meta API rate limiting
"""

import time
import random
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class RateLimitState:
    """Track rate limiting state for different API endpoints"""
    requests_made: int = 0
    requests_allowed: int = 0
    reset_time: Optional[datetime] = None
    last_request: Optional[datetime] = None
    backoff_until: Optional[datetime] = None
    consecutive_errors: int = 0
    usage_percentage: float = 0.0

class ContinuousRateLimiter:
    """
    Advanced rate limiter optimized for continuous operation
    Prevents Ads Manager UI interference while maximizing ML data collection
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.endpoints: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        
        # Continuous operation settings
        self.base_delay = float(os.getenv("META_REQUEST_DELAY", "1.2"))
        self.jitter = float(os.getenv("META_JITTER", "0.3"))
        self.max_concurrent = int(os.getenv("META_MAX_CONCURRENT_INSIGHTS", "2"))
        self.usage_threshold = float(os.getenv("META_USAGE_THRESHOLD", "0.8"))
        
        # Enhanced backoff for UI-critical errors
        self.insights_platform_wait = 120  # 2 minutes for 1504022
        self.app_level_wait = 90  # 1.5 minutes for 1504039
        
        # Usage tracking
        self.usage_history = deque(maxlen=100)  # Track last 100 requests
        self.current_concurrent = 0
        
        # Business hours optimization
        self.business_hours_start = 9  # 9 AM Amsterdam
        self.business_hours_end = 18   # 6 PM Amsterdam
        
    def is_business_hours(self) -> bool:
        """Check if we're in business hours (Amsterdam time)"""
        amsterdam_hour = datetime.now().hour
        return self.business_hours_start <= amsterdam_hour <= self.business_hours_end
    
    def get_adaptive_delay(self) -> float:
        """Calculate adaptive delay based on time of day and usage"""
        base_delay = self.base_delay
        
        # Business hours: more conservative
        if self.is_business_hours():
            base_delay *= 1.5  # 50% slower during business hours
        
        # Add jitter to prevent burst alignment
        jitter = random.uniform(-self.jitter, self.jitter)
        
        # Increase delay if usage is high
        if self.usage_history:
            avg_usage = sum(self.usage_history) / len(self.usage_history)
            if avg_usage > self.usage_threshold:
                base_delay *= 2.0  # Double delay if usage is high
        
        return max(0.5, base_delay + jitter)  # Minimum 0.5s delay
    
    def should_wait_for_concurrency(self) -> bool:
        """Check if we should wait due to concurrency limits"""
        return self.current_concurrent >= self.max_concurrent
    
    def wait_for_concurrency(self):
        """Wait for concurrency slot to become available"""
        while self.should_wait_for_concurrency():
            time.sleep(0.1)  # Check every 100ms
    
    def track_request_start(self, endpoint: str):
        """Track the start of a request"""
        with self.lock:
            self.current_concurrent += 1
            self.endpoints[endpoint].last_request = datetime.now()
            self.usage_history.append(1.0)  # Track as 100% usage
    
    def track_request_end(self, endpoint: str, success: bool = True):
        """Track the end of a request"""
        with self.lock:
            self.current_concurrent = max(0, self.current_concurrent - 1)
            
            if not success:
                self.endpoints[endpoint].consecutive_errors += 1
            else:
                self.endpoints[endpoint].consecutive_errors = 0
    
    def handle_rate_limit_error(self, error_code: int, endpoint: str) -> float:
        """
        Handle rate limit errors with enhanced backoff
        Returns wait time in seconds
        """
        with self.lock:
            state = self.endpoints[endpoint]
            state.consecutive_errors += 1
            
            # Enhanced backoff for UI-critical errors
            if error_code == 1504022:  # Insights Platform
                wait_time = self.insights_platform_wait
                logger.warning(f"🚨 UI-critical error 1504022: waiting {wait_time}s")
            elif error_code == 1504039:  # App-level
                wait_time = self.app_level_wait
                logger.warning(f"🚨 App-level error 1504039: waiting {wait_time}s")
            else:
                # Standard exponential backoff
                wait_time = min(300, 60 * (2 ** state.consecutive_errors))
                logger.warning(f"⚠️ Rate limit error {error_code}: waiting {wait_time}s")
            
            state.backoff_until = datetime.now() + timedelta(seconds=wait_time)
            return wait_time
    
    def is_backed_off(self, endpoint: str) -> bool:
        """Check if endpoint is in backoff period"""
        with self.lock:
            state = self.endpoints[endpoint]
            if state.backoff_until:
                if datetime.now() < state.backoff_until:
                    return True
                else:
                    state.backoff_until = None  # Clear expired backoff
            return False
    
    def get_wait_time(self, endpoint: str) -> float:
        """Get total wait time for a request"""
        wait_time = 0.0
        
        # Check if we're in backoff
        if self.is_backed_off(endpoint):
            with self.lock:
                state = self.endpoints[endpoint]
                if state.backoff_until:
                    wait_time = (state.backoff_until - datetime.now()).total_seconds()
                    if wait_time > 0:
                        return wait_time
        
        # Add adaptive delay
        wait_time += self.get_adaptive_delay()
        
        # Wait for concurrency if needed
        if self.should_wait_for_concurrency():
            wait_time += 1.0  # Add 1 second for concurrency wait
        
        return wait_time
    
    def wait_before_request(self, endpoint: str):
        """Wait before making a request"""
        # Wait for concurrency slot
        self.wait_for_concurrency()
        
        # Check backoff
        if self.is_backed_off(endpoint):
            wait_time = self.get_wait_time(endpoint)
            if wait_time > 0:
                logger.info(f"⏳ Waiting {wait_time:.1f}s due to backoff for {endpoint}")
                time.sleep(wait_time)
        
        # Add adaptive delay
        delay = self.get_adaptive_delay()
        if delay > 0:
            logger.debug(f"⏳ Adaptive delay: {delay:.1f}s for {endpoint}")
            time.sleep(delay)
    
    def get_status(self) -> Dict:
        """Get current rate limiting status"""
        with self.lock:
            return {
                "current_concurrent": self.current_concurrent,
                "max_concurrent": self.max_concurrent,
                "base_delay": self.base_delay,
                "business_hours": self.is_business_hours(),
                "endpoints": {
                    endpoint: {
                        "consecutive_errors": state.consecutive_errors,
                        "backoff_until": state.backoff_until.isoformat() if state.backoff_until else None,
                        "last_request": state.last_request.isoformat() if state.last_request else None
                    }
                    for endpoint, state in self.endpoints.items()
                }
            }

# Global rate limiter instance
_continuous_rate_limiter = None

def get_continuous_rate_limiter() -> ContinuousRateLimiter:
    """Get the global continuous rate limiter instance"""
    global _continuous_rate_limiter
    if _continuous_rate_limiter is None:
        _continuous_rate_limiter = ContinuousRateLimiter()
    return _continuous_rate_limiter

def wait_before_request(endpoint: str = "default"):
    """Wait before making a request (public interface)"""
    limiter = get_continuous_rate_limiter()
    limiter.wait_before_request(endpoint)

def track_request_start(endpoint: str = "default"):
    """Track request start (public interface)"""
    limiter = get_continuous_rate_limiter()
    limiter.track_request_start(endpoint)

def track_request_end(endpoint: str = "default", success: bool = True):
    """Track request end (public interface)"""
    limiter = get_continuous_rate_limiter()
    limiter.track_request_end(endpoint, success)

def handle_rate_limit_error(error_code: int, endpoint: str = "default") -> float:
    """Handle rate limit error (public interface)"""
    limiter = get_continuous_rate_limiter()
    return limiter.handle_rate_limit_error(error_code, endpoint)

def get_rate_limit_status() -> Dict:
    """Get rate limiting status (public interface)"""
    limiter = get_continuous_rate_limiter()
    return limiter.get_status()
