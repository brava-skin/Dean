"""
Caching Infrastructure
Redis and in-memory caching for performance
"""

from __future__ import annotations

import logging
import json
import hashlib
import pickle
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
import functools

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory cache only")


class InMemoryCache:
    """Simple in-memory cache."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        value, expiry = self.cache[key]
        
        if datetime.now() > expiry:
            del self.cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache."""
        ttl = ttl_seconds or self.ttl_seconds
        expiry = datetime.now() + timedelta(seconds=ttl)
        
        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, expiry)
    
    def delete(self, key: str):
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            k for k, (_, expiry) in self.cache.items()
            if now > expiry
        ]
        for key in expired_keys:
            del self.cache[key]


class RedisCache:
    """Redis-based cache."""
    
    def __init__(self, redis_client=None, default_ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")
        
        self.client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=False,  # We'll handle encoding
        )
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            data = self.client.get(key)
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in Redis."""
        try:
            data = pickle.dumps(value)
            ttl = ttl_seconds or self.default_ttl
            self.client.setex(key, ttl, data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str):
        """Delete key from Redis."""
        try:
            self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern."""
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear_pattern error: {e}")


class CacheManager:
    """Unified cache manager."""
    
    def __init__(self, use_redis: bool = True):
        self.redis_cache = None
        self.memory_cache = InMemoryCache()
        
        if use_redis and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache()
                logger.info("✅ Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis cache not available: {e}")
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache (tries Redis first, then memory)."""
        full_key = f"{namespace}:{key}"
        
        # Try Redis first
        if self.redis_cache:
            value = self.redis_cache.get(full_key)
            if value is not None:
                return value
        
        # Fallback to memory
        return self.memory_cache.get(full_key)
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        namespace: str = "default",
    ):
        """Set value in cache (both Redis and memory)."""
        full_key = f"{namespace}:{key}"
        
        # Set in Redis
        if self.redis_cache:
            try:
                self.redis_cache.set(full_key, value, ttl_seconds)
            except Exception:
                pass  # Fallback to memory only
        
        # Always set in memory as backup
        self.memory_cache.set(full_key, value, ttl_seconds)
    
    def delete(self, key: str, namespace: str = "default"):
        """Delete key from cache."""
        full_key = f"{namespace}:{key}"
        
        if self.redis_cache:
            self.redis_cache.delete(full_key)
        
        self.memory_cache.delete(full_key)
    
    def clear_namespace(self, namespace: str):
        """Clear all keys in a namespace."""
        if self.redis_cache:
            self.redis_cache.clear_pattern(f"{namespace}:*")
        
        # Clear from memory cache
        keys_to_delete = [
            k for k in self.memory_cache.cache.keys()
            if k.startswith(f"{namespace}:")
        ]
        for key in keys_to_delete:
            self.memory_cache.delete(key)


# Global cache manager
cache_manager = CacheManager()


def cached(
    ttl_seconds: int = 3600,
    namespace: str = "default",
    key_func: Optional[Callable] = None,
):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(sorted(kwargs.items())),
                }
                cache_key = hashlib.md5(
                    json.dumps(key_data, sort_keys=True).encode()
                ).hexdigest()
            
            # Try cache
            cached_value = cache_manager.get(cache_key, namespace)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_key, result, ttl_seconds, namespace)
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(pattern: str, namespace: str = "default"):
    """Invalidate cache entries matching pattern."""
    cache_manager.clear_namespace(namespace)


# Aggressive caching configurations for specific use cases
ML_PREDICTION_TTL = 3600  # 1 hour for ML predictions
CREATIVE_DNA_TTL = 86400  # 24 hours for Creative DNA
PERFORMANCE_DATA_TTL = 1800  # 30 minutes for performance data
API_RESPONSE_TTL = 300  # 5 minutes for API responses


class AggressiveCacheManager(CacheManager):
    """Enhanced cache manager with aggressive caching for ML and performance data."""
    
    def cache_ml_prediction(self, ad_id: str, prediction: Any):
        """Cache ML prediction with aggressive TTL."""
        self.set(
            key=f"ml_prediction:{ad_id}",
            value=prediction,
            ttl_seconds=ML_PREDICTION_TTL,
            namespace="ml",
        )
    
    def get_ml_prediction(self, ad_id: str) -> Optional[Any]:
        """Get cached ML prediction."""
        return self.get(key=f"ml_prediction:{ad_id}", namespace="ml")
    
    def cache_creative_dna(self, creative_id: str, dna_vector: Any):
        """Cache Creative DNA vector with long TTL."""
        self.set(
            key=f"creative_dna:{creative_id}",
            value=dna_vector,
            ttl_seconds=CREATIVE_DNA_TTL,
            namespace="creative_dna",
        )
    
    def get_creative_dna(self, creative_id: str) -> Optional[Any]:
        """Get cached Creative DNA vector."""
        return self.get(key=f"creative_dna:{creative_id}", namespace="creative_dna")
    
    def cache_performance_aggregation(
        self,
        cache_key: str,
        aggregation: Any,
        ttl_seconds: int = PERFORMANCE_DATA_TTL,
    ):
        """Cache performance data aggregation."""
        self.set(
            key=f"perf_agg:{cache_key}",
            value=aggregation,
            ttl_seconds=ttl_seconds,
            namespace="performance",
        )
    
    def get_performance_aggregation(self, cache_key: str) -> Optional[Any]:
        """Get cached performance aggregation."""
        return self.get(key=f"perf_agg:{cache_key}", namespace="performance")
    
    def cache_api_response(
        self,
        api_name: str,
        request_hash: str,
        response: Any,
        ttl_seconds: int = API_RESPONSE_TTL,
    ):
        """Cache API response."""
        self.set(
            key=f"api:{api_name}:{request_hash}",
            value=response,
            ttl_seconds=ttl_seconds,
            namespace="api",
        )
    
    def get_api_response(self, api_name: str, request_hash: str) -> Optional[Any]:
        """Get cached API response."""
        return self.get(key=f"api:{api_name}:{request_hash}", namespace="api")
    
    def invalidate_ml_cache(self):
        """Invalidate all ML prediction cache."""
        self.clear_namespace("ml")
    
    def invalidate_creative_dna_cache(self):
        """Invalidate all Creative DNA cache."""
        self.clear_namespace("creative_dna")
    
    def invalidate_performance_cache(self):
        """Invalidate all performance data cache."""
        self.clear_namespace("performance")
    
    def invalidate_api_cache(self, api_name: Optional[str] = None):
        """Invalidate API response cache."""
        if api_name:
            # Invalidate specific API cache
            pattern = f"api:{api_name}:*"
            if self.redis_cache:
                self.redis_cache.clear_pattern(pattern)
        else:
            # Invalidate all API cache
            self.clear_namespace("api")


# Enhanced global cache manager
cache_manager = AggressiveCacheManager()


__all__ = [
    "CacheManager",
    "cache_manager",
    "RedisCache",
    "InMemoryCache",
    "cached",
    "invalidate_cache",
]

