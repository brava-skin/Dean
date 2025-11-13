"""
Error Handling & Recovery System
Retry logic, circuit breakers, and graceful degradation
Enhanced with error pattern detection and self-healing
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, List, Pattern
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import functools
import re

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Open after N failures
    success_threshold: int = 2  # Close after N successes in half-open
    timeout_seconds: int = 60   # Time before trying half-open
    expected_exception: type = Exception


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)


class CircuitBreaker:
    """Circuit breaker pattern for API calls."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = False
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} CLOSED")
            
            if self.state == CircuitState.CLOSED:
                self.failure_count = 0
            
            return result
        
        except self.config.expected_exception as e:
            self._record_failure()
            raise e
        except Exception as e:
            # Check if it's a retryable exception
            if isinstance(e, self.config.expected_exception):
                self._record_failure()
            raise e
    
    def _record_failure(self):
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} back to OPEN")
        
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None


class RetryHandler:
    """Retry logic with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        self.config = config or RetryConfig()
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.config.max_retries} "
                        f"after {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries ({self.config.max_retries}) exceeded")
            
            except Exception as e:
                # Non-retryable exception
                raise e
        
        # All retries exhausted
        raise last_exception or Exception("Retry failed")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff."""
        delay = self.config.initial_delay * (
            self.config.exponential_base ** attempt
        )
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (Exception,),
):
    """Decorator for retry with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            config = RetryConfig(
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                retryable_exceptions=retryable_exceptions,
            )
            handler = RetryHandler(config)
            return handler.execute(func, *args, **kwargs)
        return wrapper
    return decorator


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self) -> None:
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


def with_circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator for circuit breaker protection."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            breaker = circuit_breaker_manager.get_breaker(name, config)
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


@dataclass
class ErrorPattern:
    """Represents a detected error pattern."""
    pattern: str  # Regex pattern or error message
    error_type: str
    count: int
    first_seen: datetime
    last_seen: datetime
    resolution_action: Optional[str] = None
    auto_resolve: bool = False


@dataclass
class DeadLetterQueueEntry:
    """Entry in dead letter queue."""
    operation: str
    data: Dict[str, Any]
    error: Exception
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    error_type: str = ""
    error_message: str = ""
    resolved: bool = False
    resolution_strategy: Optional[str] = None


class EnhancedRetryHandler:
    """Enhanced retry handler with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 300.0,  # 5 minutes
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_range: float = 0.1,  # 10% jitter
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_range = jitter_range
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        # Exponential backoff
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Add jitter
        if self.jitter:
            import random
            jitter_amount = delay * self.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        retryable_exceptions: tuple = (Exception,),
        on_retry: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Execute function with retry and exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}"
                    )
                    
                    if on_retry:
                        on_retry(attempt, e, delay)
                    
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries ({self.max_retries}) exceeded")
        
        raise last_exception


class ErrorPatternDetector:
    """Detects patterns in errors for proactive resolution."""
    
    def __init__(self):
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_history: deque = deque(maxlen=1000)  # Keep last 1000 errors
        self.pattern_threshold = 3  # Minimum occurrences to be considered a pattern
    
    def record_error(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Record an error for pattern detection."""
        error_type = type(error).__name__
        error_message = str(error)
        
        self.error_history.append({
            "error_type": error_type,
            "error_message": error_message,
            "operation": operation,
            "context": context or {},
            "timestamp": datetime.now(),
        })
        
        # Detect patterns
        self._detect_pattern(error_type, error_message, operation)
    
    def _detect_pattern(self, error_type: str, error_message: str, operation: str):
        """Detect error patterns."""
        # Create pattern key
        pattern_key = f"{error_type}:{operation}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = ErrorPattern(
                pattern=error_message[:100],  # First 100 chars
                error_type=error_type,
                count=1,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
        else:
            pattern = self.error_patterns[pattern_key]
            pattern.count += 1
            pattern.last_seen = datetime.now()
            
            # Auto-resolve if pattern detected multiple times
            if pattern.count >= self.pattern_threshold and not pattern.auto_resolve:
                pattern.auto_resolve = True
                pattern.resolution_action = self._suggest_resolution(error_type, operation)
                logger.warning(
                    f"Error pattern detected: {pattern_key} ({pattern.count} occurrences). "
                    f"Suggested resolution: {pattern.resolution_action}"
                )
    
    def _suggest_resolution(self, error_type: str, operation: str) -> str:
        """Suggest resolution for error pattern."""
        # Common resolution strategies
        resolutions = {
            "RateLimitError": "Implement rate limiting and backoff",
            "ConnectionError": "Check network connectivity and retry",
            "TimeoutError": "Increase timeout or implement retry",
            "AuthenticationError": "Refresh API credentials",
            "ValidationError": "Validate input data before processing",
            "KeyError": "Check required fields exist",
            "AttributeError": "Verify object structure",
        }
        
        # Operation-specific resolutions
        if "flux" in operation.lower():
            return "Check Flux API credits and rate limits"
        elif "meta" in operation.lower():
            return "Check Meta API access token and rate limits"
        elif "supabase" in operation.lower():
            return "Check Supabase connection and credentials"
        
        return resolutions.get(error_type, "Review error logs and implement fix")
    
    def get_patterns(self) -> List[ErrorPattern]:
        """Get all detected error patterns."""
        return list(self.error_patterns.values())
    
    def get_pattern_for_error(self, error_type: str, operation: str) -> Optional[ErrorPattern]:
        """Get pattern for specific error type and operation."""
        pattern_key = f"{error_type}:{operation}"
        return self.error_patterns.get(pattern_key)


class DeadLetterQueue:
    """Dead letter queue for failed operations."""
    
    def __init__(self, max_size: int = 1000, storage_backend=None):
        self.storage = storage_backend
        self.queue: List[DeadLetterQueueEntry] = []
        self.max_size = max_size
        self.auto_retry_enabled = True
        self.retry_interval_hours = 24
    
    def add(
        self,
        operation: str,
        data: Dict[str, Any],
        error: Exception,
        max_retries: int = 3,
    ):
        """Add entry to dead letter queue."""
        entry = DeadLetterQueueEntry(
            operation=operation,
            data=data,
            error=error,
            max_retries=max_retries,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        
        self.queue.append(entry)
        
        # Trim if over max size
        if len(self.queue) > self.max_size:
            self.queue = self.queue[-self.max_size:]
        
        # Persist if storage available
        if self.storage:
            try:
                self._persist(entry)
            except Exception as e:
                logger.error(f"Failed to persist DLQ entry: {e}")
        
        logger.warning(
            f"Added to dead letter queue: {operation} - {entry.error_type}: {entry.error_message}"
        )
    
    def _persist(self, entry: DeadLetterQueueEntry):
        """Persist entry to storage."""
        # Implementation depends on storage backend
        pass
    
    def get_entries(
        self,
        operation: Optional[str] = None,
        error_type: Optional[str] = None,
        unresolved_only: bool = True,
    ) -> List[DeadLetterQueueEntry]:
        """Get entries from dead letter queue."""
        entries = self.queue
        
        if operation:
            entries = [e for e in entries if e.operation == operation]
        if error_type:
            entries = [e for e in entries if e.error_type == error_type]
        if unresolved_only:
            entries = [e for e in entries if not e.resolved]
        
        return entries
    
    def get_failed_operations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get failed operations (backward compatibility)."""
        return [
            {
                "operation": e.operation,
                "data": e.data,
                "error": e.error_message,
                "error_type": e.error_type,
                "retry_count": e.retry_count,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in self.queue[-limit:]
        ]
    
    def retry_entry(
        self,
        entry: DeadLetterQueueEntry,
        retry_func: Callable,
    ) -> bool:
        """Retry a dead letter queue entry."""
        if entry.retry_count >= entry.max_retries:
            logger.warning(f"Entry exceeded max retries: {entry.operation}")
            return False
        
        try:
            entry.retry_count += 1
            result = retry_func(entry.data)
            
            # Mark as resolved if successful
            entry.resolved = True
            entry.resolution_strategy = "automatic_retry"
            logger.info(f"Successfully retried dead letter queue entry: {entry.operation}")
            return True
            
        except Exception as e:
            logger.error(f"Retry failed for {entry.operation}: {e}")
            return False
    
    def auto_retry_eligible_entries(self, retry_func_map: Dict[str, Callable]):
        """Automatically retry eligible entries."""
        if not self.auto_retry_enabled:
            return
        
        cutoff_time = datetime.now() - timedelta(hours=self.retry_interval_hours)
        
        for entry in self.queue:
            if entry.resolved:
                continue
            
            if entry.timestamp < cutoff_time and entry.retry_count < entry.max_retries:
                retry_func = retry_func_map.get(entry.operation)
                if retry_func:
                    self.retry_entry(entry, retry_func)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dead letter queue statistics."""
        total = len(self.queue)
        unresolved = len([e for e in self.queue if not e.resolved])
        
        by_operation = defaultdict(int)
        by_error_type = defaultdict(int)
        
        for entry in self.queue:
            by_operation[entry.operation] += 1
            by_error_type[entry.error_type] += 1
        
        return {
            "total_entries": total,
            "unresolved_entries": unresolved,
            "resolved_entries": total - unresolved,
            "by_operation": dict(by_operation),
            "by_error_type": dict(by_error_type),
        }
    
    def clear(self):
        """Clear the queue."""
        self.queue.clear()


class SelfHealingSystem:
    """Self-healing system that automatically resolves common issues."""
    
    def __init__(self):
        self.healing_strategies: Dict[str, Callable] = {}
        self.error_detector = ErrorPatternDetector()
        self.register_default_strategies()
    
    def register_default_strategies(self):
        """Register default self-healing strategies."""
        # Rate limit healing
        self.register_healing_strategy(
            "rate_limit",
            self._heal_rate_limit,
        )
        
        # Connection healing
        self.register_healing_strategy(
            "connection",
            self._heal_connection,
        )
        
        # Authentication healing
        self.register_healing_strategy(
            "authentication",
            self._heal_authentication,
        )
    
    def register_healing_strategy(self, error_type: str, strategy: Callable):
        """Register a healing strategy for an error type."""
        self.healing_strategies[error_type] = strategy
    
    def attempt_healing(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Attempt to heal an error automatically."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Record error for pattern detection
        self.error_detector.record_error(error, operation, context)
        
        # Check for known patterns with auto-resolve
        pattern = self.error_detector.get_pattern_for_error(error_type, operation)
        if pattern and pattern.auto_resolve:
            logger.info(f"Attempting auto-healing for {error_type} in {operation}")
            
            # Try healing strategies
            for heal_type, strategy in self.healing_strategies.items():
                if heal_type in error_message or error_type.lower().startswith(heal_type):
                    try:
                        if strategy(operation, context):
                            logger.info(f"Successfully healed {error_type} using {heal_type} strategy")
                            return True
                    except Exception as e:
                        logger.error(f"Healing strategy {heal_type} failed: {e}")
        
        return False
    
    def _heal_rate_limit(self, operation: str, context: Optional[Dict[str, Any]]) -> bool:
        """Heal rate limit errors by waiting and backing off."""
        logger.info("Applying rate limit healing: backing off")
        time.sleep(60)  # Wait 1 minute
        return True
    
    def _heal_connection(self, operation: str, context: Optional[Dict[str, Any]]) -> bool:
        """Heal connection errors by retrying."""
        logger.info("Applying connection healing: retrying connection")
        time.sleep(5)  # Brief wait
        return True
    
    def _heal_authentication(self, operation: str, context: Optional[Dict[str, Any]]) -> bool:
        """Heal authentication errors (requires manual intervention)."""
        logger.warning("Authentication error detected - requires manual credential refresh")
        return False


# Global dead letter queue
dead_letter_queue = DeadLetterQueue()


def graceful_degradation(fallback_value: Any = None, fallback_func: Callable = None):
    """Decorator for graceful degradation."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Graceful degradation for {func.__name__}: {e}")
                
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                
                return fallback_value
        
        return wrapper
    return decorator


# Global instances for enhanced error recovery
_enhanced_retry_handler = EnhancedRetryHandler()
_error_pattern_detector = ErrorPatternDetector()
_self_healing_system = SelfHealingSystem()


def get_enhanced_retry_handler() -> EnhancedRetryHandler:
    """Get global enhanced retry handler."""
    return _enhanced_retry_handler


def get_error_pattern_detector() -> ErrorPatternDetector:
    """Get global error pattern detector."""
    return _error_pattern_detector


def get_self_healing_system() -> SelfHealingSystem:
    """Get global self-healing system."""
    return _self_healing_system


# Enhanced retry wrapper that uses new system
def enhanced_retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = 5,
    retryable_exceptions: tuple = (Exception,),
    **kwargs,
) -> T:
    """Enhanced retry with improved exponential backoff."""
    retry_handler = get_enhanced_retry_handler()
    return retry_handler.retry_with_backoff(
        func,
        *args,
        retryable_exceptions=retryable_exceptions,
        **kwargs,
    )


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "RetryHandler",
    "RetryConfig",
    "retry_with_backoff",
    "enhanced_retry_with_backoff",
    "EnhancedRetryHandler",
    "ErrorPattern",
    "ErrorPatternDetector",
    "DeadLetterQueueEntry",
    "SelfHealingSystem",
    "get_enhanced_retry_handler",
    "get_error_pattern_detector",
    "get_self_healing_system",
    "with_circuit_breaker",
    "CircuitBreakerManager",
    "circuit_breaker_manager",
    "DeadLetterQueue",
    "dead_letter_queue",
    "graceful_degradation",
]

