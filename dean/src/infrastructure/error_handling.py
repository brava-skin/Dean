from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import functools

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    expected_exception: type = Exception


@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)


class CircuitBreaker:
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = False
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        if self.state == CircuitState.OPEN:
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
            if isinstance(e, self.config.expected_exception):
                self._record_failure()
            raise e
    
    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} back to OPEN")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")
    
    def reset(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None


class RetryHandler:
    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        self.config = config or RetryConfig()
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
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
                raise e
        raise last_exception or Exception("Retry failed")
    
    def _calculate_delay(self, attempt: int) -> float:
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
    def __init__(self) -> None:
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def reset_all(self):
        for breaker in self.breakers.values():
            breaker.reset()


circuit_breaker_manager = CircuitBreakerManager()


def with_circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            breaker = circuit_breaker_manager.get_breaker(name, config)
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


@dataclass
class ErrorPattern:
    pattern: str
    error_type: str
    count: int
    first_seen: datetime
    last_seen: datetime
    resolution_action: Optional[str] = None
    auto_resolve: bool = False


@dataclass
class DeadLetterQueueEntry:
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
    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 300.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_range: float = 0.1,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_range = jitter_range
    
    def calculate_delay(self, attempt: int) -> float:
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
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
    def __init__(self):
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.pattern_threshold = 3
    
    def record_error(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_type = type(error).__name__
        error_message = str(error)
        self.error_history.append({
            "error_type": error_type,
            "error_message": error_message,
            "operation": operation,
            "context": context or {},
            "timestamp": datetime.now(),
        })
        self._detect_pattern(error_type, error_message, operation)
    
    def _detect_pattern(self, error_type: str, error_message: str, operation: str):
        pattern_key = f"{error_type}:{operation}"
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = ErrorPattern(
                pattern=error_message[:100],
                error_type=error_type,
                count=1,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
        else:
            pattern = self.error_patterns[pattern_key]
            pattern.count += 1
            pattern.last_seen = datetime.now()
            if pattern.count >= self.pattern_threshold and not pattern.auto_resolve:
                pattern.auto_resolve = True
                pattern.resolution_action = self._suggest_resolution(error_type, operation)
                logger.warning(
                    f"Error pattern detected: {pattern_key} ({pattern.count} occurrences). "
                    f"Suggested resolution: {pattern.resolution_action}"
                )
    
    def _suggest_resolution(self, error_type: str, operation: str) -> str:
        resolutions = {
            "RateLimitError": "Implement rate limiting and backoff",
            "ConnectionError": "Check network connectivity and retry",
            "TimeoutError": "Increase timeout or implement retry",
            "AuthenticationError": "Refresh API credentials",
            "ValidationError": "Validate input data before processing",
            "KeyError": "Check required fields exist",
            "AttributeError": "Verify object structure",
        }
        if "flux" in operation.lower():
            return "Check Flux API credits and rate limits"
        elif "meta" in operation.lower():
            return "Check Meta API access token and rate limits"
        elif "supabase" in operation.lower():
            return "Check Supabase connection and credentials"
        return resolutions.get(error_type, "Review error logs and implement fix")
    
    def get_patterns(self) -> List[ErrorPattern]:
        return list(self.error_patterns.values())
    
    def get_pattern_for_error(self, error_type: str, operation: str) -> Optional[ErrorPattern]:
        pattern_key = f"{error_type}:{operation}"
        return self.error_patterns.get(pattern_key)


class DeadLetterQueue:
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
        entry = DeadLetterQueueEntry(
            operation=operation,
            data=data,
            error=error,
            max_retries=max_retries,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        self.queue.append(entry)
        if len(self.queue) > self.max_size:
            self.queue = self.queue[-self.max_size:]
        if self.storage:
            try:
                self._persist(entry)
            except Exception as e:
                logger.error(f"Failed to persist DLQ entry: {e}")
        logger.warning(
            f"Added to dead letter queue: {operation} - {entry.error_type}: {entry.error_message}"
        )
    
    def _persist(self, entry: DeadLetterQueueEntry):
        pass
    
    def get_entries(
        self,
        operation: Optional[str] = None,
        error_type: Optional[str] = None,
        unresolved_only: bool = True,
    ) -> List[DeadLetterQueueEntry]:
        entries = self.queue
        if operation:
            entries = [e for e in entries if e.operation == operation]
        if error_type:
            entries = [e for e in entries if e.error_type == error_type]
        if unresolved_only:
            entries = [e for e in entries if not e.resolved]
        return entries
    
    def get_failed_operations(self, limit: int = 100) -> List[Dict[str, Any]]:
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
        if entry.retry_count >= entry.max_retries:
            logger.warning(f"Entry exceeded max retries: {entry.operation}")
            return False
        try:
            entry.retry_count += 1
            result = retry_func(entry.data)
            entry.resolved = True
            entry.resolution_strategy = "automatic_retry"
            logger.info(f"Successfully retried dead letter queue entry: {entry.operation}")
            return True
        except Exception as e:
            logger.error(f"Retry failed for {entry.operation}: {e}")
            return False
    
    def auto_retry_eligible_entries(self, retry_func_map: Dict[str, Callable]):
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
        self.queue.clear()


class SelfHealingSystem:
    def __init__(self):
        self.healing_strategies: Dict[str, Callable] = {}
        self.error_detector = ErrorPatternDetector()
        self.register_default_strategies()
    
    def register_default_strategies(self):
        self.register_healing_strategy(
            "rate_limit",
            self._heal_rate_limit,
        )
        self.register_healing_strategy(
            "connection",
            self._heal_connection,
        )
        self.register_healing_strategy(
            "authentication",
            self._heal_authentication,
        )
    
    def register_healing_strategy(self, error_type: str, strategy: Callable):
        self.healing_strategies[error_type] = strategy
    
    def attempt_healing(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        error_type = type(error).__name__
        error_message = str(error).lower()
        self.error_detector.record_error(error, operation, context)
        pattern = self.error_detector.get_pattern_for_error(error_type, operation)
        if pattern and pattern.auto_resolve:
            logger.info(f"Attempting auto-healing for {error_type} in {operation}")
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
        logger.info("Applying rate limit healing: backing off")
        time.sleep(60)
        return True
    
    def _heal_connection(self, operation: str, context: Optional[Dict[str, Any]]) -> bool:
        logger.info("Applying connection healing: retrying connection")
        time.sleep(5)
        return True
    
    def _heal_authentication(self, operation: str, context: Optional[Dict[str, Any]]) -> bool:
        logger.warning("Authentication error detected - requires manual credential refresh")
        return False


dead_letter_queue = DeadLetterQueue()

_enhanced_retry_handler = EnhancedRetryHandler()
_error_pattern_detector = ErrorPatternDetector()
_self_healing_system = SelfHealingSystem()


def enhanced_retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = 5,
    retryable_exceptions: tuple = (Exception,),
    **kwargs,
) -> T:
    retry_handler = _enhanced_retry_handler
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
    "with_circuit_breaker",
    "CircuitBreakerManager",
    "circuit_breaker_manager",
    "DeadLetterQueue",
    "dead_letter_queue",
]

