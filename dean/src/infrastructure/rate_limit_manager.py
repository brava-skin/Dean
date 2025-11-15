from __future__ import annotations

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import queue

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    META_API = "meta_api"
    FLUX_API = "flux_api"
    SUPABASE = "supabase"
    OPENAI = "openai"


@dataclass
class RateLimitConfig:
    requests_per_window: int
    window_seconds: int
    burst_limit: Optional[int] = None
    retry_after_header: Optional[str] = None
    backoff_multiplier: float = 1.5
    max_backoff_seconds: int = 300


@dataclass
class RateLimitState:
    requests_made: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    next_available: Optional[datetime] = None
    estimated_reset: Optional[datetime] = None
    backoff_until: Optional[datetime] = None


@dataclass
class QueuedRequest:
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    max_retries: int = 3
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None


class RateLimitManager:
    def __init__(self):
        self.configs: Dict[RateLimitType, RateLimitConfig] = {
            RateLimitType.META_API: RateLimitConfig(
                requests_per_window=200,
                window_seconds=3600,
                burst_limit=50,
                retry_after_header="x-business-use-case-usage",
            ),
            RateLimitType.FLUX_API: RateLimitConfig(
                requests_per_window=100,
                window_seconds=3600,
                burst_limit=20,
            ),
            RateLimitType.SUPABASE: RateLimitConfig(
                requests_per_window=500,
                window_seconds=60,
                burst_limit=100,
            ),
            RateLimitType.OPENAI: RateLimitConfig(
                requests_per_window=5000,
                window_seconds=3600,
                burst_limit=500,
            ),
        }
        self.states: Dict[RateLimitType, RateLimitState] = {
            limit_type: RateLimitState() for limit_type in RateLimitType
        }
        self.request_queues: Dict[RateLimitType, queue.PriorityQueue] = {
            limit_type: queue.PriorityQueue() for limit_type in RateLimitType
        }
        self.locks: Dict[RateLimitType, threading.Lock] = {
            limit_type: threading.Lock() for limit_type in RateLimitType
        }
        self.worker_threads: Dict[RateLimitType, threading.Thread] = {}
        self.running = True
        for limit_type in RateLimitType:
            thread = threading.Thread(
                target=self._worker_loop,
                args=(limit_type,),
                daemon=True,
                name=f"rate_limit_worker_{limit_type.value}",
            )
            thread.start()
            self.worker_threads[limit_type] = thread
        self.request_history: Dict[RateLimitType, deque] = {
            limit_type: deque(maxlen=100) for limit_type in RateLimitType
        }
    
    def _worker_loop(self, limit_type: RateLimitType):
        while self.running:
            try:
                priority, queued_request = self.request_queues[limit_type].get(timeout=1.0)
                self._wait_for_rate_limit(limit_type)
                try:
                    result = queued_request.func(*queued_request.args, **queued_request.kwargs)
                    self._record_request(limit_type, success=True)
                    if queued_request.callback:
                        queued_request.callback(result)
                except Exception as e:
                    self._record_request(limit_type, success=False)
                    if queued_request.retry_count < queued_request.max_retries:
                        queued_request.retry_count += 1
                        self.request_queues[limit_type].put((
                            queued_request.priority - 1,
                            queued_request,
                        ))
                        logger.warning(f"Retrying request {queued_request.retry_count}/{queued_request.max_retries}: {e}")
                    else:
                        if queued_request.error_callback:
                            queued_request.error_callback(e)
                        logger.error(f"Request failed after {queued_request.max_retries} retries: {e}")
                self.request_queues[limit_type].task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in rate limit worker {limit_type.value}: {e}")
                time.sleep(1)
    
    def _wait_for_rate_limit(self, limit_type: RateLimitType):
        with self.locks[limit_type]:
            state = self.states[limit_type]
            config = self.configs[limit_type]
            now = datetime.now()
            if state.backoff_until and now < state.backoff_until:
                sleep_seconds = (state.backoff_until - now).total_seconds()
                logger.debug(f"Rate limit backoff: sleeping {sleep_seconds:.2f}s")
                time.sleep(sleep_seconds)
                now = datetime.now()
            window_age = (now - state.window_start).total_seconds()
            if window_age >= config.window_seconds:
                state.requests_made = 0
                state.window_start = now
            if state.requests_made >= config.requests_per_window:
                elapsed = (now - state.window_start).total_seconds()
                wait_seconds = config.window_seconds - elapsed
                if wait_seconds > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_seconds:.2f}s")
                    time.sleep(wait_seconds)
                    state.requests_made = 0
                    state.window_start = datetime.now()
            state.requests_made += 1
    
    def _record_request(self, limit_type: RateLimitType, success: bool):
        with self.locks[limit_type]:
            self.request_history[limit_type].append({
                "timestamp": datetime.now(),
                "success": success,
            })
    
    def queue_request(
        self,
        limit_type: RateLimitType,
        func: Callable,
        *args,
        priority: int = 0,
        max_retries: int = 3,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        **kwargs,
    ) -> QueuedRequest:
        queued_request = QueuedRequest(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            callback=callback,
            error_callback=error_callback,
        )
        self.request_queues[limit_type].put((-priority, queued_request))
        return queued_request
    
    def execute_with_rate_limit(
        self,
        limit_type: RateLimitType,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def success_callback(result):
            result_queue.put(("success", result))
        
        def error_callback(error):
            error_queue.put(("error", error))
        
        self.queue_request(
            limit_type=limit_type,
            func=func,
            *args,
            priority=0,
            callback=success_callback,
            error_callback=error_callback,
            **kwargs,
        )
        while True:
            try:
                if not error_queue.empty():
                    status, value = error_queue.get_nowait()
                    if status == "error":
                        raise value
                if not result_queue.empty():
                    status, value = result_queue.get_nowait()
                    if status == "success":
                        return value
            except queue.Empty:
                time.sleep(0.1)
    
    def predict_rate_limit_reset(self, limit_type: RateLimitType) -> Optional[datetime]:
        with self.locks[limit_type]:
            state = self.states[limit_type]
            config = self.configs[limit_type]
            if state.requests_made < config.requests_per_window:
                return None
            reset_time = state.window_start + timedelta(seconds=config.window_seconds)
            return reset_time
    
    def get_rate_limit_status(self, limit_type: RateLimitType) -> Dict[str, Any]:
        with self.locks[limit_type]:
            state = self.states[limit_type]
            config = self.configs[limit_type]
            remaining = config.requests_per_window - state.requests_made
            utilization = state.requests_made / config.requests_per_window if config.requests_per_window > 0 else 0
            return {
                "limit_type": limit_type.value,
                "requests_made": state.requests_made,
                "requests_limit": config.requests_per_window,
                "remaining": remaining,
                "utilization": utilization,
                "window_start": state.window_start.isoformat(),
                "queued_requests": self.request_queues[limit_type].qsize(),
                "predicted_reset": self.predict_rate_limit_reset(limit_type).isoformat() if self.predict_rate_limit_reset(limit_type) else None,
            }
    
    def handle_rate_limit_response(
        self,
        limit_type: RateLimitType,
        response_headers: Dict[str, str],
    ):
        with self.locks[limit_type]:
            state = self.states[limit_type]
            config = self.configs[limit_type]
            if config.retry_after_header and config.retry_after_header in response_headers:
                retry_after = int(response_headers[config.retry_after_header])
                state.backoff_until = datetime.now() + timedelta(seconds=retry_after)
                logger.warning(f"Rate limit hit, backing off for {retry_after}s")
            if "x-ratelimit-remaining" in response_headers:
                remaining = int(response_headers["x-ratelimit-remaining"])
                state.requests_made = config.requests_per_window - remaining
            if "x-ratelimit-reset" in response_headers:
                reset_timestamp = int(response_headers["x-ratelimit-reset"])
                state.estimated_reset = datetime.fromtimestamp(reset_timestamp)
    
    def batch_requests(
        self,
        limit_type: RateLimitType,
        requests: List[Tuple[Callable, tuple, dict]],
        priority: int = 0,
    ) -> List[Any]:
        results = []
        result_queue = queue.Queue()
        completed = 0
        total = len(requests)
        
        def batch_callback(result):
            nonlocal completed
            result_queue.put(result)
            completed += 1
        
        def batch_error_callback(error):
            nonlocal completed
            result_queue.put(("error", error))
            completed += 1
        
        for func, args, kwargs in requests:
            self.queue_request(
                limit_type=limit_type,
                func=func,
                *args,
                priority=priority,
                callback=batch_callback,
                error_callback=batch_error_callback,
                **kwargs,
            )
        while completed < total:
            try:
                result = result_queue.get(timeout=1.0)
                if isinstance(result, tuple) and result[0] == "error":
                    raise result[1]
                results.append(result)
            except queue.Empty:
                continue
        return results
    
    def shutdown(self):
        self.running = False
        for limit_type in RateLimitType:
            self.request_queues[limit_type].join()


_rate_limit_manager: Optional[RateLimitManager] = None


def get_rate_limit_manager() -> RateLimitManager:
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
    return _rate_limit_manager

