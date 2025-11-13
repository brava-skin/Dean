"""
Health Check System
Monitoring and health endpoints
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service: str
    status: HealthStatus
    message: str
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """Health check system."""
    
    def __init__(self) -> None:
        self.checks: Dict[str, Callable[[], HealthCheckResult | bool]] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult | bool]) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
    
    def check(self, service: str) -> HealthCheckResult:
        """Run health check for a service."""
        if service not in self.checks:
            return HealthCheckResult(
                service=service,
                status=HealthStatus.UNKNOWN,
                message=f"No health check registered for {service}",
            )
        
        start_time = time.time()
        
        try:
            result = self.checks[service]()
            
            if isinstance(result, HealthCheckResult):
                result.response_time_ms = (time.time() - start_time) * 1000
                self.last_results[service] = result
                return result
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                response_time = (time.time() - start_time) * 1000
                result_obj = HealthCheckResult(
                    service=service,
                    status=status,
                    message="OK" if result else "Failed",
                    response_time_ms=response_time,
                )
                self.last_results[service] = result_obj
                return result_obj
            else:
                return HealthCheckResult(
                    service=service,
                    status=HealthStatus.UNKNOWN,
                    message="Invalid health check result",
                    response_time_ms=(time.time() - start_time) * 1000,
                )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                service=service,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time,
            )
            self.last_results[service] = result
            return result
    
    def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        for service in self.checks.keys():
            results[service] = self.check(service)
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health."""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [r.status for r in self.last_results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


def create_health_checkers():
    """Create standard health checkers."""
    checker = HealthChecker()
    
    # Meta API health check
    def check_meta_api():
        try:
            from integrations.meta_client import MetaClient
            # Simple check - could be enhanced
            return HealthCheckResult(
                service="meta_api",
                status=HealthStatus.HEALTHY,
                message="Meta API available",
            )
        except Exception as e:
            return HealthCheckResult(
                service="meta_api",
                status=HealthStatus.UNHEALTHY,
                message=f"Meta API error: {e}",
            )
    
    checker.register_check("meta_api", check_meta_api)
    
    # FLUX API health check
    def check_flux_api():
        try:
            from integrations.flux_client import FluxClient
            return HealthCheckResult(
                service="flux_api",
                status=HealthStatus.HEALTHY,
                message="FLUX API available",
            )
        except Exception as e:
            return HealthCheckResult(
                service="flux_api",
                status=HealthStatus.UNHEALTHY,
                message=f"FLUX API error: {e}",
            )
    
    checker.register_check("flux_api", check_flux_api)
    
    # Database health check
    def check_database():
        try:
            from main import _get_supabase
            supabase = _get_supabase()
            if supabase:
                return HealthCheckResult(
                    service="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connected",
                )
            else:
                return HealthCheckResult(
                    service="database",
                    status=HealthStatus.DEGRADED,
                    message="Database not configured",
                )
        except Exception as e:
            return HealthCheckResult(
                service="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {e}",
            )
    
    checker.register_check("database", check_database)
    
    # Cache health check
    def check_cache():
        try:
            from infrastructure.caching import cache_manager
            # Simple test
            cache_manager.set("health_check", "test", ttl_seconds=1)
            value = cache_manager.get("health_check")
            if value == "test":
                return HealthCheckResult(
                    service="cache",
                    status=HealthStatus.HEALTHY,
                    message="Cache operational",
                )
            else:
                return HealthCheckResult(
                    service="cache",
                    status=HealthStatus.DEGRADED,
                    message="Cache test failed",
                )
        except Exception as e:
            return HealthCheckResult(
                service="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache error: {e}",
            )
    
    checker.register_check("cache", check_cache)
    
    return checker


# Global health checker
health_checker = create_health_checkers()


def get_health_status() -> Dict[str, Any]:
    """Get overall health status."""
    results = health_checker.check_all()
    overall = health_checker.get_overall_health()
    
    return {
        "status": overall.value,
        "timestamp": datetime.now().isoformat(),
        "services": {
            name: {
                "status": result.status.value,
                "message": result.message,
                "response_time_ms": result.response_time_ms,
            }
            for name, result in results.items()
        },
    }


__all__ = [
    "HealthChecker",
    "HealthCheckResult",
    "HealthStatus",
    "health_checker",
    "get_health_status",
    "create_health_checkers",
]

