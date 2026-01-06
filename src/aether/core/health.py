"""
AETHER Health Check System

Production-grade health monitoring with:
- Component health checks
- System diagnostics
- Resource monitoring
- Dependency verification
"""

from __future__ import annotations

import asyncio
import functools
import os
import platform
import time

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

from aether.core.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "last_check": self.last_check.isoformat(),
            "error": self.error,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    components: list[ComponentHealth]
    uptime_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = "0.1.0"
    environment: str = "production"
    system_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "components": [c.to_dict() for c in self.components],
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "environment": self.environment,
            "system_info": self.system_info,
        }


class HealthCheck:
    """
    Health check registry and executor.

    Manages registration and execution of health checks for system components.

    Usage:
        health = HealthCheck()

        @health.register("database")
        async def check_database():
            # Check database connection
            return True, "Connected"

        # Run all checks
        system_health = await health.check_all()
    """

    # Global instance
    _instance: HealthCheck | None = None

    def __init__(self):
        self._checks: dict[str, Callable[[], Any]] = {}
        self._start_time = datetime.utcnow()
        self._last_results: dict[str, ComponentHealth] = {}

    @classmethod
    def get_instance(cls) -> HealthCheck:
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        name: str,
        timeout: float = 5.0,
        critical: bool = False,
    ) -> Callable[[F], F]:
        """
        Decorator to register a health check.

        Args:
            name: Component name
            timeout: Check timeout in seconds
            critical: If True, failure makes system unhealthy

        The decorated function should return:
        - bool: True for healthy, False for unhealthy
        - tuple[bool, str]: (status, message)
        - tuple[bool, str, dict]: (status, message, details)
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def wrapper() -> ComponentHealth:
                start_time = time.perf_counter()
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await asyncio.wait_for(func(), timeout=timeout)
                    else:
                        result = func()

                    latency_ms = (time.perf_counter() - start_time) * 1000

                    # Parse result
                    if isinstance(result, bool):
                        status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                        message = None
                        details = {}
                    elif isinstance(result, tuple):
                        if len(result) == 2:
                            healthy, message = result
                            details = {}
                        else:
                            healthy, message, details = result
                        status = HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY
                    else:
                        status = HealthStatus.UNKNOWN
                        message = f"Invalid check result: {type(result)}"
                        details = {}

                    return ComponentHealth(
                        name=name,
                        status=status,
                        message=message,
                        latency_ms=latency_ms,
                        details=details,
                    )

                except asyncio.TimeoutError:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check timed out after {timeout}s",
                        latency_ms=latency_ms,
                        error="timeout",
                    )
                except Exception as e:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    logger.error(f"Health check '{name}' failed: {e}")
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=str(e),
                        latency_ms=latency_ms,
                        error=type(e).__name__,
                    )

            # Store metadata
            wrapper._health_check_name = name  # type: ignore
            wrapper._health_check_critical = critical  # type: ignore

            self._checks[name] = wrapper
            return func

        return decorator

    async def check(self, name: str) -> ComponentHealth:
        """Run a single health check by name."""
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
            )

        result = await self._checks[name]()
        self._last_results[name] = result
        return result

    async def check_all(self, parallel: bool = True) -> SystemHealth:
        """
        Run all registered health checks.

        Args:
            parallel: Run checks in parallel if True

        Returns:
            SystemHealth with aggregated status
        """
        if parallel:
            # Run all checks concurrently
            tasks = [check() for check in self._checks.values()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            components = []
            for result in results:
                if isinstance(result, Exception):
                    components.append(
                        ComponentHealth(
                            name="unknown",
                            status=HealthStatus.UNHEALTHY,
                            error=str(result),
                        )
                    )
                else:
                    components.append(result)
        else:
            components = []
            for check in self._checks.values():
                result = await check()
                components.append(result)

        # Store results
        for comp in components:
            self._last_results[comp.name] = comp

        # Determine overall status
        overall_status = self._aggregate_status(components)

        # Calculate uptime
        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        # Get system info
        system_info = self._get_system_info()

        return SystemHealth(
            status=overall_status,
            components=components,
            uptime_seconds=uptime,
            system_info=system_info,
        )

    def _aggregate_status(self, components: list[ComponentHealth]) -> HealthStatus:
        """Aggregate component statuses into overall status."""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components]

        # Check for critical failures
        for comp in components:
            check_func = self._checks.get(comp.name)
            if check_func and getattr(check_func, "_health_check_critical", False):
                if comp.status == HealthStatus.UNHEALTHY:
                    return HealthStatus.UNHEALTHY

        # Aggregate
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.DEGRADED
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    def _get_system_info(self) -> dict[str, Any]:
        """Get system resource information."""
        base_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "process": {"pid": os.getpid()},
        }

        if not HAS_PSUTIL:
            base_info["note"] = "psutil not installed, detailed metrics unavailable"
            return base_info

        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            cpu_percent = psutil.cpu_percent(interval=0.1)

            base_info.update(
                {
                    "cpu_percent": cpu_percent,
                    "memory": {
                        "total_gb": round(memory.total / (1024**3), 2),
                        "available_gb": round(memory.available / (1024**3), 2),
                        "percent_used": memory.percent,
                    },
                    "disk": {
                        "total_gb": round(disk.total / (1024**3), 2),
                        "free_gb": round(disk.free / (1024**3), 2),
                        "percent_used": round((disk.used / disk.total) * 100, 1),
                    },
                    "process": {
                        "pid": os.getpid(),
                        "memory_mb": round(psutil.Process().memory_info().rss / (1024**2), 2),
                    },
                }
            )
            return base_info
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            base_info["error"] = str(e)
            return base_info

    def get_last_result(self, name: str) -> ComponentHealth | None:
        """Get last check result for a component."""
        return self._last_results.get(name)

    def get_all_last_results(self) -> dict[str, ComponentHealth]:
        """Get all last check results."""
        return self._last_results.copy()

    def list_checks(self) -> list[str]:
        """List all registered check names."""
        return list(self._checks.keys())


# Convenience function
def health_check(
    name: str,
    timeout: float = 5.0,
    critical: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to register a health check with the global instance.

    Usage:
        @health_check("database", critical=True)
        async def check_database():
            # Check database
            return True, "Connected"
    """
    return HealthCheck.get_instance().register(name, timeout, critical)


# =============================================================================
# Built-in Health Checks
# =============================================================================


def register_default_checks(health: HealthCheck | None = None) -> None:
    """Register default health checks."""
    if health is None:
        health = HealthCheck.get_instance()

    @health.register("system_resources", timeout=2.0)
    def check_system_resources():
        """Check system resource availability."""
        if not HAS_PSUTIL:
            return True, "psutil not installed, skipping resource check", {}

        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            issues = []

            # Check memory
            if memory.percent > 90:
                issues.append(f"Memory usage critical: {memory.percent}%")
            elif memory.percent > 80:
                issues.append(f"Memory usage high: {memory.percent}%")

            # Check disk
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 95:
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > 85:
                issues.append(f"Disk usage high: {disk_percent:.1f}%")

            if issues:
                return (
                    False,
                    "; ".join(issues),
                    {
                        "memory_percent": memory.percent,
                        "disk_percent": round(disk_percent, 1),
                    },
                )

            return (
                True,
                "Resources OK",
                {
                    "memory_percent": memory.percent,
                    "disk_percent": round(disk_percent, 1),
                },
            )

        except Exception as e:
            return False, f"Failed to check resources: {e}", {}


# =============================================================================
# Liveness and Readiness Probes
# =============================================================================


class ProbeStatus(Enum):
    """Kubernetes-style probe status."""

    LIVE = "live"
    READY = "ready"
    NOT_LIVE = "not_live"
    NOT_READY = "not_ready"


@dataclass
class ProbeResult:
    """Result of a probe check."""

    status: ProbeStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        return self.status in (ProbeStatus.LIVE, ProbeStatus.READY)

    @property
    def is_ready(self) -> bool:
        return self.status == ProbeStatus.READY


class ProbeManager:
    """
    Manages liveness and readiness probes.

    Liveness: Is the application alive and should not be restarted?
    Readiness: Is the application ready to receive traffic?
    """

    def __init__(self, health: HealthCheck | None = None):
        self.health = health or HealthCheck.get_instance()
        self._ready = False
        self._live = True
        self._ready_checks: list[str] = []
        self._live_checks: list[str] = []

    def set_ready(self, ready: bool = True) -> None:
        """Set application readiness."""
        self._ready = ready
        logger.info(f"Application readiness set to: {ready}")

    def set_live(self, live: bool = True) -> None:
        """Set application liveness."""
        self._live = live
        logger.info(f"Application liveness set to: {live}")

    def add_ready_check(self, check_name: str) -> None:
        """Add a health check to readiness probe."""
        self._ready_checks.append(check_name)

    def add_live_check(self, check_name: str) -> None:
        """Add a health check to liveness probe."""
        self._live_checks.append(check_name)

    def is_ready(self) -> bool:
        """Synchronous readiness check (uses cached state)."""
        return self._ready and self._live

    def is_live(self) -> bool:
        """Synchronous liveness check (uses cached state)."""
        return self._live

    async def check_liveness(self) -> ProbeResult:
        """
        Check if application is live.

        Returns success if:
        - Manual liveness flag is True
        - All liveness checks pass
        """
        if not self._live:
            return ProbeResult(
                status=ProbeStatus.NOT_LIVE,
                message="Application marked as not live",
            )

        # Run liveness checks
        if self._live_checks:
            for check_name in self._live_checks:
                result = await self.health.check(check_name)
                if result.status == HealthStatus.UNHEALTHY:
                    return ProbeResult(
                        status=ProbeStatus.NOT_LIVE,
                        message=f"Liveness check failed: {check_name}",
                        details=result.to_dict(),
                    )

        return ProbeResult(
            status=ProbeStatus.LIVE,
            message="Application is live",
        )

    async def check_readiness(self) -> ProbeResult:
        """
        Check if application is ready.

        Returns success if:
        - Application is live
        - Manual readiness flag is True
        - All readiness checks pass
        """
        # Must be live first
        liveness = await self.check_liveness()
        if not liveness.is_live:
            return ProbeResult(
                status=ProbeStatus.NOT_READY,
                message="Application is not live",
                details=liveness.details,
            )

        if not self._ready:
            return ProbeResult(
                status=ProbeStatus.NOT_READY,
                message="Application marked as not ready",
            )

        # Run readiness checks
        if self._ready_checks:
            for check_name in self._ready_checks:
                result = await self.health.check(check_name)
                if result.status != HealthStatus.HEALTHY:
                    return ProbeResult(
                        status=ProbeStatus.NOT_READY,
                        message=f"Readiness check failed: {check_name}",
                        details=result.to_dict(),
                    )

        return ProbeResult(
            status=ProbeStatus.READY,
            message="Application is ready",
        )


# Global probe manager (legacy - prefer using AetherRuntime.probe_manager)
_probe_manager: ProbeManager | None = None


def get_probe_manager() -> ProbeManager:
    """
    Get global probe manager.

    Note: For new code, prefer using `get_runtime().probe_manager` for
    centralized lifecycle management.
    """
    global _probe_manager
    if _probe_manager is None:
        # Try to get from runtime if available
        try:
            from aether.core.runtime import get_runtime

            return get_runtime().probe_manager
        except ImportError:
            _probe_manager = ProbeManager()
    return _probe_manager
