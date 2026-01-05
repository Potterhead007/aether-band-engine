"""
AETHER Runtime Context

Unified runtime context that manages all global state and singletons.
Provides centralized lifecycle management, dependency injection, and testability.

Usage:
    # Get the global runtime
    runtime = AetherRuntime.get_instance()

    # Access subsystems
    runtime.health.check_all()
    runtime.metrics.counter("requests").inc()
    runtime.providers.get("llm")

    # For testing - create isolated runtime
    test_runtime = AetherRuntime()
    with test_runtime:
        # All operations use test_runtime
        pass
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether.core.health import HealthCheck, ProbeManager
    from aether.core.metrics import MetricsCollector
    from aether.providers.base import ProviderRegistry


# Context variable for runtime scoping (enables per-task runtime in async)
_runtime_context: ContextVar[AetherRuntime | None] = ContextVar(
    "aether_runtime", default=None
)


@dataclass
class RuntimeConfig:
    """Configuration for the AETHER runtime."""

    # Metrics configuration
    metrics_prefix: str = "aether"

    # Health check configuration
    health_check_timeout: float = 5.0

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "human"  # "human" or "json"

    # Environment
    environment: str = "production"
    version: str = "0.1.0"


class AetherRuntime:
    """
    Unified runtime context for AETHER.

    Consolidates all global singletons into a single manageable context:
    - ProviderRegistry: Plugin provider management
    - HealthCheck: Component health monitoring
    - MetricsCollector: Metrics collection and export
    - ProbeManager: Kubernetes-style liveness/readiness probes
    - Logger cache: Structured logging instances

    This enables:
    - Clean initialization/shutdown lifecycle
    - Easy dependency injection for testing
    - Scoped runtime contexts for parallel operations
    - Clear ownership of global state
    """

    # Global singleton instance
    _global_instance: AetherRuntime | None = None
    _lock = threading.Lock()

    def __init__(self, config: RuntimeConfig | None = None):
        """
        Initialize a new runtime context.

        Args:
            config: Runtime configuration. Uses defaults if not provided.
        """
        self.config = config or RuntimeConfig()
        self._start_time = datetime.utcnow()
        self._initialized = False

        # Lazy-loaded subsystems (created on first access)
        self._health: HealthCheck | None = None
        self._metrics: MetricsCollector | None = None
        self._providers: ProviderRegistry | None = None
        self._probe_manager: ProbeManager | None = None
        self._loggers: dict[str, Any] = {}

        # Subsystem locks for thread-safe lazy initialization
        self._health_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._providers_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._loggers_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> AetherRuntime:
        """
        Get the global runtime instance.

        Creates a new instance with default config if none exists.
        Thread-safe singleton access.
        """
        if cls._global_instance is None:
            with cls._lock:
                # Double-check locking
                if cls._global_instance is None:
                    cls._global_instance = cls()
        return cls._global_instance

    @classmethod
    def set_instance(cls, runtime: AetherRuntime) -> None:
        """
        Set the global runtime instance.

        Useful for testing or custom configuration.
        """
        with cls._lock:
            cls._global_instance = runtime

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the global runtime instance.

        Primarily for testing cleanup.
        """
        with cls._lock:
            if cls._global_instance is not None:
                # Attempt graceful shutdown
                try:
                    asyncio.get_event_loop().run_until_complete(
                        cls._global_instance.shutdown()
                    )
                except Exception:
                    pass
            cls._global_instance = None

    # =========================================================================
    # Subsystem Access (Lazy Initialization)
    # =========================================================================

    @property
    def health(self) -> HealthCheck:
        """Get the health check subsystem."""
        if self._health is None:
            with self._health_lock:
                if self._health is None:
                    from aether.core.health import HealthCheck

                    self._health = HealthCheck()
        return self._health

    @property
    def metrics(self) -> MetricsCollector:
        """Get the metrics collector subsystem."""
        if self._metrics is None:
            with self._metrics_lock:
                if self._metrics is None:
                    from aether.core.metrics import MetricsCollector

                    self._metrics = MetricsCollector(prefix=self.config.metrics_prefix)
        return self._metrics

    @property
    def providers(self) -> ProviderRegistry:
        """Get the provider registry subsystem."""
        if self._providers is None:
            with self._providers_lock:
                if self._providers is None:
                    from aether.providers.base import ProviderRegistry

                    self._providers = ProviderRegistry()
        return self._providers

    @property
    def probe_manager(self) -> ProbeManager:
        """Get the probe manager subsystem."""
        if self._probe_manager is None:
            with self._probe_lock:
                if self._probe_manager is None:
                    from aether.core.health import ProbeManager

                    self._probe_manager = ProbeManager(self.health)
        return self._probe_manager

    def get_logger(self, name: str = "aether") -> Any:
        """
        Get a logger instance from the runtime's logger cache.

        Args:
            name: Logger name, typically module path

        Returns:
            AetherLogger instance
        """
        if name not in self._loggers:
            with self._loggers_lock:
                if name not in self._loggers:
                    from aether.core.logging import get_logger as _get_logger

                    self._loggers[name] = _get_logger(name)
        return self._loggers[name]

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def initialize(self) -> bool:
        """
        Initialize all runtime subsystems.

        Returns:
            True if all subsystems initialized successfully
        """
        if self._initialized:
            return True

        logger = self.get_logger("aether.runtime")
        logger.info("Initializing AETHER runtime...")

        try:
            # Initialize providers
            if self._providers is not None:
                results = await self.providers.initialize_all()
                failed = [name for name, success in results.items() if not success]
                if failed:
                    logger.warning(f"Some providers failed to initialize: {failed}")

            # Register default health checks
            from aether.core.health import register_default_checks

            register_default_checks(self.health)

            # Set up default metrics
            from aether.core.metrics import setup_default_metrics

            setup_default_metrics()

            # Mark as ready
            self.probe_manager.set_ready(True)
            self._initialized = True

            logger.info("AETHER runtime initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Runtime initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown all runtime subsystems."""
        logger = self.get_logger("aether.runtime")
        logger.info("Shutting down AETHER runtime...")

        # Mark as not ready
        if self._probe_manager is not None:
            self._probe_manager.set_ready(False)

        # Shutdown providers
        if self._providers is not None:
            await self._providers.shutdown_all()

        self._initialized = False
        logger.info("AETHER runtime shutdown complete")

    # =========================================================================
    # Context Management
    # =========================================================================

    def __enter__(self) -> AetherRuntime:
        """Enter runtime context (sets this as the current runtime)."""
        self._previous_runtime = _runtime_context.get()
        _runtime_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit runtime context (restores previous runtime)."""
        _runtime_context.set(self._previous_runtime)
        return False

    @classmethod
    @contextmanager
    def scoped(cls, config: RuntimeConfig | None = None):
        """
        Create a scoped runtime context.

        Useful for testing or isolated operations.

        Usage:
            with AetherRuntime.scoped() as runtime:
                # Operations use this isolated runtime
                pass
        """
        runtime = cls(config)
        with runtime:
            yield runtime

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def uptime_seconds(self) -> float:
        """Get runtime uptime in seconds."""
        return (datetime.utcnow() - self._start_time).total_seconds()

    @property
    def is_initialized(self) -> bool:
        """Check if runtime has been initialized."""
        return self._initialized

    def get_status(self) -> dict[str, Any]:
        """Get runtime status summary."""
        return {
            "initialized": self._initialized,
            "uptime_seconds": self.uptime_seconds,
            "environment": self.config.environment,
            "version": self.config.version,
            "subsystems": {
                "health": self._health is not None,
                "metrics": self._metrics is not None,
                "providers": self._providers is not None,
                "probe_manager": self._probe_manager is not None,
            },
        }


# =============================================================================
# Convenience Functions (Backward Compatibility)
# =============================================================================


def get_runtime() -> AetherRuntime:
    """
    Get the current runtime context.

    Returns the context-local runtime if in a scoped context,
    otherwise returns the global singleton.
    """
    runtime = _runtime_context.get()
    if runtime is not None:
        return runtime
    return AetherRuntime.get_instance()
