"""
AETHER Core Module

Production-grade infrastructure for error handling, logging, and resilience.
"""

from aether.core.exceptions import (
    AetherError,
    AgentError,
    AudioProcessingError,
    CircuitBreakerOpenError,
    ConfigurationError,
    MIDIError,
    PipelineError,
    ProviderError,
    QAError,
    RenderingError,
    RetryExhaustedError,
    StorageError,
    ValidationError,
)
from aether.core.health import (
    ComponentHealth,
    HealthCheck,
    HealthStatus,
    SystemHealth,
    health_check,
)
from aether.core.logging import (
    LogContext,
    configure_logging,
    get_logger,
    log_operation,
    log_performance,
)
from aether.core.metrics import (
    Counter,
    Gauge,
    MetricsCollector,
    Timer,
    get_metrics,
)
from aether.core.resilience import (
    CircuitBreaker,
    RetryPolicy,
    circuit_breaker,
    fallback,
    retry,
    timeout,
)
from aether.core.runtime import (
    AetherRuntime,
    RuntimeConfig,
    get_runtime,
)

__all__ = [
    # Exceptions
    "AetherError",
    "ConfigurationError",
    "ProviderError",
    "PipelineError",
    "AgentError",
    "ValidationError",
    "AudioProcessingError",
    "MIDIError",
    "RenderingError",
    "QAError",
    "StorageError",
    "RetryExhaustedError",
    "CircuitBreakerOpenError",
    # Logging
    "get_logger",
    "configure_logging",
    "LogContext",
    "log_operation",
    "log_performance",
    # Resilience
    "retry",
    "circuit_breaker",
    "timeout",
    "fallback",
    "CircuitBreaker",
    "RetryPolicy",
    # Health
    "HealthCheck",
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "health_check",
    # Metrics
    "MetricsCollector",
    "Timer",
    "Counter",
    "Gauge",
    "get_metrics",
    # Runtime
    "AetherRuntime",
    "RuntimeConfig",
    "get_runtime",
]
