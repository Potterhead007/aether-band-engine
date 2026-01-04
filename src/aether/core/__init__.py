"""
AETHER Core Module

Production-grade infrastructure for error handling, logging, and resilience.
"""

from aether.core.exceptions import (
    AetherError,
    ConfigurationError,
    ProviderError,
    PipelineError,
    AgentError,
    ValidationError,
    AudioProcessingError,
    MIDIError,
    RenderingError,
    QAError,
    StorageError,
    RetryExhaustedError,
    CircuitBreakerOpenError,
)
from aether.core.logging import (
    get_logger,
    configure_logging,
    LogContext,
    log_operation,
    log_performance,
)
from aether.core.resilience import (
    retry,
    circuit_breaker,
    timeout,
    fallback,
    CircuitBreaker,
    RetryPolicy,
)
from aether.core.health import (
    HealthCheck,
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    health_check,
)
from aether.core.metrics import (
    MetricsCollector,
    Timer,
    Counter,
    Gauge,
    get_metrics,
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
]
