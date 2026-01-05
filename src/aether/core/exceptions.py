"""
AETHER Exception Hierarchy

Production-grade exception system with rich context, error codes,
and recovery hints for debugging and monitoring.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification and routing."""

    CONFIGURATION = "configuration"
    PROVIDER = "provider"
    PIPELINE = "pipeline"
    AGENT = "agent"
    VALIDATION = "validation"
    AUDIO = "audio"
    MIDI = "midi"
    RENDERING = "rendering"
    QA = "qa"
    STORAGE = "storage"
    NETWORK = "network"
    RESOURCE = "resource"
    INTERNAL = "internal"


@dataclass
class ErrorContext:
    """Rich context for error tracking and debugging."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    input_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "component": self.component,
            "operation": self.operation,
            "input_data": self.input_data,
            "metadata": self.metadata,
            "stack_trace": self.stack_trace,
        }


class AetherError(Exception):
    """
    Base exception for all AETHER errors.

    Provides:
    - Error codes for programmatic handling
    - Severity levels for monitoring
    - Recovery hints for debugging
    - Rich context for tracing
    - Chained exception support
    """

    # Class-level defaults
    default_code: str = "AETHER_ERROR"
    default_severity: ErrorSeverity = ErrorSeverity.ERROR
    default_category: ErrorCategory = ErrorCategory.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        severity: ErrorSeverity | None = None,
        category: ErrorCategory | None = None,
        context: ErrorContext | None = None,
        recovery_hints: list[str] | None = None,
        cause: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.severity = severity or self.default_severity
        self.category = category or self.default_category
        self.context = context or ErrorContext()
        self.recovery_hints = recovery_hints or []
        self.cause = cause
        self.details = details or {}

        # Capture stack trace if not provided
        if self.context.stack_trace is None:
            self.context.stack_trace = traceback.format_exc()

        # Chain exception
        if cause:
            self.__cause__ = cause

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.recovery_hints:
            parts.append(f"Recovery hints: {self.recovery_hints}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.code!r}, "
            f"message={self.message!r}, "
            f"severity={self.severity.value!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context.to_dict(),
            "recovery_hints": self.recovery_hints,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }

    def with_context(self, **kwargs) -> AetherError:
        """Add context to existing error."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.metadata[key] = value
        return self

    def with_hint(self, hint: str) -> AetherError:
        """Add a recovery hint."""
        self.recovery_hints.append(hint)
        return self


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(AetherError):
    """Configuration-related errors."""

    default_code = "CONFIG_ERROR"
    default_category = ErrorCategory.CONFIGURATION

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Any | None = None,
        expected_type: type | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.details["config_key"] = config_key
        self.details["config_value"] = config_value
        self.details["expected_type"] = str(expected_type) if expected_type else None


class MissingConfigError(ConfigurationError):
    """Required configuration value is missing."""

    default_code = "CONFIG_MISSING"

    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            f"Missing required configuration: {config_key}",
            config_key=config_key,
            recovery_hints=[
                f"Set the {config_key} configuration value",
                f"Set environment variable AETHER_{config_key.upper()}",
                "Check your config.yaml file",
            ],
            **kwargs,
        )


class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid."""

    default_code = "CONFIG_INVALID"


# =============================================================================
# Provider Errors
# =============================================================================


class ProviderError(AetherError):
    """Provider-related errors."""

    default_code = "PROVIDER_ERROR"
    default_category = ErrorCategory.PROVIDER

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        provider_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.details["provider_name"] = provider_name
        self.details["provider_type"] = provider_type


class ProviderNotFoundError(ProviderError):
    """Requested provider does not exist."""

    default_code = "PROVIDER_NOT_FOUND"

    def __init__(self, provider_type: str, provider_name: str, **kwargs):
        super().__init__(
            f"Provider not found: {provider_type}/{provider_name}",
            provider_name=provider_name,
            provider_type=provider_type,
            recovery_hints=[
                f"Register a provider for type '{provider_type}'",
                "Check provider configuration",
                "Verify provider dependencies are installed",
            ],
            **kwargs,
        )


class ProviderInitializationError(ProviderError):
    """Provider failed to initialize."""

    default_code = "PROVIDER_INIT_FAILED"
    default_severity = ErrorSeverity.CRITICAL


class ProviderUnavailableError(ProviderError):
    """Provider is temporarily unavailable."""

    default_code = "PROVIDER_UNAVAILABLE"
    default_severity = ErrorSeverity.WARNING


class LLMProviderError(ProviderError):
    """LLM provider specific errors."""

    default_code = "LLM_PROVIDER_ERROR"


class RateLimitError(LLMProviderError):
    """Rate limit exceeded."""

    default_code = "RATE_LIMIT_EXCEEDED"
    default_severity = ErrorSeverity.WARNING

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            recovery_hints=[
                (
                    f"Wait {retry_after} seconds before retrying"
                    if retry_after
                    else "Wait before retrying"
                ),
                "Reduce request frequency",
                "Consider upgrading API plan",
            ],
            **kwargs,
        )
        self.retry_after = retry_after
        self.details["retry_after"] = retry_after


# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(AetherError):
    """Pipeline execution errors."""

    default_code = "PIPELINE_ERROR"
    default_category = ErrorCategory.PIPELINE

    def __init__(
        self,
        message: str,
        pipeline_id: Optional[str] = None,
        stage: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.details["pipeline_id"] = pipeline_id
        self.details["stage"] = stage


class PipelineStageError(PipelineError):
    """Error in a specific pipeline stage."""

    default_code = "PIPELINE_STAGE_ERROR"


class PipelineTimeoutError(PipelineError):
    """Pipeline execution timed out."""

    default_code = "PIPELINE_TIMEOUT"


class PipelineAbortedError(PipelineError):
    """Pipeline was aborted."""

    default_code = "PIPELINE_ABORTED"


# =============================================================================
# Agent Errors
# =============================================================================


class AgentError(AetherError):
    """Agent execution errors."""

    default_code = "AGENT_ERROR"
    default_category = ErrorCategory.AGENT

    def __init__(
        self,
        message: str,
        agent_type: Optional[str] = None,
        agent_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.details["agent_type"] = agent_type
        self.details["agent_name"] = agent_name


class AgentNotFoundError(AgentError):
    """Agent type not registered."""

    default_code = "AGENT_NOT_FOUND"


class AgentExecutionError(AgentError):
    """Agent failed during execution."""

    default_code = "AGENT_EXECUTION_ERROR"


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""

    default_code = "AGENT_TIMEOUT"


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(AetherError):
    """Data validation errors."""

    default_code = "VALIDATION_ERROR"
    default_category = ErrorCategory.VALIDATION
    default_severity = ErrorSeverity.WARNING

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any | None = None,
        constraint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.details["field"] = field
        self.details["value"] = value
        self.details["constraint"] = constraint


class SchemaValidationError(ValidationError):
    """Schema validation failed."""

    default_code = "SCHEMA_VALIDATION_ERROR"


class RangeValidationError(ValidationError):
    """Value out of allowed range."""

    default_code = "RANGE_VALIDATION_ERROR"


# =============================================================================
# Audio Processing Errors
# =============================================================================


class AudioProcessingError(AetherError):
    """Audio processing errors."""

    default_code = "AUDIO_ERROR"
    default_category = ErrorCategory.AUDIO

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.details["operation"] = operation
        self.details["sample_rate"] = sample_rate
        self.details["channels"] = channels


class AudioFileError(AudioProcessingError):
    """Audio file I/O error."""

    default_code = "AUDIO_FILE_ERROR"


class SynthesisError(AudioProcessingError):
    """Audio synthesis error."""

    default_code = "SYNTHESIS_ERROR"


class MixingError(AudioProcessingError):
    """Mixing error."""

    default_code = "MIXING_ERROR"


class MasteringError(AudioProcessingError):
    """Mastering error."""

    default_code = "MASTERING_ERROR"


# =============================================================================
# MIDI Errors
# =============================================================================


class MIDIError(AetherError):
    """MIDI-related errors."""

    default_code = "MIDI_ERROR"
    default_category = ErrorCategory.MIDI


class MIDIFileError(MIDIError):
    """MIDI file I/O error."""

    default_code = "MIDI_FILE_ERROR"


class MIDIGenerationError(MIDIError):
    """MIDI generation error."""

    default_code = "MIDI_GENERATION_ERROR"


# =============================================================================
# Rendering Errors
# =============================================================================


class RenderingError(AetherError):
    """Rendering pipeline errors."""

    default_code = "RENDERING_ERROR"
    default_category = ErrorCategory.RENDERING


class RenderingTimeoutError(RenderingError):
    """Rendering timed out."""

    default_code = "RENDERING_TIMEOUT"


# =============================================================================
# QA Errors
# =============================================================================


class QAError(AetherError):
    """Quality assurance errors."""

    default_code = "QA_ERROR"
    default_category = ErrorCategory.QA


class OriginalityError(QAError):
    """Originality check failed."""

    default_code = "ORIGINALITY_ERROR"


class GenreAuthenticityError(QAError):
    """Genre authenticity check failed."""

    default_code = "GENRE_AUTHENTICITY_ERROR"


class TechnicalQAError(QAError):
    """Technical quality check failed."""

    default_code = "TECHNICAL_QA_ERROR"


# =============================================================================
# Storage Errors
# =============================================================================


class StorageError(AetherError):
    """Storage-related errors."""

    default_code = "STORAGE_ERROR"
    default_category = ErrorCategory.STORAGE


class ArtifactNotFoundError(StorageError):
    """Artifact does not exist."""

    default_code = "ARTIFACT_NOT_FOUND"


class ArtifactCorruptedError(StorageError):
    """Artifact data is corrupted."""

    default_code = "ARTIFACT_CORRUPTED"


# =============================================================================
# Resilience Errors
# =============================================================================


class RetryExhaustedError(AetherError):
    """All retry attempts exhausted."""

    default_code = "RETRY_EXHAUSTED"
    default_category = ErrorCategory.INTERNAL

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception | None = None,
        **kwargs,
    ):
        super().__init__(
            message,
            cause=last_error,
            recovery_hints=[
                f"Operation failed after {attempts} attempts",
                "Check underlying service availability",
                "Review error logs for root cause",
            ],
            **kwargs,
        )
        self.attempts = attempts
        self.details["attempts"] = attempts


class CircuitBreakerOpenError(AetherError):
    """Circuit breaker is open, rejecting requests."""

    default_code = "CIRCUIT_BREAKER_OPEN"
    default_category = ErrorCategory.INTERNAL
    default_severity = ErrorSeverity.WARNING

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        circuit_name: Optional[str] = None,
        reset_after: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            recovery_hints=[
                "Wait for circuit breaker to reset",
                "Check underlying service health",
            ],
            **kwargs,
        )
        self.circuit_name = circuit_name
        self.reset_after = reset_after
        self.details["circuit_name"] = circuit_name
        self.details["reset_after"] = reset_after


# =============================================================================
# Error Registry
# =============================================================================

ERROR_CODES: dict[str, type[AetherError]] = {
    # Base
    "AETHER_ERROR": AetherError,
    # Configuration
    "CONFIG_ERROR": ConfigurationError,
    "CONFIG_MISSING": MissingConfigError,
    "CONFIG_INVALID": InvalidConfigError,
    # Provider
    "PROVIDER_ERROR": ProviderError,
    "PROVIDER_NOT_FOUND": ProviderNotFoundError,
    "PROVIDER_INIT_FAILED": ProviderInitializationError,
    "PROVIDER_UNAVAILABLE": ProviderUnavailableError,
    "LLM_PROVIDER_ERROR": LLMProviderError,
    "RATE_LIMIT_EXCEEDED": RateLimitError,
    # Pipeline
    "PIPELINE_ERROR": PipelineError,
    "PIPELINE_STAGE_ERROR": PipelineStageError,
    "PIPELINE_TIMEOUT": PipelineTimeoutError,
    "PIPELINE_ABORTED": PipelineAbortedError,
    # Agent
    "AGENT_ERROR": AgentError,
    "AGENT_NOT_FOUND": AgentNotFoundError,
    "AGENT_EXECUTION_ERROR": AgentExecutionError,
    "AGENT_TIMEOUT": AgentTimeoutError,
    # Validation
    "VALIDATION_ERROR": ValidationError,
    "SCHEMA_VALIDATION_ERROR": SchemaValidationError,
    "RANGE_VALIDATION_ERROR": RangeValidationError,
    # Audio
    "AUDIO_ERROR": AudioProcessingError,
    "AUDIO_FILE_ERROR": AudioFileError,
    "SYNTHESIS_ERROR": SynthesisError,
    "MIXING_ERROR": MixingError,
    "MASTERING_ERROR": MasteringError,
    # MIDI
    "MIDI_ERROR": MIDIError,
    "MIDI_FILE_ERROR": MIDIFileError,
    "MIDI_GENERATION_ERROR": MIDIGenerationError,
    # Rendering
    "RENDERING_ERROR": RenderingError,
    "RENDERING_TIMEOUT": RenderingTimeoutError,
    # QA
    "QA_ERROR": QAError,
    "ORIGINALITY_ERROR": OriginalityError,
    "GENRE_AUTHENTICITY_ERROR": GenreAuthenticityError,
    "TECHNICAL_QA_ERROR": TechnicalQAError,
    # Storage
    "STORAGE_ERROR": StorageError,
    "ARTIFACT_NOT_FOUND": ArtifactNotFoundError,
    "ARTIFACT_CORRUPTED": ArtifactCorruptedError,
    # Resilience
    "RETRY_EXHAUSTED": RetryExhaustedError,
    "CIRCUIT_BREAKER_OPEN": CircuitBreakerOpenError,
}


def get_exception_class(code: str) -> type[AetherError]:
    """Get exception class by error code."""
    return ERROR_CODES.get(code, AetherError)
