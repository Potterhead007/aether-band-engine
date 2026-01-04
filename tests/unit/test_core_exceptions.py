"""
Unit tests for AETHER exception hierarchy.
"""

import pytest
from datetime import datetime

from aether.core.exceptions import (
    AetherError,
    ConfigurationError,
    MissingConfigError,
    InvalidConfigError,
    ProviderError,
    ProviderNotFoundError,
    ProviderInitializationError,
    RateLimitError,
    PipelineError,
    PipelineStageError,
    AgentError,
    AgentNotFoundError,
    ValidationError,
    AudioProcessingError,
    MIDIError,
    RenderingError,
    QAError,
    StorageError,
    RetryExhaustedError,
    CircuitBreakerOpenError,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    get_exception_class,
)


class TestErrorContext:
    """Tests for ErrorContext."""

    def test_default_values(self):
        """Test default context values."""
        ctx = ErrorContext()
        assert ctx.trace_id is None
        assert ctx.span_id is None
        assert ctx.component is None
        assert ctx.operation is None
        assert ctx.input_data is None
        assert ctx.metadata == {}
        assert ctx.stack_trace is None
        assert isinstance(ctx.timestamp, datetime)

    def test_to_dict(self):
        """Test context serialization."""
        ctx = ErrorContext(
            trace_id="abc123",
            component="pipeline",
            operation="generate",
            metadata={"key": "value"},
        )
        d = ctx.to_dict()
        assert d["trace_id"] == "abc123"
        assert d["component"] == "pipeline"
        assert d["operation"] == "generate"
        assert d["metadata"] == {"key": "value"}
        assert "timestamp" in d


class TestAetherError:
    """Tests for base AetherError."""

    def test_basic_error(self):
        """Test basic error creation."""
        err = AetherError("Something went wrong")
        assert str(err) == "[AETHER_ERROR] Something went wrong"
        assert err.message == "Something went wrong"
        assert err.code == "AETHER_ERROR"
        assert err.severity == ErrorSeverity.ERROR
        assert err.category == ErrorCategory.INTERNAL

    def test_custom_code_and_severity(self):
        """Test custom error code and severity."""
        err = AetherError(
            "Custom error",
            code="CUSTOM_001",
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.VALIDATION,
        )
        assert err.code == "CUSTOM_001"
        assert err.severity == ErrorSeverity.WARNING
        assert err.category == ErrorCategory.VALIDATION

    def test_recovery_hints(self):
        """Test recovery hints."""
        err = AetherError(
            "Error with hints",
            recovery_hints=["Try X", "Try Y"],
        )
        assert err.recovery_hints == ["Try X", "Try Y"]
        assert "Try X" in str(err)

    def test_chained_exception(self):
        """Test exception chaining."""
        original = ValueError("Original error")
        err = AetherError("Wrapped error", cause=original)
        assert err.cause is original
        assert err.__cause__ is original

    def test_with_context(self):
        """Test adding context to error."""
        err = AetherError("Error").with_context(
            trace_id="abc123",
            component="test",
            custom_field="custom_value",
        )
        assert err.context.trace_id == "abc123"
        assert err.context.component == "test"
        assert err.context.metadata["custom_field"] == "custom_value"

    def test_with_hint(self):
        """Test adding hints to error."""
        err = AetherError("Error").with_hint("Try this").with_hint("Or that")
        assert len(err.recovery_hints) == 2

    def test_to_dict(self):
        """Test error serialization."""
        err = AetherError(
            "Test error",
            code="TEST_001",
            details={"key": "value"},
        )
        d = err.to_dict()
        assert d["error_type"] == "AetherError"
        assert d["code"] == "TEST_001"
        assert d["message"] == "Test error"
        assert d["details"]["key"] == "value"


class TestConfigurationErrors:
    """Tests for configuration errors."""

    def test_missing_config_error(self):
        """Test missing config error."""
        err = MissingConfigError("api_key")
        assert "api_key" in err.message
        assert err.code == "CONFIG_MISSING"
        assert err.category == ErrorCategory.CONFIGURATION
        assert len(err.recovery_hints) > 0
        assert "api_key" in str(err.recovery_hints)

    def test_invalid_config_error(self):
        """Test invalid config error."""
        err = InvalidConfigError(
            "Invalid port number",
            config_key="port",
            config_value=-1,
            expected_type=int,
        )
        assert err.code == "CONFIG_INVALID"
        assert err.details["config_key"] == "port"
        assert err.details["config_value"] == -1


class TestProviderErrors:
    """Tests for provider errors."""

    def test_provider_not_found(self):
        """Test provider not found error."""
        err = ProviderNotFoundError("llm", "gpt4")
        assert "gpt4" in err.message
        assert err.code == "PROVIDER_NOT_FOUND"
        assert err.details["provider_type"] == "llm"
        assert err.details["provider_name"] == "gpt4"

    def test_provider_initialization_error(self):
        """Test provider init error severity."""
        err = ProviderInitializationError("Failed to connect")
        assert err.severity == ErrorSeverity.CRITICAL

    def test_rate_limit_error(self):
        """Test rate limit error."""
        err = RateLimitError(retry_after=60)
        assert err.code == "RATE_LIMIT_EXCEEDED"
        assert err.severity == ErrorSeverity.WARNING
        assert err.retry_after == 60
        assert "60" in str(err.recovery_hints)


class TestPipelineErrors:
    """Tests for pipeline errors."""

    def test_pipeline_stage_error(self):
        """Test pipeline stage error."""
        err = PipelineStageError(
            "Stage failed",
            pipeline_id="pipe-123",
            stage="mixing",
        )
        assert err.code == "PIPELINE_STAGE_ERROR"
        assert err.details["pipeline_id"] == "pipe-123"
        assert err.details["stage"] == "mixing"


class TestAgentErrors:
    """Tests for agent errors."""

    def test_agent_not_found(self):
        """Test agent not found error."""
        err = AgentNotFoundError(
            "Agent not registered",
            agent_type="custom",
            agent_name="CustomAgent",
        )
        assert err.code == "AGENT_NOT_FOUND"
        assert err.details["agent_type"] == "custom"


class TestValidationErrors:
    """Tests for validation errors."""

    def test_validation_error(self):
        """Test validation error."""
        err = ValidationError(
            "Value out of range",
            field="tempo",
            value=300,
            constraint="max=200",
        )
        assert err.code == "VALIDATION_ERROR"
        assert err.severity == ErrorSeverity.WARNING
        assert err.details["field"] == "tempo"
        assert err.details["value"] == 300


class TestResilienceErrors:
    """Tests for resilience errors."""

    def test_retry_exhausted_error(self):
        """Test retry exhausted error."""
        original = ConnectionError("Connection failed")
        err = RetryExhaustedError(
            "All retries failed",
            attempts=5,
            last_error=original,
        )
        assert err.code == "RETRY_EXHAUSTED"
        assert err.attempts == 5
        assert err.cause is original
        assert "5" in str(err.recovery_hints)

    def test_circuit_breaker_open_error(self):
        """Test circuit breaker open error."""
        err = CircuitBreakerOpenError(
            circuit_name="external_api",
            reset_after=30.0,
        )
        assert err.code == "CIRCUIT_BREAKER_OPEN"
        assert err.severity == ErrorSeverity.WARNING
        assert err.circuit_name == "external_api"
        assert err.reset_after == 30.0


class TestErrorRegistry:
    """Tests for error registry."""

    def test_get_exception_class(self):
        """Test getting exception class by code."""
        assert get_exception_class("CONFIG_MISSING") is MissingConfigError
        assert get_exception_class("RATE_LIMIT_EXCEEDED") is RateLimitError
        assert get_exception_class("UNKNOWN_CODE") is AetherError

    def test_all_errors_have_default_code(self):
        """Test all error classes have default codes."""
        error_classes = [
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
        ]
        for cls in error_classes:
            assert hasattr(cls, "default_code")
            assert cls.default_code is not None
