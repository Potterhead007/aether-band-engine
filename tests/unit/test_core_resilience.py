"""
Unit tests for AETHER resilience patterns.
"""

import asyncio
import pytest
import time

from aether.core.resilience import (
    retry,
    RetryPolicy,
    BackoffStrategy,
    circuit_breaker,
    CircuitBreaker,
    CircuitState,
    timeout,
    fallback,
    Bulkhead,
    resilient,
)
from aether.core.exceptions import (
    RetryExhaustedError,
    CircuitBreakerOpenError,
)


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_default_values(self):
        """Test default policy values."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.backoff == BackoffStrategy.EXPONENTIAL
        assert policy.jitter is True

    def test_constant_backoff(self):
        """Test constant backoff calculation."""
        policy = RetryPolicy(
            base_delay=2.0,
            backoff=BackoffStrategy.CONSTANT,
            jitter=False,
        )
        assert policy.calculate_delay(1) == 2.0
        assert policy.calculate_delay(2) == 2.0
        assert policy.calculate_delay(5) == 2.0

    def test_linear_backoff(self):
        """Test linear backoff calculation."""
        policy = RetryPolicy(
            base_delay=1.0,
            backoff=BackoffStrategy.LINEAR,
            jitter=False,
        )
        assert policy.calculate_delay(1) == 1.0
        assert policy.calculate_delay(2) == 2.0
        assert policy.calculate_delay(3) == 3.0

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy(
            base_delay=1.0,
            backoff=BackoffStrategy.EXPONENTIAL,
            jitter=False,
        )
        assert policy.calculate_delay(1) == 1.0
        assert policy.calculate_delay(2) == 2.0
        assert policy.calculate_delay(3) == 4.0
        assert policy.calculate_delay(4) == 8.0

    def test_max_delay_cap(self):
        """Test max delay cap."""
        policy = RetryPolicy(
            base_delay=10.0,
            max_delay=30.0,
            backoff=BackoffStrategy.EXPONENTIAL,
            jitter=False,
        )
        assert policy.calculate_delay(1) == 10.0
        assert policy.calculate_delay(2) == 20.0
        assert policy.calculate_delay(3) == 30.0  # Capped
        assert policy.calculate_delay(4) == 30.0  # Still capped

    def test_jitter(self):
        """Test jitter adds variance."""
        policy = RetryPolicy(
            base_delay=10.0,
            backoff=BackoffStrategy.CONSTANT,
            jitter=True,
            jitter_factor=0.2,
        )
        delays = [policy.calculate_delay(1) for _ in range(100)]
        # With 20% jitter on 10.0, range should be 8.0-12.0
        assert any(d != 10.0 for d in delays)  # At least some variance
        assert all(6.0 <= d <= 14.0 for d in delays)  # Within bounds

    def test_should_retry_retryable(self):
        """Test retryable exception check."""
        policy = RetryPolicy(
            retryable_exceptions=(ValueError, TypeError),
        )
        assert policy.should_retry(ValueError("test"))
        assert policy.should_retry(TypeError("test"))
        assert not policy.should_retry(KeyError("test"))

    def test_should_retry_non_retryable(self):
        """Test non-retryable exception check."""
        policy = RetryPolicy(
            retryable_exceptions=(Exception,),
            non_retryable_exceptions=(ValueError,),
        )
        assert policy.should_retry(TypeError("test"))
        assert not policy.should_retry(ValueError("test"))


class TestRetryDecorator:
    """Tests for retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_success_first_try(self):
        """Test successful call on first try."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await succeed()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test success after initial failures."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await fail_then_succeed()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry exhaustion."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await always_fail()

        assert call_count == 3
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.cause, ValueError)

    @pytest.mark.asyncio
    async def test_retry_non_retryable(self):
        """Test non-retryable exception."""
        call_count = 0

        @retry(
            max_attempts=3,
            base_delay=0.01,
            non_retryable_exceptions=(KeyError,),
        )
        async def fail_with_key_error():
            nonlocal call_count
            call_count += 1
            raise KeyError("Not retryable")

        with pytest.raises(KeyError):
            await fail_with_key_error()

        assert call_count == 1  # Only one attempt

    @pytest.mark.asyncio
    async def test_retry_on_callback(self):
        """Test on_retry callback."""
        retry_exceptions = []

        def on_retry(exc, attempt):
            retry_exceptions.append((exc, attempt))

        @retry(max_attempts=3, base_delay=0.01, on_retry=on_retry)
        async def fail_twice():
            if len(retry_exceptions) < 2:
                raise ValueError("Fail")
            return "success"

        await fail_twice()
        assert len(retry_exceptions) == 2
        assert all(isinstance(e[0], ValueError) for e in retry_exceptions)

    def test_retry_sync_function(self):
        """Test retry on sync function."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def sync_fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "success"

        result = sync_fail_then_succeed()
        assert result == "success"
        assert call_count == 2


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self):
        """Test initial state is closed."""
        cb = CircuitBreaker("test_closed", failure_threshold=5)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker("test_opens", failure_threshold=3)

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    def test_success_resets_count(self):
        """Test success resets failure count."""
        cb = CircuitBreaker("test_reset", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Reset
        cb.record_failure()

        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        """Test transition to half-open after timeout."""
        cb = CircuitBreaker("test_half_open", failure_threshold=1, timeout=0.1)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        """Test success in half-open closes circuit."""
        cb = CircuitBreaker(
            "test_half_close",
            failure_threshold=1,
            success_threshold=1,
            timeout=0.1,
        )

        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """Test failure in half-open reopens circuit."""
        cb = CircuitBreaker("test_half_reopen", failure_threshold=1, timeout=0.1)

        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_excluded_exceptions(self):
        """Test excluded exceptions don't count."""
        cb = CircuitBreaker(
            "test_excluded",
            failure_threshold=1,
            excluded_exceptions=(ValueError,),
        )

        cb.record_failure(ValueError("excluded"))
        assert cb.state == CircuitState.CLOSED

        cb.record_failure(TypeError("not excluded"))
        assert cb.state == CircuitState.OPEN

    def test_manual_reset(self):
        """Test manual circuit reset."""
        cb = CircuitBreaker("test_manual_reset", failure_threshold=1)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_stats(self):
        """Test getting circuit stats."""
        cb = CircuitBreaker("test_stats", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()

        stats = cb.get_stats()
        assert stats["name"] == "test_stats"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator."""
        cb = CircuitBreaker("test_decorator", failure_threshold=2)
        call_count = 0

        @cb
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Fail")

        # First two calls fail and open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await failing_function()

        assert call_count == 2

        # Third call should be rejected by circuit
        with pytest.raises(CircuitBreakerOpenError):
            await failing_function()

        assert call_count == 2  # Not called


class TestTimeout:
    """Tests for timeout decorator."""

    @pytest.mark.asyncio
    async def test_timeout_not_exceeded(self):
        """Test function completes within timeout."""

        @timeout(1.0)
        async def fast_function():
            await asyncio.sleep(0.1)
            return "done"

        result = await fast_function()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_timeout_exceeded(self):
        """Test function exceeds timeout."""
        from aether.core.resilience import TimeoutError

        @timeout(0.1)
        async def slow_function():
            await asyncio.sleep(1.0)
            return "done"

        with pytest.raises(TimeoutError) as exc_info:
            await slow_function()

        assert "timed out" in str(exc_info.value)


class TestFallback:
    """Tests for fallback decorator."""

    @pytest.mark.asyncio
    async def test_fallback_not_triggered(self):
        """Test fallback not triggered on success."""

        @fallback(fallback_value="fallback")
        async def succeed():
            return "success"

        result = await succeed()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_fallback_value(self):
        """Test fallback value on failure."""

        @fallback(fallback_value="fallback")
        async def fail():
            raise ValueError("Fail")

        result = await fail()
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_fallback_function(self):
        """Test fallback function on failure."""

        async def get_fallback():
            return "from_fallback"

        @fallback(fallback_func=get_fallback)
        async def fail():
            raise ValueError("Fail")

        result = await fail()
        assert result == "from_fallback"

    @pytest.mark.asyncio
    async def test_fallback_specific_exceptions(self):
        """Test fallback only on specific exceptions."""

        @fallback(fallback_value="fallback", exceptions=(ValueError,))
        async def fail_with_key_error():
            raise KeyError("Not caught")

        with pytest.raises(KeyError):
            await fail_with_key_error()

    def test_fallback_sync(self):
        """Test fallback on sync function."""

        @fallback(fallback_value=[])
        def fail_sync():
            raise ValueError("Fail")

        result = fail_sync()
        assert result == []


class TestBulkhead:
    """Tests for Bulkhead pattern."""

    @pytest.mark.asyncio
    async def test_bulkhead_limits_concurrency(self):
        """Test bulkhead limits concurrent executions."""
        bulkhead = Bulkhead("test_bulkhead", max_concurrent=2, max_wait=1.0)
        active = 0
        max_active = 0

        @bulkhead
        async def slow_function():
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.1)
            active -= 1
            return "done"

        # Start 5 concurrent tasks
        tasks = [slow_function() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Max concurrency should be limited to 2
        assert max_active == 2

    def test_bulkhead_stats(self):
        """Test bulkhead statistics."""
        bulkhead = Bulkhead("test_stats", max_concurrent=10)
        stats = bulkhead.get_stats()
        assert stats["name"] == "test_stats"
        assert stats["max_concurrent"] == 10
        assert stats["active"] == 0
        assert stats["available"] == 10


class TestResilient:
    """Tests for combined resilient decorator."""

    @pytest.mark.asyncio
    async def test_resilient_success(self):
        """Test resilient decorator on success."""

        @resilient("test_success", max_retries=2)
        async def succeed():
            return "success"

        result = await succeed()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_resilient_retry(self):
        """Test resilient decorator with retry."""
        call_count = 0

        @resilient("test_retry_combined", max_retries=3, retry_delay=0.01)
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "success"

        result = await fail_then_succeed()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_resilient_fallback(self):
        """Test resilient decorator with fallback."""

        @resilient(
            "test_fallback_combined",
            max_retries=2,
            retry_delay=0.01,
            fallback_value="fallback",
        )
        async def always_fail():
            raise ValueError("Always fails")

        result = await always_fail()
        assert result == "fallback"
