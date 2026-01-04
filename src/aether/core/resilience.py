"""
AETHER Resilience Patterns

Production-grade resilience with:
- Retry with exponential backoff
- Circuit breaker pattern
- Timeout handling
- Fallback strategies
- Bulkhead isolation
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union

from aether.core.exceptions import (
    AetherError,
    CircuitBreakerOpenError,
    RetryExhaustedError,
)
from aether.core.logging import get_logger, LogContext

logger = get_logger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Retry Policy
# =============================================================================


class BackoffStrategy(Enum):
    """Backoff strategies for retries."""

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    DECORRELATED_JITTER = "decorrelated_jitter"


@dataclass
class RetryPolicy:
    """
    Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay cap (seconds)
        backoff: Backoff strategy
        jitter: Add random jitter to delays
        retryable_exceptions: Exception types to retry on
        non_retryable_exceptions: Exception types to NOT retry on
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_factor: float = 0.2
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff == BackoffStrategy.CONSTANT:
            delay = self.base_delay
        elif self.backoff == BackoffStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.backoff == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** (attempt - 1))
        elif self.backoff == BackoffStrategy.DECORRELATED_JITTER:
            # AWS-style decorrelated jitter
            delay = min(self.max_delay, random.uniform(self.base_delay, self.base_delay * 3))
        else:
            delay = self.base_delay

        # Apply cap
        delay = min(delay, self.max_delay)

        # Apply jitter
        if self.jitter and self.backoff != BackoffStrategy.DECORRELATED_JITTER:
            jitter_range = delay * self.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        # Check non-retryable first
        if isinstance(exception, self.non_retryable_exceptions):
            return False

        # Check retryable
        return isinstance(exception, self.retryable_exceptions)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[F], F]:
    """
    Decorator for automatic retries with backoff.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay cap
        backoff: Backoff strategy to use
        jitter: Add jitter to delays
        retryable_exceptions: Exceptions to retry on
        non_retryable_exceptions: Exceptions to NOT retry on
        on_retry: Callback called on each retry

    Usage:
        @retry(max_attempts=3, backoff=BackoffStrategy.EXPONENTIAL)
        async def call_external_api():
            ...
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff=backoff,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(e):
                        logger.warning(
                            f"Non-retryable exception in {func.__name__}: {e}",
                            attempt=attempt,
                            error_type=type(e).__name__,
                        )
                        raise

                    if attempt == policy.max_attempts:
                        logger.error(
                            f"Retry exhausted for {func.__name__} after {attempt} attempts",
                            error=str(e),
                        )
                        raise RetryExhaustedError(
                            f"Retry exhausted for {func.__name__}",
                            attempts=attempt,
                            last_error=e,
                        )

                    delay = policy.calculate_delay(attempt)
                    logger.warning(
                        f"Retry {attempt}/{policy.max_attempts} for {func.__name__}, "
                        f"waiting {delay:.2f}s",
                        attempt=attempt,
                        max_attempts=policy.max_attempts,
                        delay=delay,
                        error=str(e),
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    await asyncio.sleep(delay)

            # Should not reach here
            raise RetryExhaustedError(
                f"Retry exhausted for {func.__name__}",
                attempts=policy.max_attempts,
                last_error=last_exception,
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(e):
                        raise

                    if attempt == policy.max_attempts:
                        raise RetryExhaustedError(
                            f"Retry exhausted for {func.__name__}",
                            attempts=attempt,
                            last_error=e,
                        )

                    delay = policy.calculate_delay(attempt)
                    logger.warning(
                        f"Retry {attempt}/{policy.max_attempts} for {func.__name__}",
                        delay=delay,
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

            raise RetryExhaustedError(
                f"Retry exhausted for {func.__name__}",
                attempts=policy.max_attempts,
                last_error=last_exception,
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0  # Seconds before attempting reset
    half_open_max_calls: int = 1


class CircuitBreaker:
    """
    Circuit breaker implementation.

    Prevents cascading failures by failing fast when a service is unhealthy.

    States:
    - CLOSED: Normal operation, counting failures
    - OPEN: Service unhealthy, rejecting all calls
    - HALF_OPEN: Testing if service recovered

    Usage:
        breaker = CircuitBreaker("external_api")

        @breaker
        async def call_api():
            ...

        # Or manually:
        if breaker.allow_request():
            try:
                result = await call_api()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
    """

    # Global registry of circuit breakers
    _registry: Dict[str, "CircuitBreaker"] = {}

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
        half_open_max_calls: int = 1,
        excluded_exceptions: Tuple[Type[Exception], ...] = (),
    ):
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            half_open_max_calls=half_open_max_calls,
        )
        self.excluded_exceptions = excluded_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0

        # Register
        CircuitBreaker._registry[name] = self

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout transition."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
        return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = datetime.utcnow() - self._last_failure_time
        return elapsed.total_seconds() >= self.config.timeout

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        logger.info(
            f"Circuit breaker '{self.name}' transitioning to HALF_OPEN",
            circuit=self.name,
        )
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._success_count = 0

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        logger.warning(
            f"Circuit breaker '{self.name}' OPENED after {self._failure_count} failures",
            circuit=self.name,
            failures=self._failure_count,
        )
        self._state = CircuitState.OPEN
        self._last_failure_time = datetime.utcnow()

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        logger.info(
            f"Circuit breaker '{self.name}' CLOSED after recovery",
            circuit=self.name,
        )
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        state = self.state  # Triggers timeout check

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        # Half-open: allow limited requests
        if self._half_open_calls < self.config.half_open_max_calls:
            self._half_open_calls += 1
            return True

        return False

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed call."""
        # Check if exception is excluded
        if exception and isinstance(exception, self.excluded_exceptions):
            return

        self._last_failure_time = datetime.utcnow()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to_open()
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to_open()

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        logger.info(f"Circuit breaker '{self.name}' manually reset", circuit=self.name)
        self._transition_to_closed()

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": (
                self._last_failure_time.isoformat() if self._last_failure_time else None
            ),
        }

    @classmethod
    def get(cls, name: str) -> Optional["CircuitBreaker"]:
        """Get circuit breaker by name."""
        return cls._registry.get(name)

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {name: cb.get_stats() for name, cb in cls._registry.items()}

    def __call__(self, func: F) -> F:
        """Use as decorator."""

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self.allow_request():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    circuit_name=self.name,
                    reset_after=self.config.timeout,
                )

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not self.allow_request():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    circuit_name=self.name,
                    reset_after=self.config.timeout,
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 30.0,
    excluded_exceptions: Tuple[Type[Exception], ...] = (),
) -> Callable[[F], F]:
    """
    Decorator to apply circuit breaker pattern.

    Args:
        name: Circuit breaker name (for monitoring)
        failure_threshold: Failures before opening
        success_threshold: Successes before closing
        timeout: Seconds before attempting reset
        excluded_exceptions: Exceptions that don't count as failures

    Usage:
        @circuit_breaker("external_api", failure_threshold=3)
        async def call_api():
            ...
    """
    # Get or create circuit breaker
    cb = CircuitBreaker.get(name)
    if cb is None:
        cb = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            excluded_exceptions=excluded_exceptions,
        )

    return cb


# =============================================================================
# Timeout
# =============================================================================


class TimeoutError(AetherError):
    """Operation timed out."""

    default_code = "TIMEOUT"


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Decorator to apply timeout to function execution.

    Args:
        seconds: Maximum execution time

    Usage:
        @timeout(30.0)
        async def slow_operation():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Operation {func.__name__} timed out after {seconds}s",
                    details={"timeout_seconds": seconds},
                )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't easily implement timeout
            # Log warning and execute normally
            logger.warning(
                f"Timeout decorator applied to sync function {func.__name__}, "
                "timeout will not be enforced"
            )
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Fallback
# =============================================================================


def fallback(
    fallback_func: Optional[Callable[..., T]] = None,
    fallback_value: Optional[T] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to provide fallback on failure.

    Args:
        fallback_func: Function to call on failure (receives same args)
        fallback_value: Static value to return on failure
        exceptions: Exception types to catch

    Usage:
        @fallback(fallback_value=[])
        async def get_items():
            ...

        @fallback(fallback_func=get_cached_items)
        async def get_items_from_api():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                logger.warning(
                    f"Fallback triggered for {func.__name__}: {e}",
                    error=str(e),
                )
                if fallback_func is not None:
                    result = fallback_func(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                return fallback_value  # type: ignore

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.warning(
                    f"Fallback triggered for {func.__name__}: {e}",
                    error=str(e),
                )
                if fallback_func is not None:
                    return fallback_func(*args, **kwargs)
                return fallback_value  # type: ignore

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Bulkhead (Concurrency Limiter)
# =============================================================================


class Bulkhead:
    """
    Bulkhead pattern for concurrency isolation.

    Limits concurrent executions to prevent resource exhaustion.

    Usage:
        bulkhead = Bulkhead("external_api", max_concurrent=10)

        @bulkhead
        async def call_api():
            ...
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_wait: float = 30.0,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait = max_wait
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0
        self._waiting = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "active": self._active,
            "available": self.max_concurrent - self._active,
            "waiting": self._waiting,
        }

    def __call__(self, func: F) -> F:
        """Use as decorator."""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            self._waiting += 1
            try:
                acquired = await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.max_wait,
                )
            except asyncio.TimeoutError:
                self._waiting -= 1
                raise TimeoutError(
                    f"Bulkhead '{self.name}' wait timeout exceeded",
                    details={
                        "max_wait": self.max_wait,
                        "waiting": self._waiting,
                    },
                )

            self._waiting -= 1
            self._active += 1
            try:
                return await func(*args, **kwargs)
            finally:
                self._active -= 1
                self._semaphore.release()

        return wrapper  # type: ignore


# =============================================================================
# Combined Resilience
# =============================================================================


def resilient(
    name: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    circuit_failure_threshold: int = 5,
    circuit_timeout: float = 30.0,
    execution_timeout: Optional[float] = None,
    fallback_value: Optional[Any] = None,
) -> Callable[[F], F]:
    """
    Combined resilience decorator applying multiple patterns.

    Applies in order: timeout -> circuit breaker -> retry -> fallback

    Args:
        name: Name for circuit breaker and logging
        max_retries: Maximum retry attempts
        retry_delay: Base delay between retries
        circuit_failure_threshold: Failures before circuit opens
        circuit_timeout: Seconds before circuit reset attempt
        execution_timeout: Maximum execution time (optional)
        fallback_value: Value to return if all else fails (optional)

    Usage:
        @resilient("external_api", max_retries=3, fallback_value={})
        async def call_api():
            ...
    """

    def decorator(func: F) -> F:
        # Build decorator chain from inside out
        wrapped = func

        # 1. Innermost: fallback (if provided)
        if fallback_value is not None:
            wrapped = fallback(fallback_value=fallback_value)(wrapped)

        # 2. Retry
        wrapped = retry(
            max_attempts=max_retries,
            base_delay=retry_delay,
            non_retryable_exceptions=(CircuitBreakerOpenError,),
        )(wrapped)

        # 3. Circuit breaker
        wrapped = circuit_breaker(
            name=name,
            failure_threshold=circuit_failure_threshold,
            timeout=circuit_timeout,
        )(wrapped)

        # 4. Outermost: timeout (if provided)
        if execution_timeout is not None:
            wrapped = timeout(execution_timeout)(wrapped)

        return wrapped  # type: ignore

    return decorator
