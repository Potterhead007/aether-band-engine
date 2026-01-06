"""
Unified Retry Logic for AETHER Providers

Provides consistent retry behavior across all provider types with:
- Exponential backoff with jitter
- Configurable max attempts and delays
- Specific exception handling
- Logging integration
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 30.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Multiplier for exponential backoff
    jitter: float = 0.1  # Random jitter factor (0-1)
    retryable_exceptions: tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )

    @classmethod
    def default(cls) -> "RetryConfig":
        """Default retry configuration."""
        return cls()

    @classmethod
    def aggressive(cls) -> "RetryConfig":
        """Aggressive retry for critical operations."""
        return cls(
            max_attempts=5,
            base_delay=0.5,
            max_delay=60.0,
        )

    @classmethod
    def conservative(cls) -> "RetryConfig":
        """Conservative retry for rate-limited APIs."""
        return cls(
            max_attempts=3,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
        )


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


def compute_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Compute delay for given attempt using exponential backoff with jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Exponential backoff
    delay = config.base_delay * (config.exponential_base ** attempt)

    # Cap at max delay
    delay = min(delay, config.max_delay)

    # Add jitter (± jitter%)
    jitter_range = delay * config.jitter
    delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def retry(
    config: Optional[RetryConfig] = None,
    retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding retry logic to async functions.

    Args:
        config: Retry configuration (uses default if not provided)
        retryable_exceptions: Override retryable exceptions from config
        on_retry: Callback called before each retry (attempt, exception, delay)

    Returns:
        Decorated function with retry logic

    Usage:
        @retry()
        async def call_api():
            ...

        @retry(config=RetryConfig.aggressive())
        async def critical_operation():
            ...

        @retry(retryable_exceptions=(RateLimitError, TimeoutError))
        async def rate_limited_api():
            ...
    """
    if config is None:
        config = RetryConfig.default()

    exceptions = retryable_exceptions or config.retryable_exceptions

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    attempts_left = config.max_attempts - attempt - 1

                    if attempts_left > 0:
                        delay = compute_delay(attempt, config)

                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__}: "
                            f"{type(e).__name__}: {e}. Retrying in {delay:.2f}s..."
                        )

                        if on_retry:
                            on_retry(attempt + 1, e, delay)

                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} retry attempts exhausted for {func.__name__}: "
                            f"{type(e).__name__}: {e}"
                        )

                except Exception as e:
                    # Non-retryable exception - raise immediately
                    logger.error(
                        f"Non-retryable exception in {func.__name__}: {type(e).__name__}: {e}"
                    )
                    raise

            # All retries exhausted
            raise RetryExhaustedError(
                f"All {config.max_attempts} retry attempts exhausted for {func.__name__}",
                attempts=config.max_attempts,
                last_exception=last_exception,
            )

        return wrapper

    return decorator


def retry_sync(
    config: Optional[RetryConfig] = None,
    retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding retry logic to synchronous functions.

    Same as retry() but for non-async functions.
    """
    import time

    if config is None:
        config = RetryConfig.default()

    exceptions = retryable_exceptions or config.retryable_exceptions

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    attempts_left = config.max_attempts - attempt - 1

                    if attempts_left > 0:
                        delay = compute_delay(attempt, config)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__}: "
                            f"{type(e).__name__}: {e}. Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} retry attempts exhausted for {func.__name__}"
                        )

                except Exception:
                    raise

            raise RetryExhaustedError(
                f"All {config.max_attempts} retry attempts exhausted for {func.__name__}",
                attempts=config.max_attempts,
                last_exception=last_exception,
            )

        return wrapper

    return decorator


# Common retryable exceptions for different provider types
class RateLimitError(Exception):
    """Rate limit exceeded."""
    pass


class ProviderUnavailableError(Exception):
    """Provider temporarily unavailable."""
    pass


class QuotaExceededError(Exception):
    """API quota exceeded."""
    pass


# Pre-configured retry decorators for common use cases
LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        RateLimitError,
        ProviderUnavailableError,
    ),
)

EMBEDDING_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=10.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        RateLimitError,
    ),
)

AUDIO_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=1.0,
    max_delay=5.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    ),
)


# Convenience decorators
def llm_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Retry decorator configured for LLM API calls."""
    return retry(config=LLM_RETRY_CONFIG)(func)


def embedding_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Retry decorator configured for embedding API calls."""
    return retry(config=EMBEDDING_RETRY_CONFIG)(func)


def audio_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Retry decorator configured for audio processing."""
    return retry(config=AUDIO_RETRY_CONFIG)(func)
