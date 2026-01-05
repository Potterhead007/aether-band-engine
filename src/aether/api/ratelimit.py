"""
AETHER API Rate Limiting

Production-grade rate limiting with multiple backends:
- In-memory (default, single-instance)
- Redis (distributed, multi-instance)

Configuration via environment:
- AETHER_RATE_LIMIT_RPS: Requests per second (default: 10)
- AETHER_RATE_LIMIT_BURST: Burst capacity (default: 20)
- AETHER_RATE_LIMIT_ENABLED: Enable rate limiting (default: true)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_second: float = 10.0
    burst_capacity: int = 20
    enabled: bool = True
    # Paths to exclude from rate limiting
    exclude_paths: set = field(
        default_factory=lambda: {
            "/health",
            "/ready",
            "/live",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
    )

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Create config from environment variables."""
        return cls(
            requests_per_second=float(os.environ.get("AETHER_RATE_LIMIT_RPS", "10")),
            burst_capacity=int(os.environ.get("AETHER_RATE_LIMIT_BURST", "20")),
            enabled=os.environ.get("AETHER_RATE_LIMIT_ENABLED", "true").lower() == "true",
        )


class TokenBucket:
    """
    Token bucket rate limiter.

    Allows burst capacity while maintaining average rate.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens (burst capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.

        Returns True if tokens acquired, False if rate limited.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Add tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        return self.tokens


class RateLimiter:
    """
    Rate limiter with per-client tracking.

    Uses token bucket algorithm for smooth rate limiting.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(config.requests_per_second, config.burst_capacity)
        )
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.monotonic()

    def _get_client_key(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try X-Forwarded-For first (for proxied requests)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Try X-Real-IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    async def is_allowed(self, request: Request) -> tuple[bool, Dict[str, str]]:
        """
        Check if request is allowed.

        Returns:
            Tuple of (allowed, headers) where headers contain rate limit info
        """
        if not self.config.enabled:
            return True, {}

        if request.url.path in self.config.exclude_paths:
            return True, {}

        client_key = self._get_client_key(request)
        bucket = self._buckets[client_key]

        allowed = await bucket.acquire()

        # Build rate limit headers
        headers = {
            "X-RateLimit-Limit": str(self.config.burst_capacity),
            "X-RateLimit-Remaining": str(int(bucket.available_tokens)),
            "X-RateLimit-Reset": str(int(time.time() + 1)),
        }

        if not allowed:
            headers["Retry-After"] = "1"

        # Periodic cleanup of old buckets
        await self._maybe_cleanup()

        return allowed, headers

    async def _maybe_cleanup(self) -> None:
        """Remove stale client buckets."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        # Remove buckets that are at full capacity (inactive)
        stale_keys = [
            key
            for key, bucket in self._buckets.items()
            if bucket.tokens >= self.config.burst_capacity
        ]
        for key in stale_keys:
            del self._buckets[key]

        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale rate limit buckets")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Usage:
        app.add_middleware(RateLimitMiddleware, config=RateLimitConfig.from_env())
    """

    def __init__(self, app, config: Optional[RateLimitConfig] = None):
        super().__init__(app)
        self.config = config or RateLimitConfig.from_env()
        self.limiter = RateLimiter(self.config)

    async def dispatch(self, request: Request, call_next: Callable):
        """Process request through rate limiter."""
        # Skip rate limiting for CORS preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        allowed, headers = await self.limiter.is_allowed(request)

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please slow down.",
                headers=headers,
            )

        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response


# Convenience function for simple rate limiting
def create_rate_limiter() -> RateLimitMiddleware:
    """Create rate limiter with default configuration."""
    return RateLimitMiddleware(None, RateLimitConfig.from_env())


class SlidingWindowCounter:
    """
    Sliding window rate limiter for expensive operations.

    More accurate than fixed windows, with O(1) space per client.
    Used for per-endpoint rate limiting on expensive operations like /v1/generate.
    """

    def __init__(
        self,
        window_size_seconds: float = 3600.0,
        max_requests: int = 10,
    ):
        self.window_size = window_size_seconds
        self.max_requests = max_requests
        self._windows: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if request is allowed.

        Returns (allowed, remaining_requests).
        """
        now = time.monotonic()
        current_window = int(now / self.window_size)
        window_progress = (now % self.window_size) / self.window_size

        async with self._lock:
            if key not in self._windows:
                self._windows[key] = {
                    "current_window": current_window,
                    "current_count": 0,
                    "prev_count": 0,
                }

            state = self._windows[key]

            # Roll over to new window
            if state["current_window"] < current_window:
                state["prev_count"] = (
                    state["current_count"] if state["current_window"] == current_window - 1 else 0
                )
                state["current_count"] = 0
                state["current_window"] = current_window

            # Calculate weighted count
            weighted_count = state["prev_count"] * (1 - window_progress) + state["current_count"]

            if weighted_count >= self.max_requests:
                remaining = 0
                return False, remaining

            state["current_count"] += 1
            remaining = int(self.max_requests - weighted_count - 1)
            return True, max(0, remaining)
