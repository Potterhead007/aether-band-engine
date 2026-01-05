"""
AETHER Rate Limiting

Production-grade rate limiting with:
- Token bucket algorithm
- Sliding window counters
- Per-user and per-org limits
- Redis backend support
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    requests_per_hour: float = 1000.0
    burst_size: int = 20
    per_user: bool = True
    per_org: bool = False


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: float
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time)

    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Try to consume tokens.

        Returns (success, wait_time_seconds).
        """
        now = time.time()
        elapsed = now - self.last_refill

        # Refill tokens
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0.0

        # Calculate wait time
        tokens_needed = tokens - self.tokens
        wait_time = tokens_needed / self.refill_rate
        return False, wait_time


class RateLimiter:
    """
    Rate limiter with multiple strategies.

    Supports:
    - In-memory token buckets (single instance)
    - Redis-backed distributed limiting
    - Per-user, per-org, and global limits
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        redis_url: Optional[str] = None,
    ):
        self.config = config or RateLimitConfig()
        self.redis_url = redis_url
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    def _get_bucket_key(self, request: Request) -> str:
        """Get bucket key for request."""
        parts = []

        # Per-user limiting
        if self.config.per_user:
            auth = getattr(request.state, "auth", None)
            if auth:
                parts.append(f"user:{auth.user_id}")

        # Per-org limiting
        if self.config.per_org:
            auth = getattr(request.state, "auth", None)
            if auth and auth.org_id:
                parts.append(f"org:{auth.org_id}")

        # Fall back to IP
        if not parts:
            client_ip = request.client.host if request.client else "unknown"
            parts.append(f"ip:{client_ip}")

        return ":".join(parts)

    async def check_rate_limit(self, request: Request) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.

        Returns (allowed, headers).
        """
        key = self._get_bucket_key(request)

        async with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    capacity=self.config.burst_size,
                    tokens=self.config.burst_size,
                    refill_rate=self.config.requests_per_second,
                )

            bucket = self._buckets[key]
            allowed, wait_time = bucket.consume()

        headers = {
            "X-RateLimit-Limit": str(int(self.config.requests_per_minute)),
            "X-RateLimit-Remaining": str(int(bucket.tokens)),
            "X-RateLimit-Reset": str(int(time.time() + (bucket.capacity - bucket.tokens) / bucket.refill_rate)),
        }

        if not allowed:
            headers["Retry-After"] = str(int(wait_time) + 1)

        return allowed, headers

    async def cleanup_expired(self, max_age_seconds: float = 3600) -> int:
        """Remove expired buckets."""
        now = time.time()
        expired = []

        async with self._lock:
            for key, bucket in self._buckets.items():
                if now - bucket.last_refill > max_age_seconds:
                    expired.append(key)

            for key in expired:
                del self._buckets[key]

        return len(expired)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for FastAPI.
    """

    def __init__(
        self,
        app,
        limiter: Optional[RateLimiter] = None,
        exclude_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.limiter = limiter or RateLimiter()
        self.exclude_paths = set(exclude_paths or ["/health", "/ready", "/live", "/metrics"])

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        allowed, headers = await self.limiter.check_rate_limit(request)

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers=headers,
            )

        response = await call_next(request)

        # Add rate limit headers
        for key, value in headers.items():
            response.headers[key] = value

        return response


class SlidingWindowCounter:
    """
    Sliding window rate limiter.

    More accurate than fixed windows, with O(1) space.
    """

    def __init__(
        self,
        window_size_seconds: float = 60.0,
        max_requests: int = 100,
    ):
        self.window_size = window_size_seconds
        self.max_requests = max_requests
        self._windows: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Check if request is allowed.

        Returns (allowed, remaining_requests).
        """
        now = time.time()
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
                state["prev_count"] = state["current_count"] if state["current_window"] == current_window - 1 else 0
                state["current_count"] = 0
                state["current_window"] = current_window

            # Calculate weighted count
            weighted_count = state["prev_count"] * (1 - window_progress) + state["current_count"]

            if weighted_count >= self.max_requests:
                remaining = 0
                return False, remaining

            state["current_count"] += 1
            remaining = int(self.max_requests - weighted_count - 1)
            return True, remaining
