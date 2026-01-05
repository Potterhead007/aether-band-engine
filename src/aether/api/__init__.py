"""
AETHER API Service Layer

Production-grade REST API for institutional deployment.
"""

from aether.api.app import create_app
from aether.api.auth import AuthMiddleware, APIKeyAuth
from aether.api.ratelimit import RateLimiter

__all__ = [
    "create_app",
    "AuthMiddleware",
    "APIKeyAuth",
    "RateLimiter",
]
