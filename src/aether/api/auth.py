"""
AETHER Authentication Middleware

Enterprise-grade authentication supporting:
- API Key authentication
- JWT Bearer tokens
- SSO/SAML integration hooks
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class AuthContext:
    """Authentication context attached to requests."""

    user_id: str
    org_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    authenticated_at: float = field(default_factory=time.time)


class AuthProvider(ABC):
    """Base class for authentication providers."""

    @abstractmethod
    async def authenticate(self, request: Request) -> Optional[AuthContext]:
        """
        Authenticate a request.

        Returns AuthContext if authenticated, None otherwise.
        """
        pass


class APIKeyAuth(AuthProvider):
    """
    API Key authentication provider.

    Supports:
    - Header-based API keys (X-API-Key)
    - Query parameter API keys
    - Hashed key storage for security
    """

    def __init__(
        self,
        header_name: str = "X-API-Key",
        query_param: Optional[str] = "api_key",
        key_store: Optional[Dict[str, AuthContext]] = None,
    ):
        self.header_name = header_name
        self.query_param = query_param
        self._key_store: Dict[str, AuthContext] = key_store or {}
        self._header_scheme = APIKeyHeader(name=header_name, auto_error=False)

    def register_key(
        self,
        api_key: str,
        user_id: str,
        org_id: Optional[str] = None,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
    ) -> str:
        """
        Register an API key.

        Returns the hashed key for reference.
        """
        key_hash = self._hash_key(api_key)
        self._key_store[key_hash] = AuthContext(
            user_id=user_id,
            org_id=org_id,
            roles=roles or [],
            permissions=permissions or [],
        )
        return key_hash

    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = self._hash_key(api_key)
        if key_hash in self._key_store:
            del self._key_store[key_hash]
            return True
        return False

    def _hash_key(self, key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    async def authenticate(self, request: Request) -> Optional[AuthContext]:
        """Authenticate request using API key."""
        # Try header first
        api_key = request.headers.get(self.header_name)

        # Fall back to query param
        if not api_key and self.query_param:
            api_key = request.query_params.get(self.query_param)

        if not api_key:
            return None

        key_hash = self._hash_key(api_key)
        context = self._key_store.get(key_hash)

        if context:
            # Create fresh context with updated timestamp
            return AuthContext(
                user_id=context.user_id,
                org_id=context.org_id,
                roles=context.roles.copy(),
                permissions=context.permissions.copy(),
                metadata=context.metadata.copy(),
                authenticated_at=time.time(),
            )

        return None


class JWTAuth(AuthProvider):
    """
    JWT Bearer token authentication.

    Supports:
    - RS256 and HS256 algorithms
    - Token validation and expiry
    - Custom claims extraction
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.issuer = issuer
        self.audience = audience
        self._bearer_scheme = HTTPBearer(auto_error=False)

    async def authenticate(self, request: Request) -> Optional[AuthContext]:
        """Authenticate request using JWT."""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]

        try:
            # In production, use PyJWT or python-jose
            # This is a placeholder for the JWT verification logic
            import jwt

            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience,
            )

            return AuthContext(
                user_id=payload.get("sub", "unknown"),
                org_id=payload.get("org_id"),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                metadata={"jwt_claims": payload},
            )

        except Exception as e:
            logger.warning(f"JWT authentication failed: {e}")
            return None


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for FastAPI.

    Supports multiple auth providers with fallback chain.
    """

    def __init__(
        self,
        app,
        providers: Optional[List[AuthProvider]] = None,
        exclude_paths: Optional[List[str]] = None,
        require_auth: bool = True,
    ):
        super().__init__(app)
        self.providers = providers or []
        self.exclude_paths = set(
            exclude_paths or ["/health", "/ready", "/live", "/docs", "/redoc", "/openapi.json"]
        )
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next):
        """Process authentication for each request."""
        # Skip auth for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Try each provider
        auth_context = None
        for provider in self.providers:
            auth_context = await provider.authenticate(request)
            if auth_context:
                break

        # Require authentication
        if self.require_auth and not auth_context:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer, ApiKey"},
            )

        # Attach context to request
        request.state.auth = auth_context

        response = await call_next(request)
        # Note: User ID intentionally NOT exposed in headers for security

        return response


def require_permission(permission: str) -> Callable:
    """
    Decorator to require specific permission.

    Usage:
        @app.get("/admin")
        @require_permission("admin:read")
        async def admin_endpoint(request: Request):
            ...
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(request: Request, *args, **kwargs):
            auth: Optional[AuthContext] = getattr(request.state, "auth", None)

            if not auth:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if permission not in auth.permissions and "admin" not in auth.roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission} required",
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


def require_role(role: str) -> Callable:
    """
    Decorator to require specific role.

    Usage:
        @app.get("/admin")
        @require_role("admin")
        async def admin_endpoint(request: Request):
            ...
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(request: Request, *args, **kwargs):
            auth: Optional[AuthContext] = getattr(request.state, "auth", None)

            if not auth:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if role not in auth.roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role denied: {role} required",
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
