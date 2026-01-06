"""
AETHER FastAPI Application

Institutional-grade REST API with OpenAPI documentation,
authentication, rate limiting, and observability.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Literal

import re
from aether.core import AetherRuntime, get_runtime
from aether.api.auth import AuthMiddleware, JWTAuth, APIKeyAuth
from aether.api.ratelimit import RateLimitMiddleware, RateLimitConfig, SlidingWindowCounter
from aether.agents.creative_director import CreativeDirectorAgent, CreativeDirectorInput
from aether.agents.composition import CompositionAgent, CompositionInput
from aether.agents.arrangement import ArrangementAgent, ArrangementInput
from aether.providers import ProviderManager, ProviderConfig

logger = logging.getLogger(__name__)

# Per-endpoint rate limiters for expensive operations
_generate_limiter: Optional[SlidingWindowCounter] = None
_render_limiter: Optional[SlidingWindowCounter] = None


def get_generate_limiter() -> SlidingWindowCounter:
    """Get or create the generation rate limiter."""
    global _generate_limiter
    if _generate_limiter is None:
        # 10 generations per hour per client
        max_per_hour = int(os.environ.get("AETHER_GENERATE_RATE_LIMIT", "10"))
        _generate_limiter = SlidingWindowCounter(
            window_size_seconds=3600, max_requests=max_per_hour
        )
    return _generate_limiter


def get_render_limiter() -> SlidingWindowCounter:
    """Get or create the render rate limiter."""
    global _render_limiter
    if _render_limiter is None:
        # 20 renders per hour per client
        max_per_hour = int(os.environ.get("AETHER_RENDER_RATE_LIMIT", "20"))
        _render_limiter = SlidingWindowCounter(window_size_seconds=3600, max_requests=max_per_hour)
    return _render_limiter


def safe_path_component(value: str) -> str:
    """Sanitize path component to prevent directory traversal."""
    # Allow alphanumeric, underscore, hyphen, dots, and spaces (for filenames)
    # But don't allow .. to prevent path traversal
    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid path component: {value}")
    if not re.match(r"^[a-zA-Z0-9_. -]+$", value):
        raise ValueError(f"Invalid path component: {value}")
    return value


def get_client_key(request: Request) -> str:
    """Extract client identifier for rate limiting."""
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


# =============================================================================
# Request/Response Models - Typed for Production
# =============================================================================


class SongSpec(BaseModel):
    """Typed song specification."""
    title: str
    genre_id: str
    bpm: int = Field(ge=40, le=300)
    key: str
    mode: str = "minor"
    time_signature: List[int] = Field(default=[4, 4])
    duration_seconds: int = Field(ge=30, le=600)
    energy_curve: Optional[str] = None
    mood: Optional[str] = None
    creative_brief: Optional[str] = None

    class Config:
        extra = "allow"  # Allow additional fields for flexibility


class HarmonySpec(BaseModel):
    """Typed harmony specification."""
    key: str
    mode: str
    chord_progression: Optional[List[str]] = None
    harmonic_rhythm: Optional[float] = None
    tension_profile: Optional[str] = None

    class Config:
        extra = "allow"


class MelodySpec(BaseModel):
    """Typed melody specification."""
    scale: Optional[str] = None
    range_low: Optional[int] = None
    range_high: Optional[int] = None
    density: Optional[float] = None
    motifs: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"


class ArrangementSpec(BaseModel):
    """Typed arrangement specification."""
    sections: Optional[List[Dict[str, Any]]] = None
    instrumentation: Optional[List[str]] = None
    dynamics_curve: Optional[str] = None

    class Config:
        extra = "allow"


class GenerateRequest(BaseModel):
    """Request to generate a new track."""

    title: str = Field(..., description="Track title", min_length=1, max_length=200)
    genre: str = Field(..., description="Genre ID (e.g., 'synthwave', 'lofi-hip-hop')")
    brief: str = Field(..., description="Creative brief describing the track", min_length=1, max_length=2000)
    bpm: Optional[int] = Field(None, ge=40, le=300, description="Tempo in BPM")
    key: Optional[str] = Field(None, description="Musical key (e.g., 'Am', 'C')", pattern=r"^[A-G][#b]?m?$")
    duration_seconds: Optional[int] = Field(None, ge=30, le=600)


class GenerateResponse(BaseModel):
    """Response from track generation with typed specs."""

    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    song_spec: Optional[SongSpec] = None
    harmony_spec: Optional[HarmonySpec] = None
    melody_spec: Optional[MelodySpec] = None
    arrangement_spec: Optional[ArrangementSpec] = None

    class Config:
        # Allow backwards compatibility with untyped dicts
        extra = "allow"


class RenderRequest(BaseModel):
    """Request to render audio from specs."""

    song_spec: Dict[str, Any] = Field(..., description="Song specification")
    harmony_spec: Optional[Dict[str, Any]] = Field(None, description="Harmony specification")
    melody_spec: Optional[Dict[str, Any]] = Field(None, description="Melody specification")
    arrangement_spec: Optional[Dict[str, Any]] = Field(
        None, description="Arrangement specification"
    )
    output_formats: list[str] = Field(default=["wav", "mp3"], description="Output formats")
    render_stems: bool = Field(default=False, description="Also export individual stems")


class RenderResponse(BaseModel):
    """Response from audio rendering with typed fields."""

    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    duration_seconds: float = Field(ge=0)
    loudness_lufs: Optional[float] = Field(None, description="Integrated loudness in LUFS")
    peak_db: Optional[float] = Field(None, description="True peak in dB")
    output_files: Dict[str, str] = Field(
        default_factory=dict, description="Map of format to download URL"
    )


class FLStudioExportRequest(BaseModel):
    """Request to export to FL Studio project."""

    song_spec: Dict[str, Any] = Field(..., description="Song specification")
    harmony_spec: Optional[Dict[str, Any]] = Field(None, description="Harmony specification")
    melody_spec: Optional[Dict[str, Any]] = Field(None, description="Melody specification")
    arrangement_spec: Optional[Dict[str, Any]] = Field(None, description="Arrangement specification")
    project_name: str = Field(default="AETHER Export", description="FL Studio project name")
    artist_name: str = Field(default="AETHER", description="Artist name")


class FLStudioExportResponse(BaseModel):
    """Response from FL Studio export."""

    job_id: str
    status: str
    message: str
    output_file: str
    pyflp_available: bool = Field(description="Whether native .flp export was used")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, bool]


class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details error response."""

    type: str = Field(default="about:blank", description="URI reference for problem type")
    title: str = Field(..., description="Short human-readable summary")
    status: int = Field(..., description="HTTP status code")
    detail: Optional[str] = Field(None, description="Human-readable explanation")
    instance: Optional[str] = Field(None, description="URI reference to specific occurrence")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")


# =============================================================================
# Application Factory
# =============================================================================


def create_app(
    title: str = "AETHER Band Engine API",
    version: str = "1.0.0",
    enable_cors: bool = True,
    cors_origins: Optional[list] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title for documentation
        version: API version
        enable_cors: Enable CORS middleware
        cors_origins: Allowed CORS origins (default: localhost only)

    Returns:
        Configured FastAPI application
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler with graceful shutdown."""
        logger.info("Starting AETHER API service...")

        # Initialize runtime
        runtime = get_runtime()
        await runtime.initialize()

        # Initialize providers from environment or defaults
        provider_config = ProviderConfig(
            llm_provider=os.environ.get("AETHER_LLM_PROVIDER", "mock"),
            midi_provider=os.environ.get("AETHER_MIDI_PROVIDER", "internal"),
            audio_provider=os.environ.get("AETHER_AUDIO_PROVIDER", "synth"),
            embedding_provider=os.environ.get("AETHER_EMBEDDING_PROVIDER", "mock"),
        )
        app.state.provider_manager = ProviderManager(provider_config)
        await app.state.provider_manager.initialize()

        # Track active requests for graceful shutdown
        app.state.active_requests = 0
        app.state.shutting_down = False

        # Setup signal handlers for graceful shutdown
        shutdown_event = asyncio.Event()

        def handle_shutdown_signal(signum, frame):
            """Handle SIGTERM/SIGINT for graceful shutdown."""
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating graceful shutdown...")
            app.state.shutting_down = True
            shutdown_event.set()

        # Register signal handlers (only in main thread)
        try:
            signal.signal(signal.SIGTERM, handle_shutdown_signal)
            signal.signal(signal.SIGINT, handle_shutdown_signal)
        except ValueError:
            # Signal handlers can only be set in main thread
            pass

        logger.info("AETHER API service started")
        yield

        # Graceful shutdown sequence
        logger.info("Shutting down AETHER API service...")
        app.state.shutting_down = True

        # Wait for active requests to complete (max 30 seconds)
        shutdown_timeout = float(os.environ.get("AETHER_SHUTDOWN_TIMEOUT", "30"))
        wait_start = time.time()
        while app.state.active_requests > 0:
            elapsed = time.time() - wait_start
            if elapsed >= shutdown_timeout:
                logger.warning(
                    f"Shutdown timeout ({shutdown_timeout}s) reached with "
                    f"{app.state.active_requests} active requests"
                )
                break
            logger.info(f"Waiting for {app.state.active_requests} active requests to complete...")
            await asyncio.sleep(1)

        # Shutdown providers
        await app.state.provider_manager.shutdown()
        await runtime.shutdown()
        logger.info("AETHER API service stopped")

    app = FastAPI(
        title=title,
        version=version,
        description="AI-powered music generation engine",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS - Security: restrict origins, prevent wildcard with credentials
    if enable_cors:
        api_port = os.environ.get("AETHER_API_PORT", "8000")
        default_origins = os.environ.get(
            "AETHER_CORS_ORIGINS",
            f"http://localhost:3000,http://localhost:3001,http://localhost:{api_port}",
        ).split(",")
        origins = cors_origins or default_origins

        # Security: prevent wildcard with credentials
        has_wildcard = "*" in origins
        if has_wildcard:
            logger.warning("CORS wildcard origin detected - disabling credentials")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=not has_wildcard,  # Disable credentials if wildcard
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Accept", "Accept-Language", "Authorization", "Content-Language", "Content-Type", "X-API-Key", "X-Request-ID"],
        )

    # Security headers middleware
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # XSS protection (legacy but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Content Security Policy (API-appropriate)
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        # Strict Transport Security (HTTPS)
        if os.environ.get("AETHER_ENVIRONMENT") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    # Rate limiting middleware
    if os.environ.get("AETHER_RATE_LIMIT_ENABLED", "true").lower() == "true":
        app.add_middleware(RateLimitMiddleware, config=RateLimitConfig.from_env())

    # Authentication middleware (enabled by default in production)
    auth_enabled = os.environ.get("AETHER_AUTH_ENABLED", "true").lower() != "false"
    if auth_enabled:
        auth_providers: List = []
        jwt_secret = os.environ.get("AETHER_JWT_SECRET")
        if jwt_secret:
            # Validate JWT secret strength
            if len(jwt_secret) < 32:
                logger.warning(
                    "JWT secret is too short (< 32 chars) - consider using a stronger secret"
                )
            if jwt_secret.isalnum() and (jwt_secret.islower() or jwt_secret.isupper()):
                logger.warning(
                    "JWT secret lacks complexity - consider adding mixed case and symbols"
                )
            auth_providers.append(JWTAuth(secret_key=jwt_secret))

        # API Key auth - disable query param in production
        is_production = os.environ.get("AETHER_ENVIRONMENT", "development") == "production"
        auth_providers.append(APIKeyAuth(query_param=None if is_production else "api_key"))

        # Public endpoints exempt from authentication
        public_paths = [
            "/health",
            "/ready",
            "/live",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/v1/genres",  # Public reference data
            "/v1/generate",  # Allow demo generation without auth
            "/v1/render",  # Allow demo rendering without auth
            "/v1/download/",  # Allow audio file downloads (prefix match)
            "/v1/export/flstudio",  # Allow FL Studio export without auth
        ]
        app.add_middleware(
            AuthMiddleware,
            providers=auth_providers,
            require_auth=True,
            exclude_paths=public_paths,
        )

    # Request tracking middleware with graceful shutdown support
    @app.middleware("http")
    async def request_tracking(request: Request, call_next):
        # Reject new requests during shutdown (except health checks)
        if (
            hasattr(app.state, "shutting_down")
            and app.state.shutting_down
            and request.url.path not in ["/health", "/ready", "/live"]
        ):
            return JSONResponse(
                status_code=503,
                content={"error": "Service shutting down", "retry_after": 30},
                headers={"Retry-After": "30"},
            )

        request_id = str(uuid4())
        request.state.request_id = request_id
        start_time = time.time()

        # Track active requests for graceful shutdown
        if hasattr(app.state, "active_requests"):
            app.state.active_requests += 1

        try:
            response = await call_next(request)

            duration_ms = (time.time() - start_time) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

            # Metrics
            runtime = get_runtime()
            runtime.metrics.counter(
                "http_requests_total",
                "Total HTTP requests",
                labels={"method": request.method, "path": request.url.path},
            ).inc()

            return response
        finally:
            if hasattr(app.state, "active_requests"):
                app.state.active_requests -= 1

    # Exception handlers (RFC 7807 Problem Details format)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                type=f"https://aether.band/errors/{exc.status_code}",
                title=str(exc.detail) if exc.detail else "Error",
                status=exc.status_code,
                detail=str(exc.detail) if exc.detail else None,
                instance=str(request.url.path),
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(exclude_none=True),
            media_type="application/problem+json",
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception(f"Unhandled exception [request_id={request_id}]: {exc}")

        # Security: Only expose error details in development mode
        is_production = os.environ.get("AETHER_ENVIRONMENT", "development") == "production"
        detail = None if is_production else str(exc)

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                type="https://aether.band/errors/internal",
                title="Internal Server Error",
                status=500,
                detail=detail,
                instance=str(request.url.path),
                request_id=request_id,
            ).model_dump(exclude_none=True),
            media_type="application/problem+json",
        )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register API routes."""

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check system health with timeout protection."""
        import asyncio

        runtime = get_runtime()
        health_timeout = float(os.environ.get("AETHER_HEALTH_CHECK_TIMEOUT", "5.0"))

        try:
            system_health = await asyncio.wait_for(
                runtime.health.check_all(),
                timeout=health_timeout,
            )
            health_status = system_health.status.value
        except asyncio.TimeoutError:
            logger.warning(f"Health check timed out after {health_timeout}s")
            health_status = "degraded"

        return HealthResponse(
            status=health_status,
            version=runtime.config.version,
            uptime_seconds=runtime.uptime_seconds,
            components={
                "health": runtime._health is not None,
                "metrics": runtime._metrics is not None,
                "providers": runtime._providers is not None,
            },
        )

    @app.get("/ready", tags=["System"])
    async def readiness_check():
        """Kubernetes readiness probe."""
        runtime = get_runtime()
        if runtime.probe_manager.is_ready():
            return {"ready": True}
        raise HTTPException(status_code=503, detail="Not ready")

    @app.get("/live", tags=["System"])
    async def liveness_check():
        """Kubernetes liveness probe."""
        runtime = get_runtime()
        if runtime.probe_manager.is_live():
            return {"live": True}
        raise HTTPException(status_code=503, detail="Not live")

    @app.post(
        "/v1/generate",
        response_model=GenerateResponse,
        tags=["Generation"],
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def generate_track(request: GenerateRequest, http_request: Request):
        """
        Generate a new music track.

        This endpoint starts the music generation pipeline and returns
        the initial song specification. For full track generation,
        use the async job endpoint.
        """
        # Per-endpoint rate limiting for expensive generation
        client_key = get_client_key(http_request)
        limiter = get_generate_limiter()
        allowed, remaining = await limiter.is_allowed(client_key)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Generation rate limit exceeded. Please try again later.",
                headers={"X-RateLimit-Remaining": "0", "Retry-After": "3600"},
            )

        job_id = str(uuid4())

        try:
            # Creative Director
            cd_agent = CreativeDirectorAgent()
            cd_input = CreativeDirectorInput(
                title=request.title,
                genre_id=request.genre,
                creative_brief=request.brief,
                bpm=request.bpm,
                key=request.key,
                duration_seconds=request.duration_seconds,
            )
            cd_result = await cd_agent.process(cd_input, context={})

            # Composition
            comp_agent = CompositionAgent()
            comp_input = CompositionInput(
                song_spec=cd_result.song_spec,
                genre_profile_id=request.genre,
            )
            comp_result = await comp_agent.process(comp_input, context={})

            # Arrangement
            arr_agent = ArrangementAgent()
            arr_input = ArrangementInput(
                song_spec=cd_result.song_spec,
                harmony_spec=comp_result.harmony_spec,
                melody_spec=comp_result.melody_spec,
                genre_profile_id=request.genre,
            )
            arr_result = await arr_agent.process(arr_input, context={})

            return GenerateResponse(
                job_id=job_id,
                status="completed",
                song_spec=cd_result.song_spec,
                harmony_spec=comp_result.harmony_spec,
                melody_spec=comp_result.melody_spec,
                arrangement_spec=arr_result.arrangement_spec,
            )

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            # Security: Don't expose internal details in production
            is_production = os.environ.get("AETHER_ENVIRONMENT", "development") == "production"
            detail = "Generation failed. Please try again." if is_production else f"Generation failed: {str(e)}"
            raise HTTPException(
                status_code=500,
                detail=detail,
            )

    @app.post(
        "/v1/render",
        response_model=RenderResponse,
        tags=["Rendering"],
        status_code=status.HTTP_200_OK,
    )
    async def render_audio(request: RenderRequest, http_request: Request):
        """
        Render audio from music specifications.

        Takes song/harmony/melody/arrangement specs and generates
        actual WAV and MP3 audio files.
        """
        from pathlib import Path
        from aether.rendering.engine import RenderingEngine, RenderingConfig

        # Per-endpoint rate limiting for expensive rendering
        client_key = get_client_key(http_request)
        limiter = get_render_limiter()
        allowed, remaining = await limiter.is_allowed(client_key)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Render rate limit exceeded. Please try again later.",
                headers={"X-RateLimit-Remaining": "0", "Retry-After": "3600"},
            )

        job_id = str(uuid4())

        try:
            # Validate and sanitize path component
            safe_job_id = safe_path_component(job_id)
            # Configure rendering
            output_dir = Path.home() / ".aether" / "output" / safe_job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            config = RenderingConfig(
                sample_rate=48000,
                bit_depth=24,
                output_dir=output_dir,
                render_stems=request.render_stems,
                apply_mastering=True,
                target_lufs=-14.0,
                true_peak_ceiling=-1.0,
                export_formats=request.output_formats,
            )

            # Prepare pipeline output
            pipeline_output = {
                "song_id": job_id,
                "song_spec": request.song_spec,
                "harmony_spec": request.harmony_spec or {},
                "melody_spec": request.melody_spec or {},
                "arrangement_spec": request.arrangement_spec or {},
            }

            # Render
            engine = RenderingEngine(config)
            result = await engine.render(pipeline_output)

            if not result.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Rendering failed: {', '.join(result.errors)}",
                )

            # Convert file paths to download URLs
            from pathlib import Path
            output_urls = {}
            for key, path in result.output_paths.items():
                filename = Path(path).name
                output_urls[key] = f"/v1/download/{job_id}/{filename}"

            return RenderResponse(
                job_id=job_id,
                status="completed",
                duration_seconds=result.duration_seconds,
                loudness_lufs=result.loudness_lufs if result.loudness_lufs else None,
                peak_db=result.peak_db if result.peak_db else None,
                output_files=output_urls,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Rendering failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Rendering failed: {str(e)}",
            )

    @app.get(
        "/v1/download/{job_id}/{filename}",
        tags=["Rendering"],
        response_class=FileResponse,
    )
    async def download_audio(job_id: str, filename: str):
        """
        Download a rendered audio file.

        Use the job_id and filename from the render response.
        """
        from pathlib import Path

        # Validate job_id format (UUID)
        try:
            from uuid import UUID
            UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")

        # Sanitize filename to prevent path traversal
        safe_filename = safe_path_component(filename)
        if not safe_filename or safe_filename != filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Only allow audio and FL Studio file extensions
        allowed_extensions = {".wav", ".mp3", ".flac", ".aiff", ".flp", ".json", ".zip", ".mid"}
        ext = Path(safe_filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Construct file path - check both direct and nested paths
        # The render engine may nest files under song_id (which equals job_id)
        base_dir = Path.home() / ".aether" / "output" / job_id
        file_path = base_dir / safe_filename

        # Also check nested path (output/{job_id}/{job_id}/filename)
        if not file_path.exists():
            nested_path = base_dir / job_id / safe_filename
            if nested_path.exists():
                file_path = nested_path
            else:
                raise HTTPException(status_code=404, detail="File not found")

        # Determine media type
        media_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".aiff": "audio/aiff",
            ".flp": "application/octet-stream",
            ".json": "application/json",
            ".zip": "application/zip",
            ".mid": "audio/midi",
        }
        media_type = media_types.get(ext, "application/octet-stream")

        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=safe_filename,
        )

    @app.get("/v1/genres", tags=["Reference"])
    async def list_genres():
        """List available genres."""
        try:
            from aether.knowledge import get_genre_manager

            manager = get_genre_manager()
            return {
                "genres": [
                    {
                        "id": profile.genre_id,
                        "name": profile.name,
                        "aliases": profile.aliases,
                    }
                    for profile in manager.list_profiles()
                ]
            }
        except Exception as e:
            logger.error(f"Failed to list genres: {e}", exc_info=True)
            raise

    @app.post(
        "/v1/export/flstudio",
        response_model=FLStudioExportResponse,
        tags=["Export"],
        status_code=status.HTTP_200_OK,
    )
    async def export_to_flstudio(request: FLStudioExportRequest, http_request: Request):
        """
        Export music specifications to FL Studio project package.

        Creates an institutional-grade FL Studio project package containing:
        - MIDI files with full track separation
        - Genre-optimized mixer routing presets
        - Plugin recommendations with specific presets
        - Effects chain templates
        - Human-readable import instructions
        """
        from pathlib import Path
        from aether.providers import FLStudioProvider, FLStudioExportConfig
        from aether.providers.midi import AlgorithmicMIDIProvider

        job_id = str(uuid4())

        try:
            # Initialize providers
            fl_provider = FLStudioProvider()
            await fl_provider.initialize()

            midi_provider = AlgorithmicMIDIProvider()
            await midi_provider.initialize()

            # Extract song parameters
            song_spec = request.song_spec
            bpm = song_spec.get("bpm", 120)
            key = song_spec.get("key", "C")
            time_sig = song_spec.get("time_signature", [4, 4])
            duration = song_spec.get("duration_seconds", 180)
            genre = song_spec.get("genre_id", "electronic")

            # Generate MIDI from specs using generate_from_spec
            harmony_spec = request.harmony_spec or {"key": key, "mode": "minor"}
            melody_spec = request.melody_spec or {}
            rhythm_spec = {"bpm": bpm, "time_signature": time_sig}
            arrangement_spec = request.arrangement_spec or {"duration_seconds": duration}

            midi_data = await midi_provider.generate_from_spec(
                harmony_spec=harmony_spec,
                melody_spec=melody_spec,
                rhythm_spec=rhythm_spec,
                arrangement_spec=arrangement_spec,
            )

            # Configure export
            config = FLStudioExportConfig(
                project_name=request.project_name,
                artist_name=request.artist_name,
                genre=genre,
                tempo_bpm=bpm,
                time_signature=tuple(time_sig),
                create_zip_package=True,
            )

            # Set output directory
            safe_job_id = safe_path_component(job_id)
            output_dir = Path.home() / ".aether" / "output" / safe_job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Export to FL Studio package (returns .zip path)
            result_path = await fl_provider.export_package(
                midi_data=midi_data,
                output_dir=output_dir,
                genre=genre,
                project_name=request.project_name,
                config=config,
            )

            # Determine download URL
            filename = result_path.name
            download_url = f"/v1/download/{job_id}/{filename}"

            return FLStudioExportResponse(
                job_id=job_id,
                status="completed",
                message=f"FL Studio project package exported for {genre} genre with mixer presets",
                output_file=download_url,
                pyflp_available=fl_provider._midiutil_available,
            )

        except Exception as e:
            logger.exception(f"FL Studio export failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"FL Studio export failed: {str(e)}",
            )

    @app.get("/metrics", tags=["System"], include_in_schema=False)
    async def prometheus_metrics(request: Request):
        """Prometheus metrics endpoint (internal network only)."""
        # Restrict to internal networks for security
        client_ip = request.client.host if request.client else ""
        is_internal = (
            client_ip.startswith("10.")
            or client_ip.startswith("172.16.")
            or client_ip.startswith("172.17.")
            or client_ip.startswith("172.18.")
            or client_ip.startswith("172.19.")
            or client_ip.startswith("172.2")
            or client_ip.startswith("172.3")
            or client_ip.startswith("192.168.")
            or client_ip.startswith("127.")
            or client_ip == "::1"
            or os.environ.get("AETHER_METRICS_PUBLIC", "false").lower() == "true"
        )
        if not is_internal:
            raise HTTPException(status_code=403, detail="Metrics endpoint is internal only")

        runtime = get_runtime()
        metrics = runtime.metrics.collect()

        # Format as Prometheus text
        lines = []
        for name, value in metrics.get("counters", {}).items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        for name, value in metrics.get("gauges", {}).items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        return Response(
            content="\n".join(lines),
            media_type="text/plain; charset=utf-8",
        )


# Create module-level app instance for uvicorn
app = create_app()
