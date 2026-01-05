"""
AETHER FastAPI Application

Institutional-grade REST API with OpenAPI documentation,
authentication, rate limiting, and observability.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aether.core import AetherRuntime, get_runtime
from aether.api.auth import AuthMiddleware, JWTAuth, APIKeyAuth
from aether.api.ratelimit import RateLimitMiddleware, RateLimitConfig
from aether.agents.creative_director import CreativeDirectorAgent, CreativeDirectorInput
from aether.agents.composition import CompositionAgent, CompositionInput
from aether.agents.arrangement import ArrangementAgent, ArrangementInput
from aether.providers import ProviderManager, ProviderConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class GenerateRequest(BaseModel):
    """Request to generate a new track."""

    title: str = Field(..., description="Track title")
    genre: str = Field(..., description="Genre ID (e.g., 'synthwave', 'lo-fi-hip-hop')")
    brief: str = Field(..., description="Creative brief describing the track")
    bpm: Optional[int] = Field(None, ge=40, le=300, description="Tempo in BPM")
    key: Optional[str] = Field(None, description="Musical key (e.g., 'Am', 'C')")
    duration_seconds: Optional[int] = Field(None, ge=30, le=600)


class GenerateResponse(BaseModel):
    """Response from track generation."""

    job_id: str
    status: str
    song_spec: Optional[Dict[str, Any]] = None
    harmony_spec: Optional[Dict[str, Any]] = None
    melody_spec: Optional[Dict[str, Any]] = None
    arrangement_spec: Optional[Dict[str, Any]] = None


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
    """Response from audio rendering."""

    job_id: str
    status: str
    duration_seconds: float
    loudness_lufs: Optional[float] = None
    peak_db: Optional[float] = None
    output_files: Dict[str, str] = Field(
        default_factory=dict, description="Paths to rendered files"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


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
        """Application lifespan handler."""
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

        logger.info("AETHER API service started")
        yield

        # Shutdown
        logger.info("Shutting down AETHER API service...")
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

    # CORS - Security: restrict origins, no wildcard with credentials
    if enable_cors:
        default_origins = os.environ.get(
            "AETHER_CORS_ORIGINS",
            "http://localhost:3000,http://localhost:3001,http://localhost:8000",
        ).split(",")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or default_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
        )

    # Rate limiting middleware
    if os.environ.get("AETHER_RATE_LIMIT_ENABLED", "true").lower() == "true":
        app.add_middleware(RateLimitMiddleware, config=RateLimitConfig.from_env())

    # Authentication middleware
    if os.environ.get("AETHER_AUTH_ENABLED", "false").lower() == "true":
        auth_providers: List = []
        jwt_secret = os.environ.get("AETHER_JWT_SECRET")
        if jwt_secret:
            auth_providers.append(JWTAuth(secret_key=jwt_secret))
        auth_providers.append(APIKeyAuth())

        app.add_middleware(
            AuthMiddleware,
            providers=auth_providers,
            require_auth=True,
        )

    # Request tracking middleware
    @app.middleware("http")
    async def request_tracking(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        start_time = time.time()

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

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc) if app.debug else None,
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register API routes."""

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check system health."""
        runtime = get_runtime()
        system_health = await runtime.health.check_all()

        return HealthResponse(
            status=system_health.status.value,
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
    async def generate_track(request: GenerateRequest):
        """
        Generate a new music track.

        This endpoint starts the music generation pipeline and returns
        the initial song specification. For full track generation,
        use the async job endpoint.
        """
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
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {str(e)}",
            )

    @app.post(
        "/v1/render",
        response_model=RenderResponse,
        tags=["Rendering"],
        status_code=status.HTTP_200_OK,
    )
    async def render_audio(request: RenderRequest):
        """
        Render audio from music specifications.

        Takes song/harmony/melody/arrangement specs and generates
        actual WAV and MP3 audio files.
        """
        from pathlib import Path
        from aether.rendering.engine import RenderingEngine, RenderingConfig

        job_id = str(uuid4())

        try:
            # Configure rendering
            output_dir = Path.home() / ".aether" / "output" / job_id
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

            return RenderResponse(
                job_id=job_id,
                status="completed",
                duration_seconds=result.duration_seconds,
                loudness_lufs=result.loudness_lufs if result.loudness_lufs else None,
                peak_db=result.peak_db if result.peak_db else None,
                output_files={k: str(v) for k, v in result.output_paths.items()},
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Rendering failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Rendering failed: {str(e)}",
            )

    @app.get("/v1/genres", tags=["Reference"])
    async def list_genres():
        """List available genres."""
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

    @app.get("/metrics", tags=["System"])
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
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
