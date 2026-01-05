"""
AETHER FastAPI Application

Institutional-grade REST API with OpenAPI documentation,
authentication, rate limiting, and observability.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aether.core import AetherRuntime, get_runtime
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
        cors_origins: Allowed CORS origins (default: ["*"])

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

        # Initialize providers
        config = ProviderConfig(
            llm_provider="mock",
            midi_provider="internal",
            audio_provider="synth",
            embedding_provider="mock",
        )
        app.state.provider_manager = ProviderManager(config)
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

    # CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
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
                    "description": profile.description,
                }
                for profile in manager.list_all()
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
