"""
Voice Synthesis API

REST API endpoints for voice synthesis.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

try:
    from fastapi import APIRouter, HTTPException, BackgroundTasks
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from aether.voice.integration.pipeline import (
    VoiceSynthesisPipeline,
    VoiceSynthesisInput,
    VoiceSynthesisOutput,
)
from aether.voice.identity.blueprint import AVU1Identity
from aether.voice.performance.profiles import GENRE_PROFILES
from aether.voice.quality.thresholds import ReleaseStage


logger = logging.getLogger(__name__)


# Pydantic models for API
if FASTAPI_AVAILABLE:
    class MelodyNote(BaseModel):
        """A single melody note."""
        pitch: int = Field(..., ge=0, le=127, description="MIDI pitch (0-127)")
        start_beat: float = Field(..., ge=0, description="Start time in beats")
        duration_beats: float = Field(..., gt=0, description="Duration in beats")
        velocity: int = Field(100, ge=0, le=127, description="Velocity (0-127)")

    class SynthesisRequest(BaseModel):
        """Request for voice synthesis."""
        lyrics: str = Field(..., min_length=1, description="Lyrics to sing")
        melody: List[MelodyNote] = Field(..., min_items=1, description="Melody notes")
        tempo: float = Field(120.0, gt=0, le=300, description="Tempo in BPM")
        key_root: int = Field(0, ge=0, le=11, description="Key root (0=C, 11=B)")
        scale_type: str = Field("major", description="Scale type (major/minor)")
        genre: str = Field("pop", description="Genre for performance style")
        language: str = Field("en", description="Language (en/es/mixed)")
        emotion: Optional[str] = Field(None, description="Emotion (happy/sad/angry/etc)")
        emotion_intensity: float = Field(0.5, ge=0, le=1, description="Emotion intensity")
        generate_harmonies: bool = Field(False, description="Generate harmony vocals")
        generate_doubles: bool = Field(False, description="Generate vocal doubles")
        arrangement_density: float = Field(0.5, ge=0, le=1, description="Arrangement fullness")
        quality_check: bool = Field(True, description="Run quality evaluation")
        release_stage: str = Field("beta", description="Quality threshold stage")
        output_format: str = Field("wav", description="Output format (wav/mp3/base64)")

    class SynthesisResponse(BaseModel):
        """Response from voice synthesis."""
        success: bool
        job_id: Optional[str] = None
        audio_url: Optional[str] = None
        audio_base64: Optional[str] = None
        duration_seconds: float = 0.0
        sample_rate: int = 48000
        quality_score: Optional[float] = None
        quality_passed: bool = True
        warnings: List[str] = []
        error: Optional[str] = None

    class JobStatus(BaseModel):
        """Status of a synthesis job."""
        job_id: str
        status: str  # pending, processing, completed, failed
        progress: float = 0.0
        message: str = ""
        result: Optional[SynthesisResponse] = None


# Job storage (in-memory for now, would use Redis in production)
_jobs: Dict[str, dict] = {}


def create_voice_router() -> "APIRouter":
    """
    Create FastAPI router for voice synthesis.

    Returns:
        APIRouter with voice endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi")

    router = APIRouter(prefix="/voice", tags=["voice"])
    pipeline = VoiceSynthesisPipeline()

    @router.post("/synthesize", response_model=SynthesisResponse)
    async def synthesize_voice(
        request: SynthesisRequest,
        background_tasks: BackgroundTasks,
    ) -> SynthesisResponse:
        """
        Synthesize vocals from lyrics and melody.

        This is the main endpoint for voice synthesis. It can run
        synchronously for short songs or asynchronously for longer ones.
        """
        try:
            # Convert request to input spec
            input_spec = VoiceSynthesisInput(
                lyrics=request.lyrics,
                melody=[
                    {
                        "pitch": note.pitch,
                        "start_beat": note.start_beat,
                        "duration_beats": note.duration_beats,
                        "velocity": note.velocity,
                    }
                    for note in request.melody
                ],
                tempo=request.tempo,
                key_root=request.key_root,
                scale_type=request.scale_type,
                genre=request.genre,
                language=request.language,
                emotion=request.emotion,
                emotion_intensity=request.emotion_intensity,
                generate_harmonies=request.generate_harmonies,
                generate_doubles=request.generate_doubles,
                arrangement_density=request.arrangement_density,
                quality_check=request.quality_check,
                release_stage=request.release_stage,
            )

            # Synthesize
            output = await pipeline.synthesize(input_spec)

            # Format audio output
            audio_base64 = None
            if request.output_format == "base64":
                audio_bytes = _audio_to_bytes(
                    output.mixed_audio or output.lead_audio,
                    output.sample_rate,
                )
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            return SynthesisResponse(
                success=True,
                audio_base64=audio_base64,
                duration_seconds=output.duration_seconds,
                sample_rate=output.sample_rate,
                quality_score=output.quality_score,
                quality_passed=output.quality_passed,
                warnings=output.quality_warnings,
            )

        except Exception as e:
            logger.exception("Synthesis failed")
            return SynthesisResponse(
                success=False,
                error=str(e),
            )

    @router.post("/synthesize/async", response_model=JobStatus)
    async def synthesize_voice_async(
        request: SynthesisRequest,
        background_tasks: BackgroundTasks,
    ) -> JobStatus:
        """
        Start asynchronous voice synthesis.

        Returns a job ID that can be used to poll for status.
        """
        import uuid
        job_id = str(uuid.uuid4())

        # Initialize job
        _jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "message": "Job queued",
            "result": None,
        }

        # Start background task
        background_tasks.add_task(
            _run_synthesis_job,
            job_id,
            request,
            pipeline,
        )

        return JobStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Job queued",
        )

    @router.get("/jobs/{job_id}", response_model=JobStatus)
    async def get_job_status(job_id: str) -> JobStatus:
        """Get status of a synthesis job."""
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = _jobs[job_id]
        return JobStatus(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            message=job["message"],
            result=job.get("result"),
        )

    @router.get("/genres")
    async def list_genres() -> List[dict]:
        """List available genres with their profiles."""
        return [
            {
                "id": genre_id,
                "name": profile.name,
                "description": profile.description,
            }
            for genre_id, profile in GENRE_PROFILES.items()
        ]

    @router.get("/identity")
    async def get_identity() -> dict:
        """Get current vocal identity information."""
        identity = AVU1Identity
        return {
            "name": identity.name,
            "classification": identity.classification.value,
            "vocal_range": {
                "comfortable_low": identity.vocal_range.comfortable_low,
                "comfortable_high": identity.vocal_range.comfortable_high,
                "tessitura_low": identity.vocal_range.tessitura_low,
                "tessitura_high": identity.vocal_range.tessitura_high,
            },
            "languages": identity.supported_languages,
            "timbre": {
                "brightness": identity.timbre.brightness,
                "breathiness": identity.timbre.breathiness,
                "grit": identity.timbre.grit,
            },
        }

    @router.get("/health")
    async def voice_health() -> dict:
        """Health check for voice synthesis service."""
        return {
            "status": "healthy",
            "pipeline_initialized": pipeline is not None,
            "identity": AVU1Identity.name,
        }

    return router


async def _run_synthesis_job(
    job_id: str,
    request: "SynthesisRequest",
    pipeline: VoiceSynthesisPipeline,
) -> None:
    """Run synthesis job in background."""
    try:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["message"] = "Starting synthesis..."

        # Set up progress callback
        def progress_callback(progress):
            _jobs[job_id]["progress"] = progress.progress_pct
            _jobs[job_id]["message"] = progress.stage_message

        pipeline.set_progress_callback(progress_callback)

        # Convert and run
        input_spec = VoiceSynthesisInput(
            lyrics=request.lyrics,
            melody=[
                {
                    "pitch": note.pitch,
                    "start_beat": note.start_beat,
                    "duration_beats": note.duration_beats,
                    "velocity": note.velocity,
                }
                for note in request.melody
            ],
            tempo=request.tempo,
            key_root=request.key_root,
            scale_type=request.scale_type,
            genre=request.genre,
            language=request.language,
            emotion=request.emotion,
            emotion_intensity=request.emotion_intensity,
            generate_harmonies=request.generate_harmonies,
            generate_doubles=request.generate_doubles,
            arrangement_density=request.arrangement_density,
            quality_check=request.quality_check,
            release_stage=request.release_stage,
        )

        output = await pipeline.synthesize(input_spec)

        # Format result
        audio_bytes = _audio_to_bytes(
            output.mixed_audio or output.lead_audio,
            output.sample_rate,
        )
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["progress"] = 100.0
        _jobs[job_id]["message"] = "Synthesis complete"
        _jobs[job_id]["result"] = SynthesisResponse(
            success=True,
            audio_base64=audio_base64,
            duration_seconds=output.duration_seconds,
            sample_rate=output.sample_rate,
            quality_score=output.quality_score,
            quality_passed=output.quality_passed,
            warnings=output.quality_warnings,
        )

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["message"] = str(e)
        _jobs[job_id]["result"] = SynthesisResponse(
            success=False,
            error=str(e),
        )


def _audio_to_bytes(
    audio: np.ndarray,
    sample_rate: int,
    format: str = "wav",
) -> bytes:
    """Convert audio array to bytes."""
    try:
        import scipy.io.wavfile as wavfile
        buffer = io.BytesIO()
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(buffer, sample_rate, audio_int16)
        return buffer.getvalue()
    except ImportError:
        # Fallback: raw PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()


# Standalone API functions for non-FastAPI use
async def synthesize_voice_standalone(
    lyrics: str,
    melody: List[dict],
    tempo: float = 120.0,
    genre: str = "pop",
    **kwargs,
) -> dict:
    """
    Standalone voice synthesis function.

    Can be used without FastAPI.

    Args:
        lyrics: Lyrics to sing
        melody: List of melody note dicts
        tempo: Tempo in BPM
        genre: Genre for performance style
        **kwargs: Additional options

    Returns:
        Dict with audio data and metadata
    """
    pipeline = VoiceSynthesisPipeline()

    input_spec = VoiceSynthesisInput(
        lyrics=lyrics,
        melody=melody,
        tempo=tempo,
        genre=genre,
        **kwargs,
    )

    output = await pipeline.synthesize(input_spec)

    return {
        "audio": output.mixed_audio or output.lead_audio,
        "sample_rate": output.sample_rate,
        "duration_seconds": output.duration_seconds,
        "quality_score": output.quality_score,
        "quality_passed": output.quality_passed,
        "warnings": output.quality_warnings,
    }


def synthesize_voice_sync(
    lyrics: str,
    melody: List[dict],
    tempo: float = 120.0,
    genre: str = "pop",
    **kwargs,
) -> dict:
    """
    Synchronous wrapper for voice synthesis.

    Args:
        lyrics: Lyrics to sing
        melody: List of melody note dicts
        tempo: Tempo in BPM
        genre: Genre for performance style
        **kwargs: Additional options

    Returns:
        Dict with audio data and metadata
    """
    return asyncio.run(synthesize_voice_standalone(
        lyrics=lyrics,
        melody=melody,
        tempo=tempo,
        genre=genre,
        **kwargs,
    ))
