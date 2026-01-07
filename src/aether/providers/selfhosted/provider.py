"""
Self-Hosted Vocal Provider

Main orchestrator for XTTS + RVC voice synthesis pipeline.
Implements the VocalProvider interface for seamless integration.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import tempfile
import time
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from aether.providers.base import (
    VocalProvider,
    VoiceProfile,
    VocalRequest,
    AudioBuffer,
    ProviderInfo,
    ProviderStatus,
)

from .config import SelfHostedConfig, AVU_VOICE_CONFIGS
from .xtts import XTTSEngine
from .rvc import RVCEngine
from .bark import BarkSingingProvider, BarkConfig, SingingPhrase, get_bark_provider

logger = logging.getLogger(__name__)


class SynthesisStage(str, Enum):
    """Stages of the synthesis pipeline."""

    INITIALIZING = "initializing"
    TEXT_PROCESSING = "text_processing"
    XTTS_GENERATION = "xtts_generation"
    RVC_CONVERSION = "rvc_conversion"
    POST_PROCESSING = "post_processing"
    OUTPUT = "output"


@dataclass
class SynthesisProgress:
    """Progress tracking for synthesis."""

    stage: SynthesisStage
    progress_pct: float
    message: str
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None


class SelfHostedVocalProvider(VocalProvider):
    """
    Self-hosted vocal synthesis using XTTS + RVC pipeline.

    Two-stage pipeline:
    1. XTTS: Text-to-speech with speaker conditioning
    2. RVC: Voice conversion to match AVU identities

    Features:
    - No external API dependencies
    - High-quality singing synthesis
    - GPU acceleration (CUDA/MPS)
    - Progress reporting for long operations
    - Caching for efficient preview generation
    """

    CACHE_DIR = Path(tempfile.gettempdir()) / "aether_selfhosted_cache"

    def __init__(self, config: SelfHostedConfig | dict[str, Any] | None = None):
        # Handle dict config
        if isinstance(config, dict):
            super().__init__(config)
            self.config = SelfHostedConfig.from_env()
        else:
            super().__init__(config.__dict__ if config else None)
            self.config = config or SelfHostedConfig.from_env()

        # Initialize engines (not loaded yet)
        self.xtts = XTTSEngine(self.config.xtts)
        self.rvc = RVCEngine(self.config.rvc) if self.config.enable_rvc else None

        # Setup cache
        cache_dir = self.config.cache_dir or self.CACHE_DIR
        cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir = cache_dir

        # Progress callback
        self._progress_callback: Optional[Callable[[SynthesisProgress], None]] = None

    def get_info(self) -> ProviderInfo:
        """Get provider information."""
        capabilities = [
            "text_to_speech",
            "voice_conversion",
            "multi_language",
            "gpu_acceleration",
        ]

        if self.rvc:
            capabilities.append("rvc_voice_matching")

        return ProviderInfo(
            name="SelfHosted",
            version="1.0.0",
            provider_type="vocal",
            status=self._status,
            capabilities=capabilities,
            config={
                "xtts_loaded": self.xtts.is_loaded,
                "rvc_enabled": self.rvc is not None,
                "rvc_loaded": self.rvc.is_loaded if self.rvc else False,
                "device": self.xtts.device,
                "available_voices": self.get_available_voices(),
            },
        )

    async def initialize(self) -> bool:
        """Initialize XTTS and RVC engines."""
        logger.info("Initializing self-hosted vocal provider...")

        # Validate configuration
        issues = self.config.validate()
        if issues:
            for issue in issues:
                logger.warning(f"Config issue: {issue}")

        try:
            # Load XTTS (required)
            if not await self.xtts.load():
                logger.error("Failed to load XTTS engine")
                self._status = ProviderStatus.UNAVAILABLE
                return False

            # Load RVC (optional but recommended)
            if self.rvc:
                if not await self.rvc.load():
                    logger.warning("RVC engine failed to load, voice conversion disabled")
                    # Continue in degraded mode
                    self._status = ProviderStatus.DEGRADED
                else:
                    self._status = ProviderStatus.AVAILABLE
            else:
                self._status = ProviderStatus.AVAILABLE

            logger.info(f"Self-hosted vocal provider initialized (status: {self._status.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize self-hosted provider: {e}")
            self._status = ProviderStatus.UNAVAILABLE
            return False

    async def shutdown(self) -> None:
        """Shutdown and release resources."""
        logger.info("Shutting down self-hosted vocal provider...")

        await self.xtts.unload()
        if self.rvc:
            await self.rvc.unload()

        self._status = ProviderStatus.UNAVAILABLE
        logger.info("Self-hosted vocal provider shutdown complete")

    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        return self.xtts.is_loaded

    def is_available(self) -> bool:
        """Check if provider is ready for synthesis."""
        return self._status in (ProviderStatus.AVAILABLE, ProviderStatus.DEGRADED) and self.xtts.is_loaded

    def set_progress_callback(self, callback: Callable[[SynthesisProgress], None]) -> None:
        """Set callback for progress updates during synthesis."""
        self._progress_callback = callback

    def _report_progress(
        self,
        stage: SynthesisStage,
        progress: float,
        message: str,
        elapsed: float = 0.0,
    ) -> None:
        """Report progress to callback."""
        if self._progress_callback:
            self._progress_callback(
                SynthesisProgress(
                    stage=stage,
                    progress_pct=progress,
                    message=message,
                    elapsed_seconds=elapsed,
                )
            )

    async def synthesize(self, request: VocalRequest) -> AudioBuffer:
        """
        Synthesize vocals using XTTS + RVC pipeline.

        Args:
            request: Vocal synthesis request

        Returns:
            AudioBuffer with synthesized audio

        Raises:
            RuntimeError: If provider not initialized
        """
        if not self.xtts.is_loaded:
            raise RuntimeError("Self-hosted provider not initialized")

        start_time = time.time()

        # Map voice profile to AVU voice
        voice_name = self._get_voice_for_profile(request.voice_profile)
        logger.info(f"Synthesizing with voice: {voice_name}")

        # Stage 1: Text processing
        self._report_progress(SynthesisStage.TEXT_PROCESSING, 5, "Processing text...")
        processed_text = self._preprocess_text(request.text)

        # Stage 2: XTTS generation
        self._report_progress(SynthesisStage.XTTS_GENERATION, 10, "Running XTTS synthesis...")

        def xtts_progress(pct, msg):
            # Scale XTTS progress to 10-50% range
            scaled = 10 + (pct / 100) * 40
            self._report_progress(SynthesisStage.XTTS_GENERATION, scaled, msg)

        xtts_audio = await self.xtts.synthesize(
            text=processed_text,
            voice_name=voice_name,
            progress_callback=xtts_progress,
        )

        elapsed = time.time() - start_time
        self._report_progress(
            SynthesisStage.XTTS_GENERATION,
            50,
            f"XTTS complete ({elapsed:.1f}s)",
            elapsed,
        )

        # Stage 3: RVC conversion (if enabled and available)
        if self.rvc and self.rvc.is_loaded and voice_name in self.rvc.get_available_voices():
            self._report_progress(SynthesisStage.RVC_CONVERSION, 55, "Applying voice character...")

            def rvc_progress(pct, msg):
                scaled = 55 + (pct / 100) * 30
                self._report_progress(SynthesisStage.RVC_CONVERSION, scaled, msg)

            rvc_audio = await self.rvc.convert(
                audio=xtts_audio,
                sample_rate=self.xtts.sample_rate,
                target_voice=voice_name,
                progress_callback=rvc_progress,
            )

            elapsed = time.time() - start_time
            self._report_progress(
                SynthesisStage.RVC_CONVERSION,
                85,
                f"RVC complete ({elapsed:.1f}s)",
                elapsed,
            )
        else:
            rvc_audio = xtts_audio
            if not self.rvc:
                logger.debug("RVC disabled, skipping voice conversion")
            elif not self.rvc.is_loaded:
                logger.debug("RVC not loaded, skipping voice conversion")
            else:
                logger.debug(f"No RVC model for {voice_name}, skipping conversion")

        # Stage 4: Post-processing
        self._report_progress(SynthesisStage.POST_PROCESSING, 90, "Post-processing...")
        final_audio = self._post_process(rvc_audio)

        # Stage 5: Output
        elapsed = time.time() - start_time
        self._report_progress(SynthesisStage.OUTPUT, 100, f"Complete ({elapsed:.1f}s)", elapsed)

        logger.info(f"Synthesis complete in {elapsed:.1f}s")

        return AudioBuffer(
            data=final_audio,
            sample_rate=self.config.sample_rate,
            channels=1,
        )

    async def synthesize_singing(
        self,
        lyrics: list[dict],
        voice: str = "AVU-1",
        tempo_bpm: float = 120.0,
        total_duration_seconds: Optional[float] = None,
    ) -> AudioBuffer:
        """
        Synthesize singing vocals using Bark neural synthesis.

        This produces REAL singing voices (not pitch-shifted TTS) by using
        Bark's neural singing mode with musical notation markers.

        Args:
            lyrics: List of lyric dicts with keys:
                - text: The words to sing
                - start_beat: Beat number where phrase starts
                - duration_beats: How many beats the phrase lasts
            voice: AVU voice name (AVU-1, AVU-2, AVU-3, AVU-4)
            tempo_bpm: Track tempo for beat-to-time conversion
            total_duration_seconds: Total track length (optional)

        Returns:
            AudioBuffer with synthesized singing audio

        Example:
            await provider.synthesize_singing(
                lyrics=[
                    {"text": "Feel the beat", "start_beat": 0, "duration_beats": 4},
                    {"text": "Move your body", "start_beat": 8, "duration_beats": 4},
                ],
                voice="AVU-1",
                tempo_bpm=128.0,
            )
        """
        # Get Bark provider
        bark = get_bark_provider()

        # Forward progress callback
        if self._progress_callback:
            bark.set_progress_callback(
                lambda p, m: self._report_progress(
                    SynthesisStage.XTTS_GENERATION,
                    int(p * 100),
                    f"Bark: {m}"
                )
            )

        # Convert lyric dicts to SingingPhrase objects
        phrases = [
            SingingPhrase(
                text=lyric["text"],
                start_beat=lyric["start_beat"],
                duration_beats=lyric["duration_beats"],
                emotion=lyric.get("emotion", "neutral"),
                style=lyric.get("style", "default"),
            )
            for lyric in lyrics
        ]

        # Synthesize with Bark
        self._report_progress(SynthesisStage.XTTS_GENERATION, 0, "Starting Bark singing synthesis...")

        audio = await bark.synthesize_lyrics(
            phrases=phrases,
            voice=voice,
            tempo_bpm=tempo_bpm,
            total_duration_seconds=total_duration_seconds,
        )

        self._report_progress(SynthesisStage.OUTPUT, 100, "Singing synthesis complete")

        return AudioBuffer(
            data=audio,
            sample_rate=bark.config.target_sample_rate,
            channels=1,
        )

    async def generate_preview(
        self,
        voice_name: str,
        preview_type: str = "default",
        custom_params: Optional[dict] = None,
    ) -> Optional[Path]:
        """
        Generate a preview audio file for a voice.

        Args:
            voice_name: AVU voice name (e.g., "AVU-1")
            preview_type: Type of preview ("default", "emotional", "range")
            custom_params: Optional custom voice parameters

        Returns:
            Path to generated audio file, or None if generation failed
        """
        # Check cache
        cache_key = hashlib.md5(
            f"{voice_name}:{preview_type}:{custom_params}".encode()
        ).hexdigest()[:16]

        cache_path = self.cache_dir / f"preview_{cache_key}.wav"

        if self.config.enable_caching and cache_path.exists():
            logger.debug(f"Using cached preview: {cache_path}")
            return cache_path

        # Preview phrases
        preview_phrases = {
            "default": {
                "AVU-1": "Feel the warmth of my voice, rich and expressive, ready to bring your music to life.",
                "AVU-2": "Smooth and versatile, this voice flows like honey through every melody.",
                "AVU-3": "Deep resonance and power define this voice, grounded and commanding.",
                "AVU-4": "Crystal clear and agile, this voice sparkles with brilliance and precision.",
            },
            "emotional": {
                "AVU-1": "Every note carries emotion, from whispered intimacy to soaring power.",
                "AVU-2": "With warmth and sincerity, I'll capture the heart of every song.",
                "AVU-3": "Strength meets sensitivity in every phrase I deliver.",
                "AVU-4": "Light as air yet full of feeling, every high note shines.",
            },
            "range": {
                "AVU-1": "From the depths of my range to the heights, my voice tells your story.",
                "AVU-2": "Whether soft and intimate or bold and commanding, I adapt to your vision.",
                "AVU-3": "Rich bass tones and warm midrange create an unforgettable presence.",
                "AVU-4": "From gentle whispers to soaring heights, clarity is my signature.",
            },
        }

        # Get preview text
        phrases = preview_phrases.get(preview_type, preview_phrases["default"])
        text = phrases.get(voice_name, phrases.get("AVU-1", "Testing voice synthesis."))

        # Create voice profile
        voice_config = AVU_VOICE_CONFIGS.get(voice_name)
        if custom_params and "timbre" in custom_params:
            brightness = custom_params["timbre"].get("brightness", 0.5)
            breathiness = custom_params["timbre"].get("breathiness", 0.2)
        else:
            brightness = 0.5
            breathiness = 0.2

        profile = VoiceProfile(
            gender="feminine" if voice_name in ["AVU-2", "AVU-4"] else "masculine",
            age="adult",
            brightness=brightness,
            breathiness=breathiness,
            vibrato_depth=0.3,
            vibrato_rate=5.5,
        )

        # Create minimal request
        request = VocalRequest(
            text=text,
            voice_profile=profile,
            melody_pitches=[],
            melody_durations=[],
            emotion="neutral",
            intensity=0.7,
        )

        try:
            result = await self.synthesize(request)

            # Save to cache
            self._save_audio(result.data, result.sample_rate, cache_path)

            logger.info(f"Generated preview: {cache_path}")
            return cache_path

        except Exception as e:
            logger.error(f"Preview generation failed: {e}")
            return None

    async def list_voices(self) -> list[VoiceProfile]:
        """List available voice profiles."""
        profiles = []

        for voice_name, config in AVU_VOICE_CONFIGS.items():
            # Determine gender from voice name
            is_feminine = voice_name in ["AVU-2", "AVU-4"]

            profiles.append(
                VoiceProfile(
                    gender="feminine" if is_feminine else "masculine",
                    age="adult",
                    brightness=0.5,
                    breathiness=0.2,
                    vibrato_depth=0.3,
                    vibrato_rate=5.5,
                )
            )

        return profiles

    async def create_voice(self, profile: VoiceProfile) -> str:
        """Map profile to best matching AVU voice."""
        return self._get_voice_for_profile(profile)

    def get_available_voices(self) -> list[str]:
        """Get list of available voice names."""
        voices = list(AVU_VOICE_CONFIGS.keys())

        # Filter by what's actually available
        if self.rvc and self.rvc.is_loaded:
            rvc_voices = self.rvc.get_available_voices()
            # Return intersection, keeping AVU order
            voices = [v for v in voices if v in rvc_voices or not self.config.enable_rvc]

        return voices

    def _get_voice_for_profile(self, profile: VoiceProfile) -> str:
        """Map VoiceProfile to best matching AVU voice."""
        if profile.gender == "feminine":
            # Feminine voices
            if profile.brightness > 0.65:
                return "AVU-4"  # Soprano - bright
            return "AVU-2"  # Mezzo-soprano - warm
        else:
            # Masculine voices
            if profile.brightness < 0.5:
                return "AVU-3"  # Baritone - dark
            return "AVU-1"  # Lyric Tenor - balanced

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for synthesis."""
        import re

        # Clean whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove unsupported characters
        text = re.sub(r"[^\w\s.,!?'\"-]", "", text)

        return text

    def _post_process(self, audio: np.ndarray) -> np.ndarray:
        """Post-process synthesized audio."""
        # Ensure float32
        audio = audio.astype(np.float32)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        # Resample if needed
        if self.xtts.sample_rate != self.config.sample_rate:
            audio = self._resample(audio, self.xtts.sample_rate, self.config.sample_rate)

        return audio

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if from_sr == to_sr:
            return audio

        try:
            import scipy.signal as signal

            num_samples = int(len(audio) * to_sr / from_sr)
            resampled = signal.resample(audio, num_samples)
            return resampled.astype(np.float32)
        except ImportError:
            logger.warning("scipy not available for resampling")
            return audio

    def _save_audio(self, audio: np.ndarray, sample_rate: int, path: Path) -> None:
        """Save audio to file."""
        try:
            import scipy.io.wavfile as wavfile

            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(str(path), sample_rate, audio_int16)
        except ImportError:
            # Fallback to soundfile
            try:
                import soundfile as sf

                sf.write(str(path), audio, sample_rate)
            except ImportError:
                logger.error("No audio library available for saving (need scipy or soundfile)")

    def clear_cache(self) -> int:
        """Clear preview cache. Returns number of files deleted."""
        count = 0
        for file in self.cache_dir.glob("*.wav"):
            file.unlink()
            count += 1
        logger.info(f"Cleared {count} cached files")
        return count


# =============================================================================
# Singleton Management
# =============================================================================

_provider_instance: Optional[SelfHostedVocalProvider] = None


async def get_selfhosted_provider() -> Optional[SelfHostedVocalProvider]:
    """Get or create the self-hosted provider instance."""
    global _provider_instance

    if _provider_instance is None:
        config = SelfHostedConfig.from_env()

        # Check if configuration is sufficient
        if not config.xtts.model_path and not config.xtts.model_name:
            logger.debug("Self-hosted provider not configured (no XTTS model)")
            return None

        _provider_instance = SelfHostedVocalProvider(config)
        if not await _provider_instance.initialize():
            _provider_instance = None

    return _provider_instance


def is_selfhosted_configured() -> bool:
    """Check if self-hosted provider is configured (without loading models)."""
    import os

    # Check for explicit configuration
    if os.environ.get("XTTS_MODEL_PATH"):
        return True
    if os.environ.get("AETHER_VOICE_PROVIDER") == "self_hosted":
        return True

    # Check if TTS is available (we can use built-in XTTS speakers)
    try:
        # Check for TTS in the voice venv
        venv_python = Path(__file__).parents[4] / ".venv-voice" / "bin" / "python"
        if venv_python.exists():
            return True

        # Check if TTS is installed in current environment
        import TTS
        return True
    except ImportError:
        pass

    return False


def get_voice_venv_python() -> Optional[Path]:
    """Get path to Python in the voice venv (for subprocess calls)."""
    venv_python = Path(__file__).parents[4] / ".venv-voice" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return None
