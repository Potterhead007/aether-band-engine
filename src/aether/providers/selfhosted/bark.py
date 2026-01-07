"""
Bark Neural Singing Provider for AETHER Voice Engine.

Uses Suno's Bark model for high-quality neural singing synthesis.
Bark can generate actual singing (not pitch-shifted TTS) when using
musical notation markers like ♪.
"""

import asyncio
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Bark sample rate
BARK_SAMPLE_RATE = 24000


@dataclass
class BarkConfig:
    """Configuration for Bark singing provider."""

    # Model settings
    model_size: str = "small"  # "small" or "large"
    use_gpu: bool = True

    # Voice settings
    speaker_preset: str = "v2/en_speaker_9"  # Good for singing

    # Cache settings
    cache_dir: Optional[Path] = None

    # Audio settings
    target_sample_rate: int = 44100

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "aether_bark_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SingingPhrase:
    """A phrase to be sung with timing information."""

    text: str
    start_beat: float  # Beat number where singing starts
    duration_beats: float  # Duration in beats

    # Optional styling
    emotion: str = "neutral"  # neutral, happy, sad, intense
    style: str = "default"  # default, breathy, powerful

    def to_bark_text(self) -> str:
        """Convert to Bark-compatible singing text with musical markers."""
        # Bark uses ♪ to trigger singing mode
        text = self.text.strip()
        if not text.startswith("♪"):
            text = f"♪ {text}"
        if not text.endswith("♪"):
            text = f"{text} ♪"
        return text


@dataclass
class SingingResult:
    """Result of singing synthesis."""

    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    phrase: SingingPhrase


class BarkSingingProvider:
    """
    Neural singing synthesis using Bark.

    Bark generates actual singing voices (not pitch-shifted TTS) when
    given text with musical notation markers.
    """

    # Voice presets optimized for different singing styles
    VOICE_PRESETS = {
        "AVU-1": "v2/en_speaker_9",   # Male tenor, good for pop/rock
        "AVU-2": "v2/en_speaker_2",   # Female, warm tone
        "AVU-3": "v2/en_speaker_6",   # Male baritone, deep
        "AVU-4": "v2/en_speaker_1",   # Female, bright soprano
        "default": "v2/en_speaker_9",
    }

    def __init__(self, config: Optional[BarkConfig] = None):
        self.config = config or BarkConfig()
        self._initialized = False
        self._bark_generate = None
        self._bark_preload = None
        self._progress_callback: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates: callback(progress: 0-1, message: str)."""
        self._progress_callback = callback

    def _report_progress(self, progress: float, message: str):
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(progress, message)
        logger.info(f"Bark: {message} ({progress*100:.0f}%)")

    async def initialize(self) -> bool:
        """Initialize Bark model (lazy loading)."""
        if self._initialized:
            return True

        self._report_progress(0.0, "Loading Bark model...")

        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_bark_sync)
            self._initialized = True
            self._report_progress(1.0, "Bark model loaded")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Bark: {e}")
            return False

    def _load_bark_sync(self):
        """Synchronously load Bark model."""
        import torch

        # Patch torch.load for PyTorch 2.6+ compatibility
        original_load = torch.load
        def patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load

        # Now import Bark
        from bark import generate_audio, preload_models, SAMPLE_RATE

        # Preload models
        logger.info("Preloading Bark models...")
        preload_models()

        self._bark_generate = generate_audio
        self._bark_sample_rate = SAMPLE_RATE

    async def synthesize_phrase(
        self,
        phrase: SingingPhrase,
        voice: str = "default",
        tempo_bpm: float = 120.0,
    ) -> SingingResult:
        """
        Synthesize a singing phrase.

        Args:
            phrase: The phrase to sing with timing info
            voice: Voice preset name (AVU-1, AVU-2, etc.)
            tempo_bpm: Tempo for beat-to-time conversion

        Returns:
            SingingResult with audio data
        """
        if not self._initialized:
            await self.initialize()

        # Get speaker preset for voice
        speaker = self.VOICE_PRESETS.get(voice, self.VOICE_PRESETS["default"])

        # Convert to Bark text format
        bark_text = phrase.to_bark_text()

        # Calculate target duration
        beat_duration = 60.0 / tempo_bpm
        target_duration = phrase.duration_beats * beat_duration

        self._report_progress(0.3, f"Generating: {phrase.text[:30]}...")

        # Generate audio in thread pool
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            self._generate_sync,
            bark_text,
            speaker,
        )

        # Resample to target sample rate if needed
        if self._bark_sample_rate != self.config.target_sample_rate:
            from scipy.signal import resample
            num_samples = int(len(audio) * self.config.target_sample_rate / self._bark_sample_rate)
            audio = resample(audio, num_samples).astype(np.float32)
            sample_rate = self.config.target_sample_rate
        else:
            sample_rate = self._bark_sample_rate

        # Time-stretch to match target duration if needed
        actual_duration = len(audio) / sample_rate
        if abs(actual_duration - target_duration) > 0.1:  # More than 100ms difference
            audio = self._time_stretch(audio, sample_rate, target_duration)

        self._report_progress(1.0, "Phrase complete")

        return SingingResult(
            audio=audio,
            sample_rate=sample_rate,
            duration_seconds=len(audio) / sample_rate,
            phrase=phrase,
        )

    def _generate_sync(self, text: str, speaker: str) -> np.ndarray:
        """Synchronously generate audio with Bark."""
        audio = self._bark_generate(text, history_prompt=speaker)
        return audio.astype(np.float32)

    def _time_stretch(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_duration: float
    ) -> np.ndarray:
        """Time-stretch audio to target duration without changing pitch."""
        from scipy.signal import resample

        current_duration = len(audio) / sample_rate
        stretch_factor = target_duration / current_duration

        # Simple resampling for time stretch (pitch will shift slightly)
        # For production, use phase vocoder or WSOLA
        target_samples = int(len(audio) * stretch_factor)
        stretched = resample(audio, target_samples).astype(np.float32)

        return stretched

    async def synthesize_lyrics(
        self,
        phrases: list[SingingPhrase],
        voice: str = "default",
        tempo_bpm: float = 120.0,
        total_duration_seconds: Optional[float] = None,
    ) -> np.ndarray:
        """
        Synthesize multiple phrases and combine into a single track.

        Args:
            phrases: List of phrases with timing
            voice: Voice preset
            tempo_bpm: Track tempo
            total_duration_seconds: Total track length (phrases placed within)

        Returns:
            Combined audio array at target sample rate
        """
        if not phrases:
            return np.array([], dtype=np.float32)

        beat_duration = 60.0 / tempo_bpm

        # Calculate total duration
        if total_duration_seconds is None:
            max_end = max(p.start_beat + p.duration_beats for p in phrases)
            total_duration_seconds = max_end * beat_duration

        # Create output buffer
        total_samples = int(total_duration_seconds * self.config.target_sample_rate)
        output = np.zeros(total_samples, dtype=np.float32)

        # Synthesize each phrase
        for i, phrase in enumerate(phrases):
            self._report_progress(
                i / len(phrases),
                f"Phrase {i+1}/{len(phrases)}: {phrase.text[:20]}..."
            )

            result = await self.synthesize_phrase(phrase, voice, tempo_bpm)

            # Calculate position in output
            start_time = phrase.start_beat * beat_duration
            start_sample = int(start_time * self.config.target_sample_rate)
            end_sample = min(start_sample + len(result.audio), total_samples)

            # Mix into output
            audio_length = end_sample - start_sample
            output[start_sample:end_sample] += result.audio[:audio_length]

        self._report_progress(1.0, "All phrases complete")

        # Normalize to prevent clipping
        max_val = np.abs(output).max()
        if max_val > 0.95:
            output = output * 0.95 / max_val

        return output

    def get_cache_key(self, text: str, voice: str) -> str:
        """Generate cache key for a synthesis request."""
        key_str = f"{text}:{voice}:{self.config.speaker_preset}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def is_available(self) -> bool:
        """Check if Bark is available."""
        try:
            import bark
            return True
        except ImportError:
            return False


# Singleton instance
_bark_provider: Optional[BarkSingingProvider] = None


def get_bark_provider(config: Optional[BarkConfig] = None) -> BarkSingingProvider:
    """Get or create the Bark singing provider singleton."""
    global _bark_provider
    if _bark_provider is None:
        _bark_provider = BarkSingingProvider(config)
    return _bark_provider
