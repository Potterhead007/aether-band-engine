"""
AETHER Provider Interfaces

Abstract base classes for all providers (MIDI, Audio, Vocal, LLM).
Providers are pluggable components that handle specific generation tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ProviderStatus(str, Enum):
    """Provider operational status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    INITIALIZING = "initializing"


@dataclass
class ProviderInfo:
    """Information about a provider."""

    name: str
    version: str
    provider_type: str
    status: ProviderStatus
    capabilities: list[str]
    config: dict[str, Any]


class BaseProvider(ABC):
    """Base class for all AETHER providers."""

    provider_type: str = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._status = ProviderStatus.INITIALIZING

    @property
    def status(self) -> ProviderStatus:
        return self._status

    @abstractmethod
    def get_info(self) -> ProviderInfo:
        """Get provider information."""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider. Returns True if successful."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of the provider."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy and operational."""
        pass


# ============================================================================
# MIDI Provider
# ============================================================================


@dataclass
class MIDINote:
    """A single MIDI note."""

    pitch: int  # 0-127
    velocity: int  # 0-127
    start_time: float  # In beats
    duration: float  # In beats
    channel: int = 0


@dataclass
class MIDITrack:
    """A MIDI track containing notes."""

    name: str
    notes: list[MIDINote]
    program: int = 0  # MIDI program number
    channel: int = 0


@dataclass
class MIDIFile:
    """Complete MIDI file data."""

    tracks: list[MIDITrack]
    tempo_bpm: float
    time_signature: tuple  # (numerator, denominator)
    ticks_per_beat: int = 480


class MIDIProvider(BaseProvider):
    """Abstract MIDI generation and manipulation provider."""

    provider_type: str = "midi"

    @abstractmethod
    async def generate_from_spec(
        self,
        harmony_spec: Any,
        melody_spec: Any,
        rhythm_spec: Any,
        arrangement_spec: Any,
    ) -> MIDIFile:
        """Generate MIDI from musical specifications."""
        pass

    @abstractmethod
    async def render_to_file(
        self,
        midi_data: MIDIFile,
        output_path: Path,
    ) -> Path:
        """Write MIDI data to a file."""
        pass

    @abstractmethod
    async def load_from_file(self, path: Path) -> MIDIFile:
        """Load MIDI from file."""
        pass

    @abstractmethod
    async def transpose(
        self,
        midi_data: MIDIFile,
        semitones: int,
    ) -> MIDIFile:
        """Transpose MIDI by semitones."""
        pass

    @abstractmethod
    async def quantize(
        self,
        midi_data: MIDIFile,
        grid: float,  # In beats (e.g., 0.25 for 16th notes)
        strength: float = 1.0,
    ) -> MIDIFile:
        """Quantize MIDI to grid."""
        pass


# ============================================================================
# Audio Provider
# ============================================================================


@dataclass
class AudioBuffer:
    """Audio data buffer."""

    data: Any  # numpy array
    sample_rate: int
    channels: int

    @property
    def duration_seconds(self) -> float:
        return len(self.data) / self.sample_rate if self.sample_rate > 0 else 0


@dataclass
class AudioStem:
    """A single audio stem."""

    name: str
    buffer: AudioBuffer
    category: str  # drums, bass, keys, vocals, etc.


class AudioProvider(BaseProvider):
    """Abstract audio rendering and processing provider."""

    provider_type: str = "audio"

    @abstractmethod
    async def render_midi(
        self,
        midi_data: MIDIFile,
        soundfont_path: Optional[Path] = None,
    ) -> AudioBuffer:
        """Render MIDI to audio using soundfont."""
        pass

    @abstractmethod
    async def load_file(self, path: Path) -> AudioBuffer:
        """Load audio from file."""
        pass

    @abstractmethod
    async def save_file(
        self,
        buffer: AudioBuffer,
        path: Path,
        format: str = "wav",
        bit_depth: int = 24,
    ) -> Path:
        """Save audio to file."""
        pass

    @abstractmethod
    async def mix_stems(
        self,
        stems: list[AudioStem],
        levels_db: dict[str, float] | None = None,
        pans: dict[str, float] | None = None,
    ) -> AudioBuffer:
        """Mix multiple stems into a single buffer."""
        pass

    @abstractmethod
    async def apply_effect(
        self,
        buffer: AudioBuffer,
        effect_type: str,
        params: dict[str, Any],
    ) -> AudioBuffer:
        """Apply an audio effect."""
        pass

    @abstractmethod
    async def analyze_loudness(
        self,
        buffer: AudioBuffer,
    ) -> dict[str, float]:
        """Analyze audio loudness (LUFS, peak, etc.)."""
        pass


# ============================================================================
# Vocal Provider
# ============================================================================


@dataclass
class VoiceProfile:
    """Parametric voice definition (NOT a clone)."""

    gender: str  # masculine, feminine, androgynous
    age: str  # young, adult, mature
    brightness: float  # 0-1
    breathiness: float  # 0-1
    vibrato_depth: float  # 0-1
    vibrato_rate: float  # Hz


@dataclass
class VocalRequest:
    """Request for vocal synthesis."""

    text: str
    voice_profile: VoiceProfile
    melody_pitches: list[int]  # MIDI pitches
    melody_durations: list[float]  # In seconds
    emotion: str
    intensity: float  # 0-1


class VocalProvider(BaseProvider):
    """Abstract vocal synthesis provider."""

    provider_type: str = "vocal"

    @abstractmethod
    async def synthesize(
        self,
        request: VocalRequest,
    ) -> AudioBuffer:
        """Synthesize vocals from request."""
        pass

    @abstractmethod
    async def list_voices(self) -> list[VoiceProfile]:
        """List available voice profiles."""
        pass

    @abstractmethod
    async def create_voice(
        self,
        profile: VoiceProfile,
    ) -> str:
        """Create a new parametric voice. Returns voice ID."""
        pass


# ============================================================================
# LLM Provider
# ============================================================================


@dataclass
class LLMMessage:
    """A message in an LLM conversation."""

    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    usage: dict[str, int]  # tokens used
    finish_reason: str


class LLMProvider(BaseProvider):
    """Abstract LLM provider for creative generation."""

    provider_type: str = "llm"

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion from messages."""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate structured output matching schema."""
        pass


# ============================================================================
# Embedding Provider
# ============================================================================


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embedding: list[float]
    model: str
    dimensions: int


class EmbeddingProvider(BaseProvider):
    """Abstract embedding provider for similarity checks."""

    provider_type: str = "embedding"

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    async def similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity between embeddings."""
        pass


# ============================================================================
# Provider Registry
# ============================================================================


class ProviderRegistry:
    """Registry for managing providers."""

    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}

    def register(self, name: str, provider: BaseProvider) -> None:
        """Register a provider."""
        self._providers[name] = provider

    def get(self, name: str) -> BaseProvider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_by_type(self, provider_type: str) -> list[BaseProvider]:
        """Get all providers of a type."""
        return [p for p in self._providers.values() if p.provider_type == provider_type]

    async def initialize_all(self) -> dict[str, bool]:
        """Initialize all providers."""
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.initialize()
            except Exception:
                results[name] = False
        return results

    async def shutdown_all(self) -> None:
        """Shutdown all providers."""
        for provider in self._providers.values():
            try:
                await provider.shutdown()
            except Exception:
                pass

    def list_all(self) -> list[ProviderInfo]:
        """List all registered providers."""
        return [p.get_info() for p in self._providers.values()]


# Global registry (legacy - prefer using AetherRuntime.providers)
_registry: ProviderRegistry | None = None


def get_provider_registry() -> ProviderRegistry:
    """
    Get the global provider registry.

    Note: For new code, prefer using `get_runtime().providers` for
    centralized lifecycle management.
    """
    global _registry
    if _registry is None:
        # Try to get from runtime if available
        try:
            from aether.core.runtime import get_runtime

            return get_runtime().providers
        except ImportError:
            _registry = ProviderRegistry()
    return _registry
