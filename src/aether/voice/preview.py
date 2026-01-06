"""
Voice Preview Generator

Professional voice preview system with multiple backends:
1. ElevenLabs (Primary) - High-quality realistic human voices
2. MIDI/FluidSynth (Fallback) - Instrumental approximation

Automatically selects the best available provider based on configuration.
"""

from __future__ import annotations

import os
import tempfile
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, Tuple, Literal
from enum import Enum
import logging

from midiutil import MIDIFile

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class PreviewBackend(str, Enum):
    """Available preview backends."""
    SELF_HOSTED = "self_hosted"  # XTTS + RVC (highest quality, no API costs)
    ELEVENLABS = "elevenlabs"    # External API (high quality, requires key)
    MIDI = "midi"                # Instrumental approximation (always available)
    AUTO = "auto"                # Try SELF_HOSTED -> ELEVENLABS -> MIDI


# Cache directory for generated previews
PREVIEW_CACHE_DIR = Path(tempfile.gettempdir()) / "aether_voice_previews"
PREVIEW_CACHE_DIR.mkdir(exist_ok=True)


# =============================================================================
# Unified Preview Interface
# =============================================================================

async def generate_voice_preview(
    voice_name: str,
    preview_type: Literal["default", "emotional", "range"] = "default",
    backend: PreviewBackend = PreviewBackend.AUTO,
    custom_params: Optional[dict] = None,
) -> Optional[Tuple[Path, str]]:
    """
    Generate a professional voice preview.

    Args:
        voice_name: AVU voice identifier (e.g., "AVU-1")
        preview_type: Type of preview to generate
        backend: Which backend to use (auto, elevenlabs, midi)
        custom_params: Optional custom voice parameters for fine-tuning

    Returns:
        Tuple of (audio_path, backend_used) or None if all backends fail
    """
    from aether.voice.identity import VOICE_REGISTRY

    # Validate voice
    if voice_name not in VOICE_REGISTRY:
        logger.error(f"Unknown voice: {voice_name}")
        return None

    voice = VOICE_REGISTRY[voice_name]

    # Try backends in order
    backends_to_try = []

    if backend == PreviewBackend.AUTO:
        # Fallback chain: Self-hosted (best) -> ElevenLabs -> MIDI (always works)
        backends_to_try = [
            PreviewBackend.SELF_HOSTED,
            PreviewBackend.ELEVENLABS,
            PreviewBackend.MIDI,
        ]
    else:
        backends_to_try = [backend]

    for try_backend in backends_to_try:
        result = await _generate_with_backend(
            voice_name, voice, preview_type, try_backend, custom_params
        )
        if result:
            return (Path(result), try_backend.value)

    logger.error(f"All backends failed for voice preview: {voice_name}")
    return None


async def _generate_with_backend(
    voice_name: str,
    voice,
    preview_type: str,
    backend: PreviewBackend,
    custom_params: Optional[dict],
) -> Optional[str]:
    """Generate preview with a specific backend."""

    if backend == PreviewBackend.SELF_HOSTED:
        return await _generate_selfhosted_preview(
            voice_name, preview_type, custom_params
        )
    elif backend == PreviewBackend.ELEVENLABS:
        return await _generate_elevenlabs_preview(
            voice_name, preview_type, custom_params
        )
    elif backend == PreviewBackend.MIDI:
        return await _generate_midi_preview(voice, custom_params)

    return None


# =============================================================================
# Self-Hosted Backend (XTTS + RVC)
# =============================================================================

async def _generate_selfhosted_preview(
    voice_name: str,
    preview_type: str,
    custom_params: Optional[dict],
) -> Optional[str]:
    """Generate preview using self-hosted XTTS + RVC pipeline."""
    try:
        from aether.providers.selfhosted import (
            get_selfhosted_provider,
            is_selfhosted_configured,
        )

        # Quick check if configured (without loading models)
        if not is_selfhosted_configured():
            logger.debug("Self-hosted provider not configured")
            return None

        provider = await get_selfhosted_provider()
        if not provider:
            logger.debug("Self-hosted provider not available")
            return None

        if not provider.is_available():
            logger.debug("Self-hosted provider not ready")
            return None

        # Generate preview
        result = await provider.generate_preview(
            voice_name=voice_name,
            preview_type=preview_type,
            custom_params=custom_params,
        )

        return str(result) if result else None

    except ImportError:
        logger.debug("Self-hosted provider not installed")
        return None
    except Exception as e:
        logger.warning(f"Self-hosted preview generation failed: {e}")
        return None


# =============================================================================
# ElevenLabs Backend
# =============================================================================

async def _generate_elevenlabs_preview(
    voice_name: str,
    preview_type: str,
    custom_params: Optional[dict],
) -> Optional[str]:
    """Generate preview using ElevenLabs API."""
    try:
        from aether.providers.elevenlabs import get_elevenlabs_provider

        provider = await get_elevenlabs_provider()
        if not provider:
            logger.debug("ElevenLabs provider not available")
            return None

        if custom_params:
            # Generate with custom parameters
            result = await provider.generate_custom_preview(
                voice_name=voice_name,
                timbre_params=custom_params.get("timbre", {}),
                emotion_params=custom_params.get("emotion", {}),
                vibrato_params=custom_params.get("vibrato", {}),
            )
        else:
            result = await provider.generate_preview(
                voice_name=voice_name,
                preview_type=preview_type,
            )

        return str(result) if result else None

    except ImportError:
        logger.debug("ElevenLabs provider not installed")
        return None
    except Exception as e:
        logger.warning(f"ElevenLabs preview generation failed: {e}")
        return None


# =============================================================================
# MIDI Backend (Fallback)
# =============================================================================

# General MIDI voice/choir patches
GM_PATCHES = {
    "soprano": 52,        # Choir Aahs
    "mezzo_soprano": 52,  # Choir Aahs
    "alto": 53,           # Voice Oohs
    "tenor": 52,          # Choir Aahs
    "lyric_tenor": 52,    # Choir Aahs
    "baritone": 53,       # Voice Oohs
    "bass": 53,           # Voice Oohs
    "default": 52,        # Choir Aahs
}


def get_gm_patch_for_voice(classification: str) -> int:
    """Get appropriate GM patch for voice type."""
    return GM_PATCHES.get(classification.lower(), GM_PATCHES["default"])


def generate_preview_melody(
    tessitura_low: int,
    tessitura_high: int,
    brightness: float = 0.5,
    duration_seconds: float = 4.0,
) -> list:
    """
    Generate a melodic phrase showcasing the voice's range.

    Returns list of (pitch, start_beat, duration_beats, velocity) tuples.
    """
    # Calculate comfortable middle of range
    mid = (tessitura_low + tessitura_high) // 2
    range_span = tessitura_high - tessitura_low

    # Create a simple ascending/descending phrase
    if range_span >= 12:  # Octave or more
        notes = [
            (mid - 4, 0.0, 0.75, 90),
            (mid, 0.75, 0.75, 100),
            (mid + 3, 1.5, 0.5, 95),
            (mid + 5, 2.0, 0.5, 100),
            (mid + 7, 2.5, 0.75, 105),
            (mid + 5, 3.25, 0.5, 95),
            (mid + 3, 3.75, 0.5, 90),
            (mid, 4.25, 1.0, 100),
            (mid - 2, 5.25, 0.75, 85),
            (mid - 4, 6.0, 1.5, 90),
        ]
    else:
        notes = [
            (mid - 2, 0.0, 1.0, 90),
            (mid, 1.0, 1.0, 100),
            (mid + 2, 2.0, 1.0, 105),
            (mid, 3.0, 1.0, 100),
            (mid - 2, 4.0, 1.5, 90),
        ]

    # Adjust velocities based on brightness
    velocity_mod = int((brightness - 0.5) * 20)
    notes = [
        (p, s, d, min(127, max(60, v + velocity_mod)))
        for p, s, d, v in notes
    ]

    return notes


def create_preview_midi(
    voice_name: str,
    classification: str,
    tessitura_low: int,
    tessitura_high: int,
    brightness: float = 0.5,
    vibrato_rate: float = 5.5,
) -> Tuple[bytes, str]:
    """Create a MIDI file for voice preview."""
    cache_key = hashlib.md5(
        f"{voice_name}:{classification}:{tessitura_low}:{tessitura_high}:{brightness}".encode()
    ).hexdigest()[:12]

    midi_path = PREVIEW_CACHE_DIR / f"preview_{cache_key}.mid"

    if midi_path.exists():
        logger.debug(f"Using cached MIDI preview: {midi_path}")
        return midi_path.read_bytes(), str(midi_path)

    midi = MIDIFile(1)
    track = 0
    channel = 0
    tempo = 80

    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, get_gm_patch_for_voice(classification))
    midi.addControllerEvent(track, channel, 0, 1, int(vibrato_rate * 10))

    notes = generate_preview_melody(tessitura_low, tessitura_high, brightness)

    for pitch, start, duration, velocity in notes:
        pitch = max(0, min(127, pitch))
        midi.addNote(track, channel, pitch, start, duration, velocity)

    with open(midi_path, 'wb') as f:
        midi.writeFile(f)

    logger.info(f"Generated MIDI preview: {midi_path}")
    return midi_path.read_bytes(), str(midi_path)


async def _generate_midi_preview(voice, custom_params: Optional[dict]) -> Optional[str]:
    """Generate preview using MIDI + FluidSynth."""
    # Extract voice parameters
    classification = voice.classification.value
    tessitura_low = voice.vocal_range.tessitura_low
    tessitura_high = voice.vocal_range.tessitura_high
    brightness = voice.timbre.brightness
    vibrato_rate = (voice.vibrato_rate_hz[0] + voice.vibrato_rate_hz[1]) / 2

    # Apply custom params if provided
    if custom_params:
        timbre = custom_params.get("timbre", {})
        brightness = timbre.get("brightness", brightness)

    return await render_preview_audio(
        voice.name,
        classification,
        tessitura_low,
        tessitura_high,
        brightness,
        vibrato_rate,
    )


async def render_preview_audio(
    voice_name: str,
    classification: str,
    tessitura_low: int,
    tessitura_high: int,
    brightness: float = 0.5,
    vibrato_rate: float = 5.5,
    output_format: str = "wav",
) -> Optional[str]:
    """Render MIDI preview to audio using FluidSynth."""
    midi_bytes, midi_path = create_preview_midi(
        voice_name, classification, tessitura_low, tessitura_high,
        brightness, vibrato_rate
    )

    cache_key = hashlib.md5(
        f"{voice_name}:{classification}:{tessitura_low}:{tessitura_high}:{brightness}:{output_format}".encode()
    ).hexdigest()[:12]
    audio_path = PREVIEW_CACHE_DIR / f"preview_{cache_key}.{output_format}"

    if audio_path.exists():
        logger.debug(f"Using cached audio preview: {audio_path}")
        return str(audio_path)

    soundfont_path = _find_soundfont()
    if not soundfont_path:
        logger.error("No SoundFont found for audio rendering")
        return None

    try:
        cmd = [
            "fluidsynth",
            "-ni",
            "-g", "0.8",
            "-F", str(audio_path),
            soundfont_path,
            midi_path,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=30.0
        )

        if process.returncode != 0:
            logger.error(f"FluidSynth failed: {stderr.decode()}")
            return None

        if not audio_path.exists():
            logger.error("FluidSynth did not create output file")
            return None

        logger.info(f"Rendered audio preview: {audio_path}")
        return str(audio_path)

    except asyncio.TimeoutError:
        logger.error("FluidSynth rendering timed out")
        return None
    except FileNotFoundError:
        logger.error("FluidSynth not found - install with: brew install fluid-synth")
        return None
    except Exception as e:
        logger.error(f"Audio rendering failed: {e}")
        return None


def _find_soundfont() -> Optional[str]:
    """Find available SoundFont file."""
    search_paths = [
        Path.home() / "soundfonts",
        Path("/usr/share/sounds/sf2"),
        Path("/usr/share/soundfonts"),
        Path("/usr/local/share/soundfonts"),
    ]

    env_sf = os.environ.get("SOUNDFONT_PATH")
    if env_sf and Path(env_sf).exists():
        return env_sf

    for search_dir in search_paths:
        if search_dir.exists():
            for sf_file in search_dir.glob("*.sf2"):
                return str(sf_file)
            for sf_file in search_dir.glob("*.sf3"):
                return str(sf_file)

    return None


# =============================================================================
# Utility Functions
# =============================================================================

def get_preview_for_voice(voice) -> dict:
    """Get preview metadata for a voice identity."""
    return {
        "voice_name": voice.name,
        "classification": voice.classification.value,
        "tessitura_low": voice.vocal_range.tessitura_low,
        "tessitura_high": voice.vocal_range.tessitura_high,
        "brightness": voice.timbre.brightness,
        "vibrato_rate": (voice.vibrato_rate_hz[0] + voice.vibrato_rate_hz[1]) / 2,
    }


async def generate_all_previews(backend: PreviewBackend = PreviewBackend.AUTO):
    """Pre-generate previews for all built-in voices."""
    from aether.voice.identity import VOICE_REGISTRY

    results = {}
    for name, voice in VOICE_REGISTRY.items():
        result = await generate_voice_preview(
            voice_name=name,
            preview_type="default",
            backend=backend,
        )
        results[name] = {
            "success": result is not None,
            "audio_path": str(result[0]) if result else None,
            "backend_used": result[1] if result else None,
        }
        logger.info(f"Generated preview for {name}: {results[name]}")

    return results


def get_available_backends() -> list[str]:
    """Get list of available preview backends."""
    backends = ["midi"]  # MIDI is always available

    # Check if ElevenLabs is configured
    if os.environ.get("ELEVENLABS_API_KEY"):
        backends.insert(0, "elevenlabs")

    # Check if self-hosted is configured (without loading models)
    try:
        from aether.providers.selfhosted import is_selfhosted_configured

        if is_selfhosted_configured():
            # Insert at beginning (highest priority)
            backends.insert(0, "self_hosted")
    except ImportError:
        pass

    return backends


def clear_preview_cache() -> int:
    """Clear all cached preview files. Returns count of files deleted."""
    count = 0
    for file in PREVIEW_CACHE_DIR.glob("*"):
        if file.is_file():
            file.unlink()
            count += 1
    logger.info(f"Cleared {count} cached preview files")
    return count
