"""
Voice Preview Generator

Generates audio previews for voice identities using MIDI synthesis.
Creates short melodic phrases in each voice's range to demonstrate character.
"""

from __future__ import annotations

import os
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import logging

from midiutil import MIDIFile

logger = logging.getLogger(__name__)

# Cache directory for generated previews
PREVIEW_CACHE_DIR = Path(tempfile.gettempdir()) / "aether_voice_previews"
PREVIEW_CACHE_DIR.mkdir(exist_ok=True)

# General MIDI voice/choir patches
GM_PATCHES = {
    "soprano": 52,      # Choir Aahs
    "mezzo_soprano": 52,  # Choir Aahs
    "alto": 53,         # Voice Oohs
    "tenor": 52,        # Choir Aahs
    "lyric_tenor": 52,  # Choir Aahs
    "baritone": 53,     # Voice Oohs
    "bass": 53,         # Voice Oohs
    "default": 52,      # Choir Aahs
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
    # Adjust complexity based on range
    if range_span >= 12:  # Octave or more
        # Full arpeggio pattern
        notes = [
            (mid - 4, 0.0, 0.75, 90),      # Start below middle
            (mid, 0.75, 0.75, 100),         # Middle
            (mid + 3, 1.5, 0.5, 95),        # Third above
            (mid + 5, 2.0, 0.5, 100),       # Fifth above
            (mid + 7, 2.5, 0.75, 105),      # Seventh (peak)
            (mid + 5, 3.25, 0.5, 95),       # Back down
            (mid + 3, 3.75, 0.5, 90),
            (mid, 4.25, 1.0, 100),          # Resolve to middle
            (mid - 2, 5.25, 0.75, 85),      # Step down
            (mid - 4, 6.0, 1.5, 90),        # Final note
        ]
    else:
        # Simpler pattern for smaller ranges
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
    """
    Create a MIDI file for voice preview.

    Returns:
        Tuple of (midi_bytes, cache_path)
    """
    # Generate cache key
    cache_key = hashlib.md5(
        f"{voice_name}:{classification}:{tessitura_low}:{tessitura_high}:{brightness}".encode()
    ).hexdigest()[:12]

    midi_path = PREVIEW_CACHE_DIR / f"preview_{cache_key}.mid"

    # Check cache
    if midi_path.exists():
        logger.debug(f"Using cached MIDI preview: {midi_path}")
        return midi_path.read_bytes(), str(midi_path)

    # Create MIDI file
    midi = MIDIFile(1)  # One track
    track = 0
    channel = 0
    tempo = 80  # BPM - moderate tempo for vocal feel
    volume = 100

    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, get_gm_patch_for_voice(classification))

    # Add expression/modulation for vibrato feel
    # CC1 = Modulation wheel
    midi.addControllerEvent(track, channel, 0, 1, int(vibrato_rate * 10))

    # Generate and add notes
    notes = generate_preview_melody(tessitura_low, tessitura_high, brightness)

    for pitch, start, duration, velocity in notes:
        # Clamp pitch to valid MIDI range
        pitch = max(0, min(127, pitch))
        midi.addNote(track, channel, pitch, start, duration, velocity)

    # Write to file
    with open(midi_path, 'wb') as f:
        midi.writeFile(f)

    logger.info(f"Generated MIDI preview: {midi_path}")
    return midi_path.read_bytes(), str(midi_path)


async def render_preview_audio(
    voice_name: str,
    classification: str,
    tessitura_low: int,
    tessitura_high: int,
    brightness: float = 0.5,
    vibrato_rate: float = 5.5,
    output_format: str = "wav",
) -> Optional[str]:
    """
    Render a voice preview to audio.

    Returns path to the rendered audio file, or None if rendering fails.
    """
    import asyncio

    # Generate MIDI
    midi_bytes, midi_path = create_preview_midi(
        voice_name, classification, tessitura_low, tessitura_high,
        brightness, vibrato_rate
    )

    # Output path
    cache_key = hashlib.md5(
        f"{voice_name}:{classification}:{tessitura_low}:{tessitura_high}:{brightness}:{output_format}".encode()
    ).hexdigest()[:12]
    audio_path = PREVIEW_CACHE_DIR / f"preview_{cache_key}.{output_format}"

    # Check audio cache
    if audio_path.exists():
        logger.debug(f"Using cached audio preview: {audio_path}")
        return str(audio_path)

    # Find SoundFont
    soundfont_path = _find_soundfont()
    if not soundfont_path:
        logger.error("No SoundFont found for audio rendering")
        return None

    # Render with FluidSynth
    try:
        cmd = [
            "fluidsynth",
            "-ni",                    # Non-interactive
            "-g", "0.8",              # Gain
            "-F", str(audio_path),    # Output file
            soundfont_path,           # SoundFont
            midi_path,                # MIDI file
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

    # Check environment variable first
    env_sf = os.environ.get("SOUNDFONT_PATH")
    if env_sf and Path(env_sf).exists():
        return env_sf

    # Search common locations
    for search_dir in search_paths:
        if search_dir.exists():
            for sf_file in search_dir.glob("*.sf2"):
                return str(sf_file)
            for sf_file in search_dir.glob("*.sf3"):
                return str(sf_file)

    return None


def get_preview_for_voice(voice) -> dict:
    """
    Get preview info for a voice identity.

    Args:
        voice: VocalIdentity instance

    Returns:
        Dict with preview metadata
    """
    return {
        "voice_name": voice.name,
        "classification": voice.classification.value,
        "tessitura_low": voice.vocal_range.tessitura_low,
        "tessitura_high": voice.vocal_range.tessitura_high,
        "brightness": voice.timbre.brightness,
        "vibrato_rate": (voice.vibrato_rate_hz[0] + voice.vibrato_rate_hz[1]) / 2,
    }


async def generate_all_previews():
    """Pre-generate previews for all built-in voices."""
    from aether.voice.identity import VOICE_REGISTRY

    results = {}
    for name, voice in VOICE_REGISTRY.items():
        info = get_preview_for_voice(voice)
        audio_path = await render_preview_audio(**info)
        results[name] = {
            "midi_generated": True,
            "audio_path": audio_path,
            "audio_generated": audio_path is not None,
        }
        logger.info(f"Generated preview for {name}: {results[name]}")

    return results
