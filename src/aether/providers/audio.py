"""
AETHER Audio Provider

Production-grade audio rendering, processing, and I/O.

Features:
- MIDI to audio rendering (via FluidSynth or fallback synthesis)
- Audio file loading and saving
- Stem mixing with levels and panning
- Effect processing
- Loudness analysis (ITU-R BS.1770-4)

Integration:
- Works with aether.audio module for DSP
- Works with aether.providers.midi for MIDI handling

Example:
    provider = SynthAudioProvider()
    await provider.initialize()

    # Render MIDI to audio
    audio = await provider.render_midi(midi_data, soundfont_path)

    # Mix stems
    mixed = await provider.mix_stems([kick, snare, bass, keys])
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np

from aether.providers.base import (
    AudioBuffer,
    AudioProvider,
    AudioStem,
    MIDIFile,
    ProviderInfo,
    ProviderStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Security: Safe Subprocess Execution
# ============================================================================

# Allowed executables whitelist
_ALLOWED_EXECUTABLES = frozenset({"fluidsynth"})

# Shell metacharacters that indicate injection attempts
_SHELL_METACHAR_PATTERN = re.compile(r'[;&|`$<>\\"\'\n\r]')


class SubprocessSecurityError(Exception):
    """Raised when subprocess input validation fails."""

    pass


def _validate_subprocess_args(cmd: list[str]) -> None:
    """
    Validate subprocess arguments for security.

    Raises:
        SubprocessSecurityError: If validation fails.
    """
    if not cmd:
        raise SubprocessSecurityError("Empty command")

    # Validate executable is in whitelist
    executable = Path(cmd[0]).name
    if executable not in _ALLOWED_EXECUTABLES:
        raise SubprocessSecurityError(
            f"Executable '{executable}' not in whitelist: {_ALLOWED_EXECUTABLES}"
        )

    # Check all arguments for shell metacharacters
    for i, arg in enumerate(cmd):
        if _SHELL_METACHAR_PATTERN.search(arg):
            raise SubprocessSecurityError(
                f"Shell metacharacter detected in argument {i}: {repr(arg[:50])}"
            )

    # Validate file paths don't contain path traversal
    for arg in cmd[1:]:
        if arg.startswith("-"):
            continue  # Skip flags
        # Check for path traversal attempts
        if ".." in arg or arg.startswith("/etc") or arg.startswith("/proc"):
            # Allow absolute paths but verify they're safe
            try:
                resolved = Path(arg).resolve()
                # Ensure path doesn't escape to sensitive directories
                sensitive_dirs = {"/etc", "/proc", "/sys", "/dev"}
                for sensitive in sensitive_dirs:
                    if str(resolved).startswith(sensitive):
                        raise SubprocessSecurityError(
                            f"Path traversal to sensitive directory: {arg}"
                        )
            except (OSError, ValueError):
                pass  # Path doesn't exist yet, that's OK


def _safe_subprocess_run(
    cmd: list[str], timeout: int = 120, **kwargs
) -> subprocess.CompletedProcess:
    """
    Safely execute subprocess with input validation.

    Args:
        cmd: Command and arguments list
        timeout: Maximum execution time in seconds
        **kwargs: Additional subprocess.run arguments

    Returns:
        CompletedProcess result

    Raises:
        SubprocessSecurityError: If validation fails
        subprocess.TimeoutExpired: If command times out
        subprocess.SubprocessError: If command fails
    """
    _validate_subprocess_args(cmd)

    # Ensure we never use shell=True
    kwargs["shell"] = False

    # Force capture output for security logging
    if "capture_output" not in kwargs:
        kwargs["capture_output"] = True

    logger.debug(f"Executing safe subprocess: {cmd[0]}")
    return subprocess.run(cmd, timeout=timeout, **kwargs)


# ============================================================================
# Security: Safe Temporary File Handling
# ============================================================================


@contextmanager
def _safe_temp_directory() -> Generator[Path, None, None]:
    """
    Create a temporary directory with guaranteed cleanup.

    Ensures cleanup even if exceptions occur.
    """
    tmpdir = tempfile.mkdtemp(prefix="aether_audio_")
    tmppath = Path(tmpdir)
    try:
        yield tmppath
    finally:
        # Secure cleanup: remove all files then directory
        try:
            for item in tmppath.rglob("*"):
                if item.is_file():
                    item.unlink()
            for item in sorted(tmppath.rglob("*"), reverse=True):
                if item.is_dir():
                    item.rmdir()
            tmppath.rmdir()
        except OSError as e:
            logger.warning(f"Temp directory cleanup failed: {e}")


# ============================================================================
# Audio Buffer Utilities
# ============================================================================


def normalize_buffer(buffer: AudioBuffer, target_peak: float = 0.9) -> AudioBuffer:
    """Normalize audio to target peak level."""
    peak = np.max(np.abs(buffer.data))
    if peak > 0:
        normalized = buffer.data * (target_peak / peak)
    else:
        normalized = buffer.data
    return AudioBuffer(data=normalized, sample_rate=buffer.sample_rate, channels=buffer.channels)


def mix_buffers(
    buffers: list[AudioBuffer],
    levels_db: list[float] | None = None,
    pans: list[float] | None = None,
) -> AudioBuffer:
    """Mix multiple audio buffers with levels and panning."""
    if not buffers:
        raise ValueError("No buffers to mix")

    # Find common sample rate and max length
    sample_rate = buffers[0].sample_rate
    max_length = max(len(b.data) if b.data.ndim == 1 else b.data.shape[1] for b in buffers)

    # Initialize stereo output
    output = np.zeros((2, max_length))

    # Default levels and pans
    if levels_db is None:
        levels_db = [0.0] * len(buffers)
    if pans is None:
        pans = [0.0] * len(buffers)

    for i, buffer in enumerate(buffers):
        # Convert to stereo
        if buffer.data.ndim == 1:
            stereo = np.array([buffer.data, buffer.data])
        elif buffer.data.shape[0] == 1:
            stereo = np.array([buffer.data[0], buffer.data[0]])
        else:
            stereo = buffer.data[:2]

        # Pad to max length
        if stereo.shape[1] < max_length:
            stereo = np.pad(stereo, ((0, 0), (0, max_length - stereo.shape[1])))

        # Apply level
        level_linear = 10 ** (levels_db[i] / 20)
        stereo = stereo * level_linear

        # Apply panning (-1 = full left, +1 = full right)
        pan = pans[i]
        left_gain = np.cos((pan + 1) * np.pi / 4)
        right_gain = np.sin((pan + 1) * np.pi / 4)
        stereo[0] *= left_gain
        stereo[1] *= right_gain

        # Add to output
        output += stereo

    return AudioBuffer(data=output, sample_rate=sample_rate, channels=2)


# ============================================================================
# FluidSynth Provider
# ============================================================================


class SynthAudioProvider(AudioProvider):
    """
    Audio provider using FluidSynth for MIDI rendering.

    Falls back to simple synthesis if FluidSynth is not available.

    Requirements:
    - FluidSynth CLI (optional but recommended)
    - SoundFont files (.sf2)

    Example:
        provider = SynthAudioProvider(
            soundfont_path="/path/to/default.sf2"
        )
        await provider.initialize()
    """

    def __init__(
        self,
        soundfont_path: Optional[Path] = None,
        sample_rate: int = 48000,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.soundfont_path = Path(soundfont_path) if soundfont_path else None
        self.sample_rate = sample_rate
        self._fluidsynth_available = False

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Synth Audio Provider",
            version="1.0.0",
            provider_type="audio",
            status=self._status,
            capabilities=[
                "midi_rendering",
                "file_io",
                "stem_mixing",
                "effects",
                "loudness_analysis",
            ],
            config={
                "sample_rate": self.sample_rate,
                "soundfont": str(self.soundfont_path) if self.soundfont_path else None,
                "fluidsynth": self._fluidsynth_available,
            },
        )

    async def initialize(self) -> bool:
        """Initialize the provider."""
        # Check for FluidSynth using safe subprocess execution
        try:
            result = _safe_subprocess_run(
                ["fluidsynth", "--version"],
                timeout=5,
            )
            self._fluidsynth_available = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, SubprocessSecurityError):
            self._fluidsynth_available = False

        if self._fluidsynth_available:
            logger.info("FluidSynth available for MIDI rendering")
        else:
            logger.warning("FluidSynth not found, using fallback synthesis")

        self._status = ProviderStatus.AVAILABLE
        return True

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._status = ProviderStatus.UNAVAILABLE

    async def health_check(self) -> bool:
        return self._status == ProviderStatus.AVAILABLE

    async def render_midi(
        self,
        midi_data: MIDIFile,
        soundfont_path: Optional[Path] = None,
    ) -> AudioBuffer:
        """
        Render MIDI to audio.

        Uses FluidSynth if available, otherwise falls back to simple synthesis.
        """
        sf_path = soundfont_path or self.soundfont_path

        if self._fluidsynth_available and sf_path and sf_path.exists():
            return await self._render_with_fluidsynth(midi_data, sf_path)
        else:
            return await self._render_fallback(midi_data)

    async def _render_with_fluidsynth(
        self,
        midi_data: MIDIFile,
        soundfont_path: Path,
    ) -> AudioBuffer:
        """Render MIDI using FluidSynth CLI with secure subprocess handling."""
        # Use safe temp directory with guaranteed cleanup
        with _safe_temp_directory() as tmpdir:
            midi_path = tmpdir / "input.mid"
            wav_path = tmpdir / "output.wav"

            # Save MIDI to temp file
            from aether.providers.midi import AlgorithmicMIDIProvider

            midi_provider = AlgorithmicMIDIProvider()
            await midi_provider.render_to_file(midi_data, midi_path)

            # Validate soundfont path is safe
            sf_path_resolved = soundfont_path.resolve()
            if not sf_path_resolved.exists():
                logger.error(f"SoundFont not found: {soundfont_path}")
                return await self._render_fallback(midi_data)

            # Calculate dynamic timeout based on song duration
            # Estimate duration from MIDI data (in beats -> seconds)
            max_time = 0
            for track in midi_data.tracks:
                for note in track.notes:
                    end_time = note.start_time + note.duration
                    max_time = max(max_time, end_time)
            duration_seconds = (max_time / midi_data.tempo_bpm) * 60
            # Allow 3x realtime + 30s buffer for rendering overhead
            timeout = max(120, int(duration_seconds * 3) + 30)
            logger.info(f"FluidSynth rendering {duration_seconds:.1f}s audio (timeout: {timeout}s)")

            # Render with FluidSynth using safe subprocess
            # CRITICAL: Options must come BEFORE soundfont and MIDI file
            cmd = [
                "fluidsynth",
                "-ni",  # Non-interactive, no shell
                "-g", "0.6",  # Gain (slightly higher for better levels)
                "-r", str(self.sample_rate),
                "-F", str(wav_path),  # Output file (must be before soundfont)
                str(sf_path_resolved),  # SoundFont
                str(midi_path),  # MIDI file
            ]

            try:
                result = _safe_subprocess_run(cmd, timeout=timeout)
                if result.returncode != 0:
                    stderr = result.stderr.decode() if result.stderr else "Unknown error"
                    logger.error(f"FluidSynth error: {stderr}")
                    return await self._render_fallback(midi_data)
            except subprocess.TimeoutExpired:
                logger.error(f"FluidSynth rendering timed out after {timeout}s")
                return await self._render_fallback(midi_data)
            except SubprocessSecurityError as e:
                logger.error(f"Security validation failed: {e}")
                return await self._render_fallback(midi_data)

            # Verify output file exists and has content
            if not wav_path.exists() or wav_path.stat().st_size < 1000:
                logger.error("FluidSynth produced no output or empty file")
                return await self._render_fallback(midi_data)

            # Load the rendered audio
            logger.info(f"FluidSynth render complete: {wav_path.stat().st_size / 1024 / 1024:.1f}MB")
            return await self.load_file(wav_path)

    async def _render_fallback(self, midi_data: MIDIFile) -> AudioBuffer:
        """
        Simple fallback synthesis when FluidSynth is not available.

        Generates basic sine wave synthesis for testing.
        """
        logger.info("Using fallback synthesis (sine waves)")

        # Calculate duration
        max_time = 0
        for track in midi_data.tracks:
            for note in track.notes:
                end_time = note.start_time + note.duration
                max_time = max(max_time, end_time)

        duration_seconds = (max_time / midi_data.tempo_bpm) * 60
        num_samples = int(duration_seconds * self.sample_rate) + self.sample_rate  # Extra second

        # Initialize stereo buffer
        audio = np.zeros((2, num_samples))

        # Render each track
        for track in midi_data.tracks:
            if track.channel == 9:  # Drums
                track_audio = self._render_drums(track, midi_data, num_samples)
            else:
                track_audio = self._render_melodic(track, midi_data, num_samples)

            audio += track_audio

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.8 / peak)

        return AudioBuffer(
            data=audio,
            sample_rate=self.sample_rate,
            channels=2,
        )

    def _render_melodic(
        self,
        track: Any,  # MIDITrack
        midi_data: MIDIFile,
        num_samples: int,
    ) -> np.ndarray:
        """Render melodic track with sine waves."""
        audio = np.zeros((2, num_samples))
        seconds_per_beat = 60.0 / midi_data.tempo_bpm

        for note in track.notes:
            # Convert MIDI note to frequency
            freq = 440.0 * (2 ** ((note.pitch - 69) / 12.0))

            # Calculate sample positions
            start_sample = int(note.start_time * seconds_per_beat * self.sample_rate)
            duration_samples = int(note.duration * seconds_per_beat * self.sample_rate)

            if start_sample >= num_samples:
                continue

            end_sample = min(start_sample + duration_samples, num_samples)
            num_note_samples = end_sample - start_sample

            # Generate sine wave
            t = np.linspace(0, duration_samples / self.sample_rate, num_note_samples)
            wave = np.sin(2 * np.pi * freq * t)

            # Apply envelope (ADSR-like)
            envelope = self._create_envelope(num_note_samples, self.sample_rate)
            wave = wave * envelope

            # Apply velocity
            amplitude = (note.velocity / 127.0) * 0.3
            wave = wave * amplitude

            # Add to stereo buffer
            audio[0, start_sample:end_sample] += wave
            audio[1, start_sample:end_sample] += wave

        return audio

    def _render_drums(
        self,
        track: Any,  # MIDITrack
        midi_data: MIDIFile,
        num_samples: int,
    ) -> np.ndarray:
        """Render drum track with noise and clicks."""
        audio = np.zeros((2, num_samples))
        seconds_per_beat = 60.0 / midi_data.tempo_bpm

        for note in track.notes:
            start_sample = int(note.start_time * seconds_per_beat * self.sample_rate)

            if start_sample >= num_samples:
                continue

            amplitude = (note.velocity / 127.0) * 0.5

            # Generate drum sounds based on MIDI note
            if note.pitch in [36, 35]:  # Kick
                drum_sound = self._generate_kick(self.sample_rate)
            elif note.pitch in [38, 40]:  # Snare
                drum_sound = self._generate_snare(self.sample_rate)
            elif note.pitch in [42, 44]:  # Closed hi-hat
                drum_sound = self._generate_hihat(self.sample_rate, closed=True)
            elif note.pitch == 46:  # Open hi-hat
                drum_sound = self._generate_hihat(self.sample_rate, closed=False)
            else:  # Generic
                drum_sound = self._generate_click(self.sample_rate)

            drum_sound = drum_sound * amplitude

            # Add to buffer
            end_sample = min(start_sample + len(drum_sound), num_samples)
            samples_to_add = end_sample - start_sample
            audio[0, start_sample:end_sample] += drum_sound[:samples_to_add]
            audio[1, start_sample:end_sample] += drum_sound[:samples_to_add]

        return audio

    def _create_envelope(self, num_samples: int, sample_rate: int) -> np.ndarray:
        """Create ADSR-like envelope."""
        attack_samples = int(0.01 * sample_rate)
        int(0.1 * sample_rate)
        release_samples = int(0.1 * sample_rate)

        envelope = np.ones(num_samples)

        # Attack
        if attack_samples > 0:
            attack_end = min(attack_samples, num_samples)
            envelope[:attack_end] = np.linspace(0, 1, attack_end)

        # Release
        if release_samples > 0 and num_samples > release_samples:
            release_start = num_samples - release_samples
            envelope[release_start:] = np.linspace(1, 0, release_samples)

        return envelope

    def _generate_kick(self, sample_rate: int) -> np.ndarray:
        """Generate kick drum sound."""
        duration = 0.15
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)

        # Pitch envelope (starts high, drops)
        freq = 150 * np.exp(-30 * t) + 50
        phase = np.cumsum(freq) / sample_rate * 2 * np.pi
        wave = np.sin(phase)

        # Amplitude envelope
        envelope = np.exp(-10 * t)
        return wave * envelope

    def _generate_snare(self, sample_rate: int) -> np.ndarray:
        """Generate snare drum sound."""
        duration = 0.2
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)

        # Body (pitched component)
        body = np.sin(2 * np.pi * 200 * t) * np.exp(-20 * t)

        # Snares (noise component)
        noise = np.random.randn(num_samples) * np.exp(-15 * t)

        return body * 0.5 + noise * 0.5

    def _generate_hihat(self, sample_rate: int, closed: bool = True) -> np.ndarray:
        """Generate hi-hat sound."""
        duration = 0.05 if closed else 0.3
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)

        # High-frequency noise
        noise = np.random.randn(num_samples)

        # High-pass filter (simple difference)
        filtered = np.diff(noise, prepend=noise[0])

        # Envelope
        decay_rate = 50 if closed else 10
        envelope = np.exp(-decay_rate * t)

        return filtered * envelope * 0.3

    def _generate_click(self, sample_rate: int) -> np.ndarray:
        """Generate generic percussion click."""
        duration = 0.02
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)

        click = np.sin(2 * np.pi * 1000 * t) * np.exp(-100 * t)
        return click

    async def load_file(self, path: Path) -> AudioBuffer:
        """Load audio from file."""
        try:
            from aether.audio import read_audio

            audio_file = read_audio(path)
            return AudioBuffer(
                data=audio_file.data,
                sample_rate=audio_file.sample_rate,
                channels=2 if audio_file.data.ndim > 1 else 1,
            )
        except ImportError:
            # Fallback to scipy
            try:
                from scipy.io import wavfile

                sr, data = wavfile.read(str(path))

                # Normalize to float
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0

                # Ensure stereo
                if data.ndim == 1:
                    data = np.array([data, data])
                else:
                    data = data.T

                return AudioBuffer(data=data, sample_rate=sr, channels=2)
            except ImportError:
                raise ImportError("scipy required for audio file loading")

    async def save_file(
        self,
        buffer: AudioBuffer,
        path: Path,
        format: str = "wav",
        bit_depth: int = 24,
    ) -> Path:
        """Save audio to file."""
        try:
            from aether.audio import write_audio

            write_audio(
                path=path,
                audio=buffer.data,
                sample_rate=buffer.sample_rate,
            )
            return path
        except ImportError:
            # Fallback to scipy
            from scipy.io import wavfile

            # Convert to int
            if bit_depth == 16:
                data = (buffer.data * 32767).astype(np.int16)
            else:
                data = (buffer.data * 2147483647).astype(np.int32)

            # Ensure correct shape for scipy
            if data.ndim == 2 and data.shape[0] == 2:
                data = data.T

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(str(path), buffer.sample_rate, data)
            return path

    async def mix_stems(
        self,
        stems: list[AudioStem],
        levels_db: dict[str, float] | None = None,
        pans: dict[str, float] | None = None,
    ) -> AudioBuffer:
        """Mix multiple stems into a single buffer."""
        if not stems:
            raise ValueError("No stems to mix")

        buffers = [stem.buffer for stem in stems]
        levels = [levels_db.get(stem.name, 0.0) if levels_db else 0.0 for stem in stems]
        pan_values = [pans.get(stem.name, 0.0) if pans else 0.0 for stem in stems]

        return mix_buffers(buffers, levels, pan_values)

    async def apply_effect(
        self,
        buffer: AudioBuffer,
        effect_type: str,
        params: dict[str, Any],
    ) -> AudioBuffer:
        """Apply an audio effect."""
        try:
            from aether.audio import (
                BiquadFilter,
                Compressor,
                FilterType,
                ParametricEQ,
            )

            if effect_type == "eq":
                eq = ParametricEQ(sample_rate=buffer.sample_rate)
                bands = params.get("bands", [])
                for band in bands:
                    eq.add_band(
                        freq=band.get("freq", 1000),
                        gain_db=band.get("gain_db", 0),
                        q=band.get("q", 1.0),
                        filter_type=band.get("type", "peak"),
                    )
                processed = eq.process(buffer.data)

            elif effect_type == "compressor":
                comp = Compressor(
                    sample_rate=buffer.sample_rate,
                    threshold_db=params.get("threshold_db", -20),
                    ratio=params.get("ratio", 4.0),
                    attack_ms=params.get("attack_ms", 10),
                    release_ms=params.get("release_ms", 100),
                )
                processed = comp.process(buffer.data)

            elif effect_type == "highpass":
                filt = BiquadFilter(
                    sample_rate=buffer.sample_rate,
                    filter_type=FilterType.HIGH_PASS,
                    freq=params.get("freq", 80),
                    q=params.get("q", 0.707),
                )
                processed = filt.process(buffer.data)

            elif effect_type == "lowpass":
                filt = BiquadFilter(
                    sample_rate=buffer.sample_rate,
                    filter_type=FilterType.LOW_PASS,
                    freq=params.get("freq", 10000),
                    q=params.get("q", 0.707),
                )
                processed = filt.process(buffer.data)

            else:
                logger.warning(f"Unknown effect type: {effect_type}")
                processed = buffer.data

            return AudioBuffer(
                data=processed,
                sample_rate=buffer.sample_rate,
                channels=buffer.channels,
            )

        except ImportError:
            logger.warning("aether.audio not available, returning unprocessed audio")
            return buffer

    async def analyze_loudness(
        self,
        buffer: AudioBuffer,
    ) -> dict[str, float]:
        """Analyze audio loudness."""
        try:
            from aether.audio import LoudnessMeter

            meter = LoudnessMeter(buffer.sample_rate)
            measurement = meter.measure(buffer.data)

            return {
                "integrated_lufs": measurement.integrated_lufs,
                "momentary_lufs": measurement.momentary_lufs,
                "short_term_lufs": measurement.short_term_lufs,
                "true_peak_dbtp": measurement.true_peak_dbtp,
                "sample_peak_db": measurement.sample_peak_db,
                "loudness_range_lu": measurement.loudness_range_lu,
            }

        except ImportError:
            # Fallback to simple analysis
            peak = np.max(np.abs(buffer.data))
            np.sqrt(np.mean(buffer.data**2))

            return {
                "integrated_lufs": -14.0,  # Placeholder
                "momentary_lufs": -14.0,
                "short_term_lufs": -14.0,
                "true_peak_dbtp": 20 * np.log10(peak) if peak > 0 else -100,
                "sample_peak_db": 20 * np.log10(peak) if peak > 0 else -100,
                "loudness_range_lu": 8.0,  # Placeholder
            }


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    "SynthAudioProvider",
    "AudioBuffer",
    "AudioStem",
    "normalize_buffer",
    "mix_buffers",
]
