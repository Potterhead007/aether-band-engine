"""
Breath Model

Models realistic breath sounds and phrasing for singing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from aether.voice.identity.blueprint import VocalIdentity


@dataclass
class BreathEvent:
    """A breath event in the vocal performance."""
    position: float  # Position in beats
    duration_ms: float  # Breath duration
    intensity: float  # 0.0-1.0
    audible: bool = True  # Whether breath sound is audible
    breath_type: str = "inhale"  # inhale or exhale


@dataclass
class BreathType:
    """Type of breath with characteristics."""
    name: str
    duration_range: Tuple[float, float]  # ms
    intensity_range: Tuple[float, float]  # 0.0-1.0
    audible: bool = True


# Breath types based on context
BREATH_TYPES: Dict[str, BreathType] = {
    "quick": BreathType(
        name="quick",
        duration_range=(100, 200),
        intensity_range=(0.2, 0.4),
        audible=True,
    ),
    "normal": BreathType(
        name="normal",
        duration_range=(200, 350),
        intensity_range=(0.3, 0.5),
        audible=True,
    ),
    "deep": BreathType(
        name="deep",
        duration_range=(350, 500),
        intensity_range=(0.5, 0.7),
        audible=True,
    ),
    "silent": BreathType(
        name="silent",
        duration_range=(150, 300),
        intensity_range=(0.0, 0.1),
        audible=False,
    ),
}


class BreathModel:
    """
    Models breath placement and synthesis for singing.

    Features:
    - Automatic breath placement based on phrase structure
    - Physiological constraints (can't sing forever)
    - Genre-appropriate breath audibility
    - Natural breath sound synthesis
    """

    # Maximum phrase duration before forced breath (seconds)
    MAX_PHRASE_DURATION = 8.0

    def __init__(
        self,
        identity: VocalIdentity,
        sample_rate: int = 48000,
    ):
        """
        Initialize breath model.

        Args:
            identity: Vocal identity for breath characteristics
            sample_rate: Audio sample rate
        """
        self.identity = identity
        self.sample_rate = sample_rate

        # Get breath profile from identity
        self.breath_profile = identity.breath_sound_profile

    def plan_breaths(
        self,
        lyrics: List,  # LyricToken
        melody: List,  # MelodyNote
        section_map: List,  # (start, end, type)
        tempo: float,
    ) -> List[BreathEvent]:
        """
        Determine breath placement for the performance.

        Args:
            lyrics: Lyric tokens
            melody: Melody notes
            section_map: Section boundaries
            tempo: BPM

        Returns:
            List of breath events
        """
        breaths = []
        accumulated_duration = 0.0
        beat_duration_s = 60 / tempo

        for i, note in enumerate(melody):
            note_duration_s = note.duration_beats * beat_duration_s
            accumulated_duration += note_duration_s

            # Check for natural breath points
            is_phrase_end = self._is_phrase_boundary(lyrics, melody, i)
            has_gap = self._has_sufficient_gap(melody, i)
            needs_breath = accumulated_duration > self.MAX_PHRASE_DURATION

            if (is_phrase_end or needs_breath) and has_gap:
                # Determine breath type based on context
                breath_type = self._select_breath_type(
                    accumulated_duration,
                    self._get_next_intensity(section_map, note.start_beat),
                )

                duration = np.random.uniform(*BREATH_TYPES[breath_type].duration_range)
                intensity = np.random.uniform(*BREATH_TYPES[breath_type].intensity_range)

                breaths.append(BreathEvent(
                    position=note.start_beat + note.duration_beats,
                    duration_ms=duration,
                    intensity=intensity,
                    audible=BREATH_TYPES[breath_type].audible,
                    breath_type="inhale",
                ))
                accumulated_duration = 0.0

        return breaths

    def _is_phrase_boundary(
        self,
        lyrics: List,
        melody: List,
        note_idx: int,
    ) -> bool:
        """Check if this note is at a phrase boundary."""
        if note_idx >= len(melody):
            return False

        # Check for punctuation in lyrics
        if note_idx < len(lyrics):
            text = lyrics[note_idx].text if hasattr(lyrics[note_idx], 'text') else ""
            if any(p in text for p in ".,!?;"):
                return True

        # Check melody for phrase end markers
        note = melody[note_idx]
        if hasattr(note, 'is_phrase_end') and note.is_phrase_end:
            return True

        return False

    def _has_sufficient_gap(
        self,
        melody: List,
        note_idx: int,
        min_gap_beats: float = 0.25,
    ) -> bool:
        """Check if there's enough gap for a breath."""
        if note_idx >= len(melody) - 1:
            return True  # End of melody

        current = melody[note_idx]
        next_note = melody[note_idx + 1]

        gap = next_note.start_beat - (current.start_beat + current.duration_beats)
        return gap >= min_gap_beats

    def _select_breath_type(
        self,
        accumulated_duration: float,
        next_intensity: float,
    ) -> str:
        """Select appropriate breath type based on context."""
        if accumulated_duration > 6.0:
            return "deep"  # Long phrase needs deep breath
        elif next_intensity > 0.7:
            return "deep"  # High energy coming needs prep
        elif accumulated_duration < 3.0:
            return "quick"  # Short phrase, quick breath
        else:
            return "normal"

    def _get_next_intensity(
        self,
        section_map: List,
        current_beat: float,
    ) -> float:
        """Get intensity of upcoming section."""
        for start, end, section_type in section_map:
            if start > current_beat:
                # Map section type to intensity
                intensity_map = {
                    "chorus": 0.8,
                    "verse": 0.5,
                    "bridge": 0.6,
                    "intro": 0.4,
                    "outro": 0.3,
                    "pre_chorus": 0.7,
                }
                return intensity_map.get(section_type, 0.5)
        return 0.5

    def synthesize_breath(
        self,
        event: BreathEvent,
    ) -> np.ndarray:
        """
        Synthesize breath audio.

        Args:
            event: Breath event specification

        Returns:
            Audio samples (mono, float32)
        """
        if not event.audible or event.intensity < 0.05:
            # Return silence
            return np.zeros(int(event.duration_ms * self.sample_rate / 1000), dtype=np.float32)

        duration_samples = int(event.duration_ms * self.sample_rate / 1000)

        # Generate breath sound from filtered noise
        noise = self._generate_breath_noise(duration_samples)

        # Apply formant shaping
        shaped = self._apply_breath_formants(noise)

        # Apply envelope
        envelope = self._generate_breath_envelope(duration_samples, event.intensity)
        audio = shaped * envelope

        return audio.astype(np.float32)

    def _generate_breath_noise(self, samples: int) -> np.ndarray:
        """Generate pink-ish noise for breath base."""
        # White noise
        white = np.random.randn(samples)

        # Simple pinking filter (approximate)
        # Cumulative sum creates brown noise, mix for pink-ish
        brown = np.cumsum(white) / 10
        pink = 0.7 * white + 0.3 * brown

        # Normalize
        pink = pink / np.max(np.abs(pink))

        return pink

    def _apply_breath_formants(self, noise: np.ndarray) -> np.ndarray:
        """Apply formant filtering to breath noise."""
        # Simple lowpass filter for breath character
        # In production, would use proper formant filter

        # Moving average for low-pass effect
        kernel_size = 50
        kernel = np.ones(kernel_size) / kernel_size
        filtered = np.convolve(noise, kernel, mode='same')

        # Add some high frequencies back for breathiness
        shaped = 0.7 * filtered + 0.3 * noise

        return shaped

    def _generate_breath_envelope(
        self,
        samples: int,
        intensity: float,
    ) -> np.ndarray:
        """Generate amplitude envelope for breath."""
        # Breath typically has fast attack, medium decay
        attack = int(samples * 0.15)
        sustain = int(samples * 0.5)
        release = samples - attack - sustain

        envelope = np.concatenate([
            np.linspace(0, 1, max(1, attack)),
            np.ones(max(1, sustain)),
            np.linspace(1, 0, max(1, release)),
        ])

        # Adjust length if needed
        if len(envelope) < samples:
            envelope = np.pad(envelope, (0, samples - len(envelope)))
        elif len(envelope) > samples:
            envelope = envelope[:samples]

        return envelope * intensity

    def insert_breaths(
        self,
        audio: np.ndarray,
        breaths: List[BreathEvent],
        tempo: float,
    ) -> np.ndarray:
        """
        Insert breath sounds into audio.

        Args:
            audio: Original audio
            breaths: Breath events
            tempo: BPM

        Returns:
            Audio with breaths mixed in
        """
        beat_duration_samples = int(60 / tempo * self.sample_rate)
        result = audio.copy()

        for breath in breaths:
            breath_audio = self.synthesize_breath(breath)
            start_sample = int(breath.position * beat_duration_samples)

            if start_sample >= 0 and start_sample < len(result):
                end_sample = min(start_sample + len(breath_audio), len(result))
                breath_len = end_sample - start_sample

                if breath_len > 0:
                    result[start_sample:end_sample] += breath_audio[:breath_len] * 0.3

        return result
