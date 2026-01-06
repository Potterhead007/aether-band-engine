"""
Pitch Controller

Generates pitch contours for singing with expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from aether.voice.identity.blueprint import VocalIdentity


@dataclass
class PitchContext:
    """Context for pitch generation."""
    prev_pitch: Optional[int] = None
    next_pitch: Optional[int] = None
    is_phrase_start: bool = False
    is_phrase_end: bool = False
    genre: str = "pop"
    emotion_intensity: float = 0.5


class PitchController:
    """
    Controls pitch generation for singing.

    Balances pitch accuracy with musical expression through:
    - Scoops and falls
    - Vibrato (handled separately)
    - Micro-pitch variations
    """

    # Deviation rules by note position
    DEVIATION_RULES = {
        "note_attack": {
            "scoop_up_cents": (-50, 0),  # Range of scoop from below
            "scoop_down_cents": (0, 30),  # Range of scoop from above
            "duration_ms": (20, 80),
        },
        "note_sustain": {
            "drift_range_cents": (-10, 10),
        },
        "note_release": {
            "fall_range_cents": (-100, 0),
            "rise_range_cents": (0, 50),
        },
    }

    # Genre-specific pitch behaviors
    GENRE_PITCH_STYLE = {
        "pop": {
            "scoop_probability": 0.3,
            "fall_probability": 0.2,
            "drift_amount": 0.3,
        },
        "r-and-b": {
            "scoop_probability": 0.5,
            "fall_probability": 0.4,
            "drift_amount": 0.5,
        },
        "rock": {
            "scoop_probability": 0.2,
            "fall_probability": 0.3,
            "drift_amount": 0.2,
        },
        "jazz": {
            "scoop_probability": 0.4,
            "fall_probability": 0.3,
            "drift_amount": 0.4,
        },
        "house": {
            "scoop_probability": 0.1,
            "fall_probability": 0.1,
            "drift_amount": 0.1,
        },
    }

    def __init__(
        self,
        identity: VocalIdentity,
        frame_rate: float = 100.0,  # Frames per second
    ):
        """
        Initialize pitch controller.

        Args:
            identity: Vocal identity for range constraints
            frame_rate: Pitch contour frame rate
        """
        self.identity = identity
        self.frame_rate = frame_rate

    def generate_pitch_contour(
        self,
        unit,  # AlignedUnit
        tempo: float,
        genre: str = "pop",
        context: Optional[PitchContext] = None,
    ) -> np.ndarray:
        """
        Generate pitch contour for a note.

        Args:
            unit: Aligned unit with pitch and timing
            tempo: BPM
            genre: Genre for style
            context: Optional context for expression

        Returns:
            Pitch contour in Hz, one value per frame
        """
        # Calculate duration in frames
        beat_duration_s = 60 / tempo
        duration_s = unit.duration_beats * beat_duration_s
        num_frames = max(1, int(duration_s * self.frame_rate))

        # Base pitch
        target_hz = self._midi_to_hz(unit.pitch)
        contour = np.full(num_frames, target_hz)

        # Get genre style
        style = self.GENRE_PITCH_STYLE.get(genre, self.GENRE_PITCH_STYLE["pop"])

        # Apply attack behavior
        if context is None or not context.is_phrase_start or np.random.random() < style["scoop_probability"]:
            contour = self._apply_attack(contour, style)

        # Apply sustain drift
        contour = self._apply_drift(contour, style)

        # Apply release behavior
        if context is None or context.is_phrase_end or np.random.random() < style["fall_probability"]:
            contour = self._apply_release(contour, style)

        return contour

    def _midi_to_hz(self, midi_note: int) -> float:
        """Convert MIDI note to frequency in Hz."""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def _hz_to_cents(self, hz: float, reference_hz: float) -> float:
        """Convert Hz difference to cents."""
        if reference_hz <= 0 or hz <= 0:
            return 0.0
        return 1200 * np.log2(hz / reference_hz)

    def _cents_to_ratio(self, cents: float) -> float:
        """Convert cents to frequency ratio."""
        return 2.0 ** (cents / 1200.0)

    def _apply_attack(
        self,
        contour: np.ndarray,
        style: Dict,
    ) -> np.ndarray:
        """Apply attack pitch behavior (scoops)."""
        if len(contour) < 3:
            return contour

        # Attack duration (first 5-15% of note)
        attack_frames = max(1, int(len(contour) * np.random.uniform(0.05, 0.15)))

        # Scoop amount
        scoop_cents = np.random.uniform(-40, -10)  # Scoop up from below
        scoop_ratio = self._cents_to_ratio(scoop_cents)

        # Generate scoop curve
        attack_curve = np.linspace(scoop_ratio, 1.0, attack_frames)

        # Apply to contour
        contour[:attack_frames] *= attack_curve

        return contour

    def _apply_drift(
        self,
        contour: np.ndarray,
        style: Dict,
    ) -> np.ndarray:
        """Apply sustain pitch drift."""
        drift_amount = style.get("drift_amount", 0.3)
        if drift_amount <= 0:
            return contour

        # Generate slow drift using low-frequency noise
        drift_cents = np.random.uniform(-8, 8, len(contour)) * drift_amount

        # Smooth the drift
        kernel_size = max(3, len(contour) // 10)
        kernel = np.ones(kernel_size) / kernel_size
        drift_cents = np.convolve(drift_cents, kernel, mode='same')

        # Convert to ratio and apply
        drift_ratio = self._cents_to_ratio(drift_cents)
        contour *= drift_ratio

        return contour

    def _apply_release(
        self,
        contour: np.ndarray,
        style: Dict,
    ) -> np.ndarray:
        """Apply release pitch behavior (falls)."""
        if len(contour) < 3:
            return contour

        # Release duration (last 10-20% of note)
        release_frames = max(1, int(len(contour) * np.random.uniform(0.1, 0.2)))

        # Fall amount (usually downward)
        fall_cents = np.random.uniform(-50, -10)
        fall_ratio = self._cents_to_ratio(fall_cents)

        # Generate fall curve
        release_curve = np.linspace(1.0, fall_ratio, release_frames)

        # Apply to contour
        contour[-release_frames:] *= release_curve

        return contour

    def constrain_to_range(
        self,
        contour: np.ndarray,
        allow_strain: bool = False,
    ) -> np.ndarray:
        """
        Constrain pitch contour to vocalist's range.

        Args:
            contour: Pitch contour in Hz
            allow_strain: Allow extended range with strain

        Returns:
            Constrained contour
        """
        if allow_strain:
            low_hz = self._midi_to_hz(self.identity.vocal_range.extended_low)
            high_hz = self._midi_to_hz(self.identity.vocal_range.extended_high)
        else:
            low_hz = self._midi_to_hz(self.identity.vocal_range.comfortable_low)
            high_hz = self._midi_to_hz(self.identity.vocal_range.comfortable_high)

        return np.clip(contour, low_hz, high_hz)


class PitchContourGenerator:
    """
    Higher-level pitch contour generator for phrases.
    """

    def __init__(self, identity: VocalIdentity):
        """Initialize generator."""
        self.controller = PitchController(identity)
        self.identity = identity

    def generate_phrase_contour(
        self,
        units: List,  # List[AlignedUnit]
        tempo: float,
        genre: str = "pop",
    ) -> List[np.ndarray]:
        """
        Generate pitch contours for an entire phrase.

        Ensures smooth transitions between notes.
        """
        contours = []

        for i, unit in enumerate(units):
            context = PitchContext(
                prev_pitch=units[i-1].pitch if i > 0 else None,
                next_pitch=units[i+1].pitch if i < len(units)-1 else None,
                is_phrase_start=(i == 0),
                is_phrase_end=(i == len(units)-1),
                genre=genre,
            )

            contour = self.controller.generate_pitch_contour(
                unit, tempo, genre, context
            )
            contours.append(contour)

        return contours
