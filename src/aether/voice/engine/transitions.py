"""
Transition Engine

Models transitions between notes with singing-appropriate behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class TransitionType(Enum):
    """Types of note transitions."""
    LEGATO = "legato"
    PORTAMENTO = "portamento"
    STACCATO = "staccato"
    BREATH = "breath"
    SCOOP = "scoop"
    FALL = "fall"


@dataclass
class TransitionSpec:
    """Specification for a transition type."""
    name: str
    description: str
    pitch_behavior: str  # glide, step, scoop_up, fall_off
    glide_duration_ratio: float = 0.0  # % of note duration
    note_shortening: float = 0.0  # How much to shorten the note
    breath_break: bool = False
    scoop_cents: Tuple[float, float] = (0, 0)
    fall_cents: Tuple[float, float] = (0, 0)
    duration_ms: Tuple[float, float] = (0, 0)


# Transition type definitions
TRANSITION_SPECS: Dict[TransitionType, TransitionSpec] = {
    TransitionType.LEGATO: TransitionSpec(
        name="legato",
        description="Smooth, connected transition",
        pitch_behavior="glide",
        glide_duration_ratio=0.15,
    ),
    TransitionType.PORTAMENTO: TransitionSpec(
        name="portamento",
        description="Deliberate pitch slide",
        pitch_behavior="slide",
        glide_duration_ratio=0.25,
    ),
    TransitionType.STACCATO: TransitionSpec(
        name="staccato",
        description="Separated, punctuated",
        pitch_behavior="step",
        note_shortening=0.3,
    ),
    TransitionType.BREATH: TransitionSpec(
        name="breath",
        description="Breath-separated phrases",
        pitch_behavior="step",
        breath_break=True,
        duration_ms=(150, 400),
    ),
    TransitionType.SCOOP: TransitionSpec(
        name="scoop",
        description="Approach from below",
        pitch_behavior="scoop_up",
        scoop_cents=(-100, -30),
        duration_ms=(30, 80),
    ),
    TransitionType.FALL: TransitionSpec(
        name="fall",
        description="Release downward",
        pitch_behavior="fall_off",
        fall_cents=(-200, -50),
        duration_ms=(50, 150),
    ),
}


# Genre-specific transition preferences
GENRE_TRANSITION_STYLE: Dict[str, Dict] = {
    "pop": {
        "default": TransitionType.LEGATO,
        "portamento_threshold": 5,  # Semitones
        "allows_portamento": True,
        "phrase_break_threshold": 0.5,  # Beats
    },
    "r-and-b": {
        "default": TransitionType.LEGATO,
        "portamento_threshold": 3,
        "allows_portamento": True,
        "phrase_break_threshold": 0.4,
    },
    "rock": {
        "default": TransitionType.STACCATO,
        "portamento_threshold": 7,
        "allows_portamento": False,
        "phrase_break_threshold": 0.3,
    },
    "jazz": {
        "default": TransitionType.LEGATO,
        "portamento_threshold": 4,
        "allows_portamento": True,
        "phrase_break_threshold": 0.5,
    },
    "house": {
        "default": TransitionType.STACCATO,
        "portamento_threshold": 6,
        "allows_portamento": False,
        "phrase_break_threshold": 0.25,
    },
    "trap": {
        "default": TransitionType.STACCATO,
        "portamento_threshold": 5,
        "allows_portamento": True,
        "phrase_break_threshold": 0.3,
    },
    "funk": {
        "default": TransitionType.STACCATO,
        "portamento_threshold": 4,
        "allows_portamento": False,
        "phrase_break_threshold": 0.25,
    },
    "ambient": {
        "default": TransitionType.LEGATO,
        "portamento_threshold": 2,
        "allows_portamento": True,
        "phrase_break_threshold": 1.0,
    },
}


class TransitionEngine:
    """
    Selects and generates appropriate transitions between notes.
    """

    def __init__(self):
        """Initialize transition engine."""
        pass

    def select_transition(
        self,
        prev_unit,  # AlignedUnit
        next_unit,  # AlignedUnit
        genre: str = "pop",
    ) -> TransitionType:
        """
        Select appropriate transition type based on context.

        Args:
            prev_unit: Previous aligned unit
            next_unit: Next aligned unit
            genre: Genre for style

        Returns:
            Selected transition type
        """
        style = GENRE_TRANSITION_STYLE.get(genre, GENRE_TRANSITION_STYLE["pop"])

        # Calculate interval and time gap
        interval = abs(next_unit.pitch - prev_unit.pitch)
        time_gap = next_unit.start_beat - (prev_unit.start_beat + prev_unit.duration_beats)

        # Check for phrase boundary
        if time_gap > style["phrase_break_threshold"]:
            return TransitionType.BREATH

        # Check for large intervals with portamento
        if interval > style["portamento_threshold"] and style["allows_portamento"]:
            return TransitionType.PORTAMENTO

        # Check for phrase end (might want fall)
        if hasattr(prev_unit, 'is_phrase_end') and prev_unit.is_phrase_end:
            if np.random.random() < 0.3:  # 30% chance
                return TransitionType.FALL

        # Check for phrase start (might want scoop)
        if hasattr(next_unit, 'is_phrase_start') and next_unit.is_phrase_start:
            if np.random.random() < 0.3:
                return TransitionType.SCOOP

        # Default to genre preference
        return style["default"]

    def generate_transition(
        self,
        transition_type: TransitionType,
        prev_pitch_hz: float,
        next_pitch_hz: float,
        duration_frames: int,
        frame_rate: float = 100.0,
    ) -> np.ndarray:
        """
        Generate pitch transition curve.

        Args:
            transition_type: Type of transition
            prev_pitch_hz: Previous note pitch
            next_pitch_hz: Next note pitch
            duration_frames: Transition duration in frames
            frame_rate: Frame rate

        Returns:
            Pitch curve in Hz
        """
        spec = TRANSITION_SPECS[transition_type]

        if spec.pitch_behavior == "step":
            # Immediate step
            curve = np.full(duration_frames, next_pitch_hz)

        elif spec.pitch_behavior == "glide":
            # Linear glide
            curve = np.linspace(prev_pitch_hz, next_pitch_hz, duration_frames)

        elif spec.pitch_behavior == "slide":
            # S-curve slide (smoother)
            t = np.linspace(0, 1, duration_frames)
            # Sigmoid-like curve
            s = t * t * (3 - 2 * t)
            curve = prev_pitch_hz + (next_pitch_hz - prev_pitch_hz) * s

        elif spec.pitch_behavior == "scoop_up":
            # Start below target, curve up
            scoop_amount = np.random.uniform(*spec.scoop_cents)
            scoop_ratio = 2 ** (scoop_amount / 1200)
            start_hz = next_pitch_hz * scoop_ratio

            t = np.linspace(0, 1, duration_frames)
            curve = start_hz + (next_pitch_hz - start_hz) * t

        elif spec.pitch_behavior == "fall_off":
            # Start at pitch, fall down
            fall_amount = np.random.uniform(*spec.fall_cents)
            fall_ratio = 2 ** (fall_amount / 1200)
            end_hz = prev_pitch_hz * fall_ratio

            t = np.linspace(0, 1, duration_frames)
            curve = prev_pitch_hz + (end_hz - prev_pitch_hz) * t

        else:
            curve = np.full(duration_frames, next_pitch_hz)

        return curve

    def get_note_shortening(
        self,
        transition_type: TransitionType,
    ) -> float:
        """Get how much to shorten a note for this transition."""
        spec = TRANSITION_SPECS[transition_type]
        return spec.note_shortening

    def needs_breath(self, transition_type: TransitionType) -> bool:
        """Check if transition requires a breath."""
        spec = TRANSITION_SPECS[transition_type]
        return spec.breath_break

    def get_breath_duration_ms(
        self,
        transition_type: TransitionType,
    ) -> float:
        """Get breath duration for breath transitions."""
        spec = TRANSITION_SPECS[transition_type]
        if spec.breath_break:
            return np.random.uniform(*spec.duration_ms)
        return 0.0
