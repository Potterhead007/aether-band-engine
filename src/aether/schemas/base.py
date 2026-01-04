"""
Base schema classes and common types for AETHER.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class AetherBaseModel(BaseModel):
    """Base model for all AETHER schemas with common configuration."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        # Note: NOT using use_enum_values=True so that enum fields
        # retain their type and .value can be called on them
    )


class TimestampedModel(AetherBaseModel):
    """Base model with automatic timestamps."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class IdentifiableModel(TimestampedModel):
    """Base model with UUID and timestamps."""

    id: UUID = Field(default_factory=uuid4)


# Common Enums


class NoteName(str, Enum):
    """Musical note names."""

    C = "C"
    C_SHARP = "C#"
    D = "D"
    D_SHARP = "D#"
    E = "E"
    F = "F"
    F_SHARP = "F#"
    G = "G"
    G_SHARP = "G#"
    A = "A"
    A_SHARP = "A#"
    B = "B"


class Mode(str, Enum):
    """Musical modes."""

    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"


class TimeSignature(str, Enum):
    """Common time signatures."""

    FOUR_FOUR = "4/4"
    THREE_FOUR = "3/4"
    TWO_FOUR = "2/4"
    SIX_EIGHT = "6/8"
    TWELVE_EIGHT = "12/8"
    FIVE_FOUR = "5/4"
    SEVEN_EIGHT = "7/8"


class SectionType(str, Enum):
    """Song section types."""

    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    POST_CHORUS = "post_chorus"
    BRIDGE = "bridge"
    BREAKDOWN = "breakdown"
    BUILD = "build"
    DROP = "drop"
    OUTRO = "outro"
    INTERLUDE = "interlude"
    SOLO = "solo"


class Feel(str, Enum):
    """Rhythmic feel types."""

    STRAIGHT = "straight"
    SWING = "swing"
    SHUFFLE = "shuffle"
    TRIPLET = "triplet"


class EnergyLevel(str, Enum):
    """Energy level descriptors."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM_LOW = "medium_low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MoodCategory(str, Enum):
    """Mood categories for tracks."""

    HAPPY = "happy"
    SAD = "sad"
    ENERGETIC = "energetic"
    CALM = "calm"
    AGGRESSIVE = "aggressive"
    MELANCHOLIC = "melancholic"
    UPLIFTING = "uplifting"
    DARK = "dark"
    NOSTALGIC = "nostalgic"
    ETHEREAL = "ethereal"
    PLAYFUL = "playful"
    INTENSE = "intense"


# Common Types


class KeySignature(AetherBaseModel):
    """Musical key signature."""

    root: NoteName
    mode: Mode

    def __str__(self) -> str:
        return f"{self.root.value} {self.mode.value}"


class TempoRange(AetherBaseModel):
    """BPM range specification."""

    min_bpm: int = Field(ge=20, le=300)
    max_bpm: int = Field(ge=20, le=300)
    typical_bpm: int = Field(ge=20, le=300)


class DurationRange(AetherBaseModel):
    """Duration range in seconds."""

    min_seconds: int = Field(ge=0)
    max_seconds: int = Field(ge=0)


class LoudnessTarget(AetherBaseModel):
    """Loudness targeting specification."""

    target_lufs: float = Field(ge=-30.0, le=0.0, default=-14.0)
    tolerance: float = Field(ge=0.0, le=3.0, default=0.5)


class TruePeakTarget(AetherBaseModel):
    """True peak ceiling specification."""

    ceiling_dbtp: float = Field(ge=-10.0, le=0.0, default=-1.0)


class DynamicRangeTarget(AetherBaseModel):
    """Dynamic range specification in LU."""

    minimum_lu: float = Field(ge=0.0, le=20.0, default=6.0)
    target_lu: float = Field(ge=0.0, le=20.0, default=8.0)
