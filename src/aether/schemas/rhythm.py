"""
RhythmSpec - Rhythmic patterns and groove.

Purpose: Defines rhythmic patterns, groove feel, and humanization parameters.
"""

from typing import Optional

from pydantic import Field

from aether.schemas.base import (
    AetherBaseModel,
    Feel,
    IdentifiableModel,
    SectionType,
    TimeSignature,
)


class DrumHit(AetherBaseModel):
    """A single drum hit in a pattern."""

    instrument: str = Field(description="kick, snare, hihat, tom, cymbal, etc.")
    position_beats: float = Field(ge=0.0)
    velocity: int = Field(ge=1, le=127, default=100)
    is_accent: bool = Field(default=False)
    is_ghost: bool = Field(default=False)


class DrumPattern(AetherBaseModel):
    """A drum pattern for a section or variation."""

    name: str
    length_bars: int = Field(ge=1, default=1)
    time_signature: TimeSignature = Field(default=TimeSignature.FOUR_FOUR)
    hits: list[DrumHit] = Field(min_length=1)
    feel: Feel = Field(default=Feel.STRAIGHT)


class GrooveTemplate(AetherBaseModel):
    """Overall groove characteristics."""

    feel: Feel
    swing_amount: float = Field(
        ge=0.0, le=1.0, default=0.0, description="0=straight, 0.66=triplet swing"
    )
    push_pull: float = Field(
        ge=-1.0, le=1.0, default=0.0, description="-1=behind, 0=on grid, 1=ahead"
    )
    pocket_tightness: float = Field(
        ge=0.0, le=1.0, default=0.8, description="Kick/bass alignment tightness"
    )


class Humanization(AetherBaseModel):
    """Humanization parameters for natural feel."""

    timing_variation_ms: float = Field(
        ge=0.0, le=50.0, default=5.0, description="Random timing offset range"
    )
    velocity_variation: int = Field(
        ge=0, le=30, default=10, description="Random velocity offset range"
    )
    swing_variation: float = Field(
        ge=0.0, le=0.1, default=0.02, description="Random swing amount variation"
    )


class SectionRhythm(AetherBaseModel):
    """Rhythm specification for a song section."""

    section_type: SectionType
    drum_pattern_name: str = Field(description="Reference to DrumPattern")
    groove_intensity: float = Field(
        ge=0.0, le=1.0, default=0.7, description="How busy/active the groove is"
    )
    fills_allowed: bool = Field(default=True)


class RhythmSpec(IdentifiableModel):
    """
    Complete rhythmic specification for a song.

    Defines groove templates, drum patterns, and humanization for natural feel.
    """

    # Reference
    song_id: str = Field(description="Reference to parent SongSpec")

    # Time
    time_signature: TimeSignature
    bpm: int = Field(ge=20, le=300)

    # Groove
    groove_template: GrooveTemplate
    humanization: Humanization

    # Patterns
    drum_patterns: list[DrumPattern] = Field(min_length=1)

    # Section assignments
    section_rhythms: list[SectionRhythm] = Field(min_length=1)

    # Signature elements
    signature_pattern: Optional[str] = Field(
        default=None, description="Genre-defining rhythm element (e.g., four_on_floor)"
    )

    # Originality
    groove_fingerprint: Optional[str] = Field(
        default=None, description="Pattern hash for similarity check"
    )
    originality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "example-song-id",
                "time_signature": "4/4",
                "bpm": 92,
                "groove_template": {
                    "feel": "straight",
                    "swing_amount": 0.08,
                    "push_pull": -0.1,
                    "pocket_tightness": 0.9,
                },
                "humanization": {
                    "timing_variation_ms": 8.0,
                    "velocity_variation": 15,
                    "swing_variation": 0.03,
                },
                "drum_patterns": [
                    {
                        "name": "main_beat",
                        "length_bars": 1,
                        "time_signature": "4/4",
                        "hits": [
                            {"instrument": "kick", "position_beats": 0.0, "velocity": 110},
                            {"instrument": "snare", "position_beats": 1.0, "velocity": 100},
                            {"instrument": "kick", "position_beats": 2.0, "velocity": 100},
                            {"instrument": "snare", "position_beats": 3.0, "velocity": 105},
                        ],
                        "feel": "straight",
                    }
                ],
                "signature_pattern": "boom_bap",
            }
        }
