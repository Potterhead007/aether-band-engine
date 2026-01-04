"""
MelodySpec - Melodic content specification.

Purpose: Defines all melodic content including hooks, motifs, and section melodies.
"""

from typing import Optional

from pydantic import Field

from aether.schemas.base import (
    AetherBaseModel,
    IdentifiableModel,
    NoteName,
    SectionType,
)


class MelodicInterval(AetherBaseModel):
    """A melodic interval in a phrase."""

    interval: str = Field(description="Interval name (m2, M2, m3, M3, P4, P5, etc.)")
    direction: str = Field(description="up, down, or same")
    duration_beats: float = Field(ge=0.0)


class MelodicContour(AetherBaseModel):
    """Shape description of a melody."""

    contour_type: str = Field(
        description="arch, ascending, descending, wave, static"
    )
    peak_position: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Position of melodic peak (0-1) for arch contours",
    )
    range_semitones: int = Field(ge=1, le=36)


class Motif(AetherBaseModel):
    """A short melodic idea that recurs throughout the song."""

    name: str = Field(description="Identifier for this motif")
    notes: list[NoteName] = Field(min_length=2)
    rhythm_pattern: list[float] = Field(
        min_length=2, description="Note durations in beats"
    )
    intervals: list[MelodicInterval] = Field(default_factory=list)
    contour: MelodicContour
    is_hook: bool = Field(default=False)
    hook_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Memorability score"
    )


class Hook(AetherBaseModel):
    """Primary hook - the most memorable melodic element."""

    melody_notes: list[NoteName] = Field(min_length=2)
    rhythm_pattern: list[float] = Field(min_length=2)
    lyrics_snippet: Optional[str] = Field(default=None, max_length=50)
    placement: SectionType = Field(default=SectionType.CHORUS)
    contour: MelodicContour
    singability_score: float = Field(ge=0.0, le=1.0, default=0.8)
    memorability_score: float = Field(ge=0.0, le=1.0, default=0.8)


class SectionMelody(AetherBaseModel):
    """Melody for a specific song section."""

    section_type: SectionType
    phrases: list[list[NoteName]] = Field(min_length=1)
    phrase_rhythms: list[list[float]] = Field(min_length=1)
    contour: MelodicContour
    uses_motifs: list[str] = Field(
        default_factory=list, description="Motif names used in this section"
    )
    register: str = Field(
        default="mid", description="low, mid, high - vocal/instrument register"
    )


class MelodySpec(IdentifiableModel):
    """
    Complete melodic specification for a song.

    Defines hooks, motifs, and section-by-section melodies with
    contour and interval analysis for originality verification.
    """

    # Reference
    song_id: str = Field(description="Reference to parent SongSpec")
    harmony_id: str = Field(description="Reference to HarmonySpec for alignment")

    # Core melodic elements
    primary_hook: Hook
    secondary_hooks: list[Hook] = Field(default_factory=list, max_length=3)
    motifs: list[Motif] = Field(min_length=1)

    # Section melodies
    section_melodies: list[SectionMelody] = Field(min_length=1)

    # Constraints
    lowest_note: NoteName
    highest_note: NoteName
    typical_range_octaves: float = Field(ge=0.5, le=3.0, default=1.5)

    # Development techniques used
    development_techniques: list[str] = Field(
        default_factory=list,
        description="Techniques: sequence, inversion, retrograde, augmentation, etc.",
    )

    # Originality
    interval_hash: Optional[str] = Field(
        default=None, description="Hash of interval sequence for plagiarism check"
    )
    contour_fingerprint: Optional[str] = Field(
        default=None, description="Encoded contour for similarity check"
    )
    originality_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Melody originality score"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "example-song-id",
                "harmony_id": "example-harmony-id",
                "primary_hook": {
                    "melody_notes": ["C", "E", "G", "A", "G"],
                    "rhythm_pattern": [0.5, 0.5, 0.5, 0.25, 0.25],
                    "placement": "chorus",
                    "contour": {
                        "contour_type": "arch",
                        "peak_position": 0.6,
                        "range_semitones": 9,
                    },
                    "singability_score": 0.9,
                    "memorability_score": 0.85,
                },
                "motifs": [
                    {
                        "name": "main_motif",
                        "notes": ["C", "E", "G"],
                        "rhythm_pattern": [0.5, 0.5, 1.0],
                        "contour": {
                            "contour_type": "ascending",
                            "range_semitones": 7,
                        },
                        "is_hook": True,
                    }
                ],
                "lowest_note": "C",
                "highest_note": "G",
            }
        }
