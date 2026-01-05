"""
HarmonySpec - Harmonic content specification.

Purpose: Defines all harmonic content for a song including progressions,
voice leading, and modulations.
"""

from typing import Optional

from pydantic import Field, field_validator

from aether.schemas.base import (
    AetherBaseModel,
    IdentifiableModel,
    KeySignature,
    NoteName,
    SectionType,
)


class ChordVoicing(AetherBaseModel):
    """Specific voicing for a chord."""

    root: NoteName
    quality: str = Field(description="Chord quality (maj, min, dim, aug, 7, maj7, etc.)")
    bass_note: Optional[NoteName] = Field(default=None, description="Slash chord bass")
    extensions: list[str] = Field(default_factory=list, description="Extensions (9, 11, 13)")
    alterations: list[str] = Field(default_factory=list, description="Alterations (b5, #9)")

    def __str__(self) -> str:
        base = f"{self.root.value}{self.quality}"
        if self.extensions:
            base += "".join(self.extensions)
        if self.alterations:
            base += f"({','.join(self.alterations)})"
        if self.bass_note:
            base += f"/{self.bass_note.value}"
        return base


class ChordProgression(AetherBaseModel):
    """A chord progression for a song section."""

    section_type: SectionType
    chords: list[ChordVoicing] = Field(min_length=1)
    roman_numerals: list[str] = Field(
        min_length=1, description="Roman numeral analysis (I, ii, V7, etc.)"
    )
    durations_beats: list[float] = Field(
        min_length=1, description="Duration of each chord in beats"
    )
    repeat_count: int = Field(default=1, ge=1)

    @field_validator("durations_beats")
    @classmethod
    def validate_durations_match_chords(cls, v: list[float], info) -> list[float]:
        if "chords" in info.data and len(v) != len(info.data["chords"]):
            raise ValueError("Duration count must match chord count")
        return v


class VoiceLeadingRule(AetherBaseModel):
    """Voice leading validation rule."""

    name: str
    description: str
    is_violated: bool = Field(default=False)
    violation_details: Optional[str] = Field(default=None)


class Modulation(AetherBaseModel):
    """Key change specification."""

    from_key: KeySignature
    to_key: KeySignature
    pivot_chord: Optional[ChordVoicing] = Field(default=None)
    technique: str = Field(
        default="direct",
        description="Modulation technique (direct, pivot, chromatic, etc.)",
    )
    at_section: SectionType
    at_bar: int = Field(ge=1)


class HarmonySpec(IdentifiableModel):
    """
    Complete harmonic specification for a song.

    Defines the harmonic content including progressions for each section,
    voice leading rules, and any modulations.
    """

    # Reference
    song_id: str = Field(description="Reference to parent SongSpec")

    # Key Information
    primary_key: KeySignature
    secondary_keys: list[KeySignature] = Field(
        default_factory=list, description="Keys used in modulations"
    )

    # Progressions
    progressions: list[ChordProgression] = Field(min_length=1)

    # Harmonic characteristics
    tension_level: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Overall harmonic tension (0=consonant, 1=tense)"
    )
    jazz_influence: float = Field(
        ge=0.0, le=1.0, default=0.0, description="Jazz harmony influence level"
    )
    modal_interchange_used: bool = Field(default=False)

    # Modulations
    modulations: list[Modulation] = Field(default_factory=list)

    # Voice Leading
    voice_leading_rules: list[VoiceLeadingRule] = Field(default_factory=list)

    # Validation scores
    originality_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Harmonic originality score"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "example-song-id",
                "primary_key": {"root": "C", "mode": "minor"},
                "progressions": [
                    {
                        "section_type": "verse",
                        "chords": [
                            {"root": "C", "quality": "min"},
                            {"root": "A#", "quality": "maj"},
                            {"root": "G#", "quality": "maj"},
                            {"root": "G", "quality": "maj"},
                        ],
                        "roman_numerals": ["i", "VII", "VI", "V"],
                        "durations_beats": [4.0, 4.0, 4.0, 4.0],
                    }
                ],
                "tension_level": 0.4,
            }
        }
