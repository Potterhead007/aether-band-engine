"""
ArrangementSpec - Song structure and instrumentation.

Purpose: Defines song structure, energy curve, and instrumentation per section.
"""

from typing import Optional

from pydantic import Field

from aether.schemas.base import (
    AetherBaseModel,
    EnergyLevel,
    IdentifiableModel,
    SectionType,
)


class Instrument(AetherBaseModel):
    """An instrument in the arrangement."""

    name: str = Field(description="Instrument name")
    category: str = Field(
        description="drums, bass, keys, synth, guitar, strings, brass, vocal, fx"
    )
    role: str = Field(description="lead, rhythm, pad, texture, accent")
    midi_program: Optional[int] = Field(default=None, ge=0, le=127)
    is_essential: bool = Field(default=False)


class SectionDefinition(AetherBaseModel):
    """Definition of a song section."""

    section_type: SectionType
    label: str = Field(description="e.g., 'Verse 1', 'Chorus 2'")
    start_bar: int = Field(ge=1)
    length_bars: int = Field(ge=1)
    energy_level: EnergyLevel
    instruments: list[str] = Field(
        min_length=1, description="Instrument names active in this section"
    )
    dynamics: str = Field(
        default="mf", description="Dynamic marking (pp, p, mp, mf, f, ff)"
    )


class Transition(AetherBaseModel):
    """Transition between sections."""

    from_section: str = Field(description="Label of source section")
    to_section: str = Field(description="Label of target section")
    technique: str = Field(
        description="fill, riser, drop, filter_sweep, silence, crash, reverse"
    )
    duration_beats: float = Field(ge=0.0, default=4.0)
    has_fill: bool = Field(default=False)


class EnergyPoint(AetherBaseModel):
    """A point on the energy curve."""

    position_percent: float = Field(ge=0.0, le=100.0)
    energy_level: EnergyLevel
    section_label: str


class ArrangementSpec(IdentifiableModel):
    """
    Complete arrangement specification for a song.

    Defines the structure, energy curve, instrumentation, and transitions.
    """

    # Reference
    song_id: str = Field(description="Reference to parent SongSpec")

    # Instrumentation
    instruments: list[Instrument] = Field(min_length=1)

    # Structure
    sections: list[SectionDefinition] = Field(min_length=1)
    total_bars: int = Field(ge=1)
    total_duration_seconds: float = Field(ge=0.0)

    # Energy curve
    energy_curve: list[EnergyPoint] = Field(min_length=2)
    energy_curve_type: str = Field(
        default="build_release",
        description="build, maintain, build_release, wave, climactic",
    )

    # Transitions
    transitions: list[Transition] = Field(default_factory=list)

    # Ear candy
    ear_candy: list[str] = Field(
        default_factory=list,
        description="Special production elements: risers, impacts, sweeps",
    )

    # Validation
    structure_archetype: Optional[str] = Field(
        default=None,
        description="Structure pattern e.g., 'intro-verse-chorus-verse-chorus-bridge-chorus-outro'",
    )
    genre_structure_match: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="How well structure matches genre norms"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "example-song-id",
                "instruments": [
                    {"name": "kick", "category": "drums", "role": "rhythm", "is_essential": True},
                    {"name": "bass_synth", "category": "synth", "role": "rhythm", "is_essential": True},
                    {"name": "lead_synth", "category": "synth", "role": "lead"},
                    {"name": "pad", "category": "synth", "role": "pad"},
                ],
                "sections": [
                    {
                        "section_type": "intro",
                        "label": "Intro",
                        "start_bar": 1,
                        "length_bars": 8,
                        "energy_level": "low",
                        "instruments": ["pad"],
                    },
                    {
                        "section_type": "verse",
                        "label": "Verse 1",
                        "start_bar": 9,
                        "length_bars": 16,
                        "energy_level": "medium",
                        "instruments": ["kick", "bass_synth", "pad"],
                    },
                ],
                "total_bars": 120,
                "total_duration_seconds": 240.0,
                "energy_curve": [
                    {"position_percent": 0, "energy_level": "low", "section_label": "Intro"},
                    {"position_percent": 50, "energy_level": "high", "section_label": "Chorus"},
                    {"position_percent": 100, "energy_level": "medium", "section_label": "Outro"},
                ],
            }
        }
