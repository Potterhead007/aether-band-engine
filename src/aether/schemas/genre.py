"""
GenreRootProfile - Complete genre definition.

Purpose: Comprehensive genre profile including history, musical parameters,
production characteristics, and authenticity scoring rubric.

This is the most critical schema - it defines what makes a genre authentic.
"""

from typing import Optional

from pydantic import Field, field_validator

from aether.schemas.base import (
    AetherBaseModel,
    DurationRange,
    IdentifiableModel,
    Mode,
    TempoRange,
)


# Genealogy


class GenreLineage(AetherBaseModel):
    """Genre family tree."""

    ancestors: list[str] = Field(description="Parent genres")
    influences: list[str] = Field(default_factory=list, description="Adjacent influences")
    descendants: list[str] = Field(default_factory=list, description="Child genres")
    siblings: list[str] = Field(default_factory=list, description="Parallel developments")


class HistoricalContext(AetherBaseModel):
    """Historical emergence context."""

    emergence_era: str = Field(description="e.g., 'late 1980s'")
    emergence_year: int = Field(ge=1900, le=2030)
    geographic_origin: str
    cultural_context: str
    socioeconomic_factors: list[str] = Field(default_factory=list)
    key_innovations: list[str] = Field(min_length=1)


class EvolutionPeriod(AetherBaseModel):
    """A period in the genre's evolution."""

    period_name: str
    years: str = Field(description="e.g., '1987-1992'")
    characteristics: list[str]
    production_norms: dict[str, str] = Field(default_factory=dict)


# Musical Parameters


class RhythmProfile(AetherBaseModel):
    """Rhythmic characteristics of the genre."""

    time_signatures: list[str] = Field(min_length=1)
    feels: list[str] = Field(min_length=1, description="straight, swing, shuffle")
    swing_amount_min: float = Field(ge=0.0, le=1.0)
    swing_amount_max: float = Field(ge=0.0, le=1.0)
    swing_amount_typical: float = Field(ge=0.0, le=1.0)
    signature_patterns: list[str] = Field(
        default_factory=list, description="e.g., 'four_on_floor', 'boom_bap'"
    )
    drum_characteristics: dict[str, str] = Field(
        default_factory=dict, description="kick, snare, hihat descriptions"
    )


class HarmonyProfile(AetherBaseModel):
    """Harmonic characteristics of the genre."""

    common_modes: list[Mode]
    typical_progressions: list[str] = Field(min_length=1, description="Roman numeral progressions")
    tension_level: float = Field(ge=0.0, le=1.0)
    jazz_influence: float = Field(ge=0.0, le=1.0, default=0.0)
    modal_interchange_common: bool = Field(default=False)


class MelodyProfile(AetherBaseModel):
    """Melodic characteristics of the genre."""

    typical_range_octaves: float = Field(ge=0.5, le=3.0)
    interval_vocabulary: list[str] = Field(description="Common intervals")
    contour_preferences: list[str] = Field(description="arch, ascending, etc.")
    phrase_lengths: list[int] = Field(description="Typical phrase lengths in bars")


class ArrangementProfile(AetherBaseModel):
    """Arrangement characteristics of the genre."""

    typical_duration: DurationRange
    common_structures: list[str] = Field(
        min_length=1, description="e.g., 'intro-verse-chorus-verse-chorus-bridge-chorus-outro'"
    )
    energy_curve_type: str = Field(description="build, maintain, build_release, wave")


class InstrumentationProfile(AetherBaseModel):
    """Instrumentation characteristics."""

    essential: list[str] = Field(min_length=1)
    common: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list, description="Anachronistic instruments")


# Production


class MixCharacteristics(AetherBaseModel):
    """Mix aesthetic targets."""

    low_end_emphasis: float = Field(ge=0.0, le=1.0)
    vocal_forwardness: float = Field(ge=0.0, le=1.0)
    brightness: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    vintage_warmth: float = Field(ge=0.0, le=1.0, default=0.0)


class MasterTargets(AetherBaseModel):
    """Mastering targets for the genre."""

    loudness_lufs_min: float = Field(ge=-20.0, le=-6.0)
    loudness_lufs_max: float = Field(ge=-20.0, le=-6.0)
    dynamic_range_lu_min: float = Field(ge=3.0, le=15.0)
    dynamic_range_lu_max: float = Field(ge=3.0, le=15.0)


class ProductionProfile(AetherBaseModel):
    """Production aesthetic for the genre."""

    era_reference: str = Field(description="e.g., '1990s'")
    mix_characteristics: MixCharacteristics
    master_targets: MasterTargets
    signature_effects: list[str] = Field(
        default_factory=list, description="e.g., 'gated reverb', 'tape saturation'"
    )


# Authenticity Rubric


class RubricDimension(AetherBaseModel):
    """A single scoring dimension in the authenticity rubric."""

    name: str
    weight: float = Field(ge=0.0, le=1.0)
    criteria: list[str] = Field(min_length=1)
    scoring_guide: str


class AuthenticityRubric(AetherBaseModel):
    """Complete scoring rubric for genre authenticity."""

    dimensions: list[RubricDimension] = Field(min_length=3)
    minimum_passing_score: float = Field(ge=0.5, le=1.0, default=0.8)

    @field_validator("dimensions")
    @classmethod
    def validate_weights_sum(cls, v: list[RubricDimension]) -> list[RubricDimension]:
        """Ensure weights sum to approximately 1.0."""
        total = sum(d.weight for d in v)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Dimension weights must sum to 1.0, got {total}")
        return v


# Main Schema


class GenreRootProfile(IdentifiableModel):
    """
    Complete genre definition.

    This is the authoritative source for all genre-specific parameters,
    production norms, and authenticity evaluation criteria.

    Every genre MUST have a complete profile to ensure authentic output.
    """

    # Identity
    genre_id: str = Field(description="Unique identifier (e.g., 'hip-hop-boom-bap')")
    name: str = Field(description="Display name")
    aliases: list[str] = Field(default_factory=list)

    # Genealogy
    lineage: GenreLineage
    historical_context: HistoricalContext
    evolution_timeline: list[EvolutionPeriod] = Field(default_factory=list)

    # Musical Parameters
    tempo: TempoRange
    rhythm: RhythmProfile
    harmony: HarmonyProfile
    melody: MelodyProfile
    arrangement: ArrangementProfile
    instrumentation: InstrumentationProfile

    # Production
    production: ProductionProfile

    # Authenticity
    authenticity_rubric: AuthenticityRubric

    # Metadata
    version: str = Field(default="1.0.0")
    last_validated: Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "genre_id": "hip-hop-boom-bap",
                "name": "Boom Bap",
                "aliases": ["East Coast Hip-Hop", "Golden Age Hip-Hop"],
                "lineage": {
                    "ancestors": ["funk", "soul", "jazz"],
                    "influences": ["disco", "reggae"],
                    "descendants": ["abstract-hip-hop", "conscious-hip-hop"],
                    "siblings": ["g-funk"],
                },
                "historical_context": {
                    "emergence_era": "late 1980s",
                    "emergence_year": 1987,
                    "geographic_origin": "New York City",
                    "cultural_context": "Post-golden age hip-hop",
                    "key_innovations": ["Chopped samples", "MPC workflow"],
                },
                "tempo": {"min_bpm": 80, "max_bpm": 100, "typical_bpm": 90},
                "rhythm": {
                    "time_signatures": ["4/4"],
                    "feels": ["straight", "slight_swing"],
                    "swing_amount_min": 0.0,
                    "swing_amount_max": 0.2,
                    "swing_amount_typical": 0.08,
                    "signature_patterns": ["boom_bap"],
                },
                "harmony": {
                    "common_modes": ["minor", "dorian"],
                    "typical_progressions": ["i-VII-VI-VII"],
                    "tension_level": 0.4,
                },
                "melody": {
                    "typical_range_octaves": 1.5,
                    "interval_vocabulary": ["m2", "M2", "m3", "P4", "P5"],
                    "contour_preferences": ["descending", "wave"],
                    "phrase_lengths": [2, 4],
                },
                "arrangement": {
                    "typical_duration": {"min_seconds": 180, "max_seconds": 300},
                    "common_structures": ["intro-verse-chorus-verse-chorus-verse-outro"],
                    "energy_curve_type": "maintain",
                },
                "instrumentation": {
                    "essential": ["drums", "bass", "samples"],
                    "common": ["scratching", "piano"],
                    "forbidden": ["trap_808", "EDM_synths"],
                },
                "production": {
                    "era_reference": "1990s",
                    "mix_characteristics": {
                        "low_end_emphasis": 0.7,
                        "vocal_forwardness": 0.8,
                        "brightness": 0.4,
                        "width": 0.4,
                    },
                    "master_targets": {
                        "loudness_lufs_min": -12.0,
                        "loudness_lufs_max": -9.0,
                        "dynamic_range_lu_min": 6.0,
                        "dynamic_range_lu_max": 10.0,
                    },
                },
                "authenticity_rubric": {
                    "dimensions": [
                        {
                            "name": "Drum Sound",
                            "weight": 0.3,
                            "criteria": ["Punchy drums", "Vinyl texture"],
                            "scoring_guide": "5=classic, 1=wrong style",
                        },
                        {
                            "name": "Sample Aesthetic",
                            "weight": 0.25,
                            "criteria": ["Warm samples", "Proper chops"],
                            "scoring_guide": "5=authentic, 1=too modern",
                        },
                        {
                            "name": "Groove",
                            "weight": 0.25,
                            "criteria": ["80-100 BPM", "Head-nod factor"],
                            "scoring_guide": "5=perfect pocket, 1=wrong tempo",
                        },
                        {
                            "name": "Mix Character",
                            "weight": 0.2,
                            "criteria": ["Lo-fi warmth", "Not over-produced"],
                            "scoring_guide": "5=authentic, 1=too clean",
                        },
                    ],
                    "minimum_passing_score": 0.8,
                },
            }
        }
