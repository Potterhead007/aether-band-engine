"""
VocalSpec - Vocal performance parameters.

Purpose: Defines vocal persona, performance style, and vocal arrangement.
CRITICAL: Uses parametric voice design only - NO voice cloning.
"""

from typing import Optional

from pydantic import Field, field_validator

from aether.schemas.base import (
    AetherBaseModel,
    IdentifiableModel,
    NoteName,
    SectionType,
)


class VoicePersona(AetherBaseModel):
    """
    Parametric voice design.

    CRITICAL: This is NOT voice cloning. All parameters are abstract
    descriptions that will be interpreted by the vocal provider.
    """

    # Fundamental characteristics
    gender_presentation: str = Field(description="masculine, feminine, androgynous")
    age_range: str = Field(description="young, adult, mature")
    vocal_weight: str = Field(description="light, medium, heavy")

    # Timbral qualities
    brightness: float = Field(ge=0.0, le=1.0, default=0.5, description="Dark to bright spectrum")
    breathiness: float = Field(ge=0.0, le=1.0, default=0.2, description="Amount of air in voice")
    nasality: float = Field(ge=0.0, le=1.0, default=0.3, description="Nasal resonance")
    vibrato_depth: float = Field(ge=0.0, le=1.0, default=0.3, description="Vibrato intensity")
    vibrato_rate: float = Field(ge=3.0, le=8.0, default=5.5, description="Vibrato frequency in Hz")

    # Range
    lowest_note: NoteName
    highest_note: NoteName
    comfortable_low: NoteName
    comfortable_high: NoteName

    @field_validator("gender_presentation")
    @classmethod
    def validate_no_artist_reference(cls, v: str) -> str:
        """Ensure no real artist references."""
        banned = ["like", "similar to", "sound like", "voice of"]
        v_lower = v.lower()
        for phrase in banned:
            if phrase in v_lower:
                raise ValueError(f"Cannot use '{phrase}' - no artist references allowed")
        return v


class VocalDouble(AetherBaseModel):
    """Vocal double/layer specification."""

    name: str = Field(description="e.g., 'main_double', 'whisper_layer'")
    offset_cents: int = Field(ge=-50, le=50, default=0, description="Pitch offset for thickening")
    delay_ms: float = Field(ge=0.0, le=50.0, default=15.0, description="Timing offset")
    level_db: float = Field(ge=-20.0, le=0.0, default=-6.0, description="Relative level")
    pan: float = Field(ge=-1.0, le=1.0, default=0.0, description="-1=left, 0=center, 1=right")


class VocalHarmony(AetherBaseModel):
    """Backing vocal harmony specification."""

    interval: str = Field(description="3rd, 5th, octave, etc.")
    direction: str = Field(description="above or below lead")
    sections: list[SectionType] = Field(min_length=1)
    level_db: float = Field(ge=-20.0, le=0.0, default=-8.0)


class AdLib(AetherBaseModel):
    """Ad-lib specification."""

    type: str = Field(description="yeah, uh, oh, hey, etc.")
    placement: str = Field(description="Description of where it occurs")
    energy: float = Field(ge=0.0, le=1.0, default=0.7)


class EmotionMarker(AetherBaseModel):
    """Emotional intensity at a point in the song."""

    section: SectionType
    emotion: str = Field(description="joy, sadness, anger, tenderness, power, etc.")
    intensity: float = Field(ge=0.0, le=1.0)


class VocalSpec(IdentifiableModel):
    """
    Complete vocal specification for a song.

    Defines the voice persona, performance parameters, and vocal arrangement.
    CRITICAL: Uses parametric design only - NO voice cloning or artist references.
    """

    # Reference
    song_id: str = Field(description="Reference to parent SongSpec")
    lyric_id: str = Field(description="Reference to LyricSpec")
    melody_id: str = Field(description="Reference to MelodySpec")

    # Voice
    voice_persona: VoicePersona

    # Doubles and layers
    doubles: list[VocalDouble] = Field(default_factory=list)
    harmonies: list[VocalHarmony] = Field(default_factory=list)

    # Ad-libs
    ad_libs: list[AdLib] = Field(default_factory=list)

    # Emotional arc
    emotion_arc: list[EmotionMarker] = Field(min_length=1)

    # Performance style
    delivery_style: str = Field(
        default="sung", description="sung, rapped, spoken, whispered, belted"
    )
    articulation: str = Field(default="clear", description="clear, lazy, staccato, legato")

    # Processing hints
    autotune_amount: float = Field(
        ge=0.0, le=1.0, default=0.3, description="0=natural, 1=hard tuned"
    )
    reverb_send: float = Field(ge=0.0, le=1.0, default=0.3, description="Reverb amount")
    delay_send: float = Field(ge=0.0, le=1.0, default=0.2, description="Delay amount")

    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "example-song-id",
                "lyric_id": "example-lyric-id",
                "melody_id": "example-melody-id",
                "voice_persona": {
                    "gender_presentation": "feminine",
                    "age_range": "adult",
                    "vocal_weight": "medium",
                    "brightness": 0.6,
                    "breathiness": 0.3,
                    "vibrato_depth": 0.4,
                    "lowest_note": "G",
                    "highest_note": "E",
                    "comfortable_low": "C",
                    "comfortable_high": "C",
                },
                "emotion_arc": [
                    {"section": "verse", "emotion": "contemplative", "intensity": 0.4},
                    {"section": "chorus", "emotion": "joy", "intensity": 0.8},
                ],
                "delivery_style": "sung",
            }
        }
