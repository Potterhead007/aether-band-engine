"""
SongSpec - Complete song specification schema.

Purpose: Complete song specification that initiates the pipeline.
"""

from typing import Optional
from uuid import UUID

from pydantic import Field

from aether.schemas.base import (
    IdentifiableModel,
    KeySignature,
    MoodCategory,
    TimeSignature,
)


class CreativeBrief(IdentifiableModel):
    """Creative direction for a song."""

    theme: str = Field(description="Main theme or concept")
    mood: MoodCategory
    energy_description: str = Field(description="Energy arc description")
    lyrical_direction: Optional[str] = Field(
        default=None, description="Direction for lyrics if applicable"
    )
    reference_era: Optional[str] = Field(
        default=None, description="Production era reference (e.g., '1990s')"
    )
    special_requests: list[str] = Field(
        default_factory=list, description="Special production requests"
    )


class SongSpec(IdentifiableModel):
    """
    Complete song specification.

    This is the primary input to the AETHER pipeline, defining all
    high-level parameters for a track.
    """

    # Identity
    title: str = Field(min_length=1, max_length=200)
    artist_name: str = Field(default="AETHER", min_length=1, max_length=100)

    # Genre
    genre_id: str = Field(description="Reference to GenreRootProfile ID")
    subgenre: Optional[str] = Field(default=None)

    # Musical Parameters
    bpm: int = Field(ge=20, le=300)
    key: KeySignature
    time_signature: TimeSignature = Field(default=TimeSignature.FOUR_FOUR)

    # Duration
    target_duration_seconds: int = Field(ge=60, le=600, default=210)

    # Creative Direction
    creative_brief: CreativeBrief

    # Mood and Energy
    primary_mood: MoodCategory
    secondary_moods: list[MoodCategory] = Field(default_factory=list, max_length=3)

    # Flags
    has_vocals: bool = Field(default=True)
    is_instrumental: bool = Field(default=False)

    # Reproducibility
    random_seed: Optional[int] = Field(default=None)

    # Lineage
    album_id: Optional[UUID] = Field(default=None)
    track_number: Optional[int] = Field(default=None, ge=1)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Midnight Protocol",
                "artist_name": "Neon Circuit",
                "genre_id": "synthwave",
                "bpm": 118,
                "key": {"root": "A", "mode": "minor"},
                "time_signature": "4/4",
                "target_duration_seconds": 240,
                "creative_brief": {
                    "theme": "Digital escapism and neon dreams",
                    "mood": "nostalgic",
                    "energy_description": "Builds from contemplative to driving",
                    "reference_era": "1980s",
                },
                "primary_mood": "nostalgic",
                "has_vocals": True,
                "is_instrumental": False,
            }
        }
