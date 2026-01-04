"""
QAReport - Quality verification results.

Purpose: Comprehensive quality assurance report covering originality,
technical specs, and genre authenticity.
"""

from datetime import datetime
from typing import Optional

from pydantic import Field

from aether.schemas.base import (
    AetherBaseModel,
    IdentifiableModel,
)


class OriginalityCheck(AetherBaseModel):
    """Result of an originality check."""

    check_name: str
    check_type: str = Field(
        description="melody, lyrics, harmony, rhythm"
    )
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool
    details: Optional[str] = Field(default=None)
    similar_matches: list[str] = Field(
        default_factory=list, description="IDs of similar items if any"
    )


class TechnicalCheck(AetherBaseModel):
    """Result of a technical audio check."""

    check_name: str
    measured_value: float
    target_value: float
    tolerance: float
    passed: bool
    unit: str = Field(description="LUFS, dBTP, LU, Hz, etc.")


class GenreRubricScore(AetherBaseModel):
    """Score for a single genre authenticity dimension."""

    dimension_name: str
    weight: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0, le=1.0)
    feedback: str


class GenreAuthenticityResult(AetherBaseModel):
    """Complete genre authenticity evaluation."""

    genre_id: str
    dimension_scores: list[GenreRubricScore]
    total_score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0, default=0.8)
    passed: bool
    improvement_suggestions: list[str] = Field(default_factory=list)


class QAReport(IdentifiableModel):
    """
    Complete quality assurance report for a song.

    Covers all verification including originality, technical specs,
    and genre authenticity. This is the final gate before release.
    """

    # Reference
    song_id: str = Field(description="Reference to parent SongSpec")

    # Timestamps
    qa_started: datetime = Field(default_factory=datetime.utcnow)
    qa_completed: Optional[datetime] = Field(default=None)

    # Originality
    originality_checks: list[OriginalityCheck] = Field(min_length=1)
    originality_passed: bool = Field(default=False)
    overall_originality_score: float = Field(ge=0.0, le=1.0)

    # Technical
    technical_checks: list[TechnicalCheck] = Field(min_length=1)
    technical_passed: bool = Field(default=False)

    # Genre
    genre_authenticity: GenreAuthenticityResult

    # Final verdict
    all_passed: bool = Field(default=False)
    rejection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Manual overrides
    human_reviewed: bool = Field(default=False)
    human_approved: Optional[bool] = Field(default=None)
    human_notes: Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "example-song-id",
                "originality_checks": [
                    {
                        "check_name": "melody_interval_hash",
                        "check_type": "melody",
                        "score": 0.92,
                        "threshold": 0.85,
                        "passed": True,
                    },
                    {
                        "check_name": "lyric_ngram",
                        "check_type": "lyrics",
                        "score": 0.02,
                        "threshold": 0.03,
                        "passed": True,
                    },
                ],
                "originality_passed": True,
                "overall_originality_score": 0.91,
                "technical_checks": [
                    {
                        "check_name": "loudness",
                        "measured_value": -14.2,
                        "target_value": -14.0,
                        "tolerance": 0.5,
                        "passed": True,
                        "unit": "LUFS",
                    }
                ],
                "technical_passed": True,
                "genre_authenticity": {
                    "genre_id": "boom-bap",
                    "dimension_scores": [
                        {
                            "dimension_name": "Drum Sound",
                            "weight": 0.25,
                            "score": 0.88,
                            "weighted_score": 0.22,
                            "feedback": "Punchy drums with good vinyl texture",
                        }
                    ],
                    "total_score": 0.86,
                    "threshold": 0.8,
                    "passed": True,
                },
                "all_passed": True,
            }
        }
