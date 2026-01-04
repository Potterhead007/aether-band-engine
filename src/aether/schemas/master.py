"""
MasterSpec - Mastering targets and settings.

Purpose: Defines mastering targets for loudness, dynamic range, and tonal balance.
"""

from typing import Optional

from pydantic import Field

from aether.schemas.base import (
    AetherBaseModel,
    DynamicRangeTarget,
    IdentifiableModel,
    LoudnessTarget,
    TruePeakTarget,
)


class TonalTarget(AetherBaseModel):
    """Tonal balance targets for mastering."""

    low_end_emphasis: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Sub/bass emphasis"
    )
    brightness: float = Field(
        ge=0.0, le=1.0, default=0.5, description="High frequency presence"
    )
    warmth: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Low-mid warmth"
    )
    presence: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Vocal presence range (2-5kHz)"
    )
    air: float = Field(
        ge=0.0, le=1.0, default=0.3, description="Ultra-high shimmer (10kHz+)"
    )


class MultibandSettings(AetherBaseModel):
    """Multi-band compression settings."""

    band_name: str = Field(description="low, low_mid, mid, high_mid, high")
    crossover_low_hz: int = Field(ge=20, le=20000)
    crossover_high_hz: int = Field(ge=20, le=20000)
    threshold_db: float = Field(ge=-60.0, le=0.0, default=-20.0)
    ratio: float = Field(ge=1.0, le=20.0, default=2.0)
    attack_ms: float = Field(ge=0.1, le=500.0, default=20.0)
    release_ms: float = Field(ge=10.0, le=3000.0, default=200.0)
    gain_db: float = Field(ge=-12.0, le=12.0, default=0.0)


class LimiterSettings(AetherBaseModel):
    """Final limiter settings."""

    ceiling_dbtp: float = Field(ge=-3.0, le=0.0, default=-1.0)
    release_ms: float = Field(ge=10.0, le=1000.0, default=100.0)
    lookahead_ms: float = Field(ge=0.0, le=10.0, default=5.0)


class MasterSpec(IdentifiableModel):
    """
    Complete mastering specification for a song.

    Defines all mastering targets and processing settings to achieve
    commercial-ready audio for streaming platforms.
    """

    # Reference
    song_id: str = Field(description="Reference to parent SongSpec")
    mix_id: str = Field(description="Reference to MixSpec")
    genre_id: str = Field(description="Reference to GenreRootProfile for targets")

    # Primary targets
    loudness: LoudnessTarget = Field(default_factory=LoudnessTarget)
    true_peak: TruePeakTarget = Field(default_factory=TruePeakTarget)
    dynamic_range: DynamicRangeTarget = Field(default_factory=DynamicRangeTarget)

    # Tonal targets
    tonal_target: TonalTarget = Field(default_factory=TonalTarget)

    # Processing chain
    multiband_compression: list[MultibandSettings] = Field(
        default_factory=list, max_length=5
    )
    limiter: LimiterSettings = Field(default_factory=LimiterSettings)

    # Stereo processing
    stereo_enhancement: float = Field(
        ge=0.0, le=1.0, default=0.0, description="Stereo widening amount"
    )
    mid_side_eq: bool = Field(default=False, description="Use M/S processing")

    # Delivery formats
    formats: list[str] = Field(
        default=["wav_24_48", "flac_24_48", "wav_16_44"],
        description="Output format identifiers",
    )

    # Verification
    measured_lufs: Optional[float] = Field(default=None)
    measured_true_peak: Optional[float] = Field(default=None)
    measured_dynamic_range: Optional[float] = Field(default=None)
    passed_qc: Optional[bool] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "example-song-id",
                "mix_id": "example-mix-id",
                "genre_id": "boom-bap",
                "loudness": {"target_lufs": -14.0, "tolerance": 0.5},
                "true_peak": {"ceiling_dbtp": -1.0},
                "dynamic_range": {"minimum_lu": 6.0, "target_lu": 8.0},
                "tonal_target": {
                    "low_end_emphasis": 0.7,
                    "brightness": 0.4,
                    "warmth": 0.6,
                },
            }
        }
