"""
MixSpec - Mix parameters and routing.

Purpose: Defines bus structure, track settings, automation, and spatial processing.
"""

from typing import Optional

from pydantic import Field

from aether.schemas.base import (
    AetherBaseModel,
    IdentifiableModel,
)


class EQBand(AetherBaseModel):
    """A single EQ band."""

    band_type: str = Field(description="lowshelf, highshelf, peak, lowpass, highpass, notch")
    frequency_hz: int = Field(ge=20, le=20000)
    gain_db: float = Field(ge=-24.0, le=24.0, default=0.0)
    q: float = Field(ge=0.1, le=10.0, default=1.0)
    enabled: bool = Field(default=True)


class Compressor(AetherBaseModel):
    """Compressor settings."""

    threshold_db: float = Field(ge=-60.0, le=0.0, default=-20.0)
    ratio: float = Field(ge=1.0, le=20.0, default=4.0)
    attack_ms: float = Field(ge=0.1, le=500.0, default=10.0)
    release_ms: float = Field(ge=10.0, le=3000.0, default=100.0)
    knee_db: float = Field(ge=0.0, le=12.0, default=3.0)
    makeup_gain_db: float = Field(ge=0.0, le=24.0, default=0.0)
    enabled: bool = Field(default=True)


class TrackSettings(AetherBaseModel):
    """Settings for an individual track in the mix."""

    track_name: str
    gain_db: float = Field(ge=-60.0, le=12.0, default=0.0)
    pan: float = Field(ge=-1.0, le=1.0, default=0.0)
    mute: bool = Field(default=False)
    solo: bool = Field(default=False)

    # Processing
    eq_bands: list[EQBand] = Field(default_factory=list, max_length=8)
    compressor: Optional[Compressor] = Field(default=None)

    # Sends
    reverb_send_db: float = Field(ge=-60.0, le=0.0, default=-20.0)
    delay_send_db: float = Field(ge=-60.0, le=0.0, default=-30.0)

    # Routing
    output_bus: str = Field(default="master")


class BusSettings(AetherBaseModel):
    """Settings for a mix bus (submix)."""

    bus_name: str
    gain_db: float = Field(ge=-60.0, le=12.0, default=0.0)
    eq_bands: list[EQBand] = Field(default_factory=list, max_length=8)
    compressor: Optional[Compressor] = Field(default=None)
    output_bus: str = Field(default="master")


class AutomationPoint(AetherBaseModel):
    """A single automation point."""

    time_seconds: float = Field(ge=0.0)
    value: float


class Automation(AetherBaseModel):
    """Automation for a parameter."""

    target_track: str
    parameter: str = Field(description="gain, pan, send, filter_cutoff, etc.")
    points: list[AutomationPoint] = Field(min_length=2)
    curve_type: str = Field(default="linear", description="linear, exponential, s-curve")


class SpatialSettings(AetherBaseModel):
    """Spatial/stereo processing settings."""

    stereo_width: float = Field(
        ge=0.0, le=2.0, default=1.0, description="0=mono, 1=stereo, 2=widened"
    )
    mid_side_balance: float = Field(
        ge=-1.0, le=1.0, default=0.0, description="-1=mid only, 0=balanced, 1=side only"
    )
    haas_delay_ms: Optional[float] = Field(
        default=None, ge=0.0, le=35.0, description="Haas effect delay"
    )


class MixSpec(IdentifiableModel):
    """
    Complete mix specification for a song.

    Defines the mixing parameters including track settings, bus routing,
    automation, and spatial processing.
    """

    # Reference
    song_id: str = Field(description="Reference to parent SongSpec")
    sound_design_id: str = Field(description="Reference to SoundDesignSpec")

    # Bus structure
    buses: list[BusSettings] = Field(default_factory=list)

    # Track settings
    tracks: list[TrackSettings] = Field(min_length=1)

    # Automation
    automations: list[Automation] = Field(default_factory=list)

    # Spatial
    spatial: SpatialSettings = Field(default_factory=SpatialSettings)

    # Master section
    master_eq: list[EQBand] = Field(default_factory=list)
    master_compressor: Optional[Compressor] = Field(default=None)

    # Headroom
    target_headroom_db: float = Field(
        ge=-12.0, le=-1.0, default=-6.0, description="Headroom before mastering"
    )

    # Quality checks
    mono_compatible: Optional[bool] = Field(default=None)
    phase_correlation: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    peak_db: Optional[float] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "example-song-id",
                "sound_design_id": "example-sd-id",
                "tracks": [
                    {"track_name": "kick", "gain_db": 0.0, "pan": 0.0},
                    {"track_name": "snare", "gain_db": -2.0, "pan": 0.0},
                    {"track_name": "bass", "gain_db": -3.0, "pan": 0.0},
                ],
                "target_headroom_db": -6.0,
            }
        }
