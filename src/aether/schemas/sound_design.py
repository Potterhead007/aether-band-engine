"""
SoundDesignSpec - Sound palette and patches.

Purpose: Defines instrument assignments, synth patches, and sample sources.
"""

from typing import Optional

from pydantic import Field

from aether.schemas.base import (
    AetherBaseModel,
    IdentifiableModel,
)


class SynthPatch(AetherBaseModel):
    """Synthesizer patch specification."""

    name: str
    synth_type: str = Field(
        description="subtractive, fm, wavetable, granular, additive"
    )
    # Oscillators
    osc1_waveform: str = Field(default="saw", description="sine, saw, square, triangle, noise")
    osc2_waveform: Optional[str] = Field(default=None)
    osc_mix: float = Field(ge=0.0, le=1.0, default=0.5)
    detune_cents: int = Field(ge=-100, le=100, default=0)

    # Filter
    filter_type: str = Field(default="lowpass", description="lowpass, highpass, bandpass, notch")
    filter_cutoff_hz: int = Field(ge=20, le=20000, default=5000)
    filter_resonance: float = Field(ge=0.0, le=1.0, default=0.3)
    filter_envelope_amount: float = Field(ge=0.0, le=1.0, default=0.5)

    # Envelopes
    amp_attack_ms: float = Field(ge=0.0, le=5000.0, default=10.0)
    amp_decay_ms: float = Field(ge=0.0, le=5000.0, default=100.0)
    amp_sustain: float = Field(ge=0.0, le=1.0, default=0.7)
    amp_release_ms: float = Field(ge=0.0, le=10000.0, default=200.0)

    # Effects (send amounts)
    reverb_send: float = Field(ge=0.0, le=1.0, default=0.2)
    delay_send: float = Field(ge=0.0, le=1.0, default=0.1)
    chorus_amount: float = Field(ge=0.0, le=1.0, default=0.0)
    distortion_amount: float = Field(ge=0.0, le=1.0, default=0.0)


class SampleSource(AetherBaseModel):
    """Audio sample source specification."""

    name: str
    source_type: str = Field(
        description="soundfont, oneshot, loop, recorded"
    )
    soundfont_bank: Optional[int] = Field(default=None, ge=0)
    soundfont_preset: Optional[int] = Field(default=None, ge=0)
    file_path: Optional[str] = Field(default=None)
    license: str = Field(default="royalty_free")


class InstrumentAssignment(AetherBaseModel):
    """Maps an arrangement instrument to a sound source."""

    instrument_name: str = Field(description="From ArrangementSpec.instruments")
    source_type: str = Field(description="synth, sample, vocal")
    patch_name: Optional[str] = Field(
        default=None, description="Reference to SynthPatch if synth"
    )
    sample_name: Optional[str] = Field(
        default=None, description="Reference to SampleSource if sample"
    )
    velocity_curve: str = Field(
        default="linear", description="linear, soft, hard, s-curve"
    )
    layer_count: int = Field(ge=1, le=8, default=1)


class SoundDesignSpec(IdentifiableModel):
    """
    Complete sound design specification for a song.

    Defines the sonic palette including synth patches, samples, and instrument assignments.
    """

    # Reference
    song_id: str = Field(description="Reference to parent SongSpec")
    arrangement_id: str = Field(description="Reference to ArrangementSpec")
    rhythm_id: str = Field(description="Reference to RhythmSpec")

    # Patches
    synth_patches: list[SynthPatch] = Field(default_factory=list)

    # Samples
    sample_sources: list[SampleSource] = Field(default_factory=list)

    # Assignments
    instrument_assignments: list[InstrumentAssignment] = Field(min_length=1)

    # Global settings
    master_tuning_hz: float = Field(ge=400.0, le=480.0, default=440.0)
    global_reverb_type: str = Field(
        default="hall", description="room, hall, plate, spring, chamber"
    )
    global_reverb_size: float = Field(ge=0.0, le=1.0, default=0.5)
    global_reverb_decay: float = Field(ge=0.1, le=10.0, default=2.0)

    # Era-appropriate processing
    vintage_warmth: float = Field(
        ge=0.0, le=1.0, default=0.0, description="Analog warmth/saturation"
    )
    tape_saturation: float = Field(ge=0.0, le=1.0, default=0.0)
    vinyl_texture: float = Field(ge=0.0, le=1.0, default=0.0)

    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "example-song-id",
                "arrangement_id": "example-arrangement-id",
                "rhythm_id": "example-rhythm-id",
                "synth_patches": [
                    {
                        "name": "lead_synth",
                        "synth_type": "subtractive",
                        "osc1_waveform": "saw",
                        "filter_cutoff_hz": 3000,
                        "reverb_send": 0.3,
                    }
                ],
                "sample_sources": [
                    {
                        "name": "drums",
                        "source_type": "soundfont",
                        "soundfont_bank": 0,
                        "soundfont_preset": 0,
                    }
                ],
                "instrument_assignments": [
                    {
                        "instrument_name": "lead_synth",
                        "source_type": "synth",
                        "patch_name": "lead_synth",
                    }
                ],
            }
        }
