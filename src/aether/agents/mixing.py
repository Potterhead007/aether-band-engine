"""
Mixing Agent

Creates the mix specification with levels, panning, EQ, and dynamics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel

from aether.agents.base import BaseAgent, AgentRegistry
from aether.knowledge import get_genre_manager
from aether.schemas.mix import (
    MixSpec,
    TrackSettings,
    BusSettings,
    EQBand,
    Compressor,
    SpatialSettings,
    Automation,
    AutomationPoint,
)
from aether.schemas.base import SectionType
from aether.storage import ArtifactType

logger = logging.getLogger(__name__)


class MixingInput(BaseModel):
    song_spec: Dict[str, Any]
    arrangement_spec: Dict[str, Any]
    sound_design_spec: Dict[str, Any]
    genre_profile_id: str


class MixingOutput(BaseModel):
    mix_spec: Dict[str, Any]


@AgentRegistry.register("mixing")
class MixingAgent(BaseAgent[MixingInput, MixingOutput]):
    """
    Mixing Agent.

    Responsibilities:
    - Set track levels and panning
    - Apply EQ per track/bus
    - Configure compression
    - Create bus structure (drums, music, vocals, master)
    - Design automation for dynamic movement
    - Set spatial parameters
    """

    agent_type = "mixing"
    agent_name = "Mixing Agent"
    input_schema = MixingInput
    output_schema = MixingOutput

    async def process(
        self,
        input_data: MixingInput,
        context: Dict[str, Any],
    ) -> MixingOutput:
        song_spec = input_data.song_spec
        arrangement = input_data.arrangement_spec
        sound_design = input_data.sound_design_spec
        genre_manager = get_genre_manager()
        profile = genre_manager.get(input_data.genre_profile_id)

        instruments = arrangement.get("instruments", [])
        sections = arrangement.get("sections", [])

        # Create buses
        buses = self._create_bus_structure(profile)

        # Create track settings
        tracks = []
        for inst in instruments:
            track = self._create_track_settings(inst, profile)
            tracks.append(track)

        # Create automation
        automations = self._create_automation(sections, instruments)

        # Spatial settings
        spatial = self._create_spatial_settings(profile)

        # Master EQ and compression
        master_eq = self._create_master_eq(profile)
        master_comp = self._create_master_compressor(profile)

        mix_spec = MixSpec(
            song_id=str(song_spec["id"]),
            sound_design_id=str(sound_design.get("id", "sound_design")),
            buses=buses,
            tracks=tracks,
            automations=automations,
            spatial=spatial,
            master_eq=master_eq,
            master_compressor=master_comp,
            target_headroom_db=-6.0,
        )

        self.log_decision(
            decision_type="mixing",
            input_summary=f"Tracks: {len(instruments)}",
            output_summary=f"Created mix with {len(buses)} buses, {len(tracks)} tracks",
            reasoning="Following genre mixing conventions",
            confidence=0.85,
        )

        return MixingOutput(mix_spec=mix_spec.model_dump())

    def _create_bus_structure(self, profile) -> List[BusSettings]:
        """Create standard bus structure."""
        buses = []

        # Drum bus
        buses.append(
            BusSettings(
                bus_name="drums",
                gain_db=0.0,
                eq_bands=[
                    EQBand(band_type="highpass", frequency_hz=40, gain_db=0.0, q=0.7),
                    EQBand(band_type="peak", frequency_hz=4000, gain_db=2.0, q=1.5),
                ],
                compressor=Compressor(
                    threshold_db=-18.0,
                    ratio=3.0,
                    attack_ms=20.0,
                    release_ms=150.0,
                ),
                output_bus="master",
            )
        )

        # Bass bus
        buses.append(
            BusSettings(
                bus_name="bass",
                gain_db=0.0,
                eq_bands=[
                    EQBand(band_type="highpass", frequency_hz=30, gain_db=0.0, q=0.7),
                    EQBand(band_type="lowshelf", frequency_hz=80, gain_db=2.0, q=0.8),
                ],
                compressor=Compressor(
                    threshold_db=-15.0,
                    ratio=4.0,
                    attack_ms=10.0,
                    release_ms=100.0,
                ),
                output_bus="master",
            )
        )

        # Music bus (synths, keys, guitars)
        buses.append(
            BusSettings(
                bus_name="music",
                gain_db=-2.0,
                eq_bands=[
                    EQBand(band_type="highpass", frequency_hz=100, gain_db=0.0, q=0.7),
                ],
                compressor=Compressor(
                    threshold_db=-20.0,
                    ratio=2.5,
                    attack_ms=30.0,
                    release_ms=200.0,
                ),
                output_bus="master",
            )
        )

        # Vocal bus
        buses.append(
            BusSettings(
                bus_name="vocals",
                gain_db=1.0,
                eq_bands=[
                    EQBand(band_type="highpass", frequency_hz=80, gain_db=0.0, q=0.7),
                    EQBand(band_type="peak", frequency_hz=3000, gain_db=2.0, q=2.0),
                    EQBand(band_type="highshelf", frequency_hz=10000, gain_db=1.5, q=0.7),
                ],
                compressor=Compressor(
                    threshold_db=-16.0,
                    ratio=4.0,
                    attack_ms=5.0,
                    release_ms=80.0,
                ),
                output_bus="master",
            )
        )

        return buses

    def _create_track_settings(self, instrument: Dict, profile) -> TrackSettings:
        """Create settings for a single track."""
        name = instrument.get("name", "unknown")
        category = instrument.get("category", "synth")

        # Base level and pan by category
        if category == "drums":
            gain = 0.0
            pan = 0.0
            output_bus = "drums"
            reverb_send = -25.0
            delay_send = -30.0
        elif category == "bass":
            gain = -2.0
            pan = 0.0
            output_bus = "bass"
            reverb_send = -40.0
            delay_send = -40.0
        elif category == "vocal":
            gain = 0.0
            pan = 0.0
            output_bus = "vocals"
            reverb_send = -15.0
            delay_send = -20.0
        elif category in ["synth", "keys"]:
            gain = -4.0
            pan = 0.0
            output_bus = "music"
            reverb_send = -18.0
            delay_send = -22.0
        else:
            gain = -6.0
            pan = 0.0
            output_bus = "music"
            reverb_send = -20.0
            delay_send = -25.0

        # Panning for doubles
        if "left" in name.lower() or "_L" in name:
            pan = -0.6
        elif "right" in name.lower() or "_R" in name:
            pan = 0.6

        # Create EQ based on category
        eq_bands = self._get_track_eq(category)

        # Create compressor if needed
        compressor = None
        if category in ["drums", "bass", "vocal"]:
            compressor = Compressor(
                threshold_db=-18.0,
                ratio=3.0,
                attack_ms=15.0,
                release_ms=100.0,
            )

        return TrackSettings(
            track_name=name,
            gain_db=gain,
            pan=pan,
            mute=False,
            solo=False,
            eq_bands=eq_bands,
            compressor=compressor,
            reverb_send_db=reverb_send,
            delay_send_db=delay_send,
            output_bus=output_bus,
        )

    def _get_track_eq(self, category: str) -> List[EQBand]:
        """Get EQ settings for track category."""
        if category == "drums":
            return [
                EQBand(band_type="highpass", frequency_hz=40, gain_db=0.0, q=0.7),
                EQBand(band_type="peak", frequency_hz=100, gain_db=2.0, q=2.0),
                EQBand(band_type="peak", frequency_hz=5000, gain_db=1.5, q=1.5),
            ]
        elif category == "bass":
            return [
                EQBand(band_type="highpass", frequency_hz=30, gain_db=0.0, q=0.7),
                EQBand(band_type="peak", frequency_hz=60, gain_db=2.0, q=1.5),
                EQBand(band_type="lowpass", frequency_hz=8000, gain_db=0.0, q=0.7),
            ]
        elif category == "vocal":
            return [
                EQBand(band_type="highpass", frequency_hz=100, gain_db=0.0, q=0.7),
                EQBand(band_type="peak", frequency_hz=300, gain_db=-2.0, q=2.0),
                EQBand(band_type="peak", frequency_hz=3000, gain_db=2.0, q=2.0),
            ]
        elif category in ["synth", "keys"]:
            return [
                EQBand(band_type="highpass", frequency_hz=100, gain_db=0.0, q=0.7),
                EQBand(band_type="peak", frequency_hz=2500, gain_db=1.0, q=1.5),
            ]
        else:
            return [
                EQBand(band_type="highpass", frequency_hz=80, gain_db=0.0, q=0.7),
            ]

    def _create_automation(self, sections: List[Dict], instruments: List[Dict]) -> List[Automation]:
        """Create mix automation for dynamic movement."""
        automations = []

        # Calculate approximate section times
        total_duration = sum(s.get("length_bars", 8) * 2 for s in sections)  # Approx 2s per bar

        current_time = 0.0
        for section in sections:
            section_type = section.get("section_type", "verse")
            duration = section.get("length_bars", 8) * 2.0

            # Master gain automation for section dynamics
            if section_type == SectionType.CHORUS.value:
                # Slight boost for chorus
                automations.append(
                    Automation(
                        target_track="master",
                        parameter="gain",
                        points=[
                            AutomationPoint(time_seconds=current_time, value=0.0),
                            AutomationPoint(time_seconds=current_time + 1.0, value=1.0),
                        ],
                        curve_type="exponential",
                    )
                )
            elif section_type == SectionType.BRIDGE.value:
                # Pull back for bridge
                automations.append(
                    Automation(
                        target_track="master",
                        parameter="gain",
                        points=[
                            AutomationPoint(time_seconds=current_time, value=0.0),
                            AutomationPoint(time_seconds=current_time + 1.0, value=-2.0),
                        ],
                        curve_type="linear",
                    )
                )

            current_time += duration

        return automations

    def _create_spatial_settings(self, profile) -> SpatialSettings:
        """Create spatial/stereo settings."""
        return SpatialSettings(
            stereo_width=1.0,
            mid_side_balance=0.0,
            haas_delay_ms=None,
        )

    def _create_master_eq(self, profile) -> List[EQBand]:
        """Create master bus EQ."""
        return [
            EQBand(band_type="highpass", frequency_hz=30, gain_db=0.0, q=0.7),
            EQBand(band_type="highshelf", frequency_hz=12000, gain_db=0.5, q=0.7),
        ]

    def _create_master_compressor(self, profile) -> Compressor:
        """Create master bus compressor."""
        return Compressor(
            threshold_db=-12.0,
            ratio=2.0,
            attack_ms=30.0,
            release_ms=200.0,
            knee_db=6.0,
            makeup_gain_db=0.0,
        )
