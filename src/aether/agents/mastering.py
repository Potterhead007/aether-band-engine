"""
Mastering Agent

Applies final processing for loudness, dynamics, and tonal balance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel

from aether.agents.base import BaseAgent, AgentRegistry
from aether.knowledge import get_genre_manager
from aether.schemas.master import MasterSpec, TonalTarget, MultibandSettings, LimiterSettings
from aether.schemas.base import LoudnessTarget, TruePeakTarget, DynamicRangeTarget
from aether.storage import ArtifactType

logger = logging.getLogger(__name__)


class MasteringInput(BaseModel):
    song_spec: Dict[str, Any]
    mix_spec: Dict[str, Any]
    genre_profile_id: str


class MasteringOutput(BaseModel):
    master_spec: Dict[str, Any]


@AgentRegistry.register("mastering")
class MasteringAgent(BaseAgent[MasteringInput, MasteringOutput]):
    """
    Mastering Agent.

    Responsibilities:
    - Set loudness targets (LUFS) per genre
    - Configure multiband compression
    - Set true peak ceiling
    - Define tonal balance targets
    - Configure limiter settings
    - Define delivery formats
    """

    agent_type = "mastering"
    agent_name = "Mastering Agent"
    input_schema = MasteringInput
    output_schema = MasteringOutput

    async def process(
        self,
        input_data: MasteringInput,
        context: Dict[str, Any],
    ) -> MasteringOutput:
        song_spec = input_data.song_spec
        mix_spec = input_data.mix_spec
        genre_manager = get_genre_manager()
        profile = genre_manager.get(input_data.genre_profile_id)

        mood = song_spec.get("primary_mood", "energetic")

        # Determine loudness target from profile
        loudness = self._determine_loudness_target(profile)

        # True peak target
        true_peak = TruePeakTarget(ceiling_dbtp=-1.0)

        # Dynamic range target
        dynamic_range = self._determine_dynamic_range(profile, mood)

        # Tonal target
        tonal = self._determine_tonal_target(profile, mood)

        # Multiband compression
        multiband = self._create_multiband_settings(profile)

        # Limiter
        limiter = LimiterSettings(
            ceiling_dbtp=-1.0,
            release_ms=100.0,
            lookahead_ms=5.0,
        )

        # Stereo enhancement
        stereo_enhancement = self._determine_stereo_enhancement(mood)

        master_spec = MasterSpec(
            song_id=str(song_spec["id"]),
            mix_id=str(mix_spec.get("id", "mix")),
            genre_id=input_data.genre_profile_id,
            loudness=loudness,
            true_peak=true_peak,
            dynamic_range=dynamic_range,
            tonal_target=tonal,
            multiband_compression=multiband,
            limiter=limiter,
            stereo_enhancement=stereo_enhancement,
            mid_side_eq=True,
            formats=["wav_24_48", "flac_24_48", "wav_16_44", "mp3_320"],
        )

        self.log_decision(
            decision_type="mastering",
            input_summary=f"Genre: {input_data.genre_profile_id}, Mood: {mood}",
            output_summary=f"Target: {loudness.target_lufs} LUFS, DR: {dynamic_range.target_lu} LU",
            reasoning=f"Following streaming platform standards and genre conventions",
            confidence=0.9,
        )

        return MasteringOutput(master_spec=master_spec.model_dump())

    def _determine_loudness_target(self, profile) -> LoudnessTarget:
        """Determine loudness target from genre profile."""
        # Use profile target or default to streaming standard
        target = getattr(profile.production, "target_lufs", -14.0)
        if target is None:
            target = -14.0

        return LoudnessTarget(
            target_lufs=target,
            tolerance=0.5,
        )

    def _determine_dynamic_range(self, profile, mood: str) -> DynamicRangeTarget:
        """Determine dynamic range target."""
        # Genre and mood affect dynamic range
        base_dr = 8.0

        # Electronic tends to be more compressed
        if "electronic" in profile.lineage.primary_parent.lower():
            base_dr = 6.0
        # Acoustic/organic genres have more dynamics
        elif "jazz" in profile.lineage.primary_parent.lower():
            base_dr = 10.0

        # Mood adjustments
        if mood in ["calm", "ethereal"]:
            base_dr += 2.0
        elif mood in ["aggressive", "energetic"]:
            base_dr -= 1.0

        return DynamicRangeTarget(
            minimum_lu=max(4.0, base_dr - 2),
            target_lu=base_dr,
        )

    def _determine_tonal_target(self, profile, mood: str) -> TonalTarget:
        """Determine tonal balance targets."""
        # Base from genre
        low_end = 0.5
        brightness = 0.5
        warmth = 0.5
        presence = 0.5
        air = 0.3

        # Genre adjustments
        genre_id = profile.id.lower()
        if "hip-hop" in genre_id or "boom-bap" in genre_id:
            low_end = 0.7
            warmth = 0.6
        elif "edm" in genre_id or "synthwave" in genre_id:
            brightness = 0.6
            air = 0.4
        elif "lo-fi" in genre_id:
            low_end = 0.6
            warmth = 0.7
            brightness = 0.3
            air = 0.2

        # Mood adjustments
        if mood in ["dark", "melancholic"]:
            brightness -= 0.1
            warmth += 0.1
        elif mood in ["happy", "energetic"]:
            brightness += 0.1
            presence += 0.1

        return TonalTarget(
            low_end_emphasis=max(0.0, min(1.0, low_end)),
            brightness=max(0.0, min(1.0, brightness)),
            warmth=max(0.0, min(1.0, warmth)),
            presence=max(0.0, min(1.0, presence)),
            air=max(0.0, min(1.0, air)),
        )

    def _create_multiband_settings(self, profile) -> List[MultibandSettings]:
        """Create multiband compression settings."""
        return [
            MultibandSettings(
                band_name="low",
                crossover_low_hz=20,
                crossover_high_hz=120,
                threshold_db=-18.0,
                ratio=2.5,
                attack_ms=30.0,
                release_ms=300.0,
                gain_db=0.0,
            ),
            MultibandSettings(
                band_name="low_mid",
                crossover_low_hz=120,
                crossover_high_hz=500,
                threshold_db=-20.0,
                ratio=2.0,
                attack_ms=25.0,
                release_ms=200.0,
                gain_db=0.0,
            ),
            MultibandSettings(
                band_name="mid",
                crossover_low_hz=500,
                crossover_high_hz=2000,
                threshold_db=-22.0,
                ratio=1.8,
                attack_ms=20.0,
                release_ms=150.0,
                gain_db=0.5,
            ),
            MultibandSettings(
                band_name="high_mid",
                crossover_low_hz=2000,
                crossover_high_hz=8000,
                threshold_db=-24.0,
                ratio=1.5,
                attack_ms=15.0,
                release_ms=100.0,
                gain_db=0.0,
            ),
            MultibandSettings(
                band_name="high",
                crossover_low_hz=8000,
                crossover_high_hz=20000,
                threshold_db=-26.0,
                ratio=1.3,
                attack_ms=10.0,
                release_ms=80.0,
                gain_db=0.5,
            ),
        ]

    def _determine_stereo_enhancement(self, mood: str) -> float:
        """Determine stereo enhancement amount."""
        if mood in ["ethereal", "uplifting"]:
            return 0.2
        elif mood in ["aggressive", "energetic"]:
            return 0.1
        else:
            return 0.05
