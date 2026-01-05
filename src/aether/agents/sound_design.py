"""
Sound Design Agent

Assigns sounds/presets to instruments and designs the sonic palette.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from aether.agents.base import AgentRegistry, BaseAgent
from aether.knowledge import get_genre_manager
from aether.schemas.sound_design import (
    InstrumentAssignment,
    SampleSource,
    SoundDesignSpec,
    SynthPatch,
)

logger = logging.getLogger(__name__)


class SoundDesignInput(BaseModel):
    song_spec: dict[str, Any]
    arrangement_spec: dict[str, Any]
    rhythm_spec: dict[str, Any]
    genre_profile_id: str


class SoundDesignOutput(BaseModel):
    sound_design_spec: dict[str, Any]


@AgentRegistry.register("sound_design")
class SoundDesignAgent(BaseAgent[SoundDesignInput, SoundDesignOutput]):
    """
    Sound Design Agent.

    Responsibilities:
    - Assign sounds/patches to each instrument
    - Design synth patches with genre-appropriate parameters
    - Select samples for drums and texture
    - Configure global reverb and processing
    - Apply era-appropriate vintage processing
    """

    agent_type = "sound_design"
    agent_name = "Sound Design Agent"
    input_schema = SoundDesignInput
    output_schema = SoundDesignOutput

    async def process(
        self,
        input_data: SoundDesignInput,
        context: dict[str, Any],
    ) -> SoundDesignOutput:
        song_spec = input_data.song_spec
        arrangement = input_data.arrangement_spec
        rhythm = input_data.rhythm_spec
        genre_manager = get_genre_manager()
        profile = genre_manager.get(input_data.genre_profile_id)

        instruments = arrangement.get("instruments", [])
        mood = song_spec.get("primary_mood", "energetic")

        # Create synth patches
        synth_patches = []
        sample_sources = []
        assignments = []

        for inst in instruments:
            inst_name = inst.get("name", "unknown")
            category = inst.get("category", "synth")

            if category in ["drums", "percussion"]:
                # Sample-based
                sample = self._create_sample_source(inst_name, profile)
                sample_sources.append(sample)
                assignments.append(
                    InstrumentAssignment(
                        instrument_name=inst_name,
                        source_type="sample",
                        sample_name=sample.name,
                        velocity_curve="linear",
                        layer_count=1,
                    )
                )
            elif category in ["synth", "keys", "bass"]:
                # Synth-based
                patch = self._create_synth_patch(inst_name, category, profile, mood)
                synth_patches.append(patch)
                assignments.append(
                    InstrumentAssignment(
                        instrument_name=inst_name,
                        source_type="synth",
                        patch_name=patch.name,
                        velocity_curve="soft" if category == "keys" else "linear",
                        layer_count=2 if category == "synth" else 1,
                    )
                )
            else:
                # Default to sample
                sample = self._create_sample_source(inst_name, profile)
                sample_sources.append(sample)
                assignments.append(
                    InstrumentAssignment(
                        instrument_name=inst_name,
                        source_type="sample",
                        sample_name=sample.name,
                        velocity_curve="linear",
                        layer_count=1,
                    )
                )

        # Determine era-appropriate processing
        vintage_warmth, tape_sat, vinyl = self._determine_vintage_processing(profile)

        # Determine reverb settings from genre
        reverb_type, reverb_size, reverb_decay = self._determine_reverb(profile, mood)

        sound_design_spec = SoundDesignSpec(
            song_id=str(song_spec["id"]),
            arrangement_id=str(arrangement.get("id", "arrangement")),
            rhythm_id=str(rhythm.get("id", "rhythm")),
            synth_patches=synth_patches,
            sample_sources=sample_sources,
            instrument_assignments=assignments,
            master_tuning_hz=440.0,
            global_reverb_type=reverb_type,
            global_reverb_size=reverb_size,
            global_reverb_decay=reverb_decay,
            vintage_warmth=vintage_warmth,
            tape_saturation=tape_sat,
            vinyl_texture=vinyl,
        )

        self.log_decision(
            decision_type="sound_design",
            input_summary=f"Instruments: {len(instruments)}, Mood: {mood}",
            output_summary=f"Created {len(synth_patches)} patches, {len(sample_sources)} samples",
            reasoning=f"Following genre era: {profile.production.era_reference}",
            confidence=0.85,
        )

        return SoundDesignOutput(sound_design_spec=sound_design_spec.model_dump())

    def _create_synth_patch(self, name: str, category: str, profile, mood: str) -> SynthPatch:
        """Create a synth patch specification."""
        # Base parameters by category
        if category == "bass":
            osc1 = "saw"
            osc2 = "square"
            cutoff = 800 if mood in ["dark", "aggressive"] else 1200
            resonance = 0.4
            attack = 10.0
            release = 300.0
            reverb = 0.05
            chorus = 0.0
        elif category == "keys":
            osc1 = "square"
            osc2 = "sine"
            cutoff = 2000
            resonance = 0.2
            attack = 20.0
            release = 500.0
            reverb = 0.3
            chorus = 0.2
        else:  # lead/pad synth
            osc1 = "saw"
            osc2 = "saw"
            cutoff = 3000
            resonance = 0.3
            attack = 100.0 if "pad" in name else 10.0
            release = 2000.0 if "pad" in name else 400.0
            reverb = 0.4
            chorus = 0.3

        # Apply genre characteristics
        if profile.production.era_reference in ["1980s", "1970s"]:
            chorus += 0.1

        return SynthPatch(
            name=f"{name}_patch",
            synth_type="subtractive",
            osc1_waveform=osc1,
            osc2_waveform=osc2,
            osc_mix=0.6,
            detune_cents=7 if category == "synth" else 0,
            filter_type="lowpass",
            filter_cutoff_hz=cutoff,
            filter_resonance=resonance,
            filter_envelope_amount=0.4,
            amp_attack_ms=attack,
            amp_decay_ms=100.0,
            amp_sustain=0.7,
            amp_release_ms=release,
            reverb_send=reverb,
            delay_send=0.15 if category == "synth" else 0.0,
            chorus_amount=min(1.0, chorus),
            distortion_amount=0.1 if mood == "aggressive" else 0.0,
        )

    def _create_sample_source(self, inst_name: str, profile) -> SampleSource:
        """Create sample source specification."""
        # Determine source type and bank/preset
        name_lower = inst_name.lower()

        if "kick" in name_lower or "drum" in name_lower:
            return SampleSource(
                name=f"{inst_name}_sample",
                source_type="soundfont",
                soundfont_bank=0,
                soundfont_preset=0,  # GM Standard Kit
                license="royalty_free",
            )
        elif "snare" in name_lower:
            return SampleSource(
                name=f"{inst_name}_sample",
                source_type="soundfont",
                soundfont_bank=0,
                soundfont_preset=0,
                license="royalty_free",
            )
        elif "hat" in name_lower or "hihat" in name_lower:
            return SampleSource(
                name=f"{inst_name}_sample",
                source_type="soundfont",
                soundfont_bank=0,
                soundfont_preset=0,
                license="royalty_free",
            )
        elif "guitar" in name_lower:
            return SampleSource(
                name=f"{inst_name}_sample",
                source_type="soundfont",
                soundfont_bank=0,
                soundfont_preset=25,  # Acoustic Guitar
                license="royalty_free",
            )
        elif "string" in name_lower:
            return SampleSource(
                name=f"{inst_name}_sample",
                source_type="soundfont",
                soundfont_bank=0,
                soundfont_preset=48,  # Strings
                license="royalty_free",
            )
        else:
            return SampleSource(
                name=f"{inst_name}_sample",
                source_type="soundfont",
                soundfont_bank=0,
                soundfont_preset=0,
                license="royalty_free",
            )

    def _determine_vintage_processing(self, profile) -> tuple:
        """Determine vintage processing amounts based on era."""
        era = profile.production.era_reference

        if era in ["1970s", "1960s"]:
            return (0.6, 0.4, 0.0)
        elif era == "1980s":
            return (0.4, 0.2, 0.0)
        elif era == "1990s":
            return (0.3, 0.1, 0.0)
        elif "lo-fi" in profile.id.lower():
            return (0.5, 0.3, 0.4)
        else:
            return (0.1, 0.0, 0.0)

    def _determine_reverb(self, profile, mood: str) -> tuple:
        """Determine global reverb settings."""
        # Base settings from production profile
        reverb_type = "hall"
        reverb_size = 0.5
        reverb_decay = 2.0

        # Adjust by mood
        if mood in ["ethereal", "calm"]:
            reverb_size = 0.7
            reverb_decay = 3.0
            reverb_type = "hall"
        elif mood in ["aggressive", "energetic"]:
            reverb_size = 0.3
            reverb_decay = 1.0
            reverb_type = "room"
        elif mood in ["dark", "melancholic"]:
            reverb_size = 0.6
            reverb_decay = 2.5
            reverb_type = "plate"

        return (reverb_type, reverb_size, reverb_decay)
