"""
Arrangement Agent

Designs song structure, instrumentation, and energy curves.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from aether.agents.base import AgentRegistry, BaseAgent
from aether.knowledge import get_genre_manager
from aether.schemas.arrangement import (
    ArrangementSpec,
    EnergyPoint,
    Instrument,
    SectionDefinition,
    Transition,
)
from aether.schemas.base import EnergyLevel, Feel, SectionType, TimeSignature
from aether.schemas.rhythm import (
    DrumHit,
    DrumPattern,
    GrooveTemplate,
    Humanization,
    RhythmSpec,
    SectionRhythm,
)

logger = logging.getLogger(__name__)


class ArrangementInput(BaseModel):
    song_spec: dict[str, Any]
    harmony_spec: dict[str, Any]
    melody_spec: dict[str, Any]
    genre_profile_id: str


class ArrangementOutput(BaseModel):
    arrangement_spec: dict[str, Any]
    rhythm_spec: dict[str, Any]


@AgentRegistry.register("arrangement")
class ArrangementAgent(BaseAgent[ArrangementInput, ArrangementOutput]):
    """
    Arrangement Agent.

    Responsibilities:
    - Design song structure (intro, verse, chorus, etc.)
    - Assign instruments to sections
    - Create energy curve
    - Design transitions
    - Define drum patterns and groove
    """

    agent_type = "arrangement"
    agent_name = "Arrangement Agent"
    input_schema = ArrangementInput
    output_schema = ArrangementOutput

    async def process(
        self,
        input_data: ArrangementInput,
        context: dict[str, Any],
    ) -> ArrangementOutput:
        song_spec = input_data.song_spec
        genre_manager = get_genre_manager()
        profile = genre_manager.get(input_data.genre_profile_id)

        bpm = song_spec["bpm"]
        duration = song_spec["target_duration_seconds"]

        # Calculate total bars
        beats_per_bar = 4  # Assuming 4/4
        total_beats = (duration / 60) * bpm
        total_bars = int(total_beats / beats_per_bar)

        # Create instruments
        instruments = self._create_instruments(profile)

        # Create structure
        sections = self._create_structure(profile, total_bars, instruments)

        # Create energy curve
        energy_curve = self._create_energy_curve(sections)

        # Create transitions
        transitions = self._create_transitions(sections)

        arrangement_spec = ArrangementSpec(
            song_id=str(song_spec["id"]),
            instruments=instruments,
            sections=sections,
            total_bars=total_bars,
            total_duration_seconds=float(duration),
            energy_curve=energy_curve,
            energy_curve_type=profile.arrangement.energy_curve_type,
            transitions=transitions,
        )

        # Create rhythm spec
        rhythm_spec = self._create_rhythm_spec(song_spec, profile, sections)

        self.log_decision(
            decision_type="structure",
            input_summary=f"Duration: {duration}s, {total_bars} bars",
            output_summary=f"Created {len(sections)} sections",
            reasoning=f"Following genre archetype: {profile.arrangement.common_structures[0]}",
            confidence=0.85,
        )

        return ArrangementOutput(
            arrangement_spec=arrangement_spec.model_dump(),
            rhythm_spec=rhythm_spec.model_dump(),
        )

    def _create_instruments(self, profile) -> list[Instrument]:
        """Create instrument list from genre profile."""
        instruments = []

        # Essential instruments
        for name in profile.instrumentation.essential:
            category = self._categorize_instrument(name)
            instruments.append(
                Instrument(
                    name=name.replace(" ", "_").lower(),
                    category=category,
                    role="rhythm" if category in ["drums", "bass"] else "lead",
                    is_essential=True,
                )
            )

        # Common instruments (add some)
        for name in profile.instrumentation.common[:3]:
            category = self._categorize_instrument(name)
            instruments.append(
                Instrument(
                    name=name.replace(" ", "_").lower(),
                    category=category,
                    role="texture",
                    is_essential=False,
                )
            )

        return instruments

    def _categorize_instrument(self, name: str) -> str:
        """Categorize an instrument."""
        name_lower = name.lower()
        if any(x in name_lower for x in ["drum", "kick", "snare", "hat", "cymbal"]):
            return "drums"
        if any(x in name_lower for x in ["bass"]):
            return "bass"
        if any(x in name_lower for x in ["piano", "keys", "rhodes", "organ"]):
            return "keys"
        if any(x in name_lower for x in ["synth", "pad", "lead"]):
            return "synth"
        if any(x in name_lower for x in ["guitar"]):
            return "guitar"
        if any(x in name_lower for x in ["string", "violin", "cello"]):
            return "strings"
        if any(x in name_lower for x in ["vocal", "voice"]):
            return "vocal"
        return "fx"

    def _create_structure(
        self,
        profile,
        total_bars: int,
        instruments: list[Instrument],
    ) -> list[SectionDefinition]:
        """Create song structure."""
        # Parse a structure archetype
        structure_str = profile.arrangement.common_structures[0]
        section_names = structure_str.split("-")

        # Calculate bar distribution
        avg_bars_per_section = total_bars // len(section_names)
        sections = []
        current_bar = 1

        essential_names = [i.name for i in instruments if i.is_essential]
        all_names = [i.name for i in instruments]

        for idx, section_name in enumerate(section_names):
            section_type = self._parse_section_type(section_name)

            # Determine length
            if section_type == SectionType.INTRO:
                length = min(8, avg_bars_per_section)
            elif section_type == SectionType.OUTRO:
                length = min(8, avg_bars_per_section)
            elif section_type == SectionType.CHORUS:
                length = avg_bars_per_section + 4
            else:
                length = avg_bars_per_section

            # Determine energy
            energy = self._section_energy(section_type, idx, len(section_names))

            # Determine instruments
            if section_type == SectionType.INTRO:
                section_instruments = essential_names[:2]
            elif section_type == SectionType.CHORUS:
                section_instruments = all_names
            elif section_type == SectionType.BRIDGE:
                section_instruments = essential_names
            else:
                section_instruments = essential_names + all_names[:1]

            sections.append(
                SectionDefinition(
                    section_type=section_type,
                    label=f"{section_name.title()} {idx + 1}",
                    start_bar=current_bar,
                    length_bars=length,
                    energy_level=energy,
                    instruments=section_instruments,
                )
            )

            current_bar += length

        return sections

    def _parse_section_type(self, name: str) -> SectionType:
        """Parse section name to type."""
        name = name.lower().strip()
        mapping = {
            "intro": SectionType.INTRO,
            "verse": SectionType.VERSE,
            "chorus": SectionType.CHORUS,
            "hook": SectionType.CHORUS,
            "bridge": SectionType.BRIDGE,
            "outro": SectionType.OUTRO,
            "breakdown": SectionType.BREAKDOWN,
            "build": SectionType.BUILD,
            "drop": SectionType.DROP,
        }
        return mapping.get(name, SectionType.VERSE)

    def _section_energy(self, section_type: SectionType, idx: int, total: int) -> EnergyLevel:
        """Determine energy level for section."""
        if section_type == SectionType.INTRO:
            return EnergyLevel.LOW
        elif section_type == SectionType.CHORUS:
            return EnergyLevel.HIGH
        elif section_type == SectionType.BRIDGE:
            return EnergyLevel.MEDIUM_LOW
        elif section_type == SectionType.OUTRO:
            return EnergyLevel.MEDIUM
        elif section_type == SectionType.DROP:
            return EnergyLevel.VERY_HIGH
        else:
            return EnergyLevel.MEDIUM

    def _create_energy_curve(self, sections: list[SectionDefinition]) -> list[EnergyPoint]:
        """Create energy curve from sections."""
        curve = []
        total_bars = sum(s.length_bars for s in sections)

        accumulated = 0
        for section in sections:
            position = (accumulated / total_bars) * 100 if total_bars > 0 else 0
            curve.append(
                EnergyPoint(
                    position_percent=position,
                    energy_level=section.energy_level,
                    section_label=section.label,
                )
            )
            accumulated += section.length_bars

        return curve

    def _create_transitions(self, sections: list[SectionDefinition]) -> list[Transition]:
        """Create transitions between sections."""
        transitions = []
        for i in range(len(sections) - 1):
            from_section = sections[i]
            to_section = sections[i + 1]

            # Determine transition type
            if to_section.section_type == SectionType.CHORUS:
                technique = "riser"
            elif from_section.section_type == SectionType.CHORUS:
                technique = "drop"
            else:
                technique = "fill"

            transitions.append(
                Transition(
                    from_section=from_section.label,
                    to_section=to_section.label,
                    technique=technique,
                    duration_beats=4.0,
                    has_fill=True,
                )
            )

        return transitions

    def _create_rhythm_spec(
        self,
        song_spec: dict,
        profile,
        sections: list[SectionDefinition],
    ) -> RhythmSpec:
        """Create rhythm specification."""
        bpm = song_spec["bpm"]

        # Groove template from profile
        groove = GrooveTemplate(
            feel=Feel(profile.rhythm.feels[0]) if profile.rhythm.feels else Feel.STRAIGHT,
            swing_amount=profile.rhythm.swing_amount_typical,
            push_pull=0.0,
            pocket_tightness=0.85,
        )

        humanization = Humanization(
            timing_variation_ms=8.0,
            velocity_variation=12,
            swing_variation=0.02,
        )

        # Create basic drum pattern
        patterns = [self._create_drum_pattern(profile, "main_beat")]

        # Section rhythms
        section_rhythms = [
            SectionRhythm(
                section_type=s.section_type,
                drum_pattern_name="main_beat",
                groove_intensity=0.8 if s.section_type == SectionType.CHORUS else 0.6,
            )
            for s in sections
        ]

        return RhythmSpec(
            song_id=str(song_spec["id"]),
            time_signature=TimeSignature.FOUR_FOUR,
            bpm=bpm,
            groove_template=groove,
            humanization=humanization,
            drum_patterns=patterns,
            section_rhythms=section_rhythms,
            signature_pattern=(
                profile.rhythm.signature_patterns[0] if profile.rhythm.signature_patterns else None
            ),
        )

    def _create_drum_pattern(self, profile, name: str) -> DrumPattern:
        """Create a basic drum pattern."""
        hits = [
            DrumHit(instrument="kick", position_beats=0.0, velocity=110),
            DrumHit(instrument="snare", position_beats=1.0, velocity=100),
            DrumHit(instrument="kick", position_beats=2.0, velocity=100),
            DrumHit(instrument="snare", position_beats=3.0, velocity=105),
            # Hi-hats
            DrumHit(instrument="hihat", position_beats=0.0, velocity=80),
            DrumHit(instrument="hihat", position_beats=0.5, velocity=60),
            DrumHit(instrument="hihat", position_beats=1.0, velocity=80),
            DrumHit(instrument="hihat", position_beats=1.5, velocity=60),
            DrumHit(instrument="hihat", position_beats=2.0, velocity=80),
            DrumHit(instrument="hihat", position_beats=2.5, velocity=60),
            DrumHit(instrument="hihat", position_beats=3.0, velocity=80),
            DrumHit(instrument="hihat", position_beats=3.5, velocity=60),
        ]

        return DrumPattern(
            name=name,
            length_bars=1,
            time_signature=TimeSignature.FOUR_FOUR,
            hits=hits,
            feel=Feel.STRAIGHT,
        )
