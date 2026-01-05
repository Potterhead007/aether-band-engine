"""
Vocal Agent

Plans vocal performance including voice design, harmonies, and emotional arc.
CRITICAL: Uses parametric voice design only - NO voice cloning.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from aether.agents.base import AgentRegistry, BaseAgent
from aether.schemas.base import NoteName, SectionType
from aether.schemas.vocal import (
    AdLib,
    EmotionMarker,
    VocalDouble,
    VocalHarmony,
    VocalSpec,
    VoicePersona,
)

logger = logging.getLogger(__name__)


class VocalInput(BaseModel):
    song_spec: dict[str, Any]
    lyric_spec: dict[str, Any]
    melody_spec: dict[str, Any]
    genre_profile_id: str


class VocalOutput(BaseModel):
    vocal_spec: dict[str, Any]


@AgentRegistry.register("vocal")
class VocalAgent(BaseAgent[VocalInput, VocalOutput]):
    """
    Vocal Agent.

    Responsibilities:
    - Design voice persona (parametric, NOT cloning)
    - Plan vocal doubles and harmonies
    - Map emotional arc across sections
    - Design ad-libs and embellishments
    """

    agent_type = "vocal"
    agent_name = "Vocal Agent"
    input_schema = VocalInput
    output_schema = VocalOutput

    async def process(
        self,
        input_data: VocalInput,
        context: dict[str, Any],
    ) -> VocalOutput:
        song_spec = input_data.song_spec
        lyric_spec = input_data.lyric_spec
        melody_spec = input_data.melody_spec
        mood = song_spec.get("primary_mood", "energetic")

        # Design voice persona
        voice_persona = self._design_voice_persona(mood)

        # Create doubles
        doubles = self._create_doubles()

        # Create harmonies
        harmonies = self._create_harmonies()

        # Create ad-libs
        ad_libs = self._create_ad_libs(mood)

        # Create emotion arc
        emotion_arc = self._create_emotion_arc(lyric_spec)

        # Determine delivery style from mood
        delivery_style = self._determine_delivery_style(mood)

        vocal_spec = VocalSpec(
            song_id=str(song_spec["id"]),
            lyric_id=str(lyric_spec.get("id", "lyrics")),
            melody_id=str(melody_spec.get("id", "melody")),
            voice_persona=voice_persona,
            doubles=doubles,
            harmonies=harmonies,
            ad_libs=ad_libs,
            emotion_arc=emotion_arc,
            delivery_style=delivery_style,
            articulation="clear",
            autotune_amount=0.3,
            reverb_send=0.35,
            delay_send=0.2,
        )

        self.log_decision(
            decision_type="voice_design",
            input_summary=f"Mood: {mood}",
            output_summary=f"Designed {voice_persona.gender_presentation} voice, {delivery_style} delivery",
            reasoning="Parametric voice design matching song mood (NO cloning)",
            confidence=0.85,
        )

        return VocalOutput(vocal_spec=vocal_spec.model_dump())

    def _design_voice_persona(self, mood: str) -> VoicePersona:
        """
        Design a parametric voice persona.

        CRITICAL: This creates abstract parameters, NOT a clone of any real voice.
        """
        # Determine characteristics based on mood
        if mood in ["aggressive", "intense", "energetic"]:
            brightness = 0.7
            breathiness = 0.2
            vibrato_depth = 0.3
        elif mood in ["calm", "ethereal", "melancholic"]:
            brightness = 0.4
            breathiness = 0.5
            vibrato_depth = 0.5
        else:
            brightness = 0.5
            breathiness = 0.3
            vibrato_depth = 0.4

        return VoicePersona(
            gender_presentation="feminine",  # Default, can be configured
            age_range="adult",
            vocal_weight="medium",
            brightness=brightness,
            breathiness=breathiness,
            nasality=0.3,
            vibrato_depth=vibrato_depth,
            vibrato_rate=5.5,
            lowest_note=NoteName.G,  # G3
            highest_note=NoteName.E,  # E5
            comfortable_low=NoteName.C,  # C4
            comfortable_high=NoteName.C,  # C5
        )

    def _create_doubles(self) -> list[VocalDouble]:
        """Create vocal double tracks."""
        return [
            VocalDouble(
                name="main_double_L",
                offset_cents=-8,
                delay_ms=18.0,
                level_db=-8.0,
                pan=-0.5,
            ),
            VocalDouble(
                name="main_double_R",
                offset_cents=8,
                delay_ms=22.0,
                level_db=-8.0,
                pan=0.5,
            ),
        ]

    def _create_harmonies(self) -> list[VocalHarmony]:
        """Create backing vocal harmonies."""
        return [
            VocalHarmony(
                interval="3rd",
                direction="above",
                sections=[SectionType.CHORUS],
                level_db=-10.0,
            ),
            VocalHarmony(
                interval="5th",
                direction="above",
                sections=[SectionType.CHORUS],
                level_db=-12.0,
            ),
        ]

    def _create_ad_libs(self, mood: str) -> list[AdLib]:
        """Create ad-lib specifications."""
        ad_libs = []

        if mood in ["energetic", "aggressive", "happy"]:
            ad_libs.extend(
                [
                    AdLib(type="yeah", placement="End of chorus", energy=0.8),
                    AdLib(type="hey", placement="Pre-chorus buildup", energy=0.7),
                ]
            )
        elif mood in ["calm", "ethereal"]:
            ad_libs.append(AdLib(type="oh", placement="End of verse", energy=0.4))

        return ad_libs

    def _create_emotion_arc(self, lyric_spec: dict) -> list[EmotionMarker]:
        """Create emotional arc across sections."""
        markers = []

        sections = lyric_spec.get("sections", [])
        for section in sections:
            section_type = SectionType(section["section_type"])

            if section_type == SectionType.VERSE:
                markers.append(
                    EmotionMarker(
                        section=section_type,
                        emotion="contemplative",
                        intensity=0.5,
                    )
                )
            elif section_type == SectionType.CHORUS:
                markers.append(
                    EmotionMarker(
                        section=section_type,
                        emotion="powerful",
                        intensity=0.85,
                    )
                )
            elif section_type == SectionType.BRIDGE:
                markers.append(
                    EmotionMarker(
                        section=section_type,
                        emotion="vulnerable",
                        intensity=0.6,
                    )
                )

        return markers

    def _determine_delivery_style(self, mood: str) -> str:
        """Determine vocal delivery style from mood."""
        style_map = {
            "energetic": "belted",
            "aggressive": "belted",
            "calm": "sung",
            "ethereal": "whispered",
            "happy": "sung",
            "sad": "sung",
            "dark": "spoken",
            "melancholic": "sung",
        }
        return style_map.get(mood, "sung")
