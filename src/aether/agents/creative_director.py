"""
Creative Director Agent

Translates creative briefs into complete SongSpec with genre-appropriate parameters.
First agent in the pipeline - sets the creative direction for everything that follows.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from aether.agents.base import AgentRegistry, BaseAgent
from aether.knowledge import get_genre_manager
from aether.schemas.base import KeySignature, Mode, MoodCategory, NoteName, TimeSignature
from aether.schemas.song import CreativeBrief, SongSpec
from aether.storage import ArtifactType

logger = logging.getLogger(__name__)


class CreativeDirectorInput(BaseModel):
    """Input for Creative Director."""

    title: str
    genre_id: str
    creative_brief: str = Field(description="Text description of the song")
    bpm: Optional[int] = None
    key: Optional[str] = None
    mood: Optional[str] = None
    duration_seconds: Optional[int] = None
    has_vocals: bool = True
    random_seed: Optional[int] = None


class CreativeDirectorOutput(BaseModel):
    """Output from Creative Director."""

    song_spec: dict[str, Any]
    genre_profile_id: str
    creative_decisions: list


@AgentRegistry.register("creative_director")
class CreativeDirectorAgent(BaseAgent[CreativeDirectorInput, CreativeDirectorOutput]):
    """
    Creative Director Agent.

    Responsibilities:
    - Load and validate genre profile
    - Interpret creative brief
    - Generate complete SongSpec
    - Set musical parameters within genre bounds
    - Establish mood and energy arc
    """

    agent_type = "creative_director"
    agent_name = "Creative Director"
    input_schema = CreativeDirectorInput
    output_schema = CreativeDirectorOutput

    async def process(
        self,
        input_data: CreativeDirectorInput,
        context: dict[str, Any],
    ) -> CreativeDirectorOutput:
        """Process creative direction."""
        genre_manager = get_genre_manager()
        profile = genre_manager.get(input_data.genre_id)

        # Determine BPM
        if input_data.bpm:
            bpm = input_data.bpm
            # Validate against genre
            if not (profile.tempo.min_bpm <= bpm <= profile.tempo.max_bpm):
                logger.warning(
                    f"BPM {bpm} outside genre range ({profile.tempo.min_bpm}-{profile.tempo.max_bpm})"
                )
        else:
            bpm = profile.tempo.typical_bpm

        self.log_decision(
            decision_type="bpm_selection",
            input_summary=f"Requested: {input_data.bpm or 'auto'}",
            output_summary=f"Selected: {bpm} BPM",
            reasoning=f"Genre typical: {profile.tempo.typical_bpm}, range: {profile.tempo.min_bpm}-{profile.tempo.max_bpm}",
            confidence=0.9 if input_data.bpm else 0.8,
        )

        # Determine key
        if input_data.key:
            key = self._parse_key(input_data.key)
        else:
            # Select from genre-appropriate modes
            preferred_mode = profile.harmony.common_modes[0]
            key = KeySignature(root=NoteName.A, mode=preferred_mode)

        self.log_decision(
            decision_type="key_selection",
            input_summary=f"Requested: {input_data.key or 'auto'}",
            output_summary=f"Selected: {key}",
            reasoning=f"Genre common modes: {[m.value for m in profile.harmony.common_modes]}",
            alternatives=[
                f"{NoteName.C.value} {m.value}" for m in profile.harmony.common_modes[:3]
            ],
            confidence=0.85,
        )

        # Determine mood
        if input_data.mood:
            try:
                mood = MoodCategory(input_data.mood.lower())
            except ValueError:
                mood = MoodCategory.ENERGETIC
        else:
            mood = self._infer_mood_from_brief(input_data.creative_brief)

        self.log_decision(
            decision_type="mood_selection",
            input_summary=f"Brief: {input_data.creative_brief[:50]}...",
            output_summary=f"Selected mood: {mood.value}",
            reasoning="Inferred from creative brief keywords and tone",
            confidence=0.75,
        )

        # Determine duration
        duration = input_data.duration_seconds
        if not duration:
            duration = (
                profile.arrangement.typical_duration.min_seconds
                + profile.arrangement.typical_duration.max_seconds
            ) // 2

        # Create creative brief object
        brief = CreativeBrief(
            theme=self._extract_theme(input_data.creative_brief),
            mood=mood,
            energy_description=self._generate_energy_arc(mood, profile),
            lyrical_direction=input_data.creative_brief if input_data.has_vocals else None,
            reference_era=profile.production.era_reference,
            special_requests=[],
        )

        # Build SongSpec
        song_spec = SongSpec(
            id=uuid4(),
            title=input_data.title,
            artist_name="AETHER",
            genre_id=input_data.genre_id,
            bpm=bpm,
            key=key,
            time_signature=TimeSignature.FOUR_FOUR,
            target_duration_seconds=duration,
            creative_brief=brief,
            primary_mood=mood,
            has_vocals=input_data.has_vocals,
            is_instrumental=not input_data.has_vocals,
            random_seed=input_data.random_seed,
        )

        # Store artifact
        self.store_artifact(
            data=song_spec.model_dump(),
            artifact_type=ArtifactType.SONG_SPEC,
            name="song_spec",
            song_id=str(song_spec.id),
        )

        return CreativeDirectorOutput(
            song_spec=song_spec.model_dump(),
            genre_profile_id=input_data.genre_id,
            creative_decisions=[d.__dict__ for d in self.decisions],
        )

    def _parse_key(self, key_str: str) -> KeySignature:
        """Parse key string like 'Am' or 'C major'."""
        key_str = key_str.strip()

        # Simple parsing
        if len(key_str) >= 2:
            root_char = key_str[0].upper()
            rest = key_str[1:].lower()

            # Handle sharps/flats
            if rest.startswith("#") or rest.startswith("b"):
                root = NoteName(root_char + rest[0])
                rest = rest[1:]
            else:
                root = NoteName(root_char)

            # Determine mode
            if "m" in rest or "min" in rest:
                mode = Mode.MINOR
            elif "dor" in rest:
                mode = Mode.DORIAN
            elif "phry" in rest:
                mode = Mode.PHRYGIAN
            elif "lyd" in rest:
                mode = Mode.LYDIAN
            elif "mix" in rest:
                mode = Mode.MIXOLYDIAN
            else:
                mode = Mode.MAJOR

            return KeySignature(root=root, mode=mode)

        return KeySignature(root=NoteName.C, mode=Mode.MAJOR)

    def _infer_mood_from_brief(self, brief: str) -> MoodCategory:
        """Infer mood from creative brief text."""
        brief_lower = brief.lower()

        mood_keywords = {
            MoodCategory.HAPPY: ["happy", "joy", "celebration", "upbeat", "bright"],
            MoodCategory.SAD: ["sad", "melancholy", "heartbreak", "loss", "tears"],
            MoodCategory.ENERGETIC: ["energy", "power", "drive", "intense", "pump"],
            MoodCategory.CALM: ["calm", "peaceful", "serene", "gentle", "soft"],
            MoodCategory.AGGRESSIVE: ["aggressive", "angry", "rage", "hard", "brutal"],
            MoodCategory.MELANCHOLIC: ["bittersweet", "longing", "wistful", "nostalgic"],
            MoodCategory.UPLIFTING: ["hope", "inspire", "rise", "triumph", "victory"],
            MoodCategory.DARK: ["dark", "shadow", "night", "sinister", "ominous"],
            MoodCategory.NOSTALGIC: ["memory", "past", "remember", "vintage", "retro"],
            MoodCategory.ETHEREAL: ["dream", "float", "ethereal", "ambient", "space"],
        }

        scores = {mood: 0 for mood in MoodCategory}
        for mood, keywords in mood_keywords.items():
            for keyword in keywords:
                if keyword in brief_lower:
                    scores[mood] += 1

        best_mood = max(scores, key=scores.get)
        if scores[best_mood] == 0:
            return MoodCategory.ENERGETIC  # Default

        return best_mood

    def _extract_theme(self, brief: str) -> str:
        """Extract main theme from brief."""
        # Simple extraction - first sentence or first 100 chars
        sentences = brief.split(".")
        if sentences:
            return sentences[0].strip()[:100]
        return brief[:100]

    def _generate_energy_arc(self, mood: MoodCategory, profile) -> str:
        """Generate energy arc description based on mood and genre."""
        arc_type = profile.arrangement.energy_curve_type

        if arc_type == "build":
            return "Starts low, builds continuously to peak at the end"
        elif arc_type == "build_release":
            return "Builds through verse to chorus peak, releases in bridge, final peak"
        elif arc_type == "maintain":
            return "Consistent energy throughout with subtle variations"
        elif arc_type == "wave":
            return "Alternating peaks and valleys throughout"
        else:
            return "Dynamic energy curve matching song sections"
