"""
Creative Director Agent - Melody Expert Edition

World-class creative agent with orchestral-level experience, producer mindset,
and innovation-focused approach. Translates creative briefs into complete SongSpec
with genre-appropriate parameters and professional musical intelligence.

This agent embodies:
- Deep music theory knowledge (scales, modes, voicings, counterpoint)
- Industry-proven melodic techniques
- Professional orchestration understanding
- Genre-specific expertise across 18+ genres
- Innovation-focused approach with producer mindset
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from aether.agents.base import AgentRegistry, BaseAgent
from aether.knowledge import get_genre_manager
from aether.knowledge.theory import (
    get_scale_for_genre,
    get_chord_extensions_for_genre,
    get_genre_progression,
    get_harmonic_rhythm_for_genre,
    CADENCES,
    GENRE_PROGRESSIONS,
)
from aether.knowledge.melody_expert import (
    MelodyExpert,
    get_archetype_for_genre,
    get_hook_formula_for_genre,
    PRODUCER_INSIGHTS,
    MelodicArchetype,
)
from aether.knowledge.orchestration import (
    get_arrangement_for_genre,
    get_structure_for_genre,
    get_dynamic_layers,
    get_wisdom_for_genre,
)
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
    # New expert parameters
    melodic_archetype: Optional[str] = None  # arch, wave, ascending, etc.
    harmonic_complexity: Optional[float] = None  # 0-1, simple to complex
    production_style: Optional[str] = None  # modern, vintage, experimental


class CreativeDirectorOutput(BaseModel):
    """Output from Creative Director."""

    song_spec: dict[str, Any]
    genre_profile_id: str
    creative_decisions: list
    # New expert outputs
    melodic_strategy: dict[str, Any] = Field(default_factory=dict)
    arrangement_plan: dict[str, Any] = Field(default_factory=dict)
    production_notes: list[str] = Field(default_factory=list)


@AgentRegistry.register("creative_director")
class CreativeDirectorAgent(BaseAgent[CreativeDirectorInput, CreativeDirectorOutput]):
    """
    Creative Director Agent - Melody Expert Edition.

    World-class creative intelligence with:
    - Orchestral-level composition expertise
    - Producer mindset and industry knowledge
    - Innovation-focused approach
    - Deep integration with music theory systems

    Responsibilities:
    - Load and validate genre profile with deep understanding
    - Interpret creative brief using producer expertise
    - Generate complete SongSpec with professional parameters
    - Design melodic strategy based on proven formulas
    - Plan arrangement using orchestration knowledge
    - Set musical parameters with theory-backed decisions
    - Establish mood and energy arc for maximum impact
    """

    agent_type = "creative_director"
    agent_name = "Creative Director (Melody Expert)"
    input_schema = CreativeDirectorInput
    output_schema = CreativeDirectorOutput

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.melody_expert: Optional[MelodyExpert] = None

    async def process(
        self,
        input_data: CreativeDirectorInput,
        context: dict[str, Any],
    ) -> CreativeDirectorOutput:
        """
        Process creative direction with expert-level musical intelligence.

        This method leverages:
        - Genre-specific knowledge systems
        - Melodic archetype expertise
        - Professional orchestration understanding
        - Producer mindset and industry wisdom
        """
        genre_manager = get_genre_manager()
        profile = genre_manager.get(input_data.genre_id)

        # Initialize melody expert for this genre
        self.melody_expert = MelodyExpert(genre=input_data.genre_id)

        # ================================================================
        # BPM SELECTION - Theory-backed tempo decision
        # ================================================================
        if input_data.bpm:
            bpm = input_data.bpm
            if not (profile.tempo.min_bpm <= bpm <= profile.tempo.max_bpm):
                logger.warning(
                    f"BPM {bpm} outside genre range ({profile.tempo.min_bpm}-{profile.tempo.max_bpm})"
                )
        else:
            # Select tempo based on mood for more nuanced choice
            bpm = self._select_optimal_bpm(profile, input_data.mood, input_data.creative_brief)

        self.log_decision(
            decision_type="bpm_selection",
            input_summary=f"Requested: {input_data.bpm or 'auto'}",
            output_summary=f"Selected: {bpm} BPM",
            reasoning=f"Genre typical: {profile.tempo.typical_bpm}, range: {profile.tempo.min_bpm}-{profile.tempo.max_bpm}. "
                      f"Mood-adjusted for optimal energy.",
            confidence=0.9 if input_data.bpm else 0.85,
        )

        # ================================================================
        # KEY SELECTION - Scale/mode expertise
        # ================================================================
        if input_data.key:
            key = self._parse_key(input_data.key)
        else:
            key = self._select_optimal_key(input_data.genre_id, profile, input_data.mood)

        # Get recommended scale for melody generation
        recommended_scale = get_scale_for_genre(input_data.genre_id)

        self.log_decision(
            decision_type="key_selection",
            input_summary=f"Requested: {input_data.key or 'auto'}",
            output_summary=f"Selected: {key}",
            reasoning=f"Genre prefers {recommended_scale} scale. Common modes: "
                      f"{[m.value for m in profile.harmony.common_modes]}",
            alternatives=[
                f"{NoteName.C.value} {m.value}" for m in profile.harmony.common_modes[:3]
            ],
            confidence=0.9,
        )

        # ================================================================
        # MOOD ANALYSIS - Deep semantic understanding
        # ================================================================
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
            reasoning="Inferred from creative brief keywords, emotional tone, and genre context",
            confidence=0.8,
        )

        # ================================================================
        # MELODIC STRATEGY - Expert melody design
        # ================================================================
        melodic_archetype = self._determine_melodic_archetype(
            input_data.melodic_archetype,
            input_data.genre_id,
            mood
        )
        hook_formula = get_hook_formula_for_genre(input_data.genre_id)
        harmonic_rhythm = get_harmonic_rhythm_for_genre(input_data.genre_id)

        melodic_strategy = {
            "archetype": melodic_archetype.value,
            "hook_formula": hook_formula.name,
            "recommended_scale": recommended_scale,
            "chord_extensions": get_chord_extensions_for_genre(input_data.genre_id),
            "harmonic_rhythm": harmonic_rhythm.name,
            "changes_per_bar": harmonic_rhythm.changes_per_bar,
            "singability_target": 0.8 if input_data.has_vocals else 0.5,
            "memorability_target": 0.85,
        }

        self.log_decision(
            decision_type="melodic_strategy",
            input_summary=f"Genre: {input_data.genre_id}, Mood: {mood.value}",
            output_summary=f"Archetype: {melodic_archetype.value}, Hook: {hook_formula.name}",
            reasoning=f"Selected {melodic_archetype.value} archetype for {mood.value} mood in "
                      f"{input_data.genre_id}. Using {hook_formula.name} hook formula (effectiveness: "
                      f"{hook_formula.effectiveness:.0%})",
            confidence=0.88,
        )

        # ================================================================
        # ARRANGEMENT PLAN - Orchestration expertise
        # ================================================================
        arrangement = get_arrangement_for_genre(input_data.genre_id)
        structure = get_structure_for_genre(input_data.genre_id)
        dynamic_layers = get_dynamic_layers(input_data.genre_id)

        arrangement_plan = {
            "core_instruments": arrangement.core_instruments if arrangement else ["piano", "bass", "drums"],
            "optional_instruments": arrangement.optional_instruments if arrangement else [],
            "texture": arrangement.typical_texture if arrangement else "melody_chords",
            "density": arrangement.density if arrangement else 0.6,
            "frequency_distribution": arrangement.frequency_distribution if arrangement else {"low": 0.25, "mid": 0.5, "high": 0.25},
            "structure": structure.name if structure else "verse_chorus_pop",
            "total_bars": structure.total_bars if structure else 120,
            "sections": [s.name for s in structure.sections] if structure else ["intro", "verse", "chorus", "verse", "chorus", "outro"],
            "dynamic_layers": len(dynamic_layers),
        }

        self.log_decision(
            decision_type="arrangement_plan",
            input_summary=f"Genre: {input_data.genre_id}",
            output_summary=f"Structure: {arrangement_plan['structure']}, Core: {arrangement_plan['core_instruments'][:3]}",
            reasoning=f"Using {arrangement_plan['structure']} form with {arrangement_plan['texture']} texture. "
                      f"Density: {arrangement_plan['density']:.0%}",
            confidence=0.85,
        )

        # ================================================================
        # PRODUCTION NOTES - Producer wisdom
        # ================================================================
        production_wisdom = get_wisdom_for_genre(input_data.genre_id)
        production_notes = [tip.tip for tip in production_wisdom[:5]]

        # Add genre-specific progression suggestions
        if input_data.genre_id in GENRE_PROGRESSIONS:
            top_prog = GENRE_PROGRESSIONS[input_data.genre_id][0]
            production_notes.append(f"Recommended progression: {top_prog['name']} ({top_prog['style']})")

        # ================================================================
        # DURATION CALCULATION
        # ================================================================
        duration = input_data.duration_seconds
        if not duration:
            if structure:
                # Calculate from structure
                bars = structure.total_bars
                beats_per_bar = 4  # Assuming 4/4
                seconds_per_beat = 60 / bpm
                duration = int(bars * beats_per_bar * seconds_per_beat)
            else:
                duration = (
                    profile.arrangement.typical_duration.min_seconds
                    + profile.arrangement.typical_duration.max_seconds
                ) // 2

        # ================================================================
        # CREATE CREATIVE BRIEF
        # ================================================================
        brief = CreativeBrief(
            theme=self._extract_theme(input_data.creative_brief),
            mood=mood,
            energy_description=self._generate_energy_arc(mood, profile),
            lyrical_direction=input_data.creative_brief if input_data.has_vocals else None,
            reference_era=profile.production.era_reference,
            special_requests=[],
        )

        # ================================================================
        # BUILD SONG SPEC
        # ================================================================
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

        logger.info(
            f"[{self.agent_name}] Created SongSpec with expert parameters: "
            f"BPM={bpm}, Key={key}, Archetype={melodic_archetype.value}, "
            f"Structure={arrangement_plan['structure']}"
        )

        return CreativeDirectorOutput(
            song_spec=song_spec.model_dump(),
            genre_profile_id=input_data.genre_id,
            creative_decisions=[d.__dict__ for d in self.decisions],
            melodic_strategy=melodic_strategy,
            arrangement_plan=arrangement_plan,
            production_notes=production_notes,
        )

    def _select_optimal_bpm(
        self,
        profile,
        mood: Optional[str],
        brief: str,
    ) -> int:
        """Select optimal BPM based on genre, mood, and brief analysis."""
        base_bpm = profile.tempo.typical_bpm

        # Adjust based on mood keywords
        mood_lower = (mood or "").lower()
        brief_lower = brief.lower()

        # High energy moods
        if any(word in mood_lower + brief_lower for word in ["energetic", "intense", "powerful", "fast", "aggressive"]):
            adjustment = min(15, (profile.tempo.max_bpm - base_bpm) // 2)
            return base_bpm + adjustment

        # Low energy moods
        if any(word in mood_lower + brief_lower for word in ["calm", "peaceful", "slow", "relaxed", "ambient"]):
            adjustment = min(15, (base_bpm - profile.tempo.min_bpm) // 2)
            return base_bpm - adjustment

        return base_bpm

    def _select_optimal_key(
        self,
        genre_id: str,
        profile,
        mood: Optional[str],
    ) -> KeySignature:
        """Select optimal key based on genre and mood."""
        preferred_mode = profile.harmony.common_modes[0]

        # Select root based on mood
        mood_lower = (mood or "").lower()

        # Bright moods prefer sharp keys
        if any(word in mood_lower for word in ["happy", "bright", "uplifting", "energetic"]):
            roots = [NoteName.G, NoteName.D, NoteName.A, NoteName.E]
        # Dark moods prefer flat keys
        elif any(word in mood_lower for word in ["dark", "sad", "melancholic", "mysterious"]):
            roots = [NoteName.F, NoteName.Bb, NoteName.Eb, NoteName.Ab]
        else:
            # Neutral - use common keys
            roots = [NoteName.C, NoteName.G, NoteName.A, NoteName.E]

        # Select based on mode
        if preferred_mode in [Mode.MINOR, Mode.DORIAN, Mode.PHRYGIAN]:
            preferred_roots = [NoteName.A, NoteName.E, NoteName.D]
        else:
            preferred_roots = [NoteName.C, NoteName.G, NoteName.F]

        # Find intersection or use first available
        for root in preferred_roots:
            if root in roots:
                return KeySignature(root=root, mode=preferred_mode)

        return KeySignature(root=roots[0], mode=preferred_mode)

    def _determine_melodic_archetype(
        self,
        requested: Optional[str],
        genre_id: str,
        mood: MoodCategory,
    ) -> MelodicArchetype:
        """Determine the best melodic archetype."""
        if requested:
            try:
                return MelodicArchetype(requested)
            except ValueError:
                pass

        # Get genre-appropriate archetype
        archetype = get_archetype_for_genre(genre_id)

        # Adjust based on mood
        mood_overrides = {
            MoodCategory.HAPPY: MelodicArchetype.ARCH,
            MoodCategory.SAD: MelodicArchetype.DESCENDING,
            MoodCategory.ENERGETIC: MelodicArchetype.ASCENDING,
            MoodCategory.CALM: MelodicArchetype.WAVE,
            MoodCategory.AGGRESSIVE: MelodicArchetype.ROCKET,
            MoodCategory.UPLIFTING: MelodicArchetype.ASCENDING,
            MoodCategory.DARK: MelodicArchetype.DESCENDING,
            MoodCategory.ETHEREAL: MelodicArchetype.WAVE,
        }

        return mood_overrides.get(mood, archetype)

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
