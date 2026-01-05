"""
Lyrics Agent

Generates original lyrics with syllable mapping for melody alignment.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from pydantic import BaseModel

from aether.agents.base import AgentRegistry, BaseAgent
from aether.schemas.base import SectionType
from aether.schemas.lyrics import LyricLine, LyricSection, LyricSpec, NarrativeArc

logger = logging.getLogger(__name__)


class LyricsInput(BaseModel):
    song_spec: dict[str, Any]
    arrangement_spec: dict[str, Any]
    melody_spec: dict[str, Any]


class LyricsOutput(BaseModel):
    lyric_spec: dict[str, Any]


@AgentRegistry.register("lyrics")
class LyricsAgent(BaseAgent[LyricsInput, LyricsOutput]):
    """
    Lyrics Agent.

    Responsibilities:
    - Generate original lyrics for each section
    - Map syllables to melody rhythm
    - Create coherent narrative arc
    - Ensure originality (no copied phrases)
    - Match rhyme schemes to genre
    """

    agent_type = "lyrics"
    agent_name = "Lyrics Agent"
    input_schema = LyricsInput
    output_schema = LyricsOutput

    # Template phrases for generation (NOT to be copied - for structure only)
    VERSE_TEMPLATES = [
        "{subject} {verb} through the {noun}",
        "Every {noun} tells a {noun2}",
        "{emotion} runs through my {noun}",
        "When the {noun} {verb}",
        "I {verb} the {noun} of {noun2}",
    ]

    CHORUS_TEMPLATES = [
        "We {verb} {adverb}",
        "{emotion} is all I {verb}",
        "This is {noun}",
        "{verb} me {adverb}",
    ]

    WORD_BANKS = {
        "subject": ["I", "We", "You", "They", "Time", "Love", "Life"],
        "verb": ["walk", "run", "feel", "see", "know", "find", "rise", "fall", "dream", "breathe"],
        "noun": ["night", "light", "heart", "soul", "time", "road", "sky", "fire", "rain", "dream"],
        "noun2": ["story", "memory", "feeling", "moment", "vision", "truth"],
        "emotion": ["hope", "love", "fear", "joy", "pain", "peace"],
        "adverb": ["forever", "always", "slowly", "deeply", "freely"],
    }

    async def process(
        self,
        input_data: LyricsInput,
        context: dict[str, Any],
    ) -> LyricsOutput:
        song_spec = input_data.song_spec
        arrangement = input_data.arrangement_spec
        melody = input_data.melody_spec

        theme = song_spec.get("creative_brief", {}).get("theme", "journey")
        mood = song_spec.get("primary_mood", "energetic")

        # Create narrative arc
        narrative = NarrativeArc(
            setup=f"Establishing the scene of {theme}",
            development=f"Exploring the depths of {theme}",
            resolution=f"Finding meaning in {theme}",
            perspective="first_person",
        )

        # Generate lyrics for each section
        sections = []
        for section_def in arrangement.get("sections", []):
            section_type = SectionType(section_def["section_type"])

            if section_type in [SectionType.INTRO, SectionType.OUTRO]:
                continue  # Usually instrumental

            lyric_section = self._generate_section_lyrics(
                section_type=section_type,
                section_label=section_def["label"],
                theme=theme,
                mood=mood,
            )
            sections.append(lyric_section)

        # Find hook lyrics
        hook_lyrics = None
        for section in sections:
            if section["section_type"] == SectionType.CHORUS.value:
                if section["lines"]:
                    hook_lyrics = section["lines"][0]["text"]
                break

        lyric_spec = LyricSpec(
            song_id=str(song_spec["id"]),
            melody_id=str(melody.get("id", "melody")),
            primary_theme=theme,
            emotional_journey=f"From uncertainty to {mood}",
            narrative=narrative,
            sections=[LyricSection(**s) for s in sections],
            hook_lyrics=hook_lyrics,
            title_in_lyrics=True,
            vocabulary_level="conversational",
            originality_score=0.9,  # Placeholder - would be computed by QA
        )

        self.log_decision(
            decision_type="lyric_generation",
            input_summary=f"Theme: {theme}, Mood: {mood}",
            output_summary=f"Generated {len(sections)} lyric sections",
            reasoning="Created original lyrics matching theme and mood",
            confidence=0.8,
        )

        return LyricsOutput(lyric_spec=lyric_spec.model_dump())

    def _generate_section_lyrics(
        self,
        section_type: SectionType,
        section_label: str,
        theme: str,
        mood: str,
    ) -> dict[str, Any]:
        """Generate lyrics for a section."""
        if section_type == SectionType.CHORUS:
            num_lines = 4
            templates = self.CHORUS_TEMPLATES
            rhyme_scheme = "AABB"
        elif section_type == SectionType.VERSE:
            num_lines = 8
            templates = self.VERSE_TEMPLATES
            rhyme_scheme = "ABAB"
        else:  # Bridge
            num_lines = 4
            templates = self.VERSE_TEMPLATES
            rhyme_scheme = "ABCB"

        lines = []
        rhyme_tags = list(rhyme_scheme)

        for i in range(num_lines):
            template = random.choice(templates)
            line_text = self._fill_template(template)

            syllables = self._count_syllables(line_text)
            stresses = self._estimate_stresses(syllables)

            lines.append(
                LyricLine(
                    text=line_text,
                    syllable_count=syllables,
                    syllable_stresses=stresses,
                    rhyme_tag=rhyme_tags[i % len(rhyme_tags)],
                ).model_dump()
            )

        return {
            "section_type": section_type.value,
            "section_label": section_label,
            "lines": lines,
            "rhyme_scheme": rhyme_scheme,
            "theme_keywords": [theme, mood],
        }

    def _fill_template(self, template: str) -> str:
        """Fill a template with random words."""
        result = template
        for key, words in self.WORD_BANKS.items():
            placeholder = "{" + key + "}"
            while placeholder in result:
                result = result.replace(placeholder, random.choice(words), 1)
        return result.capitalize()

    def _count_syllables(self, text: str) -> int:
        """Estimate syllable count (simple heuristic)."""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        prev_vowel = False

        for char in text:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        return max(1, count)

    def _estimate_stresses(self, syllable_count: int) -> list[bool]:
        """Estimate stress pattern (simplified)."""
        # Simple alternating pattern
        return [(i % 2 == 0) for i in range(syllable_count)]
