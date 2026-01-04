"""
LyricSpec - Lyrics with syllable mapping.

Purpose: Defines lyrics, themes, rhyme schemes, and syllable-rhythm alignment.
"""

from typing import Optional

from pydantic import Field

from aether.schemas.base import (
    AetherBaseModel,
    IdentifiableModel,
    SectionType,
)


class LyricLine(AetherBaseModel):
    """A single line of lyrics."""

    text: str = Field(min_length=1)
    syllable_count: int = Field(ge=1)
    syllable_stresses: list[bool] = Field(
        default_factory=list, description="Stress pattern for each syllable"
    )
    rhyme_tag: Optional[str] = Field(
        default=None, description="Rhyme group identifier (A, B, C, etc.)"
    )


class LyricSection(AetherBaseModel):
    """Lyrics for a song section."""

    section_type: SectionType
    section_label: str = Field(description="e.g., 'Verse 1', 'Chorus'")
    lines: list[LyricLine] = Field(min_length=1)
    rhyme_scheme: str = Field(default="AABB", description="Rhyme scheme pattern")
    theme_keywords: list[str] = Field(
        default_factory=list, description="Key thematic words in this section"
    )


class NarrativeArc(AetherBaseModel):
    """Overall narrative structure of lyrics."""

    setup: str = Field(description="Opening premise/situation")
    development: str = Field(description="How the story/theme develops")
    resolution: str = Field(description="How it concludes/resolves")
    perspective: str = Field(
        default="first_person", description="first_person, second_person, third_person"
    )


class LyricSpec(IdentifiableModel):
    """
    Complete lyric specification for a song.

    Defines lyrics with syllable-rhythm mapping, rhyme schemes, and thematic content.
    """

    # Reference
    song_id: str = Field(description="Reference to parent SongSpec")
    melody_id: str = Field(description="Reference to MelodySpec for rhythm alignment")

    # Theme
    primary_theme: str = Field(description="Main lyrical theme")
    secondary_themes: list[str] = Field(default_factory=list, max_length=3)
    emotional_journey: str = Field(description="Emotional arc description")
    narrative: NarrativeArc

    # Content
    sections: list[LyricSection] = Field(min_length=1)

    # Hook lyrics
    hook_lyrics: Optional[str] = Field(
        default=None, max_length=100, description="The main lyrical hook"
    )
    title_in_lyrics: bool = Field(default=True, description="Does the title appear in lyrics?")

    # Vocabulary
    vocabulary_level: str = Field(
        default="conversational",
        description="academic, literary, conversational, slang, poetic",
    )
    banned_words: list[str] = Field(default_factory=list, description="Words to avoid")

    # Originality
    ngram_overlap_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="N-gram overlap with corpus (lower is better)"
    )
    semantic_distance: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Semantic distance from known lyrics"
    )
    originality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "example-song-id",
                "melody_id": "example-melody-id",
                "primary_theme": "Finding hope in darkness",
                "emotional_journey": "From despair through struggle to hope",
                "narrative": {
                    "setup": "Protagonist in a dark place",
                    "development": "Searching for light, finding strength",
                    "resolution": "Emerging with renewed purpose",
                    "perspective": "first_person",
                },
                "sections": [
                    {
                        "section_type": "verse",
                        "section_label": "Verse 1",
                        "lines": [
                            {
                                "text": "Walking through the shadows",
                                "syllable_count": 6,
                                "rhyme_tag": "A",
                            },
                            {
                                "text": "Searching for the light",
                                "syllable_count": 5,
                                "rhyme_tag": "B",
                            },
                        ],
                        "rhyme_scheme": "ABAB",
                    }
                ],
                "hook_lyrics": "We rise from the ashes",
                "title_in_lyrics": True,
            }
        }
