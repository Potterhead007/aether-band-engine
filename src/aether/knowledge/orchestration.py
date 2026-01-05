"""
AETHER Orchestration & Arrangement Knowledge Base

Professional orchestration techniques, arrangement patterns, instrument
combinations, and production knowledge for world-class music generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================================
# INSTRUMENT RANGES - Professional Orchestral Ranges
# ============================================================================

@dataclass
class InstrumentRange:
    """MIDI range and optimal tessitura for an instrument."""
    name: str
    lowest_note: int  # MIDI number
    highest_note: int  # MIDI number
    tessitura_low: int  # Optimal low
    tessitura_high: int  # Optimal high
    transposition: int = 0  # Semitones (for transposing instruments)
    clef: str = "treble"
    description: str = ""


ORCHESTRAL_RANGES: dict[str, InstrumentRange] = {
    # ---- Strings ----
    "violin": InstrumentRange(
        "Violin", 55, 103, 60, 88,
        description="G3 to G7, sweet spot D4-D6"
    ),
    "viola": InstrumentRange(
        "Viola", 48, 91, 55, 79,
        clef="alto",
        description="C3 to G6, sweet spot G3-G5"
    ),
    "cello": InstrumentRange(
        "Cello", 36, 76, 43, 67,
        clef="bass",
        description="C2 to E5, sweet spot G2-G4"
    ),
    "double_bass": InstrumentRange(
        "Double Bass", 28, 60, 33, 55,
        clef="bass",
        transposition=-12,
        description="E1 to C4, sounds octave lower"
    ),

    # ---- Woodwinds ----
    "flute": InstrumentRange(
        "Flute", 60, 96, 67, 86,
        description="C4 to C7, sweet spot G4-D6"
    ),
    "piccolo": InstrumentRange(
        "Piccolo", 74, 108, 79, 98,
        transposition=12,
        description="D5 to C8, sounds octave higher"
    ),
    "oboe": InstrumentRange(
        "Oboe", 58, 91, 62, 82,
        description="Bb3 to G6, sweet spot D4-A5"
    ),
    "english_horn": InstrumentRange(
        "English Horn", 52, 81, 57, 74,
        transposition=-7,
        description="E3 to A5, sounds P5 lower"
    ),
    "clarinet_bb": InstrumentRange(
        "Clarinet (Bb)", 50, 94, 57, 84,
        transposition=-2,
        description="D3 to Bb6, sounds M2 lower"
    ),
    "bass_clarinet": InstrumentRange(
        "Bass Clarinet", 38, 77, 43, 67,
        transposition=-14,
        clef="bass",
        description="Db2 to F5, sounds M9 lower"
    ),
    "bassoon": InstrumentRange(
        "Bassoon", 34, 75, 41, 65,
        clef="bass",
        description="Bb1 to Eb5, sweet spot F2-Bb4"
    ),
    "contrabassoon": InstrumentRange(
        "Contrabassoon", 22, 53, 29, 48,
        transposition=-12,
        clef="bass",
        description="Bb0 to F3, sounds octave lower"
    ),

    # ---- Brass ----
    "french_horn": InstrumentRange(
        "French Horn", 34, 77, 43, 67,
        transposition=-7,
        description="B1 to F5, sounds P5 lower"
    ),
    "trumpet_bb": InstrumentRange(
        "Trumpet (Bb)", 55, 84, 60, 77,
        transposition=-2,
        description="G3 to C6, sounds M2 lower"
    ),
    "trombone": InstrumentRange(
        "Trombone", 40, 72, 46, 65,
        clef="bass",
        description="E2 to C5, sweet spot Bb2-Bb4"
    ),
    "bass_trombone": InstrumentRange(
        "Bass Trombone", 34, 65, 38, 58,
        clef="bass",
        description="Bb1 to F4, extends low range"
    ),
    "tuba": InstrumentRange(
        "Tuba", 28, 58, 33, 52,
        clef="bass",
        description="E1 to Bb3, foundation of brass"
    ),

    # ---- Keyboards ----
    "piano": InstrumentRange(
        "Piano", 21, 108, 36, 96,
        description="A0 to C8, full 88 keys"
    ),
    "organ": InstrumentRange(
        "Organ", 24, 108, 36, 96,
        description="C1 to C8, varies by stops"
    ),

    # ---- Voices ----
    "soprano": InstrumentRange(
        "Soprano", 60, 84, 65, 79,
        description="C4 to C6, sweet spot F4-G5"
    ),
    "mezzo_soprano": InstrumentRange(
        "Mezzo Soprano", 57, 81, 62, 76,
        description="A3 to A5"
    ),
    "alto": InstrumentRange(
        "Alto", 53, 77, 58, 72,
        description="F3 to F5"
    ),
    "tenor": InstrumentRange(
        "Tenor", 48, 72, 53, 67,
        description="C3 to C5, sweet spot F3-G4"
    ),
    "baritone": InstrumentRange(
        "Baritone", 45, 69, 50, 64,
        description="A2 to A4"
    ),
    "bass": InstrumentRange(
        "Bass Voice", 40, 64, 45, 60,
        clef="bass",
        description="E2 to E4, rich low range"
    ),

    # ---- Guitar Family ----
    "acoustic_guitar": InstrumentRange(
        "Acoustic Guitar", 40, 88, 45, 79,
        description="E2 to E6, 6 strings"
    ),
    "electric_guitar": InstrumentRange(
        "Electric Guitar", 40, 88, 45, 84,
        description="E2 to E6, extended with effects"
    ),
    "bass_guitar": InstrumentRange(
        "Bass Guitar", 28, 60, 33, 55,
        clef="bass",
        description="E1 to C4, 4-string standard"
    ),
    "bass_guitar_5string": InstrumentRange(
        "5-String Bass", 23, 60, 28, 55,
        clef="bass",
        description="B0 to C4, extended low range"
    ),
}


# ============================================================================
# TEXTURE TYPES
# ============================================================================

class TextureType(Enum):
    """Musical texture classifications."""
    MONOPHONIC = "monophonic"  # Single melodic line
    HOMOPHONIC = "homophonic"  # Melody with accompaniment
    POLYPHONIC = "polyphonic"  # Multiple independent melodies
    HETEROPHONIC = "heterophonic"  # Same melody with variations
    HOMORHYTHMIC = "homorhythmic"  # Same rhythm, different notes
    ANTIPHONAL = "antiphonal"  # Call and response


@dataclass
class TextureDefinition:
    """Detailed texture definition."""
    type: TextureType
    description: str
    voice_count_min: int
    voice_count_max: int
    genre_affinity: dict[str, float]
    complexity: float  # 0-1


TEXTURES: dict[str, TextureDefinition] = {
    "unison": TextureDefinition(
        TextureType.MONOPHONIC,
        "All instruments playing same notes",
        1, 1,
        {"orchestral": 0.8, "rock": 0.6, "metal": 0.9},
        0.1
    ),
    "octave_unison": TextureDefinition(
        TextureType.MONOPHONIC,
        "Same melody in octaves",
        2, 4,
        {"orchestral": 0.9, "rock": 0.8, "pop": 0.7},
        0.2
    ),
    "melody_chords": TextureDefinition(
        TextureType.HOMOPHONIC,
        "Single melody with chord accompaniment",
        2, 5,
        {"pop": 1.0, "rock": 0.9, "jazz": 0.7, "classical": 0.8},
        0.3
    ),
    "melody_bass": TextureDefinition(
        TextureType.HOMOPHONIC,
        "Melody with bass line only",
        2, 2,
        {"jazz": 0.8, "funk": 0.9, "r-and-b": 0.8},
        0.3
    ),
    "block_chords": TextureDefinition(
        TextureType.HOMORHYTHMIC,
        "Chords moving in same rhythm",
        3, 6,
        {"jazz": 0.9, "gospel": 0.9, "pop": 0.7},
        0.4
    ),
    "counterpoint_2voice": TextureDefinition(
        TextureType.POLYPHONIC,
        "Two independent melodic lines",
        2, 2,
        {"classical": 0.9, "baroque": 1.0, "jazz": 0.6},
        0.6
    ),
    "counterpoint_3voice": TextureDefinition(
        TextureType.POLYPHONIC,
        "Three independent melodic lines",
        3, 3,
        {"classical": 0.9, "baroque": 1.0},
        0.7
    ),
    "fugal": TextureDefinition(
        TextureType.POLYPHONIC,
        "Imitative counterpoint (fugue style)",
        3, 5,
        {"baroque": 1.0, "classical": 0.8},
        0.9
    ),
    "call_response": TextureDefinition(
        TextureType.ANTIPHONAL,
        "Question-answer between voices/sections",
        2, 8,
        {"gospel": 1.0, "funk": 0.8, "jazz": 0.7, "blues": 0.9},
        0.4
    ),
    "layered": TextureDefinition(
        TextureType.HETEROPHONIC,
        "Multiple layers of varied activity",
        3, 8,
        {"ambient": 0.9, "electronic": 0.8, "cinematic": 0.9},
        0.5
    ),
    "pad_melody": TextureDefinition(
        TextureType.HOMOPHONIC,
        "Sustained pad with floating melody",
        2, 4,
        {"ambient": 1.0, "new_age": 0.9, "cinematic": 0.8},
        0.3
    ),
}


# ============================================================================
# ARRANGEMENT TECHNIQUES
# ============================================================================

@dataclass
class ArrangementTechnique:
    """An orchestration/arrangement technique."""
    name: str
    description: str
    instruments_required: int  # Minimum instruments
    complexity: float  # 0-1
    effect: str  # sonic result
    genre_affinity: dict[str, float]


ARRANGEMENT_TECHNIQUES: dict[str, ArrangementTechnique] = {
    # ---- Doubling Techniques ----
    "unison_doubling": ArrangementTechnique(
        "Unison Doubling",
        "Two or more instruments playing exact same notes",
        2, 0.2,
        "Thickens sound, increases presence",
        {"orchestral": 0.9, "rock": 0.8, "pop": 0.7}
    ),
    "octave_doubling": ArrangementTechnique(
        "Octave Doubling",
        "Same melody played in different octaves",
        2, 0.3,
        "Extends range, adds fullness",
        {"orchestral": 1.0, "rock": 0.9, "cinematic": 0.9}
    ),
    "third_doubling": ArrangementTechnique(
        "Third Doubling",
        "Melody doubled a third above or below",
        2, 0.4,
        "Sweet, consonant harmonization",
        {"pop": 0.9, "country": 0.9, "gospel": 0.8}
    ),
    "sixth_doubling": ArrangementTechnique(
        "Sixth Doubling",
        "Melody doubled a sixth above or below",
        2, 0.4,
        "Rich, full harmonization",
        {"jazz": 0.8, "r-and-b": 0.9, "pop": 0.8}
    ),

    # ---- Voicing Techniques ----
    "close_voicing": ArrangementTechnique(
        "Close Voicing",
        "Notes within one octave",
        3, 0.3,
        "Compact, dense sound",
        {"jazz": 0.8, "pop": 0.9, "r-and-b": 0.8}
    ),
    "open_voicing": ArrangementTechnique(
        "Open Voicing",
        "Notes spread across multiple octaves",
        3, 0.4,
        "Spacious, orchestral sound",
        {"orchestral": 1.0, "cinematic": 0.9, "ambient": 0.8}
    ),
    "drop2_voicing": ArrangementTechnique(
        "Drop 2 Voicing",
        "Second voice from top dropped an octave",
        4, 0.5,
        "Jazz guitar/piano sound",
        {"jazz": 1.0, "neo-soul": 0.8}
    ),
    "spread_voicing": ArrangementTechnique(
        "Spread Voicing",
        "Wide intervals between voices",
        4, 0.5,
        "Dramatic, cinematic sound",
        {"cinematic": 1.0, "orchestral": 0.9}
    ),

    # ---- Textural Techniques ----
    "pedal_point": ArrangementTechnique(
        "Pedal Point",
        "Sustained bass note while harmony moves above",
        2, 0.3,
        "Creates tension, builds anticipation",
        {"classical": 0.9, "rock": 0.8, "electronic": 0.7}
    ),
    "ostinato": ArrangementTechnique(
        "Ostinato",
        "Repeated musical pattern throughout section",
        1, 0.3,
        "Creates drive, hypnotic effect",
        {"electronic": 1.0, "minimalist": 1.0, "rock": 0.7}
    ),
    "alberti_bass": ArrangementTechnique(
        "Alberti Bass",
        "Broken chord pattern in left hand",
        1, 0.3,
        "Classical accompaniment feel",
        {"classical": 1.0, "romantic": 0.9}
    ),
    "stride": ArrangementTechnique(
        "Stride",
        "Alternating bass note and chord",
        1, 0.5,
        "Jazz piano feel, rhythmic drive",
        {"jazz": 1.0, "ragtime": 1.0, "swing": 0.9}
    ),
    "arpeggiation": ArrangementTechnique(
        "Arpeggiation",
        "Chord notes played in sequence",
        1, 0.2,
        "Creates movement, fills space",
        {"pop": 0.9, "ambient": 0.8, "classical": 0.9}
    ),

    # ---- Section Writing ----
    "soli": ArrangementTechnique(
        "Soli",
        "Entire section plays melody together in harmony",
        4, 0.6,
        "Big band section sound",
        {"jazz": 1.0, "big_band": 1.0, "latin": 0.8}
    ),
    "tutti": ArrangementTechnique(
        "Tutti",
        "Full ensemble plays together",
        6, 0.5,
        "Maximum power and impact",
        {"orchestral": 1.0, "cinematic": 0.9, "rock": 0.7}
    ),
    "concertino": ArrangementTechnique(
        "Concertino",
        "Small group featured against full ensemble",
        4, 0.7,
        "Contrast, featured soloists",
        {"baroque": 1.0, "classical": 0.9}
    ),

    # ---- Modern/Electronic ----
    "layering": ArrangementTechnique(
        "Layering",
        "Multiple sounds stacked for thickness",
        2, 0.4,
        "Modern, full sound",
        {"electronic": 1.0, "pop": 0.9, "hip-hop": 0.8}
    ),
    "frequency_stacking": ArrangementTechnique(
        "Frequency Stacking",
        "Different instruments in different frequency ranges",
        3, 0.5,
        "Clear mix, full spectrum",
        {"electronic": 1.0, "pop": 0.9, "hip-hop": 0.9}
    ),
    "call_response": ArrangementTechnique(
        "Call and Response",
        "Musical dialogue between parts",
        2, 0.4,
        "Dynamic, conversational feel",
        {"gospel": 1.0, "funk": 0.9, "blues": 0.9, "jazz": 0.8}
    ),
}


# ============================================================================
# ORCHESTRAL SECTIONS
# ============================================================================

@dataclass
class OrchestraSection:
    """An orchestral section definition."""
    name: str
    instruments: list[str]
    role: str
    frequency_range: str  # low, mid-low, mid, mid-high, high
    color: str  # warm, bright, dark, neutral
    blend_with: list[str]  # sections that blend well


ORCHESTRA_SECTIONS: dict[str, OrchestraSection] = {
    "strings": OrchestraSection(
        "Strings",
        ["violin", "viola", "cello", "double_bass"],
        "Foundation, melody, harmony, everything",
        "full",
        "warm",
        ["woodwinds", "brass", "choir"]
    ),
    "high_strings": OrchestraSection(
        "High Strings",
        ["violin"],
        "Melody, soaring lines, shimmer",
        "high",
        "bright",
        ["flute", "oboe", "clarinet"]
    ),
    "low_strings": OrchestraSection(
        "Low Strings",
        ["cello", "double_bass"],
        "Bass foundation, dark colors",
        "low",
        "dark",
        ["bassoon", "trombone", "tuba"]
    ),
    "woodwinds": OrchestraSection(
        "Woodwinds",
        ["flute", "oboe", "clarinet_bb", "bassoon"],
        "Color, melody, countermelody",
        "mid-high",
        "neutral",
        ["strings", "brass"]
    ),
    "high_woodwinds": OrchestraSection(
        "High Woodwinds",
        ["flute", "piccolo", "oboe"],
        "Sparkle, bird-like effects",
        "high",
        "bright",
        ["violin", "trumpet"]
    ),
    "low_woodwinds": OrchestraSection(
        "Low Woodwinds",
        ["bass_clarinet", "bassoon", "contrabassoon"],
        "Dark colors, bass reinforcement",
        "low",
        "dark",
        ["cello", "tuba", "trombone"]
    ),
    "brass": OrchestraSection(
        "Brass",
        ["french_horn", "trumpet_bb", "trombone", "tuba"],
        "Power, fanfares, majesty",
        "mid",
        "bright",
        ["strings", "woodwinds"]
    ),
    "high_brass": OrchestraSection(
        "High Brass",
        ["trumpet_bb", "french_horn"],
        "Fanfares, heroic themes",
        "mid-high",
        "bright",
        ["high_strings", "high_woodwinds"]
    ),
    "low_brass": OrchestraSection(
        "Low Brass",
        ["trombone", "bass_trombone", "tuba"],
        "Power, weight, darkness",
        "low",
        "dark",
        ["low_strings", "low_woodwinds"]
    ),
    "percussion": OrchestraSection(
        "Percussion",
        ["timpani", "snare", "cymbals", "bass_drum"],
        "Rhythm, accents, color",
        "full",
        "neutral",
        ["brass", "full_orchestra"]
    ),
    "choir": OrchestraSection(
        "Choir",
        ["soprano", "alto", "tenor", "bass"],
        "Text, emotion, power",
        "full",
        "warm",
        ["strings", "brass"]
    ),
}


# ============================================================================
# GENRE-SPECIFIC ARRANGEMENTS
# ============================================================================

@dataclass
class GenreArrangement:
    """Typical arrangement pattern for a genre."""
    genre: str
    core_instruments: list[str]
    optional_instruments: list[str]
    typical_texture: str
    typical_techniques: list[str]
    frequency_distribution: dict[str, float]  # low/mid/high balance
    density: float  # 0-1, how full the arrangement is
    description: str


GENRE_ARRANGEMENTS: dict[str, GenreArrangement] = {
    "pop": GenreArrangement(
        "pop",
        ["piano", "acoustic_guitar", "bass_guitar", "drums", "vocals"],
        ["synth_pad", "strings", "electric_guitar"],
        "melody_chords",
        ["layering", "arpeggiation", "octave_doubling"],
        {"low": 0.2, "mid": 0.5, "high": 0.3},
        0.6,
        "Clean, polished, vocal-forward"
    ),
    "rock": GenreArrangement(
        "rock",
        ["electric_guitar", "bass_guitar", "drums", "vocals"],
        ["piano", "organ", "synth"],
        "melody_chords",
        ["octave_doubling", "unison_doubling", "pedal_point"],
        {"low": 0.3, "mid": 0.4, "high": 0.3},
        0.7,
        "Guitar-driven, powerful rhythm section"
    ),
    "jazz": GenreArrangement(
        "jazz",
        ["piano", "double_bass", "drums", "saxophone"],
        ["trumpet", "trombone", "guitar", "vibraphone"],
        "melody_bass",
        ["drop2_voicing", "stride", "call_response"],
        {"low": 0.25, "mid": 0.45, "high": 0.3},
        0.5,
        "Interactive, improvisation-ready, sophisticated harmony"
    ),
    "orchestral": GenreArrangement(
        "orchestral",
        ["strings", "woodwinds", "brass", "percussion"],
        ["harp", "piano", "choir"],
        "layered",
        ["tutti", "soli", "spread_voicing", "octave_doubling"],
        {"low": 0.3, "mid": 0.4, "high": 0.3},
        0.8,
        "Full, cinematic, dynamic range"
    ),
    "electronic": GenreArrangement(
        "electronic",
        ["synth_lead", "synth_bass", "drums", "synth_pad"],
        ["vocals", "arpeggios", "fx"],
        "layered",
        ["layering", "frequency_stacking", "ostinato"],
        {"low": 0.35, "mid": 0.35, "high": 0.3},
        0.7,
        "Synthetic, layered, frequency-conscious"
    ),
    "hip-hop": GenreArrangement(
        "hip-hop",
        ["bass_808", "drums", "samples", "vocals"],
        ["synth", "piano", "strings"],
        "melody_bass",
        ["layering", "frequency_stacking"],
        {"low": 0.4, "mid": 0.35, "high": 0.25},
        0.5,
        "Bass-heavy, vocal-focused, sample-based"
    ),
    "r-and-b": GenreArrangement(
        "r-and-b",
        ["keys", "bass", "drums", "vocals"],
        ["guitar", "strings", "horns"],
        "melody_chords",
        ["sixth_doubling", "close_voicing", "layering"],
        {"low": 0.3, "mid": 0.45, "high": 0.25},
        0.6,
        "Smooth, lush harmonies, groove-focused"
    ),
    "funk": GenreArrangement(
        "funk",
        ["bass", "drums", "guitar", "keys"],
        ["horns", "vocals", "percussion"],
        "call_response",
        ["call_response", "pedal_point", "ostinato"],
        {"low": 0.35, "mid": 0.4, "high": 0.25},
        0.7,
        "Rhythmic, syncopated, interactive"
    ),
    "ambient": GenreArrangement(
        "ambient",
        ["synth_pad", "piano", "strings"],
        ["vocals", "guitar", "bells"],
        "pad_melody",
        ["open_voicing", "arpeggiation", "pedal_point"],
        {"low": 0.2, "mid": 0.5, "high": 0.3},
        0.3,
        "Spacious, atmospheric, minimal"
    ),
    "trap": GenreArrangement(
        "trap",
        ["bass_808", "drums", "hi_hats", "vocals"],
        ["synth", "bells", "pad"],
        "melody_bass",
        ["layering", "frequency_stacking"],
        {"low": 0.45, "mid": 0.3, "high": 0.25},
        0.4,
        "Heavy bass, rapid hi-hats, minimal melody"
    ),
    "house": GenreArrangement(
        "house",
        ["drums", "bass", "synth", "vocals"],
        ["piano", "strings", "brass"],
        "layered",
        ["arpeggiation", "ostinato", "layering"],
        {"low": 0.3, "mid": 0.4, "high": 0.3},
        0.65,
        "Four-on-floor, building energy"
    ),
    "neo-soul": GenreArrangement(
        "neo-soul",
        ["keys", "bass", "drums", "vocals"],
        ["guitar", "horns", "strings"],
        "melody_chords",
        ["drop2_voicing", "third_doubling", "arpeggiation"],
        {"low": 0.25, "mid": 0.5, "high": 0.25},
        0.55,
        "Warm, organic, jazz-influenced"
    ),
}


# ============================================================================
# SONG STRUCTURE TEMPLATES
# ============================================================================

@dataclass
class SongSection:
    """A section of a song."""
    name: str
    bars: int
    energy: float  # 0-1
    density: float  # 0-1
    instruments_active: list[str]  # which instruments play
    role: str  # purpose of this section


@dataclass
class SongStructure:
    """Complete song structure template."""
    name: str
    genre: str
    sections: list[SongSection]
    total_bars: int
    description: str


SONG_STRUCTURES: dict[str, SongStructure] = {
    "verse_chorus_pop": SongStructure(
        "Standard Pop",
        "pop",
        [
            SongSection("intro", 8, 0.3, 0.3, ["drums", "bass", "keys"], "establish groove"),
            SongSection("verse1", 16, 0.4, 0.5, ["drums", "bass", "keys", "guitar"], "story begins"),
            SongSection("prechorus", 8, 0.6, 0.6, ["drums", "bass", "keys", "guitar", "strings"], "build tension"),
            SongSection("chorus", 16, 0.8, 0.8, ["all"], "release, hook"),
            SongSection("verse2", 16, 0.5, 0.5, ["drums", "bass", "keys", "guitar"], "develop story"),
            SongSection("prechorus", 8, 0.6, 0.65, ["drums", "bass", "keys", "guitar", "strings"], "build again"),
            SongSection("chorus", 16, 0.85, 0.85, ["all"], "bigger release"),
            SongSection("bridge", 8, 0.5, 0.4, ["keys", "vocals", "strings"], "contrast"),
            SongSection("chorus", 16, 0.9, 0.9, ["all"], "final chorus"),
            SongSection("outro", 8, 0.4, 0.3, ["drums", "bass", "keys"], "resolve"),
        ],
        120,
        "Standard radio pop structure"
    ),

    "aaba_jazz": SongStructure(
        "AABA (32-bar)",
        "jazz",
        [
            SongSection("head_a1", 8, 0.5, 0.4, ["all"], "main theme"),
            SongSection("head_a2", 8, 0.5, 0.4, ["all"], "theme repeat"),
            SongSection("head_b", 8, 0.6, 0.5, ["all"], "bridge/contrast"),
            SongSection("head_a3", 8, 0.5, 0.4, ["all"], "theme return"),
            SongSection("solo1", 32, 0.6, 0.5, ["piano", "bass", "drums"], "piano solo"),
            SongSection("solo2", 32, 0.65, 0.5, ["saxophone", "bass", "drums"], "sax solo"),
            SongSection("head_out", 32, 0.5, 0.5, ["all"], "out head"),
        ],
        128,
        "Standard jazz standard form"
    ),

    "edm_drop": SongStructure(
        "EDM Build-Drop",
        "electronic",
        [
            SongSection("intro", 16, 0.3, 0.2, ["pad", "fx"], "atmosphere"),
            SongSection("buildup1", 16, 0.5, 0.4, ["drums_light", "bass", "synth"], "energy rise"),
            SongSection("drop1", 16, 0.9, 0.9, ["all"], "main drop"),
            SongSection("breakdown", 16, 0.3, 0.2, ["pad", "vocals", "fx"], "contrast"),
            SongSection("buildup2", 16, 0.6, 0.5, ["drums", "bass", "synth", "riser"], "bigger build"),
            SongSection("drop2", 16, 1.0, 1.0, ["all"], "massive drop"),
            SongSection("outro", 16, 0.4, 0.3, ["drums", "pad"], "wind down"),
        ],
        112,
        "Standard EDM structure with build-drop"
    ),

    "blues_12bar": SongStructure(
        "12-Bar Blues",
        "blues",
        [
            SongSection("intro", 4, 0.4, 0.4, ["guitar", "drums", "bass"], "establish feel"),
            SongSection("verse1", 12, 0.5, 0.5, ["guitar", "drums", "bass", "vocals"], "first verse"),
            SongSection("verse2", 12, 0.55, 0.55, ["guitar", "drums", "bass", "vocals"], "second verse"),
            SongSection("solo", 12, 0.7, 0.6, ["guitar", "drums", "bass"], "guitar solo"),
            SongSection("verse3", 12, 0.6, 0.6, ["guitar", "drums", "bass", "vocals"], "final verse"),
            SongSection("outro", 4, 0.5, 0.4, ["guitar", "drums", "bass"], "turnaround"),
        ],
        56,
        "Traditional 12-bar blues structure"
    ),

    "trap_beat": SongStructure(
        "Trap Beat",
        "trap",
        [
            SongSection("intro", 8, 0.3, 0.2, ["melody", "fx"], "dark atmosphere"),
            SongSection("verse1", 16, 0.5, 0.5, ["drums", "bass", "melody", "vocals"], "verse"),
            SongSection("hook", 8, 0.7, 0.6, ["drums", "bass", "melody", "vocals"], "hook"),
            SongSection("verse2", 16, 0.55, 0.55, ["drums", "bass", "melody", "vocals"], "verse 2"),
            SongSection("hook", 8, 0.75, 0.65, ["drums", "bass", "melody", "vocals"], "hook repeat"),
            SongSection("bridge", 8, 0.4, 0.3, ["melody", "vocals", "fx"], "contrast"),
            SongSection("hook", 8, 0.8, 0.7, ["drums", "bass", "melody", "vocals"], "final hook"),
            SongSection("outro", 4, 0.3, 0.2, ["melody", "fx"], "fade out"),
        ],
        76,
        "Modern trap/hip-hop structure"
    ),
}


# ============================================================================
# DYNAMIC LAYERS
# ============================================================================

@dataclass
class DynamicLayer:
    """A layer of instrumentation at a specific energy level."""
    energy_level: float  # 0-1
    instruments: list[str]
    techniques: list[str]
    description: str


def get_dynamic_layers(genre: str) -> list[DynamicLayer]:
    """Get dynamic layering guide for a genre."""
    layers = {
        "pop": [
            DynamicLayer(0.2, ["pad", "bass"], ["pedal_point"], "minimal, atmospheric"),
            DynamicLayer(0.4, ["drums", "bass", "keys"], ["arpeggiation"], "groove established"),
            DynamicLayer(0.6, ["drums", "bass", "keys", "guitar"], ["layering"], "full verse"),
            DynamicLayer(0.8, ["all"], ["octave_doubling", "layering"], "full chorus"),
            DynamicLayer(1.0, ["all", "extra_percussion"], ["tutti"], "maximum impact"),
        ],
        "electronic": [
            DynamicLayer(0.2, ["pad", "fx"], ["pedal_point"], "atmosphere only"),
            DynamicLayer(0.4, ["drums_filtered", "bass", "pad"], ["ostinato"], "filtered groove"),
            DynamicLayer(0.6, ["drums", "bass", "synth"], ["arpeggiation"], "building"),
            DynamicLayer(0.8, ["drums", "bass", "synth", "lead"], ["layering"], "pre-drop"),
            DynamicLayer(1.0, ["all", "fx", "extra_bass"], ["frequency_stacking"], "full drop"),
        ],
        "orchestral": [
            DynamicLayer(0.2, ["solo_strings"], ["open_voicing"], "solo, intimate"),
            DynamicLayer(0.4, ["strings"], ["close_voicing"], "string section"),
            DynamicLayer(0.6, ["strings", "woodwinds"], ["layering"], "adding color"),
            DynamicLayer(0.8, ["strings", "woodwinds", "brass"], ["spread_voicing"], "building power"),
            DynamicLayer(1.0, ["full_orchestra", "percussion"], ["tutti"], "full tutti"),
        ],
        "jazz": [
            DynamicLayer(0.2, ["piano", "bass"], ["stride"], "duo, intimate"),
            DynamicLayer(0.4, ["piano", "bass", "drums_brushes"], ["close_voicing"], "trio, subtle"),
            DynamicLayer(0.6, ["piano", "bass", "drums", "horn"], ["drop2_voicing"], "quartet"),
            DynamicLayer(0.8, ["full_rhythm", "horns"], ["soli"], "section trading"),
            DynamicLayer(1.0, ["big_band"], ["tutti"], "full ensemble"),
        ],
    }

    return layers.get(genre, layers["pop"])


# ============================================================================
# INSTRUMENT COMBINATION RULES
# ============================================================================

@dataclass
class CombinationRule:
    """Rule for combining instruments."""
    name: str
    instruments: list[str]
    effect: str
    quality: float  # 0-1, how well they combine
    genre_affinity: dict[str, float]
    notes: str


COMBINATION_RULES: list[CombinationRule] = [
    # Classic doublings
    CombinationRule(
        "Flute + Violin Unison",
        ["flute", "violin"],
        "Bright, ethereal melody",
        0.95,
        {"orchestral": 1.0, "classical": 0.9, "cinematic": 0.8},
        "Classic orchestral doubling, sweet in high register"
    ),
    CombinationRule(
        "Oboe + Clarinet Thirds",
        ["oboe", "clarinet_bb"],
        "Warm, blended color",
        0.9,
        {"orchestral": 0.9, "classical": 0.9},
        "Excellent blend in mid-range"
    ),
    CombinationRule(
        "French Horn + Cello",
        ["french_horn", "cello"],
        "Noble, warm, full",
        0.95,
        {"orchestral": 1.0, "cinematic": 0.9},
        "Beautiful natural blend"
    ),
    CombinationRule(
        "Trumpet + Trombone Power",
        ["trumpet_bb", "trombone"],
        "Powerful brass",
        0.9,
        {"orchestral": 0.9, "jazz": 0.9, "big_band": 1.0},
        "Bright and punchy section"
    ),

    # Modern combinations
    CombinationRule(
        "Synth Bass + 808",
        ["synth_bass", "bass_808"],
        "Massive low end",
        0.85,
        {"trap": 1.0, "hip-hop": 0.9, "electronic": 0.8},
        "Layer for weight, watch phase issues"
    ),
    CombinationRule(
        "Piano + Strings Pad",
        ["piano", "strings"],
        "Lush, emotional",
        0.95,
        {"pop": 1.0, "r-and-b": 0.9, "neo-soul": 0.9},
        "Classic ballad combination"
    ),
    CombinationRule(
        "Electric Guitar + Organ",
        ["electric_guitar", "organ"],
        "Classic rock fullness",
        0.9,
        {"rock": 1.0, "blues": 0.9},
        "Fill mid-range, watch for muddiness"
    ),

    # Avoid combinations
    CombinationRule(
        "Piccolo + Soprano Voice",
        ["piccolo", "soprano"],
        "High frequency clash",
        0.3,
        {"orchestral": 0.3},
        "Both compete for same register, use carefully"
    ),
    CombinationRule(
        "Tuba + Double Bass Unison",
        ["tuba", "double_bass"],
        "Muddy low end",
        0.5,
        {"orchestral": 0.5},
        "Can work in octaves, unison is muddy"
    ),
]


def get_good_combinations_for_instrument(instrument: str) -> list[CombinationRule]:
    """Get combinations that work well with a specific instrument."""
    return [
        rule for rule in COMBINATION_RULES
        if instrument in rule.instruments and rule.quality > 0.7
    ]


def get_combinations_for_genre(genre: str) -> list[CombinationRule]:
    """Get recommended combinations for a genre."""
    return sorted(
        [rule for rule in COMBINATION_RULES if rule.genre_affinity.get(genre, 0) > 0.7],
        key=lambda r: r.genre_affinity.get(genre, 0),
        reverse=True
    )


# ============================================================================
# FREQUENCY DISTRIBUTION GUIDE
# ============================================================================

@dataclass
class FrequencyBand:
    """A frequency band for arrangement."""
    name: str
    low_hz: float
    high_hz: float
    midi_range: tuple[int, int]
    role: str
    typical_instruments: list[str]


FREQUENCY_BANDS: list[FrequencyBand] = [
    FrequencyBand(
        "sub_bass",
        20, 60,
        (24, 36),
        "Physical impact, felt more than heard",
        ["bass_808", "synth_sub", "contrabassoon"]
    ),
    FrequencyBand(
        "bass",
        60, 250,
        (36, 55),
        "Harmonic foundation, groove",
        ["bass_guitar", "double_bass", "tuba", "bass_synth"]
    ),
    FrequencyBand(
        "low_mid",
        250, 500,
        (55, 67),
        "Body, warmth, can get muddy",
        ["cello", "baritone", "guitar_low", "trombone"]
    ),
    FrequencyBand(
        "mid",
        500, 2000,
        (67, 84),
        "Presence, intelligibility, most melodic content",
        ["vocals", "piano", "guitar", "violin", "trumpet"]
    ),
    FrequencyBand(
        "high_mid",
        2000, 6000,
        (84, 96),
        "Clarity, definition, can be harsh",
        ["flute", "violin_high", "cymbals", "hi_hat"]
    ),
    FrequencyBand(
        "high",
        6000, 20000,
        (96, 108),
        "Air, brilliance, sparkle",
        ["piccolo", "cymbals_shimmer", "synth_air"]
    ),
]


def get_frequency_band_for_midi(midi_note: int) -> FrequencyBand:
    """Get the frequency band for a MIDI note."""
    for band in FREQUENCY_BANDS:
        if band.midi_range[0] <= midi_note <= band.midi_range[1]:
            return band
    return FREQUENCY_BANDS[2]  # Default to mid


def check_arrangement_balance(
    instruments_with_ranges: list[tuple[str, int, int]]
) -> dict[str, float]:
    """
    Check if an arrangement is well-balanced across frequency spectrum.

    Returns: dict with band names and coverage percentage
    """
    band_coverage = {band.name: 0.0 for band in FREQUENCY_BANDS}

    for _, low, high in instruments_with_ranges:
        for band in FREQUENCY_BANDS:
            # Check overlap
            overlap_low = max(low, band.midi_range[0])
            overlap_high = min(high, band.midi_range[1])
            if overlap_low < overlap_high:
                overlap_amount = (overlap_high - overlap_low) / (band.midi_range[1] - band.midi_range[0])
                band_coverage[band.name] = max(band_coverage[band.name], overlap_amount)

    return band_coverage


# ============================================================================
# PRODUCTION WISDOM
# ============================================================================

@dataclass
class ProductionTip:
    """Production wisdom and tips."""
    category: str
    tip: str
    genre_relevance: list[str]
    importance: float  # 0-1


PRODUCTION_WISDOM: list[ProductionTip] = [
    # Arrangement
    ProductionTip(
        "arrangement",
        "Less is more: leave space for each element to breathe",
        ["all"],
        0.9
    ),
    ProductionTip(
        "arrangement",
        "Each instrument should have its own frequency/rhythmic space",
        ["pop", "electronic", "hip-hop"],
        0.85
    ),
    ProductionTip(
        "arrangement",
        "Build arrangements in layers: start minimal, add elements gradually",
        ["electronic", "ambient", "cinematic"],
        0.8
    ),
    ProductionTip(
        "arrangement",
        "Strip out competing elements during vocals",
        ["pop", "r-and-b", "hip-hop"],
        0.85
    ),

    # Dynamics
    ProductionTip(
        "dynamics",
        "Save your biggest moment for the final chorus or drop",
        ["pop", "electronic", "rock"],
        0.9
    ),
    ProductionTip(
        "dynamics",
        "Contrast is key: quiet sections make loud sections feel bigger",
        ["all"],
        0.95
    ),
    ProductionTip(
        "dynamics",
        "Use automation to create movement and interest",
        ["electronic", "pop"],
        0.8
    ),

    # Bass
    ProductionTip(
        "bass",
        "Keep bass mono below 100Hz for club playback",
        ["electronic", "trap", "house"],
        0.85
    ),
    ProductionTip(
        "bass",
        "Don't double the bass with another bass instrument in the same octave",
        ["all"],
        0.9
    ),

    # Melody
    ProductionTip(
        "melody",
        "The hook should be instantly memorable - if you can't hum it, simplify it",
        ["pop", "rock", "electronic"],
        0.9
    ),
    ProductionTip(
        "melody",
        "Use repetition with variation to build familiarity",
        ["all"],
        0.85
    ),

    # Rhythm
    ProductionTip(
        "rhythm",
        "Lock the kick and bass together rhythmically",
        ["all"],
        0.9
    ),
    ProductionTip(
        "rhythm",
        "Subtle timing variations (humanization) prevent mechanical feel",
        ["jazz", "funk", "r-and-b", "neo-soul"],
        0.85
    ),
]


def get_wisdom_for_genre(genre: str) -> list[ProductionTip]:
    """Get relevant production wisdom for a genre."""
    return [
        tip for tip in PRODUCTION_WISDOM
        if genre in tip.genre_relevance or "all" in tip.genre_relevance
    ]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_arrangement_for_genre(genre: str) -> Optional[GenreArrangement]:
    """Get the arrangement definition for a genre."""
    return GENRE_ARRANGEMENTS.get(genre)


def get_structure_for_genre(genre: str) -> Optional[SongStructure]:
    """Get a song structure template for a genre."""
    genre_to_structure = {
        "pop": "verse_chorus_pop",
        "rock": "verse_chorus_pop",
        "jazz": "aaba_jazz",
        "electronic": "edm_drop",
        "house": "edm_drop",
        "techno": "edm_drop",
        "blues": "blues_12bar",
        "trap": "trap_beat",
        "hip-hop": "trap_beat",
    }

    structure_name = genre_to_structure.get(genre, "verse_chorus_pop")
    return SONG_STRUCTURES.get(structure_name)


def get_instrument_range(instrument: str) -> Optional[InstrumentRange]:
    """Get the range for an instrument."""
    return ORCHESTRAL_RANGES.get(instrument)


def suggest_instruments_for_role(
    role: str,
    genre: str,
    existing_instruments: list[str]
) -> list[str]:
    """
    Suggest instruments for a specific role in an arrangement.

    Roles: melody, harmony, bass, rhythm, color
    """
    suggestions = {
        "melody": {
            "pop": ["vocals", "synth_lead", "guitar"],
            "jazz": ["saxophone", "trumpet", "piano"],
            "orchestral": ["violin", "flute", "oboe"],
            "electronic": ["synth_lead", "vocals", "pluck"],
            "default": ["piano", "guitar", "synth"],
        },
        "harmony": {
            "pop": ["piano", "guitar", "strings"],
            "jazz": ["piano", "guitar", "vibraphone"],
            "orchestral": ["strings", "woodwinds", "brass"],
            "electronic": ["synth_pad", "strings", "piano"],
            "default": ["piano", "strings", "synth_pad"],
        },
        "bass": {
            "pop": ["bass_guitar", "synth_bass"],
            "jazz": ["double_bass", "bass_guitar"],
            "orchestral": ["double_bass", "cello", "tuba"],
            "electronic": ["synth_bass", "bass_808"],
            "trap": ["bass_808", "synth_sub"],
            "default": ["bass_guitar", "synth_bass"],
        },
        "rhythm": {
            "pop": ["drums", "percussion"],
            "jazz": ["drums_jazz", "percussion"],
            "orchestral": ["timpani", "percussion"],
            "electronic": ["drums_electronic", "hi_hats"],
            "default": ["drums", "percussion"],
        },
        "color": {
            "pop": ["strings", "synth_pad", "bells"],
            "jazz": ["vibraphone", "piano_fills", "percussion"],
            "orchestral": ["harp", "celesta", "glockenspiel"],
            "electronic": ["arpeggios", "fx", "vocoder"],
            "default": ["strings", "bells", "fx"],
        },
    }

    role_suggestions = suggestions.get(role, suggestions.get("color", {}))
    genre_instruments = role_suggestions.get(genre, role_suggestions.get("default", []))

    # Filter out already used instruments
    return [inst for inst in genre_instruments if inst not in existing_instruments]
