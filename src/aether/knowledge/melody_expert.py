"""
AETHER Melody Expert Knowledge System

World-class melody generation expertise with orchestral-level experience,
producer mindset, and innovation-focused techniques. This module provides
the intelligence for creating memorable, genre-appropriate melodies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from aether.knowledge.theory import (
    SCALE_PATTERNS,
    CHORD_PATTERNS,
    get_scale,
    get_chord_notes,
    note_to_midi,
    midi_to_note,
    transpose_note,
    get_interval_sequence,
    get_contour,
    calculate_singability,
    sequence_melody,
    invert_melody,
    create_motif_variation,
    embellish_melody,
)


# ============================================================================
# MELODIC ARCHETYPES - Classic Melody Shapes
# ============================================================================

class MelodicArchetype(Enum):
    """Classic melodic archetypes used across genres."""
    ARCH = "arch"  # Up then down
    INVERTED_ARCH = "inverted_arch"  # Down then up
    ASCENDING = "ascending"  # Generally upward
    DESCENDING = "descending"  # Generally downward
    WAVE = "wave"  # Oscillating
    TERRACED = "terraced"  # Step-wise with plateaus
    PENDULUM = "pendulum"  # Swinging around center
    ROCKET = "rocket"  # Quick ascent, slow descent
    CASCADE = "cascade"  # Stepwise descent with jumps up


@dataclass
class ArchetypeDefinition:
    """Definition of a melodic archetype."""
    archetype: MelodicArchetype
    description: str
    contour_pattern: list[str]  # up/down/same sequence
    emotional_quality: str
    genre_affinity: dict[str, float]
    typical_range_semitones: int
    phrase_length_beats: int = 8


ARCHETYPE_DEFINITIONS: dict[MelodicArchetype, ArchetypeDefinition] = {
    MelodicArchetype.ARCH: ArchetypeDefinition(
        MelodicArchetype.ARCH,
        "Rises to a peak then falls - classic satisfying contour",
        ["up", "up", "up", "same", "down", "down", "down"],
        "hopeful, reaching, resolved",
        {"pop": 1.0, "classical": 0.9, "r-and-b": 0.8, "rock": 0.7},
        10
    ),
    MelodicArchetype.INVERTED_ARCH: ArchetypeDefinition(
        MelodicArchetype.INVERTED_ARCH,
        "Falls to a low point then rises - tension to release",
        ["down", "down", "down", "same", "up", "up", "up"],
        "introspective, building, triumphant",
        {"classical": 0.9, "cinematic": 1.0, "rock": 0.7},
        10
    ),
    MelodicArchetype.ASCENDING: ArchetypeDefinition(
        MelodicArchetype.ASCENDING,
        "Generally upward motion - building energy",
        ["up", "up", "same", "up", "up", "same", "up"],
        "hopeful, energizing, aspirational",
        {"electronic": 0.9, "pop": 0.8, "gospel": 0.9},
        12
    ),
    MelodicArchetype.DESCENDING: ArchetypeDefinition(
        MelodicArchetype.DESCENDING,
        "Generally downward motion - resolution, melancholy",
        ["down", "down", "same", "down", "down", "same", "down"],
        "peaceful, melancholic, resolving",
        {"jazz": 0.9, "blues": 0.9, "ambient": 0.8},
        10
    ),
    MelodicArchetype.WAVE: ArchetypeDefinition(
        MelodicArchetype.WAVE,
        "Oscillating up and down - fluid, continuous",
        ["up", "down", "up", "down", "up", "down", "same"],
        "flowing, natural, conversational",
        {"pop": 0.9, "r-and-b": 0.9, "neo-soul": 0.9},
        8
    ),
    MelodicArchetype.TERRACED: ArchetypeDefinition(
        MelodicArchetype.TERRACED,
        "Step-wise with plateaus - dramatic, baroque",
        ["up", "same", "same", "up", "same", "same", "down"],
        "dramatic, ceremonial, structured",
        {"classical": 1.0, "baroque": 1.0, "cinematic": 0.8},
        12
    ),
    MelodicArchetype.PENDULUM: ArchetypeDefinition(
        MelodicArchetype.PENDULUM,
        "Swinging around a central note - hypnotic",
        ["up", "down", "down", "up", "up", "down", "same"],
        "hypnotic, meditative, cyclical",
        {"ambient": 0.9, "electronic": 0.8, "minimalist": 1.0},
        6
    ),
    MelodicArchetype.ROCKET: ArchetypeDefinition(
        MelodicArchetype.ROCKET,
        "Quick ascent, slow descent - dramatic climax",
        ["up", "up", "up", "up", "down", "down", "same"],
        "explosive, dramatic, powerful",
        {"rock": 0.9, "metal": 0.9, "electronic": 0.8},
        14
    ),
    MelodicArchetype.CASCADE: ArchetypeDefinition(
        MelodicArchetype.CASCADE,
        "Stepwise descent with occasional jumps up",
        ["down", "down", "up", "down", "down", "up", "down"],
        "flowing, waterfall-like, graceful",
        {"classical": 0.9, "jazz": 0.8, "ambient": 0.8},
        12
    ),
}


def get_archetype_for_genre(genre: str) -> MelodicArchetype:
    """Get the best melodic archetype for a genre."""
    best_archetype = MelodicArchetype.ARCH
    best_affinity = 0.0

    for archetype, definition in ARCHETYPE_DEFINITIONS.items():
        affinity = definition.genre_affinity.get(genre, 0.0)
        if affinity > best_affinity:
            best_affinity = affinity
            best_archetype = archetype

    return best_archetype


# ============================================================================
# HOOK FORMULAS - Memorable Melody Patterns
# ============================================================================

@dataclass
class HookFormula:
    """A formula for creating memorable hooks."""
    name: str
    description: str
    interval_pattern: list[int]  # Relative intervals
    rhythm_pattern: list[float]  # Relative durations
    repetition_type: str  # exact, varied, sequential
    effectiveness: float  # 0-1
    genre_affinity: dict[str, float]


HOOK_FORMULAS: list[HookFormula] = [
    HookFormula(
        "Repeating Third",
        "Same note repeated then up/down a third",
        [0, 0, 0, 3, 0, 0, 0, -3],
        [1, 1, 1, 1, 1, 1, 1, 1],
        "exact",
        0.95,
        {"pop": 1.0, "rock": 0.9, "r-and-b": 0.8}
    ),
    HookFormula(
        "Call and Response",
        "Two-part phrase: question (ends high) and answer (ends low)",
        [0, 2, 4, 5, 4, 2, 0, -2],
        [1, 1, 1, 2, 1, 1, 1, 2],
        "varied",
        0.9,
        {"pop": 0.9, "gospel": 1.0, "funk": 0.9}
    ),
    HookFormula(
        "Pentatonic Jump",
        "Pentatonic scale with signature leap",
        [0, 2, 5, 7, 5, 2, 0, 0],
        [0.5, 0.5, 1, 2, 1, 0.5, 0.5, 2],
        "sequential",
        0.85,
        {"pop": 0.9, "rock": 0.9, "blues": 0.8}
    ),
    HookFormula(
        "Stepwise Descent",
        "Memorable descending line with final resolution",
        [0, -1, -2, -3, -4, -5, -4, -5],
        [1, 1, 1, 1, 1, 1, 0.5, 1.5],
        "exact",
        0.88,
        {"pop": 0.9, "jazz": 0.8, "classical": 0.9}
    ),
    HookFormula(
        "Rhythmic Repetition",
        "Same pitch, interesting rhythm",
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0.25, 0.25, 1, 0.5, 0.25, 0.25, 1],
        "exact",
        0.85,
        {"hip-hop": 1.0, "trap": 0.9, "funk": 0.8}
    ),
    HookFormula(
        "Octave Frame",
        "Opens with octave leap, fills in between",
        [0, 12, 11, 9, 7, 5, 4, 0],
        [2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2],
        "varied",
        0.82,
        {"pop": 0.8, "r-and-b": 0.9, "neo-soul": 0.9}
    ),
    HookFormula(
        "Syncopated Anticipation",
        "Notes arrive just before the beat",
        [0, 2, 4, 5, 4, 2, 0, 0],
        [0.75, 0.75, 0.5, 1, 0.75, 0.75, 0.5, 1],
        "exact",
        0.87,
        {"funk": 1.0, "r-and-b": 0.9, "neo-soul": 0.9}
    ),
    HookFormula(
        "Rising Question",
        "Ascending line ending on leading tone",
        [0, 2, 4, 5, 7, 9, 11, 11],
        [1, 1, 1, 1, 1, 1, 1, 2],
        "sequential",
        0.8,
        {"pop": 0.9, "rock": 0.8, "classical": 0.7}
    ),
]


def get_hook_formula_for_genre(genre: str) -> HookFormula:
    """Get the best hook formula for a genre."""
    best_formula = HOOK_FORMULAS[0]
    best_affinity = 0.0

    for formula in HOOK_FORMULAS:
        affinity = formula.genre_affinity.get(genre, 0.0)
        if affinity > best_affinity:
            best_affinity = affinity
            best_formula = formula

    return best_formula


# ============================================================================
# PHRASE CONSTRUCTION - Musical Sentence Building
# ============================================================================

@dataclass
class PhraseStructure:
    """Structure for building musical phrases."""
    name: str
    bar_structure: list[str]  # antecedent, consequent, extension, etc.
    total_bars: int
    cadence_points: list[int]  # Bar numbers where cadences occur
    melodic_rhythm: str  # dense, sparse, varied
    description: str


PHRASE_STRUCTURES: dict[str, PhraseStructure] = {
    "period_8bar": PhraseStructure(
        "Period (8-bar)",
        ["antecedent", "antecedent", "antecedent", "half_cadence",
         "consequent", "consequent", "consequent", "full_cadence"],
        8,
        [4, 8],
        "varied",
        "Classic sentence structure: question and answer"
    ),
    "sentence_8bar": PhraseStructure(
        "Sentence (8-bar)",
        ["idea", "idea_repeat", "fragmentation", "fragmentation",
         "continuation", "continuation", "cadential", "cadential"],
        8,
        [8],
        "accelerating",
        "Beethoven-style: idea, repeat, development, cadence"
    ),
    "verse_16bar": PhraseStructure(
        "Verse (16-bar)",
        ["intro_phrase", "intro_phrase", "main_idea", "main_idea",
         "main_idea_var", "main_idea_var", "pre_cadence", "half_cadence",
         "response", "response", "response_var", "response_var",
         "development", "development", "cadential", "full_cadence"],
        16,
        [8, 16],
        "varied",
        "Extended verse structure with clear sections"
    ),
    "aaba_32bar": PhraseStructure(
        "AABA (32-bar)",
        ["A1", "A1", "A1", "A1_cadence", "A2", "A2", "A2", "A2_cadence",
         "B1", "B1", "B1", "B1_contrast", "B2", "B2", "B2", "B2_return",
         "A3", "A3", "A3", "A3_cadence", "A4", "A4", "A4", "A4_final",
         "tag", "tag", "tag", "tag", "outro", "outro", "outro", "end"],
        32,
        [8, 16, 24, 32],
        "varied",
        "Classic jazz standard form"
    ),
    "hook_4bar": PhraseStructure(
        "Hook (4-bar)",
        ["hook_statement", "hook_statement", "variation", "resolution"],
        4,
        [4],
        "dense",
        "Short, memorable hook phrase"
    ),
    "edm_buildup": PhraseStructure(
        "EDM Buildup (8-bar)",
        ["intro", "intro", "rising", "rising",
         "tension", "tension", "peak_prep", "drop_lead"],
        8,
        [8],
        "accelerating",
        "Building tension toward a drop"
    ),
}


# ============================================================================
# MELODIC DEVELOPMENT STRATEGIES
# ============================================================================

class DevelopmentStrategy(Enum):
    """Strategies for developing melodic material."""
    REPETITION = "repetition"  # Exact repeat
    SEQUENCE = "sequence"  # Same pattern, different pitch
    FRAGMENTATION = "fragmentation"  # Using smaller parts
    EXTENSION = "extension"  # Making it longer
    TRUNCATION = "truncation"  # Making it shorter
    INVERSION = "inversion"  # Flipping intervals
    RETROGRADE = "retrograde"  # Playing backwards
    AUGMENTATION = "augmentation"  # Longer note values
    DIMINUTION = "diminution"  # Shorter note values
    ORNAMENTATION = "ornamentation"  # Adding embellishments
    VARIATION = "variation"  # Changing some elements
    INTERPOLATION = "interpolation"  # Adding notes between
    TRANSPOSITION = "transposition"  # Different key


@dataclass
class DevelopmentTechnique:
    """A technique for developing melodic material."""
    strategy: DevelopmentStrategy
    description: str
    complexity: float  # 0-1
    recognition_preserved: float  # 0-1, how recognizable is the result
    genre_affinity: dict[str, float]


DEVELOPMENT_TECHNIQUES: list[DevelopmentTechnique] = [
    DevelopmentTechnique(
        DevelopmentStrategy.REPETITION,
        "Exact repetition for reinforcement",
        0.1, 1.0,
        {"pop": 1.0, "electronic": 0.9, "hip-hop": 0.9}
    ),
    DevelopmentTechnique(
        DevelopmentStrategy.SEQUENCE,
        "Same pattern at different pitch levels",
        0.3, 0.9,
        {"classical": 1.0, "pop": 0.8, "jazz": 0.7}
    ),
    DevelopmentTechnique(
        DevelopmentStrategy.FRAGMENTATION,
        "Using only part of the original",
        0.4, 0.7,
        {"classical": 0.9, "jazz": 0.8, "electronic": 0.7}
    ),
    DevelopmentTechnique(
        DevelopmentStrategy.INVERSION,
        "Flipping the melody upside down",
        0.6, 0.6,
        {"classical": 1.0, "jazz": 0.7}
    ),
    DevelopmentTechnique(
        DevelopmentStrategy.ORNAMENTATION,
        "Adding passing tones, trills, turns",
        0.5, 0.85,
        {"classical": 0.9, "jazz": 1.0, "neo-soul": 0.8}
    ),
    DevelopmentTechnique(
        DevelopmentStrategy.VARIATION,
        "Changing rhythm or some pitches",
        0.4, 0.8,
        {"jazz": 1.0, "r-and-b": 0.9, "pop": 0.8}
    ),
    DevelopmentTechnique(
        DevelopmentStrategy.AUGMENTATION,
        "Stretching note durations",
        0.3, 0.85,
        {"classical": 0.9, "ambient": 0.9, "cinematic": 0.8}
    ),
    DevelopmentTechnique(
        DevelopmentStrategy.TRANSPOSITION,
        "Same melody in different key",
        0.2, 0.95,
        {"pop": 0.9, "classical": 0.8, "rock": 0.8}
    ),
]


def get_development_for_genre(genre: str) -> list[DevelopmentTechnique]:
    """Get appropriate development techniques for a genre."""
    return sorted(
        [t for t in DEVELOPMENT_TECHNIQUES if t.genre_affinity.get(genre, 0) > 0.6],
        key=lambda t: t.genre_affinity.get(genre, 0),
        reverse=True
    )


# ============================================================================
# PRODUCER MINDSET - Industry Wisdom
# ============================================================================

@dataclass
class ProducerInsight:
    """Production wisdom for melody creation."""
    category: str
    insight: str
    application: str
    importance: float  # 0-1


PRODUCER_INSIGHTS: list[ProducerInsight] = [
    # Memorability
    ProducerInsight(
        "memorability",
        "If you can't hum it after one listen, simplify it",
        "Test melodies by humming them away from the instrument",
        0.95
    ),
    ProducerInsight(
        "memorability",
        "Repetition legitimizes - repeat the hook at least 3 times",
        "Ensure main hook appears multiple times in the song",
        0.9
    ),
    ProducerInsight(
        "memorability",
        "The best melodies use only 5-7 unique pitches",
        "Limit pitch vocabulary for stronger hooks",
        0.85
    ),

    # Range and Singability
    ProducerInsight(
        "singability",
        "Keep vocal melodies within an octave for most phrases",
        "Check range of each phrase; larger ranges are for climax only",
        0.9
    ),
    ProducerInsight(
        "singability",
        "Stepwise motion is easier to sing and remember",
        "Use leaps sparingly and balance with steps",
        0.85
    ),
    ProducerInsight(
        "singability",
        "End phrases on stable notes (1, 3, 5) for resolution",
        "Choose landing notes that feel finished",
        0.8
    ),

    # Rhythm
    ProducerInsight(
        "rhythm",
        "Rhythmic interest can make a simple melody memorable",
        "Try syncopation, unexpected rests, varied durations",
        0.9
    ),
    ProducerInsight(
        "rhythm",
        "Leave space for the listener to breathe",
        "Include rests; constant motion fatigues the ear",
        0.85
    ),
    ProducerInsight(
        "rhythm",
        "The strongest notes land on the strongest beats",
        "Place important melodic notes on beat 1 and 3",
        0.8
    ),

    # Contrast
    ProducerInsight(
        "contrast",
        "Verse melodies should contrast with chorus",
        "If verse is low/sparse, chorus should be high/dense",
        0.9
    ),
    ProducerInsight(
        "contrast",
        "Save your highest note for the most emotional moment",
        "The peak note should appear only once or twice",
        0.85
    ),
    ProducerInsight(
        "contrast",
        "Follow tension with release",
        "Dissonant notes should resolve to consonant ones",
        0.8
    ),

    # Genre
    ProducerInsight(
        "genre",
        "Study the top 10 hits in your genre - find the patterns",
        "Analyze successful melodies for common intervals and rhythms",
        0.85
    ),
    ProducerInsight(
        "genre",
        "Genre expectations exist for a reason - meet them first, then innovate",
        "Establish genre conventions before subverting them",
        0.8
    ),

    # Innovation
    ProducerInsight(
        "innovation",
        "Innovation comes from constraint - limit yourself",
        "Try writing with only 4 notes, or no leaps larger than a 3rd",
        0.75
    ),
    ProducerInsight(
        "innovation",
        "Borrow patterns from other genres",
        "Try jazz intervals in pop, or classical phrasing in electronic",
        0.7
    ),
]


def get_insights_for_category(category: str) -> list[ProducerInsight]:
    """Get producer insights for a specific category."""
    return [i for i in PRODUCER_INSIGHTS if i.category == category]


# ============================================================================
# MELODY GENERATION INTELLIGENCE
# ============================================================================

@dataclass
class MelodyParameters:
    """Parameters for intelligent melody generation."""
    root_note: str
    scale_type: str
    range_low_midi: int
    range_high_midi: int
    archetype: MelodicArchetype
    phrase_bars: int
    note_density: float  # 0-1, notes per beat on average
    leap_probability: float  # 0-1
    syncopation_amount: float  # 0-1
    repetition_level: float  # 0-1, how much to repeat
    genre: str


@dataclass
class GeneratedMelody:
    """A generated melody with metadata."""
    midi_notes: list[int]
    durations: list[float]
    velocities: list[int]
    phrase_structure: str
    archetype_used: MelodicArchetype
    hook_formula_used: Optional[str]
    development_techniques: list[str]
    singability_score: float
    memorability_score: float


class MelodyExpert:
    """
    World-class melody expert with orchestral experience and producer mindset.

    This class embodies:
    - Deep music theory knowledge
    - Industry-proven melodic techniques
    - Genre-specific expertise
    - Innovation-focused approach
    """

    def __init__(self, genre: str = "pop"):
        self.genre = genre
        self.preferred_archetype = get_archetype_for_genre(genre)
        self.preferred_hook = get_hook_formula_for_genre(genre)
        self.development_techniques = get_development_for_genre(genre)

    def analyze_melody(self, midi_notes: list[int]) -> dict:
        """Comprehensive melody analysis."""
        if not midi_notes:
            return {"error": "No notes to analyze"}

        intervals = get_interval_sequence(midi_notes)
        contour = get_contour(midi_notes)
        singability = calculate_singability(midi_notes)

        # Determine archetype
        archetype = self._identify_archetype(contour)

        # Calculate memorability
        memorability = self._calculate_memorability(midi_notes, intervals)

        # Identify patterns
        patterns = self._identify_patterns(midi_notes)

        return {
            "note_count": len(midi_notes),
            "range": max(midi_notes) - min(midi_notes),
            "intervals": intervals,
            "contour": contour,
            "archetype": archetype.value if archetype else "unknown",
            "singability": singability,
            "memorability": memorability,
            "patterns_found": patterns,
            "suggestions": self._generate_suggestions(midi_notes, singability, memorability),
        }

    def generate_hook(
        self,
        root: str = "C",
        scale_type: str = "major",
        octave: int = 4,
    ) -> GeneratedMelody:
        """Generate a memorable hook based on genre-appropriate formulas."""
        formula = self.preferred_hook
        root_midi = note_to_midi(root, octave)

        # Get scale notes
        scale_notes = get_scale(root, scale_type)
        scale_midi = [note_to_midi(n, octave) for n in scale_notes]

        # Apply formula intervals
        midi_notes = []
        for interval in formula.interval_pattern:
            # Snap to scale
            target = root_midi + interval
            closest = min(scale_midi, key=lambda x: abs(x - target))
            midi_notes.append(closest)

        # Generate velocities (accent pattern)
        velocities = [90 if i % 2 == 0 else 75 for i in range(len(midi_notes))]

        return GeneratedMelody(
            midi_notes=midi_notes,
            durations=formula.rhythm_pattern,
            velocities=velocities,
            phrase_structure="hook_4bar",
            archetype_used=MelodicArchetype.WAVE,
            hook_formula_used=formula.name,
            development_techniques=[],
            singability_score=calculate_singability(midi_notes),
            memorability_score=0.85,
        )

    def generate_melody(
        self,
        params: MelodyParameters,
        seed_notes: Optional[list[int]] = None,
    ) -> GeneratedMelody:
        """Generate a complete melody with expert intelligence."""
        archetype_def = ARCHETYPE_DEFINITIONS[params.archetype]

        # Get scale
        scale_notes = get_scale(params.root_note, params.scale_type)
        root_midi = note_to_midi(params.root_note, 4)
        scale_midi_octave = [note_to_midi(n, 4) for n in scale_notes]

        # Build scale across full range
        scale_midi = []
        for octave in range(2, 7):
            for note in scale_notes:
                midi = note_to_midi(note, octave)
                if params.range_low_midi <= midi <= params.range_high_midi:
                    scale_midi.append(midi)
        scale_midi = sorted(set(scale_midi))

        if not scale_midi:
            scale_midi = list(range(params.range_low_midi, params.range_high_midi + 1))

        # Generate based on archetype contour
        contour = archetype_def.contour_pattern
        notes_per_direction = max(1, params.phrase_bars * int(params.note_density * 4) // len(contour))

        midi_notes = []
        current_note = scale_midi[len(scale_midi) // 2]  # Start in middle
        midi_notes.append(current_note)

        for direction in contour:
            for _ in range(notes_per_direction):
                if direction == "up":
                    candidates = [n for n in scale_midi if n > current_note]
                    if candidates:
                        if random.random() < params.leap_probability:
                            current_note = random.choice(candidates[:min(3, len(candidates))])
                        else:
                            current_note = candidates[0]
                elif direction == "down":
                    candidates = [n for n in scale_midi if n < current_note]
                    if candidates:
                        if random.random() < params.leap_probability:
                            current_note = random.choice(candidates[-min(3, len(candidates)):])
                        else:
                            current_note = candidates[-1]
                # "same" keeps current note

                midi_notes.append(current_note)

        # Apply repetition for memorability
        if params.repetition_level > 0.5:
            # Repeat first phrase
            phrase_len = len(midi_notes) // 2
            midi_notes = midi_notes[:phrase_len] + midi_notes[:phrase_len] + midi_notes[phrase_len:]

        # Generate rhythms
        base_duration = 1.0  # Quarter note
        durations = []
        for i in range(len(midi_notes)):
            if random.random() < params.syncopation_amount:
                durations.append(base_duration * 0.75)
            else:
                durations.append(base_duration)

        # Generate velocities with accent pattern
        velocities = []
        for i in range(len(midi_notes)):
            if i % 4 == 0:  # Downbeat accent
                velocities.append(95)
            elif i % 2 == 0:
                velocities.append(85)
            else:
                velocities.append(75)

        singability = calculate_singability(midi_notes)
        memorability = self._calculate_memorability(midi_notes, get_interval_sequence(midi_notes))

        return GeneratedMelody(
            midi_notes=midi_notes,
            durations=durations,
            velocities=velocities,
            phrase_structure=f"phrase_{params.phrase_bars}bar",
            archetype_used=params.archetype,
            hook_formula_used=None,
            development_techniques=[],
            singability_score=singability,
            memorability_score=memorability,
        )

    def develop_melody(
        self,
        original: list[int],
        strategy: DevelopmentStrategy,
        amount: float = 0.5,
    ) -> list[int]:
        """Apply a development strategy to evolve a melody."""
        if strategy == DevelopmentStrategy.REPETITION:
            return original + original

        elif strategy == DevelopmentStrategy.SEQUENCE:
            # Transpose by a third
            interval = 3 if amount > 0.5 else -3
            return sequence_melody(original, interval, 1)

        elif strategy == DevelopmentStrategy.FRAGMENTATION:
            # Use first half
            return original[:len(original) // 2]

        elif strategy == DevelopmentStrategy.INVERSION:
            return invert_melody(original)

        elif strategy == DevelopmentStrategy.AUGMENTATION:
            # Return notes (rhythm handled separately)
            return original

        elif strategy == DevelopmentStrategy.TRANSPOSITION:
            interval = int(amount * 12)  # Up to an octave
            return [n + interval for n in original]

        elif strategy == DevelopmentStrategy.VARIATION:
            return create_motif_variation(original, "transpose")

        return original

    def suggest_countermelody(
        self,
        main_melody: list[int],
        root: str,
        scale_type: str,
    ) -> list[int]:
        """Suggest a countermelody that complements the main melody."""
        scale_notes = get_scale(root, scale_type)

        # Basic approach: contrary motion in thirds/sixths
        counter = []
        for note in main_melody:
            note_name, octave = midi_to_note(note)
            note_idx = scale_notes.index(note_name) if note_name in scale_notes else 0

            # A third below
            counter_idx = (note_idx - 2) % len(scale_notes)
            counter_octave = octave if counter_idx < note_idx else octave - 1
            counter_midi = note_to_midi(scale_notes[counter_idx], counter_octave)
            counter.append(counter_midi)

        return counter

    def _identify_archetype(self, contour: list[str]) -> Optional[MelodicArchetype]:
        """Identify the melodic archetype from a contour."""
        if not contour:
            return None

        # Count directions
        ups = contour.count("up")
        downs = contour.count("down")
        total = len(contour)

        # Check patterns
        first_half = contour[:len(contour) // 2]
        second_half = contour[len(contour) // 2:]

        first_ups = first_half.count("up")
        second_downs = second_half.count("down")

        if first_ups > len(first_half) * 0.6 and second_downs > len(second_half) * 0.6:
            return MelodicArchetype.ARCH

        if ups > total * 0.7:
            return MelodicArchetype.ASCENDING

        if downs > total * 0.7:
            return MelodicArchetype.DESCENDING

        return MelodicArchetype.WAVE

    def _calculate_memorability(self, notes: list[int], intervals: list[int]) -> float:
        """Calculate how memorable a melody is likely to be."""
        score = 0.5  # Base

        # Repetition increases memorability
        unique_notes = len(set(notes))
        if unique_notes <= 5:
            score += 0.15
        elif unique_notes <= 7:
            score += 0.1

        # Stepwise motion is more memorable
        stepwise = sum(1 for i in intervals if abs(i) <= 2)
        stepwise_ratio = stepwise / len(intervals) if intervals else 0
        score += stepwise_ratio * 0.15

        # Moderate range is more memorable
        range_semitones = max(notes) - min(notes) if notes else 0
        if 5 <= range_semitones <= 12:
            score += 0.1
        elif range_semitones > 15:
            score -= 0.1

        # Pattern repetition
        if len(notes) >= 4:
            first_four = tuple(notes[:4])
            if notes[4:8] == list(first_four):
                score += 0.15

        return min(1.0, max(0.0, score))

    def _identify_patterns(self, notes: list[int]) -> list[dict]:
        """Identify repeated patterns in the melody."""
        patterns = []

        # Check for 2-4 note repeated patterns
        for pattern_len in range(2, 5):
            for i in range(len(notes) - pattern_len * 2):
                pattern = notes[i:i + pattern_len]
                # Check if pattern repeats
                next_segment = notes[i + pattern_len:i + pattern_len * 2]
                if pattern == next_segment:
                    patterns.append({
                        "type": "exact_repetition",
                        "length": pattern_len,
                        "position": i,
                    })
                # Check for sequence (same intervals, different pitches)
                elif len(pattern) > 1:
                    pattern_intervals = get_interval_sequence(pattern)
                    next_intervals = get_interval_sequence(next_segment) if len(next_segment) > 1 else []
                    if pattern_intervals == next_intervals:
                        patterns.append({
                            "type": "sequence",
                            "length": pattern_len,
                            "position": i,
                        })

        return patterns[:5]  # Limit to top 5

    def _generate_suggestions(
        self,
        notes: list[int],
        singability: float,
        memorability: float,
    ) -> list[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []

        if singability < 0.6:
            suggestions.append("Consider reducing large leaps for better singability")

        if memorability < 0.6:
            suggestions.append("Add more repetition or limit pitch variety for memorability")

        range_semitones = max(notes) - min(notes) if notes else 0
        if range_semitones > 15:
            suggestions.append("Range is wide - consider constraining to 12 semitones for verses")

        intervals = get_interval_sequence(notes)
        if intervals:
            avg_interval = sum(abs(i) for i in intervals) / len(intervals)
            if avg_interval > 4:
                suggestions.append("Many large intervals - add more stepwise motion")

        unique_notes = len(set(notes))
        if unique_notes > 10:
            suggestions.append(f"Using {unique_notes} unique pitches - simpler hooks use 5-7")

        return suggestions


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_melody_expert(genre: str = "pop") -> MelodyExpert:
    """Create a melody expert configured for a specific genre."""
    return MelodyExpert(genre=genre)


def quick_analyze(midi_notes: list[int], genre: str = "pop") -> dict:
    """Quick melody analysis."""
    expert = MelodyExpert(genre)
    return expert.analyze_melody(midi_notes)


def quick_hook(root: str = "C", scale: str = "major", genre: str = "pop") -> GeneratedMelody:
    """Quickly generate a hook."""
    expert = MelodyExpert(genre)
    return expert.generate_hook(root, scale)
