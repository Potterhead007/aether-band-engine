"""
AETHER Genre Differentiation System v0.1

Institutional-grade genre conditioning and enforcement to eliminate genre collapse.

Components:
- GenreDNA: 48-dimensional genre fingerprint
- GenreConditioner: Translates user selection to generation constraints
- RhythmPatternMatcher: Detects rhythm patterns for genre classification
- HarmonicAnalyzer: Analyzes harmonic content for genre classification
- GenreDetector: Ensemble genre detection from audio/MIDI
- GenreEnforcer: Iterative correction loop for genre drift

Target Genres (v0.1):
- lofi-hip-hop, trap, drill, boom-bap
- synthwave, house, techno, drum-and-bass
- reggaeton, afrobeat, pop, cinematic
"""

from aether.genre.dna import (
    GenreDNA,
    RhythmDNA,
    HarmonyDNA,
    MelodyDNA,
    TimbreDNA,
    StructureDNA,
    GENRE_DNA_LIBRARY,
    get_genre_dna,
    list_genres,
)

from aether.genre.conditioner import (
    GenreConditioner,
    GenerationConstraints,
    HardConstraints,
    SoftPreferences,
    RejectionCriteria,
)

from aether.genre.detection import (
    GenreDetector,
    GenreDetectionResult,
    RhythmPatternMatcher,
    HarmonicAnalyzer,
    RhythmFeatures,
    HarmonicFeatures,
)

from aether.genre.motif import (
    MotifGenerator,
    MotifRules,
    Motif,
    GENRE_MOTIF_RULES,
)

from aether.genre.rhythm import (
    RhythmGrammar,
    GENRE_RHYTHM_GRAMMAR,
    generate_rhythm_pattern,
)

from aether.genre.harmony import (
    HarmonicGrammar,
    GENRE_HARMONIC_GRAMMAR,
    generate_chord_progression,
)

from aether.genre.metrics import (
    GenreDifferentiationMetrics,
    MetricsReport,
    compute_embedding_separation,
    compute_rhythm_accuracy,
)

__all__ = [
    # DNA
    "GenreDNA",
    "RhythmDNA",
    "HarmonyDNA",
    "MelodyDNA",
    "TimbreDNA",
    "StructureDNA",
    "GENRE_DNA_LIBRARY",
    "get_genre_dna",
    "list_genres",
    # Conditioning
    "GenreConditioner",
    "GenerationConstraints",
    "HardConstraints",
    "SoftPreferences",
    "RejectionCriteria",
    # Detection
    "GenreDetector",
    "GenreDetectionResult",
    "RhythmPatternMatcher",
    "HarmonicAnalyzer",
    "RhythmFeatures",
    "HarmonicFeatures",
    # Motif
    "MotifGenerator",
    "MotifRules",
    "Motif",
    "GENRE_MOTIF_RULES",
    # Rhythm
    "RhythmGrammar",
    "GENRE_RHYTHM_GRAMMAR",
    "generate_rhythm_pattern",
    # Harmony
    "HarmonicGrammar",
    "GENRE_HARMONIC_GRAMMAR",
    "generate_chord_progression",
    # Metrics
    "GenreDifferentiationMetrics",
    "MetricsReport",
    "compute_embedding_separation",
    "compute_rhythm_accuracy",
]
