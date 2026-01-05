"""
AETHER Knowledge Layer

Genre profiles, music theory, and domain knowledge.
"""

from aether.knowledge.genres import (
    BUILTIN_PROFILES,
    GenreNotFoundError,
    GenreProfileManager,
    GenreValidationError,
    get_genre_manager,
)
from aether.knowledge.theory import (
    CHORD_PATTERNS,
    INTERVALS,
    # Constants
    NOTE_NAMES,
    SCALE_PATTERNS,
    VoiceLeadingViolation,
    analyze_melody_range,
    # Rhythm
    beats_to_seconds,
    calculate_singability,
    cents_difference,
    # Voice leading
    check_voice_leading,
    contour_to_hash,
    frequency_to_midi,
    get_chord_midi,
    get_chord_notes,
    get_contour,
    get_diatonic_chord,
    get_diatonic_progression,
    # Melody analysis
    get_interval_sequence,
    # Scale/chord construction
    get_scale,
    get_scale_degree,
    get_swing_offset,
    interval_between,
    # Frequency/tuning
    midi_to_frequency,
    midi_to_note,
    normalize_note,
    # Note operations
    note_to_midi,
    parse_progression,
    quantize_to_grid,
    seconds_to_beats,
    smooth_voice_leading_distance,
    transpose_note,
)

__all__ = [
    # Genre management
    "GenreProfileManager",
    "GenreNotFoundError",
    "GenreValidationError",
    "get_genre_manager",
    "BUILTIN_PROFILES",
    # Theory - Notes
    "note_to_midi",
    "midi_to_note",
    "normalize_note",
    "transpose_note",
    "interval_between",
    # Theory - Scales/Chords
    "get_scale",
    "get_chord_notes",
    "get_chord_midi",
    "get_scale_degree",
    "get_diatonic_chord",
    "get_diatonic_progression",
    "parse_progression",
    # Theory - Voice Leading
    "check_voice_leading",
    "smooth_voice_leading_distance",
    "VoiceLeadingViolation",
    # Theory - Melody
    "get_interval_sequence",
    "get_contour",
    "contour_to_hash",
    "analyze_melody_range",
    "calculate_singability",
    # Theory - Frequency
    "midi_to_frequency",
    "frequency_to_midi",
    "cents_difference",
    # Theory - Rhythm
    "beats_to_seconds",
    "seconds_to_beats",
    "quantize_to_grid",
    "get_swing_offset",
    # Constants
    "NOTE_NAMES",
    "INTERVALS",
    "SCALE_PATTERNS",
    "CHORD_PATTERNS",
]
