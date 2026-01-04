"""
AETHER Knowledge Layer

Genre profiles, music theory, and domain knowledge.
"""

from aether.knowledge.genres import (
    GenreProfileManager,
    GenreNotFoundError,
    GenreValidationError,
    get_genre_manager,
    BUILTIN_PROFILES,
)

from aether.knowledge.theory import (
    # Note operations
    note_to_midi,
    midi_to_note,
    normalize_note,
    transpose_note,
    interval_between,
    # Scale/chord construction
    get_scale,
    get_chord_notes,
    get_chord_midi,
    get_scale_degree,
    get_diatonic_chord,
    get_diatonic_progression,
    parse_progression,
    # Voice leading
    check_voice_leading,
    smooth_voice_leading_distance,
    VoiceLeadingViolation,
    # Melody analysis
    get_interval_sequence,
    get_contour,
    contour_to_hash,
    analyze_melody_range,
    calculate_singability,
    # Frequency/tuning
    midi_to_frequency,
    frequency_to_midi,
    cents_difference,
    # Rhythm
    beats_to_seconds,
    seconds_to_beats,
    quantize_to_grid,
    get_swing_offset,
    # Constants
    NOTE_NAMES,
    INTERVALS,
    SCALE_PATTERNS,
    CHORD_PATTERNS,
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
