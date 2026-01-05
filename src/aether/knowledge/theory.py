"""
AETHER Music Theory Utilities

Comprehensive music theory library for harmonic analysis, scale construction,
chord voicing, and melodic operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# Constants
# ============================================================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC_MAP = {
    "Db": "C#",
    "Eb": "D#",
    "Fb": "E",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "B#": "C",
    "E#": "F",
    "Cb": "B",
}

# Intervals in semitones
INTERVALS = {
    "P1": 0,
    "m2": 1,
    "M2": 2,
    "m3": 3,
    "M3": 4,
    "P4": 5,
    "TT": 6,
    "d5": 6,
    "A4": 6,
    "P5": 7,
    "m6": 8,
    "M6": 9,
    "m7": 10,
    "M7": 11,
    "P8": 12,
    "m9": 13,
    "M9": 14,
    "m10": 15,
    "M10": 16,
    "P11": 17,
    "A11": 18,
    "P12": 19,
    "m13": 20,
    "M13": 21,
}

# Scale patterns (intervals from root)
SCALE_PATTERNS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],  # Same as natural minor
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
}

# Chord patterns (intervals from root)
CHORD_PATTERNS = {
    # Triads
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    # Sevenths
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "7": [0, 4, 7, 10],  # Dominant 7
    "dim7": [0, 3, 6, 9],
    "m7b5": [0, 3, 6, 10],  # Half-diminished
    "minmaj7": [0, 3, 7, 11],
    "aug7": [0, 4, 8, 10],
    # Extended
    "9": [0, 4, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
    "min9": [0, 3, 7, 10, 14],
    "11": [0, 4, 7, 10, 14, 17],
    "13": [0, 4, 7, 10, 14, 17, 21],
    # Add chords
    "add9": [0, 4, 7, 14],
    "add11": [0, 4, 7, 17],
    "6": [0, 4, 7, 9],
    "min6": [0, 3, 7, 9],
}


# ============================================================================
# Core Functions
# ============================================================================


def note_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name and octave to MIDI number."""
    note = normalize_note(note)
    if note not in NOTE_NAMES:
        raise ValueError(f"Unknown note: {note}")
    return NOTE_NAMES.index(note) + (octave + 1) * 12


def midi_to_note(midi: int) -> tuple[str, int]:
    """Convert MIDI number to note name and octave."""
    octave = (midi // 12) - 1
    note_idx = midi % 12
    return NOTE_NAMES[note_idx], octave


def normalize_note(note: str) -> str:
    """Normalize note name to sharp notation."""
    note = note.strip()
    if len(note) > 1:
        note = note[0].upper() + note[1:]
    else:
        note = note.upper()
    return ENHARMONIC_MAP.get(note, note)


def transpose_note(note: str, semitones: int) -> str:
    """Transpose a note by semitones."""
    note = normalize_note(note)
    idx = NOTE_NAMES.index(note)
    new_idx = (idx + semitones) % 12
    return NOTE_NAMES[new_idx]


def interval_between(note1: str, note2: str) -> int:
    """Get interval in semitones between two notes."""
    n1 = normalize_note(note1)
    n2 = normalize_note(note2)
    idx1 = NOTE_NAMES.index(n1)
    idx2 = NOTE_NAMES.index(n2)
    return (idx2 - idx1) % 12


def get_scale(root: str, scale_type: str) -> list[str]:
    """Get notes in a scale."""
    root = normalize_note(root)
    if scale_type not in SCALE_PATTERNS:
        raise ValueError(f"Unknown scale type: {scale_type}")

    pattern = SCALE_PATTERNS[scale_type]
    root_idx = NOTE_NAMES.index(root)
    return [NOTE_NAMES[(root_idx + interval) % 12] for interval in pattern]


def get_chord_notes(root: str, chord_type: str) -> list[str]:
    """Get notes in a chord."""
    root = normalize_note(root)
    if chord_type not in CHORD_PATTERNS:
        raise ValueError(f"Unknown chord type: {chord_type}")

    pattern = CHORD_PATTERNS[chord_type]
    root_idx = NOTE_NAMES.index(root)
    return [NOTE_NAMES[(root_idx + interval) % 12] for interval in pattern]


def get_chord_midi(root: str, chord_type: str, octave: int = 4) -> list[int]:
    """Get MIDI note numbers for a chord."""
    root = normalize_note(root)
    if chord_type not in CHORD_PATTERNS:
        raise ValueError(f"Unknown chord type: {chord_type}")

    root_midi = note_to_midi(root, octave)
    pattern = CHORD_PATTERNS[chord_type]
    return [root_midi + interval for interval in pattern]


# ============================================================================
# Diatonic Functions
# ============================================================================


def get_scale_degree(root: str, scale_type: str, degree: int) -> str:
    """Get the nth scale degree (1-indexed)."""
    scale = get_scale(root, scale_type)
    return scale[(degree - 1) % len(scale)]


def get_diatonic_chord(root: str, scale_type: str, degree: int) -> tuple[str, str]:
    """
    Get the diatonic chord for a scale degree.

    Returns (chord_root, chord_type).
    """
    scale = get_scale(root, scale_type)
    chord_root = scale[(degree - 1) % len(scale)]

    # Determine chord quality based on scale type and degree
    if scale_type in ["major", "lydian", "mixolydian"]:
        qualities = ["maj", "min", "min", "maj", "maj", "min", "dim"]
    elif scale_type in ["minor", "aeolian", "dorian", "phrygian"]:
        qualities = ["min", "dim", "maj", "min", "min", "maj", "maj"]
    else:
        qualities = ["maj", "min", "min", "maj", "maj", "min", "dim"]

    return chord_root, qualities[(degree - 1) % 7]


def get_diatonic_progression(
    root: str, scale_type: str, degrees: list[int]
) -> list[tuple[str, str]]:
    """Get chords for a progression given in scale degrees."""
    return [get_diatonic_chord(root, scale_type, d) for d in degrees]


def roman_to_degree(roman: str) -> tuple[int, str]:
    """
    Convert Roman numeral to degree and quality.

    Examples: "I" -> (1, "maj"), "ii" -> (2, "min"), "V7" -> (5, "7")
    """
    roman = roman.strip()

    # Check for quality modifiers
    quality_suffix = ""
    base_roman = roman

    for suffix in ["maj7", "min7", "dim7", "7", "6", "9", "11", "13", "m7b5"]:
        if roman.endswith(suffix):
            quality_suffix = suffix
            base_roman = roman[: -len(suffix)]
            break

    # Handle augmented/diminished
    if base_roman.endswith("°") or base_roman.endswith("o"):
        quality_suffix = quality_suffix or "dim"
        base_roman = base_roman[:-1]
    elif base_roman.endswith("+"):
        quality_suffix = quality_suffix or "aug"
        base_roman = base_roman[:-1]

    # Parse degree
    roman_map = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
        "vii": 7,
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
    }

    degree = roman_map.get(base_roman.lower().replace("b", "").replace("#", ""))
    if degree is None:
        raise ValueError(f"Unknown Roman numeral: {roman}")

    # Determine quality from case if not specified
    if not quality_suffix:
        if base_roman[0].isupper():
            quality_suffix = "maj"
        else:
            quality_suffix = "min"

    return degree, quality_suffix


def parse_progression(root: str, scale_type: str, roman_numerals: str) -> list[tuple[str, str]]:
    """
    Parse a Roman numeral progression string.

    Example: "I-vi-IV-V" -> [("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj")]
    """
    numerals = [n.strip() for n in roman_numerals.replace("-", " ").split()]
    chords = []

    scale = get_scale(root, scale_type)

    for numeral in numerals:
        degree, quality = roman_to_degree(numeral)
        chord_root = scale[(degree - 1) % len(scale)]

        # Handle accidentals in Roman numerals
        if "b" in numeral and not numeral.startswith("b"):
            chord_root = transpose_note(chord_root, -1)
        elif "#" in numeral:
            chord_root = transpose_note(chord_root, 1)

        chords.append((chord_root, quality))

    return chords


# ============================================================================
# Voice Leading
# ============================================================================


@dataclass
class VoiceLeadingViolation:
    """A voice leading violation."""

    type: str
    description: str
    voice1: Optional[int] = None
    voice2: Optional[int] = None
    severity: float = 1.0


def check_voice_leading(
    chord1_midi: list[int],
    chord2_midi: list[int],
) -> list[VoiceLeadingViolation]:
    """
    Check for voice leading violations between two chords.

    Returns list of violations found.
    """
    violations = []

    # Assume voices are sorted from bass to soprano
    c1 = sorted(chord1_midi)
    c2 = sorted(chord2_midi)

    # Match voices (simplistic: by position)
    min_voices = min(len(c1), len(c2))

    for i in range(min_voices - 1):
        for j in range(i + 1, min_voices):
            # Check for parallel fifths
            interval1 = (c1[j] - c1[i]) % 12
            interval2 = (c2[j] - c2[i]) % 12

            if interval1 == 7 and interval2 == 7:  # Both are P5
                motion1 = c2[i] - c1[i]
                motion2 = c2[j] - c1[j]
                if motion1 != 0 and motion1 == motion2:  # Parallel motion
                    violations.append(
                        VoiceLeadingViolation(
                            type="parallel_fifths",
                            description=f"Parallel fifths between voices {i} and {j}",
                            voice1=i,
                            voice2=j,
                        )
                    )

            # Check for parallel octaves
            if interval1 == 0 and interval2 == 0:
                motion1 = c2[i] - c1[i]
                motion2 = c2[j] - c1[j]
                if motion1 != 0 and motion1 == motion2:
                    violations.append(
                        VoiceLeadingViolation(
                            type="parallel_octaves",
                            description=f"Parallel octaves between voices {i} and {j}",
                            voice1=i,
                            voice2=j,
                        )
                    )

    # Check for voice crossing
    for i in range(min_voices - 1):
        if c2[i] > c2[i + 1]:
            violations.append(
                VoiceLeadingViolation(
                    type="voice_crossing",
                    description=f"Voice {i} crosses above voice {i + 1}",
                    voice1=i,
                    voice2=i + 1,
                    severity=0.5,
                )
            )

    # Check for large leaps (> octave)
    for i in range(min_voices):
        leap = abs(c2[i] - c1[i])
        if leap > 12:
            violations.append(
                VoiceLeadingViolation(
                    type="large_leap",
                    description=f"Voice {i} leaps more than an octave ({leap} semitones)",
                    voice1=i,
                    severity=0.7,
                )
            )

    return violations


def smooth_voice_leading_distance(
    chord1_midi: list[int],
    chord2_midi: list[int],
) -> float:
    """
    Calculate the total voice leading distance (in semitones).

    Lower is smoother voice leading.
    """
    c1 = sorted(chord1_midi)
    c2 = sorted(chord2_midi)
    min_voices = min(len(c1), len(c2))

    return sum(abs(c2[i] - c1[i]) for i in range(min_voices))


# ============================================================================
# Melody Analysis
# ============================================================================


def get_interval_sequence(midi_notes: list[int]) -> list[int]:
    """Get sequence of intervals between consecutive notes."""
    return [midi_notes[i + 1] - midi_notes[i] for i in range(len(midi_notes) - 1)]


def get_contour(midi_notes: list[int]) -> list[str]:
    """Get melodic contour as up/down/same sequence."""
    contour = []
    for i in range(len(midi_notes) - 1):
        diff = midi_notes[i + 1] - midi_notes[i]
        if diff > 0:
            contour.append("up")
        elif diff < 0:
            contour.append("down")
        else:
            contour.append("same")
    return contour


def contour_to_hash(contour: list[str]) -> str:
    """Convert contour to a hash string for comparison."""
    mapping = {"up": "U", "down": "D", "same": "S"}
    return "".join(mapping.get(c, "?") for c in contour)


def analyze_melody_range(midi_notes: list[int]) -> dict[str, int]:
    """Analyze the range of a melody."""
    if not midi_notes:
        return {"lowest": 0, "highest": 0, "range": 0}

    lowest = min(midi_notes)
    highest = max(midi_notes)
    return {
        "lowest": lowest,
        "highest": highest,
        "range": highest - lowest,
        "lowest_note": midi_to_note(lowest),
        "highest_note": midi_to_note(highest),
    }


def calculate_singability(midi_notes: list[int]) -> float:
    """
    Calculate singability score (0-1).

    Higher score = more singable (stepwise motion, comfortable range).
    """
    if len(midi_notes) < 2:
        return 1.0

    intervals = get_interval_sequence(midi_notes)

    # Penalize large intervals
    large_interval_penalty = sum(1 for i in intervals if abs(i) > 5) / len(intervals)

    # Reward stepwise motion
    stepwise_ratio = sum(1 for i in intervals if abs(i) <= 2) / len(intervals)

    # Check range (comfortable = ~12 semitones / 1 octave)
    range_info = analyze_melody_range(midi_notes)
    range_penalty = max(0, (range_info["range"] - 12) / 24)

    score = (
        (stepwise_ratio * 0.5) + ((1 - large_interval_penalty) * 0.3) + ((1 - range_penalty) * 0.2)
    )
    return max(0, min(1, score))


# ============================================================================
# Frequency / Tuning
# ============================================================================


def midi_to_frequency(midi: int, a4_freq: float = 440.0) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return a4_freq * (2 ** ((midi - 69) / 12))


def frequency_to_midi(freq: float, a4_freq: float = 440.0) -> float:
    """Convert frequency to MIDI note number (may be fractional)."""
    return 69 + 12 * math.log2(freq / a4_freq)


def cents_difference(freq1: float, freq2: float) -> float:
    """Calculate the difference in cents between two frequencies."""
    return 1200 * math.log2(freq2 / freq1)


# ============================================================================
# Rhythm Utilities
# ============================================================================


def beats_to_seconds(beats: float, bpm: float) -> float:
    """Convert beats to seconds."""
    return beats * 60.0 / bpm


def seconds_to_beats(seconds: float, bpm: float) -> float:
    """Convert seconds to beats."""
    return seconds * bpm / 60.0


def quantize_to_grid(position: float, grid: float, strength: float = 1.0) -> float:
    """
    Quantize a position to a grid.

    Args:
        position: Position in beats
        grid: Grid size in beats (e.g., 0.25 for 16th notes)
        strength: Quantize strength (0-1)
    """
    quantized = round(position / grid) * grid
    return position + (quantized - position) * strength


def get_swing_offset(position: float, swing_amount: float, grid: float = 0.5) -> float:
    """
    Calculate swing offset for a position.

    Swing delays every other grid position.
    """
    grid_pos = position / grid
    is_offbeat = (int(grid_pos) % 2) == 1

    if is_offbeat:
        # swing_amount: 0 = straight, 0.33 = light, 0.66 = triplet
        return swing_amount * grid * 0.5
    return 0.0


# ============================================================================
# EXTENDED SCALE LIBRARY - World-Class Coverage
# ============================================================================

EXTENDED_SCALES = {
    # ---- Western Classical Modes ----
    "ionian": [0, 2, 4, 5, 7, 9, 11],  # Major
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
    "locrian": [0, 1, 3, 5, 6, 8, 10],

    # ---- Jazz Modes (Melodic Minor) ----
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "dorian_b2": [0, 1, 3, 5, 7, 9, 10],  # Phrygian #6
    "lydian_augmented": [0, 2, 4, 6, 8, 9, 11],
    "lydian_dominant": [0, 2, 4, 6, 7, 9, 10],  # Lydian b7
    "mixolydian_b6": [0, 2, 4, 5, 7, 8, 10],  # Hindu scale
    "locrian_natural2": [0, 2, 3, 5, 6, 8, 10],  # Half-diminished
    "altered": [0, 1, 3, 4, 6, 8, 10],  # Super Locrian

    # ---- Jazz Modes (Harmonic Minor) ----
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "locrian_natural6": [0, 1, 3, 5, 6, 9, 10],
    "ionian_augmented": [0, 2, 4, 5, 8, 9, 11],
    "dorian_sharp4": [0, 2, 3, 6, 7, 9, 10],  # Romanian
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],  # Spanish/Jewish
    "lydian_sharp2": [0, 3, 4, 6, 7, 9, 11],
    "ultralocrian": [0, 1, 3, 4, 6, 8, 9],  # Altered bb7

    # ---- Harmonic Major Modes ----
    "harmonic_major": [0, 2, 4, 5, 7, 8, 11],
    "dorian_b5": [0, 2, 3, 5, 6, 9, 10],
    "phrygian_b4": [0, 1, 3, 4, 7, 8, 10],
    "lydian_b3": [0, 2, 3, 6, 7, 9, 11],
    "mixolydian_b2": [0, 1, 4, 5, 7, 9, 10],
    "lydian_augmented_sharp2": [0, 3, 4, 6, 8, 9, 11],
    "locrian_bb7": [0, 1, 3, 5, 6, 8, 9],

    # ---- Bebop Scales ----
    "bebop_dominant": [0, 2, 4, 5, 7, 9, 10, 11],
    "bebop_major": [0, 2, 4, 5, 7, 8, 9, 11],
    "bebop_dorian": [0, 2, 3, 4, 5, 7, 9, 10],
    "bebop_melodic_minor": [0, 2, 3, 5, 7, 8, 9, 11],
    "bebop_harmonic_minor": [0, 2, 3, 5, 7, 8, 10, 11],

    # ---- Pentatonic & Blues ----
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "major_blues": [0, 2, 3, 4, 7, 9],
    "minor_blues": [0, 3, 5, 6, 7, 10],
    "blues_hexatonic": [0, 3, 4, 6, 7, 10],

    # ---- Symmetric Scales ----
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "diminished_hw": [0, 1, 3, 4, 6, 7, 9, 10],  # Half-whole
    "diminished_wh": [0, 2, 3, 5, 6, 8, 9, 11],  # Whole-half
    "augmented": [0, 3, 4, 7, 8, 11],
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "tritone": [0, 1, 4, 6, 7, 10],

    # ---- World Scales - Middle Eastern ----
    "arabic": [0, 2, 4, 5, 6, 8, 10],
    "persian": [0, 1, 4, 5, 6, 8, 11],
    "byzantine": [0, 1, 4, 5, 7, 8, 11],  # Double harmonic major
    "hungarian_minor": [0, 2, 3, 6, 7, 8, 11],
    "hungarian_major": [0, 3, 4, 6, 7, 9, 10],
    "double_harmonic": [0, 1, 4, 5, 7, 8, 11],
    "maqam_hijaz": [0, 1, 4, 5, 7, 8, 10],
    "maqam_bayati": [0, 1.5, 3, 5, 7, 8, 10],  # Quarter tone
    "maqam_saba": [0, 1.5, 3, 4, 6, 8, 10],
    "maqam_sikah": [0, 1.5, 4, 5, 7, 8.5, 10],

    # ---- World Scales - Asian ----
    "chinese": [0, 4, 6, 7, 11],
    "japanese": [0, 1, 5, 7, 8],  # In scale
    "hirajoshi": [0, 2, 3, 7, 8],
    "kumoi": [0, 2, 3, 7, 9],
    "iwato": [0, 1, 5, 6, 10],
    "yo": [0, 2, 5, 7, 9],  # Japanese major pentatonic
    "pelog": [0, 1, 3, 7, 8],  # Indonesian
    "slendro": [0, 2, 5, 7, 9],  # Indonesian
    "raga_bhairav": [0, 1, 4, 5, 7, 8, 11],
    "raga_todi": [0, 1, 3, 6, 7, 8, 11],
    "raga_purvi": [0, 1, 4, 6, 7, 8, 11],
    "raga_marwa": [0, 1, 4, 6, 7, 9, 11],
    "raga_kafi": [0, 2, 3, 5, 7, 9, 10],  # Same as Dorian

    # ---- World Scales - Other ----
    "spanish_gypsy": [0, 1, 4, 5, 7, 8, 10],
    "flamenco": [0, 1, 4, 5, 7, 8, 11],
    "jewish": [0, 1, 4, 5, 7, 8, 10],
    "gypsy": [0, 2, 3, 6, 7, 8, 11],
    "romanian": [0, 2, 3, 6, 7, 9, 10],
    "neapolitan_minor": [0, 1, 3, 5, 7, 8, 11],
    "neapolitan_major": [0, 1, 3, 5, 7, 9, 11],
    "enigmatic": [0, 1, 4, 6, 8, 10, 11],
    "prometheus": [0, 2, 4, 6, 9, 10],
    "prometheus_neapolitan": [0, 1, 4, 6, 9, 10],

    # ---- African/Caribbean ----
    "ethiopian": [0, 2, 3, 5, 7, 8, 10],
    "algerian": [0, 2, 3, 6, 7, 8, 11],
    "cuban": [0, 3, 4, 6, 7, 9, 10],  # Minor bebop

    # ---- Contemporary/Experimental ----
    "super_locrian": [0, 1, 3, 4, 6, 8, 10],  # Altered dominant
    "leading_whole_tone": [0, 2, 4, 6, 8, 10, 11],
    "synthetic_mixture_5": [0, 1, 4, 5, 7, 9, 10],
    "composite_blues": [0, 2, 3, 4, 5, 6, 7, 9, 10],
}

# Merge extended scales into main SCALE_PATTERNS
SCALE_PATTERNS.update(EXTENDED_SCALES)


# ============================================================================
# EXTENDED CHORD VOCABULARY - Professional Level
# ============================================================================

EXTENDED_CHORDS = {
    # ---- Basic Triads ----
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "5": [0, 7],  # Power chord

    # ---- Seventh Chords ----
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "7": [0, 4, 7, 10],  # Dominant 7
    "dim7": [0, 3, 6, 9],
    "m7b5": [0, 3, 6, 10],  # Half-diminished (ø7)
    "minmaj7": [0, 3, 7, 11],  # Minor-major 7
    "aug7": [0, 4, 8, 10],  # Augmented 7
    "augmaj7": [0, 4, 8, 11],  # Augmented major 7
    "7sus4": [0, 5, 7, 10],
    "7sus2": [0, 2, 7, 10],

    # ---- Sixth Chords ----
    "6": [0, 4, 7, 9],
    "min6": [0, 3, 7, 9],
    "6/9": [0, 4, 7, 9, 14],
    "min6/9": [0, 3, 7, 9, 14],

    # ---- Extended Chords ----
    "9": [0, 4, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
    "min9": [0, 3, 7, 10, 14],
    "9sus4": [0, 5, 7, 10, 14],
    "11": [0, 4, 7, 10, 14, 17],
    "maj11": [0, 4, 7, 11, 14, 17],
    "min11": [0, 3, 7, 10, 14, 17],
    "13": [0, 4, 7, 10, 14, 17, 21],
    "maj13": [0, 4, 7, 11, 14, 17, 21],
    "min13": [0, 3, 7, 10, 14, 17, 21],

    # ---- Altered Dominant Chords ----
    "7b5": [0, 4, 6, 10],
    "7#5": [0, 4, 8, 10],
    "7b9": [0, 4, 7, 10, 13],
    "7#9": [0, 4, 7, 10, 15],
    "7b5b9": [0, 4, 6, 10, 13],
    "7b5#9": [0, 4, 6, 10, 15],
    "7#5b9": [0, 4, 8, 10, 13],
    "7#5#9": [0, 4, 8, 10, 15],
    "7#11": [0, 4, 7, 10, 14, 18],
    "7b13": [0, 4, 7, 10, 14, 20],
    "13b9": [0, 4, 7, 10, 13, 17, 21],
    "13#9": [0, 4, 7, 10, 15, 17, 21],
    "13#11": [0, 4, 7, 10, 14, 18, 21],
    "7alt": [0, 4, 8, 10, 13, 15],  # Generic altered

    # ---- Add Chords ----
    "add9": [0, 4, 7, 14],
    "add11": [0, 4, 7, 17],
    "add13": [0, 4, 7, 21],
    "minadd9": [0, 3, 7, 14],
    "minadd11": [0, 3, 7, 17],

    # ---- Clusters & Modern ----
    "quartal": [0, 5, 10],  # Stacked 4ths
    "quartal_4voice": [0, 5, 10, 15],
    "quintal": [0, 7, 14],  # Stacked 5ths
    "mu": [0, 2, 4, 7],  # Steely Dan chord
    "so_what": [0, 5, 10, 15, 19],  # Miles Davis "So What" voicing

    # ---- Jazz Voicing Shapes ----
    "rootless_maj9": [4, 7, 11, 14],  # Type A
    "rootless_min9": [3, 7, 10, 14],  # Type A
    "rootless_dom9": [4, 7, 10, 14],  # Type A
    "rootless_maj9_b": [7, 11, 14, 16],  # Type B
    "rootless_min9_b": [7, 10, 14, 15],  # Type B
    "rootless_dom9_b": [10, 14, 16, 19],  # Type B
}

# Merge extended chords into main CHORD_PATTERNS
CHORD_PATTERNS.update(EXTENDED_CHORDS)


# ============================================================================
# CHORD VOICINGS - Drop Voicings & Inversions
# ============================================================================

@dataclass
class ChordVoicing:
    """Represents a specific chord voicing with MIDI notes."""
    name: str
    intervals: list[int]
    description: str = ""
    style: str = "standard"  # standard, jazz, classical, modern
    difficulty: float = 0.5  # 0-1


def get_drop2_voicing(chord_type: str, root_midi: int) -> list[int]:
    """
    Create drop-2 voicing: drop the 2nd highest note down an octave.
    Standard jazz guitar/piano voicing.
    """
    if chord_type not in CHORD_PATTERNS:
        raise ValueError(f"Unknown chord type: {chord_type}")

    intervals = sorted(CHORD_PATTERNS[chord_type])
    if len(intervals) < 4:
        # For triads, just return close voicing
        return [root_midi + i for i in intervals]

    # Build close position voicing
    notes = [root_midi + i for i in intervals]

    # Drop the 2nd highest note down an octave
    if len(notes) >= 2:
        second_highest_idx = len(notes) - 2
        notes[second_highest_idx] -= 12

    return sorted(notes)


def get_drop3_voicing(chord_type: str, root_midi: int) -> list[int]:
    """
    Create drop-3 voicing: drop the 3rd highest note down an octave.
    Creates wider spacing, good for big band voicings.
    """
    if chord_type not in CHORD_PATTERNS:
        raise ValueError(f"Unknown chord type: {chord_type}")

    intervals = sorted(CHORD_PATTERNS[chord_type])
    if len(intervals) < 4:
        return [root_midi + i for i in intervals]

    notes = [root_midi + i for i in intervals]

    if len(notes) >= 3:
        third_highest_idx = len(notes) - 3
        notes[third_highest_idx] -= 12

    return sorted(notes)


def get_drop24_voicing(chord_type: str, root_midi: int) -> list[int]:
    """
    Create drop-2&4 voicing: drop both 2nd and 4th highest notes.
    Very open spacing for orchestral/big band.
    """
    if chord_type not in CHORD_PATTERNS:
        raise ValueError(f"Unknown chord type: {chord_type}")

    intervals = sorted(CHORD_PATTERNS[chord_type])
    if len(intervals) < 4:
        return [root_midi + i for i in intervals]

    notes = [root_midi + i for i in intervals]

    if len(notes) >= 2:
        notes[len(notes) - 2] -= 12
    if len(notes) >= 4:
        notes[len(notes) - 4] -= 12

    return sorted(notes)


def get_spread_voicing(chord_type: str, root_midi: int, spread: int = 2) -> list[int]:
    """
    Create spread voicing: distribute notes across octaves.
    Spread=2 means notes span 2 octaves.
    """
    if chord_type not in CHORD_PATTERNS:
        raise ValueError(f"Unknown chord type: {chord_type}")

    intervals = CHORD_PATTERNS[chord_type]
    notes = []
    octave_offset = 0

    for i, interval in enumerate(intervals):
        if i > 0 and i % 2 == 0:
            octave_offset += 12
        notes.append(root_midi + interval + octave_offset)

    return notes


def get_shell_voicing(chord_type: str, root_midi: int) -> list[int]:
    """
    Create shell voicing: root + 3rd + 7th only.
    Essential jazz piano voicing for comping.
    """
    if chord_type not in CHORD_PATTERNS:
        raise ValueError(f"Unknown chord type: {chord_type}")

    intervals = CHORD_PATTERNS[chord_type]

    # Extract essential tones
    root = 0
    third = None
    seventh = None

    for i in intervals:
        if i in [3, 4]:  # Minor or major 3rd
            third = i
        elif i in [10, 11]:  # Minor or major 7th
            seventh = i

    shell = [root_midi]
    if third:
        shell.append(root_midi + third)
    if seventh:
        shell.append(root_midi + seventh)
    elif len(intervals) > 2:  # Use 5th if no 7th
        shell.append(root_midi + 7)

    return shell


def get_rootless_voicing(chord_type: str, root_midi: int, type_a: bool = True) -> list[int]:
    """
    Create rootless voicing for jazz piano.
    Type A: 3-5-7-9 (bass plays root)
    Type B: 7-9-3-5 (different inversion)
    """
    intervals = CHORD_PATTERNS.get(chord_type, [0, 4, 7, 10])

    # Build with extensions
    third = 4 if any(i == 4 for i in intervals) else 3
    fifth = 7
    seventh = 11 if any(i == 11 for i in intervals) else 10
    ninth = 14

    if type_a:
        # Type A: 3-5-7-9 from bottom
        return [root_midi + third, root_midi + fifth, root_midi + seventh, root_midi + ninth]
    else:
        # Type B: 7-9-3-5 from bottom
        return [root_midi + seventh - 12, root_midi + ninth - 12, root_midi + third, root_midi + fifth]


def invert_chord(midi_notes: list[int], inversion: int) -> list[int]:
    """
    Get chord inversion.

    inversion=0: root position
    inversion=1: first inversion (root up octave)
    inversion=2: second inversion (root & 3rd up octave)
    etc.
    """
    notes = sorted(midi_notes)
    for _ in range(inversion):
        if notes:
            lowest = notes.pop(0)
            notes.append(lowest + 12)
    return notes


# ============================================================================
# COUNTERPOINT RULES - Classical Voice Leading
# ============================================================================

@dataclass
class CounterpointRule:
    """A rule for counterpoint composition."""
    name: str
    description: str
    severity: float = 1.0  # 0-1, how strict
    style: str = "species"  # species, free, modern


COUNTERPOINT_RULES: list[CounterpointRule] = [
    CounterpointRule(
        "no_parallel_fifths",
        "Avoid parallel perfect fifths between any two voices",
        severity=1.0,
        style="species"
    ),
    CounterpointRule(
        "no_parallel_octaves",
        "Avoid parallel octaves/unisons between any two voices",
        severity=1.0,
        style="species"
    ),
    CounterpointRule(
        "no_hidden_fifths",
        "Avoid approaching P5 by similar motion in outer voices",
        severity=0.8,
        style="species"
    ),
    CounterpointRule(
        "no_hidden_octaves",
        "Avoid approaching P8 by similar motion in outer voices",
        severity=0.8,
        style="species"
    ),
    CounterpointRule(
        "contrary_motion_preferred",
        "Prefer contrary or oblique motion over parallel/similar",
        severity=0.5,
        style="species"
    ),
    CounterpointRule(
        "resolve_leading_tone",
        "Leading tone (scale degree 7) should resolve up to tonic",
        severity=0.9,
        style="species"
    ),
    CounterpointRule(
        "resolve_tritone",
        "Tritone should resolve: aug4 outward, dim5 inward",
        severity=0.8,
        style="species"
    ),
    CounterpointRule(
        "avoid_augmented_second",
        "Avoid melodic augmented second (3 semitones with accidental)",
        severity=0.7,
        style="species"
    ),
    CounterpointRule(
        "limit_leaps",
        "Avoid leaps larger than an octave; compensate large leaps",
        severity=0.6,
        style="species"
    ),
    CounterpointRule(
        "balance_leaps",
        "After a leap, move by step in the opposite direction",
        severity=0.5,
        style="species"
    ),
]


def analyze_motion_type(
    voice1_start: int, voice1_end: int,
    voice2_start: int, voice2_end: int
) -> str:
    """
    Analyze the type of motion between two voices.

    Returns: "parallel", "similar", "contrary", "oblique"
    """
    motion1 = voice1_end - voice1_start
    motion2 = voice2_end - voice2_start

    if motion1 == 0 and motion2 == 0:
        return "oblique"  # Both stationary
    elif motion1 == 0 or motion2 == 0:
        return "oblique"  # One voice stationary
    elif motion1 > 0 and motion2 < 0:
        return "contrary"
    elif motion1 < 0 and motion2 > 0:
        return "contrary"
    elif motion1 == motion2:
        return "parallel"
    else:
        return "similar"  # Same direction, different amount


def check_species_counterpoint(
    cantus_firmus: list[int],
    counterpoint: list[int]
) -> list[VoiceLeadingViolation]:
    """
    Check a counterpoint line against a cantus firmus for species violations.
    """
    violations = []

    for i in range(len(cantus_firmus) - 1):
        if i >= len(counterpoint) - 1:
            break

        cf_curr, cf_next = cantus_firmus[i], cantus_firmus[i + 1]
        cp_curr, cp_next = counterpoint[i], counterpoint[i + 1]

        # Check interval at current position
        interval_curr = abs(cp_curr - cf_curr) % 12
        interval_next = abs(cp_next - cf_next) % 12

        # Check for parallel fifths
        if interval_curr == 7 and interval_next == 7:
            motion = analyze_motion_type(cf_curr, cf_next, cp_curr, cp_next)
            if motion == "parallel":
                violations.append(VoiceLeadingViolation(
                    type="parallel_fifths",
                    description=f"Parallel fifths at position {i}",
                    severity=1.0
                ))

        # Check for parallel octaves
        if interval_curr == 0 and interval_next == 0:
            motion = analyze_motion_type(cf_curr, cf_next, cp_curr, cp_next)
            if motion == "parallel":
                violations.append(VoiceLeadingViolation(
                    type="parallel_octaves",
                    description=f"Parallel octaves at position {i}",
                    severity=1.0
                ))

        # Check for large leaps
        cp_leap = abs(cp_next - cp_curr)
        if cp_leap > 12:
            violations.append(VoiceLeadingViolation(
                type="large_leap",
                description=f"Leap of {cp_leap} semitones at position {i}",
                severity=0.7
            ))

    return violations


# ============================================================================
# CADENCES - Musical Phrase Endings
# ============================================================================

@dataclass
class Cadence:
    """Represents a musical cadence."""
    name: str
    roman_numerals: str
    description: str
    finality: float  # 0-1, how conclusive
    genre_affinity: dict[str, float]


CADENCES: dict[str, Cadence] = {
    # ---- Authentic Cadences ----
    "perfect_authentic": Cadence(
        name="Perfect Authentic Cadence",
        roman_numerals="V-I",
        description="Dominant to tonic, both in root position, tonic in soprano",
        finality=1.0,
        genre_affinity={"classical": 1.0, "pop": 0.8, "jazz": 0.6, "rock": 0.7}
    ),
    "imperfect_authentic": Cadence(
        name="Imperfect Authentic Cadence",
        roman_numerals="V-I",
        description="Dominant to tonic, one or both inverted or 3rd/5th in soprano",
        finality=0.8,
        genre_affinity={"classical": 1.0, "pop": 0.8, "jazz": 0.7, "rock": 0.7}
    ),

    # ---- Half Cadences ----
    "half": Cadence(
        name="Half Cadence",
        roman_numerals="?-V",
        description="Any chord moving to dominant, feels incomplete",
        finality=0.3,
        genre_affinity={"classical": 1.0, "pop": 0.7, "jazz": 0.8, "rock": 0.6}
    ),
    "phrygian_half": Cadence(
        name="Phrygian Half Cadence",
        roman_numerals="iv6-V",
        description="Minor iv6 to V in minor, distinctive sound",
        finality=0.3,
        genre_affinity={"classical": 1.0, "flamenco": 1.0, "jazz": 0.5}
    ),

    # ---- Plagal Cadences ----
    "plagal": Cadence(
        name="Plagal Cadence (Amen)",
        roman_numerals="IV-I",
        description="Subdominant to tonic, 'Amen' cadence",
        finality=0.7,
        genre_affinity={"classical": 0.8, "gospel": 1.0, "pop": 0.6, "rock": 0.7}
    ),
    "minor_plagal": Cadence(
        name="Minor Plagal Cadence",
        roman_numerals="iv-I",
        description="Minor subdominant to major tonic",
        finality=0.7,
        genre_affinity={"pop": 0.9, "rock": 0.8, "gospel": 0.9}
    ),

    # ---- Deceptive Cadences ----
    "deceptive": Cadence(
        name="Deceptive Cadence",
        roman_numerals="V-vi",
        description="Dominant to submediant instead of tonic",
        finality=0.2,
        genre_affinity={"classical": 1.0, "pop": 0.9, "jazz": 0.7, "rock": 0.6}
    ),
    "deceptive_iv": Cadence(
        name="Deceptive to IV",
        roman_numerals="V-IV",
        description="Dominant resolving to subdominant (rock cliché)",
        finality=0.2,
        genre_affinity={"rock": 1.0, "pop": 0.8, "blues": 0.7}
    ),

    # ---- Jazz Cadences ----
    "ii_v_i": Cadence(
        name="ii-V-I",
        roman_numerals="ii7-V7-Imaj7",
        description="The essential jazz cadence",
        finality=0.9,
        genre_affinity={"jazz": 1.0, "r-and-b": 0.8, "neo-soul": 0.9, "pop": 0.5}
    ),
    "tritone_sub": Cadence(
        name="Tritone Substitution",
        roman_numerals="bII7-I",
        description="Dominant substituted by chord a tritone away",
        finality=0.8,
        genre_affinity={"jazz": 1.0, "neo-soul": 0.8, "r-and-b": 0.6}
    ),
    "backdoor": Cadence(
        name="Backdoor Cadence",
        roman_numerals="bVII7-I",
        description="Flat VII dominant to tonic",
        finality=0.7,
        genre_affinity={"jazz": 1.0, "pop": 0.7, "r-and-b": 0.7}
    ),

    # ---- Modal Cadences ----
    "aeolian": Cadence(
        name="Aeolian Cadence",
        roman_numerals="bVII-i",
        description="Subtonic to minor tonic",
        finality=0.6,
        genre_affinity={"rock": 0.9, "metal": 0.8, "pop": 0.7}
    ),
    "dorian": Cadence(
        name="Dorian Cadence",
        roman_numerals="IV-i",
        description="Major IV to minor tonic in Dorian mode",
        finality=0.6,
        genre_affinity={"jazz": 0.8, "funk": 0.9, "neo-soul": 0.8}
    ),
}


def get_cadence_chords(
    cadence_name: str,
    root: str,
    scale_type: str = "major"
) -> list[tuple[str, str]]:
    """
    Get the actual chords for a cadence in a given key.
    """
    if cadence_name not in CADENCES:
        raise ValueError(f"Unknown cadence: {cadence_name}")

    cadence = CADENCES[cadence_name]
    return parse_progression(root, scale_type, cadence.roman_numerals)


# ============================================================================
# GENRE-SPECIFIC PROGRESSIONS
# ============================================================================

GENRE_PROGRESSIONS: dict[str, list[dict]] = {
    "pop": [
        {"name": "I-V-vi-IV", "numerals": "I V vi IV", "style": "happy", "popularity": 1.0},
        {"name": "vi-IV-I-V", "numerals": "vi IV I V", "style": "melancholic", "popularity": 0.9},
        {"name": "I-IV-vi-V", "numerals": "I IV vi V", "style": "uplifting", "popularity": 0.8},
        {"name": "I-vi-IV-V", "numerals": "I vi IV V", "style": "classic", "popularity": 0.8},
        {"name": "IV-I-V-vi", "numerals": "IV I V vi", "style": "anthem", "popularity": 0.7},
    ],
    "jazz": [
        {"name": "ii-V-I", "numerals": "ii7 V7 Imaj7", "style": "standard", "popularity": 1.0},
        {"name": "I-vi-ii-V", "numerals": "Imaj7 vi7 ii7 V7", "style": "turnaround", "popularity": 0.95},
        {"name": "iii-vi-ii-V", "numerals": "iii7 vi7 ii7 V7", "style": "rhythm_changes", "popularity": 0.85},
        {"name": "Imaj7-IV7-iii7-bIII°7-ii7-V7", "numerals": "Imaj7 IV7 iii7 bIII°7 ii7 V7", "style": "chromatic", "popularity": 0.7},
        {"name": "Coltrane_changes", "numerals": "Imaj7 bIII7 V7 bVII7", "style": "giant_steps", "popularity": 0.6},
    ],
    "blues": [
        {"name": "12_bar_basic", "numerals": "I7 I7 I7 I7 IV7 IV7 I7 I7 V7 IV7 I7 V7", "style": "standard", "popularity": 1.0},
        {"name": "12_bar_quick_iv", "numerals": "I7 IV7 I7 I7 IV7 IV7 I7 I7 V7 IV7 I7 V7", "style": "quick_change", "popularity": 0.9},
        {"name": "minor_blues", "numerals": "i7 i7 i7 i7 iv7 iv7 i7 i7 VI7 V7 i7 V7", "style": "minor", "popularity": 0.8},
        {"name": "jazz_blues", "numerals": "I7 IV7 I7 I7 IV7 #iv°7 I7 vi7 ii7 V7 I7 V7", "style": "jazz", "popularity": 0.85},
    ],
    "rock": [
        {"name": "I-bVII-IV", "numerals": "I bVII IV", "style": "classic_rock", "popularity": 1.0},
        {"name": "I-IV-V", "numerals": "I IV V", "style": "basic", "popularity": 0.95},
        {"name": "i-bVI-bIII-bVII", "numerals": "i bVI bIII bVII", "style": "aeolian", "popularity": 0.85},
        {"name": "i-bVII-bVI-V", "numerals": "i bVII bVI V", "style": "andalusian", "popularity": 0.8},
        {"name": "I-bIII-IV", "numerals": "I bIII IV", "style": "mixolydian", "popularity": 0.75},
    ],
    "r-and-b": [
        {"name": "I-IV-I-V", "numerals": "Imaj7 IVmaj7 Imaj7 V7", "style": "smooth", "popularity": 0.9},
        {"name": "ii-V-I-IV", "numerals": "ii9 V13 Imaj9 IVmaj9", "style": "neo_soul", "popularity": 0.95},
        {"name": "vi-ii-V-I", "numerals": "vi9 ii9 V13 Imaj9", "style": "sophisticated", "popularity": 0.85},
        {"name": "I-iii-IV-iv", "numerals": "Imaj7 iii7 IVmaj7 iv7", "style": "borrowed", "popularity": 0.8},
    ],
    "funk": [
        {"name": "i7_vamp", "numerals": "i9 i9 i9 i9", "style": "one_chord", "popularity": 1.0},
        {"name": "i-IV", "numerals": "i7 IV7", "style": "two_chord", "popularity": 0.95},
        {"name": "i-bVII-IV", "numerals": "i7 bVII7 IV7", "style": "dorian", "popularity": 0.85},
    ],
    "trap": [
        {"name": "i-bVI-bVII", "numerals": "i bVI bVII", "style": "dark", "popularity": 1.0},
        {"name": "i-iv-bVI", "numerals": "i iv bVI", "style": "minor", "popularity": 0.9},
        {"name": "i-bVI-bIII-bVII", "numerals": "i bVI bIII bVII", "style": "melodic", "popularity": 0.85},
    ],
    "house": [
        {"name": "I-V-vi-IV", "numerals": "I V vi IV", "style": "progressive", "popularity": 0.9},
        {"name": "vi-IV-I-V", "numerals": "vi IV I V", "style": "emotional", "popularity": 0.95},
        {"name": "i_vamp", "numerals": "i i i i", "style": "minimal", "popularity": 0.85},
    ],
    "ambient": [
        {"name": "Imaj9_sus", "numerals": "Imaj9 Imaj9 Imaj9 Imaj9", "style": "static", "popularity": 0.9},
        {"name": "I-bVII-IV", "numerals": "Imaj9 bVIImaj9 IVmaj9", "style": "floating", "popularity": 0.85},
        {"name": "quartal", "numerals": "I5 bVII5 IV5", "style": "open", "popularity": 0.8},
    ],
}


def get_genre_progression(
    genre: str,
    root: str,
    style: Optional[str] = None
) -> list[tuple[str, str]]:
    """
    Get a chord progression appropriate for a genre.
    """
    if genre not in GENRE_PROGRESSIONS:
        # Default to pop progressions
        genre = "pop"

    progressions = GENRE_PROGRESSIONS[genre]

    # Filter by style if specified
    if style:
        matching = [p for p in progressions if p.get("style") == style]
        if matching:
            progressions = matching

    # Choose based on popularity (weighted random would be better, but deterministic for now)
    prog = max(progressions, key=lambda p: p.get("popularity", 0))

    # Parse into actual chords
    scale_type = "minor" if any(prog["numerals"].startswith(x) for x in ["i", "vi"]) else "major"
    return parse_progression(root, scale_type, prog["numerals"])


# ============================================================================
# TENSION & RESOLUTION
# ============================================================================

@dataclass
class TensionNote:
    """A tension/extension note and its resolution."""
    name: str
    interval: int  # From root
    resolution_interval: int
    tension_level: float  # 0-1
    description: str


TENSION_NOTES: dict[str, TensionNote] = {
    "b9": TensionNote("flat 9", 13, 12, 0.9, "Very dissonant, resolves to root"),
    "9": TensionNote("9", 14, 12, 0.3, "Color tone, resolves to root"),
    "#9": TensionNote("sharp 9", 15, 16, 0.8, "Hendrix chord sound, resolves to 3rd"),
    "11": TensionNote("11", 17, 16, 0.5, "Suspended feel, resolves to 3rd"),
    "#11": TensionNote("sharp 11", 18, 19, 0.6, "Lydian sound, resolves to 5th"),
    "b13": TensionNote("flat 13", 20, 19, 0.7, "Dark color, resolves to 5th"),
    "13": TensionNote("13", 21, 19, 0.4, "Sweet extension, resolves to 5th"),
}


def calculate_chord_tension(chord_midi: list[int], root_midi: int) -> float:
    """
    Calculate overall tension level of a chord (0-1).
    Higher = more tension/dissonance.
    """
    intervals = sorted([(n - root_midi) % 12 for n in chord_midi])
    tension = 0.0

    # Minor 2nd (1 semitone) = high tension
    if 1 in intervals:
        tension += 0.3

    # Tritone (6 semitones) = significant tension
    if 6 in intervals:
        tension += 0.25

    # Minor 9th within chord
    for i, int1 in enumerate(intervals):
        for int2 in intervals[i+1:]:
            if (int2 - int1) % 12 == 1:  # Minor 2nd apart
                tension += 0.2

    # Added extensions add mild tension
    extension_count = len([i for i in intervals if i > 11])
    tension += extension_count * 0.05

    return min(1.0, tension)


def suggest_resolution(chord_type: str, root: str) -> list[tuple[str, str]]:
    """
    Suggest possible resolution chords for a given chord.
    """
    resolutions = []

    # Dominant chords want to resolve down a fifth
    if "7" in chord_type and "maj7" not in chord_type:
        resolution_root = transpose_note(root, -7)  # Down a P5
        resolutions.append((resolution_root, "maj"))
        resolutions.append((resolution_root, "maj7"))
        resolutions.append((resolution_root, "min"))  # Deceptive to relative minor

    # Diminished chords typically resolve up a half step
    if "dim" in chord_type:
        resolution_root = transpose_note(root, 1)
        resolutions.append((resolution_root, "min"))
        resolutions.append((resolution_root, "maj"))

    # Augmented chords can resolve multiple ways
    if "aug" in chord_type:
        # Up a half step
        resolutions.append((transpose_note(root, 1), "maj"))
        # Down a half step
        resolutions.append((transpose_note(root, -1), "maj"))

    return resolutions


# ============================================================================
# MELODIC DEVELOPMENT TECHNIQUES
# ============================================================================

def sequence_melody(
    melody_midi: list[int],
    interval: int,
    iterations: int = 2
) -> list[int]:
    """
    Create a melodic sequence by transposing a pattern.

    Common sequences: -2 (descending 2nd), -3 (descending 3rd), +4 (ascending 4th)
    """
    result = list(melody_midi)
    pattern = list(melody_midi)

    for i in range(iterations):
        pattern = [n + interval for n in pattern]
        result.extend(pattern)

    return result


def invert_melody(melody_midi: list[int], axis: Optional[int] = None) -> list[int]:
    """
    Invert a melody around an axis note.
    If no axis given, uses the first note.
    """
    if not melody_midi:
        return []

    if axis is None:
        axis = melody_midi[0]

    return [axis - (n - axis) for n in melody_midi]


def retrograde_melody(melody_midi: list[int]) -> list[int]:
    """Reverse a melody (play backwards)."""
    return list(reversed(melody_midi))


def augment_melody(melody_midi: list[int], factor: float = 2.0) -> list[tuple[int, float]]:
    """
    Augment melody rhythmically (return notes with relative durations).
    Factor > 1 = slower, Factor < 1 = faster
    """
    # Returns list of (midi_note, relative_duration)
    return [(n, factor) for n in melody_midi]


def diminish_melody(melody_midi: list[int], factor: float = 0.5) -> list[tuple[int, float]]:
    """
    Diminish melody rhythmically (faster).
    """
    return augment_melody(melody_midi, factor)


def embellish_melody(
    melody_midi: list[int],
    scale_notes: list[int],
    embellishment_type: str = "neighbor"
) -> list[int]:
    """
    Add embellishments to a melody.

    Types:
    - neighbor: add upper/lower neighbor tones
    - passing: add passing tones between leaps
    - turn: add turn ornaments
    """
    if not melody_midi:
        return []

    result = []
    scale_set = set(scale_notes)

    for i, note in enumerate(melody_midi):
        if embellishment_type == "neighbor" and i > 0:
            # Add upper neighbor then main note
            upper_neighbor = note + 1 if (note + 1) in scale_set else note + 2
            result.append(upper_neighbor)
            result.append(note)
        elif embellishment_type == "passing" and i < len(melody_midi) - 1:
            result.append(note)
            next_note = melody_midi[i + 1]
            leap = next_note - note
            if abs(leap) > 2:
                # Add passing tone(s)
                direction = 1 if leap > 0 else -1
                passing = note + direction * 2
                if passing in scale_set:
                    result.append(passing)
        elif embellishment_type == "turn":
            # Turn: upper neighbor, main, lower neighbor, main
            upper = note + 2
            lower = note - 2
            result.extend([upper, note, lower, note])
        else:
            result.append(note)

    return result


def create_motif_variation(
    motif_midi: list[int],
    variation_type: str,
    scale_root: int = 60,
    scale_type: str = "major"
) -> list[int]:
    """
    Create a variation of a melodic motif.

    Variation types:
    - transpose: move to different pitch level
    - invert: mirror around axis
    - retrograde: play backwards
    - augment_interval: stretch intervals
    - contract_interval: shrink intervals
    - change_octave: move octave
    """
    if not motif_midi:
        return []

    if variation_type == "transpose":
        # Transpose up a third
        return [n + 3 for n in motif_midi]

    elif variation_type == "invert":
        return invert_melody(motif_midi)

    elif variation_type == "retrograde":
        return retrograde_melody(motif_midi)

    elif variation_type == "augment_interval":
        # Double all intervals
        result = [motif_midi[0]]
        for i in range(1, len(motif_midi)):
            interval = motif_midi[i] - motif_midi[i-1]
            result.append(result[-1] + interval * 2)
        return result

    elif variation_type == "contract_interval":
        # Halve all intervals
        result = [motif_midi[0]]
        for i in range(1, len(motif_midi)):
            interval = motif_midi[i] - motif_midi[i-1]
            result.append(result[-1] + interval // 2)
        return result

    elif variation_type == "change_octave":
        return [n + 12 for n in motif_midi]

    return motif_midi


# ============================================================================
# HARMONIC RHYTHM
# ============================================================================

@dataclass
class HarmonicRhythm:
    """Defines how often chords change."""
    name: str
    changes_per_bar: float
    description: str
    genre_affinity: dict[str, float]


HARMONIC_RHYTHMS: dict[str, HarmonicRhythm] = {
    "very_slow": HarmonicRhythm(
        "very_slow", 0.25,
        "Chord changes every 4 bars",
        {"ambient": 1.0, "drone": 0.9, "cinematic": 0.7}
    ),
    "slow": HarmonicRhythm(
        "slow", 0.5,
        "Chord changes every 2 bars",
        {"ambient": 0.8, "r-and-b": 0.7, "neo-soul": 0.8, "trap": 0.7}
    ),
    "standard": HarmonicRhythm(
        "standard", 1.0,
        "Chord changes every bar",
        {"pop": 1.0, "rock": 0.9, "jazz": 0.6, "funk": 0.7}
    ),
    "moderate": HarmonicRhythm(
        "moderate", 2.0,
        "Chord changes twice per bar",
        {"jazz": 0.8, "pop": 0.7, "latin": 0.8}
    ),
    "fast": HarmonicRhythm(
        "fast", 4.0,
        "Chord changes on every beat",
        {"jazz": 0.9, "bebop": 1.0, "latin": 0.7}
    ),
    "very_fast": HarmonicRhythm(
        "very_fast", 8.0,
        "Chord changes on every 8th note",
        {"bebop": 1.0, "jazz": 0.5}
    ),
}


def get_harmonic_rhythm_for_genre(genre: str) -> HarmonicRhythm:
    """Get the most appropriate harmonic rhythm for a genre."""
    best_match = "standard"
    best_affinity = 0.0

    for name, hr in HARMONIC_RHYTHMS.items():
        affinity = hr.genre_affinity.get(genre, 0.0)
        if affinity > best_affinity:
            best_affinity = affinity
            best_match = name

    return HARMONIC_RHYTHMS[best_match]


# ============================================================================
# HELPER EXPORTS
# ============================================================================

def get_all_scale_names() -> list[str]:
    """Get all available scale names."""
    return list(SCALE_PATTERNS.keys())


def get_all_chord_names() -> list[str]:
    """Get all available chord type names."""
    return list(CHORD_PATTERNS.keys())


def get_scale_for_genre(genre: str, mode: str = "default") -> str:
    """
    Get a recommended scale for a genre.
    """
    genre_scales = {
        "jazz": ["dorian", "mixolydian", "bebop_dominant", "lydian"],
        "blues": ["blues", "minor_pentatonic", "mixolydian"],
        "rock": ["minor_pentatonic", "aeolian", "mixolydian"],
        "pop": ["major", "minor", "mixolydian"],
        "r-and-b": ["dorian", "mixolydian", "pentatonic_minor"],
        "funk": ["dorian", "mixolydian", "minor_pentatonic"],
        "trap": ["minor", "phrygian", "harmonic_minor"],
        "house": ["minor", "major", "dorian"],
        "techno": ["minor", "phrygian", "whole_tone"],
        "ambient": ["lydian", "whole_tone", "major"],
        "neo-soul": ["dorian", "lydian", "mixolydian"],
        "metal": ["phrygian", "harmonic_minor", "locrian"],
        "flamenco": ["phrygian_dominant", "harmonic_minor", "spanish_gypsy"],
        "latin": ["dorian", "mixolydian", "harmonic_minor"],
        "classical": ["major", "minor", "harmonic_minor"],
    }

    scales = genre_scales.get(genre, ["major", "minor"])
    return scales[0] if mode == "default" else scales


def get_chord_extensions_for_genre(genre: str) -> list[str]:
    """
    Get recommended chord extensions for a genre.
    """
    extensions = {
        "jazz": ["maj9", "min9", "13", "7#11", "7alt"],
        "neo-soul": ["maj9", "min11", "9", "6/9"],
        "r-and-b": ["maj7", "min7", "9", "add9"],
        "pop": ["maj", "min", "sus4", "add9"],
        "rock": ["5", "maj", "min", "sus4"],
        "funk": ["7", "9", "min7", "6"],
        "blues": ["7", "9", "13"],
        "trap": ["min", "min7", "5"],
        "house": ["min7", "maj7", "min9"],
        "ambient": ["maj9", "sus2", "add9", "quartal"],
    }

    return extensions.get(genre, ["maj", "min", "7"])
