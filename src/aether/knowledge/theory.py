"""
AETHER Music Theory Utilities

Comprehensive music theory library for harmonic analysis, scale construction,
chord voicing, and melodic operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math


# ============================================================================
# Constants
# ============================================================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC_MAP = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "B#": "C", "E#": "F", "Cb": "B",
}

# Intervals in semitones
INTERVALS = {
    "P1": 0, "m2": 1, "M2": 2, "m3": 3, "M3": 4, "P4": 5,
    "TT": 6, "d5": 6, "A4": 6, "P5": 7, "m6": 8, "M6": 9,
    "m7": 10, "M7": 11, "P8": 12, "m9": 13, "M9": 14,
    "m10": 15, "M10": 16, "P11": 17, "A11": 18, "P12": 19,
    "m13": 20, "M13": 21,
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


def midi_to_note(midi: int) -> Tuple[str, int]:
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


def get_scale(root: str, scale_type: str) -> List[str]:
    """Get notes in a scale."""
    root = normalize_note(root)
    if scale_type not in SCALE_PATTERNS:
        raise ValueError(f"Unknown scale type: {scale_type}")

    pattern = SCALE_PATTERNS[scale_type]
    root_idx = NOTE_NAMES.index(root)
    return [NOTE_NAMES[(root_idx + interval) % 12] for interval in pattern]


def get_chord_notes(root: str, chord_type: str) -> List[str]:
    """Get notes in a chord."""
    root = normalize_note(root)
    if chord_type not in CHORD_PATTERNS:
        raise ValueError(f"Unknown chord type: {chord_type}")

    pattern = CHORD_PATTERNS[chord_type]
    root_idx = NOTE_NAMES.index(root)
    return [NOTE_NAMES[(root_idx + interval) % 12] for interval in pattern]


def get_chord_midi(root: str, chord_type: str, octave: int = 4) -> List[int]:
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


def get_diatonic_chord(root: str, scale_type: str, degree: int) -> Tuple[str, str]:
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


def get_diatonic_progression(root: str, scale_type: str, degrees: List[int]) -> List[Tuple[str, str]]:
    """Get chords for a progression given in scale degrees."""
    return [get_diatonic_chord(root, scale_type, d) for d in degrees]


def roman_to_degree(roman: str) -> Tuple[int, str]:
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
            base_roman = roman[:-len(suffix)]
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
        "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
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


def parse_progression(root: str, scale_type: str, roman_numerals: str) -> List[Tuple[str, str]]:
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
    chord1_midi: List[int],
    chord2_midi: List[int],
) -> List[VoiceLeadingViolation]:
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
                    violations.append(VoiceLeadingViolation(
                        type="parallel_fifths",
                        description=f"Parallel fifths between voices {i} and {j}",
                        voice1=i,
                        voice2=j,
                    ))

            # Check for parallel octaves
            if interval1 == 0 and interval2 == 0:
                motion1 = c2[i] - c1[i]
                motion2 = c2[j] - c1[j]
                if motion1 != 0 and motion1 == motion2:
                    violations.append(VoiceLeadingViolation(
                        type="parallel_octaves",
                        description=f"Parallel octaves between voices {i} and {j}",
                        voice1=i,
                        voice2=j,
                    ))

    # Check for voice crossing
    for i in range(min_voices - 1):
        if c2[i] > c2[i + 1]:
            violations.append(VoiceLeadingViolation(
                type="voice_crossing",
                description=f"Voice {i} crosses above voice {i + 1}",
                voice1=i,
                voice2=i + 1,
                severity=0.5,
            ))

    # Check for large leaps (> octave)
    for i in range(min_voices):
        leap = abs(c2[i] - c1[i])
        if leap > 12:
            violations.append(VoiceLeadingViolation(
                type="large_leap",
                description=f"Voice {i} leaps more than an octave ({leap} semitones)",
                voice1=i,
                severity=0.7,
            ))

    return violations


def smooth_voice_leading_distance(
    chord1_midi: List[int],
    chord2_midi: List[int],
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


def get_interval_sequence(midi_notes: List[int]) -> List[int]:
    """Get sequence of intervals between consecutive notes."""
    return [midi_notes[i + 1] - midi_notes[i] for i in range(len(midi_notes) - 1)]


def get_contour(midi_notes: List[int]) -> List[str]:
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


def contour_to_hash(contour: List[str]) -> str:
    """Convert contour to a hash string for comparison."""
    mapping = {"up": "U", "down": "D", "same": "S"}
    return "".join(mapping.get(c, "?") for c in contour)


def analyze_melody_range(midi_notes: List[int]) -> Dict[str, int]:
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


def calculate_singability(midi_notes: List[int]) -> float:
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

    score = (stepwise_ratio * 0.5) + ((1 - large_interval_penalty) * 0.3) + ((1 - range_penalty) * 0.2)
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
