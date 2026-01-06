"""
Harmonic Grammar - Genre-specific chord progression generation.

Defines harmonic grammars for each genre that generate characteristic
chord progressions, ensuring genre authenticity through rule-based generation.
"""

from dataclasses import dataclass, field
from typing import Optional
import random

from aether.genre.dna import (
    GenreDNA,
    VoicingStyle,
    TensionProfile,
    get_genre_dna,
)


@dataclass
class Chord:
    """A chord with root, quality, and extensions."""
    root: int  # Scale degree (0=I, 1=II, etc.) or semitone offset
    quality: str  # major, minor, dim, aug, sus2, sus4
    extensions: list[str] = field(default_factory=list)  # 7, 9, 11, 13
    bass: Optional[int] = None  # Slash chord bass note

    def to_symbol(self, key: str = "C") -> str:
        """Convert to chord symbol (e.g., Am7, Cmaj9)."""
        # Roman numeral or note name
        roots = ["I", "II", "III", "IV", "V", "VI", "VII"]
        root_str = roots[self.root % 7]

        if self.quality == "minor":
            root_str = root_str.lower()
        elif self.quality == "dim":
            root_str = root_str.lower() + "°"
        elif self.quality == "aug":
            root_str += "+"

        ext_str = "".join(self.extensions)

        return root_str + ext_str

    def to_pitches(
        self,
        root_pitch: int = 60,
        scale: list[int] = None,
    ) -> list[int]:
        """Convert to MIDI pitches."""
        if scale is None:
            scale = [0, 2, 4, 5, 7, 9, 11]  # Major scale

        # Get root pitch in scale
        root = root_pitch + scale[self.root % len(scale)]

        # Build chord tones
        pitches = [root]

        # Quality determines 3rd
        if self.quality == "major":
            pitches.append(root + 4)  # Major 3rd
            pitches.append(root + 7)  # Perfect 5th
        elif self.quality == "minor":
            pitches.append(root + 3)  # Minor 3rd
            pitches.append(root + 7)  # Perfect 5th
        elif self.quality == "dim":
            pitches.append(root + 3)  # Minor 3rd
            pitches.append(root + 6)  # Diminished 5th
        elif self.quality == "aug":
            pitches.append(root + 4)  # Major 3rd
            pitches.append(root + 8)  # Augmented 5th
        elif self.quality == "sus2":
            pitches.append(root + 2)  # Major 2nd
            pitches.append(root + 7)  # Perfect 5th
        elif self.quality == "sus4":
            pitches.append(root + 5)  # Perfect 4th
            pitches.append(root + 7)  # Perfect 5th

        # Extensions
        for ext in self.extensions:
            if "7" in ext:
                if "maj" in ext.lower():
                    pitches.append(root + 11)  # Major 7th
                else:
                    pitches.append(root + 10)  # Minor 7th
            if "9" in ext:
                pitches.append(root + 14)  # 9th
            if "11" in ext:
                pitches.append(root + 17)  # 11th
            if "13" in ext:
                pitches.append(root + 21)  # 13th

        return sorted(set(pitches))


@dataclass
class ChordProgression:
    """A sequence of chords with timing."""
    chords: list[tuple[Chord, float]]  # (chord, duration_beats)
    key_root: int = 0  # Key root as semitone (0=C)
    mode: str = "major"  # major, minor, dorian, etc.

    def total_beats(self) -> float:
        """Total duration in beats."""
        return sum(dur for _, dur in self.chords)

    def to_midi_notes(
        self,
        root_pitch: int = 48,
        ticks_per_beat: int = 480,
        velocity: int = 80,
    ) -> list[dict]:
        """Convert progression to MIDI notes."""
        # Get scale for mode
        mode_scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "phrygian": [0, 1, 3, 5, 7, 8, 10],
            "lydian": [0, 2, 4, 6, 7, 9, 11],
            "mixolydian": [0, 2, 4, 5, 7, 9, 10],
            "locrian": [0, 1, 3, 5, 6, 8, 10],
        }
        scale = mode_scales.get(self.mode, mode_scales["minor"])

        notes = []
        current_tick = 0
        actual_root = root_pitch + self.key_root

        for chord, duration in self.chords:
            pitches = chord.to_pitches(actual_root, scale)
            duration_ticks = int(duration * ticks_per_beat)

            for pitch in pitches:
                notes.append({
                    "pitch": pitch,
                    "start_tick": current_tick,
                    "duration": duration_ticks - 10,  # Slight gap
                    "velocity": velocity,
                    "channel": 0,
                })

            current_tick += duration_ticks

        return notes


@dataclass
class HarmonicGrammar:
    """
    Grammar rules for generating chord progressions in a genre.

    Each genre has characteristic progressions, voicing styles,
    and harmonic rhythm patterns.
    """
    # Common progressions (as chord symbol patterns)
    primary_progressions: list[str]  # Most characteristic
    secondary_progressions: list[str]  # Also acceptable
    avoid_progressions: list[str]  # Genre-inappropriate

    # Chord vocabulary
    primary_mode: str  # major, minor, dorian, etc.
    chord_qualities: list[str]  # Allowed chord qualities
    extension_probability: float  # 0-1, likelihood of extensions
    allowed_extensions: list[str]

    # Harmonic rhythm
    typical_chord_durations: list[float]  # In beats
    changes_per_bar: float  # Average chord changes per bar

    # Voicing
    voicing_style: VoicingStyle
    register: tuple[int, int]  # MIDI pitch range

    # Tension
    tension_profile: TensionProfile
    chromaticism_level: float  # 0-1


# Genre-specific harmonic grammars
GENRE_HARMONIC_GRAMMAR: dict[str, HarmonicGrammar] = {
    "lofi-hip-hop": HarmonicGrammar(
        primary_progressions=[
            "i-VI-III-VII",
            "i-iv-VII-III",
            "ii-V-I-vi",
            "i-V-VI-IV",
        ],
        secondary_progressions=[
            "I-vi-ii-V",
            "i-VII-VI-V",
        ],
        avoid_progressions=["I-IV-V-I"],
        primary_mode="minor",
        chord_qualities=["minor", "major", "minor7", "maj7"],
        extension_probability=0.7,
        allowed_extensions=["7", "9", "maj7"],
        typical_chord_durations=[2.0, 4.0],
        changes_per_bar=0.5,
        voicing_style=VoicingStyle.CLOSE,
        register=(48, 72),
        tension_profile=TensionProfile.GENTLE_SUSPEND,
        chromaticism_level=0.2,
    ),
    "trap": HarmonicGrammar(
        primary_progressions=[
            "i-VI-III-VII",
            "i-iv-i-VII",
            "i-i-iv-iv",
        ],
        secondary_progressions=[
            "i-VII-VI-VII",
            "i-III-VII-VI",
        ],
        avoid_progressions=["I-IV-V-I", "ii-V-I"],
        primary_mode="minor",
        chord_qualities=["minor", "major"],
        extension_probability=0.3,
        allowed_extensions=["7"],
        typical_chord_durations=[4.0, 8.0],
        changes_per_bar=0.25,
        voicing_style=VoicingStyle.SHELL,
        register=(36, 60),
        tension_profile=TensionProfile.DARK_UNRESOLVED,
        chromaticism_level=0.1,
    ),
    "drill": HarmonicGrammar(
        primary_progressions=[
            "i-i-i-i",
            "i-VII-i-VII",
            "i-VI-VII-i",
        ],
        secondary_progressions=[
            "i-iv-VII-i",
        ],
        avoid_progressions=["I-IV-V-I", "ii-V-I"],
        primary_mode="phrygian",
        chord_qualities=["minor", "major"],
        extension_probability=0.1,
        allowed_extensions=["7"],
        typical_chord_durations=[4.0, 8.0],
        changes_per_bar=0.25,
        voicing_style=VoicingStyle.SHELL,
        register=(36, 60),
        tension_profile=TensionProfile.DARK_UNRESOLVED,
        chromaticism_level=0.15,
    ),
    "boom-bap": HarmonicGrammar(
        primary_progressions=[
            "i-IV-i-IV",
            "i-VI-III-VII",
            "ii-V-I-vi",
        ],
        secondary_progressions=[
            "i-iv-V-i",
            "I-vi-IV-V",
        ],
        avoid_progressions=[],
        primary_mode="minor",
        chord_qualities=["minor", "major", "minor7", "dom7"],
        extension_probability=0.5,
        allowed_extensions=["7", "9"],
        typical_chord_durations=[2.0, 4.0],
        changes_per_bar=0.5,
        voicing_style=VoicingStyle.CLOSE,
        register=(48, 72),
        tension_profile=TensionProfile.RESOLVE_QUICK,
        chromaticism_level=0.25,
    ),
    "synthwave": HarmonicGrammar(
        primary_progressions=[
            "I-V-vi-IV",
            "vi-IV-I-V",
            "I-vi-IV-V",
        ],
        secondary_progressions=[
            "I-III-IV-iv",
            "i-VII-VI-V",
        ],
        avoid_progressions=[],
        primary_mode="major",
        chord_qualities=["major", "minor", "sus4"],
        extension_probability=0.4,
        allowed_extensions=["7", "9", "add9"],
        typical_chord_durations=[4.0, 8.0],
        changes_per_bar=0.25,
        voicing_style=VoicingStyle.SPREAD,
        register=(48, 84),
        tension_profile=TensionProfile.SUSPEND,
        chromaticism_level=0.15,
    ),
    "house": HarmonicGrammar(
        primary_progressions=[
            "i-iv-i-iv",
            "I-vi-ii-V",
            "i-VII-VI-VII",
        ],
        secondary_progressions=[
            "I-V-vi-IV",
            "ii-V-I-vi",
        ],
        avoid_progressions=[],
        primary_mode="minor",
        chord_qualities=["minor", "major", "minor7", "dom7"],
        extension_probability=0.6,
        allowed_extensions=["7", "9"],
        typical_chord_durations=[4.0, 8.0],
        changes_per_bar=0.25,
        voicing_style=VoicingStyle.SPREAD,
        register=(48, 72),
        tension_profile=TensionProfile.RESOLVE_QUICK,
        chromaticism_level=0.2,
    ),
    "techno": HarmonicGrammar(
        primary_progressions=[
            "i-i-i-i",
            "i-VII-i-VII",
        ],
        secondary_progressions=[
            "i-iv-i-iv",
        ],
        avoid_progressions=["I-IV-V-I"],
        primary_mode="minor",
        chord_qualities=["minor", "sus4", "power"],
        extension_probability=0.2,
        allowed_extensions=["7"],
        typical_chord_durations=[8.0, 16.0],
        changes_per_bar=0.125,
        voicing_style=VoicingStyle.SHELL,
        register=(36, 60),
        tension_profile=TensionProfile.SUSPEND,
        chromaticism_level=0.1,
    ),
    "drum-and-bass": HarmonicGrammar(
        primary_progressions=[
            "i-VI-III-VII",
            "i-iv-VII-III",
            "i-VII-i-VII",
        ],
        secondary_progressions=[
            "i-V-VI-IV",
            "vi-IV-I-V",
        ],
        avoid_progressions=[],
        primary_mode="minor",
        chord_qualities=["minor", "major", "minor7"],
        extension_probability=0.5,
        allowed_extensions=["7", "9"],
        typical_chord_durations=[4.0, 8.0],
        changes_per_bar=0.25,
        voicing_style=VoicingStyle.SPREAD,
        register=(48, 72),
        tension_profile=TensionProfile.RESOLVE_QUICK,
        chromaticism_level=0.15,
    ),
    "reggaeton": HarmonicGrammar(
        primary_progressions=[
            "i-iv-VII-III",
            "i-VII-VI-VII",
            "i-iv-i-VII",
        ],
        secondary_progressions=[
            "I-V-vi-IV",
            "i-VI-III-VII",
        ],
        avoid_progressions=[],
        primary_mode="minor",
        chord_qualities=["minor", "major"],
        extension_probability=0.3,
        allowed_extensions=["7"],
        typical_chord_durations=[4.0],
        changes_per_bar=0.25,
        voicing_style=VoicingStyle.CLOSE,
        register=(48, 72),
        tension_profile=TensionProfile.RESOLVE_QUICK,
        chromaticism_level=0.1,
    ),
    "afrobeat": HarmonicGrammar(
        primary_progressions=[
            "I-IV-I-IV",
            "i-iv-VII-III",
            "I-V-vi-IV",
        ],
        secondary_progressions=[
            "ii-V-I-vi",
            "I-vi-ii-V",
        ],
        avoid_progressions=[],
        primary_mode="mixolydian",
        chord_qualities=["major", "minor", "dom7", "minor7"],
        extension_probability=0.6,
        allowed_extensions=["7", "9", "11"],
        typical_chord_durations=[2.0, 4.0],
        changes_per_bar=0.5,
        voicing_style=VoicingStyle.SPREAD,
        register=(48, 72),
        tension_profile=TensionProfile.RESOLVE_QUICK,
        chromaticism_level=0.25,
    ),
    "pop": HarmonicGrammar(
        primary_progressions=[
            "I-V-vi-IV",
            "I-vi-IV-V",
            "vi-IV-I-V",
        ],
        secondary_progressions=[
            "I-IV-vi-V",
            "I-V-IV-V",
        ],
        avoid_progressions=[],
        primary_mode="major",
        chord_qualities=["major", "minor"],
        extension_probability=0.3,
        allowed_extensions=["7", "add9"],
        typical_chord_durations=[2.0, 4.0],
        changes_per_bar=0.5,
        voicing_style=VoicingStyle.CLOSE,
        register=(48, 72),
        tension_profile=TensionProfile.RESOLVE_QUICK,
        chromaticism_level=0.1,
    ),
    "cinematic": HarmonicGrammar(
        primary_progressions=[
            "I-V-vi-IV",
            "i-VI-III-VII",
            "I-III-IV-iv",
        ],
        secondary_progressions=[
            "i-VII-VI-V",
            "I-vi-ii-V",
            "i-iv-i-V",
        ],
        avoid_progressions=[],
        primary_mode="minor",
        chord_qualities=["major", "minor", "dim", "aug", "sus4"],
        extension_probability=0.7,
        allowed_extensions=["7", "maj7", "9", "11", "13"],
        typical_chord_durations=[4.0, 8.0, 16.0],
        changes_per_bar=0.25,
        voicing_style=VoicingStyle.SPREAD,
        register=(36, 96),
        tension_profile=TensionProfile.SUSPEND,
        chromaticism_level=0.35,
    ),
}


class HarmonyGenerator:
    """Generates chord progressions according to genre grammar."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_progression(
        self,
        genre_id: str,
        num_bars: int = 4,
        key_root: int = 0,
    ) -> ChordProgression:
        """
        Generate a chord progression for the genre.

        Args:
            genre_id: Target genre
            num_bars: Number of bars
            key_root: Key root as semitone (0=C, 2=D, etc.)

        Returns:
            ChordProgression
        """
        grammar = GENRE_HARMONIC_GRAMMAR.get(genre_id)
        if grammar is None:
            grammar = GENRE_HARMONIC_GRAMMAR["pop"]

        # Select a progression pattern
        if self.rng.random() < 0.7:
            pattern = self.rng.choice(grammar.primary_progressions)
        else:
            if grammar.secondary_progressions:
                pattern = self.rng.choice(grammar.secondary_progressions)
            else:
                pattern = self.rng.choice(grammar.primary_progressions)

        # Parse pattern into chords
        chords = self._parse_progression(pattern, grammar)

        # Extend or repeat to fill bars
        beats_needed = num_bars * 4
        chords = self._fit_to_length(chords, beats_needed, grammar)

        return ChordProgression(
            chords=chords,
            key_root=key_root,
            mode=grammar.primary_mode,
        )

    def _parse_progression(
        self,
        pattern: str,
        grammar: HarmonicGrammar,
    ) -> list[tuple[Chord, float]]:
        """Parse a progression pattern string into chords."""
        chords = []
        symbols = pattern.split("-")

        for symbol in symbols:
            chord = self._parse_chord_symbol(symbol, grammar)
            duration = self.rng.choice(grammar.typical_chord_durations)
            chords.append((chord, duration))

        return chords

    def _parse_chord_symbol(
        self,
        symbol: str,
        grammar: HarmonicGrammar,
    ) -> Chord:
        """Parse a chord symbol (e.g., 'i', 'IV', 'VII') into a Chord."""
        # Determine root and quality from Roman numeral
        symbol = symbol.strip()

        # Check for lowercase (minor) or uppercase (major)
        is_minor = symbol[0].islower()

        # Extract Roman numeral
        numeral_map = {
            "i": 0, "I": 0,
            "ii": 1, "II": 1,
            "iii": 2, "III": 2,
            "iv": 3, "IV": 3,
            "v": 4, "V": 4,
            "vi": 5, "VI": 5,
            "vii": 6, "VII": 6,
        }

        # Find the numeral
        root = 0
        for num, deg in numeral_map.items():
            if symbol.upper().startswith(num.upper()):
                root = deg
                break

        # Determine quality
        if is_minor:
            quality = "minor"
        else:
            quality = "major"

        # Add extensions probabilistically
        extensions = []
        if self.rng.random() < grammar.extension_probability:
            ext = self.rng.choice(grammar.allowed_extensions)
            extensions.append(ext)

        return Chord(
            root=root,
            quality=quality,
            extensions=extensions,
        )

    def _fit_to_length(
        self,
        chords: list[tuple[Chord, float]],
        target_beats: float,
        grammar: HarmonicGrammar,
    ) -> list[tuple[Chord, float]]:
        """Adjust progression to fit target length."""
        current_beats = sum(dur for _, dur in chords)

        if current_beats >= target_beats:
            # Trim to fit
            result = []
            remaining = target_beats
            for chord, dur in chords:
                if remaining <= 0:
                    break
                actual_dur = min(dur, remaining)
                result.append((chord, actual_dur))
                remaining -= actual_dur
            return result

        # Repeat to fill
        result = []
        remaining = target_beats
        chord_idx = 0

        while remaining > 0:
            chord, dur = chords[chord_idx % len(chords)]
            actual_dur = min(dur, remaining)
            result.append((chord, actual_dur))
            remaining -= actual_dur
            chord_idx += 1

        return result

    def generate_modulation(
        self,
        from_key: int,
        target_key: int,
        num_chords: int = 2,
    ) -> list[tuple[Chord, float]]:
        """Generate a modulation passage between keys."""
        # Simple pivot chord modulation
        chords = []

        # Pivot chord (works in both keys)
        pivot = Chord(root=4, quality="major", extensions=["7"])  # V7
        chords.append((pivot, 2.0))

        # Target dominant
        dom = Chord(root=4, quality="major", extensions=["7"])
        chords.append((dom, 2.0))

        return chords


def generate_chord_progression(
    genre_id: str,
    num_bars: int = 4,
    key_root: int = 0,
    seed: Optional[int] = None,
) -> ChordProgression:
    """
    Generate a chord progression for a genre.

    Args:
        genre_id: Target genre
        num_bars: Number of bars
        key_root: Key root (0=C, 2=D, etc.)
        seed: Random seed

    Returns:
        ChordProgression
    """
    generator = HarmonyGenerator(seed)
    return generator.generate_progression(genre_id, num_bars, key_root)


def get_harmonic_grammar(genre_id: str) -> Optional[HarmonicGrammar]:
    """Get harmonic grammar for a genre."""
    return GENRE_HARMONIC_GRAMMAR.get(genre_id)
