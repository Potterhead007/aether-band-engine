"""
Genre-Specific Pattern Libraries for Professional Music Generation

Each genre has unique:
- Drum patterns (multiple variations)
- Bass patterns (rhythmic and melodic)
- Chord voicing styles
- Melody characteristics
- Velocity/dynamics profiles
- Groove/swing templates
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class DrumHit:
    """Single drum hit with timing and velocity."""
    beat: float  # Beat position (0-based within bar)
    drum: str    # Drum name (kick, snare, hihat_closed, etc.)
    velocity: int  # 0-127


@dataclass
class DrumPattern:
    """Complete drum pattern for a bar."""
    name: str
    hits: List[DrumHit]
    time_signature: Tuple[int, int] = (4, 4)


@dataclass
class BassPattern:
    """Bass pattern with note positions relative to chord root."""
    name: str
    # List of (beat_offset, interval_from_root, duration, velocity_range)
    notes: List[Tuple[float, int, float, Tuple[int, int]]]


@dataclass
class ChordStyle:
    """Chord voicing and rhythm style."""
    name: str
    voicing: str  # "close", "spread", "power", "shell", "extended"
    rhythm: List[Tuple[float, float]]  # (beat_offset, duration)
    velocity_range: Tuple[int, int]
    use_inversions: bool = True


@dataclass
class MelodyStyle:
    """Melody generation characteristics."""
    note_density: float  # 0-1, how many notes per beat
    preferred_durations: List[float]
    duration_weights: List[float]
    rest_probability: float
    leap_probability: float  # Large interval jumps
    max_leap: int  # Semitones
    rhythmic_patterns: List[List[float]]  # Common rhythmic figures
    articulation: str  # "legato", "staccato", "normal"
    velocity_range: Tuple[int, int]
    octave_range: Tuple[int, int]  # Relative to base octave


@dataclass
class GrooveTemplate:
    """Timing and feel adjustments."""
    swing_amount: float  # 0-1
    push_pull: Dict[float, float]  # beat -> timing offset in beats
    velocity_accents: Dict[float, float]  # beat -> velocity multiplier
    humanize_timing: float  # Random timing variation
    humanize_velocity: float  # Random velocity variation


@dataclass
class GenreTemplate:
    """Complete generation template for a genre."""
    genre_id: str
    drum_patterns: List[DrumPattern]
    fill_patterns: List[DrumPattern]
    bass_patterns: List[BassPattern]
    chord_style: ChordStyle
    melody_style: MelodyStyle
    groove: GrooveTemplate
    # Section-specific energy modifiers
    section_dynamics: Dict[str, float]  # section_type -> energy multiplier


# ============================================================================
# Drum Patterns by Genre
# ============================================================================

def _create_house_drums() -> List[DrumPattern]:
    """Classic house: 4-on-floor with offbeat hats."""
    patterns = []

    # Main pattern
    main = DrumPattern(
        name="house_main",
        hits=[
            # Kick: 4-on-floor
            DrumHit(0.0, "kick", 110),
            DrumHit(1.0, "kick", 105),
            DrumHit(2.0, "kick", 110),
            DrumHit(3.0, "kick", 105),
            # Clap on 2 and 4
            DrumHit(1.0, "clap", 95),
            DrumHit(3.0, "clap", 100),
            # Offbeat open hats
            DrumHit(0.5, "hihat_open", 70),
            DrumHit(1.5, "hihat_open", 65),
            DrumHit(2.5, "hihat_open", 70),
            DrumHit(3.5, "hihat_open", 65),
            # Closed hats on beats
            DrumHit(0.0, "hihat_closed", 50),
            DrumHit(1.0, "hihat_closed", 45),
            DrumHit(2.0, "hihat_closed", 50),
            DrumHit(3.0, "hihat_closed", 45),
        ]
    )
    patterns.append(main)

    # Variation with percussion
    variation = DrumPattern(
        name="house_perc",
        hits=main.hits + [
            DrumHit(0.75, "shaker", 40),
            DrumHit(1.75, "shaker", 35),
            DrumHit(2.75, "shaker", 40),
            DrumHit(3.75, "shaker", 35),
            DrumHit(2.5, "conga_high", 55),
        ]
    )
    patterns.append(variation)

    return patterns


def _create_techno_drums() -> List[DrumPattern]:
    """Industrial techno: heavy kick, sparse but powerful."""
    patterns = []

    # Driving pattern
    main = DrumPattern(
        name="techno_main",
        hits=[
            # Heavy kicks
            DrumHit(0.0, "kick", 120),
            DrumHit(1.0, "kick", 115),
            DrumHit(2.0, "kick", 120),
            DrumHit(3.0, "kick", 115),
            # Clap/snare on 2 and 4
            DrumHit(1.0, "clap", 90),
            DrumHit(3.0, "clap", 95),
            # Relentless 16th hats
            *[DrumHit(i * 0.25, "hihat_closed", 50 + (10 if i % 4 == 0 else 0))
              for i in range(16)],
        ]
    )
    patterns.append(main)

    # Minimal pattern
    minimal = DrumPattern(
        name="techno_minimal",
        hits=[
            DrumHit(0.0, "kick", 120),
            DrumHit(1.0, "kick", 110),
            DrumHit(2.0, "kick", 120),
            DrumHit(3.0, "kick", 110),
            DrumHit(1.0, "rim", 80),
            DrumHit(3.0, "rim", 85),
            *[DrumHit(i * 0.5 + 0.25, "hihat_closed", 45) for i in range(8)],
        ]
    )
    patterns.append(minimal)

    return patterns


def _create_jazz_drums() -> List[DrumPattern]:
    """Jazz swing: ride pattern, kick/snare comping."""
    patterns = []

    # Swing ride pattern
    main = DrumPattern(
        name="jazz_swing",
        hits=[
            # Ride pattern (swing 8ths)
            DrumHit(0.0, "ride", 75),
            DrumHit(0.67, "ride", 55),  # Swung
            DrumHit(1.0, "ride", 70),
            DrumHit(1.67, "ride", 55),
            DrumHit(2.0, "ride", 75),
            DrumHit(2.67, "ride", 55),
            DrumHit(3.0, "ride", 70),
            DrumHit(3.67, "ride", 55),
            # Hi-hat on 2 and 4
            DrumHit(1.0, "hihat_pedal", 65),
            DrumHit(3.0, "hihat_pedal", 65),
            # Ghost notes on snare
            DrumHit(1.33, "snare", 35),  # Ghost
            DrumHit(2.67, "snare", 30),  # Ghost
            # Light kick accents
            DrumHit(0.0, "kick", 60),
            DrumHit(2.5, "kick", 50),
        ]
    )
    patterns.append(main)

    # Brushes pattern
    brushes = DrumPattern(
        name="jazz_brushes",
        hits=[
            DrumHit(0.0, "ride", 55),
            DrumHit(1.0, "ride", 50),
            DrumHit(2.0, "ride", 55),
            DrumHit(3.0, "ride", 50),
            DrumHit(1.0, "sidestick", 45),
            DrumHit(3.0, "sidestick", 50),
            DrumHit(0.0, "kick", 45),
            DrumHit(2.0, "kick", 40),
        ]
    )
    patterns.append(brushes)

    return patterns


def _create_hiphop_drums() -> List[DrumPattern]:
    """Boom bap: hard kicks and snares, vinyl feel."""
    patterns = []

    main = DrumPattern(
        name="boombap_main",
        hits=[
            DrumHit(0.0, "kick", 115),
            DrumHit(0.75, "kick", 100),  # Syncopated
            DrumHit(2.25, "kick", 105),
            DrumHit(1.0, "snare", 110),
            DrumHit(3.0, "snare", 115),
            # Hats
            *[DrumHit(i * 0.5, "hihat_closed", 60 + random.randint(-5, 5))
              for i in range(8)],
            DrumHit(2.5, "hihat_open", 55),
        ]
    )
    patterns.append(main)

    # Variation
    var = DrumPattern(
        name="boombap_var",
        hits=[
            DrumHit(0.0, "kick", 115),
            DrumHit(1.5, "kick", 95),
            DrumHit(2.5, "kick", 100),
            DrumHit(1.0, "snare", 110),
            DrumHit(2.75, "snare", 85),  # Ghost
            DrumHit(3.0, "snare", 115),
            *[DrumHit(i * 0.5, "hihat_closed", 55) for i in range(8)],
        ]
    )
    patterns.append(var)

    return patterns


def _create_trap_drums() -> List[DrumPattern]:
    """Trap: 808 kicks, hi-hat rolls, sparse but hard."""
    patterns = []

    main = DrumPattern(
        name="trap_main",
        hits=[
            # 808 kicks (half-time feel)
            DrumHit(0.0, "kick", 127),
            DrumHit(2.75, "kick", 120),
            # Snare on 3 (half-time)
            DrumHit(2.0, "snare", 120),
            # Hi-hat pattern with rolls
            DrumHit(0.0, "hihat_closed", 70),
            DrumHit(0.5, "hihat_closed", 65),
            DrumHit(1.0, "hihat_closed", 75),
            # Roll before beat 2
            DrumHit(1.25, "hihat_closed", 50),
            DrumHit(1.375, "hihat_closed", 55),
            DrumHit(1.5, "hihat_closed", 60),
            DrumHit(1.625, "hihat_closed", 65),
            DrumHit(1.75, "hihat_closed", 70),
            DrumHit(1.875, "hihat_closed", 75),
            DrumHit(2.0, "hihat_open", 80),
            DrumHit(2.5, "hihat_closed", 60),
            DrumHit(3.0, "hihat_closed", 70),
            # Another roll
            DrumHit(3.25, "hihat_closed", 55),
            DrumHit(3.375, "hihat_closed", 60),
            DrumHit(3.5, "hihat_closed", 65),
            DrumHit(3.625, "hihat_closed", 70),
            DrumHit(3.75, "hihat_closed", 75),
            DrumHit(3.875, "hihat_closed", 80),
        ]
    )
    patterns.append(main)

    return patterns


def _create_rock_drums() -> List[DrumPattern]:
    """Rock: driving backbeat, fills."""
    patterns = []

    main = DrumPattern(
        name="rock_main",
        hits=[
            DrumHit(0.0, "kick", 110),
            DrumHit(2.0, "kick", 110),
            DrumHit(1.0, "snare", 115),
            DrumHit(3.0, "snare", 115),
            *[DrumHit(i * 0.5, "hihat_closed", 70) for i in range(8)],
        ]
    )
    patterns.append(main)

    # Driving 8ths variation
    driving = DrumPattern(
        name="rock_driving",
        hits=[
            DrumHit(0.0, "kick", 110),
            DrumHit(1.0, "kick", 100),
            DrumHit(2.0, "kick", 110),
            DrumHit(2.5, "kick", 95),
            DrumHit(1.0, "snare", 115),
            DrumHit(3.0, "snare", 115),
            *[DrumHit(i * 0.5, "hihat_closed", 75) for i in range(8)],
            DrumHit(3.5, "crash", 90),
        ]
    )
    patterns.append(driving)

    return patterns


def _create_funk_drums() -> List[DrumPattern]:
    """Funk: syncopated, ghost notes, emphasis on the ONE."""
    patterns = []

    main = DrumPattern(
        name="funk_main",
        hits=[
            # THE ONE (emphasized)
            DrumHit(0.0, "kick", 120),
            DrumHit(0.0, "crash", 70),
            # Syncopated kicks
            DrumHit(0.75, "kick", 90),
            DrumHit(2.5, "kick", 100),
            DrumHit(3.25, "kick", 85),
            # Backbeat with ghosts
            DrumHit(1.0, "snare", 110),
            DrumHit(1.5, "snare", 40),  # Ghost
            DrumHit(2.25, "snare", 35),  # Ghost
            DrumHit(3.0, "snare", 110),
            DrumHit(3.5, "snare", 45),  # Ghost
            # 16th hats with accents
            *[DrumHit(i * 0.25, "hihat_closed", 55 + (20 if i % 4 == 0 else (10 if i % 2 == 0 else 0)))
              for i in range(16)],
            DrumHit(2.5, "hihat_open", 65),
        ]
    )
    patterns.append(main)

    return patterns


def _create_disco_drums() -> List[DrumPattern]:
    """Disco: 4-on-floor, open hats, orchestral feel."""
    patterns = []

    main = DrumPattern(
        name="disco_main",
        hits=[
            # Four on the floor
            DrumHit(0.0, "kick", 105),
            DrumHit(1.0, "kick", 100),
            DrumHit(2.0, "kick", 105),
            DrumHit(3.0, "kick", 100),
            # Clap/snare with layering
            DrumHit(1.0, "clap", 90),
            DrumHit(1.0, "snare", 80),
            DrumHit(3.0, "clap", 95),
            DrumHit(3.0, "snare", 85),
            # Sizzling open hats
            DrumHit(0.5, "hihat_open", 80),
            DrumHit(1.5, "hihat_open", 75),
            DrumHit(2.5, "hihat_open", 80),
            DrumHit(3.5, "hihat_open", 75),
            # Closed hats on beats
            DrumHit(0.0, "hihat_closed", 60),
            DrumHit(1.0, "hihat_closed", 55),
            DrumHit(2.0, "hihat_closed", 60),
            DrumHit(3.0, "hihat_closed", 55),
            # Tambourine
            *[DrumHit(i * 0.5, "tambourine", 40) for i in range(8)],
        ]
    )
    patterns.append(main)

    return patterns


def _create_dnb_drums() -> List[DrumPattern]:
    """Drum and Bass: fast breakbeats, syncopated."""
    patterns = []

    # Two-step pattern
    main = DrumPattern(
        name="dnb_twostep",
        hits=[
            DrumHit(0.0, "kick", 115),
            DrumHit(1.5, "kick", 110),
            DrumHit(0.5, "snare", 105),
            DrumHit(1.0, "snare", 115),
            DrumHit(2.0, "snare", 110),
            DrumHit(2.5, "snare", 100),
            DrumHit(3.0, "kick", 105),
            DrumHit(3.5, "snare", 95),
            # Fast hats
            *[DrumHit(i * 0.25, "hihat_closed", 50 + (15 if i % 2 == 0 else 0))
              for i in range(16)],
        ]
    )
    patterns.append(main)

    return patterns


def _create_dubstep_drums() -> List[DrumPattern]:
    """Dubstep: half-time, heavy, spacious."""
    patterns = []

    main = DrumPattern(
        name="dubstep_main",
        hits=[
            # Heavy kick on 1
            DrumHit(0.0, "kick", 127),
            # Snare on 3 (half-time)
            DrumHit(2.0, "snare", 120),
            DrumHit(2.0, "clap", 100),
            # Sparse hats
            DrumHit(0.5, "hihat_closed", 60),
            DrumHit(1.0, "hihat_closed", 65),
            DrumHit(1.5, "hihat_closed", 55),
            DrumHit(2.5, "hihat_open", 70),
            DrumHit(3.0, "hihat_closed", 60),
            DrumHit(3.5, "hihat_closed", 55),
        ]
    )
    patterns.append(main)

    return patterns


def _create_lofi_drums() -> List[DrumPattern]:
    """Lo-fi: relaxed, swung, dusty feel."""
    patterns = []

    main = DrumPattern(
        name="lofi_main",
        hits=[
            DrumHit(0.0, "kick", 85),
            DrumHit(1.75, "kick", 75),
            DrumHit(2.5, "kick", 80),
            DrumHit(1.0, "snare", 75),
            DrumHit(3.0, "snare", 80),
            # Swung hats (lazy feel)
            DrumHit(0.33, "hihat_closed", 45),
            DrumHit(0.67, "hihat_closed", 40),
            DrumHit(1.33, "hihat_closed", 45),
            DrumHit(1.67, "hihat_closed", 40),
            DrumHit(2.33, "hihat_closed", 45),
            DrumHit(2.67, "hihat_closed", 40),
            DrumHit(3.33, "hihat_closed", 45),
            DrumHit(3.67, "hihat_closed", 40),
        ]
    )
    patterns.append(main)

    return patterns


def _create_folk_drums() -> List[DrumPattern]:
    """Folk: minimal or no drums, light percussion."""
    patterns = []

    main = DrumPattern(
        name="folk_minimal",
        hits=[
            DrumHit(0.0, "kick", 55),
            DrumHit(2.0, "kick", 50),
            DrumHit(1.0, "tambourine", 45),
            DrumHit(3.0, "tambourine", 50),
            DrumHit(2.0, "shaker", 35),
        ]
    )
    patterns.append(main)

    # No drums option
    silent = DrumPattern(name="folk_none", hits=[])
    patterns.append(silent)

    return patterns


def _create_cinematic_drums() -> List[DrumPattern]:
    """Cinematic: epic, orchestral percussion."""
    patterns = []

    main = DrumPattern(
        name="cinematic_epic",
        hits=[
            DrumHit(0.0, "kick", 120),  # Use as taiko
            DrumHit(0.0, "crash", 100),
            DrumHit(2.0, "kick", 115),
            DrumHit(2.0, "snare", 100),  # Use as orchestral snare
            DrumHit(3.0, "kick", 90),
            DrumHit(3.5, "kick", 95),
        ]
    )
    patterns.append(main)

    # Tension build
    tension = DrumPattern(
        name="cinematic_tension",
        hits=[
            *[DrumHit(i * 0.5, "snare", 40 + i * 5) for i in range(8)],  # Snare roll crescendo
        ]
    )
    patterns.append(tension)

    return patterns


def _create_ambient_drums() -> List[DrumPattern]:
    """Ambient: minimal or none."""
    return [DrumPattern(name="ambient_none", hits=[])]


def _create_chillwave_drums() -> List[DrumPattern]:
    """Chillwave: soft, washed out."""
    patterns = []

    main = DrumPattern(
        name="chillwave_main",
        hits=[
            DrumHit(0.0, "kick", 70),
            DrumHit(2.0, "kick", 65),
            DrumHit(1.0, "snare", 55),
            DrumHit(3.0, "snare", 60),
            DrumHit(0.5, "hihat_closed", 35),
            DrumHit(1.5, "hihat_closed", 30),
            DrumHit(2.5, "hihat_closed", 35),
            DrumHit(3.5, "hihat_closed", 30),
        ]
    )
    patterns.append(main)

    return patterns


def _create_neosoul_drums() -> List[DrumPattern]:
    """Neo-soul: J Dilla-style, behind the beat."""
    patterns = []

    main = DrumPattern(
        name="neosoul_main",
        hits=[
            # Slightly late/drunk feel
            DrumHit(0.05, "kick", 85),
            DrumHit(1.55, "kick", 75),
            DrumHit(2.55, "kick", 80),
            DrumHit(1.05, "snare", 80),
            DrumHit(1.8, "snare", 35),  # Ghost
            DrumHit(3.1, "snare", 85),
            # Swung hats
            DrumHit(0.35, "hihat_closed", 50),
            DrumHit(0.7, "hihat_closed", 45),
            DrumHit(1.35, "hihat_closed", 50),
            DrumHit(1.7, "hihat_closed", 45),
            DrumHit(2.35, "hihat_closed", 50),
            DrumHit(2.7, "hihat_closed", 45),
            DrumHit(3.35, "hihat_closed", 50),
            DrumHit(3.7, "hihat_closed", 45),
        ]
    )
    patterns.append(main)

    return patterns


def _create_rnb_drums() -> List[DrumPattern]:
    """R&B: smooth, pocket groove."""
    patterns = []

    main = DrumPattern(
        name="rnb_main",
        hits=[
            DrumHit(0.0, "kick", 95),
            DrumHit(1.5, "kick", 85),
            DrumHit(2.75, "kick", 80),
            DrumHit(1.0, "snare", 90),
            DrumHit(2.5, "snare", 40),  # Ghost
            DrumHit(3.0, "snare", 95),
            # Subtle hats with swing
            *[DrumHit(i * 0.5 + (0.08 if i % 2 == 1 else 0), "hihat_closed", 55)
              for i in range(8)],
        ]
    )
    patterns.append(main)

    return patterns


def _create_synthwave_drums() -> List[DrumPattern]:
    """Synthwave: gated reverb, 80s feel."""
    patterns = []

    main = DrumPattern(
        name="synthwave_main",
        hits=[
            DrumHit(0.0, "kick", 105),
            DrumHit(2.0, "kick", 105),
            DrumHit(1.0, "snare", 110),  # Big gated snare
            DrumHit(3.0, "snare", 110),
            # Bright hats
            *[DrumHit(i * 0.5, "hihat_closed", 65) for i in range(8)],
            DrumHit(3.5, "hihat_open", 70),
            # Tom fill hint
            DrumHit(3.75, "tom_high", 75),
        ]
    )
    patterns.append(main)

    return patterns


# ============================================================================
# Bass Patterns by Genre
# ============================================================================

BASS_PATTERNS: Dict[str, List[BassPattern]] = {
    "house": [
        BassPattern("house_pump", [
            (0.0, 0, 0.4, (100, 115)),
            (0.5, 0, 0.4, (90, 105)),
            (1.0, 0, 0.4, (100, 115)),
            (1.5, 0, 0.4, (90, 105)),
            (2.0, 0, 0.4, (100, 115)),
            (2.5, 0, 0.4, (90, 105)),
            (3.0, 0, 0.4, (100, 115)),
            (3.5, 0, 0.4, (90, 105)),
        ]),
        BassPattern("house_octave", [
            (0.0, 0, 0.9, (105, 115)),
            (1.0, 12, 0.9, (95, 105)),  # Octave up
            (2.0, 0, 0.9, (105, 115)),
            (3.0, 7, 0.9, (95, 105)),  # Fifth
        ]),
    ],
    "techno": [
        BassPattern("techno_pulse", [
            (0.0, 0, 0.4, (110, 125)),
            (0.5, 0, 0.4, (105, 120)),
            (1.0, 0, 0.4, (110, 125)),
            (1.5, 0, 0.4, (105, 120)),
            (2.0, 0, 0.4, (110, 125)),
            (2.5, 0, 0.4, (105, 120)),
            (3.0, 0, 0.4, (110, 125)),
            (3.5, 0, 0.4, (105, 120)),
        ]),
        BassPattern("techno_drone", [
            (0.0, 0, 3.9, (100, 115)),
        ]),
    ],
    "jazz": [
        BassPattern("walking", [
            (0.0, 0, 0.9, (80, 95)),
            (1.0, 4, 0.9, (75, 90)),   # Third
            (2.0, 7, 0.9, (80, 95)),   # Fifth
            (3.0, 5, 0.9, (75, 90)),   # Fourth (approach)
        ]),
        BassPattern("walking_chromatic", [
            (0.0, 0, 0.9, (80, 95)),
            (1.0, 3, 0.9, (75, 90)),
            (2.0, 5, 0.9, (80, 95)),
            (3.0, 6, 0.9, (75, 90)),  # Chromatic approach
        ]),
    ],
    "hip-hop-boom-bap": [
        BassPattern("boombap_simple", [
            (0.0, 0, 1.4, (100, 115)),
            (2.0, 0, 1.4, (95, 110)),
        ]),
        BassPattern("boombap_syncopated", [
            (0.0, 0, 0.9, (105, 115)),
            (0.75, 0, 0.5, (90, 100)),
            (2.25, 0, 1.2, (100, 110)),
        ]),
    ],
    "trap": [
        BassPattern("808_sustain", [
            (0.0, 0, 3.9, (115, 127)),  # Long 808
        ]),
        BassPattern("808_slide", [
            (0.0, 0, 1.9, (120, 127)),
            (2.0, -2, 1.9, (115, 125)),  # Slide down
        ]),
    ],
    "funk": [
        BassPattern("slap_main", [
            (0.0, 0, 0.3, (110, 120)),  # Thumb
            (0.5, 12, 0.2, (90, 100)),  # Pop octave
            (0.75, 0, 0.3, (100, 110)),
            (1.5, 7, 0.3, (95, 105)),   # Fifth
            (2.0, 0, 0.2, (110, 120)),
            (2.25, 0, 0.2, (100, 110)),
            (2.5, 12, 0.2, (90, 100)),  # Pop
            (3.0, 5, 0.3, (105, 115)),  # Fourth
            (3.5, 7, 0.3, (100, 110)),
        ]),
    ],
    "disco": [
        BassPattern("disco_octave", [
            (0.0, 0, 0.4, (100, 110)),
            (0.5, 12, 0.4, (90, 100)),
            (1.0, 0, 0.4, (100, 110)),
            (1.5, 12, 0.4, (90, 100)),
            (2.0, 0, 0.4, (100, 110)),
            (2.5, 12, 0.4, (90, 100)),
            (3.0, 7, 0.4, (95, 105)),
            (3.5, 12, 0.4, (90, 100)),
        ]),
    ],
    "rock": [
        BassPattern("rock_root", [
            (0.0, 0, 1.9, (100, 115)),
            (2.0, 0, 1.4, (100, 115)),
            (3.5, 7, 0.4, (90, 105)),
        ]),
        BassPattern("rock_driving", [
            (0.0, 0, 0.9, (105, 115)),
            (1.0, 0, 0.9, (100, 110)),
            (2.0, 0, 0.9, (105, 115)),
            (3.0, 7, 0.9, (100, 110)),
        ]),
    ],
    "drum-and-bass": [
        BassPattern("reese", [
            (0.0, 0, 1.9, (110, 125)),
            (2.0, -5, 1.9, (105, 120)),  # Movement
        ]),
    ],
    "dubstep": [
        BassPattern("wobble", [
            (0.0, 0, 0.4, (115, 127)),
            (0.5, 0, 0.4, (100, 115)),
            (1.0, 0, 0.4, (115, 127)),
            (1.5, 0, 0.4, (100, 115)),
            (2.0, 0, 0.4, (115, 127)),
            (2.5, 0, 0.4, (100, 115)),
            (3.0, 0, 0.4, (115, 127)),
            (3.5, 0, 0.4, (100, 115)),
        ]),
    ],
    "lo-fi-hip-hop": [
        BassPattern("lofi_mellow", [
            (0.0, 0, 1.9, (75, 90)),
            (2.0, 5, 1.4, (70, 85)),
            (3.5, 7, 0.4, (65, 80)),
        ]),
    ],
    "neo-soul": [
        BassPattern("neosoul_groove", [
            (0.05, 0, 0.9, (80, 95)),  # Slightly late
            (1.55, 3, 0.4, (70, 85)),
            (2.1, 5, 0.8, (75, 90)),
            (3.0, 7, 0.9, (80, 95)),
        ]),
    ],
    "r-and-b": [
        BassPattern("rnb_smooth", [
            (0.0, 0, 1.4, (85, 100)),
            (1.5, 3, 0.4, (75, 90)),
            (2.0, 5, 0.9, (80, 95)),
            (3.0, 7, 0.9, (80, 95)),
        ]),
    ],
    "ambient": [
        BassPattern("ambient_pad", [
            (0.0, 0, 3.9, (50, 70)),  # Soft sustained
        ]),
    ],
    "acoustic-folk": [
        BassPattern("folk_root", [
            (0.0, 0, 1.9, (70, 85)),
            (2.0, 7, 1.9, (65, 80)),
        ]),
    ],
    "cinematic": [
        BassPattern("epic_pulse", [
            (0.0, 0, 0.9, (90, 110)),
            (1.0, 0, 0.9, (85, 105)),
            (2.0, 0, 0.9, (90, 110)),
            (3.0, 0, 0.9, (85, 105)),
        ]),
    ],
    "chillwave": [
        BassPattern("chillwave_soft", [
            (0.0, 0, 1.9, (60, 80)),
            (2.0, 0, 1.9, (55, 75)),
        ]),
    ],
    "synthwave": [
        BassPattern("synthwave_arp", [
            (0.0, 0, 0.4, (95, 110)),
            (0.5, 0, 0.4, (90, 105)),
            (1.0, 12, 0.4, (85, 100)),
            (1.5, 7, 0.4, (90, 105)),
            (2.0, 0, 0.4, (95, 110)),
            (2.5, 0, 0.4, (90, 105)),
            (3.0, 5, 0.4, (85, 100)),
            (3.5, 7, 0.4, (90, 105)),
        ]),
    ],
}


# ============================================================================
# Melody Styles by Genre
# ============================================================================

MELODY_STYLES: Dict[str, MelodyStyle] = {
    "house": MelodyStyle(
        note_density=0.4,
        preferred_durations=[0.5, 1.0, 2.0],
        duration_weights=[0.3, 0.5, 0.2],
        rest_probability=0.3,
        leap_probability=0.2,
        max_leap=7,
        rhythmic_patterns=[[0.5, 0.5, 1.0], [1.0, 0.5, 0.5]],
        articulation="normal",
        velocity_range=(70, 95),
        octave_range=(-1, 1),
    ),
    "techno": MelodyStyle(
        note_density=0.3,
        preferred_durations=[0.25, 0.5, 1.0],
        duration_weights=[0.4, 0.4, 0.2],
        rest_probability=0.4,
        leap_probability=0.3,
        max_leap=12,
        rhythmic_patterns=[[0.25, 0.25, 0.5], [0.5, 0.25, 0.25]],
        articulation="staccato",
        velocity_range=(60, 90),
        octave_range=(0, 2),
    ),
    "jazz": MelodyStyle(
        note_density=0.7,
        preferred_durations=[0.33, 0.67, 1.0, 2.0],
        duration_weights=[0.3, 0.3, 0.25, 0.15],
        rest_probability=0.15,
        leap_probability=0.35,
        max_leap=12,
        rhythmic_patterns=[[0.33, 0.33, 0.34], [0.67, 0.33]],
        articulation="legato",
        velocity_range=(55, 100),
        octave_range=(-1, 2),
    ),
    "hip-hop-boom-bap": MelodyStyle(
        note_density=0.35,
        preferred_durations=[0.5, 1.0, 1.5],
        duration_weights=[0.4, 0.4, 0.2],
        rest_probability=0.35,
        leap_probability=0.15,
        max_leap=5,
        rhythmic_patterns=[[1.0, 0.5, 0.5], [0.5, 1.0, 0.5]],
        articulation="normal",
        velocity_range=(65, 90),
        octave_range=(-1, 1),
    ),
    "trap": MelodyStyle(
        note_density=0.25,
        preferred_durations=[0.5, 1.0, 2.0],
        duration_weights=[0.3, 0.4, 0.3],
        rest_probability=0.4,
        leap_probability=0.25,
        max_leap=7,
        rhythmic_patterns=[[1.0, 1.0], [2.0]],
        articulation="staccato",
        velocity_range=(70, 100),
        octave_range=(0, 2),
    ),
    "rock": MelodyStyle(
        note_density=0.5,
        preferred_durations=[0.5, 1.0, 2.0],
        duration_weights=[0.35, 0.45, 0.2],
        rest_probability=0.2,
        leap_probability=0.25,
        max_leap=7,
        rhythmic_patterns=[[1.0, 1.0, 2.0], [0.5, 0.5, 1.0]],
        articulation="normal",
        velocity_range=(75, 110),
        octave_range=(-1, 1),
    ),
    "funk": MelodyStyle(
        note_density=0.55,
        preferred_durations=[0.25, 0.5, 0.75],
        duration_weights=[0.4, 0.4, 0.2],
        rest_probability=0.25,
        leap_probability=0.2,
        max_leap=5,
        rhythmic_patterns=[[0.25, 0.25, 0.5], [0.5, 0.25, 0.25]],
        articulation="staccato",
        velocity_range=(70, 105),
        octave_range=(-1, 1),
    ),
    "disco": MelodyStyle(
        note_density=0.5,
        preferred_durations=[0.5, 1.0, 2.0],
        duration_weights=[0.4, 0.4, 0.2],
        rest_probability=0.2,
        leap_probability=0.3,
        max_leap=7,
        rhythmic_patterns=[[0.5, 0.5, 1.0], [1.0, 0.5, 0.5]],
        articulation="legato",
        velocity_range=(75, 100),
        octave_range=(0, 1),
    ),
    "drum-and-bass": MelodyStyle(
        note_density=0.35,
        preferred_durations=[0.25, 0.5, 1.0],
        duration_weights=[0.3, 0.4, 0.3],
        rest_probability=0.35,
        leap_probability=0.3,
        max_leap=12,
        rhythmic_patterns=[[0.25, 0.25, 0.5], [0.5, 0.5]],
        articulation="staccato",
        velocity_range=(65, 95),
        octave_range=(0, 2),
    ),
    "dubstep": MelodyStyle(
        note_density=0.25,
        preferred_durations=[0.5, 1.0, 2.0],
        duration_weights=[0.3, 0.4, 0.3],
        rest_probability=0.45,
        leap_probability=0.3,
        max_leap=12,
        rhythmic_patterns=[[1.0, 1.0], [2.0]],
        articulation="normal",
        velocity_range=(70, 100),
        octave_range=(0, 2),
    ),
    "lo-fi-hip-hop": MelodyStyle(
        note_density=0.3,
        preferred_durations=[0.67, 1.0, 1.33],
        duration_weights=[0.35, 0.4, 0.25],
        rest_probability=0.35,
        leap_probability=0.15,
        max_leap=5,
        rhythmic_patterns=[[0.67, 0.33, 1.0], [1.0, 0.67, 0.33]],
        articulation="legato",
        velocity_range=(50, 80),
        octave_range=(-1, 1),
    ),
    "neo-soul": MelodyStyle(
        note_density=0.45,
        preferred_durations=[0.33, 0.67, 1.0, 1.33],
        duration_weights=[0.25, 0.35, 0.25, 0.15],
        rest_probability=0.25,
        leap_probability=0.25,
        max_leap=7,
        rhythmic_patterns=[[0.33, 0.67, 1.0], [0.67, 0.33, 1.0]],
        articulation="legato",
        velocity_range=(55, 90),
        octave_range=(-1, 1),
    ),
    "r-and-b": MelodyStyle(
        note_density=0.5,
        preferred_durations=[0.5, 0.75, 1.0, 1.5],
        duration_weights=[0.3, 0.3, 0.25, 0.15],
        rest_probability=0.2,
        leap_probability=0.3,
        max_leap=7,
        rhythmic_patterns=[[0.5, 0.5, 1.0], [0.75, 0.25, 1.0]],
        articulation="legato",
        velocity_range=(60, 95),
        octave_range=(-1, 1),
    ),
    "ambient": MelodyStyle(
        note_density=0.15,
        preferred_durations=[2.0, 4.0, 8.0],
        duration_weights=[0.3, 0.4, 0.3],
        rest_probability=0.5,
        leap_probability=0.4,
        max_leap=12,
        rhythmic_patterns=[[4.0], [2.0, 2.0]],
        articulation="legato",
        velocity_range=(35, 70),
        octave_range=(-1, 2),
    ),
    "acoustic-folk": MelodyStyle(
        note_density=0.45,
        preferred_durations=[0.5, 1.0, 2.0],
        duration_weights=[0.35, 0.45, 0.2],
        rest_probability=0.2,
        leap_probability=0.2,
        max_leap=5,
        rhythmic_patterns=[[1.0, 0.5, 0.5], [0.5, 1.0, 0.5]],
        articulation="legato",
        velocity_range=(55, 85),
        octave_range=(-1, 1),
    ),
    "cinematic": MelodyStyle(
        note_density=0.35,
        preferred_durations=[1.0, 2.0, 4.0],
        duration_weights=[0.3, 0.45, 0.25],
        rest_probability=0.25,
        leap_probability=0.35,
        max_leap=12,
        rhythmic_patterns=[[2.0, 2.0], [1.0, 1.0, 2.0]],
        articulation="legato",
        velocity_range=(50, 100),
        octave_range=(-1, 2),
    ),
    "chillwave": MelodyStyle(
        note_density=0.3,
        preferred_durations=[1.0, 2.0, 3.0],
        duration_weights=[0.35, 0.4, 0.25],
        rest_probability=0.35,
        leap_probability=0.2,
        max_leap=7,
        rhythmic_patterns=[[1.0, 1.0, 2.0], [2.0, 1.0, 1.0]],
        articulation="legato",
        velocity_range=(45, 75),
        octave_range=(-1, 1),
    ),
    "synthwave": MelodyStyle(
        note_density=0.45,
        preferred_durations=[0.5, 1.0, 2.0],
        duration_weights=[0.35, 0.45, 0.2],
        rest_probability=0.2,
        leap_probability=0.3,
        max_leap=7,
        rhythmic_patterns=[[0.5, 0.5, 1.0], [1.0, 0.5, 0.5]],
        articulation="legato",
        velocity_range=(65, 95),
        octave_range=(0, 1),
    ),
}


# ============================================================================
# Groove Templates by Genre
# ============================================================================

GROOVE_TEMPLATES: Dict[str, GrooveTemplate] = {
    "house": GrooveTemplate(
        swing_amount=0.05,
        push_pull={0.5: 0.02, 1.5: 0.02, 2.5: 0.02, 3.5: 0.02},
        velocity_accents={0.0: 1.1, 1.0: 1.0, 2.0: 1.1, 3.0: 1.0},
        humanize_timing=0.01,
        humanize_velocity=0.05,
    ),
    "techno": GrooveTemplate(
        swing_amount=0.0,
        push_pull={},
        velocity_accents={0.0: 1.15, 1.0: 1.0, 2.0: 1.1, 3.0: 1.0},
        humanize_timing=0.005,
        humanize_velocity=0.03,
    ),
    "jazz": GrooveTemplate(
        swing_amount=0.35,
        push_pull={0.67: 0.05, 1.67: 0.05, 2.67: 0.05, 3.67: 0.05},
        velocity_accents={0.0: 1.0, 1.0: 0.85, 2.0: 1.05, 3.0: 0.85},
        humanize_timing=0.03,
        humanize_velocity=0.1,
    ),
    "hip-hop-boom-bap": GrooveTemplate(
        swing_amount=0.15,
        push_pull={0.5: 0.03, 2.5: 0.03},
        velocity_accents={0.0: 1.1, 1.0: 1.15, 2.0: 0.95, 3.0: 1.15},
        humanize_timing=0.02,
        humanize_velocity=0.08,
    ),
    "trap": GrooveTemplate(
        swing_amount=0.0,
        push_pull={},
        velocity_accents={0.0: 1.2, 2.0: 1.15},
        humanize_timing=0.01,
        humanize_velocity=0.05,
    ),
    "rock": GrooveTemplate(
        swing_amount=0.05,
        push_pull={},
        velocity_accents={0.0: 1.05, 1.0: 1.15, 2.0: 1.05, 3.0: 1.15},
        humanize_timing=0.015,
        humanize_velocity=0.07,
    ),
    "funk": GrooveTemplate(
        swing_amount=0.12,
        push_pull={0.25: -0.02, 0.75: 0.02, 2.25: -0.02, 2.75: 0.02},
        velocity_accents={0.0: 1.2, 1.0: 1.0, 2.0: 0.9, 3.0: 1.0},  # THE ONE
        humanize_timing=0.02,
        humanize_velocity=0.1,
    ),
    "disco": GrooveTemplate(
        swing_amount=0.05,
        push_pull={0.5: 0.02, 1.5: 0.02, 2.5: 0.02, 3.5: 0.02},
        velocity_accents={0.0: 1.05, 1.0: 1.1, 2.0: 1.05, 3.0: 1.1},
        humanize_timing=0.01,
        humanize_velocity=0.05,
    ),
    "drum-and-bass": GrooveTemplate(
        swing_amount=0.08,
        push_pull={0.5: -0.02, 1.5: 0.02},
        velocity_accents={0.0: 1.1, 0.5: 1.15, 2.0: 1.05},
        humanize_timing=0.01,
        humanize_velocity=0.05,
    ),
    "dubstep": GrooveTemplate(
        swing_amount=0.0,
        push_pull={},
        velocity_accents={0.0: 1.2, 2.0: 1.25},
        humanize_timing=0.005,
        humanize_velocity=0.03,
    ),
    "lo-fi-hip-hop": GrooveTemplate(
        swing_amount=0.25,
        push_pull={0.33: 0.05, 0.67: 0.03, 1.33: 0.05, 1.67: 0.03},
        velocity_accents={0.0: 0.95, 1.0: 1.0, 2.0: 0.9, 3.0: 1.0},
        humanize_timing=0.04,
        humanize_velocity=0.12,
    ),
    "neo-soul": GrooveTemplate(
        swing_amount=0.2,
        push_pull={0.0: 0.05, 1.0: 0.03, 2.0: 0.05, 3.0: 0.03},  # Behind beat
        velocity_accents={0.0: 0.95, 1.0: 1.0, 2.0: 0.9, 3.0: 1.05},
        humanize_timing=0.04,
        humanize_velocity=0.1,
    ),
    "r-and-b": GrooveTemplate(
        swing_amount=0.15,
        push_pull={0.5: 0.03, 1.5: 0.02, 2.5: 0.03, 3.5: 0.02},
        velocity_accents={0.0: 1.0, 1.0: 1.05, 2.0: 0.95, 3.0: 1.05},
        humanize_timing=0.025,
        humanize_velocity=0.08,
    ),
    "ambient": GrooveTemplate(
        swing_amount=0.0,
        push_pull={},
        velocity_accents={},
        humanize_timing=0.05,
        humanize_velocity=0.15,
    ),
    "acoustic-folk": GrooveTemplate(
        swing_amount=0.1,
        push_pull={},
        velocity_accents={0.0: 1.05, 2.0: 1.0},
        humanize_timing=0.03,
        humanize_velocity=0.1,
    ),
    "cinematic": GrooveTemplate(
        swing_amount=0.0,
        push_pull={},
        velocity_accents={0.0: 1.1, 2.0: 1.15},
        humanize_timing=0.02,
        humanize_velocity=0.08,
    ),
    "chillwave": GrooveTemplate(
        swing_amount=0.1,
        push_pull={0.5: 0.03, 1.5: 0.03, 2.5: 0.03, 3.5: 0.03},
        velocity_accents={},
        humanize_timing=0.03,
        humanize_velocity=0.1,
    ),
    "synthwave": GrooveTemplate(
        swing_amount=0.05,
        push_pull={},
        velocity_accents={0.0: 1.05, 1.0: 1.1, 2.0: 1.05, 3.0: 1.1},
        humanize_timing=0.015,
        humanize_velocity=0.05,
    ),
}


# ============================================================================
# Main Access Functions
# ============================================================================

def get_drum_patterns(genre_id: str) -> List[DrumPattern]:
    """Get drum patterns for a genre."""
    pattern_creators = {
        "house": _create_house_drums,
        "techno": _create_techno_drums,
        "jazz": _create_jazz_drums,
        "hip-hop-boom-bap": _create_hiphop_drums,
        "trap": _create_trap_drums,
        "rock": _create_rock_drums,
        "funk": _create_funk_drums,
        "disco": _create_disco_drums,
        "drum-and-bass": _create_dnb_drums,
        "dubstep": _create_dubstep_drums,
        "lo-fi-hip-hop": _create_lofi_drums,
        "acoustic-folk": _create_folk_drums,
        "cinematic": _create_cinematic_drums,
        "ambient": _create_ambient_drums,
        "chillwave": _create_chillwave_drums,
        "neo-soul": _create_neosoul_drums,
        "r-and-b": _create_rnb_drums,
        "synthwave": _create_synthwave_drums,
    }

    creator = pattern_creators.get(genre_id, _create_rock_drums)
    return creator()


def get_bass_patterns(genre_id: str) -> List[BassPattern]:
    """Get bass patterns for a genre."""
    return BASS_PATTERNS.get(genre_id, BASS_PATTERNS.get("rock", []))


def get_melody_style(genre_id: str) -> MelodyStyle:
    """Get melody style for a genre."""
    return MELODY_STYLES.get(genre_id, MELODY_STYLES.get("rock"))


def get_groove_template(genre_id: str) -> GrooveTemplate:
    """Get groove template for a genre."""
    return GROOVE_TEMPLATES.get(genre_id, GROOVE_TEMPLATES.get("rock"))


# Chord styles per genre
CHORD_STYLES: Dict[str, ChordStyle] = {
    "house": ChordStyle("house", "spread", [(0.0, 3.5)], (65, 85), True),
    "techno": ChordStyle("techno", "power", [(0.0, 0.4), (0.5, 0.4), (1.0, 0.4), (1.5, 0.4), (2.0, 0.4), (2.5, 0.4), (3.0, 0.4), (3.5, 0.4)], (55, 75), False),
    "jazz": ChordStyle("jazz", "extended", [(0.0, 2.0), (2.0, 1.5), (3.5, 0.5)], (50, 75), True),
    "hip-hop-boom-bap": ChordStyle("boombap", "close", [(0.0, 1.5), (2.0, 1.5)], (60, 80), True),
    "trap": ChordStyle("trap", "spread", [(0.0, 3.9)], (60, 80), False),
    "rock": ChordStyle("rock", "power", [(0.0, 3.5)], (85, 105), False),
    "funk": ChordStyle("funk", "close", [(0.0, 0.3), (0.5, 0.3), (0.75, 0.2), (1.5, 0.3), (2.0, 0.3), (2.5, 0.3), (3.0, 0.3), (3.5, 0.2)], (65, 90), True),
    "disco": ChordStyle("disco", "spread", [(0.0, 0.4), (0.5, 0.4), (1.0, 0.4), (1.5, 0.4), (2.0, 0.4), (2.5, 0.4), (3.0, 0.4), (3.5, 0.4)], (70, 90), True),
    "drum-and-bass": ChordStyle("dnb", "spread", [(0.0, 1.9), (2.0, 1.9)], (60, 80), True),
    "dubstep": ChordStyle("dubstep", "power", [(0.0, 3.9)], (70, 90), False),
    "lo-fi-hip-hop": ChordStyle("lofi", "extended", [(0.0, 3.5)], (45, 65), True),
    "neo-soul": ChordStyle("neosoul", "extended", [(0.0, 2.0), (2.0, 1.5), (3.5, 0.5)], (50, 75), True),
    "r-and-b": ChordStyle("rnb", "extended", [(0.0, 1.5), (1.5, 1.0), (2.5, 1.0)], (55, 80), True),
    "ambient": ChordStyle("ambient", "spread", [(0.0, 7.9)], (35, 55), True),
    "acoustic-folk": ChordStyle("folk", "close", [(0.0, 3.5)], (55, 75), True),
    "cinematic": ChordStyle("cinematic", "spread", [(0.0, 3.5)], (60, 100), True),
    "chillwave": ChordStyle("chillwave", "spread", [(0.0, 3.5)], (45, 65), True),
    "synthwave": ChordStyle("synthwave", "spread", [(0.0, 3.5)], (65, 85), True),
}


def get_chord_style(genre_id: str) -> ChordStyle:
    """Get chord style for a genre."""
    return CHORD_STYLES.get(genre_id, CHORD_STYLES.get("rock"))
