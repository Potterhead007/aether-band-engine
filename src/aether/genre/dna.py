"""
Genre DNA - 48-dimensional genre fingerprint definitions.

Each genre is defined by a comprehensive DNA vector covering:
- Rhythm (12 dimensions): tempo, swing, syncopation, patterns
- Harmony (12 dimensions): modes, progressions, voicings, tension
- Melody (12 dimensions): scales, density, intervals, contour
- Timbre (6 dimensions): instruments, texture, frequency balance
- Structure (6 dimensions): sections, length, dynamics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class KickPattern(Enum):
    """Kick drum pattern types."""
    FOUR_ON_FLOOR = "four_on_floor"
    BREAKBEAT = "breakbeat"
    TRAP = "trap"
    BOOM_BAP = "boom_bap"
    UK_DRILL = "uk_drill"
    DEMBOW = "dembow"
    AFROBEAT = "afrobeat"
    HALFTIME = "halftime"
    SPARSE = "sparse"
    CINEMATIC = "cinematic"


class SnarePosition(Enum):
    """Snare placement patterns."""
    TWO_FOUR = "2_4"          # Standard backbeat
    THREE = "3"               # Halftime feel
    OFFBEAT = "offbeat"       # Syncopated
    ROLLING = "rolling"       # Trap rolls
    SPARSE = "sparse"         # Minimal
    POLYRHYTHMIC = "polyrhythmic"


class TimeFeel(Enum):
    """Rhythmic time feel."""
    STRAIGHT = "straight"
    SWING = "swing"
    SHUFFLE = "shuffle"
    TRIPLET = "triplet"
    BEHIND = "behind"         # Laid back
    AHEAD = "ahead"           # Pushing


class TensionProfile(Enum):
    """Harmonic tension handling."""
    RESOLVE_QUICK = "resolve_quick"
    SUSPEND = "suspend"
    CHROMATIC = "chromatic"
    STATIC = "static"
    DARK_UNRESOLVED = "dark_unresolved"
    EUPHORIC_RELEASE = "euphoric_release"
    GENTLE_SUSPEND = "gentle_suspend"


class VoicingStyle(Enum):
    """Chord voicing styles."""
    CLOSE = "close"
    DROP2 = "drop2"
    DROP3 = "drop3"
    SPREAD = "spread"
    SHELL = "shell"
    ROOTLESS = "rootless"
    POWER = "power"
    STAB = "stab"


class ContourType(Enum):
    """Melodic contour shapes."""
    ARCH = "arch"
    DESCENT = "descent"
    ASCENT = "ascent"
    WAVE = "wave"
    STATIC = "static"
    QUESTION_ANSWER = "question_answer"


class DynamicRange(Enum):
    """Dynamic range characteristics."""
    COMPRESSED = "compressed"
    MODERATE = "moderate"
    WIDE = "wide"
    CINEMATIC = "cinematic"


# =============================================================================
# Deterministic Index Mappings for Vector Encoding
# =============================================================================
# These replace hash() for reproducible 48-dimensional genre fingerprints

KICK_PATTERN_INDEX: dict[KickPattern, float] = {
    KickPattern.FOUR_ON_FLOOR: 0.0,
    KickPattern.BREAKBEAT: 0.1,
    KickPattern.TRAP: 0.2,
    KickPattern.BOOM_BAP: 0.3,
    KickPattern.UK_DRILL: 0.4,
    KickPattern.DEMBOW: 0.5,
    KickPattern.AFROBEAT: 0.6,
    KickPattern.HALFTIME: 0.7,
    KickPattern.SPARSE: 0.8,
    KickPattern.CINEMATIC: 0.9,
}

SNARE_POSITION_INDEX: dict[SnarePosition, float] = {
    SnarePosition.TWO_FOUR: 0.0,
    SnarePosition.THREE: 0.2,
    SnarePosition.OFFBEAT: 0.4,
    SnarePosition.ROLLING: 0.6,
    SnarePosition.SPARSE: 0.8,
    SnarePosition.POLYRHYTHMIC: 1.0,
}

TIME_FEEL_INDEX: dict[TimeFeel, float] = {
    TimeFeel.STRAIGHT: 0.0,
    TimeFeel.SWING: 0.2,
    TimeFeel.SHUFFLE: 0.4,
    TimeFeel.TRIPLET: 0.6,
    TimeFeel.BEHIND: 0.8,
    TimeFeel.AHEAD: 1.0,
}

TENSION_PROFILE_INDEX: dict[TensionProfile, float] = {
    TensionProfile.RESOLVE_QUICK: 0.0,
    TensionProfile.SUSPEND: 0.15,
    TensionProfile.CHROMATIC: 0.30,
    TensionProfile.STATIC: 0.45,
    TensionProfile.DARK_UNRESOLVED: 0.60,
    TensionProfile.EUPHORIC_RELEASE: 0.75,
    TensionProfile.GENTLE_SUSPEND: 0.90,
}

VOICING_STYLE_INDEX: dict[VoicingStyle, float] = {
    VoicingStyle.CLOSE: 0.0,
    VoicingStyle.DROP2: 0.125,
    VoicingStyle.DROP3: 0.25,
    VoicingStyle.SPREAD: 0.375,
    VoicingStyle.SHELL: 0.5,
    VoicingStyle.ROOTLESS: 0.625,
    VoicingStyle.POWER: 0.75,
    VoicingStyle.STAB: 0.875,
}

CONTOUR_TYPE_INDEX: dict[ContourType, float] = {
    ContourType.ARCH: 0.0,
    ContourType.DESCENT: 0.2,
    ContourType.ASCENT: 0.4,
    ContourType.WAVE: 0.6,
    ContourType.STATIC: 0.8,
    ContourType.QUESTION_ANSWER: 1.0,
}

DYNAMIC_RANGE_INDEX: dict[DynamicRange, float] = {
    DynamicRange.COMPRESSED: 0.0,
    DynamicRange.MODERATE: 0.33,
    DynamicRange.WIDE: 0.66,
    DynamicRange.CINEMATIC: 1.0,
}

# String-based indices for non-enum fields
GROOVE_TEMPLATE_INDEX: dict[str, float] = {
    "lofi_lazy": 0.0,
    "trap_bounce": 0.1,
    "drill_slide": 0.2,
    "boom_bap_swing": 0.3,
    "synthwave_drive": 0.4,
    "house_groove": 0.5,
    "techno_hypnotic": 0.6,
    "amen_break": 0.7,
    "dembow": 0.8,
    "afrobeat_12_8": 0.85,
    "pop_standard": 0.9,
    "cinematic_rubato": 1.0,
}

BASS_BEHAVIOR_INDEX: dict[str, float] = {
    "root": 0.0,
    "walking": 0.25,
    "pedal": 0.5,
    "glide": 0.75,
}

RHYTHMIC_MOTIF_INDEX: dict[str, float] = {
    "on_beat": 0.0,
    "syncopated": 0.5,
    "triplet": 1.0,
}

TARGET_NOTES_INDEX: dict[str, float] = {
    "chord_tones": 0.0,
    "tensions": 0.5,
    "chromatic": 0.75,
    "root": 1.0,
}

ARTICULATION_INDEX: dict[str, float] = {
    "legato": 0.0,
    "staccato": 0.5,
    "mixed": 1.0,
}

FREQUENCY_BALANCE_INDEX: dict[str, float] = {
    "bass_heavy": 0.0,
    "mid_focus": 0.33,
    "balanced": 0.66,
    "bright": 1.0,
}

REVERB_CHARACTER_INDEX: dict[str, float] = {
    "dry": 0.0,
    "room": 0.33,
    "hall": 0.66,
    "infinite": 1.0,
}

INTRO_STYLE_INDEX: dict[str, float] = {
    "ambient": 0.0,
    "drop": 0.25,
    "buildup": 0.5,
    "direct": 0.75,
}

ARRANGEMENT_CURVE_INDEX: dict[str, float] = {
    "flat": 0.0,
    "build": 0.5,
    "wave": 1.0,
}


@dataclass
class RhythmDNA:
    """Rhythm characteristics (12 dimensions)."""

    tempo_center: float              # BPM center
    tempo_variance: float            # Allowed deviation ±
    swing_amount: float              # 0.0 = straight, 0.33 = triplet swing
    syncopation_density: float       # 0-1, notes off the grid
    hihat_subdivision: int           # 8, 16, 32
    kick_pattern: KickPattern
    snare_position: SnarePosition
    ghost_note_density: float        # 0-1
    polyrhythm_layers: int           # 1-4
    groove_template: str             # Reference pattern ID
    time_feel: TimeFeel
    rhythmic_complexity: float       # 0-1

    @property
    def tempo_range(self) -> tuple[float, float]:
        """Compute tempo range from center and variance."""
        return (self.tempo_center - self.tempo_variance, self.tempo_center + self.tempo_variance)


@dataclass
class HarmonyDNA:
    """Harmonic characteristics (12 dimensions)."""

    mode_preferences: list[str]      # ["minor", "dorian", "phrygian"]
    chord_complexity: float          # 0=triads, 1=extended+altered
    harmonic_rhythm: float           # Chord changes per bar
    tension_profile: TensionProfile
    bass_behavior: str               # "root", "walking", "pedal", "glide"
    voicing_style: VoicingStyle
    primary_cadences: list[str]      # ["ii-V-I", "IV-V-I"]
    avoid_cadences: list[str]        # Cadences to avoid
    borrowed_chords: list[str]       # ["bVII", "bVI", "iv"]
    use_secondary_dominants: bool
    use_tritone_subs: bool
    common_progressions: list[list[str]]

    @property
    def primary_modes(self) -> list[str]:
        """Alias for mode_preferences."""
        return self.mode_preferences[:2] if len(self.mode_preferences) > 1 else self.mode_preferences

    @property
    def secondary_modes(self) -> list[str]:
        """Secondary modes from mode_preferences."""
        return self.mode_preferences[2:] if len(self.mode_preferences) > 2 else []

    @property
    def cadence_preferences(self) -> list[str]:
        """Alias for primary_cadences."""
        return self.primary_cadences


@dataclass
class MelodyDNA:
    """Melodic characteristics (12 dimensions)."""

    scale_preferences: list[str]     # ["minor_pentatonic", "harmonic_minor"]
    note_density: float              # Notes per beat (0-2)
    leap_probability: float          # 0-1
    max_interval: int                # Semitones
    phrase_lengths: list[int]        # Bars [2, 4, 8]
    use_call_response: bool
    motif_repetition: float          # 0-1
    ornamentation: list[str]         # ["slides", "bends", "trills"]
    rhythmic_motif: str              # "on_beat", "syncopated", "triplet"
    contour_type: ContourType
    target_notes: str                # "chord_tones", "tensions", "chromatic"
    articulation: str                # "legato", "staccato", "mixed"


@dataclass
class TimbreDNA:
    """Timbral characteristics (6 dimensions)."""

    primary_instruments: list[str]   # Core instruments
    secondary_instruments: list[str] # Supporting instruments
    texture_density: float           # 0=sparse, 1=dense
    frequency_balance: str           # "bass_heavy", "mid_focus", "bright"
    stereo_width: float              # 0-1
    reverb_character: str            # "dry", "room", "hall", "infinite"


@dataclass
class StructureDNA:
    """Structural characteristics (6 dimensions)."""

    section_types: list[str]         # Available sections
    typical_length_bars: int         # 64, 128, 256
    intro_style: str                 # "ambient", "drop", "buildup", "direct"
    use_build_release: bool          # EDM-style builds
    dynamic_range: DynamicRange
    arrangement_curve: str           # "flat", "build", "wave"


@dataclass
class GenreDNA:
    """
    Complete 48-dimensional genre fingerprint.

    Combines all DNA components into a single profile that
    fully characterizes a genre's musical identity.
    """

    genre_id: str
    genre_name: str
    parent_genre: Optional[str]

    rhythm: RhythmDNA
    harmony: HarmonyDNA
    melody: MelodyDNA
    timbre: TimbreDNA
    structure: StructureDNA

    # Anti-patterns: what this genre should NOT sound like
    avoid_genres: list[str] = field(default_factory=list)

    # Compatibility for blending
    compatible_genres: list[str] = field(default_factory=list)

    def to_vector(self) -> list[float]:
        """
        Convert DNA to deterministic 48-dimensional float vector.

        Uses explicit index mappings instead of hash() for reproducibility
        across Python versions and runs.
        """
        vector = []

        # Rhythm (12 dimensions)
        vector.extend([
            self.rhythm.tempo_center / 200,  # Normalize to 0-1
            self.rhythm.tempo_variance / 30,
            self.rhythm.swing_amount,
            self.rhythm.syncopation_density,
            self.rhythm.hihat_subdivision / 32,
            KICK_PATTERN_INDEX.get(self.rhythm.kick_pattern, 0.5),
            SNARE_POSITION_INDEX.get(self.rhythm.snare_position, 0.5),
            self.rhythm.ghost_note_density,
            self.rhythm.polyrhythm_layers / 4,
            GROOVE_TEMPLATE_INDEX.get(self.rhythm.groove_template, 0.5),
            TIME_FEEL_INDEX.get(self.rhythm.time_feel, 0.5),
            self.rhythm.rhythmic_complexity,
        ])

        # Harmony (12 dimensions)
        vector.extend([
            len(self.harmony.mode_preferences) / 5,
            self.harmony.chord_complexity,
            self.harmony.harmonic_rhythm / 2,
            TENSION_PROFILE_INDEX.get(self.harmony.tension_profile, 0.5),
            BASS_BEHAVIOR_INDEX.get(self.harmony.bass_behavior, 0.5),
            VOICING_STYLE_INDEX.get(self.harmony.voicing_style, 0.5),
            len(self.harmony.primary_cadences) / 5,
            len(self.harmony.avoid_cadences) / 5,
            len(self.harmony.borrowed_chords) / 5,
            float(self.harmony.use_secondary_dominants),
            float(self.harmony.use_tritone_subs),
            len(self.harmony.common_progressions) / 5,
        ])

        # Melody (12 dimensions)
        vector.extend([
            len(self.melody.scale_preferences) / 5,
            self.melody.note_density / 2,
            self.melody.leap_probability,
            self.melody.max_interval / 12,
            len(self.melody.phrase_lengths) / 4,
            float(self.melody.use_call_response),
            self.melody.motif_repetition,
            len(self.melody.ornamentation) / 5,
            RHYTHMIC_MOTIF_INDEX.get(self.melody.rhythmic_motif, 0.5),
            CONTOUR_TYPE_INDEX.get(self.melody.contour_type, 0.5),
            TARGET_NOTES_INDEX.get(self.melody.target_notes, 0.5),
            ARTICULATION_INDEX.get(self.melody.articulation, 0.5),
        ])

        # Timbre (6 dimensions)
        vector.extend([
            len(self.timbre.primary_instruments) / 8,
            len(self.timbre.secondary_instruments) / 8,
            self.timbre.texture_density,
            FREQUENCY_BALANCE_INDEX.get(self.timbre.frequency_balance, 0.5),
            self.timbre.stereo_width,
            REVERB_CHARACTER_INDEX.get(self.timbre.reverb_character, 0.5),
        ])

        # Structure (6 dimensions)
        vector.extend([
            len(self.structure.section_types) / 8,
            self.structure.typical_length_bars / 256,
            INTRO_STYLE_INDEX.get(self.structure.intro_style, 0.5),
            float(self.structure.use_build_release),
            DYNAMIC_RANGE_INDEX.get(self.structure.dynamic_range, 0.5),
            ARRANGEMENT_CURVE_INDEX.get(self.structure.arrangement_curve, 0.5),
        ])

        return vector


# =============================================================================
# Genre DNA Library - 12 Target Genres
# =============================================================================

GENRE_DNA_LIBRARY: dict[str, GenreDNA] = {

    # -------------------------------------------------------------------------
    # HIP-HOP FAMILY
    # -------------------------------------------------------------------------

    "lofi-hip-hop": GenreDNA(
        genre_id="lofi-hip-hop",
        genre_name="Lo-Fi Hip Hop",
        parent_genre="hip-hop",
        rhythm=RhythmDNA(
            tempo_center=82,
            tempo_variance=10,
            swing_amount=0.15,
            syncopation_density=0.3,
            hihat_subdivision=8,
            kick_pattern=KickPattern.BOOM_BAP,
            snare_position=SnarePosition.TWO_FOUR,
            ghost_note_density=0.2,
            polyrhythm_layers=1,
            groove_template="lofi_lazy",
            time_feel=TimeFeel.BEHIND,
            rhythmic_complexity=0.3,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["minor", "dorian"],
            chord_complexity=0.7,  # 7ths and 9ths
            harmonic_rhythm=0.5,
            tension_profile=TensionProfile.GENTLE_SUSPEND,
            bass_behavior="root",
            voicing_style=VoicingStyle.ROOTLESS,
            primary_cadences=["ii7-V7-Imaj7", "IVmaj7-iii7-vi7"],
            avoid_cadences=[],
            borrowed_chords=["bVII", "bVI", "iv"],
            use_secondary_dominants=True,
            use_tritone_subs=False,
            common_progressions=[
                ["ii7", "V7", "Imaj7", "vi7"],
                ["Imaj7", "vi7", "ii7", "V7"],
                ["IVmaj7", "iii7", "vi7", "V7"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["minor_pentatonic", "dorian"],
            note_density=0.4,
            leap_probability=0.2,
            max_interval=7,
            phrase_lengths=[2, 4],
            use_call_response=False,
            motif_repetition=0.6,
            ornamentation=["pitch_drift", "lazy_release"],
            rhythmic_motif="syncopated",
            contour_type=ContourType.WAVE,
            target_notes="chord_tones",
            articulation="legato",
        ),
        timbre=TimbreDNA(
            primary_instruments=["piano", "rhodes", "guitar_clean"],
            secondary_instruments=["vinyl_noise", "tape_hiss", "ambient_pad"],
            texture_density=0.4,
            frequency_balance="mid_focus",
            stereo_width=0.6,
            reverb_character="room",
        ),
        structure=StructureDNA(
            section_types=["intro", "verse", "outro"],
            typical_length_bars=64,
            intro_style="ambient",
            use_build_release=False,
            dynamic_range=DynamicRange.COMPRESSED,
            arrangement_curve="flat",
        ),
        avoid_genres=["trap", "drill", "techno"],
        compatible_genres=["boom-bap", "jazz", "ambient"],
    ),

    "trap": GenreDNA(
        genre_id="trap",
        genre_name="Trap",
        parent_genre="hip-hop",
        rhythm=RhythmDNA(
            tempo_center=145,
            tempo_variance=20,
            swing_amount=0.0,
            syncopation_density=0.2,
            hihat_subdivision=32,
            kick_pattern=KickPattern.TRAP,
            snare_position=SnarePosition.THREE,
            ghost_note_density=0.1,
            polyrhythm_layers=1,
            groove_template="trap_bounce",
            time_feel=TimeFeel.STRAIGHT,
            rhythmic_complexity=0.5,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["minor", "phrygian"],
            chord_complexity=0.3,
            harmonic_rhythm=0.25,
            tension_profile=TensionProfile.DARK_UNRESOLVED,
            bass_behavior="glide",
            voicing_style=VoicingStyle.POWER,
            primary_cadences=["VII-i"],
            avoid_cadences=["V-I"],
            borrowed_chords=["bII"],
            use_secondary_dominants=False,
            use_tritone_subs=False,
            common_progressions=[
                ["i", "i", "i", "i"],
                ["i", "VI", "III", "VII"],
                ["i", "VII", "VI", "VII"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["minor_pentatonic", "phrygian"],
            note_density=0.3,
            leap_probability=0.3,
            max_interval=7,
            phrase_lengths=[4, 8],
            use_call_response=False,
            motif_repetition=0.7,
            ornamentation=["pitch_bend_down", "staccato"],
            rhythmic_motif="triplet",
            contour_type=ContourType.DESCENT,
            target_notes="chord_tones",
            articulation="staccato",
        ),
        timbre=TimbreDNA(
            primary_instruments=["808_bass", "trap_lead", "bell"],
            secondary_instruments=["dark_pad", "fx_riser", "vocal_chop"],
            texture_density=0.5,
            frequency_balance="bass_heavy",
            stereo_width=0.7,
            reverb_character="hall",
        ),
        structure=StructureDNA(
            section_types=["intro", "verse", "hook", "bridge", "outro"],
            typical_length_bars=128,
            intro_style="drop",
            use_build_release=True,
            dynamic_range=DynamicRange.COMPRESSED,
            arrangement_curve="build",
        ),
        avoid_genres=["lofi-hip-hop", "jazz", "ambient"],
        compatible_genres=["drill", "reggaeton"],
    ),

    "drill": GenreDNA(
        genre_id="drill",
        genre_name="Drill",
        parent_genre="hip-hop",
        rhythm=RhythmDNA(
            tempo_center=142,
            tempo_variance=5,
            swing_amount=0.0,
            syncopation_density=0.4,
            hihat_subdivision=16,
            kick_pattern=KickPattern.UK_DRILL,
            snare_position=SnarePosition.ROLLING,
            ghost_note_density=0.3,
            polyrhythm_layers=1,
            groove_template="drill_slide",
            time_feel=TimeFeel.STRAIGHT,
            rhythmic_complexity=0.6,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["phrygian", "phrygian_dominant"],
            chord_complexity=0.4,
            harmonic_rhythm=0.5,
            tension_profile=TensionProfile.DARK_UNRESOLVED,
            bass_behavior="glide",
            voicing_style=VoicingStyle.CLOSE,
            primary_cadences=["bII-i"],
            avoid_cadences=["V-I"],
            borrowed_chords=["bII", "bVII"],
            use_secondary_dominants=False,
            use_tritone_subs=False,
            common_progressions=[
                ["i", "bII", "i", "bII"],
                ["i", "VII", "bVI", "bII"],
                ["i", "bII", "VII", "i"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["phrygian_dominant", "harmonic_minor"],
            note_density=0.5,
            leap_probability=0.25,
            max_interval=5,
            phrase_lengths=[2, 4],
            use_call_response=False,
            motif_repetition=0.5,
            ornamentation=["slide_up", "slide_down", "pitch_warble"],
            rhythmic_motif="triplet",
            contour_type=ContourType.WAVE,
            target_notes="tensions",
            articulation="mixed",
        ),
        timbre=TimbreDNA(
            primary_instruments=["808_slide", "drill_lead", "piano_dark"],
            secondary_instruments=["string_stab", "fx_impact", "vocal_chop"],
            texture_density=0.5,
            frequency_balance="bass_heavy",
            stereo_width=0.6,
            reverb_character="room",
        ),
        structure=StructureDNA(
            section_types=["intro", "verse", "hook", "outro"],
            typical_length_bars=96,
            intro_style="direct",
            use_build_release=False,
            dynamic_range=DynamicRange.COMPRESSED,
            arrangement_curve="flat",
        ),
        avoid_genres=["lofi-hip-hop", "house", "pop"],
        compatible_genres=["trap", "afrobeat"],
    ),

    "boom-bap": GenreDNA(
        genre_id="boom-bap",
        genre_name="Boom Bap",
        parent_genre="hip-hop",
        rhythm=RhythmDNA(
            tempo_center=92,
            tempo_variance=8,
            swing_amount=0.18,
            syncopation_density=0.4,
            hihat_subdivision=16,
            kick_pattern=KickPattern.BOOM_BAP,
            snare_position=SnarePosition.TWO_FOUR,
            ghost_note_density=0.4,
            polyrhythm_layers=1,
            groove_template="boom_bap_swing",
            time_feel=TimeFeel.SWING,
            rhythmic_complexity=0.5,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["minor", "dorian", "mixolydian"],
            chord_complexity=0.5,
            harmonic_rhythm=0.5,
            tension_profile=TensionProfile.RESOLVE_QUICK,
            bass_behavior="root",
            voicing_style=VoicingStyle.SHELL,
            primary_cadences=["ii-V-I", "iv-V-i"],
            avoid_cadences=[],
            borrowed_chords=["bVII", "IV"],
            use_secondary_dominants=True,
            use_tritone_subs=False,
            common_progressions=[
                ["i", "iv", "i", "V"],
                ["ii7", "V7", "I", "vi"],
                ["i", "VII", "VI", "V"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["minor_pentatonic", "blues"],
            note_density=0.6,
            leap_probability=0.3,
            max_interval=7,
            phrase_lengths=[2, 4],
            use_call_response=True,
            motif_repetition=0.5,
            ornamentation=["sample_chop", "vinyl_scratch"],
            rhythmic_motif="syncopated",
            contour_type=ContourType.ARCH,
            target_notes="chord_tones",
            articulation="mixed",
        ),
        timbre=TimbreDNA(
            primary_instruments=["sampled_drums", "bass_synth", "piano_sample"],
            secondary_instruments=["horn_stab", "scratch", "vocal_sample"],
            texture_density=0.5,
            frequency_balance="mid_focus",
            stereo_width=0.5,
            reverb_character="dry",
        ),
        structure=StructureDNA(
            section_types=["intro", "verse", "hook", "bridge", "outro"],
            typical_length_bars=96,
            intro_style="direct",
            use_build_release=False,
            dynamic_range=DynamicRange.MODERATE,
            arrangement_curve="wave",
        ),
        avoid_genres=["trap", "techno", "house"],
        compatible_genres=["lofi-hip-hop", "jazz"],
    ),

    # -------------------------------------------------------------------------
    # ELECTRONIC FAMILY
    # -------------------------------------------------------------------------

    "synthwave": GenreDNA(
        genre_id="synthwave",
        genre_name="Synthwave",
        parent_genre="electronic",
        rhythm=RhythmDNA(
            tempo_center=105,
            tempo_variance=15,
            swing_amount=0.0,
            syncopation_density=0.2,
            hihat_subdivision=16,
            kick_pattern=KickPattern.FOUR_ON_FLOOR,
            snare_position=SnarePosition.TWO_FOUR,
            ghost_note_density=0.1,
            polyrhythm_layers=1,
            groove_template="synthwave_drive",
            time_feel=TimeFeel.STRAIGHT,
            rhythmic_complexity=0.3,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["minor", "dorian"],
            chord_complexity=0.4,
            harmonic_rhythm=0.5,
            tension_profile=TensionProfile.RESOLVE_QUICK,
            bass_behavior="root",
            voicing_style=VoicingStyle.POWER,
            primary_cadences=["VI-VII-i", "iv-V-i"],
            avoid_cadences=[],
            borrowed_chords=["IV", "bVII"],
            use_secondary_dominants=False,
            use_tritone_subs=False,
            common_progressions=[
                ["i", "VI", "III", "VII"],
                ["i", "iv", "VI", "V"],
                ["I", "V", "vi", "IV"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["natural_minor", "dorian"],
            note_density=0.6,
            leap_probability=0.4,
            max_interval=12,
            phrase_lengths=[4, 8],
            use_call_response=False,
            motif_repetition=0.7,
            ornamentation=["arpeggio", "portamento"],
            rhythmic_motif="on_beat",
            contour_type=ContourType.ARCH,
            target_notes="chord_tones",
            articulation="legato",
        ),
        timbre=TimbreDNA(
            primary_instruments=["analog_pad", "synth_bass", "lead_saw"],
            secondary_instruments=["arpeggiator", "gated_reverb_drums", "choir_pad"],
            texture_density=0.6,
            frequency_balance="mid_focus",
            stereo_width=0.8,
            reverb_character="hall",
        ),
        structure=StructureDNA(
            section_types=["intro", "verse", "chorus", "bridge", "outro"],
            typical_length_bars=128,
            intro_style="buildup",
            use_build_release=True,
            dynamic_range=DynamicRange.MODERATE,
            arrangement_curve="build",
        ),
        avoid_genres=["trap", "drill", "afrobeat"],
        compatible_genres=["house", "pop", "cinematic"],
    ),

    "house": GenreDNA(
        genre_id="house",
        genre_name="House",
        parent_genre="electronic",
        rhythm=RhythmDNA(
            tempo_center=124,
            tempo_variance=6,
            swing_amount=0.08,
            syncopation_density=0.5,
            hihat_subdivision=16,
            kick_pattern=KickPattern.FOUR_ON_FLOOR,
            snare_position=SnarePosition.OFFBEAT,
            ghost_note_density=0.2,
            polyrhythm_layers=1,
            groove_template="house_groove",
            time_feel=TimeFeel.AHEAD,
            rhythmic_complexity=0.4,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["minor", "dorian"],
            chord_complexity=0.5,
            harmonic_rhythm=1.0,
            tension_profile=TensionProfile.EUPHORIC_RELEASE,
            bass_behavior="root",
            voicing_style=VoicingStyle.STAB,
            primary_cadences=["VII-i", "IV-V-I"],
            avoid_cadences=[],
            borrowed_chords=["IV", "bVII"],
            use_secondary_dominants=False,
            use_tritone_subs=False,
            common_progressions=[
                ["i", "VI", "III", "VII"],
                ["i", "iv", "VII", "III"],
                ["I", "V", "vi", "IV"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["natural_minor", "pentatonic"],
            note_density=0.5,
            leap_probability=0.3,
            max_interval=7,
            phrase_lengths=[4, 8],
            use_call_response=True,
            motif_repetition=0.6,
            ornamentation=["filter_sweep", "stab_decay"],
            rhythmic_motif="syncopated",
            contour_type=ContourType.WAVE,
            target_notes="chord_tones",
            articulation="staccato",
        ),
        timbre=TimbreDNA(
            primary_instruments=["house_kick", "piano_stab", "synth_bass"],
            secondary_instruments=["vocal_chop", "clap", "shaker"],
            texture_density=0.5,
            frequency_balance="bass_heavy",
            stereo_width=0.7,
            reverb_character="room",
        ),
        structure=StructureDNA(
            section_types=["intro", "buildup", "drop", "breakdown", "outro"],
            typical_length_bars=128,
            intro_style="buildup",
            use_build_release=True,
            dynamic_range=DynamicRange.COMPRESSED,
            arrangement_curve="wave",
        ),
        avoid_genres=["boom-bap", "drill", "cinematic"],
        compatible_genres=["techno", "synthwave", "pop"],
    ),

    "techno": GenreDNA(
        genre_id="techno",
        genre_name="Techno",
        parent_genre="electronic",
        rhythm=RhythmDNA(
            tempo_center=132,
            tempo_variance=10,
            swing_amount=0.0,
            syncopation_density=0.3,
            hihat_subdivision=16,
            kick_pattern=KickPattern.FOUR_ON_FLOOR,
            snare_position=SnarePosition.OFFBEAT,
            ghost_note_density=0.3,
            polyrhythm_layers=2,
            groove_template="techno_hypnotic",
            time_feel=TimeFeel.STRAIGHT,
            rhythmic_complexity=0.5,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["phrygian", "locrian"],
            chord_complexity=0.2,
            harmonic_rhythm=0.125,
            tension_profile=TensionProfile.STATIC,
            bass_behavior="pedal",
            voicing_style=VoicingStyle.POWER,
            primary_cadences=[],
            avoid_cadences=["V-I"],
            borrowed_chords=[],
            use_secondary_dominants=False,
            use_tritone_subs=False,
            common_progressions=[
                ["i", "i", "i", "i"],
                ["i", "bII", "i", "bII"],
                ["i", "iv", "i", "iv"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["phrygian", "chromatic"],
            note_density=0.3,
            leap_probability=0.2,
            max_interval=5,
            phrase_lengths=[8, 16],
            use_call_response=False,
            motif_repetition=0.9,
            ornamentation=["filter_mod", "delay_feedback"],
            rhythmic_motif="on_beat",
            contour_type=ContourType.STATIC,
            target_notes="root",
            articulation="staccato",
        ),
        timbre=TimbreDNA(
            primary_instruments=["techno_kick", "industrial_perc", "acid_bass"],
            secondary_instruments=["noise_sweep", "metallic_hit", "reverb_tail"],
            texture_density=0.6,
            frequency_balance="bass_heavy",
            stereo_width=0.5,
            reverb_character="hall",
        ),
        structure=StructureDNA(
            section_types=["intro", "build", "peak", "breakdown", "outro"],
            typical_length_bars=192,
            intro_style="buildup",
            use_build_release=True,
            dynamic_range=DynamicRange.COMPRESSED,
            arrangement_curve="build",
        ),
        avoid_genres=["lofi-hip-hop", "pop", "jazz"],
        compatible_genres=["house", "drum-and-bass"],
    ),

    "drum-and-bass": GenreDNA(
        genre_id="drum-and-bass",
        genre_name="Drum & Bass",
        parent_genre="electronic",
        rhythm=RhythmDNA(
            tempo_center=174,
            tempo_variance=6,
            swing_amount=0.0,
            syncopation_density=0.7,
            hihat_subdivision=16,
            kick_pattern=KickPattern.BREAKBEAT,
            snare_position=SnarePosition.OFFBEAT,
            ghost_note_density=0.5,
            polyrhythm_layers=2,
            groove_template="amen_break",
            time_feel=TimeFeel.STRAIGHT,
            rhythmic_complexity=0.8,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["minor", "dorian"],
            chord_complexity=0.4,
            harmonic_rhythm=0.25,
            tension_profile=TensionProfile.SUSPEND,
            bass_behavior="glide",
            voicing_style=VoicingStyle.SPREAD,
            primary_cadences=["iv-VII-i"],
            avoid_cadences=[],
            borrowed_chords=["bVII", "bVI"],
            use_secondary_dominants=False,
            use_tritone_subs=False,
            common_progressions=[
                ["i", "VII", "VI", "VII"],
                ["i", "iv", "VII", "i"],
                ["i", "VI", "iv", "VII"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["natural_minor", "pentatonic"],
            note_density=0.7,
            leap_probability=0.4,
            max_interval=12,
            phrase_lengths=[2, 4],
            use_call_response=True,
            motif_repetition=0.5,
            ornamentation=["reese_unison", "octave_jump"],
            rhythmic_motif="syncopated",
            contour_type=ContourType.WAVE,
            target_notes="chord_tones",
            articulation="mixed",
        ),
        timbre=TimbreDNA(
            primary_instruments=["break_drums", "reese_bass", "pad"],
            secondary_instruments=["vocal_chop", "stab", "fx_riser"],
            texture_density=0.7,
            frequency_balance="bass_heavy",
            stereo_width=0.7,
            reverb_character="room",
        ),
        structure=StructureDNA(
            section_types=["intro", "drop", "breakdown", "drop2", "outro"],
            typical_length_bars=128,
            intro_style="buildup",
            use_build_release=True,
            dynamic_range=DynamicRange.MODERATE,
            arrangement_curve="wave",
        ),
        avoid_genres=["lofi-hip-hop", "ambient"],
        compatible_genres=["techno", "house"],
    ),

    # -------------------------------------------------------------------------
    # LATIN / WORLD
    # -------------------------------------------------------------------------

    "reggaeton": GenreDNA(
        genre_id="reggaeton",
        genre_name="Reggaeton",
        parent_genre="latin",
        rhythm=RhythmDNA(
            tempo_center=95,
            tempo_variance=7,
            swing_amount=0.0,
            syncopation_density=0.3,
            hihat_subdivision=16,
            kick_pattern=KickPattern.DEMBOW,
            snare_position=SnarePosition.TWO_FOUR,
            ghost_note_density=0.2,
            polyrhythm_layers=1,
            groove_template="dembow",
            time_feel=TimeFeel.STRAIGHT,
            rhythmic_complexity=0.4,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["minor"],
            chord_complexity=0.3,
            harmonic_rhythm=0.5,
            tension_profile=TensionProfile.RESOLVE_QUICK,
            bass_behavior="root",
            voicing_style=VoicingStyle.CLOSE,
            primary_cadences=["iv-V-i"],
            avoid_cadences=[],
            borrowed_chords=["bVII"],
            use_secondary_dominants=False,
            use_tritone_subs=False,
            common_progressions=[
                ["i", "VI", "III", "VII"],
                ["i", "iv", "VII", "III"],
                ["i", "VII", "VI", "VII"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["natural_minor", "harmonic_minor"],
            note_density=0.4,
            leap_probability=0.3,
            max_interval=7,
            phrase_lengths=[2, 4],
            use_call_response=True,
            motif_repetition=0.6,
            ornamentation=["vocal_style", "melisma"],
            rhythmic_motif="on_beat",
            contour_type=ContourType.WAVE,
            target_notes="chord_tones",
            articulation="legato",
        ),
        timbre=TimbreDNA(
            primary_instruments=["dembow_drums", "synth_bass", "pluck"],
            secondary_instruments=["vocal_chop", "brass_stab", "fx"],
            texture_density=0.5,
            frequency_balance="bass_heavy",
            stereo_width=0.6,
            reverb_character="room",
        ),
        structure=StructureDNA(
            section_types=["intro", "verse", "chorus", "bridge", "outro"],
            typical_length_bars=96,
            intro_style="direct",
            use_build_release=False,
            dynamic_range=DynamicRange.COMPRESSED,
            arrangement_curve="flat",
        ),
        avoid_genres=["techno", "ambient", "jazz"],
        compatible_genres=["trap", "pop", "house"],
    ),

    "afrobeat": GenreDNA(
        genre_id="afrobeat",
        genre_name="Afrobeat",
        parent_genre="world",
        rhythm=RhythmDNA(
            tempo_center=108,
            tempo_variance=12,
            swing_amount=0.0,
            syncopation_density=0.7,
            hihat_subdivision=16,
            kick_pattern=KickPattern.AFROBEAT,
            snare_position=SnarePosition.POLYRHYTHMIC,
            ghost_note_density=0.5,
            polyrhythm_layers=3,
            groove_template="afrobeat_12_8",
            time_feel=TimeFeel.TRIPLET,
            rhythmic_complexity=0.8,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["dorian", "mixolydian"],
            chord_complexity=0.5,
            harmonic_rhythm=0.5,
            tension_profile=TensionProfile.RESOLVE_QUICK,
            bass_behavior="root",
            voicing_style=VoicingStyle.SPREAD,
            primary_cadences=["V7-I", "IV-I"],
            avoid_cadences=[],
            borrowed_chords=["bVII", "II7"],
            use_secondary_dominants=True,
            use_tritone_subs=False,
            common_progressions=[
                ["I", "IV", "I", "V"],
                ["i", "iv", "i", "VII"],
                ["I7", "IV7", "I7", "V7"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["dorian", "mixolydian", "pentatonic"],
            note_density=0.7,
            leap_probability=0.3,
            max_interval=7,
            phrase_lengths=[2, 4],
            use_call_response=True,
            motif_repetition=0.5,
            ornamentation=["horn_stab", "guitar_lick"],
            rhythmic_motif="syncopated",
            contour_type=ContourType.ARCH,
            target_notes="chord_tones",
            articulation="mixed",
        ),
        timbre=TimbreDNA(
            primary_instruments=["african_drums", "bass_guitar", "horn_section"],
            secondary_instruments=["rhythm_guitar", "organ", "shaker"],
            texture_density=0.7,
            frequency_balance="mid_focus",
            stereo_width=0.6,
            reverb_character="room",
        ),
        structure=StructureDNA(
            section_types=["intro", "groove", "verse", "chorus", "breakdown", "outro"],
            typical_length_bars=192,
            intro_style="direct",
            use_build_release=False,
            dynamic_range=DynamicRange.MODERATE,
            arrangement_curve="wave",
        ),
        avoid_genres=["techno", "ambient"],
        compatible_genres=["house", "reggaeton", "pop"],
    ),

    # -------------------------------------------------------------------------
    # POP / CINEMATIC
    # -------------------------------------------------------------------------

    "pop": GenreDNA(
        genre_id="pop",
        genre_name="Pop",
        parent_genre="pop",
        rhythm=RhythmDNA(
            tempo_center=115,
            tempo_variance=20,
            swing_amount=0.0,
            syncopation_density=0.3,
            hihat_subdivision=8,
            kick_pattern=KickPattern.FOUR_ON_FLOOR,
            snare_position=SnarePosition.TWO_FOUR,
            ghost_note_density=0.2,
            polyrhythm_layers=1,
            groove_template="pop_standard",
            time_feel=TimeFeel.STRAIGHT,
            rhythmic_complexity=0.3,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["major", "minor"],
            chord_complexity=0.4,
            harmonic_rhythm=1.0,
            tension_profile=TensionProfile.RESOLVE_QUICK,
            bass_behavior="root",
            voicing_style=VoicingStyle.CLOSE,
            primary_cadences=["V-I", "IV-I"],
            avoid_cadences=[],
            borrowed_chords=["bVII", "iv"],
            use_secondary_dominants=True,
            use_tritone_subs=False,
            common_progressions=[
                ["I", "V", "vi", "IV"],
                ["I", "IV", "vi", "V"],
                ["vi", "IV", "I", "V"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["major", "pentatonic"],
            note_density=0.6,
            leap_probability=0.4,
            max_interval=10,
            phrase_lengths=[4, 8],
            use_call_response=True,
            motif_repetition=0.7,
            ornamentation=["clean", "vocal_friendly"],
            rhythmic_motif="on_beat",
            contour_type=ContourType.ARCH,
            target_notes="chord_tones",
            articulation="legato",
        ),
        timbre=TimbreDNA(
            primary_instruments=["piano", "synth_pad", "acoustic_guitar"],
            secondary_instruments=["strings", "brass", "vocal"],
            texture_density=0.5,
            frequency_balance="balanced",
            stereo_width=0.7,
            reverb_character="room",
        ),
        structure=StructureDNA(
            section_types=["intro", "verse", "prechorus", "chorus", "bridge", "outro"],
            typical_length_bars=96,
            intro_style="direct",
            use_build_release=True,
            dynamic_range=DynamicRange.MODERATE,
            arrangement_curve="build",
        ),
        avoid_genres=["techno", "drill"],
        compatible_genres=["house", "synthwave", "afrobeat"],
    ),

    "cinematic": GenreDNA(
        genre_id="cinematic",
        genre_name="Cinematic",
        parent_genre="orchestral",
        rhythm=RhythmDNA(
            tempo_center=90,
            tempo_variance=50,
            swing_amount=0.0,
            syncopation_density=0.2,
            hihat_subdivision=8,
            kick_pattern=KickPattern.CINEMATIC,
            snare_position=SnarePosition.SPARSE,
            ghost_note_density=0.1,
            polyrhythm_layers=1,
            groove_template="cinematic_rubato",
            time_feel=TimeFeel.STRAIGHT,
            rhythmic_complexity=0.4,
        ),
        harmony=HarmonyDNA(
            mode_preferences=["minor", "harmonic_minor", "phrygian"],
            chord_complexity=0.7,
            harmonic_rhythm=0.5,
            tension_profile=TensionProfile.SUSPEND,
            bass_behavior="pedal",
            voicing_style=VoicingStyle.SPREAD,
            primary_cadences=["VI-VII-i", "iv-V"],
            avoid_cadences=[],
            borrowed_chords=["bVI", "bIII", "bVII", "#iv°"],
            use_secondary_dominants=True,
            use_tritone_subs=True,
            common_progressions=[
                ["i", "VI", "III", "VII"],
                ["i", "v", "VI", "III"],
                ["I", "iii", "vi", "IV"],
            ],
        ),
        melody=MelodyDNA(
            scale_preferences=["harmonic_minor", "natural_minor", "dorian"],
            note_density=0.4,
            leap_probability=0.5,
            max_interval=12,
            phrase_lengths=[4, 8, 16],
            use_call_response=True,
            motif_repetition=0.6,
            ornamentation=["legato_strings", "brass_swells"],
            rhythmic_motif="on_beat",
            contour_type=ContourType.ARCH,
            target_notes="chord_tones",
            articulation="legato",
        ),
        timbre=TimbreDNA(
            primary_instruments=["strings", "brass", "piano"],
            secondary_instruments=["choir", "percussion_epic", "synth_pad"],
            texture_density=0.6,
            frequency_balance="balanced",
            stereo_width=0.9,
            reverb_character="hall",
        ),
        structure=StructureDNA(
            section_types=["opening", "development", "climax", "resolution", "coda"],
            typical_length_bars=128,
            intro_style="ambient",
            use_build_release=True,
            dynamic_range=DynamicRange.CINEMATIC,
            arrangement_curve="build",
        ),
        avoid_genres=["trap", "drill", "reggaeton"],
        compatible_genres=["ambient", "synthwave", "pop"],
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_genre_dna(genre_id: str) -> GenreDNA:
    """Get GenreDNA for a specific genre."""
    if genre_id not in GENRE_DNA_LIBRARY:
        raise ValueError(f"Unknown genre: {genre_id}. Available: {list(GENRE_DNA_LIBRARY.keys())}")
    return GENRE_DNA_LIBRARY[genre_id]


def list_genres() -> list[str]:
    """List all available genre IDs."""
    return list(GENRE_DNA_LIBRARY.keys())


def get_genre_vector(genre_id: str) -> list[float]:
    """Get 48-dimensional vector for a genre."""
    return get_genre_dna(genre_id).to_vector()


def compute_genre_similarity(genre_a: str, genre_b: str) -> float:
    """Compute cosine similarity between two genre DNA vectors."""
    import math

    vec_a = get_genre_vector(genre_a)
    vec_b = get_genre_vector(genre_b)

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)
