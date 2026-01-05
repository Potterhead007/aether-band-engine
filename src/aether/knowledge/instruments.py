"""
AETHER Instrument Library

Comprehensive instrument knowledge base covering:
- General MIDI (GM) standard instruments
- Analog/acoustic instruments (orchestral, acoustic, vintage)
- Digital/electronic instruments (synths, drum machines)
- Genre-specific instrument palettes
- Articulations and playing techniques
- Frequency ranges and timbral characteristics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================================
# Enums
# ============================================================================

class InstrumentFamily(Enum):
    """High-level instrument family classification."""
    KEYBOARDS = "keyboards"
    STRINGS = "strings"
    BRASS = "brass"
    WOODWINDS = "woodwinds"
    PERCUSSION = "percussion"
    SYNTH = "synth"
    BASS = "bass"
    GUITAR = "guitar"
    VOCALS = "vocals"
    ETHNIC = "ethnic"
    SFX = "sfx"


class InstrumentEra(Enum):
    """Historical era of the instrument."""
    CLASSICAL = "classical"  # Pre-1900
    VINTAGE = "vintage"      # 1900-1970
    MODERN = "modern"        # 1970-2000
    CONTEMPORARY = "contemporary"  # 2000+


class InstrumentType(Enum):
    """Analog vs Digital classification."""
    ACOUSTIC = "acoustic"
    ELECTRIC = "electric"
    ELECTRONIC = "electronic"
    HYBRID = "hybrid"


class Articulation(Enum):
    """Playing articulations."""
    SUSTAIN = "sustain"
    STACCATO = "staccato"
    LEGATO = "legato"
    PIZZICATO = "pizzicato"
    TREMOLO = "tremolo"
    VIBRATO = "vibrato"
    GLISSANDO = "glissando"
    PORTAMENTO = "portamento"
    SPICCATO = "spiccato"
    MARCATO = "marcato"
    TENUTO = "tenuto"
    SFORZANDO = "sforzando"
    HARMONICS = "harmonics"
    MUTED = "muted"
    PALM_MUTE = "palm_mute"
    HAMMER_ON = "hammer_on"
    PULL_OFF = "pull_off"
    BEND = "bend"
    SLIDE = "slide"
    WAH = "wah"
    SLAP = "slap"
    POP = "pop"
    GHOST_NOTE = "ghost_note"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FrequencyRange:
    """Frequency range of an instrument."""
    low_hz: float
    high_hz: float
    sweet_spot_low_hz: float
    sweet_spot_high_hz: float

    @property
    def low_midi(self) -> int:
        """Convert low frequency to MIDI note."""
        import math
        return int(69 + 12 * math.log2(self.low_hz / 440))

    @property
    def high_midi(self) -> int:
        """Convert high frequency to MIDI note."""
        import math
        return int(69 + 12 * math.log2(self.high_hz / 440))


@dataclass
class Instrument:
    """Complete instrument definition."""
    id: str
    name: str
    family: InstrumentFamily
    instrument_type: InstrumentType
    era: InstrumentEra
    gm_program: Optional[int]  # General MIDI program number (0-127)
    gm_bank: int = 0

    # Range
    frequency_range: Optional[FrequencyRange] = None
    typical_octave: int = 4

    # Characteristics
    articulations: list[Articulation] = field(default_factory=list)
    polyphonic: bool = True
    max_polyphony: int = 16

    # Timbral
    attack_time_ms: float = 10.0
    decay_time_ms: float = 100.0
    sustain_level: float = 0.7
    release_time_ms: float = 200.0
    brightness: float = 0.5  # 0-1
    warmth: float = 0.5  # 0-1

    # Genre affinity (0-1 for each genre)
    genre_affinity: dict[str, float] = field(default_factory=dict)

    # Layering
    layer_priority: int = 5  # 1-10, higher = more prominent
    can_double: bool = True
    doubles_well_with: list[str] = field(default_factory=list)

    # Description
    description: str = ""
    playing_tips: list[str] = field(default_factory=list)


# ============================================================================
# General MIDI Instrument Definitions
# ============================================================================

# Piano Family (0-7)
GM_INSTRUMENTS: dict[int, Instrument] = {
    0: Instrument(
        id="acoustic_grand_piano",
        name="Acoustic Grand Piano",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=0,
        frequency_range=FrequencyRange(27.5, 4186, 65, 1047),
        articulations=[Articulation.SUSTAIN, Articulation.STACCATO, Articulation.LEGATO],
        attack_time_ms=5,
        brightness=0.6,
        warmth=0.7,
        genre_affinity={"classical": 1.0, "jazz": 0.9, "pop": 0.7, "ballad": 0.9},
        description="Full concert grand piano with rich harmonics",
    ),
    1: Instrument(
        id="bright_acoustic_piano",
        name="Bright Acoustic Piano",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=1,
        frequency_range=FrequencyRange(27.5, 4186, 65, 1047),
        brightness=0.8,
        warmth=0.5,
        genre_affinity={"pop": 0.8, "rock": 0.6},
    ),
    2: Instrument(
        id="electric_grand_piano",
        name="Electric Grand Piano",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.VINTAGE,
        gm_program=2,
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"jazz": 0.8, "soul": 0.9, "r-and-b": 0.8},
    ),
    3: Instrument(
        id="honky_tonk_piano",
        name="Honky-tonk Piano",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=3,
        brightness=0.7,
        warmth=0.4,
        genre_affinity={"country": 0.9, "blues": 0.7, "ragtime": 1.0},
    ),
    4: Instrument(
        id="electric_piano_1",
        name="Electric Piano 1 (Rhodes)",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.VINTAGE,
        gm_program=4,
        brightness=0.5,
        warmth=0.8,
        genre_affinity={"jazz": 0.9, "soul": 1.0, "r-and-b": 0.9, "neo-soul": 1.0, "lo-fi-hip-hop": 0.9},
        description="Fender Rhodes electric piano - warm, bell-like tones",
    ),
    5: Instrument(
        id="electric_piano_2",
        name="Electric Piano 2 (DX)",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=5,
        brightness=0.7,
        warmth=0.4,
        genre_affinity={"pop": 0.8, "synthwave": 0.6, "80s": 1.0},
        description="Yamaha DX7 FM electric piano - crystalline, glassy",
    ),
    6: Instrument(
        id="harpsichord",
        name="Harpsichord",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=6,
        brightness=0.9,
        warmth=0.3,
        genre_affinity={"baroque": 1.0, "classical": 0.7},
    ),
    7: Instrument(
        id="clavinet",
        name="Clavinet",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.VINTAGE,
        gm_program=7,
        articulations=[Articulation.STACCATO, Articulation.WAH, Articulation.MUTED],
        brightness=0.8,
        warmth=0.5,
        genre_affinity={"funk": 1.0, "soul": 0.8, "disco": 0.7},
        description="Hohner Clavinet - funky, percussive keyboard",
    ),

    # Chromatic Percussion (8-15)
    8: Instrument(
        id="celesta",
        name="Celesta",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=8,
        frequency_range=FrequencyRange(261, 4186, 523, 2093),
        brightness=0.9,
        warmth=0.4,
        genre_affinity={"classical": 0.9, "cinematic": 0.8, "ambient": 0.6},
    ),
    9: Instrument(
        id="glockenspiel",
        name="Glockenspiel",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=9,
        brightness=1.0,
        warmth=0.2,
        genre_affinity={"classical": 0.8, "cinematic": 0.7, "indie": 0.6},
    ),
    10: Instrument(
        id="music_box",
        name="Music Box",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=10,
        brightness=0.9,
        warmth=0.3,
        genre_affinity={"ambient": 0.7, "cinematic": 0.6, "lo-fi-hip-hop": 0.5},
    ),
    11: Instrument(
        id="vibraphone",
        name="Vibraphone",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=11,
        articulations=[Articulation.SUSTAIN, Articulation.MUTED, Articulation.TREMOLO],
        brightness=0.6,
        warmth=0.7,
        genre_affinity={"jazz": 1.0, "lounge": 0.9, "ambient": 0.6},
        description="Jazz vibraphone with motor vibrato",
    ),
    12: Instrument(
        id="marimba",
        name="Marimba",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=12,
        brightness=0.4,
        warmth=0.8,
        genre_affinity={"classical": 0.8, "world": 0.9, "ambient": 0.6},
    ),
    13: Instrument(
        id="xylophone",
        name="Xylophone",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=13,
        brightness=0.9,
        warmth=0.2,
        genre_affinity={"classical": 0.8, "cinematic": 0.6},
    ),
    14: Instrument(
        id="tubular_bells",
        name="Tubular Bells",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=14,
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"classical": 0.9, "cinematic": 1.0, "ambient": 0.7},
    ),
    15: Instrument(
        id="dulcimer",
        name="Dulcimer",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=15,
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"folk": 1.0, "country": 0.7, "world": 0.6},
    ),

    # Organ (16-23)
    16: Instrument(
        id="drawbar_organ",
        name="Drawbar Organ (Hammond)",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.VINTAGE,
        gm_program=16,
        articulations=[Articulation.SUSTAIN, Articulation.VIBRATO],
        brightness=0.6,
        warmth=0.7,
        genre_affinity={"jazz": 0.9, "blues": 0.9, "soul": 1.0, "rock": 0.7, "funk": 0.8},
        description="Hammond B3 organ with Leslie speaker simulation",
    ),
    17: Instrument(
        id="percussive_organ",
        name="Percussive Organ",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.VINTAGE,
        gm_program=17,
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"rock": 0.8, "pop": 0.6},
    ),
    18: Instrument(
        id="rock_organ",
        name="Rock Organ",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.VINTAGE,
        gm_program=18,
        brightness=0.8,
        warmth=0.5,
        genre_affinity={"rock": 1.0, "blues": 0.7},
    ),
    19: Instrument(
        id="church_organ",
        name="Church Organ (Pipe)",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=19,
        brightness=0.5,
        warmth=0.6,
        genre_affinity={"classical": 1.0, "cinematic": 0.8, "gothic": 0.9},
    ),
    20: Instrument(
        id="reed_organ",
        name="Reed Organ",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=20,
        brightness=0.5,
        warmth=0.6,
        genre_affinity={"folk": 0.7, "country": 0.6},
    ),
    21: Instrument(
        id="accordion",
        name="Accordion",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=21,
        brightness=0.6,
        warmth=0.7,
        genre_affinity={"folk": 0.9, "world": 0.8, "tango": 1.0, "polka": 1.0},
    ),
    22: Instrument(
        id="harmonica",
        name="Harmonica",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=22,
        articulations=[Articulation.BEND, Articulation.VIBRATO, Articulation.TREMOLO],
        brightness=0.6,
        warmth=0.7,
        genre_affinity={"blues": 1.0, "folk": 0.8, "country": 0.7, "rock": 0.5},
    ),
    23: Instrument(
        id="tango_accordion",
        name="Tango Accordion (Bandoneon)",
        family=InstrumentFamily.KEYBOARDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=23,
        brightness=0.5,
        warmth=0.8,
        genre_affinity={"tango": 1.0, "world": 0.7},
    ),

    # Guitar (24-31)
    24: Instrument(
        id="acoustic_guitar_nylon",
        name="Acoustic Guitar (Nylon/Classical)",
        family=InstrumentFamily.GUITAR,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=24,
        frequency_range=FrequencyRange(82, 880, 110, 660),
        articulations=[Articulation.SUSTAIN, Articulation.STACCATO, Articulation.HARMONICS, Articulation.TREMOLO],
        brightness=0.4,
        warmth=0.9,
        genre_affinity={"classical": 1.0, "flamenco": 1.0, "bossa-nova": 1.0, "latin": 0.9, "folk": 0.7},
    ),
    25: Instrument(
        id="acoustic_guitar_steel",
        name="Acoustic Guitar (Steel)",
        family=InstrumentFamily.GUITAR,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=25,
        frequency_range=FrequencyRange(82, 880, 110, 660),
        articulations=[Articulation.SUSTAIN, Articulation.STACCATO, Articulation.HAMMER_ON, Articulation.PULL_OFF, Articulation.SLIDE],
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"folk": 1.0, "country": 1.0, "pop": 0.8, "rock": 0.6, "acoustic-folk": 1.0},
    ),
    26: Instrument(
        id="electric_guitar_jazz",
        name="Electric Guitar (Jazz)",
        family=InstrumentFamily.GUITAR,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.VINTAGE,
        gm_program=26,
        articulations=[Articulation.SUSTAIN, Articulation.LEGATO, Articulation.VIBRATO],
        brightness=0.4,
        warmth=0.8,
        genre_affinity={"jazz": 1.0, "blues": 0.8, "soul": 0.7},
        description="Hollow-body jazz guitar - warm, round tone",
    ),
    27: Instrument(
        id="electric_guitar_clean",
        name="Electric Guitar (Clean)",
        family=InstrumentFamily.GUITAR,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.MODERN,
        gm_program=27,
        articulations=[Articulation.SUSTAIN, Articulation.STACCATO, Articulation.HAMMER_ON, Articulation.PULL_OFF],
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"pop": 0.9, "funk": 0.8, "r-and-b": 0.7, "indie": 0.8},
    ),
    28: Instrument(
        id="electric_guitar_muted",
        name="Electric Guitar (Muted)",
        family=InstrumentFamily.GUITAR,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.MODERN,
        gm_program=28,
        articulations=[Articulation.PALM_MUTE, Articulation.STACCATO],
        brightness=0.5,
        warmth=0.5,
        genre_affinity={"funk": 1.0, "disco": 0.8, "pop": 0.6},
    ),
    29: Instrument(
        id="electric_guitar_overdrive",
        name="Electric Guitar (Overdriven)",
        family=InstrumentFamily.GUITAR,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.MODERN,
        gm_program=29,
        articulations=[Articulation.SUSTAIN, Articulation.BEND, Articulation.VIBRATO],
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"rock": 1.0, "blues": 0.9},
    ),
    30: Instrument(
        id="electric_guitar_distortion",
        name="Electric Guitar (Distortion)",
        family=InstrumentFamily.GUITAR,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.MODERN,
        gm_program=30,
        articulations=[Articulation.SUSTAIN, Articulation.PALM_MUTE, Articulation.BEND],
        brightness=0.8,
        warmth=0.4,
        genre_affinity={"rock": 1.0, "metal": 1.0, "punk": 0.9},
    ),
    31: Instrument(
        id="electric_guitar_harmonics",
        name="Electric Guitar (Harmonics)",
        family=InstrumentFamily.GUITAR,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.MODERN,
        gm_program=31,
        articulations=[Articulation.HARMONICS],
        brightness=0.9,
        warmth=0.3,
        genre_affinity={"ambient": 0.7, "post-rock": 0.8},
    ),

    # Bass (32-39)
    32: Instrument(
        id="acoustic_bass",
        name="Acoustic Bass (Upright)",
        family=InstrumentFamily.BASS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=32,
        frequency_range=FrequencyRange(41, 294, 41, 196),
        articulations=[Articulation.SUSTAIN, Articulation.PIZZICATO, Articulation.SLIDE],
        polyphonic=False,
        brightness=0.3,
        warmth=0.9,
        genre_affinity={"jazz": 1.0, "classical": 0.9, "folk": 0.7, "rockabilly": 0.9},
        description="Double bass / upright bass - warm, woody tone",
    ),
    33: Instrument(
        id="electric_bass_finger",
        name="Electric Bass (Finger)",
        family=InstrumentFamily.BASS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.MODERN,
        gm_program=33,
        frequency_range=FrequencyRange(41, 392, 41, 294),
        articulations=[Articulation.SUSTAIN, Articulation.SLIDE, Articulation.HAMMER_ON],
        polyphonic=False,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"rock": 0.9, "pop": 0.9, "funk": 0.8, "soul": 0.9, "r-and-b": 0.9},
    ),
    34: Instrument(
        id="electric_bass_pick",
        name="Electric Bass (Pick)",
        family=InstrumentFamily.BASS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.MODERN,
        gm_program=34,
        articulations=[Articulation.STACCATO],
        polyphonic=False,
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"rock": 1.0, "punk": 1.0, "metal": 0.8},
    ),
    35: Instrument(
        id="fretless_bass",
        name="Fretless Bass",
        family=InstrumentFamily.BASS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.MODERN,
        gm_program=35,
        articulations=[Articulation.LEGATO, Articulation.SLIDE, Articulation.VIBRATO],
        polyphonic=False,
        brightness=0.4,
        warmth=0.8,
        genre_affinity={"jazz": 0.9, "fusion": 1.0, "world": 0.7},
    ),
    36: Instrument(
        id="slap_bass_1",
        name="Slap Bass 1",
        family=InstrumentFamily.BASS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.MODERN,
        gm_program=36,
        articulations=[Articulation.SLAP, Articulation.POP, Articulation.GHOST_NOTE],
        polyphonic=False,
        brightness=0.8,
        warmth=0.5,
        genre_affinity={"funk": 1.0, "disco": 0.9, "slap": 1.0, "r-and-b": 0.7},
        description="Slap bass technique - percussive, funky",
    ),
    37: Instrument(
        id="slap_bass_2",
        name="Slap Bass 2",
        family=InstrumentFamily.BASS,
        instrument_type=InstrumentType.ELECTRIC,
        era=InstrumentEra.MODERN,
        gm_program=37,
        articulations=[Articulation.SLAP, Articulation.POP],
        polyphonic=False,
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"funk": 0.9, "pop": 0.7},
    ),
    38: Instrument(
        id="synth_bass_1",
        name="Synth Bass 1",
        family=InstrumentFamily.BASS,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=38,
        polyphonic=False,
        brightness=0.6,
        warmth=0.5,
        genre_affinity={"synthwave": 1.0, "house": 0.9, "techno": 0.8, "pop": 0.7, "edm": 0.9},
    ),
    39: Instrument(
        id="synth_bass_2",
        name="Synth Bass 2",
        family=InstrumentFamily.BASS,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=39,
        polyphonic=False,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"synthwave": 0.9, "electronic": 0.8, "trap": 0.7},
    ),

    # Strings (40-47)
    40: Instrument(
        id="violin",
        name="Violin",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=40,
        frequency_range=FrequencyRange(196, 3136, 261, 1568),
        articulations=[Articulation.SUSTAIN, Articulation.LEGATO, Articulation.STACCATO,
                      Articulation.PIZZICATO, Articulation.TREMOLO, Articulation.SPICCATO,
                      Articulation.VIBRATO, Articulation.HARMONICS],
        polyphonic=False,
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"classical": 1.0, "cinematic": 0.9, "folk": 0.7, "country": 0.6},
    ),
    41: Instrument(
        id="viola",
        name="Viola",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=41,
        frequency_range=FrequencyRange(130, 1175, 174, 880),
        articulations=[Articulation.SUSTAIN, Articulation.LEGATO, Articulation.STACCATO,
                      Articulation.PIZZICATO, Articulation.TREMOLO],
        polyphonic=False,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"classical": 1.0, "cinematic": 0.8},
    ),
    42: Instrument(
        id="cello",
        name="Cello",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=42,
        frequency_range=FrequencyRange(65, 988, 98, 523),
        articulations=[Articulation.SUSTAIN, Articulation.LEGATO, Articulation.STACCATO,
                      Articulation.PIZZICATO, Articulation.TREMOLO, Articulation.VIBRATO],
        polyphonic=False,
        brightness=0.4,
        warmth=0.9,
        genre_affinity={"classical": 1.0, "cinematic": 1.0, "ambient": 0.7},
        description="Rich, expressive cello - emotional depth",
    ),
    43: Instrument(
        id="contrabass",
        name="Contrabass",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=43,
        frequency_range=FrequencyRange(41, 294, 41, 196),
        articulations=[Articulation.SUSTAIN, Articulation.PIZZICATO],
        polyphonic=False,
        brightness=0.3,
        warmth=0.9,
        genre_affinity={"classical": 1.0, "cinematic": 0.8},
    ),
    44: Instrument(
        id="tremolo_strings",
        name="Tremolo Strings",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=44,
        articulations=[Articulation.TREMOLO],
        brightness=0.6,
        warmth=0.6,
        genre_affinity={"classical": 0.8, "cinematic": 1.0, "horror": 1.0},
    ),
    45: Instrument(
        id="pizzicato_strings",
        name="Pizzicato Strings",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=45,
        articulations=[Articulation.PIZZICATO],
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"classical": 0.9, "cinematic": 0.7, "lo-fi-hip-hop": 0.5},
    ),
    46: Instrument(
        id="orchestral_harp",
        name="Orchestral Harp",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=46,
        articulations=[Articulation.SUSTAIN, Articulation.GLISSANDO],
        brightness=0.7,
        warmth=0.7,
        genre_affinity={"classical": 1.0, "cinematic": 0.9, "ambient": 0.8, "new-age": 0.9},
    ),
    47: Instrument(
        id="timpani",
        name="Timpani",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=47,
        articulations=[Articulation.SUSTAIN, Articulation.TREMOLO],
        brightness=0.4,
        warmth=0.8,
        genre_affinity={"classical": 1.0, "cinematic": 1.0},
    ),

    # Ensemble (48-55)
    48: Instrument(
        id="string_ensemble_1",
        name="String Ensemble 1",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=48,
        brightness=0.5,
        warmth=0.8,
        genre_affinity={"classical": 0.9, "cinematic": 1.0, "pop": 0.6, "ballad": 0.8},
        description="Full orchestral string section",
    ),
    49: Instrument(
        id="string_ensemble_2",
        name="String Ensemble 2 (Slow)",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=49,
        attack_time_ms=200,
        brightness=0.4,
        warmth=0.9,
        genre_affinity={"ambient": 0.9, "cinematic": 0.8},
    ),
    50: Instrument(
        id="synth_strings_1",
        name="Synth Strings 1",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=50,
        brightness=0.6,
        warmth=0.6,
        genre_affinity={"synthwave": 0.9, "disco": 0.8, "pop": 0.7, "80s": 1.0},
    ),
    51: Instrument(
        id="synth_strings_2",
        name="Synth Strings 2",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=51,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"ambient": 0.8, "synthwave": 0.7},
    ),
    52: Instrument(
        id="choir_aahs",
        name="Choir Aahs",
        family=InstrumentFamily.VOCALS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=52,
        brightness=0.5,
        warmth=0.8,
        genre_affinity={"classical": 0.9, "cinematic": 1.0, "ambient": 0.8, "new-age": 0.9},
    ),
    53: Instrument(
        id="voice_oohs",
        name="Voice Oohs",
        family=InstrumentFamily.VOCALS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=53,
        brightness=0.4,
        warmth=0.9,
        genre_affinity={"ambient": 0.9, "cinematic": 0.8, "soul": 0.6},
    ),
    54: Instrument(
        id="synth_choir",
        name="Synth Choir",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=54,
        brightness=0.6,
        warmth=0.6,
        genre_affinity={"synthwave": 0.8, "trance": 0.7, "ambient": 0.7},
    ),
    55: Instrument(
        id="orchestra_hit",
        name="Orchestra Hit",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=55,
        brightness=0.8,
        warmth=0.5,
        genre_affinity={"80s": 1.0, "hip-hop": 0.7, "edm": 0.5},
    ),

    # Brass (56-63)
    56: Instrument(
        id="trumpet",
        name="Trumpet",
        family=InstrumentFamily.BRASS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=56,
        frequency_range=FrequencyRange(165, 988, 233, 698),
        articulations=[Articulation.SUSTAIN, Articulation.STACCATO, Articulation.MUTED,
                      Articulation.SFORZANDO, Articulation.VIBRATO],
        polyphonic=False,
        brightness=0.8,
        warmth=0.5,
        genre_affinity={"jazz": 1.0, "classical": 0.9, "funk": 0.8, "latin": 0.9, "ska": 0.9},
    ),
    57: Instrument(
        id="trombone",
        name="Trombone",
        family=InstrumentFamily.BRASS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=57,
        frequency_range=FrequencyRange(82, 493, 98, 349),
        articulations=[Articulation.SUSTAIN, Articulation.STACCATO, Articulation.GLISSANDO],
        polyphonic=False,
        brightness=0.6,
        warmth=0.7,
        genre_affinity={"jazz": 1.0, "classical": 0.9, "ska": 0.9, "funk": 0.7},
    ),
    58: Instrument(
        id="tuba",
        name="Tuba",
        family=InstrumentFamily.BRASS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=58,
        frequency_range=FrequencyRange(29, 349, 41, 196),
        polyphonic=False,
        brightness=0.3,
        warmth=0.9,
        genre_affinity={"classical": 1.0, "marching": 1.0},
    ),
    59: Instrument(
        id="muted_trumpet",
        name="Muted Trumpet",
        family=InstrumentFamily.BRASS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=59,
        articulations=[Articulation.MUTED, Articulation.WAH],
        polyphonic=False,
        brightness=0.5,
        warmth=0.6,
        genre_affinity={"jazz": 1.0, "film-noir": 1.0},
    ),
    60: Instrument(
        id="french_horn",
        name="French Horn",
        family=InstrumentFamily.BRASS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=60,
        frequency_range=FrequencyRange(62, 698, 87, 523),
        polyphonic=False,
        brightness=0.5,
        warmth=0.8,
        genre_affinity={"classical": 1.0, "cinematic": 1.0},
        description="Noble, majestic French horn",
    ),
    61: Instrument(
        id="brass_section",
        name="Brass Section",
        family=InstrumentFamily.BRASS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=61,
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"funk": 1.0, "soul": 0.9, "disco": 0.8, "jazz": 0.7, "ska": 0.8},
    ),
    62: Instrument(
        id="synth_brass_1",
        name="Synth Brass 1",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=62,
        brightness=0.8,
        warmth=0.4,
        genre_affinity={"synthwave": 1.0, "80s": 1.0, "disco": 0.7},
    ),
    63: Instrument(
        id="synth_brass_2",
        name="Synth Brass 2",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=63,
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"synthwave": 0.9, "house": 0.6},
    ),

    # Reed (64-71)
    64: Instrument(
        id="soprano_sax",
        name="Soprano Saxophone",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=64,
        frequency_range=FrequencyRange(233, 1175, 277, 880),
        articulations=[Articulation.SUSTAIN, Articulation.LEGATO, Articulation.VIBRATO, Articulation.BEND],
        polyphonic=False,
        brightness=0.8,
        warmth=0.5,
        genre_affinity={"jazz": 1.0, "smooth-jazz": 0.9},
    ),
    65: Instrument(
        id="alto_sax",
        name="Alto Saxophone",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=65,
        frequency_range=FrequencyRange(139, 831, 185, 622),
        articulations=[Articulation.SUSTAIN, Articulation.LEGATO, Articulation.VIBRATO, Articulation.BEND],
        polyphonic=False,
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"jazz": 1.0, "soul": 0.8, "funk": 0.7, "r-and-b": 0.7},
    ),
    66: Instrument(
        id="tenor_sax",
        name="Tenor Saxophone",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=66,
        frequency_range=FrequencyRange(104, 622, 139, 466),
        articulations=[Articulation.SUSTAIN, Articulation.LEGATO, Articulation.VIBRATO, Articulation.BEND],
        polyphonic=False,
        brightness=0.6,
        warmth=0.7,
        genre_affinity={"jazz": 1.0, "soul": 0.9, "r-and-b": 0.8, "rock": 0.5},
        description="Classic tenor sax - smooth, soulful",
    ),
    67: Instrument(
        id="baritone_sax",
        name="Baritone Saxophone",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=67,
        frequency_range=FrequencyRange(69, 415, 87, 311),
        polyphonic=False,
        brightness=0.4,
        warmth=0.8,
        genre_affinity={"jazz": 0.9, "funk": 0.8},
    ),
    68: Instrument(
        id="oboe",
        name="Oboe",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=68,
        frequency_range=FrequencyRange(233, 1397, 277, 988),
        polyphonic=False,
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"classical": 1.0, "cinematic": 0.8},
    ),
    69: Instrument(
        id="english_horn",
        name="English Horn",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=69,
        polyphonic=False,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"classical": 1.0, "cinematic": 0.9},
    ),
    70: Instrument(
        id="bassoon",
        name="Bassoon",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=70,
        frequency_range=FrequencyRange(58, 622, 87, 440),
        polyphonic=False,
        brightness=0.4,
        warmth=0.8,
        genre_affinity={"classical": 1.0, "cinematic": 0.7},
    ),
    71: Instrument(
        id="clarinet",
        name="Clarinet",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=71,
        frequency_range=FrequencyRange(147, 2093, 165, 1175),
        articulations=[Articulation.SUSTAIN, Articulation.LEGATO, Articulation.STACCATO],
        polyphonic=False,
        brightness=0.6,
        warmth=0.6,
        genre_affinity={"classical": 1.0, "jazz": 0.9, "klezmer": 1.0},
    ),

    # Pipe (72-79)
    72: Instrument(
        id="piccolo",
        name="Piccolo",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=72,
        frequency_range=FrequencyRange(587, 4186, 698, 2637),
        polyphonic=False,
        brightness=1.0,
        warmth=0.2,
        genre_affinity={"classical": 1.0, "marching": 0.9},
    ),
    73: Instrument(
        id="flute",
        name="Flute",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=73,
        frequency_range=FrequencyRange(262, 2349, 294, 1760),
        articulations=[Articulation.SUSTAIN, Articulation.LEGATO, Articulation.STACCATO, Articulation.TREMOLO],
        polyphonic=False,
        brightness=0.8,
        warmth=0.4,
        genre_affinity={"classical": 1.0, "jazz": 0.7, "new-age": 0.8, "celtic": 0.9},
    ),
    74: Instrument(
        id="recorder",
        name="Recorder",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.CLASSICAL,
        gm_program=74,
        polyphonic=False,
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"baroque": 1.0, "medieval": 0.9, "folk": 0.6},
    ),
    75: Instrument(
        id="pan_flute",
        name="Pan Flute",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=75,
        polyphonic=False,
        brightness=0.6,
        warmth=0.7,
        genre_affinity={"world": 1.0, "new-age": 0.9, "ambient": 0.7},
    ),
    76: Instrument(
        id="blown_bottle",
        name="Blown Bottle",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=76,
        polyphonic=False,
        brightness=0.5,
        warmth=0.6,
        genre_affinity={"ambient": 0.7, "experimental": 0.8},
    ),
    77: Instrument(
        id="shakuhachi",
        name="Shakuhachi",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=77,
        articulations=[Articulation.BEND, Articulation.VIBRATO],
        polyphonic=False,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"world": 1.0, "ambient": 0.8, "meditation": 1.0},
        description="Japanese bamboo flute - meditative, expressive",
    ),
    78: Instrument(
        id="whistle",
        name="Whistle",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=78,
        polyphonic=False,
        brightness=0.9,
        warmth=0.3,
        genre_affinity={"celtic": 1.0, "folk": 0.9},
    ),
    79: Instrument(
        id="ocarina",
        name="Ocarina",
        family=InstrumentFamily.WOODWINDS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=79,
        polyphonic=False,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"world": 0.8, "folk": 0.7, "game-music": 1.0},
    ),

    # Synth Lead (80-87)
    80: Instrument(
        id="lead_square",
        name="Lead 1 (Square)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=80,
        polyphonic=False,
        brightness=0.8,
        warmth=0.4,
        genre_affinity={"synthwave": 1.0, "chiptune": 1.0, "edm": 0.7},
    ),
    81: Instrument(
        id="lead_sawtooth",
        name="Lead 2 (Sawtooth)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=81,
        polyphonic=False,
        brightness=0.9,
        warmth=0.3,
        genre_affinity={"synthwave": 1.0, "trance": 0.9, "edm": 0.8},
        description="Classic sawtooth lead - bright, cutting",
    ),
    82: Instrument(
        id="lead_calliope",
        name="Lead 3 (Calliope)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=82,
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"circus": 0.9, "experimental": 0.6},
    ),
    83: Instrument(
        id="lead_chiff",
        name="Lead 4 (Chiff)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=83,
        brightness=0.8,
        warmth=0.4,
        genre_affinity={"80s": 0.8, "synthpop": 0.7},
    ),
    84: Instrument(
        id="lead_charang",
        name="Lead 5 (Charang)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=84,
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"rock": 0.6, "fusion": 0.7},
    ),
    85: Instrument(
        id="lead_voice",
        name="Lead 6 (Voice)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=85,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"ambient": 0.8, "new-age": 0.7},
    ),
    86: Instrument(
        id="lead_fifths",
        name="Lead 7 (Fifths)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=86,
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"rock": 0.7, "metal": 0.6},
    ),
    87: Instrument(
        id="lead_bass_lead",
        name="Lead 8 (Bass + Lead)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=87,
        brightness=0.6,
        warmth=0.6,
        genre_affinity={"synthwave": 0.8, "electronic": 0.7},
    ),

    # Synth Pad (88-95)
    88: Instrument(
        id="pad_new_age",
        name="Pad 1 (New Age)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=88,
        attack_time_ms=500,
        release_time_ms=1000,
        brightness=0.4,
        warmth=0.8,
        genre_affinity={"new-age": 1.0, "ambient": 0.9, "meditation": 1.0},
    ),
    89: Instrument(
        id="pad_warm",
        name="Pad 2 (Warm)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=89,
        attack_time_ms=300,
        brightness=0.3,
        warmth=1.0,
        genre_affinity={"ambient": 1.0, "chillout": 0.9, "lo-fi-hip-hop": 0.8},
        description="Warm analog-style pad - lush, enveloping",
    ),
    90: Instrument(
        id="pad_polysynth",
        name="Pad 3 (Polysynth)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=90,
        brightness=0.6,
        warmth=0.6,
        genre_affinity={"synthwave": 0.9, "80s": 0.8},
    ),
    91: Instrument(
        id="pad_choir",
        name="Pad 4 (Choir)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=91,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"ambient": 0.8, "cinematic": 0.7},
    ),
    92: Instrument(
        id="pad_bowed",
        name="Pad 5 (Bowed)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=92,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"ambient": 0.9, "cinematic": 0.8},
    ),
    93: Instrument(
        id="pad_metallic",
        name="Pad 6 (Metallic)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=93,
        brightness=0.8,
        warmth=0.3,
        genre_affinity={"industrial": 0.8, "experimental": 0.7},
    ),
    94: Instrument(
        id="pad_halo",
        name="Pad 7 (Halo)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=94,
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"ambient": 0.9, "new-age": 0.8},
    ),
    95: Instrument(
        id="pad_sweep",
        name="Pad 8 (Sweep)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=95,
        brightness=0.6,
        warmth=0.6,
        genre_affinity={"trance": 0.8, "edm": 0.7},
    ),

    # Synth Effects (96-103)
    96: Instrument(
        id="fx_rain",
        name="FX 1 (Rain)",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=96,
        genre_affinity={"ambient": 0.9},
    ),
    97: Instrument(
        id="fx_soundtrack",
        name="FX 2 (Soundtrack)",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=97,
        genre_affinity={"cinematic": 0.8},
    ),
    98: Instrument(
        id="fx_crystal",
        name="FX 3 (Crystal)",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=98,
        brightness=1.0,
        warmth=0.3,
        genre_affinity={"ambient": 0.8, "new-age": 0.7},
    ),
    99: Instrument(
        id="fx_atmosphere",
        name="FX 4 (Atmosphere)",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=99,
        genre_affinity={"ambient": 1.0, "cinematic": 0.8},
    ),
    100: Instrument(
        id="fx_brightness",
        name="FX 5 (Brightness)",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=100,
        brightness=1.0,
        genre_affinity={"ambient": 0.7},
    ),
    101: Instrument(
        id="fx_goblins",
        name="FX 6 (Goblins)",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=101,
        genre_affinity={"horror": 0.9, "experimental": 0.7},
    ),
    102: Instrument(
        id="fx_echoes",
        name="FX 7 (Echoes)",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=102,
        genre_affinity={"ambient": 0.9},
    ),
    103: Instrument(
        id="fx_sci_fi",
        name="FX 8 (Sci-Fi)",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=103,
        genre_affinity={"sci-fi": 1.0, "electronic": 0.7},
    ),

    # Ethnic (104-111)
    104: Instrument(
        id="sitar",
        name="Sitar",
        family=InstrumentFamily.ETHNIC,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=104,
        articulations=[Articulation.BEND, Articulation.VIBRATO, Articulation.GLISSANDO],
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"world": 1.0, "indian": 1.0, "psychedelic": 0.8},
    ),
    105: Instrument(
        id="banjo",
        name="Banjo",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=105,
        brightness=0.9,
        warmth=0.4,
        genre_affinity={"country": 1.0, "bluegrass": 1.0, "folk": 0.8},
    ),
    106: Instrument(
        id="shamisen",
        name="Shamisen",
        family=InstrumentFamily.ETHNIC,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=106,
        brightness=0.8,
        warmth=0.4,
        genre_affinity={"world": 1.0, "japanese": 1.0},
    ),
    107: Instrument(
        id="koto",
        name="Koto",
        family=InstrumentFamily.ETHNIC,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=107,
        articulations=[Articulation.GLISSANDO, Articulation.BEND],
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"world": 1.0, "japanese": 1.0, "ambient": 0.6},
    ),
    108: Instrument(
        id="kalimba",
        name="Kalimba",
        family=InstrumentFamily.ETHNIC,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=108,
        brightness=0.8,
        warmth=0.6,
        genre_affinity={"world": 1.0, "african": 1.0, "ambient": 0.7, "lo-fi-hip-hop": 0.6},
    ),
    109: Instrument(
        id="bagpipe",
        name="Bagpipe",
        family=InstrumentFamily.ETHNIC,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=109,
        brightness=0.7,
        warmth=0.5,
        genre_affinity={"celtic": 1.0, "scottish": 1.0, "folk": 0.7},
    ),
    110: Instrument(
        id="fiddle",
        name="Fiddle",
        family=InstrumentFamily.STRINGS,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=110,
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"country": 1.0, "bluegrass": 1.0, "celtic": 0.9, "folk": 0.9},
    ),
    111: Instrument(
        id="shanai",
        name="Shanai",
        family=InstrumentFamily.ETHNIC,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=111,
        brightness=0.8,
        warmth=0.5,
        genre_affinity={"world": 1.0, "indian": 1.0},
    ),

    # Percussive (112-119)
    112: Instrument(
        id="tinkle_bell",
        name="Tinkle Bell",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=112,
        brightness=1.0,
        warmth=0.3,
    ),
    113: Instrument(
        id="agogo",
        name="Agogo",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=113,
        brightness=0.8,
        warmth=0.4,
        genre_affinity={"latin": 1.0, "brazilian": 1.0, "world": 0.8},
    ),
    114: Instrument(
        id="steel_drums",
        name="Steel Drums",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=114,
        brightness=0.8,
        warmth=0.5,
        genre_affinity={"caribbean": 1.0, "reggae": 0.8, "world": 0.7},
    ),
    115: Instrument(
        id="woodblock",
        name="Woodblock",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=115,
        brightness=0.7,
        warmth=0.4,
        genre_affinity={"latin": 0.8, "world": 0.7},
    ),
    116: Instrument(
        id="taiko_drum",
        name="Taiko Drum",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.VINTAGE,
        gm_program=116,
        brightness=0.4,
        warmth=0.9,
        genre_affinity={"world": 1.0, "japanese": 1.0, "cinematic": 0.9},
    ),
    117: Instrument(
        id="melodic_tom",
        name="Melodic Tom",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=117,
        brightness=0.5,
        warmth=0.7,
        genre_affinity={"cinematic": 0.8},
    ),
    118: Instrument(
        id="synth_drum",
        name="Synth Drum",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=118,
        brightness=0.6,
        warmth=0.5,
        genre_affinity={"electronic": 0.9, "80s": 0.8},
    ),
    119: Instrument(
        id="reverse_cymbal",
        name="Reverse Cymbal",
        family=InstrumentFamily.PERCUSSION,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=119,
        brightness=0.7,
        warmth=0.4,
        genre_affinity={"cinematic": 0.9, "edm": 0.8},
    ),

    # Sound Effects (120-127)
    120: Instrument(
        id="guitar_fret_noise",
        name="Guitar Fret Noise",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=120,
    ),
    121: Instrument(
        id="breath_noise",
        name="Breath Noise",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=121,
    ),
    122: Instrument(
        id="seashore",
        name="Seashore",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=122,
        genre_affinity={"ambient": 0.9},
    ),
    123: Instrument(
        id="bird_tweet",
        name="Bird Tweet",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=123,
        genre_affinity={"ambient": 0.8},
    ),
    124: Instrument(
        id="telephone_ring",
        name="Telephone Ring",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=124,
    ),
    125: Instrument(
        id="helicopter",
        name="Helicopter",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=125,
    ),
    126: Instrument(
        id="applause",
        name="Applause",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=126,
    ),
    127: Instrument(
        id="gunshot",
        name="Gunshot",
        family=InstrumentFamily.SFX,
        instrument_type=InstrumentType.ACOUSTIC,
        era=InstrumentEra.MODERN,
        gm_program=127,
    ),
}


# ============================================================================
# Electronic/Synth Instrument Extensions (Beyond GM)
# ============================================================================

SYNTH_INSTRUMENTS: dict[str, Instrument] = {
    # Classic Analog Synthesizers
    "minimoog": Instrument(
        id="minimoog",
        name="Minimoog",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.VINTAGE,
        gm_program=81,  # Map to sawtooth lead
        polyphonic=False,
        brightness=0.7,
        warmth=0.9,
        genre_affinity={"synthwave": 1.0, "prog-rock": 0.9, "funk": 0.8},
        description="Classic Moog monophonic synthesizer - fat, warm bass and leads",
    ),
    "prophet_5": Instrument(
        id="prophet_5",
        name="Prophet-5",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.VINTAGE,
        gm_program=90,
        max_polyphony=5,
        brightness=0.6,
        warmth=0.8,
        genre_affinity={"synthwave": 1.0, "80s": 1.0, "new-wave": 0.9},
        description="Sequential Circuits Prophet-5 - lush pads and brass",
    ),
    "juno_106": Instrument(
        id="juno_106",
        name="Roland Juno-106",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.VINTAGE,
        gm_program=89,
        brightness=0.5,
        warmth=0.9,
        genre_affinity={"synthwave": 1.0, "house": 0.9, "lo-fi-hip-hop": 0.8, "chillwave": 0.9},
        description="Roland Juno-106 - warm pads, classic chorus",
    ),
    "dx7": Instrument(
        id="dx7",
        name="Yamaha DX7",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.MODERN,
        gm_program=5,
        brightness=0.8,
        warmth=0.3,
        genre_affinity={"80s": 1.0, "pop": 0.8, "synthpop": 0.9},
        description="Yamaha DX7 FM synthesis - bells, electric pianos, brass",
    ),
    "tb_303": Instrument(
        id="tb_303",
        name="Roland TB-303",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.VINTAGE,
        gm_program=38,
        polyphonic=False,
        brightness=0.9,
        warmth=0.4,
        genre_affinity={"acid": 1.0, "techno": 0.9, "house": 0.8},
        description="Roland TB-303 acid bass - squelchy, resonant",
    ),
    "sh_101": Instrument(
        id="sh_101",
        name="Roland SH-101",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.VINTAGE,
        gm_program=81,
        polyphonic=False,
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"techno": 0.9, "house": 0.8, "synthwave": 0.8},
        description="Roland SH-101 - punchy bass, classic leads",
    ),
    "jupiter_8": Instrument(
        id="jupiter_8",
        name="Roland Jupiter-8",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.VINTAGE,
        gm_program=90,
        max_polyphony=8,
        brightness=0.6,
        warmth=0.9,
        genre_affinity={"synthwave": 1.0, "80s": 1.0, "trance": 0.7},
        description="Roland Jupiter-8 - lush pads, iconic strings",
    ),
    "oberheim_ob_xa": Instrument(
        id="oberheim_ob_xa",
        name="Oberheim OB-Xa",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.VINTAGE,
        gm_program=50,
        brightness=0.7,
        warmth=0.8,
        genre_affinity={"80s": 1.0, "synthpop": 0.9, "van-halen": 1.0},
        description="Oberheim OB-Xa - Jump sound, big brass stabs",
    ),
    "arp_odyssey": Instrument(
        id="arp_odyssey",
        name="ARP Odyssey",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.VINTAGE,
        gm_program=81,
        polyphonic=False,
        brightness=0.8,
        warmth=0.6,
        genre_affinity={"prog-rock": 0.9, "synthwave": 0.8},
        description="ARP Odyssey duophonic synth - aggressive leads",
    ),
    "moog_sub_37": Instrument(
        id="moog_sub_37",
        name="Moog Sub 37",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.CONTEMPORARY,
        gm_program=38,
        polyphonic=False,
        brightness=0.6,
        warmth=1.0,
        genre_affinity={"bass-music": 1.0, "dubstep": 0.8, "electronic": 0.9},
        description="Moog Sub 37 - massive bass, modern Moog sound",
    ),

    # Modern Digital/Software Synths
    "serum": Instrument(
        id="serum",
        name="Serum (Wavetable)",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.CONTEMPORARY,
        gm_program=81,
        brightness=0.8,
        warmth=0.5,
        genre_affinity={"edm": 1.0, "dubstep": 1.0, "future-bass": 1.0, "trap": 0.8},
        description="Xfer Serum wavetable synth - modern EDM standard",
    ),
    "massive": Instrument(
        id="massive",
        name="Massive",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.CONTEMPORARY,
        gm_program=38,
        brightness=0.7,
        warmth=0.6,
        genre_affinity={"dubstep": 1.0, "dnb": 0.9, "bass-music": 1.0},
        description="NI Massive - wobble bass, aggressive leads",
    ),
    "sylenth1": Instrument(
        id="sylenth1",
        name="Sylenth1",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.CONTEMPORARY,
        gm_program=81,
        brightness=0.7,
        warmth=0.7,
        genre_affinity={"trance": 1.0, "progressive-house": 0.9, "edm": 0.8},
        description="Sylenth1 - supersaw leads, trance plucks",
    ),
    "analog_lab": Instrument(
        id="analog_lab",
        name="Arturia Analog Lab",
        family=InstrumentFamily.SYNTH,
        instrument_type=InstrumentType.ELECTRONIC,
        era=InstrumentEra.CONTEMPORARY,
        gm_program=89,
        brightness=0.6,
        warmth=0.8,
        genre_affinity={"synthwave": 0.9, "80s": 0.9, "ambient": 0.7},
        description="Arturia vintage synth emulations",
    ),
}


# ============================================================================
# Lookup Functions
# ============================================================================

def get_instrument_by_gm(program: int, bank: int = 0) -> Optional[Instrument]:
    """Get instrument by General MIDI program number."""
    return GM_INSTRUMENTS.get(program)


def get_instrument_by_id(instrument_id: str) -> Optional[Instrument]:
    """Get instrument by ID."""
    # Check GM instruments
    for inst in GM_INSTRUMENTS.values():
        if inst.id == instrument_id:
            return inst

    # Check synth instruments
    return SYNTH_INSTRUMENTS.get(instrument_id)


def get_instruments_by_family(family: InstrumentFamily) -> list[Instrument]:
    """Get all instruments in a family."""
    instruments = []
    for inst in GM_INSTRUMENTS.values():
        if inst.family == family:
            instruments.append(inst)
    for inst in SYNTH_INSTRUMENTS.values():
        if inst.family == family:
            instruments.append(inst)
    return instruments


def get_instruments_by_genre(genre: str, min_affinity: float = 0.7) -> list[Instrument]:
    """Get instruments well-suited for a genre."""
    instruments = []
    for inst in GM_INSTRUMENTS.values():
        if inst.genre_affinity.get(genre, 0) >= min_affinity:
            instruments.append(inst)
    for inst in SYNTH_INSTRUMENTS.values():
        if inst.genre_affinity.get(genre, 0) >= min_affinity:
            instruments.append(inst)
    return sorted(instruments, key=lambda x: x.genre_affinity.get(genre, 0), reverse=True)


def get_instruments_by_type(inst_type: InstrumentType) -> list[Instrument]:
    """Get all instruments of a type (acoustic, electric, electronic)."""
    instruments = []
    for inst in GM_INSTRUMENTS.values():
        if inst.instrument_type == inst_type:
            instruments.append(inst)
    for inst in SYNTH_INSTRUMENTS.values():
        if inst.instrument_type == inst_type:
            instruments.append(inst)
    return instruments


def get_recommended_doublings(instrument: Instrument) -> list[Instrument]:
    """Get instruments that double well with the given instrument."""
    recommendations = []
    for other_id in instrument.doubles_well_with:
        other = get_instrument_by_id(other_id)
        if other:
            recommendations.append(other)
    return recommendations


# ============================================================================
# Genre Instrument Palettes
# ============================================================================

GENRE_PALETTES: dict[str, dict[str, list[int]]] = {
    "synthwave": {
        "lead": [81, 80],  # Sawtooth, square leads
        "pad": [89, 90, 50],  # Warm pad, polysynth, synth strings
        "bass": [38, 39],  # Synth bass
        "keys": [4, 5],  # Rhodes, DX EP
        "drums": [118],  # Synth drums (use drum kit separately)
    },
    "jazz": {
        "lead": [65, 66, 56],  # Alto sax, tenor sax, trumpet
        "pad": [48, 52],  # Strings, choir
        "bass": [32, 33],  # Upright, electric finger
        "keys": [0, 4, 16],  # Piano, Rhodes, Hammond
        "drums": [47],  # Timpani for accents
    },
    "funk": {
        "lead": [7, 65, 56],  # Clavinet, alto sax, trumpet
        "pad": [48],  # Strings
        "bass": [36, 37],  # Slap bass
        "keys": [4, 7, 16],  # Rhodes, Clavinet, Hammond
        "brass": [61],  # Brass section
    },
    "techno": {
        "lead": [80, 81],  # Square, sawtooth
        "pad": [93, 95],  # Metallic, sweep
        "bass": [38],  # Synth bass
        "fx": [99, 103],  # Atmosphere, sci-fi
    },
    "ambient": {
        "lead": [77, 75],  # Shakuhachi, pan flute
        "pad": [88, 89, 91, 92],  # New age, warm, choir, bowed
        "texture": [99, 102],  # Atmosphere, echoes
        "keys": [10, 46],  # Music box, harp
    },
    "rock": {
        "lead": [29, 30],  # Overdriven, distortion guitar
        "rhythm": [27, 28],  # Clean, muted guitar
        "bass": [33, 34],  # Finger, pick bass
        "keys": [18, 16],  # Rock organ, Hammond
    },
    "classical": {
        "strings": [40, 41, 42, 43],  # Violin, viola, cello, contrabass
        "woodwinds": [73, 68, 71, 70],  # Flute, oboe, clarinet, bassoon
        "brass": [56, 57, 60, 58],  # Trumpet, trombone, horn, tuba
        "percussion": [47, 14, 9],  # Timpani, bells, glockenspiel
        "keys": [19, 6],  # Church organ, harpsichord
    },
    "lo-fi-hip-hop": {
        "keys": [4, 10, 11],  # Rhodes, music box, vibraphone
        "pad": [89],  # Warm pad
        "bass": [33],  # Electric finger
        "texture": [45],  # Pizzicato strings
    },
    "trap": {
        "lead": [80, 81],  # Square, saw leads
        "pad": [89, 54],  # Warm, synth choir
        "bass": [38, 39],  # Synth bass (808 style)
        "keys": [4],  # Rhodes for melodies
    },
    "house": {
        "lead": [4, 80],  # Rhodes, square
        "pad": [89, 50],  # Warm, synth strings
        "bass": [38],  # Synth bass
        "stab": [55, 62],  # Orch hit, synth brass
    },
    "cinematic": {
        "strings": [48, 44, 45],  # String ensemble, tremolo, pizz
        "brass": [60, 56, 61],  # French horn, trumpet, brass section
        "percussion": [47, 116, 14],  # Timpani, taiko, tubular bells
        "choir": [52, 53],  # Aahs, oohs
        "texture": [99, 46],  # Atmosphere, harp
    },
}


def get_genre_palette(genre: str) -> dict[str, list[Instrument]]:
    """Get the instrument palette for a genre."""
    palette_programs = GENRE_PALETTES.get(genre, {})
    palette = {}

    for role, programs in palette_programs.items():
        instruments = []
        for prog in programs:
            inst = get_instrument_by_gm(prog)
            if inst:
                instruments.append(inst)
        palette[role] = instruments

    return palette
