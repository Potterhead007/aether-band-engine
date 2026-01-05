"""
AETHER Percussion & Drums Knowledge Base

Comprehensive percussion library covering:
- General MIDI Drum Map (Channel 10)
- Acoustic Drum Kits (jazz, rock, orchestral)
- Electronic Drum Machines (808, 909, LinnDrum, etc.)
- World Percussion (Latin, African, Asian)
- Orchestral Percussion (timpani, cymbals, mallet instruments)
- Genre-specific drum patterns and velocities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================================
# Enums
# ============================================================================

class DrumCategory(Enum):
    """High-level drum/percussion category."""
    KICK = "kick"
    SNARE = "snare"
    HIHAT = "hihat"
    TOM = "tom"
    CYMBAL = "cymbal"
    CLAP = "clap"
    PERCUSSION = "percussion"
    ELECTRONIC = "electronic"
    WORLD = "world"
    ORCHESTRAL = "orchestral"


class DrumMachine(Enum):
    """Classic drum machine types."""
    TR808 = "tr808"
    TR909 = "tr909"
    LINNDRUM = "linndrum"
    DMX = "dmx"
    CR78 = "cr78"
    DRUMTRAKS = "drumtraks"
    SP1200 = "sp1200"
    MPC = "mpc"
    ACOUSTIC = "acoustic"


class DrumKitStyle(Enum):
    """Acoustic drum kit styles."""
    JAZZ = "jazz"
    ROCK = "rock"
    METAL = "metal"
    FUNK = "funk"
    STUDIO = "studio"
    VINTAGE = "vintage"
    BRUSH = "brush"
    ORCHESTRAL = "orchestral"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DrumSound:
    """Individual drum/percussion sound definition."""
    id: str
    name: str
    category: DrumCategory
    gm_note: int  # General MIDI note number (channel 10)

    # Sound characteristics
    pitch_hz: Optional[float] = None
    pitch_midi: Optional[int] = None
    decay_ms: float = 200.0
    attack_ms: float = 1.0

    # Timbral
    brightness: float = 0.5
    body: float = 0.5
    click: float = 0.0  # Attack transient
    sustain: float = 0.0

    # Velocity response
    velocity_curve: str = "linear"  # linear, soft, hard, fixed
    default_velocity: int = 100
    ghost_velocity: int = 40
    accent_velocity: int = 120

    # Genre affinity
    genre_affinity: dict[str, float] = field(default_factory=dict)

    # Layering
    can_layer_with: list[str] = field(default_factory=list)
    exclusive_group: Optional[str] = None  # Sounds that cut each other off

    description: str = ""


@dataclass
class DrumKit:
    """Complete drum kit definition."""
    id: str
    name: str
    style: DrumKitStyle
    machine: Optional[DrumMachine] = None

    # Kit pieces
    sounds: dict[str, DrumSound] = field(default_factory=dict)

    # Overall characteristics
    overall_brightness: float = 0.5
    overall_punch: float = 0.5
    room_amount: float = 0.3

    # Genre affinity
    genre_affinity: dict[str, float] = field(default_factory=dict)

    description: str = ""


# ============================================================================
# General MIDI Drum Map (Channel 10)
# Standard drum sounds mapped to MIDI note numbers
# ============================================================================

GM_DRUM_MAP: dict[int, DrumSound] = {
    # Bass Drums
    35: DrumSound(
        id="acoustic_bass_drum",
        name="Acoustic Bass Drum",
        category=DrumCategory.KICK,
        gm_note=35,
        pitch_hz=60,
        decay_ms=400,
        body=0.9,
        click=0.3,
        genre_affinity={"rock": 0.9, "jazz": 0.7, "pop": 0.8},
    ),
    36: DrumSound(
        id="bass_drum_1",
        name="Bass Drum 1",
        category=DrumCategory.KICK,
        gm_note=36,
        pitch_hz=55,
        decay_ms=350,
        body=0.8,
        click=0.4,
        genre_affinity={"rock": 1.0, "pop": 0.9, "funk": 0.8},
        description="Standard kick drum",
    ),

    # Snares
    38: DrumSound(
        id="acoustic_snare",
        name="Acoustic Snare",
        category=DrumCategory.SNARE,
        gm_note=38,
        decay_ms=200,
        brightness=0.7,
        body=0.6,
        click=0.5,
        genre_affinity={"rock": 1.0, "pop": 0.9, "funk": 0.8},
        description="Standard snare drum",
    ),
    40: DrumSound(
        id="electric_snare",
        name="Electric Snare",
        category=DrumCategory.SNARE,
        gm_note=40,
        decay_ms=180,
        brightness=0.8,
        click=0.6,
        genre_affinity={"electronic": 0.9, "pop": 0.8},
    ),
    37: DrumSound(
        id="side_stick",
        name="Side Stick (Rimshot)",
        category=DrumCategory.SNARE,
        gm_note=37,
        decay_ms=100,
        brightness=0.9,
        click=1.0,
        genre_affinity={"reggae": 1.0, "jazz": 0.8, "r-and-b": 0.7},
        description="Cross-stick / rimshot",
    ),

    # Hi-Hats
    42: DrumSound(
        id="closed_hihat",
        name="Closed Hi-Hat",
        category=DrumCategory.HIHAT,
        gm_note=42,
        decay_ms=50,
        brightness=0.9,
        click=0.8,
        exclusive_group="hihat",
        genre_affinity={"rock": 0.9, "pop": 0.9, "funk": 1.0, "disco": 1.0},
    ),
    44: DrumSound(
        id="pedal_hihat",
        name="Pedal Hi-Hat",
        category=DrumCategory.HIHAT,
        gm_note=44,
        decay_ms=80,
        brightness=0.7,
        exclusive_group="hihat",
        genre_affinity={"jazz": 1.0, "funk": 0.7},
    ),
    46: DrumSound(
        id="open_hihat",
        name="Open Hi-Hat",
        category=DrumCategory.HIHAT,
        gm_note=46,
        decay_ms=400,
        brightness=0.8,
        sustain=0.7,
        exclusive_group="hihat",
        genre_affinity={"rock": 0.9, "disco": 1.0, "house": 0.9},
    ),

    # Toms
    41: DrumSound(
        id="low_floor_tom",
        name="Low Floor Tom",
        category=DrumCategory.TOM,
        gm_note=41,
        pitch_hz=80,
        decay_ms=350,
        body=0.9,
        genre_affinity={"rock": 0.9, "cinematic": 0.8},
    ),
    43: DrumSound(
        id="high_floor_tom",
        name="High Floor Tom",
        category=DrumCategory.TOM,
        gm_note=43,
        pitch_hz=100,
        decay_ms=300,
        body=0.8,
        genre_affinity={"rock": 0.9},
    ),
    45: DrumSound(
        id="low_tom",
        name="Low Tom",
        category=DrumCategory.TOM,
        gm_note=45,
        pitch_hz=120,
        decay_ms=280,
        body=0.7,
        genre_affinity={"rock": 0.9},
    ),
    47: DrumSound(
        id="low_mid_tom",
        name="Low-Mid Tom",
        category=DrumCategory.TOM,
        gm_note=47,
        pitch_hz=150,
        decay_ms=250,
        body=0.6,
        genre_affinity={"rock": 0.9},
    ),
    48: DrumSound(
        id="hi_mid_tom",
        name="Hi-Mid Tom",
        category=DrumCategory.TOM,
        gm_note=48,
        pitch_hz=180,
        decay_ms=220,
        body=0.5,
        genre_affinity={"rock": 0.9},
    ),
    50: DrumSound(
        id="high_tom",
        name="High Tom",
        category=DrumCategory.TOM,
        gm_note=50,
        pitch_hz=220,
        decay_ms=200,
        body=0.4,
        genre_affinity={"rock": 0.9},
    ),

    # Cymbals
    49: DrumSound(
        id="crash_cymbal_1",
        name="Crash Cymbal 1",
        category=DrumCategory.CYMBAL,
        gm_note=49,
        decay_ms=2000,
        brightness=0.9,
        sustain=0.8,
        genre_affinity={"rock": 1.0, "metal": 1.0, "pop": 0.7},
    ),
    51: DrumSound(
        id="ride_cymbal_1",
        name="Ride Cymbal 1",
        category=DrumCategory.CYMBAL,
        gm_note=51,
        decay_ms=1500,
        brightness=0.7,
        sustain=0.6,
        genre_affinity={"jazz": 1.0, "rock": 0.8, "pop": 0.6},
        description="Main ride cymbal",
    ),
    52: DrumSound(
        id="chinese_cymbal",
        name="Chinese Cymbal",
        category=DrumCategory.CYMBAL,
        gm_note=52,
        decay_ms=1800,
        brightness=0.8,
        genre_affinity={"metal": 0.9, "rock": 0.6},
    ),
    53: DrumSound(
        id="ride_bell",
        name="Ride Bell",
        category=DrumCategory.CYMBAL,
        gm_note=53,
        decay_ms=800,
        brightness=1.0,
        click=0.9,
        genre_affinity={"jazz": 0.9, "latin": 0.8},
    ),
    55: DrumSound(
        id="splash_cymbal",
        name="Splash Cymbal",
        category=DrumCategory.CYMBAL,
        gm_note=55,
        decay_ms=1000,
        brightness=1.0,
        genre_affinity={"rock": 0.8, "punk": 0.9},
    ),
    57: DrumSound(
        id="crash_cymbal_2",
        name="Crash Cymbal 2",
        category=DrumCategory.CYMBAL,
        gm_note=57,
        decay_ms=2200,
        brightness=0.85,
        sustain=0.8,
        genre_affinity={"rock": 0.9, "metal": 0.9},
    ),
    59: DrumSound(
        id="ride_cymbal_2",
        name="Ride Cymbal 2",
        category=DrumCategory.CYMBAL,
        gm_note=59,
        decay_ms=1600,
        brightness=0.75,
        genre_affinity={"jazz": 0.8, "rock": 0.7},
    ),

    # Claps & Electronic
    39: DrumSound(
        id="hand_clap",
        name="Hand Clap",
        category=DrumCategory.CLAP,
        gm_note=39,
        decay_ms=150,
        brightness=0.8,
        click=0.9,
        genre_affinity={"disco": 1.0, "house": 1.0, "pop": 0.8, "funk": 0.7},
    ),

    # Latin Percussion
    54: DrumSound(
        id="tambourine",
        name="Tambourine",
        category=DrumCategory.PERCUSSION,
        gm_note=54,
        decay_ms=300,
        brightness=1.0,
        genre_affinity={"pop": 0.8, "rock": 0.7, "folk": 0.9},
    ),
    56: DrumSound(
        id="cowbell",
        name="Cowbell",
        category=DrumCategory.PERCUSSION,
        gm_note=56,
        decay_ms=200,
        brightness=0.9,
        click=0.8,
        genre_affinity={"funk": 0.9, "disco": 0.8, "latin": 1.0, "rock": 0.6},
        description="More cowbell!",
    ),
    58: DrumSound(
        id="vibraslap",
        name="Vibraslap",
        category=DrumCategory.PERCUSSION,
        gm_note=58,
        decay_ms=600,
        genre_affinity={"latin": 0.7, "funk": 0.5},
    ),
    60: DrumSound(
        id="hi_bongo",
        name="Hi Bongo",
        category=DrumCategory.WORLD,
        gm_note=60,
        pitch_hz=400,
        decay_ms=150,
        genre_affinity={"latin": 1.0, "jazz": 0.7, "world": 0.9},
    ),
    61: DrumSound(
        id="low_bongo",
        name="Low Bongo",
        category=DrumCategory.WORLD,
        gm_note=61,
        pitch_hz=300,
        decay_ms=180,
        genre_affinity={"latin": 1.0, "jazz": 0.7, "world": 0.9},
    ),
    62: DrumSound(
        id="mute_hi_conga",
        name="Mute Hi Conga",
        category=DrumCategory.WORLD,
        gm_note=62,
        decay_ms=80,
        genre_affinity={"latin": 1.0, "funk": 0.8, "disco": 0.7},
    ),
    63: DrumSound(
        id="open_hi_conga",
        name="Open Hi Conga",
        category=DrumCategory.WORLD,
        gm_note=63,
        decay_ms=250,
        genre_affinity={"latin": 1.0, "funk": 0.8, "disco": 0.7},
    ),
    64: DrumSound(
        id="low_conga",
        name="Low Conga",
        category=DrumCategory.WORLD,
        gm_note=64,
        pitch_hz=180,
        decay_ms=280,
        genre_affinity={"latin": 1.0, "funk": 0.8},
    ),
    65: DrumSound(
        id="high_timbale",
        name="High Timbale",
        category=DrumCategory.WORLD,
        gm_note=65,
        decay_ms=150,
        brightness=0.9,
        genre_affinity={"latin": 1.0, "salsa": 1.0},
    ),
    66: DrumSound(
        id="low_timbale",
        name="Low Timbale",
        category=DrumCategory.WORLD,
        gm_note=66,
        decay_ms=180,
        genre_affinity={"latin": 1.0, "salsa": 1.0},
    ),
    67: DrumSound(
        id="high_agogo",
        name="High Agogo",
        category=DrumCategory.WORLD,
        gm_note=67,
        decay_ms=100,
        brightness=1.0,
        genre_affinity={"latin": 1.0, "brazilian": 1.0, "samba": 1.0},
    ),
    68: DrumSound(
        id="low_agogo",
        name="Low Agogo",
        category=DrumCategory.WORLD,
        gm_note=68,
        decay_ms=120,
        genre_affinity={"latin": 1.0, "brazilian": 1.0, "samba": 1.0},
    ),
    69: DrumSound(
        id="cabasa",
        name="Cabasa",
        category=DrumCategory.WORLD,
        gm_note=69,
        decay_ms=80,
        genre_affinity={"latin": 0.9, "brazilian": 1.0},
    ),
    70: DrumSound(
        id="maracas",
        name="Maracas",
        category=DrumCategory.WORLD,
        gm_note=70,
        decay_ms=60,
        genre_affinity={"latin": 1.0, "salsa": 1.0, "pop": 0.5},
    ),
    71: DrumSound(
        id="short_whistle",
        name="Short Whistle",
        category=DrumCategory.PERCUSSION,
        gm_note=71,
        decay_ms=100,
        genre_affinity={"latin": 0.6},
    ),
    72: DrumSound(
        id="long_whistle",
        name="Long Whistle",
        category=DrumCategory.PERCUSSION,
        gm_note=72,
        decay_ms=400,
        genre_affinity={"latin": 0.6},
    ),
    73: DrumSound(
        id="short_guiro",
        name="Short Guiro",
        category=DrumCategory.WORLD,
        gm_note=73,
        decay_ms=100,
        genre_affinity={"latin": 1.0, "salsa": 0.9},
    ),
    74: DrumSound(
        id="long_guiro",
        name="Long Guiro",
        category=DrumCategory.WORLD,
        gm_note=74,
        decay_ms=400,
        genre_affinity={"latin": 1.0, "salsa": 0.9},
    ),
    75: DrumSound(
        id="claves",
        name="Claves",
        category=DrumCategory.WORLD,
        gm_note=75,
        decay_ms=100,
        brightness=1.0,
        click=1.0,
        genre_affinity={"latin": 1.0, "salsa": 1.0, "cuban": 1.0},
        description="Essential for clave rhythm patterns",
    ),
    76: DrumSound(
        id="hi_wood_block",
        name="Hi Wood Block",
        category=DrumCategory.PERCUSSION,
        gm_note=76,
        decay_ms=50,
        brightness=0.9,
        genre_affinity={"latin": 0.8, "classical": 0.6},
    ),
    77: DrumSound(
        id="low_wood_block",
        name="Low Wood Block",
        category=DrumCategory.PERCUSSION,
        gm_note=77,
        decay_ms=60,
        genre_affinity={"latin": 0.8, "classical": 0.6},
    ),
    78: DrumSound(
        id="mute_cuica",
        name="Mute Cuica",
        category=DrumCategory.WORLD,
        gm_note=78,
        decay_ms=80,
        genre_affinity={"brazilian": 1.0, "samba": 1.0},
    ),
    79: DrumSound(
        id="open_cuica",
        name="Open Cuica",
        category=DrumCategory.WORLD,
        gm_note=79,
        decay_ms=300,
        genre_affinity={"brazilian": 1.0, "samba": 1.0},
    ),
    80: DrumSound(
        id="mute_triangle",
        name="Mute Triangle",
        category=DrumCategory.PERCUSSION,
        gm_note=80,
        decay_ms=50,
        brightness=1.0,
        genre_affinity={"classical": 0.8, "latin": 0.7},
    ),
    81: DrumSound(
        id="open_triangle",
        name="Open Triangle",
        category=DrumCategory.PERCUSSION,
        gm_note=81,
        decay_ms=800,
        brightness=1.0,
        sustain=0.8,
        genre_affinity={"classical": 0.9, "latin": 0.7, "pop": 0.5},
    ),
    82: DrumSound(
        id="shaker",
        name="Shaker",
        category=DrumCategory.PERCUSSION,
        gm_note=82,
        decay_ms=60,
        genre_affinity={"pop": 0.8, "acoustic-folk": 0.9, "latin": 0.8},
    ),
    83: DrumSound(
        id="jingle_bell",
        name="Jingle Bell",
        category=DrumCategory.PERCUSSION,
        gm_note=83,
        decay_ms=200,
        brightness=1.0,
        genre_affinity={"christmas": 1.0, "pop": 0.4},
    ),
    84: DrumSound(
        id="bell_tree",
        name="Bell Tree",
        category=DrumCategory.PERCUSSION,
        gm_note=84,
        decay_ms=1500,
        brightness=1.0,
        sustain=0.9,
        genre_affinity={"orchestral": 0.8, "cinematic": 0.7},
    ),
    85: DrumSound(
        id="castanets",
        name="Castanets",
        category=DrumCategory.WORLD,
        gm_note=85,
        decay_ms=50,
        brightness=0.9,
        genre_affinity={"flamenco": 1.0, "spanish": 1.0, "latin": 0.7},
    ),
    86: DrumSound(
        id="mute_surdo",
        name="Mute Surdo",
        category=DrumCategory.WORLD,
        gm_note=86,
        pitch_hz=60,
        decay_ms=150,
        body=1.0,
        genre_affinity={"brazilian": 1.0, "samba": 1.0},
    ),
    87: DrumSound(
        id="open_surdo",
        name="Open Surdo",
        category=DrumCategory.WORLD,
        gm_note=87,
        pitch_hz=50,
        decay_ms=350,
        body=1.0,
        genre_affinity={"brazilian": 1.0, "samba": 1.0},
    ),
}


# ============================================================================
# Electronic Drum Machine Sounds
# ============================================================================

TR808_SOUNDS: dict[str, DrumSound] = {
    "kick": DrumSound(
        id="808_kick",
        name="808 Kick",
        category=DrumCategory.KICK,
        gm_note=36,
        pitch_hz=45,
        decay_ms=800,
        body=1.0,
        click=0.2,
        sustain=0.9,
        genre_affinity={"trap": 1.0, "hip-hop": 1.0, "r-and-b": 0.8, "house": 0.7},
        description="Legendary Roland TR-808 kick - deep, boomy, tuneable",
    ),
    "snare": DrumSound(
        id="808_snare",
        name="808 Snare",
        category=DrumCategory.SNARE,
        gm_note=38,
        decay_ms=200,
        brightness=0.6,
        body=0.5,
        genre_affinity={"trap": 0.9, "hip-hop": 0.9, "electro": 0.8},
    ),
    "clap": DrumSound(
        id="808_clap",
        name="808 Clap",
        category=DrumCategory.CLAP,
        gm_note=39,
        decay_ms=180,
        brightness=0.7,
        genre_affinity={"trap": 1.0, "hip-hop": 1.0, "house": 0.8, "electro": 0.9},
    ),
    "hihat_closed": DrumSound(
        id="808_hihat_closed",
        name="808 Closed Hi-Hat",
        category=DrumCategory.HIHAT,
        gm_note=42,
        decay_ms=30,
        brightness=0.8,
        exclusive_group="hihat_808",
        genre_affinity={"trap": 1.0, "hip-hop": 0.9},
    ),
    "hihat_open": DrumSound(
        id="808_hihat_open",
        name="808 Open Hi-Hat",
        category=DrumCategory.HIHAT,
        gm_note=46,
        decay_ms=200,
        brightness=0.7,
        exclusive_group="hihat_808",
        genre_affinity={"trap": 1.0, "hip-hop": 0.9},
    ),
    "tom_low": DrumSound(
        id="808_tom_low",
        name="808 Low Tom",
        category=DrumCategory.TOM,
        gm_note=41,
        pitch_hz=100,
        decay_ms=400,
        body=0.9,
        genre_affinity={"trap": 0.8, "electro": 0.7},
    ),
    "tom_mid": DrumSound(
        id="808_tom_mid",
        name="808 Mid Tom",
        category=DrumCategory.TOM,
        gm_note=47,
        pitch_hz=150,
        decay_ms=350,
        genre_affinity={"trap": 0.8, "electro": 0.7},
    ),
    "tom_high": DrumSound(
        id="808_tom_high",
        name="808 High Tom",
        category=DrumCategory.TOM,
        gm_note=50,
        pitch_hz=200,
        decay_ms=300,
        genre_affinity={"trap": 0.8, "electro": 0.7},
    ),
    "cowbell": DrumSound(
        id="808_cowbell",
        name="808 Cowbell",
        category=DrumCategory.PERCUSSION,
        gm_note=56,
        decay_ms=150,
        brightness=1.0,
        click=0.9,
        genre_affinity={"electro": 1.0, "hip-hop": 0.7},
    ),
    "rimshot": DrumSound(
        id="808_rimshot",
        name="808 Rimshot",
        category=DrumCategory.SNARE,
        gm_note=37,
        decay_ms=80,
        brightness=0.9,
        click=1.0,
        genre_affinity={"trap": 0.8, "hip-hop": 0.8},
    ),
    "clave": DrumSound(
        id="808_clave",
        name="808 Clave",
        category=DrumCategory.PERCUSSION,
        gm_note=75,
        decay_ms=50,
        brightness=1.0,
        genre_affinity={"electro": 0.8, "latin-electronic": 0.9},
    ),
    "maracas": DrumSound(
        id="808_maracas",
        name="808 Maracas",
        category=DrumCategory.PERCUSSION,
        gm_note=70,
        decay_ms=40,
        brightness=0.9,
        genre_affinity={"electro": 0.7},
    ),
    "cymbal": DrumSound(
        id="808_cymbal",
        name="808 Cymbal",
        category=DrumCategory.CYMBAL,
        gm_note=49,
        decay_ms=1500,
        brightness=0.6,
        genre_affinity={"electro": 0.8, "hip-hop": 0.6},
    ),
    "conga_low": DrumSound(
        id="808_conga_low",
        name="808 Low Conga",
        category=DrumCategory.WORLD,
        gm_note=64,
        decay_ms=200,
        genre_affinity={"electro": 0.7, "latin-electronic": 0.9},
    ),
    "conga_mid": DrumSound(
        id="808_conga_mid",
        name="808 Mid Conga",
        category=DrumCategory.WORLD,
        gm_note=63,
        decay_ms=180,
        genre_affinity={"electro": 0.7, "latin-electronic": 0.9},
    ),
    "conga_high": DrumSound(
        id="808_conga_high",
        name="808 High Conga",
        category=DrumCategory.WORLD,
        gm_note=62,
        decay_ms=100,
        genre_affinity={"electro": 0.7, "latin-electronic": 0.9},
    ),
}


TR909_SOUNDS: dict[str, DrumSound] = {
    "kick": DrumSound(
        id="909_kick",
        name="909 Kick",
        category=DrumCategory.KICK,
        gm_note=36,
        pitch_hz=50,
        decay_ms=300,
        body=0.8,
        click=0.6,
        genre_affinity={"house": 1.0, "techno": 1.0, "trance": 0.9, "edm": 0.9},
        description="Roland TR-909 kick - punchy, tight, iconic",
    ),
    "snare": DrumSound(
        id="909_snare",
        name="909 Snare",
        category=DrumCategory.SNARE,
        gm_note=38,
        decay_ms=180,
        brightness=0.8,
        body=0.6,
        genre_affinity={"house": 1.0, "techno": 1.0, "trance": 0.8},
    ),
    "clap": DrumSound(
        id="909_clap",
        name="909 Clap",
        category=DrumCategory.CLAP,
        gm_note=39,
        decay_ms=200,
        brightness=0.8,
        genre_affinity={"house": 1.0, "techno": 1.0, "trance": 0.9},
        description="Iconic house/techno clap",
    ),
    "hihat_closed": DrumSound(
        id="909_hihat_closed",
        name="909 Closed Hi-Hat",
        category=DrumCategory.HIHAT,
        gm_note=42,
        decay_ms=40,
        brightness=0.9,
        exclusive_group="hihat_909",
        genre_affinity={"house": 1.0, "techno": 1.0},
    ),
    "hihat_open": DrumSound(
        id="909_hihat_open",
        name="909 Open Hi-Hat",
        category=DrumCategory.HIHAT,
        gm_note=46,
        decay_ms=300,
        brightness=0.85,
        exclusive_group="hihat_909",
        genre_affinity={"house": 1.0, "techno": 0.9, "disco": 0.8},
    ),
    "ride": DrumSound(
        id="909_ride",
        name="909 Ride",
        category=DrumCategory.CYMBAL,
        gm_note=51,
        decay_ms=500,
        brightness=0.7,
        genre_affinity={"house": 0.8, "techno": 0.7},
    ),
    "crash": DrumSound(
        id="909_crash",
        name="909 Crash",
        category=DrumCategory.CYMBAL,
        gm_note=49,
        decay_ms=1200,
        brightness=0.75,
        genre_affinity={"house": 0.8, "techno": 0.7, "trance": 0.9},
    ),
    "tom_low": DrumSound(
        id="909_tom_low",
        name="909 Low Tom",
        category=DrumCategory.TOM,
        gm_note=41,
        decay_ms=250,
        genre_affinity={"house": 0.7, "techno": 0.8},
    ),
    "tom_mid": DrumSound(
        id="909_tom_mid",
        name="909 Mid Tom",
        category=DrumCategory.TOM,
        gm_note=47,
        decay_ms=220,
        genre_affinity={"house": 0.7, "techno": 0.8},
    ),
    "tom_high": DrumSound(
        id="909_tom_high",
        name="909 High Tom",
        category=DrumCategory.TOM,
        gm_note=50,
        decay_ms=180,
        genre_affinity={"house": 0.7, "techno": 0.8},
    ),
    "rimshot": DrumSound(
        id="909_rimshot",
        name="909 Rimshot",
        category=DrumCategory.SNARE,
        gm_note=37,
        decay_ms=60,
        brightness=0.95,
        click=1.0,
        genre_affinity={"house": 0.8, "techno": 0.9},
    ),
}


LINNDRUM_SOUNDS: dict[str, DrumSound] = {
    "kick": DrumSound(
        id="linn_kick",
        name="LinnDrum Kick",
        category=DrumCategory.KICK,
        gm_note=36,
        decay_ms=350,
        body=0.7,
        click=0.5,
        genre_affinity={"80s": 1.0, "synthpop": 1.0, "pop": 0.8},
        description="LM-1/LinnDrum kick - the 80s sound",
    ),
    "snare": DrumSound(
        id="linn_snare",
        name="LinnDrum Snare",
        category=DrumCategory.SNARE,
        gm_note=38,
        decay_ms=220,
        brightness=0.7,
        body=0.6,
        genre_affinity={"80s": 1.0, "synthpop": 1.0, "prince": 1.0},
    ),
    "hihat": DrumSound(
        id="linn_hihat",
        name="LinnDrum Hi-Hat",
        category=DrumCategory.HIHAT,
        gm_note=42,
        decay_ms=50,
        brightness=0.75,
        genre_affinity={"80s": 1.0, "synthpop": 0.9},
    ),
    "clap": DrumSound(
        id="linn_clap",
        name="LinnDrum Clap",
        category=DrumCategory.CLAP,
        gm_note=39,
        decay_ms=160,
        genre_affinity={"80s": 1.0, "synthpop": 0.9},
    ),
    "tom_low": DrumSound(
        id="linn_tom_low",
        name="LinnDrum Low Tom",
        category=DrumCategory.TOM,
        gm_note=41,
        decay_ms=300,
        genre_affinity={"80s": 0.9},
    ),
    "tom_high": DrumSound(
        id="linn_tom_high",
        name="LinnDrum High Tom",
        category=DrumCategory.TOM,
        gm_note=50,
        decay_ms=200,
        genre_affinity={"80s": 0.9},
    ),
    "cabasa": DrumSound(
        id="linn_cabasa",
        name="LinnDrum Cabasa",
        category=DrumCategory.PERCUSSION,
        gm_note=69,
        decay_ms=60,
        genre_affinity={"80s": 0.8},
    ),
    "tambourine": DrumSound(
        id="linn_tambourine",
        name="LinnDrum Tambourine",
        category=DrumCategory.PERCUSSION,
        gm_note=54,
        decay_ms=200,
        genre_affinity={"80s": 0.9, "pop": 0.7},
    ),
    "cowbell": DrumSound(
        id="linn_cowbell",
        name="LinnDrum Cowbell",
        category=DrumCategory.PERCUSSION,
        gm_note=56,
        decay_ms=180,
        genre_affinity={"80s": 0.8},
    ),
    "ride": DrumSound(
        id="linn_ride",
        name="LinnDrum Ride",
        category=DrumCategory.CYMBAL,
        gm_note=51,
        decay_ms=800,
        genre_affinity={"80s": 0.7},
    ),
    "crash": DrumSound(
        id="linn_crash",
        name="LinnDrum Crash",
        category=DrumCategory.CYMBAL,
        gm_note=49,
        decay_ms=1500,
        genre_affinity={"80s": 0.8},
    ),
}


# ============================================================================
# Drum Kit Definitions
# ============================================================================

DRUM_KITS: dict[str, DrumKit] = {
    "standard": DrumKit(
        id="standard",
        name="Standard Kit",
        style=DrumKitStyle.STUDIO,
        overall_brightness=0.6,
        overall_punch=0.7,
        room_amount=0.3,
        genre_affinity={"rock": 0.9, "pop": 1.0, "funk": 0.8},
        description="Versatile studio drum kit",
    ),
    "jazz": DrumKit(
        id="jazz",
        name="Jazz Kit",
        style=DrumKitStyle.JAZZ,
        overall_brightness=0.5,
        overall_punch=0.4,
        room_amount=0.5,
        genre_affinity={"jazz": 1.0, "swing": 0.9, "lounge": 0.8},
        description="Warm, dynamic jazz kit with brush options",
    ),
    "rock": DrumKit(
        id="rock",
        name="Rock Kit",
        style=DrumKitStyle.ROCK,
        overall_brightness=0.7,
        overall_punch=0.9,
        room_amount=0.4,
        genre_affinity={"rock": 1.0, "metal": 0.7, "punk": 0.8},
        description="Punchy rock kit with tight kick and crisp snare",
    ),
    "metal": DrumKit(
        id="metal",
        name="Metal Kit",
        style=DrumKitStyle.METAL,
        overall_brightness=0.8,
        overall_punch=1.0,
        room_amount=0.2,
        genre_affinity={"metal": 1.0, "rock": 0.7},
        description="Heavy, aggressive kit with fast response",
    ),
    "funk": DrumKit(
        id="funk",
        name="Funk Kit",
        style=DrumKitStyle.FUNK,
        overall_brightness=0.6,
        overall_punch=0.7,
        room_amount=0.3,
        genre_affinity={"funk": 1.0, "soul": 0.9, "disco": 0.8},
        description="Tight, dry kit perfect for ghost notes",
    ),
    "vintage": DrumKit(
        id="vintage",
        name="Vintage Kit",
        style=DrumKitStyle.VINTAGE,
        overall_brightness=0.4,
        overall_punch=0.5,
        room_amount=0.6,
        genre_affinity={"vintage": 1.0, "soul": 0.9, "motown": 1.0},
        description="60s/70s style kit with warm, rounded tones",
    ),
    "brush": DrumKit(
        id="brush",
        name="Brush Kit",
        style=DrumKitStyle.BRUSH,
        overall_brightness=0.3,
        overall_punch=0.2,
        room_amount=0.5,
        genre_affinity={"jazz": 1.0, "ballad": 0.9, "lounge": 0.9},
        description="Brush technique on snare, soft mallets on cymbals",
    ),
    "tr808": DrumKit(
        id="tr808",
        name="TR-808",
        style=DrumKitStyle.STUDIO,
        machine=DrumMachine.TR808,
        overall_brightness=0.5,
        overall_punch=0.8,
        room_amount=0.0,
        genre_affinity={"trap": 1.0, "hip-hop": 1.0, "r-and-b": 0.8, "electro": 0.9},
        description="Roland TR-808 drum machine - hip-hop essential",
    ),
    "tr909": DrumKit(
        id="tr909",
        name="TR-909",
        style=DrumKitStyle.STUDIO,
        machine=DrumMachine.TR909,
        overall_brightness=0.7,
        overall_punch=0.9,
        room_amount=0.0,
        genre_affinity={"house": 1.0, "techno": 1.0, "trance": 0.9, "edm": 0.8},
        description="Roland TR-909 - house and techno foundation",
    ),
    "linndrum": DrumKit(
        id="linndrum",
        name="LinnDrum",
        style=DrumKitStyle.STUDIO,
        machine=DrumMachine.LINNDRUM,
        overall_brightness=0.6,
        overall_punch=0.6,
        room_amount=0.1,
        genre_affinity={"80s": 1.0, "synthpop": 1.0, "pop": 0.8},
        description="LinnDrum - the definitive 80s sound",
    ),
    "orchestral": DrumKit(
        id="orchestral",
        name="Orchestral Percussion",
        style=DrumKitStyle.ORCHESTRAL,
        overall_brightness=0.5,
        overall_punch=0.6,
        room_amount=0.7,
        genre_affinity={"classical": 1.0, "cinematic": 1.0, "orchestral": 1.0},
        description="Concert hall percussion ensemble",
    ),
    "lo-fi": DrumKit(
        id="lo-fi",
        name="Lo-Fi Kit",
        style=DrumKitStyle.VINTAGE,
        overall_brightness=0.3,
        overall_punch=0.4,
        room_amount=0.2,
        genre_affinity={"lo-fi-hip-hop": 1.0, "chillhop": 0.9, "boom-bap": 0.8},
        description="Dusty, vinyl-sampled drums",
    ),
}


# ============================================================================
# World Percussion Collections
# ============================================================================

LATIN_PERCUSSION: dict[str, DrumSound] = {
    "clave_3_2": DrumSound(
        id="clave_3_2",
        name="Clave (3-2 Son)",
        category=DrumCategory.WORLD,
        gm_note=75,
        decay_ms=80,
        brightness=1.0,
        genre_affinity={"salsa": 1.0, "latin": 1.0, "cuban": 1.0},
        description="Foundational Cuban rhythm pattern",
    ),
    "tumbadora_open": DrumSound(
        id="tumbadora_open",
        name="Tumbadora (Open)",
        category=DrumCategory.WORLD,
        gm_note=63,
        pitch_hz=180,
        decay_ms=300,
        body=0.9,
        genre_affinity={"salsa": 1.0, "latin": 1.0},
    ),
    "tumbadora_slap": DrumSound(
        id="tumbadora_slap",
        name="Tumbadora (Slap)",
        category=DrumCategory.WORLD,
        gm_note=62,
        decay_ms=80,
        brightness=0.9,
        click=0.9,
        genre_affinity={"salsa": 1.0, "latin": 1.0},
    ),
    "bongo_martillo": DrumSound(
        id="bongo_martillo",
        name="Bongo Martillo",
        category=DrumCategory.WORLD,
        gm_note=60,
        decay_ms=100,
        genre_affinity={"salsa": 1.0, "cuban": 1.0},
    ),
    "timbale_cascara": DrumSound(
        id="timbale_cascara",
        name="Timbale Cascara",
        category=DrumCategory.WORLD,
        gm_note=65,
        decay_ms=50,
        brightness=0.95,
        genre_affinity={"salsa": 1.0, "latin": 0.9},
    ),
    "guiro": DrumSound(
        id="guiro",
        name="Guiro",
        category=DrumCategory.WORLD,
        gm_note=74,
        decay_ms=200,
        genre_affinity={"salsa": 0.9, "latin": 0.9},
    ),
}


AFRICAN_PERCUSSION: dict[str, DrumSound] = {
    "djembe_bass": DrumSound(
        id="djembe_bass",
        name="Djembe (Bass)",
        category=DrumCategory.WORLD,
        gm_note=64,
        pitch_hz=100,
        decay_ms=250,
        body=1.0,
        genre_affinity={"african": 1.0, "world": 0.9},
        description="Deep bass tone from djembe center",
    ),
    "djembe_tone": DrumSound(
        id="djembe_tone",
        name="Djembe (Tone)",
        category=DrumCategory.WORLD,
        gm_note=63,
        pitch_hz=200,
        decay_ms=180,
        body=0.7,
        genre_affinity={"african": 1.0, "world": 0.9},
    ),
    "djembe_slap": DrumSound(
        id="djembe_slap",
        name="Djembe (Slap)",
        category=DrumCategory.WORLD,
        gm_note=62,
        decay_ms=80,
        brightness=0.9,
        click=1.0,
        genre_affinity={"african": 1.0, "world": 0.9},
    ),
    "dundun": DrumSound(
        id="dundun",
        name="Dundun",
        category=DrumCategory.WORLD,
        gm_note=41,
        pitch_hz=70,
        decay_ms=400,
        body=1.0,
        genre_affinity={"african": 1.0, "world": 0.8},
    ),
    "sangban": DrumSound(
        id="sangban",
        name="Sangban",
        category=DrumCategory.WORLD,
        gm_note=47,
        pitch_hz=100,
        decay_ms=350,
        body=0.9,
        genre_affinity={"african": 1.0},
    ),
    "kenkeni": DrumSound(
        id="kenkeni",
        name="Kenkeni",
        category=DrumCategory.WORLD,
        gm_note=50,
        pitch_hz=150,
        decay_ms=280,
        body=0.8,
        genre_affinity={"african": 1.0},
    ),
    "shekere": DrumSound(
        id="shekere",
        name="Shekere",
        category=DrumCategory.WORLD,
        gm_note=82,
        decay_ms=100,
        brightness=0.8,
        genre_affinity={"african": 1.0, "afrobeat": 0.9},
    ),
    "talking_drum": DrumSound(
        id="talking_drum",
        name="Talking Drum",
        category=DrumCategory.WORLD,
        gm_note=117,
        pitch_hz=200,
        decay_ms=300,
        genre_affinity={"african": 1.0, "world": 0.8},
        description="Variable pitch drum - can 'talk'",
    ),
}


ASIAN_PERCUSSION: dict[str, DrumSound] = {
    "taiko_odaiko": DrumSound(
        id="taiko_odaiko",
        name="Taiko Odaiko",
        category=DrumCategory.WORLD,
        gm_note=116,
        pitch_hz=50,
        decay_ms=600,
        body=1.0,
        genre_affinity={"japanese": 1.0, "cinematic": 0.9, "world": 0.8},
        description="Large Japanese taiko drum - powerful, cinematic",
    ),
    "taiko_shime": DrumSound(
        id="taiko_shime",
        name="Taiko Shime-daiko",
        category=DrumCategory.WORLD,
        gm_note=117,
        pitch_hz=250,
        decay_ms=150,
        brightness=0.8,
        genre_affinity={"japanese": 1.0},
    ),
    "tabla_dayan": DrumSound(
        id="tabla_dayan",
        name="Tabla Dayan",
        category=DrumCategory.WORLD,
        gm_note=60,
        pitch_hz=300,
        decay_ms=150,
        brightness=0.7,
        genre_affinity={"indian": 1.0, "world": 0.8},
        description="Small tabla drum - melodic, expressive",
    ),
    "tabla_bayan": DrumSound(
        id="tabla_bayan",
        name="Tabla Bayan",
        category=DrumCategory.WORLD,
        gm_note=64,
        pitch_hz=80,
        decay_ms=350,
        body=0.9,
        genre_affinity={"indian": 1.0, "world": 0.8},
    ),
    "dholak": DrumSound(
        id="dholak",
        name="Dholak",
        category=DrumCategory.WORLD,
        gm_note=61,
        decay_ms=200,
        body=0.8,
        genre_affinity={"indian": 0.9, "bhangra": 1.0},
    ),
    "gamelan_gong": DrumSound(
        id="gamelan_gong",
        name="Gamelan Gong",
        category=DrumCategory.WORLD,
        gm_note=84,
        decay_ms=3000,
        brightness=0.6,
        sustain=0.9,
        genre_affinity={"indonesian": 1.0, "world": 0.7, "ambient": 0.6},
    ),
    "gamelan_kenong": DrumSound(
        id="gamelan_kenong",
        name="Gamelan Kenong",
        category=DrumCategory.WORLD,
        gm_note=9,
        decay_ms=1500,
        brightness=0.8,
        genre_affinity={"indonesian": 1.0, "world": 0.6},
    ),
}


ORCHESTRAL_PERCUSSION: dict[str, DrumSound] = {
    "timpani_low": DrumSound(
        id="timpani_low",
        name="Timpani (Low)",
        category=DrumCategory.ORCHESTRAL,
        gm_note=47,
        pitch_hz=73,
        decay_ms=800,
        body=1.0,
        genre_affinity={"classical": 1.0, "cinematic": 1.0, "orchestral": 1.0},
    ),
    "timpani_mid": DrumSound(
        id="timpani_mid",
        name="Timpani (Mid)",
        category=DrumCategory.ORCHESTRAL,
        gm_note=47,
        pitch_hz=98,
        decay_ms=700,
        body=0.9,
        genre_affinity={"classical": 1.0, "cinematic": 1.0},
    ),
    "timpani_high": DrumSound(
        id="timpani_high",
        name="Timpani (High)",
        category=DrumCategory.ORCHESTRAL,
        gm_note=47,
        pitch_hz=131,
        decay_ms=600,
        body=0.8,
        genre_affinity={"classical": 1.0, "cinematic": 1.0},
    ),
    "timpani_roll": DrumSound(
        id="timpani_roll",
        name="Timpani Roll",
        category=DrumCategory.ORCHESTRAL,
        gm_note=47,
        decay_ms=3000,
        sustain=0.9,
        genre_affinity={"classical": 1.0, "cinematic": 1.0},
    ),
    "bass_drum_concert": DrumSound(
        id="bass_drum_concert",
        name="Concert Bass Drum",
        category=DrumCategory.ORCHESTRAL,
        gm_note=35,
        pitch_hz=40,
        decay_ms=1000,
        body=1.0,
        genre_affinity={"classical": 1.0, "cinematic": 1.0},
    ),
    "snare_concert": DrumSound(
        id="snare_concert",
        name="Concert Snare",
        category=DrumCategory.ORCHESTRAL,
        gm_note=38,
        decay_ms=180,
        brightness=0.6,
        genre_affinity={"classical": 1.0, "marching": 0.9},
    ),
    "snare_roll": DrumSound(
        id="snare_roll",
        name="Snare Roll",
        category=DrumCategory.ORCHESTRAL,
        gm_note=38,
        decay_ms=2000,
        sustain=0.9,
        genre_affinity={"classical": 1.0, "cinematic": 0.8},
    ),
    "cymbal_crash_orchestral": DrumSound(
        id="cymbal_crash_orchestral",
        name="Orchestral Crash Cymbals",
        category=DrumCategory.ORCHESTRAL,
        gm_note=49,
        decay_ms=3000,
        brightness=0.8,
        sustain=0.9,
        genre_affinity={"classical": 1.0, "cinematic": 1.0},
    ),
    "cymbal_suspended": DrumSound(
        id="cymbal_suspended",
        name="Suspended Cymbal",
        category=DrumCategory.ORCHESTRAL,
        gm_note=51,
        decay_ms=4000,
        brightness=0.7,
        sustain=0.95,
        genre_affinity={"classical": 1.0, "cinematic": 0.9},
    ),
    "cymbal_roll": DrumSound(
        id="cymbal_roll",
        name="Cymbal Roll",
        category=DrumCategory.ORCHESTRAL,
        gm_note=51,
        decay_ms=5000,
        sustain=1.0,
        genre_affinity={"classical": 1.0, "cinematic": 1.0},
    ),
    "tam_tam": DrumSound(
        id="tam_tam",
        name="Tam-Tam (Gong)",
        category=DrumCategory.ORCHESTRAL,
        gm_note=84,
        decay_ms=8000,
        brightness=0.4,
        sustain=1.0,
        genre_affinity={"classical": 0.9, "cinematic": 1.0},
    ),
    "triangle": DrumSound(
        id="triangle",
        name="Triangle",
        category=DrumCategory.ORCHESTRAL,
        gm_note=81,
        decay_ms=2000,
        brightness=1.0,
        sustain=0.8,
        genre_affinity={"classical": 1.0, "cinematic": 0.7},
    ),
    "castanets": DrumSound(
        id="castanets",
        name="Castanets",
        category=DrumCategory.ORCHESTRAL,
        gm_note=85,
        decay_ms=40,
        brightness=0.9,
        genre_affinity={"spanish": 1.0, "classical": 0.7},
    ),
    "wood_block": DrumSound(
        id="wood_block",
        name="Wood Block",
        category=DrumCategory.ORCHESTRAL,
        gm_note=76,
        decay_ms=50,
        brightness=0.9,
        genre_affinity={"classical": 0.8, "cinematic": 0.6},
    ),
    "temple_blocks": DrumSound(
        id="temple_blocks",
        name="Temple Blocks",
        category=DrumCategory.ORCHESTRAL,
        gm_note=76,
        decay_ms=80,
        brightness=0.7,
        genre_affinity={"classical": 0.8, "world": 0.7},
    ),
    "tubular_bells": DrumSound(
        id="tubular_bells",
        name="Tubular Bells",
        category=DrumCategory.ORCHESTRAL,
        gm_note=14,
        decay_ms=4000,
        brightness=0.8,
        sustain=0.9,
        genre_affinity={"classical": 1.0, "cinematic": 1.0},
    ),
    "glockenspiel": DrumSound(
        id="glockenspiel",
        name="Glockenspiel",
        category=DrumCategory.ORCHESTRAL,
        gm_note=9,
        decay_ms=1500,
        brightness=1.0,
        genre_affinity={"classical": 1.0, "cinematic": 0.8},
    ),
    "xylophone": DrumSound(
        id="xylophone",
        name="Xylophone",
        category=DrumCategory.ORCHESTRAL,
        gm_note=13,
        decay_ms=500,
        brightness=0.95,
        genre_affinity={"classical": 1.0, "cinematic": 0.7},
    ),
    "vibraphone": DrumSound(
        id="vibraphone",
        name="Vibraphone",
        category=DrumCategory.ORCHESTRAL,
        gm_note=11,
        decay_ms=2500,
        brightness=0.7,
        sustain=0.8,
        genre_affinity={"jazz": 1.0, "classical": 0.8},
    ),
    "marimba": DrumSound(
        id="marimba",
        name="Marimba",
        category=DrumCategory.ORCHESTRAL,
        gm_note=12,
        decay_ms=800,
        brightness=0.5,
        body=0.8,
        genre_affinity={"classical": 0.9, "world": 0.8},
    ),
    "crotales": DrumSound(
        id="crotales",
        name="Crotales (Antique Cymbals)",
        category=DrumCategory.ORCHESTRAL,
        gm_note=9,
        decay_ms=3000,
        brightness=1.0,
        sustain=0.9,
        genre_affinity={"classical": 0.9, "cinematic": 0.8},
    ),
}


# ============================================================================
# Lookup Functions
# ============================================================================

def get_drum_sound_by_gm(note: int) -> Optional[DrumSound]:
    """Get drum sound by GM note number."""
    return GM_DRUM_MAP.get(note)


def get_drum_kit(kit_id: str) -> Optional[DrumKit]:
    """Get drum kit by ID."""
    return DRUM_KITS.get(kit_id)


def get_drum_machine_sounds(machine: DrumMachine) -> dict[str, DrumSound]:
    """Get all sounds for a drum machine."""
    if machine == DrumMachine.TR808:
        return TR808_SOUNDS
    elif machine == DrumMachine.TR909:
        return TR909_SOUNDS
    elif machine == DrumMachine.LINNDRUM:
        return LINNDRUM_SOUNDS
    return {}


def get_drums_by_category(category: DrumCategory) -> list[DrumSound]:
    """Get all drum sounds in a category."""
    sounds = []
    for sound in GM_DRUM_MAP.values():
        if sound.category == category:
            sounds.append(sound)
    return sounds


def get_drums_for_genre(genre: str, min_affinity: float = 0.7) -> list[DrumSound]:
    """Get drum sounds suited for a genre."""
    sounds = []
    for sound in GM_DRUM_MAP.values():
        if sound.genre_affinity.get(genre, 0) >= min_affinity:
            sounds.append(sound)
    for sound in TR808_SOUNDS.values():
        if sound.genre_affinity.get(genre, 0) >= min_affinity:
            sounds.append(sound)
    for sound in TR909_SOUNDS.values():
        if sound.genre_affinity.get(genre, 0) >= min_affinity:
            sounds.append(sound)
    return sounds


def get_kit_for_genre(genre: str) -> Optional[DrumKit]:
    """Get the best drum kit for a genre."""
    best_kit = None
    best_affinity = 0.0

    for kit in DRUM_KITS.values():
        affinity = kit.genre_affinity.get(genre, 0)
        if affinity > best_affinity:
            best_affinity = affinity
            best_kit = kit

    return best_kit


def get_world_percussion(region: str) -> dict[str, DrumSound]:
    """Get world percussion sounds by region."""
    region = region.lower()
    if region in ["latin", "cuban", "salsa", "brazilian"]:
        return LATIN_PERCUSSION
    elif region in ["african", "west-african", "afrobeat"]:
        return AFRICAN_PERCUSSION
    elif region in ["asian", "japanese", "indian", "indonesian"]:
        return ASIAN_PERCUSSION
    return {}


# ============================================================================
# Velocity & Dynamics Helpers
# ============================================================================

def get_ghost_note_velocity(sound: DrumSound) -> int:
    """Get appropriate velocity for ghost notes."""
    return sound.ghost_velocity


def get_accent_velocity(sound: DrumSound) -> int:
    """Get appropriate velocity for accents."""
    return sound.accent_velocity


def humanize_velocity(base_velocity: int, amount: float = 0.1) -> int:
    """Add human-like velocity variation."""
    import random
    variation = int(base_velocity * amount)
    return max(1, min(127, base_velocity + random.randint(-variation, variation)))


def apply_velocity_curve(velocity: int, curve: str) -> int:
    """Apply a velocity curve transformation."""
    if curve == "soft":
        # Compress dynamics - softer feel
        return int(64 + (velocity - 64) * 0.7)
    elif curve == "hard":
        # Expand dynamics - more aggressive
        return int(64 + (velocity - 64) * 1.3)
    elif curve == "fixed":
        return 100
    return velocity  # linear
