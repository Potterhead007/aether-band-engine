"""
AETHER FL Studio Integration Provider - Institutional Grade

Professional-grade FL Studio project package generation with:
- MIDI export with full track separation
- Comprehensive mixer routing presets per genre
- Plugin recommendations with specific presets
- Effects chain templates
- Automation suggestions
- Color-coded channel organization

The package format (.flpkg) includes:
- project.mid - Full MIDI with all tracks
- manifest.json - Complete project metadata and setup instructions
- mixer_preset.json - Genre-optimized mixer routing
- channels/ - Individual track MIDI files

Example:
    provider = FLStudioProvider()
    await provider.initialize()

    package_path = await provider.export_package(
        midi_data=midi_file,
        output_dir=Path("output"),
        genre="synthwave",
        project_name="My Song",
    )
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from aether.providers.base import (
    BaseProvider,
    MIDIFile,
    MIDINote,
    MIDITrack,
    ProviderInfo,
    ProviderStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# FL Studio Constants - Institutional Grade
# ============================================================================

class FLPluginCategory(Enum):
    """FL Studio native plugin categories."""
    SYNTHESIZER = "synthesizer"
    SAMPLER = "sampler"
    EFFECT = "effect"
    GENERATOR = "generator"


# FL Studio native plugins with presets and capabilities
FL_NATIVE_PLUGINS = {
    # Synthesizers
    "Sytrus": {
        "category": FLPluginCategory.SYNTHESIZER,
        "best_for": ["bass", "lead", "pad", "pluck", "fm"],
        "presets": {
            "bass": ["Sub Bass", "Reese Bass", "Acid Bass", "Wobble Bass"],
            "lead": ["Saw Lead", "Square Lead", "Supersaw", "Trance Lead"],
            "pad": ["Warm Pad", "String Pad", "Atmosphere", "Evolving Pad"],
            "pluck": ["Pluck Lead", "Bell Pluck", "Marimba"],
        },
    },
    "Harmor": {
        "category": FLPluginCategory.SYNTHESIZER,
        "best_for": ["pad", "texture", "ambient", "experimental"],
        "presets": {
            "pad": ["Warm Strings", "Glass Pad", "Vocal Pad"],
            "texture": ["Atmosphere", "Drone", "Evolve"],
        },
    },
    "FLEX": {
        "category": FLPluginCategory.SYNTHESIZER,
        "best_for": ["strings", "brass", "woodwind", "ethnic", "orchestral"],
        "presets": {
            "strings": ["Full Strings", "Violin Section", "Cello"],
            "brass": ["Brass Section", "French Horns", "Trumpet"],
            "woodwind": ["Flute", "Clarinet", "Oboe"],
        },
    },
    "FL Keys": {
        "category": FLPluginCategory.SYNTHESIZER,
        "best_for": ["piano", "keys", "organ", "electric_piano"],
        "presets": {
            "piano": ["Grand Piano", "Bright Piano", "Soft Piano"],
            "organ": ["B3 Organ", "Church Organ", "Jazz Organ"],
            "electric_piano": ["Rhodes", "Wurlitzer", "DX Piano"],
        },
    },
    "3x Osc": {
        "category": FLPluginCategory.SYNTHESIZER,
        "best_for": ["bass", "lead", "simple"],
        "presets": {
            "bass": ["Sub", "Square Bass"],
            "lead": ["Saw Lead", "PWM Lead"],
        },
    },
    "GMS": {
        "category": FLPluginCategory.SYNTHESIZER,
        "best_for": ["all-purpose", "gm"],
        "presets": {},
    },
    "FPC": {
        "category": FLPluginCategory.SAMPLER,
        "best_for": ["drums", "percussion", "one-shots"],
        "kits": {
            "electronic": ["808", "909", "707", "Trap Kit"],
            "acoustic": ["Standard Kit", "Jazz Kit", "Rock Kit"],
            "hybrid": ["Hybrid Kit", "Lo-Fi Kit"],
        },
    },
    "DirectWave": {
        "category": FLPluginCategory.SAMPLER,
        "best_for": ["samples", "multisamples", "realistic"],
        "presets": {},
    },
    "Slicex": {
        "category": FLPluginCategory.SAMPLER,
        "best_for": ["loops", "breaks", "chopped"],
        "presets": {},
    },
}

# Effects with genre-specific recommendations
FL_EFFECTS = {
    "Fruity Parametric EQ 2": {
        "type": "eq",
        "presets": {
            "kick_punch": {"low_shelf": 60, "low_boost": 3, "mid_cut": 400, "mid_q": 2},
            "snare_crack": {"high_shelf": 5000, "high_boost": 2, "mid_boost": 200},
            "bass_clarity": {"high_pass": 30, "low_boost": 80, "mid_cut": 250},
            "vocal_presence": {"high_shelf": 3000, "high_boost": 2, "mid_cut": 400},
        },
    },
    "Fruity Compressor": {
        "type": "dynamics",
        "presets": {
            "drums_punch": {"ratio": 4, "attack": 10, "release": 100, "threshold": -12},
            "bass_control": {"ratio": 3, "attack": 20, "release": 200, "threshold": -10},
            "glue": {"ratio": 2, "attack": 30, "release": 300, "threshold": -8},
        },
    },
    "Fruity Reverb 2": {
        "type": "space",
        "presets": {
            "room": {"size": 30, "decay": 1.2, "damping": 0.5, "wet": 20},
            "hall": {"size": 70, "decay": 2.5, "damping": 0.3, "wet": 25},
            "plate": {"size": 50, "decay": 1.8, "damping": 0.4, "wet": 30},
        },
    },
    "Fruity Delay 3": {
        "type": "time",
        "presets": {
            "quarter": {"time": "1/4", "feedback": 30, "wet": 20},
            "eighth": {"time": "1/8", "feedback": 40, "wet": 25},
            "dotted_eighth": {"time": "1/8D", "feedback": 35, "wet": 22},
            "pingpong": {"time": "1/4", "feedback": 45, "wet": 30, "pingpong": True},
        },
    },
    "Fruity Limiter": {
        "type": "dynamics",
        "presets": {
            "master": {"ceiling": -0.3, "release": 100, "gain": 0},
            "bus": {"ceiling": -1, "release": 150, "gain": 2},
        },
    },
    "Soundgoodizer": {
        "type": "enhancer",
        "presets": {"A": 50, "B": 40, "C": 60, "D": 30},
    },
    "Maximus": {
        "type": "multiband",
        "presets": {
            "master": {"low": {"comp": 2, "gain": 0}, "mid": {"comp": 1.5, "gain": 1}, "high": {"comp": 2, "gain": 0.5}},
        },
    },
    "Fruity Soft Clipper": {
        "type": "saturation",
        "presets": {"subtle": {"threshold": -3, "post_gain": 2}, "aggressive": {"threshold": -6, "post_gain": 4}},
    },
    "Vocodex": {
        "type": "vocoder",
        "presets": {},
    },
    "Gross Beat": {
        "type": "time",
        "presets": {"half_speed": {}, "stutter": {}, "tape_stop": {}},
    },
}

# FL Studio track colors (RGB hex values)
FL_TRACK_COLORS = {
    "drums": 0xFF5722,      # Orange
    "kick": 0xFF5722,       # Orange
    "snare": 0xFF7043,      # Light Orange
    "hihat": 0xFFAB91,      # Pale Orange
    "percussion": 0xFFCCBC, # Very Light Orange
    "bass": 0x9C27B0,       # Purple
    "sub": 0x7B1FA2,        # Dark Purple
    "keys": 0x2196F3,       # Blue
    "piano": 0x1976D2,      # Dark Blue
    "organ": 0x42A5F5,      # Light Blue
    "guitar": 0x4CAF50,     # Green
    "acoustic": 0x66BB6A,   # Light Green
    "electric": 0x388E3C,   # Dark Green
    "strings": 0xE91E63,    # Pink
    "violin": 0xEC407A,     # Light Pink
    "cello": 0xC2185B,      # Dark Pink
    "brass": 0xFFEB3B,      # Yellow
    "horn": 0xFDD835,       # Dark Yellow
    "trumpet": 0xFFEE58,    # Light Yellow
    "synth": 0x00BCD4,      # Cyan
    "pad": 0x673AB7,        # Deep Purple
    "lead": 0xFF9800,       # Amber
    "pluck": 0xFFB74D,      # Light Amber
    "arp": 0xFFA726,        # Orange Amber
    "fx": 0x607D8B,         # Blue Grey
    "riser": 0x78909C,      # Light Blue Grey
    "impact": 0x455A64,     # Dark Blue Grey
    "vocals": 0xF44336,     # Red
    "choir": 0xEF5350,      # Light Red
    "ambient": 0x795548,    # Brown
    "texture": 0x8D6E63,    # Light Brown
    "master": 0x212121,     # Near Black
    "bus": 0x424242,        # Dark Grey
    "default": 0x9E9E9E,    # Grey
}

# GM Program to FL Studio plugin mapping
GM_TO_FL_PLUGIN = {
    # Pianos (0-7)
    range(0, 8): ("FL Keys", "piano"),
    # Chromatic Percussion (8-15)
    range(8, 16): ("FLEX", "mallet"),
    # Organ (16-23)
    range(16, 24): ("FL Keys", "organ"),
    # Guitar (24-31)
    range(24, 32): ("FLEX", "guitar"),
    # Bass (32-39)
    range(32, 40): ("Sytrus", "bass"),
    # Strings (40-47)
    range(40, 48): ("FLEX", "strings"),
    # Ensemble (48-55)
    range(48, 56): ("FLEX", "ensemble"),
    # Brass (56-63)
    range(56, 64): ("FLEX", "brass"),
    # Reed (64-71)
    range(64, 72): ("FLEX", "woodwind"),
    # Pipe (72-79)
    range(72, 80): ("FLEX", "woodwind"),
    # Synth Lead (80-87)
    range(80, 88): ("Sytrus", "lead"),
    # Synth Pad (88-95)
    range(88, 96): ("Sytrus", "pad"),
    # Synth Effects (96-103)
    range(96, 104): ("Sytrus", "fx"),
    # Ethnic (104-111)
    range(104, 112): ("FLEX", "ethnic"),
    # Percussive (112-119)
    range(112, 120): ("FPC", "percussion"),
    # Sound Effects (120-127)
    range(120, 128): ("Sytrus", "fx"),
}

# FPC drum pad mapping (GM drums to FPC pads)
FPC_PAD_MAP = {
    35: 0,   # Acoustic Bass Drum -> Pad 1
    36: 0,   # Bass Drum 1 -> Pad 1
    37: 10,  # Side Stick -> Pad 11
    38: 1,   # Acoustic Snare -> Pad 2
    39: 9,   # Hand Clap -> Pad 10
    40: 1,   # Electric Snare -> Pad 2
    41: 4,   # Low Floor Tom -> Pad 5
    42: 2,   # Closed Hi-Hat -> Pad 3
    43: 4,   # High Floor Tom -> Pad 5
    44: 2,   # Pedal Hi-Hat -> Pad 3
    45: 5,   # Low Tom -> Pad 6
    46: 3,   # Open Hi-Hat -> Pad 4
    47: 5,   # Low-Mid Tom -> Pad 6
    48: 6,   # Hi-Mid Tom -> Pad 7
    49: 7,   # Crash Cymbal 1 -> Pad 8
    50: 6,   # High Tom -> Pad 7
    51: 8,   # Ride Cymbal 1 -> Pad 9
    52: 7,   # Chinese Cymbal -> Pad 8
    53: 15,  # Ride Bell -> Pad 16
    54: 12,  # Tambourine -> Pad 13
    55: 7,   # Splash Cymbal -> Pad 8
    56: 11,  # Cowbell -> Pad 12
    57: 7,   # Crash Cymbal 2 -> Pad 8
    63: 13,  # High Conga -> Pad 14
    64: 14,  # Low Conga -> Pad 15
}


# ============================================================================
# Genre-Specific Mixer Templates
# ============================================================================

GENRE_MIXER_TEMPLATES = {
    "synthwave": {
        "description": "80s-inspired analog warmth with lush reverb",
        "master_chain": ["Fruity Parametric EQ 2", "Fruity Soft Clipper", "Fruity Limiter"],
        "bus_groups": {
            "drums": {
                "tracks": ["kick", "snare", "hihat", "percussion"],
                "effects": [
                    ("Fruity Parametric EQ 2", "drums_punch"),
                    ("Fruity Compressor", "drums_punch"),
                ],
                "send_reverb": 15,
                "send_delay": 10,
            },
            "bass": {
                "tracks": ["bass", "sub"],
                "effects": [
                    ("Fruity Parametric EQ 2", "bass_clarity"),
                    ("Fruity Soft Clipper", "subtle"),
                ],
                "send_reverb": 0,
                "send_delay": 0,
            },
            "synths": {
                "tracks": ["lead", "pad", "arp"],
                "effects": [
                    ("Fruity Parametric EQ 2", None),
                    ("Fruity Compressor", "glue"),
                ],
                "send_reverb": 35,
                "send_delay": 25,
            },
        },
        "reverb_settings": {"type": "hall", "decay": 2.8, "wet": 100},
        "delay_settings": {"time": "1/8D", "feedback": 35, "wet": 100},
        "recommended_tempo": (85, 118),
    },
    "house": {
        "description": "Four-on-the-floor energy with punchy drums",
        "master_chain": ["Fruity Parametric EQ 2", "Maximus", "Fruity Limiter"],
        "bus_groups": {
            "drums": {
                "tracks": ["kick", "snare", "hihat", "percussion"],
                "effects": [
                    ("Fruity Parametric EQ 2", "kick_punch"),
                    ("Fruity Compressor", "drums_punch"),
                ],
                "send_reverb": 10,
                "send_delay": 5,
            },
            "bass": {
                "tracks": ["bass"],
                "effects": [
                    ("Fruity Parametric EQ 2", "bass_clarity"),
                    ("Fruity Compressor", "bass_control"),
                ],
                "send_reverb": 0,
                "send_delay": 0,
                "sidechain_to": "kick",
            },
            "synths": {
                "tracks": ["lead", "pad", "stab"],
                "effects": [("Fruity Compressor", "glue")],
                "send_reverb": 25,
                "send_delay": 20,
                "sidechain_to": "kick",
            },
        },
        "reverb_settings": {"type": "room", "decay": 1.5, "wet": 100},
        "delay_settings": {"time": "1/4", "feedback": 25, "wet": 100},
        "recommended_tempo": (120, 130),
    },
    "trap": {
        "description": "Hard-hitting 808s with crispy hi-hats",
        "master_chain": ["Fruity Parametric EQ 2", "Fruity Soft Clipper", "Maximus", "Fruity Limiter"],
        "bus_groups": {
            "drums": {
                "tracks": ["kick", "snare", "hihat", "percussion"],
                "effects": [
                    ("Fruity Parametric EQ 2", "drums_punch"),
                ],
                "send_reverb": 5,
                "send_delay": 0,
            },
            "808": {
                "tracks": ["bass", "sub"],
                "effects": [
                    ("Fruity Parametric EQ 2", "bass_clarity"),
                    ("Fruity Soft Clipper", "aggressive"),
                ],
                "send_reverb": 0,
                "send_delay": 0,
            },
            "melodic": {
                "tracks": ["lead", "pad", "keys"],
                "effects": [("Fruity Compressor", "glue")],
                "send_reverb": 40,
                "send_delay": 30,
            },
        },
        "reverb_settings": {"type": "hall", "decay": 3.0, "wet": 100},
        "delay_settings": {"time": "1/4", "feedback": 40, "wet": 100},
        "recommended_tempo": (130, 175),
    },
    "lo-fi-hip-hop": {
        "description": "Warm, dusty vibes with vinyl character",
        "master_chain": ["Fruity Parametric EQ 2", "Fruity Soft Clipper", "Fruity Limiter"],
        "bus_groups": {
            "drums": {
                "tracks": ["kick", "snare", "hihat"],
                "effects": [
                    ("Fruity Parametric EQ 2", None),
                    ("Fruity Compressor", "drums_punch"),
                ],
                "send_reverb": 20,
                "send_delay": 15,
            },
            "bass": {
                "tracks": ["bass"],
                "effects": [("Fruity Soft Clipper", "subtle")],
                "send_reverb": 10,
                "send_delay": 0,
            },
            "melodic": {
                "tracks": ["keys", "guitar", "pad"],
                "effects": [("Fruity Compressor", "glue")],
                "send_reverb": 35,
                "send_delay": 25,
            },
        },
        "reverb_settings": {"type": "room", "decay": 1.8, "wet": 100},
        "delay_settings": {"time": "1/8", "feedback": 30, "wet": 100},
        "recommended_tempo": (70, 95),
        "special_fx": ["vinyl_noise", "tape_wobble", "low_pass_filter"],
    },
    "techno": {
        "description": "Driving industrial energy with hypnotic elements",
        "master_chain": ["Fruity Parametric EQ 2", "Maximus", "Fruity Limiter"],
        "bus_groups": {
            "drums": {
                "tracks": ["kick", "snare", "hihat", "percussion"],
                "effects": [
                    ("Fruity Parametric EQ 2", "kick_punch"),
                    ("Fruity Compressor", "drums_punch"),
                ],
                "send_reverb": 15,
                "send_delay": 20,
            },
            "bass": {
                "tracks": ["bass"],
                "effects": [
                    ("Fruity Parametric EQ 2", "bass_clarity"),
                ],
                "send_reverb": 5,
                "send_delay": 10,
            },
            "synths": {
                "tracks": ["lead", "pad", "arp"],
                "effects": [("Fruity Compressor", "glue")],
                "send_reverb": 30,
                "send_delay": 35,
            },
        },
        "reverb_settings": {"type": "hall", "decay": 2.5, "wet": 100},
        "delay_settings": {"time": "1/8", "feedback": 45, "wet": 100, "pingpong": True},
        "recommended_tempo": (125, 150),
    },
    "jazz": {
        "description": "Warm acoustic space with natural dynamics",
        "master_chain": ["Fruity Parametric EQ 2", "Fruity Compressor", "Fruity Limiter"],
        "bus_groups": {
            "drums": {
                "tracks": ["kick", "snare", "hihat", "ride"],
                "effects": [("Fruity Parametric EQ 2", None)],
                "send_reverb": 25,
                "send_delay": 0,
            },
            "bass": {
                "tracks": ["bass", "upright"],
                "effects": [("Fruity Parametric EQ 2", "bass_clarity")],
                "send_reverb": 15,
                "send_delay": 0,
            },
            "melodic": {
                "tracks": ["piano", "keys", "guitar", "brass", "woodwind"],
                "effects": [("Fruity Compressor", "glue")],
                "send_reverb": 30,
                "send_delay": 10,
            },
        },
        "reverb_settings": {"type": "hall", "decay": 2.0, "wet": 100},
        "delay_settings": {"time": "1/4", "feedback": 20, "wet": 100},
        "recommended_tempo": (60, 180),
    },
    "drum-and-bass": {
        "description": "Fast breakbeats with deep sub bass",
        "master_chain": ["Fruity Parametric EQ 2", "Maximus", "Fruity Limiter"],
        "bus_groups": {
            "drums": {
                "tracks": ["kick", "snare", "hihat", "break"],
                "effects": [
                    ("Fruity Parametric EQ 2", "drums_punch"),
                    ("Fruity Compressor", "drums_punch"),
                ],
                "send_reverb": 10,
                "send_delay": 15,
            },
            "bass": {
                "tracks": ["bass", "sub", "reese"],
                "effects": [
                    ("Fruity Parametric EQ 2", "bass_clarity"),
                    ("Fruity Soft Clipper", "subtle"),
                ],
                "send_reverb": 5,
                "send_delay": 0,
            },
            "synths": {
                "tracks": ["lead", "pad", "stab"],
                "effects": [("Fruity Compressor", "glue")],
                "send_reverb": 35,
                "send_delay": 25,
            },
        },
        "reverb_settings": {"type": "plate", "decay": 1.5, "wet": 100},
        "delay_settings": {"time": "1/8", "feedback": 35, "wet": 100},
        "recommended_tempo": (160, 180),
    },
    "ambient": {
        "description": "Expansive soundscapes with ethereal textures",
        "master_chain": ["Fruity Parametric EQ 2", "Fruity Limiter"],
        "bus_groups": {
            "pads": {
                "tracks": ["pad", "drone", "texture"],
                "effects": [("Fruity Parametric EQ 2", None)],
                "send_reverb": 60,
                "send_delay": 40,
            },
            "melodic": {
                "tracks": ["lead", "bell", "keys"],
                "effects": [],
                "send_reverb": 50,
                "send_delay": 45,
            },
        },
        "reverb_settings": {"type": "hall", "decay": 5.0, "wet": 100},
        "delay_settings": {"time": "1/4", "feedback": 50, "wet": 100, "pingpong": True},
        "recommended_tempo": (60, 100),
    },
    "rock": {
        "description": "Punchy drums with guitar-driven energy",
        "master_chain": ["Fruity Parametric EQ 2", "Fruity Compressor", "Fruity Limiter"],
        "bus_groups": {
            "drums": {
                "tracks": ["kick", "snare", "hihat", "toms", "cymbals"],
                "effects": [
                    ("Fruity Parametric EQ 2", "drums_punch"),
                    ("Fruity Compressor", "drums_punch"),
                ],
                "send_reverb": 20,
                "send_delay": 5,
            },
            "bass": {
                "tracks": ["bass"],
                "effects": [
                    ("Fruity Parametric EQ 2", "bass_clarity"),
                    ("Fruity Compressor", "bass_control"),
                ],
                "send_reverb": 10,
                "send_delay": 0,
            },
            "guitars": {
                "tracks": ["guitar", "rhythm", "lead_guitar"],
                "effects": [("Fruity Compressor", "glue")],
                "send_reverb": 25,
                "send_delay": 15,
            },
        },
        "reverb_settings": {"type": "room", "decay": 1.5, "wet": 100},
        "delay_settings": {"time": "1/8", "feedback": 25, "wet": 100},
        "recommended_tempo": (100, 180),
    },
    "electronic": {
        "description": "Modern electronic production with clean separation",
        "master_chain": ["Fruity Parametric EQ 2", "Maximus", "Fruity Limiter"],
        "bus_groups": {
            "drums": {
                "tracks": ["kick", "snare", "hihat", "percussion"],
                "effects": [
                    ("Fruity Parametric EQ 2", "drums_punch"),
                    ("Fruity Compressor", "drums_punch"),
                ],
                "send_reverb": 15,
                "send_delay": 10,
            },
            "bass": {
                "tracks": ["bass", "sub"],
                "effects": [
                    ("Fruity Parametric EQ 2", "bass_clarity"),
                ],
                "send_reverb": 5,
                "send_delay": 0,
                "sidechain_to": "kick",
            },
            "synths": {
                "tracks": ["lead", "pad", "arp", "pluck"],
                "effects": [("Fruity Compressor", "glue")],
                "send_reverb": 30,
                "send_delay": 25,
                "sidechain_to": "kick",
            },
        },
        "reverb_settings": {"type": "plate", "decay": 2.0, "wet": 100},
        "delay_settings": {"time": "1/8", "feedback": 35, "wet": 100},
        "recommended_tempo": (110, 140),
    },
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FLStudioExportConfig:
    """Configuration for FL Studio export."""

    project_name: str = "AETHER Project"
    artist_name: str = "AETHER"
    genre: str = "electronic"
    tempo_bpm: float = 120.0
    time_signature: tuple[int, int] = (4, 4)

    # Channel settings
    default_volume: float = 0.78  # -2.5dB headroom
    default_pan: float = 0.5      # Center

    # Pattern settings
    beats_per_pattern: int = 16
    ppq: int = 96  # Pulses per quarter note

    # Export options
    include_mixer_routing: bool = True
    include_effects_chain: bool = True
    include_automation: bool = True
    export_separate_tracks: bool = True
    create_zip_package: bool = True

    # Quality settings
    bit_depth: int = 24
    sample_rate: int = 48000


@dataclass
class MixerChannel:
    """FL Studio mixer channel configuration."""

    index: int
    name: str
    color: int
    volume: float = 0.78
    pan: float = 0.5
    stereo_separation: float = 0.0
    effects: list[tuple[str, Optional[str]]] = field(default_factory=list)
    sends: dict[str, float] = field(default_factory=dict)
    sidechain_source: Optional[str] = None


@dataclass
class ChannelRackItem:
    """FL Studio channel rack item."""

    index: int
    name: str
    plugin: str
    plugin_category: str
    color: int
    mixer_track: int
    preset: Optional[str] = None
    note_count: int = 0


@dataclass
class FLProjectPackage:
    """Complete FL Studio project package."""

    name: str
    artist: str
    genre: str
    tempo: float
    time_signature: tuple[int, int]
    ppq: int
    created_at: str
    channels: list[ChannelRackItem]
    mixer: list[MixerChannel]
    patterns: list[dict]
    arrangement: list[dict]
    midi_file: str
    track_files: list[str]


class FLStudioProvider(BaseProvider):
    """
    Institutional-grade FL Studio integration provider.

    Creates comprehensive FL Studio project packages including:
    - MIDI files with full track separation
    - Genre-optimized mixer routing presets
    - Plugin recommendations with specific presets
    - Effects chain templates
    - Sidechain routing suggestions
    - Color-coded channel organization
    """

    provider_type: str = "flstudio"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._export_config = FLStudioExportConfig()
        self._midiutil_available = False

    def get_info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="FL Studio Export Provider (Institutional)",
            version="2.0.0",
            provider_type=self.provider_type,
            status=self._status,
            capabilities=[
                "midi_export",
                "project_package",
                "mixer_routing",
                "effects_chain",
                "genre_templates",
                "sidechain_routing",
                "track_separation",
            ],
            config=self.config,
        )

    async def initialize(self) -> bool:
        """Initialize the FL Studio provider."""
        try:
            from midiutil import MIDIFile as MIDIUtilFile
            self._midiutil_available = True
            self._status = ProviderStatus.AVAILABLE
            logger.info("FL Studio provider initialized with MIDI export support")
            return True
        except ImportError:
            logger.warning(
                "midiutil not installed. Install with: pip install midiutil"
            )
            self._midiutil_available = False
            self._status = ProviderStatus.DEGRADED
            return True

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._status = ProviderStatus.UNAVAILABLE

    async def health_check(self) -> bool:
        """Check provider health."""
        return self._status in [ProviderStatus.AVAILABLE, ProviderStatus.DEGRADED]

    async def export_package(
        self,
        midi_data: MIDIFile,
        output_dir: Path,
        genre: Optional[str] = None,
        project_name: Optional[str] = None,
        config: Optional[FLStudioExportConfig] = None,
    ) -> Path:
        """
        Export complete FL Studio project package.

        Creates a .flpkg directory or .zip containing:
        - project.mid - Full MIDI with all tracks
        - manifest.json - Complete project metadata
        - mixer_preset.json - Genre-optimized mixer routing
        - channels/ - Individual track MIDI files

        Args:
            midi_data: MIDI file data to export
            output_dir: Directory for output package
            genre: Genre for mixer template selection
            project_name: Project name override
            config: Export configuration

        Returns:
            Path to created package
        """
        export_config = config or FLStudioExportConfig()
        if project_name:
            export_config.project_name = project_name
        if genre:
            export_config.genre = genre

        # Update from MIDI data
        export_config.tempo_bpm = midi_data.tempo_bpm
        export_config.time_signature = midi_data.time_signature

        # Create package directory
        safe_name = export_config.project_name.replace(" ", "_").replace("/", "-")
        package_dir = Path(output_dir) / f"{safe_name}.flpkg"
        package_dir.mkdir(parents=True, exist_ok=True)

        # Generate all components
        channels = self._create_channel_rack(midi_data, export_config)
        mixer = self._create_mixer_routing(channels, export_config)
        patterns = self._create_patterns(midi_data, export_config)

        # Export MIDI files
        midi_path = await self._export_midi(midi_data, package_dir / "project.mid")
        track_paths = []
        if export_config.export_separate_tracks:
            tracks_dir = package_dir / "channels"
            tracks_dir.mkdir(exist_ok=True)
            track_paths = await self._export_separate_tracks(midi_data, tracks_dir)

        # Create manifest
        package = FLProjectPackage(
            name=export_config.project_name,
            artist=export_config.artist_name,
            genre=export_config.genre,
            tempo=export_config.tempo_bpm,
            time_signature=export_config.time_signature,
            ppq=export_config.ppq,
            created_at=datetime.now().isoformat(),
            channels=[self._channel_to_dict(ch) for ch in channels],
            mixer=[self._mixer_to_dict(m) for m in mixer],
            patterns=patterns,
            arrangement=self._create_arrangement(midi_data, patterns),
            midi_file="project.mid",
            track_files=[p.name for p in track_paths],
        )

        # Write manifest
        manifest_path = package_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self._package_to_dict(package), f, indent=2)

        # Write mixer preset
        mixer_preset_path = package_dir / "mixer_preset.json"
        mixer_template = GENRE_MIXER_TEMPLATES.get(
            export_config.genre,
            GENRE_MIXER_TEMPLATES.get("electronic")
        )
        with open(mixer_preset_path, "w") as f:
            json.dump({
                "genre": export_config.genre,
                "template": mixer_template,
                "channels": [self._mixer_to_dict(m) for m in mixer],
            }, f, indent=2)

        # Write import instructions
        instructions_path = package_dir / "FL_STUDIO_IMPORT.txt"
        self._write_import_instructions(instructions_path, export_config, package)

        # Optionally create zip
        if export_config.create_zip_package:
            zip_path = package_dir.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in package_dir.rglob("*"):
                    if file.is_file():
                        zf.write(file, file.relative_to(package_dir))
            logger.info(f"Created FL Studio package: {zip_path}")
            return zip_path

        logger.info(f"Created FL Studio package: {package_dir}")
        return package_dir

    async def export_to_flp(
        self,
        midi_data: MIDIFile,
        output_path: Path,
        project_name: Optional[str] = None,
        config: Optional[FLStudioExportConfig] = None,
    ) -> Path:
        """
        Legacy API - exports package to specified location.

        For backward compatibility with existing code.
        """
        output_dir = Path(output_path).parent
        return await self.export_package(
            midi_data=midi_data,
            output_dir=output_dir,
            project_name=project_name or Path(output_path).stem,
            config=config,
        )

    def _create_channel_rack(
        self,
        midi_data: MIDIFile,
        config: FLStudioExportConfig,
    ) -> list[ChannelRackItem]:
        """Create FL Studio channel rack items from MIDI tracks."""
        channels = []

        for idx, track in enumerate(midi_data.tracks):
            plugin, category = self._get_plugin_for_track(track)
            color = self._get_track_color(track.name)
            preset = self._get_preset_for_track(track, plugin, config.genre)

            channel = ChannelRackItem(
                index=idx,
                name=track.name,
                plugin=plugin,
                plugin_category=category,
                color=color,
                mixer_track=idx + 1,  # 0 is master
                preset=preset,
                note_count=len(track.notes),
            )
            channels.append(channel)

        return channels

    def _create_mixer_routing(
        self,
        channels: list[ChannelRackItem],
        config: FLStudioExportConfig,
    ) -> list[MixerChannel]:
        """Create FL Studio mixer routing based on genre template."""
        mixer = []
        template = GENRE_MIXER_TEMPLATES.get(
            config.genre,
            GENRE_MIXER_TEMPLATES.get("electronic")
        )

        # Create master channel
        master = MixerChannel(
            index=0,
            name="Master",
            color=FL_TRACK_COLORS["master"],
            volume=0.8,
            effects=[(e, None) for e in template.get("master_chain", [])],
        )
        mixer.append(master)

        # Create send channels (reverb, delay)
        reverb_idx = 100
        delay_idx = 101
        mixer.append(MixerChannel(
            index=reverb_idx,
            name="Reverb Send",
            color=FL_TRACK_COLORS["fx"],
            effects=[("Fruity Reverb 2", template.get("reverb_settings", {}).get("type", "hall"))],
        ))
        mixer.append(MixerChannel(
            index=delay_idx,
            name="Delay Send",
            color=FL_TRACK_COLORS["fx"],
            effects=[("Fruity Delay 3", template.get("delay_settings", {}).get("time", "1/8"))],
        ))

        # Map channels to mixer tracks with bus routing
        bus_groups = template.get("bus_groups", {})
        bus_indices = {}

        # Create bus channels
        bus_start_idx = 90
        for i, (bus_name, bus_config) in enumerate(bus_groups.items()):
            bus_idx = bus_start_idx + i
            bus_indices[bus_name] = bus_idx

            effects = []
            for effect_name, preset in bus_config.get("effects", []):
                effects.append((effect_name, preset))

            sends = {}
            if bus_config.get("send_reverb", 0) > 0:
                sends["reverb"] = bus_config["send_reverb"] / 100
            if bus_config.get("send_delay", 0) > 0:
                sends["delay"] = bus_config["send_delay"] / 100

            sidechain = None
            if "sidechain_to" in bus_config:
                sidechain = bus_config["sidechain_to"]

            mixer.append(MixerChannel(
                index=bus_idx,
                name=f"{bus_name.title()} Bus",
                color=FL_TRACK_COLORS.get(bus_name, FL_TRACK_COLORS["bus"]),
                effects=effects,
                sends=sends,
                sidechain_source=sidechain,
            ))

        # Create individual channel mixer tracks
        for channel in channels:
            track_lower = channel.name.lower()

            # Find which bus this channel belongs to
            assigned_bus = None
            for bus_name, bus_config in bus_groups.items():
                for track_type in bus_config.get("tracks", []):
                    if track_type in track_lower:
                        assigned_bus = bus_name
                        break
                if assigned_bus:
                    break

            mixer_channel = MixerChannel(
                index=channel.mixer_track,
                name=channel.name,
                color=channel.color,
                volume=config.default_volume,
                pan=config.default_pan,
            )

            if assigned_bus and assigned_bus in bus_indices:
                mixer_channel.sends[assigned_bus] = 1.0  # Route to bus

            mixer.append(mixer_channel)

        return mixer

    def _create_patterns(
        self,
        midi_data: MIDIFile,
        config: FLStudioExportConfig,
    ) -> list[dict]:
        """Create FL Studio pattern data from MIDI."""
        patterns = []

        for track_idx, track in enumerate(midi_data.tracks):
            if not track.notes:
                continue

            # Group notes into patterns
            max_time = max(n.start_time + n.duration for n in track.notes)
            num_patterns = int(max_time / config.beats_per_pattern) + 1

            for pattern_idx in range(num_patterns):
                start_beat = pattern_idx * config.beats_per_pattern
                end_beat = start_beat + config.beats_per_pattern

                pattern_notes = [
                    {
                        "pitch": note.pitch,
                        "velocity": note.velocity,
                        "position": note.start_time - start_beat,
                        "length": note.duration,
                        "channel": note.channel,
                    }
                    for note in track.notes
                    if start_beat <= note.start_time < end_beat
                ]

                if pattern_notes:
                    patterns.append({
                        "index": len(patterns),
                        "name": f"{track.name} {pattern_idx + 1}",
                        "channel_index": track_idx,
                        "color": self._get_track_color(track.name),
                        "length_beats": config.beats_per_pattern,
                        "notes": pattern_notes,
                    })

        return patterns

    def _create_arrangement(
        self,
        midi_data: MIDIFile,
        patterns: list[dict],
    ) -> list[dict]:
        """Create FL Studio arrangement/playlist data."""
        arrangement = []

        # Group patterns by channel
        channel_patterns = {}
        for pattern in patterns:
            ch_idx = pattern["channel_index"]
            if ch_idx not in channel_patterns:
                channel_patterns[ch_idx] = []
            channel_patterns[ch_idx].append(pattern)

        # Create playlist items
        for ch_idx, ch_patterns in channel_patterns.items():
            for pattern in ch_patterns:
                # Extract pattern number from name (e.g., "Drums 1" -> 0)
                name_parts = pattern["name"].rsplit(" ", 1)
                if len(name_parts) == 2 and name_parts[1].isdigit():
                    pattern_num = int(name_parts[1]) - 1
                    start_bar = pattern_num

                    arrangement.append({
                        "pattern_index": pattern["index"],
                        "track": ch_idx,
                        "position_bars": start_bar,
                        "length_bars": 1,
                    })

        return arrangement

    async def _export_midi(self, midi_data: MIDIFile, output_path: Path) -> Path:
        """Export MIDI data to file using midiutil."""
        if not self._midiutil_available:
            # Fallback: create empty file with metadata
            output_path.write_text(f"# MIDI placeholder for {len(midi_data.tracks)} tracks\n")
            return output_path

        from midiutil import MIDIFile as MIDIUtilFile

        num_tracks = len(midi_data.tracks)
        midi = MIDIUtilFile(num_tracks, deinterleave=False)

        for track_idx, track in enumerate(midi_data.tracks):
            midi.addTempo(track_idx, 0, midi_data.tempo_bpm)
            midi.addTrackName(track_idx, 0, track.name)
            midi.addProgramChange(track_idx, track.channel, 0, track.program)

            for note in track.notes:
                midi.addNote(
                    track=track_idx,
                    channel=track.channel,
                    pitch=note.pitch,
                    time=note.start_time,
                    duration=note.duration,
                    volume=note.velocity,
                )

        with open(output_path, "wb") as f:
            midi.writeFile(f)

        return output_path

    async def _export_separate_tracks(
        self,
        midi_data: MIDIFile,
        output_dir: Path,
    ) -> list[Path]:
        """Export each MIDI track as a separate file."""
        paths = []

        if not self._midiutil_available:
            return paths

        from midiutil import MIDIFile as MIDIUtilFile

        for track_idx, track in enumerate(midi_data.tracks):
            if not track.notes:
                continue

            midi = MIDIUtilFile(1)
            midi.addTempo(0, 0, midi_data.tempo_bpm)
            midi.addTrackName(0, 0, track.name)
            midi.addProgramChange(0, track.channel, 0, track.program)

            for note in track.notes:
                midi.addNote(
                    track=0,
                    channel=track.channel,
                    pitch=note.pitch,
                    time=note.start_time,
                    duration=note.duration,
                    volume=note.velocity,
                )

            safe_name = track.name.replace(" ", "_").replace("/", "-")
            output_path = output_dir / f"{track_idx:02d}_{safe_name}.mid"

            with open(output_path, "wb") as f:
                midi.writeFile(f)

            paths.append(output_path)

        return paths

    def _get_plugin_for_track(self, track: MIDITrack) -> tuple[str, str]:
        """Get recommended FL Studio plugin for a track."""
        track_lower = track.name.lower()

        # Check track name patterns
        if any(k in track_lower for k in ["drum", "perc", "kick", "snare", "hat"]):
            return ("FPC", "drums")
        if "bass" in track_lower or "sub" in track_lower:
            return ("Sytrus", "bass")
        if any(k in track_lower for k in ["piano", "keys", "organ"]):
            return ("FL Keys", "keys")
        if any(k in track_lower for k in ["synth", "lead"]):
            return ("Sytrus", "lead")
        if "pad" in track_lower:
            return ("Sytrus", "pad")
        if any(k in track_lower for k in ["string", "violin", "cello"]):
            return ("FLEX", "strings")
        if any(k in track_lower for k in ["brass", "horn", "trumpet"]):
            return ("FLEX", "brass")
        if any(k in track_lower for k in ["guitar"]):
            return ("FLEX", "guitar")

        # Fall back to GM program mapping
        for program_range, (plugin, category) in GM_TO_FL_PLUGIN.items():
            if track.program in program_range:
                return (plugin, category)

        return ("FLEX", "general")

    def _get_preset_for_track(
        self,
        track: MIDITrack,
        plugin: str,
        genre: str,
    ) -> Optional[str]:
        """Get recommended preset for a track."""
        if plugin not in FL_NATIVE_PLUGINS:
            return None

        plugin_info = FL_NATIVE_PLUGINS[plugin]
        track_lower = track.name.lower()

        # Find matching preset category
        for category, presets in plugin_info.get("presets", {}).items():
            if category in track_lower and presets:
                return presets[0]  # Return first preset

        return None

    def _get_track_color(self, track_name: str) -> int:
        """Get FL Studio color for track based on name."""
        track_lower = track_name.lower()

        for keyword, color in FL_TRACK_COLORS.items():
            if keyword in track_lower:
                return color

        return FL_TRACK_COLORS["default"]

    def _channel_to_dict(self, channel: ChannelRackItem) -> dict:
        """Convert channel to dictionary."""
        return {
            "index": channel.index,
            "name": channel.name,
            "plugin": channel.plugin,
            "plugin_category": channel.plugin_category,
            "color": f"#{channel.color:06X}",
            "mixer_track": channel.mixer_track,
            "preset": channel.preset,
            "note_count": channel.note_count,
        }

    def _mixer_to_dict(self, mixer: MixerChannel) -> dict:
        """Convert mixer channel to dictionary."""
        return {
            "index": mixer.index,
            "name": mixer.name,
            "color": f"#{mixer.color:06X}",
            "volume": mixer.volume,
            "pan": mixer.pan,
            "stereo_separation": mixer.stereo_separation,
            "effects": [{"plugin": e[0], "preset": e[1]} for e in mixer.effects],
            "sends": mixer.sends,
            "sidechain_source": mixer.sidechain_source,
        }

    def _package_to_dict(self, package: FLProjectPackage) -> dict:
        """Convert package to dictionary."""
        return {
            "format": "aether_flstudio_package",
            "version": "2.0.0",
            "project": {
                "name": package.name,
                "artist": package.artist,
                "genre": package.genre,
                "tempo": package.tempo,
                "time_signature": list(package.time_signature),
                "ppq": package.ppq,
                "created_at": package.created_at,
            },
            "channels": package.channels,
            "mixer": package.mixer,
            "patterns": package.patterns,
            "arrangement": package.arrangement,
            "files": {
                "midi": package.midi_file,
                "tracks": package.track_files,
            },
        }

    def _write_import_instructions(
        self,
        path: Path,
        config: FLStudioExportConfig,
        package: FLProjectPackage,
    ) -> None:
        """Write human-readable import instructions."""
        template = GENRE_MIXER_TEMPLATES.get(config.genre, {})

        instructions = f"""
================================================================================
AETHER FL STUDIO PROJECT PACKAGE
================================================================================

Project: {config.project_name}
Artist: {config.artist_name}
Genre: {config.genre}
Tempo: {config.tempo_bpm} BPM
Time Signature: {config.time_signature[0]}/{config.time_signature[1]}

================================================================================
QUICK START - IMPORT IN 3 STEPS
================================================================================

1. IMPORT MIDI
   - Open FL Studio
   - File > Import > MIDI file
   - Select "project.mid" from this package
   - Check "Start new project" and click OK

2. ASSIGN INSTRUMENTS
   For each channel in the Channel Rack, right-click and replace with:

"""
        for ch in package.channels:
            instructions += f"   {ch['name']:20} -> {ch['plugin']}"
            if ch.get('preset'):
                instructions += f" (Preset: {ch['preset']})"
            instructions += "\n"

        instructions += f"""
3. SET UP MIXER
   Route each channel to its mixer track, then add effects:

"""
        for mx in package.mixer:
            if mx.get('effects'):
                instructions += f"   Track {mx['index']:3} ({mx['name']:20}):\n"
                for effect in mx['effects']:
                    instructions += f"      + {effect['plugin']}"
                    if effect.get('preset'):
                        instructions += f" [{effect['preset']}]"
                    instructions += "\n"

        instructions += f"""
================================================================================
GENRE TEMPLATE: {config.genre.upper()}
================================================================================

{template.get('description', 'Professional mixing template')}

Recommended Tempo Range: {template.get('recommended_tempo', (100, 130))}

MASTER CHAIN:
{chr(10).join(f'  {i+1}. {e}' for i, e in enumerate(template.get('master_chain', [])))}

REVERB SEND SETTINGS:
  Type: {template.get('reverb_settings', {}).get('type', 'hall')}
  Decay: {template.get('reverb_settings', {}).get('decay', 2.0)}s

DELAY SEND SETTINGS:
  Time: {template.get('delay_settings', {}).get('time', '1/8')}
  Feedback: {template.get('delay_settings', {}).get('feedback', 30)}%

================================================================================
SIDECHAIN SETUP (if applicable)
================================================================================

For genres like {config.genre}, sidechain compression creates pumping effects:

1. On bass/synth mixer tracks, add Fruity Limiter
2. Enable sidechain input from Kick track
3. Set ratio to 4:1, attack 0.5ms, release 100-200ms
4. Adjust threshold to taste

================================================================================
FILES IN THIS PACKAGE
================================================================================

project.mid          - Complete MIDI file with all tracks
manifest.json        - Machine-readable project metadata
mixer_preset.json    - Mixer routing configuration
FL_STUDIO_IMPORT.txt - This file
channels/            - Individual track MIDI files (for selective import)

================================================================================
Generated by AETHER Band Engine
https://github.com/aether-band-engine
================================================================================
"""
        path.write_text(instructions)

    async def create_project_template(
        self,
        genre: str,
        output_path: Path,
        config: Optional[FLStudioExportConfig] = None,
    ) -> Path:
        """
        Create an FL Studio project template for a specific genre.

        Returns path to template JSON file.
        """
        export_config = config or FLStudioExportConfig()
        export_config.genre = genre

        template = GENRE_MIXER_TEMPLATES.get(genre, GENRE_MIXER_TEMPLATES["electronic"])

        template_data = {
            "format": "aether_flstudio_template",
            "version": "2.0.0",
            "genre": genre,
            "description": template.get("description", ""),
            "recommended_tempo": template.get("recommended_tempo", (100, 130)),
            "master_chain": template.get("master_chain", []),
            "bus_groups": template.get("bus_groups", {}),
            "reverb_settings": template.get("reverb_settings", {}),
            "delay_settings": template.get("delay_settings", {}),
            "plugins": FL_NATIVE_PLUGINS,
            "effects": FL_EFFECTS,
        }

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".fltemplate.json")

        with open(output_path, "w") as f:
            json.dump(template_data, f, indent=2)

        logger.info(f"Created FL Studio template for {genre}: {output_path}")
        return output_path


# ============================================================================
# Helper Functions
# ============================================================================

def get_flstudio_provider(config: dict[str, Any] | None = None) -> FLStudioProvider:
    """Get an FL Studio provider instance."""
    return FLStudioProvider(config)


async def quick_export_to_flp(
    midi_data: MIDIFile,
    output_path: Path,
    project_name: str = "AETHER Export",
    genre: str = "electronic",
) -> Path:
    """Quick helper to export MIDI to FL Studio package."""
    provider = FLStudioProvider()
    await provider.initialize()
    return await provider.export_package(
        midi_data=midi_data,
        output_dir=Path(output_path).parent,
        project_name=project_name,
        genre=genre,
    )


def get_available_genres() -> list[str]:
    """Get list of genres with mixer templates."""
    return list(GENRE_MIXER_TEMPLATES.keys())


def get_genre_template(genre: str) -> Optional[dict]:
    """Get mixer template for a genre."""
    return GENRE_MIXER_TEMPLATES.get(genre)
