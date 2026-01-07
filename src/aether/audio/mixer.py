"""
AETHER Professional Mixing Console

Industry-standard mixing system with:
- Per-track gain staging in dB
- Bus routing (drums, bass, synths, vocals)
- Pan controls
- Solo/mute functionality
- Metering (peak, RMS, LUFS)
- Genre-specific mix presets

Reference Levels:
- Track peaks: -18 to -12 dBFS (K-20 metering)
- Mix bus: -6 to -3 dBFS pre-master
- Headroom for mastering: 6 dB minimum
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type aliases
AudioBuffer = NDArray[np.float64]
StereoBuffer = NDArray[np.float64]


def db_to_linear(db: float) -> float:
    """Convert decibels to linear gain."""
    return 10 ** (db / 20)


def linear_to_db(linear: float) -> float:
    """Convert linear gain to decibels."""
    if linear <= 0:
        return -120.0  # Floor
    return 20 * math.log10(linear)


class TrackType(str, Enum):
    """Track categories for routing and processing."""
    KICK = "kick"
    SNARE = "snare"
    HIHAT = "hihat"
    PERCUSSION = "percussion"
    BASS = "bass"
    PAD = "pad"
    LEAD = "lead"
    VOCAL = "vocal"
    FX = "fx"
    MASTER = "master"


class BusType(str, Enum):
    """Mix bus categories."""
    DRUMS = "drums"
    BASS = "bass"
    SYNTHS = "synths"
    VOCALS = "vocals"
    FX = "fx"
    MASTER = "master"


@dataclass
class TrackMeter:
    """Real-time metering for a track."""
    peak_db: float = -120.0
    rms_db: float = -120.0
    peak_hold_db: float = -120.0
    clip_count: int = 0

    def update(self, audio: AudioBuffer) -> None:
        """Update meters with new audio."""
        if len(audio) == 0:
            return

        # Peak
        peak = np.max(np.abs(audio))
        self.peak_db = linear_to_db(peak)

        # Peak hold (decay slowly)
        if self.peak_db > self.peak_hold_db:
            self.peak_hold_db = self.peak_db
        else:
            self.peak_hold_db -= 0.5  # Decay 0.5 dB per update

        # RMS
        rms = np.sqrt(np.mean(audio ** 2))
        self.rms_db = linear_to_db(rms)

        # Clip detection
        if peak > 1.0:
            self.clip_count += 1


@dataclass
class Track:
    """
    Individual mix track with full channel strip.

    Gain staging:
    - input_gain: Adjust incoming signal level
    - fader: Main volume control (0 dB = unity)
    - pan: Stereo position (-1 left, 0 center, +1 right)
    """
    name: str
    track_type: TrackType
    bus: BusType = BusType.MASTER

    # Gain staging (all in dB)
    input_gain_db: float = 0.0      # Pre-fader gain
    fader_db: float = 0.0           # Main fader (0 = unity)

    # Pan (-1 to +1)
    pan: float = 0.0

    # Solo/Mute
    solo: bool = False
    mute: bool = False

    # Audio buffer
    audio: StereoBuffer | None = None

    # Metering
    meter: TrackMeter = field(default_factory=TrackMeter)

    # Processing chain (optional)
    pre_fader_fx: list[Callable] = field(default_factory=list)
    post_fader_fx: list[Callable] = field(default_factory=list)

    def set_audio(self, audio: AudioBuffer | StereoBuffer) -> None:
        """Set track audio, converting mono to stereo if needed."""
        if audio.ndim == 1:
            self.audio = np.array([audio, audio])
        else:
            self.audio = audio.copy()

    def get_output(self) -> StereoBuffer | None:
        """Get processed track output."""
        if self.audio is None or self.mute:
            return None

        output = self.audio.copy()

        # Apply input gain
        if self.input_gain_db != 0:
            output *= db_to_linear(self.input_gain_db)

        # Pre-fader processing
        for fx in self.pre_fader_fx:
            output = fx(output)

        # Update meter (pre-fader)
        self.meter.update(output[0])

        # Apply fader
        if self.fader_db != 0:
            output *= db_to_linear(self.fader_db)

        # Apply pan (constant power panning)
        if self.pan != 0:
            # Convert pan (-1 to 1) to angle (0 to pi/2)
            angle = (self.pan + 1) * math.pi / 4
            left_gain = math.cos(angle)
            right_gain = math.sin(angle)
            output[0] *= left_gain
            output[1] *= right_gain

        # Post-fader processing
        for fx in self.post_fader_fx:
            output = fx(output)

        return output

    @property
    def output_level_db(self) -> float:
        """Get current output level in dB."""
        return self.input_gain_db + self.fader_db


@dataclass
class Bus:
    """Mix bus for grouping tracks."""
    name: str
    bus_type: BusType

    # Bus processing
    fader_db: float = 0.0
    mute: bool = False

    # Tracks routed to this bus
    track_names: list[str] = field(default_factory=list)

    # Metering
    meter: TrackMeter = field(default_factory=TrackMeter)


# =============================================================================
# GENRE-SPECIFIC MIX PRESETS
# =============================================================================

@dataclass
class MixPreset:
    """Predefined mix settings for a genre."""
    name: str
    genre: str
    description: str

    # Track levels in dB (relative to 0 dB reference)
    track_levels: dict[TrackType, float] = field(default_factory=dict)

    # Bus levels in dB
    bus_levels: dict[BusType, float] = field(default_factory=dict)

    # Pan positions
    pan_positions: dict[TrackType, float] = field(default_factory=dict)


# Industry-standard mix presets
MIX_PRESETS = {
    "house": MixPreset(
        name="House",
        genre="house",
        description="Four-on-the-floor with prominent kick and driving bass",
        track_levels={
            TrackType.KICK: 0.0,       # Reference level
            TrackType.SNARE: -4.0,     # Clap sits back
            TrackType.HIHAT: -8.0,     # Hats subtle
            TrackType.BASS: -3.0,      # Strong bass
            TrackType.PAD: -10.0,      # Pads way back
            TrackType.LEAD: -6.0,      # Lead present but not dominant
        },
        bus_levels={
            BusType.DRUMS: 0.0,
            BusType.BASS: -1.0,
            BusType.SYNTHS: -3.0,
        },
        pan_positions={
            TrackType.KICK: 0.0,
            TrackType.SNARE: 0.0,
            TrackType.HIHAT: 0.15,
            TrackType.BASS: 0.0,
            TrackType.PAD: 0.0,
            TrackType.LEAD: 0.0,
        },
    ),

    "techno": MixPreset(
        name="Techno",
        genre="techno",
        description="Hard-hitting drums with hypnotic elements",
        track_levels={
            TrackType.KICK: 0.0,
            TrackType.SNARE: -3.0,
            TrackType.HIHAT: -6.0,
            TrackType.BASS: -2.0,
            TrackType.PAD: -12.0,
            TrackType.LEAD: -8.0,
        },
        bus_levels={
            BusType.DRUMS: 0.0,
            BusType.BASS: 0.0,
            BusType.SYNTHS: -4.0,
        },
        pan_positions={
            TrackType.KICK: 0.0,
            TrackType.SNARE: 0.0,
            TrackType.HIHAT: 0.1,
            TrackType.BASS: 0.0,
            TrackType.PAD: 0.0,
            TrackType.LEAD: 0.0,
        },
    ),

    "trap": MixPreset(
        name="Trap",
        genre="trap",
        description="Heavy 808s with crisp hats and hard-hitting snares",
        track_levels={
            TrackType.KICK: 0.0,       # 808 is king
            TrackType.SNARE: -2.0,     # Snare hits hard
            TrackType.HIHAT: -5.0,     # Hats present for rolls
            TrackType.BASS: -6.0,      # Bass under 808
            TrackType.PAD: -14.0,      # Minimal pads
            TrackType.LEAD: -4.0,      # Lead more prominent
        },
        bus_levels={
            BusType.DRUMS: 0.0,
            BusType.BASS: -2.0,
            BusType.SYNTHS: -3.0,
        },
        pan_positions={
            TrackType.KICK: 0.0,
            TrackType.SNARE: 0.0,
            TrackType.HIHAT: 0.0,      # Centered for impact
            TrackType.BASS: 0.0,
            TrackType.PAD: 0.0,
            TrackType.LEAD: 0.0,
        },
    ),

    "hip_hop": MixPreset(
        name="Hip-Hop",
        genre="hip_hop",
        description="Vocal-forward with punchy drums and warm bass",
        track_levels={
            TrackType.KICK: -2.0,
            TrackType.SNARE: -1.0,     # Snare prominent
            TrackType.HIHAT: -8.0,
            TrackType.BASS: -3.0,
            TrackType.PAD: -12.0,
            TrackType.LEAD: -6.0,
            TrackType.VOCAL: 0.0,      # Vocals on top
        },
        bus_levels={
            BusType.DRUMS: -1.0,
            BusType.BASS: -2.0,
            BusType.SYNTHS: -4.0,
            BusType.VOCALS: 0.0,
        },
        pan_positions={
            TrackType.KICK: 0.0,
            TrackType.SNARE: 0.0,
            TrackType.HIHAT: 0.2,
            TrackType.BASS: 0.0,
            TrackType.PAD: 0.0,
            TrackType.LEAD: -0.1,
            TrackType.VOCAL: 0.0,
        },
    ),

    "synthwave": MixPreset(
        name="Synthwave",
        genre="synthwave",
        description="Lush synths with gated drums and prominent leads",
        track_levels={
            TrackType.KICK: -2.0,
            TrackType.SNARE: -3.0,
            TrackType.HIHAT: -9.0,
            TrackType.BASS: -4.0,
            TrackType.PAD: -6.0,       # Pads more prominent
            TrackType.LEAD: -3.0,      # Lead shines
        },
        bus_levels={
            BusType.DRUMS: -2.0,
            BusType.BASS: -2.0,
            BusType.SYNTHS: 0.0,       # Synths forward
        },
        pan_positions={
            TrackType.KICK: 0.0,
            TrackType.SNARE: 0.0,
            TrackType.HIHAT: 0.25,
            TrackType.BASS: 0.0,
            TrackType.PAD: 0.0,
            TrackType.LEAD: 0.0,
        },
    ),

    "ambient": MixPreset(
        name="Ambient",
        genre="ambient",
        description="Atmospheric with wide stereo field",
        track_levels={
            TrackType.KICK: -8.0,
            TrackType.SNARE: -10.0,
            TrackType.HIHAT: -12.0,
            TrackType.BASS: -6.0,
            TrackType.PAD: 0.0,        # Pads are the focus
            TrackType.LEAD: -4.0,
        },
        bus_levels={
            BusType.DRUMS: -6.0,
            BusType.BASS: -4.0,
            BusType.SYNTHS: 0.0,
        },
        pan_positions={
            TrackType.KICK: 0.0,
            TrackType.SNARE: 0.0,
            TrackType.HIHAT: 0.4,
            TrackType.BASS: 0.0,
            TrackType.PAD: 0.0,
            TrackType.LEAD: 0.1,
        },
    ),

    "dnb": MixPreset(
        name="Drum & Bass",
        genre="dnb",
        description="Punchy breaks with heavy bass",
        track_levels={
            TrackType.KICK: 0.0,
            TrackType.SNARE: -1.0,
            TrackType.HIHAT: -5.0,
            TrackType.BASS: -2.0,
            TrackType.PAD: -10.0,
            TrackType.LEAD: -6.0,
        },
        bus_levels={
            BusType.DRUMS: 0.0,
            BusType.BASS: -1.0,
            BusType.SYNTHS: -4.0,
        },
        pan_positions={
            TrackType.KICK: 0.0,
            TrackType.SNARE: 0.0,
            TrackType.HIHAT: 0.2,
            TrackType.BASS: 0.0,
            TrackType.PAD: 0.0,
            TrackType.LEAD: 0.0,
        },
    ),
}


def get_mix_preset(genre: str) -> MixPreset | None:
    """Get mix preset for a genre."""
    return MIX_PRESETS.get(genre.lower().replace("-", "_").replace(" ", "_"))


# =============================================================================
# MIXING CONSOLE
# =============================================================================

class MixingConsole:
    """
    Professional mixing console for AETHER.

    Features:
    - Multi-track mixing with gain staging
    - Bus routing
    - Genre-specific presets
    - Metering and headroom management

    Usage:
        console = MixingConsole(sample_rate=44100)
        console.add_track("kick", TrackType.KICK, kick_audio)
        console.add_track("bass", TrackType.BASS, bass_audio)
        console.apply_preset("house")
        stereo_mix = console.render()
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.tracks: dict[str, Track] = {}
        self.buses: dict[BusType, Bus] = {}
        self.master_fader_db: float = 0.0
        self.master_meter = TrackMeter()

        # Initialize default buses
        self._init_buses()

    def _init_buses(self) -> None:
        """Initialize mix buses."""
        for bus_type in BusType:
            self.buses[bus_type] = Bus(
                name=bus_type.value,
                bus_type=bus_type,
            )

    def add_track(
        self,
        name: str,
        track_type: TrackType,
        audio: AudioBuffer | StereoBuffer,
        bus: BusType | None = None,
        fader_db: float = 0.0,
        pan: float = 0.0,
    ) -> Track:
        """
        Add a track to the console.

        Args:
            name: Unique track name
            track_type: Type of track for preset routing
            audio: Mono or stereo audio buffer
            bus: Target bus (auto-assigned if None)
            fader_db: Initial fader level
            pan: Pan position (-1 to +1)

        Returns:
            Created Track object
        """
        # Auto-assign bus based on track type
        if bus is None:
            bus = self._get_default_bus(track_type)

        track = Track(
            name=name,
            track_type=track_type,
            bus=bus,
            fader_db=fader_db,
            pan=pan,
        )
        track.set_audio(audio)

        self.tracks[name] = track
        self.buses[bus].track_names.append(name)

        logger.debug(f"Added track '{name}' ({track_type.value}) to bus {bus.value}")
        return track

    def _get_default_bus(self, track_type: TrackType) -> BusType:
        """Get default bus for a track type."""
        drum_types = {TrackType.KICK, TrackType.SNARE, TrackType.HIHAT, TrackType.PERCUSSION}
        synth_types = {TrackType.PAD, TrackType.LEAD}

        if track_type in drum_types:
            return BusType.DRUMS
        elif track_type == TrackType.BASS:
            return BusType.BASS
        elif track_type in synth_types:
            return BusType.SYNTHS
        elif track_type == TrackType.VOCAL:
            return BusType.VOCALS
        elif track_type == TrackType.FX:
            return BusType.FX
        else:
            return BusType.MASTER

    def set_track_level(self, name: str, fader_db: float) -> None:
        """Set track fader level."""
        if name in self.tracks:
            self.tracks[name].fader_db = fader_db

    def set_track_pan(self, name: str, pan: float) -> None:
        """Set track pan position."""
        if name in self.tracks:
            self.tracks[name].pan = np.clip(pan, -1.0, 1.0)

    def set_bus_level(self, bus_type: BusType, fader_db: float) -> None:
        """Set bus fader level."""
        if bus_type in self.buses:
            self.buses[bus_type].fader_db = fader_db

    def apply_preset(self, genre: str) -> bool:
        """
        Apply a genre mix preset.

        Args:
            genre: Genre name (e.g., "house", "trap", "techno")

        Returns:
            True if preset was applied
        """
        preset = get_mix_preset(genre)
        if preset is None:
            logger.warning(f"No preset found for genre: {genre}")
            return False

        logger.info(f"Applying mix preset: {preset.name}")

        # Apply track levels and pans
        for track in self.tracks.values():
            if track.track_type in preset.track_levels:
                track.fader_db = preset.track_levels[track.track_type]
                logger.debug(f"  {track.name}: {track.fader_db:+.1f} dB")

            if track.track_type in preset.pan_positions:
                track.pan = preset.pan_positions[track.track_type]

        # Apply bus levels
        for bus_type, level in preset.bus_levels.items():
            if bus_type in self.buses:
                self.buses[bus_type].fader_db = level

        return True

    def get_levels_summary(self) -> str:
        """Get a formatted summary of current mix levels."""
        lines = ["MIX LEVELS", "=" * 50]

        for track in self.tracks.values():
            status = "M" if track.mute else ("S" if track.solo else " ")
            pan_str = f"L{abs(track.pan)*100:.0f}" if track.pan < -0.05 else (
                f"R{track.pan*100:.0f}" if track.pan > 0.05 else "C"
            )
            lines.append(
                f"  [{status}] {track.name:12s} {track.fader_db:+6.1f} dB  {pan_str:>4s}  "
                f"({track.track_type.value})"
            )

        lines.append("-" * 50)

        for bus_type, bus in self.buses.items():
            if bus.track_names:
                lines.append(f"  BUS {bus_type.value:8s} {bus.fader_db:+6.1f} dB")

        lines.append("-" * 50)
        lines.append(f"  MASTER        {self.master_fader_db:+6.1f} dB")

        return "\n".join(lines)

    def render(self, normalize: bool = True, target_peak_db: float = -1.0) -> StereoBuffer:
        """
        Render the final stereo mix.

        Args:
            normalize: Whether to normalize the output
            target_peak_db: Target peak level if normalizing

        Returns:
            Stereo audio buffer
        """
        if not self.tracks:
            raise ValueError("No tracks to render")

        # Get max length
        max_length = max(
            t.audio.shape[1] for t in self.tracks.values()
            if t.audio is not None
        )

        # Check for solo
        any_solo = any(t.solo for t in self.tracks.values())

        # Initialize bus buffers
        bus_outputs: dict[BusType, StereoBuffer] = {
            bus_type: np.zeros((2, max_length))
            for bus_type in self.buses
        }

        # Process tracks to buses
        for track in self.tracks.values():
            # Skip muted tracks, or non-solo tracks if any are soloed
            if track.mute:
                continue
            if any_solo and not track.solo:
                continue

            output = track.get_output()
            if output is None:
                continue

            # Pad if needed
            if output.shape[1] < max_length:
                padded = np.zeros((2, max_length))
                padded[:, :output.shape[1]] = output
                output = padded

            # Route to bus
            bus_outputs[track.bus] += output

        # Apply bus faders and sum to master
        master = np.zeros((2, max_length))

        for bus_type, bus in self.buses.items():
            if bus.mute or bus_type == BusType.MASTER:
                continue

            bus_audio = bus_outputs[bus_type]

            # Update bus meter
            bus.meter.update(bus_audio[0])

            # Apply bus fader
            if bus.fader_db != 0:
                bus_audio = bus_audio * db_to_linear(bus.fader_db)

            master += bus_audio

        # Add any direct-to-master tracks
        master += bus_outputs[BusType.MASTER]

        # Apply master fader
        if self.master_fader_db != 0:
            master *= db_to_linear(self.master_fader_db)

        # Update master meter
        self.master_meter.update(master[0])

        # Normalize if requested
        if normalize:
            peak = np.max(np.abs(master))
            if peak > 0:
                target_linear = db_to_linear(target_peak_db)
                master = master * (target_linear / peak)

        return master

    def render_stems(self) -> dict[str, StereoBuffer]:
        """
        Render individual track stems for export.

        Returns:
            Dictionary mapping track names to stereo buffers
        """
        stems = {}

        for name, track in self.tracks.items():
            output = track.get_output()
            if output is not None:
                stems[name] = output

        return stems


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_mix(
    tracks: dict[str, tuple[TrackType, AudioBuffer | StereoBuffer]],
    genre: str,
    sample_rate: int = 44100,
) -> StereoBuffer:
    """
    Quick mix multiple tracks with genre preset.

    Args:
        tracks: Dict mapping track names to (TrackType, audio) tuples
        genre: Genre for preset selection
        sample_rate: Sample rate

    Returns:
        Mixed stereo audio

    Example:
        mix = quick_mix({
            "kick": (TrackType.KICK, kick_audio),
            "bass": (TrackType.BASS, bass_audio),
            "lead": (TrackType.LEAD, lead_audio),
        }, genre="house")
    """
    console = MixingConsole(sample_rate)

    for name, (track_type, audio) in tracks.items():
        console.add_track(name, track_type, audio)

    console.apply_preset(genre)

    return console.render()
