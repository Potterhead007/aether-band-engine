"""
AETHER Professional Mixing Engine

Production-grade audio mixing with signal routing, bus architecture,
insert processing, send effects, and automation playback.

Architecture:
    Tracks → Bus Inserts → Bus Summing → Master Inserts → Output

Features:
- Gain staging with headroom management
- Per-track EQ and dynamics
- Hierarchical bus routing
- Send/return effects architecture
- Parameter automation with multiple curve types
- Phase correlation monitoring
- Mono compatibility checking
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from aether.audio.dsp import (
    AudioBuffer,
    StereoBuffer,
    BiquadFilter,
    FilterType,
    ParametricEQ,
    Compressor,
    StereoProcessor,
    LoudnessMeter,
    db_to_linear,
    linear_to_db,
)

logger = logging.getLogger(__name__)


class AutomationCurve(str, Enum):
    """Automation interpolation curves."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    S_CURVE = "s-curve"
    STEP = "step"


@dataclass
class AutomationPoint:
    """A single automation point."""

    time_seconds: float
    value: float
    curve: AutomationCurve = AutomationCurve.LINEAR


@dataclass
class AutomationLane:
    """Automation data for a single parameter."""

    target: str  # "track:kick:gain" or "bus:drums:pan"
    parameter: str
    points: List[AutomationPoint] = field(default_factory=list)

    def get_value_at(self, time_seconds: float) -> Optional[float]:
        """Get interpolated value at a given time."""
        if not self.points:
            return None

        # Sort points by time
        sorted_points = sorted(self.points, key=lambda p: p.time_seconds)

        # Before first point
        if time_seconds <= sorted_points[0].time_seconds:
            return sorted_points[0].value

        # After last point
        if time_seconds >= sorted_points[-1].time_seconds:
            return sorted_points[-1].value

        # Find surrounding points
        for i in range(len(sorted_points) - 1):
            p1 = sorted_points[i]
            p2 = sorted_points[i + 1]

            if p1.time_seconds <= time_seconds <= p2.time_seconds:
                # Calculate interpolation factor
                duration = p2.time_seconds - p1.time_seconds
                if duration < 1e-6:
                    return p1.value

                t = (time_seconds - p1.time_seconds) / duration

                # Apply curve
                if p1.curve == AutomationCurve.LINEAR:
                    factor = t
                elif p1.curve == AutomationCurve.EXPONENTIAL:
                    factor = t**2
                elif p1.curve == AutomationCurve.LOGARITHMIC:
                    factor = math.sqrt(t)
                elif p1.curve == AutomationCurve.S_CURVE:
                    # Smoothstep
                    factor = t * t * (3 - 2 * t)
                elif p1.curve == AutomationCurve.STEP:
                    factor = 0.0 if t < 1.0 else 1.0
                else:
                    factor = t

                return p1.value + (p2.value - p1.value) * factor

        return sorted_points[-1].value


@dataclass
class TrackState:
    """Runtime state for a mix track."""

    name: str
    audio: Optional[StereoBuffer] = None
    gain_db: float = 0.0
    pan: float = 0.0  # -1 to +1
    mute: bool = False
    solo: bool = False
    output_bus: str = "master"

    # Insert chain
    eq: Optional[ParametricEQ] = None
    compressor: Optional[Compressor] = None

    # Send levels
    sends: Dict[str, float] = field(default_factory=dict)  # bus_name -> level_db

    def get_panned_gain(self) -> Tuple[float, float]:
        """Calculate L/R gains from gain_db and pan (constant power)."""
        linear_gain = db_to_linear(self.gain_db)

        # Constant power panning
        angle = (self.pan + 1) * math.pi / 4  # 0 to pi/2
        gain_l = linear_gain * math.cos(angle)
        gain_r = linear_gain * math.sin(angle)

        return gain_l, gain_r


@dataclass
class BusState:
    """Runtime state for a mix bus."""

    name: str
    gain_db: float = 0.0
    output_bus: str = "master"

    # Insert chain
    eq: Optional[ParametricEQ] = None
    compressor: Optional[Compressor] = None

    # Accumulated audio
    buffer: Optional[StereoBuffer] = None


@dataclass
class SendEffect:
    """A send effect (reverb, delay, etc.)."""

    name: str
    process_fn: Callable[[StereoBuffer], StereoBuffer]
    return_gain_db: float = 0.0


class MixingEngine:
    """
    Professional mixing engine with full signal routing.

    Processes audio through the following chain:
    1. Track processing (gain, pan, EQ, compression)
    2. Track sends to effects
    3. Bus summing
    4. Bus processing
    5. Effect returns
    6. Master bus processing
    7. Output metering

    Usage:
        engine = MixingEngine(sample_rate=48000)
        engine.add_track("kick", audio_data, gain_db=-3, output_bus="drums")
        engine.add_bus("drums", gain_db=0, output_bus="master")
        engine.configure_track_eq("kick", bands=[...])
        output = engine.render()
    """

    def __init__(
        self,
        sample_rate: float = 48000,
        block_size: int = 512,
        target_headroom_db: float = -6.0,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.target_headroom_db = target_headroom_db

        # Tracks and buses
        self.tracks: Dict[str, TrackState] = {}
        self.buses: Dict[str, BusState] = {}

        # Always have a master bus
        self.buses["master"] = BusState(name="master", output_bus="")

        # Send effects
        self.send_effects: Dict[str, SendEffect] = {}

        # Automation
        self.automation_lanes: List[AutomationLane] = []

        # Master section
        self.master_eq: Optional[ParametricEQ] = None
        self.master_compressor: Optional[Compressor] = None

        # Metering
        self.loudness_meter = LoudnessMeter(sample_rate)

        # Output info
        self.output_peak_db: float = -100.0
        self.phase_correlation: float = 1.0
        self.mono_compatible: bool = True

    # =========================================================================
    # Track Management
    # =========================================================================

    def add_track(
        self,
        name: str,
        audio: StereoBuffer,
        gain_db: float = 0.0,
        pan: float = 0.0,
        output_bus: str = "master",
    ) -> None:
        """Add a track to the mix."""
        # Ensure stereo format (2, samples)
        if audio.ndim == 1:
            audio = np.array([audio, audio])
        elif audio.shape[0] != 2:
            audio = audio.T if audio.shape[1] == 2 else np.array([audio[0], audio[0]])

        self.tracks[name] = TrackState(
            name=name,
            audio=audio,
            gain_db=gain_db,
            pan=pan,
            output_bus=output_bus,
        )
        logger.debug(f"Added track: {name} ({audio.shape[1]} samples)")

    def set_track_gain(self, name: str, gain_db: float) -> None:
        """Set track gain."""
        if name in self.tracks:
            self.tracks[name].gain_db = gain_db

    def set_track_pan(self, name: str, pan: float) -> None:
        """Set track pan (-1 to +1)."""
        if name in self.tracks:
            self.tracks[name].pan = max(-1.0, min(1.0, pan))

    def mute_track(self, name: str, muted: bool = True) -> None:
        """Mute/unmute a track."""
        if name in self.tracks:
            self.tracks[name].mute = muted

    def solo_track(self, name: str, soloed: bool = True) -> None:
        """Solo/unsolo a track."""
        if name in self.tracks:
            self.tracks[name].solo = soloed

    def configure_track_eq(
        self,
        name: str,
        bands: List[Dict[str, Any]],
    ) -> None:
        """Configure track EQ from band definitions."""
        if name not in self.tracks:
            return

        eq = ParametricEQ(self.sample_rate)
        for band in bands:
            filter_type = band.get("band_type", "peak")
            freq = band.get("frequency_hz", 1000)
            gain = band.get("gain_db", 0.0)
            q = band.get("q", 1.0)

            if band.get("enabled", True):
                eq.add_band(filter_type, freq, gain, q)

        self.tracks[name].eq = eq

    def configure_track_compressor(
        self,
        name: str,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        knee_db: float = 3.0,
        makeup_gain_db: float = 0.0,
    ) -> None:
        """Configure track compressor."""
        if name not in self.tracks:
            return

        self.tracks[name].compressor = Compressor(
            sample_rate=self.sample_rate,
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
            knee_db=knee_db,
            makeup_gain_db=makeup_gain_db,
        )

    def set_track_send(self, track_name: str, effect_name: str, level_db: float) -> None:
        """Set send level from track to effect."""
        if track_name in self.tracks:
            self.tracks[track_name].sends[effect_name] = level_db

    # =========================================================================
    # Bus Management
    # =========================================================================

    def add_bus(
        self,
        name: str,
        gain_db: float = 0.0,
        output_bus: str = "master",
    ) -> None:
        """Add a bus to the mix."""
        self.buses[name] = BusState(
            name=name,
            gain_db=gain_db,
            output_bus=output_bus,
        )

    def configure_bus_eq(
        self,
        name: str,
        bands: List[Dict[str, Any]],
    ) -> None:
        """Configure bus EQ."""
        if name not in self.buses:
            return

        eq = ParametricEQ(self.sample_rate)
        for band in bands:
            filter_type = band.get("band_type", "peak")
            freq = band.get("frequency_hz", 1000)
            gain = band.get("gain_db", 0.0)
            q = band.get("q", 1.0)

            if band.get("enabled", True):
                eq.add_band(filter_type, freq, gain, q)

        self.buses[name].eq = eq

    def configure_bus_compressor(
        self,
        name: str,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        knee_db: float = 3.0,
        makeup_gain_db: float = 0.0,
    ) -> None:
        """Configure bus compressor."""
        if name not in self.buses:
            return

        self.buses[name].compressor = Compressor(
            sample_rate=self.sample_rate,
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
            knee_db=knee_db,
            makeup_gain_db=makeup_gain_db,
        )

    # =========================================================================
    # Master Section
    # =========================================================================

    def configure_master_eq(self, bands: List[Dict[str, Any]]) -> None:
        """Configure master bus EQ."""
        eq = ParametricEQ(self.sample_rate)
        for band in bands:
            filter_type = band.get("band_type", "peak")
            freq = band.get("frequency_hz", 1000)
            gain = band.get("gain_db", 0.0)
            q = band.get("q", 1.0)

            if band.get("enabled", True):
                eq.add_band(filter_type, freq, gain, q)

        self.master_eq = eq

    def configure_master_compressor(
        self,
        threshold_db: float = -12.0,
        ratio: float = 2.0,
        attack_ms: float = 30.0,
        release_ms: float = 200.0,
        knee_db: float = 6.0,
        makeup_gain_db: float = 0.0,
    ) -> None:
        """Configure master bus compressor."""
        self.master_compressor = Compressor(
            sample_rate=self.sample_rate,
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
            knee_db=knee_db,
            makeup_gain_db=makeup_gain_db,
        )

    # =========================================================================
    # Effects
    # =========================================================================

    def add_send_effect(
        self,
        name: str,
        process_fn: Callable[[StereoBuffer], StereoBuffer],
        return_gain_db: float = 0.0,
    ) -> None:
        """Add a send effect."""
        self.send_effects[name] = SendEffect(
            name=name,
            process_fn=process_fn,
            return_gain_db=return_gain_db,
        )

    # =========================================================================
    # Automation
    # =========================================================================

    def add_automation(
        self,
        target: str,
        parameter: str,
        points: List[Tuple[float, float]],
        curve: AutomationCurve = AutomationCurve.LINEAR,
    ) -> None:
        """
        Add automation for a parameter.

        Args:
            target: Target identifier (e.g., "track:kick" or "bus:drums")
            parameter: Parameter name (e.g., "gain", "pan")
            points: List of (time_seconds, value) tuples
            curve: Interpolation curve type
        """
        lane = AutomationLane(
            target=target,
            parameter=parameter,
            points=[AutomationPoint(time_seconds=t, value=v, curve=curve) for t, v in points],
        )
        self.automation_lanes.append(lane)

    def _apply_automation(self, time_seconds: float) -> None:
        """Apply automation values at a given time."""
        for lane in self.automation_lanes:
            value = lane.get_value_at(time_seconds)
            if value is None:
                continue

            # Parse target
            parts = lane.target.split(":")
            if len(parts) < 2:
                continue

            target_type, target_name = parts[0], parts[1]

            if target_type == "track" and target_name in self.tracks:
                track = self.tracks[target_name]
                if lane.parameter == "gain":
                    track.gain_db = value
                elif lane.parameter == "pan":
                    track.pan = value

            elif target_type == "bus" and target_name in self.buses:
                bus = self.buses[target_name]
                if lane.parameter == "gain":
                    bus.gain_db = value

            elif target_type == "master":
                if lane.parameter == "gain":
                    self.buses["master"].gain_db = value

    # =========================================================================
    # Rendering
    # =========================================================================

    def _get_max_length(self) -> int:
        """Get maximum track length in samples."""
        max_len = 0
        for track in self.tracks.values():
            if track.audio is not None:
                max_len = max(max_len, track.audio.shape[1])
        return max_len

    def _process_track(
        self,
        track: TrackState,
        start_sample: int,
        num_samples: int,
    ) -> StereoBuffer:
        """Process a track for a block of samples."""
        # Get audio slice
        if track.audio is None:
            return np.zeros((2, num_samples))

        end_sample = min(start_sample + num_samples, track.audio.shape[1])
        if start_sample >= track.audio.shape[1]:
            return np.zeros((2, num_samples))

        audio = track.audio[:, start_sample:end_sample]

        # Pad if needed
        if audio.shape[1] < num_samples:
            audio = np.pad(audio, ((0, 0), (0, num_samples - audio.shape[1])))

        # Apply EQ
        if track.eq is not None:
            audio = track.eq.process(audio)

        # Apply compressor
        if track.compressor is not None:
            audio, _ = track.compressor.process_stereo(audio)

        # Apply gain and pan
        gain_l, gain_r = track.get_panned_gain()
        audio[0] *= gain_l
        audio[1] *= gain_r

        return audio

    def _topological_sort_buses(self) -> List[str]:
        """Sort buses in processing order (leaves first, master last)."""
        # Build dependency graph
        dependencies: Dict[str, List[str]] = {}
        for name, bus in self.buses.items():
            dependencies[name] = []
            for other_name, other_bus in self.buses.items():
                if other_bus.output_bus == name:
                    dependencies[name].append(other_name)

        # Topological sort
        result = []
        visited = set()
        temp_mark = set()

        def visit(name: str) -> None:
            if name in temp_mark:
                raise ValueError("Circular bus routing detected")
            if name in visited:
                return

            temp_mark.add(name)
            for dep in dependencies.get(name, []):
                visit(dep)
            temp_mark.remove(name)
            visited.add(name)
            result.append(name)

        for name in dependencies:
            if name not in visited:
                visit(name)

        return result

    def render(self) -> StereoBuffer:
        """
        Render the complete mix.

        Returns:
            Stereo audio buffer (2, samples)
        """
        total_samples = self._get_max_length()
        if total_samples == 0:
            return np.zeros((2, 0))

        logger.info(f"Rendering mix: {len(self.tracks)} tracks, {total_samples} samples")

        # Check for solo
        has_solo = any(t.solo for t in self.tracks.values())

        # Get bus processing order
        bus_order = self._topological_sort_buses()

        # Initialize bus buffers
        for bus in self.buses.values():
            bus.buffer = np.zeros((2, total_samples))

        # Initialize effect send buffers
        effect_sends: Dict[str, StereoBuffer] = {
            name: np.zeros((2, total_samples)) for name in self.send_effects
        }

        # Process in blocks for automation
        for block_start in range(0, total_samples, self.block_size):
            block_end = min(block_start + self.block_size, total_samples)
            num_samples = block_end - block_start

            # Apply automation at block start
            time_seconds = block_start / self.sample_rate
            self._apply_automation(time_seconds)

            # Process tracks
            for track in self.tracks.values():
                # Skip muted tracks (unless solo active and track is soloed)
                if track.mute:
                    continue
                if has_solo and not track.solo:
                    continue

                # Process track
                audio = self._process_track(track, block_start, num_samples)

                # Route to output bus
                if track.output_bus in self.buses:
                    self.buses[track.output_bus].buffer[:, block_start:block_end] += audio

                # Send to effects
                for effect_name, send_level_db in track.sends.items():
                    if effect_name in effect_sends:
                        send_gain = db_to_linear(send_level_db)
                        effect_sends[effect_name][:, block_start:block_end] += audio * send_gain

        # Process effects
        effect_returns = np.zeros((2, total_samples))
        for name, effect in self.send_effects.items():
            if name in effect_sends:
                processed = effect.process_fn(effect_sends[name])
                return_gain = db_to_linear(effect.return_gain_db)
                effect_returns += processed * return_gain

        # Process buses in order (excluding master)
        for bus_name in bus_order:
            if bus_name == "master":
                continue

            bus = self.buses[bus_name]
            if bus.buffer is None:
                continue

            audio = bus.buffer

            # Apply bus EQ
            if bus.eq is not None:
                audio = bus.eq.process(audio)

            # Apply bus compressor
            if bus.compressor is not None:
                audio, _ = bus.compressor.process_stereo(audio)

            # Apply bus gain
            gain = db_to_linear(bus.gain_db)
            audio *= gain

            # Route to output bus
            if bus.output_bus in self.buses:
                self.buses[bus.output_bus].buffer += audio

        # Add effect returns to master
        self.buses["master"].buffer += effect_returns

        # Process master bus
        master = self.buses["master"]
        output = master.buffer.copy()

        # Master EQ
        if self.master_eq is not None:
            output = self.master_eq.process(output)

        # Master compressor
        if self.master_compressor is not None:
            output, _ = self.master_compressor.process_stereo(output)

        # Master gain
        master_gain = db_to_linear(master.gain_db)
        output *= master_gain

        # Apply headroom target
        peak = max(np.max(np.abs(output[0])), np.max(np.abs(output[1])))
        if peak > 0:
            target_peak = db_to_linear(self.target_headroom_db)
            if peak > target_peak:
                output *= target_peak / peak
                logger.warning(
                    f"Applied headroom limiting: {linear_to_db(peak):.1f} dB -> {self.target_headroom_db} dB"
                )

        # Output metering
        self.output_peak_db = linear_to_db(
            max(np.max(np.abs(output[0])), np.max(np.abs(output[1])))
        )
        self.mono_compatible, self.phase_correlation = StereoProcessor.check_mono_compatibility(
            output
        )

        logger.info(
            f"Mix complete: peak={self.output_peak_db:.1f} dB, "
            f"correlation={self.phase_correlation:.2f}, "
            f"mono_compatible={self.mono_compatible}"
        )

        return output

    def get_loudness_measurement(self, audio: StereoBuffer) -> Dict[str, float]:
        """Measure loudness of rendered audio."""
        self.loudness_meter.reset()
        measurement = self.loudness_meter.measure(audio)

        return {
            "integrated_lufs": measurement.integrated_lufs,
            "short_term_lufs": measurement.short_term_lufs,
            "momentary_lufs": measurement.momentary_lufs,
            "loudness_range_lu": measurement.loudness_range_lu,
            "true_peak_dbtp": measurement.true_peak_dbtp,
            "sample_peak_db": measurement.sample_peak_db,
        }


def create_basic_reverb(
    sample_rate: float,
    decay_seconds: float = 1.5,
    wet_level: float = 0.3,
) -> Callable[[StereoBuffer], StereoBuffer]:
    """Create a simple algorithmic reverb effect."""

    # Comb filter delays (in samples)
    comb_delays = [
        int(0.0297 * sample_rate),
        int(0.0371 * sample_rate),
        int(0.0411 * sample_rate),
        int(0.0437 * sample_rate),
    ]

    # All-pass filter delays
    allpass_delays = [
        int(0.0050 * sample_rate),
        int(0.0017 * sample_rate),
    ]

    # Calculate feedback based on decay time
    feedback = 0.84 ** (1.0 / (decay_seconds * sample_rate / comb_delays[0]))

    def process(audio: StereoBuffer) -> StereoBuffer:
        """Process audio through reverb."""
        output = np.zeros_like(audio)

        for ch in range(2):
            # Parallel comb filters
            comb_sum = np.zeros(audio.shape[1])
            for delay in comb_delays:
                # Simple feedback comb filter
                comb_out = np.zeros(audio.shape[1])
                buffer = np.zeros(delay)
                buf_idx = 0

                for i in range(audio.shape[1]):
                    comb_out[i] = buffer[buf_idx]
                    buffer[buf_idx] = audio[ch, i] + comb_out[i] * feedback
                    buf_idx = (buf_idx + 1) % delay

                comb_sum += comb_out

            comb_sum /= len(comb_delays)

            # Series all-pass filters
            ap_out = comb_sum
            for delay in allpass_delays:
                buffer = np.zeros(delay)
                buf_idx = 0
                new_ap_out = np.zeros(len(ap_out))

                for i in range(len(ap_out)):
                    delayed = buffer[buf_idx]
                    new_ap_out[i] = -0.7 * ap_out[i] + delayed + 0.7 * delayed
                    buffer[buf_idx] = ap_out[i]
                    buf_idx = (buf_idx + 1) % delay

                ap_out = new_ap_out

            output[ch] = ap_out

        # Mix wet/dry
        return audio * (1 - wet_level) + output * wet_level

    return process


def create_basic_delay(
    sample_rate: float,
    delay_time_ms: float = 375.0,
    feedback: float = 0.4,
    wet_level: float = 0.3,
) -> Callable[[StereoBuffer], StereoBuffer]:
    """Create a simple stereo delay effect."""

    delay_samples = int(delay_time_ms * sample_rate / 1000)

    def process(audio: StereoBuffer) -> StereoBuffer:
        """Process audio through delay."""
        output = np.zeros_like(audio)

        for ch in range(2):
            buffer = np.zeros(delay_samples)
            buf_idx = 0
            delay_out = np.zeros(audio.shape[1])

            for i in range(audio.shape[1]):
                delay_out[i] = buffer[buf_idx]
                buffer[buf_idx] = audio[ch, i] + delay_out[i] * feedback
                buf_idx = (buf_idx + 1) % delay_samples

            output[ch] = delay_out

        return audio * (1 - wet_level) + output * wet_level

    return process
