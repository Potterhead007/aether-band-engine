"""
Vocal Ornamentation Engine

Generates runs, bends, scoops, falls, and other vocal ornaments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from aether.voice.performance.profiles import VocalPerformanceProfile, OrnamentationProfile


class OrnamentType(Enum):
    """Types of vocal ornaments."""
    RUN = "run"
    BEND = "bend"
    SCOOP = "scoop"
    FALL = "fall"
    TRILL = "trill"
    MORDENT = "mordent"
    TURN = "turn"
    GRACE_NOTE = "grace_note"
    FLIP = "flip"
    RIFF = "riff"


@dataclass
class OrnamentSpec:
    """Specification for an ornament."""
    ornament_type: OrnamentType
    start_beat: float
    duration_beats: float
    target_pitch: int  # MIDI note
    pitches: List[int]  # Sequence of MIDI notes
    velocities: List[int]  # Velocity for each note
    timing_ratios: List[float]  # Duration ratios

    def to_pitch_sequence(self, tempo: float) -> List[Tuple[int, float, float]]:
        """
        Convert to pitch sequence.

        Returns:
            List of (midi_note, start_ms, duration_ms) tuples
        """
        beat_ms = 60000 / tempo
        total_ms = self.duration_beats * beat_ms

        result = []
        current_ms = self.start_beat * beat_ms

        for i, (pitch, ratio) in enumerate(zip(self.pitches, self.timing_ratios)):
            duration_ms = total_ms * ratio
            result.append((pitch, current_ms, duration_ms))
            current_ms += duration_ms

        return result


class OrnamentationEngine:
    """
    Generates vocal ornaments appropriate for genre and context.

    Features:
    - Genre-appropriate run generation
    - Note bending with proper curves
    - Scoops and falls
    - Trills and mordents
    - Context-aware placement
    """

    # Scale patterns for runs
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
    PENTATONIC_MAJOR = [0, 2, 4, 7, 9]
    PENTATONIC_MINOR = [0, 3, 5, 7, 10]
    BLUES_SCALE = [0, 3, 5, 6, 7, 10]

    # Genre-to-scale mapping
    GENRE_SCALES: Dict[str, List[int]] = {
        "pop": PENTATONIC_MAJOR,
        "r-and-b": PENTATONIC_MINOR,
        "rock": PENTATONIC_MINOR,
        "jazz": MAJOR_SCALE,
        "house": MAJOR_SCALE,
        "trap": PENTATONIC_MINOR,
        "funk": BLUES_SCALE,
        "ambient": MAJOR_SCALE,
        "neo-soul": BLUES_SCALE,
    }

    def __init__(self, profile: VocalPerformanceProfile):
        """
        Initialize ornamentation engine.

        Args:
            profile: Performance profile for genre
        """
        self.profile = profile
        self.orn_profile = profile.ornamentation

    def should_ornament(
        self,
        context: dict,
        ornament_type: OrnamentType,
    ) -> bool:
        """
        Decide whether to apply ornamentation.

        Args:
            context: Note context (position, duration, etc.)
            ornament_type: Type of ornament to consider

        Returns:
            Whether to apply the ornament
        """
        # Get frequency for this ornament type
        frequency_map = {
            OrnamentType.RUN: self.orn_profile.run_frequency,
            OrnamentType.BEND: self.orn_profile.bend_frequency,
            OrnamentType.SCOOP: self.orn_profile.scoop_frequency,
            OrnamentType.FALL: self.orn_profile.fall_frequency,
            OrnamentType.TRILL: 0.1 if self.orn_profile.trill_allowed else 0.0,
        }

        frequency = frequency_map.get(ornament_type, 0.1)

        # Adjust based on context
        if context.get("is_phrase_end") and ornament_type == OrnamentType.FALL:
            frequency *= 1.5
        if context.get("is_phrase_start") and ornament_type == OrnamentType.SCOOP:
            frequency *= 1.3
        if context.get("duration_beats", 0) < 0.5 and ornament_type == OrnamentType.RUN:
            frequency *= 0.3  # Reduce runs on short notes

        return np.random.random() < frequency

    def generate_run(
        self,
        start_pitch: int,
        end_pitch: int,
        duration_beats: float,
        genre: str = "pop",
        complexity: Optional[int] = None,
    ) -> OrnamentSpec:
        """
        Generate a melodic run.

        Args:
            start_pitch: Starting MIDI note
            end_pitch: Target MIDI note
            duration_beats: Duration in beats
            genre: Genre for scale selection
            complexity: 1-5, overrides profile if set

        Returns:
            OrnamentSpec for the run
        """
        complexity = complexity or self.orn_profile.run_complexity
        scale = self.GENRE_SCALES.get(genre, self.PENTATONIC_MAJOR)

        # Determine direction
        direction = 1 if end_pitch > start_pitch else -1
        interval = abs(end_pitch - start_pitch)

        # Generate notes based on complexity
        num_notes = min(complexity + 2, int(duration_beats * 8))

        # Get scale degrees between start and end
        pitches = self._get_scale_run(start_pitch, end_pitch, scale, num_notes)

        # Generate velocities (slight accent on downbeats)
        velocities = []
        for i in range(len(pitches)):
            base_vel = 90
            if i == 0 or i == len(pitches) - 1:
                base_vel = 100
            elif i % 2 == 0:
                base_vel = 95
            velocities.append(base_vel + np.random.randint(-5, 5))

        # Generate timing ratios (slightly uneven for human feel)
        base_ratio = 1.0 / len(pitches)
        timing_ratios = []
        for i in range(len(pitches)):
            ratio = base_ratio * (1 + np.random.uniform(-0.1, 0.1))
            timing_ratios.append(ratio)

        # Normalize ratios
        total = sum(timing_ratios)
        timing_ratios = [r / total for r in timing_ratios]

        return OrnamentSpec(
            ornament_type=OrnamentType.RUN,
            start_beat=0,
            duration_beats=duration_beats,
            target_pitch=end_pitch,
            pitches=pitches,
            velocities=velocities,
            timing_ratios=timing_ratios,
        )

    def _get_scale_run(
        self,
        start: int,
        end: int,
        scale: List[int],
        num_notes: int,
    ) -> List[int]:
        """Generate scale-based run between two pitches."""
        direction = 1 if end > start else -1

        # Get all scale tones in range
        scale_tones = []
        octave = start // 12

        for offset in range(-12, abs(end - start) + 12):
            for degree in scale:
                note = (octave * 12) + degree + (offset // len(scale)) * 12
                if direction > 0:
                    if start <= note <= end:
                        scale_tones.append(note)
                else:
                    if end <= note <= start:
                        scale_tones.append(note)

        scale_tones = sorted(set(scale_tones))
        if direction < 0:
            scale_tones = scale_tones[::-1]

        if not scale_tones:
            # Fallback to chromatic
            if direction > 0:
                scale_tones = list(range(start, end + 1))
            else:
                scale_tones = list(range(start, end - 1, -1))

        # Select subset of notes
        if len(scale_tones) <= num_notes:
            return scale_tones

        # Interpolate to get desired number
        indices = np.linspace(0, len(scale_tones) - 1, num_notes).astype(int)
        return [scale_tones[i] for i in indices]

    def generate_bend(
        self,
        start_pitch: int,
        bend_cents: float,
        duration_beats: float,
        return_to_pitch: bool = True,
    ) -> OrnamentSpec:
        """
        Generate a pitch bend.

        Args:
            start_pitch: Starting MIDI note
            bend_cents: Bend amount in cents (+/-)
            duration_beats: Duration of bend
            return_to_pitch: Whether to return to original

        Returns:
            OrnamentSpec for the bend
        """
        # Bend is a continuous pitch curve, represented as dense points
        num_points = max(4, int(duration_beats * 16))

        if return_to_pitch:
            # Up and back
            t = np.linspace(0, 1, num_points)
            bend_curve = np.sin(t * np.pi) * bend_cents
        else:
            # Just bend
            t = np.linspace(0, 1, num_points)
            bend_curve = t * bend_cents

        # Convert cents to MIDI pitch offset
        pitches = [int(start_pitch + c / 100) for c in bend_curve]
        velocities = [90] * len(pitches)
        timing_ratios = [1.0 / len(pitches)] * len(pitches)

        return OrnamentSpec(
            ornament_type=OrnamentType.BEND,
            start_beat=0,
            duration_beats=duration_beats,
            target_pitch=start_pitch,
            pitches=pitches,
            velocities=velocities,
            timing_ratios=timing_ratios,
        )

    def generate_scoop(
        self,
        target_pitch: int,
        duration_beats: float = 0.125,
        scoop_semitones: float = -2.0,
    ) -> OrnamentSpec:
        """
        Generate a scoop (approach from below).

        Args:
            target_pitch: Target MIDI note
            duration_beats: Duration of scoop
            scoop_semitones: How far below to start

        Returns:
            OrnamentSpec for the scoop
        """
        start_pitch = int(target_pitch + scoop_semitones)
        num_points = max(3, int(duration_beats * 16))

        # Exponential curve up
        t = np.linspace(0, 1, num_points)
        curve = t ** 0.5  # Fast start, slow finish

        pitches = [int(start_pitch + (target_pitch - start_pitch) * c) for c in curve]
        velocities = [80 + int(20 * c) for c in curve]
        timing_ratios = [1.0 / len(pitches)] * len(pitches)

        return OrnamentSpec(
            ornament_type=OrnamentType.SCOOP,
            start_beat=0,
            duration_beats=duration_beats,
            target_pitch=target_pitch,
            pitches=pitches,
            velocities=velocities,
            timing_ratios=timing_ratios,
        )

    def generate_fall(
        self,
        start_pitch: int,
        duration_beats: float = 0.25,
        fall_semitones: float = -4.0,
    ) -> OrnamentSpec:
        """
        Generate a fall (release downward).

        Args:
            start_pitch: Starting MIDI note
            duration_beats: Duration of fall
            fall_semitones: How far to fall

        Returns:
            OrnamentSpec for the fall
        """
        end_pitch = int(start_pitch + fall_semitones)
        num_points = max(3, int(duration_beats * 16))

        # Logarithmic curve down
        t = np.linspace(0, 1, num_points)
        curve = 1 - (1 - t) ** 2  # Slow start, fast end

        pitches = [int(start_pitch + (end_pitch - start_pitch) * c) for c in curve]
        velocities = [100 - int(30 * c) for c in curve]  # Fade out
        timing_ratios = [1.0 / len(pitches)] * len(pitches)

        return OrnamentSpec(
            ornament_type=OrnamentType.FALL,
            start_beat=0,
            duration_beats=duration_beats,
            target_pitch=end_pitch,
            pitches=pitches,
            velocities=velocities,
            timing_ratios=timing_ratios,
        )

    def generate_trill(
        self,
        main_pitch: int,
        duration_beats: float,
        interval: int = 2,  # Semitones
        speed: float = 8.0,  # Oscillations per beat
    ) -> OrnamentSpec:
        """
        Generate a trill.

        Args:
            main_pitch: Main MIDI note
            duration_beats: Duration
            interval: Upper note interval in semitones
            speed: Oscillation speed

        Returns:
            OrnamentSpec for the trill
        """
        upper_pitch = main_pitch + interval
        num_oscillations = int(duration_beats * speed)

        pitches = []
        velocities = []

        for i in range(num_oscillations * 2):
            pitch = main_pitch if i % 2 == 0 else upper_pitch
            pitches.append(pitch)
            velocities.append(85 + np.random.randint(-5, 5))

        timing_ratios = [1.0 / len(pitches)] * len(pitches)

        return OrnamentSpec(
            ornament_type=OrnamentType.TRILL,
            start_beat=0,
            duration_beats=duration_beats,
            target_pitch=main_pitch,
            pitches=pitches,
            velocities=velocities,
            timing_ratios=timing_ratios,
        )

    def generate_mordent(
        self,
        main_pitch: int,
        duration_beats: float = 0.125,
        upper: bool = True,
    ) -> OrnamentSpec:
        """
        Generate a mordent (single oscillation).

        Args:
            main_pitch: Main MIDI note
            duration_beats: Duration
            upper: Upper mordent if True, lower if False

        Returns:
            OrnamentSpec for the mordent
        """
        auxiliary = main_pitch + (2 if upper else -2)

        pitches = [main_pitch, auxiliary, main_pitch]
        velocities = [95, 85, 90]
        timing_ratios = [0.35, 0.3, 0.35]

        return OrnamentSpec(
            ornament_type=OrnamentType.MORDENT,
            start_beat=0,
            duration_beats=duration_beats,
            target_pitch=main_pitch,
            pitches=pitches,
            velocities=velocities,
            timing_ratios=timing_ratios,
        )

    def generate_grace_note(
        self,
        target_pitch: int,
        grace_pitch: int,
        duration_beats: float = 0.0625,
    ) -> OrnamentSpec:
        """
        Generate a grace note.

        Args:
            target_pitch: Main MIDI note
            grace_pitch: Grace note pitch
            duration_beats: Grace note duration

        Returns:
            OrnamentSpec for the grace note
        """
        pitches = [grace_pitch, target_pitch]
        velocities = [80, 95]
        timing_ratios = [0.25, 0.75]

        return OrnamentSpec(
            ornament_type=OrnamentType.GRACE_NOTE,
            start_beat=0,
            duration_beats=duration_beats,
            target_pitch=target_pitch,
            pitches=pitches,
            velocities=velocities,
            timing_ratios=timing_ratios,
        )

    def generate_riff(
        self,
        root_pitch: int,
        duration_beats: float,
        genre: str = "r-and-b",
    ) -> OrnamentSpec:
        """
        Generate a genre-specific riff pattern.

        Args:
            root_pitch: Root MIDI note
            duration_beats: Duration
            genre: Genre for pattern selection

        Returns:
            OrnamentSpec for the riff
        """
        scale = self.GENRE_SCALES.get(genre, self.PENTATONIC_MINOR)

        # Generate riff pattern based on genre
        if genre in ["r-and-b", "neo-soul"]:
            # Melismatic pattern
            intervals = np.random.choice(scale[:5], size=6, replace=True)
            pattern = [root_pitch + i for i in intervals]
        elif genre == "jazz":
            # Bebop-style pattern
            intervals = [0, 2, 4, 7, 9, 7, 4, 2]
            pattern = [root_pitch + scale[i % len(scale)] for i in intervals[:6]]
        elif genre == "funk":
            # Syncopated pattern
            intervals = [0, 3, 5, 0, 7, 5]
            pattern = [root_pitch + i for i in intervals]
        else:
            # Generic pattern
            intervals = np.random.choice(scale, size=4, replace=True)
            pattern = [root_pitch + i for i in intervals]

        pitches = pattern
        velocities = [90 + np.random.randint(-5, 10) for _ in pitches]

        # Syncopated timing
        timing_ratios = []
        for i in range(len(pitches)):
            if i % 2 == 0:
                timing_ratios.append(0.15)
            else:
                timing_ratios.append(0.18)

        total = sum(timing_ratios)
        timing_ratios = [r / total for r in timing_ratios]

        return OrnamentSpec(
            ornament_type=OrnamentType.RIFF,
            start_beat=0,
            duration_beats=duration_beats,
            target_pitch=root_pitch,
            pitches=pitches,
            velocities=velocities,
            timing_ratios=timing_ratios,
        )

    def apply_ornaments(
        self,
        notes: List[dict],
        genre: str = "pop",
    ) -> List[dict]:
        """
        Apply appropriate ornaments to a note sequence.

        Args:
            notes: List of note dicts with pitch, start, duration
            genre: Genre for style

        Returns:
            Notes with ornaments inserted
        """
        result = []

        for i, note in enumerate(notes):
            context = {
                "index": i,
                "is_phrase_start": note.get("is_phrase_start", i == 0),
                "is_phrase_end": note.get("is_phrase_end", i == len(notes) - 1),
                "duration_beats": note.get("duration_beats", 1.0),
                "pitch": note.get("pitch", 60),
            }

            # Check for scoop on phrase start
            if self.should_ornament(context, OrnamentType.SCOOP):
                scoop = self.generate_scoop(
                    note["pitch"],
                    duration_beats=min(0.125, note.get("duration_beats", 1.0) * 0.2),
                )
                result.append({
                    "type": "ornament",
                    "ornament": scoop,
                    "start_beat": note.get("start_beat", 0) - scoop.duration_beats,
                })

            # Add the main note
            result.append(note)

            # Check for fall on phrase end
            if self.should_ornament(context, OrnamentType.FALL):
                fall = self.generate_fall(
                    note["pitch"],
                    duration_beats=min(0.25, note.get("duration_beats", 1.0) * 0.3),
                )
                result.append({
                    "type": "ornament",
                    "ornament": fall,
                    "start_beat": (
                        note.get("start_beat", 0) +
                        note.get("duration_beats", 1.0) -
                        fall.duration_beats
                    ),
                })

        return result
