"""
Motif Generator - Genre-specific melodic pattern generation.

Creates characteristic melodic motifs based on genre DNA specifications.
Each genre has distinct motif rules governing:
- Note density and rhythm
- Interval preferences
- Phrase length and contour
- Articulation patterns
"""

from dataclasses import dataclass, field
from typing import Optional
import random

from aether.genre.dna import (
    GenreDNA,
    ContourType,
    get_genre_dna,
)


@dataclass
class Motif:
    """A melodic motif with notes and metadata."""
    # Note data: list of (pitch_offset, duration_beats, velocity)
    notes: list[tuple[int, float, int]]

    # Metadata
    contour: ContourType
    length_beats: float
    root_pitch: int = 60  # Middle C default

    # Genre source
    genre_id: Optional[str] = None

    def to_absolute_pitches(self, root: int = 60, scale: list[int] = None) -> list[dict]:
        """Convert to absolute pitch notes."""
        if scale is None:
            scale = [0, 2, 4, 5, 7, 9, 11]  # Major scale

        result = []
        current_time = 0.0

        for offset, duration, velocity in self.notes:
            # Map offset to scale degree
            octave = offset // len(scale)
            degree = offset % len(scale)
            pitch = root + scale[degree] + (octave * 12)

            result.append({
                "pitch": pitch,
                "start_beat": current_time,
                "duration_beats": duration,
                "velocity": velocity,
            })
            current_time += duration

        return result

    def transpose(self, semitones: int) -> "Motif":
        """Return transposed copy."""
        return Motif(
            notes=[(p + semitones, d, v) for p, d, v in self.notes],
            contour=self.contour,
            length_beats=self.length_beats,
            root_pitch=self.root_pitch + semitones,
            genre_id=self.genre_id,
        )

    def augment(self, factor: float) -> "Motif":
        """Return time-stretched copy."""
        return Motif(
            notes=[(p, d * factor, v) for p, d, v in self.notes],
            contour=self.contour,
            length_beats=self.length_beats * factor,
            root_pitch=self.root_pitch,
            genre_id=self.genre_id,
        )


@dataclass
class MotifRules:
    """Rules for generating motifs in a specific genre."""
    # Timing
    min_notes: int = 3
    max_notes: int = 8
    min_length_beats: float = 1.0
    max_length_beats: float = 4.0

    # Rhythm
    preferred_durations: list[float] = field(default_factory=lambda: [0.5, 1.0])
    syncopation_probability: float = 0.2
    rest_probability: float = 0.1

    # Pitch
    preferred_intervals: list[int] = field(default_factory=lambda: [1, 2, 3])  # Scale degrees
    max_leap: int = 5  # Scale degrees
    contour_types: list[ContourType] = field(default_factory=lambda: [ContourType.ARCH])

    # Dynamics
    velocity_range: tuple[int, int] = (70, 100)
    accent_beats: list[float] = field(default_factory=list)  # Beats to accent

    # Repetition
    repetition_probability: float = 0.3
    variation_probability: float = 0.2


# Genre-specific motif rules
GENRE_MOTIF_RULES: dict[str, MotifRules] = {
    "lofi-hip-hop": MotifRules(
        min_notes=4,
        max_notes=8,
        min_length_beats=2.0,
        max_length_beats=4.0,
        preferred_durations=[0.5, 0.75, 1.0],
        syncopation_probability=0.4,
        rest_probability=0.2,
        preferred_intervals=[1, 2, 3, 5],
        max_leap=4,
        contour_types=[ContourType.DESCENT, ContourType.WAVE],
        velocity_range=(50, 85),
        repetition_probability=0.4,
    ),
    "trap": MotifRules(
        min_notes=3,
        max_notes=6,
        min_length_beats=1.0,
        max_length_beats=2.0,
        preferred_durations=[0.25, 0.5],
        syncopation_probability=0.5,
        rest_probability=0.3,
        preferred_intervals=[1, 3, 4],
        max_leap=5,
        contour_types=[ContourType.DESCENT, ContourType.STATIC],
        velocity_range=(80, 120),
        repetition_probability=0.5,
    ),
    "drill": MotifRules(
        min_notes=3,
        max_notes=5,
        min_length_beats=0.5,
        max_length_beats=2.0,
        preferred_durations=[0.25, 0.5],
        syncopation_probability=0.6,
        rest_probability=0.2,
        preferred_intervals=[1, 2, 7],  # Minor 2nd emphasis
        max_leap=7,
        contour_types=[ContourType.DESCENT, ContourType.STATIC],
        velocity_range=(90, 127),
        repetition_probability=0.6,
    ),
    "boom-bap": MotifRules(
        min_notes=4,
        max_notes=8,
        min_length_beats=2.0,
        max_length_beats=4.0,
        preferred_durations=[0.5, 1.0, 1.5],
        syncopation_probability=0.3,
        rest_probability=0.15,
        preferred_intervals=[2, 3, 4, 5],
        max_leap=5,
        contour_types=[ContourType.ARCH, ContourType.WAVE],
        velocity_range=(70, 100),
        repetition_probability=0.3,
    ),
    "synthwave": MotifRules(
        min_notes=4,
        max_notes=10,
        min_length_beats=2.0,
        max_length_beats=8.0,
        preferred_durations=[0.5, 1.0, 2.0],
        syncopation_probability=0.2,
        rest_probability=0.1,
        preferred_intervals=[2, 4, 5, 7],
        max_leap=7,
        contour_types=[ContourType.ASCENT, ContourType.ARCH],
        velocity_range=(80, 110),
        repetition_probability=0.4,
    ),
    "house": MotifRules(
        min_notes=3,
        max_notes=6,
        min_length_beats=1.0,
        max_length_beats=4.0,
        preferred_durations=[0.5, 1.0],
        syncopation_probability=0.4,
        rest_probability=0.15,
        preferred_intervals=[1, 3, 5],
        max_leap=5,
        contour_types=[ContourType.WAVE, ContourType.ARCH],
        velocity_range=(80, 110),
        accent_beats=[0.0, 1.0, 2.0, 3.0],
        repetition_probability=0.5,
    ),
    "techno": MotifRules(
        min_notes=2,
        max_notes=6,
        min_length_beats=1.0,
        max_length_beats=4.0,
        preferred_durations=[0.25, 0.5, 1.0],
        syncopation_probability=0.3,
        rest_probability=0.2,
        preferred_intervals=[1, 2, 5],
        max_leap=4,
        contour_types=[ContourType.STATIC, ContourType.WAVE],
        velocity_range=(90, 120),
        repetition_probability=0.7,
    ),
    "drum-and-bass": MotifRules(
        min_notes=4,
        max_notes=8,
        min_length_beats=1.0,
        max_length_beats=2.0,
        preferred_durations=[0.25, 0.5],
        syncopation_probability=0.5,
        rest_probability=0.15,
        preferred_intervals=[2, 3, 5],
        max_leap=5,
        contour_types=[ContourType.ASCENT, ContourType.WAVE],
        velocity_range=(85, 115),
        repetition_probability=0.4,
    ),
    "reggaeton": MotifRules(
        min_notes=4,
        max_notes=8,
        min_length_beats=2.0,
        max_length_beats=4.0,
        preferred_durations=[0.5, 0.75, 1.0],
        syncopation_probability=0.5,
        rest_probability=0.1,
        preferred_intervals=[1, 2, 3, 5],
        max_leap=4,
        contour_types=[ContourType.WAVE, ContourType.ARCH],
        velocity_range=(80, 105),
        repetition_probability=0.5,
    ),
    "afrobeat": MotifRules(
        min_notes=5,
        max_notes=10,
        min_length_beats=2.0,
        max_length_beats=8.0,
        preferred_durations=[0.5, 0.75, 1.0],
        syncopation_probability=0.6,
        rest_probability=0.1,
        preferred_intervals=[1, 2, 3, 4, 5],
        max_leap=5,
        contour_types=[ContourType.WAVE, ContourType.ASCENT],
        velocity_range=(75, 100),
        repetition_probability=0.4,
    ),
    "pop": MotifRules(
        min_notes=4,
        max_notes=8,
        min_length_beats=2.0,
        max_length_beats=4.0,
        preferred_durations=[0.5, 1.0, 1.5],
        syncopation_probability=0.25,
        rest_probability=0.1,
        preferred_intervals=[1, 2, 3, 5],
        max_leap=5,
        contour_types=[ContourType.ARCH, ContourType.ASCENT],
        velocity_range=(75, 105),
        repetition_probability=0.4,
    ),
    "cinematic": MotifRules(
        min_notes=3,
        max_notes=8,
        min_length_beats=2.0,
        max_length_beats=8.0,
        preferred_durations=[1.0, 2.0, 4.0],
        syncopation_probability=0.1,
        rest_probability=0.2,
        preferred_intervals=[2, 4, 5, 7],
        max_leap=7,
        contour_types=[ContourType.ASCENT, ContourType.ARCH],
        velocity_range=(50, 100),
        repetition_probability=0.3,
        variation_probability=0.4,
    ),
}


class MotifGenerator:
    """
    Generates melodic motifs according to genre-specific rules.

    Motifs are short melodic phrases that capture the essence of a genre.
    They can be used as building blocks for larger melodies.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self.rng = random.Random(seed)

    def generate(
        self,
        genre_id: str,
        length_beats: Optional[float] = None,
        contour: Optional[ContourType] = None,
        root_pitch: int = 60,
    ) -> Motif:
        """
        Generate a motif for the specified genre.

        Args:
            genre_id: Target genre
            length_beats: Desired length (or random within rules)
            contour: Desired contour shape (or random)
            root_pitch: Root pitch for the motif

        Returns:
            Generated Motif
        """
        rules = GENRE_MOTIF_RULES.get(genre_id)
        if rules is None:
            # Fall back to pop rules
            rules = GENRE_MOTIF_RULES["pop"]

        # Determine length
        if length_beats is None:
            length_beats = self.rng.uniform(rules.min_length_beats, rules.max_length_beats)

        # Determine contour
        if contour is None:
            contour = self.rng.choice(rules.contour_types)

        # Generate notes
        notes = self._generate_notes(rules, length_beats, contour)

        return Motif(
            notes=notes,
            contour=contour,
            length_beats=length_beats,
            root_pitch=root_pitch,
            genre_id=genre_id,
        )

    def _generate_notes(
        self,
        rules: MotifRules,
        length_beats: float,
        contour: ContourType,
    ) -> list[tuple[int, float, int]]:
        """Generate note sequence based on rules and contour."""
        notes = []
        current_time = 0.0
        current_pitch = 0  # Scale degree offset

        # Target number of notes
        num_notes = self.rng.randint(rules.min_notes, rules.max_notes)

        # Generate contour shape
        contour_shape = self._generate_contour_shape(contour, num_notes)

        for i, target_pitch in enumerate(contour_shape):
            # Check if we should add a rest
            if self.rng.random() < rules.rest_probability and i > 0:
                rest_duration = self.rng.choice(rules.preferred_durations)
                current_time += rest_duration
                continue

            # Duration
            duration = self.rng.choice(rules.preferred_durations)

            # Ensure we don't exceed length
            if current_time + duration > length_beats:
                duration = length_beats - current_time
                if duration <= 0:
                    break

            # Pitch movement towards contour target
            if i > 0:
                interval = self.rng.choice(rules.preferred_intervals)
                if target_pitch > current_pitch:
                    current_pitch += interval
                elif target_pitch < current_pitch:
                    current_pitch -= interval
                # else stay at current pitch
            else:
                current_pitch = target_pitch

            # Clamp leap
            if i > 0:
                prev_pitch = notes[-1][0]
                if abs(current_pitch - prev_pitch) > rules.max_leap:
                    direction = 1 if current_pitch > prev_pitch else -1
                    current_pitch = prev_pitch + direction * rules.max_leap

            # Velocity
            base_velocity = self.rng.randint(*rules.velocity_range)

            # Accent on specified beats
            if rules.accent_beats and any(
                abs(current_time - b) < 0.1 for b in rules.accent_beats
            ):
                base_velocity = min(127, base_velocity + 15)

            # Syncopation: offset timing slightly
            if self.rng.random() < rules.syncopation_probability:
                # Add slight swing/push
                pass  # Would require fractional beat handling

            notes.append((current_pitch, duration, base_velocity))
            current_time += duration

            if current_time >= length_beats:
                break

        return notes

    def _generate_contour_shape(
        self,
        contour: ContourType,
        num_points: int,
    ) -> list[int]:
        """Generate target pitch contour."""
        if contour == ContourType.ASCENT:
            return list(range(0, num_points))

        elif contour == ContourType.DESCENT:
            return list(range(num_points - 1, -1, -1))

        elif contour == ContourType.ARCH:
            mid = num_points // 2
            ascending = list(range(0, mid + 1))
            descending = list(range(mid - 1, -1, -1))
            return ascending + descending[:num_points - len(ascending)]

        elif contour == ContourType.WAVE:
            import math
            return [int(2 * math.sin(i * math.pi / 2)) for i in range(num_points)]

        elif contour == ContourType.STATIC:
            return [0] * num_points

        elif contour == ContourType.QUESTION_ANSWER:
            half = num_points // 2
            call = [0, 2, 4, 2][:half]
            response = [3, 1, -1, 0][:num_points - half]
            return call + response

        else:
            return [0] * num_points

    def generate_variation(
        self,
        original: Motif,
        variation_type: str = "rhythmic",
    ) -> Motif:
        """Generate a variation of an existing motif."""
        notes = list(original.notes)

        if variation_type == "rhythmic":
            # Modify durations
            notes = [
                (p, d * self.rng.choice([0.5, 1.0, 1.5, 2.0]), v)
                for p, d, v in notes
            ]

        elif variation_type == "melodic":
            # Transpose some notes
            notes = [
                (p + self.rng.choice([-2, -1, 0, 1, 2]), d, v)
                for p, d, v in notes
            ]

        elif variation_type == "dynamic":
            # Vary velocities
            notes = [
                (p, d, max(1, min(127, v + self.rng.randint(-20, 20))))
                for p, d, v in notes
            ]

        elif variation_type == "retrograde":
            # Reverse the motif
            notes = notes[::-1]

        elif variation_type == "inversion":
            # Invert intervals
            center = notes[0][0]
            notes = [
                (center - (p - center), d, v)
                for p, d, v in notes
            ]

        # Recalculate length
        new_length = sum(d for _, d, _ in notes)

        return Motif(
            notes=notes,
            contour=original.contour,
            length_beats=new_length,
            root_pitch=original.root_pitch,
            genre_id=original.genre_id,
        )

    def generate_phrase(
        self,
        genre_id: str,
        num_motifs: int = 4,
        include_variation: bool = True,
    ) -> list[Motif]:
        """Generate a phrase (sequence of related motifs)."""
        phrase = []

        # Generate base motif
        base = self.generate(genre_id)
        phrase.append(base)

        for i in range(1, num_motifs):
            if include_variation and self.rng.random() < 0.5:
                # Variation of previous
                variation_type = self.rng.choice([
                    "rhythmic", "melodic", "dynamic", "retrograde"
                ])
                motif = self.generate_variation(phrase[-1], variation_type)
            else:
                # New contrasting motif
                motif = self.generate(genre_id)

            phrase.append(motif)

        return phrase


def get_motif_rules(genre_id: str) -> Optional[MotifRules]:
    """Get motif rules for a genre."""
    return GENRE_MOTIF_RULES.get(genre_id)
