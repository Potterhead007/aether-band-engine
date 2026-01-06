"""
Genre Conditioner - Translates genre selection into generation constraints.

Converts high-level genre DNA into actionable constraints for the generation pipeline:
- Hard constraints: Must be satisfied or output is rejected
- Soft preferences: Guide generation with weighted targets
- Rejection criteria: Automatic rejection rules for off-genre output
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from aether.genre.dna import (
    GenreDNA,
    get_genre_dna,
    KickPattern,
    SnarePosition,
    TimeFeel,
)


class ConstraintPriority(Enum):
    """Priority levels for constraint enforcement."""
    CRITICAL = 1      # Must satisfy or reject
    HIGH = 2          # Strong preference, slight deviation OK
    MEDIUM = 3        # Target value, moderate deviation OK
    LOW = 4           # Nice to have


@dataclass
class TempoConstraint:
    """Tempo constraint with range and target."""
    min_bpm: float
    max_bpm: float
    target_bpm: float
    priority: ConstraintPriority = ConstraintPriority.HIGH

    def validate(self, bpm: float) -> tuple[bool, float]:
        """Check if BPM is valid and return deviation score."""
        valid = self.min_bpm <= bpm <= self.max_bpm
        if not valid:
            return False, 1.0
        deviation = abs(bpm - self.target_bpm) / (self.max_bpm - self.min_bpm)
        return True, deviation


@dataclass
class SwingConstraint:
    """Swing feel constraint."""
    min_swing: float
    max_swing: float
    target_swing: float
    priority: ConstraintPriority = ConstraintPriority.MEDIUM

    def validate(self, swing: float) -> tuple[bool, float]:
        """Check if swing amount is valid."""
        valid = self.min_swing <= swing <= self.max_swing
        if not valid:
            return False, 1.0
        if self.max_swing == self.min_swing:
            return True, 0.0
        deviation = abs(swing - self.target_swing) / max(0.01, self.max_swing - self.min_swing)
        return True, deviation


@dataclass
class KeyConstraint:
    """Key/mode constraint for harmonic content."""
    allowed_modes: list[str]
    preferred_modes: list[str]
    priority: ConstraintPriority = ConstraintPriority.MEDIUM

    def validate(self, mode: str) -> tuple[bool, float]:
        """Check if mode is valid."""
        if mode not in self.allowed_modes:
            return False, 1.0
        score = 0.0 if mode in self.preferred_modes else 0.3
        return True, score


@dataclass
class RhythmConstraint:
    """Rhythm pattern constraints."""
    allowed_kick_patterns: list[KickPattern]
    allowed_snare_positions: list[SnarePosition]
    allowed_time_feels: list[TimeFeel]
    preferred_kick: KickPattern
    preferred_snare: SnarePosition
    preferred_feel: TimeFeel
    priority: ConstraintPriority = ConstraintPriority.HIGH

    def validate_kick(self, pattern: KickPattern) -> tuple[bool, float]:
        """Validate kick pattern."""
        if pattern not in self.allowed_kick_patterns:
            return False, 1.0
        return True, 0.0 if pattern == self.preferred_kick else 0.2

    def validate_snare(self, position: SnarePosition) -> tuple[bool, float]:
        """Validate snare position."""
        if position not in self.allowed_snare_positions:
            return False, 1.0
        return True, 0.0 if position == self.preferred_snare else 0.2

    def validate_feel(self, feel: TimeFeel) -> tuple[bool, float]:
        """Validate time feel."""
        if feel not in self.allowed_time_feels:
            return False, 1.0
        return True, 0.0 if feel == self.preferred_feel else 0.2


@dataclass
class HarmonyConstraint:
    """Harmonic content constraints."""
    min_chord_complexity: float
    max_chord_complexity: float
    target_complexity: float
    allowed_cadences: list[str]
    required_progressions: list[str]  # At least one must appear
    priority: ConstraintPriority = ConstraintPriority.MEDIUM

    def validate_complexity(self, complexity: float) -> tuple[bool, float]:
        """Validate chord complexity."""
        valid = self.min_chord_complexity <= complexity <= self.max_chord_complexity
        if not valid:
            return False, 1.0
        deviation = abs(complexity - self.target_complexity) / max(0.01, self.max_chord_complexity - self.min_chord_complexity)
        return True, deviation


@dataclass
class MelodyConstraint:
    """Melodic content constraints."""
    min_note_density: float
    max_note_density: float
    target_density: float
    allowed_scales: list[str]
    max_leap_interval: int  # semitones
    priority: ConstraintPriority = ConstraintPriority.MEDIUM

    def validate_density(self, density: float) -> tuple[bool, float]:
        """Validate note density."""
        valid = self.min_note_density <= density <= self.max_note_density
        if not valid:
            return False, 1.0
        deviation = abs(density - self.target_density) / max(0.01, self.max_note_density - self.min_note_density)
        return True, deviation


@dataclass
class InstrumentConstraint:
    """Instrumentation constraints."""
    required_instruments: list[str]  # Must have at least one
    forbidden_instruments: list[str]  # Never use these
    preferred_instruments: list[str]  # Use when possible
    priority: ConstraintPriority = ConstraintPriority.HIGH

    def validate_instruments(self, instruments: list[str]) -> tuple[bool, float]:
        """Validate instrument selection."""
        # Check forbidden
        for inst in instruments:
            if inst in self.forbidden_instruments:
                return False, 1.0

        # Check required (at least one)
        if self.required_instruments:
            has_required = any(inst in instruments for inst in self.required_instruments)
            if not has_required:
                return False, 1.0

        # Score preferred
        preferred_count = sum(1 for inst in instruments if inst in self.preferred_instruments)
        score = 1.0 - (preferred_count / max(1, len(self.preferred_instruments)))
        return True, score * 0.3


@dataclass
class StructureConstraint:
    """Song structure constraints."""
    min_sections: int
    max_sections: int
    required_section_types: list[str]
    min_length_bars: int
    max_length_bars: int
    priority: ConstraintPriority = ConstraintPriority.LOW

    def validate_sections(self, sections: list[str], total_bars: int) -> tuple[bool, float]:
        """Validate section structure."""
        if not (self.min_sections <= len(sections) <= self.max_sections):
            return False, 1.0
        if not (self.min_length_bars <= total_bars <= self.max_length_bars):
            return False, 1.0

        # Check required sections
        for req in self.required_section_types:
            if req not in sections:
                return False, 0.8

        return True, 0.0


@dataclass
class HardConstraints:
    """
    Hard constraints that MUST be satisfied.
    Violation of any hard constraint results in rejection.
    """
    tempo: TempoConstraint
    rhythm: RhythmConstraint
    key: KeyConstraint

    def validate(self, generation_params: dict) -> tuple[bool, list[str]]:
        """Validate all hard constraints. Returns (valid, list of violations)."""
        violations = []

        # Tempo check
        if "bpm" in generation_params:
            valid, _ = self.tempo.validate(generation_params["bpm"])
            if not valid:
                violations.append(f"Tempo {generation_params['bpm']} outside range [{self.tempo.min_bpm}, {self.tempo.max_bpm}]")

        # Mode check
        if "mode" in generation_params:
            valid, _ = self.key.validate(generation_params["mode"])
            if not valid:
                violations.append(f"Mode '{generation_params['mode']}' not allowed, use: {self.key.allowed_modes}")

        # Kick pattern check
        if "kick_pattern" in generation_params:
            valid, _ = self.rhythm.validate_kick(generation_params["kick_pattern"])
            if not valid:
                violations.append(f"Kick pattern not allowed for genre")

        return len(violations) == 0, violations


@dataclass
class SoftPreferences:
    """
    Soft preferences that guide generation.
    Deviations result in lower quality scores, not rejection.
    """
    harmony: HarmonyConstraint
    melody: MelodyConstraint
    instruments: InstrumentConstraint
    structure: StructureConstraint
    swing: SwingConstraint

    # Target values for scoring
    target_energy: float = 0.5          # 0-1 energy level
    target_brightness: float = 0.5      # 0-1 spectral brightness
    target_density: float = 0.5         # 0-1 overall density

    def score(self, generation_params: dict) -> float:
        """Score generation against soft preferences. Returns 0-1 (1 is perfect match)."""
        scores = []
        weights = []

        # Swing score
        if "swing" in generation_params:
            _, deviation = self.swing.validate(generation_params["swing"])
            scores.append(1.0 - deviation)
            weights.append(1.0)

        # Complexity score
        if "chord_complexity" in generation_params:
            _, deviation = self.harmony.validate_complexity(generation_params["chord_complexity"])
            scores.append(1.0 - deviation)
            weights.append(1.5)

        # Density score
        if "note_density" in generation_params:
            _, deviation = self.melody.validate_density(generation_params["note_density"])
            scores.append(1.0 - deviation)
            weights.append(1.0)

        # Instrument score
        if "instruments" in generation_params:
            _, deviation = self.instruments.validate_instruments(generation_params["instruments"])
            scores.append(1.0 - deviation)
            weights.append(2.0)

        if not scores:
            return 0.5  # Neutral score if no params to check

        # Weighted average
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


@dataclass
class RejectionCriteria:
    """
    Automatic rejection criteria based on genre DNA deviation.
    These are computed dynamically from DNA similarity scores.
    """
    # Maximum deviation from target DNA vector
    max_rhythm_deviation: float = 0.4
    max_harmony_deviation: float = 0.5
    max_melody_deviation: float = 0.5
    max_overall_deviation: float = 0.35

    # Minimum similarity to target genre
    min_genre_similarity: float = 0.6

    # Cross-genre contamination limits
    max_wrong_genre_similarity: float = 0.7  # If too similar to wrong genre

    def should_reject(
        self,
        target_genre: str,
        rhythm_deviation: float,
        harmony_deviation: float,
        melody_deviation: float,
        detected_genre: Optional[str] = None,
        genre_similarity: float = 1.0,
    ) -> tuple[bool, str]:
        """Check if output should be rejected."""

        # Check component deviations
        if rhythm_deviation > self.max_rhythm_deviation:
            return True, f"Rhythm deviation {rhythm_deviation:.2f} exceeds max {self.max_rhythm_deviation}"

        if harmony_deviation > self.max_harmony_deviation:
            return True, f"Harmony deviation {harmony_deviation:.2f} exceeds max {self.max_harmony_deviation}"

        if melody_deviation > self.max_melody_deviation:
            return True, f"Melody deviation {melody_deviation:.2f} exceeds max {self.max_melody_deviation}"

        # Check overall similarity
        if genre_similarity < self.min_genre_similarity:
            return True, f"Genre similarity {genre_similarity:.2f} below minimum {self.min_genre_similarity}"

        # Check for wrong genre
        if detected_genre and detected_genre != target_genre:
            return True, f"Detected as '{detected_genre}' instead of '{target_genre}'"

        return False, ""


@dataclass
class GenerationConstraints:
    """
    Complete constraint package for genre-conditioned generation.
    Produced by GenreConditioner from genre DNA.
    """
    genre_id: str
    hard: HardConstraints
    soft: SoftPreferences
    rejection: RejectionCriteria

    # Genre DNA for reference
    source_dna: GenreDNA

    # Blend weights if multiple genres
    blend_weights: dict[str, float] = field(default_factory=dict)

    # Mood modifiers
    mood_modifiers: dict[str, float] = field(default_factory=dict)

    def validate_and_score(self, generation_params: dict) -> tuple[bool, float, list[str]]:
        """
        Validate generation against all constraints.
        Returns: (is_valid, score, list of issues)
        """
        # Check hard constraints first
        hard_valid, violations = self.hard.validate(generation_params)
        if not hard_valid:
            return False, 0.0, violations

        # Score soft preferences
        score = self.soft.score(generation_params)

        return True, score, []


class GenreConditioner:
    """
    Translates user genre selection into generation constraints.

    The conditioner converts high-level genre DNA specifications into
    actionable constraints that guide the generation pipeline while
    preventing genre collapse.
    """

    def __init__(self):
        """Initialize the genre conditioner."""
        self._constraint_cache: dict[str, GenerationConstraints] = {}

    def condition(
        self,
        genre_id: str,
        mood_descriptors: Optional[list[str]] = None,
        blend_genres: Optional[list[tuple[str, float]]] = None,
    ) -> GenerationConstraints:
        """
        Generate constraints from genre selection.

        Args:
            genre_id: Primary genre identifier
            mood_descriptors: Optional mood modifiers (e.g., ["dark", "energetic"])
            blend_genres: Optional list of (genre_id, weight) for genre blending

        Returns:
            GenerationConstraints ready for the pipeline
        """
        # Get primary genre DNA
        dna = get_genre_dna(genre_id)
        if dna is None:
            raise ValueError(f"Unknown genre: {genre_id}")

        # Build constraints from DNA
        constraints = self._build_constraints(dna)

        # Apply mood modifiers
        if mood_descriptors:
            constraints = self._apply_mood_modifiers(constraints, mood_descriptors)

        # Handle genre blending
        if blend_genres:
            constraints = self._apply_genre_blend(constraints, blend_genres)

        return constraints

    def _build_constraints(self, dna: GenreDNA) -> GenerationConstraints:
        """Build constraint set from genre DNA."""

        # Tempo constraint with ±15% flexibility
        # RhythmDNA has tempo_center and tempo_variance
        tempo_min = dna.rhythm.tempo_center - dna.rhythm.tempo_variance
        tempo_max = dna.rhythm.tempo_center + dna.rhythm.tempo_variance
        tempo_flex = dna.rhythm.tempo_variance * 0.15

        tempo_constraint = TempoConstraint(
            min_bpm=tempo_min - tempo_flex,
            max_bpm=tempo_max + tempo_flex,
            target_bpm=dna.rhythm.tempo_center,
            priority=ConstraintPriority.HIGH,
        )

        # Swing constraint
        swing_constraint = SwingConstraint(
            min_swing=max(0.0, dna.rhythm.swing_amount - 0.1),
            max_swing=min(1.0, dna.rhythm.swing_amount + 0.1),
            target_swing=dna.rhythm.swing_amount,
            priority=ConstraintPriority.MEDIUM,
        )

        # Rhythm constraint
        rhythm_constraint = RhythmConstraint(
            allowed_kick_patterns=list(KickPattern),  # Allow all but prefer target
            allowed_snare_positions=list(SnarePosition),
            allowed_time_feels=[dna.rhythm.time_feel],  # Strict on feel
            preferred_kick=dna.rhythm.kick_pattern,
            preferred_snare=dna.rhythm.snare_position,
            preferred_feel=dna.rhythm.time_feel,
            priority=ConstraintPriority.HIGH,
        )

        # Key constraint
        key_constraint = KeyConstraint(
            allowed_modes=dna.harmony.primary_modes + dna.harmony.secondary_modes,
            preferred_modes=dna.harmony.primary_modes,
            priority=ConstraintPriority.MEDIUM,
        )

        # Hard constraints
        hard = HardConstraints(
            tempo=tempo_constraint,
            rhythm=rhythm_constraint,
            key=key_constraint,
        )

        # Harmony constraint
        # chord_complexity is a float 0-1, create a range around it
        complexity_variance = 0.2
        harmony_constraint = HarmonyConstraint(
            min_chord_complexity=max(0.0, dna.harmony.chord_complexity - complexity_variance),
            max_chord_complexity=min(1.0, dna.harmony.chord_complexity + complexity_variance),
            target_complexity=dna.harmony.chord_complexity,
            allowed_cadences=dna.harmony.cadence_preferences,
            # common_progressions is list[list[str]], flatten first items for required
            required_progressions=[p[0] for p in dna.harmony.common_progressions[:2] if p],
            priority=ConstraintPriority.MEDIUM,
        )

        # Melody constraint
        # note_density is a float, create a range around it
        density_variance = 0.3
        melody_constraint = MelodyConstraint(
            min_note_density=max(0.0, dna.melody.note_density - density_variance),
            max_note_density=min(2.0, dna.melody.note_density + density_variance),
            target_density=dna.melody.note_density,
            allowed_scales=dna.melody.scale_preferences,
            max_leap_interval=dna.melody.max_interval,
            priority=ConstraintPriority.MEDIUM,
        )

        # Instrument constraint
        instrument_constraint = InstrumentConstraint(
            required_instruments=dna.timbre.primary_instruments[:3],  # Require at least one of core
            forbidden_instruments=[],  # Could add genre-specific forbidden instruments
            preferred_instruments=dna.timbre.primary_instruments,
            priority=ConstraintPriority.HIGH,
        )

        # Structure constraint
        structure_constraint = StructureConstraint(
            min_sections=2,
            max_sections=len(dna.structure.section_types),
            required_section_types=dna.structure.section_types[:2],  # Require first section types
            min_length_bars=8,
            max_length_bars=dna.structure.typical_length_bars,
            priority=ConstraintPriority.LOW,
        )

        # Soft preferences
        soft = SoftPreferences(
            harmony=harmony_constraint,
            melody=melody_constraint,
            instruments=instrument_constraint,
            structure=structure_constraint,
            swing=swing_constraint,
            target_energy=self._compute_energy_target(dna),
            target_brightness=self._compute_brightness_target(dna),
            target_density=dna.melody.note_density,
        )

        # Rejection criteria (slightly relaxed for v0.1)
        rejection = RejectionCriteria(
            max_rhythm_deviation=0.45,
            max_harmony_deviation=0.55,
            max_melody_deviation=0.55,
            max_overall_deviation=0.40,
            min_genre_similarity=0.55,
        )

        return GenerationConstraints(
            genre_id=dna.genre_id,
            hard=hard,
            soft=soft,
            rejection=rejection,
            source_dna=dna,
        )

    def _compute_energy_target(self, dna: GenreDNA) -> float:
        """Compute target energy level from DNA."""
        # Based on tempo, density, and dynamic range
        tempo_factor = (dna.rhythm.tempo_range[1] - 60) / 140  # Normalize 60-200 BPM
        density_factor = dna.melody.note_density  # note_density is a float

        # Dynamic range affects energy perception
        from aether.genre.dna import DynamicRange
        dynamic_map = {
            DynamicRange.COMPRESSED: 0.8,
            DynamicRange.MODERATE: 0.5,
            DynamicRange.WIDE: 0.4,
            DynamicRange.CINEMATIC: 0.3,
        }
        dynamic_factor = dynamic_map.get(dna.structure.dynamic_range, 0.5)

        return min(1.0, (tempo_factor + density_factor + dynamic_factor) / 3)

    def _compute_brightness_target(self, dna: GenreDNA) -> float:
        """Compute target spectral brightness from DNA."""
        # Simplified brightness estimation
        freq_map = {
            "sub-heavy": 0.2,
            "bass-focused": 0.3,
            "balanced": 0.5,
            "mid-forward": 0.6,
            "bright": 0.7,
            "airy": 0.8,
        }
        return freq_map.get(dna.timbre.frequency_balance, 0.5)

    def _apply_mood_modifiers(
        self,
        constraints: GenerationConstraints,
        mood_descriptors: list[str],
    ) -> GenerationConstraints:
        """Apply mood-based modifications to constraints."""

        mood_effects = {
            "dark": {"brightness": -0.2, "mode_shift": ["minor", "phrygian", "locrian"]},
            "bright": {"brightness": 0.2, "mode_shift": ["major", "lydian", "mixolydian"]},
            "energetic": {"tempo_shift": 1.1, "density_shift": 0.1},
            "chill": {"tempo_shift": 0.9, "density_shift": -0.1},
            "aggressive": {"energy": 0.2, "density_shift": 0.15},
            "melancholic": {"brightness": -0.1, "tempo_shift": 0.95},
            "uplifting": {"brightness": 0.15, "tempo_shift": 1.05},
            "atmospheric": {"density_shift": -0.2, "reverb": 0.3},
        }

        for mood in mood_descriptors:
            mood_lower = mood.lower()
            if mood_lower in mood_effects:
                effects = mood_effects[mood_lower]
                constraints.mood_modifiers[mood_lower] = 1.0

                # Apply brightness shift
                if "brightness" in effects:
                    constraints.soft.target_brightness = max(0, min(1,
                        constraints.soft.target_brightness + effects["brightness"]))

                # Apply tempo shift
                if "tempo_shift" in effects:
                    factor = effects["tempo_shift"]
                    constraints.hard.tempo.min_bpm *= factor
                    constraints.hard.tempo.max_bpm *= factor
                    constraints.hard.tempo.target_bpm *= factor

                # Apply density shift
                if "density_shift" in effects:
                    shift = effects["density_shift"]
                    constraints.soft.target_density = max(0, min(1,
                        constraints.soft.target_density + shift))

                # Apply energy shift
                if "energy" in effects:
                    constraints.soft.target_energy = max(0, min(1,
                        constraints.soft.target_energy + effects["energy"]))

        return constraints

    def _apply_genre_blend(
        self,
        constraints: GenerationConstraints,
        blend_genres: list[tuple[str, float]],
    ) -> GenerationConstraints:
        """Apply genre blending to constraints."""

        # Normalize weights
        primary_weight = 1.0 - sum(w for _, w in blend_genres)
        if primary_weight < 0.5:
            # Primary genre should remain dominant
            scale = 0.5 / (1.0 - primary_weight)
            blend_genres = [(g, w * scale) for g, w in blend_genres]
            primary_weight = 0.5

        constraints.blend_weights[constraints.genre_id] = primary_weight

        for blend_genre, weight in blend_genres:
            blend_dna = get_genre_dna(blend_genre)
            if blend_dna is None:
                continue

            constraints.blend_weights[blend_genre] = weight

            # Expand allowed modes
            for mode in blend_dna.harmony.primary_modes:
                if mode not in constraints.hard.key.allowed_modes:
                    constraints.hard.key.allowed_modes.append(mode)

            # Expand tempo range
            constraints.hard.tempo.min_bpm = min(
                constraints.hard.tempo.min_bpm,
                blend_dna.rhythm.tempo_range[0]
            )
            constraints.hard.tempo.max_bpm = max(
                constraints.hard.tempo.max_bpm,
                blend_dna.rhythm.tempo_range[1]
            )

            # Blend target values
            constraints.soft.target_energy = (
                constraints.soft.target_energy * primary_weight +
                self._compute_energy_target(blend_dna) * weight
            )

            # Add preferred instruments from blend genre
            for inst in blend_dna.timbre.core_instruments:
                if inst not in constraints.soft.instruments.preferred_instruments:
                    constraints.soft.instruments.preferred_instruments.append(inst)

        # Relax rejection criteria for blends
        constraints.rejection.max_overall_deviation = 0.5
        constraints.rejection.min_genre_similarity = 0.45

        return constraints

    def get_generation_params(self, constraints: GenerationConstraints) -> dict:
        """
        Extract concrete generation parameters from constraints.
        These can be passed directly to the MIDI provider.
        """
        dna = constraints.source_dna

        return {
            # Tempo
            "bpm": constraints.hard.tempo.target_bpm,
            "tempo_range": (constraints.hard.tempo.min_bpm, constraints.hard.tempo.max_bpm),

            # Rhythm
            "swing": constraints.soft.swing.target_swing,
            "time_feel": constraints.hard.rhythm.preferred_feel.value,
            "kick_pattern": constraints.hard.rhythm.preferred_kick.value,
            "snare_position": constraints.hard.rhythm.preferred_snare.value,

            # Harmony
            "mode": constraints.hard.key.preferred_modes[0] if constraints.hard.key.preferred_modes else "minor",
            "chord_complexity": constraints.soft.harmony.target_complexity,
            "progressions": constraints.soft.harmony.required_progressions,

            # Melody
            "note_density": constraints.soft.melody.target_density,
            "scales": constraints.soft.melody.allowed_scales,
            "max_leap": constraints.soft.melody.max_leap_interval,

            # Instrumentation
            "instruments": constraints.soft.instruments.preferred_instruments,

            # Structure
            "sections": dna.structure.section_types,
            "bars_per_section": dna.structure.typical_length_bars // max(1, len(dna.structure.section_types)),

            # Targets
            "target_energy": constraints.soft.target_energy,
            "target_brightness": constraints.soft.target_brightness,

            # Metadata
            "genre_id": constraints.genre_id,
            "blend_weights": constraints.blend_weights,
            "mood_modifiers": constraints.mood_modifiers,
        }


# Convenience function
def condition_for_genre(
    genre_id: str,
    mood: Optional[list[str]] = None,
    blend: Optional[list[tuple[str, float]]] = None,
) -> GenerationConstraints:
    """Quick constraint generation for a genre."""
    conditioner = GenreConditioner()
    return conditioner.condition(genre_id, mood, blend)
