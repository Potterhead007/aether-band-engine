"""
AETHER Genre Authenticity Evaluator

Professional-grade authenticity scoring against genre rubrics.

Features:
- Multi-dimensional evaluation (rhythm, harmony, melody, production, arrangement)
- Genre-specific rubric scoring (1-5 scale)
- Weighted score aggregation
- Automated improvement suggestions
- Pass/fail determination against thresholds

Architecture:
- DimensionEvaluator: Base class for dimension-specific evaluation
- TempoGrooveEvaluator: BPM, swing, rhythmic feel
- HarmonicEvaluator: Mode, progressions, tension
- MelodicEvaluator: Intervals, contour, phrasing
- ProductionEvaluator: Mix characteristics, effects, vintage feel
- ArrangementEvaluator: Structure, energy curve, duration
- GenreAuthenticityEvaluator: Orchestrates full evaluation

This is institutional-grade quality assurance.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


class ScoreLevel(Enum):
    """Authenticity score levels (1-5 scale)."""

    EXCELLENT = 5  # Perfect genre authenticity
    GOOD = 4       # Strong adherence with minor deviations
    ACCEPTABLE = 3  # Meets minimum standards
    POOR = 2       # Noticeable issues
    FAILING = 1    # Wrong style entirely


@dataclass
class DimensionScore:
    """Score for a single rubric dimension."""

    dimension_name: str
    weight: float
    raw_score: float  # 1-5
    weighted_score: float  # raw_score * weight / 5.0
    criteria_met: List[str]
    criteria_failed: List[str]
    improvement_suggestions: List[str]
    confidence: float = 1.0  # How confident we are in this score

    @property
    def normalized_score(self) -> float:
        """Score normalized to 0-1 range."""
        return self.raw_score / 5.0

    @property
    def passes_threshold(self) -> bool:
        """Whether this dimension meets acceptable threshold (3+)."""
        return self.raw_score >= 3.0


@dataclass
class AuthenticityResult:
    """Complete authenticity evaluation result."""

    genre_id: str
    genre_name: str
    overall_score: float  # 0-1
    passing_threshold: float  # From genre rubric
    passed: bool
    dimension_scores: List[DimensionScore]
    top_strengths: List[str]
    top_weaknesses: List[str]
    improvement_priority: List[str]
    evaluation_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "genre_id": self.genre_id,
            "genre_name": self.genre_name,
            "overall_score": round(self.overall_score, 4),
            "passing_threshold": self.passing_threshold,
            "passed": self.passed,
            "dimension_scores": [
                {
                    "name": ds.dimension_name,
                    "weight": ds.weight,
                    "raw_score": ds.raw_score,
                    "weighted_score": round(ds.weighted_score, 4),
                    "criteria_met": ds.criteria_met,
                    "criteria_failed": ds.criteria_failed,
                    "suggestions": ds.improvement_suggestions,
                }
                for ds in self.dimension_scores
            ],
            "top_strengths": self.top_strengths,
            "top_weaknesses": self.top_weaknesses,
            "improvement_priority": self.improvement_priority,
            "notes": self.evaluation_notes,
        }


# ============================================================================
# Track Analysis Data (input to evaluator)
# ============================================================================


@dataclass
class RhythmAnalysis:
    """Analyzed rhythm characteristics of a track."""

    bpm: float
    time_signature: str = "4/4"
    swing_amount: float = 0.0  # 0-1
    feel: str = "straight"  # straight, swing, shuffle, triplet
    groove_pocket_deviation: float = 0.0  # How much notes deviate from grid (0-1)
    drum_pattern_type: str = ""  # e.g., "boom_bap", "four_on_floor"
    kick_characteristics: Dict[str, Any] = field(default_factory=dict)
    snare_characteristics: Dict[str, Any] = field(default_factory=dict)
    hihat_characteristics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HarmonyAnalysis:
    """Analyzed harmony characteristics of a track."""

    key_root: str  # e.g., "C", "F#"
    mode: str  # e.g., "minor", "dorian"
    detected_progressions: List[str]  # Roman numeral progressions
    jazz_chord_ratio: float = 0.0  # Ratio of 7th/9th/extended chords
    tension_level: float = 0.0  # 0-1
    modal_interchange_used: bool = False
    chord_vocabulary_complexity: float = 0.0  # 0-1


@dataclass
class MelodyAnalysis:
    """Analyzed melody characteristics of a track."""

    range_octaves: float
    intervals_used: List[str]  # e.g., ["M2", "m3", "P4"]
    contour_type: str  # ascending, descending, arch, wave, static
    average_phrase_length_bars: float
    note_density: float  # Notes per beat
    repetition_ratio: float = 0.0  # How much melody repeats


@dataclass
class ProductionAnalysis:
    """Analyzed production characteristics of a track."""

    # Mix characteristics (0-1)
    low_end_emphasis: float = 0.5
    brightness: float = 0.5
    stereo_width: float = 0.5
    vintage_warmth: float = 0.0

    # Loudness
    integrated_lufs: float = -14.0
    dynamic_range_lu: float = 8.0

    # Effects detected
    effects_detected: List[str] = field(default_factory=list)

    # Texture
    has_vinyl_texture: bool = False
    has_tape_saturation: bool = False
    has_lo_fi_filtering: bool = False
    has_gated_reverb: bool = False


@dataclass
class ArrangementAnalysis:
    """Analyzed arrangement characteristics of a track."""

    duration_seconds: float
    structure: str  # e.g., "intro-verse-chorus-verse-chorus-outro"
    section_types: List[str]
    energy_curve_type: str  # build, maintain, build_release, wave
    energy_peak_position: float = 0.5  # 0-1 position in track
    has_intro: bool = True
    has_outro: bool = True


@dataclass
class TrackAnalysis:
    """Complete analysis of a track for authenticity evaluation."""

    rhythm: RhythmAnalysis
    harmony: HarmonyAnalysis
    melody: MelodyAnalysis
    production: ProductionAnalysis
    arrangement: ArrangementAnalysis

    # Instrumentation
    instruments_detected: List[str] = field(default_factory=list)

    # Additional metadata
    genre_id: str = ""
    track_title: str = ""


# ============================================================================
# Dimension Evaluators
# ============================================================================


class DimensionEvaluator(ABC):
    """Base class for dimension-specific evaluation."""

    @abstractmethod
    def evaluate(
        self,
        analysis: TrackAnalysis,
        criteria: List[str],
        genre_profile: Any,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """
        Evaluate the dimension.

        Returns:
            Tuple of (score, criteria_met, criteria_failed, suggestions)
        """
        pass

    def _calculate_score(
        self,
        checks: List[Tuple[bool, str, str, str]],
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """
        Calculate score from a list of checks.

        Each check is: (passed, criterion, failure_reason, suggestion)
        """
        criteria_met = []
        criteria_failed = []
        suggestions = []

        for passed, criterion, failure_reason, suggestion in checks:
            if passed:
                criteria_met.append(criterion)
            else:
                criteria_failed.append(f"{criterion}: {failure_reason}")
                if suggestion:
                    suggestions.append(suggestion)

        if not checks:
            return 3.0, [], [], []

        pass_ratio = len(criteria_met) / len(checks)

        # Map ratio to 1-5 score
        if pass_ratio >= 0.9:
            score = 5.0
        elif pass_ratio >= 0.75:
            score = 4.0
        elif pass_ratio >= 0.5:
            score = 3.0
        elif pass_ratio >= 0.25:
            score = 2.0
        else:
            score = 1.0

        return score, criteria_met, criteria_failed, suggestions


class TempoGrooveEvaluator(DimensionEvaluator):
    """Evaluates tempo, swing, and groove authenticity."""

    def evaluate(
        self,
        analysis: TrackAnalysis,
        criteria: List[str],
        genre_profile: Any,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        checks = []
        rhythm = analysis.rhythm
        tempo_range = genre_profile.tempo
        rhythm_profile = genre_profile.rhythm

        # Check 1: BPM in range
        bpm_in_range = tempo_range.min_bpm <= rhythm.bpm <= tempo_range.max_bpm
        bpm_diff = 0
        if not bpm_in_range:
            if rhythm.bpm < tempo_range.min_bpm:
                bpm_diff = tempo_range.min_bpm - rhythm.bpm
            else:
                bpm_diff = rhythm.bpm - tempo_range.max_bpm

        checks.append((
            bpm_in_range,
            f"BPM within range ({tempo_range.min_bpm}-{tempo_range.max_bpm})",
            f"BPM {rhythm.bpm:.1f} is {bpm_diff:.1f} BPM outside range",
            f"Adjust tempo to {tempo_range.typical_bpm} BPM",
        ))

        # Check 2: Swing amount appropriate
        swing_in_range = (
            rhythm_profile.swing_amount_min <= rhythm.swing_amount
            <= rhythm_profile.swing_amount_max
        )
        checks.append((
            swing_in_range,
            f"Swing amount appropriate ({rhythm_profile.swing_amount_min}-{rhythm_profile.swing_amount_max})",
            f"Swing {rhythm.swing_amount:.2f} outside expected range",
            f"Set swing to approximately {rhythm_profile.swing_amount_typical:.2f}",
        ))

        # Check 3: Feel matches genre
        feel_matches = rhythm.feel in rhythm_profile.feels or rhythm.feel.replace("_", " ") in rhythm_profile.feels
        checks.append((
            feel_matches,
            f"Rhythmic feel matches genre",
            f"Feel '{rhythm.feel}' not typical (expected: {rhythm_profile.feels})",
            f"Use {rhythm_profile.feels[0]} feel for this genre",
        ))

        # Check 4: Time signature
        time_sig_matches = rhythm.time_signature in rhythm_profile.time_signatures
        checks.append((
            time_sig_matches,
            "Time signature appropriate",
            f"Time signature {rhythm.time_signature} uncommon (expected: {rhythm_profile.time_signatures})",
            f"Use {rhythm_profile.time_signatures[0]} time signature",
        ))

        # Check 5: Groove pocket (not over-quantized)
        has_pocket = rhythm.groove_pocket_deviation > 0.01 or rhythm.swing_amount > 0
        checks.append((
            has_pocket,
            "Has natural groove/pocket",
            "Rhythm sounds too quantized/mechanical",
            "Add subtle timing variations for human feel",
        ))

        return self._calculate_score(checks)


class HarmonicEvaluator(DimensionEvaluator):
    """Evaluates harmonic/chord progression authenticity."""

    def evaluate(
        self,
        analysis: TrackAnalysis,
        criteria: List[str],
        genre_profile: Any,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        checks = []
        harmony = analysis.harmony
        harmony_profile = genre_profile.harmony

        # Check 1: Mode appropriate
        mode_appropriate = harmony.mode.lower() in [m.value.lower() for m in harmony_profile.common_modes]
        checks.append((
            mode_appropriate,
            f"Mode appropriate for genre",
            f"Mode '{harmony.mode}' uncommon (expected: {[m.value for m in harmony_profile.common_modes]})",
            f"Consider using {harmony_profile.common_modes[0].value} mode",
        ))

        # Check 2: Progression fits style
        progression_matches = False
        matched_prog = None
        for prog in harmony.detected_progressions:
            prog_normalized = prog.lower().replace(" ", "")
            for expected in harmony_profile.typical_progressions:
                expected_normalized = expected.lower().replace(" ", "")
                if prog_normalized == expected_normalized or prog_normalized.startswith(expected_normalized[:6]):
                    progression_matches = True
                    matched_prog = prog
                    break
            if progression_matches:
                break

        checks.append((
            progression_matches,
            "Chord progression fits genre style",
            f"Progressions {harmony.detected_progressions[:2]} not typical",
            f"Try progressions like: {harmony_profile.typical_progressions[:2]}",
        ))

        # Check 3: Jazz influence level
        expected_jazz = harmony_profile.jazz_influence
        jazz_diff = abs(harmony.jazz_chord_ratio - expected_jazz)
        jazz_appropriate = jazz_diff < 0.3
        checks.append((
            jazz_appropriate,
            "Jazz influence level appropriate",
            f"Jazz chord ratio {harmony.jazz_chord_ratio:.2f} differs from expected {expected_jazz:.2f}",
            f"{'Add more extended chords' if harmony.jazz_chord_ratio < expected_jazz else 'Simplify chord voicings'}",
        ))

        # Check 4: Tension level
        expected_tension = harmony_profile.tension_level
        tension_diff = abs(harmony.tension_level - expected_tension)
        tension_appropriate = tension_diff < 0.25
        checks.append((
            tension_appropriate,
            "Harmonic tension level fits genre",
            f"Tension level {harmony.tension_level:.2f} differs from expected {expected_tension:.2f}",
            f"{'Increase harmonic tension' if harmony.tension_level < expected_tension else 'Reduce harmonic tension'}",
        ))

        return self._calculate_score(checks)


class MelodicEvaluator(DimensionEvaluator):
    """Evaluates melodic characteristics authenticity."""

    def evaluate(
        self,
        analysis: TrackAnalysis,
        criteria: List[str],
        genre_profile: Any,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        checks = []
        melody = analysis.melody
        melody_profile = genre_profile.melody

        # Check 1: Range appropriate
        range_diff = abs(melody.range_octaves - melody_profile.typical_range_octaves)
        range_ok = range_diff < 0.75
        checks.append((
            range_ok,
            f"Melodic range fits genre ({melody_profile.typical_range_octaves:.1f} octaves typical)",
            f"Range {melody.range_octaves:.1f} octaves differs from typical {melody_profile.typical_range_octaves:.1f}",
            f"{'Expand' if melody.range_octaves < melody_profile.typical_range_octaves else 'Constrain'} melodic range",
        ))

        # Check 2: Interval vocabulary
        expected_intervals = set(melody_profile.interval_vocabulary)
        used_intervals = set(melody.intervals_used)
        interval_overlap = len(expected_intervals & used_intervals) / max(len(expected_intervals), 1)
        intervals_ok = interval_overlap >= 0.4 or len(used_intervals) <= 3
        checks.append((
            intervals_ok,
            "Interval vocabulary matches genre",
            f"Intervals {list(used_intervals)[:5]} differ from typical {list(expected_intervals)[:5]}",
            "Adjust melodic intervals to match genre conventions",
        ))

        # Check 3: Contour preference
        contour_matches = melody.contour_type in melody_profile.contour_preferences
        checks.append((
            contour_matches,
            "Melodic contour fits genre style",
            f"Contour '{melody.contour_type}' uncommon (expected: {melody_profile.contour_preferences})",
            f"Reshape melody with {melody_profile.contour_preferences[0]} contour",
        ))

        # Check 4: Phrase length
        phrase_ok = any(
            abs(melody.average_phrase_length_bars - pl) < 1
            for pl in melody_profile.phrase_lengths
        )
        checks.append((
            phrase_ok,
            f"Phrase lengths appropriate ({melody_profile.phrase_lengths} bars typical)",
            f"Phrase length {melody.average_phrase_length_bars:.1f} bars unusual",
            f"Structure phrases in {melody_profile.phrase_lengths[0]}-bar units",
        ))

        return self._calculate_score(checks)


class ProductionEvaluator(DimensionEvaluator):
    """Evaluates production/mix characteristics authenticity."""

    def evaluate(
        self,
        analysis: TrackAnalysis,
        criteria: List[str],
        genre_profile: Any,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        checks = []
        production = analysis.production
        prod_profile = genre_profile.production
        mix_chars = prod_profile.mix_characteristics
        master_targets = prod_profile.master_targets

        # Check 1: Low-end emphasis
        low_end_diff = abs(production.low_end_emphasis - mix_chars.low_end_emphasis)
        low_end_ok = low_end_diff < 0.25
        checks.append((
            low_end_ok,
            "Low-end emphasis matches genre",
            f"Low-end {production.low_end_emphasis:.2f} differs from target {mix_chars.low_end_emphasis:.2f}",
            f"{'Boost' if production.low_end_emphasis < mix_chars.low_end_emphasis else 'Reduce'} low frequencies",
        ))

        # Check 2: Brightness
        brightness_diff = abs(production.brightness - mix_chars.brightness)
        brightness_ok = brightness_diff < 0.25
        checks.append((
            brightness_ok,
            "Brightness level appropriate",
            f"Brightness {production.brightness:.2f} differs from target {mix_chars.brightness:.2f}",
            f"{'Brighten' if production.brightness < mix_chars.brightness else 'Darken'} the mix",
        ))

        # Check 3: Stereo width
        width_diff = abs(production.stereo_width - mix_chars.width)
        width_ok = width_diff < 0.25
        checks.append((
            width_ok,
            "Stereo width fits genre",
            f"Width {production.stereo_width:.2f} differs from target {mix_chars.width:.2f}",
            f"{'Widen' if production.stereo_width < mix_chars.width else 'Narrow'} the stereo image",
        ))

        # Check 4: Vintage warmth
        warmth_diff = abs(production.vintage_warmth - mix_chars.vintage_warmth)
        warmth_ok = warmth_diff < 0.3
        checks.append((
            warmth_ok,
            "Vintage warmth level appropriate",
            f"Warmth {production.vintage_warmth:.2f} differs from target {mix_chars.vintage_warmth:.2f}",
            f"{'Add' if production.vintage_warmth < mix_chars.vintage_warmth else 'Reduce'} analog saturation/warmth",
        ))

        # Check 5: Signature effects present
        expected_effects = set(e.lower() for e in prod_profile.signature_effects)
        detected_effects = set(e.lower() for e in production.effects_detected)
        effect_overlap = len(expected_effects & detected_effects)
        effects_ok = effect_overlap >= min(2, len(expected_effects)) or len(expected_effects) == 0
        checks.append((
            effects_ok,
            "Signature effects present",
            f"Missing expected effects: {list(expected_effects - detected_effects)[:3]}",
            f"Add characteristic effects: {prod_profile.signature_effects[:2]}",
        ))

        # Check 6: Loudness in genre range
        lufs_ok = master_targets.loudness_lufs_min <= production.integrated_lufs <= master_targets.loudness_lufs_max
        checks.append((
            lufs_ok,
            f"Loudness in genre range ({master_targets.loudness_lufs_min} to {master_targets.loudness_lufs_max} LUFS)",
            f"Loudness {production.integrated_lufs:.1f} LUFS outside range",
            f"Target {(master_targets.loudness_lufs_min + master_targets.loudness_lufs_max) / 2:.1f} LUFS",
        ))

        # Check 7: Dynamic range
        dr_ok = master_targets.dynamic_range_lu_min <= production.dynamic_range_lu <= master_targets.dynamic_range_lu_max
        checks.append((
            dr_ok,
            f"Dynamic range appropriate ({master_targets.dynamic_range_lu_min}-{master_targets.dynamic_range_lu_max} LU)",
            f"DR {production.dynamic_range_lu:.1f} LU outside range",
            f"Target {(master_targets.dynamic_range_lu_min + master_targets.dynamic_range_lu_max) / 2:.1f} LU",
        ))

        return self._calculate_score(checks)


class ArrangementEvaluator(DimensionEvaluator):
    """Evaluates arrangement and structure authenticity."""

    def evaluate(
        self,
        analysis: TrackAnalysis,
        criteria: List[str],
        genre_profile: Any,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        checks = []
        arrangement = analysis.arrangement
        arr_profile = genre_profile.arrangement

        # Check 1: Duration in range
        duration_ok = arr_profile.typical_duration.min_seconds <= arrangement.duration_seconds <= arr_profile.typical_duration.max_seconds
        checks.append((
            duration_ok,
            f"Duration in genre range ({arr_profile.typical_duration.min_seconds}-{arr_profile.typical_duration.max_seconds}s)",
            f"Duration {arrangement.duration_seconds:.0f}s outside typical range",
            f"Target {(arr_profile.typical_duration.min_seconds + arr_profile.typical_duration.max_seconds) // 2}s",
        ))

        # Check 2: Structure matches
        structure_normalized = arrangement.structure.lower().replace(" ", "-")
        structure_matches = any(
            self._structure_similarity(structure_normalized, expected.lower().replace(" ", "-")) > 0.5
            for expected in arr_profile.common_structures
        )
        checks.append((
            structure_matches,
            "Song structure fits genre conventions",
            f"Structure '{arrangement.structure}' unusual for genre",
            f"Consider structure like: {arr_profile.common_structures[0]}",
        ))

        # Check 3: Energy curve type
        energy_matches = arrangement.energy_curve_type == arr_profile.energy_curve_type
        checks.append((
            energy_matches,
            f"Energy curve matches genre ({arr_profile.energy_curve_type})",
            f"Energy curve '{arrangement.energy_curve_type}' differs from expected",
            f"Shape dynamics with {arr_profile.energy_curve_type} energy curve",
        ))

        # Check 4: Has proper intro/outro
        has_bookends = arrangement.has_intro and arrangement.has_outro
        checks.append((
            has_bookends,
            "Has proper intro and outro",
            "Missing intro or outro section",
            "Add intro and outro sections",
        ))

        return self._calculate_score(checks)

    def _structure_similarity(self, s1: str, s2: str) -> float:
        """Calculate structural similarity between two song structures."""
        parts1 = set(s1.split("-"))
        parts2 = set(s2.split("-"))
        if not parts1 or not parts2:
            return 0.0
        overlap = len(parts1 & parts2)
        return overlap / max(len(parts1), len(parts2))


class InstrumentationEvaluator(DimensionEvaluator):
    """Evaluates instrumentation authenticity."""

    def evaluate(
        self,
        analysis: TrackAnalysis,
        criteria: List[str],
        genre_profile: Any,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        checks = []
        instruments = set(i.lower() for i in analysis.instruments_detected)
        instr_profile = genre_profile.instrumentation

        essential = set(i.lower() for i in instr_profile.essential)
        common = set(i.lower() for i in instr_profile.common)
        forbidden = set(i.lower() for i in instr_profile.forbidden)

        # Check 1: Essential instruments present
        essential_present = sum(1 for e in essential if any(e in inst or inst in e for inst in instruments))
        essential_ok = essential_present >= len(essential) * 0.6
        checks.append((
            essential_ok,
            "Essential instruments present",
            f"Missing essential: {list(essential - instruments)[:3]}",
            f"Add essential instruments: {list(essential)[:3]}",
        ))

        # Check 2: No forbidden instruments
        forbidden_found = [f for f in forbidden if any(f in inst or inst in f for inst in instruments)]
        no_forbidden = len(forbidden_found) == 0
        checks.append((
            no_forbidden,
            "No anachronistic/forbidden instruments",
            f"Contains inappropriate: {forbidden_found[:3]}",
            f"Remove: {forbidden_found[:2]}" if forbidden_found else "",
        ))

        # Check 3: Has common instruments
        common_present = sum(1 for c in common if any(c in inst or inst in c for inst in instruments))
        common_ok = common_present >= 1 or len(common) == 0
        checks.append((
            common_ok,
            "Uses common genre instruments",
            "Missing common genre instruments",
            f"Consider adding: {list(common)[:2]}",
        ))

        return self._calculate_score(checks)


# ============================================================================
# Main Evaluator
# ============================================================================


class GenreAuthenticityEvaluator:
    """
    Professional-grade genre authenticity evaluation.

    Evaluates tracks against genre-specific rubrics using multiple
    dimension evaluators, calculates weighted scores, and provides
    actionable improvement suggestions.

    Example:
        evaluator = GenreAuthenticityEvaluator()
        result = evaluator.evaluate(track_analysis, genre_profile)
        if result.passed:
            print(f"Track passes with {result.overall_score:.0%}")
        else:
            print("Improvements needed:")
            for item in result.improvement_priority:
                print(f"  - {item}")
    """

    def __init__(self):
        """Initialize evaluator with dimension evaluators."""
        self._evaluators: Dict[str, DimensionEvaluator] = {
            "tempo": TempoGrooveEvaluator(),
            "groove": TempoGrooveEvaluator(),
            "rhythm": TempoGrooveEvaluator(),
            "drum": TempoGrooveEvaluator(),
            "harmony": HarmonicEvaluator(),
            "harmonic": HarmonicEvaluator(),
            "chord": HarmonicEvaluator(),
            "melody": MelodicEvaluator(),
            "melodic": MelodicEvaluator(),
            "production": ProductionEvaluator(),
            "mix": ProductionEvaluator(),
            "master": ProductionEvaluator(),
            "synth": ProductionEvaluator(),
            "aesthetic": ProductionEvaluator(),
            "sound": ProductionEvaluator(),
            "arrangement": ArrangementEvaluator(),
            "structure": ArrangementEvaluator(),
            "mood": ProductionEvaluator(),
            "atmosphere": ProductionEvaluator(),
            "instrument": InstrumentationEvaluator(),
            "sample": ProductionEvaluator(),
            "lo-fi": ProductionEvaluator(),
            "lofi": ProductionEvaluator(),
            "jazz": HarmonicEvaluator(),
        }

    def _get_evaluator_for_dimension(self, dimension_name: str) -> DimensionEvaluator:
        """Get appropriate evaluator for a dimension name."""
        name_lower = dimension_name.lower()

        for key, evaluator in self._evaluators.items():
            if key in name_lower:
                return evaluator

        # Default to production evaluator
        return ProductionEvaluator()

    def evaluate(
        self,
        analysis: TrackAnalysis,
        genre_profile: Any,
    ) -> AuthenticityResult:
        """
        Evaluate track authenticity against genre rubric.

        Args:
            analysis: Complete track analysis
            genre_profile: GenreRootProfile with authenticity rubric

        Returns:
            AuthenticityResult with scores and suggestions
        """
        rubric = genre_profile.authenticity_rubric
        dimension_scores: List[DimensionScore] = []

        # Evaluate each rubric dimension
        for dimension in rubric.dimensions:
            evaluator = self._get_evaluator_for_dimension(dimension.name)

            raw_score, criteria_met, criteria_failed, suggestions = evaluator.evaluate(
                analysis,
                dimension.criteria,
                genre_profile,
            )

            # Apply dimension-specific criteria evaluation
            criteria_check_score = self._evaluate_specific_criteria(
                dimension.criteria,
                dimension.name,
                analysis,
                genre_profile,
            )

            # Blend evaluator score with criteria check
            blended_score = (raw_score * 0.7) + (criteria_check_score * 0.3)

            weighted_score = (blended_score / 5.0) * dimension.weight

            dimension_scores.append(DimensionScore(
                dimension_name=dimension.name,
                weight=dimension.weight,
                raw_score=blended_score,
                weighted_score=weighted_score,
                criteria_met=criteria_met,
                criteria_failed=criteria_failed,
                improvement_suggestions=suggestions,
            ))

        # Calculate overall score (sum of weighted scores)
        overall_score = sum(ds.weighted_score for ds in dimension_scores)

        # Determine pass/fail
        passed = overall_score >= rubric.minimum_passing_score

        # Identify strengths and weaknesses
        sorted_by_score = sorted(dimension_scores, key=lambda d: d.raw_score, reverse=True)
        top_strengths = [
            f"{d.dimension_name}: {d.raw_score:.1f}/5"
            for d in sorted_by_score if d.raw_score >= 4.0
        ][:3]

        top_weaknesses = [
            f"{d.dimension_name}: {d.raw_score:.1f}/5"
            for d in sorted_by_score if d.raw_score < 3.0
        ][:3]

        # Prioritize improvements by impact (weight * deficit)
        improvement_priority = []
        for ds in dimension_scores:
            if ds.raw_score < 4.0:
                deficit = 5.0 - ds.raw_score
                impact = deficit * ds.weight
                for suggestion in ds.improvement_suggestions[:2]:
                    improvement_priority.append((impact, suggestion))

        improvement_priority.sort(key=lambda x: -x[0])
        priority_list = [item[1] for item in improvement_priority[:5]]

        # Generate evaluation notes
        notes = []
        if passed:
            notes.append(f"Track passes genre authenticity threshold ({rubric.minimum_passing_score:.0%})")
        else:
            deficit = rubric.minimum_passing_score - overall_score
            notes.append(f"Track needs {deficit:.0%} improvement to pass threshold")

        notes.append(f"Evaluated against {len(dimension_scores)} dimensions")

        return AuthenticityResult(
            genre_id=genre_profile.genre_id,
            genre_name=genre_profile.name,
            overall_score=overall_score,
            passing_threshold=rubric.minimum_passing_score,
            passed=passed,
            dimension_scores=dimension_scores,
            top_strengths=top_strengths,
            top_weaknesses=top_weaknesses,
            improvement_priority=priority_list,
            evaluation_notes=notes,
        )

    def _evaluate_specific_criteria(
        self,
        criteria: List[str],
        dimension_name: str,
        analysis: TrackAnalysis,
        genre_profile: Any,
    ) -> float:
        """Evaluate specific criteria strings against analysis."""
        if not criteria:
            return 3.0

        passed = 0
        total = len(criteria)

        for criterion in criteria:
            criterion_lower = criterion.lower()

            # Match criteria to analysis features
            if self._criterion_matches(criterion_lower, analysis, genre_profile):
                passed += 1

        ratio = passed / total if total > 0 else 0.5

        # Map to 1-5 scale
        if ratio >= 0.8:
            return 5.0
        elif ratio >= 0.6:
            return 4.0
        elif ratio >= 0.4:
            return 3.0
        elif ratio >= 0.2:
            return 2.0
        else:
            return 1.0

    def _criterion_matches(
        self,
        criterion: str,
        analysis: TrackAnalysis,
        genre_profile: Any,
    ) -> bool:
        """Check if a specific criterion is met."""
        # Tempo/BPM checks
        if "bpm" in criterion or "tempo" in criterion:
            tempo = genre_profile.tempo
            if analysis.rhythm.bpm >= tempo.min_bpm and analysis.rhythm.bpm <= tempo.max_bpm:
                return True
            return False

        # Swing/groove checks
        if "swing" in criterion or "groove" in criterion or "pocket" in criterion:
            return analysis.rhythm.swing_amount > 0 or analysis.rhythm.groove_pocket_deviation > 0

        # Lo-fi/vintage checks
        if "vinyl" in criterion or "lo-fi" in criterion or "lofi" in criterion or "dust" in criterion:
            return analysis.production.has_vinyl_texture or analysis.production.vintage_warmth > 0.5

        # Filter/warmth checks
        if "warm" in criterion or "filter" in criterion or "muffled" in criterion:
            return analysis.production.vintage_warmth > 0.4 or analysis.production.has_lo_fi_filtering

        # Gated reverb checks
        if "gated" in criterion:
            return analysis.production.has_gated_reverb

        # Jazz checks
        if "jazz" in criterion or "soul" in criterion:
            return analysis.harmony.jazz_chord_ratio > 0.3

        # Dynamic checks
        if "dynamic" in criterion:
            return analysis.production.dynamic_range_lu > 6.0

        # Stereo/width checks
        if "wide" in criterion or "stereo" in criterion:
            return analysis.production.stereo_width > 0.5

        # Energy checks
        if "energy" in criterion or "calm" in criterion or "relax" in criterion:
            # Context dependent - check energy curve
            return True  # Assume pass if not analyzable

        # Default: assume met (conservative)
        return True


# ============================================================================
# Convenience Functions
# ============================================================================


def evaluate_genre_authenticity(
    analysis: TrackAnalysis,
    genre_profile: Any,
) -> AuthenticityResult:
    """
    Evaluate track genre authenticity.

    Convenience function for quick evaluation.

    Args:
        analysis: Complete track analysis
        genre_profile: GenreRootProfile to evaluate against

    Returns:
        AuthenticityResult with scores and suggestions
    """
    evaluator = GenreAuthenticityEvaluator()
    return evaluator.evaluate(analysis, genre_profile)


def create_track_analysis_from_artifacts(
    rhythm_data: Dict[str, Any],
    harmony_data: Dict[str, Any],
    melody_data: Dict[str, Any],
    production_data: Dict[str, Any],
    arrangement_data: Dict[str, Any],
    instruments: List[str] = None,
    genre_id: str = "",
    title: str = "",
) -> TrackAnalysis:
    """
    Create TrackAnalysis from pipeline artifacts.

    Helper function to construct analysis from typical pipeline outputs.
    """
    return TrackAnalysis(
        rhythm=RhythmAnalysis(
            bpm=rhythm_data.get("bpm", 120),
            time_signature=rhythm_data.get("time_signature", "4/4"),
            swing_amount=rhythm_data.get("swing_amount", 0.0),
            feel=rhythm_data.get("feel", "straight"),
            groove_pocket_deviation=rhythm_data.get("pocket_deviation", 0.0),
            drum_pattern_type=rhythm_data.get("pattern_type", ""),
        ),
        harmony=HarmonyAnalysis(
            key_root=harmony_data.get("key_root", "C"),
            mode=harmony_data.get("mode", "minor"),
            detected_progressions=harmony_data.get("progressions", []),
            jazz_chord_ratio=harmony_data.get("jazz_ratio", 0.0),
            tension_level=harmony_data.get("tension", 0.5),
        ),
        melody=MelodyAnalysis(
            range_octaves=melody_data.get("range_octaves", 1.5),
            intervals_used=melody_data.get("intervals", []),
            contour_type=melody_data.get("contour", "wave"),
            average_phrase_length_bars=melody_data.get("phrase_length", 4),
            note_density=melody_data.get("note_density", 0.5),
        ),
        production=ProductionAnalysis(
            low_end_emphasis=production_data.get("low_end", 0.5),
            brightness=production_data.get("brightness", 0.5),
            stereo_width=production_data.get("width", 0.5),
            vintage_warmth=production_data.get("warmth", 0.0),
            integrated_lufs=production_data.get("lufs", -14.0),
            dynamic_range_lu=production_data.get("dynamic_range", 8.0),
            effects_detected=production_data.get("effects", []),
            has_vinyl_texture=production_data.get("has_vinyl", False),
            has_tape_saturation=production_data.get("has_tape", False),
            has_lo_fi_filtering=production_data.get("has_lofi", False),
            has_gated_reverb=production_data.get("has_gated_reverb", False),
        ),
        arrangement=ArrangementAnalysis(
            duration_seconds=arrangement_data.get("duration", 180),
            structure=arrangement_data.get("structure", "intro-verse-chorus-verse-outro"),
            section_types=arrangement_data.get("sections", ["intro", "verse", "chorus", "outro"]),
            energy_curve_type=arrangement_data.get("energy_curve", "maintain"),
        ),
        instruments_detected=instruments or [],
        genre_id=genre_id,
        track_title=title,
    )


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Results
    "ScoreLevel",
    "DimensionScore",
    "AuthenticityResult",
    # Analysis data
    "RhythmAnalysis",
    "HarmonyAnalysis",
    "MelodyAnalysis",
    "ProductionAnalysis",
    "ArrangementAnalysis",
    "TrackAnalysis",
    # Evaluators
    "DimensionEvaluator",
    "TempoGrooveEvaluator",
    "HarmonicEvaluator",
    "MelodicEvaluator",
    "ProductionEvaluator",
    "ArrangementEvaluator",
    "InstrumentationEvaluator",
    "GenreAuthenticityEvaluator",
    # Functions
    "evaluate_genre_authenticity",
    "create_track_analysis_from_artifacts",
]
