"""
Vocal Quality Metrics

Automated metrics for evaluating synthesized vocal quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class MetricCategory(Enum):
    """Categories of quality metrics."""
    PITCH = "pitch"
    TIMING = "timing"
    PHONETIC = "phonetic"
    TIMBRE = "timbre"
    EXPRESSION = "expression"
    TECHNICAL = "technical"


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    name: str
    category: MetricCategory
    value: float  # Raw metric value
    score: float  # Normalized 0-100 score
    threshold: float  # Passing threshold
    passed: bool
    details: Optional[str] = None


@dataclass
class MetricSummary:
    """Summary of all metric evaluations."""
    overall_score: float
    category_scores: Dict[MetricCategory, float]
    metrics: List[MetricResult]
    passed: bool
    failure_reasons: List[str]


class PitchMetrics:
    """
    Pitch accuracy and expression metrics.
    """

    def __init__(self):
        """Initialize pitch metrics."""
        pass

    def calculate_pitch_accuracy(
        self,
        generated_pitch: np.ndarray,
        target_pitch: np.ndarray,
        tolerance_cents: float = 50.0,
    ) -> MetricResult:
        """
        Calculate pitch accuracy.

        Args:
            generated_pitch: Generated pitch contour (Hz)
            target_pitch: Target pitch contour (Hz)
            tolerance_cents: Acceptable deviation in cents

        Returns:
            MetricResult with accuracy score
        """
        if len(generated_pitch) == 0 or len(target_pitch) == 0:
            return MetricResult(
                name="pitch_accuracy",
                category=MetricCategory.PITCH,
                value=0,
                score=0,
                threshold=80,
                passed=False,
                details="Empty pitch data",
            )

        # Resample if different lengths
        if len(generated_pitch) != len(target_pitch):
            target_pitch = np.interp(
                np.linspace(0, 1, len(generated_pitch)),
                np.linspace(0, 1, len(target_pitch)),
                target_pitch,
            )

        # Calculate cents deviation
        cents_deviation = self._hz_to_cents_deviation(generated_pitch, target_pitch)

        # Calculate percentage within tolerance
        within_tolerance = np.abs(cents_deviation) <= tolerance_cents
        accuracy = np.mean(within_tolerance) * 100

        return MetricResult(
            name="pitch_accuracy",
            category=MetricCategory.PITCH,
            value=accuracy,
            score=accuracy,
            threshold=80,
            passed=accuracy >= 80,
            details=f"Mean deviation: {np.mean(np.abs(cents_deviation)):.1f} cents",
        )

    def calculate_pitch_stability(
        self,
        pitch_contour: np.ndarray,
        window_ms: float = 100,
        sample_rate: float = 100,  # Frames per second
    ) -> MetricResult:
        """
        Calculate pitch stability (lack of unwanted wobble).

        Args:
            pitch_contour: Pitch contour (Hz)
            window_ms: Analysis window
            sample_rate: Contour sample rate

        Returns:
            MetricResult with stability score
        """
        if len(pitch_contour) < 10:
            return MetricResult(
                name="pitch_stability",
                category=MetricCategory.PITCH,
                value=0,
                score=0,
                threshold=70,
                passed=False,
                details="Insufficient data",
            )

        # Calculate local variance
        window_samples = max(3, int(window_ms * sample_rate / 1000))

        variances = []
        for i in range(0, len(pitch_contour) - window_samples, window_samples // 2):
            window = pitch_contour[i:i + window_samples]
            if np.mean(window) > 0:
                # Convert to cents variance
                cents_var = np.var(1200 * np.log2(window / np.mean(window) + 1e-10))
                variances.append(cents_var)

        if not variances:
            return MetricResult(
                name="pitch_stability",
                category=MetricCategory.PITCH,
                value=0,
                score=50,
                threshold=70,
                passed=False,
            )

        avg_variance = np.mean(variances)
        # Convert variance to stability score (lower variance = higher score)
        stability_score = max(0, 100 - avg_variance)

        return MetricResult(
            name="pitch_stability",
            category=MetricCategory.PITCH,
            value=avg_variance,
            score=stability_score,
            threshold=70,
            passed=stability_score >= 70,
            details=f"Avg cents variance: {avg_variance:.1f}",
        )

    def calculate_vibrato_quality(
        self,
        pitch_contour: np.ndarray,
        expected_rate: float = 5.5,
        expected_depth: float = 40,
        sample_rate: float = 100,
    ) -> MetricResult:
        """
        Evaluate vibrato quality.

        Args:
            pitch_contour: Pitch contour (Hz)
            expected_rate: Expected vibrato rate (Hz)
            expected_depth: Expected depth (cents)
            sample_rate: Contour sample rate

        Returns:
            MetricResult with vibrato quality score
        """
        if len(pitch_contour) < 50:
            return MetricResult(
                name="vibrato_quality",
                category=MetricCategory.PITCH,
                value=0,
                score=50,
                threshold=60,
                passed=False,
                details="Insufficient data for vibrato analysis",
            )

        # Convert to cents from mean
        mean_pitch = np.mean(pitch_contour)
        cents = 1200 * np.log2(pitch_contour / mean_pitch + 1e-10)

        # FFT to find vibrato rate
        fft = np.abs(np.fft.rfft(cents - np.mean(cents)))
        freqs = np.fft.rfftfreq(len(cents), 1 / sample_rate)

        # Find peak in vibrato range (4-8 Hz)
        vibrato_mask = (freqs >= 4) & (freqs <= 8)
        if not np.any(vibrato_mask):
            return MetricResult(
                name="vibrato_quality",
                category=MetricCategory.PITCH,
                value=0,
                score=50,
                threshold=60,
                passed=False,
            )

        vibrato_fft = fft.copy()
        vibrato_fft[~vibrato_mask] = 0

        peak_idx = np.argmax(vibrato_fft)
        detected_rate = freqs[peak_idx]
        detected_depth = np.std(cents) * 2  # Approximate peak-to-peak

        # Score based on closeness to expected
        rate_score = max(0, 100 - abs(detected_rate - expected_rate) * 20)
        depth_score = max(0, 100 - abs(detected_depth - expected_depth))

        overall_score = (rate_score + depth_score) / 2

        return MetricResult(
            name="vibrato_quality",
            category=MetricCategory.PITCH,
            value=overall_score,
            score=overall_score,
            threshold=60,
            passed=overall_score >= 60,
            details=f"Rate: {detected_rate:.1f}Hz, Depth: {detected_depth:.0f}cents",
        )

    def _hz_to_cents_deviation(
        self,
        generated: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Calculate cents deviation between pitch arrays."""
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = generated / (target + 1e-10)
            cents = 1200 * np.log2(ratio + 1e-10)
        return np.nan_to_num(cents, nan=0, posinf=0, neginf=0)


class TimingMetrics:
    """
    Timing accuracy and rhythm metrics.
    """

    def __init__(self):
        """Initialize timing metrics."""
        pass

    def calculate_timing_accuracy(
        self,
        generated_onsets: List[float],
        target_onsets: List[float],
        tolerance_ms: float = 50.0,
    ) -> MetricResult:
        """
        Calculate onset timing accuracy.

        Args:
            generated_onsets: Generated onset times (ms)
            target_onsets: Target onset times (ms)
            tolerance_ms: Acceptable deviation

        Returns:
            MetricResult with timing accuracy
        """
        if not generated_onsets or not target_onsets:
            return MetricResult(
                name="timing_accuracy",
                category=MetricCategory.TIMING,
                value=0,
                score=0,
                threshold=80,
                passed=False,
                details="No onset data",
            )

        # Match onsets
        matched = 0
        deviations = []

        for target in target_onsets:
            # Find closest generated onset
            closest_idx = np.argmin([abs(g - target) for g in generated_onsets])
            closest = generated_onsets[closest_idx]
            deviation = abs(closest - target)

            if deviation <= tolerance_ms:
                matched += 1
            deviations.append(deviation)

        accuracy = (matched / len(target_onsets)) * 100
        mean_deviation = np.mean(deviations)

        return MetricResult(
            name="timing_accuracy",
            category=MetricCategory.TIMING,
            value=accuracy,
            score=accuracy,
            threshold=80,
            passed=accuracy >= 80,
            details=f"Mean deviation: {mean_deviation:.1f}ms",
        )

    def calculate_rhythm_consistency(
        self,
        inter_onset_intervals: List[float],
        expected_intervals: List[float],
    ) -> MetricResult:
        """
        Calculate rhythm consistency.

        Args:
            inter_onset_intervals: Generated IOIs
            expected_intervals: Expected IOIs

        Returns:
            MetricResult with rhythm consistency
        """
        if not inter_onset_intervals or not expected_intervals:
            return MetricResult(
                name="rhythm_consistency",
                category=MetricCategory.TIMING,
                value=0,
                score=50,
                threshold=70,
                passed=False,
            )

        # Normalize to compare patterns
        gen_normalized = np.array(inter_onset_intervals) / np.mean(inter_onset_intervals)
        exp_normalized = np.array(expected_intervals) / np.mean(expected_intervals)

        # Resample if needed
        if len(gen_normalized) != len(exp_normalized):
            exp_normalized = np.interp(
                np.linspace(0, 1, len(gen_normalized)),
                np.linspace(0, 1, len(exp_normalized)),
                exp_normalized,
            )

        # Calculate correlation
        correlation = np.corrcoef(gen_normalized, exp_normalized)[0, 1]
        score = max(0, correlation * 100)

        return MetricResult(
            name="rhythm_consistency",
            category=MetricCategory.TIMING,
            value=correlation,
            score=score,
            threshold=70,
            passed=score >= 70,
            details=f"Pattern correlation: {correlation:.2f}",
        )


class PhoneticMetrics:
    """
    Phonetic accuracy and intelligibility metrics.
    """

    def __init__(self):
        """Initialize phonetic metrics."""
        pass

    def calculate_phoneme_accuracy(
        self,
        generated_phonemes: List[str],
        target_phonemes: List[str],
    ) -> MetricResult:
        """
        Calculate phoneme sequence accuracy.

        Args:
            generated_phonemes: Generated phoneme sequence
            target_phonemes: Target phoneme sequence

        Returns:
            MetricResult with phoneme accuracy
        """
        if not target_phonemes:
            return MetricResult(
                name="phoneme_accuracy",
                category=MetricCategory.PHONETIC,
                value=0,
                score=0,
                threshold=85,
                passed=False,
            )

        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(generated_phonemes, target_phonemes)
        max_len = max(len(generated_phonemes), len(target_phonemes))

        accuracy = (1 - distance / max_len) * 100 if max_len > 0 else 0

        return MetricResult(
            name="phoneme_accuracy",
            category=MetricCategory.PHONETIC,
            value=accuracy,
            score=accuracy,
            threshold=85,
            passed=accuracy >= 85,
            details=f"Edit distance: {distance}",
        )

    def calculate_formant_accuracy(
        self,
        generated_formants: List[Tuple[float, float]],  # (F1, F2) pairs
        target_formants: List[Tuple[float, float]],
        tolerance_hz: float = 100,
    ) -> MetricResult:
        """
        Calculate formant accuracy for vowels.

        Args:
            generated_formants: Generated F1/F2 values
            target_formants: Target F1/F2 values
            tolerance_hz: Acceptable deviation

        Returns:
            MetricResult with formant accuracy
        """
        if not generated_formants or not target_formants:
            return MetricResult(
                name="formant_accuracy",
                category=MetricCategory.PHONETIC,
                value=0,
                score=50,
                threshold=70,
                passed=False,
            )

        # Match by position
        min_len = min(len(generated_formants), len(target_formants))
        matches = 0

        for i in range(min_len):
            gen_f1, gen_f2 = generated_formants[i]
            tgt_f1, tgt_f2 = target_formants[i]

            f1_close = abs(gen_f1 - tgt_f1) <= tolerance_hz
            f2_close = abs(gen_f2 - tgt_f2) <= tolerance_hz

            if f1_close and f2_close:
                matches += 1

        accuracy = (matches / min_len) * 100 if min_len > 0 else 0

        return MetricResult(
            name="formant_accuracy",
            category=MetricCategory.PHONETIC,
            value=accuracy,
            score=accuracy,
            threshold=70,
            passed=accuracy >= 70,
            details=f"Matched {matches}/{min_len} vowels",
        )

    def _levenshtein_distance(self, s1: List, s2: List) -> int:
        """Calculate Levenshtein distance between sequences."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


class TimbreMetrics:
    """
    Timbre and voice quality metrics.
    """

    def __init__(self):
        """Initialize timbre metrics."""
        pass

    def calculate_timbre_consistency(
        self,
        audio_segments: List[np.ndarray],
        sample_rate: int = 48000,
    ) -> MetricResult:
        """
        Calculate timbre consistency across segments.

        Args:
            audio_segments: Audio segments to analyze
            sample_rate: Audio sample rate

        Returns:
            MetricResult with consistency score
        """
        if len(audio_segments) < 2:
            return MetricResult(
                name="timbre_consistency",
                category=MetricCategory.TIMBRE,
                value=0,
                score=50,
                threshold=70,
                passed=False,
            )

        # Extract simple spectral features
        features = []
        for segment in audio_segments:
            if len(segment) > 0:
                spec = np.abs(np.fft.rfft(segment))
                # Normalize
                spec = spec / (np.sum(spec) + 1e-10)
                features.append(spec[:100])  # First 100 bins

        if len(features) < 2:
            return MetricResult(
                name="timbre_consistency",
                category=MetricCategory.TIMBRE,
                value=0,
                score=50,
                threshold=70,
                passed=False,
            )

        # Calculate pairwise correlations
        correlations = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                min_len = min(len(features[i]), len(features[j]))
                corr = np.corrcoef(features[i][:min_len], features[j][:min_len])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        avg_correlation = np.mean(correlations) if correlations else 0
        score = max(0, avg_correlation * 100)

        return MetricResult(
            name="timbre_consistency",
            category=MetricCategory.TIMBRE,
            value=avg_correlation,
            score=score,
            threshold=70,
            passed=score >= 70,
            details=f"Avg spectral correlation: {avg_correlation:.2f}",
        )

    def calculate_breathiness(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
    ) -> MetricResult:
        """
        Estimate breathiness level.

        Args:
            audio: Audio signal
            sample_rate: Sample rate

        Returns:
            MetricResult with breathiness level
        """
        if len(audio) == 0:
            return MetricResult(
                name="breathiness",
                category=MetricCategory.TIMBRE,
                value=0,
                score=50,
                threshold=0,  # Informational only
                passed=True,
            )

        # Simple HNR-based estimation
        # High HNR = less breathy
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sample_rate)

        # Estimate harmonic vs noise energy
        # (Simplified - real implementation would use proper HNR)
        harmonic_bins = spec[::10]  # Every 10th bin (rough harmonic estimate)
        total_energy = np.sum(spec ** 2)
        harmonic_energy = np.sum(harmonic_bins ** 2) * 10

        hnr = harmonic_energy / (total_energy - harmonic_energy + 1e-10)
        breathiness = 1 / (1 + hnr)  # Inverse relationship

        return MetricResult(
            name="breathiness",
            category=MetricCategory.TIMBRE,
            value=breathiness,
            score=breathiness * 100,
            threshold=0,
            passed=True,
            details=f"Breathiness level: {breathiness:.2f}",
        )


class QualityMetricsCollector:
    """
    Collects and aggregates all quality metrics.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.pitch_metrics = PitchMetrics()
        self.timing_metrics = TimingMetrics()
        self.phonetic_metrics = PhoneticMetrics()
        self.timbre_metrics = TimbreMetrics()

    def collect_all_metrics(
        self,
        generated_audio: np.ndarray,
        generated_pitch: np.ndarray,
        target_pitch: np.ndarray,
        generated_onsets: List[float],
        target_onsets: List[float],
        generated_phonemes: List[str],
        target_phonemes: List[str],
        sample_rate: int = 48000,
    ) -> MetricSummary:
        """
        Collect all quality metrics.

        Args:
            generated_audio: Generated audio
            generated_pitch: Generated pitch contour
            target_pitch: Target pitch contour
            generated_onsets: Generated onset times
            target_onsets: Target onset times
            generated_phonemes: Generated phoneme sequence
            target_phonemes: Target phoneme sequence
            sample_rate: Audio sample rate

        Returns:
            MetricSummary with all results
        """
        metrics = []

        # Pitch metrics
        metrics.append(self.pitch_metrics.calculate_pitch_accuracy(
            generated_pitch, target_pitch
        ))
        metrics.append(self.pitch_metrics.calculate_pitch_stability(
            generated_pitch
        ))
        metrics.append(self.pitch_metrics.calculate_vibrato_quality(
            generated_pitch
        ))

        # Timing metrics
        metrics.append(self.timing_metrics.calculate_timing_accuracy(
            generated_onsets, target_onsets
        ))

        # Phonetic metrics
        metrics.append(self.phonetic_metrics.calculate_phoneme_accuracy(
            generated_phonemes, target_phonemes
        ))

        # Timbre metrics
        if len(generated_audio) > 0:
            segments = np.array_split(generated_audio, 5)
            metrics.append(self.timbre_metrics.calculate_timbre_consistency(
                segments, sample_rate
            ))

        # Calculate category scores
        category_scores = {}
        for category in MetricCategory:
            cat_metrics = [m for m in metrics if m.category == category]
            if cat_metrics:
                category_scores[category] = np.mean([m.score for m in cat_metrics])

        # Overall score (weighted average)
        weights = {
            MetricCategory.PITCH: 0.35,
            MetricCategory.TIMING: 0.25,
            MetricCategory.PHONETIC: 0.25,
            MetricCategory.TIMBRE: 0.15,
        }

        overall = 0
        total_weight = 0
        for category, score in category_scores.items():
            weight = weights.get(category, 0.1)
            overall += score * weight
            total_weight += weight

        overall_score = overall / total_weight if total_weight > 0 else 0

        # Check pass/fail
        failures = [m for m in metrics if not m.passed]
        failure_reasons = [f"{m.name}: {m.details}" for m in failures]

        return MetricSummary(
            overall_score=overall_score,
            category_scores=category_scores,
            metrics=metrics,
            passed=len(failures) == 0,
            failure_reasons=failure_reasons,
        )
