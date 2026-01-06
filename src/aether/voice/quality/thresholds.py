"""
Quality Thresholds

Defines pass/fail thresholds for different release stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ReleaseStage(Enum):
    """Release stages with different quality requirements."""
    DEVELOPMENT = "development"
    ALPHA = "alpha"
    BETA = "beta"
    PRODUCTION = "production"


@dataclass
class QualityThresholds:
    """
    Quality thresholds for a release stage.

    All values are minimum scores (0-100) required to pass.
    """
    # Overall
    overall_minimum: float

    # Pitch thresholds
    pitch_accuracy: float
    pitch_stability: float
    vibrato_quality: float

    # Timing thresholds
    timing_accuracy: float
    rhythm_consistency: float

    # Phonetic thresholds
    phoneme_accuracy: float
    formant_accuracy: float

    # Timbre thresholds
    timbre_consistency: float

    # Human evaluation minimums (MOS 1-5 scale)
    mos_naturalness: float
    mos_intelligibility: float
    mos_expressiveness: float

    def get_threshold(self, metric_name: str) -> Optional[float]:
        """
        Get threshold for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Threshold value or None if not defined
        """
        mapping = {
            "pitch_accuracy": self.pitch_accuracy,
            "pitch_stability": self.pitch_stability,
            "vibrato_quality": self.vibrato_quality,
            "timing_accuracy": self.timing_accuracy,
            "rhythm_consistency": self.rhythm_consistency,
            "phoneme_accuracy": self.phoneme_accuracy,
            "formant_accuracy": self.formant_accuracy,
            "timbre_consistency": self.timbre_consistency,
        }
        return mapping.get(metric_name)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "overall_minimum": self.overall_minimum,
            "pitch_accuracy": self.pitch_accuracy,
            "pitch_stability": self.pitch_stability,
            "vibrato_quality": self.vibrato_quality,
            "timing_accuracy": self.timing_accuracy,
            "rhythm_consistency": self.rhythm_consistency,
            "phoneme_accuracy": self.phoneme_accuracy,
            "formant_accuracy": self.formant_accuracy,
            "timbre_consistency": self.timbre_consistency,
            "mos_naturalness": self.mos_naturalness,
            "mos_intelligibility": self.mos_intelligibility,
            "mos_expressiveness": self.mos_expressiveness,
        }


# Stage-specific thresholds
STAGE_THRESHOLDS: Dict[ReleaseStage, QualityThresholds] = {
    ReleaseStage.DEVELOPMENT: QualityThresholds(
        overall_minimum=40,
        pitch_accuracy=50,
        pitch_stability=40,
        vibrato_quality=30,
        timing_accuracy=50,
        rhythm_consistency=40,
        phoneme_accuracy=60,
        formant_accuracy=40,
        timbre_consistency=40,
        mos_naturalness=2.0,
        mos_intelligibility=2.5,
        mos_expressiveness=2.0,
    ),

    ReleaseStage.ALPHA: QualityThresholds(
        overall_minimum=60,
        pitch_accuracy=70,
        pitch_stability=60,
        vibrato_quality=50,
        timing_accuracy=70,
        rhythm_consistency=60,
        phoneme_accuracy=75,
        formant_accuracy=60,
        timbre_consistency=60,
        mos_naturalness=2.5,
        mos_intelligibility=3.0,
        mos_expressiveness=2.5,
    ),

    ReleaseStage.BETA: QualityThresholds(
        overall_minimum=75,
        pitch_accuracy=80,
        pitch_stability=75,
        vibrato_quality=65,
        timing_accuracy=80,
        rhythm_consistency=75,
        phoneme_accuracy=85,
        formant_accuracy=70,
        timbre_consistency=75,
        mos_naturalness=3.5,
        mos_intelligibility=4.0,
        mos_expressiveness=3.5,
    ),

    ReleaseStage.PRODUCTION: QualityThresholds(
        overall_minimum=85,
        pitch_accuracy=90,
        pitch_stability=85,
        vibrato_quality=75,
        timing_accuracy=90,
        rhythm_consistency=85,
        phoneme_accuracy=92,
        formant_accuracy=80,
        timbre_consistency=85,
        mos_naturalness=4.0,
        mos_intelligibility=4.5,
        mos_expressiveness=4.0,
    ),
}


def get_thresholds(stage: ReleaseStage) -> QualityThresholds:
    """
    Get thresholds for a release stage.

    Args:
        stage: Release stage

    Returns:
        QualityThresholds for that stage
    """
    return STAGE_THRESHOLDS.get(stage, STAGE_THRESHOLDS[ReleaseStage.ALPHA])


def check_production_ready(
    metrics: Dict[str, float],
) -> tuple[bool, list[str]]:
    """
    Check if metrics meet production requirements.

    Args:
        metrics: Dict of metric names to scores

    Returns:
        Tuple of (is_ready, list of failing metrics)
    """
    prod_thresholds = get_thresholds(ReleaseStage.PRODUCTION)
    failures = []

    threshold_dict = prod_thresholds.to_dict()

    for metric_name, score in metrics.items():
        if metric_name in threshold_dict:
            threshold = threshold_dict[metric_name]
            if score < threshold:
                failures.append(
                    f"{metric_name}: {score:.1f} < {threshold:.1f} required"
                )

    return len(failures) == 0, failures


class ThresholdProgressTracker:
    """
    Tracks progress toward production thresholds.
    """

    def __init__(self, current_stage: ReleaseStage = ReleaseStage.ALPHA):
        """
        Initialize tracker.

        Args:
            current_stage: Current release stage
        """
        self.current_stage = current_stage
        self.history: list[Dict[str, float]] = []

    def record(self, metrics: Dict[str, float]) -> None:
        """Record a metrics snapshot."""
        self.history.append(metrics.copy())

    def get_progress_to_next_stage(
        self,
        current_metrics: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Get progress toward next stage thresholds.

        Args:
            current_metrics: Current metric values

        Returns:
            Dict mapping metrics to progress info
        """
        # Determine next stage
        stage_order = [
            ReleaseStage.DEVELOPMENT,
            ReleaseStage.ALPHA,
            ReleaseStage.BETA,
            ReleaseStage.PRODUCTION,
        ]

        current_idx = stage_order.index(self.current_stage)
        if current_idx >= len(stage_order) - 1:
            return {}  # Already at production

        next_stage = stage_order[current_idx + 1]
        next_thresholds = get_thresholds(next_stage)
        threshold_dict = next_thresholds.to_dict()

        progress = {}
        for metric_name, current_value in current_metrics.items():
            if metric_name in threshold_dict:
                target = threshold_dict[metric_name]
                if current_value >= target:
                    pct = 100.0
                else:
                    pct = (current_value / target) * 100 if target > 0 else 0

                progress[metric_name] = {
                    "current": current_value,
                    "target": target,
                    "progress_pct": pct,
                    "gap": max(0, target - current_value),
                }

        return progress

    def get_improvement_trend(
        self,
        metric_name: str,
        window: int = 10,
    ) -> Optional[float]:
        """
        Get improvement trend for a metric.

        Args:
            metric_name: Metric to analyze
            window: Number of recent samples to use

        Returns:
            Trend slope (positive = improving) or None
        """
        if len(self.history) < 2:
            return None

        recent = self.history[-window:]
        values = [h.get(metric_name, 0) for h in recent if metric_name in h]

        if len(values) < 2:
            return None

        # Simple linear regression slope
        x = range(len(values))
        x_mean = sum(x) / len(x)
        y_mean = sum(values) / len(values)

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def generate_readiness_report(
        self,
        current_metrics: Dict[str, float],
    ) -> str:
        """
        Generate a readiness report.

        Args:
            current_metrics: Current metrics

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 50,
            "PRODUCTION READINESS REPORT",
            "=" * 50,
            f"Current Stage: {self.current_stage.value}",
            "",
        ]

        # Check production readiness
        is_ready, failures = check_production_ready(current_metrics)

        if is_ready:
            lines.append("STATUS: PRODUCTION READY ✓")
        else:
            lines.append("STATUS: NOT READY")
            lines.append("")
            lines.append("Failing Metrics:")
            for failure in failures:
                lines.append(f"  - {failure}")

        # Progress to next stage
        progress = self.get_progress_to_next_stage(current_metrics)
        if progress:
            lines.extend([
                "",
                "-" * 50,
                f"Progress to Next Stage",
                "-" * 50,
            ])
            for metric, info in progress.items():
                status = "✓" if info["progress_pct"] >= 100 else "○"
                lines.append(
                    f"  {status} {metric}: {info['current']:.1f}/{info['target']:.1f} "
                    f"({info['progress_pct']:.0f}%)"
                )

        # Trends
        lines.extend([
            "",
            "-" * 50,
            "Improvement Trends",
            "-" * 50,
        ])
        for metric in current_metrics:
            trend = self.get_improvement_trend(metric)
            if trend is not None:
                direction = "↑" if trend > 0.5 else "↓" if trend < -0.5 else "→"
                lines.append(f"  {metric}: {direction} ({trend:+.2f}/sample)")

        lines.append("=" * 50)
        return "\n".join(lines)


# Convenience function for quick quality checks
def quick_quality_check(
    pitch_accuracy: float,
    timing_accuracy: float,
    phoneme_accuracy: float,
    stage: ReleaseStage = ReleaseStage.BETA,
) -> tuple[bool, str]:
    """
    Quick quality check for core metrics.

    Args:
        pitch_accuracy: Pitch accuracy score
        timing_accuracy: Timing accuracy score
        phoneme_accuracy: Phoneme accuracy score
        stage: Release stage to check against

    Returns:
        Tuple of (passed, message)
    """
    thresholds = get_thresholds(stage)

    failures = []
    if pitch_accuracy < thresholds.pitch_accuracy:
        failures.append(f"pitch ({pitch_accuracy:.0f} < {thresholds.pitch_accuracy:.0f})")
    if timing_accuracy < thresholds.timing_accuracy:
        failures.append(f"timing ({timing_accuracy:.0f} < {thresholds.timing_accuracy:.0f})")
    if phoneme_accuracy < thresholds.phoneme_accuracy:
        failures.append(f"phoneme ({phoneme_accuracy:.0f} < {thresholds.phoneme_accuracy:.0f})")

    if failures:
        return False, f"Failed: {', '.join(failures)}"
    return True, "Passed all core metrics"
