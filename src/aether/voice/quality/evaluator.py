"""
Vocal Quality Evaluator

Comprehensive evaluation system for synthesized vocals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from aether.voice.quality.metrics import (
    QualityMetricsCollector,
    MetricSummary,
    MetricCategory,
)
from aether.voice.quality.thresholds import (
    QualityThresholds,
    ReleaseStage,
    get_thresholds,
)


class EvaluationGrade(Enum):
    """Quality grades."""
    A_PLUS = "A+"  # 95-100
    A = "A"        # 90-94
    B_PLUS = "B+"  # 85-89
    B = "B"        # 80-84
    C_PLUS = "C+"  # 75-79
    C = "C"        # 70-74
    D = "D"        # 60-69
    F = "F"        # < 60


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    # Overall
    grade: EvaluationGrade
    score: float
    passed: bool
    release_stage: ReleaseStage

    # Detailed metrics
    metrics_summary: MetricSummary

    # Human evaluation scores (if available)
    human_scores: Optional[Dict[str, float]] = None

    # Recommendations
    improvements: List[str] = None
    strengths: List[str] = None


class HumanEvaluationRubric:
    """
    Rubric for human evaluation of synthesized vocals.

    Based on Mean Opinion Score (MOS) methodology.
    """

    CRITERIA = {
        "naturalness": {
            "description": "How natural and human-like does the voice sound?",
            "weight": 0.25,
            "scale": (1, 5),
            "anchors": {
                1: "Clearly synthetic, robotic",
                2: "Somewhat unnatural, noticeable artifacts",
                3: "Acceptable, minor synthetic qualities",
                4: "Natural, occasional minor issues",
                5: "Indistinguishable from human",
            },
        },
        "intelligibility": {
            "description": "How clearly can you understand the lyrics?",
            "weight": 0.20,
            "scale": (1, 5),
            "anchors": {
                1: "Unintelligible",
                2: "Difficult to understand",
                3: "Understandable with effort",
                4: "Clear with minor issues",
                5: "Perfectly clear",
            },
        },
        "expressiveness": {
            "description": "How expressive and emotionally appropriate is the performance?",
            "weight": 0.20,
            "scale": (1, 5),
            "anchors": {
                1: "Flat, monotone",
                2: "Minimal expression",
                3: "Some expression, somewhat generic",
                4: "Good expression, fits the song",
                5: "Highly expressive, emotionally compelling",
            },
        },
        "pitch_quality": {
            "description": "How accurate and pleasing is the pitch?",
            "weight": 0.15,
            "scale": (1, 5),
            "anchors": {
                1: "Severely off-pitch",
                2: "Noticeably out of tune",
                3: "Acceptable pitch, minor issues",
                4: "Good pitch accuracy",
                5: "Perfect pitch, beautiful intonation",
            },
        },
        "timing_quality": {
            "description": "How well does the timing fit the music?",
            "weight": 0.10,
            "scale": (1, 5),
            "anchors": {
                1: "Completely off rhythm",
                2: "Noticeably out of time",
                3: "Acceptable timing",
                4: "Good groove and timing",
                5: "Perfect rhythm, great feel",
            },
        },
        "genre_fit": {
            "description": "How well does the vocal style fit the genre?",
            "weight": 0.10,
            "scale": (1, 5),
            "anchors": {
                1: "Completely inappropriate style",
                2: "Mostly inappropriate",
                3: "Acceptable, somewhat generic",
                4: "Good fit for genre",
                5: "Perfect genre authenticity",
            },
        },
    }

    def __init__(self):
        """Initialize human evaluation rubric."""
        pass

    def get_rubric(self) -> Dict:
        """Get the evaluation rubric."""
        return self.CRITERIA

    def calculate_weighted_score(
        self,
        scores: Dict[str, float],
    ) -> float:
        """
        Calculate weighted average from human scores.

        Args:
            scores: Dict mapping criteria to scores (1-5)

        Returns:
            Weighted score (0-100)
        """
        total = 0
        total_weight = 0

        for criterion, weight_info in self.CRITERIA.items():
            if criterion in scores:
                # Normalize 1-5 to 0-100
                normalized = (scores[criterion] - 1) * 25
                total += normalized * weight_info["weight"]
                total_weight += weight_info["weight"]

        return total / total_weight if total_weight > 0 else 0


class VocalQualityEvaluator:
    """
    Main evaluation system for synthesized vocals.

    Combines automated metrics with human evaluation framework.
    """

    def __init__(self, release_stage: ReleaseStage = ReleaseStage.ALPHA):
        """
        Initialize evaluator.

        Args:
            release_stage: Current release stage for thresholds
        """
        self.release_stage = release_stage
        self.thresholds = get_thresholds(release_stage)
        self.metrics_collector = QualityMetricsCollector()
        self.human_rubric = HumanEvaluationRubric()

    def evaluate(
        self,
        generated_audio: np.ndarray,
        generated_pitch: np.ndarray,
        target_pitch: np.ndarray,
        generated_onsets: List[float],
        target_onsets: List[float],
        generated_phonemes: List[str],
        target_phonemes: List[str],
        human_scores: Optional[Dict[str, float]] = None,
        sample_rate: int = 48000,
    ) -> EvaluationResult:
        """
        Perform complete evaluation.

        Args:
            generated_audio: Generated audio
            generated_pitch: Generated pitch contour
            target_pitch: Target pitch contour
            generated_onsets: Generated onset times
            target_onsets: Target onset times
            generated_phonemes: Generated phonemes
            target_phonemes: Target phonemes
            human_scores: Optional human evaluation scores
            sample_rate: Audio sample rate

        Returns:
            Complete evaluation result
        """
        # Collect automated metrics
        metrics_summary = self.metrics_collector.collect_all_metrics(
            generated_audio=generated_audio,
            generated_pitch=generated_pitch,
            target_pitch=target_pitch,
            generated_onsets=generated_onsets,
            target_onsets=target_onsets,
            generated_phonemes=generated_phonemes,
            target_phonemes=target_phonemes,
            sample_rate=sample_rate,
        )

        # Calculate combined score
        auto_score = metrics_summary.overall_score

        if human_scores:
            human_score = self.human_rubric.calculate_weighted_score(human_scores)
            # Weight: 60% automated, 40% human
            combined_score = auto_score * 0.6 + human_score * 0.4
        else:
            combined_score = auto_score

        # Determine grade
        grade = self._score_to_grade(combined_score)

        # Check pass/fail against thresholds
        passed = self._check_thresholds(metrics_summary)

        # Generate recommendations
        improvements, strengths = self._analyze_results(metrics_summary)

        return EvaluationResult(
            grade=grade,
            score=combined_score,
            passed=passed,
            release_stage=self.release_stage,
            metrics_summary=metrics_summary,
            human_scores=human_scores,
            improvements=improvements,
            strengths=strengths,
        )

    def _score_to_grade(self, score: float) -> EvaluationGrade:
        """Convert numeric score to letter grade."""
        if score >= 95:
            return EvaluationGrade.A_PLUS
        elif score >= 90:
            return EvaluationGrade.A
        elif score >= 85:
            return EvaluationGrade.B_PLUS
        elif score >= 80:
            return EvaluationGrade.B
        elif score >= 75:
            return EvaluationGrade.C_PLUS
        elif score >= 70:
            return EvaluationGrade.C
        elif score >= 60:
            return EvaluationGrade.D
        else:
            return EvaluationGrade.F

    def _check_thresholds(self, summary: MetricSummary) -> bool:
        """Check if metrics meet thresholds for release stage."""
        for metric in summary.metrics:
            threshold_value = self.thresholds.get_threshold(metric.name)
            if threshold_value is not None:
                if metric.score < threshold_value:
                    return False
        return True

    def _analyze_results(
        self,
        summary: MetricSummary,
    ) -> Tuple[List[str], List[str]]:
        """Analyze results to generate recommendations."""
        improvements = []
        strengths = []

        for metric in summary.metrics:
            if metric.score >= 90:
                strengths.append(f"{metric.name}: Excellent ({metric.score:.0f})")
            elif metric.score < 70:
                improvements.append(
                    f"{metric.name}: Needs improvement ({metric.score:.0f}) - {metric.details or ''}"
                )

        # Category-level analysis
        for category, score in summary.category_scores.items():
            if score < 70:
                improvements.append(
                    f"Focus on {category.value} quality (current: {score:.0f})"
                )

        return improvements, strengths

    def evaluate_batch(
        self,
        samples: List[dict],
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate a batch of samples.

        Args:
            samples: List of sample dicts with audio/pitch/etc data

        Returns:
            Dict mapping sample IDs to evaluation results
        """
        results = {}

        for sample in samples:
            sample_id = sample.get("id", str(len(results)))

            result = self.evaluate(
                generated_audio=sample.get("audio", np.array([])),
                generated_pitch=sample.get("pitch", np.array([])),
                target_pitch=sample.get("target_pitch", np.array([])),
                generated_onsets=sample.get("onsets", []),
                target_onsets=sample.get("target_onsets", []),
                generated_phonemes=sample.get("phonemes", []),
                target_phonemes=sample.get("target_phonemes", []),
                human_scores=sample.get("human_scores"),
                sample_rate=sample.get("sample_rate", 48000),
            )

            results[sample_id] = result

        return results

    def generate_report(
        self,
        result: EvaluationResult,
    ) -> str:
        """
        Generate human-readable evaluation report.

        Args:
            result: Evaluation result

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 50,
            "VOCAL QUALITY EVALUATION REPORT",
            "=" * 50,
            "",
            f"Overall Grade: {result.grade.value}",
            f"Score: {result.score:.1f}/100",
            f"Release Stage: {result.release_stage.value}",
            f"Status: {'PASS' if result.passed else 'FAIL'}",
            "",
            "-" * 50,
            "CATEGORY SCORES",
            "-" * 50,
        ]

        for category, score in result.metrics_summary.category_scores.items():
            lines.append(f"  {category.value}: {score:.1f}")

        lines.extend([
            "",
            "-" * 50,
            "INDIVIDUAL METRICS",
            "-" * 50,
        ])

        for metric in result.metrics_summary.metrics:
            status = "✓" if metric.passed else "✗"
            lines.append(f"  {status} {metric.name}: {metric.score:.1f}")
            if metric.details:
                lines.append(f"      {metric.details}")

        if result.strengths:
            lines.extend([
                "",
                "-" * 50,
                "STRENGTHS",
                "-" * 50,
            ])
            for strength in result.strengths:
                lines.append(f"  + {strength}")

        if result.improvements:
            lines.extend([
                "",
                "-" * 50,
                "AREAS FOR IMPROVEMENT",
                "-" * 50,
            ])
            for improvement in result.improvements:
                lines.append(f"  - {improvement}")

        if result.human_scores:
            lines.extend([
                "",
                "-" * 50,
                "HUMAN EVALUATION SCORES",
                "-" * 50,
            ])
            for criterion, score in result.human_scores.items():
                lines.append(f"  {criterion}: {score:.1f}/5")

        lines.extend([
            "",
            "=" * 50,
        ])

        return "\n".join(lines)


class ABTestEvaluator:
    """
    A/B testing framework for comparing voice models.
    """

    def __init__(self):
        """Initialize A/B test evaluator."""
        self.base_evaluator = VocalQualityEvaluator()

    def compare(
        self,
        model_a_output: dict,
        model_b_output: dict,
        target_data: dict,
    ) -> Dict[str, any]:
        """
        Compare two model outputs.

        Args:
            model_a_output: Output from model A
            model_b_output: Output from model B
            target_data: Ground truth data

        Returns:
            Comparison results
        """
        # Evaluate both
        result_a = self.base_evaluator.evaluate(
            generated_audio=model_a_output.get("audio", np.array([])),
            generated_pitch=model_a_output.get("pitch", np.array([])),
            target_pitch=target_data.get("pitch", np.array([])),
            generated_onsets=model_a_output.get("onsets", []),
            target_onsets=target_data.get("onsets", []),
            generated_phonemes=model_a_output.get("phonemes", []),
            target_phonemes=target_data.get("phonemes", []),
        )

        result_b = self.base_evaluator.evaluate(
            generated_audio=model_b_output.get("audio", np.array([])),
            generated_pitch=model_b_output.get("pitch", np.array([])),
            target_pitch=target_data.get("pitch", np.array([])),
            generated_onsets=model_b_output.get("onsets", []),
            target_onsets=target_data.get("onsets", []),
            generated_phonemes=model_b_output.get("phonemes", []),
            target_phonemes=target_data.get("phonemes", []),
        )

        # Compare
        score_diff = result_a.score - result_b.score

        # Per-category comparison
        category_diffs = {}
        for category in result_a.metrics_summary.category_scores:
            score_a = result_a.metrics_summary.category_scores.get(category, 0)
            score_b = result_b.metrics_summary.category_scores.get(category, 0)
            category_diffs[category.value] = score_a - score_b

        return {
            "winner": "A" if score_diff > 0 else "B" if score_diff < 0 else "tie",
            "score_difference": abs(score_diff),
            "model_a_score": result_a.score,
            "model_b_score": result_b.score,
            "model_a_grade": result_a.grade.value,
            "model_b_grade": result_b.grade.value,
            "category_differences": category_diffs,
            "model_a_result": result_a,
            "model_b_result": result_b,
        }
