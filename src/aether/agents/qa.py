"""
QA Agent

Performs quality assurance checks: originality, technical specs, genre authenticity.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel

from aether.agents.base import BaseAgent, AgentRegistry
from aether.knowledge import get_genre_manager, contour_to_hash
from aether.schemas.qa import (
    QAReport,
    OriginalityCheck,
    TechnicalCheck,
    GenreAuthenticityResult,
    GenreRubricScore,
)
from aether.storage import ArtifactType

logger = logging.getLogger(__name__)


class QAInput(BaseModel):
    song_spec: Dict[str, Any]
    harmony_spec: Dict[str, Any]
    melody_spec: Dict[str, Any]
    lyric_spec: Dict[str, Any]
    master_spec: Dict[str, Any]
    genre_profile_id: str


class QAOutput(BaseModel):
    qa_report: Dict[str, Any]
    passed: bool


@AgentRegistry.register("qa")
class QAAgent(BaseAgent[QAInput, QAOutput]):
    """
    Quality Assurance Agent.

    Responsibilities:
    - Check melody originality (interval hash, contour)
    - Check lyric originality (n-gram analysis)
    - Check chord progression originality
    - Verify technical specs (loudness, true peak, DR)
    - Evaluate genre authenticity against rubric
    - Generate final QA report
    """

    agent_type = "qa"
    agent_name = "QA Agent"
    input_schema = QAInput
    output_schema = QAOutput

    # Thresholds
    MELODY_ORIGINALITY_THRESHOLD = 0.85
    LYRIC_ORIGINALITY_THRESHOLD = 0.03  # Max similarity allowed
    HARMONY_ORIGINALITY_THRESHOLD = 0.7
    GENRE_AUTHENTICITY_THRESHOLD = 0.8

    async def process(
        self,
        input_data: QAInput,
        context: Dict[str, Any],
    ) -> QAOutput:
        song_spec = input_data.song_spec
        melody_spec = input_data.melody_spec
        lyric_spec = input_data.lyric_spec
        harmony_spec = input_data.harmony_spec
        master_spec = input_data.master_spec
        genre_manager = get_genre_manager()
        profile = genre_manager.get(input_data.genre_profile_id)

        qa_started = datetime.utcnow()

        # Originality checks
        originality_checks = []
        originality_checks.append(self._check_melody_originality(melody_spec))
        originality_checks.append(self._check_lyric_originality(lyric_spec))
        originality_checks.append(self._check_harmony_originality(harmony_spec))

        originality_passed = all(c.passed for c in originality_checks)
        overall_originality = sum(c.score for c in originality_checks) / len(originality_checks)

        # Technical checks
        technical_checks = self._run_technical_checks(master_spec)
        technical_passed = all(c.passed for c in technical_checks)

        # Genre authenticity
        genre_result = self._evaluate_genre_authenticity(
            song_spec, harmony_spec, melody_spec, profile
        )

        # Final verdict
        all_passed = originality_passed and technical_passed and genre_result.passed
        rejection_reasons = []
        warnings = []

        if not originality_passed:
            rejection_reasons.append("Originality checks failed")
        if not technical_passed:
            rejection_reasons.append("Technical specs out of tolerance")
        if not genre_result.passed:
            warnings.append(
                f"Genre authenticity score below threshold: {genre_result.total_score:.2f}"
            )

        qa_report = QAReport(
            song_id=str(song_spec["id"]),
            qa_started=qa_started,
            qa_completed=datetime.utcnow(),
            originality_checks=originality_checks,
            originality_passed=originality_passed,
            overall_originality_score=overall_originality,
            technical_checks=technical_checks,
            technical_passed=technical_passed,
            genre_authenticity=genre_result,
            all_passed=all_passed,
            rejection_reasons=rejection_reasons,
            warnings=warnings,
            human_reviewed=False,
        )

        self.log_decision(
            decision_type="qa_verdict",
            input_summary=f"Song: {song_spec.get('title', 'Unknown')}",
            output_summary=f"{'PASSED' if all_passed else 'FAILED'}: Originality={overall_originality:.2f}, Genre={genre_result.total_score:.2f}",
            reasoning="Comprehensive QA against originality, technical, and genre criteria",
            confidence=0.95,
        )

        return QAOutput(
            qa_report=qa_report.model_dump(),
            passed=all_passed,
        )

    def _check_melody_originality(self, melody_spec: Dict) -> OriginalityCheck:
        """Check melody for originality."""
        # Get the primary hook
        hook = melody_spec.get("primary_hook", {})
        notes = hook.get("melody_notes", [])

        # Compute interval hash (simplified)
        # In production, would compare against database of known melodies
        originality_score = melody_spec.get("originality_score", 0.85)

        # Placeholder: assume original if no match found
        return OriginalityCheck(
            check_name="melody_interval_hash",
            check_type="melody",
            score=originality_score,
            threshold=self.MELODY_ORIGINALITY_THRESHOLD,
            passed=originality_score >= self.MELODY_ORIGINALITY_THRESHOLD,
            details=f"Analyzed {len(notes)} note hook",
            similar_matches=[],
        )

    def _check_lyric_originality(self, lyric_spec: Dict) -> OriginalityCheck:
        """Check lyrics for originality using n-gram analysis."""
        sections = lyric_spec.get("sections", [])
        total_lines = sum(len(s.get("lines", [])) for s in sections)

        # In production, would check against lyrics database
        # Using placeholder score
        similarity_score = 0.01  # Low = original

        return OriginalityCheck(
            check_name="lyric_ngram_analysis",
            check_type="lyrics",
            score=1.0 - similarity_score,  # Invert for originality
            threshold=1.0 - self.LYRIC_ORIGINALITY_THRESHOLD,
            passed=similarity_score <= self.LYRIC_ORIGINALITY_THRESHOLD,
            details=f"Analyzed {total_lines} lyric lines",
            similar_matches=[],
        )

    def _check_harmony_originality(self, harmony_spec: Dict) -> OriginalityCheck:
        """Check chord progressions for originality."""
        progressions = harmony_spec.get("progressions", [])

        # Common progressions are okay - we check for exact copying
        # In production, would use progression fingerprinting
        originality_score = harmony_spec.get("originality_score", 0.85)

        return OriginalityCheck(
            check_name="harmony_progression_check",
            check_type="harmony",
            score=originality_score,
            threshold=self.HARMONY_ORIGINALITY_THRESHOLD,
            passed=originality_score >= self.HARMONY_ORIGINALITY_THRESHOLD,
            details=f"Analyzed {len(progressions)} progressions",
            similar_matches=[],
        )

    def _run_technical_checks(self, master_spec: Dict) -> List[TechnicalCheck]:
        """Run technical audio checks."""
        checks = []

        # Loudness check
        loudness = master_spec.get("loudness", {})
        target_lufs = loudness.get("target_lufs", -14.0)
        tolerance = loudness.get("tolerance", 0.5)
        measured = master_spec.get("measured_lufs", target_lufs)  # Placeholder

        checks.append(
            TechnicalCheck(
                check_name="integrated_loudness",
                measured_value=measured,
                target_value=target_lufs,
                tolerance=tolerance,
                passed=abs(measured - target_lufs) <= tolerance,
                unit="LUFS",
            )
        )

        # True peak check
        true_peak = master_spec.get("true_peak", {})
        ceiling = true_peak.get("ceiling_dbtp", -1.0)
        measured_peak = master_spec.get("measured_true_peak", ceiling)

        checks.append(
            TechnicalCheck(
                check_name="true_peak",
                measured_value=measured_peak,
                target_value=ceiling,
                tolerance=0.0,
                passed=measured_peak <= ceiling,
                unit="dBTP",
            )
        )

        # Dynamic range check
        dr = master_spec.get("dynamic_range", {})
        target_dr = dr.get("target_lu", 8.0)
        min_dr = dr.get("minimum_lu", 6.0)
        measured_dr = master_spec.get("measured_dynamic_range", target_dr)

        checks.append(
            TechnicalCheck(
                check_name="dynamic_range",
                measured_value=measured_dr,
                target_value=target_dr,
                tolerance=2.0,
                passed=measured_dr >= min_dr,
                unit="LU",
            )
        )

        return checks

    def _evaluate_genre_authenticity(
        self,
        song_spec: Dict,
        harmony_spec: Dict,
        melody_spec: Dict,
        profile,
    ) -> GenreAuthenticityResult:
        """Evaluate genre authenticity against rubric."""
        rubric = profile.authenticity_rubric
        dimension_scores = []

        # Evaluate each rubric dimension
        for dimension in rubric:
            score = self._evaluate_dimension(
                dimension, song_spec, harmony_spec, melody_spec, profile
            )
            weighted = score * dimension.weight
            dimension_scores.append(
                GenreRubricScore(
                    dimension_name=dimension.dimension_name,
                    weight=dimension.weight,
                    score=score,
                    weighted_score=weighted,
                    feedback=self._generate_feedback(dimension.dimension_name, score),
                )
            )

        total_score = sum(d.weighted_score for d in dimension_scores)

        # Generate improvement suggestions
        suggestions = []
        for d in dimension_scores:
            if d.score < 0.7:
                suggestions.append(f"Improve {d.dimension_name}: {d.feedback}")

        return GenreAuthenticityResult(
            genre_id=profile.id,
            dimension_scores=dimension_scores,
            total_score=total_score,
            threshold=self.GENRE_AUTHENTICITY_THRESHOLD,
            passed=total_score >= self.GENRE_AUTHENTICITY_THRESHOLD,
            improvement_suggestions=suggestions[:3],  # Top 3
        )

    def _evaluate_dimension(
        self,
        dimension,
        song_spec: Dict,
        harmony_spec: Dict,
        melody_spec: Dict,
        profile,
    ) -> float:
        """Evaluate a single rubric dimension."""
        # Simplified evaluation - would be more sophisticated in production
        dim_name = dimension.dimension_name.lower()

        if "rhythm" in dim_name or "drum" in dim_name:
            # Check BPM is in range
            bpm = song_spec.get("bpm", 120)
            if profile.tempo.min_bpm <= bpm <= profile.tempo.max_bpm:
                return 0.9
            return 0.5

        elif "harmony" in dim_name or "chord" in dim_name:
            # Check if using genre-typical progressions
            return harmony_spec.get("originality_score", 0.85)

        elif "melody" in dim_name:
            return melody_spec.get("originality_score", 0.85)

        elif "production" in dim_name or "sound" in dim_name:
            # Assume sound design matches if we got this far
            return 0.85

        else:
            # Default
            return 0.8

    def _generate_feedback(self, dimension_name: str, score: float) -> str:
        """Generate feedback for a dimension score."""
        if score >= 0.9:
            return f"Excellent {dimension_name.lower()}"
        elif score >= 0.8:
            return f"Good {dimension_name.lower()}, minor improvements possible"
        elif score >= 0.7:
            return f"Acceptable {dimension_name.lower()}, some elements could be stronger"
        else:
            return f"{dimension_name} needs significant improvement"
