"""
AETHER QA Report Generator

Comprehensive quality assurance reporting for AI-generated music.

Features:
- Multi-component aggregation (technical, originality, authenticity)
- Executive summaries with key metrics
- Detailed breakdown by category
- Prioritized improvement suggestions
- Multiple output formats (Markdown, JSON, dict)
- Pass/fail gate determination
- Configurable severity levels and thresholds

This is the final integration point for all QA components.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class Severity(Enum):
    """Issue severity levels."""

    CRITICAL = "critical"  # Fails release, must fix
    MAJOR = "major"  # Significant issue, should fix
    MINOR = "minor"  # Small issue, consider fixing
    INFO = "info"  # Informational, no action required


class QACategory(Enum):
    """QA evaluation categories."""

    TECHNICAL = "technical"
    ORIGINALITY = "originality"
    AUTHENTICITY = "authenticity"
    OVERALL = "overall"


class PassFailStatus(Enum):
    """Overall pass/fail status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"  # Passes with concerns


# ============================================================================
# Report Components
# ============================================================================


@dataclass
class QAIssue:
    """A single QA issue or finding."""

    category: QACategory
    severity: Severity
    title: str
    description: str
    suggestion: str = ""
    metric_name: str = ""
    actual_value: Optional[float] = None
    expected_value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "suggestion": self.suggestion,
            "metric_name": self.metric_name,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
            "threshold": self.threshold,
        }


@dataclass
class MetricResult:
    """Result for a single metric."""

    name: str
    value: float
    unit: str = ""
    target: Optional[float] = None
    tolerance: Optional[float] = None
    passed: bool = True
    severity: Severity = Severity.INFO

    @property
    def formatted_value(self) -> str:
        """Format value with unit."""
        if self.unit:
            return f"{self.value:.2f} {self.unit}"
        return f"{self.value:.2f}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "unit": self.unit,
            "target": self.target,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "severity": self.severity.value,
        }


@dataclass
class CategoryReport:
    """Report for a single QA category."""

    category: QACategory
    passed: bool
    score: float  # 0-1
    metrics: list[MetricResult] = field(default_factory=list)
    issues: list[QAIssue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def critical_issues(self) -> list[QAIssue]:
        """Get critical issues only."""
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    @property
    def major_issues(self) -> list[QAIssue]:
        """Get major issues only."""
        return [i for i in self.issues if i.severity == Severity.MAJOR]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "passed": self.passed,
            "score": round(self.score, 4),
            "metrics": [m.to_dict() for m in self.metrics],
            "issues": [i.to_dict() for i in self.issues],
            "notes": self.notes,
        }


@dataclass
class ExecutiveSummary:
    """High-level executive summary."""

    status: PassFailStatus
    overall_score: float  # 0-1
    technical_score: float
    originality_score: float
    authenticity_score: float
    critical_count: int
    major_count: int
    minor_count: int
    top_issues: list[str]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "overall_score": round(self.overall_score, 4),
            "scores": {
                "technical": round(self.technical_score, 4),
                "originality": round(self.originality_score, 4),
                "authenticity": round(self.authenticity_score, 4),
            },
            "issue_counts": {
                "critical": self.critical_count,
                "major": self.major_count,
                "minor": self.minor_count,
            },
            "top_issues": self.top_issues,
            "recommendation": self.recommendation,
        }


# ============================================================================
# Main Report
# ============================================================================


@dataclass
class QAReport:
    """
    Comprehensive QA report for a track.

    Aggregates results from all QA components and provides
    actionable insights for quality improvement.
    """

    # Metadata
    report_id: UUID = field(default_factory=uuid4)
    track_id: str = ""
    track_title: str = ""
    genre_id: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Summary
    executive_summary: ExecutiveSummary | None = None

    # Category reports
    technical_report: CategoryReport | None = None
    originality_report: CategoryReport | None = None
    authenticity_report: CategoryReport | None = None

    # Overall
    overall_passed: bool = False
    overall_score: float = 0.0

    # All issues aggregated
    all_issues: list[QAIssue] = field(default_factory=list)

    # Improvement suggestions (prioritized)
    improvement_priority: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert entire report to dictionary."""
        return {
            "metadata": {
                "report_id": str(self.report_id),
                "track_id": self.track_id,
                "track_title": self.track_title,
                "genre_id": self.genre_id,
                "generated_at": self.generated_at.isoformat(),
            },
            "executive_summary": (
                self.executive_summary.to_dict() if self.executive_summary else None
            ),
            "categories": {
                "technical": self.technical_report.to_dict() if self.technical_report else None,
                "originality": (
                    self.originality_report.to_dict() if self.originality_report else None
                ),
                "authenticity": (
                    self.authenticity_report.to_dict() if self.authenticity_report else None
                ),
            },
            "overall": {
                "passed": self.overall_passed,
                "score": round(self.overall_score, 4),
            },
            "issues": [i.to_dict() for i in self.all_issues],
            "improvement_priority": self.improvement_priority,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []

        # Header
        lines.append("# AETHER QA Report")
        lines.append("")
        lines.append(f"**Track:** {self.track_title or 'Untitled'}")
        lines.append(f"**Genre:** {self.genre_id}")
        lines.append(f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Report ID:** `{self.report_id}`")
        lines.append("")

        # Executive Summary
        if self.executive_summary:
            es = self.executive_summary
            status_emoji = {
                PassFailStatus.PASS: "✅",
                PassFailStatus.FAIL: "❌",
                PassFailStatus.WARNING: "⚠️",
            }[es.status]

            lines.append("## Executive Summary")
            lines.append("")
            lines.append(f"**Status:** {status_emoji} {es.status.value.upper()}")
            lines.append(f"**Overall Score:** {es.overall_score:.0%}")
            lines.append("")
            lines.append("### Scores by Category")
            lines.append("")
            lines.append("| Category | Score | Status |")
            lines.append("|----------|-------|--------|")

            for cat, score in [
                ("Technical", es.technical_score),
                ("Originality", es.originality_score),
                ("Authenticity", es.authenticity_score),
            ]:
                status = "✅" if score >= 0.8 else ("⚠️" if score >= 0.6 else "❌")
                lines.append(f"| {cat} | {score:.0%} | {status} |")

            lines.append("")
            lines.append("### Issue Summary")
            lines.append("")
            lines.append(f"- **Critical:** {es.critical_count}")
            lines.append(f"- **Major:** {es.major_count}")
            lines.append(f"- **Minor:** {es.minor_count}")
            lines.append("")

            if es.top_issues:
                lines.append("### Top Issues")
                lines.append("")
                for issue in es.top_issues[:5]:
                    lines.append(f"- {issue}")
                lines.append("")

            lines.append("### Recommendation")
            lines.append("")
            lines.append(f"> {es.recommendation}")
            lines.append("")

        # Technical Report
        if self.technical_report:
            lines.append("---")
            lines.append("")
            lines.append("## Technical Quality")
            lines.append("")
            lines.append(self._category_report_markdown(self.technical_report))

        # Originality Report
        if self.originality_report:
            lines.append("---")
            lines.append("")
            lines.append("## Originality Analysis")
            lines.append("")
            lines.append(self._category_report_markdown(self.originality_report))

        # Authenticity Report
        if self.authenticity_report:
            lines.append("---")
            lines.append("")
            lines.append("## Genre Authenticity")
            lines.append("")
            lines.append(self._category_report_markdown(self.authenticity_report))

        # Improvement Priority
        if self.improvement_priority:
            lines.append("---")
            lines.append("")
            lines.append("## Improvement Priority")
            lines.append("")
            lines.append("Actions ordered by impact:")
            lines.append("")
            for i, item in enumerate(self.improvement_priority[:10], 1):
                lines.append(f"{i}. {item}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by AETHER QA System*")

        return "\n".join(lines)

    def _category_report_markdown(self, report: CategoryReport) -> str:
        """Generate markdown for a category report."""
        lines = []

        status = "✅ PASS" if report.passed else "❌ FAIL"
        lines.append(f"**Status:** {status}")
        lines.append(f"**Score:** {report.score:.0%}")
        lines.append("")

        # Metrics table
        if report.metrics:
            lines.append("### Metrics")
            lines.append("")
            lines.append("| Metric | Value | Target | Status |")
            lines.append("|--------|-------|--------|--------|")
            for m in report.metrics:
                target = f"{m.target:.2f}" if m.target is not None else "-"
                status = "✅" if m.passed else "❌"
                lines.append(f"| {m.name} | {m.formatted_value} | {target} | {status} |")
            lines.append("")

        # Issues
        if report.issues:
            lines.append("### Issues")
            lines.append("")
            for issue in report.issues:
                severity_icon = {
                    Severity.CRITICAL: "🔴",
                    Severity.MAJOR: "🟠",
                    Severity.MINOR: "🟡",
                    Severity.INFO: "🔵",
                }[issue.severity]
                lines.append(f"- {severity_icon} **{issue.title}**: {issue.description}")
                if issue.suggestion:
                    lines.append(f"  - *Suggestion:* {issue.suggestion}")
            lines.append("")

        # Notes
        if report.notes:
            lines.append("### Notes")
            lines.append("")
            for note in report.notes:
                lines.append(f"- {note}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# Report Generator
# ============================================================================


class QAReportGenerator:
    """
    Generates comprehensive QA reports from component results.

    Aggregates technical validation, originality checks, and authenticity
    evaluation into unified reports with actionable insights.

    Example:
        generator = QAReportGenerator()
        report = generator.generate(
            track_id="track_123",
            title="My Track",
            genre_id="lo-fi-hip-hop",
            technical_result=tech_result,
            originality_result=orig_result,
            authenticity_result=auth_result,
        )
        print(report.to_markdown())
    """

    def __init__(
        self,
        technical_weight: float = 0.35,
        originality_weight: float = 0.30,
        authenticity_weight: float = 0.35,
    ):
        """
        Initialize report generator.

        Args:
            technical_weight: Weight for technical score (0-1)
            originality_weight: Weight for originality score (0-1)
            authenticity_weight: Weight for authenticity score (0-1)
        """
        # Normalize weights
        total = technical_weight + originality_weight + authenticity_weight
        self.technical_weight = technical_weight / total
        self.originality_weight = originality_weight / total
        self.authenticity_weight = authenticity_weight / total

    def generate(
        self,
        track_id: str = "",
        title: str = "",
        genre_id: str = "",
        technical_result: Any | None = None,
        originality_result: Any | None = None,
        authenticity_result: Any | None = None,
    ) -> QAReport:
        """
        Generate comprehensive QA report.

        Args:
            track_id: Track identifier
            title: Track title
            genre_id: Genre identifier
            technical_result: TechnicalValidationResult
            originality_result: OriginalityResult
            authenticity_result: AuthenticityResult

        Returns:
            Complete QAReport
        """
        report = QAReport(
            track_id=track_id,
            track_title=title,
            genre_id=genre_id,
        )

        # Process each category
        if technical_result:
            report.technical_report = self._process_technical(technical_result)

        if originality_result:
            report.originality_report = self._process_originality(originality_result)

        if authenticity_result:
            report.authenticity_report = self._process_authenticity(authenticity_result)

        # Calculate overall scores
        scores = []
        weights = []

        if report.technical_report:
            scores.append(report.technical_report.score)
            weights.append(self.technical_weight)

        if report.originality_report:
            scores.append(report.originality_report.score)
            weights.append(self.originality_weight)

        if report.authenticity_report:
            scores.append(report.authenticity_report.score)
            weights.append(self.authenticity_weight)

        if scores:
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                report.overall_score = sum(s * w / total_weight for s, w in zip(scores, weights))
            else:
                report.overall_score = sum(scores) / len(scores)

        # Aggregate issues
        if report.technical_report:
            report.all_issues.extend(report.technical_report.issues)
        if report.originality_report:
            report.all_issues.extend(report.originality_report.issues)
        if report.authenticity_report:
            report.all_issues.extend(report.authenticity_report.issues)

        # Determine pass/fail
        critical_issues = [i for i in report.all_issues if i.severity == Severity.CRITICAL]
        [i for i in report.all_issues if i.severity == Severity.MAJOR]

        # Pass conditions: no critical, score >= 0.8
        report.overall_passed = len(critical_issues) == 0 and report.overall_score >= 0.8

        # Generate executive summary
        report.executive_summary = self._generate_executive_summary(report)

        # Prioritize improvements
        report.improvement_priority = self._prioritize_improvements(report)

        return report

    def _process_technical(self, result: Any) -> CategoryReport:
        """Process technical validation result."""
        metrics = []
        issues = []

        # Handle TechnicalReport (has checks list)
        if hasattr(result, "checks"):
            for check in result.checks:
                # Convert check to metric
                severity_map = {
                    "critical": Severity.CRITICAL,
                    "warning": Severity.MAJOR,
                    "info": Severity.INFO,
                }
                severity = severity_map.get(check.severity.value, Severity.INFO)

                metrics.append(
                    MetricResult(
                        name=check.check_type.value.replace("_", " ").title(),
                        value=check.measured_value,
                        unit=check.unit,
                        target=check.target_value,
                        tolerance=check.tolerance,
                        passed=check.passed,
                        severity=severity if not check.passed else Severity.INFO,
                    )
                )

                # Add issue for failed checks
                if not check.passed:
                    issues.append(
                        QAIssue(
                            category=QACategory.TECHNICAL,
                            severity=severity,
                            title=check.check_type.value.replace("_", " ").title(),
                            description=check.details,
                            suggestion=check.recommendation or "",
                            metric_name=check.check_type.value,
                            actual_value=check.measured_value,
                            expected_value=check.target_value,
                            threshold=check.tolerance,
                        )
                    )

        # TechnicalReport uses all_critical_passed and all_passed
        if hasattr(result, "all_critical_passed"):
            passed = result.all_critical_passed
            # Calculate score from passed/failed checks
            total_checks = len(result.checks) if result.checks else 1
            passed_checks = sum(1 for c in result.checks if c.passed)
            score = passed_checks / total_checks
        else:
            passed = result.passed if hasattr(result, "passed") else True
            score = (
                result.overall_score
                if hasattr(result, "overall_score")
                else (1.0 if passed else 0.5)
            )

        return CategoryReport(
            category=QACategory.TECHNICAL,
            passed=passed,
            score=score,
            metrics=metrics,
            issues=issues,
        )

    def _process_originality(self, result: Any) -> CategoryReport:
        """Process originality check result."""
        metrics = []
        issues = []

        # Handle List[OriginalityResult] from OriginalityChecker.check_all()
        if isinstance(result, list):
            for r in result:
                if hasattr(r, "check_type") and hasattr(r, "score"):
                    check_name = r.check_type.value.replace("_", " ").title()
                    target = (
                        0.85
                        if "melody" in r.check_type.value
                        else 0.97 if "lyric" in r.check_type.value else 0.80
                    )

                    metrics.append(
                        MetricResult(
                            name=check_name,
                            value=r.score,
                            target=target,
                            passed=r.passed,
                            severity=Severity.CRITICAL if not r.passed else Severity.INFO,
                        )
                    )

                    if not r.passed:
                        issues.append(
                            QAIssue(
                                category=QACategory.ORIGINALITY,
                                severity=Severity.CRITICAL,
                                title=f"{check_name} Issue",
                                description=(
                                    r.details if r.details else f"{check_name} check failed"
                                ),
                                suggestion=f"Review and modify {check_name.lower()} content",
                            )
                        )

        # Handle single OriginalityResult (legacy support)
        elif hasattr(result, "overall_score"):
            metrics.append(
                MetricResult(
                    name="Overall Originality",
                    value=result.overall_score,
                    target=0.85,
                    passed=result.overall_score >= 0.85,
                )
            )

        if hasattr(result, "melody_originality"):
            passed = result.melody_originality >= 0.85
            metrics.append(
                MetricResult(
                    name="Melody Originality",
                    value=result.melody_originality,
                    target=0.85,
                    passed=passed,
                    severity=Severity.CRITICAL if not passed else Severity.INFO,
                )
            )

            if not passed:
                issues.append(
                    QAIssue(
                        category=QACategory.ORIGINALITY,
                        severity=Severity.CRITICAL,
                        title="Melody Similarity Detected",
                        description="Melody shows high similarity to existing works",
                        suggestion="Modify melodic intervals or contour for uniqueness",
                    )
                )

        if hasattr(result, "lyric_originality"):
            passed = result.lyric_originality >= 0.97  # N-gram threshold
            metrics.append(
                MetricResult(
                    name="Lyric Originality",
                    value=result.lyric_originality,
                    target=0.97,
                    passed=passed,
                    severity=Severity.CRITICAL if not passed else Severity.INFO,
                )
            )

            if not passed:
                issues.append(
                    QAIssue(
                        category=QACategory.ORIGINALITY,
                        severity=Severity.CRITICAL,
                        title="Lyric Overlap Detected",
                        description="Lyrics show significant n-gram overlap with references",
                        suggestion="Rewrite flagged phrases for originality",
                    )
                )

        if hasattr(result, "harmony_originality"):
            metrics.append(
                MetricResult(
                    name="Harmony Originality",
                    value=result.harmony_originality,
                    target=0.80,
                    passed=result.harmony_originality >= 0.80,
                )
            )

        if hasattr(result, "audio_similarity"):
            passed = result.audio_similarity <= 0.15
            metrics.append(
                MetricResult(
                    name="Audio Similarity",
                    value=result.audio_similarity,
                    target=0.15,
                    passed=passed,
                    severity=Severity.CRITICAL if not passed else Severity.INFO,
                )
            )

            if not passed:
                issues.append(
                    QAIssue(
                        category=QACategory.ORIGINALITY,
                        severity=Severity.CRITICAL,
                        title="Audio Similarity Detected",
                        description="Audio embedding shows high similarity to reference",
                        suggestion="Modify production elements or arrangement",
                    )
                )

        # Calculate pass/score for list results
        if isinstance(result, list) and result:
            passed = all(r.passed for r in result if hasattr(r, "passed"))
            total_score = sum(r.score for r in result if hasattr(r, "score"))
            score = total_score / len(result)
        else:
            passed = result.passed if hasattr(result, "passed") else True
            score = (
                result.overall_score
                if hasattr(result, "overall_score")
                else (1.0 if passed else 0.5)
            )

        return CategoryReport(
            category=QACategory.ORIGINALITY,
            passed=passed,
            score=score,
            metrics=metrics,
            issues=issues,
        )

    def _process_authenticity(self, result: Any) -> CategoryReport:
        """Process authenticity evaluation result."""
        metrics = []
        issues = []
        notes = []

        # Handle AuthenticityResult
        if hasattr(result, "overall_score"):
            metrics.append(
                MetricResult(
                    name="Overall Authenticity",
                    value=result.overall_score,
                    target=0.80,
                    passed=result.overall_score >= 0.80,
                )
            )

        # Dimension scores
        if hasattr(result, "dimension_scores"):
            for ds in result.dimension_scores:
                passed = ds.raw_score >= 3.0
                metrics.append(
                    MetricResult(
                        name=ds.dimension_name,
                        value=ds.raw_score / 5.0,  # Normalize to 0-1
                        target=0.6,  # 3/5
                        passed=passed,
                        severity=Severity.MAJOR if not passed else Severity.INFO,
                    )
                )

                if not passed:
                    for suggestion in ds.improvement_suggestions[:1]:
                        issues.append(
                            QAIssue(
                                category=QACategory.AUTHENTICITY,
                                severity=Severity.MAJOR,
                                title=f"{ds.dimension_name} Below Standard",
                                description=f"Score: {ds.raw_score:.1f}/5",
                                suggestion=suggestion,
                            )
                        )

        # Strengths and weaknesses
        if hasattr(result, "top_strengths") and result.top_strengths:
            notes.append(f"Strengths: {', '.join(result.top_strengths[:3])}")

        if hasattr(result, "top_weaknesses") and result.top_weaknesses:
            notes.append(f"Weaknesses: {', '.join(result.top_weaknesses[:3])}")

        passed = result.passed if hasattr(result, "passed") else True
        score = (
            result.overall_score if hasattr(result, "overall_score") else (1.0 if passed else 0.5)
        )

        return CategoryReport(
            category=QACategory.AUTHENTICITY,
            passed=passed,
            score=score,
            metrics=metrics,
            issues=issues,
            notes=notes,
        )

    def _generate_executive_summary(self, report: QAReport) -> ExecutiveSummary:
        """Generate executive summary from report."""
        # Count issues by severity
        critical = len([i for i in report.all_issues if i.severity == Severity.CRITICAL])
        major = len([i for i in report.all_issues if i.severity == Severity.MAJOR])
        minor = len([i for i in report.all_issues if i.severity == Severity.MINOR])

        # Determine status
        if critical > 0:
            status = PassFailStatus.FAIL
        elif major > 2 or report.overall_score < 0.8:
            status = PassFailStatus.WARNING
        else:
            status = PassFailStatus.PASS

        # Get category scores
        tech_score = report.technical_report.score if report.technical_report else 0.0
        orig_score = report.originality_report.score if report.originality_report else 0.0
        auth_score = report.authenticity_report.score if report.authenticity_report else 0.0

        # Top issues (prioritize by severity)
        top_issues = []
        sorted_issues = sorted(
            report.all_issues,
            key=lambda i: {
                Severity.CRITICAL: 0,
                Severity.MAJOR: 1,
                Severity.MINOR: 2,
                Severity.INFO: 3,
            }[i.severity],
        )
        for issue in sorted_issues[:5]:
            top_issues.append(f"[{issue.severity.value.upper()}] {issue.title}")

        # Generate recommendation
        if status == PassFailStatus.PASS:
            recommendation = "Track meets all quality standards and is ready for release."
        elif status == PassFailStatus.WARNING:
            recommendation = "Track is acceptable but has room for improvement. Review major issues before release."
        else:
            recommendation = (
                "Track requires fixes before release. Address critical issues immediately."
            )

        return ExecutiveSummary(
            status=status,
            overall_score=report.overall_score,
            technical_score=tech_score,
            originality_score=orig_score,
            authenticity_score=auth_score,
            critical_count=critical,
            major_count=major,
            minor_count=minor,
            top_issues=top_issues,
            recommendation=recommendation,
        )

    def _prioritize_improvements(self, report: QAReport) -> list[str]:
        """Generate prioritized list of improvements."""
        improvements = []

        # Sort issues by severity and category importance
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.MAJOR: 1,
            Severity.MINOR: 2,
            Severity.INFO: 3,
        }

        sorted_issues = sorted(
            [i for i in report.all_issues if i.suggestion], key=lambda i: severity_order[i.severity]
        )

        for issue in sorted_issues:
            if issue.suggestion and issue.suggestion not in improvements:
                improvements.append(issue.suggestion)

        return improvements[:10]


# ============================================================================
# Convenience Functions
# ============================================================================


def generate_qa_report(
    track_id: str = "",
    title: str = "",
    genre_id: str = "",
    technical_result: Any | None = None,
    originality_result: Any | None = None,
    authenticity_result: Any | None = None,
) -> QAReport:
    """
    Generate QA report with default settings.

    Convenience function for quick report generation.
    """
    generator = QAReportGenerator()
    return generator.generate(
        track_id=track_id,
        title=title,
        genre_id=genre_id,
        technical_result=technical_result,
        originality_result=originality_result,
        authenticity_result=authenticity_result,
    )


def quick_check(
    technical_passed: bool = True,
    originality_passed: bool = True,
    authenticity_passed: bool = True,
) -> bool:
    """
    Quick pass/fail check without full report generation.

    Returns True only if all checks pass.
    """
    return technical_passed and originality_passed and authenticity_passed


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Enums
    "Severity",
    "QACategory",
    "PassFailStatus",
    # Data structures
    "QAIssue",
    "MetricResult",
    "CategoryReport",
    "ExecutiveSummary",
    "QAReport",
    # Generator
    "QAReportGenerator",
    # Functions
    "generate_qa_report",
    "quick_check",
]
