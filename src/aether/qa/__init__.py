"""
AETHER Quality Assurance System

Production-grade QA for AI-generated music ensuring commercial viability.

Components:
- Technical Validation: Loudness, peak, phase, spectrum analysis
- Originality Checking: Melody fingerprinting, lyric n-gram, audio embedding
- Genre Authenticity: Multi-dimensional rubric evaluation
- Report Generation: Comprehensive QA reports with actionable insights

Standards:
- Technical: ITU-R BS.1770-4 loudness, AES-17 true peak
- Originality: Embedding similarity < 0.15, n-gram overlap < 3%
- Authenticity: Genre rubric score >= 80%

Example Usage:
    from aether.qa import QAAgent, QAConfig

    # Initialize QA system
    qa = QAAgent(config=QAConfig())

    # Full QA evaluation
    report = qa.evaluate_track(
        audio_path="track.wav",
        genre_id="lo-fi-hip-hop",
        melody_data=melody_artifacts,
        lyrics=lyrics_text,
    )

    if report.overall_passed:
        print("Track ready for release!")
    else:
        print("Issues found:")
        for issue in report.improvement_priority:
            print(f"  - {issue}")
"""

# Technical Validation
# Main Integration
from aether.qa.agent import (
    QAAgent,
    QAConfig,
    QAThresholds,
)

# Genre Authenticity
from aether.qa.authenticity import (
    ArrangementAnalysis,
    AuthenticityResult,
    DimensionScore,
    GenreAuthenticityEvaluator,
    HarmonyAnalysis,
    MelodyAnalysis,
    ProductionAnalysis,
    RhythmAnalysis,
    ScoreLevel,
    TrackAnalysis,
    create_track_analysis_from_artifacts,
    evaluate_genre_authenticity,
)

# Originality Checking
from aether.qa.originality import (
    AudioEmbeddingChecker,
    HarmonyOriginalityChecker,
    LyricOriginalityChecker,
    MelodyFingerprint,
    MelodyOriginalityChecker,
    OriginalityChecker,
    OriginalityCheckType,
    OriginalityResult,
)

# Report Generation
from aether.qa.reports import (
    CategoryReport,
    ExecutiveSummary,
    MetricResult,
    PassFailStatus,
    QACategory,
    QAIssue,
    QAReport,
    QAReportGenerator,
    Severity,
    generate_qa_report,
)
from aether.qa.technical import (
    AudioQualityValidator,
    CheckSeverity,
    LoudnessValidator,
    SpectralValidator,
    StereoValidator,
    TechnicalCheckResult,
    TechnicalCheckType,
    TechnicalReport,
    TechnicalValidator,
)

__all__ = [
    # Main Agent
    "QAAgent",
    "QAConfig",
    "QAThresholds",
    # Technical
    "TechnicalValidator",
    "TechnicalReport",
    "TechnicalCheckResult",
    "TechnicalCheckType",
    "CheckSeverity",
    "LoudnessValidator",
    "StereoValidator",
    "AudioQualityValidator",
    "SpectralValidator",
    # Originality
    "OriginalityChecker",
    "OriginalityResult",
    "OriginalityCheckType",
    "MelodyOriginalityChecker",
    "LyricOriginalityChecker",
    "HarmonyOriginalityChecker",
    "AudioEmbeddingChecker",
    "MelodyFingerprint",
    # Authenticity
    "GenreAuthenticityEvaluator",
    "AuthenticityResult",
    "DimensionScore",
    "ScoreLevel",
    "TrackAnalysis",
    "RhythmAnalysis",
    "HarmonyAnalysis",
    "MelodyAnalysis",
    "ProductionAnalysis",
    "ArrangementAnalysis",
    "evaluate_genre_authenticity",
    "create_track_analysis_from_artifacts",
    # Reports
    "QAReportGenerator",
    "QAReport",
    "CategoryReport",
    "ExecutiveSummary",
    "QAIssue",
    "MetricResult",
    "Severity",
    "QACategory",
    "PassFailStatus",
    "generate_qa_report",
]
