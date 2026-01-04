"""
AETHER QA Agent

The main integration point for the quality assurance system.
Orchestrates technical validation, originality checking, authenticity
evaluation, and report generation.

This agent is called at the end of the music production pipeline to
ensure tracks meet commercial release standards.

Example:
    qa_agent = QAAgent()

    # Full evaluation
    report = qa_agent.evaluate_track(
        audio=audio_buffer,
        sample_rate=48000,
        genre_id="lo-fi-hip-hop",
        melody_data={"intervals": [...], "contour": "wave"},
        harmony_data={"mode": "minor", "progressions": ["ii-V-I"]},
        lyrics="Verse lyrics here...",
    )

    if report.overall_passed:
        print("Ready for release!")
    else:
        print(f"Score: {report.overall_score:.0%}")
        for fix in report.improvement_priority[:3]:
            print(f"  - {fix}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from aether.qa.technical import (
    TechnicalValidator,
    TechnicalReport,
)
from aether.qa.originality import (
    OriginalityChecker,
    OriginalityResult,
)
from typing import Tuple
from aether.qa.authenticity import (
    GenreAuthenticityEvaluator,
    AuthenticityResult,
    TrackAnalysis,
    RhythmAnalysis,
    HarmonyAnalysis,
    MelodyAnalysis,
    ProductionAnalysis,
    ArrangementAnalysis,
    create_track_analysis_from_artifacts,
)
from aether.qa.reports import (
    QAReportGenerator,
    QAReport,
    generate_qa_report,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class QAThresholds:
    """Thresholds for QA pass/fail decisions."""

    # Technical thresholds
    target_lufs: float = -14.0
    lufs_tolerance: float = 0.5
    max_true_peak_dbtp: float = -1.0
    min_dynamic_range_lu: float = 6.0
    min_stereo_correlation: float = 0.2
    max_dc_offset: float = 0.001

    # Originality thresholds
    max_melody_similarity: float = 0.15
    max_lyric_ngram_overlap: float = 0.03  # 3%
    max_audio_embedding_similarity: float = 0.15

    # Authenticity thresholds
    min_authenticity_score: float = 0.80

    # Overall
    min_overall_score: float = 0.80


@dataclass
class QAConfig:
    """Configuration for QA Agent."""

    thresholds: QAThresholds = field(default_factory=QAThresholds)

    # Weight distribution for overall score
    technical_weight: float = 0.35
    originality_weight: float = 0.30
    authenticity_weight: float = 0.35

    # Component toggles
    enable_technical: bool = True
    enable_originality: bool = True
    enable_authenticity: bool = True

    # Report settings
    verbose_reports: bool = True
    include_suggestions: bool = True

    # Reference database paths (for originality checking)
    melody_reference_db: Optional[Path] = None
    lyric_reference_db: Optional[Path] = None
    audio_embedding_db: Optional[Path] = None


# ============================================================================
# QA Agent
# ============================================================================


class QAAgent:
    """
    Orchestrates quality assurance for AI-generated music.

    This is the main entry point for the QA system, coordinating:
    - Technical validation (loudness, peak, phase, spectrum)
    - Originality checking (melody, lyrics, audio)
    - Genre authenticity evaluation (rubric scoring)
    - Report generation (comprehensive analysis)

    The agent is designed to be called at the end of the production
    pipeline, providing a gate decision on whether a track is ready
    for commercial release.

    Example:
        # Initialize with defaults
        qa = QAAgent()

        # Or with custom config
        config = QAConfig(
            thresholds=QAThresholds(target_lufs=-14.0),
            enable_originality=True,
        )
        qa = QAAgent(config=config)

        # Evaluate a track
        report = qa.evaluate_track(
            audio=audio_array,
            sample_rate=48000,
            genre_id="synthwave",
        )

        if report.overall_passed:
            print("Approved for release!")
    """

    def __init__(self, config: Optional[QAConfig] = None):
        """
        Initialize QA Agent.

        Args:
            config: QA configuration (uses defaults if not provided)
        """
        self.config = config or QAConfig()

        # Initialize component validators
        self._technical_validator = TechnicalValidator(
            sample_rate=48000,  # Default, will be updated per call
            target_lufs=self.config.thresholds.target_lufs,
            tolerance_lufs=self.config.thresholds.lufs_tolerance,
            target_peak_dbtp=self.config.thresholds.max_true_peak_dbtp,
        )

        self._originality_checker = OriginalityChecker()

        self._authenticity_evaluator = GenreAuthenticityEvaluator()

        self._report_generator = QAReportGenerator(
            technical_weight=self.config.technical_weight,
            originality_weight=self.config.originality_weight,
            authenticity_weight=self.config.authenticity_weight,
        )

        # Cache for genre profiles
        self._genre_cache: Dict[str, Any] = {}

        logger.info("QA Agent initialized")

    def evaluate_track(
        self,
        audio: Optional[np.ndarray] = None,
        sample_rate: float = 48000,
        audio_path: Optional[Union[str, Path]] = None,
        genre_id: str = "",
        track_id: str = "",
        title: str = "",
        # Melody data for originality
        melody_data: Optional[Dict[str, Any]] = None,
        # Harmony data
        harmony_data: Optional[Dict[str, Any]] = None,
        # Lyrics for originality
        lyrics: Optional[str] = None,
        # Production data
        production_data: Optional[Dict[str, Any]] = None,
        # Arrangement data
        arrangement_data: Optional[Dict[str, Any]] = None,
        # Rhythm data
        rhythm_data: Optional[Dict[str, Any]] = None,
        # Instruments detected
        instruments: Optional[List[str]] = None,
        # Reference data for originality comparison
        reference_melodies: Optional[List[Dict[str, Any]]] = None,
        reference_lyrics: Optional[List[str]] = None,
        reference_embeddings: Optional[np.ndarray] = None,
    ) -> QAReport:
        """
        Perform full QA evaluation on a track.

        Args:
            audio: Audio buffer (stereo, shape: [2, samples] or [samples, 2])
            sample_rate: Sample rate in Hz
            audio_path: Path to audio file (alternative to audio buffer)
            genre_id: Genre identifier for authenticity evaluation
            track_id: Track identifier for reporting
            title: Track title for reporting
            melody_data: Melody analysis data
            harmony_data: Harmony analysis data
            lyrics: Track lyrics (if any)
            production_data: Production analysis data
            arrangement_data: Arrangement analysis data
            rhythm_data: Rhythm analysis data
            instruments: List of detected instruments
            reference_melodies: Reference melodies for originality
            reference_lyrics: Reference lyrics for originality
            reference_embeddings: Reference audio embeddings

        Returns:
            QAReport with full evaluation results
        """
        logger.info(f"Starting QA evaluation for: {title or track_id or 'Unknown'}")

        # Load audio from path if needed
        if audio is None and audio_path:
            audio = self._load_audio(audio_path, sample_rate)

        # Ensure audio is in correct format [2, samples]
        if audio is not None:
            audio = self._normalize_audio_format(audio)

        # Technical validation
        technical_result = None
        if self.config.enable_technical and audio is not None:
            logger.debug("Running technical validation")
            technical_result = self._run_technical_validation(audio, sample_rate)

        # Originality checking
        originality_result = None
        if self.config.enable_originality:
            logger.debug("Running originality check")
            originality_result = self._run_originality_check(
                melody_data=melody_data,
                lyrics=lyrics,
                audio=audio,
                sample_rate=sample_rate,
                reference_melodies=reference_melodies,
                reference_lyrics=reference_lyrics,
                reference_embeddings=reference_embeddings,
            )

        # Authenticity evaluation
        authenticity_result = None
        if self.config.enable_authenticity and genre_id:
            logger.debug("Running authenticity evaluation")
            authenticity_result = self._run_authenticity_evaluation(
                genre_id=genre_id,
                rhythm_data=rhythm_data,
                harmony_data=harmony_data,
                melody_data=melody_data,
                production_data=production_data,
                arrangement_data=arrangement_data,
                instruments=instruments,
                technical_result=technical_result,
            )

        # Generate report
        logger.debug("Generating QA report")
        report = self._report_generator.generate(
            track_id=track_id,
            title=title,
            genre_id=genre_id,
            technical_result=technical_result,
            originality_result=originality_result,
            authenticity_result=authenticity_result,
        )

        logger.info(
            f"QA evaluation complete. "
            f"Status: {'PASS' if report.overall_passed else 'FAIL'}, "
            f"Score: {report.overall_score:.0%}"
        )

        return report

    def quick_validate(
        self,
        audio: np.ndarray,
        sample_rate: float = 48000,
    ) -> bool:
        """
        Quick technical-only validation.

        Returns True if audio passes basic technical standards.

        Args:
            audio: Audio buffer
            sample_rate: Sample rate in Hz

        Returns:
            True if passes technical validation
        """
        audio = self._normalize_audio_format(audio)
        result = self._run_technical_validation(audio, sample_rate)
        return result.passed if result else False

    def validate_for_platform(
        self,
        audio: np.ndarray,
        sample_rate: float,
        platform: str,
    ) -> QAReport:
        """
        Validate audio for a specific streaming platform.

        Adjusts thresholds based on platform requirements.

        Args:
            audio: Audio buffer
            sample_rate: Sample rate in Hz
            platform: Platform name (spotify, apple, youtube, etc.)

        Returns:
            QAReport with platform-specific validation
        """
        # Platform-specific thresholds
        platform_thresholds = {
            "spotify": QAThresholds(target_lufs=-14.0, max_true_peak_dbtp=-1.0),
            "apple": QAThresholds(target_lufs=-16.0, max_true_peak_dbtp=-1.0),
            "youtube": QAThresholds(target_lufs=-14.0, max_true_peak_dbtp=-1.0),
            "tidal": QAThresholds(target_lufs=-14.0, max_true_peak_dbtp=-1.0),
            "broadcast": QAThresholds(
                target_lufs=-23.0, lufs_tolerance=1.0, max_true_peak_dbtp=-1.0
            ),
        }

        original_thresholds = self.config.thresholds

        try:
            # Apply platform thresholds
            if platform.lower() in platform_thresholds:
                self.config.thresholds = platform_thresholds[platform.lower()]
                self._technical_validator = TechnicalValidator(
                    target_lufs=self.config.thresholds.target_lufs,
                    lufs_tolerance=self.config.thresholds.lufs_tolerance,
                    max_true_peak=self.config.thresholds.max_true_peak_dbtp,
                )

            # Run validation
            return self.evaluate_track(
                audio=audio,
                sample_rate=sample_rate,
                title=f"Platform validation: {platform}",
            )

        finally:
            # Restore original thresholds
            self.config.thresholds = original_thresholds

    def _run_technical_validation(
        self,
        audio: np.ndarray,
        sample_rate: float,
    ) -> TechnicalReport:
        """Run technical validation on audio."""
        # Set the sample rate on the validator
        self._technical_validator.sample_rate = sample_rate
        return self._technical_validator.validate(audio)

    def _run_originality_check(
        self,
        melody_data: Optional[Dict[str, Any]],
        lyrics: Optional[str],
        audio: Optional[np.ndarray],
        sample_rate: float,
        reference_melodies: Optional[List[Dict[str, Any]]],
        reference_lyrics: Optional[List[str]],
        reference_embeddings: Optional[np.ndarray],
    ) -> List[OriginalityResult]:
        """Run originality checking."""
        # Build spec dicts for the checker
        melody_spec = melody_data if melody_data else None
        lyric_spec = {"lyrics": lyrics} if lyrics else None
        harmony_spec = None  # Can be extracted from melody_data if available

        return self._originality_checker.check_all(
            melody_spec=melody_spec,
            lyric_spec=lyric_spec,
            harmony_spec=harmony_spec,
            audio=audio,
            sample_rate=int(sample_rate),
        )

    def _run_authenticity_evaluation(
        self,
        genre_id: str,
        rhythm_data: Optional[Dict[str, Any]],
        harmony_data: Optional[Dict[str, Any]],
        melody_data: Optional[Dict[str, Any]],
        production_data: Optional[Dict[str, Any]],
        arrangement_data: Optional[Dict[str, Any]],
        instruments: Optional[List[str]],
        technical_result: Optional[TechnicalReport],
    ) -> Optional[AuthenticityResult]:
        """Run genre authenticity evaluation."""
        # Get genre profile
        genre_profile = self._get_genre_profile(genre_id)
        if not genre_profile:
            logger.warning(f"Genre profile not found: {genre_id}")
            return None

        # Build track analysis
        analysis = self._build_track_analysis(
            rhythm_data=rhythm_data or {},
            harmony_data=harmony_data or {},
            melody_data=melody_data or {},
            production_data=production_data or {},
            arrangement_data=arrangement_data or {},
            instruments=instruments or [],
            technical_result=technical_result,
            genre_id=genre_id,
        )

        return self._authenticity_evaluator.evaluate(analysis, genre_profile)

    def _get_genre_profile(self, genre_id: str) -> Optional[Any]:
        """Get genre profile, using cache if available."""
        if genre_id in self._genre_cache:
            return self._genre_cache[genre_id]

        try:
            from aether.knowledge import get_genre_manager

            manager = get_genre_manager()
            profile = manager.get(genre_id)
            self._genre_cache[genre_id] = profile
            return profile
        except Exception as e:
            logger.error(f"Failed to load genre profile {genre_id}: {e}")
            return None

    def _build_track_analysis(
        self,
        rhythm_data: Dict[str, Any],
        harmony_data: Dict[str, Any],
        melody_data: Dict[str, Any],
        production_data: Dict[str, Any],
        arrangement_data: Dict[str, Any],
        instruments: List[str],
        technical_result: Optional[TechnicalReport],
        genre_id: str,
    ) -> TrackAnalysis:
        """Build TrackAnalysis from available data."""
        # Extract production data from technical result if available
        if technical_result and technical_result.checks:
            # Extract values from check results
            for check in technical_result.checks:
                if check.check_type.value == "loudness_integrated":
                    production_data.setdefault("lufs", check.measured_value)
                elif check.check_type.value == "loudness_range":
                    production_data.setdefault("dynamic_range", check.measured_value)
                elif check.check_type.value == "phase_correlation":
                    production_data.setdefault("width", check.measured_value)

        return create_track_analysis_from_artifacts(
            rhythm_data=rhythm_data,
            harmony_data=harmony_data,
            melody_data=melody_data,
            production_data=production_data,
            arrangement_data=arrangement_data,
            instruments=instruments,
            genre_id=genre_id,
        )

    def _load_audio(self, path: Union[str, Path], sample_rate: float) -> np.ndarray:
        """Load audio from file."""
        try:
            from aether.audio import read_audio

            audio_file = read_audio(Path(path))
            return audio_file.data
        except ImportError:
            # Fallback to basic loading
            import wave

            with wave.open(str(path), "rb") as f:
                frames = f.readframes(f.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                audio = audio / 32768.0
                if f.getnchannels() == 2:
                    audio = audio.reshape(-1, 2).T
                else:
                    audio = np.stack([audio, audio])
                return audio

    def _normalize_audio_format(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is in [2, samples] format."""
        if audio.ndim == 1:
            # Mono to stereo
            return np.stack([audio, audio])
        elif audio.shape[0] == 2:
            return audio
        elif audio.shape[1] == 2:
            # [samples, 2] to [2, samples]
            return audio.T
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")


# ============================================================================
# Convenience Functions
# ============================================================================


def create_qa_agent(
    target_lufs: float = -14.0,
    enable_originality: bool = True,
    enable_authenticity: bool = True,
) -> QAAgent:
    """
    Create QA agent with common settings.

    Args:
        target_lufs: Target loudness in LUFS
        enable_originality: Enable originality checking
        enable_authenticity: Enable authenticity evaluation

    Returns:
        Configured QAAgent
    """
    config = QAConfig(
        thresholds=QAThresholds(target_lufs=target_lufs),
        enable_originality=enable_originality,
        enable_authenticity=enable_authenticity,
    )
    return QAAgent(config)


def evaluate_track_quick(
    audio: np.ndarray,
    sample_rate: float = 48000,
    genre_id: str = "",
) -> QAReport:
    """
    Quick track evaluation with default settings.

    Args:
        audio: Audio buffer
        sample_rate: Sample rate in Hz
        genre_id: Genre identifier

    Returns:
        QAReport
    """
    agent = QAAgent()
    return agent.evaluate_track(
        audio=audio,
        sample_rate=sample_rate,
        genre_id=genre_id,
    )


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    "QAAgent",
    "QAConfig",
    "QAThresholds",
    "create_qa_agent",
    "evaluate_track_quick",
]
