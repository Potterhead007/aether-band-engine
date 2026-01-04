"""
AETHER Technical Validator

Production-grade technical audio quality verification.
Validates against broadcast and streaming platform specifications.

Checks:
- Loudness (ITU-R BS.1770-4, EBU R128)
- True Peak (AES-17)
- Dynamic Range
- Phase Correlation / Mono Compatibility
- DC Offset
- Clipping Detection
- Frequency Response (spectral balance)
- Silence Detection
- Noise Floor

Target Specifications (Mission Requirements):
- Integrated Loudness: -14.0 LUFS (±0.5)
- True Peak: < -1.0 dBTP
- Dynamic Range: > 6 LU
- Phase Correlation: > 0.3
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aether.audio.dsp import (
    LoudnessMeter,
    LoudnessMeasurement,
    StereoProcessor,
    BiquadFilter,
    FilterType,
    db_to_linear,
    linear_to_db,
)

logger = logging.getLogger(__name__)


class TechnicalCheckType(str, Enum):
    """Types of technical checks."""
    LOUDNESS_INTEGRATED = "loudness_integrated"
    LOUDNESS_SHORT_TERM = "loudness_short_term"
    LOUDNESS_RANGE = "loudness_range"
    TRUE_PEAK = "true_peak"
    SAMPLE_PEAK = "sample_peak"
    DYNAMIC_RANGE = "dynamic_range"
    PHASE_CORRELATION = "phase_correlation"
    MONO_COMPATIBILITY = "mono_compatibility"
    DC_OFFSET = "dc_offset"
    CLIPPING = "clipping"
    NOISE_FLOOR = "noise_floor"
    SILENCE = "silence"
    SPECTRAL_BALANCE = "spectral_balance"
    CREST_FACTOR = "crest_factor"


class CheckSeverity(str, Enum):
    """Severity of check failures."""
    CRITICAL = "critical"   # Must pass for release
    WARNING = "warning"     # Should pass, can release with note
    INFO = "info"          # Informational only


@dataclass
class TechnicalCheckResult:
    """Result of a technical check."""
    check_type: TechnicalCheckType
    severity: CheckSeverity
    measured_value: float
    target_value: float
    tolerance: float
    passed: bool
    unit: str
    details: str
    recommendation: Optional[str] = None


@dataclass
class TechnicalReport:
    """Complete technical validation report."""
    sample_rate: int
    bit_depth: int
    channels: int
    duration_seconds: float
    checks: List[TechnicalCheckResult]
    all_critical_passed: bool
    all_passed: bool
    summary: str


class LoudnessValidator:
    """
    Validates loudness per ITU-R BS.1770-4 and EBU R128.

    Target Specifications:
    - Streaming: -14.0 LUFS (±0.5)
    - Broadcast EU: -23.0 LUFS (±1.0)
    - Broadcast US: -24.0 LKFS (±2.0)
    """

    def __init__(
        self,
        sample_rate: float,
        target_lufs: float = -14.0,
        tolerance_lufs: float = 0.5,
        target_peak_dbtp: float = -1.0,
        min_dynamic_range_lu: float = 6.0,
    ):
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.tolerance_lufs = tolerance_lufs
        self.target_peak_dbtp = target_peak_dbtp
        self.min_dynamic_range_lu = min_dynamic_range_lu

        self.meter = LoudnessMeter(sample_rate)

    def validate(self, audio: np.ndarray) -> List[TechnicalCheckResult]:
        """
        Validate audio loudness.

        Returns list of check results.
        """
        results = []

        # Measure
        self.meter.reset()
        measurement = self.meter.measure(audio)

        # Integrated loudness
        lufs_diff = abs(measurement.integrated_lufs - self.target_lufs)
        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.LOUDNESS_INTEGRATED,
            severity=CheckSeverity.CRITICAL,
            measured_value=measurement.integrated_lufs,
            target_value=self.target_lufs,
            tolerance=self.tolerance_lufs,
            passed=lufs_diff <= self.tolerance_lufs,
            unit="LUFS",
            details=f"Integrated loudness: {measurement.integrated_lufs:.1f} LUFS",
            recommendation="Adjust limiter threshold or input gain" if lufs_diff > self.tolerance_lufs else None,
        ))

        # True peak
        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.TRUE_PEAK,
            severity=CheckSeverity.CRITICAL,
            measured_value=measurement.true_peak_dbtp,
            target_value=self.target_peak_dbtp,
            tolerance=0.0,
            passed=measurement.true_peak_dbtp <= self.target_peak_dbtp,
            unit="dBTP",
            details=f"True peak: {measurement.true_peak_dbtp:.1f} dBTP",
            recommendation="Lower ceiling or add limiting" if measurement.true_peak_dbtp > self.target_peak_dbtp else None,
        ))

        # Sample peak
        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.SAMPLE_PEAK,
            severity=CheckSeverity.WARNING,
            measured_value=measurement.sample_peak_db,
            target_value=-0.1,
            tolerance=0.0,
            passed=measurement.sample_peak_db <= -0.1,
            unit="dBFS",
            details=f"Sample peak: {measurement.sample_peak_db:.1f} dBFS",
        ))

        # Loudness range
        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.LOUDNESS_RANGE,
            severity=CheckSeverity.WARNING,
            measured_value=measurement.loudness_range_lu,
            target_value=8.0,
            tolerance=2.0,
            passed=measurement.loudness_range_lu >= self.min_dynamic_range_lu,
            unit="LU",
            details=f"Loudness range: {measurement.loudness_range_lu:.1f} LU",
            recommendation="Reduce compression to increase dynamics" if measurement.loudness_range_lu < self.min_dynamic_range_lu else None,
        ))

        return results


class StereoValidator:
    """
    Validates stereo characteristics.

    Checks:
    - Phase correlation
    - Mono compatibility
    - Stereo width
    """

    def validate(self, audio: np.ndarray) -> List[TechnicalCheckResult]:
        """Validate stereo characteristics."""
        results = []

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.array([audio, audio])
        elif audio.shape[0] != 2:
            audio = audio.T if audio.shape[1] == 2 else np.array([audio[0], audio[0]])

        # Phase correlation
        correlation = StereoProcessor.calculate_correlation(audio)
        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.PHASE_CORRELATION,
            severity=CheckSeverity.WARNING,
            measured_value=correlation,
            target_value=0.5,
            tolerance=0.2,
            passed=correlation > 0.3,
            unit="",
            details=f"Stereo correlation: {correlation:.3f}",
            recommendation="Check for phase issues between L/R channels" if correlation < 0.3 else None,
        ))

        # Mono compatibility
        mono_ok, _ = StereoProcessor.check_mono_compatibility(audio)
        mono_sum = (audio[0] + audio[1]) / 2
        mono_level = linear_to_db(np.max(np.abs(mono_sum)))
        stereo_level = linear_to_db(max(np.max(np.abs(audio[0])), np.max(np.abs(audio[1]))))
        mono_loss = stereo_level - mono_level

        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.MONO_COMPATIBILITY,
            severity=CheckSeverity.WARNING,
            measured_value=mono_loss,
            target_value=0.0,
            tolerance=3.0,
            passed=mono_ok and mono_loss < 3.0,
            unit="dB",
            details=f"Mono fold-down loss: {mono_loss:.1f} dB",
            recommendation="Phase cancellation detected - check stereo processing" if mono_loss >= 3.0 else None,
        ))

        return results


class AudioQualityValidator:
    """
    Validates general audio quality metrics.

    Checks:
    - DC offset
    - Clipping
    - Noise floor
    - Silence detection
    - Crest factor
    """

    def validate(
        self,
        audio: np.ndarray,
        sample_rate: float,
    ) -> List[TechnicalCheckResult]:
        """Validate audio quality."""
        results = []

        # Ensure correct format
        if audio.ndim == 1:
            audio = np.array([audio, audio])
        elif audio.shape[0] != 2:
            audio = audio.T if audio.shape[1] == 2 else np.array([audio[0], audio[0]])

        # DC offset check
        dc_offset_l = np.mean(audio[0])
        dc_offset_r = np.mean(audio[1])
        max_dc = max(abs(dc_offset_l), abs(dc_offset_r))
        dc_db = linear_to_db(max_dc) if max_dc > 0 else -100

        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.DC_OFFSET,
            severity=CheckSeverity.WARNING,
            measured_value=max_dc,
            target_value=0.0,
            tolerance=0.001,
            passed=max_dc < 0.001,
            unit="",
            details=f"DC offset: {max_dc:.6f} ({dc_db:.1f} dB)",
            recommendation="Apply DC offset removal filter" if max_dc >= 0.001 else None,
        ))

        # Clipping detection
        clip_threshold = 0.9999
        clipped_samples_l = np.sum(np.abs(audio[0]) >= clip_threshold)
        clipped_samples_r = np.sum(np.abs(audio[1]) >= clip_threshold)
        total_clipped = clipped_samples_l + clipped_samples_r
        clip_percentage = (total_clipped / (audio.shape[1] * 2)) * 100

        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.CLIPPING,
            severity=CheckSeverity.CRITICAL,
            measured_value=clip_percentage,
            target_value=0.0,
            tolerance=0.01,
            passed=clip_percentage < 0.01,
            unit="%",
            details=f"Clipping: {total_clipped} samples ({clip_percentage:.4f}%)",
            recommendation="Reduce gain to eliminate clipping" if total_clipped > 0 else None,
        ))

        # Noise floor estimation (using quietest portions)
        block_size = int(0.1 * sample_rate)  # 100ms blocks
        n_blocks = audio.shape[1] // block_size

        if n_blocks > 0:
            block_rms = []
            for i in range(n_blocks):
                start = i * block_size
                block = audio[:, start:start + block_size]
                rms = np.sqrt(np.mean(block ** 2))
                if rms > 1e-10:
                    block_rms.append(rms)

            if block_rms:
                # 5th percentile as noise floor estimate
                noise_floor = np.percentile(block_rms, 5)
                noise_floor_db = linear_to_db(noise_floor)
            else:
                noise_floor_db = -100
        else:
            noise_floor_db = -100

        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.NOISE_FLOOR,
            severity=CheckSeverity.INFO,
            measured_value=noise_floor_db,
            target_value=-60.0,
            tolerance=10.0,
            passed=noise_floor_db < -50.0,
            unit="dBFS",
            details=f"Estimated noise floor: {noise_floor_db:.1f} dBFS",
            recommendation="Apply noise reduction" if noise_floor_db > -50.0 else None,
        ))

        # Silence detection at start/end
        silence_threshold = 0.001
        start_silence = 0
        end_silence = 0

        for i in range(min(audio.shape[1], int(sample_rate))):  # First second
            if max(abs(audio[0, i]), abs(audio[1, i])) < silence_threshold:
                start_silence += 1
            else:
                break

        for i in range(audio.shape[1] - 1, max(0, audio.shape[1] - int(sample_rate)), -1):
            if max(abs(audio[0, i]), abs(audio[1, i])) < silence_threshold:
                end_silence += 1
            else:
                break

        start_silence_ms = (start_silence / sample_rate) * 1000
        end_silence_ms = (end_silence / sample_rate) * 1000

        # Some silence is expected for fades
        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.SILENCE,
            severity=CheckSeverity.INFO,
            measured_value=start_silence_ms,
            target_value=0.0,
            tolerance=500.0,
            passed=start_silence_ms < 1000,  # Less than 1 second
            unit="ms",
            details=f"Start silence: {start_silence_ms:.0f}ms, End silence: {end_silence_ms:.0f}ms",
            recommendation="Trim excessive silence" if start_silence_ms > 500 else None,
        ))

        # Crest factor (peak to RMS ratio)
        peak = max(np.max(np.abs(audio[0])), np.max(np.abs(audio[1])))
        rms = np.sqrt(np.mean(audio ** 2))
        crest_factor = peak / rms if rms > 0 else 0
        crest_factor_db = 20 * math.log10(crest_factor) if crest_factor > 0 else 0

        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.CREST_FACTOR,
            severity=CheckSeverity.INFO,
            measured_value=crest_factor_db,
            target_value=12.0,
            tolerance=6.0,
            passed=crest_factor_db >= 6.0,
            unit="dB",
            details=f"Crest factor: {crest_factor_db:.1f} dB",
            recommendation="Audio may be over-compressed" if crest_factor_db < 6.0 else None,
        ))

        return results


class SpectralValidator:
    """
    Validates frequency response and spectral balance.

    Checks for:
    - Sub-bass content (20-60 Hz)
    - Bass (60-250 Hz)
    - Mids (250-4000 Hz)
    - Presence (4000-8000 Hz)
    - Brilliance (8000-20000 Hz)
    """

    BANDS = [
        ("sub_bass", 20, 60),
        ("bass", 60, 250),
        ("low_mid", 250, 500),
        ("mid", 500, 2000),
        ("high_mid", 2000, 4000),
        ("presence", 4000, 8000),
        ("brilliance", 8000, 20000),
    ]

    def validate(
        self,
        audio: np.ndarray,
        sample_rate: float,
    ) -> List[TechnicalCheckResult]:
        """Validate spectral balance."""
        results = []

        # Mix to mono for spectral analysis
        if audio.ndim > 1:
            mono = np.mean(audio, axis=0)
        else:
            mono = audio

        # Compute spectrum
        n_fft = min(len(mono), 8192)
        spectrum = np.abs(np.fft.rfft(mono, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)

        # Calculate energy per band
        band_energies = {}
        total_energy = np.sum(spectrum ** 2)

        for name, low, high in self.BANDS:
            mask = (freqs >= low) & (freqs < high)
            band_energy = np.sum(spectrum[mask] ** 2)
            band_energies[name] = band_energy / total_energy if total_energy > 0 else 0

        # Check for spectral issues
        # Sub-bass should be present but not dominant
        results.append(TechnicalCheckResult(
            check_type=TechnicalCheckType.SPECTRAL_BALANCE,
            severity=CheckSeverity.INFO,
            measured_value=band_energies.get("sub_bass", 0) * 100,
            target_value=10.0,
            tolerance=10.0,
            passed=band_energies.get("sub_bass", 0) < 0.3,  # Less than 30%
            unit="%",
            details=f"Sub-bass energy: {band_energies.get('sub_bass', 0)*100:.1f}%",
            recommendation="Excessive sub-bass detected" if band_energies.get("sub_bass", 0) > 0.3 else None,
        ))

        # Build spectral summary
        spectral_summary = ", ".join([
            f"{name}: {energy*100:.1f}%"
            for name, energy in band_energies.items()
        ])

        logger.debug(f"Spectral balance: {spectral_summary}")

        return results


class TechnicalValidator:
    """
    Main technical validator combining all checks.

    Usage:
        validator = TechnicalValidator(sample_rate=48000)
        report = validator.validate(audio)
    """

    def __init__(
        self,
        sample_rate: float = 48000,
        target_lufs: float = -14.0,
        tolerance_lufs: float = 0.5,
        target_peak_dbtp: float = -1.0,
        min_dynamic_range_lu: float = 6.0,
    ):
        self.sample_rate = sample_rate

        self.loudness_validator = LoudnessValidator(
            sample_rate=sample_rate,
            target_lufs=target_lufs,
            tolerance_lufs=tolerance_lufs,
            target_peak_dbtp=target_peak_dbtp,
            min_dynamic_range_lu=min_dynamic_range_lu,
        )
        self.stereo_validator = StereoValidator()
        self.quality_validator = AudioQualityValidator()
        self.spectral_validator = SpectralValidator()

    def validate(
        self,
        audio: np.ndarray,
        bit_depth: int = 24,
    ) -> TechnicalReport:
        """
        Run all technical validations.

        Args:
            audio: Audio data (2, samples) or (samples,)
            bit_depth: Bit depth of audio

        Returns:
            TechnicalReport with all check results
        """
        # Normalize format
        if audio.ndim == 1:
            audio = np.array([audio, audio])
        elif audio.shape[0] != 2:
            audio = audio.T if audio.shape[1] == 2 else np.array([audio[0], audio[0]])

        duration = audio.shape[1] / self.sample_rate

        logger.info(f"Running technical validation: {duration:.1f}s, {self.sample_rate}Hz")

        # Collect all checks
        all_checks = []

        # Loudness checks (CRITICAL)
        all_checks.extend(self.loudness_validator.validate(audio))

        # Stereo checks (WARNING)
        all_checks.extend(self.stereo_validator.validate(audio))

        # Quality checks (mixed severity)
        all_checks.extend(self.quality_validator.validate(audio, self.sample_rate))

        # Spectral checks (INFO)
        all_checks.extend(self.spectral_validator.validate(audio, self.sample_rate))

        # Determine overall pass/fail
        all_critical_passed = all(
            c.passed for c in all_checks
            if c.severity == CheckSeverity.CRITICAL
        )
        all_passed = all(c.passed for c in all_checks)

        # Build summary
        failed_critical = [c for c in all_checks if c.severity == CheckSeverity.CRITICAL and not c.passed]
        failed_warnings = [c for c in all_checks if c.severity == CheckSeverity.WARNING and not c.passed]

        if all_passed:
            summary = "All technical checks passed"
        elif all_critical_passed:
            summary = f"Passed with {len(failed_warnings)} warnings"
        else:
            summary = f"FAILED: {len(failed_critical)} critical issues"

        return TechnicalReport(
            sample_rate=int(self.sample_rate),
            bit_depth=bit_depth,
            channels=2,
            duration_seconds=duration,
            checks=all_checks,
            all_critical_passed=all_critical_passed,
            all_passed=all_passed,
            summary=summary,
        )

    def get_measurements(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Get raw measurements without pass/fail evaluation.

        Useful for displaying meters.
        """
        if audio.ndim == 1:
            audio = np.array([audio, audio])
        elif audio.shape[0] != 2:
            audio = audio.T if audio.shape[1] == 2 else np.array([audio[0], audio[0]])

        self.loudness_validator.meter.reset()
        measurement = self.loudness_validator.meter.measure(audio)

        correlation = StereoProcessor.calculate_correlation(audio)

        peak = max(np.max(np.abs(audio[0])), np.max(np.abs(audio[1])))
        rms = np.sqrt(np.mean(audio ** 2))

        return {
            "integrated_lufs": measurement.integrated_lufs,
            "short_term_lufs": measurement.short_term_lufs,
            "momentary_lufs": measurement.momentary_lufs,
            "true_peak_dbtp": measurement.true_peak_dbtp,
            "sample_peak_db": measurement.sample_peak_db,
            "loudness_range_lu": measurement.loudness_range_lu,
            "phase_correlation": correlation,
            "peak_linear": peak,
            "rms_linear": rms,
            "crest_factor_db": 20 * math.log10(peak / rms) if rms > 0 else 0,
        }
