"""
AETHER Broadcast-Grade Mastering Chain

Production-grade mastering processor meeting international broadcast standards.

Standards Compliance:
- ITU-R BS.1770-4 (Loudness measurement)
- EBU R128 (European loudness normalization)
- ATSC A/85 (North American broadcast)
- AES-17 (True peak metering)

Processing Chain:
1. Input gain staging
2. Linear phase EQ (optional)
3. Multiband dynamics
4. Stereo enhancement (M/S processing)
5. Harmonic exciter (optional)
6. Brick wall limiter (true peak aware)
7. Loudness normalization
8. Dithering and noise shaping (for bit depth reduction)

Target Specifications:
- Integrated loudness: -14.0 LUFS (±0.5 tolerance)
- True peak: < -1.0 dBTP
- Dynamic range: 6-12 LU depending on genre
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from aether.audio.dsp import (
    AnalogSaturator,
    BiquadFilter,
    Compressor,
    ExciterEnhancer,
    FilterType,
    LoudnessMeasurement,
    LoudnessMeter,
    ParametricEQ,
    SaturationType,
    StereoBuffer,
    StereoProcessor,
    SubBassEnhancer,
    TransientShaper,
    TruePeakLimiter,
    db_to_linear,
    linear_to_db,
)

logger = logging.getLogger(__name__)


class DeliveryPlatform(str, Enum):
    """Target delivery platform with loudness specs."""

    SPOTIFY = "spotify"  # -14 LUFS
    APPLE_MUSIC = "apple_music"  # -16 LUFS (with sound check)
    YOUTUBE = "youtube"  # -14 LUFS
    AMAZON_MUSIC = "amazon_music"  # -14 LUFS
    TIDAL = "tidal"  # -14 LUFS
    BROADCAST_EU = "broadcast_eu"  # -23 LUFS (EBU R128)
    BROADCAST_US = "broadcast_us"  # -24 LKFS (ATSC A/85)
    CLUB = "club"  # -9 to -6 LUFS
    CD = "cd"  # -12 to -9 LUFS


@dataclass
class MasteringTarget:
    """Target specifications for mastering."""

    target_lufs: float = -14.0
    tolerance_lufs: float = 0.5
    true_peak_ceiling_dbtp: float = -1.0
    min_dynamic_range_lu: float = 6.0
    target_dynamic_range_lu: float = 8.0

    @classmethod
    def for_platform(cls, platform: DeliveryPlatform) -> MasteringTarget:
        """Get target for specific platform."""
        if platform == DeliveryPlatform.SPOTIFY:
            return cls(target_lufs=-14.0, true_peak_ceiling_dbtp=-1.0)
        elif platform == DeliveryPlatform.APPLE_MUSIC:
            return cls(target_lufs=-16.0, true_peak_ceiling_dbtp=-1.0)
        elif platform == DeliveryPlatform.YOUTUBE:
            return cls(target_lufs=-14.0, true_peak_ceiling_dbtp=-1.0)
        elif platform == DeliveryPlatform.BROADCAST_EU:
            return cls(target_lufs=-23.0, true_peak_ceiling_dbtp=-1.0, target_dynamic_range_lu=12.0)
        elif platform == DeliveryPlatform.BROADCAST_US:
            return cls(target_lufs=-24.0, true_peak_ceiling_dbtp=-2.0, target_dynamic_range_lu=12.0)
        elif platform == DeliveryPlatform.CLUB:
            return cls(target_lufs=-8.0, true_peak_ceiling_dbtp=-0.3, min_dynamic_range_lu=4.0)
        elif platform == DeliveryPlatform.CD:
            return cls(target_lufs=-10.0, true_peak_ceiling_dbtp=-0.1, min_dynamic_range_lu=5.0)
        else:
            return cls()  # Default streaming target


@dataclass
class MultibandCompressorBand:
    """Settings for a single band of multiband compression."""

    name: str
    low_freq: float
    high_freq: float
    threshold_db: float = -20.0
    ratio: float = 2.0
    attack_ms: float = 20.0
    release_ms: float = 200.0
    makeup_gain_db: float = 0.0
    enabled: bool = True


class MultibandCompressor:
    """
    Professional multiband dynamics processor.

    Uses Linkwitz-Riley crossover filters for phase-coherent band splitting.
    Each band has independent dynamics processing.
    """

    def __init__(
        self,
        sample_rate: float,
        bands: list[MultibandCompressorBand],
    ):
        self.sample_rate = sample_rate
        self.band_configs = bands

        # Initialize crossover filters and compressors
        self.crossover_filters: list[tuple[BiquadFilter, BiquadFilter]] = []
        self.compressors: list[Compressor | None] = []

        self._init_crossovers()
        self._init_compressors()

    def _init_crossovers(self) -> None:
        """Initialize Linkwitz-Riley crossover filters."""
        for band in self.band_configs:
            # Linkwitz-Riley 4th order (2x 2nd order Butterworth)
            # Low pass for bottom of band
            if band.low_freq > 20:
                lp1 = BiquadFilter(FilterType.LOWPASS, band.low_freq, self.sample_rate, q=0.707)
                lp2 = BiquadFilter(FilterType.LOWPASS, band.low_freq, self.sample_rate, q=0.707)
            else:
                lp1 = lp2 = None

            # High pass for top of band
            if band.high_freq < 20000:
                hp1 = BiquadFilter(FilterType.HIGHPASS, band.high_freq, self.sample_rate, q=0.707)
                hp2 = BiquadFilter(FilterType.HIGHPASS, band.high_freq, self.sample_rate, q=0.707)
            else:
                hp1 = hp2 = None

            self.crossover_filters.append(((lp1, lp2), (hp1, hp2)))

    def _init_compressors(self) -> None:
        """Initialize per-band compressors."""
        for band in self.band_configs:
            if band.enabled:
                self.compressors.append(
                    Compressor(
                        sample_rate=self.sample_rate,
                        threshold_db=band.threshold_db,
                        ratio=band.ratio,
                        attack_ms=band.attack_ms,
                        release_ms=band.release_ms,
                        makeup_gain_db=band.makeup_gain_db,
                    )
                )
            else:
                self.compressors.append(None)

    def _split_bands(self, audio: StereoBuffer) -> list[StereoBuffer]:
        """Split audio into frequency bands."""
        bands = []

        for i, band_cfg in enumerate(self.band_configs):
            band_audio = audio.copy()

            # Apply highpass (cut frequencies below this band)
            if band_cfg.low_freq > 20:
                hp = BiquadFilter(FilterType.HIGHPASS, band_cfg.low_freq, self.sample_rate, q=0.707)
                band_audio = hp.process_stereo(band_audio)
                # Second pass for LR4
                hp2 = BiquadFilter(
                    FilterType.HIGHPASS, band_cfg.low_freq, self.sample_rate, q=0.707
                )
                band_audio = hp2.process_stereo(band_audio)

            # Apply lowpass (cut frequencies above this band)
            if band_cfg.high_freq < 20000:
                lp = BiquadFilter(FilterType.LOWPASS, band_cfg.high_freq, self.sample_rate, q=0.707)
                band_audio = lp.process_stereo(band_audio)
                # Second pass for LR4
                lp2 = BiquadFilter(
                    FilterType.LOWPASS, band_cfg.high_freq, self.sample_rate, q=0.707
                )
                band_audio = lp2.process_stereo(band_audio)

            bands.append(band_audio)

        return bands

    def process(self, audio: StereoBuffer) -> tuple[StereoBuffer, list[float]]:
        """
        Process audio through multiband compression.

        Returns:
            Tuple of (processed audio, per-band gain reduction in dB)
        """
        # Split into bands
        bands = self._split_bands(audio)

        # Process each band
        processed_bands = []
        gain_reductions = []

        for i, (band_audio, compressor) in enumerate(zip(bands, self.compressors)):
            if compressor is not None:
                processed, gr = compressor.process_stereo(band_audio)
                processed_bands.append(processed)
                gain_reductions.append(np.mean(gr))
            else:
                processed_bands.append(band_audio)
                gain_reductions.append(0.0)

        # Sum bands back together
        output = np.zeros_like(audio)
        for band in processed_bands:
            output += band

        return output, gain_reductions


class HarmonicExciter:
    """
    Subtle harmonic enhancement for adding presence and air.

    Uses soft saturation to generate even and odd harmonics.
    """

    def __init__(
        self,
        sample_rate: float,
        low_freq: float = 300.0,
        high_freq: float = 8000.0,
        drive: float = 0.2,
        mix: float = 0.1,
    ):
        self.sample_rate = sample_rate
        self.drive = drive
        self.mix = mix

        # Bandpass to isolate frequencies for excitation
        self.hp = BiquadFilter(FilterType.HIGHPASS, low_freq, sample_rate, q=0.707)
        self.lp = BiquadFilter(FilterType.LOWPASS, high_freq, sample_rate, q=0.707)

    def _soft_saturate(self, x: float) -> float:
        """Soft saturation function (tanh-like)."""
        return math.tanh(x * self.drive) / self.drive

    def process(self, audio: StereoBuffer) -> StereoBuffer:
        """Apply harmonic excitation."""
        # Extract frequency range
        filtered = self.hp.process_stereo(audio)
        filtered = self.lp.process_stereo(filtered)

        # Apply saturation
        harmonics = np.zeros_like(filtered)
        for ch in range(2):
            for i in range(filtered.shape[1]):
                harmonics[ch, i] = self._soft_saturate(filtered[ch, i])

        # Mix back with original
        return audio + (harmonics - filtered) * self.mix


class StereoEnhancer:
    """
    M/S based stereo enhancement.

    Features:
    - Width control
    - Independent M/S EQ
    - Bass mono-ization
    - Correlation monitoring
    """

    def __init__(
        self,
        sample_rate: float,
        width: float = 1.0,  # 0=mono, 1=original, 2=wide
        bass_mono_freq: float = 120.0,  # Make bass mono below this
        side_high_boost_db: float = 0.0,  # Boost highs on side for air
    ):
        self.sample_rate = sample_rate
        self.width = width
        self.bass_mono_freq = bass_mono_freq

        # Bass mono filter
        self.bass_lp = BiquadFilter(FilterType.LOWPASS, bass_mono_freq, sample_rate, q=0.707)

        # Side high shelf for air
        if side_high_boost_db != 0:
            self.side_shelf = BiquadFilter(
                FilterType.HIGHSHELF, 8000, sample_rate, q=0.707, gain_db=side_high_boost_db
            )
        else:
            self.side_shelf = None

    def process(self, audio: StereoBuffer) -> StereoBuffer:
        """Apply stereo enhancement."""
        # Convert to M/S
        ms = StereoProcessor.to_mid_side(audio)

        # Apply width
        ms[1] *= self.width

        # Make bass mono (reduce side below bass_mono_freq)
        bass = self.bass_lp.process_stereo(np.array([ms[1], ms[1]]))
        ms[1] = ms[1] - bass[0] * 0.8  # Reduce bass in side channel

        # Boost highs on side for air
        if self.side_shelf is not None:
            side_boosted = self.side_shelf.process_stereo(np.array([ms[1], ms[1]]))
            ms[1] = side_boosted[0]

        # Convert back to stereo
        return StereoProcessor.from_mid_side(ms)


class Ditherer:
    """
    TPDF dithering with optional noise shaping for bit depth reduction.

    Implements:
    - TPDF (Triangular Probability Density Function) dither
    - Optional noise shaping (psychoacoustic curve)
    """

    def __init__(
        self,
        target_bits: int = 16,
        noise_shaping: bool = True,
    ):
        self.target_bits = target_bits
        self.noise_shaping = noise_shaping

        # Quantization step
        self.quant_step = 2.0 / (2**target_bits)

        # Noise shaping filter coefficients (F-weighted curve approximation)
        self.ns_coeffs = [0.5, -0.25, 0.125] if noise_shaping else []
        self.error_buffer = [0.0, 0.0, 0.0]

    def process(self, audio: StereoBuffer) -> StereoBuffer:
        """Apply dithering and quantization."""
        output = np.zeros_like(audio)

        for ch in range(2):
            error_history = self.error_buffer.copy()

            for i in range(audio.shape[1]):
                sample = audio[ch, i]

                # Apply noise shaping
                shaped_error = 0.0
                if self.noise_shaping:
                    for j, coeff in enumerate(self.ns_coeffs):
                        if j < len(error_history):
                            shaped_error += coeff * error_history[j]

                sample_ns = sample + shaped_error

                # Add TPDF dither
                dither = (np.random.random() - np.random.random()) * self.quant_step

                # Quantize
                quantized = np.round((sample_ns + dither) / self.quant_step) * self.quant_step

                # Calculate quantization error
                error = sample - quantized

                # Update error history
                error_history = [error] + error_history[:-1]

                output[ch, i] = quantized

        return output


@dataclass
class MasteringResult:
    """Result of mastering processing."""

    audio: StereoBuffer
    input_measurement: LoudnessMeasurement
    output_measurement: LoudnessMeasurement
    gain_applied_db: float
    peak_reduction_db: float
    meets_target: bool
    warnings: list[str] = field(default_factory=list)


class MasteringChain:
    """
    Complete broadcast-grade mastering chain.

    Implements a professional mastering workflow with metering
    and quality control at each stage.

    Usage:
        chain = MasteringChain(sample_rate=48000)
        chain.set_target(MasteringTarget.for_platform(DeliveryPlatform.SPOTIFY))
        chain.configure_multiband([...])
        result = chain.process(audio)
    """

    def __init__(
        self,
        sample_rate: float = 48000,
        target: MasteringTarget | None = None,
    ):
        self.sample_rate = sample_rate
        self.target = target or MasteringTarget()

        # Processing components
        self.input_eq: ParametricEQ | None = None
        self.saturator: AnalogSaturator | None = None
        self.transient_shaper: TransientShaper | None = None
        self.multiband: MultibandCompressor | None = None
        self.exciter: HarmonicExciter | None = None
        self.sub_enhancer: SubBassEnhancer | None = None
        self.high_exciter: ExciterEnhancer | None = None
        self.stereo_enhancer: StereoEnhancer | None = None
        self.limiter: TruePeakLimiter | None = None

        # Ditherer (for format conversion)
        self.ditherer: Ditherer | None = None

        # Metering
        self.meter = LoudnessMeter(sample_rate)

        # Default limiter
        self._init_limiter()

    def _init_limiter(self) -> None:
        """Initialize the true peak limiter."""
        self.limiter = TruePeakLimiter(
            sample_rate=self.sample_rate,
            ceiling_dbtp=self.target.true_peak_ceiling_dbtp,
            release_ms=100.0,
            lookahead_ms=5.0,
        )

    def set_target(self, target: MasteringTarget) -> None:
        """Set mastering target."""
        self.target = target
        self._init_limiter()

    def configure_eq(self, bands: list[dict[str, Any]]) -> None:
        """Configure input EQ."""
        eq = ParametricEQ(self.sample_rate)
        for band in bands:
            eq.add_band(
                band.get("band_type", "peak"),
                band.get("frequency_hz", 1000),
                band.get("gain_db", 0.0),
                band.get("q", 1.0),
            )
        self.input_eq = eq

    def configure_multiband(self, bands: list[dict[str, Any]]) -> None:
        """Configure multiband compression."""
        band_configs = []
        for band in bands:
            band_configs.append(
                MultibandCompressorBand(
                    name=band.get("band_name", "band"),
                    low_freq=band.get("crossover_low_hz", 20),
                    high_freq=band.get("crossover_high_hz", 20000),
                    threshold_db=band.get("threshold_db", -20.0),
                    ratio=band.get("ratio", 2.0),
                    attack_ms=band.get("attack_ms", 20.0),
                    release_ms=band.get("release_ms", 200.0),
                    makeup_gain_db=band.get("gain_db", 0.0),
                )
            )

        self.multiband = MultibandCompressor(self.sample_rate, band_configs)

    def configure_stereo(
        self,
        width: float = 1.0,
        bass_mono_freq: float = 120.0,
        side_high_boost_db: float = 0.0,
    ) -> None:
        """Configure stereo enhancement."""
        self.stereo_enhancer = StereoEnhancer(
            self.sample_rate,
            width=width,
            bass_mono_freq=bass_mono_freq,
            side_high_boost_db=side_high_boost_db,
        )

    def configure_exciter(
        self,
        low_freq: float = 300.0,
        high_freq: float = 8000.0,
        drive: float = 0.2,
        mix: float = 0.1,
    ) -> None:
        """Configure harmonic exciter."""
        self.exciter = HarmonicExciter(
            self.sample_rate,
            low_freq=low_freq,
            high_freq=high_freq,
            drive=drive,
            mix=mix,
        )

    def configure_dither(self, target_bits: int = 16, noise_shaping: bool = True) -> None:
        """Configure dithering for bit depth reduction."""
        self.ditherer = Ditherer(target_bits, noise_shaping)

    def configure_saturation(
        self,
        saturation_type: SaturationType = SaturationType.TAPE,
        drive: float = 0.3,
        mix: float = 0.5,
        output_gain_db: float = 0.0,
    ) -> None:
        """Configure analog saturation for warmth and glue."""
        self.saturator = AnalogSaturator(
            self.sample_rate,
            saturation_type=saturation_type,
            drive=drive,
            mix=mix,
            output_gain_db=output_gain_db,
        )

    def configure_transient_shaper(
        self,
        attack: float = 0.0,
        sustain: float = 0.0,
        sensitivity: float = 1.0,
    ) -> None:
        """Configure transient shaping for punch control."""
        self.transient_shaper = TransientShaper(
            self.sample_rate,
            attack=attack,
            sustain=sustain,
            sensitivity=sensitivity,
        )

    def configure_sub_bass(
        self,
        frequency: float = 60.0,
        amount: float = 0.3,
        drive: float = 0.2,
    ) -> None:
        """Configure sub-bass enhancement."""
        self.sub_enhancer = SubBassEnhancer(
            self.sample_rate,
            frequency=frequency,
            amount=amount,
            drive=drive,
        )

    def configure_high_exciter(
        self,
        frequency: float = 3000.0,
        amount: float = 0.2,
        harmonics: float = 0.4,
    ) -> None:
        """Configure high-frequency exciter for air and presence."""
        self.high_exciter = ExciterEnhancer(
            self.sample_rate,
            frequency=frequency,
            amount=amount,
            harmonics=harmonics,
        )

    def process(self, audio: StereoBuffer) -> MasteringResult:
        """
        Process audio through the complete mastering chain.

        Returns:
            MasteringResult with processed audio and analysis
        """
        logger.info(f"Mastering: {audio.shape[1]} samples, target {self.target.target_lufs} LUFS")

        warnings = []

        # Input measurement
        self.meter.reset()
        input_measurement = self.meter.measure(audio)

        logger.info(
            f"Input: {input_measurement.integrated_lufs:.1f} LUFS, "
            f"TP: {input_measurement.true_peak_dbtp:.1f} dBTP"
        )

        # Working copy
        processed = audio.copy()

        # Stage 1: Input EQ
        if self.input_eq is not None:
            processed = self.input_eq.process(processed)

        # Stage 2: Analog saturation (for warmth and glue)
        if self.saturator is not None:
            processed = self.saturator.process_stereo(processed)
            logger.debug("Applied analog saturation")

        # Stage 3: Transient shaping (for punch)
        if self.transient_shaper is not None:
            processed = self.transient_shaper.process_stereo(processed)
            logger.debug("Applied transient shaping")

        # Stage 4: Multiband compression
        if self.multiband is not None:
            processed, band_gr = self.multiband.process(processed)
            logger.debug(f"Multiband GR: {[f'{gr:.1f} dB' for gr in band_gr]}")

        # Stage 5: Sub-bass enhancement
        if self.sub_enhancer is not None:
            processed = self.sub_enhancer.process_stereo(processed)
            logger.debug("Applied sub-bass enhancement")

        # Stage 6: High-frequency exciter
        if self.high_exciter is not None:
            processed = self.high_exciter.process_stereo(processed)
            logger.debug("Applied high-frequency exciter")

        # Stage 7: Stereo enhancement
        if self.stereo_enhancer is not None:
            processed = self.stereo_enhancer.process(processed)

        # Stage 8: Harmonic exciter (legacy)
        if self.exciter is not None:
            processed = self.exciter.process(processed)

        # Stage 9: Normalize to target loudness (pre-limiter)
        # Calculate how much gain we need
        self.meter.reset()
        pre_limit_measurement = self.meter.measure(processed)

        if pre_limit_measurement.integrated_lufs > -70:
            target_with_headroom = self.target.target_lufs + 1.0  # Leave headroom for limiter
            gain_needed_db = target_with_headroom - pre_limit_measurement.integrated_lufs

            # Limit gain to reasonable range
            gain_needed_db = max(-20.0, min(20.0, gain_needed_db))

            processed = processed * db_to_linear(gain_needed_db)
            logger.info(f"Applied {gain_needed_db:.1f} dB gain for loudness target")
        else:
            gain_needed_db = 0.0
            warnings.append("Input too quiet to measure loudness")

        # Stage 6: True peak limiting
        if self.limiter is not None:
            # Check if we need limiting
            pre_peak = max(np.max(np.abs(processed[0])), np.max(np.abs(processed[1])))
            pre_peak_db = linear_to_db(pre_peak)

            processed = self.limiter.process_stereo(processed)

            post_peak = max(np.max(np.abs(processed[0])), np.max(np.abs(processed[1])))
            post_peak_db = linear_to_db(post_peak)

            peak_reduction = pre_peak_db - post_peak_db
            if peak_reduction > 3.0:
                warnings.append(f"Heavy limiting applied ({peak_reduction:.1f} dB)")
        else:
            peak_reduction = 0.0

        # Stage 7: Dithering (if enabled)
        if self.ditherer is not None:
            processed = self.ditherer.process(processed)

        # Output measurement
        self.meter.reset()
        output_measurement = self.meter.measure(processed)

        logger.info(
            f"Output: {output_measurement.integrated_lufs:.1f} LUFS, "
            f"TP: {output_measurement.true_peak_dbtp:.1f} dBTP, "
            f"LRA: {output_measurement.loudness_range_lu:.1f} LU"
        )

        # Check if we meet target
        meets_target = True
        lufs_diff = abs(output_measurement.integrated_lufs - self.target.target_lufs)
        if lufs_diff > self.target.tolerance_lufs:
            warnings.append(
                f"Loudness {output_measurement.integrated_lufs:.1f} LUFS "
                f"outside target {self.target.target_lufs} ±{self.target.tolerance_lufs}"
            )
            meets_target = False

        if output_measurement.true_peak_dbtp > self.target.true_peak_ceiling_dbtp:
            warnings.append(
                f"True peak {output_measurement.true_peak_dbtp:.1f} dBTP "
                f"exceeds ceiling {self.target.true_peak_ceiling_dbtp}"
            )
            meets_target = False

        if output_measurement.loudness_range_lu < self.target.min_dynamic_range_lu:
            warnings.append(
                f"Dynamic range {output_measurement.loudness_range_lu:.1f} LU "
                f"below minimum {self.target.min_dynamic_range_lu}"
            )

        # Check mono compatibility
        mono_ok, correlation = StereoProcessor.check_mono_compatibility(processed)
        if not mono_ok:
            warnings.append(f"Poor mono compatibility (correlation: {correlation:.2f})")

        return MasteringResult(
            audio=processed,
            input_measurement=input_measurement,
            output_measurement=output_measurement,
            gain_applied_db=gain_needed_db,
            peak_reduction_db=peak_reduction,
            meets_target=meets_target,
            warnings=warnings,
        )


def create_streaming_master(
    audio: StereoBuffer,
    sample_rate: float,
    platform: DeliveryPlatform = DeliveryPlatform.SPOTIFY,
) -> MasteringResult:
    """
    Convenience function to create a streaming-ready master.

    Applies sensible defaults for the specified platform.
    """
    chain = MasteringChain(sample_rate)
    chain.set_target(MasteringTarget.for_platform(platform))

    # Configure standard multiband
    chain.configure_multiband(
        [
            {
                "band_name": "low",
                "crossover_low_hz": 20,
                "crossover_high_hz": 120,
                "threshold_db": -18.0,
                "ratio": 2.5,
                "attack_ms": 30.0,
                "release_ms": 300.0,
            },
            {
                "band_name": "low_mid",
                "crossover_low_hz": 120,
                "crossover_high_hz": 500,
                "threshold_db": -20.0,
                "ratio": 2.0,
                "attack_ms": 25.0,
                "release_ms": 200.0,
            },
            {
                "band_name": "mid",
                "crossover_low_hz": 500,
                "crossover_high_hz": 2000,
                "threshold_db": -22.0,
                "ratio": 1.8,
                "attack_ms": 20.0,
                "release_ms": 150.0,
            },
            {
                "band_name": "high_mid",
                "crossover_low_hz": 2000,
                "crossover_high_hz": 8000,
                "threshold_db": -24.0,
                "ratio": 1.5,
                "attack_ms": 15.0,
                "release_ms": 100.0,
            },
            {
                "band_name": "high",
                "crossover_low_hz": 8000,
                "crossover_high_hz": 20000,
                "threshold_db": -26.0,
                "ratio": 1.3,
                "attack_ms": 10.0,
                "release_ms": 80.0,
            },
        ]
    )

    # Subtle stereo enhancement
    chain.configure_stereo(width=1.05, bass_mono_freq=120.0, side_high_boost_db=0.5)

    return chain.process(audio)


def create_genre_master(
    audio: StereoBuffer,
    sample_rate: float,
    genre_id: str,
    target_lufs: float = -14.0,
) -> MasteringResult:
    """
    Create a mastered version with genre-appropriate settings.

    Args:
        audio: Input stereo audio
        sample_rate: Sample rate
        genre_id: Genre identifier for style-specific processing
        target_lufs: Target loudness in LUFS

    Returns:
        MasteringResult with processed audio and analysis
    """
    chain = MasteringChain(sample_rate)
    chain.set_target(
        MasteringTarget(
            target_lufs=target_lufs,
            true_peak_ceiling_dbtp=-1.0,
        )
    )

    genre_lower = genre_id.lower()

    # Genre-specific multiband settings
    if "trap" in genre_lower:
        # Trap: heavy 808s, crisp hats, punchy
        chain.configure_multiband(
            [
                {
                    "band_name": "sub",
                    "crossover_low_hz": 20,
                    "crossover_high_hz": 50,
                    "threshold_db": -12.0,
                    "ratio": 4.0,
                    "attack_ms": 50.0,
                    "release_ms": 500.0,
                },
                {
                    "band_name": "bass",
                    "crossover_low_hz": 50,
                    "crossover_high_hz": 150,
                    "threshold_db": -14.0,
                    "ratio": 3.0,
                    "attack_ms": 30.0,
                    "release_ms": 300.0,
                },
                {
                    "band_name": "low_mid",
                    "crossover_low_hz": 150,
                    "crossover_high_hz": 600,
                    "threshold_db": -20.0,
                    "ratio": 2.0,
                    "attack_ms": 20.0,
                    "release_ms": 150.0,
                },
                {
                    "band_name": "mid",
                    "crossover_low_hz": 600,
                    "crossover_high_hz": 4000,
                    "threshold_db": -22.0,
                    "ratio": 1.8,
                    "attack_ms": 15.0,
                    "release_ms": 100.0,
                },
                {
                    "band_name": "high",
                    "crossover_low_hz": 4000,
                    "crossover_high_hz": 20000,
                    "threshold_db": -20.0,
                    "ratio": 1.5,
                    "attack_ms": 10.0,
                    "release_ms": 80.0,
                },
            ]
        )
        # Heavy sub-bass enhancement for 808s
        chain.configure_sub_bass(frequency=50.0, amount=0.5, drive=0.4)
        # Transient shaping for punchy hats and snares
        chain.configure_transient_shaper(attack=30, sustain=-10, sensitivity=1.2)
        # Subtle tape saturation for warmth
        chain.configure_saturation(SaturationType.TAPE, drive=0.25, mix=0.4)
        # Crisp highs
        chain.configure_high_exciter(frequency=4000.0, amount=0.25, harmonics=0.5)
        chain.configure_stereo(width=1.1, bass_mono_freq=120.0)

    elif "hip-hop" in genre_lower or "boom-bap" in genre_lower:
        # Hip-hop: strong low end, controlled mids
        chain.configure_multiband(
            [
                {
                    "band_name": "sub",
                    "crossover_low_hz": 20,
                    "crossover_high_hz": 60,
                    "threshold_db": -15.0,
                    "ratio": 3.0,
                    "attack_ms": 40.0,
                    "release_ms": 400.0,
                },
                {
                    "band_name": "bass",
                    "crossover_low_hz": 60,
                    "crossover_high_hz": 200,
                    "threshold_db": -16.0,
                    "ratio": 2.5,
                    "attack_ms": 30.0,
                    "release_ms": 300.0,
                },
                {
                    "band_name": "low_mid",
                    "crossover_low_hz": 200,
                    "crossover_high_hz": 800,
                    "threshold_db": -20.0,
                    "ratio": 2.0,
                    "attack_ms": 25.0,
                    "release_ms": 200.0,
                },
                {
                    "band_name": "mid",
                    "crossover_low_hz": 800,
                    "crossover_high_hz": 4000,
                    "threshold_db": -22.0,
                    "ratio": 1.8,
                    "attack_ms": 20.0,
                    "release_ms": 150.0,
                },
                {
                    "band_name": "high",
                    "crossover_low_hz": 4000,
                    "crossover_high_hz": 20000,
                    "threshold_db": -24.0,
                    "ratio": 1.5,
                    "attack_ms": 15.0,
                    "release_ms": 100.0,
                },
            ]
        )
        chain.configure_stereo(width=1.0, bass_mono_freq=150.0)
        # Warm tape saturation for that classic hip-hop sound
        chain.configure_saturation(SaturationType.TAPE, drive=0.35, mix=0.5)
        # Sub-bass punch
        chain.configure_sub_bass(frequency=60.0, amount=0.4, drive=0.3)

    elif "house" in genre_lower:
        # House: pumping bass, clean mids, sparkling highs
        chain.configure_multiband(
            [
                {
                    "band_name": "sub",
                    "crossover_low_hz": 20,
                    "crossover_high_hz": 80,
                    "threshold_db": -14.0,
                    "ratio": 3.0,
                    "attack_ms": 25.0,
                    "release_ms": 200.0,
                },
                {
                    "band_name": "bass",
                    "crossover_low_hz": 80,
                    "crossover_high_hz": 250,
                    "threshold_db": -16.0,
                    "ratio": 2.5,
                    "attack_ms": 20.0,
                    "release_ms": 150.0,
                },
                {
                    "band_name": "mid",
                    "crossover_low_hz": 250,
                    "crossover_high_hz": 3000,
                    "threshold_db": -20.0,
                    "ratio": 1.8,
                    "attack_ms": 15.0,
                    "release_ms": 100.0,
                },
                {
                    "band_name": "high",
                    "crossover_low_hz": 3000,
                    "crossover_high_hz": 20000,
                    "threshold_db": -22.0,
                    "ratio": 1.5,
                    "attack_ms": 10.0,
                    "release_ms": 80.0,
                },
            ]
        )
        chain.configure_stereo(width=1.15, bass_mono_freq=100.0, side_high_boost_db=1.0)
        # Punch for the kick
        chain.configure_transient_shaper(attack=20, sustain=0, sensitivity=1.0)
        # Air and sparkle
        chain.configure_high_exciter(frequency=5000.0, amount=0.2, harmonics=0.4)
        # Tube warmth for that analog feel
        chain.configure_saturation(SaturationType.TUBE, drive=0.2, mix=0.3)

    elif "techno" in genre_lower:
        # Techno: hard-hitting, industrial, relentless
        chain.configure_multiband(
            [
                {
                    "band_name": "sub",
                    "crossover_low_hz": 20,
                    "crossover_high_hz": 80,
                    "threshold_db": -12.0,
                    "ratio": 4.0,
                    "attack_ms": 15.0,
                    "release_ms": 150.0,
                },
                {
                    "band_name": "bass",
                    "crossover_low_hz": 80,
                    "crossover_high_hz": 300,
                    "threshold_db": -14.0,
                    "ratio": 3.0,
                    "attack_ms": 10.0,
                    "release_ms": 100.0,
                },
                {
                    "band_name": "mid",
                    "crossover_low_hz": 300,
                    "crossover_high_hz": 4000,
                    "threshold_db": -18.0,
                    "ratio": 2.0,
                    "attack_ms": 10.0,
                    "release_ms": 80.0,
                },
                {
                    "band_name": "high",
                    "crossover_low_hz": 4000,
                    "crossover_high_hz": 20000,
                    "threshold_db": -20.0,
                    "ratio": 1.5,
                    "attack_ms": 8.0,
                    "release_ms": 60.0,
                },
            ]
        )
        chain.configure_stereo(width=1.0, bass_mono_freq=120.0)
        # Aggressive transients
        chain.configure_transient_shaper(attack=40, sustain=-20, sensitivity=1.5)
        # Transistor distortion for that industrial edge
        chain.configure_saturation(SaturationType.TRANSISTOR, drive=0.4, mix=0.4)

    elif "edm" in genre_lower or "electronic" in genre_lower:
        # EDM: punchy, wide, loud
        chain.configure_multiband(
            [
                {
                    "band_name": "sub",
                    "crossover_low_hz": 20,
                    "crossover_high_hz": 80,
                    "threshold_db": -12.0,
                    "ratio": 4.0,
                    "attack_ms": 20.0,
                    "release_ms": 200.0,
                },
                {
                    "band_name": "bass",
                    "crossover_low_hz": 80,
                    "crossover_high_hz": 300,
                    "threshold_db": -14.0,
                    "ratio": 3.0,
                    "attack_ms": 15.0,
                    "release_ms": 150.0,
                },
                {
                    "band_name": "mid",
                    "crossover_low_hz": 300,
                    "crossover_high_hz": 3000,
                    "threshold_db": -18.0,
                    "ratio": 2.0,
                    "attack_ms": 15.0,
                    "release_ms": 100.0,
                },
                {
                    "band_name": "high",
                    "crossover_low_hz": 3000,
                    "crossover_high_hz": 20000,
                    "threshold_db": -20.0,
                    "ratio": 1.5,
                    "attack_ms": 10.0,
                    "release_ms": 80.0,
                },
            ]
        )
        chain.configure_stereo(width=1.15, bass_mono_freq=120.0, side_high_boost_db=1.0)
        chain.configure_exciter(low_freq=500, high_freq=12000, drive=0.15, mix=0.08)

    elif "lo-fi" in genre_lower or "lofi" in genre_lower:
        # Lo-fi: warm, relaxed dynamics
        chain.configure_multiband(
            [
                {
                    "band_name": "low",
                    "crossover_low_hz": 20,
                    "crossover_high_hz": 200,
                    "threshold_db": -20.0,
                    "ratio": 1.5,
                    "attack_ms": 50.0,
                    "release_ms": 500.0,
                },
                {
                    "band_name": "mid",
                    "crossover_low_hz": 200,
                    "crossover_high_hz": 3000,
                    "threshold_db": -22.0,
                    "ratio": 1.3,
                    "attack_ms": 40.0,
                    "release_ms": 400.0,
                },
                {
                    "band_name": "high",
                    "crossover_low_hz": 3000,
                    "crossover_high_hz": 20000,
                    "threshold_db": -26.0,
                    "ratio": 1.2,
                    "attack_ms": 30.0,
                    "release_ms": 300.0,
                },
            ]
        )
        chain.configure_stereo(width=0.95, bass_mono_freq=100.0)
        # Lo-fi often has rolled off highs
        chain.configure_eq(
            [
                {"band_type": "lowshelf", "frequency_hz": 200, "gain_db": 2.0, "q": 0.7},
                {"band_type": "highshelf", "frequency_hz": 8000, "gain_db": -2.0, "q": 0.7},
            ]
        )

    elif "synthwave" in genre_lower or "retrowave" in genre_lower:
        # Synthwave: bright, wide, punchy
        chain.configure_multiband(
            [
                {
                    "band_name": "sub",
                    "crossover_low_hz": 20,
                    "crossover_high_hz": 100,
                    "threshold_db": -16.0,
                    "ratio": 3.0,
                    "attack_ms": 25.0,
                    "release_ms": 250.0,
                },
                {
                    "band_name": "bass",
                    "crossover_low_hz": 100,
                    "crossover_high_hz": 400,
                    "threshold_db": -18.0,
                    "ratio": 2.5,
                    "attack_ms": 20.0,
                    "release_ms": 200.0,
                },
                {
                    "band_name": "mid",
                    "crossover_low_hz": 400,
                    "crossover_high_hz": 4000,
                    "threshold_db": -20.0,
                    "ratio": 2.0,
                    "attack_ms": 15.0,
                    "release_ms": 150.0,
                },
                {
                    "band_name": "high",
                    "crossover_low_hz": 4000,
                    "crossover_high_hz": 20000,
                    "threshold_db": -22.0,
                    "ratio": 1.5,
                    "attack_ms": 10.0,
                    "release_ms": 100.0,
                },
            ]
        )
        chain.configure_stereo(width=1.2, bass_mono_freq=100.0, side_high_boost_db=1.5)
        chain.configure_exciter(low_freq=400, high_freq=10000, drive=0.2, mix=0.1)
        chain.configure_eq(
            [
                {"band_type": "highshelf", "frequency_hz": 10000, "gain_db": 1.5, "q": 0.7},
            ]
        )

    else:
        # Default: balanced approach
        chain.configure_multiband(
            [
                {
                    "band_name": "low",
                    "crossover_low_hz": 20,
                    "crossover_high_hz": 120,
                    "threshold_db": -18.0,
                    "ratio": 2.5,
                    "attack_ms": 30.0,
                    "release_ms": 300.0,
                },
                {
                    "band_name": "low_mid",
                    "crossover_low_hz": 120,
                    "crossover_high_hz": 500,
                    "threshold_db": -20.0,
                    "ratio": 2.0,
                    "attack_ms": 25.0,
                    "release_ms": 200.0,
                },
                {
                    "band_name": "mid",
                    "crossover_low_hz": 500,
                    "crossover_high_hz": 2000,
                    "threshold_db": -22.0,
                    "ratio": 1.8,
                    "attack_ms": 20.0,
                    "release_ms": 150.0,
                },
                {
                    "band_name": "high_mid",
                    "crossover_low_hz": 2000,
                    "crossover_high_hz": 8000,
                    "threshold_db": -24.0,
                    "ratio": 1.5,
                    "attack_ms": 15.0,
                    "release_ms": 100.0,
                },
                {
                    "band_name": "high",
                    "crossover_low_hz": 8000,
                    "crossover_high_hz": 20000,
                    "threshold_db": -26.0,
                    "ratio": 1.3,
                    "attack_ms": 10.0,
                    "release_ms": 80.0,
                },
            ]
        )
        chain.configure_stereo(width=1.05, bass_mono_freq=120.0, side_high_boost_db=0.5)

    return chain.process(audio)
