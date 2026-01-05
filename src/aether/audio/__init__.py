"""
AETHER Audio Processing Module

Production-grade audio processing for mixing, mastering, and export.

Components:
- dsp: Core DSP utilities (filters, dynamics, meters)
- mixing: Professional mixing engine
- mastering: Broadcast-grade mastering chain
- io: Audio file I/O

Standards Compliance:
- ITU-R BS.1770-4 (Loudness measurement)
- EBU R128 (European loudness normalization)
- AES-17 (True peak metering)

Example Usage:
    from aether.audio import MixingEngine, MasteringChain, write_wav

    # Create a mix
    engine = MixingEngine(sample_rate=48000)
    engine.add_track("kick", kick_audio)
    engine.add_track("bass", bass_audio, pan=-0.2)
    mix = engine.render()

    # Master it
    chain = MasteringChain(sample_rate=48000)
    result = chain.process(mix)

    # Export
    write_wav("output.wav", result.audio, 48000, bit_depth=24)
"""

# DSP Utilities
from aether.audio.dsp import (
    # Types
    AudioBuffer,
    BiquadCoefficients,
    BiquadFilter,
    # Dynamics
    Compressor,
    # Filters
    FilterType,
    LoudnessMeasurement,
    # Metering
    LoudnessMeter,
    ParametricEQ,
    StereoBuffer,
    # Stereo
    StereoProcessor,
    TruePeakLimiter,
    # Utilities
    db_to_linear,
    linear_to_db,
    normalize_loudness,
    normalize_peak,
)

# I/O
from aether.audio.io import (
    AudioFile,
    AudioFormat,
    AudioFormatSpec,
    AudioMetadata,
    BatchExporter,
    BitDepthConverter,
    SampleRateConverter,
    generate_silence,
    generate_test_tone,
    normalize_audio,
    read_audio,
    # Functions
    read_wav,
    write_audio,
    write_wav,
)

# Mastering Chain
from aether.audio.mastering import (
    DeliveryPlatform,
    Ditherer,
    HarmonicExciter,
    MasteringChain,
    MasteringResult,
    MasteringTarget,
    MultibandCompressor,
    MultibandCompressorBand,
    StereoEnhancer,
    create_genre_master,
    # Convenience functions
    create_streaming_master,
)

# Mixing Engine
from aether.audio.mixing import (
    AutomationCurve,
    AutomationLane,
    AutomationPoint,
    BusState,
    MixingEngine,
    TrackState,
    create_basic_delay,
    # Effects
    create_basic_reverb,
)

__all__ = [
    # DSP Types
    "AudioBuffer",
    "StereoBuffer",
    # Filters
    "FilterType",
    "BiquadFilter",
    "BiquadCoefficients",
    "ParametricEQ",
    # Dynamics
    "Compressor",
    "TruePeakLimiter",
    # Metering
    "LoudnessMeter",
    "LoudnessMeasurement",
    # Stereo
    "StereoProcessor",
    # DSP Utilities
    "db_to_linear",
    "linear_to_db",
    "normalize_peak",
    "normalize_loudness",
    # Mixing
    "MixingEngine",
    "TrackState",
    "BusState",
    "AutomationLane",
    "AutomationPoint",
    "AutomationCurve",
    "create_basic_reverb",
    "create_basic_delay",
    # Mastering
    "MasteringChain",
    "MasteringTarget",
    "MasteringResult",
    "DeliveryPlatform",
    "MultibandCompressor",
    "MultibandCompressorBand",
    "HarmonicExciter",
    "StereoEnhancer",
    "Ditherer",
    "create_streaming_master",
    "create_genre_master",
    # I/O
    "AudioFormat",
    "AudioFormatSpec",
    "AudioMetadata",
    "AudioFile",
    "SampleRateConverter",
    "BitDepthConverter",
    "BatchExporter",
    "read_wav",
    "write_wav",
    "read_audio",
    "write_audio",
    "generate_test_tone",
    "generate_silence",
    "normalize_audio",
]
