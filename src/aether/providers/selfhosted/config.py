"""
Self-Hosted Voice Provider Configuration

Configuration dataclasses for XTTS and RVC voice synthesis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class XTTSConfig:
    """Configuration for Coqui XTTS text-to-speech engine."""

    # Model settings
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_path: Optional[Path] = None  # Custom model path (optional)

    # Device settings
    device: str = "auto"  # auto, cuda, mps, cpu
    compute_type: str = "float16"  # float16, float32, int8

    # Speaker settings
    speaker_embeddings_dir: Optional[Path] = None  # Directory with AVU-*.npy files
    default_speaker_wav: Optional[Path] = None  # Fallback speaker reference

    # Generation parameters
    language: str = "en"
    temperature: float = 0.7
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.85

    # Performance
    enable_text_splitting: bool = True
    stream: bool = False

    @classmethod
    def from_env(cls) -> "XTTSConfig":
        """Create config from environment variables."""
        return cls(
            model_name=os.environ.get("XTTS_MODEL_NAME", cls.model_name),
            model_path=Path(os.environ["XTTS_MODEL_PATH"]) if os.environ.get("XTTS_MODEL_PATH") else None,
            device=os.environ.get("AETHER_VOICE_DEVICE", "auto"),
            compute_type=os.environ.get("XTTS_COMPUTE_TYPE", "float16"),
            speaker_embeddings_dir=Path(os.environ["XTTS_SPEAKER_DIR"]) if os.environ.get("XTTS_SPEAKER_DIR") else None,
            default_speaker_wav=Path(os.environ["XTTS_SPEAKER_WAV"]) if os.environ.get("XTTS_SPEAKER_WAV") else None,
            language=os.environ.get("XTTS_LANGUAGE", "en"),
            temperature=float(os.environ.get("XTTS_TEMPERATURE", "0.7")),
        )


@dataclass
class RVCConfig:
    """Configuration for RVC (Retrieval-based Voice Conversion) engine."""

    # Model directory containing AVU-*/model.pth files
    models_dir: Optional[Path] = None

    # Device settings
    device: str = "auto"  # auto, cuda, mps, cpu

    # Conversion parameters
    f0_method: str = "rmvpe"  # pm, harvest, crepe, rmvpe (best quality)
    index_rate: float = 0.75  # 0.0-1.0, higher = more like training voice
    filter_radius: int = 3  # Median filtering for pitch
    resample_rate: int = 0  # 0 = auto
    rms_mix_rate: float = 0.25  # 0.0-1.0, mix of original and converted RMS
    protect: float = 0.33  # Protect voiceless consonants (0.0-0.5)

    # Performance
    use_index: bool = True  # Use FAISS index for retrieval

    @classmethod
    def from_env(cls) -> "RVCConfig":
        """Create config from environment variables."""
        return cls(
            models_dir=Path(os.environ["RVC_MODELS_PATH"]) if os.environ.get("RVC_MODELS_PATH") else None,
            device=os.environ.get("AETHER_VOICE_DEVICE", "auto"),
            f0_method=os.environ.get("RVC_F0_METHOD", "rmvpe"),
            index_rate=float(os.environ.get("RVC_INDEX_RATE", "0.75")),
            protect=float(os.environ.get("RVC_PROTECT", "0.33")),
        )


@dataclass
class AVUVoiceMapping:
    """Mapping configuration for an AVU voice identity."""

    name: str  # AVU-1, AVU-2, etc.

    # XTTS settings
    xtts_speaker_embedding: Optional[Path] = None  # .npy file
    xtts_speaker_wav: Optional[Path] = None  # Reference audio
    xtts_language: str = "en"
    xtts_temperature: float = 0.7

    # RVC settings
    rvc_model_path: Optional[Path] = None  # .pth file
    rvc_index_path: Optional[Path] = None  # .index file
    rvc_pitch_shift: int = 0  # Semitones adjustment
    rvc_index_rate: float = 0.75


# Default AVU voice configurations
# These will be populated when models are discovered
AVU_VOICE_CONFIGS: dict[str, AVUVoiceMapping] = {
    "AVU-1": AVUVoiceMapping(
        name="AVU-1",
        xtts_temperature=0.65,  # More controlled for lyric tenor
        rvc_pitch_shift=0,
        rvc_index_rate=0.8,
    ),
    "AVU-2": AVUVoiceMapping(
        name="AVU-2",
        xtts_temperature=0.70,  # Warm, versatile mezzo
        rvc_pitch_shift=0,
        rvc_index_rate=0.75,
    ),
    "AVU-3": AVUVoiceMapping(
        name="AVU-3",
        xtts_temperature=0.60,  # Controlled baritone
        rvc_pitch_shift=0,
        rvc_index_rate=0.80,
    ),
    "AVU-4": AVUVoiceMapping(
        name="AVU-4",
        xtts_temperature=0.75,  # Expressive soprano
        rvc_pitch_shift=0,
        rvc_index_rate=0.75,
    ),
}


@dataclass
class SelfHostedConfig:
    """Master configuration for self-hosted vocal provider."""

    # Sub-component configs
    xtts: XTTSConfig = field(default_factory=XTTSConfig)
    rvc: RVCConfig = field(default_factory=RVCConfig)

    # General settings
    sample_rate: int = 48000
    cache_dir: Optional[Path] = None

    # Feature toggles
    enable_rvc: bool = True  # Can disable RVC for faster (lower quality) output
    enable_caching: bool = True

    # Timeouts
    synthesis_timeout_seconds: float = 120.0
    model_load_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "SelfHostedConfig":
        """Create full config from environment variables."""
        return cls(
            xtts=XTTSConfig.from_env(),
            rvc=RVCConfig.from_env(),
            sample_rate=int(os.environ.get("AETHER_VOICE_SAMPLE_RATE", "48000")),
            cache_dir=Path(os.environ["AETHER_VOICE_CACHE"]) if os.environ.get("AETHER_VOICE_CACHE") else None,
            enable_rvc=os.environ.get("AETHER_VOICE_ENABLE_RVC", "true").lower() == "true",
            enable_caching=os.environ.get("AETHER_VOICE_ENABLE_CACHE", "true").lower() == "true",
            synthesis_timeout_seconds=float(os.environ.get("AETHER_VOICE_SYNTHESIS_TIMEOUT", "120")),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check XTTS requirements
        if not self.xtts.model_path and not self.xtts.model_name:
            issues.append("No XTTS model specified (set XTTS_MODEL_PATH or XTTS_MODEL_NAME)")

        if self.xtts.model_path and not self.xtts.model_path.exists():
            issues.append(f"XTTS model path does not exist: {self.xtts.model_path}")

        # Check RVC requirements (if enabled)
        if self.enable_rvc:
            if not self.rvc.models_dir:
                issues.append("RVC enabled but no models directory specified (set RVC_MODELS_PATH)")
            elif not self.rvc.models_dir.exists():
                issues.append(f"RVC models directory does not exist: {self.rvc.models_dir}")

        return issues
