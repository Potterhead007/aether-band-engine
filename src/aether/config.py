"""
AETHER Configuration System

Manages all configuration including paths, providers, and defaults.
Supports environment variables, YAML files, and programmatic override.
"""

from pathlib import Path
from typing import ClassVar, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseSettings):
    """Path configuration for AETHER."""

    model_config = SettingsConfigDict(env_prefix="AETHER_")

    # Base paths
    base_dir: Path = Field(
        default_factory=lambda: Path.home() / ".aether",
        description="Base directory for AETHER data",
    )

    # Data paths
    genres_dir: Path = Field(default=Path("data/genres"))
    instruments_dir: Path = Field(default=Path("data/instruments"))
    samples_dir: Path = Field(default=Path("data/samples"))
    soundfonts_dir: Path = Field(default=Path("data/soundfonts"))

    # Output paths
    output_dir: Path = Field(default=Path("output"))
    projects_dir: Path = Field(default=Path("projects"))

    # Cache
    cache_dir: Path = Field(default=Path("cache"))

    def get_absolute(self, relative_path: Path) -> Path:
        """Convert relative path to absolute based on base_dir."""
        if relative_path.is_absolute():
            return relative_path
        return self.base_dir / relative_path

    def ensure_directories(self) -> None:
        """Create all required directories."""
        for attr in [
            "genres_dir",
            "instruments_dir",
            "samples_dir",
            "soundfonts_dir",
            "output_dir",
            "projects_dir",
            "cache_dir",
        ]:
            path = self.get_absolute(getattr(self, attr))
            path.mkdir(parents=True, exist_ok=True)


class ProviderConfig(BaseSettings):
    """Provider configuration."""

    model_config = SettingsConfigDict(env_prefix="AETHER_PROVIDER_")

    # LLM Provider
    llm_provider: str = Field(default="anthropic", description="LLM provider: anthropic, openai")
    llm_model: str = Field(default="claude-sonnet-4-20250514", description="LLM model name")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API key")

    # MIDI Provider
    midi_provider: str = Field(default="internal", description="MIDI generation provider")

    # Audio Provider
    audio_provider: str = Field(default="soundfont", description="Audio rendering provider")
    default_soundfont: str = Field(default="FluidR3_GM.sf2", description="Default soundfont")

    # Vocal Provider
    vocal_provider: str = Field(default="tts", description="Vocal synthesis provider")

    # Embedding Provider
    embedding_provider: str = Field(default="sentence-transformers")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")


class AudioConfig(BaseSettings):
    """Audio processing configuration."""

    model_config = SettingsConfigDict(env_prefix="AETHER_AUDIO_")

    # Sample rates
    working_sample_rate: int = Field(default=48000)
    output_sample_rate: int = Field(default=44100)

    # Bit depths
    working_bit_depth: int = Field(default=32)  # Float
    output_bit_depth: int = Field(default=24)

    # Buffer sizes
    buffer_size: int = Field(default=1024)

    # Mastering defaults
    default_lufs: float = Field(default=-14.0)
    default_true_peak: float = Field(default=-1.0)


class QAConfig(BaseSettings):
    """Quality assurance thresholds."""

    model_config = SettingsConfigDict(env_prefix="AETHER_QA_")

    # Originality thresholds
    melody_similarity_threshold: float = Field(default=0.85)
    lyric_ngram_threshold: float = Field(default=0.03)
    semantic_distance_threshold: float = Field(default=0.7)

    # Genre authenticity
    genre_authenticity_threshold: float = Field(default=0.80)

    # Technical
    lufs_tolerance: float = Field(default=0.5)
    phase_correlation_minimum: float = Field(default=0.5)


class AetherConfig(BaseSettings):
    """
    Main AETHER configuration.

    Aggregates all configuration sections and provides loading/saving.
    """

    model_config = SettingsConfigDict(
        env_prefix="AETHER_",
        env_nested_delimiter="__",
    )

    # Sub-configs
    paths: PathsConfig = Field(default_factory=PathsConfig)
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    qa: QAConfig = Field(default_factory=QAConfig)

    # General
    debug: bool = Field(default=False)
    verbose: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Reproducibility
    default_seed: Optional[int] = Field(default=None)

    @classmethod
    def load_from_yaml(cls, path: Path) -> "AetherConfig":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    # Fields that should never be serialized to disk
    SENSITIVE_FIELDS: ClassVar[set[str]] = {"llm_api_key", "embedding_api_key", "api_key"}

    def save_to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file, excluding sensitive fields."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get config data excluding sensitive fields
        data = self.model_dump()

        # Remove sensitive fields from providers section
        if "providers" in data:
            for sensitive_key in self.SENSITIVE_FIELDS:
                data["providers"].pop(sensitive_key, None)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def ensure_setup(self) -> None:
        """Ensure all required directories and files exist."""
        self.paths.ensure_directories()


# Global config instance
_config: Optional[AetherConfig] = None


def get_config() -> AetherConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AetherConfig()
    return _config


def init_config(config_path: Optional[Path] = None) -> AetherConfig:
    """Initialize configuration from file or defaults."""
    global _config

    if config_path and config_path.exists():
        _config = AetherConfig.load_from_yaml(config_path)
    else:
        _config = AetherConfig()

    _config.ensure_setup()
    return _config


def set_config(config: AetherConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
