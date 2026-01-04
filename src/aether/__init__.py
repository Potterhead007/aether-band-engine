"""
AETHER Band Engine

Autonomous Ensemble for Thoughtful Harmonic Expression and Rendering

A production-grade AI music factory capable of generating commercially viable,
fully original music across any genre.
"""

__version__ = "0.1.0"
__author__ = "AETHER Team"

from aether.config import AetherConfig, get_config, init_config


# Pipeline exports (lazy import to avoid circular dependencies)
def generate_track(*args, **kwargs):
    """Generate a complete track through the AETHER pipeline."""
    from aether.orchestration import generate_track as _generate_track

    return _generate_track(*args, **kwargs)


def create_pipeline(*args, **kwargs):
    """Create a new MusicPipeline instance."""
    from aether.orchestration import MusicPipeline

    return MusicPipeline(*args, **kwargs)


__all__ = [
    "__version__",
    # Config
    "AetherConfig",
    "get_config",
    "init_config",
    # Pipeline
    "generate_track",
    "create_pipeline",
]
