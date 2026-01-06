"""AETHER voice synthesis pipeline integration."""

from aether.voice.integration.pipeline import (
    VoiceSynthesisPipeline,
    VoiceSynthesisInput,
    VoiceSynthesisOutput,
    PipelineStage,
    PipelineProgress,
    BatchSynthesisPipeline,
)
from aether.voice.integration.api import (
    synthesize_voice_sync,
    synthesize_voice_standalone,
)

# Conditionally export FastAPI router
try:
    from aether.voice.integration.api import create_voice_router
    __all_api__ = ["create_voice_router"]
except ImportError:
    __all_api__ = []

__all__ = [
    "VoiceSynthesisPipeline",
    "VoiceSynthesisInput",
    "VoiceSynthesisOutput",
    "PipelineStage",
    "PipelineProgress",
    "BatchSynthesisPipeline",
    "synthesize_voice_sync",
    "synthesize_voice_standalone",
] + __all_api__
