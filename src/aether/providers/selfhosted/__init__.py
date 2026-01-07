"""
Self-Hosted Voice Provider Package

Provides local XTTS + RVC voice synthesis without external API dependencies.
"""

from .config import (
    SelfHostedConfig,
    XTTSConfig,
    RVCConfig,
    AVUVoiceMapping,
    AVU_VOICE_CONFIGS,
)
from .provider import (
    SelfHostedVocalProvider,
    SynthesisStage,
    SynthesisProgress,
    get_selfhosted_provider,
    is_selfhosted_configured,
)
from .xtts import XTTSEngine
from .rvc import RVCEngine
from .bark import (
    BarkSingingProvider,
    BarkConfig,
    SingingPhrase,
    SingingResult,
    get_bark_provider,
)

__all__ = [
    # Main provider
    "SelfHostedVocalProvider",
    "get_selfhosted_provider",
    "is_selfhosted_configured",
    # Engines
    "XTTSEngine",
    "RVCEngine",
    # Config
    "SelfHostedConfig",
    "XTTSConfig",
    "RVCConfig",
    "AVUVoiceMapping",
    "AVU_VOICE_CONFIGS",
    # Progress tracking
    "SynthesisStage",
    "SynthesisProgress",
    # Bark singing
    "BarkSingingProvider",
    "BarkConfig",
    "SingingPhrase",
    "SingingResult",
    "get_bark_provider",
]
