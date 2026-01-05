"""
AETHER Providers Layer

Pluggable providers for MIDI, Audio, Vocal, LLM, and Embedding generation.
Includes concrete implementations for production use.
"""

from aether.providers.audio import (
    SynthAudioProvider,
)
from aether.providers.base import (
    AudioBuffer,
    # Audio
    AudioProvider,
    AudioStem,
    BaseProvider,
    # Embedding
    EmbeddingProvider,
    EmbeddingResult,
    LLMMessage,
    # LLM
    LLMProvider,
    LLMResponse,
    MIDIFile,
    MIDINote,
    # MIDI
    MIDIProvider,
    MIDITrack,
    ProviderInfo,
    ProviderRegistry,
    ProviderStatus,
    # Vocal
    VocalProvider,
    VocalRequest,
    VoiceProfile,
    get_provider_registry,
)
from aether.providers.embedding import (
    AudioEmbeddingProvider,
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_provider,
)

# Concrete implementations
from aether.providers.llm import (
    ClaudeLLMProvider,
    CreativePrompts,
    MockLLMProvider,
    OpenAILLMProvider,
    create_llm_provider,
)
from aether.providers.manager import (
    ProviderConfig,
    ProviderManager,
    setup_providers,
)
from aether.providers.midi import (
    CHORD_INTERVALS,
    GM_DRUMS,
    SCALE_INTERVALS,
    AlgorithmicMIDIProvider,
)

__all__ = [
    # Base classes and types
    "BaseProvider",
    "ProviderInfo",
    "ProviderStatus",
    "ProviderRegistry",
    "get_provider_registry",
    # MIDI types
    "MIDIProvider",
    "MIDIFile",
    "MIDITrack",
    "MIDINote",
    # Audio types
    "AudioProvider",
    "AudioBuffer",
    "AudioStem",
    # Vocal types
    "VocalProvider",
    "VoiceProfile",
    "VocalRequest",
    # LLM types
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    # Embedding types
    "EmbeddingProvider",
    "EmbeddingResult",
    # LLM implementations
    "ClaudeLLMProvider",
    "OpenAILLMProvider",
    "MockLLMProvider",
    "CreativePrompts",
    "create_llm_provider",
    # MIDI implementations
    "AlgorithmicMIDIProvider",
    "CHORD_INTERVALS",
    "SCALE_INTERVALS",
    "GM_DRUMS",
    # Audio implementations
    "SynthAudioProvider",
    # Embedding implementations
    "SentenceTransformerEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "MockEmbeddingProvider",
    "AudioEmbeddingProvider",
    "create_embedding_provider",
    # Provider management
    "ProviderManager",
    "ProviderConfig",
    "setup_providers",
]
