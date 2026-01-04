"""
AETHER Providers Layer

Pluggable providers for MIDI, Audio, Vocal, LLM, and Embedding generation.
Includes concrete implementations for production use.
"""

from aether.providers.base import (
    BaseProvider,
    ProviderInfo,
    ProviderStatus,
    ProviderRegistry,
    get_provider_registry,
    # MIDI
    MIDIProvider,
    MIDIFile,
    MIDITrack,
    MIDINote,
    # Audio
    AudioProvider,
    AudioBuffer,
    AudioStem,
    # Vocal
    VocalProvider,
    VoiceProfile,
    VocalRequest,
    # LLM
    LLMProvider,
    LLMMessage,
    LLMResponse,
    # Embedding
    EmbeddingProvider,
    EmbeddingResult,
)

# Concrete implementations
from aether.providers.llm import (
    ClaudeLLMProvider,
    OpenAILLMProvider,
    MockLLMProvider,
    CreativePrompts,
    create_llm_provider,
)

from aether.providers.midi import (
    AlgorithmicMIDIProvider,
    CHORD_INTERVALS,
    SCALE_INTERVALS,
    GM_DRUMS,
)

from aether.providers.audio import (
    SynthAudioProvider,
)

from aether.providers.embedding import (
    SentenceTransformerEmbeddingProvider,
    OpenAIEmbeddingProvider,
    MockEmbeddingProvider,
    AudioEmbeddingProvider,
    create_embedding_provider,
)

from aether.providers.manager import (
    ProviderManager,
    ProviderConfig,
    setup_providers,
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
