"""
AETHER Provider Manager

Centralized management of all providers - initialization, configuration,
and lifecycle management for the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from aether.providers.base import (
    ProviderRegistry,
    get_provider_registry,
    BaseProvider,
)
from aether.providers.llm import (
    MockLLMProvider,
    ClaudeLLMProvider,
    OpenAILLMProvider,
)
from aether.providers.midi import AlgorithmicMIDIProvider
from aether.providers.audio import SynthAudioProvider
from aether.providers.embedding import (
    MockEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    OpenAIEmbeddingProvider,
)

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for provider selection and settings."""

    # LLM provider
    llm_provider: str = "mock"  # mock, claude, openai
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None

    # MIDI provider
    midi_provider: str = "algorithmic"  # algorithmic

    # Audio provider
    audio_provider: str = "synth"  # synth
    audio_sample_rate: int = 48000
    soundfont_path: Optional[str] = None

    # Embedding provider
    embedding_provider: str = "mock"  # mock, sentence_transformer, openai
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None

    # Additional settings
    extra: Dict[str, Any] = field(default_factory=dict)


class ProviderManager:
    """
    Manages provider lifecycle for the AETHER pipeline.

    Usage:
        manager = ProviderManager(config)
        await manager.initialize()

        # Providers now available via get_provider_registry()
        llm = get_provider_registry().get("llm")

        await manager.shutdown()
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config or ProviderConfig()
        self.registry = get_provider_registry()
        self._initialized = False
        self._providers_created: Dict[str, BaseProvider] = {}

    async def initialize(self) -> Dict[str, bool]:
        """
        Initialize all configured providers and register them.

        Returns:
            Dict mapping provider name to initialization success status.
        """
        if self._initialized:
            logger.warning("ProviderManager already initialized")
            return {}

        results = {}

        # Create and register LLM provider
        llm = self._create_llm_provider()
        if llm:
            self.registry.register("llm", llm)
            self._providers_created["llm"] = llm
            try:
                await llm.initialize()
                results["llm"] = True
                logger.info(f"LLM provider initialized: {self.config.llm_provider}")
            except Exception as e:
                logger.error(f"LLM provider failed to initialize: {e}")
                results["llm"] = False

        # Create and register MIDI provider
        midi = self._create_midi_provider()
        if midi:
            self.registry.register("midi", midi)
            self._providers_created["midi"] = midi
            try:
                await midi.initialize()
                results["midi"] = True
                logger.info(f"MIDI provider initialized: {self.config.midi_provider}")
            except Exception as e:
                logger.error(f"MIDI provider failed to initialize: {e}")
                results["midi"] = False

        # Create and register Audio provider
        audio = self._create_audio_provider()
        if audio:
            self.registry.register("audio", audio)
            self._providers_created["audio"] = audio
            try:
                await audio.initialize()
                results["audio"] = True
                logger.info(f"Audio provider initialized: {self.config.audio_provider}")
            except Exception as e:
                logger.error(f"Audio provider failed to initialize: {e}")
                results["audio"] = False

        # Create and register Embedding provider
        embedding = self._create_embedding_provider()
        if embedding:
            self.registry.register("embedding", embedding)
            self._providers_created["embedding"] = embedding
            try:
                await embedding.initialize()
                results["embedding"] = True
                logger.info(f"Embedding provider initialized: {self.config.embedding_provider}")
            except Exception as e:
                logger.error(f"Embedding provider failed to initialize: {e}")
                results["embedding"] = False

        self._initialized = True
        logger.info(f"Provider initialization complete: {sum(results.values())}/{len(results)} successful")

        return results

    async def shutdown(self) -> None:
        """Shutdown all providers."""
        for name, provider in self._providers_created.items():
            try:
                await provider.shutdown()
                logger.debug(f"Provider {name} shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")

        self._initialized = False
        self._providers_created.clear()

    def _create_llm_provider(self) -> Optional[BaseProvider]:
        """Create LLM provider based on config."""
        provider_type = self.config.llm_provider.lower()

        if provider_type == "mock":
            return MockLLMProvider()

        elif provider_type == "claude":
            return ClaudeLLMProvider(
                model=self.config.llm_model or "claude-sonnet-4-20250514",
                api_key=self.config.llm_api_key,
            )

        elif provider_type == "openai":
            return OpenAILLMProvider(
                model=self.config.llm_model or "gpt-4",
                api_key=self.config.llm_api_key,
            )

        else:
            logger.warning(f"Unknown LLM provider: {provider_type}, using mock")
            return MockLLMProvider()

    def _create_midi_provider(self) -> Optional[BaseProvider]:
        """Create MIDI provider based on config."""
        provider_type = self.config.midi_provider.lower()

        if provider_type == "algorithmic":
            return AlgorithmicMIDIProvider()

        else:
            logger.warning(f"Unknown MIDI provider: {provider_type}, using algorithmic")
            return AlgorithmicMIDIProvider()

    def _create_audio_provider(self) -> Optional[BaseProvider]:
        """Create Audio provider based on config."""
        provider_type = self.config.audio_provider.lower()

        if provider_type == "synth":
            return SynthAudioProvider(
                sample_rate=self.config.audio_sample_rate,
            )

        else:
            logger.warning(f"Unknown Audio provider: {provider_type}, using synth")
            return SynthAudioProvider()

    def _create_embedding_provider(self) -> Optional[BaseProvider]:
        """Create Embedding provider based on config."""
        provider_type = self.config.embedding_provider.lower()

        if provider_type == "mock":
            return MockEmbeddingProvider()

        elif provider_type == "sentence_transformer":
            return SentenceTransformerEmbeddingProvider(
                model_name=self.config.embedding_model or "all-MiniLM-L6-v2",
            )

        elif provider_type == "openai":
            return OpenAIEmbeddingProvider(
                model=self.config.embedding_model or "text-embedding-3-small",
                api_key=self.config.embedding_api_key,
            )

        else:
            logger.warning(f"Unknown Embedding provider: {provider_type}, using mock")
            return MockEmbeddingProvider()

    @property
    def is_initialized(self) -> bool:
        """Check if providers are initialized."""
        return self._initialized

    def get_status(self) -> Dict[str, str]:
        """Get status of all providers."""
        return {
            name: provider.status.value
            for name, provider in self._providers_created.items()
        }


# Convenience function
async def setup_providers(config: Optional[ProviderConfig] = None) -> ProviderManager:
    """
    Setup and initialize all providers.

    Returns initialized ProviderManager.
    """
    manager = ProviderManager(config)
    await manager.initialize()
    return manager
