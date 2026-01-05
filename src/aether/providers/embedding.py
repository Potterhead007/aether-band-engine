"""
AETHER Embedding Provider

Production-grade embedding generation for similarity and originality checks.

Supported Providers:
- SentenceTransformers (local, recommended)
- OpenAI Embeddings (API-based)
- Mock (for testing)

Features:
- Text embedding for lyrics/concepts
- Audio embedding for musical similarity
- Batch processing with caching
- Cosine similarity computation
- Vector database integration ready

Example:
    provider = SentenceTransformerEmbeddingProvider()
    await provider.initialize()

    embedding = await provider.embed_text("Walking in the rain")
    similarity = await provider.similarity(embed1, embed2)
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import numpy as np

from aether.core.exceptions import MissingConfigError
from aether.providers.base import (
    EmbeddingProvider,
    EmbeddingResult,
    ProviderInfo,
    ProviderStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Embedding Cache
# ============================================================================


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: dict[str, EmbeddingResult] = {}

    def _hash_text(self, text: str, model: str) -> str:
        """Create cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, text: str, model: str) -> EmbeddingResult | None:
        """Get cached embedding."""
        key = self._hash_text(text, model)
        return self._cache.get(key)

    def set(self, text: str, model: str, result: EmbeddingResult) -> None:
        """Cache embedding result."""
        if len(self._cache) >= self.max_size:
            # Simple eviction: remove oldest 10%
            keys = list(self._cache.keys())[: int(self.max_size * 0.1)]
            for key in keys:
                del self._cache[key]

        key = self._hash_text(text, model)
        self._cache[key] = result

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()


# ============================================================================
# Sentence Transformer Provider
# ============================================================================


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using sentence-transformers library.

    Recommended models:
    - all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
    - all-mpnet-base-v2 (slower, best quality, 768 dimensions)
    - paraphrase-multilingual-MiniLM-L12-v2 (multilingual)

    Example:
        provider = SentenceTransformerEmbeddingProvider(
            model_name="all-MiniLM-L6-v2"
        )
        await provider.initialize()
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_enabled: bool = True,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self._model = None
        self._cache = EmbeddingCache() if cache_enabled else None

    def get_info(self) -> ProviderInfo:
        dimensions = 384 if "MiniLM" in self.model_name else 768
        return ProviderInfo(
            name="SentenceTransformer Embedding Provider",
            version="1.0.0",
            provider_type="embedding",
            status=self._status,
            capabilities=["text_embedding", "batch_embedding", "similarity"],
            config={
                "model": self.model_name,
                "dimensions": dimensions,
                "cache_enabled": self.cache_enabled,
            },
        )

    async def initialize(self) -> bool:
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._status = ProviderStatus.AVAILABLE
            logger.info(f"SentenceTransformer initialized with model: {self.model_name}")
            return True
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install 'aether-band-engine[ml]' or pip install sentence-transformers"
            )
            self._status = ProviderStatus.UNAVAILABLE
            return False
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            self._status = ProviderStatus.UNAVAILABLE
            return False

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._model = None
        if self._cache:
            self._cache.clear()
        self._status = ProviderStatus.UNAVAILABLE

    async def health_check(self) -> bool:
        return self._model is not None

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for text."""
        if not self._model:
            raise RuntimeError("Provider not initialized")

        # Check cache
        if self._cache:
            cached = self._cache.get(text, self.model_name)
            if cached:
                return cached

        # Generate embedding
        embedding = self._model.encode(text, convert_to_numpy=True)

        result = EmbeddingResult(
            embedding=embedding.tolist(),
            model=self.model_name,
            dimensions=len(embedding),
        )

        # Cache result
        if self._cache:
            self._cache.set(text, self.model_name, result)

        return result

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        if not self._model:
            raise RuntimeError("Provider not initialized")

        results = []
        to_embed = []
        to_embed_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if self._cache:
                cached = self._cache.get(text, self.model_name)
                if cached:
                    results.append((i, cached))
                    continue
            to_embed.append(text)
            to_embed_indices.append(i)

        # Batch encode uncached texts
        if to_embed:
            embeddings = self._model.encode(to_embed, convert_to_numpy=True)
            for idx, embedding in zip(to_embed_indices, embeddings):
                result = EmbeddingResult(
                    embedding=embedding.tolist(),
                    model=self.model_name,
                    dimensions=len(embedding),
                )
                results.append((idx, result))

                # Cache
                if self._cache:
                    self._cache.set(texts[idx], self.model_name, result)

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    async def similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity between embeddings."""
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)

        dot = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))


# ============================================================================
# OpenAI Embedding Provider
# ============================================================================


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using OpenAI API.

    Models:
    - text-embedding-3-small (1536 dimensions, fast)
    - text-embedding-3-large (3072 dimensions, best quality)
    - text-embedding-ada-002 (legacy, 1536 dimensions)

    Example:
        provider = OpenAIEmbeddingProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model="text-embedding-3-small",
        )
        await provider.initialize()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        cache_enabled: bool = True,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise MissingConfigError(
                "OPENAI_API_KEY",
                recovery_hints=[
                    "Set OPENAI_API_KEY environment variable",
                    "Pass api_key parameter to OpenAIEmbeddingProvider()",
                    "Use SentenceTransformerEmbeddingProvider for local embeddings",
                    "Use MockEmbeddingProvider for testing",
                ],
            )
        self.model = model
        self.cache_enabled = cache_enabled
        self._client = None
        self._cache = EmbeddingCache() if cache_enabled else None

    def get_info(self) -> ProviderInfo:
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }.get(self.model, 1536)

        return ProviderInfo(
            name="OpenAI Embedding Provider",
            version="1.0.0",
            provider_type="embedding",
            status=self._status,
            capabilities=["text_embedding", "batch_embedding", "similarity"],
            config={
                "model": self.model,
                "dimensions": dimensions,
            },
        )

    async def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            import openai

            self._client = openai.AsyncOpenAI(api_key=self.api_key)
            self._status = ProviderStatus.AVAILABLE
            logger.info(f"OpenAI Embedding provider initialized with model: {self.model}")
            return True
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            self._status = ProviderStatus.UNAVAILABLE
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI Embedding provider: {e}")
            self._status = ProviderStatus.UNAVAILABLE
            return False

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._client = None
        if self._cache:
            self._cache.clear()
        self._status = ProviderStatus.UNAVAILABLE

    async def health_check(self) -> bool:
        return self._client is not None

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for text."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        # Check cache
        if self._cache:
            cached = self._cache.get(text, self.model)
            if cached:
                return cached

        response = await self._client.embeddings.create(
            model=self.model,
            input=text,
        )

        embedding = response.data[0].embedding

        result = EmbeddingResult(
            embedding=embedding,
            model=self.model,
            dimensions=len(embedding),
        )

        # Cache result
        if self._cache:
            self._cache.set(text, self.model, result)

        return result

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        results = []
        to_embed = []
        to_embed_indices = []

        # Check cache
        for i, text in enumerate(texts):
            if self._cache:
                cached = self._cache.get(text, self.model)
                if cached:
                    results.append((i, cached))
                    continue
            to_embed.append(text)
            to_embed_indices.append(i)

        # Batch API call
        if to_embed:
            response = await self._client.embeddings.create(
                model=self.model,
                input=to_embed,
            )

            for idx, data in zip(to_embed_indices, response.data):
                result = EmbeddingResult(
                    embedding=data.embedding,
                    model=self.model,
                    dimensions=len(data.embedding),
                )
                results.append((idx, result))

                if self._cache:
                    self._cache.set(texts[idx], self.model, result)

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    async def similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity."""
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)

        dot = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))


# ============================================================================
# Mock Embedding Provider
# ============================================================================


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing.

    Generates deterministic pseudo-random embeddings based on text content.
    """

    def __init__(
        self,
        dimensions: int = 384,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.dimensions = dimensions

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Mock Embedding Provider",
            version="1.0.0",
            provider_type="embedding",
            status=self._status,
            capabilities=["text_embedding", "batch_embedding", "similarity"],
            config={"dimensions": self.dimensions, "mock": True},
        )

    async def initialize(self) -> bool:
        self._status = ProviderStatus.AVAILABLE
        return True

    async def shutdown(self) -> None:
        self._status = ProviderStatus.UNAVAILABLE

    async def health_check(self) -> bool:
        return self._status == ProviderStatus.AVAILABLE

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate deterministic mock embedding."""
        # Use text hash as seed for reproducibility
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)

        # Generate normalized embedding
        embedding = rng.randn(self.dimensions)
        embedding = embedding / np.linalg.norm(embedding)

        return EmbeddingResult(
            embedding=embedding.tolist(),
            model="mock",
            dimensions=self.dimensions,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate mock embeddings for batch."""
        return [await self.embed_text(text) for text in texts]

    async def similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity."""
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)

        dot = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))


# ============================================================================
# Audio Embedding Provider
# ============================================================================


class AudioEmbeddingProvider:
    """
    Audio embedding for musical similarity checks.

    Uses spectral features to create audio fingerprints.
    This is a simple implementation - for production, consider using
    dedicated audio fingerprinting libraries.
    """

    def __init__(self, sample_rate: int = 48000, n_mels: int = 128):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    async def embed_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Generate embedding from audio.

        Returns fixed-size embedding vector based on mel spectrogram statistics.
        """
        try:
            import librosa
        except ImportError:
            # Fallback: simple statistical features
            return self._embed_simple(audio)

        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)

        # Convert stereo to mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
        )

        # Log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Extract statistics as embedding
        embedding = np.concatenate(
            [
                np.mean(mel_spec_db, axis=1),  # Mean per band
                np.std(mel_spec_db, axis=1),  # Std per band
                np.max(mel_spec_db, axis=1),  # Max per band
                np.min(mel_spec_db, axis=1),  # Min per band
            ]
        )

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        return embedding

    def _embed_simple(self, audio: np.ndarray) -> np.ndarray:
        """Simple embedding without librosa."""
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Basic spectral features using numpy FFT
        n_fft = 2048
        hop_length = 512
        n_frames = (len(audio) - n_fft) // hop_length + 1

        if n_frames < 1:
            return np.zeros(256)

        features = []
        for i in range(min(n_frames, 100)):  # Limit frames
            start = i * hop_length
            frame = audio[start : start + n_fft]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))

            # Band energies
            n_bands = 32
            band_size = len(spectrum) // n_bands
            band_energies = [
                np.sum(spectrum[j * band_size : (j + 1) * band_size] ** 2) for j in range(n_bands)
            ]
            features.append(band_energies)

        features = np.array(features)

        # Statistics
        embedding = np.concatenate(
            [
                np.mean(features, axis=0),
                np.std(features, axis=0),
                np.max(features, axis=0),
                np.min(features, axis=0),
            ]
        )

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        # Pad/truncate to fixed size
        target_size = 256
        if len(embedding) < target_size:
            embedding = np.pad(embedding, (0, target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]

        return embedding

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between audio embeddings."""
        dot = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))


# ============================================================================
# Provider Factory
# ============================================================================


def create_embedding_provider(
    provider_type: str = "sentence_transformer",
    **kwargs,
) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.

    Args:
        provider_type: "sentence_transformer", "openai", or "mock"
        **kwargs: Provider-specific configuration

    Returns:
        Configured embedding provider
    """
    providers = {
        "sentence_transformer": SentenceTransformerEmbeddingProvider,
        "st": SentenceTransformerEmbeddingProvider,
        "openai": OpenAIEmbeddingProvider,
        "mock": MockEmbeddingProvider,
    }

    provider_class = providers.get(provider_type.lower())
    if not provider_class:
        raise ValueError(
            f"Unknown provider type: {provider_type}. " f"Available: {list(providers.keys())}"
        )

    return provider_class(**kwargs)


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Providers
    "SentenceTransformerEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "MockEmbeddingProvider",
    "AudioEmbeddingProvider",
    # Utilities
    "EmbeddingCache",
    # Factory
    "create_embedding_provider",
]
