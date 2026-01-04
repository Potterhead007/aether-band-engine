"""
Integration tests for AETHER Providers.

Tests the full provider stack including:
- LLM providers (Mock, Claude, OpenAI)
- MIDI generation (Algorithmic)
- Audio synthesis and mixing
- Embedding providers (Mock, Audio)
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path

from aether.providers import (
    # Base types
    MIDIFile,
    MIDITrack,
    MIDINote,
    AudioBuffer,
    AudioStem,
    LLMMessage,
    EmbeddingResult,
    # LLM
    MockLLMProvider,
    CreativePrompts,
    # MIDI
    AlgorithmicMIDIProvider,
    CHORD_INTERVALS,
    SCALE_INTERVALS,
    GM_DRUMS,
    # Audio
    SynthAudioProvider,
    # Embedding
    MockEmbeddingProvider,
    AudioEmbeddingProvider,
)


# ============================================================================
# LLM Provider Tests
# ============================================================================


class TestMockLLMProvider:
    """Test MockLLMProvider functionality."""

    @pytest.fixture
    async def provider(self):
        """Create and initialize provider."""
        provider = MockLLMProvider()
        await provider.initialize()
        yield provider
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_shutdown(self):
        """Test provider lifecycle."""
        provider = MockLLMProvider()
        await provider.initialize()
        # Provider should be ready after initialize
        response = await provider.complete([LLMMessage(role="user", content="test")])
        assert response.content is not None
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_complete_simple(self, provider):
        """Test simple completion."""
        response = await provider.complete([LLMMessage(role="user", content="Hello")])
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_complete_with_system(self, provider):
        """Test completion with system message."""
        response = await provider.complete(
            [
                LLMMessage(role="system", content="You are a helpful assistant."),
                LLMMessage(role="user", content="Write a haiku"),
            ]
        )
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_complete_json_mode(self, provider):
        """Test JSON mode completion."""
        response = await provider.complete(
            [LLMMessage(role="user", content="Return a JSON object")], json_mode=True
        )
        assert response.content is not None


class TestCreativePrompts:
    """Test creative prompt generation."""

    def test_lyrics_generation_prompt(self):
        """Test lyrics generation prompt structure."""
        messages = CreativePrompts.lyrics_generation(
            genre="boom-bap",
            mood="introspective",
            theme="urban struggle",
            structure=["verse", "chorus", "verse"],
        )
        assert len(messages) >= 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"

    def test_concept_album_prompt(self):
        """Test concept album prompt structure."""
        messages = CreativePrompts.concept_album(
            genre="synthwave", concept="neon nights", track_count=8
        )
        assert len(messages) >= 2


# ============================================================================
# MIDI Provider Tests
# ============================================================================


class TestAlgorithmicMIDIProvider:
    """Test AlgorithmicMIDIProvider functionality."""

    @pytest.fixture
    async def provider(self):
        """Create and initialize provider."""
        provider = AlgorithmicMIDIProvider()
        await provider.initialize()
        yield provider
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_shutdown(self):
        """Test provider lifecycle."""
        provider = AlgorithmicMIDIProvider()
        await provider.initialize()
        # Provider should be ready after initialize
        midi = await provider.generate_from_spec(
            harmony_spec={"progression": ["C"]},
            melody_spec={},
            rhythm_spec={"bpm": 120},
            arrangement_spec={"sections": ["verse"]},
        )
        assert midi is not None
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_generate_from_spec(self, provider):
        """Test MIDI generation from spec."""
        midi = await provider.generate_from_spec(
            harmony_spec={"progression": ["Cm", "Ab", "Eb", "Bb"], "key": "C", "mode": "minor"},
            melody_spec={"contour": "arch", "range_octaves": 1.5},
            rhythm_spec={"bpm": 90, "time_signature": (4, 4)},
            arrangement_spec={"sections": ["verse"]},
        )

        assert isinstance(midi, MIDIFile)
        assert len(midi.tracks) > 0

        # Check track structure
        for track in midi.tracks:
            assert hasattr(track, "name")
            assert hasattr(track, "notes")
            assert len(track.notes) > 0

    @pytest.mark.asyncio
    async def test_generate_drums(self, provider):
        """Test drum generation."""
        midi = await provider.generate_from_spec(
            harmony_spec={"progression": ["C"]},
            melody_spec={},
            rhythm_spec={"bpm": 120, "time_signature": (4, 4)},
            arrangement_spec={"sections": ["intro"]},
        )

        # Find drums track
        drums = next((t for t in midi.tracks if "drum" in t.name.lower()), None)
        assert drums is not None
        assert len(drums.notes) > 0


class TestMusicTheoryConstants:
    """Test music theory constants."""

    def test_chord_intervals(self):
        """Test chord interval definitions."""
        assert "major" in CHORD_INTERVALS
        assert "minor" in CHORD_INTERVALS
        assert CHORD_INTERVALS["major"] == [0, 4, 7]
        assert CHORD_INTERVALS["minor"] == [0, 3, 7]

    def test_scale_intervals(self):
        """Test scale interval definitions."""
        assert "major" in SCALE_INTERVALS
        assert "minor" in SCALE_INTERVALS
        assert len(SCALE_INTERVALS["major"]) == 7

    def test_gm_drums(self):
        """Test GM drum map."""
        assert "kick" in GM_DRUMS
        assert "snare" in GM_DRUMS
        assert GM_DRUMS["kick"] == 36


# ============================================================================
# Audio Provider Tests
# ============================================================================


class TestSynthAudioProvider:
    """Test SynthAudioProvider functionality."""

    @pytest.fixture
    async def provider(self):
        """Create and initialize provider."""
        provider = SynthAudioProvider()
        await provider.initialize()
        yield provider
        await provider.shutdown()

    @pytest.fixture
    def simple_midi(self):
        """Create simple MIDI for testing."""
        return MIDIFile(
            tracks=[
                MIDITrack(
                    name="Test",
                    channel=0,
                    notes=[
                        MIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.5),
                        MIDINote(pitch=64, velocity=100, start_time=0.5, duration=0.5),
                        MIDINote(pitch=67, velocity=100, start_time=1.0, duration=0.5),
                    ],
                )
            ],
            tempo_bpm=120.0,
            time_signature=(4, 4),
        )

    @pytest.mark.asyncio
    async def test_initialize_shutdown(self):
        """Test provider lifecycle."""
        provider = SynthAudioProvider()
        await provider.initialize()
        # Provider should be ready - test with simple MIDI
        midi = MIDIFile(
            tracks=[
                MIDITrack(
                    name="Test",
                    channel=0,
                    notes=[MIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.5)],
                )
            ],
            tempo_bpm=120.0,
            time_signature=(4, 4),
        )
        audio = await provider.render_midi(midi)
        assert audio is not None
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_render_midi(self, provider, simple_midi):
        """Test MIDI rendering."""
        audio = await provider.render_midi(simple_midi)

        assert isinstance(audio, AudioBuffer)
        assert audio.data is not None
        assert audio.sample_rate > 0
        assert audio.data.shape[-1] > 0

    @pytest.mark.asyncio
    async def test_mix_stems(self, provider):
        """Test stem mixing."""
        # Create test stems
        sr = 44100
        duration = 1.0
        samples = int(sr * duration)

        buffer1 = AudioBuffer(data=np.random.randn(2, samples) * 0.1, sample_rate=sr, channels=2)
        buffer2 = AudioBuffer(data=np.random.randn(2, samples) * 0.1, sample_rate=sr, channels=2)

        stems = [
            AudioStem(name="lead", buffer=buffer1, category="synth"),
            AudioStem(name="bass", buffer=buffer2, category="bass"),
        ]

        mixed = await provider.mix_stems(stems, {"lead": -3, "bass": 0})

        assert isinstance(mixed, AudioBuffer)
        assert mixed.data.shape[-1] == samples


# ============================================================================
# Embedding Provider Tests
# ============================================================================


class TestMockEmbeddingProvider:
    """Test MockEmbeddingProvider functionality."""

    @pytest.fixture
    async def provider(self):
        """Create and initialize provider."""
        provider = MockEmbeddingProvider()
        await provider.initialize()
        yield provider
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_shutdown(self):
        """Test provider lifecycle."""
        provider = MockEmbeddingProvider()
        await provider.initialize()
        # Provider should be ready
        result = await provider.embed_text("test")
        assert result is not None
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_embed_text(self, provider):
        """Test text embedding."""
        result = await provider.embed_text("Hello, world!")

        assert isinstance(result, EmbeddingResult)
        assert result.embedding is not None
        assert len(result.embedding) > 0

    @pytest.mark.asyncio
    async def test_embed_batch(self, provider):
        """Test batch text embedding."""
        texts = ["Hello", "World", "Test"]
        results = await provider.embed_batch(texts)

        assert len(results) == 3
        for result in results:
            assert len(result.embedding) > 0

    @pytest.mark.asyncio
    async def test_similarity(self, provider):
        """Test similarity computation."""
        r1 = await provider.embed_text("cat")
        r2 = await provider.embed_text("dog")

        sim = await provider.similarity(np.array(r1.embedding), np.array(r2.embedding))

        assert -1 <= sim <= 1


class TestAudioEmbeddingProvider:
    """Test AudioEmbeddingProvider functionality."""

    @pytest.fixture
    def provider(self):
        """Create provider."""
        return AudioEmbeddingProvider()

    @pytest.mark.asyncio
    async def test_embed_audio(self, provider):
        """Test audio embedding."""
        # Create test audio (1 second of noise)
        audio = np.random.randn(44100)

        embedding = await provider.embed_audio(audio, 44100)

        assert embedding is not None
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0

    @pytest.mark.asyncio
    async def test_embed_audio_stereo(self, provider):
        """Test stereo audio embedding."""
        # Create stereo test audio
        audio = np.random.randn(2, 44100)

        embedding = await provider.embed_audio(audio, 44100)

        assert embedding is not None
        assert len(embedding.shape) == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestFullPipeline:
    """Test full provider pipeline integration."""

    @pytest.mark.asyncio
    async def test_llm_to_midi_to_audio(self):
        """Test full pipeline: LLM -> MIDI -> Audio."""
        # 1. Get song concept from LLM
        llm = MockLLMProvider()
        await llm.initialize()

        response = await llm.complete(
            [LLMMessage(role="user", content="Generate a song concept for boom-bap")]
        )
        assert response.content is not None
        await llm.shutdown()

        # 2. Generate MIDI
        midi_provider = AlgorithmicMIDIProvider()
        await midi_provider.initialize()

        midi = await midi_provider.generate_from_spec(
            harmony_spec={"progression": ["Am", "F", "C", "G"], "key": "A", "mode": "minor"},
            melody_spec={"contour": "wave"},
            rhythm_spec={"bpm": 90, "time_signature": (4, 4)},
            arrangement_spec={"sections": ["verse"]},
        )
        assert len(midi.tracks) > 0
        await midi_provider.shutdown()

        # 3. Render to audio
        audio_provider = SynthAudioProvider()
        await audio_provider.initialize()

        audio = await audio_provider.render_midi(midi)
        assert audio.data.shape[-1] > 0
        await audio_provider.shutdown()

        # 4. Generate embedding for similarity check
        emb_provider = MockEmbeddingProvider()
        await emb_provider.initialize()

        result = await emb_provider.embed_text(response.content)
        assert len(result.embedding) > 0
        await emb_provider.shutdown()

    @pytest.mark.asyncio
    async def test_provider_graceful_operation(self):
        """Test provider operates gracefully."""
        # Mock providers are designed to work without external dependencies
        provider = MockLLMProvider()
        await provider.initialize()

        # Should work with various inputs
        response = await provider.complete([LLMMessage(role="user", content="")])  # Empty content
        assert response is not None

        # Should handle multiple calls
        for i in range(3):
            response = await provider.complete([LLMMessage(role="user", content=f"Test {i}")])
            assert response.content is not None

        await provider.shutdown()
