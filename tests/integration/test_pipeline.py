"""
Integration tests for AETHER music generation pipeline.
"""

import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np


class TestPipelineIntegration:
    """Integration tests for full pipeline."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        mock = MagicMock()
        mock.generate = AsyncMock(return_value="Generated content")
        mock.generate_structured = AsyncMock(return_value={"key": "value"})
        return mock

    @pytest.fixture
    def mock_midi_provider(self):
        """Create mock MIDI provider."""
        mock = MagicMock()
        mock.generate_from_spec = AsyncMock()
        return mock

    @pytest.fixture
    def mock_audio_provider(self):
        """Create mock audio provider."""
        mock = MagicMock()
        # Return stereo audio
        mock.render_midi = AsyncMock(return_value=np.random.randn(2, 48000 * 30).astype(np.float32))
        mock.sample_rate = 48000
        return mock

    @pytest.mark.asyncio
    async def test_provider_manager_initialization(self):
        """Test provider manager initializes all providers."""
        from aether.providers import ProviderManager, ProviderConfig

        config = ProviderConfig(
            llm_provider="mock",
            midi_provider="internal",
            audio_provider="mock",
            embedding_provider="mock",
        )

        manager = ProviderManager(config)
        results = await manager.initialize()

        # At least some providers should initialize
        assert isinstance(results, dict)
        await manager.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="T-002: Requires deep mocking of mastering chain AudioBuffer types")
    async def test_rendering_engine_mock(self):
        """Test rendering engine with mock data."""
        from aether.rendering import RenderingEngine, RenderingConfig

        config = RenderingConfig(
            output_dir=Path(tempfile.mkdtemp()),
            sample_rate=48000,
            export_formats=["wav"],
        )

        engine = RenderingEngine(config)

        # Create minimal pipeline output
        pipeline_output = {
            "song_spec": {"id": "test-song", "title": "Test Song"},
            "arrangement_spec": {
                "sections": [{"section_type": "verse", "length_bars": 8}],
                "instruments": [{"name": "kick", "category": "drums"}],
            },
            "rhythm_spec": {"tempo_bpm": 120},
            "harmony_spec": {"key": "C", "mode": "major"},
            "melody_spec": {"phrases": []},
            "sound_design_spec": {"instrument_patches": []},
            "mix_spec": {
                "buses": [],
                "tracks": [],
                "master_eq": [],
            },
            "master_spec": {
                "loudness": {"target_lufs": -14.0},
                "multiband_compression": [],
                "limiter": {"ceiling_dbtp": -1.0},
            },
        }

        # Mock the internal methods
        with patch.object(engine, "_generate_midi") as mock_midi:
            with patch.object(engine, "_render_to_stems") as mock_stems:
                with patch.object(engine, "_mix_stems") as mock_mix:
                    mock_midi.return_value = MagicMock()
                    mock_stems.return_value = {"kick": np.random.randn(2, 48000 * 10)}
                    mock_mix.return_value = np.random.randn(2, 48000 * 10)

                    result = await engine.render(pipeline_output)

                    assert result.success
                    assert mock_midi.called
                    assert mock_stems.called
                    assert mock_mix.called


class TestAgentPipelineIntegration:
    """Test agent pipeline integration."""

    @pytest.mark.asyncio
    async def test_creative_director_to_composition(self):
        """Test creative director output feeds into composition."""
        from aether.agents.creative_director import CreativeDirectorAgent, CreativeDirectorInput
        from aether.agents.composition import CompositionAgent, CompositionInput

        # Run creative director with proper input schema using process() method
        # Use 'synthwave' genre which is available in the genre manager
        cd_agent = CreativeDirectorAgent()
        cd_input = CreativeDirectorInput(
            title="Synthwave Banger",
            genre_id="synthwave",
            creative_brief="upbeat electronic synthwave track",
        )
        cd_result = await cd_agent.process(cd_input, context={})

        song_spec = cd_result.song_spec

        # Verify song spec has required fields for composition
        assert "id" in song_spec
        assert "tempo_bpm" in song_spec or "primary_mood" in song_spec

        # Run composition with song spec - use actual genre profile since it exists
        comp_agent = CompositionAgent()
        comp_input = CompositionInput(song_spec=song_spec, genre_profile_id="synthwave")
        comp_result = await comp_agent.process(comp_input, context={})

        # CompositionOutput has harmony_spec and melody_spec (not rhythm_spec)
        assert "harmony_spec" in comp_result.model_dump()
        assert "melody_spec" in comp_result.model_dump()

    @pytest.mark.asyncio
    async def test_full_agent_chain(self):
        """Test complete agent chain execution."""
        from aether.agents import (
            CreativeDirectorAgent,
            CompositionAgent,
            ArrangementAgent,
            LyricsAgent,
        )
        from aether.agents.creative_director import CreativeDirectorInput
        from aether.agents.composition import CompositionInput
        from aether.agents.arrangement import ArrangementInput
        from aether.agents.lyrics import LyricsInput

        # Creative Director - use process() method with available genre
        cd_agent = CreativeDirectorAgent()
        cd_input = CreativeDirectorInput(
            title="Sad Lo-Fi Beat",
            genre_id="lo-fi-hip-hop",  # Available genre
            creative_brief="sad introspective track about loss",
        )
        cd_result = await cd_agent.process(cd_input, context={})
        song_spec = cd_result.song_spec

        # Composition - use actual genre profile since it exists
        comp_agent = CompositionAgent()
        comp_input = CompositionInput(song_spec=song_spec, genre_profile_id="lo-fi-hip-hop")
        comp_result = await comp_agent.process(comp_input, context={})

        # Arrangement - use actual genre profile since it exists
        # ArrangementInput needs harmony_spec not rhythm_spec from composition
        arr_agent = ArrangementAgent()
        arr_input = ArrangementInput(
            song_spec=song_spec,
            harmony_spec=comp_result.harmony_spec,
            melody_spec=comp_result.melody_spec,
            genre_profile_id="lo-fi-hip-hop",
        )
        arr_result = await arr_agent.process(arr_input, context={})

        # Lyrics
        lyrics_agent = LyricsAgent()
        lyrics_input = LyricsInput(
            song_spec=song_spec,
            arrangement_spec=arr_result.arrangement_spec,
            melody_spec=comp_result.melody_spec,
        )
        lyrics_result = await lyrics_agent.process(lyrics_input, context={})

        # Verify chain completed successfully
        assert "sections" in lyrics_result.lyric_spec
        assert len(lyrics_result.lyric_spec["sections"]) > 0


class TestAudioProcessingIntegration:
    """Test audio processing chain integration."""

    @pytest.mark.asyncio
    async def test_dsp_chain(self):
        """Test DSP processing chain."""
        from aether.audio.dsp import (
            BiquadFilter,
            FilterType,
            ParametricEQ,
            Compressor,
        )

        # Create test audio
        sample_rate = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Apply highpass filter using BiquadFilter class
        hp_filter = BiquadFilter(
            filter_type=FilterType.HIGHPASS,
            sample_rate=sample_rate,
            frequency=100,
            q=0.707,
        )
        filtered = hp_filter.process_mono(audio)
        assert filtered.shape == audio.shape

        # Apply parametric EQ with stereo audio
        stereo_audio = np.array([audio, audio])
        eq = ParametricEQ(sample_rate)
        eq.add_band(FilterType.HIGHPASS, 80, 0, 0.707)
        eq.add_band(FilterType.PEAK, 3000, 2.0, 1.5)
        eq_audio = eq.process(stereo_audio)
        assert eq_audio.shape == stereo_audio.shape

        # Apply compression
        comp = Compressor(
            sample_rate=sample_rate,
            threshold_db=-20,
            ratio=4.0,
            attack_ms=10,
            release_ms=100,
        )
        compressed, _ = comp.process_stereo(stereo_audio)
        assert compressed.shape == stereo_audio.shape

    @pytest.mark.asyncio
    async def test_mixing_engine(self):
        """Test mixing engine."""
        from aether.audio.mixing import MixingEngine

        sample_rate = 48000
        engine = MixingEngine(sample_rate=sample_rate)

        # Create test stems and add tracks
        duration = 1.0
        samples = int(sample_rate * duration)

        engine.add_track("kick", np.random.randn(samples) * 0.5, output_bus="drums")
        engine.add_track("bass", np.random.randn(samples) * 0.3, output_bus="bass")
        engine.add_track("synth", np.random.randn(samples) * 0.2, output_bus="music")

        # Configure tracks
        engine.set_track_gain("kick", 0)
        engine.set_track_gain("bass", -2)
        engine.set_track_gain("synth", -4)
        engine.set_track_pan("synth", 0.3)

        # Add buses
        engine.add_bus("drums")
        engine.add_bus("bass")
        engine.add_bus("music")

        # Render the mix
        mixed = engine.render()

        # Should return stereo mix
        assert mixed.ndim == 2
        assert mixed.shape[0] == 2  # Stereo


class TestQAIntegration:
    """Test QA system integration."""

    @pytest.mark.asyncio
    async def test_technical_validation(self):
        """Test technical audio validation."""
        from aether.qa.technical import TechnicalValidator, TechnicalReport

        validator = TechnicalValidator(sample_rate=48000)

        # Create test audio (properly normalized)
        sample_rate = 48000
        duration = 5.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples)

        # Create stereo audio with reasonable levels
        audio = np.stack(
            [
                np.sin(2 * np.pi * 440 * t) * 0.5,
                np.sin(2 * np.pi * 440 * t) * 0.5,
            ]
        )

        # TechnicalValidator.validate returns a TechnicalReport object
        result = validator.validate(audio)

        # Access attributes of TechnicalReport
        assert isinstance(result, TechnicalReport)
        assert result.sample_rate == sample_rate
        assert result.duration_seconds > 0
        assert len(result.checks) > 0  # Should have some checks

    @pytest.mark.asyncio
    async def test_originality_checker(self):
        """Test originality checking."""
        from aether.qa.originality import OriginalityChecker

        checker = OriginalityChecker()

        # Test lyric checking with proper structure (lyric_spec dict)
        lyric_spec = {
            "sections": [
                {
                    "type": "verse",
                    "lines": [
                        "This is a completely original song",
                        "About unique experiences and memories",
                    ],
                }
            ]
        }
        results = checker.check_lyrics(lyric_spec)

        # Returns list of OriginalityResult
        assert isinstance(results, list)


class TestStorageIntegration:
    """Test storage system integration."""

    @pytest.mark.asyncio
    async def test_artifact_store(self):
        """Test artifact storage."""
        from aether.storage import ArtifactStore, ArtifactType

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_path=Path(tmpdir))

            # Store artifact with correct API
            metadata = store.store(
                data={"id": "test", "title": "Test Song"},
                artifact_type=ArtifactType.SONG_SPEC,
                name="Test Song Spec",
                song_id="test-song-123",
                tags={"genre": "pop"},
            )

            assert metadata is not None
            assert metadata.artifact_id is not None

            # Retrieve artifact as JSON
            retrieved = store.get_json(metadata.artifact_id)
            assert retrieved is not None
            assert retrieved["id"] == "test"
            assert retrieved["title"] == "Test Song"

            # List artifacts by song
            artifacts = store.list_by_song("test-song-123")
            assert len(artifacts) >= 1


class TestHealthAndMetrics:
    """Test health check and metrics integration."""

    @pytest.mark.asyncio
    async def test_health_check_system(self):
        """Test health check system."""
        from aether.core.health import HealthCheck, HealthStatus

        health = HealthCheck()

        @health.register("test_component")
        def check_test():
            return True, "Component healthy"

        @health.register("failing_component")
        def check_failing():
            return False, "Component unhealthy"

        # Run all checks
        system_health = await health.check_all()

        assert system_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert len(system_health.components) == 2

    def test_metrics_collection(self):
        """Test metrics collection."""
        from aether.core.metrics import MetricsCollector

        metrics = MetricsCollector("test")

        # Create and use counter
        counter = metrics.counter("requests_total", "Total requests")
        counter.inc()
        counter.inc(5)
        assert counter.get() == 6

        # Create and use gauge
        gauge = metrics.gauge("active_connections", "Active connections")
        gauge.set(10)
        gauge.inc(5)
        gauge.dec(3)
        assert gauge.get() == 12

        # Create and use histogram
        histogram = metrics.histogram("request_duration", "Request duration")
        histogram.observe(0.1)
        histogram.observe(0.2)
        histogram.observe(0.5)

        stats = histogram.get_stats()
        assert stats["count"] == 3
        assert stats["mean"] == pytest.approx(0.267, rel=0.1)

        # Collect all metrics
        all_metrics = metrics.collect()
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics


class TestResilienceIntegration:
    """Test resilience patterns integration."""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry combined with circuit breaker."""
        from aether.core.resilience import retry, CircuitBreaker

        call_count = 0
        cb = CircuitBreaker("test_integration", failure_threshold=3, timeout=0.1)

        @cb
        @retry(max_attempts=2, base_delay=0.01)
        async def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ValueError("Temporary failure")
            return "success"

        # First call: retry exhausted
        from aether.core.exceptions import RetryExhaustedError

        with pytest.raises(RetryExhaustedError):
            await flaky_service()

        assert call_count == 2  # Retried once

        # Reset for next test
        cb.reset()
        call_count = 3

        # Second call: should succeed
        result = await flaky_service()
        assert result == "success"
