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
        from aether.agents.creative_director import CreativeDirectorAgent
        from aether.agents.composition import CompositionAgent

        # Run creative director
        cd_agent = CreativeDirectorAgent()
        cd_result = await cd_agent.execute(
            {"prompt": "upbeat electronic dance track", "genre_id": "edm"},
            context={},
        )

        song_spec = cd_result.song_spec

        # Verify song spec has required fields for composition
        assert "id" in song_spec
        assert "tempo_bpm" in song_spec or "primary_mood" in song_spec

        # Run composition with song spec
        comp_agent = CompositionAgent()
        with patch.object(comp_agent, "_get_genre_profile") as mock_profile:
            mock_profile.return_value = MagicMock(
                rhythm=MagicMock(
                    tempo_range=(120, 150),
                    time_signatures=["4/4"],
                ),
                harmony=MagicMock(
                    common_keys=["Am", "Em"],
                    common_modes=["minor", "phrygian"],
                ),
            )

            comp_result = await comp_agent.execute(
                {"song_spec": song_spec, "genre_profile_id": "edm"},
                context={},
            )

            assert "rhythm_spec" in comp_result.model_dump()
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

        # Creative Director
        cd_agent = CreativeDirectorAgent()
        cd_result = await cd_agent.execute(
            {"prompt": "sad ballad about loss", "genre_id": "pop"},
            context={},
        )
        song_spec = cd_result.song_spec

        # Composition
        comp_agent = CompositionAgent()
        with patch.object(comp_agent, "_get_genre_profile") as mock:
            mock.return_value = MagicMock(
                rhythm=MagicMock(tempo_range=(60, 80), time_signatures=["4/4"]),
                harmony=MagicMock(common_keys=["C", "Am"], common_modes=["major", "minor"]),
            )
            comp_result = await comp_agent.execute(
                {"song_spec": song_spec, "genre_profile_id": "pop"},
                context={},
            )

        # Arrangement
        arr_agent = ArrangementAgent()
        with patch.object(arr_agent, "_get_genre_profile") as mock:
            mock.return_value = MagicMock(
                arrangement=MagicMock(
                    typical_structures=[
                        {
                            "sections": [
                                {"type": "intro", "length_bars": 4},
                                {"type": "verse", "length_bars": 16},
                                {"type": "chorus", "length_bars": 8},
                            ]
                        }
                    ]
                ),
                instrumentation=MagicMock(
                    core_instruments=["piano", "strings"],
                    optional_instruments=["guitar"],
                ),
            )
            arr_result = await arr_agent.execute(
                {
                    "song_spec": song_spec,
                    "rhythm_spec": comp_result.rhythm_spec,
                    "harmony_spec": comp_result.harmony_spec,
                    "genre_profile_id": "pop",
                },
                context={},
            )

        # Lyrics
        lyrics_agent = LyricsAgent()
        lyrics_result = await lyrics_agent.execute(
            {
                "song_spec": song_spec,
                "arrangement_spec": arr_result.arrangement_spec,
                "melody_spec": comp_result.melody_spec,
            },
            context={},
        )

        # Verify chain completed successfully
        assert "sections" in lyrics_result.lyric_spec
        assert len(lyrics_result.lyric_spec["sections"]) > 0


class TestAudioProcessingIntegration:
    """Test audio processing chain integration."""

    @pytest.mark.asyncio
    async def test_dsp_chain(self):
        """Test DSP processing chain."""
        from aether.audio.dsp import (
            create_biquad_filter,
            apply_filter,
            ParametricEQ,
            Compressor,
        )

        # Create test audio
        sample_rate = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Apply highpass filter
        hp_coeffs = create_biquad_filter("highpass", sample_rate, 100, 0.707)
        filtered = apply_filter(audio, hp_coeffs)
        assert filtered.shape == audio.shape

        # Apply parametric EQ
        eq = ParametricEQ(sample_rate)
        eq.add_band("highpass", 80, 0, 0.707)
        eq.add_band("peak", 3000, 2.0, 1.5)
        eq_audio = eq.process(audio)
        assert eq_audio.shape == audio.shape

        # Apply compression
        comp = Compressor(
            sample_rate=sample_rate,
            threshold_db=-20,
            ratio=4.0,
            attack_ms=10,
            release_ms=100,
        )
        compressed = comp.process(audio)
        assert compressed.shape == audio.shape

    @pytest.mark.asyncio
    async def test_mixing_engine(self):
        """Test mixing engine."""
        from aether.audio.mixing import MixingEngine

        engine = MixingEngine(sample_rate=48000)

        # Create test stems
        duration = 1.0
        samples = int(48000 * duration)
        stems = {
            "kick": np.random.randn(samples) * 0.5,
            "bass": np.random.randn(samples) * 0.3,
            "synth": np.random.randn(samples) * 0.2,
        }

        # Create minimal mix spec
        mix_spec = {
            "tracks": [
                {"track_name": "kick", "gain_db": 0, "pan": 0, "output_bus": "drums"},
                {"track_name": "bass", "gain_db": -2, "pan": 0, "output_bus": "bass"},
                {"track_name": "synth", "gain_db": -4, "pan": 0.3, "output_bus": "music"},
            ],
            "buses": [
                {"bus_name": "drums", "gain_db": 0, "output_bus": "master"},
                {"bus_name": "bass", "gain_db": 0, "output_bus": "master"},
                {"bus_name": "music", "gain_db": -2, "output_bus": "master"},
            ],
        }

        mixed = engine.mix(stems, mix_spec)

        # Should return stereo mix
        assert mixed.ndim == 2
        assert mixed.shape[0] == 2  # Stereo


class TestQAIntegration:
    """Test QA system integration."""

    @pytest.mark.asyncio
    async def test_technical_validation(self):
        """Test technical audio validation."""
        from aether.qa.technical import TechnicalValidator

        validator = TechnicalValidator()

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

        result = validator.validate(audio, sample_rate)

        assert "loudness" in result
        assert "true_peak" in result
        assert "dynamic_range" in result
        assert "phase_correlation" in result

    @pytest.mark.asyncio
    async def test_originality_checker(self):
        """Test originality checking."""
        from aether.qa.originality import OriginalityChecker

        checker = OriginalityChecker()

        # Test melody fingerprinting
        melody_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
        fingerprint = checker._melody_to_intervals(melody_notes)

        assert len(fingerprint) == len(melody_notes) - 1
        assert all(isinstance(i, int) for i in fingerprint)

        # Test lyric checking
        lyrics = "This is a completely original song about unique experiences"
        result = checker.check_lyrics(lyrics)

        assert "ngram_matches" in result
        assert "similarity_score" in result


class TestStorageIntegration:
    """Test storage system integration."""

    @pytest.mark.asyncio
    async def test_artifact_store(self):
        """Test artifact storage."""
        from aether.storage import ArtifactStore, ArtifactType

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_path=Path(tmpdir))

            # Store artifact
            artifact_id = store.store(
                artifact_type=ArtifactType.SONG_SPEC,
                data={"id": "test", "title": "Test Song"},
                metadata={"genre": "pop"},
            )

            assert artifact_id is not None

            # Retrieve artifact
            retrieved = store.get(artifact_id)
            assert retrieved["id"] == "test"
            assert retrieved["title"] == "Test Song"

            # List artifacts
            artifacts = store.list(artifact_type=ArtifactType.SONG_SPEC)
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
