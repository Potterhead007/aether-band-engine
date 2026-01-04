"""
Pytest configuration and shared fixtures for AETHER tests.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Async Support
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Temporary Directories
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_output_dir(temp_dir: Path) -> Path:
    """Create output directory in temp."""
    output = temp_dir / "output"
    output.mkdir(parents=True, exist_ok=True)
    return output


# =============================================================================
# Audio Fixtures
# =============================================================================

@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for tests."""
    return 48000


@pytest.fixture
def mono_audio(sample_rate: int) -> np.ndarray:
    """Generate mono test audio (1 second 440Hz sine wave)."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


@pytest.fixture
def stereo_audio(sample_rate: int) -> np.ndarray:
    """Generate stereo test audio."""
    duration = 1.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)

    left = np.sin(2 * np.pi * 440 * t) * 0.5
    right = np.sin(2 * np.pi * 440 * t + np.pi / 4) * 0.5

    return np.stack([left, right]).astype(np.float32)


@pytest.fixture
def long_stereo_audio(sample_rate: int) -> np.ndarray:
    """Generate longer stereo audio for mastering tests (10 seconds)."""
    duration = 10.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)

    # Mix of frequencies
    signal = (
        np.sin(2 * np.pi * 100 * t) * 0.3 +  # Bass
        np.sin(2 * np.pi * 440 * t) * 0.4 +  # Mid
        np.sin(2 * np.pi * 2000 * t) * 0.2   # High
    )

    # Add some dynamics
    envelope = np.concatenate([
        np.linspace(0, 1, samples // 10),
        np.ones(samples * 8 // 10),
        np.linspace(1, 0, samples // 10),
    ])[:samples]

    signal = signal * envelope

    return np.stack([signal, signal]).astype(np.float32) * 0.5


# =============================================================================
# Spec Fixtures
# =============================================================================

@pytest.fixture
def song_spec() -> dict:
    """Sample song specification."""
    return {
        "id": "test-song-001",
        "title": "Test Song",
        "primary_mood": "energetic",
        "tempo_bpm": 120,
        "key_signature": "C",
        "time_signature": "4/4",
        "target_duration_seconds": 180,
        "creative_brief": {
            "theme": "celebration",
            "style_references": ["upbeat pop"],
        },
    }


@pytest.fixture
def rhythm_spec() -> dict:
    """Sample rhythm specification."""
    return {
        "id": "rhythm-001",
        "tempo_bpm": 120,
        "time_signature": "4/4",
        "swing_amount": 0.0,
        "groove_template": "straight",
    }


@pytest.fixture
def harmony_spec() -> dict:
    """Sample harmony specification."""
    return {
        "id": "harmony-001",
        "key": "C",
        "mode": "major",
        "chord_progression": ["C", "Am", "F", "G"],
    }


@pytest.fixture
def melody_spec() -> dict:
    """Sample melody specification."""
    return {
        "id": "melody-001",
        "phrases": [
            {
                "notes": [60, 62, 64, 65, 67],
                "durations": [0.5, 0.5, 0.5, 0.5, 1.0],
            }
        ],
    }


@pytest.fixture
def arrangement_spec() -> dict:
    """Sample arrangement specification."""
    return {
        "id": "arrangement-001",
        "sections": [
            {
                "section_type": "intro",
                "label": "Intro",
                "start_bar": 1,
                "length_bars": 4,
            },
            {
                "section_type": "verse",
                "label": "Verse 1",
                "start_bar": 5,
                "length_bars": 16,
            },
            {
                "section_type": "chorus",
                "label": "Chorus",
                "start_bar": 21,
                "length_bars": 8,
            },
        ],
        "instruments": [
            {"name": "kick", "category": "drums", "role": "foundation"},
            {"name": "snare", "category": "drums", "role": "foundation"},
            {"name": "bass", "category": "bass", "role": "foundation"},
            {"name": "lead_synth", "category": "synth", "role": "melody"},
            {"name": "pad", "category": "synth", "role": "harmonic"},
        ],
    }


@pytest.fixture
def mix_spec() -> dict:
    """Sample mix specification."""
    return {
        "id": "mix-001",
        "buses": [
            {"bus_name": "drums", "gain_db": 0, "output_bus": "master"},
            {"bus_name": "bass", "gain_db": 0, "output_bus": "master"},
            {"bus_name": "music", "gain_db": -2, "output_bus": "master"},
        ],
        "tracks": [
            {
                "track_name": "kick",
                "gain_db": 0,
                "pan": 0,
                "output_bus": "drums",
                "eq_bands": [],
            },
            {
                "track_name": "bass",
                "gain_db": -2,
                "pan": 0,
                "output_bus": "bass",
                "eq_bands": [],
            },
        ],
        "master_eq": [],
        "target_headroom_db": -6.0,
    }


@pytest.fixture
def master_spec() -> dict:
    """Sample mastering specification."""
    return {
        "id": "master-001",
        "loudness": {
            "target_lufs": -14.0,
            "tolerance": 0.5,
        },
        "true_peak": {
            "ceiling_dbtp": -1.0,
        },
        "dynamic_range": {
            "minimum_lu": 6.0,
            "target_lu": 8.0,
        },
        "multiband_compression": [],
        "limiter": {
            "ceiling_dbtp": -1.0,
            "release_ms": 100,
            "lookahead_ms": 5,
        },
        "formats": ["wav_24_48", "mp3_320"],
    }


# =============================================================================
# Mock Providers
# =============================================================================

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider."""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value="Generated text content")
    mock.generate_structured = AsyncMock(return_value={"result": "structured"})
    mock.is_available = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_midi_provider():
    """Mock MIDI provider."""
    mock = MagicMock()
    mock.generate_from_spec = AsyncMock()
    mock.create_midi_file = MagicMock()
    return mock


@pytest.fixture
def mock_audio_provider(sample_rate: int):
    """Mock audio provider."""
    mock = MagicMock()
    mock.sample_rate = sample_rate
    mock.render_midi = AsyncMock(
        return_value=np.random.randn(2, sample_rate * 30).astype(np.float32) * 0.5
    )
    mock.synthesize = AsyncMock(
        return_value=np.random.randn(sample_rate).astype(np.float32) * 0.5
    )
    return mock


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider."""
    mock = MagicMock()
    mock.embed_text = AsyncMock(return_value=np.random.randn(384).astype(np.float32))
    mock.embed_audio = AsyncMock(return_value=np.random.randn(512).astype(np.float32))
    mock.similarity = MagicMock(return_value=0.5)
    return mock


# =============================================================================
# Provider Registry Mock
# =============================================================================

@pytest.fixture
def mock_provider_registry(
    mock_llm_provider,
    mock_midi_provider,
    mock_audio_provider,
    mock_embedding_provider,
):
    """Mock provider registry with all providers."""
    registry = MagicMock()
    registry.get.side_effect = lambda name: {
        "llm": mock_llm_provider,
        "midi": mock_midi_provider,
        "audio": mock_audio_provider,
        "embedding": mock_embedding_provider,
    }.get(name)
    return registry


# =============================================================================
# Genre Profile Mock
# =============================================================================

@pytest.fixture
def mock_genre_profile():
    """Mock genre profile."""
    return MagicMock(
        id="test-genre",
        name="Test Genre",
        lineage=MagicMock(
            primary_parent="electronic",
            secondary_influences=["pop"],
        ),
        rhythm=MagicMock(
            tempo_range=(100, 140),
            time_signatures=["4/4"],
            swing_range=(0, 0.1),
        ),
        harmony=MagicMock(
            common_keys=["C", "Am", "G"],
            common_modes=["major", "minor"],
            typical_progressions=[["I", "V", "vi", "IV"]],
        ),
        arrangement=MagicMock(
            typical_structures=[{
                "name": "standard",
                "sections": [
                    {"type": "intro", "length_bars": 4},
                    {"type": "verse", "length_bars": 16},
                    {"type": "chorus", "length_bars": 8},
                ]
            }],
        ),
        instrumentation=MagicMock(
            core_instruments=["drums", "bass", "synth"],
            optional_instruments=["piano", "guitar"],
        ),
        production=MagicMock(
            target_lufs=-14.0,
            mix_characteristics={
                "drums": {"punch": 0.7},
                "bass": {"warmth": 0.6},
            },
        ),
    )


# =============================================================================
# Configuration
# =============================================================================

@pytest.fixture
def test_config(temp_dir: Path):
    """Test configuration."""
    from aether.config import AetherConfig, PathsConfig

    return AetherConfig(
        paths=PathsConfig(
            base_dir=temp_dir,
            output_dir=temp_dir / "output",
            cache_dir=temp_dir / "cache",
        ),
        debug=True,
        verbose=True,
        log_level="DEBUG",
    )


# =============================================================================
# Test Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_soundfont: marks tests requiring soundfont"
    )
