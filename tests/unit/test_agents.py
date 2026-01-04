"""
Unit tests for AETHER agents.
"""

import pytest
import uuid
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

from aether.agents.base import BaseAgent, AgentRegistry
from aether.agents.creative_director import CreativeDirectorAgent
from aether.agents.composition import CompositionAgent
from aether.agents.arrangement import ArrangementAgent
from aether.agents.sound_design import SoundDesignAgent
from aether.agents.lyrics import LyricsAgent
from aether.agents.vocal import VocalAgent
from aether.agents.mixing import MixingAgent
from aether.agents.mastering import MasteringAgent
from aether.schemas.base import SectionType


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_register_agent(self):
        """Test agent registration."""
        # Registry should have agents registered via decorators
        assert len(AgentRegistry._agents) > 0

    def test_get_existing_agent(self):
        """Test getting existing agent."""
        # Should be able to get creative_director agent
        cls = AgentRegistry.get("creative_director")
        assert cls is not None
        assert cls.agent_type == "creative_director"

    def test_get_nonexistent_agent(self):
        """Test getting non-existent agent."""
        result = AgentRegistry._agents.get("nonexistent")
        assert result is None


class TestBaseAgent:
    """Tests for BaseAgent."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        class MockInput(BaseModel):
            data: str

        class MockOutput(BaseModel):
            result: str

        class MockAgent(BaseAgent[MockInput, MockOutput]):
            agent_type = "mock"
            agent_name = "Mock Agent"
            input_schema = MockInput
            output_schema = MockOutput

            async def process(self, input_data, context):
                return MockOutput(result=f"processed: {input_data.data}")

        return MockAgent()

    @pytest.mark.asyncio
    async def test_agent_process(self, mock_agent):
        """Test agent process method."""
        input_data = mock_agent.input_schema(data="test")
        result = await mock_agent.process(input_data, context={})
        assert result.result == "processed: test"

    def test_agent_log_decision(self, mock_agent):
        """Test logging decisions."""
        mock_agent.log_decision(
            decision_type="test",
            input_summary="input",
            output_summary="output",
            reasoning="because",
            confidence=0.9,
        )
        assert len(mock_agent.decisions) == 1
        assert mock_agent.decisions[0].decision_type == "test"
        assert mock_agent.decisions[0].confidence == 0.9


class TestCreativeDirectorAgent:
    """Tests for CreativeDirectorAgent."""

    @pytest.fixture
    def agent(self):
        return CreativeDirectorAgent()

    @pytest.fixture
    def input_data(self, agent):
        return agent.input_schema(
            prompt="upbeat pop song about summer",
            genre_id="pop",
        )

    @pytest.mark.asyncio
    async def test_generate_song_spec(self, agent, input_data):
        """Test song spec generation."""
        result = await agent.process(input_data, context={})

        assert hasattr(result, "song_spec")
        song = result.song_spec
        assert "id" in song
        assert "title" in song
        assert "primary_mood" in song

    @pytest.mark.asyncio
    async def test_mood_detection(self, agent):
        """Test mood detection from prompt."""
        # Energetic prompt
        input_data = agent.input_schema(
            prompt="happy upbeat party dance track",
            genre_id="pop",
        )
        result = await agent.process(input_data, context={})
        mood = result.song_spec["primary_mood"]
        assert mood in ["energetic", "happy", "uplifting"]


class TestCompositionAgent:
    """Tests for CompositionAgent."""

    @pytest.fixture
    def agent(self):
        return CompositionAgent()

    @pytest.fixture
    def input_data(self, agent):
        return agent.input_schema(
            song_spec={
                "id": str(uuid.uuid4()),
                "title": "Test Song",
                "primary_mood": "energetic",
                "tempo_bpm": 120,
                "key_signature": "C",
                "time_signature": "4/4",
                "target_duration_seconds": 180,
            },
            genre_profile_id="boom-bap",
        )

    @pytest.mark.asyncio
    async def test_generate_composition(self, agent, input_data):
        """Test composition generation."""
        with patch("aether.agents.composition.get_genre_manager") as mock_manager:
            mock_profile = MagicMock()
            mock_profile.rhythm = MagicMock(
                tempo_range=(80, 140),
                time_signatures=["4/4"],
            )
            mock_profile.harmony = MagicMock(
                common_keys=["C", "G"],
                common_modes=["major"],
            )
            mock_manager.return_value.get.return_value = mock_profile

            result = await agent.process(input_data, context={})

            assert hasattr(result, "rhythm_spec")
            assert hasattr(result, "harmony_spec")
            assert hasattr(result, "melody_spec")


class TestArrangementAgent:
    """Tests for ArrangementAgent."""

    @pytest.fixture
    def agent(self):
        return ArrangementAgent()

    @pytest.fixture
    def input_data(self, agent):
        return agent.input_schema(
            song_spec={
                "id": str(uuid.uuid4()),
                "title": "Test",
                "primary_mood": "energetic",
                "target_duration_seconds": 180,
            },
            rhythm_spec={"tempo_bpm": 120},
            harmony_spec={"key": "C", "mode": "major"},
            genre_profile_id="boom-bap",
        )

    @pytest.mark.asyncio
    async def test_generate_arrangement(self, agent, input_data):
        """Test arrangement generation."""
        with patch("aether.agents.arrangement.get_genre_manager") as mock_manager:
            mock_profile = MagicMock()
            mock_profile.arrangement = MagicMock(
                typical_structures=[{
                    "sections": [
                        {"type": "intro", "length_bars": 4},
                        {"type": "verse", "length_bars": 16},
                        {"type": "chorus", "length_bars": 8},
                    ]
                }]
            )
            mock_profile.instrumentation = MagicMock(
                core_instruments=["drums", "bass"],
                optional_instruments=["synth"],
            )
            mock_manager.return_value.get.return_value = mock_profile

            result = await agent.process(input_data, context={})

            assert hasattr(result, "arrangement_spec")
            arr = result.arrangement_spec
            assert "sections" in arr
            assert "instruments" in arr


class TestLyricsAgent:
    """Tests for LyricsAgent."""

    @pytest.fixture
    def agent(self):
        return LyricsAgent()

    @pytest.fixture
    def input_data(self, agent):
        return agent.input_schema(
            song_spec={
                "id": str(uuid.uuid4()),
                "creative_brief": {"theme": "love"},
                "primary_mood": "happy",
            },
            arrangement_spec={
                "sections": [
                    {"section_type": "verse", "label": "Verse 1", "length_bars": 8},
                    {"section_type": "chorus", "label": "Chorus", "length_bars": 8},
                ]
            },
            melody_spec={"id": "melody-1"},
        )

    @pytest.mark.asyncio
    async def test_generate_lyrics(self, agent, input_data):
        """Test lyrics generation."""
        result = await agent.process(input_data, context={})

        assert hasattr(result, "lyric_spec")
        lyrics = result.lyric_spec
        assert "sections" in lyrics
        assert len(lyrics["sections"]) == 2  # verse and chorus

    def test_syllable_counting(self, agent):
        """Test syllable counting heuristic."""
        test_cases = [
            ("hello", 2),
            ("I", 1),
            ("love", 1),
        ]
        for word, expected in test_cases:
            result = agent._count_syllables(word)
            assert result >= 1  # At least 1 syllable


class TestVocalAgent:
    """Tests for VocalAgent."""

    @pytest.fixture
    def agent(self):
        return VocalAgent()

    @pytest.fixture
    def input_data(self, agent):
        return agent.input_schema(
            song_spec={
                "id": str(uuid.uuid4()),
                "primary_mood": "energetic",
            },
            lyric_spec={
                "id": "lyrics-1",
                "sections": [
                    {"section_type": "verse"},
                    {"section_type": "chorus"},
                ]
            },
            melody_spec={"id": "melody-1"},
            genre_profile_id="pop",
        )

    @pytest.mark.asyncio
    async def test_generate_vocal_spec(self, agent, input_data):
        """Test vocal spec generation."""
        result = await agent.process(input_data, context={})

        assert hasattr(result, "vocal_spec")
        vocal = result.vocal_spec
        assert "voice_persona" in vocal
        assert "doubles" in vocal
        assert "harmonies" in vocal

    def test_delivery_style_mapping(self, agent):
        """Test delivery style determination."""
        assert agent._determine_delivery_style("energetic") == "belted"
        assert agent._determine_delivery_style("calm") == "sung"
        assert agent._determine_delivery_style("ethereal") == "whispered"
        assert agent._determine_delivery_style("unknown") == "sung"


class TestMixingAgent:
    """Tests for MixingAgent."""

    @pytest.fixture
    def agent(self):
        return MixingAgent()

    @pytest.fixture
    def input_data(self, agent):
        return agent.input_schema(
            song_spec={"id": str(uuid.uuid4())},
            arrangement_spec={
                "sections": [{"section_type": "verse", "length_bars": 8}],
                "instruments": [
                    {"name": "kick", "category": "drums"},
                    {"name": "bass", "category": "bass"},
                    {"name": "lead_synth", "category": "synth"},
                ]
            },
            sound_design_spec={"id": "sound-1"},
            genre_profile_id="boom-bap",
        )

    @pytest.mark.asyncio
    async def test_generate_mix_spec(self, agent, input_data):
        """Test mix spec generation."""
        with patch("aether.agents.mixing.get_genre_manager") as mock_manager:
            mock_manager.return_value.get.return_value = MagicMock()

            result = await agent.process(input_data, context={})

            assert hasattr(result, "mix_spec")
            mix = result.mix_spec
            assert "buses" in mix
            assert "tracks" in mix
            assert len(mix["tracks"]) == 3

    def test_bus_creation(self, agent):
        """Test bus structure creation."""
        mock_profile = MagicMock()
        buses = agent._create_bus_structure(mock_profile)

        bus_names = [b.bus_name for b in buses]
        assert "drums" in bus_names
        assert "bass" in bus_names
        assert "music" in bus_names
        assert "vocals" in bus_names


class TestMasteringAgent:
    """Tests for MasteringAgent."""

    @pytest.fixture
    def agent(self):
        return MasteringAgent()

    @pytest.fixture
    def input_data(self, agent):
        return agent.input_schema(
            song_spec={
                "id": str(uuid.uuid4()),
                "primary_mood": "energetic",
            },
            mix_spec={"id": "mix-1"},
            genre_profile_id="boom-bap",
        )

    @pytest.mark.asyncio
    async def test_generate_master_spec(self, agent, input_data):
        """Test master spec generation."""
        with patch("aether.agents.mastering.get_genre_manager") as mock_manager:
            mock_profile = MagicMock()
            mock_profile.lineage = MagicMock(primary_parent="hip-hop")
            mock_profile.production = MagicMock(target_lufs=-14.0)
            mock_profile.id = "boom-bap"
            mock_manager.return_value.get.return_value = mock_profile

            result = await agent.process(input_data, context={})

            assert hasattr(result, "master_spec")
            master = result.master_spec
            assert "loudness" in master
            assert "multiband_compression" in master
            assert "limiter" in master

    def test_loudness_target(self, agent):
        """Test loudness target determination."""
        mock_profile = MagicMock()
        mock_profile.production = MagicMock(target_lufs=-14.0)
        target = agent._determine_loudness_target(mock_profile)
        assert target.target_lufs == -14.0

    def test_stereo_enhancement(self, agent):
        """Test stereo enhancement determination."""
        assert agent._determine_stereo_enhancement("ethereal") == 0.2
        assert agent._determine_stereo_enhancement("aggressive") == 0.1
        assert agent._determine_stereo_enhancement("other") == 0.05


class TestSoundDesignAgent:
    """Tests for SoundDesignAgent."""

    @pytest.fixture
    def agent(self):
        return SoundDesignAgent()

    @pytest.fixture
    def input_data(self, agent):
        return agent.input_schema(
            song_spec={
                "id": str(uuid.uuid4()),
                "primary_mood": "dark",
            },
            arrangement_spec={
                "id": str(uuid.uuid4()),
                "instruments": [
                    {"name": "kick", "category": "drums", "role": "foundation"},
                    {"name": "bass", "category": "bass", "role": "foundation"},
                ]
            },
            rhythm_spec={"id": str(uuid.uuid4())},
            genre_profile_id="synthwave",
        )

    @pytest.mark.asyncio
    async def test_generate_sound_design(self, agent, input_data):
        """Test sound design generation."""
        with patch("aether.agents.sound_design.get_genre_manager") as mock_manager:
            mock_profile = MagicMock()
            mock_profile.production = MagicMock(
                mix_characteristics={
                    "drums": {"punch": 0.8},
                    "bass": {"warmth": 0.7},
                }
            )
            mock_manager.return_value.get.return_value = mock_profile

            result = await agent.process(input_data, context={})

            assert hasattr(result, "sound_design_spec")
            sd = result.sound_design_spec
            assert "instrument_patches" in sd


class TestAgentIntegration:
    """Integration tests for agent pipeline."""

    @pytest.mark.asyncio
    async def test_creative_director_output(self):
        """Test creative director output format."""
        agent = CreativeDirectorAgent()
        input_data = agent.input_schema(
            prompt="test song",
            genre_id="pop",
        )

        result = await agent.process(input_data, context={})

        assert hasattr(result, "song_spec")
        song_spec = result.song_spec
        assert "id" in song_spec
        assert "title" in song_spec
