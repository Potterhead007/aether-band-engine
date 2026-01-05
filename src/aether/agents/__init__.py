"""
AETHER Agents

The 10-agent pipeline for music generation.
"""

from aether.agents.arrangement import (
    ArrangementAgent,
    ArrangementInput,
    ArrangementOutput,
)
from aether.agents.base import AgentDecision, AgentRegistry, BaseAgent
from aether.agents.composition import (
    CompositionAgent,
    CompositionInput,
    CompositionOutput,
)

# Import all agents to register them
from aether.agents.creative_director import (
    CreativeDirectorAgent,
    CreativeDirectorInput,
    CreativeDirectorOutput,
)
from aether.agents.lyrics import (
    LyricsAgent,
    LyricsInput,
    LyricsOutput,
)
from aether.agents.mastering import (
    MasteringAgent,
    MasteringInput,
    MasteringOutput,
)
from aether.agents.mixing import (
    MixingAgent,
    MixingInput,
    MixingOutput,
)
from aether.agents.qa import (
    QAAgent,
    QAInput,
    QAOutput,
)
from aether.agents.release import (
    ReleaseAgent,
    ReleaseInput,
    ReleaseOutput,
)
from aether.agents.sound_design import (
    SoundDesignAgent,
    SoundDesignInput,
    SoundDesignOutput,
)
from aether.agents.vocal import (
    VocalAgent,
    VocalInput,
    VocalOutput,
)

__all__ = [
    # Base
    "BaseAgent",
    "AgentRegistry",
    "AgentDecision",
    # Creative Director
    "CreativeDirectorAgent",
    "CreativeDirectorInput",
    "CreativeDirectorOutput",
    # Composition
    "CompositionAgent",
    "CompositionInput",
    "CompositionOutput",
    # Arrangement
    "ArrangementAgent",
    "ArrangementInput",
    "ArrangementOutput",
    # Lyrics
    "LyricsAgent",
    "LyricsInput",
    "LyricsOutput",
    # Vocal
    "VocalAgent",
    "VocalInput",
    "VocalOutput",
    # Sound Design
    "SoundDesignAgent",
    "SoundDesignInput",
    "SoundDesignOutput",
    # Mixing
    "MixingAgent",
    "MixingInput",
    "MixingOutput",
    # Mastering
    "MasteringAgent",
    "MasteringInput",
    "MasteringOutput",
    # QA
    "QAAgent",
    "QAInput",
    "QAOutput",
    # Release
    "ReleaseAgent",
    "ReleaseInput",
    "ReleaseOutput",
]


def get_pipeline_agents() -> list:
    """Get all pipeline agents in execution order."""
    return [
        "creative_director",
        "composition",
        "arrangement",
        "lyrics",
        "vocal",
        "sound_design",
        "mixing",
        "mastering",
        "qa",
        "release",
    ]


def create_agent(agent_type: str) -> BaseAgent:
    """Create an agent by type."""
    return AgentRegistry.create(agent_type)
