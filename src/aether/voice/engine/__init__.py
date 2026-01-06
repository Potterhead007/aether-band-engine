"""Singing engine core components."""

from aether.voice.engine.synthesizer import (
    SingingEngine,
    SingingEngineInput,
    SingingEngineOutput,
)
from aether.voice.engine.aligner import LyricMelodyAligner, AlignedUnit
from aether.voice.engine.pitch import (
    PitchController,
    PitchContourGenerator,
    PitchContext,
)
from aether.voice.engine.vibrato import (
    VibratoGenerator,
    VibratoParams,
    GENRE_VIBRATO_PRESETS,
)
from aether.voice.engine.transitions import (
    TransitionEngine,
    TransitionType,
    TransitionSpec,
    GENRE_TRANSITION_STYLE,
)
from aether.voice.engine.breath import (
    BreathModel,
    BreathEvent,
    BreathType,
    BREATH_TYPES,
)

__all__ = [
    "SingingEngine",
    "SingingEngineInput",
    "SingingEngineOutput",
    "LyricMelodyAligner",
    "AlignedUnit",
    "PitchController",
    "PitchContourGenerator",
    "PitchContext",
    "VibratoGenerator",
    "VibratoParams",
    "GENRE_VIBRATO_PRESETS",
    "TransitionEngine",
    "TransitionType",
    "TransitionSpec",
    "GENRE_TRANSITION_STYLE",
    "BreathModel",
    "BreathEvent",
    "BreathType",
    "BREATH_TYPES",
]
