"""Vocal arrangement system."""

from aether.voice.arrangement.layers import (
    VocalArrangementSystem,
    VocalLayer,
    VocalLayerType,
    ArrangementSection,
    GENRE_ARRANGEMENTS,
    DEFAULT_LAYERS,
    DynamicArrangementBuilder,
)
from aether.voice.arrangement.harmony import (
    HarmonyGenerator,
    ScaleHarmonizer,
    VoiceLeadingEngine,
    HarmonyVoice,
    HarmonyNote,
    HarmonyType,
)
from aether.voice.arrangement.safeguards import (
    ArrangementSafeguards,
    SafeguardResult,
    SafeguardIssue,
    SafeguardType,
    PhaseDetector,
    MaskingDetector,
    BuildupDetector,
)

__all__ = [
    "VocalArrangementSystem",
    "VocalLayer",
    "VocalLayerType",
    "ArrangementSection",
    "GENRE_ARRANGEMENTS",
    "DEFAULT_LAYERS",
    "DynamicArrangementBuilder",
    "HarmonyGenerator",
    "ScaleHarmonizer",
    "VoiceLeadingEngine",
    "HarmonyVoice",
    "HarmonyNote",
    "HarmonyType",
    "ArrangementSafeguards",
    "SafeguardResult",
    "SafeguardIssue",
    "SafeguardType",
    "PhaseDetector",
    "MaskingDetector",
    "BuildupDetector",
]
