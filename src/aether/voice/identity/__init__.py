"""Voice identity management."""

from aether.voice.identity.blueprint import (
    VocalIdentity,
    VocalRange,
    TimbreCharacteristics,
    FormantProfile,
    VocalClassification,
    EmotionalBaseline,
    AVU1Identity,
    AVU2Identity,
    AVU3Identity,
    AVU4Identity,
    VOICE_REGISTRY,
    get_voice,
    list_voices,
)
from aether.voice.identity.invariants import (
    IdentityInvariants,
    ControlledFlexibility,
    InvariantSpec,
    AVU1_INVARIANTS,
)
from aether.voice.identity.drift_monitor import (
    VoiceConsistencyMonitor,
    IdentityDriftTracker,
    DriftReport,
)

__all__ = [
    "VocalIdentity",
    "VocalRange",
    "TimbreCharacteristics",
    "FormantProfile",
    "VocalClassification",
    "EmotionalBaseline",
    "AVU1Identity",
    "AVU2Identity",
    "AVU3Identity",
    "AVU4Identity",
    "VOICE_REGISTRY",
    "get_voice",
    "list_voices",
    "IdentityInvariants",
    "ControlledFlexibility",
    "InvariantSpec",
    "AVU1_INVARIANTS",
    "VoiceConsistencyMonitor",
    "IdentityDriftTracker",
    "DriftReport",
]
