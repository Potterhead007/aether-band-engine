"""
AETHER Voice Generation & Singing Module

Production-grade AI singing voice synthesis with:
- Consistent vocal identity (AVU-1)
- Bilingual support (English/Spanish)
- Genre-aware performance adaptation
- Multi-layer vocal arrangements
- Quality control and evaluation

Usage:
    from aether.voice import SingingEngine, VocalIdentity, synthesize_voice_sync

    # Quick synthesis
    result = synthesize_voice_sync(
        lyrics="Hello world",
        melody=[{"pitch": 60, "start_beat": 0, "duration_beats": 1}],
        tempo=120,
        genre="pop",
    )
    audio = result["audio"]

    # Full engine access
    engine = SingingEngine()
    output = await engine.synthesize(input_spec)
"""

from aether.voice.identity.blueprint import (
    VocalIdentity,
    VocalRange,
    TimbreCharacteristics,
    FormantProfile,
    VocalClassification,
    AVU1Identity,
)
from aether.voice.identity.invariants import (
    IdentityInvariants,
    ControlledFlexibility,
    AVU1_INVARIANTS,
)
from aether.voice.identity.drift_monitor import (
    VoiceConsistencyMonitor,
    IdentityDriftTracker,
)

from aether.voice.engine.synthesizer import (
    SingingEngine,
    SingingEngineInput,
    SingingEngineOutput,
)
from aether.voice.engine.aligner import LyricMelodyAligner, AlignedUnit
from aether.voice.engine.pitch import PitchController, PitchContourGenerator
from aether.voice.engine.vibrato import VibratoGenerator, VibratoParams
from aether.voice.engine.transitions import TransitionEngine, TransitionType
from aether.voice.engine.breath import BreathModel, BreathEvent

from aether.voice.performance.profiles import (
    VocalPerformanceProfile,
    GENRE_PROFILES,
    get_profile,
)
from aether.voice.performance.ornamentation import OrnamentationEngine, OrnamentSpec
from aether.voice.performance.expression import (
    ExpressionMapper,
    EmotionVector,
    EmotionCategory,
    ExpressionParameters,
)

from aether.voice.arrangement.layers import (
    VocalArrangementSystem,
    VocalLayer,
    VocalLayerType,
)
from aether.voice.arrangement.harmony import HarmonyGenerator, HarmonyVoice
from aether.voice.arrangement.safeguards import (
    ArrangementSafeguards,
    SafeguardResult,
)

from aether.voice.quality.metrics import (
    QualityMetricsCollector,
    MetricSummary,
    MetricCategory,
)
from aether.voice.quality.evaluator import (
    VocalQualityEvaluator,
    EvaluationResult,
    EvaluationGrade,
)
from aether.voice.quality.thresholds import (
    QualityThresholds,
    ReleaseStage,
    get_thresholds,
)

from aether.voice.integration.pipeline import (
    VoiceSynthesisPipeline,
    VoiceSynthesisInput,
    VoiceSynthesisOutput,
)
from aether.voice.integration.api import (
    synthesize_voice_sync,
    synthesize_voice_standalone,
)

__all__ = [
    # Identity
    "VocalIdentity",
    "VocalRange",
    "TimbreCharacteristics",
    "FormantProfile",
    "VocalClassification",
    "AVU1Identity",
    "IdentityInvariants",
    "ControlledFlexibility",
    "AVU1_INVARIANTS",
    "VoiceConsistencyMonitor",
    "IdentityDriftTracker",
    # Engine
    "SingingEngine",
    "SingingEngineInput",
    "SingingEngineOutput",
    "LyricMelodyAligner",
    "AlignedUnit",
    "PitchController",
    "PitchContourGenerator",
    "VibratoGenerator",
    "VibratoParams",
    "TransitionEngine",
    "TransitionType",
    "BreathModel",
    "BreathEvent",
    # Performance
    "VocalPerformanceProfile",
    "GENRE_PROFILES",
    "get_profile",
    "OrnamentationEngine",
    "OrnamentSpec",
    "ExpressionMapper",
    "EmotionVector",
    "EmotionCategory",
    "ExpressionParameters",
    # Arrangement
    "VocalArrangementSystem",
    "VocalLayer",
    "VocalLayerType",
    "HarmonyGenerator",
    "HarmonyVoice",
    "ArrangementSafeguards",
    "SafeguardResult",
    # Quality
    "QualityMetricsCollector",
    "MetricSummary",
    "MetricCategory",
    "VocalQualityEvaluator",
    "EvaluationResult",
    "EvaluationGrade",
    "QualityThresholds",
    "ReleaseStage",
    "get_thresholds",
    # Integration
    "VoiceSynthesisPipeline",
    "VoiceSynthesisInput",
    "VoiceSynthesisOutput",
    "synthesize_voice_sync",
    "synthesize_voice_standalone",
]

__version__ = "0.1.0"
