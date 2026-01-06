"""Genre-aware vocal performance."""

from aether.voice.performance.profiles import (
    VocalPerformanceProfile,
    GENRE_PROFILES,
    get_profile,
    interpolate_profiles,
    GenreStyle,
    TimingProfile,
    ArticulationProfile,
    DynamicsProfile,
    RegisterProfile,
    OrnamentationProfile,
)
from aether.voice.performance.ornamentation import (
    OrnamentationEngine,
    OrnamentType,
    OrnamentSpec,
)
from aether.voice.performance.expression import (
    ExpressionMapper,
    EmotionVector,
    EmotionCategory,
    ExpressionParameters,
    PhraseDynamicsPlanner,
    ExpressionContourGenerator,
)

__all__ = [
    "VocalPerformanceProfile",
    "GENRE_PROFILES",
    "get_profile",
    "interpolate_profiles",
    "GenreStyle",
    "TimingProfile",
    "ArticulationProfile",
    "DynamicsProfile",
    "RegisterProfile",
    "OrnamentationProfile",
    "OrnamentationEngine",
    "OrnamentType",
    "OrnamentSpec",
    "ExpressionMapper",
    "EmotionVector",
    "EmotionCategory",
    "ExpressionParameters",
    "PhraseDynamicsPlanner",
    "ExpressionContourGenerator",
]
