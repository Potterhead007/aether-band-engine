"""Phonetics and language processing."""

from aether.voice.phonetics.english import (
    EnglishPhonetics,
    EnglishProsody,
    ENGLISH_PHONEME_FEATURES,
)
from aether.voice.phonetics.spanish import (
    SpanishPhonetics,
    SpanishProsody,
    SPANISH_PHONEME_FEATURES,
)
from aether.voice.phonetics.prosody import (
    ProsodyProcessor,
    SyllableTiming,
    EmotionProfile,
)
from aether.voice.phonetics.bilingual import (
    BilingualController,
    LanguageTransition,
    BilingualQualityChecker,
)

# Aliases for backwards compatibility
ENGLISH_PHONEMES = ENGLISH_PHONEME_FEATURES
SPANISH_PHONEMES = SPANISH_PHONEME_FEATURES

__all__ = [
    "EnglishPhonetics",
    "EnglishProsody",
    "ENGLISH_PHONEME_FEATURES",
    "ENGLISH_PHONEMES",
    "SpanishPhonetics",
    "SpanishProsody",
    "SPANISH_PHONEME_FEATURES",
    "SPANISH_PHONEMES",
    "ProsodyProcessor",
    "SyllableTiming",
    "EmotionProfile",
    "BilingualController",
    "LanguageTransition",
    "BilingualQualityChecker",
]
