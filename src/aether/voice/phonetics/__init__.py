"""Phonetics and language processing."""

from aether.voice.phonetics.english import (
    EnglishPhonetics,
    EnglishProsody,
    ENGLISH_PHONEMES,
)
from aether.voice.phonetics.spanish import (
    SpanishPhonetics,
    SpanishProsody,
    SPANISH_PHONEMES,
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

__all__ = [
    "EnglishPhonetics",
    "EnglishProsody",
    "ENGLISH_PHONEMES",
    "SpanishPhonetics",
    "SpanishProsody",
    "SPANISH_PHONEMES",
    "ProsodyProcessor",
    "SyllableTiming",
    "EmotionProfile",
    "BilingualController",
    "LanguageTransition",
    "BilingualQualityChecker",
]
