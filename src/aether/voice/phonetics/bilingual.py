"""
Bilingual Controller

Handles smooth transitions between English and Spanish
within songs, managing code-switching and accent consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from aether.voice.phonetics.english import EnglishPhonetics, EnglishProsody
from aether.voice.phonetics.spanish import SpanishPhonetics, SpanishProsody


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"


class BoundaryType(Enum):
    """Types of language boundaries."""
    PHRASE_BREAK = "phrase_break"  # Full stop, paragraph
    WORD_BOUNDARY = "word_boundary"  # Mid-phrase switch
    LOAN_WORD = "loan_word"  # Single foreign word


@dataclass
class TransitionParams:
    """Parameters for language transition."""
    blend_frames: int  # Audio frames for blending
    reset_prosody: bool  # Whether to reset prosody model
    maintain_timbre: bool  # Keep voice timbre consistent
    interpolate_formants: bool = False  # Smooth formant transition
    use_source_phonemes: bool = False  # Keep source language pronunciation
    adapt_to_target_prosody: bool = True  # Apply target language prosody


@dataclass
class LanguageSpan:
    """A span of text in a single language."""
    text: str
    language: Language
    start_idx: int
    end_idx: int
    phonemes: Optional[List[str]] = None


class BilingualController:
    """
    Manages bilingual singing with code-switching support.

    Handles:
    - Language detection
    - Phoneme conversion per language
    - Smooth transitions between languages
    - Accent consistency within each language
    """

    def __init__(self):
        """Initialize bilingual controller."""
        self.english = EnglishPhonetics()
        self.spanish = SpanishPhonetics()
        self.english_prosody = EnglishProsody()
        self.spanish_prosody = SpanishProsody()

        # Common Spanish loan words in English contexts
        self.spanish_loan_words = {
            "amigo", "amor", "corazón", "vida", "loco", "loca",
            "fiesta", "siesta", "mañana", "gracias", "hola",
            "adios", "por favor", "nada", "todo", "bueno",
        }

        # Common English loan words in Spanish contexts
        self.english_loan_words = {
            "baby", "love", "party", "cool", "yeah", "okay",
            "sorry", "please", "thanks", "hello", "bye",
        }

    def detect_language_spans(
        self,
        text: str,
        default_language: Language = Language.ENGLISH
    ) -> List[LanguageSpan]:
        """
        Detect language spans in mixed-language text.

        Args:
            text: Input text potentially containing both languages
            default_language: Assumed language when ambiguous

        Returns:
            List of language spans
        """
        words = text.split()
        spans = []
        current_lang = default_language
        current_start = 0
        current_words = []

        for i, word in enumerate(words):
            word_lower = word.lower().strip(".,!?¿¡")

            detected_lang = self._detect_word_language(word_lower, current_lang)

            if detected_lang != current_lang and current_words:
                # Save current span
                span_text = " ".join(current_words)
                spans.append(LanguageSpan(
                    text=span_text,
                    language=current_lang,
                    start_idx=current_start,
                    end_idx=current_start + len(span_text),
                ))
                current_start = current_start + len(span_text) + 1
                current_words = []
                current_lang = detected_lang

            current_words.append(word)

        # Add final span
        if current_words:
            span_text = " ".join(current_words)
            spans.append(LanguageSpan(
                text=span_text,
                language=current_lang,
                start_idx=current_start,
                end_idx=current_start + len(span_text),
            ))

        return spans

    def _detect_word_language(
        self,
        word: str,
        context_lang: Language
    ) -> Language:
        """Detect language of a single word."""
        # Check for Spanish-specific characters
        if any(c in word for c in "ñáéíóúü¿¡"):
            return Language.SPANISH

        # Check loan word lists
        if context_lang == Language.ENGLISH:
            if word in self.spanish_loan_words:
                return Language.SPANISH
        else:
            if word in self.english_loan_words:
                return Language.ENGLISH

        # Check common Spanish patterns
        spanish_patterns = ["ción", "mente", "oso", "osa", "ito", "ita"]
        if any(word.endswith(p) for p in spanish_patterns):
            return Language.SPANISH

        # Default to context language
        return context_lang

    def process_text(
        self,
        text: str,
        default_language: Language = Language.ENGLISH
    ) -> List[LanguageSpan]:
        """
        Process text and convert to phonemes per language.

        Args:
            text: Input text
            default_language: Default language

        Returns:
            Language spans with phonemes
        """
        spans = self.detect_language_spans(text, default_language)

        for span in spans:
            if span.language == Language.ENGLISH:
                phoneme_lists = self.english.text_to_phonemes(span.text)
            else:
                phoneme_lists = self.spanish.text_to_phonemes(span.text)

            # Flatten to single list
            span.phonemes = [p for word_phonemes in phoneme_lists for p in word_phonemes]

        return spans

    def handle_language_switch(
        self,
        current_lang: Language,
        next_lang: Language,
        boundary_type: BoundaryType
    ) -> TransitionParams:
        """
        Get transition parameters for a language switch.

        Args:
            current_lang: Current language
            next_lang: Target language
            boundary_type: Type of boundary at switch

        Returns:
            Parameters for the transition
        """
        if boundary_type == BoundaryType.PHRASE_BREAK:
            # Full reset at phrase boundaries
            return TransitionParams(
                blend_frames=0,
                reset_prosody=True,
                maintain_timbre=True,
            )
        elif boundary_type == BoundaryType.WORD_BOUNDARY:
            # Smooth blend for mid-phrase switches
            return TransitionParams(
                blend_frames=3,
                reset_prosody=False,
                maintain_timbre=True,
                interpolate_formants=True,
            )
        elif boundary_type == BoundaryType.LOAN_WORD:
            # Keep source pronunciation but adapt prosody
            return TransitionParams(
                blend_frames=1,
                reset_prosody=False,
                maintain_timbre=True,
                use_source_phonemes=True,
                adapt_to_target_prosody=True,
            )
        else:
            # Default safe transition
            return TransitionParams(
                blend_frames=2,
                reset_prosody=True,
                maintain_timbre=True,
            )

    def get_stress_pattern(
        self,
        spans: List[LanguageSpan]
    ) -> List[int]:
        """
        Get combined stress pattern for all spans.

        Args:
            spans: Language spans with text

        Returns:
            Combined stress pattern
        """
        stress = []

        for span in spans:
            words = span.text.split()
            for word in words:
                if span.language == Language.ENGLISH:
                    word_stress = EnglishProsody.get_word_stress(word)
                else:
                    word_stress = SpanishProsody.get_word_stress(word)
                stress.extend(word_stress)

        return stress

    def validate_transitions(
        self,
        spans: List[LanguageSpan]
    ) -> List[Tuple[int, str]]:
        """
        Validate language transitions for potential issues.

        Returns list of (index, warning) tuples.
        """
        warnings = []

        for i in range(len(spans) - 1):
            current = spans[i]
            next_span = spans[i + 1]

            # Check for rapid switches (single word spans)
            if len(current.text.split()) == 1 and len(next_span.text.split()) == 1:
                if current.language != next_span.language:
                    warnings.append((
                        i,
                        f"Rapid language switch at '{current.text}' -> '{next_span.text}'"
                    ))

            # Check for phoneme bleeding potential
            if current.phonemes and next_span.phonemes:
                last_phoneme = current.phonemes[-1]
                first_phoneme = next_span.phonemes[0]

                # Similar phonemes across languages might blend oddly
                if self._phonemes_similar(last_phoneme, first_phoneme):
                    if current.language != next_span.language:
                        warnings.append((
                            i,
                            f"Potential phoneme bleeding: '{last_phoneme}' -> '{first_phoneme}'"
                        ))

        return warnings

    def _phonemes_similar(self, p1: str, p2: str) -> bool:
        """Check if two phonemes are acoustically similar."""
        # Same phoneme
        if p1 == p2:
            return True

        # Similar vowels
        similar_vowels = [
            {"a", "ɑ", "æ"},
            {"e", "ɛ", "eɪ"},
            {"i", "ɪ"},
            {"o", "ɔ", "oʊ"},
            {"u", "ʊ"},
        ]

        for group in similar_vowels:
            if p1 in group and p2 in group:
                return True

        return False


# Acceptance criteria checker
class BilingualQualityChecker:
    """
    Checks bilingual output against acceptance criteria.
    """

    # Thresholds from spec
    THRESHOLDS = {
        "word_recognition_rate": 0.95,
        "phoneme_accuracy": 0.98,
        "cross_language_confusion": 0.02,
        "stress_placement_accuracy": 0.97,
        "language_contamination_rate": 0.01,
        "accent_consistency": 0.95,
    }

    def check_intelligibility(
        self,
        recognized_text: str,
        expected_text: str
    ) -> Tuple[bool, float]:
        """
        Check word recognition rate.

        Returns (passed, score).
        """
        expected_words = expected_text.lower().split()
        recognized_words = recognized_text.lower().split()

        if not expected_words:
            return True, 1.0

        matches = sum(
            1 for e, r in zip(expected_words, recognized_words)
            if e == r
        )
        score = matches / len(expected_words)

        return score >= self.THRESHOLDS["word_recognition_rate"], score

    def check_language_purity(
        self,
        detected_languages: List[Language],
        expected_languages: List[Language]
    ) -> Tuple[bool, float]:
        """
        Check for cross-language confusion.

        Returns (passed, contamination_rate).
        """
        if not expected_languages:
            return True, 0.0

        mismatches = sum(
            1 for d, e in zip(detected_languages, expected_languages)
            if d != e
        )
        contamination = mismatches / len(expected_languages)

        return contamination <= self.THRESHOLDS["cross_language_confusion"], contamination
