"""
Prosody Processor

Unified prosody processing for singing across languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ProsodyLanguage(Enum):
    """Supported languages for prosody processing."""
    ENGLISH = "en"
    SPANISH = "es"


@dataclass
class ProsodyParams:
    """Parameters controlling prosody generation."""
    # Timing
    tempo: float = 120.0  # BPM
    swing_amount: float = 0.0  # 0.0-1.0

    # Stress
    stress_contrast: float = 0.5  # Difference between stressed/unstressed
    reduction_strength: float = 0.5  # How much to reduce unstressed vowels

    # Phrasing
    phrase_final_lengthening: float = 1.2  # Duration multiplier at phrase end
    breath_pause_ms: float = 200  # Pause duration for breath

    # Emotion
    emotion_intensity: float = 0.5  # 0.0-1.0


@dataclass
class SyllableTiming:
    """Timing information for a syllable."""
    start_ms: float
    duration_ms: float
    stress: float  # 0.0-1.0
    is_phrase_boundary: bool = False


class ProsodyProcessor:
    """
    Unified prosody processor for singing.

    Handles timing, stress, and phrasing for both
    English and Spanish.
    """

    def __init__(self, language: ProsodyLanguage = ProsodyLanguage.ENGLISH):
        """
        Initialize prosody processor.

        Args:
            language: Target language
        """
        self.language = language

        # Language-specific settings
        if language == ProsodyLanguage.ENGLISH:
            self.is_stress_timed = True
            self.allows_reduction = True
        else:  # Spanish
            self.is_stress_timed = False
            self.allows_reduction = False

    def compute_syllable_timing(
        self,
        syllables: List[List[str]],
        stress_pattern: List[int],
        note_durations: List[float],
        params: ProsodyParams,
    ) -> List[SyllableTiming]:
        """
        Compute timing for each syllable.

        Args:
            syllables: List of syllables (each is list of phonemes)
            stress_pattern: Stress value (0 or 1) per syllable
            note_durations: Duration in beats per note
            params: Prosody parameters

        Returns:
            Timing information for each syllable
        """
        if len(syllables) != len(note_durations):
            # Adjust if mismatch
            note_durations = self._adjust_durations(syllables, note_durations)

        timings = []
        current_time = 0.0

        beat_duration_ms = 60000 / params.tempo  # ms per beat

        for i, (syllable, note_dur) in enumerate(zip(syllables, note_durations)):
            duration_ms = note_dur * beat_duration_ms

            # Apply stress-based duration adjustment
            stress = stress_pattern[i] if i < len(stress_pattern) else 0
            if self.is_stress_timed:
                # Stressed syllables slightly longer in English
                if stress > 0:
                    duration_ms *= 1.1
                else:
                    duration_ms *= 0.9

            # Phrase-final lengthening
            is_phrase_end = i == len(syllables) - 1
            if is_phrase_end:
                duration_ms *= params.phrase_final_lengthening

            timings.append(SyllableTiming(
                start_ms=current_time,
                duration_ms=duration_ms,
                stress=float(stress),
                is_phrase_boundary=is_phrase_end,
            ))

            current_time += duration_ms

        return timings

    def _adjust_durations(
        self,
        syllables: List[List[str]],
        durations: List[float]
    ) -> List[float]:
        """Adjust duration list to match syllable count."""
        if len(durations) >= len(syllables):
            return durations[:len(syllables)]

        # Extend durations by repeating last value
        result = list(durations)
        while len(result) < len(syllables):
            result.append(result[-1] if result else 1.0)
        return result

    def apply_swing(
        self,
        timings: List[SyllableTiming],
        swing_amount: float
    ) -> List[SyllableTiming]:
        """
        Apply swing timing to syllables.

        Swing delays every other beat slightly.
        """
        if swing_amount <= 0:
            return timings

        result = []
        cumulative_shift = 0.0

        for i, timing in enumerate(timings):
            new_timing = SyllableTiming(
                start_ms=timing.start_ms + cumulative_shift,
                duration_ms=timing.duration_ms,
                stress=timing.stress,
                is_phrase_boundary=timing.is_phrase_boundary,
            )

            # Apply swing on off-beats
            if i % 2 == 1:
                shift = timing.duration_ms * swing_amount * 0.33
                new_timing.start_ms += shift
                cumulative_shift += shift

            result.append(new_timing)

        return result

    def compute_phoneme_durations(
        self,
        syllable: List[str],
        syllable_duration_ms: float,
        stress: float,
    ) -> List[Tuple[str, float]]:
        """
        Distribute duration across phonemes in a syllable.

        Args:
            syllable: List of phonemes
            syllable_duration_ms: Total syllable duration
            stress: Stress level (0.0-1.0)

        Returns:
            List of (phoneme, duration_ms) tuples
        """
        if not syllable:
            return []

        # Categorize phonemes
        vowels = []
        consonants = []
        for i, p in enumerate(syllable):
            if self._is_vowel(p):
                vowels.append((i, p))
            else:
                consonants.append((i, p))

        # Allocate durations
        # Vowels get most of the duration
        vowel_ratio = 0.7 if vowels else 0.0
        consonant_ratio = 1.0 - vowel_ratio

        vowel_duration = syllable_duration_ms * vowel_ratio / max(1, len(vowels))
        consonant_duration = syllable_duration_ms * consonant_ratio / max(1, len(consonants))

        # Build result in original order
        result = []
        for i, phoneme in enumerate(syllable):
            if self._is_vowel(phoneme):
                result.append((phoneme, vowel_duration))
            else:
                result.append((phoneme, consonant_duration))

        return result

    def _is_vowel(self, phoneme: str) -> bool:
        """Check if phoneme is a vowel."""
        vowels = set("aeiouɪʊɛɔæɑʌəaɪaʊɔɪ")
        return phoneme.lower() in vowels or any(v in phoneme for v in vowels)


@dataclass
class EmotionProfile:
    """Emotion profile for prosody modulation."""
    name: str
    pitch_variation: float  # Multiplier on pitch range
    tempo_variation: float  # Multiplier on tempo
    intensity: float  # Overall energy
    breathiness: float  # Added breathiness


EMOTION_PROFILES: Dict[str, EmotionProfile] = {
    "neutral": EmotionProfile("neutral", 1.0, 1.0, 0.5, 0.0),
    "happy": EmotionProfile("happy", 1.2, 1.1, 0.7, 0.0),
    "sad": EmotionProfile("sad", 0.8, 0.9, 0.4, 0.2),
    "angry": EmotionProfile("angry", 1.3, 1.05, 0.9, 0.0),
    "tender": EmotionProfile("tender", 0.9, 0.95, 0.4, 0.3),
    "excited": EmotionProfile("excited", 1.4, 1.15, 0.8, 0.0),
    "melancholic": EmotionProfile("melancholic", 0.85, 0.85, 0.35, 0.15),
    "passionate": EmotionProfile("passionate", 1.25, 1.0, 0.85, 0.1),
}


def get_emotion_profile(emotion: str) -> EmotionProfile:
    """Get emotion profile by name."""
    return EMOTION_PROFILES.get(emotion.lower(), EMOTION_PROFILES["neutral"])
