"""
English Phonetics Module

Defines the phoneme inventory, pronunciation rules, and prosody
system for General American English singing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import re


class EnglishVowel(Enum):
    """English vowel phonemes (IPA)."""
    # Monophthongs
    IY = "i"    # beat
    IH = "ɪ"    # bit
    EY = "e"    # bait (actually diphthong but treated as long vowel)
    EH = "ɛ"    # bet
    AE = "æ"    # bat
    AA = "ɑ"    # bot (father)
    AO = "ɔ"    # bought
    OW = "o"    # boat
    UH = "ʊ"    # book
    UW = "u"    # boot
    AH = "ʌ"    # but
    AX = "ə"    # about (schwa)

    # Diphthongs
    AY = "aɪ"   # bite
    AW = "aʊ"   # bout
    OY = "ɔɪ"   # boy


class EnglishConsonant(Enum):
    """English consonant phonemes (IPA)."""
    # Stops
    P = "p"
    B = "b"
    T = "t"
    D = "d"
    K = "k"
    G = "g"

    # Fricatives
    F = "f"
    V = "v"
    TH = "θ"    # thin
    DH = "ð"    # this
    S = "s"
    Z = "z"
    SH = "ʃ"    # ship
    ZH = "ʒ"    # measure
    H = "h"

    # Affricates
    CH = "tʃ"   # church
    JH = "dʒ"   # judge

    # Nasals
    M = "m"
    N = "n"
    NG = "ŋ"    # sing

    # Liquids
    L = "l"
    R = "r"

    # Glides
    W = "w"
    Y = "j"     # yes


@dataclass
class PhonemeFeatures:
    """Articulatory features for a phoneme."""
    is_vowel: bool
    is_voiced: bool
    manner: str  # stop, fricative, nasal, etc.
    place: str   # bilabial, alveolar, etc.
    duration_factor: float = 1.0  # Relative duration


# Phoneme feature database
ENGLISH_PHONEME_FEATURES: Dict[str, PhonemeFeatures] = {
    # Vowels (all voiced)
    "i": PhonemeFeatures(True, True, "vowel", "high_front", 1.0),
    "ɪ": PhonemeFeatures(True, True, "vowel", "high_front_lax", 0.8),
    "e": PhonemeFeatures(True, True, "vowel", "mid_front", 1.2),
    "ɛ": PhonemeFeatures(True, True, "vowel", "mid_front_lax", 0.9),
    "æ": PhonemeFeatures(True, True, "vowel", "low_front", 1.0),
    "ɑ": PhonemeFeatures(True, True, "vowel", "low_back", 1.1),
    "ɔ": PhonemeFeatures(True, True, "vowel", "mid_back_round", 1.0),
    "o": PhonemeFeatures(True, True, "vowel", "mid_back_round", 1.2),
    "ʊ": PhonemeFeatures(True, True, "vowel", "high_back_lax", 0.8),
    "u": PhonemeFeatures(True, True, "vowel", "high_back", 1.0),
    "ʌ": PhonemeFeatures(True, True, "vowel", "mid_central", 0.9),
    "ə": PhonemeFeatures(True, True, "vowel", "mid_central", 0.6),
    "aɪ": PhonemeFeatures(True, True, "diphthong", "low_to_high", 1.4),
    "aʊ": PhonemeFeatures(True, True, "diphthong", "low_to_high", 1.4),
    "ɔɪ": PhonemeFeatures(True, True, "diphthong", "mid_to_high", 1.3),

    # Consonants
    "p": PhonemeFeatures(False, False, "stop", "bilabial", 0.3),
    "b": PhonemeFeatures(False, True, "stop", "bilabial", 0.3),
    "t": PhonemeFeatures(False, False, "stop", "alveolar", 0.3),
    "d": PhonemeFeatures(False, True, "stop", "alveolar", 0.3),
    "k": PhonemeFeatures(False, False, "stop", "velar", 0.3),
    "g": PhonemeFeatures(False, True, "stop", "velar", 0.3),
    "f": PhonemeFeatures(False, False, "fricative", "labiodental", 0.5),
    "v": PhonemeFeatures(False, True, "fricative", "labiodental", 0.5),
    "θ": PhonemeFeatures(False, False, "fricative", "dental", 0.5),
    "ð": PhonemeFeatures(False, True, "fricative", "dental", 0.4),
    "s": PhonemeFeatures(False, False, "fricative", "alveolar", 0.5),
    "z": PhonemeFeatures(False, True, "fricative", "alveolar", 0.5),
    "ʃ": PhonemeFeatures(False, False, "fricative", "postalveolar", 0.5),
    "ʒ": PhonemeFeatures(False, True, "fricative", "postalveolar", 0.5),
    "h": PhonemeFeatures(False, False, "fricative", "glottal", 0.3),
    "tʃ": PhonemeFeatures(False, False, "affricate", "postalveolar", 0.5),
    "dʒ": PhonemeFeatures(False, True, "affricate", "postalveolar", 0.5),
    "m": PhonemeFeatures(False, True, "nasal", "bilabial", 0.6),
    "n": PhonemeFeatures(False, True, "nasal", "alveolar", 0.5),
    "ŋ": PhonemeFeatures(False, True, "nasal", "velar", 0.5),
    "l": PhonemeFeatures(False, True, "liquid", "alveolar", 0.5),
    "r": PhonemeFeatures(False, True, "liquid", "alveolar", 0.5),
    "w": PhonemeFeatures(False, True, "glide", "bilabial", 0.3),
    "j": PhonemeFeatures(False, True, "glide", "palatal", 0.3),
}


class EnglishPhonetics:
    """
    English phonetics processor for singing.

    Handles text-to-phoneme conversion, syllabification,
    and singing-specific phoneme modifications.
    """

    # Common word pronunciations (simplified)
    LEXICON: Dict[str, List[str]] = {
        "the": ["ð", "ə"],
        "a": ["ə"],
        "an": ["æ", "n"],
        "i": ["aɪ"],
        "you": ["j", "u"],
        "love": ["l", "ʌ", "v"],
        "heart": ["h", "ɑ", "r", "t"],
        "baby": ["b", "e", "b", "i"],
        "night": ["n", "aɪ", "t"],
        "day": ["d", "e"],
        "time": ["t", "aɪ", "m"],
        "way": ["w", "e"],
        "feel": ["f", "i", "l"],
        "know": ["n", "o"],
        "want": ["w", "ɑ", "n", "t"],
        "need": ["n", "i", "d"],
        "come": ["k", "ʌ", "m"],
        "go": ["g", "o"],
        "say": ["s", "e"],
        "make": ["m", "e", "k"],
        "take": ["t", "e", "k"],
        "see": ["s", "i"],
        "yeah": ["j", "æ"],
        "oh": ["o"],
        "ooh": ["u"],
        "ah": ["ɑ"],
    }

    # Letter-to-phoneme rules (simplified)
    GRAPHEME_RULES: Dict[str, List[str]] = {
        "a": ["æ"],
        "e": ["ɛ"],
        "i": ["ɪ"],
        "o": ["ɑ"],
        "u": ["ʌ"],
        "ee": ["i"],
        "oo": ["u"],
        "ea": ["i"],
        "ai": ["e"],
        "ay": ["e"],
        "oa": ["o"],
        "ow": ["o"],
        "ou": ["aʊ"],
        "oi": ["ɔɪ"],
        "oy": ["ɔɪ"],
        "sh": ["ʃ"],
        "ch": ["tʃ"],
        "th": ["θ"],
        "ng": ["ŋ"],
        "ck": ["k"],
        "ph": ["f"],
        "wh": ["w"],
    }

    def text_to_phonemes(self, text: str) -> List[List[str]]:
        """
        Convert text to phoneme sequences.

        Args:
            text: Input text

        Returns:
            List of phoneme lists, one per word
        """
        words = self._tokenize(text)
        result = []

        for word in words:
            word_lower = word.lower()
            if word_lower in self.LEXICON:
                result.append(self.LEXICON[word_lower])
            else:
                result.append(self._grapheme_to_phoneme(word_lower))

        return result

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Remove punctuation but keep apostrophes in contractions
        text = re.sub(r"[^\w\s']", " ", text)
        return text.split()

    def _grapheme_to_phoneme(self, word: str) -> List[str]:
        """Convert a word to phonemes using rules."""
        phonemes = []
        i = 0

        while i < len(word):
            # Try two-character rules first
            if i + 1 < len(word):
                digraph = word[i:i+2]
                if digraph in self.GRAPHEME_RULES:
                    phonemes.extend(self.GRAPHEME_RULES[digraph])
                    i += 2
                    continue

            # Single character rules
            char = word[i]
            if char in self.GRAPHEME_RULES:
                phonemes.extend(self.GRAPHEME_RULES[char])
            elif char.isalpha():
                # Default consonant handling
                phonemes.append(char)
            i += 1

        return phonemes

    def syllabify(self, phonemes: List[str]) -> List[List[str]]:
        """
        Split phonemes into syllables.

        Uses the maximum onset principle.
        """
        if not phonemes:
            return []

        syllables = []
        current_syllable = []

        # Find vowel positions
        vowel_indices = [
            i for i, p in enumerate(phonemes)
            if p in ENGLISH_PHONEME_FEATURES and ENGLISH_PHONEME_FEATURES[p].is_vowel
        ]

        if not vowel_indices:
            return [phonemes]

        for i, phoneme in enumerate(phonemes):
            current_syllable.append(phoneme)

            # Check if this is a vowel and we have more phonemes
            is_vowel = (
                phoneme in ENGLISH_PHONEME_FEATURES and
                ENGLISH_PHONEME_FEATURES[phoneme].is_vowel
            )

            if is_vowel and i < len(phonemes) - 1:
                # Look ahead to determine syllable boundary
                next_vowel_idx = None
                for vi in vowel_indices:
                    if vi > i:
                        next_vowel_idx = vi
                        break

                if next_vowel_idx is not None:
                    # Consonants between this vowel and next
                    consonants_between = next_vowel_idx - i - 1

                    if consonants_between == 0:
                        # No consonants - split here
                        syllables.append(current_syllable)
                        current_syllable = []
                    elif consonants_between == 1:
                        # One consonant - goes with next syllable
                        syllables.append(current_syllable)
                        current_syllable = []
                    # else: keep consonants until later decision

        if current_syllable:
            syllables.append(current_syllable)

        return syllables

    def get_phoneme_duration(self, phoneme: str, tempo: float = 120) -> float:
        """
        Get estimated duration for a phoneme in milliseconds.

        Args:
            phoneme: The phoneme
            tempo: BPM for context

        Returns:
            Duration in milliseconds
        """
        base_duration = 80  # Base phoneme duration at 120 BPM

        if phoneme in ENGLISH_PHONEME_FEATURES:
            factor = ENGLISH_PHONEME_FEATURES[phoneme].duration_factor
        else:
            factor = 1.0

        # Adjust for tempo
        tempo_factor = 120 / tempo

        return base_duration * factor * tempo_factor


@dataclass
class EnglishProsody:
    """English prosody rules for singing."""

    # Stress-timed language
    stress_timing: bool = True

    # Unstressed vowels reduce to schwa
    reduction_enabled: bool = True

    # Linking rules
    consonant_to_vowel_linking: bool = True
    r_linking: bool = True
    glottal_insertion: bool = False  # Avoid for singing

    @staticmethod
    def get_word_stress(word: str) -> List[int]:
        """
        Get stress pattern for a word.

        Returns list where 1 = stressed, 0 = unstressed.
        """
        # Simplified stress rules
        syllable_count = EnglishProsody._estimate_syllables(word)

        if syllable_count == 1:
            return [1]
        elif syllable_count == 2:
            # Most 2-syllable words stress first syllable
            return [1, 0]
        else:
            # Alternate stress pattern
            return [1 if i % 2 == 0 else 0 for i in range(syllable_count)]

    @staticmethod
    def _estimate_syllables(word: str) -> int:
        """Estimate syllable count from spelling."""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e") and count > 1:
            count -= 1

        return max(1, count)

    @staticmethod
    def apply_reduction(phonemes: List[str], stress: List[int]) -> List[str]:
        """
        Apply vowel reduction to unstressed syllables.
        """
        result = []
        vowel_idx = 0

        for phoneme in phonemes:
            if phoneme in ENGLISH_PHONEME_FEATURES:
                if ENGLISH_PHONEME_FEATURES[phoneme].is_vowel:
                    if vowel_idx < len(stress) and stress[vowel_idx] == 0:
                        # Reduce to schwa (except already short vowels)
                        if phoneme not in ["ə", "ɪ", "ʊ"]:
                            result.append("ə")
                        else:
                            result.append(phoneme)
                    else:
                        result.append(phoneme)
                    vowel_idx += 1
                else:
                    result.append(phoneme)
            else:
                result.append(phoneme)

        return result
