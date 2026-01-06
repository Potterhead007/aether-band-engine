"""
Spanish Phonetics Module

Defines the phoneme inventory, pronunciation rules, and prosody
system for Neutral Latin American Spanish singing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import re


class SpanishVowel(Enum):
    """Spanish vowel phonemes (5 pure vowels)."""
    A = "a"   # casa
    E = "e"   # peso
    I = "i"   # piso
    O = "o"   # poco
    U = "u"   # luna


class SpanishConsonant(Enum):
    """Spanish consonant phonemes."""
    # Stops
    P = "p"
    B = "b"
    T = "t"
    D = "d"
    K = "k"
    G = "g"

    # Fricatives
    F = "f"
    S = "s"
    X = "x"    # jota (velar fricative)
    BETA = "β"  # allophone of /b/
    DELTA = "ð" # allophone of /d/
    GAMMA = "ɣ" # allophone of /g/

    # Affricate
    CH = "tʃ"

    # Nasals
    M = "m"
    N = "n"
    NY = "ɲ"   # ñ

    # Liquids
    L = "l"
    R = "r"    # trill
    RR = "ɾ"   # tap

    # Glides
    W = "w"
    Y = "j"


# Spanish phoneme features
SPANISH_PHONEME_FEATURES: Dict[str, Dict] = {
    # Vowels
    "a": {"is_vowel": True, "is_voiced": True, "height": "low", "backness": "central"},
    "e": {"is_vowel": True, "is_voiced": True, "height": "mid", "backness": "front"},
    "i": {"is_vowel": True, "is_voiced": True, "height": "high", "backness": "front"},
    "o": {"is_vowel": True, "is_voiced": True, "height": "mid", "backness": "back"},
    "u": {"is_vowel": True, "is_voiced": True, "height": "high", "backness": "back"},

    # Consonants
    "p": {"is_vowel": False, "is_voiced": False, "manner": "stop"},
    "b": {"is_vowel": False, "is_voiced": True, "manner": "stop"},
    "t": {"is_vowel": False, "is_voiced": False, "manner": "stop"},
    "d": {"is_vowel": False, "is_voiced": True, "manner": "stop"},
    "k": {"is_vowel": False, "is_voiced": False, "manner": "stop"},
    "g": {"is_vowel": False, "is_voiced": True, "manner": "stop"},
    "f": {"is_vowel": False, "is_voiced": False, "manner": "fricative"},
    "s": {"is_vowel": False, "is_voiced": False, "manner": "fricative"},
    "x": {"is_vowel": False, "is_voiced": False, "manner": "fricative"},
    "β": {"is_vowel": False, "is_voiced": True, "manner": "fricative"},
    "ð": {"is_vowel": False, "is_voiced": True, "manner": "fricative"},
    "ɣ": {"is_vowel": False, "is_voiced": True, "manner": "fricative"},
    "tʃ": {"is_vowel": False, "is_voiced": False, "manner": "affricate"},
    "m": {"is_vowel": False, "is_voiced": True, "manner": "nasal"},
    "n": {"is_vowel": False, "is_voiced": True, "manner": "nasal"},
    "ɲ": {"is_vowel": False, "is_voiced": True, "manner": "nasal"},
    "l": {"is_vowel": False, "is_voiced": True, "manner": "liquid"},
    "r": {"is_vowel": False, "is_voiced": True, "manner": "trill"},
    "ɾ": {"is_vowel": False, "is_voiced": True, "manner": "tap"},
    "w": {"is_vowel": False, "is_voiced": True, "manner": "glide"},
    "j": {"is_vowel": False, "is_voiced": True, "manner": "glide"},
}


class SpanishPhonetics:
    """
    Spanish phonetics processor for singing.

    Handles text-to-phoneme conversion with neutral Latin American
    pronunciation (seseo, yeísmo).
    """

    # Letter to phoneme mapping
    GRAPHEME_MAP: Dict[str, str] = {
        "a": "a",
        "e": "e",
        "i": "i",
        "o": "o",
        "u": "u",
        "b": "b",
        "c": "k",  # Default, modified by following vowel
        "d": "d",
        "f": "f",
        "g": "g",  # Default, modified by following vowel
        "h": "",   # Silent
        "j": "x",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "ñ": "ɲ",
        "p": "p",
        "q": "k",
        "r": "ɾ",  # Default tap, modified for trill
        "s": "s",
        "t": "t",
        "v": "b",  # Same as b in Spanish
        "w": "w",
        "x": "ks",
        "y": "j",  # Yeísmo
        "z": "s",  # Seseo
    }

    # Common word pronunciations
    LEXICON: Dict[str, List[str]] = {
        "el": ["e", "l"],
        "la": ["l", "a"],
        "de": ["d", "e"],
        "que": ["k", "e"],
        "y": ["i"],
        "en": ["e", "n"],
        "un": ["u", "n"],
        "es": ["e", "s"],
        "se": ["s", "e"],
        "no": ["n", "o"],
        "te": ["t", "e"],
        "me": ["m", "e"],
        "mi": ["m", "i"],
        "tu": ["t", "u"],
        "yo": ["j", "o"],
        "amor": ["a", "m", "o", "ɾ"],
        "corazón": ["k", "o", "ɾ", "a", "s", "o", "n"],
        "vida": ["b", "i", "ð", "a"],
        "noche": ["n", "o", "tʃ", "e"],
        "día": ["d", "i", "a"],
        "cielo": ["s", "j", "e", "l", "o"],
        "fuego": ["f", "w", "e", "ɣ", "o"],
        "tiempo": ["t", "j", "e", "m", "p", "o"],
    }

    def text_to_phonemes(self, text: str) -> List[List[str]]:
        """
        Convert Spanish text to phoneme sequences.

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
        text = re.sub(r"[^\w\sáéíóúüñ¿¡]", " ", text, flags=re.IGNORECASE)
        return text.split()

    def _grapheme_to_phoneme(self, word: str) -> List[str]:
        """Convert a Spanish word to phonemes."""
        phonemes = []
        i = 0

        while i < len(word):
            char = word[i]
            next_char = word[i + 1] if i + 1 < len(word) else ""

            # Handle digraphs and special cases
            if char == "c":
                if next_char in "ei":
                    phonemes.append("s")  # Seseo
                elif char + next_char == "ch":
                    phonemes.append("tʃ")
                    i += 1
                else:
                    phonemes.append("k")
            elif char == "g":
                if next_char in "ei":
                    phonemes.append("x")
                elif char + next_char == "gu":
                    if i + 2 < len(word) and word[i + 2] in "ei":
                        phonemes.append("g")
                        i += 1  # Skip u
                    else:
                        phonemes.append("g")
                        phonemes.append("w")
                        i += 1
                else:
                    phonemes.append("g")
            elif char == "r":
                # Trill at start of word or after n, l, s
                if i == 0 or (i > 0 and word[i-1] in "nls"):
                    phonemes.append("r")
                elif char + next_char == "rr":
                    phonemes.append("r")
                    i += 1
                else:
                    phonemes.append("ɾ")
            elif char == "l" and next_char == "l":
                phonemes.append("j")  # Yeísmo
                i += 1
            elif char == "q" and next_char == "u":
                phonemes.append("k")
                i += 1  # Skip u
            elif char in self.GRAPHEME_MAP:
                phoneme = self.GRAPHEME_MAP[char]
                if phoneme:
                    if isinstance(phoneme, str) and len(phoneme) > 1:
                        phonemes.extend(list(phoneme))
                    else:
                        phonemes.append(phoneme)
            elif char in "áéíóú":
                # Accented vowels
                vowel_map = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
                phonemes.append(vowel_map[char])
            elif char == "ü":
                phonemes.append("u")

            i += 1

        # Apply allophonic rules
        phonemes = self._apply_allophones(phonemes)

        return phonemes

    def _apply_allophones(self, phonemes: List[str]) -> List[str]:
        """Apply Spanish allophonic rules."""
        result = []

        for i, phoneme in enumerate(phonemes):
            prev = phonemes[i - 1] if i > 0 else None

            # Spirantization: /b/, /d/, /g/ become fricatives after vowels
            if phoneme == "b" and prev and self._is_vowel(prev):
                result.append("β")
            elif phoneme == "d" and prev and self._is_vowel(prev):
                result.append("ð")
            elif phoneme == "g" and prev and self._is_vowel(prev):
                result.append("ɣ")
            else:
                result.append(phoneme)

        return result

    def _is_vowel(self, phoneme: str) -> bool:
        """Check if a phoneme is a vowel."""
        return phoneme in "aeiou"

    def syllabify(self, phonemes: List[str]) -> List[List[str]]:
        """
        Split phonemes into syllables.

        Spanish has clear syllable boundaries.
        """
        if not phonemes:
            return []

        syllables = []
        current = []

        i = 0
        while i < len(phonemes):
            phoneme = phonemes[i]
            current.append(phoneme)

            if self._is_vowel(phoneme):
                # Look ahead for consonants
                j = i + 1
                consonants = []
                while j < len(phonemes) and not self._is_vowel(phonemes[j]):
                    consonants.append(phonemes[j])
                    j += 1

                if j < len(phonemes):  # More vowels coming
                    # Split consonants between syllables
                    if len(consonants) == 0:
                        syllables.append(current)
                        current = []
                    elif len(consonants) == 1:
                        syllables.append(current)
                        current = []
                    elif len(consonants) >= 2:
                        # First consonant stays, rest go to next syllable
                        current.append(consonants[0])
                        syllables.append(current)
                        current = consonants[1:]
                        i = j - 1
                else:
                    # End of word
                    current.extend(consonants)
                    i = j - 1

            i += 1

        if current:
            syllables.append(current)

        return syllables


@dataclass
class SpanishProsody:
    """Spanish prosody rules for singing."""

    # Syllable-timed language (not stress-timed)
    stress_timing: bool = False

    # No vowel reduction
    reduction_enabled: bool = False

    # Synalepha: merge vowels across word boundaries
    synalepha: bool = True

    @staticmethod
    def get_word_stress(word: str) -> List[int]:
        """
        Get stress pattern for a Spanish word.

        Spanish stress rules:
        1. Words ending in vowel, n, or s: stress penultimate
        2. Words ending in other consonants: stress ultimate
        3. Accent marks override these rules
        """
        # Check for accent mark
        accent_pos = None
        for i, char in enumerate(word):
            if char in "áéíóú":
                accent_pos = i
                break

        syllable_count = SpanishProsody._count_syllables(word)

        if syllable_count == 1:
            return [1]

        stress = [0] * syllable_count

        if accent_pos is not None:
            # Find which syllable has the accent
            syllable_idx = SpanishProsody._char_to_syllable(word, accent_pos)
            stress[syllable_idx] = 1
        else:
            # Apply default rules
            last_char = word[-1].lower()
            if last_char in "aeioun s":
                # Stress penultimate
                stress[-2] = 1
            else:
                # Stress ultimate
                stress[-1] = 1

        return stress

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a Spanish word."""
        word = word.lower()
        vowels = "aeiouáéíóúü"
        count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            # In Spanish, each vowel (or diphthong) is a syllable
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        return max(1, count)

    @staticmethod
    def _char_to_syllable(word: str, char_idx: int) -> int:
        """Map character index to syllable index."""
        vowels = "aeiouáéíóúü"
        syllable = 0
        prev_was_vowel = False

        for i, char in enumerate(word):
            if i == char_idx:
                return syllable
            is_vowel = char.lower() in vowels
            if is_vowel and not prev_was_vowel:
                syllable += 1
            prev_was_vowel = is_vowel

        return syllable

    @staticmethod
    def apply_synalepha(words: List[List[str]]) -> List[List[str]]:
        """
        Apply synalepha: merge vowels across word boundaries.

        In Spanish singing, final vowel of one word often merges
        with initial vowel of the next word.
        """
        if len(words) < 2:
            return words

        result = [words[0]]

        for word in words[1:]:
            if not result[-1] or not word:
                result.append(word)
                continue

            last_phoneme = result[-1][-1]
            first_phoneme = word[0]

            # Check if both are vowels
            if last_phoneme in "aeiou" and first_phoneme in "aeiou":
                # Merge: remove last vowel, keep first
                result[-1] = result[-1][:-1]
                if not result[-1]:
                    result[-1] = word
                else:
                    result.append(word)
            else:
                result.append(word)

        return result
