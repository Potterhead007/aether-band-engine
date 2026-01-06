"""
Lyric-Melody Aligner

Aligns lyric syllables to melody notes with singing-aware heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlignedUnit:
    """A unit of aligned lyrics and melody."""
    # From lyrics
    text: str
    phonemes: List[str]
    language: str = "en"
    stress: float = 0.5

    # From melody
    pitch: int  # MIDI note
    start_beat: float
    duration_beats: float
    velocity: int = 100

    # Alignment metadata
    is_melisma: bool = False  # Multiple notes per syllable
    extend_from: Optional[int] = None  # Index of syllable this extends
    is_phrase_start: bool = False
    is_phrase_end: bool = False


class LyricMelodyAligner:
    """
    Aligns lyric syllables to melody notes.

    Uses heuristics to handle:
    - One syllable per note (most common)
    - Melismas (multiple notes per syllable)
    - Phrase boundaries and breath points
    """

    def __init__(
        self,
        melisma_threshold: float = 0.3,
        min_syllable_duration: float = 0.125,  # beats
    ):
        """
        Initialize the aligner.

        Args:
            melisma_threshold: Likelihood threshold for melisma detection
            min_syllable_duration: Minimum duration for a syllable
        """
        self.melisma_threshold = melisma_threshold
        self.min_syllable_duration = min_syllable_duration

    def align(
        self,
        lyrics: List,  # LyricToken
        melody: List,  # MelodyNote
    ) -> List[AlignedUnit]:
        """
        Align lyrics to melody notes.

        Args:
            lyrics: List of lyric tokens
            melody: List of melody notes

        Returns:
            List of aligned units
        """
        if not lyrics or not melody:
            return []

        # Step 1: Syllabify lyrics
        syllables = self._syllabify(lyrics)

        # Step 2: Initial assignment
        assignments = self._initial_assignment(syllables, melody)

        # Step 3: Detect and handle melismas
        assignments = self._detect_melismas(assignments, melody)

        # Step 4: Mark phrase boundaries
        assignments = self._mark_boundaries(assignments)

        logger.debug(f"Aligned {len(syllables)} syllables to {len(melody)} notes")
        return assignments

    def _syllabify(self, lyrics: List) -> List[dict]:
        """
        Extract syllables from lyrics.

        Returns list of syllable dicts with text, phonemes, etc.
        """
        syllables = []

        for token in lyrics:
            text = token.text if hasattr(token, 'text') else str(token)
            phonemes = token.phonemes if hasattr(token, 'phonemes') else None
            language = token.language if hasattr(token, 'language') else "en"
            stress = token.stress if hasattr(token, 'stress') else None

            # Simple syllabification by vowel groups
            if phonemes:
                # Use provided phonemes
                token_syllables = self._split_phonemes_to_syllables(phonemes)
            else:
                # Estimate from text
                token_syllables = self._estimate_syllables(text)

            for i, syl in enumerate(token_syllables):
                syllables.append({
                    'text': syl['text'] if isinstance(syl, dict) else text,
                    'phonemes': syl['phonemes'] if isinstance(syl, dict) else [],
                    'language': language,
                    'stress': stress if stress is not None else (1.0 if i == 0 else 0.5),
                })

        return syllables

    def _split_phonemes_to_syllables(
        self,
        phonemes: List[str]
    ) -> List[dict]:
        """Split phonemes into syllables."""
        if not phonemes:
            return [{'text': '', 'phonemes': []}]

        vowels = set("aeiouɪʊɛɔæɑʌəaɪaʊɔɪ")
        syllables = []
        current = []
        has_vowel = False

        for phoneme in phonemes:
            is_vowel = any(v in phoneme.lower() for v in vowels)

            if is_vowel:
                if has_vowel and current:
                    # Start new syllable
                    syllables.append({
                        'text': ''.join(current),
                        'phonemes': current,
                    })
                    current = []
                has_vowel = True

            current.append(phoneme)

        if current:
            syllables.append({
                'text': ''.join(current),
                'phonemes': current,
            })

        return syllables if syllables else [{'text': '', 'phonemes': phonemes}]

    def _estimate_syllables(self, text: str) -> List[dict]:
        """Estimate syllables from text."""
        vowels = "aeiouyAEIOUY"
        syllables = []
        current = ""
        has_vowel = False

        for char in text:
            is_vowel = char in vowels

            if is_vowel:
                if has_vowel and len(current) > 1:
                    syllables.append({'text': current, 'phonemes': []})
                    current = ""
                has_vowel = True

            current += char

        if current:
            syllables.append({'text': current, 'phonemes': []})

        return syllables if syllables else [{'text': text, 'phonemes': []}]

    def _initial_assignment(
        self,
        syllables: List[dict],
        melody: List,  # MelodyNote
    ) -> List[AlignedUnit]:
        """
        Create initial one-to-one assignment.
        """
        assignments = []
        note_idx = 0

        for syl in syllables:
            if note_idx >= len(melody):
                # More syllables than notes - use last note
                note = melody[-1] if melody else None
            else:
                note = melody[note_idx]
                note_idx += 1

            if note is None:
                continue

            unit = AlignedUnit(
                text=syl['text'],
                phonemes=syl['phonemes'],
                language=syl.get('language', 'en'),
                stress=syl.get('stress', 0.5),
                pitch=note.pitch,
                start_beat=note.start_beat,
                duration_beats=note.duration_beats,
                velocity=note.velocity,
            )
            assignments.append(unit)

        # Handle extra notes (melismas or instrumental)
        while note_idx < len(melody):
            note = melody[note_idx]
            # These will be handled in melisma detection
            note_idx += 1

        return assignments

    def _detect_melismas(
        self,
        assignments: List[AlignedUnit],
        melody: List,
    ) -> List[AlignedUnit]:
        """
        Detect and handle melismas (multiple notes per syllable).
        """
        if len(melody) <= len(assignments):
            return assignments

        # Extra notes need to be distributed
        extra_notes = len(melody) - len(assignments)

        # Find vowel-heavy syllables that could sustain
        sustainable = []
        for i, unit in enumerate(assignments):
            vowel_count = sum(1 for p in unit.phonemes if self._is_vowel(p))
            if vowel_count > 0:
                sustainable.append((i, vowel_count))

        # Sort by vowel count descending
        sustainable.sort(key=lambda x: x[1], reverse=True)

        # Extend syllables with extra notes
        note_offset = len(assignments)
        for i in range(min(extra_notes, len(sustainable))):
            syl_idx = sustainable[i][0]
            if note_offset < len(melody):
                note = melody[note_offset]
                # Create melisma extension
                extended = AlignedUnit(
                    text="~",  # Continuation marker
                    phonemes=[],
                    language=assignments[syl_idx].language,
                    stress=0.3,
                    pitch=note.pitch,
                    start_beat=note.start_beat,
                    duration_beats=note.duration_beats,
                    velocity=note.velocity,
                    is_melisma=True,
                    extend_from=syl_idx,
                )
                assignments.append(extended)
                note_offset += 1

        # Sort by start time
        assignments.sort(key=lambda u: u.start_beat)
        return assignments

    def _is_vowel(self, phoneme: str) -> bool:
        """Check if phoneme is a vowel."""
        vowels = set("aeiouɪʊɛɔæɑʌəaɪaʊɔɪ")
        return any(v in phoneme.lower() for v in vowels)

    def _mark_boundaries(
        self,
        assignments: List[AlignedUnit]
    ) -> List[AlignedUnit]:
        """Mark phrase boundaries."""
        if not assignments:
            return assignments

        # First unit is phrase start
        assignments[0].is_phrase_start = True

        # Last unit is phrase end
        assignments[-1].is_phrase_end = True

        # Look for gaps indicating phrase breaks
        for i in range(1, len(assignments)):
            prev = assignments[i - 1]
            curr = assignments[i]

            gap = curr.start_beat - (prev.start_beat + prev.duration_beats)

            if gap > 0.5:  # Half beat gap suggests phrase break
                prev.is_phrase_end = True
                curr.is_phrase_start = True

        return assignments

    def get_phoneme_durations(
        self,
        unit: AlignedUnit,
        tempo: float,
    ) -> List[Tuple[str, float]]:
        """
        Get duration for each phoneme in a unit.

        Args:
            unit: Aligned unit
            tempo: BPM

        Returns:
            List of (phoneme, duration_ms) tuples
        """
        if not unit.phonemes:
            return []

        beat_duration_ms = 60000 / tempo
        total_ms = unit.duration_beats * beat_duration_ms

        # Distribute duration based on phoneme type
        durations = []
        vowel_count = sum(1 for p in unit.phonemes if self._is_vowel(p))
        consonant_count = len(unit.phonemes) - vowel_count

        if vowel_count == 0:
            # All consonants - equal distribution
            per_phoneme = total_ms / len(unit.phonemes)
            return [(p, per_phoneme) for p in unit.phonemes]

        # Vowels get 70% of duration
        vowel_duration = (total_ms * 0.7) / vowel_count
        consonant_duration = (total_ms * 0.3) / max(1, consonant_count)

        for phoneme in unit.phonemes:
            if self._is_vowel(phoneme):
                durations.append((phoneme, vowel_duration))
            else:
                durations.append((phoneme, consonant_duration))

        return durations
