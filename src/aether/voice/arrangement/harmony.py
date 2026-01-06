"""
Harmony Generation

Generates harmony vocal lines with proper voice leading.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class HarmonyType(Enum):
    """Types of harmony intervals."""
    UNISON = 0
    MINOR_SECOND = 1
    MAJOR_SECOND = 2
    MINOR_THIRD = 3
    MAJOR_THIRD = 4
    PERFECT_FOURTH = 5
    TRITONE = 6
    PERFECT_FIFTH = 7
    MINOR_SIXTH = 8
    MAJOR_SIXTH = 9
    MINOR_SEVENTH = 10
    MAJOR_SEVENTH = 11
    OCTAVE = 12


@dataclass
class HarmonyNote:
    """A single harmony note."""
    pitch: int  # MIDI note
    start_beat: float
    duration_beats: float
    velocity: int = 100
    interval_from_melody: int = 0  # Semitones


@dataclass
class HarmonyVoice:
    """A complete harmony voice line."""
    name: str
    notes: List[HarmonyNote]
    default_interval: int  # Default interval from melody
    volume_db: float = -6.0
    pan: float = 0.0


class ScaleHarmonizer:
    """
    Generates harmonies based on scale degrees.
    """

    # Common scales
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
    HARMONIC_MINOR = [0, 2, 3, 5, 7, 8, 11]
    DORIAN_MODE = [0, 2, 3, 5, 7, 9, 10]
    MIXOLYDIAN_MODE = [0, 2, 4, 5, 7, 9, 10]

    # Scale degree to chord quality mapping (for major scale)
    MAJOR_CHORD_QUALITIES = {
        0: "major",    # I
        1: "minor",    # ii
        2: "minor",    # iii
        3: "major",    # IV
        4: "major",    # V
        5: "minor",    # vi
        6: "dim",      # vii°
    }

    def __init__(
        self,
        root: int = 0,  # C
        scale: List[int] = None,
    ):
        """
        Initialize harmonizer.

        Args:
            root: Root note (0-11, C=0)
            scale: Scale intervals (defaults to major)
        """
        self.root = root % 12
        self.scale = scale or self.MAJOR_SCALE

    def get_scale_degree(self, pitch: int) -> Optional[int]:
        """
        Get scale degree of a pitch.

        Args:
            pitch: MIDI pitch

        Returns:
            Scale degree (0-6) or None if not in scale
        """
        pc = (pitch - self.root) % 12
        if pc in self.scale:
            return self.scale.index(pc)
        return None

    def get_diatonic_harmony(
        self,
        melody_pitch: int,
        interval_type: str = "third",
        direction: str = "above",
    ) -> int:
        """
        Get diatonic harmony note.

        Args:
            melody_pitch: Melody MIDI pitch
            interval_type: "third", "fifth", or "sixth"
            direction: "above" or "below"

        Returns:
            Harmony MIDI pitch
        """
        degree = self.get_scale_degree(melody_pitch)

        if degree is None:
            # Chromatic note - use closest scale tone
            pc = (melody_pitch - self.root) % 12
            closest = min(self.scale, key=lambda s: abs(s - pc))
            degree = self.scale.index(closest)

        # Calculate interval in scale degrees
        interval_degrees = {
            "third": 2,
            "fifth": 4,
            "sixth": 5,
        }
        offset = interval_degrees.get(interval_type, 2)

        if direction == "below":
            offset = -offset

        # Get harmony scale degree
        harmony_degree = (degree + offset) % 7
        octave_offset = (degree + offset) // 7

        if direction == "below" and offset < 0:
            octave_offset = -1 if harmony_degree > degree else 0

        # Calculate harmony pitch
        harmony_pc = self.scale[harmony_degree]
        melody_octave = melody_pitch // 12
        harmony_pitch = (melody_octave + octave_offset) * 12 + self.root + harmony_pc

        return harmony_pitch

    def harmonize_melody(
        self,
        melody: List[dict],  # [{"pitch": int, "start": float, "duration": float}]
        harmony_intervals: List[str] = None,
    ) -> List[HarmonyVoice]:
        """
        Generate harmony voices for a melody.

        Args:
            melody: Melody notes
            harmony_intervals: Intervals to generate ("third_above", "third_below", etc.)

        Returns:
            List of HarmonyVoice objects
        """
        if harmony_intervals is None:
            harmony_intervals = ["third_above", "third_below"]

        voices = []

        for interval_spec in harmony_intervals:
            parts = interval_spec.split("_")
            interval_type = parts[0] if parts else "third"
            direction = parts[1] if len(parts) > 1 else "above"

            notes = []
            for note in melody:
                harmony_pitch = self.get_diatonic_harmony(
                    note["pitch"],
                    interval_type,
                    direction,
                )

                notes.append(HarmonyNote(
                    pitch=harmony_pitch,
                    start_beat=note.get("start", note.get("start_beat", 0)),
                    duration_beats=note.get("duration", note.get("duration_beats", 1)),
                    velocity=note.get("velocity", 100),
                    interval_from_melody=harmony_pitch - note["pitch"],
                ))

            voices.append(HarmonyVoice(
                name=interval_spec,
                notes=notes,
                default_interval=notes[0].interval_from_melody if notes else 0,
            ))

        return voices


class VoiceLeadingEngine:
    """
    Applies voice leading rules to harmony lines.

    Ensures smooth, musical harmony movement.
    """

    # Voice leading rules
    MAX_LEAP = 7  # Maximum jump in semitones (perfect 5th)
    AVOID_PARALLEL_FIFTHS = True
    AVOID_PARALLEL_OCTAVES = True
    PREFER_CONTRARY_MOTION = True

    def __init__(self):
        """Initialize voice leading engine."""
        pass

    def apply_voice_leading(
        self,
        melody: List[HarmonyNote],
        harmony: List[HarmonyNote],
    ) -> List[HarmonyNote]:
        """
        Apply voice leading rules to harmony.

        Args:
            melody: Melody notes
            harmony: Raw harmony notes

        Returns:
            Voice-led harmony notes
        """
        if not harmony:
            return harmony

        result = [harmony[0]]  # Keep first note

        for i in range(1, len(harmony)):
            prev_harm = result[-1]
            curr_harm = harmony[i]
            curr_melody = melody[i] if i < len(melody) else melody[-1]
            prev_melody = melody[i - 1] if i > 0 and i - 1 < len(melody) else melody[0]

            # Check for large leap
            leap = abs(curr_harm.pitch - prev_harm.pitch)

            if leap > self.MAX_LEAP:
                # Try octave adjustment
                adjusted = self._minimize_leap(prev_harm.pitch, curr_harm.pitch)
                curr_harm = HarmonyNote(
                    pitch=adjusted,
                    start_beat=curr_harm.start_beat,
                    duration_beats=curr_harm.duration_beats,
                    velocity=curr_harm.velocity,
                    interval_from_melody=adjusted - curr_melody.pitch,
                )

            # Check for parallel fifths/octaves
            if self.AVOID_PARALLEL_FIFTHS or self.AVOID_PARALLEL_OCTAVES:
                curr_harm = self._avoid_parallels(
                    prev_melody, curr_melody, prev_harm, curr_harm
                )

            result.append(curr_harm)

        return result

    def _minimize_leap(self, prev_pitch: int, target_pitch: int) -> int:
        """Find closest octave of target pitch to prev pitch."""
        target_pc = target_pitch % 12
        prev_octave = prev_pitch // 12

        candidates = [
            prev_octave * 12 + target_pc,
            (prev_octave - 1) * 12 + target_pc,
            (prev_octave + 1) * 12 + target_pc,
        ]

        return min(candidates, key=lambda p: abs(p - prev_pitch))

    def _avoid_parallels(
        self,
        prev_melody: HarmonyNote,
        curr_melody: HarmonyNote,
        prev_harm: HarmonyNote,
        curr_harm: HarmonyNote,
    ) -> HarmonyNote:
        """Adjust harmony to avoid parallel fifths/octaves."""
        prev_interval = (prev_harm.pitch - prev_melody.pitch) % 12
        curr_interval = (curr_harm.pitch - curr_melody.pitch) % 12

        # Check for parallel fifth (7 semitones)
        if self.AVOID_PARALLEL_FIFTHS:
            if prev_interval == 7 and curr_interval == 7:
                # Move to third
                new_pitch = curr_melody.pitch + 4
                return HarmonyNote(
                    pitch=new_pitch,
                    start_beat=curr_harm.start_beat,
                    duration_beats=curr_harm.duration_beats,
                    velocity=curr_harm.velocity,
                    interval_from_melody=4,
                )

        # Check for parallel octave (0 or 12 semitones)
        if self.AVOID_PARALLEL_OCTAVES:
            if prev_interval in [0, 12] and curr_interval in [0, 12]:
                # Move to third
                new_pitch = curr_melody.pitch + 3
                return HarmonyNote(
                    pitch=new_pitch,
                    start_beat=curr_harm.start_beat,
                    duration_beats=curr_harm.duration_beats,
                    velocity=curr_harm.velocity,
                    interval_from_melody=3,
                )

        return curr_harm

    def smooth_voice(
        self,
        notes: List[HarmonyNote],
        max_leap: int = 5,
    ) -> List[HarmonyNote]:
        """
        Smooth a voice line to reduce large leaps.

        Args:
            notes: Voice line notes
            max_leap: Maximum allowed leap

        Returns:
            Smoothed voice line
        """
        if len(notes) < 2:
            return notes

        result = [notes[0]]

        for i in range(1, len(notes)):
            prev = result[-1]
            curr = notes[i]

            leap = abs(curr.pitch - prev.pitch)

            if leap > max_leap:
                # Interpolate with passing tone (conceptual)
                new_pitch = self._minimize_leap(prev.pitch, curr.pitch)
                result.append(HarmonyNote(
                    pitch=new_pitch,
                    start_beat=curr.start_beat,
                    duration_beats=curr.duration_beats,
                    velocity=curr.velocity,
                    interval_from_melody=curr.interval_from_melody,
                ))
            else:
                result.append(curr)

        return result


class HarmonyGenerator:
    """
    High-level harmony generation.

    Combines scale harmonization with voice leading.
    """

    def __init__(
        self,
        key_root: int = 0,
        scale_type: str = "major",
    ):
        """
        Initialize harmony generator.

        Args:
            key_root: Root note (0-11, C=0)
            scale_type: Scale type (major, minor, dorian, etc.)
        """
        scale_map = {
            "major": ScaleHarmonizer.MAJOR_SCALE,
            "minor": ScaleHarmonizer.MINOR_SCALE,
            "harmonic_minor": ScaleHarmonizer.HARMONIC_MINOR,
            "dorian": ScaleHarmonizer.DORIAN_MODE,
            "mixolydian": ScaleHarmonizer.MIXOLYDIAN_MODE,
        }

        scale = scale_map.get(scale_type, ScaleHarmonizer.MAJOR_SCALE)

        self.harmonizer = ScaleHarmonizer(root=key_root, scale=scale)
        self.voice_leading = VoiceLeadingEngine()

    def generate_harmony(
        self,
        melody: List[dict],
        num_voices: int = 2,
        style: str = "close",  # close, open, mixed
    ) -> List[HarmonyVoice]:
        """
        Generate multi-voice harmony.

        Args:
            melody: Melody notes
            num_voices: Number of harmony voices
            style: Voicing style

        Returns:
            List of harmony voices
        """
        # Determine intervals based on style
        if style == "close":
            intervals = ["third_above", "third_below", "sixth_above"][:num_voices]
        elif style == "open":
            intervals = ["fifth_above", "fifth_below", "third_above"][:num_voices]
        else:
            intervals = ["third_above", "fifth_below", "sixth_above"][:num_voices]

        # Generate raw harmonies
        raw_voices = self.harmonizer.harmonize_melody(melody, intervals)

        # Convert melody to HarmonyNote format for voice leading
        melody_notes = [
            HarmonyNote(
                pitch=n["pitch"],
                start_beat=n.get("start", n.get("start_beat", 0)),
                duration_beats=n.get("duration", n.get("duration_beats", 1)),
                velocity=n.get("velocity", 100),
            )
            for n in melody
        ]

        # Apply voice leading to each voice
        led_voices = []
        for voice in raw_voices:
            led_notes = self.voice_leading.apply_voice_leading(
                melody_notes,
                voice.notes,
            )
            led_voices.append(HarmonyVoice(
                name=voice.name,
                notes=led_notes,
                default_interval=voice.default_interval,
                volume_db=voice.volume_db,
                pan=voice.pan,
            ))

        return led_voices

    def generate_chord_based_harmony(
        self,
        melody: List[dict],
        chord_progression: List[Tuple[float, str]],  # [(beat, chord_name), ...]
    ) -> List[HarmonyVoice]:
        """
        Generate harmony based on chord progression.

        Args:
            melody: Melody notes
            chord_progression: Chord changes with timing

        Returns:
            Harmony voices
        """
        # Parse chords
        chord_map = self._parse_chords(chord_progression)

        # Generate context-aware harmony
        voices = []

        # High harmony voice
        high_notes = []
        for note in melody:
            beat = note.get("start", note.get("start_beat", 0))
            chord = self._get_chord_at_beat(beat, chord_map)

            harm_pitch = self._get_chord_tone_above(note["pitch"], chord)
            high_notes.append(HarmonyNote(
                pitch=harm_pitch,
                start_beat=beat,
                duration_beats=note.get("duration", note.get("duration_beats", 1)),
                velocity=note.get("velocity", 100),
                interval_from_melody=harm_pitch - note["pitch"],
            ))

        voices.append(HarmonyVoice(
            name="chord_high",
            notes=self.voice_leading.smooth_voice(high_notes),
            default_interval=4,
            volume_db=-6,
            pan=0.2,
        ))

        # Low harmony voice
        low_notes = []
        for note in melody:
            beat = note.get("start", note.get("start_beat", 0))
            chord = self._get_chord_at_beat(beat, chord_map)

            harm_pitch = self._get_chord_tone_below(note["pitch"], chord)
            low_notes.append(HarmonyNote(
                pitch=harm_pitch,
                start_beat=beat,
                duration_beats=note.get("duration", note.get("duration_beats", 1)),
                velocity=note.get("velocity", 100),
                interval_from_melody=harm_pitch - note["pitch"],
            ))

        voices.append(HarmonyVoice(
            name="chord_low",
            notes=self.voice_leading.smooth_voice(low_notes),
            default_interval=-3,
            volume_db=-6,
            pan=-0.2,
        ))

        return voices

    def _parse_chords(
        self,
        progression: List[Tuple[float, str]],
    ) -> Dict[float, List[int]]:
        """Parse chord names to pitch classes."""
        result = {}

        for beat, chord_name in progression:
            # Simple chord parsing
            root_map = {
                "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
            }

            if not chord_name:
                continue

            root = root_map.get(chord_name[0].upper(), 0)
            quality = chord_name[1:] if len(chord_name) > 1 else ""

            # Handle sharps/flats
            if "#" in quality:
                root = (root + 1) % 12
                quality = quality.replace("#", "")
            elif "b" in quality:
                root = (root - 1) % 12
                quality = quality.replace("b", "")

            # Build chord
            if "m" in quality and "maj" not in quality:
                intervals = [0, 3, 7]  # Minor
            elif "dim" in quality:
                intervals = [0, 3, 6]  # Diminished
            elif "aug" in quality:
                intervals = [0, 4, 8]  # Augmented
            else:
                intervals = [0, 4, 7]  # Major

            chord_pitches = [(root + i) % 12 for i in intervals]
            result[beat] = chord_pitches

        return result

    def _get_chord_at_beat(
        self,
        beat: float,
        chord_map: Dict[float, List[int]],
    ) -> List[int]:
        """Get active chord at a beat."""
        active_chord = [0, 4, 7]  # Default C major

        for chord_beat, pitches in sorted(chord_map.items()):
            if chord_beat <= beat:
                active_chord = pitches
            else:
                break

        return active_chord

    def _get_chord_tone_above(
        self,
        melody_pitch: int,
        chord: List[int],
    ) -> int:
        """Get nearest chord tone above melody."""
        melody_pc = melody_pitch % 12
        melody_octave = melody_pitch // 12

        # Find chord tones above
        for offset in range(1, 13):
            candidate_pc = (melody_pc + offset) % 12
            if candidate_pc in chord:
                return melody_pitch + offset

        return melody_pitch + 4  # Default to major third

    def _get_chord_tone_below(
        self,
        melody_pitch: int,
        chord: List[int],
    ) -> int:
        """Get nearest chord tone below melody."""
        melody_pc = melody_pitch % 12
        melody_octave = melody_pitch // 12

        # Find chord tones below
        for offset in range(1, 13):
            candidate_pc = (melody_pc - offset) % 12
            if candidate_pc in chord:
                return melody_pitch - offset

        return melody_pitch - 3  # Default to minor third
