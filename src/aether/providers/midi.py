"""
AETHER MIDI Provider

Production-grade MIDI generation, manipulation, and rendering.

Features:
- Algorithmic MIDI generation from musical specs
- MIDI file I/O (read/write)
- Transposition, quantization, humanization
- Chord and melody generation
- Drum pattern generation
- Integration with music theory utilities

Example:
    provider = AlgorithmicMIDIProvider()
    await provider.initialize()

    midi = await provider.generate_from_spec(
        harmony_spec={"progression": ["Cm", "Ab", "Eb", "Bb"]},
        melody_spec={"contour": "arch", "range_octaves": 1.5},
        rhythm_spec={"bpm": 90, "swing": 0.1},
        arrangement_spec={"sections": ["intro", "verse", "chorus"]},
    )
    await provider.render_to_file(midi, Path("output.mid"))
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any

from aether.providers.base import (
    MIDIFile,
    MIDINote,
    MIDIProvider,
    MIDITrack,
    ProviderInfo,
    ProviderStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================


# MIDI note numbers
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Chord intervals (semitones from root)
CHORD_INTERVALS = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "dim7": [0, 3, 6, 9],
    "m7b5": [0, 3, 6, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add9": [0, 4, 7, 14],
    "9": [0, 4, 7, 10, 14],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "add11": [0, 4, 7, 17],
    "7sus4": [0, 5, 7, 10],
}

# Scale intervals
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

# GM drum map
GM_DRUMS = {
    "kick": 36,
    "kick2": 35,
    "snare": 38,
    "snare2": 40,
    "sidestick": 37,
    "hihat_closed": 42,
    "hihat_pedal": 44,
    "hihat_open": 46,
    "tom_low": 45,
    "tom_mid": 47,
    "tom_high": 50,
    "tom_floor": 41,
    "crash": 49,
    "crash2": 57,
    "ride": 51,
    "ride_bell": 53,
    "clap": 39,
    "rim": 37,
    "cowbell": 56,
    "tambourine": 54,
    "shaker": 70,
    "conga_high": 63,
    "conga_low": 64,
    "bongo_high": 60,
    "bongo_low": 61,
}

# GM instrument programs by category
GM_INSTRUMENTS = {
    # Piano
    "acoustic_grand": 0,
    "bright_acoustic": 1,
    "electric_grand": 2,
    "honky_tonk": 3,
    "electric_piano1": 4,
    "electric_piano2": 5,
    "harpsichord": 6,
    "clavinet": 7,
    # Chromatic Percussion
    "celesta": 8,
    "glockenspiel": 9,
    "music_box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular_bells": 14,
    "dulcimer": 15,
    # Organ
    "drawbar_organ": 16,
    "percussive_organ": 17,
    "rock_organ": 18,
    "church_organ": 19,
    "reed_organ": 20,
    "accordion": 21,
    "harmonica": 22,
    "tango_accordion": 23,
    # Guitar
    "nylon_guitar": 24,
    "steel_guitar": 25,
    "jazz_guitar": 26,
    "clean_guitar": 27,
    "muted_guitar": 28,
    "overdrive_guitar": 29,
    "distortion_guitar": 30,
    "harmonics_guitar": 31,
    # Bass
    "acoustic_bass": 32,
    "finger_bass": 33,
    "pick_bass": 34,
    "fretless_bass": 35,
    "slap_bass1": 36,
    "slap_bass2": 37,
    "synth_bass1": 38,
    "synth_bass2": 39,
    # Strings
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "tremolo_strings": 44,
    "pizzicato_strings": 45,
    "orchestral_harp": 46,
    "timpani": 47,
    # Ensemble
    "string_ensemble1": 48,
    "string_ensemble2": 49,
    "synth_strings1": 50,
    "synth_strings2": 51,
    "choir_aahs": 52,
    "voice_oohs": 53,
    "synth_voice": 54,
    "orchestra_hit": 55,
    # Brass
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "muted_trumpet": 59,
    "french_horn": 60,
    "brass_section": 61,
    "synth_brass1": 62,
    "synth_brass2": 63,
    # Reed
    "soprano_sax": 64,
    "alto_sax": 65,
    "tenor_sax": 66,
    "baritone_sax": 67,
    "oboe": 68,
    "english_horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    # Pipe
    "piccolo": 72,
    "flute": 73,
    "recorder": 74,
    "pan_flute": 75,
    "blown_bottle": 76,
    "shakuhachi": 77,
    "whistle": 78,
    "ocarina": 79,
    # Synth Lead
    "lead_square": 80,
    "lead_sawtooth": 81,
    "lead_calliope": 82,
    "lead_chiff": 83,
    "lead_charang": 84,
    "lead_voice": 85,
    "lead_fifths": 86,
    "lead_bass": 87,
    # Synth Pad
    "pad_new_age": 88,
    "pad_warm": 89,
    "pad_polysynth": 90,
    "pad_choir": 91,
    "pad_bowed": 92,
    "pad_metallic": 93,
    "pad_halo": 94,
    "pad_sweep": 95,
    # Synth Effects
    "fx_rain": 96,
    "fx_soundtrack": 97,
    "fx_crystal": 98,
    "fx_atmosphere": 99,
    "fx_brightness": 100,
    "fx_goblins": 101,
    "fx_echoes": 102,
    "fx_scifi": 103,
}

# Genre-specific instrument mappings
GENRE_INSTRUMENTS = {
    "hip-hop-boom-bap": {
        "chords": GM_INSTRUMENTS["electric_piano1"],
        "bass": GM_INSTRUMENTS["synth_bass1"],
        "melody": GM_INSTRUMENTS["vibraphone"],
        "pad": GM_INSTRUMENTS["string_ensemble1"],
    },
    "synthwave": {
        "chords": GM_INSTRUMENTS["pad_warm"],
        "bass": GM_INSTRUMENTS["synth_bass1"],
        "melody": GM_INSTRUMENTS["lead_sawtooth"],
        "pad": GM_INSTRUMENTS["pad_polysynth"],
        "arp": GM_INSTRUMENTS["lead_square"],
    },
    "lo-fi-hip-hop": {
        "chords": GM_INSTRUMENTS["electric_piano2"],
        "bass": GM_INSTRUMENTS["fretless_bass"],
        "melody": GM_INSTRUMENTS["vibraphone"],
        "pad": GM_INSTRUMENTS["pad_warm"],
    },
    "house": {
        "chords": GM_INSTRUMENTS["pad_polysynth"],
        "bass": GM_INSTRUMENTS["synth_bass2"],
        "melody": GM_INSTRUMENTS["lead_sawtooth"],
        "pad": GM_INSTRUMENTS["pad_warm"],
    },
    "jazz": {
        "chords": GM_INSTRUMENTS["acoustic_grand"],
        "bass": GM_INSTRUMENTS["acoustic_bass"],
        "melody": GM_INSTRUMENTS["tenor_sax"],
        "pad": GM_INSTRUMENTS["string_ensemble1"],
    },
    "rock": {
        "chords": GM_INSTRUMENTS["overdrive_guitar"],
        "bass": GM_INSTRUMENTS["pick_bass"],
        "melody": GM_INSTRUMENTS["distortion_guitar"],
        "pad": GM_INSTRUMENTS["rock_organ"],
    },
    "techno": {
        "chords": GM_INSTRUMENTS["pad_bowed"],
        "bass": GM_INSTRUMENTS["synth_bass2"],
        "melody": GM_INSTRUMENTS["lead_square"],
        "pad": GM_INSTRUMENTS["pad_metallic"],
        "arp": GM_INSTRUMENTS["lead_charang"],
    },
    "ambient": {
        "chords": GM_INSTRUMENTS["pad_choir"],
        "bass": GM_INSTRUMENTS["synth_bass1"],
        "melody": GM_INSTRUMENTS["pad_halo"],
        "pad": GM_INSTRUMENTS["pad_warm"],
    },
    "r-and-b": {
        "chords": GM_INSTRUMENTS["electric_piano1"],
        "bass": GM_INSTRUMENTS["finger_bass"],
        "melody": GM_INSTRUMENTS["alto_sax"],
        "pad": GM_INSTRUMENTS["string_ensemble1"],
    },
    "funk": {
        "chords": GM_INSTRUMENTS["clavinet"],
        "bass": GM_INSTRUMENTS["slap_bass1"],
        "melody": GM_INSTRUMENTS["trumpet"],
        "pad": GM_INSTRUMENTS["drawbar_organ"],
    },
    "disco": {
        "chords": GM_INSTRUMENTS["electric_piano1"],
        "bass": GM_INSTRUMENTS["synth_bass1"],
        "melody": GM_INSTRUMENTS["brass_section"],
        "pad": GM_INSTRUMENTS["string_ensemble1"],
    },
    "trap": {
        "chords": GM_INSTRUMENTS["pad_bowed"],
        "bass": GM_INSTRUMENTS["synth_bass2"],
        "melody": GM_INSTRUMENTS["glockenspiel"],
        "pad": GM_INSTRUMENTS["pad_metallic"],
    },
    "drum-and-bass": {
        "chords": GM_INSTRUMENTS["pad_polysynth"],
        "bass": GM_INSTRUMENTS["synth_bass2"],
        "melody": GM_INSTRUMENTS["lead_sawtooth"],
        "pad": GM_INSTRUMENTS["pad_warm"],
    },
    "dubstep": {
        "chords": GM_INSTRUMENTS["pad_bowed"],
        "bass": GM_INSTRUMENTS["synth_bass2"],
        "melody": GM_INSTRUMENTS["lead_charang"],
        "pad": GM_INSTRUMENTS["pad_metallic"],
    },
    "acoustic-folk": {
        "chords": GM_INSTRUMENTS["nylon_guitar"],
        "bass": GM_INSTRUMENTS["acoustic_bass"],
        "melody": GM_INSTRUMENTS["steel_guitar"],
        "pad": GM_INSTRUMENTS["harmonica"],
    },
    "cinematic": {
        "chords": GM_INSTRUMENTS["string_ensemble1"],
        "bass": GM_INSTRUMENTS["contrabass"],
        "melody": GM_INSTRUMENTS["french_horn"],
        "pad": GM_INSTRUMENTS["choir_aahs"],
    },
    "chillwave": {
        "chords": GM_INSTRUMENTS["pad_warm"],
        "bass": GM_INSTRUMENTS["synth_bass1"],
        "melody": GM_INSTRUMENTS["synth_voice"],
        "pad": GM_INSTRUMENTS["pad_choir"],
    },
    "neo-soul": {
        "chords": GM_INSTRUMENTS["electric_piano2"],
        "bass": GM_INSTRUMENTS["fretless_bass"],
        "melody": GM_INSTRUMENTS["alto_sax"],
        "pad": GM_INSTRUMENTS["string_ensemble1"],
    },
    "default": {
        "chords": GM_INSTRUMENTS["acoustic_grand"],
        "bass": GM_INSTRUMENTS["finger_bass"],
        "melody": GM_INSTRUMENTS["acoustic_grand"],
        "pad": GM_INSTRUMENTS["string_ensemble1"],
    },
}


# ============================================================================
# Helper Functions
# ============================================================================


def note_name_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number."""
    note = note.upper().replace("♯", "#").replace("♭", "b")

    # Handle flats by converting to sharps
    flat_to_sharp = {
        "DB": "C#",
        "EB": "D#",
        "FB": "E",
        "GB": "F#",
        "AB": "G#",
        "BB": "A#",
        "CB": "B",
    }
    if len(note) == 2 and note[1] == "B" and note in flat_to_sharp:
        note = flat_to_sharp[note]

    # Find note index
    base_note = note[0]
    is_sharp = len(note) > 1 and note[1] == "#"

    try:
        note_index = NOTE_NAMES.index(base_note)
        if is_sharp:
            note_index += 1
    except ValueError:
        note_index = 0

    return (octave + 1) * 12 + (note_index % 12)


def midi_to_note_name(midi_note: int) -> tuple[str, int]:
    """Convert MIDI number to note name and octave."""
    octave = (midi_note // 12) - 1
    note_index = midi_note % 12
    return NOTE_NAMES[note_index], octave


def parse_chord(chord_str: str) -> tuple[int, str, int]:
    """
    Parse chord string into root MIDI note, quality, and bass note.

    Examples:
        "Cm" -> (60, "minor", 60)
        "G7" -> (67, "dom7", 67)
        "F/A" -> (65, "major", 57)
    """
    chord_str = chord_str.strip()
    if not chord_str:
        return 60, "major", 60

    # Handle slash chords
    bass_note = None
    if "/" in chord_str:
        chord_str, bass_str = chord_str.split("/")
        bass_note = note_name_to_midi(bass_str, 2)

    # Extract root note
    if len(chord_str) >= 2 and chord_str[1] in "#b":
        root = chord_str[:2]
        quality_str = chord_str[2:]
    else:
        root = chord_str[0]
        quality_str = chord_str[1:]

    root_midi = note_name_to_midi(root, 4)

    # Determine quality
    quality_str = quality_str.lower()
    if quality_str in ["", "maj"]:
        quality = "major"
    elif quality_str in ["m", "min", "-"]:
        quality = "minor"
    elif quality_str in ["dim", "o", "°"]:
        quality = "dim"
    elif quality_str in ["aug", "+", "+"]:
        quality = "aug"
    elif quality_str in ["maj7", "ma7", "△7", "δ7"]:
        quality = "maj7"
    elif quality_str in ["m7", "min7", "-7"]:
        quality = "min7"
    elif quality_str in ["7", "dom7"]:
        quality = "dom7"
    elif quality_str in ["dim7", "o7", "°7"]:
        quality = "dim7"
    elif quality_str in ["m7b5", "ø", "ø7"]:
        quality = "m7b5"
    elif quality_str in ["sus2"]:
        quality = "sus2"
    elif quality_str in ["sus4", "sus"]:
        quality = "sus4"
    elif quality_str in ["add9"]:
        quality = "add9"
    elif quality_str in ["9"]:
        quality = "9"
    elif quality_str in ["6"]:
        quality = "6"
    elif quality_str in ["m6"]:
        quality = "m6"
    else:
        quality = "major"

    if bass_note is None:
        bass_note = root_midi - 12  # Default bass is root an octave lower

    return root_midi, quality, bass_note


def get_chord_notes(root: int, quality: str, inversion: int = 0) -> list[int]:
    """Get MIDI notes for a chord."""
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["major"])
    notes = [root + interval for interval in intervals]

    # Apply inversion
    for _ in range(inversion % len(notes)):
        notes[0] += 12
        notes = notes[1:] + [notes[0]]

    return notes


def get_scale_notes(root: int, scale: str, octave_range: int = 2) -> list[int]:
    """Get MIDI notes for a scale across octaves."""
    intervals = SCALE_INTERVALS.get(scale, SCALE_INTERVALS["major"])
    notes = []
    for oct in range(octave_range):
        for interval in intervals:
            notes.append(root + interval + (oct * 12))
    return notes


def snap_to_scale(pitch: int, scale_notes: list[int]) -> int:
    """Snap a pitch to the nearest note in the scale."""
    if not scale_notes:
        return pitch

    # Find closest scale note
    closest = min(scale_notes, key=lambda n: abs((n % 12) - (pitch % 12)))
    # Preserve octave but use scale note's pitch class
    octave = pitch // 12
    return octave * 12 + (closest % 12)


def get_chord_tones_in_scale(chord_root: int, chord_quality: str, scale_root: int, scale_type: str) -> list[int]:
    """Get chord tones that fit within a scale for melodic use."""
    chord_notes = get_chord_notes(chord_root, chord_quality)
    scale_intervals = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS["major"])
    scale_pitch_classes = set((scale_root + i) % 12 for i in scale_intervals)

    # Return chord tones that are in the scale (prioritize these for melody)
    return [n for n in chord_notes if (n % 12) in scale_pitch_classes]


# ============================================================================
# Algorithmic MIDI Provider
# ============================================================================


class AlgorithmicMIDIProvider(MIDIProvider):
    """
    Algorithmic MIDI generation provider.

    Generates MIDI from musical specifications using rule-based algorithms.
    No external API dependencies - pure Python implementation.

    Capabilities:
    - Chord progressions with voicings
    - Melodic contour generation with chord-tone awareness
    - Drum pattern generation
    - Bass line generation
    - Humanization (timing/velocity variation)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._rng = random.Random()

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Algorithmic MIDI Provider",
            version="2.0.0",
            provider_type="midi",
            status=self._status,
            capabilities=[
                "chord_generation",
                "melody_generation",
                "chord_tone_melody",
                "drum_patterns",
                "bass_lines",
                "humanization",
                "transposition",
                "quantization",
                "genre_aware",
            ],
            config=self.config,
        )

    async def initialize(self) -> bool:
        """Initialize the provider."""
        self._status = ProviderStatus.AVAILABLE
        logger.info("Algorithmic MIDI provider initialized")
        return True

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._status = ProviderStatus.UNAVAILABLE

    async def health_check(self) -> bool:
        return self._status == ProviderStatus.AVAILABLE

    async def generate_from_spec(
        self,
        harmony_spec: Any,
        melody_spec: Any,
        rhythm_spec: Any,
        arrangement_spec: Any,
    ) -> MIDIFile:
        """
        Generate complete MIDI from specifications.

        Args:
            harmony_spec: Chord progression and harmonic content
            melody_spec: Melodic contour and characteristics
            rhythm_spec: Tempo, time signature, groove
            arrangement_spec: Structure, sections, dynamics

        Returns:
            Complete MIDIFile with all tracks
        """
        # Extract parameters
        bpm = rhythm_spec.get("bpm", 120)
        time_sig = rhythm_spec.get("time_signature", (4, 4))
        swing = rhythm_spec.get("swing", 0.0)
        genre = rhythm_spec.get("genre", "default")

        progression = harmony_spec.get("progression", ["C", "Am", "F", "G"])
        key_root = harmony_spec.get("key_root", "C")
        mode = harmony_spec.get("mode", "major")

        sections = arrangement_spec.get("sections", ["verse", "chorus"])
        bars_per_section = arrangement_spec.get("bars_per_section", 8)

        # Generate seed if provided for reproducibility
        seed = rhythm_spec.get("seed")
        if seed is not None:
            self._rng.seed(seed)

        # Get genre-specific instruments
        instruments = GENRE_INSTRUMENTS.get(genre, GENRE_INSTRUMENTS["default"])

        tracks = []
        total_bars = len(sections) * bars_per_section
        bars_per_chord = max(1, bars_per_section // len(progression))

        # Generate chord track
        chord_track = self._generate_chord_track(
            progression=progression,
            bars=total_bars,
            bars_per_chord=bars_per_chord,
            time_sig=time_sig,
            key_root=key_root,
            mode=mode,
            program=instruments.get("chords", 0),
        )
        tracks.append(chord_track)

        # Generate bass track with chord awareness
        bass_track = self._generate_bass_track(
            progression=progression,
            bars=total_bars,
            bars_per_chord=bars_per_chord,
            time_sig=time_sig,
            style=rhythm_spec.get("bass_style", "root_fifth"),
            key_root=key_root,
            mode=mode,
            program=instruments.get("bass", 33),
        )
        tracks.append(bass_track)

        # Generate melody track with chord-tone awareness
        melody_track = self._generate_melody_track(
            key_root=key_root,
            mode=mode,
            progression=progression,
            bars=total_bars,
            bars_per_chord=bars_per_chord,
            time_sig=time_sig,
            contour=melody_spec.get("contour", "arch"),
            density=melody_spec.get("density", 0.5),
            program=instruments.get("melody", 0),
        )
        tracks.append(melody_track)

        # Generate drum track
        drum_track = self._generate_drum_track(
            bars=total_bars,
            time_sig=time_sig,
            style=rhythm_spec.get("drum_style", "standard"),
            swing=swing,
            genre=genre,
        )
        tracks.append(drum_track)

        # Apply humanization if requested
        humanize = rhythm_spec.get("humanize", 0.0)
        if humanize > 0:
            tracks = [self._humanize_track(t, humanize) for t in tracks]

        return MIDIFile(
            tracks=tracks,
            tempo_bpm=bpm,
            time_signature=time_sig,
            ticks_per_beat=480,
        )

    def _generate_chord_track(
        self,
        progression: list[str],
        bars: int,
        bars_per_chord: int,
        time_sig: tuple,
        key_root: str,
        mode: str,
        program: int = 0,
    ) -> MIDITrack:
        """Generate chord track with proper voicings."""
        notes = []
        beats_per_bar = time_sig[0]
        current_beat = 0

        # Get scale for the key
        root_midi = note_name_to_midi(key_root, 4)
        scale_notes = get_scale_notes(root_midi, mode, 2)

        prev_chord_notes = None

        for bar in range(bars):
            chord_idx = (bar // bars_per_chord) % len(progression)
            chord_str = progression[chord_idx]
            root, quality, _ = parse_chord(chord_str)

            # Voice leading: minimize movement from previous chord
            chord_notes = get_chord_notes(root, quality)

            if prev_chord_notes:
                # Try different inversions and pick smoothest voice leading
                best_inversion = 0
                best_distance = float('inf')
                for inv in range(len(chord_notes)):
                    test_notes = get_chord_notes(root, quality, inv)
                    # Calculate total voice movement
                    distance = sum(
                        min(abs(n - p) for p in prev_chord_notes)
                        for n in test_notes
                    )
                    if distance < best_distance:
                        best_distance = distance
                        best_inversion = inv
                chord_notes = get_chord_notes(root, quality, best_inversion)

            prev_chord_notes = chord_notes

            # Only add chord on first bar of chord change
            if bar % bars_per_chord == 0:
                duration = beats_per_bar * bars_per_chord - 0.5
                for pitch in chord_notes:
                    notes.append(
                        MIDINote(
                            pitch=pitch,
                            velocity=self._rng.randint(65, 85),
                            start_time=current_beat,
                            duration=duration,
                            channel=0,
                        )
                    )

            current_beat += beats_per_bar

        return MIDITrack(
            name="Chords",
            notes=notes,
            program=program,
            channel=0,
        )

    def _generate_bass_track(
        self,
        progression: list[str],
        bars: int,
        bars_per_chord: int,
        time_sig: tuple,
        style: str,
        key_root: str,
        mode: str,
        program: int = 33,
    ) -> MIDITrack:
        """Generate bass track with proper chord awareness."""
        notes = []
        beats_per_bar = time_sig[0]
        current_beat = 0

        # Get scale
        root_midi = note_name_to_midi(key_root, 3)  # Bass in octave 3
        scale_notes = get_scale_notes(root_midi - 12, mode, 2)

        for bar in range(bars):
            chord_idx = (bar // bars_per_chord) % len(progression)
            chord_str = progression[chord_idx]
            root, quality, _ = parse_chord(chord_str)

            # Bass root 2 octaves below chord root
            bass_root = root - 24

            # Get actual chord intervals for the fifth
            chord_intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["major"])
            fifth_interval = chord_intervals[2] if len(chord_intervals) > 2 else 7
            third_interval = chord_intervals[1] if len(chord_intervals) > 1 else 4

            if style == "root_fifth":
                # Root on beat 1
                notes.append(
                    MIDINote(
                        pitch=bass_root,
                        velocity=self._rng.randint(85, 100),
                        start_time=current_beat,
                        duration=1.5,
                        channel=1,
                    )
                )
                # Fifth on beat 3 (using actual chord fifth)
                if beats_per_bar >= 4:
                    notes.append(
                        MIDINote(
                            pitch=bass_root + fifth_interval,
                            velocity=self._rng.randint(75, 90),
                            start_time=current_beat + 2,
                            duration=1.5,
                            channel=1,
                        )
                    )
            elif style == "walking":
                # Walking bass using chord tones and scale
                chord_tones = [bass_root, bass_root + third_interval, bass_root + fifth_interval]
                approach_notes = [n - 1 for n in chord_tones] + [n + 1 for n in chord_tones]

                for beat in range(beats_per_bar):
                    if beat == 0:
                        pitch = bass_root  # Always start on root
                    elif beat == beats_per_bar - 1:
                        # Approach next chord root
                        next_chord_idx = ((bar + 1) // bars_per_chord) % len(progression)
                        next_root = parse_chord(progression[next_chord_idx])[0] - 24
                        pitch = next_root - 1 if self._rng.random() > 0.5 else next_root + 1
                    else:
                        # Choose from chord tones or passing tones
                        if self._rng.random() < 0.7:
                            pitch = self._rng.choice(chord_tones)
                        else:
                            pitch = snap_to_scale(
                                self._rng.choice(approach_notes),
                                scale_notes
                            )

                    notes.append(
                        MIDINote(
                            pitch=pitch,
                            velocity=self._rng.randint(75, 95),
                            start_time=current_beat + beat,
                            duration=0.9,
                            channel=1,
                        )
                    )
            elif style == "synth":
                # Synth bass - longer sustained notes with octave
                notes.append(
                    MIDINote(
                        pitch=bass_root,
                        velocity=self._rng.randint(90, 110),
                        start_time=current_beat,
                        duration=beats_per_bar - 0.25,
                        channel=1,
                    )
                )
                # Add octave for thickness
                notes.append(
                    MIDINote(
                        pitch=bass_root + 12,
                        velocity=self._rng.randint(60, 80),
                        start_time=current_beat,
                        duration=beats_per_bar - 0.25,
                        channel=1,
                    )
                )
            else:  # Simple root
                notes.append(
                    MIDINote(
                        pitch=bass_root,
                        velocity=self._rng.randint(85, 100),
                        start_time=current_beat,
                        duration=beats_per_bar - 0.5,
                        channel=1,
                    )
                )

            current_beat += beats_per_bar

        return MIDITrack(
            name="Bass",
            notes=notes,
            program=program,
            channel=1,
        )

    def _generate_melody_track(
        self,
        key_root: str,
        mode: str,
        progression: list[str],
        bars: int,
        bars_per_chord: int,
        time_sig: tuple,
        contour: str = "arch",
        density: float = 0.5,
        program: int = 0,
    ) -> MIDITrack:
        """Generate melody track with chord-tone awareness."""
        notes = []
        beats_per_bar = time_sig[0]
        total_beats = bars * beats_per_bar

        # Get scale
        root_midi = note_name_to_midi(key_root, 5)  # Melody in octave 5
        scale_notes = get_scale_notes(root_midi, mode, 2)

        # Generate melody
        current_beat = 0
        prev_pitch = scale_notes[len(scale_notes) // 2]  # Start in middle

        while current_beat < total_beats:
            # Get current chord
            current_bar = int(current_beat / beats_per_bar)
            chord_idx = (current_bar // bars_per_chord) % len(progression)
            chord_str = progression[chord_idx]
            chord_root, chord_quality, _ = parse_chord(chord_str)

            # Get chord tones for current harmony
            chord_tones = get_chord_notes(chord_root, chord_quality)
            # Extend to melody octave
            melody_chord_tones = []
            for ct in chord_tones:
                for octave_shift in [-12, 0, 12]:
                    shifted = ct + octave_shift
                    if root_midi - 12 <= shifted <= root_midi + 24:
                        melody_chord_tones.append(shifted)

            # Determine if we play a note based on density
            if self._rng.random() < density:
                # Calculate target direction based on contour
                progress = current_beat / total_beats

                if contour == "arch":
                    target_height = 1 - 4 * (progress - 0.5) ** 2
                elif contour == "ascending":
                    target_height = progress
                elif contour == "descending":
                    target_height = 1 - progress
                elif contour == "wave":
                    target_height = 0.5 + 0.5 * math.sin(progress * math.pi * 4)
                else:
                    target_height = 0.5

                # Determine if this is a strong beat (chord tone priority)
                beat_in_bar = current_beat % beats_per_bar
                is_strong_beat = beat_in_bar in [0, 2] if beats_per_bar == 4 else beat_in_bar == 0

                # Choose pitch
                if is_strong_beat and melody_chord_tones and self._rng.random() < 0.8:
                    # Strong beats: prefer chord tones
                    target_idx = int(target_height * (len(melody_chord_tones) - 1))
                    target_idx = max(0, min(len(melody_chord_tones) - 1, target_idx))

                    # Find chord tone closest to previous pitch for smooth motion
                    candidates = sorted(melody_chord_tones, key=lambda n: abs(n - prev_pitch))
                    pitch = candidates[0] if abs(candidates[0] - prev_pitch) <= 7 else melody_chord_tones[target_idx]
                else:
                    # Weak beats or passing: use scale tones
                    target_idx = int(target_height * (len(scale_notes) - 1))
                    target_idx = max(0, min(len(scale_notes) - 1, target_idx))

                    # Smooth melodic motion
                    step = self._rng.choice([-2, -1, 0, 1, 2])
                    current_idx = min(range(len(scale_notes)), key=lambda i: abs(scale_notes[i] - prev_pitch))
                    new_idx = max(0, min(len(scale_notes) - 1, current_idx + step))

                    # Blend with contour target
                    if abs(new_idx - target_idx) > 3:
                        new_idx = new_idx + (1 if target_idx > new_idx else -1)

                    pitch = scale_notes[new_idx]

                # Ensure pitch is in valid range
                pitch = max(root_midi - 12, min(root_midi + 24, pitch))
                prev_pitch = pitch

                # Determine note duration
                durations = [0.5, 1.0, 1.5, 2.0]
                weights = [0.3, 0.4, 0.2, 0.1]
                duration = self._rng.choices(durations, weights)[0]

                notes.append(
                    MIDINote(
                        pitch=pitch,
                        velocity=self._rng.randint(70, 100),
                        start_time=current_beat,
                        duration=duration * 0.9,
                        channel=2,
                    )
                )

                current_beat += duration
            else:
                current_beat += 0.5  # Rest

        return MIDITrack(
            name="Melody",
            notes=notes,
            program=program,
            channel=2,
        )

    def _generate_drum_track(
        self,
        bars: int,
        time_sig: tuple,
        style: str = "standard",
        swing: float = 0.0,
        genre: str = "default",
    ) -> MIDITrack:
        """Generate drum track."""
        notes = []
        beats_per_bar = time_sig[0]
        current_beat = 0

        for bar in range(bars):
            if style == "standard" or style == "rock":
                # Standard rock/pop pattern
                notes.append(
                    MIDINote(
                        pitch=GM_DRUMS["kick"],
                        velocity=100,
                        start_time=current_beat,
                        duration=0.5,
                        channel=9,
                    )
                )
                notes.append(
                    MIDINote(
                        pitch=GM_DRUMS["kick"],
                        velocity=90,
                        start_time=current_beat + 2,
                        duration=0.5,
                        channel=9,
                    )
                )

                # Snare on 2 and 4
                notes.append(
                    MIDINote(
                        pitch=GM_DRUMS["snare"],
                        velocity=100,
                        start_time=current_beat + 1,
                        duration=0.5,
                        channel=9,
                    )
                )
                notes.append(
                    MIDINote(
                        pitch=GM_DRUMS["snare"],
                        velocity=95,
                        start_time=current_beat + 3,
                        duration=0.5,
                        channel=9,
                    )
                )

                # Hi-hat on every 8th
                for eighth in range(8):
                    beat = current_beat + (eighth * 0.5)
                    if eighth % 2 == 1 and swing > 0:
                        beat += swing * 0.5
                    velocity = 80 if eighth % 2 == 0 else 60
                    notes.append(
                        MIDINote(
                            pitch=GM_DRUMS["hihat_closed"],
                            velocity=velocity,
                            start_time=beat,
                            duration=0.25,
                            channel=9,
                        )
                    )

            elif style == "boom_bap":
                # Hip-hop boom bap pattern
                kick_pattern = [0, 0.75, 2.5]
                for k in kick_pattern:
                    notes.append(
                        MIDINote(
                            pitch=GM_DRUMS["kick"],
                            velocity=100,
                            start_time=current_beat + k,
                            duration=0.5,
                            channel=9,
                        )
                    )

                # Snare on 2 and 4
                notes.append(
                    MIDINote(
                        pitch=GM_DRUMS["snare"],
                        velocity=100,
                        start_time=current_beat + 1,
                        duration=0.5,
                        channel=9,
                    )
                )
                notes.append(
                    MIDINote(
                        pitch=GM_DRUMS["snare"],
                        velocity=95,
                        start_time=current_beat + 3,
                        duration=0.5,
                        channel=9,
                    )
                )

                # Hi-hat with swing
                for eighth in range(8):
                    beat = current_beat + (eighth * 0.5)
                    if eighth % 2 == 1:
                        beat += swing * 0.5
                    notes.append(
                        MIDINote(
                            pitch=GM_DRUMS["hihat_closed"],
                            velocity=self._rng.randint(60, 80),
                            start_time=beat,
                            duration=0.25,
                            channel=9,
                        )
                    )

            elif style == "four_on_floor" or style == "house":
                # Electronic/dance pattern
                for beat_offset in range(4):
                    notes.append(
                        MIDINote(
                            pitch=GM_DRUMS["kick"],
                            velocity=100,
                            start_time=current_beat + beat_offset,
                            duration=0.5,
                            channel=9,
                        )
                    )

                # Clap on 2 and 4
                notes.append(
                    MIDINote(
                        pitch=GM_DRUMS["clap"],
                        velocity=95,
                        start_time=current_beat + 1,
                        duration=0.5,
                        channel=9,
                    )
                )
                notes.append(
                    MIDINote(
                        pitch=GM_DRUMS["clap"],
                        velocity=95,
                        start_time=current_beat + 3,
                        duration=0.5,
                        channel=9,
                    )
                )

                # Open hi-hat on off-beats
                for beat_offset in range(4):
                    notes.append(
                        MIDINote(
                            pitch=GM_DRUMS["hihat_open"],
                            velocity=70,
                            start_time=current_beat + beat_offset + 0.5,
                            duration=0.25,
                            channel=9,
                        )
                    )

            elif style == "jazz":
                # Jazz swing pattern - ride cymbal focus
                for beat_offset in range(beats_per_bar):
                    # Ride pattern
                    notes.append(
                        MIDINote(
                            pitch=GM_DRUMS["ride"],
                            velocity=80,
                            start_time=current_beat + beat_offset,
                            duration=0.5,
                            channel=9,
                        )
                    )
                    # Skip beat (swung)
                    skip_time = current_beat + beat_offset + 0.66  # Swing feel
                    notes.append(
                        MIDINote(
                            pitch=GM_DRUMS["ride"],
                            velocity=60,
                            start_time=skip_time,
                            duration=0.25,
                            channel=9,
                        )
                    )

                # Hi-hat on 2 and 4 (foot)
                notes.append(
                    MIDINote(
                        pitch=GM_DRUMS["hihat_pedal"],
                        velocity=70,
                        start_time=current_beat + 1,
                        duration=0.25,
                        channel=9,
                    )
                )
                notes.append(
                    MIDINote(
                        pitch=GM_DRUMS["hihat_pedal"],
                        velocity=70,
                        start_time=current_beat + 3,
                        duration=0.25,
                        channel=9,
                    )
                )

                # Sparse kick - not every bar
                if bar % 2 == 0:
                    notes.append(
                        MIDINote(
                            pitch=GM_DRUMS["kick"],
                            velocity=75,
                            start_time=current_beat,
                            duration=0.5,
                            channel=9,
                        )
                    )

            elif style == "lo-fi":
                # Lo-fi hip hop - sparse and dusty
                # Kick pattern (less aggressive)
                kick_pattern = [0, 2.25] if bar % 2 == 0 else [0, 2.5]
                for k in kick_pattern:
                    notes.append(
                        MIDINote(
                            pitch=GM_DRUMS["kick"],
                            velocity=self._rng.randint(75, 90),
                            start_time=current_beat + k,
                            duration=0.5,
                            channel=9,
                        )
                    )

                # Snare/rim on 2 and 4 (alternating)
                snare_pitch = GM_DRUMS["snare"] if bar % 2 == 0 else GM_DRUMS["sidestick"]
                notes.append(
                    MIDINote(
                        pitch=snare_pitch,
                        velocity=self._rng.randint(70, 90),
                        start_time=current_beat + 1,
                        duration=0.5,
                        channel=9,
                    )
                )
                notes.append(
                    MIDINote(
                        pitch=snare_pitch,
                        velocity=self._rng.randint(65, 85),
                        start_time=current_beat + 3,
                        duration=0.5,
                        channel=9,
                    )
                )

                # Swung hi-hats
                for eighth in range(8):
                    beat = current_beat + (eighth * 0.5)
                    if eighth % 2 == 1:
                        beat += 0.15  # Heavy swing
                    # Skip some hits for that lo-fi feel
                    if self._rng.random() < 0.85:
                        notes.append(
                            MIDINote(
                                pitch=GM_DRUMS["hihat_closed"],
                                velocity=self._rng.randint(40, 70),
                                start_time=beat,
                                duration=0.25,
                                channel=9,
                            )
                        )

            current_beat += beats_per_bar

        return MIDITrack(
            name="Drums",
            notes=notes,
            program=0,  # Drums don't use program
            channel=9,  # MIDI channel 10 (0-indexed as 9)
        )

    def _humanize_track(self, track: MIDITrack, amount: float) -> MIDITrack:
        """Add humanization to a track (timing and velocity variation)."""
        humanized_notes = []
        for note in track.notes:
            # Timing variation (in beats)
            timing_offset = self._rng.gauss(0, amount * 0.05)

            # Velocity variation
            velocity_offset = int(self._rng.gauss(0, amount * 10))

            humanized_notes.append(
                MIDINote(
                    pitch=note.pitch,
                    velocity=max(1, min(127, note.velocity + velocity_offset)),
                    start_time=max(0, note.start_time + timing_offset),
                    duration=note.duration,
                    channel=note.channel,
                )
            )

        return MIDITrack(
            name=track.name,
            notes=humanized_notes,
            program=track.program,
            channel=track.channel,
        )

    async def render_to_file(
        self,
        midi_data: MIDIFile,
        output_path: Path,
    ) -> Path:
        """Write MIDI data to file using mido."""
        try:
            import mido
        except ImportError:
            raise ImportError("mido package required. Install with: pip install mido")

        # Create MIDI file
        mid = mido.MidiFile(ticks_per_beat=midi_data.ticks_per_beat)

        # Add tempo track
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(midi_data.tempo_bpm), time=0)
        )
        tempo_track.append(
            mido.MetaMessage(
                "time_signature",
                numerator=midi_data.time_signature[0],
                denominator=midi_data.time_signature[1],
                time=0,
            )
        )

        # Add each track
        for track in midi_data.tracks:
            midi_track = mido.MidiTrack()
            midi_track.name = track.name
            mid.tracks.append(midi_track)

            # Set program
            if track.channel != 9:  # Don't set program for drums
                midi_track.append(
                    mido.Message(
                        "program_change", program=track.program, channel=track.channel, time=0
                    )
                )

            # Sort notes by start time
            sorted_notes = sorted(track.notes, key=lambda n: n.start_time)

            # Convert to delta times and add note events
            events = []
            for note in sorted_notes:
                start_tick = int(note.start_time * midi_data.ticks_per_beat)
                duration_tick = int(note.duration * midi_data.ticks_per_beat)

                events.append((start_tick, "note_on", note.pitch, note.velocity, note.channel))
                events.append((start_tick + duration_tick, "note_off", note.pitch, 0, note.channel))

            # Sort by time
            events.sort(key=lambda e: e[0])

            # Add events with delta times
            current_tick = 0
            for event in events:
                tick, event_type, pitch, velocity, channel = event
                delta = tick - current_tick
                current_tick = tick

                midi_track.append(
                    mido.Message(
                        event_type, note=pitch, velocity=velocity, channel=channel, time=delta
                    )
                )

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mid.save(str(output_path))

        logger.info(f"MIDI saved to {output_path}")
        return output_path

    async def load_from_file(self, path: Path) -> MIDIFile:
        """Load MIDI from file."""
        try:
            import mido
        except ImportError:
            raise ImportError("mido package required. Install with: pip install mido")

        mid = mido.MidiFile(str(path))

        # Extract tempo
        tempo_bpm = 120.0  # Default
        time_sig = (4, 4)  # Default

        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    tempo_bpm = mido.tempo2bpm(msg.tempo)
                elif msg.type == "time_signature":
                    time_sig = (msg.numerator, msg.denominator)

        # Convert tracks
        tracks = []
        for mido_track in mid.tracks:
            notes = []
            current_tick = 0
            active_notes = {}  # pitch -> (start_tick, velocity, channel)

            program = 0
            channel = 0

            for msg in mido_track:
                current_tick += msg.time

                if msg.type == "program_change":
                    program = msg.program
                    channel = msg.channel
                elif msg.type == "note_on" and msg.velocity > 0:
                    key = (msg.note, msg.channel)
                    active_notes[key] = (current_tick, msg.velocity, msg.channel)
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    key = (msg.note, msg.channel)
                    if key in active_notes:
                        start_tick, velocity, ch = active_notes.pop(key)
                        duration_tick = current_tick - start_tick

                        notes.append(
                            MIDINote(
                                pitch=msg.note,
                                velocity=velocity,
                                start_time=start_tick / mid.ticks_per_beat,
                                duration=duration_tick / mid.ticks_per_beat,
                                channel=ch,
                            )
                        )

            if notes:
                tracks.append(
                    MIDITrack(
                        name=mido_track.name or f"Track {len(tracks)}",
                        notes=notes,
                        program=program,
                        channel=channel,
                    )
                )

        return MIDIFile(
            tracks=tracks,
            tempo_bpm=tempo_bpm,
            time_signature=time_sig,
            ticks_per_beat=mid.ticks_per_beat,
        )

    async def transpose(
        self,
        midi_data: MIDIFile,
        semitones: int,
    ) -> MIDIFile:
        """Transpose all non-drum tracks by semitones."""
        transposed_tracks = []
        for track in midi_data.tracks:
            if track.channel == 9:  # Don't transpose drums
                transposed_tracks.append(track)
            else:
                transposed_notes = [
                    MIDINote(
                        pitch=max(0, min(127, note.pitch + semitones)),
                        velocity=note.velocity,
                        start_time=note.start_time,
                        duration=note.duration,
                        channel=note.channel,
                    )
                    for note in track.notes
                ]
                transposed_tracks.append(
                    MIDITrack(
                        name=track.name,
                        notes=transposed_notes,
                        program=track.program,
                        channel=track.channel,
                    )
                )

        return MIDIFile(
            tracks=transposed_tracks,
            tempo_bpm=midi_data.tempo_bpm,
            time_signature=midi_data.time_signature,
            ticks_per_beat=midi_data.ticks_per_beat,
        )

    async def quantize(
        self,
        midi_data: MIDIFile,
        grid: float,
        strength: float = 1.0,
    ) -> MIDIFile:
        """Quantize MIDI to grid."""
        quantized_tracks = []
        for track in midi_data.tracks:
            quantized_notes = []
            for note in track.notes:
                # Find nearest grid point
                nearest_grid = round(note.start_time / grid) * grid
                # Apply quantization with strength
                new_start = note.start_time + (nearest_grid - note.start_time) * strength

                quantized_notes.append(
                    MIDINote(
                        pitch=note.pitch,
                        velocity=note.velocity,
                        start_time=new_start,
                        duration=note.duration,
                        channel=note.channel,
                    )
                )

            quantized_tracks.append(
                MIDITrack(
                    name=track.name,
                    notes=quantized_notes,
                    program=track.program,
                    channel=track.channel,
                )
            )

        return MIDIFile(
            tracks=quantized_tracks,
            tempo_bpm=midi_data.tempo_bpm,
            time_signature=midi_data.time_signature,
            ticks_per_beat=midi_data.ticks_per_beat,
        )


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    "AlgorithmicMIDIProvider",
    "note_name_to_midi",
    "midi_to_note_name",
    "parse_chord",
    "get_chord_notes",
    "get_scale_notes",
    "snap_to_scale",
    "CHORD_INTERVALS",
    "SCALE_INTERVALS",
    "GM_DRUMS",
    "GM_INSTRUMENTS",
    "GENRE_INSTRUMENTS",
]
