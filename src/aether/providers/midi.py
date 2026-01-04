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
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aether.providers.base import (
    MIDIProvider,
    MIDIFile,
    MIDITrack,
    MIDINote,
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
}

# GM drum map
GM_DRUMS = {
    "kick": 36,
    "snare": 38,
    "hihat_closed": 42,
    "hihat_open": 46,
    "tom_low": 45,
    "tom_mid": 47,
    "tom_high": 50,
    "crash": 49,
    "ride": 51,
    "clap": 39,
    "rim": 37,
    "cowbell": 56,
}


# ============================================================================
# Helper Functions
# ============================================================================


def note_name_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number."""
    note = note.upper().replace("♯", "#").replace("♭", "b")

    # Handle flats by converting to sharps
    flat_to_sharp = {"DB": "C#", "EB": "D#", "FB": "E", "GB": "F#", "AB": "G#", "BB": "A#", "CB": "B"}
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


def midi_to_note_name(midi_note: int) -> Tuple[str, int]:
    """Convert MIDI number to note name and octave."""
    octave = (midi_note // 12) - 1
    note_index = midi_note % 12
    return NOTE_NAMES[note_index], octave


def parse_chord(chord_str: str) -> Tuple[int, str, int]:
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
    elif quality_str in ["maj7", "ma7", "△7"]:
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
    else:
        quality = "major"

    if bass_note is None:
        bass_note = root_midi - 12  # Default bass is root an octave lower

    return root_midi, quality, bass_note


def get_chord_notes(root: int, quality: str, inversion: int = 0) -> List[int]:
    """Get MIDI notes for a chord."""
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["major"])
    notes = [root + interval for interval in intervals]

    # Apply inversion
    for _ in range(inversion % len(notes)):
        notes[0] += 12
        notes = notes[1:] + [notes[0]]

    return notes


def get_scale_notes(root: int, scale: str, octave_range: int = 2) -> List[int]:
    """Get MIDI notes for a scale across octaves."""
    intervals = SCALE_INTERVALS.get(scale, SCALE_INTERVALS["major"])
    notes = []
    for oct in range(octave_range):
        for interval in intervals:
            notes.append(root + interval + (oct * 12))
    return notes


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
    - Melodic contour generation
    - Drum pattern generation
    - Bass line generation
    - Humanization (timing/velocity variation)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._rng = random.Random()

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Algorithmic MIDI Provider",
            version="1.0.0",
            provider_type="midi",
            status=self._status,
            capabilities=[
                "chord_generation",
                "melody_generation",
                "drum_patterns",
                "bass_lines",
                "humanization",
                "transposition",
                "quantization",
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

        progression = harmony_spec.get("progression", ["C", "Am", "F", "G"])
        key_root = harmony_spec.get("key_root", "C")
        mode = harmony_spec.get("mode", "major")

        sections = arrangement_spec.get("sections", ["verse", "chorus"])
        bars_per_section = arrangement_spec.get("bars_per_section", 8)

        # Generate seed if provided for reproducibility
        seed = rhythm_spec.get("seed")
        if seed is not None:
            self._rng.seed(seed)

        tracks = []

        # Generate chord track
        chord_track = self._generate_chord_track(
            progression=progression,
            bars=len(sections) * bars_per_section,
            bars_per_chord=bars_per_section // len(progression),
            time_sig=time_sig,
        )
        tracks.append(chord_track)

        # Generate bass track
        bass_track = self._generate_bass_track(
            progression=progression,
            bars=len(sections) * bars_per_section,
            bars_per_chord=bars_per_section // len(progression),
            time_sig=time_sig,
            style=rhythm_spec.get("bass_style", "root_fifth"),
        )
        tracks.append(bass_track)

        # Generate melody track
        melody_track = self._generate_melody_track(
            key_root=key_root,
            mode=mode,
            bars=len(sections) * bars_per_section,
            time_sig=time_sig,
            contour=melody_spec.get("contour", "arch"),
            density=melody_spec.get("density", 0.5),
        )
        tracks.append(melody_track)

        # Generate drum track
        drum_track = self._generate_drum_track(
            bars=len(sections) * bars_per_section,
            time_sig=time_sig,
            style=rhythm_spec.get("drum_style", "standard"),
            swing=swing,
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
        progression: List[str],
        bars: int,
        bars_per_chord: int,
        time_sig: tuple,
    ) -> MIDITrack:
        """Generate chord track."""
        notes = []
        beats_per_bar = time_sig[0]
        current_beat = 0

        for bar in range(bars):
            chord_idx = (bar // bars_per_chord) % len(progression)
            chord_str = progression[chord_idx]
            root, quality, _ = parse_chord(chord_str)

            # Get chord notes
            chord_notes = get_chord_notes(root, quality, inversion=bar % 3)

            # Add chord
            duration = beats_per_bar * bars_per_chord if bar % bars_per_chord == 0 else 0
            if duration > 0:
                for pitch in chord_notes:
                    notes.append(MIDINote(
                        pitch=pitch,
                        velocity=self._rng.randint(70, 90),
                        start_time=current_beat,
                        duration=duration - 0.5,
                        channel=0,
                    ))

            current_beat += beats_per_bar

        return MIDITrack(
            name="Chords",
            notes=notes,
            program=4,  # Electric Piano
            channel=0,
        )

    def _generate_bass_track(
        self,
        progression: List[str],
        bars: int,
        bars_per_chord: int,
        time_sig: tuple,
        style: str = "root_fifth",
    ) -> MIDITrack:
        """Generate bass track."""
        notes = []
        beats_per_bar = time_sig[0]
        current_beat = 0

        for bar in range(bars):
            chord_idx = (bar // bars_per_chord) % len(progression)
            chord_str = progression[chord_idx]
            root, quality, bass = parse_chord(chord_str)

            # Bass is 2 octaves below chord root
            bass_root = root - 24

            if style == "root_fifth":
                # Root on beat 1, fifth on beat 3
                notes.append(MIDINote(
                    pitch=bass_root,
                    velocity=self._rng.randint(80, 100),
                    start_time=current_beat,
                    duration=1.5,
                    channel=1,
                ))
                if beats_per_bar >= 4:
                    notes.append(MIDINote(
                        pitch=bass_root + 7,  # Fifth
                        velocity=self._rng.randint(70, 90),
                        start_time=current_beat + 2,
                        duration=1.5,
                        channel=1,
                    ))
            elif style == "walking":
                # Walking bass line
                scale = get_scale_notes(bass_root, "major" if quality == "major" else "minor", 1)
                for beat in range(beats_per_bar):
                    pitch = self._rng.choice(scale)
                    notes.append(MIDINote(
                        pitch=pitch,
                        velocity=self._rng.randint(75, 95),
                        start_time=current_beat + beat,
                        duration=0.9,
                        channel=1,
                    ))
            else:  # Simple root
                notes.append(MIDINote(
                    pitch=bass_root,
                    velocity=self._rng.randint(85, 100),
                    start_time=current_beat,
                    duration=beats_per_bar - 0.5,
                    channel=1,
                ))

            current_beat += beats_per_bar

        return MIDITrack(
            name="Bass",
            notes=notes,
            program=33,  # Finger Bass
            channel=1,
        )

    def _generate_melody_track(
        self,
        key_root: str,
        mode: str,
        bars: int,
        time_sig: tuple,
        contour: str = "arch",
        density: float = 0.5,
    ) -> MIDITrack:
        """Generate melody track."""
        notes = []
        beats_per_bar = time_sig[0]
        total_beats = bars * beats_per_bar

        # Get scale
        root_midi = note_name_to_midi(key_root, 5)  # Melody in octave 5
        scale = get_scale_notes(root_midi, mode, 2)

        # Generate melody based on contour
        current_beat = 0
        current_pitch_idx = len(scale) // 2  # Start in middle of scale

        while current_beat < total_beats:
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
                    import math
                    target_height = 0.5 + 0.5 * math.sin(progress * math.pi * 4)
                else:
                    target_height = 0.5

                # Move pitch toward target
                target_idx = int(target_height * (len(scale) - 1))
                if current_pitch_idx < target_idx:
                    current_pitch_idx += self._rng.choice([0, 1, 2])
                elif current_pitch_idx > target_idx:
                    current_pitch_idx -= self._rng.choice([0, 1, 2])
                current_pitch_idx = max(0, min(len(scale) - 1, current_pitch_idx))

                # Determine note duration
                durations = [0.5, 1.0, 1.5, 2.0]
                duration = self._rng.choice(durations)

                notes.append(MIDINote(
                    pitch=scale[current_pitch_idx],
                    velocity=self._rng.randint(70, 100),
                    start_time=current_beat,
                    duration=duration * 0.9,
                    channel=2,
                ))

                current_beat += duration
            else:
                current_beat += 0.5  # Rest

        return MIDITrack(
            name="Melody",
            notes=notes,
            program=0,  # Acoustic Grand Piano
            channel=2,
        )

    def _generate_drum_track(
        self,
        bars: int,
        time_sig: tuple,
        style: str = "standard",
        swing: float = 0.0,
    ) -> MIDITrack:
        """Generate drum track."""
        notes = []
        beats_per_bar = time_sig[0]
        current_beat = 0

        for bar in range(bars):
            if style == "standard":
                # Standard rock/pop pattern
                # Kick on 1 and 3
                notes.append(MIDINote(pitch=GM_DRUMS["kick"], velocity=100, start_time=current_beat, duration=0.5, channel=9))
                notes.append(MIDINote(pitch=GM_DRUMS["kick"], velocity=90, start_time=current_beat + 2, duration=0.5, channel=9))

                # Snare on 2 and 4
                notes.append(MIDINote(pitch=GM_DRUMS["snare"], velocity=100, start_time=current_beat + 1, duration=0.5, channel=9))
                notes.append(MIDINote(pitch=GM_DRUMS["snare"], velocity=95, start_time=current_beat + 3, duration=0.5, channel=9))

                # Hi-hat on every 8th
                for eighth in range(8):
                    beat = current_beat + (eighth * 0.5)
                    # Apply swing
                    if eighth % 2 == 1 and swing > 0:
                        beat += swing * 0.5
                    velocity = 80 if eighth % 2 == 0 else 60
                    notes.append(MIDINote(pitch=GM_DRUMS["hihat_closed"], velocity=velocity, start_time=beat, duration=0.25, channel=9))

            elif style == "boom_bap":
                # Hip-hop boom bap pattern
                # Kick pattern
                kick_pattern = [0, 0.75, 2.5]
                for k in kick_pattern:
                    notes.append(MIDINote(pitch=GM_DRUMS["kick"], velocity=100, start_time=current_beat + k, duration=0.5, channel=9))

                # Snare on 2 and 4
                notes.append(MIDINote(pitch=GM_DRUMS["snare"], velocity=100, start_time=current_beat + 1, duration=0.5, channel=9))
                notes.append(MIDINote(pitch=GM_DRUMS["snare"], velocity=95, start_time=current_beat + 3, duration=0.5, channel=9))

                # Hi-hat pattern (with swing)
                for eighth in range(8):
                    beat = current_beat + (eighth * 0.5)
                    if eighth % 2 == 1:
                        beat += swing * 0.5
                    notes.append(MIDINote(pitch=GM_DRUMS["hihat_closed"], velocity=self._rng.randint(60, 80), start_time=beat, duration=0.25, channel=9))

            elif style == "four_on_floor":
                # Electronic/dance pattern
                for beat_offset in range(4):
                    notes.append(MIDINote(pitch=GM_DRUMS["kick"], velocity=100, start_time=current_beat + beat_offset, duration=0.5, channel=9))

                # Snare/clap on 2 and 4
                notes.append(MIDINote(pitch=GM_DRUMS["clap"], velocity=95, start_time=current_beat + 1, duration=0.5, channel=9))
                notes.append(MIDINote(pitch=GM_DRUMS["clap"], velocity=95, start_time=current_beat + 3, duration=0.5, channel=9))

                # Open hi-hat on off-beats
                for beat_offset in range(4):
                    notes.append(MIDINote(pitch=GM_DRUMS["hihat_open"], velocity=70, start_time=current_beat + beat_offset + 0.5, duration=0.25, channel=9))

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

            humanized_notes.append(MIDINote(
                pitch=note.pitch,
                velocity=max(1, min(127, note.velocity + velocity_offset)),
                start_time=max(0, note.start_time + timing_offset),
                duration=note.duration,
                channel=note.channel,
            ))

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
        tempo_track.append(mido.MetaMessage(
            'set_tempo',
            tempo=mido.bpm2tempo(midi_data.tempo_bpm),
            time=0
        ))
        tempo_track.append(mido.MetaMessage(
            'time_signature',
            numerator=midi_data.time_signature[0],
            denominator=midi_data.time_signature[1],
            time=0
        ))

        # Add each track
        for track in midi_data.tracks:
            midi_track = mido.MidiTrack()
            midi_track.name = track.name
            mid.tracks.append(midi_track)

            # Set program
            if track.channel != 9:  # Don't set program for drums
                midi_track.append(mido.Message(
                    'program_change',
                    program=track.program,
                    channel=track.channel,
                    time=0
                ))

            # Sort notes by start time
            sorted_notes = sorted(track.notes, key=lambda n: n.start_time)

            # Convert to delta times and add note events
            events = []
            for note in sorted_notes:
                start_tick = int(note.start_time * midi_data.ticks_per_beat)
                duration_tick = int(note.duration * midi_data.ticks_per_beat)

                events.append((start_tick, 'note_on', note.pitch, note.velocity, note.channel))
                events.append((start_tick + duration_tick, 'note_off', note.pitch, 0, note.channel))

            # Sort by time
            events.sort(key=lambda e: e[0])

            # Add events with delta times
            current_tick = 0
            for event in events:
                tick, event_type, pitch, velocity, channel = event
                delta = tick - current_tick
                current_tick = tick

                midi_track.append(mido.Message(
                    event_type,
                    note=pitch,
                    velocity=velocity,
                    channel=channel,
                    time=delta
                ))

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
                if msg.type == 'set_tempo':
                    tempo_bpm = mido.tempo2bpm(msg.tempo)
                elif msg.type == 'time_signature':
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

                if msg.type == 'program_change':
                    program = msg.program
                    channel = msg.channel
                elif msg.type == 'note_on' and msg.velocity > 0:
                    key = (msg.note, msg.channel)
                    active_notes[key] = (current_tick, msg.velocity, msg.channel)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (msg.note, msg.channel)
                    if key in active_notes:
                        start_tick, velocity, ch = active_notes.pop(key)
                        duration_tick = current_tick - start_tick

                        notes.append(MIDINote(
                            pitch=msg.note,
                            velocity=velocity,
                            start_time=start_tick / mid.ticks_per_beat,
                            duration=duration_tick / mid.ticks_per_beat,
                            channel=ch,
                        ))

            if notes:
                tracks.append(MIDITrack(
                    name=mido_track.name or f"Track {len(tracks)}",
                    notes=notes,
                    program=program,
                    channel=channel,
                ))

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
                transposed_tracks.append(MIDITrack(
                    name=track.name,
                    notes=transposed_notes,
                    program=track.program,
                    channel=track.channel,
                ))

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

                quantized_notes.append(MIDINote(
                    pitch=note.pitch,
                    velocity=note.velocity,
                    start_time=new_start,
                    duration=note.duration,
                    channel=note.channel,
                ))

            quantized_tracks.append(MIDITrack(
                name=track.name,
                notes=quantized_notes,
                program=track.program,
                channel=track.channel,
            ))

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
    "CHORD_INTERVALS",
    "SCALE_INTERVALS",
    "GM_DRUMS",
]
