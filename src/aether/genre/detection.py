"""
Genre Detection - Analyzes MIDI/audio for genre classification.

Components:
- RhythmPatternMatcher: Detects kick patterns, swing, syncopation
- HarmonicAnalyzer: Analyzes modes, chord progressions, cadences
- GenreDetector: Ensemble classifier combining multiple signals
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import math

from aether.genre.dna import (
    GenreDNA,
    KickPattern,
    SnarePosition,
    TimeFeel,
    GENRE_DNA_LIBRARY,
    get_genre_dna,
)


# ============================================================================
# Rhythm Analysis
# ============================================================================

@dataclass
class RhythmFeatures:
    """Extracted rhythm features from MIDI/audio."""
    # Timing
    tempo_bpm: float
    swing_amount: float  # 0-1, detected swing
    syncopation_level: float  # 0-1

    # Pattern detection
    detected_kick_pattern: Optional[KickPattern] = None
    detected_snare_position: Optional[SnarePosition] = None
    detected_time_feel: Optional[TimeFeel] = None

    # Pattern confidence
    kick_confidence: float = 0.0
    snare_confidence: float = 0.0
    feel_confidence: float = 0.0

    # Additional metrics
    hat_density: float = 0.0  # Hi-hat notes per beat
    ghost_note_ratio: float = 0.0  # Low-velocity notes ratio
    downbeat_emphasis: float = 0.0  # Emphasis on beat 1

    def to_vector(self) -> list[float]:
        """Convert to feature vector for similarity computation."""
        # Normalize tempo to 0-1 range (60-200 BPM)
        tempo_norm = (self.tempo_bpm - 60) / 140

        # Encode patterns as numeric
        kick_map = {p: i / len(KickPattern) for i, p in enumerate(KickPattern)}
        snare_map = {p: i / len(SnarePosition) for i, p in enumerate(SnarePosition)}
        feel_map = {p: i / len(TimeFeel) for i, p in enumerate(TimeFeel)}

        kick_val = kick_map.get(self.detected_kick_pattern, 0.5)
        snare_val = snare_map.get(self.detected_snare_position, 0.5)
        feel_val = feel_map.get(self.detected_time_feel, 0.5)

        return [
            tempo_norm,
            self.swing_amount,
            self.syncopation_level,
            kick_val,
            snare_val,
            feel_val,
            self.hat_density / 4,  # Normalize to 0-1 (4 hats/beat = max)
            self.ghost_note_ratio,
            self.downbeat_emphasis,
        ]


class RhythmPatternMatcher:
    """
    Detects rhythm patterns from MIDI note data.

    Analyzes:
    - Kick drum patterns (4-on-floor, boom-bap, trap, etc.)
    - Snare placement (backbeat, off-beat, syncopated)
    - Time feel (straight, shuffle, swing)
    - Syncopation and ghost notes
    """

    # GM Drum Map (Channel 10)
    KICK_NOTES = {35, 36}  # Acoustic/Electric bass drum
    SNARE_NOTES = {38, 40}  # Acoustic/Electric snare
    HIHAT_CLOSED = {42}
    HIHAT_OPEN = {46}
    HIHAT_PEDAL = {44}
    CLAP = {39}
    RIM = {37}

    def __init__(self):
        self._pattern_templates = self._build_pattern_templates()

    def _build_pattern_templates(self) -> dict:
        """Build reference patterns for each kick pattern type."""
        # 16-step patterns (one bar at 4/4)
        return {
            KickPattern.FOUR_ON_FLOOR: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            KickPattern.BOOM_BAP: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            KickPattern.TRAP_808: [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            KickPattern.BREAKBEAT: [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
            KickPattern.DEMBOW: [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            KickPattern.HALFTIME: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            KickPattern.SHUFFLE: [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            KickPattern.LINEAR: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            KickPattern.SYNCOPATED: [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            KickPattern.SPARSE: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }

    def analyze(
        self,
        notes: list[dict],
        tempo_bpm: float,
        ticks_per_beat: int = 480,
    ) -> RhythmFeatures:
        """
        Analyze MIDI notes to extract rhythm features.

        Args:
            notes: List of note dicts with {pitch, start_tick, duration, velocity, channel}
            tempo_bpm: Track tempo
            ticks_per_beat: MIDI resolution

        Returns:
            RhythmFeatures with detected patterns
        """
        # Separate drum notes (channel 9/10)
        drum_notes = [n for n in notes if n.get("channel", 0) in (9, 10)]

        if not drum_notes:
            # No drums, return default features
            return RhythmFeatures(
                tempo_bpm=tempo_bpm,
                swing_amount=0.0,
                syncopation_level=0.0,
            )

        # Extract patterns
        kick_pattern = self._extract_pattern(drum_notes, self.KICK_NOTES, ticks_per_beat)
        snare_pattern = self._extract_pattern(drum_notes, self.SNARE_NOTES, ticks_per_beat)
        hihat_pattern = self._extract_pattern(
            drum_notes,
            self.HIHAT_CLOSED | self.HIHAT_OPEN | self.HIHAT_PEDAL,
            ticks_per_beat
        )

        # Detect kick pattern type
        detected_kick, kick_confidence = self._match_kick_pattern(kick_pattern)

        # Detect snare position
        detected_snare, snare_confidence = self._detect_snare_position(snare_pattern)

        # Detect swing from hi-hat timing
        swing_amount = self._detect_swing(drum_notes, ticks_per_beat)

        # Detect time feel
        detected_feel, feel_confidence = self._detect_time_feel(
            drum_notes, swing_amount, ticks_per_beat
        )

        # Calculate syncopation
        syncopation = self._calculate_syncopation(kick_pattern, snare_pattern)

        # Hi-hat density
        hat_density = self._calculate_hat_density(hihat_pattern)

        # Ghost note ratio
        ghost_ratio = self._calculate_ghost_ratio(drum_notes)

        # Downbeat emphasis
        downbeat = self._calculate_downbeat_emphasis(drum_notes, ticks_per_beat)

        return RhythmFeatures(
            tempo_bpm=tempo_bpm,
            swing_amount=swing_amount,
            syncopation_level=syncopation,
            detected_kick_pattern=detected_kick,
            detected_snare_position=detected_snare,
            detected_time_feel=detected_feel,
            kick_confidence=kick_confidence,
            snare_confidence=snare_confidence,
            feel_confidence=feel_confidence,
            hat_density=hat_density,
            ghost_note_ratio=ghost_ratio,
            downbeat_emphasis=downbeat,
        )

    def _extract_pattern(
        self,
        notes: list[dict],
        pitch_set: set[int],
        ticks_per_beat: int,
    ) -> list[float]:
        """Extract 16-step pattern for given pitches."""
        # 16 steps per bar (16th notes)
        ticks_per_step = ticks_per_beat // 4
        bar_length = ticks_per_beat * 4

        pattern = [0.0] * 16

        for note in notes:
            if note["pitch"] in pitch_set:
                # Find position in bar
                pos_in_bar = note["start_tick"] % bar_length
                step = int(pos_in_bar / ticks_per_step) % 16

                # Accumulate velocity
                velocity = note.get("velocity", 100) / 127.0
                pattern[step] = max(pattern[step], velocity)

        return pattern

    def _match_kick_pattern(
        self,
        pattern: list[float],
    ) -> tuple[Optional[KickPattern], float]:
        """Match extracted pattern to known kick patterns."""
        best_match = None
        best_score = 0.0

        # Binarize pattern
        binary = [1 if v > 0.3 else 0 for v in pattern]

        for kick_type, template in self._pattern_templates.items():
            # Correlation score
            score = self._pattern_correlation(binary, template)
            if score > best_score:
                best_score = score
                best_match = kick_type

        return best_match, best_score

    def _pattern_correlation(self, pattern: list[int], template: list[int]) -> float:
        """Calculate correlation between two binary patterns."""
        if len(pattern) != len(template):
            return 0.0

        matches = sum(1 for p, t in zip(pattern, template) if p == t)
        return matches / len(pattern)

    def _detect_snare_position(
        self,
        pattern: list[float],
    ) -> tuple[Optional[SnarePosition], float]:
        """Detect snare placement from pattern."""
        # Check standard positions
        backbeat = pattern[4] + pattern[12]  # Steps 5 and 13 (beats 2 and 4)
        offbeat = sum(pattern[i] for i in [2, 6, 10, 14])  # Offbeat 16ths
        onbeat = pattern[0] + pattern[8]  # Beats 1 and 3

        # Check for syncopated patterns
        syncopated = sum(pattern[i] for i in [3, 7, 11, 15])  # Just before beats

        scores = {
            SnarePosition.BACKBEAT: backbeat,
            SnarePosition.OFFBEAT: offbeat,
            SnarePosition.ON_THREE: pattern[8],
            SnarePosition.HALFTIME_THREE: pattern[8] if sum(pattern) < 3 else 0,
            SnarePosition.SYNCOPATED: syncopated,
        }

        best_pos = max(scores, key=scores.get)
        confidence = scores[best_pos] / max(2.0, sum(pattern))

        return best_pos, min(1.0, confidence)

    def _detect_swing(
        self,
        notes: list[dict],
        ticks_per_beat: int,
    ) -> float:
        """Detect swing amount from timing deviations."""
        # Look at 8th note subdivisions
        ticks_per_8th = ticks_per_beat // 2

        offbeat_delays = []

        for note in notes:
            pos_in_beat = note["start_tick"] % ticks_per_beat

            # Check if note is near offbeat position
            expected_offbeat = ticks_per_8th
            deviation = pos_in_beat - expected_offbeat

            # If within range of an 8th note
            if abs(deviation) < ticks_per_8th * 0.4:
                # Positive deviation = swing
                offbeat_delays.append(deviation / ticks_per_8th)

        if not offbeat_delays:
            return 0.0

        # Average swing amount (clamped to 0-1)
        avg_swing = sum(offbeat_delays) / len(offbeat_delays)
        return max(0.0, min(1.0, avg_swing * 2 + 0.5))

    def _detect_time_feel(
        self,
        notes: list[dict],
        swing_amount: float,
        ticks_per_beat: int,
    ) -> tuple[Optional[TimeFeel], float]:
        """Detect overall time feel."""
        # Count notes per subdivision
        ticks_per_16th = ticks_per_beat // 4
        ticks_per_triplet = ticks_per_beat // 3

        sixteenth_hits = 0
        triplet_hits = 0
        total_notes = len(notes)

        for note in notes:
            pos = note["start_tick"] % ticks_per_beat

            # Check 16th grid alignment
            if pos % ticks_per_16th < ticks_per_16th * 0.2:
                sixteenth_hits += 1

            # Check triplet grid alignment
            if pos % ticks_per_triplet < ticks_per_triplet * 0.2:
                triplet_hits += 1

        if total_notes == 0:
            return TimeFeel.STRAIGHT, 0.5

        sixteenth_ratio = sixteenth_hits / total_notes
        triplet_ratio = triplet_hits / total_notes

        # Determine feel
        if triplet_ratio > 0.6:
            return TimeFeel.TRIPLET, triplet_ratio
        elif swing_amount > 0.3:
            if swing_amount > 0.5:
                return TimeFeel.SWING, swing_amount
            else:
                return TimeFeel.SHUFFLE, swing_amount
        elif sixteenth_ratio > 0.8:
            # Check for half-time feel based on kick/snare spacing
            return TimeFeel.STRAIGHT, sixteenth_ratio
        else:
            return TimeFeel.STRAIGHT, 0.6

    def _calculate_syncopation(
        self,
        kick_pattern: list[float],
        snare_pattern: list[float],
    ) -> float:
        """Calculate syncopation level from patterns."""
        # Notes on weak positions indicate syncopation
        strong_positions = {0, 4, 8, 12}  # Quarter note positions
        weak_positions = {2, 6, 10, 14}  # Offbeat 8ths
        very_weak = {1, 3, 5, 7, 9, 11, 13, 15}  # 16th note positions

        syncopation_score = 0.0

        for i, (k, s) in enumerate(zip(kick_pattern, snare_pattern)):
            hit = max(k, s)
            if i in very_weak:
                syncopation_score += hit * 1.0
            elif i in weak_positions:
                syncopation_score += hit * 0.5

        # Normalize to 0-1
        return min(1.0, syncopation_score / 8.0)

    def _calculate_hat_density(self, pattern: list[float]) -> float:
        """Calculate hi-hat density (hits per beat)."""
        hits = sum(1 for v in pattern if v > 0.2)
        return hits / 4.0  # Pattern is one bar = 4 beats

    def _calculate_ghost_ratio(self, notes: list[dict]) -> float:
        """Calculate ratio of ghost notes (low velocity)."""
        if not notes:
            return 0.0

        ghost_threshold = 60
        ghost_count = sum(1 for n in notes if n.get("velocity", 100) < ghost_threshold)
        return ghost_count / len(notes)

    def _calculate_downbeat_emphasis(
        self,
        notes: list[dict],
        ticks_per_beat: int,
    ) -> float:
        """Calculate emphasis on downbeat (beat 1)."""
        bar_length = ticks_per_beat * 4
        downbeat_window = ticks_per_beat // 8  # Allow slight timing variation

        total_velocity = 0
        downbeat_velocity = 0

        for note in notes:
            velocity = note.get("velocity", 100)
            total_velocity += velocity

            pos_in_bar = note["start_tick"] % bar_length
            if pos_in_bar < downbeat_window:
                downbeat_velocity += velocity

        if total_velocity == 0:
            return 0.0

        return downbeat_velocity / total_velocity


# ============================================================================
# Harmonic Analysis
# ============================================================================

@dataclass
class HarmonicFeatures:
    """Extracted harmonic features from MIDI."""
    # Mode detection
    detected_mode: str
    mode_confidence: float

    # Chord analysis
    chord_complexity: float  # 0-1
    detected_progressions: list[str]
    progression_confidence: float

    # Cadence detection
    detected_cadences: list[str]

    # Additional metrics
    chromaticism: float  # 0-1, non-diatonic note ratio
    tension_level: float  # 0-1
    root_motion: list[int]  # Intervals between chord roots

    def to_vector(self) -> list[float]:
        """Convert to feature vector."""
        mode_map = {
            "major": 0.0, "minor": 0.2, "dorian": 0.3, "mixolydian": 0.4,
            "phrygian": 0.5, "lydian": 0.6, "locrian": 0.7, "pentatonic": 0.8,
        }
        mode_val = mode_map.get(self.detected_mode, 0.5)

        return [
            mode_val,
            self.mode_confidence,
            self.chord_complexity,
            self.progression_confidence,
            self.chromaticism,
            self.tension_level,
            len(self.detected_cadences) / 4,  # Normalize
        ]


class HarmonicAnalyzer:
    """
    Analyzes harmonic content from MIDI for genre classification.

    Detects:
    - Mode/scale (major, minor, dorian, etc.)
    - Chord complexity (triads vs 7ths vs extended)
    - Common progressions (I-IV-V, ii-V-I, etc.)
    - Cadence types (perfect, plagal, deceptive)
    """

    # Pitch class profiles for modes (C as root)
    MODE_PROFILES = {
        "major": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # Ionian
        "minor": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],  # Aeolian
        "dorian": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        "phrygian": [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        "lydian": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        "mixolydian": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        "locrian": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        "pentatonic": [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],  # Major pentatonic
        "blues": [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    }

    # Common chord progressions
    PROGRESSIONS = {
        "I-IV-V-I": [0, 5, 7, 0],  # Classic
        "I-V-vi-IV": [0, 7, 9, 5],  # Pop
        "ii-V-I": [2, 7, 0],  # Jazz
        "I-vi-IV-V": [0, 9, 5, 7],  # 50s
        "i-VII-VI-VII": [0, 10, 8, 10],  # Minor
        "i-iv-VII-III": [0, 5, 10, 3],  # Minor variation
        "I-V-vi-iii-IV-I-IV-V": [0, 7, 9, 4, 5, 0, 5, 7],  # Canon
        "i-VI-III-VII": [0, 8, 3, 10],  # Trap/lofi
    }

    def __init__(self):
        pass

    def analyze(
        self,
        notes: list[dict],
        ticks_per_beat: int = 480,
    ) -> HarmonicFeatures:
        """
        Analyze MIDI notes for harmonic content.

        Args:
            notes: List of note dicts {pitch, start_tick, duration, velocity}
            ticks_per_beat: MIDI resolution

        Returns:
            HarmonicFeatures with detected harmony
        """
        # Filter out drum notes
        melodic_notes = [n for n in notes if n.get("channel", 0) not in (9, 10)]

        if not melodic_notes:
            return HarmonicFeatures(
                detected_mode="minor",
                mode_confidence=0.0,
                chord_complexity=0.5,
                detected_progressions=[],
                progression_confidence=0.0,
                detected_cadences=[],
                chromaticism=0.0,
                tension_level=0.5,
                root_motion=[],
            )

        # Detect mode
        mode, mode_conf, key_root = self._detect_mode(melodic_notes)

        # Extract chords
        chords = self._extract_chords(melodic_notes, ticks_per_beat)

        # Analyze chord complexity
        complexity = self._analyze_complexity(chords)

        # Detect progressions
        progressions, prog_conf = self._detect_progressions(chords, key_root)

        # Detect cadences
        cadences = self._detect_cadences(chords, key_root)

        # Calculate chromaticism
        chromaticism = self._calculate_chromaticism(melodic_notes, mode, key_root)

        # Calculate tension
        tension = self._calculate_tension(chords)

        # Root motion
        root_motion = self._extract_root_motion(chords)

        return HarmonicFeatures(
            detected_mode=mode,
            mode_confidence=mode_conf,
            chord_complexity=complexity,
            detected_progressions=progressions,
            progression_confidence=prog_conf,
            detected_cadences=cadences,
            chromaticism=chromaticism,
            tension_level=tension,
            root_motion=root_motion,
        )

    def _detect_mode(
        self,
        notes: list[dict],
    ) -> tuple[str, float, int]:
        """Detect mode using pitch class histogram."""
        # Build pitch class histogram
        histogram = [0.0] * 12

        for note in notes:
            pc = note["pitch"] % 12
            weight = note.get("velocity", 100) / 127.0
            weight *= note.get("duration", 480) / 480.0  # Weight by duration
            histogram[pc] += weight

        # Normalize
        total = sum(histogram)
        if total > 0:
            histogram = [h / total for h in histogram]

        # Try each possible root and mode
        best_mode = "minor"
        best_root = 0
        best_score = 0.0

        for root in range(12):
            for mode_name, profile in self.MODE_PROFILES.items():
                # Rotate profile to match root
                rotated = profile[-root:] + profile[:-root]

                # Correlation
                score = sum(h * p for h, p in zip(histogram, rotated))

                if score > best_score:
                    best_score = score
                    best_mode = mode_name
                    best_root = root

        return best_mode, min(1.0, best_score * 1.5), best_root

    def _extract_chords(
        self,
        notes: list[dict],
        ticks_per_beat: int,
    ) -> list[dict]:
        """Extract chord events from notes."""
        # Group notes by time window (1 beat)
        window_size = ticks_per_beat

        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n["start_tick"])

        if not sorted_notes:
            return []

        chords = []
        current_window_start = sorted_notes[0]["start_tick"]
        current_pitches = set()
        current_velocities = []

        for note in sorted_notes:
            if note["start_tick"] < current_window_start + window_size:
                current_pitches.add(note["pitch"] % 12)
                current_velocities.append(note.get("velocity", 100))
            else:
                # Save current chord
                if len(current_pitches) >= 2:
                    chords.append({
                        "pitches": sorted(current_pitches),
                        "start": current_window_start,
                        "velocity": sum(current_velocities) / len(current_velocities),
                    })

                # Start new window
                current_window_start = note["start_tick"]
                current_pitches = {note["pitch"] % 12}
                current_velocities = [note.get("velocity", 100)]

        # Add final chord
        if len(current_pitches) >= 2:
            chords.append({
                "pitches": sorted(current_pitches),
                "start": current_window_start,
                "velocity": sum(current_velocities) / len(current_velocities),
            })

        return chords

    def _analyze_complexity(self, chords: list[dict]) -> float:
        """Analyze chord complexity (triads vs 7ths vs extended)."""
        if not chords:
            return 0.5

        total_score = 0.0

        for chord in chords:
            n_notes = len(chord["pitches"])
            if n_notes <= 3:
                total_score += 0.3  # Triad
            elif n_notes == 4:
                total_score += 0.6  # 7th chord
            else:
                total_score += 0.9  # Extended

        return total_score / len(chords)

    def _detect_progressions(
        self,
        chords: list[dict],
        key_root: int,
    ) -> tuple[list[str], float]:
        """Detect common chord progressions."""
        if len(chords) < 3:
            return [], 0.0

        # Extract root motion (relative to key)
        roots = []
        for chord in chords:
            # Guess root as lowest pitch class
            root = min(chord["pitches"])
            roots.append((root - key_root) % 12)

        detected = []
        total_confidence = 0.0

        for prog_name, prog_pattern in self.PROGRESSIONS.items():
            # Sliding window match
            for i in range(len(roots) - len(prog_pattern) + 1):
                window = roots[i:i + len(prog_pattern)]

                # Check match (allow transposition)
                if window == prog_pattern:
                    if prog_name not in detected:
                        detected.append(prog_name)
                        total_confidence += 1.0

        confidence = min(1.0, total_confidence / 2) if detected else 0.0
        return detected, confidence

    def _detect_cadences(
        self,
        chords: list[dict],
        key_root: int,
    ) -> list[str]:
        """Detect cadence types at phrase endings."""
        if len(chords) < 2:
            return []

        cadences = []

        # Check final two chords
        final_root = min(chords[-1]["pitches"]) if chords[-1]["pitches"] else 0
        penult_root = min(chords[-2]["pitches"]) if chords[-2]["pitches"] else 0

        final_degree = (final_root - key_root) % 12
        penult_degree = (penult_root - key_root) % 12

        # Perfect cadence: V -> I
        if penult_degree == 7 and final_degree == 0:
            cadences.append("perfect")

        # Plagal cadence: IV -> I
        if penult_degree == 5 and final_degree == 0:
            cadences.append("plagal")

        # Deceptive cadence: V -> vi
        if penult_degree == 7 and final_degree == 9:
            cadences.append("deceptive")

        # Half cadence: ? -> V
        if final_degree == 7:
            cadences.append("half")

        return cadences

    def _calculate_chromaticism(
        self,
        notes: list[dict],
        mode: str,
        key_root: int,
    ) -> float:
        """Calculate ratio of non-diatonic notes."""
        if not notes:
            return 0.0

        # Get diatonic pitches for detected mode
        profile = self.MODE_PROFILES.get(mode, self.MODE_PROFILES["minor"])
        diatonic = set(i for i, v in enumerate(profile) if v == 1)

        # Rotate to actual key
        diatonic = {(pc + key_root) % 12 for pc in diatonic}

        chromatic_count = 0
        for note in notes:
            pc = note["pitch"] % 12
            if pc not in diatonic:
                chromatic_count += 1

        return chromatic_count / len(notes)

    def _calculate_tension(self, chords: list[dict]) -> float:
        """Calculate overall harmonic tension."""
        if not chords:
            return 0.5

        tension_scores = []

        for chord in chords:
            pitches = chord["pitches"]
            if len(pitches) < 2:
                continue

            # Check for dissonant intervals
            intervals = []
            for i, p1 in enumerate(pitches):
                for p2 in pitches[i + 1:]:
                    intervals.append(abs(p2 - p1) % 12)

            # Dissonance mapping
            dissonance = {
                1: 1.0,   # Minor 2nd
                2: 0.7,   # Major 2nd
                6: 0.9,   # Tritone
                10: 0.5,  # Minor 7th
                11: 0.8,  # Major 7th
            }

            chord_tension = sum(dissonance.get(i, 0.0) for i in intervals)
            chord_tension /= max(1, len(intervals))
            tension_scores.append(chord_tension)

        return sum(tension_scores) / len(tension_scores) if tension_scores else 0.5

    def _extract_root_motion(self, chords: list[dict]) -> list[int]:
        """Extract intervals between consecutive chord roots."""
        if len(chords) < 2:
            return []

        roots = [min(c["pitches"]) for c in chords if c["pitches"]]
        motion = []

        for i in range(len(roots) - 1):
            interval = (roots[i + 1] - roots[i]) % 12
            motion.append(interval)

        return motion


# ============================================================================
# Genre Detector (Ensemble)
# ============================================================================

@dataclass
class GenreDetectionResult:
    """Result of genre detection."""
    primary_genre: str
    confidence: float
    genre_scores: dict[str, float]  # All genres with scores
    rhythm_features: RhythmFeatures
    harmonic_features: HarmonicFeatures

    # Similarity to target DNA
    dna_similarity: float = 0.0

    # Feature vector for further analysis
    feature_vector: Optional[list[float]] = None


class GenreDetector:
    """
    Ensemble genre classifier combining rhythm and harmonic analysis.

    Uses rule-based matching against genre DNA profiles combined with
    weighted feature similarity scoring.
    """

    def __init__(self):
        self.rhythm_matcher = RhythmPatternMatcher()
        self.harmonic_analyzer = HarmonicAnalyzer()

        # Feature weights for similarity
        self.weights = {
            "tempo": 2.0,
            "swing": 1.5,
            "kick_pattern": 2.5,
            "mode": 1.5,
            "complexity": 1.0,
            "instruments": 1.0,
        }

    def detect(
        self,
        notes: list[dict],
        tempo_bpm: float,
        ticks_per_beat: int = 480,
        target_genre: Optional[str] = None,
    ) -> GenreDetectionResult:
        """
        Detect genre from MIDI notes.

        Args:
            notes: List of MIDI note dicts
            tempo_bpm: Track tempo
            ticks_per_beat: MIDI resolution
            target_genre: Optional target genre for similarity scoring

        Returns:
            GenreDetectionResult with classification
        """
        # Extract features
        rhythm = self.rhythm_matcher.analyze(notes, tempo_bpm, ticks_per_beat)
        harmony = self.harmonic_analyzer.analyze(notes, ticks_per_beat)

        # Score against all genres
        genre_scores = {}

        for genre_id, dna in GENRE_DNA_LIBRARY.items():
            score = self._score_genre_match(dna, rhythm, harmony)
            genre_scores[genre_id] = score

        # Find best match
        best_genre = max(genre_scores, key=genre_scores.get)
        best_score = genre_scores[best_genre]

        # Calculate target similarity if specified
        dna_similarity = 0.0
        if target_genre:
            target_dna = get_genre_dna(target_genre)
            if target_dna:
                dna_similarity = genre_scores.get(target_genre, 0.0)

        # Build feature vector
        feature_vector = rhythm.to_vector() + harmony.to_vector()

        return GenreDetectionResult(
            primary_genre=best_genre,
            confidence=best_score,
            genre_scores=genre_scores,
            rhythm_features=rhythm,
            harmonic_features=harmony,
            dna_similarity=dna_similarity,
            feature_vector=feature_vector,
        )

    def _score_genre_match(
        self,
        dna: GenreDNA,
        rhythm: RhythmFeatures,
        harmony: HarmonicFeatures,
    ) -> float:
        """Score how well features match a genre DNA."""
        scores = []
        weights = []

        # Tempo match
        tempo_range = dna.rhythm.tempo_range
        if tempo_range[0] <= rhythm.tempo_bpm <= tempo_range[1]:
            # Perfect range match
            mid = (tempo_range[0] + tempo_range[1]) / 2
            tempo_score = 1.0 - abs(rhythm.tempo_bpm - mid) / (tempo_range[1] - tempo_range[0])
            scores.append(tempo_score)
        else:
            # Out of range penalty
            if rhythm.tempo_bpm < tempo_range[0]:
                dist = tempo_range[0] - rhythm.tempo_bpm
            else:
                dist = rhythm.tempo_bpm - tempo_range[1]
            scores.append(max(0, 1.0 - dist / 50))
        weights.append(self.weights["tempo"])

        # Swing match
        swing_target = dna.rhythm.swing_amount
        swing_diff = abs(rhythm.swing_amount - swing_target)
        scores.append(1.0 - swing_diff)
        weights.append(self.weights["swing"])

        # Kick pattern match
        if rhythm.detected_kick_pattern == dna.rhythm.kick_pattern:
            scores.append(1.0)
        elif rhythm.detected_kick_pattern in [KickPattern.FOUR_ON_FLOOR, KickPattern.BOOM_BAP]:
            # Partial match for common patterns
            scores.append(0.5)
        else:
            scores.append(0.2)
        weights.append(self.weights["kick_pattern"])

        # Mode match
        if harmony.detected_mode in dna.harmony.primary_modes:
            scores.append(1.0)
        elif harmony.detected_mode in dna.harmony.secondary_modes:
            scores.append(0.7)
        else:
            scores.append(0.3)
        weights.append(self.weights["mode"])

        # Complexity match
        complexity_range = dna.harmony.chord_complexity
        if complexity_range[0] <= harmony.chord_complexity <= complexity_range[1]:
            scores.append(1.0)
        else:
            diff = min(
                abs(harmony.chord_complexity - complexity_range[0]),
                abs(harmony.chord_complexity - complexity_range[1])
            )
            scores.append(max(0, 1.0 - diff * 2))
        weights.append(self.weights["complexity"])

        # Weighted average
        if not weights:
            return 0.5

        total = sum(s * w for s, w in zip(scores, weights))
        return total / sum(weights)

    def compute_genre_similarity(
        self,
        result: GenreDetectionResult,
        target_genre: str,
    ) -> float:
        """Compute similarity score to a specific genre."""
        target_dna = get_genre_dna(target_genre)
        if target_dna is None:
            return 0.0

        return self._score_genre_match(
            target_dna,
            result.rhythm_features,
            result.harmonic_features,
        )

    def get_genre_deviations(
        self,
        result: GenreDetectionResult,
        target_genre: str,
    ) -> dict[str, float]:
        """Get specific deviations from target genre."""
        target_dna = get_genre_dna(target_genre)
        if target_dna is None:
            return {}

        rhythm = result.rhythm_features
        harmony = result.harmonic_features

        deviations = {}

        # Tempo deviation
        mid_tempo = (target_dna.rhythm.tempo_range[0] + target_dna.rhythm.tempo_range[1]) / 2
        deviations["tempo"] = abs(rhythm.tempo_bpm - mid_tempo) / 50

        # Swing deviation
        deviations["swing"] = abs(rhythm.swing_amount - target_dna.rhythm.swing_amount)

        # Pattern match (binary)
        deviations["kick_pattern"] = 0.0 if rhythm.detected_kick_pattern == target_dna.rhythm.kick_pattern else 1.0

        # Mode deviation
        if harmony.detected_mode in target_dna.harmony.primary_modes:
            deviations["mode"] = 0.0
        elif harmony.detected_mode in target_dna.harmony.secondary_modes:
            deviations["mode"] = 0.3
        else:
            deviations["mode"] = 1.0

        # Complexity deviation
        complexity_mid = (target_dna.harmony.chord_complexity[0] + target_dna.harmony.chord_complexity[1]) / 2
        deviations["complexity"] = abs(harmony.chord_complexity - complexity_mid)

        return deviations
