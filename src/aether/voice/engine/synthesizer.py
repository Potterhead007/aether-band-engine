"""
Singing Engine Synthesizer

Core singing synthesis engine that combines all components
to generate sung vocals from lyrics and melody.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import logging

from aether.voice.identity.blueprint import VocalIdentity, AVU1Identity

logger = logging.getLogger(__name__)


@dataclass
class LyricToken:
    """A token in the lyrics (word or syllable)."""
    text: str
    phonemes: Optional[List[str]] = None
    language: Literal["en", "es"] = "en"
    stress: Optional[float] = None  # 0.0-1.0


@dataclass
class MelodyNote:
    """A note in the melody."""
    pitch: int  # MIDI note number
    start_beat: float  # Position in beats
    duration_beats: float  # Length in beats
    velocity: int = 100  # 0-127 intensity
    lyric_index: Optional[int] = None  # Link to lyric token


@dataclass
class PhonemeSpan:
    """Time span for a phoneme in output."""
    phoneme: str
    start_ms: float
    end_ms: float
    confidence: float = 1.0


@dataclass
class WordSpan:
    """Time span for a word in output."""
    word: str
    start_ms: float
    end_ms: float
    phoneme_spans: List[PhonemeSpan] = field(default_factory=list)


@dataclass
class VocalRenderParams:
    """Parameters used for vocal rendering."""
    identity: str
    genre: str
    language: str
    tempo: float
    key: str
    expression_intensity: float


@dataclass
class SingingEngineInput:
    """Input specification for the singing engine."""
    # Required
    lyrics: List[LyricToken]
    melody: List[MelodyNote]
    tempo: float  # BPM
    key: str  # Musical key (e.g., "C major", "A minor")
    time_signature: Tuple[int, int] = (4, 4)

    # Required context
    genre_id: str = "pop"
    language: Literal["en", "es"] = "en"

    # Optional expression
    emotion_curve: Optional[List[Tuple[float, float]]] = None  # (beat, intensity)
    energy_curve: Optional[List[float]] = None
    section_map: Optional[List[Tuple[float, float, str]]] = None  # (start, end, type)

    # Optional control
    style_overrides: Optional[Dict[str, Any]] = None


@dataclass
class SingingEngineOutput:
    """Output from the singing engine."""
    audio: np.ndarray  # Shape: (samples,), float32, 48kHz
    sample_rate: int = 48000

    # Alignment data
    phoneme_alignment: List[PhonemeSpan] = field(default_factory=list)
    word_alignment: List[WordSpan] = field(default_factory=list)

    # Quality metrics
    pitch_confidence: List[float] = field(default_factory=list)
    phoneme_confidence: List[float] = field(default_factory=list)

    # Metadata
    duration_seconds: float = 0.0
    render_params: Optional[VocalRenderParams] = None


class SingingEngine:
    """
    Core singing synthesis engine.

    Combines lyric-melody alignment, pitch generation, expression
    modeling, and audio synthesis to produce sung vocals.
    """

    def __init__(
        self,
        identity: Optional[VocalIdentity] = None,
        sample_rate: int = 48000,
    ):
        """
        Initialize the singing engine.

        Args:
            identity: Vocal identity to use (defaults to AVU-1)
            sample_rate: Output sample rate
        """
        self.identity = identity or AVU1Identity
        self.sample_rate = sample_rate

        # Lazy-loaded components
        self._aligner = None
        self._pitch_controller = None
        self._vibrato_generator = None
        self._transition_engine = None
        self._breath_model = None

        logger.info(f"SingingEngine initialized with identity: {self.identity.name}")

    @property
    def aligner(self):
        """Lazy-load lyric-melody aligner."""
        if self._aligner is None:
            from aether.voice.engine.aligner import LyricMelodyAligner
            self._aligner = LyricMelodyAligner()
        return self._aligner

    @property
    def pitch_controller(self):
        """Lazy-load pitch controller."""
        if self._pitch_controller is None:
            from aether.voice.engine.pitch import PitchController
            self._pitch_controller = PitchController(self.identity)
        return self._pitch_controller

    @property
    def vibrato_generator(self):
        """Lazy-load vibrato generator."""
        if self._vibrato_generator is None:
            from aether.voice.engine.vibrato import VibratoGenerator
            self._vibrato_generator = VibratoGenerator(self.identity)
        return self._vibrato_generator

    @property
    def transition_engine(self):
        """Lazy-load transition engine."""
        if self._transition_engine is None:
            from aether.voice.engine.transitions import TransitionEngine
            self._transition_engine = TransitionEngine()
        return self._transition_engine

    @property
    def breath_model(self):
        """Lazy-load breath model."""
        if self._breath_model is None:
            from aether.voice.engine.breath import BreathModel
            self._breath_model = BreathModel(self.identity)
        return self._breath_model

    async def synthesize(
        self,
        input_spec: SingingEngineInput,
    ) -> SingingEngineOutput:
        """
        Synthesize sung vocals from input specification.

        Args:
            input_spec: Complete input specification

        Returns:
            Synthesized audio with alignment data
        """
        logger.info(f"Synthesizing {len(input_spec.lyrics)} lyrics, "
                   f"{len(input_spec.melody)} notes at {input_spec.tempo} BPM")

        # Step 1: Align lyrics to melody
        aligned_units = self.aligner.align(input_spec.lyrics, input_spec.melody)

        # Step 2: Plan breaths
        breaths = self.breath_model.plan_breaths(
            input_spec.lyrics,
            input_spec.melody,
            input_spec.section_map or [],
            input_spec.tempo,
        )

        # Step 3: Generate pitch contours with expression
        pitch_contours = []
        for unit in aligned_units:
            contour = self.pitch_controller.generate_pitch_contour(
                unit,
                input_spec.tempo,
                input_spec.genre_id,
            )
            pitch_contours.append(contour)

        # Step 4: Apply vibrato
        for i, contour in enumerate(pitch_contours):
            if len(contour) > 0:
                vibrato = self.vibrato_generator.generate(
                    len(contour),
                    input_spec.genre_id,
                    aligned_units[i].velocity if hasattr(aligned_units[i], 'velocity') else 100,
                )
                pitch_contours[i] = contour + vibrato

        # Step 5: Plan transitions between notes
        transitions = []
        for i in range(len(aligned_units) - 1):
            trans = self.transition_engine.select_transition(
                aligned_units[i],
                aligned_units[i + 1],
                input_spec.genre_id,
            )
            transitions.append(trans)

        # Step 6: Synthesize audio
        # Note: In production, this would use a neural vocoder
        # For now, generate placeholder audio
        total_duration = self._calculate_duration(input_spec.melody, input_spec.tempo)
        total_samples = int(total_duration * self.sample_rate)

        # Placeholder: Generate simple sine wave based on pitch
        audio = self._synthesize_placeholder(
            pitch_contours,
            aligned_units,
            input_spec.tempo,
            total_samples,
        )

        # Step 7: Build alignment data
        phoneme_alignment, word_alignment = self._build_alignment(
            aligned_units,
            input_spec.tempo,
        )

        # Step 8: Calculate confidence scores
        pitch_confidence = [0.95] * len(aligned_units)  # Placeholder
        phoneme_confidence = [0.92] * len(phoneme_alignment)

        return SingingEngineOutput(
            audio=audio,
            sample_rate=self.sample_rate,
            phoneme_alignment=phoneme_alignment,
            word_alignment=word_alignment,
            pitch_confidence=pitch_confidence,
            phoneme_confidence=phoneme_confidence,
            duration_seconds=total_duration,
            render_params=VocalRenderParams(
                identity=self.identity.name,
                genre=input_spec.genre_id,
                language=input_spec.language,
                tempo=input_spec.tempo,
                key=input_spec.key,
                expression_intensity=0.7,
            ),
        )

    def _calculate_duration(
        self,
        melody: List[MelodyNote],
        tempo: float
    ) -> float:
        """Calculate total duration in seconds."""
        if not melody:
            return 0.0

        last_note = max(melody, key=lambda n: n.start_beat + n.duration_beats)
        total_beats = last_note.start_beat + last_note.duration_beats
        return total_beats * 60 / tempo

    def _synthesize_placeholder(
        self,
        pitch_contours: List[np.ndarray],
        aligned_units: List,
        tempo: float,
        total_samples: int,
    ) -> np.ndarray:
        """
        Generate placeholder audio.

        In production, this would be replaced with a neural vocoder.
        """
        audio = np.zeros(total_samples, dtype=np.float32)

        beat_duration = 60 / tempo
        samples_per_beat = int(beat_duration * self.sample_rate)

        for i, (contour, unit) in enumerate(zip(pitch_contours, aligned_units)):
            if not hasattr(unit, 'start_beat'):
                continue

            start_sample = int(unit.start_beat * samples_per_beat)
            duration_samples = int(unit.duration_beats * samples_per_beat)

            if start_sample >= total_samples:
                continue

            end_sample = min(start_sample + duration_samples, total_samples)
            actual_samples = end_sample - start_sample

            if actual_samples <= 0:
                continue

            # Generate sine wave at pitch
            if len(contour) > 0:
                # Interpolate contour to match duration
                pitch_hz = np.interp(
                    np.linspace(0, 1, actual_samples),
                    np.linspace(0, 1, len(contour)),
                    contour,
                )
            else:
                # Use note pitch
                pitch_hz = np.full(actual_samples, 440 * (2 ** ((unit.pitch - 69) / 12)))

            # Generate waveform
            t = np.arange(actual_samples) / self.sample_rate
            phase = np.cumsum(2 * np.pi * pitch_hz / self.sample_rate)
            wave = np.sin(phase) * 0.3

            # Apply envelope
            envelope = self._generate_envelope(actual_samples)
            wave *= envelope

            # Mix into output
            audio[start_sample:end_sample] += wave.astype(np.float32)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio

    def _generate_envelope(self, samples: int) -> np.ndarray:
        """Generate ADSR envelope."""
        attack = int(samples * 0.05)
        decay = int(samples * 0.1)
        sustain_level = 0.8
        release = int(samples * 0.15)
        sustain = samples - attack - decay - release

        envelope = np.concatenate([
            np.linspace(0, 1, max(1, attack)),
            np.linspace(1, sustain_level, max(1, decay)),
            np.full(max(1, sustain), sustain_level),
            np.linspace(sustain_level, 0, max(1, release)),
        ])

        # Ensure correct length
        if len(envelope) < samples:
            envelope = np.pad(envelope, (0, samples - len(envelope)))
        elif len(envelope) > samples:
            envelope = envelope[:samples]

        return envelope

    def _build_alignment(
        self,
        aligned_units: List,
        tempo: float,
    ) -> Tuple[List[PhonemeSpan], List[WordSpan]]:
        """Build alignment data from synthesized units."""
        phoneme_spans = []
        word_spans = []

        beat_duration_ms = 60000 / tempo

        for unit in aligned_units:
            if not hasattr(unit, 'phonemes') or not hasattr(unit, 'start_beat'):
                continue

            start_ms = unit.start_beat * beat_duration_ms
            duration_ms = unit.duration_beats * beat_duration_ms

            # Create word span
            word_span = WordSpan(
                word=unit.text if hasattr(unit, 'text') else "",
                start_ms=start_ms,
                end_ms=start_ms + duration_ms,
                phoneme_spans=[],
            )

            # Distribute phonemes across duration
            if hasattr(unit, 'phonemes') and unit.phonemes:
                phoneme_duration = duration_ms / len(unit.phonemes)
                for j, phoneme in enumerate(unit.phonemes):
                    p_start = start_ms + j * phoneme_duration
                    p_span = PhonemeSpan(
                        phoneme=phoneme,
                        start_ms=p_start,
                        end_ms=p_start + phoneme_duration,
                    )
                    phoneme_spans.append(p_span)
                    word_span.phoneme_spans.append(p_span)

            word_spans.append(word_span)

        return phoneme_spans, word_spans

    def render_to_file(
        self,
        output: SingingEngineOutput,
        file_path: str,
        format: str = "wav",
    ) -> str:
        """
        Render synthesis output to audio file.

        Args:
            output: Synthesis output
            file_path: Output file path
            format: Audio format (wav, mp3, flac)

        Returns:
            Path to rendered file
        """
        import scipy.io.wavfile as wav

        # Ensure .wav extension for scipy
        if format == "wav":
            if not file_path.endswith(".wav"):
                file_path += ".wav"

            # Convert to int16
            audio_int16 = (output.audio * 32767).astype(np.int16)
            wav.write(file_path, output.sample_rate, audio_int16)

            logger.info(f"Rendered vocals to {file_path}")
            return file_path

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'wav'.")
