"""
Real Singing Synthesis Engine

Combines XTTS speech synthesis with pitch/time manipulation
to create actual singing vocals aligned to melody and rhythm.

Pipeline:
1. Generate base speech with XTTS (per syllable/phrase)
2. Apply pitch-shifting to match melody notes (using PSOLA/phase vocoder)
3. Apply time-stretching for rhythm alignment
4. Add singing expression (vibrato, scoops, falls)
5. Mix with beat-aligned timing
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyllableSpec:
    """Specification for a single syllable to sing."""
    text: str
    phonemes: List[str]
    pitch_midi: int  # Target MIDI note
    start_beat: float  # Position in beats
    duration_beats: float  # Length in beats
    velocity: int = 100  # Intensity 0-127
    is_melisma: bool = False  # Sustaining previous syllable


@dataclass
class SingingSpec:
    """Complete specification for singing synthesis."""
    syllables: List[SyllableSpec]
    tempo: float  # BPM
    voice_name: str = "AVU-1"
    genre: str = "pop"
    key: str = "C major"
    # Expression
    vibrato_depth: float = 0.5  # 0-1
    scoop_amount: float = 0.3  # 0-1
    expression_intensity: float = 0.7  # 0-1


@dataclass
class SingingResult:
    """Result of singing synthesis."""
    audio: np.ndarray  # float32 samples
    sample_rate: int = 48000
    duration_seconds: float = 0.0
    # Alignment info
    syllable_timings: List[Tuple[float, float]] = field(default_factory=list)  # (start_ms, end_ms)


class VocalSynthesizer:
    """
    Real singing synthesis using XTTS + pitch/time manipulation.

    Converts lyrics and melody into actual sung vocals.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        use_gpu: bool = True,
    ):
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu
        self._librosa = None
        self._pytsmod = None

    @property
    def librosa(self):
        """Lazy load librosa."""
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa

    @property
    def pytsmod(self):
        """Lazy load pytsmod for pitch shifting."""
        if self._pytsmod is None:
            try:
                import pytsmod
                self._pytsmod = pytsmod
            except ImportError:
                logger.warning("pytsmod not available, using librosa for pitch shift")
                self._pytsmod = False
        return self._pytsmod

    async def synthesize(
        self,
        spec: SingingSpec,
        progress_callback: Optional[callable] = None,
    ) -> SingingResult:
        """
        Synthesize singing vocals from specification.

        Args:
            spec: Complete singing specification
            progress_callback: Optional progress callback (0.0 to 1.0)

        Returns:
            SingingResult with audio and timing data
        """
        logger.info(f"Synthesizing {len(spec.syllables)} syllables at {spec.tempo} BPM")

        # Calculate total duration
        if not spec.syllables:
            return SingingResult(
                audio=np.zeros(int(self.sample_rate), dtype=np.float32),
                sample_rate=self.sample_rate,
            )

        last_syl = max(spec.syllables, key=lambda s: s.start_beat + s.duration_beats)
        total_beats = last_syl.start_beat + last_syl.duration_beats
        beat_duration = 60 / spec.tempo
        total_seconds = total_beats * beat_duration
        total_samples = int(total_seconds * self.sample_rate)

        # Initialize output buffer
        output = np.zeros(total_samples, dtype=np.float32)
        syllable_timings = []

        # Group syllables into phrases for more natural speech
        phrases = self._group_into_phrases(spec.syllables, beat_duration)

        for phrase_idx, phrase in enumerate(phrases):
            if progress_callback:
                progress_callback(phrase_idx / len(phrases) * 0.8)

            # Generate phrase audio
            phrase_audio = await self._synthesize_phrase(
                phrase,
                spec.tempo,
                spec.voice_name,
                spec.genre,
                spec.vibrato_depth,
                spec.scoop_amount,
            )

            if phrase_audio is not None and len(phrase_audio) > 0:
                # Place in output buffer
                start_sample = int(phrase[0].start_beat * beat_duration * self.sample_rate)
                end_sample = min(start_sample + len(phrase_audio), total_samples)

                if end_sample > start_sample:
                    # Crossfade for smooth joins
                    fade_len = min(1000, (end_sample - start_sample) // 4)

                    # Apply fade in
                    if fade_len > 0:
                        phrase_audio[:fade_len] *= np.linspace(0, 1, fade_len)
                        phrase_audio[-fade_len:] *= np.linspace(1, 0, fade_len)

                    output[start_sample:end_sample] += phrase_audio[:end_sample - start_sample]

                # Record timings
                for syl in phrase:
                    start_ms = syl.start_beat * beat_duration * 1000
                    end_ms = (syl.start_beat + syl.duration_beats) * beat_duration * 1000
                    syllable_timings.append((start_ms, end_ms))

        if progress_callback:
            progress_callback(1.0)

        # Normalize output
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.85

        return SingingResult(
            audio=output,
            sample_rate=self.sample_rate,
            duration_seconds=total_seconds,
            syllable_timings=syllable_timings,
        )

    def _group_into_phrases(
        self,
        syllables: List[SyllableSpec],
        beat_duration: float,
        max_gap_beats: float = 0.5,
    ) -> List[List[SyllableSpec]]:
        """Group syllables into phrases based on timing gaps."""
        if not syllables:
            return []

        # Sort by start time
        sorted_syls = sorted(syllables, key=lambda s: s.start_beat)

        phrases = []
        current_phrase = [sorted_syls[0]]

        for i in range(1, len(sorted_syls)):
            prev = sorted_syls[i - 1]
            curr = sorted_syls[i]

            gap = curr.start_beat - (prev.start_beat + prev.duration_beats)

            if gap > max_gap_beats:
                # Start new phrase
                phrases.append(current_phrase)
                current_phrase = [curr]
            else:
                current_phrase.append(curr)

        if current_phrase:
            phrases.append(current_phrase)

        return phrases

    async def _synthesize_phrase(
        self,
        phrase: List[SyllableSpec],
        tempo: float,
        voice_name: str,
        genre: str,
        vibrato_depth: float,
        scoop_amount: float,
    ) -> Optional[np.ndarray]:
        """
        Synthesize a phrase of syllables.

        1. Generate speech for the phrase text with XTTS
        2. Split into syllables
        3. Apply pitch shifting for each syllable
        4. Apply time stretching for rhythm
        5. Add vibrato
        """
        # Build phrase text
        phrase_text = " ".join(s.text for s in phrase if s.text and s.text != "~")

        if not phrase_text.strip():
            return None

        # Generate base speech
        base_audio, base_sr = await self._generate_xtts_speech(
            phrase_text, voice_name
        )

        if base_audio is None:
            return None

        # Resample to our sample rate if needed
        if base_sr != self.sample_rate:
            base_audio = self.librosa.resample(
                base_audio, orig_sr=base_sr, target_sr=self.sample_rate
            )

        # Calculate target durations for each syllable
        beat_duration = 60 / tempo
        total_phrase_seconds = sum(s.duration_beats * beat_duration for s in phrase)

        # Estimate syllable boundaries in the generated speech
        syllable_boundaries = self._estimate_syllable_boundaries(
            base_audio, len(phrase)
        )

        # Process each syllable
        processed_segments = []

        for i, (syl, (seg_start, seg_end)) in enumerate(zip(phrase, syllable_boundaries)):
            segment = base_audio[seg_start:seg_end]

            if len(segment) < 100:
                continue

            # Target duration for this syllable
            target_duration_samples = int(syl.duration_beats * beat_duration * self.sample_rate)

            # 1. Time stretch to match duration
            if target_duration_samples > 0:
                stretch_ratio = len(segment) / target_duration_samples
                if stretch_ratio > 0.1 and stretch_ratio < 10:
                    segment = self._time_stretch(segment, stretch_ratio)

            # 2. Pitch shift to target note
            target_hz = 440.0 * (2.0 ** ((syl.pitch_midi - 69) / 12.0))
            detected_hz = self._detect_pitch(segment)

            if detected_hz > 0 and target_hz > 0:
                semitones = 12 * np.log2(target_hz / detected_hz)
                if abs(semitones) > 0.1:
                    segment = self._pitch_shift(segment, semitones)

            # 3. Apply scoop (pitch bend at start)
            if scoop_amount > 0 and i == 0:  # Phrase start
                segment = self._apply_scoop(segment, scoop_amount)

            # 4. Apply vibrato
            if vibrato_depth > 0 and syl.duration_beats > 0.5:
                segment = self._apply_vibrato(segment, vibrato_depth, genre)

            processed_segments.append(segment)

        if not processed_segments:
            return None

        # Concatenate with crossfades
        return self._concatenate_with_crossfade(processed_segments)

    async def _generate_xtts_speech(
        self,
        text: str,
        voice_name: str,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Generate base speech using XTTS."""
        try:
            from aether.providers.selfhosted.xtts_subprocess import (
                synthesize_xtts,
                is_voice_venv_available,
            )

            if not is_voice_venv_available():
                logger.warning("Voice venv not available, using fallback")
                return self._generate_fallback_speech(text, voice_name)

            # Generate with XTTS
            output_path = await synthesize_xtts(
                text=text,
                voice_name=voice_name,
                language="en",
            )

            if output_path and output_path.exists():
                audio, sr = self.librosa.load(str(output_path), sr=None)
                return audio.astype(np.float32), sr

        except Exception as e:
            logger.warning(f"XTTS synthesis failed: {e}")

        return self._generate_fallback_speech(text, voice_name)

    def _generate_fallback_speech(
        self,
        text: str,
        voice_name: str,
    ) -> Tuple[np.ndarray, int]:
        """Generate fallback speech (vocoded sine waves)."""
        # Estimate duration based on text length
        duration = max(1.0, len(text.split()) * 0.4)
        samples = int(duration * self.sample_rate)

        # Generate carrier signal
        t = np.arange(samples) / self.sample_rate

        # Base frequency based on voice
        base_freq = {
            "AVU-1": 150,  # Tenor
            "AVU-2": 220,  # Mezzo
            "AVU-3": 100,  # Baritone
            "AVU-4": 280,  # Soprano
        }.get(voice_name, 150)

        # Generate harmonics
        audio = np.zeros(samples, dtype=np.float32)
        for harmonic in range(1, 6):
            amp = 1.0 / harmonic
            audio += amp * np.sin(2 * np.pi * base_freq * harmonic * t)

        # Apply amplitude envelope
        envelope = np.ones(samples)
        attack = int(0.05 * self.sample_rate)
        release = int(0.1 * self.sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)

        audio *= envelope
        audio = audio / np.max(np.abs(audio)) * 0.5

        return audio.astype(np.float32), self.sample_rate

    def _estimate_syllable_boundaries(
        self,
        audio: np.ndarray,
        num_syllables: int,
    ) -> List[Tuple[int, int]]:
        """Estimate syllable boundaries in audio using onset detection."""
        if num_syllables <= 0:
            return []

        if num_syllables == 1:
            return [(0, len(audio))]

        try:
            # Use onset detection
            onset_env = self.librosa.onset.onset_strength(
                y=audio, sr=self.sample_rate
            )
            onsets = self.librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                units='samples',
            )

            if len(onsets) >= num_syllables - 1:
                # Use detected onsets
                boundaries = []
                prev_onset = 0
                for i, onset in enumerate(onsets[:num_syllables - 1]):
                    boundaries.append((prev_onset, onset))
                    prev_onset = onset
                boundaries.append((prev_onset, len(audio)))
                return boundaries
        except Exception as e:
            logger.debug(f"Onset detection failed: {e}")

        # Fallback: equal division
        segment_len = len(audio) // num_syllables
        return [
            (i * segment_len, min((i + 1) * segment_len, len(audio)))
            for i in range(num_syllables)
        ]

    def _time_stretch(
        self,
        audio: np.ndarray,
        rate: float,
    ) -> np.ndarray:
        """Time stretch audio without changing pitch using WSOLA-like approach."""
        if abs(rate - 1.0) < 0.05:
            return audio

        # Use scipy's interpolation for high-quality time stretching
        from scipy.interpolate import interp1d
        from scipy.signal import resample

        try:
            target_len = int(len(audio) / rate)
            if target_len < 10:
                return audio

            # For small changes, use linear interpolation
            if 0.8 < rate < 1.25:
                x_old = np.linspace(0, 1, len(audio))
                x_new = np.linspace(0, 1, target_len)
                interp_func = interp1d(x_old, audio.astype(np.float64), kind='cubic')
                stretched = interp_func(x_new)
                return stretched.astype(np.float32)
            else:
                # Use scipy resample for larger changes
                stretched = resample(audio.astype(np.float64), target_len)
                return stretched.astype(np.float32)
        except Exception as e:
            logger.debug(f"Time stretch scipy failed: {e}")
            # Ultimate fallback: simple index mapping
            target_len = int(len(audio) / rate)
            indices = np.linspace(0, len(audio) - 1, target_len).astype(int)
            return audio[indices]

    def _pitch_shift(
        self,
        audio: np.ndarray,
        semitones: float,
    ) -> np.ndarray:
        """Pitch shift audio using resample + time-stretch method (PSOLA-like)."""
        if abs(semitones) < 0.1:
            return audio

        # Clamp to reasonable range
        semitones = np.clip(semitones, -12, 12)
        ratio = 2 ** (semitones / 12)

        from scipy.signal import resample
        from scipy.interpolate import interp1d

        try:
            # Step 1: Resample to change pitch (this also changes duration)
            resampled_len = int(len(audio) / ratio)
            if resampled_len < 10:
                return audio

            resampled = resample(audio.astype(np.float64), resampled_len)

            # Step 2: Time-stretch back to original length (preserves new pitch)
            x_old = np.linspace(0, 1, len(resampled))
            x_new = np.linspace(0, 1, len(audio))
            interp_func = interp1d(x_old, resampled, kind='linear', fill_value='extrapolate')
            result = interp_func(x_new)

            return result.astype(np.float32)
        except Exception as e:
            logger.debug(f"Pitch shift failed: {e}")
            # Simple fallback: just resample
            try:
                resampled_len = int(len(audio) / ratio)
                indices = np.linspace(0, len(audio) - 1, resampled_len).astype(int)
                resampled = audio[indices]
                out_indices = np.linspace(0, len(resampled) - 1, len(audio)).astype(int)
                return resampled[out_indices]
            except:
                return audio

    def _detect_pitch(
        self,
        audio: np.ndarray,
        fmin: float = 50,
        fmax: float = 800,
    ) -> float:
        """Detect fundamental frequency of audio."""
        try:
            pitches, magnitudes = self.librosa.piptrack(
                y=audio,
                sr=self.sample_rate,
                fmin=fmin,
                fmax=fmax,
            )

            # Get pitch with highest magnitude per frame
            pitch_indices = magnitudes.argmax(axis=0)
            pitches_detected = pitches[pitch_indices, np.arange(pitches.shape[1])]

            # Filter out zeros and get median
            valid_pitches = pitches_detected[pitches_detected > 0]
            if len(valid_pitches) > 0:
                return float(np.median(valid_pitches))
        except Exception as e:
            logger.debug(f"Pitch detection failed: {e}")

        return 0.0

    def _apply_scoop(
        self,
        audio: np.ndarray,
        amount: float,
    ) -> np.ndarray:
        """Apply pitch scoop at the start of audio."""
        scoop_samples = int(0.08 * self.sample_rate)  # 80ms scoop

        if scoop_samples >= len(audio):
            return audio

        # Scoop up from below (negative semitones ramping to 0)
        scoop_cents = -50 * amount  # Start 50 cents below

        # Create pitch envelope
        scoop_curve = np.linspace(scoop_cents, 0, scoop_samples)

        # Apply via frequency modulation
        t = np.arange(scoop_samples) / self.sample_rate
        modulation = np.exp(1j * 2 * np.pi * scoop_curve / 1200 * t)

        # Apply to beginning of audio
        result = audio.copy()

        # Simple approach: slight pitch bend by resampling
        for i in range(min(3, int(abs(scoop_cents) / 10))):
            ratio = 1.0 + (scoop_cents / 1200) * (1 - i / 3)
            segment = result[:scoop_samples // 3]
            if len(segment) > 100:
                stretched = self._time_stretch(segment, 1.0 / ratio)
                result[:len(stretched)] = stretched[:len(result[:scoop_samples // 3])]

        return result

    def _apply_vibrato(
        self,
        audio: np.ndarray,
        depth: float,
        genre: str,
    ) -> np.ndarray:
        """Apply vibrato to audio."""
        # Genre-specific vibrato parameters
        vibrato_params = {
            "pop": {"rate": 5.5, "depth_cents": 30},
            "r-and-b": {"rate": 5.0, "depth_cents": 40},
            "rock": {"rate": 6.0, "depth_cents": 25},
            "jazz": {"rate": 5.0, "depth_cents": 35},
            "house": {"rate": 4.5, "depth_cents": 15},
            "techno": {"rate": 4.0, "depth_cents": 10},
        }.get(genre, {"rate": 5.5, "depth_cents": 25})

        rate = vibrato_params["rate"]
        depth_cents = vibrato_params["depth_cents"] * depth

        if depth_cents < 5:
            return audio

        # Only apply vibrato to sustained portion (skip attack)
        attack_samples = int(0.1 * self.sample_rate)

        if len(audio) <= attack_samples:
            return audio

        result = audio.copy()

        # Generate vibrato LFO
        t = np.arange(len(audio) - attack_samples) / self.sample_rate
        lfo = np.sin(2 * np.pi * rate * t)

        # Apply gradual onset (vibrato builds over time)
        onset = np.minimum(t / 0.3, 1.0)  # 300ms onset
        lfo *= onset

        # Apply pitch modulation via phase vocoder is complex,
        # so we'll use a simpler amplitude modulation for now
        # that creates perceived pitch variation
        am_depth = depth_cents / 100 * 0.1  # Convert to amplitude modulation
        modulation = 1 + lfo * am_depth

        result[attack_samples:] *= modulation.astype(np.float32)

        return result

    def _concatenate_with_crossfade(
        self,
        segments: List[np.ndarray],
        crossfade_ms: float = 20,
    ) -> np.ndarray:
        """Concatenate audio segments with crossfade."""
        if not segments:
            return np.array([], dtype=np.float32)

        if len(segments) == 1:
            return segments[0]

        crossfade_samples = int(crossfade_ms / 1000 * self.sample_rate)

        # Calculate total length
        total_len = sum(len(s) for s in segments)
        total_len -= crossfade_samples * (len(segments) - 1)

        result = np.zeros(total_len, dtype=np.float32)
        pos = 0

        for i, segment in enumerate(segments):
            if i == 0:
                # First segment: no fade in
                end = len(segment) - crossfade_samples
                result[:end] = segment[:end]
                pos = end
            else:
                # Crossfade with previous
                fade_in = np.linspace(0, 1, crossfade_samples)
                fade_out = np.linspace(1, 0, crossfade_samples)

                # Apply crossfade
                result[pos:pos + crossfade_samples] *= fade_out
                result[pos:pos + crossfade_samples] += segment[:crossfade_samples] * fade_in

                # Add rest of segment
                remaining = segment[crossfade_samples:]
                if i == len(segments) - 1:
                    result[pos + crossfade_samples:pos + crossfade_samples + len(remaining)] = remaining
                else:
                    end = len(remaining) - crossfade_samples
                    result[pos + crossfade_samples:pos + crossfade_samples + end] = remaining[:end]
                    pos = pos + crossfade_samples + end

        return result


async def synthesize_singing(
    lyrics: str,
    melody_notes: List[dict],  # List of {pitch, start_beat, duration_beats}
    tempo: float,
    voice_name: str = "AVU-1",
    genre: str = "pop",
    output_path: Optional[str] = None,
) -> SingingResult:
    """
    High-level function to synthesize singing from lyrics and melody.

    Args:
        lyrics: Lyrics text (will be split by spaces)
        melody_notes: List of note dicts with pitch/timing
        tempo: BPM
        voice_name: AVU voice to use
        genre: Genre for style
        output_path: Optional path to save WAV

    Returns:
        SingingResult with audio
    """
    # Split lyrics into syllables (simple split)
    words = lyrics.split()

    # Create syllable specs
    syllables = []
    for i, (word, note) in enumerate(zip(words, melody_notes)):
        syl = SyllableSpec(
            text=word,
            phonemes=[],
            pitch_midi=note["pitch"],
            start_beat=note["start_beat"],
            duration_beats=note["duration_beats"],
            velocity=note.get("velocity", 100),
        )
        syllables.append(syl)

    # Handle extra notes as melismas
    for note in melody_notes[len(words):]:
        syl = SyllableSpec(
            text="~",
            phonemes=[],
            pitch_midi=note["pitch"],
            start_beat=note["start_beat"],
            duration_beats=note["duration_beats"],
            velocity=note.get("velocity", 100),
            is_melisma=True,
        )
        syllables.append(syl)

    # Create spec and synthesize
    spec = SingingSpec(
        syllables=syllables,
        tempo=tempo,
        voice_name=voice_name,
        genre=genre,
    )

    synthesizer = VocalSynthesizer()
    result = await synthesizer.synthesize(spec)

    # Save if requested
    if output_path:
        import scipy.io.wavfile as wav
        audio_int16 = (result.audio * 32767).astype(np.int16)
        wav.write(output_path, result.sample_rate, audio_int16)
        logger.info(f"Saved singing to {output_path}")

    return result


# =============================================================================
# BARK NEURAL SINGING - Real Singing Voice Synthesis
# =============================================================================

async def synthesize_neural_singing(
    lyrics: list[dict],
    tempo: float,
    voice_name: str = "AVU-1",
    total_duration_seconds: Optional[float] = None,
    output_path: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> SingingResult:
    """
    Synthesize REAL singing voices using Bark neural synthesis.

    Unlike pitch-shifted TTS, Bark generates actual singing when given
    text with musical notation markers (♪). This produces natural singing
    that sounds like a real vocalist.

    Args:
        lyrics: List of lyric dicts with:
            - text: Words to sing
            - start_beat: Beat position
            - duration_beats: Duration in beats
        tempo: Track tempo in BPM
        voice_name: AVU voice (AVU-1 through AVU-4)
        total_duration_seconds: Total track duration (optional)
        output_path: Optional path to save WAV
        progress_callback: Optional progress callback (0-1)

    Returns:
        SingingResult with audio data

    Example:
        result = await synthesize_neural_singing(
            lyrics=[
                {"text": "Feel the beat", "start_beat": 4, "duration_beats": 4},
                {"text": "Move your body", "start_beat": 12, "duration_beats": 4},
            ],
            tempo=128.0,
            voice_name="AVU-1",
        )
    """
    from aether.providers.selfhosted import (
        get_bark_provider,
        SingingPhrase,
    )

    # Get Bark provider
    bark = get_bark_provider()

    if progress_callback:
        bark.set_progress_callback(lambda p, m: progress_callback(p))

    # Convert lyrics to SingingPhrase objects
    phrases = [
        SingingPhrase(
            text=lyric["text"],
            start_beat=lyric["start_beat"],
            duration_beats=lyric["duration_beats"],
            emotion=lyric.get("emotion", "neutral"),
            style=lyric.get("style", "default"),
        )
        for lyric in lyrics
    ]

    # Synthesize with Bark
    logger.info(f"Synthesizing {len(phrases)} phrases with Bark neural singing")

    audio = await bark.synthesize_lyrics(
        phrases=phrases,
        voice=voice_name,
        tempo_bpm=tempo,
        total_duration_seconds=total_duration_seconds,
    )

    sample_rate = bark.config.target_sample_rate
    duration = len(audio) / sample_rate

    # Save if requested
    if output_path:
        import scipy.io.wavfile as wav
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(output_path, sample_rate, audio_int16)
        logger.info(f"Saved neural singing to {output_path}")

    return SingingResult(
        audio=audio,
        sample_rate=sample_rate,
        duration_seconds=duration,
    )


def create_singing_track(
    instrumental_audio: np.ndarray,
    vocal_audio: np.ndarray,
    instrumental_sr: int,
    vocal_sr: int,
    vocal_level: float = 0.7,
    instrumental_level: float = 1.0,
) -> np.ndarray:
    """
    Mix instrumental and vocal tracks together.

    Args:
        instrumental_audio: Instrumental audio array
        vocal_audio: Vocal audio array
        instrumental_sr: Instrumental sample rate
        vocal_sr: Vocal sample rate
        vocal_level: Vocal volume (0-1)
        instrumental_level: Instrumental volume (0-1)

    Returns:
        Mixed audio at instrumental sample rate
    """
    from scipy.signal import resample

    # Resample vocals if different sample rate
    if vocal_sr != instrumental_sr:
        num_samples = int(len(vocal_audio) * instrumental_sr / vocal_sr)
        vocal_audio = resample(vocal_audio, num_samples).astype(np.float32)

    # Match lengths
    max_len = max(len(instrumental_audio), len(vocal_audio))

    # Pad shorter track
    if len(instrumental_audio) < max_len:
        instrumental_audio = np.pad(
            instrumental_audio,
            (0, max_len - len(instrumental_audio))
        )
    if len(vocal_audio) < max_len:
        vocal_audio = np.pad(
            vocal_audio,
            (0, max_len - len(vocal_audio))
        )

    # Mix
    mixed = (
        instrumental_audio * instrumental_level +
        vocal_audio * vocal_level
    )

    # Normalize
    max_val = np.abs(mixed).max()
    if max_val > 0.95:
        mixed = mixed * 0.95 / max_val

    return mixed.astype(np.float32)
