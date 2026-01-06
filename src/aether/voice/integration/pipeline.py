"""
Voice Synthesis Pipeline

Orchestrates the complete voice synthesis process.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from aether.voice.identity.blueprint import VocalIdentity, AVU1Identity
from aether.voice.identity.invariants import IdentityInvariants, AVU1_INVARIANTS
from aether.voice.identity.drift_monitor import IdentityDriftTracker
from aether.voice.phonetics.english import EnglishPhonetics
from aether.voice.phonetics.spanish import SpanishPhonetics
from aether.voice.phonetics.bilingual import BilingualController
from aether.voice.engine.synthesizer import SingingEngine, SingingEngineInput
from aether.voice.performance.profiles import get_profile, VocalPerformanceProfile
from aether.voice.performance.expression import ExpressionMapper, EmotionVector
from aether.voice.arrangement.layers import VocalArrangementSystem
from aether.voice.arrangement.harmony import HarmonyGenerator
from aether.voice.arrangement.safeguards import ArrangementSafeguards
from aether.voice.quality.evaluator import VocalQualityEvaluator, ReleaseStage


logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Stages of the synthesis pipeline."""
    INPUT_VALIDATION = "input_validation"
    TEXT_PROCESSING = "text_processing"
    MELODY_ALIGNMENT = "melody_alignment"
    PERFORMANCE_PLANNING = "performance_planning"
    VOICE_SYNTHESIS = "voice_synthesis"
    # Self-hosted backend stages
    XTTS_GENERATION = "xtts_generation"
    RVC_CONVERSION = "rvc_conversion"
    # Continue stages
    HARMONY_GENERATION = "harmony_generation"
    ARRANGEMENT = "arrangement"
    MIXING = "mixing"
    QUALITY_CHECK = "quality_check"
    OUTPUT = "output"


class SynthesisBackend(Enum):
    """Voice synthesis backend options."""
    INTERNAL = "internal"       # Built-in SingingEngine (default)
    SELF_HOSTED = "self_hosted" # XTTS + RVC pipeline
    ELEVENLABS = "elevenlabs"   # ElevenLabs API
    AUTO = "auto"               # Auto-select best available


@dataclass
class PipelineProgress:
    """Progress tracking for pipeline execution."""
    current_stage: PipelineStage
    progress_pct: float
    stage_message: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class VoiceSynthesisInput:
    """Input specification for voice synthesis."""
    # Required
    lyrics: str
    melody: List[dict]  # List of {pitch, start_beat, duration_beats, velocity}

    # Optional
    tempo: float = 120.0
    key_root: int = 0  # C
    scale_type: str = "major"
    genre: str = "pop"
    language: str = "en"
    emotion: Optional[str] = None
    emotion_intensity: float = 0.5

    # Backend selection
    backend: str = "auto"  # "internal", "self_hosted", "elevenlabs", "auto"
    voice_name: str = "AVU-1"  # Voice identity for self-hosted/elevenlabs
    pitch_shift: Optional[int] = None  # Pitch shift in semitones

    # Arrangement
    generate_harmonies: bool = False
    generate_doubles: bool = False
    arrangement_density: float = 0.5

    # Quality
    quality_check: bool = True
    release_stage: str = "beta"


@dataclass
class VoiceSynthesisOutput:
    """Output from voice synthesis."""
    # Audio
    lead_audio: np.ndarray
    harmony_audio: Optional[np.ndarray] = None
    mixed_audio: Optional[np.ndarray] = None

    # Metadata
    sample_rate: int = 48000
    duration_seconds: float = 0.0

    # Analysis
    pitch_contour: Optional[np.ndarray] = None
    phoneme_sequence: Optional[List[str]] = None

    # Quality
    quality_score: Optional[float] = None
    quality_passed: bool = True
    quality_warnings: List[str] = field(default_factory=list)


class VoiceSynthesisPipeline:
    """
    Main pipeline for voice synthesis.

    Coordinates all components:
    - Text processing and phonetics
    - Melody alignment
    - Performance planning
    - Voice synthesis (internal, self-hosted, or external API)
    - Harmony generation
    - Arrangement and mixing
    - Quality evaluation
    """

    def __init__(
        self,
        identity: Optional[VocalIdentity] = None,
        sample_rate: int = 48000,
        default_backend: str = "auto",
    ):
        """
        Initialize pipeline.

        Args:
            identity: Vocal identity (defaults to AVU-1)
            sample_rate: Output sample rate
            default_backend: Default synthesis backend
        """
        self.identity = identity or AVU1Identity
        self.sample_rate = sample_rate
        self.default_backend = default_backend

        # Initialize components (lazy-loaded for performance)
        self._engine: Optional[SingingEngine] = None
        self._selfhosted_provider = None
        self._elevenlabs_provider = None
        self._english_phonetics: Optional[EnglishPhonetics] = None
        self._spanish_phonetics: Optional[SpanishPhonetics] = None
        self._bilingual: Optional[BilingualController] = None
        self._expression_mapper: Optional[ExpressionMapper] = None
        self._arrangement_system: Optional[VocalArrangementSystem] = None
        self._harmony_generator: Optional[HarmonyGenerator] = None
        self._safeguards: Optional[ArrangementSafeguards] = None
        self._quality_evaluator: Optional[VocalQualityEvaluator] = None
        self._drift_tracker: Optional[IdentityDriftTracker] = None

        # Progress callback
        self._progress_callback: Optional[callable] = None

        # Backend availability cache
        self._backend_availability: Dict[str, bool] = {}

    @property
    def engine(self) -> SingingEngine:
        """Get or create singing engine."""
        if self._engine is None:
            self._engine = SingingEngine(self.identity, self.sample_rate)
        return self._engine

    @property
    def english_phonetics(self) -> EnglishPhonetics:
        """Get or create English phonetics."""
        if self._english_phonetics is None:
            self._english_phonetics = EnglishPhonetics()
        return self._english_phonetics

    @property
    def spanish_phonetics(self) -> SpanishPhonetics:
        """Get or create Spanish phonetics."""
        if self._spanish_phonetics is None:
            self._spanish_phonetics = SpanishPhonetics()
        return self._spanish_phonetics

    @property
    def bilingual(self) -> BilingualController:
        """Get or create bilingual controller."""
        if self._bilingual is None:
            self._bilingual = BilingualController()
        return self._bilingual

    @property
    def expression_mapper(self) -> ExpressionMapper:
        """Get or create expression mapper."""
        if self._expression_mapper is None:
            self._expression_mapper = ExpressionMapper()
        return self._expression_mapper

    @property
    def arrangement_system(self) -> VocalArrangementSystem:
        """Get or create arrangement system."""
        if self._arrangement_system is None:
            self._arrangement_system = VocalArrangementSystem()
        return self._arrangement_system

    @property
    def harmony_generator(self) -> HarmonyGenerator:
        """Get or create harmony generator."""
        if self._harmony_generator is None:
            self._harmony_generator = HarmonyGenerator()
        return self._harmony_generator

    @property
    def safeguards(self) -> ArrangementSafeguards:
        """Get or create safeguards."""
        if self._safeguards is None:
            self._safeguards = ArrangementSafeguards()
        return self._safeguards

    @property
    def quality_evaluator(self) -> VocalQualityEvaluator:
        """Get or create quality evaluator."""
        if self._quality_evaluator is None:
            self._quality_evaluator = VocalQualityEvaluator()
        return self._quality_evaluator

    @property
    def drift_tracker(self) -> IdentityDriftTracker:
        """Get or create drift tracker."""
        if self._drift_tracker is None:
            self._drift_tracker = IdentityDriftTracker(
                AVU1_INVARIANTS,
                self.identity,
            )
        return self._drift_tracker

    async def get_selfhosted_provider(self):
        """Get or create self-hosted provider."""
        if self._selfhosted_provider is None:
            try:
                from aether.providers.selfhosted import get_selfhosted_provider
                self._selfhosted_provider = await get_selfhosted_provider()
            except ImportError:
                logger.debug("Self-hosted provider not installed")
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize self-hosted provider: {e}")
                return None
        return self._selfhosted_provider

    async def get_elevenlabs_provider(self):
        """Get or create ElevenLabs provider."""
        if self._elevenlabs_provider is None:
            try:
                from aether.providers.elevenlabs import get_elevenlabs_provider
                self._elevenlabs_provider = await get_elevenlabs_provider()
            except ImportError:
                logger.debug("ElevenLabs provider not installed")
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize ElevenLabs provider: {e}")
                return None
        return self._elevenlabs_provider

    async def _determine_backend(self, requested: str) -> str:
        """
        Determine which backend to use.

        Args:
            requested: Requested backend ("auto", "internal", "self_hosted", "elevenlabs")

        Returns:
            Resolved backend name
        """
        if requested == "internal":
            return "internal"

        if requested == "self_hosted":
            provider = await self.get_selfhosted_provider()
            if provider and provider.is_available():
                return "self_hosted"
            logger.warning("Self-hosted backend requested but not available, falling back")
            return "internal"

        if requested == "elevenlabs":
            provider = await self.get_elevenlabs_provider()
            if provider:
                return "elevenlabs"
            logger.warning("ElevenLabs backend requested but not available, falling back")
            return "internal"

        # Auto mode - try best available
        if requested == "auto":
            # Prefer self-hosted (highest quality, no API costs)
            provider = await self.get_selfhosted_provider()
            if provider and provider.is_available():
                return "self_hosted"

            # Try ElevenLabs
            provider = await self.get_elevenlabs_provider()
            if provider:
                return "elevenlabs"

            # Fall back to internal
            return "internal"

        return "internal"

    def set_progress_callback(self, callback: callable) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _report_progress(
        self,
        stage: PipelineStage,
        progress: float,
        message: str,
    ) -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(PipelineProgress(
                current_stage=stage,
                progress_pct=progress,
                stage_message=message,
            ))

    async def synthesize(
        self,
        input_spec: VoiceSynthesisInput,
    ) -> VoiceSynthesisOutput:
        """
        Execute the complete synthesis pipeline.

        Args:
            input_spec: Input specification

        Returns:
            Synthesis output
        """
        warnings = []

        # Stage 1: Input Validation
        self._report_progress(PipelineStage.INPUT_VALIDATION, 0, "Validating input...")
        validation_errors = self._validate_input(input_spec)
        if validation_errors:
            raise ValueError(f"Invalid input: {'; '.join(validation_errors)}")

        # Stage 2: Text Processing
        self._report_progress(PipelineStage.TEXT_PROCESSING, 10, "Processing lyrics...")
        lyric_tokens = await self._process_lyrics(
            input_spec.lyrics,
            input_spec.language,
        )

        # Stage 3: Performance Planning
        self._report_progress(PipelineStage.PERFORMANCE_PLANNING, 25, "Planning performance...")
        performance_profile = get_profile(input_spec.genre)
        expression_params = None

        if input_spec.emotion:
            emotion_vector = self._parse_emotion(
                input_spec.emotion,
                input_spec.emotion_intensity,
            )
            expression_params = self.expression_mapper.map_emotion(emotion_vector)

        # Determine backend
        backend = await self._determine_backend(input_spec.backend)
        logger.info(f"Using synthesis backend: {backend}")

        # Stage 4: Voice Synthesis (main stage) - route to appropriate backend
        if backend == "self_hosted":
            engine_output = await self._synthesize_selfhosted(input_spec, lyric_tokens)
        elif backend == "elevenlabs":
            engine_output = await self._synthesize_elevenlabs(input_spec, lyric_tokens)
        else:
            # Internal engine (default)
            self._report_progress(PipelineStage.VOICE_SYNTHESIS, 40, "Synthesizing voice...")

            engine_input = SingingEngineInput(
                lyrics=lyric_tokens,
                melody=self._convert_melody(input_spec.melody),
                tempo=input_spec.tempo,
                genre=input_spec.genre,
            )

            engine_output = await self.engine.synthesize(engine_input)

        # Stage 5: Harmony Generation (if requested)
        harmony_audio = None
        if input_spec.generate_harmonies:
            self._report_progress(
                PipelineStage.HARMONY_GENERATION, 60,
                "Generating harmonies..."
            )
            harmony_audio = await self._generate_harmonies(
                input_spec.melody,
                engine_output.audio,
                input_spec.key_root,
                input_spec.scale_type,
            )

        # Stage 6: Arrangement
        self._report_progress(PipelineStage.ARRANGEMENT, 75, "Arranging vocals...")
        arranged_audio = await self._apply_arrangement(
            engine_output.audio,
            harmony_audio,
            input_spec,
        )

        # Check safeguards
        melody_pitches = [m.get("pitch", 60) for m in input_spec.melody]
        melody_range = (min(melody_pitches), max(melody_pitches))

        safeguard_result = self.safeguards.analyze(
            layers=[{"name": "lead", "pitch_offset": 0}],
            melody_range=melody_range,
        )
        if not safeguard_result.passed:
            for issue in safeguard_result.issues:
                warnings.append(issue.description)

        # Stage 7: Quality Check
        quality_score = None
        quality_passed = True

        if input_spec.quality_check:
            self._report_progress(
                PipelineStage.QUALITY_CHECK, 90,
                "Evaluating quality..."
            )

            release_stage = ReleaseStage[input_spec.release_stage.upper()]
            self.quality_evaluator.release_stage = release_stage

            eval_result = self.quality_evaluator.evaluate(
                generated_audio=arranged_audio,
                generated_pitch=engine_output.pitch_contour,
                target_pitch=np.array([m.get("pitch", 60) for m in input_spec.melody]),
                generated_onsets=[],
                target_onsets=[],
                generated_phonemes=engine_output.phoneme_sequence,
                target_phonemes=[],
            )

            quality_score = eval_result.score
            quality_passed = eval_result.passed

            if not quality_passed:
                for reason in eval_result.metrics_summary.failure_reasons:
                    warnings.append(f"Quality: {reason}")

        # Stage 8: Output
        self._report_progress(PipelineStage.OUTPUT, 100, "Complete!")

        duration_seconds = len(arranged_audio) / self.sample_rate

        return VoiceSynthesisOutput(
            lead_audio=engine_output.audio,
            harmony_audio=harmony_audio,
            mixed_audio=arranged_audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds,
            pitch_contour=engine_output.pitch_contour,
            phoneme_sequence=engine_output.phoneme_sequence,
            quality_score=quality_score,
            quality_passed=quality_passed,
            quality_warnings=warnings,
        )

    def _validate_input(self, input_spec: VoiceSynthesisInput) -> List[str]:
        """Validate input specification."""
        errors = []

        if not input_spec.lyrics or not input_spec.lyrics.strip():
            errors.append("Lyrics cannot be empty")

        if not input_spec.melody:
            errors.append("Melody cannot be empty")

        if input_spec.tempo <= 0:
            errors.append("Tempo must be positive")

        if input_spec.language not in ["en", "es", "mixed"]:
            errors.append(f"Unsupported language: {input_spec.language}")

        return errors

    async def _process_lyrics(
        self,
        lyrics: str,
        language: str,
    ) -> List[dict]:
        """Process lyrics into phonetic tokens."""
        words = lyrics.split()
        tokens = []

        for word in words:
            if language == "en":
                phonemes = self.english_phonetics.text_to_phonemes(word)
            elif language == "es":
                phonemes = self.spanish_phonetics.text_to_phonemes(word)
            elif language == "mixed":
                detected = self.bilingual.detect_language(word)
                if detected == "es":
                    phonemes = self.spanish_phonetics.text_to_phonemes(word)
                else:
                    phonemes = self.english_phonetics.text_to_phonemes(word)
            else:
                phonemes = []

            tokens.append({
                "text": word,
                "phonemes": phonemes,
                "language": language,
            })

        return tokens

    def _convert_melody(self, melody: List[dict]) -> List:
        """Convert melody dict to engine format."""
        # Create simple melody note objects
        class MelodyNote:
            def __init__(self, data: dict):
                self.pitch = data.get("pitch", 60)
                self.start_beat = data.get("start_beat", data.get("start", 0))
                self.duration_beats = data.get("duration_beats", data.get("duration", 1))
                self.velocity = data.get("velocity", 100)

        return [MelodyNote(m) for m in melody]

    def _parse_emotion(
        self,
        emotion: str,
        intensity: float,
    ) -> EmotionVector:
        """Parse emotion string to vector."""
        emotion_map = {
            "happy": EmotionVector(valence=0.8, arousal=0.6, dominance=0.3, tension=0.2),
            "sad": EmotionVector(valence=-0.6, arousal=-0.4, dominance=-0.5, tension=0.4),
            "angry": EmotionVector(valence=-0.7, arousal=0.8, dominance=0.8, tension=0.9),
            "calm": EmotionVector(valence=0.3, arousal=-0.6, dominance=0.0, tension=0.1),
            "excited": EmotionVector(valence=0.7, arousal=0.9, dominance=0.5, tension=0.4),
            "tender": EmotionVector(valence=0.6, arousal=-0.3, dominance=-0.2, tension=0.2),
        }

        base = emotion_map.get(emotion.lower(), EmotionVector())

        # Scale by intensity
        return EmotionVector(
            valence=base.valence * intensity,
            arousal=base.arousal * intensity,
            dominance=base.dominance * intensity,
            tension=base.tension * intensity,
        )

    async def _generate_harmonies(
        self,
        melody: List[dict],
        lead_audio: np.ndarray,
        key_root: int,
        scale_type: str,
    ) -> np.ndarray:
        """Generate harmony vocals."""
        self.harmony_generator.harmonizer.root = key_root
        scale_map = {
            "major": self.harmony_generator.harmonizer.MAJOR_SCALE,
            "minor": self.harmony_generator.harmonizer.MINOR_SCALE,
        }
        self.harmony_generator.harmonizer.scale = scale_map.get(
            scale_type,
            self.harmony_generator.harmonizer.MAJOR_SCALE,
        )

        # Generate harmony voices
        voices = self.harmony_generator.generate_harmony(
            melody,
            num_voices=2,
            style="close",
        )

        # For now, create placeholder harmony audio
        # Real implementation would synthesize each voice
        harmony_audio = np.zeros_like(lead_audio)

        # Simulate harmony by pitch shifting (placeholder)
        for voice in voices:
            shifted = lead_audio.copy()
            # Placeholder - real implementation would resynthesize
            harmony_audio += shifted * 0.3

        return harmony_audio

    async def _apply_arrangement(
        self,
        lead_audio: np.ndarray,
        harmony_audio: Optional[np.ndarray],
        input_spec: VoiceSynthesisInput,
    ) -> np.ndarray:
        """Apply arrangement and mixing."""
        # Start with lead
        mixed = lead_audio.copy()

        # Add harmonies if present
        if harmony_audio is not None:
            # Apply volume scaling
            harmony_volume = 0.5 * input_spec.arrangement_density
            mixed = mixed + harmony_audio * harmony_volume

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * (0.95 / max_val)

        return mixed

    async def _synthesize_selfhosted(
        self,
        input_spec: VoiceSynthesisInput,
        lyric_tokens: List[dict],
    ):
        """
        Synthesize using self-hosted XTTS + RVC pipeline.

        This provides the highest quality synthesis with full
        voice conversion to match AVU identities.
        """
        from dataclasses import dataclass

        @dataclass
        class EngineOutput:
            audio: np.ndarray
            pitch_contour: Optional[np.ndarray] = None
            phoneme_sequence: Optional[List[str]] = None

        provider = await self.get_selfhosted_provider()
        if not provider:
            raise RuntimeError("Self-hosted provider not available")

        # Stage: XTTS Generation
        self._report_progress(
            PipelineStage.XTTS_GENERATION, 40,
            f"Generating speech with XTTS for {input_spec.voice_name}..."
        )

        # Convert lyric tokens to text
        text = " ".join([t.get("text", "") for t in lyric_tokens])

        # Define progress callback for provider
        def progress_callback(progress: float, stage: str, message: str):
            if stage == "xtts":
                self._report_progress(
                    PipelineStage.XTTS_GENERATION,
                    40 + (progress * 0.15),  # 40-55%
                    message,
                )
            elif stage == "rvc":
                self._report_progress(
                    PipelineStage.RVC_CONVERSION,
                    55 + (progress * 0.15),  # 55-70%
                    message,
                )

        provider.set_progress_callback(progress_callback)

        # Run synthesis
        result = await provider.synthesize(
            text=text,
            voice_name=input_spec.voice_name,
            language=input_spec.language,
            pitch_shift=input_spec.pitch_shift,
            speed=1.0,  # Speed is typically handled by melody alignment
        )

        # Stage: RVC Conversion (completed by provider)
        self._report_progress(
            PipelineStage.RVC_CONVERSION, 70,
            "Voice conversion complete"
        )

        # Load audio from result
        if result and result.audio_path:
            import soundfile as sf
            audio_data, sr = sf.read(str(result.audio_path))

            # Resample if needed
            if sr != self.sample_rate:
                import scipy.signal as signal
                num_samples = int(len(audio_data) * self.sample_rate / sr)
                audio_data = signal.resample(audio_data, num_samples)

            # Ensure mono and float32
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            audio_data = audio_data.astype(np.float32)

            return EngineOutput(
                audio=audio_data,
                pitch_contour=None,  # Self-hosted doesn't provide this yet
                phoneme_sequence=[t.get("text", "") for t in lyric_tokens],
            )

        # Fallback: generate placeholder if result failed
        logger.warning("Self-hosted synthesis returned no audio, using placeholder")
        duration_samples = int(5.0 * self.sample_rate)
        return EngineOutput(
            audio=np.zeros(duration_samples, dtype=np.float32),
            pitch_contour=None,
            phoneme_sequence=[t.get("text", "") for t in lyric_tokens],
        )

    async def _synthesize_elevenlabs(
        self,
        input_spec: VoiceSynthesisInput,
        lyric_tokens: List[dict],
    ):
        """
        Synthesize using ElevenLabs API.

        High-quality text-to-speech with voice matching.
        """
        from dataclasses import dataclass

        @dataclass
        class EngineOutput:
            audio: np.ndarray
            pitch_contour: Optional[np.ndarray] = None
            phoneme_sequence: Optional[List[str]] = None

        provider = await self.get_elevenlabs_provider()
        if not provider:
            raise RuntimeError("ElevenLabs provider not available")

        # Stage: Voice Synthesis
        self._report_progress(
            PipelineStage.VOICE_SYNTHESIS, 40,
            f"Synthesizing with ElevenLabs for {input_spec.voice_name}..."
        )

        # Convert lyric tokens to text
        text = " ".join([t.get("text", "") for t in lyric_tokens])

        # Run synthesis
        result = await provider.synthesize_text(
            text=text,
            voice_name=input_spec.voice_name,
        )

        self._report_progress(
            PipelineStage.VOICE_SYNTHESIS, 70,
            "ElevenLabs synthesis complete"
        )

        # Load audio from result
        if result:
            import soundfile as sf
            audio_data, sr = sf.read(str(result))

            # Resample if needed
            if sr != self.sample_rate:
                import scipy.signal as signal
                num_samples = int(len(audio_data) * self.sample_rate / sr)
                audio_data = signal.resample(audio_data, num_samples)

            # Ensure mono and float32
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            audio_data = audio_data.astype(np.float32)

            return EngineOutput(
                audio=audio_data,
                pitch_contour=None,
                phoneme_sequence=[t.get("text", "") for t in lyric_tokens],
            )

        # Fallback
        logger.warning("ElevenLabs synthesis returned no audio, using placeholder")
        duration_samples = int(5.0 * self.sample_rate)
        return EngineOutput(
            audio=np.zeros(duration_samples, dtype=np.float32),
            pitch_contour=None,
            phoneme_sequence=[t.get("text", "") for t in lyric_tokens],
        )


class BatchSynthesisPipeline:
    """
    Batch processing pipeline for multiple songs.
    """

    def __init__(
        self,
        max_concurrent: int = 4,
    ):
        """
        Initialize batch pipeline.

        Args:
            max_concurrent: Maximum concurrent syntheses
        """
        self.max_concurrent = max_concurrent
        self.base_pipeline = VoiceSynthesisPipeline()

    async def synthesize_batch(
        self,
        inputs: List[VoiceSynthesisInput],
        progress_callback: Optional[callable] = None,
    ) -> List[VoiceSynthesisOutput]:
        """
        Synthesize multiple songs.

        Args:
            inputs: List of input specifications
            progress_callback: Optional progress callback

        Returns:
            List of outputs
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_one(idx: int, input_spec: VoiceSynthesisInput):
            async with semaphore:
                if progress_callback:
                    progress_callback(f"Processing {idx + 1}/{len(inputs)}")
                return await self.base_pipeline.synthesize(input_spec)

        tasks = [
            process_one(i, inp)
            for i, inp in enumerate(inputs)
        ]

        return await asyncio.gather(*tasks)
