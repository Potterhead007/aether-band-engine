"""
RVC Voice Conversion Engine

Retrieval-based Voice Conversion for matching AVU voice identities.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import RVCConfig, AVU_VOICE_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class RVCVoiceModel:
    """Loaded RVC voice model."""

    name: str
    model_path: Path
    index_path: Optional[Path] = None
    pitch_shift: int = 0
    loaded: bool = False


class RVCEngine:
    """
    RVC (Retrieval-based Voice Conversion) engine.

    Converts any voice to match AVU voice identities using
    pre-trained RVC models. Each AVU voice has its own model
    trained on reference recordings.
    """

    def __init__(self, config: RVCConfig):
        self.config = config
        self._device: Optional[str] = None
        self._pipeline = None

        # Discovered voice models
        self._voice_models: dict[str, RVCVoiceModel] = {}

        # Loaded model state
        self._current_model: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """Check if RVC pipeline is initialized."""
        return self._pipeline is not None

    @property
    def device(self) -> str:
        """Get current device."""
        return self._device or "cpu"

    def _detect_device(self) -> str:
        """Detect best available compute device."""
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    async def load(self) -> bool:
        """
        Initialize RVC pipeline and discover voice models.

        Returns:
            True if initialization successful
        """
        try:
            self._device = self._detect_device()
            logger.info(f"Initializing RVC engine on {self._device}...")

            # Discover voice models
            await self._discover_models()

            if not self._voice_models:
                logger.warning("No RVC voice models found")
                return False

            # Initialize pipeline (lazy - actual model loading happens on first convert)
            self._pipeline = True  # Placeholder for actual pipeline

            logger.info(f"RVC engine initialized with {len(self._voice_models)} voice models")
            return True

        except ImportError as e:
            logger.error(f"RVC dependencies not installed: {e}")
            logger.error("Install with: pip install rvc-python faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RVC engine: {e}")
            return False

    async def _discover_models(self) -> None:
        """Discover RVC models in the configured directory."""
        if not self.config.models_dir:
            logger.warning("No RVC models directory configured")
            return

        if not self.config.models_dir.exists():
            logger.warning(f"RVC models directory does not exist: {self.config.models_dir}")
            return

        # Look for AVU voice model directories
        for voice_name in AVU_VOICE_CONFIGS.keys():
            voice_dir = self.config.models_dir / voice_name

            if not voice_dir.exists():
                logger.debug(f"No RVC model directory for {voice_name}")
                continue

            # Look for model file
            model_path = None
            for pattern in ["model.pth", "*.pth"]:
                matches = list(voice_dir.glob(pattern))
                if matches:
                    model_path = matches[0]
                    break

            if not model_path:
                logger.warning(f"No .pth model file found in {voice_dir}")
                continue

            # Look for index file (optional but improves quality)
            index_path = None
            for pattern in ["model.index", "*.index"]:
                matches = list(voice_dir.glob(pattern))
                if matches:
                    index_path = matches[0]
                    break

            # Get pitch shift from config
            voice_config = AVU_VOICE_CONFIGS.get(voice_name)
            pitch_shift = voice_config.rvc_pitch_shift if voice_config else 0

            # Register model
            self._voice_models[voice_name] = RVCVoiceModel(
                name=voice_name,
                model_path=model_path,
                index_path=index_path,
                pitch_shift=pitch_shift,
            )

            logger.info(f"Discovered RVC model for {voice_name}: {model_path.name}")

    async def unload(self) -> None:
        """Unload RVC models to free GPU memory."""
        self._pipeline = None
        self._current_model = None

        # Clear GPU cache
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("RVC engine unloaded")

    async def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_voice: str,
        pitch_shift: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """
        Convert audio to target AVU voice.

        Args:
            audio: Input audio (float32, mono)
            sample_rate: Input sample rate
            target_voice: Target AVU voice name (e.g., "AVU-1")
            pitch_shift: Optional pitch shift override (semitones)
            progress_callback: Optional callback(progress_pct, message)

        Returns:
            Converted audio (float32, mono)

        Raises:
            RuntimeError: If engine not loaded
            ValueError: If target voice not available
        """
        if not self._pipeline:
            raise RuntimeError("RVC engine not loaded. Call load() first.")

        if target_voice not in self._voice_models:
            available = list(self._voice_models.keys())
            raise ValueError(f"Voice '{target_voice}' not available. Available: {available}")

        voice_model = self._voice_models[target_voice]

        # Resolve pitch shift
        actual_pitch_shift = pitch_shift if pitch_shift is not None else voice_model.pitch_shift

        if progress_callback:
            progress_callback(10, f"Loading {target_voice} voice model...")

        # Load model if not current
        if self._current_model != target_voice:
            await self._load_voice_model(target_voice)

        if progress_callback:
            progress_callback(30, "Running voice conversion...")

        # Run conversion in executor
        loop = asyncio.get_event_loop()
        converted = await loop.run_in_executor(
            None,
            self._convert_sync,
            audio,
            sample_rate,
            voice_model,
            actual_pitch_shift,
        )

        if progress_callback:
            progress_callback(100, "Voice conversion complete")

        return converted

    async def _load_voice_model(self, voice_name: str) -> None:
        """Load a specific voice model."""
        voice_model = self._voice_models[voice_name]

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync, voice_model)

        self._current_model = voice_name
        voice_model.loaded = True
        logger.info(f"Loaded RVC model for {voice_name}")

    def _load_model_sync(self, voice_model: RVCVoiceModel) -> None:
        """Synchronous model loading."""
        # This is a placeholder for actual RVC model loading
        # In production, would use rvc-python or fairseq
        try:
            # Try to import RVC
            # from rvc_python import RVC
            # self._rvc = RVC(voice_model.model_path, device=self._device)
            logger.debug(f"Would load model from {voice_model.model_path}")
        except ImportError:
            logger.warning("rvc-python not installed, using passthrough mode")

    def _convert_sync(
        self,
        audio: np.ndarray,
        sample_rate: int,
        voice_model: RVCVoiceModel,
        pitch_shift: int,
    ) -> np.ndarray:
        """
        Synchronous voice conversion.

        This is a placeholder implementation. In production:
        1. Resample audio to RVC expected sample rate (usually 16000 or 48000)
        2. Run through RVC model
        3. Apply pitch shift
        4. Resample back to target sample rate
        """
        try:
            # Try to use actual RVC if available
            # converted = self._rvc.convert(
            #     audio,
            #     sample_rate,
            #     pitch_shift=pitch_shift,
            #     f0_method=self.config.f0_method,
            #     index_rate=self.config.index_rate,
            #     filter_radius=self.config.filter_radius,
            #     rms_mix_rate=self.config.rms_mix_rate,
            #     protect=self.config.protect,
            #     index_path=str(voice_model.index_path) if voice_model.index_path else None,
            # )
            # return converted

            # Placeholder: return audio with simulated processing
            # Apply simple pitch shift approximation
            if pitch_shift != 0:
                audio = self._simple_pitch_shift(audio, sample_rate, pitch_shift)

            logger.debug(f"RVC conversion for {voice_model.name} (pitch_shift={pitch_shift})")
            return audio

        except Exception as e:
            logger.error(f"RVC conversion failed: {e}")
            return audio  # Return original on error

    def _simple_pitch_shift(
        self,
        audio: np.ndarray,
        sample_rate: int,
        semitones: int,
    ) -> np.ndarray:
        """
        Simple pitch shift using resampling.

        This is a basic approximation. For production quality,
        use librosa or RVC's internal pitch shifting.
        """
        try:
            import scipy.signal as signal

            # Pitch shift ratio
            ratio = 2 ** (semitones / 12)

            # Resample to shift pitch
            new_length = int(len(audio) / ratio)
            shifted = signal.resample(audio, new_length)

            # Resample back to original length (changes tempo but keeps pitch)
            shifted = signal.resample(shifted, len(audio))

            return shifted.astype(np.float32)

        except ImportError:
            logger.warning("scipy not available for pitch shift")
            return audio

    def get_available_voices(self) -> list[str]:
        """Get list of available voice models."""
        return list(self._voice_models.keys())

    def get_voice_info(self, voice_name: str) -> Optional[dict]:
        """Get information about a voice model."""
        if voice_name not in self._voice_models:
            return None

        model = self._voice_models[voice_name]
        return {
            "name": model.name,
            "model_path": str(model.model_path),
            "index_path": str(model.index_path) if model.index_path else None,
            "pitch_shift": model.pitch_shift,
            "loaded": model.loaded,
        }
