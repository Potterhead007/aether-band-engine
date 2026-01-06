"""
XTTS Text-to-Speech Engine

Coqui XTTS integration for high-quality multilingual speech synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .config import XTTSConfig, AVU_VOICE_CONFIGS

logger = logging.getLogger(__name__)


# Map AVU voices to XTTS built-in speakers
# XTTS has 58 built-in speakers with pre-computed embeddings
AVU_TO_XTTS_SPEAKER = {
    "AVU-1": "Viktor Menelaos",     # Male Lyric Tenor - warm and expressive
    "AVU-2": "Claribel Dervla",     # Female Mezzo-Soprano - smooth and warm
    "AVU-3": "Damien Black",        # Male Baritone - deep and powerful
    "AVU-4": "Henriette Usha",      # Female Soprano - bright and agile
}


def _patch_torch_load():
    """Patch torch.load for PyTorch 2.9+ compatibility with TTS models."""
    try:
        import torch
        if hasattr(torch, "_original_load"):
            return  # Already patched

        original_load = torch.load
        torch._original_load = original_load

        def patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load
        logger.debug("Patched torch.load for TTS compatibility")
    except ImportError:
        pass


class XTTSEngine:
    """
    Coqui XTTS text-to-speech engine.

    Generates speech from text with speaker conditioning support.
    Can use pre-computed speaker embeddings or reference audio for voice cloning.
    """

    def __init__(self, config: XTTSConfig):
        self.config = config
        self._model = None
        self._device: Optional[str] = None
        self._sample_rate: int = 24000  # XTTS native sample rate

        # Speaker embedding cache
        self._speaker_embeddings: dict[str, np.ndarray] = {}
        self._gpt_cond_latents: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get current device."""
        return self._device or "cpu"

    @property
    def sample_rate(self) -> int:
        """Get output sample rate."""
        return self._sample_rate

    def _detect_device(self) -> str:
        """Detect best available compute device."""
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch

            if torch.cuda.is_available():
                logger.info("CUDA available, using GPU")
                return "cuda"
            # Note: MPS (Apple Silicon) has attention mask issues with XTTS/transformers
            # Use CPU for better compatibility
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("MPS available but using CPU for XTTS compatibility")
                return "cpu"  # MPS has issues with transformers attention masks
        except ImportError:
            pass

        logger.info("Using CPU for XTTS synthesis")
        return "cpu"

    async def load(self) -> bool:
        """
        Load XTTS model.

        Returns:
            True if model loaded successfully
        """
        if self._model is not None:
            logger.debug("XTTS model already loaded")
            return True

        try:
            # Apply PyTorch 2.9+ compatibility patch
            _patch_torch_load()

            # Detect device
            self._device = self._detect_device()
            logger.info(f"Loading XTTS model on {self._device}...")

            # Run model loading in executor (blocking operation)
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(None, self._load_model_sync)

            if self._model is None:
                return False

            # Load speaker embeddings if directory specified
            if self.config.speaker_embeddings_dir:
                await self._load_speaker_embeddings()

            logger.info("XTTS model loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"TTS library not installed: {e}")
            logger.error("Install with: pip install TTS torch torchaudio")
            return False
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            return False

    def _load_model_sync(self):
        """Synchronous model loading (run in executor)."""
        try:
            import warnings
            warnings.filterwarnings("ignore")

            from TTS.api import TTS

            # Load model
            if self.config.model_path:
                # Custom model path
                model = TTS(
                    model_path=str(self.config.model_path / "model.pth"),
                    config_path=str(self.config.model_path / "config.json"),
                )
            else:
                # Download/use cached model by name (XTTS v2)
                model = TTS(self.config.model_name)

            # Note: Don't call .to(device) here - TTS API handles device internally
            # and calling .to() on MPS causes attention mask issues
            logger.info(f"XTTS model loaded: {self.config.model_name}")

            return model

        except Exception as e:
            logger.error(f"Error loading XTTS model: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _load_speaker_embeddings(self) -> None:
        """Load pre-computed speaker embeddings for AVU voices."""
        if not self.config.speaker_embeddings_dir:
            return

        embeddings_dir = self.config.speaker_embeddings_dir

        for voice_name in AVU_VOICE_CONFIGS.keys():
            embedding_path = embeddings_dir / f"{voice_name}.npy"
            if embedding_path.exists():
                try:
                    embedding = np.load(str(embedding_path))
                    self._speaker_embeddings[voice_name] = embedding
                    logger.debug(f"Loaded speaker embedding for {voice_name}")
                except Exception as e:
                    logger.warning(f"Failed to load embedding for {voice_name}: {e}")

        logger.info(f"Loaded {len(self._speaker_embeddings)} speaker embeddings")

    async def unload(self) -> None:
        """Unload model to free GPU memory."""
        if self._model:
            del self._model
            self._model = None

        self._speaker_embeddings.clear()
        self._gpt_cond_latents.clear()

        # Clear GPU cache
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

        logger.info("XTTS model unloaded")

    async def synthesize(
        self,
        text: str,
        voice_name: Optional[str] = None,
        speaker_wav: Optional[Path] = None,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            voice_name: AVU voice name (e.g., "AVU-1") for speaker conditioning
            speaker_wav: Path to reference audio for voice cloning
            language: Language code (default from config)
            temperature: Generation temperature (default from config)
            progress_callback: Optional callback(progress_pct, message)

        Returns:
            Audio as numpy array (float32, mono)

        Raises:
            RuntimeError: If model not loaded
        """
        if not self._model:
            raise RuntimeError("XTTS model not loaded. Call load() first.")

        # Resolve speaker conditioning
        # First, map AVU voice to XTTS built-in speaker
        xtts_speaker = None
        voice_config = None

        if voice_name and voice_name in AVU_TO_XTTS_SPEAKER:
            xtts_speaker = AVU_TO_XTTS_SPEAKER[voice_name]
            logger.debug(f"Mapped {voice_name} to XTTS speaker: {xtts_speaker}")

        if voice_name and voice_name in AVU_VOICE_CONFIGS:
            voice_config = AVU_VOICE_CONFIGS[voice_name]

        # Fall back to speaker wav if no built-in speaker mapping
        actual_speaker_wav = speaker_wav
        if not xtts_speaker and voice_config and voice_config.xtts_speaker_wav:
            actual_speaker_wav = voice_config.xtts_speaker_wav

        # Resolve parameters
        actual_language = language or (voice_config.xtts_language if voice_config else self.config.language)

        if progress_callback:
            progress_callback(10, "Preparing synthesis...")

        # Run synthesis in executor (blocking operation)
        loop = asyncio.get_event_loop()

        def _synthesize_sync():
            if xtts_speaker:
                # Use XTTS built-in speaker (from speakers_xtts.pth)
                logger.info(f"Synthesizing with XTTS speaker: {xtts_speaker}")
                outputs = self._model.tts(
                    text=text,
                    speaker=xtts_speaker,
                    language=actual_language,
                )
            elif actual_speaker_wav and actual_speaker_wav.exists():
                # Use reference audio for speaker conditioning (voice cloning)
                logger.info(f"Synthesizing with reference audio: {actual_speaker_wav}")
                outputs = self._model.tts(
                    text=text,
                    speaker_wav=str(actual_speaker_wav),
                    language=actual_language,
                )
            else:
                # Use default built-in speaker
                default_speaker = "Claribel Dervla"
                logger.warning(f"No speaker specified, using default: {default_speaker}")
                outputs = self._model.tts(
                    text=text,
                    speaker=default_speaker,
                    language=actual_language,
                )

            return outputs

        if progress_callback:
            progress_callback(30, "Running XTTS inference...")

        audio = await loop.run_in_executor(None, _synthesize_sync)

        if progress_callback:
            progress_callback(90, "Post-processing...")

        # Convert to numpy array if needed
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        if progress_callback:
            progress_callback(100, "XTTS synthesis complete")

        return audio

    async def compute_speaker_embedding(
        self,
        audio_path: Path,
        voice_name: str,
    ) -> np.ndarray:
        """
        Compute and cache speaker embedding from reference audio.

        Args:
            audio_path: Path to reference audio file
            voice_name: Name to cache embedding under

        Returns:
            Speaker embedding as numpy array
        """
        if not self._model:
            raise RuntimeError("XTTS model not loaded")

        loop = asyncio.get_event_loop()

        def _compute_sync():
            # XTTS v2 computes latents from reference audio
            gpt_cond_latent, speaker_embedding = self._model.synthesizer.tts_model.get_conditioning_latents(
                audio_path=[str(audio_path)]
            )
            return speaker_embedding.cpu().numpy(), gpt_cond_latent

        embedding, latent = await loop.run_in_executor(None, _compute_sync)

        # Cache for later use
        self._speaker_embeddings[voice_name] = embedding
        self._gpt_cond_latents[voice_name] = latent

        return embedding

    async def save_speaker_embedding(
        self,
        voice_name: str,
        output_path: Path,
    ) -> None:
        """Save cached speaker embedding to file."""
        if voice_name not in self._speaker_embeddings:
            raise ValueError(f"No embedding cached for {voice_name}")

        np.save(str(output_path), self._speaker_embeddings[voice_name])
        logger.info(f"Saved speaker embedding to {output_path}")

    def get_available_voices(self) -> list[str]:
        """Get list of available AVU voices (mapped to XTTS speakers)."""
        return list(AVU_TO_XTTS_SPEAKER.keys())

    def get_xtts_speakers(self) -> list[str]:
        """Get list of all XTTS built-in speakers."""
        return list(AVU_TO_XTTS_SPEAKER.values())

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        # XTTS v2 supported languages
        return [
            "en",  # English
            "es",  # Spanish
            "fr",  # French
            "de",  # German
            "it",  # Italian
            "pt",  # Portuguese
            "pl",  # Polish
            "tr",  # Turkish
            "ru",  # Russian
            "nl",  # Dutch
            "cs",  # Czech
            "ar",  # Arabic
            "zh-cn",  # Chinese
            "ja",  # Japanese
            "hu",  # Hungarian
            "ko",  # Korean
        ]
