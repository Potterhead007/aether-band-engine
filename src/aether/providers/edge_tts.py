"""
Edge TTS Provider - Free Microsoft Text-to-Speech.

No API key required. High-quality human voices.
"""

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EdgeVoiceMapping:
    """Maps AVU voices to Edge TTS voices."""
    voice_id: str  # Edge TTS voice ID
    description: str
    pitch_shift: int = 0  # Semitones


# Map AVU voices to high-quality Edge TTS voices
EDGE_VOICE_MAP: dict[str, EdgeVoiceMapping] = {
    "AVU-1": EdgeVoiceMapping(
        voice_id="en-US-GuyNeural",  # Male tenor
        description="Warm male voice with natural expression",
    ),
    "AVU-2": EdgeVoiceMapping(
        voice_id="en-US-JennyNeural",  # Female mezzo
        description="Warm female voice with rich tone",
    ),
    "AVU-3": EdgeVoiceMapping(
        voice_id="en-US-ChristopherNeural",  # Male baritone (deep)
        description="Deep male voice with authority",
    ),
    "AVU-4": EdgeVoiceMapping(
        voice_id="en-US-AriaNeural",  # Female soprano
        description="Bright female voice with clarity",
    ),
}

# Preview phrases for each voice type
PREVIEW_PHRASES = {
    "AVU-1": "I'm a lyric tenor voice, warm and expressive, perfect for emotional melodies.",
    "AVU-2": "I'm a mezzo-soprano, with warmth and depth that brings soul to every song.",
    "AVU-3": "I'm a baritone voice, rich and powerful, commanding presence in every note.",
    "AVU-4": "I'm a soprano voice, bright and clear, soaring through the highest registers.",
}


class EdgeTTSProvider:
    """Edge TTS provider for human voice synthesis."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or os.environ.get(
            "AETHER_VOICE_CACHE",
            "/tmp/aether_edge_tts_cache"
        ))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._available = False

    async def initialize(self) -> bool:
        """Check if edge-tts is available."""
        try:
            import edge_tts
            self._available = True
            logger.info("Edge TTS provider initialized successfully")
            return True
        except ImportError:
            logger.warning("edge-tts not installed")
            return False

    def is_available(self) -> bool:
        return self._available

    async def generate_preview(
        self,
        voice_name: str,
        preview_type: str = "default",
        custom_params: Optional[dict] = None,
    ) -> Optional[Path]:
        """Generate voice preview using Edge TTS."""
        import edge_tts

        mapping = EDGE_VOICE_MAP.get(voice_name)
        if not mapping:
            logger.error(f"Unknown voice: {voice_name}")
            return None

        # Get text based on preview type
        if preview_type == "emotional":
            text = f"With deep feeling and passion, {PREVIEW_PHRASES.get(voice_name, 'Testing voice synthesis.')}"
        elif preview_type == "range":
            text = f"Listen to my range. {PREVIEW_PHRASES.get(voice_name, 'Testing voice synthesis.')} From low to high."
        else:
            text = PREVIEW_PHRASES.get(voice_name, "Testing voice synthesis capabilities.")

        # Apply custom text if provided
        if custom_params and custom_params.get("text"):
            text = custom_params["text"]

        # Generate cache key
        cache_key = hashlib.md5(
            f"{voice_name}:{mapping.voice_id}:{text}:{custom_params}".encode()
        ).hexdigest()[:12]

        output_path = self.cache_dir / f"edge_{cache_key}.mp3"

        # Return cached if exists
        if output_path.exists():
            logger.info(f"Using cached Edge TTS preview: {output_path}")
            return output_path

        try:
            # Configure voice settings
            voice = mapping.voice_id
            rate = "+0%"
            pitch = "+0Hz"

            # Apply custom voice parameters if provided
            if custom_params:
                # Map timbre brightness to rate (-50% to +50%)
                if "brightness" in custom_params:
                    brightness = float(custom_params["brightness"])
                    rate_adj = int((brightness - 0.5) * 40)  # -20% to +20%
                    rate = f"{rate_adj:+d}%"

                # Map emotion to pitch adjustment
                if "warmth" in custom_params:
                    warmth = float(custom_params["warmth"])
                    pitch_adj = int((warmth - 0.5) * 20)  # -10Hz to +10Hz
                    pitch = f"{pitch_adj:+d}Hz"

            # Generate speech
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch,
            )

            await communicate.save(str(output_path))

            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Generated Edge TTS preview: {output_path}")
                return output_path
            else:
                logger.error("Edge TTS generated empty file")
                return None

        except Exception as e:
            logger.error(f"Edge TTS generation failed: {e}")
            return None

    async def synthesize_text(
        self,
        text: str,
        voice_name: str = "AVU-1",
        output_path: Optional[Path] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> Optional[Path]:
        """Synthesize arbitrary text to speech."""
        import edge_tts

        mapping = EDGE_VOICE_MAP.get(voice_name, EDGE_VOICE_MAP["AVU-1"])

        if output_path is None:
            cache_key = hashlib.md5(
                f"{voice_name}:{text}:{rate}:{pitch}".encode()
            ).hexdigest()[:12]
            output_path = self.cache_dir / f"edge_synth_{cache_key}.mp3"

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=mapping.voice_id,
                rate=rate,
                pitch=pitch,
            )
            await communicate.save(str(output_path))

            if output_path.exists() and output_path.stat().st_size > 0:
                return output_path
            return None

        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}")
            return None


# Singleton instance
_provider: Optional[EdgeTTSProvider] = None


async def get_edge_tts_provider() -> EdgeTTSProvider:
    """Get or create Edge TTS provider singleton."""
    global _provider
    if _provider is None:
        _provider = EdgeTTSProvider()
        await _provider.initialize()
    return _provider
