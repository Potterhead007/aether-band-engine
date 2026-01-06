"""
XTTS Subprocess Wrapper

Runs XTTS synthesis in a subprocess using the Python 3.11 voice venv.
This allows the main AETHER app (Python 3.9) to use XTTS (requires Python 3.10+).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the Python 3.11 voice venv
VOICE_VENV_PYTHON = Path(__file__).parents[4] / ".venv-voice" / "bin" / "python"


def is_voice_venv_available() -> bool:
    """Check if the voice venv with TTS is available."""
    if not VOICE_VENV_PYTHON.exists():
        return False

    # Quick check if TTS is importable
    try:
        result = subprocess.run(
            [str(VOICE_VENV_PYTHON), "-c", "import TTS; print('ok')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


async def synthesize_xtts(
    text: str,
    voice_name: str = "AVU-1",
    language: str = "en",
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Synthesize speech using XTTS in a subprocess.

    Args:
        text: Text to synthesize
        voice_name: AVU voice name (AVU-1 to AVU-4)
        language: Language code (default: en)
        output_path: Output WAV file path (generated if not provided)

    Returns:
        Path to generated audio file, or None on failure
    """
    if not VOICE_VENV_PYTHON.exists():
        logger.error(f"Voice venv not found: {VOICE_VENV_PYTHON}")
        return None

    # Generate output path if not provided
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".wav", prefix="xtts_"))

    # XTTS speaker mappings (built-in speakers)
    speaker_map = {
        "AVU-1": "Viktor Menelaos",
        "AVU-2": "Claribel Dervla",
        "AVU-3": "Damien Black",
        "AVU-4": "Henriette Usha",
    }

    xtts_speaker = speaker_map.get(voice_name, "Claribel Dervla")

    # Escape the text for Python string
    escaped_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

    # Create the synthesis script
    script = f'''
import warnings
warnings.filterwarnings("ignore")

import torch
# Patch torch.load for PyTorch 2.9+ compatibility
original_load = torch.load
def patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from TTS.api import TTS
import sys

# Load model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Synthesize
tts.tts_to_file(
    text="{escaped_text}",
    speaker="{xtts_speaker}",
    language="{language}",
    file_path="{output_path}",
)

print("SUCCESS")
'''

    try:
        process = await asyncio.create_subprocess_exec(
            str(VOICE_VENV_PYTHON),
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=120.0  # 2 minute timeout for synthesis
        )

        if process.returncode != 0:
            logger.error(f"XTTS synthesis failed: {stderr.decode()}")
            return None

        if "SUCCESS" not in stdout.decode():
            logger.error(f"XTTS synthesis did not complete successfully")
            return None

        if not output_path.exists():
            logger.error(f"XTTS output file not created: {output_path}")
            return None

        logger.info(f"XTTS synthesis complete: {output_path}")
        return output_path

    except asyncio.TimeoutError:
        logger.error("XTTS synthesis timed out")
        return None
    except Exception as e:
        logger.error(f"XTTS synthesis error: {e}")
        return None


async def generate_xtts_preview(
    voice_name: str,
    preview_type: str = "default",
    custom_params: Optional[dict] = None,
) -> Optional[Path]:
    """
    Generate a voice preview using XTTS.

    Args:
        voice_name: AVU voice name (AVU-1 to AVU-4)
        preview_type: Type of preview (default, emotional, range)
        custom_params: Optional custom parameters (not fully supported yet)

    Returns:
        Path to generated audio file, or None on failure
    """
    # Preview phrases
    phrases = {
        "default": {
            "AVU-1": "Feel the warmth of my voice, rich and expressive, ready to bring your music to life.",
            "AVU-2": "Smooth and versatile, this voice flows like honey through every melody.",
            "AVU-3": "Deep resonance and power define this voice, grounded and commanding.",
            "AVU-4": "Crystal clear and agile, this voice sparkles with brilliance and precision.",
        },
        "emotional": {
            "AVU-1": "Every note carries emotion, from whispered intimacy to soaring power.",
            "AVU-2": "With warmth and sincerity, I'll capture the heart of every song.",
            "AVU-3": "Strength meets sensitivity in every phrase I deliver.",
            "AVU-4": "Light as air yet full of feeling, every high note shines.",
        },
        "range": {
            "AVU-1": "From the depths of my range to the heights, my voice tells your story.",
            "AVU-2": "Whether soft and intimate or bold and commanding, I adapt to your vision.",
            "AVU-3": "Rich bass tones and warm midrange create an unforgettable presence.",
            "AVU-4": "From gentle whispers to soaring heights, clarity is my signature.",
        },
    }

    # Get text for this voice and preview type
    preview_phrases = phrases.get(preview_type, phrases["default"])
    text = preview_phrases.get(voice_name, preview_phrases.get("AVU-1", "Testing voice synthesis."))

    # Generate output path with caching
    import hashlib
    cache_key = hashlib.md5(f"{voice_name}:{preview_type}".encode()).hexdigest()[:12]
    cache_dir = Path(tempfile.gettempdir()) / "aether_xtts_cache"
    cache_dir.mkdir(exist_ok=True)
    output_path = cache_dir / f"xtts_{cache_key}.wav"

    # Use cached if available
    if output_path.exists():
        logger.debug(f"Using cached XTTS preview: {output_path}")
        return output_path

    return await synthesize_xtts(
        text=text,
        voice_name=voice_name,
        language="en",
        output_path=output_path,
    )


# For direct testing
if __name__ == "__main__":
    import asyncio

    async def test():
        print(f"Voice venv available: {is_voice_venv_available()}")

        if is_voice_venv_available():
            for voice in ["AVU-1", "AVU-2", "AVU-3", "AVU-4"]:
                print(f"Testing {voice}...")
                result = await generate_xtts_preview(voice)
                if result:
                    print(f"  ✓ {result}")
                else:
                    print(f"  ✗ Failed")

    asyncio.run(test())
