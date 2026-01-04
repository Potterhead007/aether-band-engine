"""
AETHER Rendering Engine

Converts pipeline specifications into actual MIDI and audio output.

The rendering engine bridges the gap between agent-generated specifications
and actual audio files by:
1. Converting harmony/melody specs to MIDI
2. Rendering MIDI to audio via providers
3. Applying mixing and mastering
4. Exporting final audio files
"""

from __future__ import annotations

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Tuple
from uuid import uuid4

from aether.providers import (
    get_provider_registry,
    MIDIFile,
    MIDITrack,
    MIDINote,
    AudioBuffer,
    AudioStem,
)
from aether.audio.mixing import MixingEngine
from aether.audio.mastering import MasteringChain

logger = logging.getLogger(__name__)


@dataclass
class RenderingConfig:
    """Configuration for the rendering engine."""

    sample_rate: int = 48000
    bit_depth: int = 24
    output_dir: Optional[Path] = None
    render_stems: bool = True
    apply_mastering: bool = True
    target_lufs: float = -14.0
    true_peak_ceiling: float = -1.0
    export_formats: List[str] = field(default_factory=lambda: ["wav", "mp3"])


@dataclass
class RenderingResult:
    """Result of rendering pipeline."""

    success: bool
    song_id: str
    midi_file: Optional[MIDIFile] = None
    audio_buffer: Optional[AudioBuffer] = None
    master_buffer: Optional[AudioBuffer] = None
    stems: Dict[str, AudioBuffer] = field(default_factory=dict)
    output_paths: Dict[str, Path] = field(default_factory=dict)
    duration_seconds: float = 0.0
    peak_db: float = 0.0
    loudness_lufs: float = 0.0
    errors: List[str] = field(default_factory=list)


class RenderingEngine:
    """
    Main rendering engine for converting specs to audio.

    Usage:
        engine = RenderingEngine(config)
        result = await engine.render(pipeline_output)
    """

    def __init__(self, config: Optional[RenderingConfig] = None):
        self.config = config or RenderingConfig()
        self._registry = get_provider_registry()

    @property
    def midi_provider(self):
        """Get MIDI provider from registry."""
        return self._registry.get("midi")

    @property
    def audio_provider(self):
        """Get Audio provider from registry."""
        return self._registry.get("audio")

    async def render(self, pipeline_output: Dict[str, Any]) -> RenderingResult:
        """
        Render pipeline output to audio.

        Args:
            pipeline_output: Dict containing all pipeline specs:
                - song_spec
                - harmony_spec
                - melody_spec
                - arrangement_spec
                - rhythm_spec
                - sound_design_spec
                - mix_spec
                - master_spec

        Returns:
            RenderingResult with generated audio data
        """
        song_id = pipeline_output.get("song_id", str(uuid4()))
        song_spec = pipeline_output.get("song_spec", {})

        result = RenderingResult(
            success=False,
            song_id=song_id,
        )

        try:
            # Step 1: Generate MIDI from specs
            logger.info("Step 1: Generating MIDI from specifications...")
            midi_file = await self._generate_midi(
                song_spec=song_spec,
                harmony_spec=pipeline_output.get("harmony_spec", {}),
                melody_spec=pipeline_output.get("melody_spec", {}),
                arrangement_spec=pipeline_output.get("arrangement_spec", {}),
                rhythm_spec=pipeline_output.get("rhythm_spec", {}),
            )
            result.midi_file = midi_file
            logger.info(f"Generated MIDI: {len(midi_file.tracks)} tracks")

            # Step 2: Render MIDI to audio stems
            logger.info("Step 2: Rendering MIDI to audio...")
            stems = await self._render_to_stems(midi_file)
            result.stems = stems
            logger.info(f"Rendered {len(stems)} stems")

            # Step 3: Mix stems
            logger.info("Step 3: Mixing stems...")
            mix_spec = pipeline_output.get("mix_spec", {})
            mixed = await self._mix_stems(stems, mix_spec)
            result.audio_buffer = mixed
            logger.info(f"Mixed audio: {mixed.data.shape}")

            # Step 4: Apply mastering
            if self.config.apply_mastering:
                logger.info("Step 4: Applying mastering...")
                master_spec = pipeline_output.get("master_spec", {})
                mastered = await self._apply_mastering(mixed, master_spec)
                result.master_buffer = mastered
                result.duration_seconds = mastered.data.shape[-1] / mastered.sample_rate
                logger.info(f"Mastered: {result.duration_seconds:.1f}s")
            else:
                result.master_buffer = mixed
                result.duration_seconds = mixed.data.shape[-1] / mixed.sample_rate

            # Step 5: Export files
            if self.config.output_dir:
                logger.info("Step 5: Exporting files...")
                result.output_paths = await self._export_files(
                    song_id=song_id,
                    title=song_spec.get("title", "Untitled"),
                    master=result.master_buffer,
                    stems=stems if self.config.render_stems else {},
                )

            result.success = True
            logger.info("Rendering complete!")

        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            result.errors.append(str(e))

        return result

    async def _generate_midi(
        self,
        song_spec: Dict,
        harmony_spec: Dict,
        melody_spec: Dict,
        arrangement_spec: Dict,
        rhythm_spec: Dict,
    ) -> MIDIFile:
        """Generate MIDI file from specifications."""
        if not self.midi_provider:
            raise RuntimeError("MIDI provider not available")

        # Extract key parameters
        bpm = song_spec.get("bpm", 120)
        time_sig = song_spec.get("time_signature", "4/4")
        if isinstance(time_sig, str):
            parts = time_sig.split("/")
            time_signature = (int(parts[0]), int(parts[1]))
        else:
            time_signature = tuple(time_sig) if time_sig else (4, 4)

        # Get key from song spec
        key_data = song_spec.get("key", {})
        key_root = key_data.get("root", "C")
        key_mode = key_data.get("mode", "minor")

        # Get progressions from harmony spec
        progressions = harmony_spec.get("progressions", [])
        progression_strings = []
        for prog in progressions:
            chords = prog.get("chords", [])
            chord_strings = []
            for chord in chords:
                root = chord.get("root", "C")
                quality = chord.get("quality", "major")
                quality_suffix = "m" if quality == "minor" else ""
                chord_strings.append(f"{root}{quality_suffix}")
            progression_strings.extend(chord_strings)

        if not progression_strings:
            # Default progression
            if key_mode == "minor":
                progression_strings = [f"{key_root}m", "Ab", "Eb", "Bb"]
            else:
                progression_strings = [key_root, "F", "G", "Am"]

        # Generate MIDI using provider
        midi_file = await self.midi_provider.generate_from_spec(
            harmony_spec={
                "progression": progression_strings[:4],  # Use first 4 chords
                "key": key_root,
                "mode": key_mode,
            },
            melody_spec={
                "contour": melody_spec.get("primary_hook", {})
                .get("contour", {})
                .get("contour_type", "arch"),
                "range_octaves": melody_spec.get("typical_range_octaves", 1.5),
            },
            rhythm_spec={
                "bpm": bpm,
                "time_signature": time_signature,
            },
            arrangement_spec={
                "sections": [
                    s.get("section_type", "verse")
                    for s in arrangement_spec.get("sections", [{"section_type": "verse"}])
                ],
            },
        )

        return midi_file

    async def _render_to_stems(
        self,
        midi_file: MIDIFile,
    ) -> Dict[str, AudioBuffer]:
        """
        Render MIDI tracks to audio stems.

        Uses parallel rendering for improved performance - all stems are
        rendered concurrently using asyncio.gather().
        """
        if not self.audio_provider:
            raise RuntimeError("Audio provider not available")

        # Prepare render tasks for parallel execution
        async def render_track(track, name: str) -> Tuple[str, AudioBuffer]:
            """Render a single track and return (name, audio) tuple."""
            track_midi = MIDIFile(
                tracks=[track],
                tempo_bpm=midi_file.tempo_bpm,
                time_signature=midi_file.time_signature,
            )
            audio = await self.audio_provider.render_midi(track_midi)
            return (name, audio)

        # Create parallel render tasks for each track
        track_tasks: List[Coroutine[Any, Any, Tuple[str, AudioBuffer]]] = [
            render_track(track, track.name) for track in midi_file.tracks
        ]

        # Also render full MIDI in parallel
        async def render_full() -> Tuple[str, AudioBuffer]:
            audio = await self.audio_provider.render_midi(midi_file)
            return ("full", audio)

        track_tasks.append(render_full())

        # Execute all renders in parallel
        logger.info(f"Rendering {len(track_tasks)} stems in parallel...")
        results = await asyncio.gather(*track_tasks, return_exceptions=True)

        # Collect results
        stems = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Stem render failed: {result}")
                continue
            name, audio = result
            stems[name] = audio

        return stems

    async def _mix_stems(
        self,
        stems: Dict[str, AudioBuffer],
        mix_spec: Dict,
    ) -> AudioBuffer:
        """Mix rendered stems according to mix spec."""
        if not self.audio_provider:
            raise RuntimeError("Audio provider not available")

        # Convert to AudioStem format
        audio_stems = []
        for name, buffer in stems.items():
            if name == "full":
                continue  # Skip full render, use individual stems

            # Determine category from name
            name_lower = name.lower()
            if "drum" in name_lower:
                category = "drums"
            elif "bass" in name_lower:
                category = "bass"
            elif "chord" in name_lower or "pad" in name_lower:
                category = "keys"
            elif "melody" in name_lower or "lead" in name_lower:
                category = "synth"
            else:
                category = "other"

            audio_stems.append(
                AudioStem(
                    name=name,
                    buffer=buffer,
                    category=category,
                )
            )

        if not audio_stems:
            # No individual stems, use full render
            return stems.get(
                "full",
                AudioBuffer(
                    data=np.zeros((2, 44100)),
                    sample_rate=44100,
                    channels=2,
                ),
            )

        # Get levels from mix spec
        levels = {}
        for track_settings in mix_spec.get("tracks", []):
            track_name = track_settings.get("track_name", "")
            gain_db = track_settings.get("gain_db", 0.0)
            levels[track_name] = gain_db

        # Get pans from mix spec
        pans = {}
        for track_settings in mix_spec.get("tracks", []):
            track_name = track_settings.get("track_name", "")
            pan = track_settings.get("pan", 0.0)
            pans[track_name] = pan

        # Mix using audio provider
        mixed = await self.audio_provider.mix_stems(audio_stems, levels, pans)

        return mixed

    async def _apply_mastering(
        self,
        audio: AudioBuffer,
        master_spec: Dict,
    ) -> AudioBuffer:
        """Apply mastering chain to mixed audio."""
        # Extract mastering parameters
        target_lufs = master_spec.get("loudness", {}).get("target_lufs", self.config.target_lufs)
        true_peak = master_spec.get("true_peak", {}).get(
            "ceiling_dbtp", self.config.true_peak_ceiling
        )

        # Use the mastering chain from audio module
        try:
            mastering = MasteringChain(
                sample_rate=audio.sample_rate,
                target_lufs=target_lufs,
                true_peak_ceiling=true_peak,
            )

            # Process audio
            mastered_data = mastering.process(audio.data)

            return AudioBuffer(
                data=mastered_data,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
            )
        except Exception as e:
            logger.warning(f"Mastering chain failed, returning original: {e}")
            return audio

    async def _export_files(
        self,
        song_id: str,
        title: str,
        master: AudioBuffer,
        stems: Dict[str, AudioBuffer],
    ) -> Dict[str, Path]:
        """Export audio files."""
        output_paths = {}

        if not self.config.output_dir:
            return output_paths

        # Create output directory
        output_dir = Path(self.config.output_dir) / song_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Safe filename
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)

        # Export master
        try:
            from aether.audio.io import write_audio

            for fmt in self.config.export_formats:
                if fmt == "wav":
                    master_path = output_dir / f"{safe_title}_master.wav"
                    write_audio(master_path, master.data, master.sample_rate)
                    output_paths["master_wav"] = master_path
                elif fmt == "mp3":
                    # MP3 export requires additional library
                    pass

            # Export stems if requested
            if self.config.render_stems and stems:
                stems_dir = output_dir / "stems"
                stems_dir.mkdir(exist_ok=True)

                for stem_name, stem_audio in stems.items():
                    if stem_name == "full":
                        continue
                    stem_path = stems_dir / f"{stem_name}.wav"
                    write_audio(stem_path, stem_audio.data, stem_audio.sample_rate)
                    output_paths[f"stem_{stem_name}"] = stem_path

        except Exception as e:
            logger.warning(f"File export failed: {e}")

        return output_paths


async def render_pipeline_output(
    pipeline_output: Dict[str, Any],
    config: Optional[RenderingConfig] = None,
) -> RenderingResult:
    """Convenience function to render pipeline output."""
    engine = RenderingEngine(config)
    return await engine.render(pipeline_output)
