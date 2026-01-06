"""
Vibrato Generator

Generates natural, genre-appropriate vibrato for singing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from aether.voice.identity.blueprint import VocalIdentity


@dataclass
class VibratoParams:
    """Parameters for vibrato generation."""
    rate_hz: float = 5.5  # Oscillation frequency
    depth_cents: float = 40  # Pitch deviation
    onset_delay_ms: float = 200  # Time before vibrato starts
    attack_ms: float = 150  # Ramp-up time
    irregularity: float = 0.15  # Humanization factor
    intensity_scaling: bool = True  # Scale with note velocity


# Genre-specific vibrato presets
GENRE_VIBRATO_PRESETS: Dict[str, VibratoParams] = {
    "pop": VibratoParams(
        rate_hz=5.5,
        depth_cents=35,
        onset_delay_ms=180,
        attack_ms=120,
        irregularity=0.12,
    ),
    "r-and-b": VibratoParams(
        rate_hz=5.0,
        depth_cents=60,
        onset_delay_ms=150,
        attack_ms=100,
        irregularity=0.2,
    ),
    "rock": VibratoParams(
        rate_hz=6.0,
        depth_cents=25,
        onset_delay_ms=250,
        attack_ms=150,
        irregularity=0.1,
    ),
    "jazz": VibratoParams(
        rate_hz=5.2,
        depth_cents=50,
        onset_delay_ms=200,
        attack_ms=130,
        irregularity=0.18,
    ),
    "classical": VibratoParams(
        rate_hz=5.8,
        depth_cents=45,
        onset_delay_ms=100,
        attack_ms=80,
        irregularity=0.08,
    ),
    "house": VibratoParams(
        rate_hz=5.5,
        depth_cents=20,
        onset_delay_ms=300,
        attack_ms=200,
        irregularity=0.05,
    ),
    "trap": VibratoParams(
        rate_hz=5.0,
        depth_cents=30,
        onset_delay_ms=200,
        attack_ms=150,
        irregularity=0.15,
    ),
    "ambient": VibratoParams(
        rate_hz=4.5,
        depth_cents=55,
        onset_delay_ms=150,
        attack_ms=200,
        irregularity=0.2,
    ),
    "funk": VibratoParams(
        rate_hz=5.8,
        depth_cents=30,
        onset_delay_ms=180,
        attack_ms=100,
        irregularity=0.12,
    ),
}


class VibratoGenerator:
    """
    Generates natural, genre-appropriate vibrato.

    Features:
    - Genre-specific presets
    - Humanization through irregularity
    - Velocity-based intensity scaling
    - Proper onset delay and attack
    """

    def __init__(
        self,
        identity: VocalIdentity,
        frame_rate: float = 100.0,
    ):
        """
        Initialize vibrato generator.

        Args:
            identity: Vocal identity for vibrato DNA
            frame_rate: Output frame rate
        """
        self.identity = identity
        self.frame_rate = frame_rate

        # Get identity vibrato characteristics
        self.base_rate = np.mean(identity.vibrato_rate_hz)
        self.base_onset = np.mean(identity.vibrato_onset_delay_ms)

    def generate(
        self,
        duration_frames: int,
        genre: str = "pop",
        velocity: int = 100,
        custom_params: Optional[VibratoParams] = None,
    ) -> np.ndarray:
        """
        Generate vibrato modulation signal.

        Args:
            duration_frames: Number of frames to generate
            genre: Genre for preset selection
            velocity: Note velocity (0-127)
            custom_params: Override preset parameters

        Returns:
            Vibrato signal in cents (to be added to pitch)
        """
        # Get parameters
        if custom_params is not None:
            params = custom_params
        else:
            params = GENRE_VIBRATO_PRESETS.get(genre, GENRE_VIBRATO_PRESETS["pop"])

        # Blend with identity characteristics
        rate = (params.rate_hz + self.base_rate) / 2
        onset_ms = (params.onset_delay_ms + self.base_onset) / 2

        # Generate time array
        t = np.arange(duration_frames) / self.frame_rate

        # Generate phase with irregularity
        phase_noise = self._generate_phase_noise(duration_frames, params.irregularity)
        phase = 2 * np.pi * rate * t + phase_noise

        # Generate vibrato signal
        vibrato = params.depth_cents * np.sin(phase)

        # Apply onset envelope
        onset_frames = int(onset_ms * self.frame_rate / 1000)
        attack_frames = int(params.attack_ms * self.frame_rate / 1000)

        envelope = self._generate_onset_envelope(
            duration_frames, onset_frames, attack_frames
        )
        vibrato *= envelope

        # Apply velocity scaling
        if params.intensity_scaling:
            velocity_scale = 0.5 + 0.5 * (velocity / 127)
            vibrato *= velocity_scale

        return vibrato

    def _generate_phase_noise(
        self,
        length: int,
        irregularity: float
    ) -> np.ndarray:
        """Generate phase noise for humanization."""
        if irregularity <= 0:
            return np.zeros(length)

        # Low-frequency noise for gradual phase drift
        noise = np.random.randn(length) * irregularity

        # Smooth the noise
        kernel_size = max(3, length // 20)
        kernel = np.ones(kernel_size) / kernel_size
        noise = np.convolve(noise, kernel, mode='same')

        # Integrate to get phase offset
        phase_offset = np.cumsum(noise) * 0.1

        return phase_offset

    def _generate_onset_envelope(
        self,
        total_frames: int,
        onset_frames: int,
        attack_frames: int,
    ) -> np.ndarray:
        """Generate onset envelope for vibrato."""
        envelope = np.ones(total_frames)

        # Zero during onset delay
        if onset_frames > 0:
            onset_frames = min(onset_frames, total_frames)
            envelope[:onset_frames] = 0

        # Ramp up during attack
        if attack_frames > 0 and onset_frames < total_frames:
            attack_end = min(onset_frames + attack_frames, total_frames)
            attack_length = attack_end - onset_frames

            if attack_length > 0:
                envelope[onset_frames:attack_end] = np.linspace(0, 1, attack_length)

        return envelope

    def generate_expressive(
        self,
        duration_frames: int,
        genre: str = "pop",
        velocity: int = 100,
        emotion_intensity: float = 0.5,
    ) -> np.ndarray:
        """
        Generate expressive vibrato with emotion modulation.

        Higher emotion intensity increases depth and rate variation.
        """
        base_params = GENRE_VIBRATO_PRESETS.get(genre, GENRE_VIBRATO_PRESETS["pop"])

        # Modify params based on emotion
        modified = VibratoParams(
            rate_hz=base_params.rate_hz * (0.9 + 0.2 * emotion_intensity),
            depth_cents=base_params.depth_cents * (0.7 + 0.6 * emotion_intensity),
            onset_delay_ms=base_params.onset_delay_ms * (1.2 - 0.4 * emotion_intensity),
            attack_ms=base_params.attack_ms,
            irregularity=base_params.irregularity * (1 + 0.5 * emotion_intensity),
        )

        return self.generate(duration_frames, genre, velocity, modified)
