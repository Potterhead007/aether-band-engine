"""
Vocal Identity Blueprint

Defines the core characteristics of the AI singer, including:
- Vocal range and tessitura
- Timbre characteristics
- Formant profile
- Emotional baseline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class VocalClassification(Enum):
    """Standard vocal classifications."""
    SOPRANO = "soprano"
    MEZZO_SOPRANO = "mezzo_soprano"
    ALTO = "alto"
    COUNTERTENOR = "countertenor"
    TENOR = "tenor"
    BARITONE = "baritone"
    BASS = "bass"
    LYRIC_TENOR = "lyric_tenor"
    DRAMATIC_TENOR = "dramatic_tenor"


@dataclass
class FormantProfile:
    """Vocal formant frequencies defining timbre."""
    f1_range: Tuple[float, float]  # Openness (Hz)
    f2_range: Tuple[float, float]  # Frontness (Hz)
    f3_range: Tuple[float, float]  # Brightness (Hz)
    singers_formant: Tuple[float, float]  # Presence peak (Hz)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for processing."""
        return np.array([
            np.mean(self.f1_range),
            np.mean(self.f2_range),
            np.mean(self.f3_range),
            np.mean(self.singers_formant),
        ])


@dataclass
class TimbreCharacteristics:
    """Defines the timbral qualities of the voice."""
    brightness: float = 0.5  # 0.0-1.0, presence peak intensity
    breathiness: float = 0.2  # 0.0-1.0, air in tone
    grit: float = 0.1  # 0.0-1.0, raspiness
    nasality: float = 0.15  # 0.0-1.0, nasal resonance
    chest_resonance: float = 0.6  # 0.0-1.0, lower-mid body
    head_voice_blend: float = 0.5  # 0.0-1.0, mix ratio

    def __post_init__(self):
        """Validate ranges."""
        for field_name in ['brightness', 'breathiness', 'grit', 'nasality',
                          'chest_resonance', 'head_voice_blend']:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")


@dataclass
class VocalRange:
    """Defines the pitch range of the voice."""
    comfortable_low: int  # MIDI note number
    comfortable_high: int
    extended_low: int
    extended_high: int
    tessitura_low: int  # Optimal range
    tessitura_high: int

    @property
    def comfortable_range(self) -> Tuple[int, int]:
        return (self.comfortable_low, self.comfortable_high)

    @property
    def extended_range(self) -> Tuple[int, int]:
        return (self.extended_low, self.extended_high)

    @property
    def tessitura(self) -> Tuple[int, int]:
        return (self.tessitura_low, self.tessitura_high)

    def is_in_range(self, midi_note: int, allow_extended: bool = False) -> bool:
        """Check if a note is within the voice's range."""
        if allow_extended:
            return self.extended_low <= midi_note <= self.extended_high
        return self.comfortable_low <= midi_note <= self.comfortable_high

    def get_strain_factor(self, midi_note: int) -> float:
        """Get vocal strain factor for a note (0.0 = comfortable, 1.0 = max strain)."""
        if self.tessitura_low <= midi_note <= self.tessitura_high:
            return 0.0
        elif self.comfortable_low <= midi_note <= self.comfortable_high:
            # Linear interpolation in comfortable but non-tessitura range
            if midi_note < self.tessitura_low:
                return (self.tessitura_low - midi_note) / (self.tessitura_low - self.comfortable_low) * 0.3
            else:
                return (midi_note - self.tessitura_high) / (self.comfortable_high - self.tessitura_high) * 0.3
        else:
            # Extended range - higher strain
            if midi_note < self.comfortable_low:
                return 0.3 + (self.comfortable_low - midi_note) / (self.comfortable_low - self.extended_low) * 0.7
            else:
                return 0.3 + (midi_note - self.comfortable_high) / (self.extended_high - self.comfortable_high) * 0.7


@dataclass
class EmotionalBaseline:
    """Default emotional characteristics of the voice."""
    warmth: float = 0.6  # 0.0-1.0
    control: float = 0.7  # 0.0-1.0
    intimacy: float = 0.5  # 0.0-1.0
    power_reserve: float = 0.6  # 0.0-1.0, latent intensity
    sincerity: float = 0.7  # 0.0-1.0
    engagement: float = 0.8  # 0.0-1.0, never sounds bored


@dataclass
class VocalIdentity:
    """
    Complete vocal identity definition.

    This is the core specification for an AI singer, defining
    all characteristics that make up their unique voice.
    """
    name: str
    classification: VocalClassification
    vocal_range: VocalRange
    formants: FormantProfile
    timbre: TimbreCharacteristics
    emotional_baseline: EmotionalBaseline

    # Vibrato characteristics
    vibrato_rate_hz: Tuple[float, float] = (5.2, 5.8)
    vibrato_onset_delay_ms: Tuple[float, float] = (180, 280)

    # Additional characteristics
    breath_sound_profile: str = "neutral"  # Type of breath sound
    transition_smoothness: float = 0.8  # 0.0-1.0
    consonant_clarity: float = 0.7  # 0.0-1.0
    sibilance_brightness: float = 0.5  # 0.0-1.0

    def get_identity_vector(self) -> np.ndarray:
        """
        Get a numerical representation of the vocal identity.

        Returns a 24-dimensional vector for comparison and drift detection.
        """
        return np.array([
            # Range (normalized to 0-1 based on typical vocal ranges)
            (self.vocal_range.tessitura_low - 36) / 48,  # C2-C6
            (self.vocal_range.tessitura_high - 36) / 48,

            # Formants (normalized)
            np.mean(self.formants.f1_range) / 1000,
            np.mean(self.formants.f2_range) / 2000,
            np.mean(self.formants.f3_range) / 3000,
            np.mean(self.formants.singers_formant) / 4000,

            # Timbre
            self.timbre.brightness,
            self.timbre.breathiness,
            self.timbre.grit,
            self.timbre.nasality,
            self.timbre.chest_resonance,
            self.timbre.head_voice_blend,

            # Emotional baseline
            self.emotional_baseline.warmth,
            self.emotional_baseline.control,
            self.emotional_baseline.intimacy,
            self.emotional_baseline.power_reserve,
            self.emotional_baseline.sincerity,
            self.emotional_baseline.engagement,

            # Vibrato
            np.mean(self.vibrato_rate_hz) / 10,
            np.mean(self.vibrato_onset_delay_ms) / 500,

            # Other characteristics
            self.transition_smoothness,
            self.consonant_clarity,
            self.sibilance_brightness,

            # Padding to 24 dimensions
            0.5,
        ])


# =============================================================================
# AVU-1: AETHER Voice Unit 1 - The flagship AI singer
# =============================================================================

AVU1Identity = VocalIdentity(
    name="AVU-1",
    classification=VocalClassification.LYRIC_TENOR,

    vocal_range=VocalRange(
        comfortable_low=43,   # G2
        comfortable_high=72,  # C5
        extended_low=40,      # E2
        extended_high=76,     # E5
        tessitura_low=48,     # C3
        tessitura_high=67,    # G4
    ),

    formants=FormantProfile(
        f1_range=(500, 700),
        f2_range=(1400, 1800),
        f3_range=(2400, 2800),
        singers_formant=(2800, 3200),
    ),

    timbre=TimbreCharacteristics(
        brightness=0.62,
        breathiness=0.25,
        grit=0.15,
        nasality=0.18,
        chest_resonance=0.70,
        head_voice_blend=0.55,
    ),

    emotional_baseline=EmotionalBaseline(
        warmth=0.65,
        control=0.70,
        intimacy=0.60,
        power_reserve=0.65,
        sincerity=0.75,
        engagement=0.80,
    ),

    vibrato_rate_hz=(5.2, 5.8),
    vibrato_onset_delay_ms=(180, 280),
    breath_sound_profile="warm_neutral",
    transition_smoothness=0.85,
    consonant_clarity=0.72,
    sibilance_brightness=0.48,
)


def create_identity_from_params(
    name: str,
    classification: str,
    range_low: int,
    range_high: int,
    brightness: float = 0.5,
    breathiness: float = 0.2,
    warmth: float = 0.6,
) -> VocalIdentity:
    """
    Create a vocal identity from simplified parameters.

    This is a convenience function for creating custom identities
    without specifying all parameters manually.
    """
    # Estimate tessitura as middle 60% of range
    range_span = range_high - range_low
    tessitura_low = range_low + int(range_span * 0.2)
    tessitura_high = range_high - int(range_span * 0.2)

    return VocalIdentity(
        name=name,
        classification=VocalClassification(classification),
        vocal_range=VocalRange(
            comfortable_low=range_low,
            comfortable_high=range_high,
            extended_low=range_low - 3,
            extended_high=range_high + 4,
            tessitura_low=tessitura_low,
            tessitura_high=tessitura_high,
        ),
        formants=FormantProfile(
            f1_range=(500, 700),
            f2_range=(1400, 1800),
            f3_range=(2400, 2800),
            singers_formant=(2800, 3200),
        ),
        timbre=TimbreCharacteristics(
            brightness=brightness,
            breathiness=breathiness,
            grit=0.1,
            nasality=0.15,
            chest_resonance=0.6,
            head_voice_blend=0.5,
        ),
        emotional_baseline=EmotionalBaseline(
            warmth=warmth,
        ),
    )
