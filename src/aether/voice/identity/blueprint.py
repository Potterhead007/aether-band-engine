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


# =============================================================================
# AVU-2: AETHER Voice Unit 2 - Mezzo-Soprano (Versatile Female)
# =============================================================================

AVU2Identity = VocalIdentity(
    name="AVU-2",
    classification=VocalClassification.MEZZO_SOPRANO,

    vocal_range=VocalRange(
        comfortable_low=55,   # G3
        comfortable_high=79,  # G5
        extended_low=52,      # E3
        extended_high=84,     # C6
        tessitura_low=58,     # A#3
        tessitura_high=74,    # D5
    ),

    formants=FormantProfile(
        f1_range=(550, 800),    # Slightly higher F1 for female voice
        f2_range=(1600, 2100),  # Forward placement
        f3_range=(2600, 3100),  # Brighter upper harmonics
        singers_formant=(3000, 3500),
    ),

    timbre=TimbreCharacteristics(
        brightness=0.68,       # Bright but not piercing
        breathiness=0.30,      # Slightly breathy, warm
        grit=0.08,             # Smooth, minimal grit
        nasality=0.12,         # Low nasality
        chest_resonance=0.55,  # Balanced chest/head
        head_voice_blend=0.65, # Favors head voice slightly
    ),

    emotional_baseline=EmotionalBaseline(
        warmth=0.72,           # Very warm
        control=0.68,          # Good control
        intimacy=0.70,         # Intimate quality
        power_reserve=0.60,    # Moderate power
        sincerity=0.78,        # High sincerity
        engagement=0.75,       # Engaging
    ),

    vibrato_rate_hz=(5.5, 6.2),        # Slightly faster female vibrato
    vibrato_onset_delay_ms=(150, 250),
    breath_sound_profile="soft_warm",
    transition_smoothness=0.88,
    consonant_clarity=0.68,
    sibilance_brightness=0.52,
)


# =============================================================================
# AVU-3: AETHER Voice Unit 3 - Baritone (Rich Male)
# =============================================================================

AVU3Identity = VocalIdentity(
    name="AVU-3",
    classification=VocalClassification.BARITONE,

    vocal_range=VocalRange(
        comfortable_low=38,   # D2
        comfortable_high=64,  # E4
        extended_low=35,      # B1
        extended_high=69,     # A4
        tessitura_low=43,     # G2
        tessitura_high=60,    # C4
    ),

    formants=FormantProfile(
        f1_range=(450, 650),    # Lower F1 for deeper voice
        f2_range=(1200, 1600),  # Back placement
        f3_range=(2200, 2600),  # Darker harmonics
        singers_formant=(2600, 3000),
    ),

    timbre=TimbreCharacteristics(
        brightness=0.45,       # Darker tone
        breathiness=0.18,      # Minimal breathiness
        grit=0.22,             # Some character/grit
        nasality=0.15,         # Low nasality
        chest_resonance=0.82,  # Strong chest voice
        head_voice_blend=0.35, # Primarily chest
    ),

    emotional_baseline=EmotionalBaseline(
        warmth=0.70,           # Warm and rich
        control=0.75,          # Strong control
        intimacy=0.55,         # Moderate intimacy
        power_reserve=0.80,    # High power reserve
        sincerity=0.72,        # Sincere
        engagement=0.68,       # Solid engagement
    ),

    vibrato_rate_hz=(4.8, 5.4),        # Slower, wider vibrato
    vibrato_onset_delay_ms=(200, 320),
    breath_sound_profile="deep_resonant",
    transition_smoothness=0.78,
    consonant_clarity=0.75,
    sibilance_brightness=0.40,
)


# =============================================================================
# AVU-4: AETHER Voice Unit 4 - Soprano (Bright Female)
# =============================================================================

AVU4Identity = VocalIdentity(
    name="AVU-4",
    classification=VocalClassification.SOPRANO,

    vocal_range=VocalRange(
        comfortable_low=60,   # C4 (middle C)
        comfortable_high=84,  # C6
        extended_low=57,      # A3
        extended_high=88,     # E6
        tessitura_low=64,     # E4
        tessitura_high=79,    # G5
    ),

    formants=FormantProfile(
        f1_range=(600, 900),    # High F1 for soprano
        f2_range=(1800, 2300),  # Very forward
        f3_range=(2800, 3300),  # Brilliant upper partials
        singers_formant=(3200, 3800),
    ),

    timbre=TimbreCharacteristics(
        brightness=0.78,       # Very bright, crystalline
        breathiness=0.22,      # Light breathiness
        grit=0.05,             # Very clean tone
        nasality=0.10,         # Minimal nasality
        chest_resonance=0.40,  # Light chest
        head_voice_blend=0.80, # Primarily head voice
    ),

    emotional_baseline=EmotionalBaseline(
        warmth=0.58,           # Bright over warm
        control=0.72,          # Good control
        intimacy=0.65,         # Moderately intimate
        power_reserve=0.55,    # Agile over powerful
        sincerity=0.70,        # Sincere
        engagement=0.82,       # Very engaging
    ),

    vibrato_rate_hz=(5.8, 6.5),        # Fast, light vibrato
    vibrato_onset_delay_ms=(120, 220),
    breath_sound_profile="light_airy",
    transition_smoothness=0.90,        # Very smooth transitions
    consonant_clarity=0.65,
    sibilance_brightness=0.58,
)


# =============================================================================
# Voice Registry - All available voice identities
# =============================================================================

VOICE_REGISTRY = {
    "AVU-1": AVU1Identity,
    "AVU-2": AVU2Identity,
    "AVU-3": AVU3Identity,
    "AVU-4": AVU4Identity,
}


def get_voice(name: str) -> VocalIdentity:
    """Get a voice identity by name."""
    if name not in VOICE_REGISTRY:
        available = ", ".join(VOICE_REGISTRY.keys())
        raise ValueError(f"Unknown voice '{name}'. Available: {available}")
    return VOICE_REGISTRY[name]


def list_voices() -> list:
    """List all available voice identities."""
    return [
        {
            "name": v.name,
            "classification": v.classification.value,
            "range": f"{v.vocal_range.comfortable_low}-{v.vocal_range.comfortable_high}",
            "character": _describe_voice(v),
        }
        for v in VOICE_REGISTRY.values()
    ]


def _describe_voice(v: VocalIdentity) -> str:
    """Generate a brief description of a voice."""
    traits = []
    if v.timbre.brightness > 0.65:
        traits.append("bright")
    elif v.timbre.brightness < 0.5:
        traits.append("dark")
    if v.timbre.breathiness > 0.25:
        traits.append("breathy")
    if v.timbre.grit > 0.18:
        traits.append("gritty")
    if v.emotional_baseline.warmth > 0.68:
        traits.append("warm")
    if v.emotional_baseline.power_reserve > 0.70:
        traits.append("powerful")
    return ", ".join(traits) if traits else "balanced"


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
