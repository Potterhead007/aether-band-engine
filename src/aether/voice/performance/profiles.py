"""
Genre-Specific Vocal Performance Profiles

Defines performance parameters for each supported genre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class GenreStyle(Enum):
    """Supported genre styles for vocal performance."""
    POP = "pop"
    R_AND_B = "r-and-b"
    ROCK = "rock"
    JAZZ = "jazz"
    HOUSE = "house"
    TRAP = "trap"
    FUNK = "funk"
    AMBIENT = "ambient"
    NEO_SOUL = "neo-soul"


@dataclass
class TimingProfile:
    """Timing characteristics for a genre."""
    behind_beat_ms: Tuple[float, float] = (0, 0)  # Range of lag
    ahead_beat_ms: Tuple[float, float] = (0, 0)  # Range of push
    swing_amount: float = 0.0  # 0.0 = straight, 1.0 = full swing
    rubato_freedom: float = 0.0  # How much tempo variation allowed
    syncopation_frequency: float = 0.0  # 0.0-1.0


@dataclass
class ArticulationProfile:
    """Articulation characteristics for a genre."""
    default_attack: str = "normal"  # soft, normal, hard
    legato_preference: float = 0.5  # 0.0 = staccato, 1.0 = legato
    consonant_crispness: float = 0.5  # 0.0 = soft, 1.0 = crisp
    vowel_purity: float = 0.5  # 0.0 = colored, 1.0 = pure
    phrase_shaping: str = "natural"  # natural, dramatic, subtle


@dataclass
class DynamicsProfile:
    """Dynamic characteristics for a genre."""
    base_range_db: Tuple[float, float] = (-12, 0)  # Dynamic range
    crescendo_rate: float = 0.5  # How fast to build
    phrase_arc: str = "natural"  # natural, flat, dramatic
    accent_strength: float = 0.5  # 0.0-1.0
    subito_frequency: float = 0.0  # Sudden dynamic changes


@dataclass
class RegisterProfile:
    """Vocal register preferences for a genre."""
    chest_voice_preference: float = 0.5  # 0.0-1.0
    head_voice_blend: float = 0.5  # 0.0-1.0
    falsetto_allowed: bool = True
    belt_threshold_midi: int = 65  # Above this, consider belting
    mix_voice_range: Tuple[int, int] = (55, 70)  # MIDI range for mix


@dataclass
class OrnamentationProfile:
    """Ornamentation preferences for a genre."""
    run_frequency: float = 0.0  # How often to add runs
    run_complexity: int = 1  # 1-5 complexity level
    bend_frequency: float = 0.0  # Note bending
    scoop_frequency: float = 0.0  # Approaching from below
    fall_frequency: float = 0.0  # Releasing downward
    trill_allowed: bool = False
    melisma_preference: float = 0.0  # 0.0-1.0


@dataclass
class VocalPerformanceProfile:
    """Complete vocal performance profile for a genre."""
    genre: GenreStyle
    name: str
    description: str

    timing: TimingProfile = field(default_factory=TimingProfile)
    articulation: ArticulationProfile = field(default_factory=ArticulationProfile)
    dynamics: DynamicsProfile = field(default_factory=DynamicsProfile)
    register: RegisterProfile = field(default_factory=RegisterProfile)
    ornamentation: OrnamentationProfile = field(default_factory=OrnamentationProfile)

    # Additional characteristics
    vibrato_weight: float = 1.0  # Multiplier for genre vibrato
    breath_audibility: float = 0.3  # How audible breaths should be
    emotional_range: Tuple[float, float] = (0.3, 0.7)  # Intensity range

    def __post_init__(self):
        """Initialize default field values if needed."""
        if self.timing is None:
            self.timing = TimingProfile()
        if self.articulation is None:
            self.articulation = ArticulationProfile()
        if self.dynamics is None:
            self.dynamics = DynamicsProfile()
        if self.register is None:
            self.register = RegisterProfile()
        if self.ornamentation is None:
            self.ornamentation = OrnamentationProfile()


# Genre-specific profiles
GENRE_PROFILES: Dict[str, VocalPerformanceProfile] = {
    "pop": VocalPerformanceProfile(
        genre=GenreStyle.POP,
        name="Contemporary Pop",
        description="Clean, radio-friendly vocal style with moderate ornamentation",
        timing=TimingProfile(
            behind_beat_ms=(0, 20),
            swing_amount=0.0,
            rubato_freedom=0.1,
            syncopation_frequency=0.3,
        ),
        articulation=ArticulationProfile(
            default_attack="normal",
            legato_preference=0.6,
            consonant_crispness=0.7,
            vowel_purity=0.6,
            phrase_shaping="natural",
        ),
        dynamics=DynamicsProfile(
            base_range_db=(-9, 0),
            crescendo_rate=0.5,
            phrase_arc="natural",
            accent_strength=0.5,
        ),
        register=RegisterProfile(
            chest_voice_preference=0.6,
            head_voice_blend=0.5,
            falsetto_allowed=True,
            belt_threshold_midi=67,
        ),
        ornamentation=OrnamentationProfile(
            run_frequency=0.15,
            run_complexity=2,
            bend_frequency=0.1,
            scoop_frequency=0.2,
            fall_frequency=0.15,
        ),
        vibrato_weight=1.0,
        breath_audibility=0.2,
        emotional_range=(0.4, 0.75),
    ),

    "r-and-b": VocalPerformanceProfile(
        genre=GenreStyle.R_AND_B,
        name="Contemporary R&B",
        description="Soulful, melismatic style with heavy ornamentation",
        timing=TimingProfile(
            behind_beat_ms=(10, 50),
            swing_amount=0.15,
            rubato_freedom=0.3,
            syncopation_frequency=0.5,
        ),
        articulation=ArticulationProfile(
            default_attack="soft",
            legato_preference=0.8,
            consonant_crispness=0.4,
            vowel_purity=0.4,
            phrase_shaping="dramatic",
        ),
        dynamics=DynamicsProfile(
            base_range_db=(-12, 0),
            crescendo_rate=0.6,
            phrase_arc="dramatic",
            accent_strength=0.6,
        ),
        register=RegisterProfile(
            chest_voice_preference=0.5,
            head_voice_blend=0.7,
            falsetto_allowed=True,
            belt_threshold_midi=65,
            mix_voice_range=(52, 68),
        ),
        ornamentation=OrnamentationProfile(
            run_frequency=0.4,
            run_complexity=4,
            bend_frequency=0.3,
            scoop_frequency=0.4,
            fall_frequency=0.3,
            melisma_preference=0.6,
        ),
        vibrato_weight=1.3,
        breath_audibility=0.35,
        emotional_range=(0.3, 0.9),
    ),

    "rock": VocalPerformanceProfile(
        genre=GenreStyle.ROCK,
        name="Rock",
        description="Powerful, direct vocal style with grit",
        timing=TimingProfile(
            behind_beat_ms=(0, 10),
            ahead_beat_ms=(0, 15),
            swing_amount=0.0,
            rubato_freedom=0.1,
            syncopation_frequency=0.2,
        ),
        articulation=ArticulationProfile(
            default_attack="hard",
            legato_preference=0.3,
            consonant_crispness=0.8,
            vowel_purity=0.5,
            phrase_shaping="natural",
        ),
        dynamics=DynamicsProfile(
            base_range_db=(-6, 0),
            crescendo_rate=0.7,
            phrase_arc="dramatic",
            accent_strength=0.8,
            subito_frequency=0.2,
        ),
        register=RegisterProfile(
            chest_voice_preference=0.8,
            head_voice_blend=0.3,
            falsetto_allowed=False,
            belt_threshold_midi=62,
        ),
        ornamentation=OrnamentationProfile(
            run_frequency=0.05,
            run_complexity=1,
            bend_frequency=0.2,
            scoop_frequency=0.15,
            fall_frequency=0.25,
        ),
        vibrato_weight=0.7,
        breath_audibility=0.15,
        emotional_range=(0.5, 1.0),
    ),

    "jazz": VocalPerformanceProfile(
        genre=GenreStyle.JAZZ,
        name="Jazz Vocal",
        description="Sophisticated, improvisatory style with swing",
        timing=TimingProfile(
            behind_beat_ms=(20, 60),
            swing_amount=0.35,
            rubato_freedom=0.5,
            syncopation_frequency=0.6,
        ),
        articulation=ArticulationProfile(
            default_attack="soft",
            legato_preference=0.7,
            consonant_crispness=0.5,
            vowel_purity=0.3,
            phrase_shaping="natural",
        ),
        dynamics=DynamicsProfile(
            base_range_db=(-15, 0),
            crescendo_rate=0.4,
            phrase_arc="natural",
            accent_strength=0.4,
        ),
        register=RegisterProfile(
            chest_voice_preference=0.5,
            head_voice_blend=0.6,
            falsetto_allowed=True,
            belt_threshold_midi=68,
            mix_voice_range=(55, 72),
        ),
        ornamentation=OrnamentationProfile(
            run_frequency=0.25,
            run_complexity=3,
            bend_frequency=0.25,
            scoop_frequency=0.35,
            fall_frequency=0.25,
            trill_allowed=True,
            melisma_preference=0.3,
        ),
        vibrato_weight=1.1,
        breath_audibility=0.25,
        emotional_range=(0.2, 0.8),
    ),

    "house": VocalPerformanceProfile(
        genre=GenreStyle.HOUSE,
        name="House/Dance",
        description="Clean, rhythmic vocal style for electronic music",
        timing=TimingProfile(
            behind_beat_ms=(0, 5),
            swing_amount=0.0,
            rubato_freedom=0.0,
            syncopation_frequency=0.4,
        ),
        articulation=ArticulationProfile(
            default_attack="normal",
            legato_preference=0.5,
            consonant_crispness=0.8,
            vowel_purity=0.7,
            phrase_shaping="subtle",
        ),
        dynamics=DynamicsProfile(
            base_range_db=(-6, 0),
            crescendo_rate=0.3,
            phrase_arc="flat",
            accent_strength=0.6,
        ),
        register=RegisterProfile(
            chest_voice_preference=0.4,
            head_voice_blend=0.6,
            falsetto_allowed=True,
            belt_threshold_midi=70,
        ),
        ornamentation=OrnamentationProfile(
            run_frequency=0.05,
            run_complexity=1,
            bend_frequency=0.05,
            scoop_frequency=0.1,
            fall_frequency=0.1,
        ),
        vibrato_weight=0.5,
        breath_audibility=0.1,
        emotional_range=(0.4, 0.7),
    ),

    "trap": VocalPerformanceProfile(
        genre=GenreStyle.TRAP,
        name="Trap/Hip-Hop",
        description="Modern hip-hop vocal style with rhythmic emphasis",
        timing=TimingProfile(
            behind_beat_ms=(0, 30),
            swing_amount=0.1,
            rubato_freedom=0.2,
            syncopation_frequency=0.7,
        ),
        articulation=ArticulationProfile(
            default_attack="hard",
            legato_preference=0.4,
            consonant_crispness=0.9,
            vowel_purity=0.4,
            phrase_shaping="natural",
        ),
        dynamics=DynamicsProfile(
            base_range_db=(-9, 0),
            crescendo_rate=0.5,
            phrase_arc="natural",
            accent_strength=0.7,
        ),
        register=RegisterProfile(
            chest_voice_preference=0.7,
            head_voice_blend=0.4,
            falsetto_allowed=True,
            belt_threshold_midi=64,
        ),
        ornamentation=OrnamentationProfile(
            run_frequency=0.1,
            run_complexity=2,
            bend_frequency=0.2,
            scoop_frequency=0.15,
            fall_frequency=0.3,
        ),
        vibrato_weight=0.6,
        breath_audibility=0.3,
        emotional_range=(0.3, 0.85),
    ),

    "funk": VocalPerformanceProfile(
        genre=GenreStyle.FUNK,
        name="Funk/Soul",
        description="Groovy, rhythmic vocal style with attitude",
        timing=TimingProfile(
            behind_beat_ms=(5, 25),
            swing_amount=0.2,
            rubato_freedom=0.15,
            syncopation_frequency=0.8,
        ),
        articulation=ArticulationProfile(
            default_attack="hard",
            legato_preference=0.4,
            consonant_crispness=0.8,
            vowel_purity=0.5,
            phrase_shaping="natural",
        ),
        dynamics=DynamicsProfile(
            base_range_db=(-9, 0),
            crescendo_rate=0.6,
            phrase_arc="natural",
            accent_strength=0.8,
            subito_frequency=0.15,
        ),
        register=RegisterProfile(
            chest_voice_preference=0.7,
            head_voice_blend=0.5,
            falsetto_allowed=True,
            belt_threshold_midi=64,
        ),
        ornamentation=OrnamentationProfile(
            run_frequency=0.2,
            run_complexity=2,
            bend_frequency=0.25,
            scoop_frequency=0.3,
            fall_frequency=0.2,
        ),
        vibrato_weight=0.8,
        breath_audibility=0.25,
        emotional_range=(0.5, 0.9),
    ),

    "ambient": VocalPerformanceProfile(
        genre=GenreStyle.AMBIENT,
        name="Ambient/Ethereal",
        description="Atmospheric, floaty vocal style",
        timing=TimingProfile(
            behind_beat_ms=(0, 100),
            swing_amount=0.0,
            rubato_freedom=0.8,
            syncopation_frequency=0.1,
        ),
        articulation=ArticulationProfile(
            default_attack="soft",
            legato_preference=0.9,
            consonant_crispness=0.3,
            vowel_purity=0.8,
            phrase_shaping="subtle",
        ),
        dynamics=DynamicsProfile(
            base_range_db=(-18, -6),
            crescendo_rate=0.2,
            phrase_arc="subtle",
            accent_strength=0.2,
        ),
        register=RegisterProfile(
            chest_voice_preference=0.3,
            head_voice_blend=0.8,
            falsetto_allowed=True,
            belt_threshold_midi=75,
        ),
        ornamentation=OrnamentationProfile(
            run_frequency=0.02,
            run_complexity=1,
            bend_frequency=0.1,
            scoop_frequency=0.05,
            fall_frequency=0.1,
        ),
        vibrato_weight=1.2,
        breath_audibility=0.4,
        emotional_range=(0.2, 0.5),
    ),

    "neo-soul": VocalPerformanceProfile(
        genre=GenreStyle.NEO_SOUL,
        name="Neo-Soul",
        description="J Dilla-influenced soulful style",
        timing=TimingProfile(
            behind_beat_ms=(30, 80),
            swing_amount=0.25,
            rubato_freedom=0.4,
            syncopation_frequency=0.6,
        ),
        articulation=ArticulationProfile(
            default_attack="soft",
            legato_preference=0.75,
            consonant_crispness=0.4,
            vowel_purity=0.4,
            phrase_shaping="natural",
        ),
        dynamics=DynamicsProfile(
            base_range_db=(-12, 0),
            crescendo_rate=0.4,
            phrase_arc="natural",
            accent_strength=0.5,
        ),
        register=RegisterProfile(
            chest_voice_preference=0.5,
            head_voice_blend=0.65,
            falsetto_allowed=True,
            belt_threshold_midi=66,
            mix_voice_range=(53, 69),
        ),
        ornamentation=OrnamentationProfile(
            run_frequency=0.35,
            run_complexity=3,
            bend_frequency=0.3,
            scoop_frequency=0.35,
            fall_frequency=0.25,
            melisma_preference=0.5,
        ),
        vibrato_weight=1.15,
        breath_audibility=0.3,
        emotional_range=(0.3, 0.85),
    ),
}


def get_profile(genre: str) -> VocalPerformanceProfile:
    """
    Get performance profile for a genre.

    Args:
        genre: Genre name

    Returns:
        Performance profile (defaults to pop if not found)
    """
    return GENRE_PROFILES.get(genre.lower(), GENRE_PROFILES["pop"])


def interpolate_profiles(
    profile_a: VocalPerformanceProfile,
    profile_b: VocalPerformanceProfile,
    weight: float = 0.5,
) -> VocalPerformanceProfile:
    """
    Interpolate between two profiles.

    Useful for genre fusion or transitions.

    Args:
        profile_a: First profile
        profile_b: Second profile
        weight: Blend weight (0.0 = all A, 1.0 = all B)

    Returns:
        Blended profile
    """
    def lerp(a: float, b: float) -> float:
        return a + (b - a) * weight

    def lerp_tuple(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
        return (lerp(a[0], b[0]), lerp(a[1], b[1]))

    return VocalPerformanceProfile(
        genre=profile_a.genre,  # Keep first genre as identifier
        name=f"{profile_a.name} + {profile_b.name}",
        description=f"Blend of {profile_a.name} and {profile_b.name}",
        timing=TimingProfile(
            behind_beat_ms=lerp_tuple(profile_a.timing.behind_beat_ms, profile_b.timing.behind_beat_ms),
            ahead_beat_ms=lerp_tuple(profile_a.timing.ahead_beat_ms, profile_b.timing.ahead_beat_ms),
            swing_amount=lerp(profile_a.timing.swing_amount, profile_b.timing.swing_amount),
            rubato_freedom=lerp(profile_a.timing.rubato_freedom, profile_b.timing.rubato_freedom),
            syncopation_frequency=lerp(profile_a.timing.syncopation_frequency, profile_b.timing.syncopation_frequency),
        ),
        articulation=ArticulationProfile(
            default_attack=profile_a.articulation.default_attack if weight < 0.5 else profile_b.articulation.default_attack,
            legato_preference=lerp(profile_a.articulation.legato_preference, profile_b.articulation.legato_preference),
            consonant_crispness=lerp(profile_a.articulation.consonant_crispness, profile_b.articulation.consonant_crispness),
            vowel_purity=lerp(profile_a.articulation.vowel_purity, profile_b.articulation.vowel_purity),
            phrase_shaping=profile_a.articulation.phrase_shaping if weight < 0.5 else profile_b.articulation.phrase_shaping,
        ),
        dynamics=DynamicsProfile(
            base_range_db=lerp_tuple(profile_a.dynamics.base_range_db, profile_b.dynamics.base_range_db),
            crescendo_rate=lerp(profile_a.dynamics.crescendo_rate, profile_b.dynamics.crescendo_rate),
            phrase_arc=profile_a.dynamics.phrase_arc if weight < 0.5 else profile_b.dynamics.phrase_arc,
            accent_strength=lerp(profile_a.dynamics.accent_strength, profile_b.dynamics.accent_strength),
            subito_frequency=lerp(profile_a.dynamics.subito_frequency, profile_b.dynamics.subito_frequency),
        ),
        register=RegisterProfile(
            chest_voice_preference=lerp(profile_a.register.chest_voice_preference, profile_b.register.chest_voice_preference),
            head_voice_blend=lerp(profile_a.register.head_voice_blend, profile_b.register.head_voice_blend),
            falsetto_allowed=profile_a.register.falsetto_allowed or profile_b.register.falsetto_allowed,
            belt_threshold_midi=int(lerp(profile_a.register.belt_threshold_midi, profile_b.register.belt_threshold_midi)),
            mix_voice_range=(
                int(lerp(profile_a.register.mix_voice_range[0], profile_b.register.mix_voice_range[0])),
                int(lerp(profile_a.register.mix_voice_range[1], profile_b.register.mix_voice_range[1])),
            ),
        ),
        ornamentation=OrnamentationProfile(
            run_frequency=lerp(profile_a.ornamentation.run_frequency, profile_b.ornamentation.run_frequency),
            run_complexity=int(lerp(profile_a.ornamentation.run_complexity, profile_b.ornamentation.run_complexity)),
            bend_frequency=lerp(profile_a.ornamentation.bend_frequency, profile_b.ornamentation.bend_frequency),
            scoop_frequency=lerp(profile_a.ornamentation.scoop_frequency, profile_b.ornamentation.scoop_frequency),
            fall_frequency=lerp(profile_a.ornamentation.fall_frequency, profile_b.ornamentation.fall_frequency),
            trill_allowed=profile_a.ornamentation.trill_allowed or profile_b.ornamentation.trill_allowed,
            melisma_preference=lerp(profile_a.ornamentation.melisma_preference, profile_b.ornamentation.melisma_preference),
        ),
        vibrato_weight=lerp(profile_a.vibrato_weight, profile_b.vibrato_weight),
        breath_audibility=lerp(profile_a.breath_audibility, profile_b.breath_audibility),
        emotional_range=lerp_tuple(profile_a.emotional_range, profile_b.emotional_range),
    )
