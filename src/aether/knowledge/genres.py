"""
AETHER Genre Profile System

Loads, validates, and manages genre root profiles for authentic production.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from aether.schemas.genre import GenreRootProfile

logger = logging.getLogger(__name__)


class GenreNotFoundError(Exception):
    """Raised when a genre profile is not found."""
    pass


class GenreValidationError(Exception):
    """Raised when a genre profile fails validation."""
    pass


class GenreProfileManager:
    """
    Manages genre root profiles.

    Features:
    - Load from YAML/JSON files
    - Built-in genre library
    - Validation
    - Caching
    """

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir
        self._cache: Dict[str, GenreRootProfile] = {}
        self._load_builtin_profiles()

    def _load_builtin_profiles(self) -> None:
        """Load built-in genre profiles."""
        for profile in BUILTIN_PROFILES:
            self._cache[profile.genre_id] = profile
            logger.debug(f"Loaded built-in profile: {profile.genre_id}")

    def load_from_file(self, path: Path) -> GenreRootProfile:
        """Load a genre profile from file."""
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {path}")

        with open(path) as f:
            if path.suffix == ".yaml" or path.suffix == ".yml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        try:
            profile = GenreRootProfile.model_validate(data)
        except Exception as e:
            raise GenreValidationError(f"Invalid profile: {e}")

        self._cache[profile.genre_id] = profile
        logger.info(f"Loaded profile from file: {profile.genre_id}")
        return profile

    def load_directory(self, directory: Optional[Path] = None) -> int:
        """Load all profiles from a directory. Returns count loaded."""
        directory = directory or self.profiles_dir
        if not directory or not directory.exists():
            return 0

        count = 0
        for path in directory.glob("*.yaml"):
            try:
                self.load_from_file(path)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        for path in directory.glob("*.json"):
            try:
                self.load_from_file(path)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        return count

    def get(self, genre_id: str) -> GenreRootProfile:
        """Get a genre profile by ID."""
        if genre_id not in self._cache:
            # Try loading from file
            if self.profiles_dir:
                for ext in [".yaml", ".yml", ".json"]:
                    path = self.profiles_dir / f"{genre_id}{ext}"
                    if path.exists():
                        return self.load_from_file(path)

            raise GenreNotFoundError(f"Genre not found: {genre_id}")

        return self._cache[genre_id]

    def list_genres(self) -> List[str]:
        """List all available genre IDs."""
        return list(self._cache.keys())

    def list_profiles(self) -> List[GenreRootProfile]:
        """List all available profiles."""
        return list(self._cache.values())

    def search(
        self,
        tempo_bpm: Optional[int] = None,
        era: Optional[str] = None,
    ) -> List[GenreRootProfile]:
        """Search genres by criteria."""
        results = []
        for profile in self._cache.values():
            if tempo_bpm is not None:
                if not (profile.tempo.min_bpm <= tempo_bpm <= profile.tempo.max_bpm):
                    continue
            if era is not None:
                if era.lower() not in profile.production.era_reference.lower():
                    continue
            results.append(profile)
        return results

    def validate_for_song(
        self,
        genre_id: str,
        bpm: int,
        key_mode: str,
    ) -> Dict[str, bool]:
        """Validate song parameters against genre profile."""
        profile = self.get(genre_id)

        validations = {
            "tempo_in_range": profile.tempo.min_bpm <= bpm <= profile.tempo.max_bpm,
            "mode_appropriate": key_mode.lower() in [m.value for m in profile.harmony.common_modes],
        }

        return validations


# ============================================================================
# Built-in Genre Profiles
# ============================================================================

BUILTIN_PROFILES: List[GenreRootProfile] = []


def _create_boom_bap_profile() -> GenreRootProfile:
    """Create the Boom Bap genre profile."""
    from aether.schemas.genre import (
        GenreLineage,
        HistoricalContext,
        EvolutionPeriod,
        RhythmProfile,
        HarmonyProfile,
        MelodyProfile,
        ArrangementProfile,
        InstrumentationProfile,
        ProductionProfile,
        MixCharacteristics,
        MasterTargets,
        AuthenticityRubric,
        RubricDimension,
    )
    from aether.schemas.base import TempoRange, DurationRange, Mode

    return GenreRootProfile(
        genre_id="hip-hop-boom-bap",
        name="Boom Bap",
        aliases=["East Coast Hip-Hop", "Golden Age Hip-Hop", "90s Hip-Hop"],
        lineage=GenreLineage(
            ancestors=["funk", "soul", "jazz", "disco"],
            influences=["reggae", "r&b", "gospel"],
            descendants=["abstract-hip-hop", "conscious-hip-hop", "underground-hip-hop"],
            siblings=["g-funk", "gangsta-rap"],
        ),
        historical_context=HistoricalContext(
            emergence_era="late 1980s",
            emergence_year=1987,
            geographic_origin="New York City (Bronx, Queens, Brooklyn)",
            cultural_context="Post-golden age hip-hop, emphasis on lyricism and sample-based production. Born from block parties and the need for authentic street expression.",
            socioeconomic_factors=[
                "Urban poverty and marginalization",
                "Limited access to instruments",
                "Abundant vinyl records in thrift stores",
                "Rise of affordable samplers (SP-1200, MPC)",
            ],
            key_innovations=[
                "Chopped sample techniques",
                "SP-1200 and MPC workflow",
                "Jazz sample integration",
                "Complex multi-syllabic rhyme schemes",
                "Scratching as musical element",
            ],
        ),
        evolution_timeline=[
            EvolutionPeriod(
                period_name="Birth",
                years="1987-1990",
                characteristics=["Raw production", "Simple loops", "DJ-focused"],
                production_norms={"sampling": "direct loops", "drums": "breakbeat-based"},
            ),
            EvolutionPeriod(
                period_name="Golden Age",
                years="1990-1996",
                characteristics=["Complex sampling", "Jazz influence", "Peak lyricism"],
                production_norms={"sampling": "chopped and flipped", "drums": "punchy and crisp"},
            ),
            EvolutionPeriod(
                period_name="Decline",
                years="1997-2005",
                characteristics=["Commercial pressure", "G-funk dominance"],
                production_norms={"sampling": "clearance issues", "drums": "cleaner"},
            ),
            EvolutionPeriod(
                period_name="Revival",
                years="2010-present",
                characteristics=["Nostalgia", "Underground resurgence"],
                production_norms={"sampling": "classic techniques", "drums": "vintage aesthetic"},
            ),
        ],
        tempo=TempoRange(min_bpm=80, max_bpm=100, typical_bpm=90),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["straight", "slight_swing"],
            swing_amount_min=0.0,
            swing_amount_max=0.2,
            swing_amount_typical=0.08,
            signature_patterns=["boom_bap", "head_nod"],
            drum_characteristics={
                "kick": "Punchy, prominent, often on 1 with syncopation",
                "snare": "Snappy with body, on 2 and 4, sometimes rim shots",
                "hihat": "8th or 16th notes, open hat accents, not too busy",
                "overall": "Vinyl texture, sampled feel, not quantized perfectly",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MINOR, Mode.DORIAN, Mode.PHRYGIAN],
            typical_progressions=[
                "i-VII-VI-VII",
                "i-iv-i-V",
                "i-VI-VII-i",
                "ii-V-i",  # Jazz influence
                "i-VII-VI-v",
            ],
            tension_level=0.4,
            jazz_influence=0.6,
            modal_interchange_common=False,
        ),
        melody=MelodyProfile(
            typical_range_octaves=1.5,
            interval_vocabulary=["m2", "M2", "m3", "P4", "P5", "m7"],
            contour_preferences=["descending", "wave", "static"],
            phrase_lengths=[2, 4],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=180, max_seconds=300),
            common_structures=[
                "intro-verse-chorus-verse-chorus-verse-outro",
                "intro-verse-hook-verse-hook-bridge-verse-hook-outro",
                "intro-verse-verse-chorus-verse-chorus-outro",
            ],
            energy_curve_type="maintain",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "sampled/programmed drums",
                "bass (sampled or synth sub)",
                "sample chops (keys, horns, strings, vocals)",
            ],
            common=[
                "scratching",
                "piano/rhodes",
                "strings (sampled)",
                "horn stabs",
                "vocal samples",
            ],
            forbidden=[
                "modern 808 sub bass",
                "trap hi-hat rolls",
                "EDM synths",
                "heavy sidechain pumping",
                "excessive auto-tune",
            ],
        ),
        production=ProductionProfile(
            era_reference="1990s",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.7,
                vocal_forwardness=0.85,
                brightness=0.4,
                width=0.4,
                vintage_warmth=0.7,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-12.0,
                loudness_lufs_max=-9.0,
                dynamic_range_lu_min=6.0,
                dynamic_range_lu_max=10.0,
            ),
            signature_effects=[
                "vinyl crackle/dust",
                "lo-fi filtering",
                "tape saturation",
                "short room reverb",
                "slapback delay on vocals",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Drum Sound",
                    weight=0.25,
                    criteria=[
                        "Punchy, vinyl-textured drums",
                        "Proper kick/snare relationship",
                        "Appropriate swing/groove feel",
                        "Not over-quantized",
                    ],
                    scoring_guide="5=classic boom bap drums, 3=acceptable, 1=wrong style entirely",
                ),
                RubricDimension(
                    name="Sample Aesthetic",
                    weight=0.2,
                    criteria=[
                        "Warm, soulful sample choices",
                        "Appropriate chop technique",
                        "Vinyl texture/warmth present",
                        "Jazz/soul influence evident",
                    ],
                    scoring_guide="5=authentic chopped soul/jazz, 3=passable, 1=too clean/modern",
                ),
                RubricDimension(
                    name="Tempo & Groove",
                    weight=0.15,
                    criteria=[
                        "Within 80-100 BPM range",
                        "Head-nod factor present",
                        "Pocket feels right",
                    ],
                    scoring_guide="5=perfect pocket, 3=acceptable groove, 1=wrong tempo/feel",
                ),
                RubricDimension(
                    name="Harmonic Content",
                    weight=0.15,
                    criteria=[
                        "Minor/modal tonality",
                        "Jazz influence appropriate",
                        "Soulful chord choices",
                    ],
                    scoring_guide="5=classic progressions, 3=acceptable harmony, 1=wrong harmonic style",
                ),
                RubricDimension(
                    name="Arrangement",
                    weight=0.15,
                    criteria=[
                        "Standard verse-hook structure",
                        "Energy level appropriate",
                        "Room for vocals",
                    ],
                    scoring_guide="5=classic arrangement, 3=acceptable structure, 1=wrong format",
                ),
                RubricDimension(
                    name="Mix Character",
                    weight=0.1,
                    criteria=[
                        "Lo-fi warmth present",
                        "Not over-produced",
                        "Vocals forward but not sterile",
                    ],
                    scoring_guide="5=authentic 90s mix, 3=acceptable, 1=too modern/clean",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_synthwave_profile() -> GenreRootProfile:
    """Create the Synthwave genre profile."""
    from aether.schemas.genre import (
        GenreLineage,
        HistoricalContext,
        RhythmProfile,
        HarmonyProfile,
        MelodyProfile,
        ArrangementProfile,
        InstrumentationProfile,
        ProductionProfile,
        MixCharacteristics,
        MasterTargets,
        AuthenticityRubric,
        RubricDimension,
    )
    from aether.schemas.base import TempoRange, DurationRange, Mode

    return GenreRootProfile(
        genre_id="synthwave",
        name="Synthwave",
        aliases=["Retrowave", "Outrun", "Dreamwave"],
        lineage=GenreLineage(
            ancestors=["new wave", "synth-pop", "italo disco", "electronic"],
            influences=["film scores", "video game music", "80s pop"],
            descendants=["darksynth", "chillwave", "vaporwave"],
            siblings=["electro house", "nu-disco"],
        ),
        historical_context=HistoricalContext(
            emergence_era="mid 2000s",
            emergence_year=2006,
            geographic_origin="France (Kavinsky, College) and international online communities",
            cultural_context="Nostalgia for 1980s aesthetics, inspired by films like Drive, neon visuals, and retro-futurism.",
            socioeconomic_factors=[
                "Internet enabling global subcultures",
                "Affordable DAWs and VSTs",
                "80s nostalgia cycle",
                "Video game soundtrack influence",
            ],
            key_innovations=[
                "Authentic 80s synth emulation",
                "Retro-futuristic sound design",
                "Gated reverb drums revival",
                "Neon aesthetic integration",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=80, max_bpm=118, typical_bpm=100),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["straight"],
            swing_amount_min=0.0,
            swing_amount_max=0.05,
            swing_amount_typical=0.0,
            signature_patterns=["four_on_floor", "driving_eighths"],
            drum_characteristics={
                "kick": "Punchy, gated, prominent",
                "snare": "Gated reverb, big and splashy",
                "hihat": "Tight, machinelike 8ths or 16ths",
                "toms": "Big, gated, used for fills",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MINOR, Mode.DORIAN, Mode.AEOLIAN],
            typical_progressions=[
                "i-VI-III-VII",
                "i-VII-VI-VII",
                "i-III-VII-VI",
                "i-iv-VII-III",
                "vi-IV-I-V",  # For dreamier sections
            ],
            tension_level=0.5,
            jazz_influence=0.1,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=2.0,
            interval_vocabulary=["M2", "m3", "M3", "P4", "P5", "M6", "octave"],
            contour_preferences=["arch", "ascending", "wave"],
            phrase_lengths=[4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=180, max_seconds=360),
            common_structures=[
                "intro-build-drop-verse-chorus-verse-chorus-breakdown-chorus-outro",
                "intro-verse-chorus-verse-chorus-bridge-chorus-outro",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "analog-style synth bass",
                "pad synthesizers",
                "arpeggiated sequences",
                "gated reverb drums",
            ],
            common=[
                "lead synth (saw/square)",
                "electric guitar (with chorus)",
                "slap bass",
                "vocoder vocals",
                "saxophone",
            ],
            forbidden=[
                "acoustic drums (raw)",
                "modern EDM basses (dubstep wobble)",
                "trap hi-hats",
                "overly complex modern sound design",
            ],
        ),
        production=ProductionProfile(
            era_reference="1980s",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.6,
                vocal_forwardness=0.6,
                brightness=0.7,
                width=0.8,
                vintage_warmth=0.5,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-14.0,
                loudness_lufs_max=-10.0,
                dynamic_range_lu_min=6.0,
                dynamic_range_lu_max=9.0,
            ),
            signature_effects=[
                "gated reverb",
                "chorus on everything",
                "sidechain compression",
                "analog-style delay",
                "tape saturation",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Synth Sounds",
                    weight=0.3,
                    criteria=[
                        "Authentic 80s-style patches",
                        "Proper arpeggiator use",
                        "Rich pads with movement",
                        "Characterful lead sounds",
                    ],
                    scoring_guide="5=perfect 80s synths, 3=acceptable, 1=wrong era sounds",
                ),
                RubricDimension(
                    name="Drum Production",
                    weight=0.2,
                    criteria=[
                        "Gated reverb on snare",
                        "Punchy electronic kick",
                        "Machine-like precision",
                    ],
                    scoring_guide="5=classic synthwave drums, 3=passable, 1=wrong style",
                ),
                RubricDimension(
                    name="Arrangement",
                    weight=0.2,
                    criteria=[
                        "Proper build and release",
                        "Arpeggios as foundation",
                        "Epic moments",
                    ],
                    scoring_guide="5=cinematic arrangement, 3=acceptable, 1=wrong structure",
                ),
                RubricDimension(
                    name="Mix Aesthetic",
                    weight=0.15,
                    criteria=[
                        "Wide stereo image",
                        "Proper use of chorus/delay",
                        "Not overly modern/clean",
                    ],
                    scoring_guide="5=authentic 80s mix, 3=acceptable, 1=too modern",
                ),
                RubricDimension(
                    name="Mood & Atmosphere",
                    weight=0.15,
                    criteria=[
                        "Nostalgic/cinematic feel",
                        "Night drive energy",
                        "Emotional resonance",
                    ],
                    scoring_guide="5=perfect vibe, 3=acceptable mood, 1=wrong atmosphere",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_lofi_hiphop_profile() -> GenreRootProfile:
    """Create the Lo-Fi Hip Hop genre profile."""
    from aether.schemas.genre import (
        GenreLineage,
        HistoricalContext,
        RhythmProfile,
        HarmonyProfile,
        MelodyProfile,
        ArrangementProfile,
        InstrumentationProfile,
        ProductionProfile,
        MixCharacteristics,
        MasterTargets,
        AuthenticityRubric,
        RubricDimension,
    )
    from aether.schemas.base import TempoRange, DurationRange, Mode

    return GenreRootProfile(
        genre_id="lo-fi-hip-hop",
        name="Lo-Fi Hip Hop",
        aliases=["Lofi Beats", "Chillhop", "Study Beats"],
        lineage=GenreLineage(
            ancestors=["hip-hop", "jazz", "boom-bap", "trip-hop"],
            influences=["ambient", "chillwave", "J Dilla", "Nujabes"],
            descendants=["jazzhop", "chillsynth"],
            siblings=["chillhop", "instrumental hip-hop"],
        ),
        historical_context=HistoricalContext(
            emergence_era="early 2010s",
            emergence_year=2013,
            geographic_origin="Internet (YouTube, SoundCloud) - global",
            cultural_context="Study/focus music phenomenon, 24/7 livestreams, anime aesthetic, chill culture. Democratized by bedroom producers worldwide.",
            socioeconomic_factors=[
                "YouTube algorithm promotion",
                "Streaming study playlists",
                "Affordable production tools",
                "Work-from-home culture",
            ],
            key_innovations=[
                "Intentional degradation as aesthetic",
                "Vinyl simulation plugins",
                "Jazz sample revival",
                "Ambient texture layering",
                "Long-form streaming format",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=70, max_bpm=90, typical_bpm=80),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["swing", "laid_back"],
            swing_amount_min=0.1,
            swing_amount_max=0.3,
            swing_amount_typical=0.15,
            signature_patterns=["lazy_boom_bap", "head_nod_slow"],
            drum_characteristics={
                "kick": "Soft, muffled, dusty",
                "snare": "Crushed, lo-fi, rim shots",
                "hihat": "Gentle swinging 8ths, closed",
                "overall": "Heavily filtered, vinyl texture, relaxed timing",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MAJOR, Mode.MINOR, Mode.DORIAN, Mode.LYDIAN],
            typical_progressions=[
                "ii-V-I-vi",
                "I-vi-ii-V",
                "I-IV-vi-V",
                "ii-V-I",
                "i-VII-VI-VII",
            ],
            tension_level=0.3,
            jazz_influence=0.8,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=1.5,
            interval_vocabulary=["M2", "m3", "M3", "P4", "P5", "M6", "M7", "m7"],
            contour_preferences=["wave", "descending", "static"],
            phrase_lengths=[2, 4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=90, max_seconds=180),
            common_structures=[
                "intro-A-B-A-outro",
                "intro-loop-variation-loop-outro",
                "A-A-B-A",
            ],
            energy_curve_type="maintain",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "dusty drums (filtered)",
                "jazz piano/rhodes",
                "warm bass (often muted)",
                "vinyl crackle/noise",
            ],
            common=[
                "guitar (clean, jazzy)",
                "soft synth pads",
                "ambient textures",
                "rain/nature sounds",
                "vocal samples (chopped/pitched)",
                "saxophone",
                "vibraphone",
            ],
            forbidden=[
                "aggressive sounds",
                "heavy bass drops",
                "loud/harsh elements",
                "fast trap hi-hats",
                "distorted elements",
            ],
        ),
        production=ProductionProfile(
            era_reference="2010s",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.5,
                vocal_forwardness=0.3,  # Usually instrumental
                brightness=0.3,
                width=0.5,
                vintage_warmth=0.9,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-16.0,
                loudness_lufs_max=-12.0,
                dynamic_range_lu_min=8.0,
                dynamic_range_lu_max=12.0,
            ),
            signature_effects=[
                "vinyl crackle/dust",
                "heavy low-pass filtering",
                "tape wobble/wow/flutter",
                "bit crushing (subtle)",
                "room reverb",
                "sidechain (gentle)",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Lo-Fi Aesthetic",
                    weight=0.3,
                    criteria=[
                        "Proper vinyl texture",
                        "Filtered/muffled sound",
                        "Intentional imperfection",
                        "Warm, not harsh",
                    ],
                    scoring_guide="5=perfect dusty vibe, 3=acceptable, 1=too clean/harsh",
                ),
                RubricDimension(
                    name="Jazz Elements",
                    weight=0.25,
                    criteria=[
                        "Jazz chord voicings",
                        "Rhodes/piano presence",
                        "Soulful samples",
                        "Sophisticated harmony",
                    ],
                    scoring_guide="5=rich jazz influence, 3=acceptable, 1=missing jazz DNA",
                ),
                RubricDimension(
                    name="Rhythm & Groove",
                    weight=0.2,
                    criteria=[
                        "Relaxed, swung feel",
                        "Slow tempo (70-90)",
                        "Head-nod groove",
                        "Laid-back pocket",
                    ],
                    scoring_guide="5=perfect chill groove, 3=acceptable, 1=wrong feel",
                ),
                RubricDimension(
                    name="Atmosphere",
                    weight=0.15,
                    criteria=[
                        "Calming/study vibe",
                        "Ambient textures",
                        "Non-intrusive",
                    ],
                    scoring_guide="5=perfect background music, 3=acceptable, 1=too attention-grabbing",
                ),
                RubricDimension(
                    name="Mix Quality",
                    weight=0.1,
                    criteria=[
                        "Not too loud",
                        "Gentle dynamics",
                        "Cohesive low-fi sound",
                    ],
                    scoring_guide="5=professional lo-fi, 3=acceptable, 1=poor execution",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


# Initialize built-in profiles
BUILTIN_PROFILES = [
    _create_boom_bap_profile(),
    _create_synthwave_profile(),
    _create_lofi_hiphop_profile(),
]


# Global instance
_genre_manager: Optional[GenreProfileManager] = None


def get_genre_manager(profiles_dir: Optional[Path] = None) -> GenreProfileManager:
    """Get the global genre profile manager."""
    global _genre_manager
    if _genre_manager is None:
        _genre_manager = GenreProfileManager(profiles_dir)
    return _genre_manager
