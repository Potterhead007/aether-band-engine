"""
AETHER Genre Profile System

Loads, validates, and manages genre root profiles for authentic production.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

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
        self._cache: dict[str, GenreRootProfile] = {}
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

    def list_genres(self) -> list[str]:
        """List all available genre IDs."""
        return list(self._cache.keys())

    def list_profiles(self) -> list[GenreRootProfile]:
        """List all available profiles."""
        return list(self._cache.values())

    def search(
        self,
        tempo_bpm: Optional[int] = None,
        era: Optional[str] = None,
    ) -> list[GenreRootProfile]:
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
    ) -> dict[str, bool]:
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

BUILTIN_PROFILES: list[GenreRootProfile] = []


def _create_boom_bap_profile() -> GenreRootProfile:
    """Create the Boom Bap genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        EvolutionPeriod,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

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
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

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
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

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


def _create_house_profile() -> GenreRootProfile:
    """Create the House music genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="house",
        name="House",
        aliases=["Deep House", "Chicago House", "Classic House"],
        lineage=GenreLineage(
            ancestors=["disco", "synth-pop", "electro", "soul"],
            influences=["funk", "gospel", "hi-nrg"],
            descendants=["tech-house", "progressive-house", "future-house"],
            siblings=["garage", "techno"],
        ),
        historical_context=HistoricalContext(
            emergence_era="early 1980s",
            emergence_year=1984,
            geographic_origin="Chicago (The Warehouse club)",
            cultural_context="Born in Chicago's underground club scene, particularly at The Warehouse where DJ Frankie Knuckles pioneered the sound. A celebration of Black and LGBTQ+ culture.",
            socioeconomic_factors=[
                "Post-disco underground movement",
                "Affordable drum machines and synthesizers",
                "Underground club culture",
                "LGBTQ+ community expression",
            ],
            key_innovations=[
                "Four-on-the-floor kick pattern",
                "Roland TR-808 and TR-909 drum machines",
                "Synthesizer basslines",
                "Extended DJ mixes",
                "Vocal sampling and chopping",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=118, max_bpm=130, typical_bpm=124),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["straight", "slight_swing"],
            swing_amount_min=0.0,
            swing_amount_max=0.15,
            swing_amount_typical=0.05,
            signature_patterns=["four_on_floor", "offbeat_hihat"],
            drum_characteristics={
                "kick": "Punchy, consistent 4-on-floor, slight compression",
                "snare": "Clap on 2 and 4, sometimes layered with snare",
                "hihat": "Offbeat 8ths, open hats for groove",
                "overall": "Driving, hypnotic, danceable",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MINOR, Mode.DORIAN, Mode.MAJOR],
            typical_progressions=[
                "i-IV-VII-III",
                "i-VII-VI-VII",
                "ii-V-I-vi",
                "I-V-vi-IV",
                "i-iv-VII-III",
            ],
            tension_level=0.4,
            jazz_influence=0.5,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=1.5,
            interval_vocabulary=["M2", "m3", "M3", "P4", "P5", "M6"],
            contour_preferences=["wave", "ascending", "static"],
            phrase_lengths=[4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=300, max_seconds=480),
            common_structures=[
                "intro-build-drop-breakdown-build-drop-outro",
                "intro-verse-chorus-verse-chorus-breakdown-chorus-outro",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "kick drum (909/808 style)",
                "clap/snare",
                "hi-hats (open and closed)",
                "bass synth",
            ],
            common=[
                "piano/rhodes chords",
                "organ stabs",
                "vocal samples",
                "string pads",
                "percussion (congas, shakers)",
            ],
            forbidden=[
                "heavy metal guitars",
                "acoustic drum kit",
                "dubstep wobble bass",
            ],
        ),
        production=ProductionProfile(
            era_reference="1980s-present",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.8,
                vocal_forwardness=0.6,
                brightness=0.6,
                width=0.7,
                vintage_warmth=0.5,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-10.0,
                loudness_lufs_max=-6.0,
                dynamic_range_lu_min=6.0,
                dynamic_range_lu_max=9.0,
            ),
            signature_effects=[
                "sidechain compression",
                "reverb on claps",
                "filter sweeps",
                "delay throws",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Groove",
                    weight=0.3,
                    criteria=["Four-on-floor foundation", "Hypnotic feel", "Danceability"],
                    scoring_guide="5=irresistible groove, 3=acceptable, 1=not danceable",
                ),
                RubricDimension(
                    name="Sound Design",
                    weight=0.25,
                    criteria=["Classic drum sounds", "Warm bass", "Appropriate synths"],
                    scoring_guide="5=authentic house sounds, 3=acceptable, 1=wrong genre",
                ),
                RubricDimension(
                    name="Arrangement",
                    weight=0.25,
                    criteria=["Proper build/release", "DJ-friendly structure", "Energy flow"],
                    scoring_guide="5=perfect club track, 3=acceptable, 1=wrong structure",
                ),
                RubricDimension(
                    name="Mix",
                    weight=0.2,
                    criteria=["Club-ready low end", "Clear mix", "Proper dynamics"],
                    scoring_guide="5=club-ready, 3=acceptable, 1=poor mix",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_techno_profile() -> GenreRootProfile:
    """Create the Techno genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="techno",
        name="Techno",
        aliases=["Detroit Techno", "Industrial Techno", "Minimal Techno"],
        lineage=GenreLineage(
            ancestors=["electro", "synth-pop", "industrial", "funk"],
            influences=["kraftwerk", "new wave", "chicago house"],
            descendants=["minimal", "dub-techno", "hard-techno"],
            siblings=["house", "electro"],
        ),
        historical_context=HistoricalContext(
            emergence_era="mid 1980s",
            emergence_year=1985,
            geographic_origin="Detroit, Michigan",
            cultural_context="Created by the Belleville Three (Juan Atkins, Derrick May, Kevin Saunderson) as a futuristic, post-industrial expression of Detroit's changing landscape.",
            socioeconomic_factors=[
                "Post-industrial Detroit decline",
                "Affordable synthesizers",
                "Kraftwerk and European electronic influence",
                "Underground rave culture",
            ],
            key_innovations=[
                "Futuristic, machine-like aesthetics",
                "Roland TB-303 acid basslines",
                "Minimal, repetitive structures",
                "Industrial textures",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=125, max_bpm=150, typical_bpm=135),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["straight", "driving"],
            swing_amount_min=0.0,
            swing_amount_max=0.05,
            swing_amount_typical=0.0,
            signature_patterns=["four_on_floor", "industrial_pulse"],
            drum_characteristics={
                "kick": "Heavy, driving, often distorted or industrial",
                "snare": "Minimal claps or industrial hits on 2/4",
                "hihat": "Mechanical, relentless 16ths",
                "overall": "Machine-like, hypnotic, industrial",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MINOR, Mode.PHRYGIAN, Mode.LOCRIAN],
            typical_progressions=[
                "i",  # Often just one chord/drone
                "i-VII",
                "i-iv",
                "i-VII-VI-VII",
            ],
            tension_level=0.7,
            jazz_influence=0.0,
            modal_interchange_common=False,
        ),
        melody=MelodyProfile(
            typical_range_octaves=1.0,
            interval_vocabulary=["m2", "M2", "m3", "P4", "P5"],
            contour_preferences=["static", "descending", "minimal"],
            phrase_lengths=[2, 4, 8, 16],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=360, max_seconds=600),
            common_structures=[
                "intro-build-main-variation-main-breakdown-main-outro",
                "gradual-build-peak-gradual-decay",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "heavy kick drum",
                "hi-hats (mechanical)",
                "bass synth (often 303)",
                "industrial textures",
            ],
            common=[
                "percussive synth stabs",
                "noise sweeps",
                "vocal samples (processed)",
                "atonal synth lines",
            ],
            forbidden=[
                "acoustic instruments",
                "obvious melodies",
                "major key progressions",
                "soft/gentle sounds",
            ],
        ),
        production=ProductionProfile(
            era_reference="1980s-present",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.9,
                vocal_forwardness=0.2,
                brightness=0.5,
                width=0.6,
                vintage_warmth=0.3,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-9.0,
                loudness_lufs_max=-6.0,
                dynamic_range_lu_min=5.0,
                dynamic_range_lu_max=8.0,
            ),
            signature_effects=[
                "heavy compression",
                "distortion/saturation",
                "industrial reverb",
                "filter automation",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Industrial Feel",
                    weight=0.3,
                    criteria=["Machine-like groove", "Dark atmosphere", "Relentless energy"],
                    scoring_guide="5=authentic techno, 3=acceptable, 1=wrong feel",
                ),
                RubricDimension(
                    name="Sound Design",
                    weight=0.3,
                    criteria=["Industrial textures", "Heavy kicks", "Minimal but impactful"],
                    scoring_guide="5=classic techno sounds, 3=acceptable, 1=wrong genre",
                ),
                RubricDimension(
                    name="Hypnotic Quality",
                    weight=0.2,
                    criteria=["Repetitive but evolving", "Trance-inducing", "Subtle variations"],
                    scoring_guide="5=hypnotic masterpiece, 3=acceptable, 1=too busy/random",
                ),
                RubricDimension(
                    name="Mix & Master",
                    weight=0.2,
                    criteria=["Powerful low end", "Club-ready", "Industrial clarity"],
                    scoring_guide="5=warehouse-ready, 3=acceptable, 1=weak mix",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_jazz_profile() -> GenreRootProfile:
    """Create the Jazz genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="jazz",
        name="Jazz",
        aliases=["Bebop", "Hard Bop", "Post-Bop", "Modern Jazz"],
        lineage=GenreLineage(
            ancestors=["blues", "ragtime", "gospel", "marching bands"],
            influences=["classical", "african rhythms", "cuban music"],
            descendants=["fusion", "smooth-jazz", "acid-jazz"],
            siblings=["blues", "soul"],
        ),
        historical_context=HistoricalContext(
            emergence_era="early 1900s",
            emergence_year=1917,
            geographic_origin="New Orleans, Louisiana",
            cultural_context="Born from African American musical traditions, jazz became America's classical music, evolving through swing, bebop, cool jazz, and beyond.",
            socioeconomic_factors=[
                "African American cultural expression",
                "The Great Migration",
                "Prohibition-era speakeasies",
                "Civil rights movement connection",
            ],
            key_innovations=[
                "Improvisation as art form",
                "Complex harmonic language",
                "Swing rhythmic feel",
                "Extended chord voicings",
                "Modal harmony",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=60, max_bpm=280, typical_bpm=140),
        rhythm=RhythmProfile(
            time_signatures=["4/4", "3/4", "5/4"],
            feels=["swing", "straight"],
            swing_amount_min=0.2,
            swing_amount_max=0.5,
            swing_amount_typical=0.35,
            signature_patterns=["swing_ride", "walking_bass"],
            drum_characteristics={
                "kick": "Supportive, accent-based, not on every beat",
                "snare": "Comping, ghost notes, rim shots",
                "hihat": "Foot-operated, 2 and 4",
                "ride": "Primary timekeeping, swing pattern",
                "overall": "Interactive, responsive, swinging",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MAJOR, Mode.MINOR, Mode.DORIAN, Mode.MIXOLYDIAN, Mode.LYDIAN],
            typical_progressions=[
                "ii-V-I",
                "I-vi-ii-V",
                "iii-VI-ii-V-I",
                "I-IV-iii-VI-ii-V-I",
                "i-iv-VII-III",
            ],
            tension_level=0.6,
            jazz_influence=1.0,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=2.5,
            interval_vocabulary=["m2", "M2", "m3", "M3", "P4", "tritone", "P5", "m6", "M6", "m7", "M7"],
            contour_preferences=["bebop_lines", "arch", "angular"],
            phrase_lengths=[2, 4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=180, max_seconds=600),
            common_structures=[
                "head-solos-head",
                "AABA",
                "intro-head-solos-head-outro",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "rhythm section (piano/bass/drums)",
                "melodic instrument (sax/trumpet/guitar)",
            ],
            common=[
                "upright bass",
                "piano",
                "saxophone",
                "trumpet",
                "drums",
                "guitar",
                "vibraphone",
            ],
            forbidden=[
                "heavy distortion",
                "EDM synths",
                "trap drums",
                "autotune",
            ],
        ),
        production=ProductionProfile(
            era_reference="1950s-1960s",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.5,
                vocal_forwardness=0.7,
                brightness=0.5,
                width=0.6,
                vintage_warmth=0.8,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-18.0,
                loudness_lufs_max=-12.0,
                dynamic_range_lu_min=10.0,
                dynamic_range_lu_max=15.0,
            ),
            signature_effects=[
                "room reverb",
                "tape warmth",
                "minimal processing",
                "natural dynamics",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Harmonic Sophistication",
                    weight=0.3,
                    criteria=["Extended chords", "Voice leading", "Harmonic rhythm"],
                    scoring_guide="5=sophisticated jazz harmony, 3=acceptable, 1=too simple",
                ),
                RubricDimension(
                    name="Swing Feel",
                    weight=0.25,
                    criteria=["Proper swing", "Interactive rhythm", "Groove"],
                    scoring_guide="5=swings hard, 3=acceptable, 1=no swing",
                ),
                RubricDimension(
                    name="Melodic Quality",
                    weight=0.25,
                    criteria=["Bebop vocabulary", "Phrase construction", "Motivic development"],
                    scoring_guide="5=masterful lines, 3=acceptable, 1=not jazz",
                ),
                RubricDimension(
                    name="Authenticity",
                    weight=0.2,
                    criteria=["Jazz idiom", "Appropriate instrumentation", "Era-appropriate"],
                    scoring_guide="5=authentic jazz, 3=acceptable, 1=fake jazz",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_rock_profile() -> GenreRootProfile:
    """Create the Rock genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="rock",
        name="Rock",
        aliases=["Classic Rock", "Rock and Roll", "Hard Rock"],
        lineage=GenreLineage(
            ancestors=["blues", "country", "rhythm-and-blues", "gospel"],
            influences=["folk", "jazz", "classical"],
            descendants=["punk", "metal", "grunge", "indie-rock"],
            siblings=["blues-rock", "country-rock"],
        ),
        historical_context=HistoricalContext(
            emergence_era="1950s",
            emergence_year=1954,
            geographic_origin="United States (Memphis, Chicago)",
            cultural_context="Born from the fusion of African American blues and country music, rock became the voice of youth rebellion and cultural change.",
            socioeconomic_factors=[
                "Post-war prosperity",
                "Youth culture emergence",
                "Racial integration of music",
                "Electric guitar technology",
            ],
            key_innovations=[
                "Electric guitar as lead instrument",
                "Power chords",
                "Backbeat emphasis",
                "Guitar distortion",
                "Stadium sound",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=100, max_bpm=180, typical_bpm=130),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["straight", "driving"],
            swing_amount_min=0.0,
            swing_amount_max=0.1,
            swing_amount_typical=0.05,
            signature_patterns=["backbeat", "driving_eighths"],
            drum_characteristics={
                "kick": "Punchy, supportive, often 1 and 3",
                "snare": "Strong backbeat on 2 and 4",
                "hihat": "Driving 8ths or quarters",
                "overall": "Powerful, driving, energetic",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MAJOR, Mode.MINOR, Mode.MIXOLYDIAN, Mode.DORIAN],
            typical_progressions=[
                "I-IV-V-I",
                "I-V-vi-IV",
                "i-VII-VI-VII",
                "I-bVII-IV-I",
                "vi-IV-I-V",
            ],
            tension_level=0.5,
            jazz_influence=0.1,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=2.0,
            interval_vocabulary=["M2", "m3", "M3", "P4", "P5", "m7"],
            contour_preferences=["arch", "ascending", "anthemic"],
            phrase_lengths=[4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=180, max_seconds=300),
            common_structures=[
                "intro-verse-chorus-verse-chorus-solo-chorus-outro",
                "intro-verse-chorus-verse-chorus-bridge-chorus-outro",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "electric guitar",
                "bass guitar",
                "drums",
                "vocals",
            ],
            common=[
                "second guitar (rhythm)",
                "keyboard/organ",
                "acoustic guitar",
                "backing vocals",
            ],
            forbidden=[
                "EDM synths",
                "trap hi-hats",
                "autotune (excessive)",
            ],
        ),
        production=ProductionProfile(
            era_reference="1970s-1980s",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.6,
                vocal_forwardness=0.8,
                brightness=0.6,
                width=0.7,
                vintage_warmth=0.5,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-12.0,
                loudness_lufs_max=-8.0,
                dynamic_range_lu_min=7.0,
                dynamic_range_lu_max=11.0,
            ),
            signature_effects=[
                "guitar distortion/overdrive",
                "room reverb",
                "plate reverb on vocals",
                "compression",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Guitar Sound",
                    weight=0.3,
                    criteria=["Proper tone", "Right distortion level", "Riff quality"],
                    scoring_guide="5=classic rock guitar, 3=acceptable, 1=wrong genre",
                ),
                RubricDimension(
                    name="Energy & Drive",
                    weight=0.25,
                    criteria=["Powerful rhythm section", "Driving feel", "Rock attitude"],
                    scoring_guide="5=rocks hard, 3=acceptable, 1=lacks energy",
                ),
                RubricDimension(
                    name="Song Structure",
                    weight=0.25,
                    criteria=["Strong hooks", "Memorable chorus", "Good arrangement"],
                    scoring_guide="5=anthemic, 3=acceptable, 1=weak structure",
                ),
                RubricDimension(
                    name="Production",
                    weight=0.2,
                    criteria=["Appropriate mix", "Era-appropriate sound", "Punch"],
                    scoring_guide="5=classic rock production, 3=acceptable, 1=wrong sound",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_ambient_profile() -> GenreRootProfile:
    """Create the Ambient genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="ambient",
        name="Ambient",
        aliases=["Ambient Electronic", "Atmospheric", "Soundscape"],
        lineage=GenreLineage(
            ancestors=["electronic", "minimalism", "musique-concrete", "new-age"],
            influences=["classical", "world-music", "drone"],
            descendants=["dark-ambient", "ambient-techno", "drone"],
            siblings=["new-age", "downtempo"],
        ),
        historical_context=HistoricalContext(
            emergence_era="1970s",
            emergence_year=1978,
            geographic_origin="United Kingdom (Brian Eno)",
            cultural_context="Brian Eno coined the term with 'Ambient 1: Music for Airports'. Designed to be as ignorable as it is interesting.",
            socioeconomic_factors=[
                "Rise of synthesizers",
                "Conceptual art movement",
                "Environmental awareness",
                "Meditation/mindfulness culture",
            ],
            key_innovations=[
                "Music as environment",
                "Generative music concepts",
                "Extreme dynamics",
                "Sound design focus",
                "Non-linear composition",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=40, max_bpm=100, typical_bpm=60),
        rhythm=RhythmProfile(
            time_signatures=["4/4", "free"],
            feels=["free", "floating"],
            swing_amount_min=0.0,
            swing_amount_max=0.0,
            swing_amount_typical=0.0,
            signature_patterns=["no_beat", "subtle_pulse"],
            drum_characteristics={
                "overall": "Often absent or minimal. When present, soft and atmospheric.",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MAJOR, Mode.MINOR, Mode.LYDIAN, Mode.MIXOLYDIAN],
            typical_progressions=[
                "I",  # Drone/static
                "I-IV",
                "i-VII",
                "I-vi",
            ],
            tension_level=0.2,
            jazz_influence=0.0,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=3.0,
            interval_vocabulary=["M2", "M3", "P4", "P5", "octave"],
            contour_preferences=["floating", "static", "slow_wave"],
            phrase_lengths=[8, 16, 32],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=300, max_seconds=1200),
            common_structures=[
                "gradual-evolution",
                "A-A'-A''",
                "continuous-flow",
            ],
            energy_curve_type="maintain",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "synthesizer pads",
                "reverb/delay",
                "atmospheric textures",
            ],
            common=[
                "field recordings",
                "piano (processed)",
                "strings (real or synth)",
                "bells/chimes",
                "guitar (heavily processed)",
            ],
            forbidden=[
                "heavy drums",
                "distorted guitars",
                "aggressive sounds",
                "fast rhythms",
            ],
        ),
        production=ProductionProfile(
            era_reference="1970s-present",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.4,
                vocal_forwardness=0.2,
                brightness=0.4,
                width=0.9,
                vintage_warmth=0.6,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-20.0,
                loudness_lufs_max=-14.0,
                dynamic_range_lu_min=12.0,
                dynamic_range_lu_max=15.0,
            ),
            signature_effects=[
                "long reverb tails",
                "extreme delay",
                "granular processing",
                "subtle modulation",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Atmosphere",
                    weight=0.35,
                    criteria=["Immersive space", "Emotional depth", "Transportive quality"],
                    scoring_guide="5=transcendent, 3=acceptable, 1=not ambient",
                ),
                RubricDimension(
                    name="Sound Design",
                    weight=0.25,
                    criteria=["Textural interest", "Evolving sounds", "Cohesive palette"],
                    scoring_guide="5=beautiful textures, 3=acceptable, 1=boring/harsh",
                ),
                RubricDimension(
                    name="Dynamics",
                    weight=0.2,
                    criteria=["Breathing room", "Gradual evolution", "Not fatiguing"],
                    scoring_guide="5=perfect dynamics, 3=acceptable, 1=too loud/static",
                ),
                RubricDimension(
                    name="Functionality",
                    weight=0.2,
                    criteria=["Works as background", "Non-intrusive", "Enhances space"],
                    scoring_guide="5=perfect ambient, 3=acceptable, 1=too demanding",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_rnb_profile() -> GenreRootProfile:
    """Create the R&B genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="r-and-b",
        name="R&B",
        aliases=["Rhythm and Blues", "Contemporary R&B", "Soul"],
        lineage=GenreLineage(
            ancestors=["gospel", "blues", "jazz", "doo-wop"],
            influences=["pop", "funk", "hip-hop"],
            descendants=["neo-soul", "alternative-r&b", "pbr&b"],
            siblings=["soul", "funk"],
        ),
        historical_context=HistoricalContext(
            emergence_era="1940s",
            emergence_year=1947,
            geographic_origin="United States (multiple cities)",
            cultural_context="Originally a marketing term for African American popular music, evolved through soul, quiet storm, new jack swing, to contemporary R&B.",
            socioeconomic_factors=[
                "African American artistic expression",
                "Record industry marketing",
                "Civil rights movement",
                "Hip-hop crossover",
            ],
            key_innovations=[
                "Melismatic vocal techniques",
                "Gospel-influenced harmonies",
                "Syncopated rhythms",
                "Production sophistication",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=60, max_bpm=110, typical_bpm=85),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["swing", "laid_back"],
            swing_amount_min=0.1,
            swing_amount_max=0.25,
            swing_amount_typical=0.15,
            signature_patterns=["groove_pocket", "syncopated"],
            drum_characteristics={
                "kick": "Deep, punchy, often syncopated",
                "snare": "Snappy with ghost notes, on 2 and 4",
                "hihat": "16th note patterns, subtle swing",
                "overall": "Smooth, groovy, pocket-focused",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MAJOR, Mode.MINOR, Mode.DORIAN, Mode.MIXOLYDIAN],
            typical_progressions=[
                "I-vi-IV-V",
                "ii-V-I-vi",
                "I-V-vi-IV",
                "iii-vi-ii-V",
                "I-IV-vi-V",
            ],
            tension_level=0.4,
            jazz_influence=0.7,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=2.5,
            interval_vocabulary=["M2", "m3", "M3", "P4", "P5", "M6", "m7", "M7"],
            contour_preferences=["melismatic", "arch", "wave"],
            phrase_lengths=[2, 4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=180, max_seconds=300),
            common_structures=[
                "intro-verse-prechorus-chorus-verse-prechorus-chorus-bridge-chorus-outro",
                "intro-verse-chorus-verse-chorus-bridge-chorus-outro",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "vocals (lead + harmonies)",
                "bass (synth or electric)",
                "drums/beats",
                "keys (piano/rhodes)",
            ],
            common=[
                "strings (real or synth)",
                "guitar (clean)",
                "synth pads",
                "horn section",
            ],
            forbidden=[
                "heavy metal guitars",
                "aggressive sounds",
                "punk aesthetics",
            ],
        ),
        production=ProductionProfile(
            era_reference="1990s-present",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.7,
                vocal_forwardness=0.9,
                brightness=0.5,
                width=0.7,
                vintage_warmth=0.6,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-12.0,
                loudness_lufs_max=-8.0,
                dynamic_range_lu_min=7.0,
                dynamic_range_lu_max=10.0,
            ),
            signature_effects=[
                "vocal reverb",
                "smooth compression",
                "subtle saturation",
                "stereo widening",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Vocal Performance",
                    weight=0.35,
                    criteria=["Soulful delivery", "Proper technique", "Emotional connection"],
                    scoring_guide="5=exceptional vocals, 3=acceptable, 1=not R&B style",
                ),
                RubricDimension(
                    name="Groove",
                    weight=0.25,
                    criteria=["Pocket feel", "Swing/syncopation", "Smoothness"],
                    scoring_guide="5=irresistible groove, 3=acceptable, 1=stiff",
                ),
                RubricDimension(
                    name="Harmony",
                    weight=0.2,
                    criteria=["Rich chords", "Vocal harmonies", "Jazz influence"],
                    scoring_guide="5=sophisticated harmony, 3=acceptable, 1=too basic",
                ),
                RubricDimension(
                    name="Production",
                    weight=0.2,
                    criteria=["Smooth mix", "Vocals forward", "Modern sound"],
                    scoring_guide="5=radio-ready, 3=acceptable, 1=amateur",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_funk_profile() -> GenreRootProfile:
    """Create the Funk genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="funk",
        name="Funk",
        aliases=["Classic Funk", "P-Funk", "Boogie"],
        lineage=GenreLineage(
            ancestors=["soul", "r&b", "jazz", "gospel"],
            influences=["blues", "psychedelic-rock"],
            descendants=["g-funk", "electro-funk", "nu-funk"],
            siblings=["disco", "soul"],
        ),
        historical_context=HistoricalContext(
            emergence_era="mid 1960s",
            emergence_year=1965,
            geographic_origin="United States (James Brown's influence)",
            cultural_context="James Brown's rhythmic innovations created the foundation. Parliament-Funkadelic expanded it into a cosmic philosophy. Rhythm became the message.",
            socioeconomic_factors=[
                "Civil rights movement",
                "Black pride and expression",
                "Polyrhythmic African heritage",
                "Live performance culture",
            ],
            key_innovations=[
                "The 'One' emphasis",
                "Syncopated bass lines",
                "Horn stabs",
                "Rhythmic guitar scratching",
                "Extended grooves",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=95, max_bpm=125, typical_bpm=108),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["syncopated", "swing"],
            swing_amount_min=0.1,
            swing_amount_max=0.2,
            swing_amount_typical=0.15,
            signature_patterns=["the_one", "syncopated_sixteenths"],
            drum_characteristics={
                "kick": "Emphasis on the ONE, syncopated patterns",
                "snare": "Ghost notes essential, snappy backbeat",
                "hihat": "16th notes, open accents",
                "overall": "Tight, syncopated, groove-heavy",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MINOR, Mode.DORIAN, Mode.MIXOLYDIAN],
            typical_progressions=[
                "i",  # One chord vamps common
                "i-IV",
                "i-iv7",
                "I7-IV7",
                "i-VII-IV",
            ],
            tension_level=0.5,
            jazz_influence=0.6,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=1.5,
            interval_vocabulary=["M2", "m3", "P4", "P5", "m7"],
            contour_preferences=["rhythmic", "repetitive", "call_response"],
            phrase_lengths=[2, 4],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=240, max_seconds=420),
            common_structures=[
                "intro-groove-verse-chorus-groove-verse-chorus-breakdown-groove-outro",
                "intro-vamp-variations-breakdown-vamp-outro",
            ],
            energy_curve_type="maintain",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "bass (slap/fingerstyle)",
                "drums",
                "rhythm guitar (clean/wah)",
                "keyboards (clavinet/organ)",
            ],
            common=[
                "horn section",
                "lead guitar",
                "congas/percussion",
                "backing vocals",
            ],
            forbidden=[
                "heavy distortion",
                "EDM elements",
                "autotune",
            ],
        ),
        production=ProductionProfile(
            era_reference="1970s",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.8,
                vocal_forwardness=0.7,
                brightness=0.6,
                width=0.6,
                vintage_warmth=0.8,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-14.0,
                loudness_lufs_max=-10.0,
                dynamic_range_lu_min=8.0,
                dynamic_range_lu_max=12.0,
            ),
            signature_effects=[
                "wah-wah guitar",
                "room reverb",
                "tape saturation",
                "phaser on keys",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Groove/Pocket",
                    weight=0.35,
                    criteria=["The ONE emphasis", "Syncopation", "Tight rhythm section"],
                    scoring_guide="5=undeniable funk, 3=acceptable, 1=no groove",
                ),
                RubricDimension(
                    name="Bass Line",
                    weight=0.25,
                    criteria=["Syncopated", "Melodic but rhythmic", "Groove-driving"],
                    scoring_guide="5=classic funk bass, 3=acceptable, 1=wrong style",
                ),
                RubricDimension(
                    name="Instrumentation",
                    weight=0.2,
                    criteria=["Proper instruments", "Horn stabs", "Rhythm guitar"],
                    scoring_guide="5=authentic funk band, 3=acceptable, 1=wrong instruments",
                ),
                RubricDimension(
                    name="Feel",
                    weight=0.2,
                    criteria=["Makes you move", "Head-nod factor", "Infectious energy"],
                    scoring_guide="5=impossible not to dance, 3=acceptable, 1=lifeless",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_disco_profile() -> GenreRootProfile:
    """Create the Disco genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="disco",
        name="Disco",
        aliases=["Classic Disco", "Eurodisco", "Nu-Disco"],
        lineage=GenreLineage(
            ancestors=["soul", "funk", "philly-soul", "motown"],
            influences=["latin", "european-electronic"],
            descendants=["house", "nu-disco", "italo-disco"],
            siblings=["funk", "boogie"],
        ),
        historical_context=HistoricalContext(
            emergence_era="early 1970s",
            emergence_year=1973,
            geographic_origin="New York City (underground clubs)",
            cultural_context="Born in New York's underground LGBTQ+ and Black/Latino club scenes. Became a symbol of liberation and hedonism in the late 70s.",
            socioeconomic_factors=[
                "Post-Vietnam/Watergate escapism",
                "LGBTQ+ liberation movement",
                "Urban club culture",
                "Record industry boom",
            ],
            key_innovations=[
                "Four-on-floor kick",
                "Orchestral arrangements in pop",
                "Extended 12-inch mixes",
                "DJ culture emergence",
                "Synthesizer integration",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=110, max_bpm=135, typical_bpm=120),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["straight", "driving"],
            swing_amount_min=0.0,
            swing_amount_max=0.1,
            swing_amount_typical=0.05,
            signature_patterns=["four_on_floor", "offbeat_hihat"],
            drum_characteristics={
                "kick": "Four-on-floor, punchy and consistent",
                "snare": "On 2 and 4, often layered with handclaps",
                "hihat": "Open on offbeats, sizzling 16ths",
                "overall": "Driving, danceable, energetic",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MAJOR, Mode.MINOR, Mode.DORIAN],
            typical_progressions=[
                "I-vi-IV-V",
                "i-VII-VI-VII",
                "ii-V-I-vi",
                "I-IV-vi-V",
                "i-iv-VII-III",
            ],
            tension_level=0.4,
            jazz_influence=0.5,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=2.0,
            interval_vocabulary=["M2", "M3", "P4", "P5", "M6", "octave"],
            contour_preferences=["ascending", "arch", "euphoric"],
            phrase_lengths=[4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=300, max_seconds=480),
            common_structures=[
                "intro-verse-chorus-verse-chorus-breakdown-chorus-outro",
                "intro-build-verse-chorus-instrumental-chorus-outro",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "drums (four-on-floor)",
                "bass (electric or synth)",
                "strings (orchestral)",
                "vocals",
            ],
            common=[
                "horns",
                "piano/rhodes",
                "rhythm guitar",
                "synth leads",
                "backing vocals",
                "percussion (congas, timbales)",
            ],
            forbidden=[
                "heavy metal guitars",
                "aggressive sounds",
                "slow tempos",
            ],
        ),
        production=ProductionProfile(
            era_reference="1970s",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.7,
                vocal_forwardness=0.8,
                brightness=0.7,
                width=0.8,
                vintage_warmth=0.6,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-12.0,
                loudness_lufs_max=-8.0,
                dynamic_range_lu_min=7.0,
                dynamic_range_lu_max=10.0,
            ),
            signature_effects=[
                "lush reverb",
                "string arrangements",
                "phasing",
                "compression",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Danceability",
                    weight=0.3,
                    criteria=["Four-on-floor groove", "Uplifting energy", "Club-ready"],
                    scoring_guide="5=irresistible dance track, 3=acceptable, 1=not danceable",
                ),
                RubricDimension(
                    name="Production Style",
                    weight=0.25,
                    criteria=["Orchestral elements", "Lush arrangements", "Period-appropriate"],
                    scoring_guide="5=classic disco production, 3=acceptable, 1=wrong era",
                ),
                RubricDimension(
                    name="Melody & Vocals",
                    weight=0.25,
                    criteria=["Catchy hooks", "Euphoric melodies", "Strong vocal"],
                    scoring_guide="5=anthemic, 3=acceptable, 1=forgettable",
                ),
                RubricDimension(
                    name="Arrangement",
                    weight=0.2,
                    criteria=["Build and release", "String swells", "Dynamic structure"],
                    scoring_guide="5=perfect disco arrangement, 3=acceptable, 1=flat",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_trap_profile() -> GenreRootProfile:
    """Create the Trap genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="trap",
        name="Trap",
        aliases=["Trap Music", "Atlanta Trap", "Modern Hip-Hop"],
        lineage=GenreLineage(
            ancestors=["southern-hip-hop", "crunk", "gangsta-rap"],
            influences=["edm", "drill", "bounce"],
            descendants=["drill", "phonk", "hyperpop"],
            siblings=["drill", "cloud-rap"],
        ),
        historical_context=HistoricalContext(
            emergence_era="early 2000s",
            emergence_year=2003,
            geographic_origin="Atlanta, Georgia",
            cultural_context="Originated in Atlanta's Southern hip-hop scene, named after trap houses. T.I., Gucci Mane, and Young Jeezy pioneered the sound.",
            socioeconomic_factors=[
                "Southern hip-hop dominance",
                "Street culture documentation",
                "DIY production tools",
                "Streaming era influence",
            ],
            key_innovations=[
                "808 sub bass dominance",
                "Hi-hat rolls and triplets",
                "Pitched snares",
                "Dark, minor key atmospheres",
                "Ad-lib culture",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=130, max_bpm=175, typical_bpm=145),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["half_time", "triplet"],
            swing_amount_min=0.0,
            swing_amount_max=0.1,
            swing_amount_typical=0.0,
            signature_patterns=["trap_hihat_rolls", "half_time_kick"],
            drum_characteristics={
                "kick": "808 sub bass kicks, half-time feel",
                "snare": "Pitched snares/claps on 3, sometimes with reverb",
                "hihat": "Rapid triplet rolls, open hat accents",
                "overall": "Hard-hitting, bass-heavy, aggressive",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MINOR, Mode.PHRYGIAN, Mode.AEOLIAN],
            typical_progressions=[
                "i-VI-VII-i",
                "i-VII-VI-VII",
                "i-iv-VII-VI",
                "i",  # Drone common
            ],
            tension_level=0.7,
            jazz_influence=0.1,
            modal_interchange_common=False,
        ),
        melody=MelodyProfile(
            typical_range_octaves=1.5,
            interval_vocabulary=["m2", "m3", "P4", "P5", "m7"],
            contour_preferences=["descending", "static", "minimal"],
            phrase_lengths=[2, 4],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=150, max_seconds=240),
            common_structures=[
                "intro-verse-hook-verse-hook-verse-hook-outro",
                "intro-hook-verse-hook-verse-hook-outro",
            ],
            energy_curve_type="maintain",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "808 sub bass",
                "hi-hats (rapid)",
                "snare/clap",
                "dark synth pads",
            ],
            common=[
                "piano (dark)",
                "bell melodies",
                "orchestral hits",
                "vocal chops",
                "strings (ominous)",
            ],
            forbidden=[
                "acoustic drums",
                "major key melodies",
                "boom-bap drums",
            ],
        ),
        production=ProductionProfile(
            era_reference="2010s-present",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.95,
                vocal_forwardness=0.8,
                brightness=0.5,
                width=0.6,
                vintage_warmth=0.2,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-10.0,
                loudness_lufs_max=-6.0,
                dynamic_range_lu_min=5.0,
                dynamic_range_lu_max=8.0,
            ),
            signature_effects=[
                "808 distortion",
                "reverb on snares",
                "pitch-shifting",
                "hard compression",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="808 Bass",
                    weight=0.3,
                    criteria=["Powerful sub", "Proper tuning", "Pattern complexity"],
                    scoring_guide="5=chest-rattling 808s, 3=acceptable, 1=weak bass",
                ),
                RubricDimension(
                    name="Hi-Hat Patterns",
                    weight=0.25,
                    criteria=["Triplet rolls", "Velocity variation", "Signature patterns"],
                    scoring_guide="5=authentic trap hats, 3=acceptable, 1=wrong style",
                ),
                RubricDimension(
                    name="Dark Atmosphere",
                    weight=0.25,
                    criteria=["Minor key", "Ominous vibe", "Hard-hitting"],
                    scoring_guide="5=authentic trap mood, 3=acceptable, 1=too bright",
                ),
                RubricDimension(
                    name="Arrangement",
                    weight=0.2,
                    criteria=["Hook-focused", "Hard drops", "Energy maintenance"],
                    scoring_guide="5=classic trap structure, 3=acceptable, 1=wrong format",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_drum_and_bass_profile() -> GenreRootProfile:
    """Create the Drum and Bass genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="drum-and-bass",
        name="Drum and Bass",
        aliases=["DnB", "Jungle", "Liquid DnB"],
        lineage=GenreLineage(
            ancestors=["jungle", "breakbeat", "rave", "reggae"],
            influences=["dub", "hip-hop", "techno"],
            descendants=["neurofunk", "liquid-dnb", "jump-up"],
            siblings=["breakbeat", "jungle"],
        ),
        historical_context=HistoricalContext(
            emergence_era="early 1990s",
            emergence_year=1992,
            geographic_origin="United Kingdom (London)",
            cultural_context="Evolved from jungle and breakbeat hardcore in the UK rave scene. Characterized by fast breakbeats and heavy bass.",
            socioeconomic_factors=[
                "UK rave culture",
                "Jamaican sound system influence",
                "Working class youth expression",
                "Pirate radio culture",
            ],
            key_innovations=[
                "Fast breakbeat manipulation",
                "Heavy sub bass",
                "Time-stretching technology",
                "Reese bass",
                "Amen break variations",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=160, max_bpm=180, typical_bpm=174),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["broken", "syncopated"],
            swing_amount_min=0.0,
            swing_amount_max=0.15,
            swing_amount_typical=0.05,
            signature_patterns=["two_step", "amen_break"],
            drum_characteristics={
                "kick": "Punchy, fast, often syncopated",
                "snare": "Sharp, on 2 and 4 but heavily syncopated",
                "hihat": "Fast patterns, ghost notes",
                "overall": "Breakbeat-based, fast, energetic",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MINOR, Mode.DORIAN, Mode.PHRYGIAN],
            typical_progressions=[
                "i-VII-VI-VII",
                "i-iv-VII-III",
                "i-VI-III-VII",
                "i",  # Bass-focused tracks
            ],
            tension_level=0.6,
            jazz_influence=0.3,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=2.0,
            interval_vocabulary=["m2", "M2", "m3", "P4", "P5", "m7"],
            contour_preferences=["descending", "wave", "angular"],
            phrase_lengths=[4, 8, 16],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=300, max_seconds=420),
            common_structures=[
                "intro-build-drop-breakdown-build-drop-outro",
                "intro-drop-break-drop-break-drop-outro",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "breakbeats",
                "sub bass (reese/wobble)",
                "synth stabs",
            ],
            common=[
                "pads",
                "vocal samples",
                "piano",
                "strings",
                "fx/risers",
            ],
            forbidden=[
                "acoustic drums (raw)",
                "slow tempos",
                "four-on-floor kick",
            ],
        ),
        production=ProductionProfile(
            era_reference="1990s-present",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.9,
                vocal_forwardness=0.5,
                brightness=0.6,
                width=0.7,
                vintage_warmth=0.3,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-10.0,
                loudness_lufs_max=-6.0,
                dynamic_range_lu_min=6.0,
                dynamic_range_lu_max=9.0,
            ),
            signature_effects=[
                "heavy compression on drums",
                "distortion on bass",
                "reverb throws",
                "filter automation",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Breakbeat Quality",
                    weight=0.3,
                    criteria=["Proper tempo", "Good breaks", "Syncopation"],
                    scoring_guide="5=authentic dnb breaks, 3=acceptable, 1=wrong rhythm",
                ),
                RubricDimension(
                    name="Bass Design",
                    weight=0.3,
                    criteria=["Sub weight", "Movement", "Proper tone"],
                    scoring_guide="5=system-shaking bass, 3=acceptable, 1=weak bass",
                ),
                RubricDimension(
                    name="Energy",
                    weight=0.2,
                    criteria=["Fast pace", "Intensity", "Build/release"],
                    scoring_guide="5=high energy, 3=acceptable, 1=flat",
                ),
                RubricDimension(
                    name="Production",
                    weight=0.2,
                    criteria=["Clean mix", "Punchy drums", "Modern sound"],
                    scoring_guide="5=club-ready, 3=acceptable, 1=amateur",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_dubstep_profile() -> GenreRootProfile:
    """Create the Dubstep genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="dubstep",
        name="Dubstep",
        aliases=["Brostep", "Bass Music", "UK Dubstep"],
        lineage=GenreLineage(
            ancestors=["dub", "2-step-garage", "grime", "drum-and-bass"],
            influences=["reggae", "jungle", "uk-garage"],
            descendants=["riddim", "brostep", "melodic-dubstep"],
            siblings=["grime", "uk-bass"],
        ),
        historical_context=HistoricalContext(
            emergence_era="late 1990s",
            emergence_year=1998,
            geographic_origin="South London, UK",
            cultural_context="Emerged from the UK garage scene in South London. Pioneered by producers like Skream, Benga, and Digital Mystikz.",
            socioeconomic_factors=[
                "UK garage evolution",
                "Pirate radio culture",
                "Club culture",
                "Global internet spread",
            ],
            key_innovations=[
                "Wobble bass",
                "Half-time rhythms",
                "Heavy sub bass focus",
                "Sound design as composition",
                "The drop",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=138, max_bpm=150, typical_bpm=140),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["half_time"],
            swing_amount_min=0.0,
            swing_amount_max=0.1,
            swing_amount_typical=0.0,
            signature_patterns=["half_time_snare", "wobble_bass"],
            drum_characteristics={
                "kick": "Heavy, punchy, on 1",
                "snare": "Powerful, on 3 (half-time feel)",
                "hihat": "Sparse or busy depending on section",
                "overall": "Heavy, spacious, bass-focused",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MINOR, Mode.PHRYGIAN],
            typical_progressions=[
                "i-VII-VI-VII",
                "i-iv-VII-VI",
                "i",  # Drone/single chord drops
            ],
            tension_level=0.8,
            jazz_influence=0.0,
            modal_interchange_common=False,
        ),
        melody=MelodyProfile(
            typical_range_octaves=2.0,
            interval_vocabulary=["m2", "m3", "P4", "P5", "m7"],
            contour_preferences=["descending", "angular", "minimal"],
            phrase_lengths=[4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=240, max_seconds=360),
            common_structures=[
                "intro-build-drop-break-build-drop-outro",
                "intro-verse-build-drop-verse-build-drop-outro",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "sub bass",
                "wobble/growl bass",
                "drums (sparse)",
                "fx/risers",
            ],
            common=[
                "synth leads",
                "vocal chops",
                "pads",
                "piano",
            ],
            forbidden=[
                "acoustic drums",
                "major key melodies (mostly)",
                "soft sounds during drops",
            ],
        ),
        production=ProductionProfile(
            era_reference="2000s-present",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.95,
                vocal_forwardness=0.4,
                brightness=0.5,
                width=0.7,
                vintage_warmth=0.1,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-9.0,
                loudness_lufs_max=-6.0,
                dynamic_range_lu_min=5.0,
                dynamic_range_lu_max=8.0,
            ),
            signature_effects=[
                "heavy bass processing",
                "formant filters",
                "distortion",
                "sidechain compression",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Bass Design",
                    weight=0.35,
                    criteria=["Wobble quality", "Sub weight", "Sound design creativity"],
                    scoring_guide="5=face-melting bass, 3=acceptable, 1=weak/generic",
                ),
                RubricDimension(
                    name="The Drop",
                    weight=0.25,
                    criteria=["Impact", "Build quality", "Energy release"],
                    scoring_guide="5=devastating drop, 3=acceptable, 1=anticlimactic",
                ),
                RubricDimension(
                    name="Rhythm",
                    weight=0.2,
                    criteria=["Half-time feel", "Proper tempo", "Groove"],
                    scoring_guide="5=authentic dubstep rhythm, 3=acceptable, 1=wrong feel",
                ),
                RubricDimension(
                    name="Production Quality",
                    weight=0.2,
                    criteria=["Clean mix", "Powerful master", "Professional sound"],
                    scoring_guide="5=release-ready, 3=acceptable, 1=amateur",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_acoustic_folk_profile() -> GenreRootProfile:
    """Create the Acoustic Folk genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="acoustic-folk",
        name="Acoustic Folk",
        aliases=["Folk", "Indie Folk", "Contemporary Folk"],
        lineage=GenreLineage(
            ancestors=["traditional-folk", "blues", "country", "appalachian"],
            influences=["protest-music", "singer-songwriter"],
            descendants=["indie-folk", "folk-rock", "americana"],
            siblings=["country", "bluegrass"],
        ),
        historical_context=HistoricalContext(
            emergence_era="1940s-1960s",
            emergence_year=1958,
            geographic_origin="United States and United Kingdom",
            cultural_context="The folk revival brought traditional music to popular audiences. Woody Guthrie, Pete Seeger, Bob Dylan, and Joan Baez shaped its modern form.",
            socioeconomic_factors=[
                "Civil rights movement",
                "Anti-war protests",
                "Counterculture",
                "Authenticity seeking",
            ],
            key_innovations=[
                "Protest song tradition",
                "Personal storytelling",
                "Fingerpicking patterns",
                "Harmonica integration",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=60, max_bpm=140, typical_bpm=100),
        rhythm=RhythmProfile(
            time_signatures=["4/4", "3/4", "6/8"],
            feels=["straight", "light_swing"],
            swing_amount_min=0.0,
            swing_amount_max=0.15,
            swing_amount_typical=0.05,
            signature_patterns=["fingerpicking", "strumming"],
            drum_characteristics={
                "overall": "Often minimal or absent. When present, brushes or light percussion.",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MAJOR, Mode.MINOR, Mode.MIXOLYDIAN, Mode.DORIAN],
            typical_progressions=[
                "I-IV-V-I",
                "I-V-vi-IV",
                "I-vi-IV-V",
                "i-VII-VI-VII",
                "I-IV-I-V",
            ],
            tension_level=0.3,
            jazz_influence=0.2,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=1.5,
            interval_vocabulary=["M2", "m3", "M3", "P4", "P5", "M6"],
            contour_preferences=["arch", "wave", "storytelling"],
            phrase_lengths=[4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=150, max_seconds=300),
            common_structures=[
                "intro-verse-chorus-verse-chorus-verse-chorus-outro",
                "intro-verse-verse-chorus-verse-chorus-outro",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "acoustic guitar",
                "vocals",
            ],
            common=[
                "harmonica",
                "banjo",
                "mandolin",
                "fiddle",
                "upright bass",
                "piano",
                "light percussion",
            ],
            forbidden=[
                "heavy electric guitars",
                "synthesizers",
                "electronic drums",
                "autotune",
            ],
        ),
        production=ProductionProfile(
            era_reference="1960s-present",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.4,
                vocal_forwardness=0.85,
                brightness=0.5,
                width=0.5,
                vintage_warmth=0.7,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-18.0,
                loudness_lufs_max=-12.0,
                dynamic_range_lu_min=10.0,
                dynamic_range_lu_max=15.0,
            ),
            signature_effects=[
                "room reverb",
                "minimal processing",
                "tape warmth",
                "natural dynamics",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Acoustic Sound",
                    weight=0.3,
                    criteria=["Natural tones", "Proper recording", "Organic feel"],
                    scoring_guide="5=beautiful acoustic sound, 3=acceptable, 1=too processed",
                ),
                RubricDimension(
                    name="Songwriting",
                    weight=0.3,
                    criteria=["Storytelling", "Emotional depth", "Lyrical quality"],
                    scoring_guide="5=compelling story, 3=acceptable, 1=weak writing",
                ),
                RubricDimension(
                    name="Performance",
                    weight=0.2,
                    criteria=["Authentic delivery", "Technical skill", "Emotional connection"],
                    scoring_guide="5=moving performance, 3=acceptable, 1=lifeless",
                ),
                RubricDimension(
                    name="Arrangement",
                    weight=0.2,
                    criteria=["Appropriate instrumentation", "Not over-produced", "Serves the song"],
                    scoring_guide="5=perfect arrangement, 3=acceptable, 1=wrong style",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_cinematic_profile() -> GenreRootProfile:
    """Create the Cinematic/Film Score genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="cinematic",
        name="Cinematic",
        aliases=["Film Score", "Epic Music", "Trailer Music", "Orchestral"],
        lineage=GenreLineage(
            ancestors=["classical", "romantic-era", "opera"],
            influences=["electronic", "world-music", "minimalism"],
            descendants=["trailer-music", "epic-orchestral"],
            siblings=["neo-classical", "ambient"],
        ),
        historical_context=HistoricalContext(
            emergence_era="1930s",
            emergence_year=1933,
            geographic_origin="Hollywood, United States",
            cultural_context="Film music evolved from silent film accompaniment to a sophisticated art form. John Williams, Hans Zimmer, and Ennio Morricone defined the modern era.",
            socioeconomic_factors=[
                "Hollywood studio system",
                "Global film industry",
                "Trailer music industry",
                "Game/media music expansion",
            ],
            key_innovations=[
                "Leitmotif technique",
                "Orchestral + electronic hybrid",
                "Emotional manipulation through music",
                "Tempo and key changes for drama",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=40, max_bpm=180, typical_bpm=90),
        rhythm=RhythmProfile(
            time_signatures=["4/4", "3/4", "6/8", "12/8"],
            feels=["straight", "epic"],
            swing_amount_min=0.0,
            swing_amount_max=0.05,
            swing_amount_typical=0.0,
            signature_patterns=["epic_pulse", "dramatic_hits"],
            drum_characteristics={
                "overall": "Orchestral percussion: timpani, taikos, cymbals, snare rolls. Epic hits and builds.",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MAJOR, Mode.MINOR, Mode.LYDIAN, Mode.PHRYGIAN],
            typical_progressions=[
                "i-VII-VI-VII",
                "I-V-vi-IV",
                "i-VI-III-VII",
                "I-IV-V-I",
                "i-iv-VII-III",
            ],
            tension_level=0.6,
            jazz_influence=0.1,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=3.0,
            interval_vocabulary=["M2", "m3", "M3", "P4", "P5", "M6", "octave"],
            contour_preferences=["heroic", "sweeping", "emotional"],
            phrase_lengths=[4, 8, 16],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=120, max_seconds=600),
            common_structures=[
                "intro-theme-development-climax-resolution",
                "build-build-climax-denouement",
                "A-B-A-development-climax",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "strings (orchestra)",
                "brass",
                "woodwinds",
                "percussion (orchestral)",
            ],
            common=[
                "choir",
                "piano",
                "synthesizers",
                "ethnic instruments",
                "hybrid percussion",
            ],
            forbidden=[
                "trap beats",
                "autotune vocals",
                "distorted guitars (usually)",
            ],
        ),
        production=ProductionProfile(
            era_reference="2000s-present",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.6,
                vocal_forwardness=0.5,
                brightness=0.6,
                width=0.9,
                vintage_warmth=0.4,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-18.0,
                loudness_lufs_max=-10.0,
                dynamic_range_lu_min=10.0,
                dynamic_range_lu_max=15.0,
            ),
            signature_effects=[
                "hall reverb",
                "orchestral space",
                "dynamic contrast",
                "subtle compression",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Emotional Impact",
                    weight=0.3,
                    criteria=["Evokes emotion", "Cinematic feel", "Memorable themes"],
                    scoring_guide="5=goosebumps, 3=acceptable, 1=flat/boring",
                ),
                RubricDimension(
                    name="Orchestration",
                    weight=0.3,
                    criteria=["Proper voicing", "Balance", "Instrument choices"],
                    scoring_guide="5=masterful orchestration, 3=acceptable, 1=poor choices",
                ),
                RubricDimension(
                    name="Production Quality",
                    weight=0.2,
                    criteria=["Professional sound", "Proper space", "Clean mix"],
                    scoring_guide="5=Hollywood quality, 3=acceptable, 1=amateur",
                ),
                RubricDimension(
                    name="Structure",
                    weight=0.2,
                    criteria=["Dramatic arc", "Build and release", "Satisfying resolution"],
                    scoring_guide="5=perfect storytelling, 3=acceptable, 1=no arc",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_chillwave_profile() -> GenreRootProfile:
    """Create the Chillwave genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="chillwave",
        name="Chillwave",
        aliases=["Glo-Fi", "Hypnagogic Pop", "Dream Pop Electronic"],
        lineage=GenreLineage(
            ancestors=["synth-pop", "dream-pop", "shoegaze", "new-wave"],
            influences=["ambient", "lo-fi", "vaporwave"],
            descendants=["vaporwave", "synth-pop-revival"],
            siblings=["dream-pop", "synthwave"],
        ),
        historical_context=HistoricalContext(
            emergence_era="late 2000s",
            emergence_year=2009,
            geographic_origin="United States (internet-based)",
            cultural_context="Emerged from the blogosphere. Washed Out, Toro y Moi, and Neon Indian defined the hazy, nostalgic sound.",
            socioeconomic_factors=[
                "Music blog culture",
                "DIY bedroom production",
                "80s nostalgia",
                "Digital distribution",
            ],
            key_innovations=[
                "Intentional lo-fi aesthetic",
                "Heavy reverb/wash",
                "Nostalgic synthesizer tones",
                "Hazy, dreamlike production",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=80, max_bpm=120, typical_bpm=100),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["laid_back", "dreamy"],
            swing_amount_min=0.0,
            swing_amount_max=0.1,
            swing_amount_typical=0.05,
            signature_patterns=["four_on_floor_soft", "shuffle"],
            drum_characteristics={
                "kick": "Soft, often buried in mix",
                "snare": "Washed out, reverbed",
                "hihat": "Gentle, swishing",
                "overall": "Hazy, distant, lo-fi feel",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MAJOR, Mode.MINOR, Mode.LYDIAN, Mode.MIXOLYDIAN],
            typical_progressions=[
                "I-V-vi-IV",
                "I-IV-vi-V",
                "vi-IV-I-V",
                "I-vi-IV-V",
            ],
            tension_level=0.3,
            jazz_influence=0.2,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=1.5,
            interval_vocabulary=["M2", "M3", "P4", "P5", "M6"],
            contour_preferences=["wave", "floating", "dreamy"],
            phrase_lengths=[4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=180, max_seconds=300),
            common_structures=[
                "intro-verse-chorus-verse-chorus-outro",
                "intro-A-B-A-B-outro",
            ],
            energy_curve_type="maintain",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "synthesizers (vintage)",
                "drum machine",
                "vocals (washed)",
                "reverb/delay",
            ],
            common=[
                "guitar (clean, reverbed)",
                "samples",
                "bass synth",
                "pads",
            ],
            forbidden=[
                "aggressive sounds",
                "heavy distortion",
                "clean/dry production",
            ],
        ),
        production=ProductionProfile(
            era_reference="2010s",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.5,
                vocal_forwardness=0.5,
                brightness=0.4,
                width=0.8,
                vintage_warmth=0.8,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-16.0,
                loudness_lufs_max=-10.0,
                dynamic_range_lu_min=8.0,
                dynamic_range_lu_max=12.0,
            ),
            signature_effects=[
                "heavy reverb",
                "tape saturation",
                "bit crushing",
                "chorus/flanger",
                "lo-fi filtering",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Hazy Aesthetic",
                    weight=0.3,
                    criteria=["Washed out sound", "Nostalgic feel", "Dreamy quality"],
                    scoring_guide="5=perfect haze, 3=acceptable, 1=too clean",
                ),
                RubricDimension(
                    name="Synth Tones",
                    weight=0.25,
                    criteria=["Vintage sounds", "Warm textures", "80s influence"],
                    scoring_guide="5=perfect tones, 3=acceptable, 1=wrong era",
                ),
                RubricDimension(
                    name="Mood",
                    weight=0.25,
                    criteria=["Nostalgic", "Summery", "Melancholic beauty"],
                    scoring_guide="5=evocative mood, 3=acceptable, 1=wrong vibe",
                ),
                RubricDimension(
                    name="Production",
                    weight=0.2,
                    criteria=["Intentional lo-fi", "Cohesive sound", "Not over-produced"],
                    scoring_guide="5=authentic chillwave, 3=acceptable, 1=too polished",
                ),
            ],
            minimum_passing_score=0.8,
        ),
        version="1.0.0",
    )


def _create_neo_soul_profile() -> GenreRootProfile:
    """Create the Neo-Soul genre profile."""
    from aether.schemas.base import DurationRange, Mode, TempoRange
    from aether.schemas.genre import (
        ArrangementProfile,
        AuthenticityRubric,
        GenreLineage,
        HarmonyProfile,
        HistoricalContext,
        InstrumentationProfile,
        MasterTargets,
        MelodyProfile,
        MixCharacteristics,
        ProductionProfile,
        RhythmProfile,
        RubricDimension,
    )

    return GenreRootProfile(
        genre_id="neo-soul",
        name="Neo-Soul",
        aliases=["Nu-Soul", "Progressive Soul", "Conscious R&B"],
        lineage=GenreLineage(
            ancestors=["soul", "r&b", "funk", "jazz"],
            influences=["hip-hop", "gospel", "afrobeat"],
            descendants=["alternative-r&b", "indie-soul"],
            siblings=["r&b", "jazz-fusion"],
        ),
        historical_context=HistoricalContext(
            emergence_era="mid 1990s",
            emergence_year=1994,
            geographic_origin="United States (Philadelphia, Atlanta)",
            cultural_context="A return to classic soul aesthetics with modern production. D'Angelo, Erykah Badu, Lauryn Hill, and Maxwell pioneered the movement.",
            socioeconomic_factors=[
                "Reaction to new jack swing",
                "Hip-hop production influence",
                "Conscious/socially aware themes",
                "Live instrumentation revival",
            ],
            key_innovations=[
                "Jazz-influenced harmony in R&B",
                "Live instrumentation emphasis",
                "Complex rhythmic feel",
                "Conscious lyrics",
                "Vintage production aesthetic",
            ],
        ),
        evolution_timeline=[],
        tempo=TempoRange(min_bpm=65, max_bpm=105, typical_bpm=85),
        rhythm=RhythmProfile(
            time_signatures=["4/4"],
            feels=["swing", "laid_back"],
            swing_amount_min=0.15,
            swing_amount_max=0.3,
            swing_amount_typical=0.2,
            signature_patterns=["neo_soul_groove", "j_dilla_feel"],
            drum_characteristics={
                "kick": "Warm, slightly behind beat",
                "snare": "Ghost notes, rim shots, brushes",
                "hihat": "Swung 16ths, subtle dynamics",
                "overall": "Organic, live feel, J Dilla influence",
            },
        ),
        harmony=HarmonyProfile(
            common_modes=[Mode.MINOR, Mode.DORIAN, Mode.MIXOLYDIAN, Mode.MAJOR],
            typical_progressions=[
                "ii-V-I-vi",
                "I-IV-iii-vi",
                "ii7-V7-Imaj7",
                "i-VII-VI-VII",
                "iii-vi-ii-V",
            ],
            tension_level=0.5,
            jazz_influence=0.9,
            modal_interchange_common=True,
        ),
        melody=MelodyProfile(
            typical_range_octaves=2.0,
            interval_vocabulary=["M2", "m3", "M3", "P4", "P5", "M6", "m7", "M7"],
            contour_preferences=["melismatic", "wave", "soulful"],
            phrase_lengths=[2, 4, 8],
        ),
        arrangement=ArrangementProfile(
            typical_duration=DurationRange(min_seconds=240, max_seconds=360),
            common_structures=[
                "intro-verse-chorus-verse-chorus-bridge-chorus-outro",
                "intro-verse-verse-chorus-verse-chorus-outro",
            ],
            energy_curve_type="build_release",
        ),
        instrumentation=InstrumentationProfile(
            essential=[
                "rhodes/wurlitzer",
                "bass (often moog or upright)",
                "drums (live feel)",
                "vocals",
            ],
            common=[
                "guitar (clean, jazz tone)",
                "strings",
                "horns",
                "background vocals",
                "percussion",
            ],
            forbidden=[
                "aggressive sounds",
                "heavy EDM elements",
                "autotune (usually)",
            ],
        ),
        production=ProductionProfile(
            era_reference="1990s-2000s",
            mix_characteristics=MixCharacteristics(
                low_end_emphasis=0.7,
                vocal_forwardness=0.85,
                brightness=0.4,
                width=0.6,
                vintage_warmth=0.9,
            ),
            master_targets=MasterTargets(
                loudness_lufs_min=-14.0,
                loudness_lufs_max=-10.0,
                dynamic_range_lu_min=9.0,
                dynamic_range_lu_max=13.0,
            ),
            signature_effects=[
                "tape saturation",
                "vintage compression",
                "plate reverb",
                "subtle chorus on keys",
            ],
        ),
        authenticity_rubric=AuthenticityRubric(
            dimensions=[
                RubricDimension(
                    name="Groove/Feel",
                    weight=0.3,
                    criteria=["Behind-the-beat feel", "Organic groove", "J Dilla influence"],
                    scoring_guide="5=perfect neo-soul pocket, 3=acceptable, 1=stiff",
                ),
                RubricDimension(
                    name="Harmony",
                    weight=0.25,
                    criteria=["Jazz chords", "Extended voicings", "Sophisticated progressions"],
                    scoring_guide="5=rich harmony, 3=acceptable, 1=too simple",
                ),
                RubricDimension(
                    name="Vocal/Melody",
                    weight=0.25,
                    criteria=["Soulful delivery", "Melismatic technique", "Emotional depth"],
                    scoring_guide="5=transcendent vocals, 3=acceptable, 1=not soulful",
                ),
                RubricDimension(
                    name="Production",
                    weight=0.2,
                    criteria=["Warm/vintage feel", "Live instrumentation", "Organic sound"],
                    scoring_guide="5=authentic neo-soul, 3=acceptable, 1=too digital",
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
    _create_house_profile(),
    _create_techno_profile(),
    _create_jazz_profile(),
    _create_rock_profile(),
    _create_ambient_profile(),
    _create_rnb_profile(),
    _create_funk_profile(),
    _create_disco_profile(),
    _create_trap_profile(),
    _create_drum_and_bass_profile(),
    _create_dubstep_profile(),
    _create_acoustic_folk_profile(),
    _create_cinematic_profile(),
    _create_chillwave_profile(),
    _create_neo_soul_profile(),
]


# Global instance
_genre_manager: GenreProfileManager | None = None


def get_genre_manager(profiles_dir: Optional[Path] = None) -> GenreProfileManager:
    """Get the global genre profile manager."""
    global _genre_manager
    if _genre_manager is None:
        _genre_manager = GenreProfileManager(profiles_dir)
    return _genre_manager
