"""
AETHER Band Engine - Core Data Schemas

All data flows through strongly-typed Pydantic schemas.
These schemas define the contracts between agents in the pipeline.
"""

from aether.schemas.song import SongSpec
from aether.schemas.harmony import HarmonySpec
from aether.schemas.melody import MelodySpec
from aether.schemas.arrangement import ArrangementSpec
from aether.schemas.rhythm import RhythmSpec
from aether.schemas.lyrics import LyricSpec
from aether.schemas.vocal import VocalSpec
from aether.schemas.sound_design import SoundDesignSpec
from aether.schemas.mix import MixSpec
from aether.schemas.master import MasterSpec
from aether.schemas.qa import QAReport
from aether.schemas.genre import GenreRootProfile

__all__ = [
    "SongSpec",
    "HarmonySpec",
    "MelodySpec",
    "ArrangementSpec",
    "RhythmSpec",
    "LyricSpec",
    "VocalSpec",
    "SoundDesignSpec",
    "MixSpec",
    "MasterSpec",
    "QAReport",
    "GenreRootProfile",
]
