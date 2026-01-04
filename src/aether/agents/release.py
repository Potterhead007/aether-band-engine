"""
Release Agent

Packages the final track with metadata for distribution.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from aether.agents.base import BaseAgent, AgentRegistry
from aether.storage import ArtifactType

logger = logging.getLogger(__name__)


class ReleaseMetadata(BaseModel):
    """Metadata for release."""

    title: str
    artist: str
    album: Optional[str] = None
    genre: str
    subgenre: Optional[str] = None
    year: int
    duration_seconds: float
    bpm: int
    key: str
    mood: str
    tags: List[str] = Field(default_factory=list)
    isrc: Optional[str] = None
    upc: Optional[str] = None
    copyright_holder: str = "AETHER"
    release_date: Optional[datetime] = None
    explicit: bool = False


class ReleaseAsset(BaseModel):
    """A single release asset."""

    asset_type: str = Field(description="audio, artwork, lyrics, video")
    format: str
    file_path: Optional[str] = None
    checksum: Optional[str] = None
    size_bytes: Optional[int] = None


class ReleasePackage(BaseModel):
    """Complete release package."""

    release_id: str
    song_id: str
    metadata: ReleaseMetadata
    assets: List[ReleaseAsset]
    platforms: List[str]
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    qa_passed: bool
    notes: List[str] = Field(default_factory=list)


class ReleaseInput(BaseModel):
    song_spec: Dict[str, Any]
    master_spec: Dict[str, Any]
    lyric_spec: Dict[str, Any]
    qa_report: Dict[str, Any]
    genre_profile_id: str


class ReleaseOutput(BaseModel):
    release_package: Dict[str, Any]
    ready_for_distribution: bool


@AgentRegistry.register("release")
class ReleaseAgent(BaseAgent[ReleaseInput, ReleaseOutput]):
    """
    Release Agent.

    Responsibilities:
    - Generate release metadata (ID3 tags, etc.)
    - Package audio assets in multiple formats
    - Create distribution packages per platform
    - Generate ISRC codes (placeholder)
    - Prepare for content ID registration
    """

    agent_type = "release"
    agent_name = "Release Agent"
    input_schema = ReleaseInput
    output_schema = ReleaseOutput

    # Target platforms
    PLATFORMS = [
        "spotify",
        "apple_music",
        "youtube_music",
        "tidal",
        "amazon_music",
        "deezer",
        "soundcloud",
    ]

    async def process(
        self,
        input_data: ReleaseInput,
        context: Dict[str, Any],
    ) -> ReleaseOutput:
        song_spec = input_data.song_spec
        master_spec = input_data.master_spec
        lyric_spec = input_data.lyric_spec
        qa_report = input_data.qa_report

        qa_passed = qa_report.get("all_passed", False)

        # Generate metadata
        metadata = self._generate_metadata(song_spec, master_spec, input_data.genre_profile_id)

        # Create release assets
        assets = self._create_assets(master_spec, lyric_spec)

        # Create release package
        release_package = ReleasePackage(
            release_id=str(uuid4()),
            song_id=str(song_spec["id"]),
            metadata=metadata,
            assets=assets,
            platforms=self.PLATFORMS if qa_passed else [],
            status="ready" if qa_passed else "blocked",
            qa_passed=qa_passed,
            notes=self._generate_notes(qa_report),
        )

        # Store release artifact
        self.store_artifact(
            data=release_package.model_dump(),
            artifact_type=ArtifactType.RELEASE_PACKAGE,
            name="release_package",
            song_id=str(song_spec["id"]),
        )

        self.log_decision(
            decision_type="release_packaging",
            input_summary=f"Song: {song_spec.get('title', 'Unknown')}, QA: {'PASSED' if qa_passed else 'FAILED'}",
            output_summary=f"Package {'ready' if qa_passed else 'blocked'}, {len(assets)} assets, {len(self.PLATFORMS) if qa_passed else 0} platforms",
            reasoning="Packaged for distribution based on QA status",
            confidence=0.95,
        )

        return ReleaseOutput(
            release_package=release_package.model_dump(),
            ready_for_distribution=qa_passed,
        )

    def _generate_metadata(
        self,
        song_spec: Dict,
        master_spec: Dict,
        genre_id: str,
    ) -> ReleaseMetadata:
        """Generate release metadata."""
        key_info = song_spec.get("key", {})
        key_str = f"{key_info.get('root', 'C')} {key_info.get('mode', 'major')}"

        mood = song_spec.get("primary_mood", "energetic")
        if hasattr(mood, "value"):
            mood = mood.value

        # Generate tags from song attributes
        tags = self._generate_tags(song_spec, genre_id)

        return ReleaseMetadata(
            title=song_spec.get("title", "Untitled"),
            artist=song_spec.get("artist_name", "AETHER"),
            album=None,  # Single release
            genre=genre_id.replace("-", " ").title(),
            year=datetime.utcnow().year,
            duration_seconds=float(song_spec.get("target_duration_seconds", 180)),
            bpm=song_spec.get("bpm", 120),
            key=key_str,
            mood=mood,
            tags=tags,
            isrc=self._generate_placeholder_isrc(),
            copyright_holder="AETHER Music Generation",
            release_date=datetime.utcnow(),
            explicit=False,
        )

    def _create_assets(
        self,
        master_spec: Dict,
        lyric_spec: Dict,
    ) -> List[ReleaseAsset]:
        """Create list of release assets."""
        assets = []

        # Audio formats from master spec
        formats = master_spec.get("formats", ["wav_24_48", "flac_24_48", "mp3_320"])

        for fmt in formats:
            assets.append(
                ReleaseAsset(
                    asset_type="audio",
                    format=fmt,
                    file_path=None,  # Would be set during actual rendering
                )
            )

        # Lyrics asset
        if lyric_spec.get("sections"):
            assets.append(
                ReleaseAsset(
                    asset_type="lyrics",
                    format="txt",
                    file_path=None,
                )
            )
            assets.append(
                ReleaseAsset(
                    asset_type="lyrics",
                    format="lrc",  # Synced lyrics
                    file_path=None,
                )
            )

        # Artwork placeholder
        assets.append(
            ReleaseAsset(
                asset_type="artwork",
                format="jpg",
                file_path=None,
            )
        )

        return assets

    def _generate_tags(self, song_spec: Dict, genre_id: str) -> List[str]:
        """Generate descriptive tags for the release."""
        tags = []

        # Genre tag
        tags.append(genre_id.replace("-", " "))

        # Mood tag
        mood = song_spec.get("primary_mood", "")
        if hasattr(mood, "value"):
            mood = mood.value
        if mood:
            tags.append(mood)

        # Tempo category
        bpm = song_spec.get("bpm", 120)
        if bpm < 80:
            tags.append("slow")
        elif bpm < 120:
            tags.append("mid-tempo")
        elif bpm < 140:
            tags.append("upbeat")
        else:
            tags.append("fast")

        # Vocal/instrumental
        if song_spec.get("has_vocals"):
            tags.append("vocal")
        else:
            tags.append("instrumental")

        # Creative brief keywords
        brief = song_spec.get("creative_brief", {})
        if isinstance(brief, dict):
            theme = brief.get("theme", "")
            if theme:
                tags.append(theme[:20])  # First 20 chars

        return tags[:10]  # Max 10 tags

    def _generate_placeholder_isrc(self) -> str:
        """Generate a placeholder ISRC code."""
        # Format: CC-XXX-YY-NNNNN
        # CC = country, XXX = registrant, YY = year, NNNNN = designation
        import random

        year = str(datetime.utcnow().year)[2:]
        designation = str(random.randint(10000, 99999))
        return f"US-AET-{year}-{designation}"

    def _generate_notes(self, qa_report: Dict) -> List[str]:
        """Generate release notes from QA report."""
        notes = []

        if qa_report.get("all_passed"):
            notes.append("All QA checks passed")
        else:
            notes.append("QA checks incomplete or failed")

            # Add rejection reasons
            for reason in qa_report.get("rejection_reasons", []):
                notes.append(f"BLOCKED: {reason}")

        # Add warnings
        for warning in qa_report.get("warnings", []):
            notes.append(f"WARNING: {warning}")

        # Genre authenticity
        genre_auth = qa_report.get("genre_authenticity", {})
        if genre_auth:
            score = genre_auth.get("total_score", 0)
            notes.append(f"Genre authenticity: {score:.0%}")

        return notes
