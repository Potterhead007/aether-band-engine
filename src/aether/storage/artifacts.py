"""
AETHER Artifact Store

Production-grade artifact storage for tracking all pipeline outputs.
Supports versioning, checksums, metadata, and efficient retrieval.

Features:
- Async-safe SQLite operations via thread pool
- Content-addressable blob storage
- Artifact versioning and lineage tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import uuid4

logger = logging.getLogger(__name__)

# Type variable for async wrapper
T = TypeVar("T")

# Thread pool for async SQLite operations (SQLite is not async-safe)
# Using max_workers=4 to allow concurrent reads while writes are serialized by SQLite
_DB_EXECUTOR: ThreadPoolExecutor | None = None
_DB_EXECUTOR_MAX_WORKERS = 4


def _get_db_executor() -> ThreadPoolExecutor:
    """Get or create the database thread pool executor."""
    global _DB_EXECUTOR
    if _DB_EXECUTOR is None:
        _DB_EXECUTOR = ThreadPoolExecutor(
            max_workers=_DB_EXECUTOR_MAX_WORKERS, thread_name_prefix="aether_db_"
        )
    return _DB_EXECUTOR


def shutdown_db_executor() -> None:
    """Shutdown the database thread pool (call on application exit)."""
    global _DB_EXECUTOR
    if _DB_EXECUTOR is not None:
        _DB_EXECUTOR.shutdown(wait=True)
        _DB_EXECUTOR = None


async def _run_in_executor(func: Callable[..., T], *args, **kwargs) -> T:
    """Run a blocking function in the thread pool executor."""
    loop = asyncio.get_event_loop()
    executor = _get_db_executor()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(executor, func, *args)


class ArtifactType(str, Enum):
    """Types of artifacts in the pipeline."""

    # Specs
    SONG_SPEC = "song_spec"
    HARMONY_SPEC = "harmony_spec"
    MELODY_SPEC = "melody_spec"
    ARRANGEMENT_SPEC = "arrangement_spec"
    RHYTHM_SPEC = "rhythm_spec"
    LYRIC_SPEC = "lyric_spec"
    VOCAL_SPEC = "vocal_spec"
    SOUND_DESIGN_SPEC = "sound_design_spec"
    MIX_SPEC = "mix_spec"
    MASTER_SPEC = "master_spec"
    QA_REPORT = "qa_report"

    # Audio
    MIDI = "midi"
    STEM = "stem"
    MIX = "mix"
    MASTER = "master"

    # Data
    GENRE_PROFILE = "genre_profile"
    RECIPE = "recipe"
    METADATA = "metadata"
    RELEASE_PACKAGE = "release_package"


@dataclass
class ArtifactMetadata:
    """Metadata for an artifact."""

    artifact_id: str
    artifact_type: ArtifactType
    song_id: Optional[str]
    name: str
    version: int
    checksum: str
    size_bytes: int
    mime_type: str
    created_at: datetime
    created_by: str
    tags: dict[str, str]
    parent_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "song_id": self.song_id,
            "name": self.name,
            "version": self.version,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
            "parent_ids": self.parent_ids,
        }


class ArtifactStore:
    """
    Production artifact storage system.

    Features:
    - Content-addressable storage
    - Version tracking
    - Integrity verification (checksums)
    - Metadata indexing via SQLite
    - Efficient blob storage
    - Garbage collection
    """

    SCHEMA_VERSION = 1

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.blobs_path = self.base_path / "blobs"
        self.db_path = self.base_path / "artifacts.db"

        self._ensure_directories()
        self._init_database()

    def _ensure_directories(self) -> None:
        """Create required directories."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.blobs_path.mkdir(exist_ok=True)

    def _init_database(self) -> None:
        """Initialize SQLite database for metadata."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );

                INSERT OR IGNORE INTO schema_version (version) VALUES (1);

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    song_id TEXT,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mime_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    blob_path TEXT NOT NULL,
                    is_deleted INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS artifact_tags (
                    artifact_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (artifact_id, key),
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
                );

                CREATE TABLE IF NOT EXISTS artifact_parents (
                    artifact_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    PRIMARY KEY (artifact_id, parent_id),
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_song
                    ON artifacts(song_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_type
                    ON artifacts(artifact_type);
                CREATE INDEX IF NOT EXISTS idx_artifacts_checksum
                    ON artifacts(checksum);
                CREATE INDEX IF NOT EXISTS idx_artifacts_created
                    ON artifacts(created_at);
            """
            )

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()

    def _get_blob_path(self, checksum: str) -> Path:
        """Get blob storage path from checksum (content-addressable)."""
        # Use first 2 chars as directory for sharding
        return self.blobs_path / checksum[:2] / checksum

    def _store_blob(self, data: bytes, checksum: str) -> Path:
        """Store blob data, deduplicating by checksum."""
        blob_path = self._get_blob_path(checksum)

        if blob_path.exists():
            # Already have this content
            return blob_path

        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(data)
        return blob_path

    def store(
        self,
        data: bytes | str | dict[str, Any],
        artifact_type: ArtifactType,
        name: str,
        song_id: Optional[str] = None,
        created_by: str = "aether",
        tags: dict[str, str] | None = None,
        parent_ids: list[str] | None = None,
        mime_type: Optional[str] = None,
    ) -> ArtifactMetadata:
        """
        Store an artifact.

        Args:
            data: Raw bytes, string, or dict (will be JSON-serialized)
            artifact_type: Type of artifact
            name: Human-readable name
            song_id: Associated song ID if applicable
            created_by: Agent/process that created this
            tags: Key-value tags for filtering
            parent_ids: IDs of parent artifacts (lineage)
            mime_type: MIME type (auto-detected if not provided)

        Returns:
            ArtifactMetadata for the stored artifact
        """
        tags = tags or {}
        parent_ids = parent_ids or []

        # Ensure song_id is a string (handle UUID objects)
        if song_id is not None:
            song_id = str(song_id)

        # Serialize data
        if isinstance(data, dict):
            data_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
            mime_type = mime_type or "application/json"
        elif isinstance(data, str):
            data_bytes = data.encode("utf-8")
            mime_type = mime_type or "text/plain"
        else:
            data_bytes = data
            mime_type = mime_type or "application/octet-stream"

        # Compute checksum and store blob
        checksum = self._compute_checksum(data_bytes)
        blob_path = self._store_blob(data_bytes, checksum)

        # Determine version
        version = self._get_next_version(song_id, artifact_type, name)

        # Generate ID
        artifact_id = str(uuid4())
        created_at = datetime.utcnow()

        # Store metadata
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, artifact_type, song_id, name, version,
                    checksum, size_bytes, mime_type, created_at, created_by,
                    blob_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    artifact_id,
                    artifact_type.value,
                    song_id,
                    name,
                    version,
                    checksum,
                    len(data_bytes),
                    mime_type,
                    created_at.isoformat(),
                    created_by,
                    str(blob_path.relative_to(self.base_path)),
                ),
            )

            # Store tags
            for key, value in tags.items():
                conn.execute(
                    """
                    INSERT INTO artifact_tags (artifact_id, key, value)
                    VALUES (?, ?, ?)
                """,
                    (artifact_id, key, value),
                )

            # Store parent relationships
            for parent_id in parent_ids:
                conn.execute(
                    """
                    INSERT INTO artifact_parents (artifact_id, parent_id)
                    VALUES (?, ?)
                """,
                    (artifact_id, parent_id),
                )

        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            song_id=song_id,
            name=name,
            version=version,
            checksum=checksum,
            size_bytes=len(data_bytes),
            mime_type=mime_type,
            created_at=created_at,
            created_by=created_by,
            tags=tags,
            parent_ids=parent_ids,
        )

        logger.info(f"Stored artifact: {artifact_id} ({artifact_type.value}/{name} v{version})")
        return metadata

    def _get_next_version(
        self,
        song_id: Optional[str],
        artifact_type: ArtifactType,
        name: str,
    ) -> int:
        """Get next version number for an artifact."""
        with self._get_connection() as conn:
            if song_id:
                row = conn.execute(
                    """
                    SELECT MAX(version) as max_version FROM artifacts
                    WHERE song_id = ? AND artifact_type = ? AND name = ?
                    AND is_deleted = 0
                """,
                    (song_id, artifact_type.value, name),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT MAX(version) as max_version FROM artifacts
                    WHERE song_id IS NULL AND artifact_type = ? AND name = ?
                    AND is_deleted = 0
                """,
                    (artifact_type.value, name),
                ).fetchone()

            max_version = row["max_version"] if row["max_version"] else 0
            return max_version + 1

    def get(self, artifact_id: str) -> bytes | None:
        """Get artifact data by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT blob_path FROM artifacts
                WHERE artifact_id = ? AND is_deleted = 0
            """,
                (artifact_id,),
            ).fetchone()

            if not row:
                return None

            blob_path = self.base_path / row["blob_path"]
            if not blob_path.exists():
                logger.error(f"Blob missing for artifact {artifact_id}")
                return None

            return blob_path.read_bytes()

    def get_json(self, artifact_id: str) -> dict[str, Any] | None:
        """Get artifact data as JSON dict."""
        data = self.get(artifact_id)
        if data is None:
            return None
        return json.loads(data.decode("utf-8"))

    def get_metadata(self, artifact_id: str) -> ArtifactMetadata | None:
        """Get artifact metadata by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM artifacts WHERE artifact_id = ? AND is_deleted = 0
            """,
                (artifact_id,),
            ).fetchone()

            if not row:
                return None

            # Get tags
            tags = {}
            for tag_row in conn.execute(
                """
                SELECT key, value FROM artifact_tags WHERE artifact_id = ?
            """,
                (artifact_id,),
            ):
                tags[tag_row["key"]] = tag_row["value"]

            # Get parents
            parents = [
                r["parent_id"]
                for r in conn.execute(
                    """
                    SELECT parent_id FROM artifact_parents WHERE artifact_id = ?
                """,
                    (artifact_id,),
                )
            ]

            return ArtifactMetadata(
                artifact_id=row["artifact_id"],
                artifact_type=ArtifactType(row["artifact_type"]),
                song_id=row["song_id"],
                name=row["name"],
                version=row["version"],
                checksum=row["checksum"],
                size_bytes=row["size_bytes"],
                mime_type=row["mime_type"],
                created_at=datetime.fromisoformat(row["created_at"]),
                created_by=row["created_by"],
                tags=tags,
                parent_ids=parents,
            )

    def list_by_song(
        self,
        song_id: str,
        artifact_type: ArtifactType | None = None,
    ) -> list[ArtifactMetadata]:
        """List all artifacts for a song."""
        with self._get_connection() as conn:
            if artifact_type:
                rows = conn.execute(
                    """
                    SELECT artifact_id FROM artifacts
                    WHERE song_id = ? AND artifact_type = ? AND is_deleted = 0
                    ORDER BY created_at DESC
                """,
                    (song_id, artifact_type.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT artifact_id FROM artifacts
                    WHERE song_id = ? AND is_deleted = 0
                    ORDER BY created_at DESC
                """,
                    (song_id,),
                ).fetchall()

        return [self.get_metadata(row["artifact_id"]) for row in rows]

    def get_latest(
        self,
        song_id: str,
        artifact_type: ArtifactType,
        name: str,
    ) -> ArtifactMetadata | None:
        """Get the latest version of an artifact."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT artifact_id FROM artifacts
                WHERE song_id = ? AND artifact_type = ? AND name = ?
                AND is_deleted = 0
                ORDER BY version DESC LIMIT 1
            """,
                (song_id, artifact_type.value, name),
            ).fetchone()

            if not row:
                return None

            return self.get_metadata(row["artifact_id"])

    def verify_integrity(self, artifact_id: str) -> bool:
        """Verify artifact integrity via checksum."""
        metadata = self.get_metadata(artifact_id)
        if not metadata:
            return False

        data = self.get(artifact_id)
        if data is None:
            return False

        actual_checksum = self._compute_checksum(data)
        return actual_checksum == metadata.checksum

    def delete(self, artifact_id: str, soft: bool = True) -> bool:
        """Delete an artifact (soft delete by default)."""
        with self._get_connection() as conn:
            if soft:
                result = conn.execute(
                    """
                    UPDATE artifacts SET is_deleted = 1
                    WHERE artifact_id = ? AND is_deleted = 0
                """,
                    (artifact_id,),
                )
            else:
                # Hard delete - remove blob and metadata
                row = conn.execute(
                    """
                    SELECT blob_path, checksum FROM artifacts WHERE artifact_id = ?
                """,
                    (artifact_id,),
                ).fetchone()

                if row:
                    # Check if blob is used by other artifacts
                    other = conn.execute(
                        """
                        SELECT COUNT(*) as count FROM artifacts
                        WHERE checksum = ? AND artifact_id != ? AND is_deleted = 0
                    """,
                        (row["checksum"], artifact_id),
                    ).fetchone()

                    if other["count"] == 0:
                        # Safe to delete blob
                        blob_path = self.base_path / row["blob_path"]
                        if blob_path.exists():
                            blob_path.unlink()

                    conn.execute("DELETE FROM artifact_tags WHERE artifact_id = ?", (artifact_id,))
                    conn.execute(
                        "DELETE FROM artifact_parents WHERE artifact_id = ?", (artifact_id,)
                    )
                    result = conn.execute(
                        "DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,)
                    )

            return result.rowcount > 0

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        with self._get_connection() as conn:
            total = conn.execute(
                """
                SELECT COUNT(*) as count, SUM(size_bytes) as total_size
                FROM artifacts WHERE is_deleted = 0
            """
            ).fetchone()

            by_type = {}
            for row in conn.execute(
                """
                SELECT artifact_type, COUNT(*) as count, SUM(size_bytes) as total_size
                FROM artifacts WHERE is_deleted = 0
                GROUP BY artifact_type
            """
            ):
                by_type[row["artifact_type"]] = {
                    "count": row["count"],
                    "size_bytes": row["total_size"] or 0,
                }

        return {
            "total_artifacts": total["count"],
            "total_size_bytes": total["total_size"] or 0,
            "by_type": by_type,
            "db_path": str(self.db_path),
            "blobs_path": str(self.blobs_path),
        }

    # =========================================================================
    # Async-safe methods (run blocking SQLite in thread pool)
    # =========================================================================

    async def store_async(
        self,
        data: bytes | str | dict[str, Any],
        artifact_type: ArtifactType,
        name: str,
        song_id: Optional[str] = None,
        created_by: str = "aether",
        tags: dict[str, str] | None = None,
        parent_ids: list[str] | None = None,
        mime_type: Optional[str] = None,
    ) -> ArtifactMetadata:
        """
        Async-safe version of store().

        Runs the blocking SQLite operation in a thread pool to avoid
        blocking the event loop.
        """
        return await _run_in_executor(
            self.store,
            data,
            artifact_type,
            name,
            song_id,
            created_by,
            tags,
            parent_ids,
            mime_type,
        )

    async def get_async(self, artifact_id: str) -> bytes | None:
        """Async-safe version of get()."""
        return await _run_in_executor(self.get, artifact_id)

    async def get_json_async(self, artifact_id: str) -> dict[str, Any] | None:
        """Async-safe version of get_json()."""
        return await _run_in_executor(self.get_json, artifact_id)

    async def get_metadata_async(self, artifact_id: str) -> ArtifactMetadata | None:
        """Async-safe version of get_metadata()."""
        return await _run_in_executor(self.get_metadata, artifact_id)

    async def list_by_song_async(
        self,
        song_id: str,
        artifact_type: ArtifactType | None = None,
    ) -> list[ArtifactMetadata]:
        """Async-safe version of list_by_song()."""
        return await _run_in_executor(self.list_by_song, song_id, artifact_type)

    async def get_latest_async(
        self,
        song_id: str,
        artifact_type: ArtifactType,
        name: str,
    ) -> ArtifactMetadata | None:
        """Async-safe version of get_latest()."""
        return await _run_in_executor(self.get_latest, song_id, artifact_type, name)

    async def verify_integrity_async(self, artifact_id: str) -> bool:
        """Async-safe version of verify_integrity()."""
        return await _run_in_executor(self.verify_integrity, artifact_id)

    async def delete_async(self, artifact_id: str, soft: bool = True) -> bool:
        """Async-safe version of delete()."""
        return await _run_in_executor(self.delete, artifact_id, soft)

    async def get_stats_async(self) -> dict[str, Any]:
        """Async-safe version of get_stats()."""
        return await _run_in_executor(self.get_stats)


def create_artifact_store(base_path: Optional[Path] = None) -> ArtifactStore:
    """Create an artifact store with default path if not specified."""
    if base_path is None:
        base_path = Path.home() / ".aether" / "artifacts"
    return ArtifactStore(base_path)
