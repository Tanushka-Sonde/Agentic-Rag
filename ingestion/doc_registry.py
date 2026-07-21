"""
ingestion/doc_registry.py
──────────────────────────
Change 5 — Shared document-level deduplication registry.

Stores a SHA-256 hash of every ingested file in a Postgres table
(`doc_registry`). This table lives in the same shared Postgres instance
as `chunks`, so it is visible to every machine that connects to the DB.

Usage in ingest.py:
    from ingestion.doc_registry import DocRegistry
    registry = DocRegistry(engine)
    await registry.init()

    status = await registry.check(file_path)
    # status.already_ingested → True  → skip
    # status.same_name_new_hash → True → re-ingest as new version

    await registry.record(file_path, metadata)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


CREATE_REGISTRY_TABLE = """
CREATE TABLE IF NOT EXISTS doc_registry (
    content_hash    TEXT PRIMARY KEY,       -- SHA-256 of raw file bytes
    file_name       TEXT NOT NULL,          -- original filename (no path)
    source_path     TEXT NOT NULL,          -- full path at ingestion time
    engagement_id   TEXT,
    client          TEXT,
    country         TEXT,
    practice        TEXT,
    year            INTEGER,
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
)
"""

CREATE_REGISTRY_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_registry_name ON doc_registry(file_name)"
)


@dataclass
class RegistryStatus:
    already_ingested: bool      # exact same content already in DB → skip
    same_name_new_hash: bool    # same filename but different content → re-ingest
    existing_hash: str | None   # the hash that was previously stored


class DocRegistry:
    """Centralised (Postgres-backed) document deduplication registry."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def init(self) -> None:
        """Create the registry table if it doesn't exist."""
        async with self.engine.begin() as conn:
            await conn.execute(text(CREATE_REGISTRY_TABLE))
            await conn.execute(text(CREATE_REGISTRY_INDEX))

    @staticmethod
    def file_hash(file_path: Path) -> str:
        """Compute SHA-256 of the file's raw bytes."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    async def check(self, file_path: Path) -> RegistryStatus:
        """
        Check whether this file (by content hash) is already ingested.

        Returns RegistryStatus:
          - already_ingested=True  → identical content exists → skip
          - same_name_new_hash=True → same filename, different content → re-ingest
          - both False             → brand-new file → ingest normally
        """
        content_hash = self.file_hash(file_path)
        file_name    = file_path.name

        async with AsyncSession(self.engine) as session:
            # Check by exact content hash
            row_by_hash = (await session.execute(
                text("SELECT file_name FROM doc_registry WHERE content_hash = :h"),
                {"h": content_hash},
            )).first()

            if row_by_hash:
                return RegistryStatus(
                    already_ingested=True,
                    same_name_new_hash=False,
                    existing_hash=content_hash,
                )

            # Check by filename (same name, new content)
            row_by_name = (await session.execute(
                text("SELECT content_hash FROM doc_registry WHERE file_name = :n"),
                {"n": file_name},
            )).first()

            return RegistryStatus(
                already_ingested=False,
                same_name_new_hash=row_by_name is not None,
                existing_hash=row_by_name[0] if row_by_name else None,
            )

    async def record(self, file_path: Path, metadata: dict) -> str:
        """
        Record a successfully ingested document.
        Returns the content hash.
        """
        content_hash = self.file_hash(file_path)
        file_name    = file_path.name

        async with AsyncSession(self.engine) as session:
            async with session.begin():
                await session.execute(text("""
                    INSERT INTO doc_registry
                        (content_hash, file_name, source_path,
                        engagement_id, client, country, practice, year)
                    VALUES
                        (:hash, :name, :path,
                        :eid, :client, :country, :practice, :year)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        file_name    = EXCLUDED.file_name,
                        source_path  = EXCLUDED.source_path,
                        ingested_at  = NOW()
                """), {
                    "hash":    content_hash,
                    "name":    file_name,
                    "path":    str(file_path),
                    "eid":     self._as_text(metadata.get("engagement_id", "")),
                    "client":  self._as_text(metadata.get("client", "")),
                    "country": self._as_text(metadata.get("country", "")),
                    "practice": self._as_text(metadata.get("practice", "")),
                    "year":    self._as_int(metadata.get("year")),
                })
        return content_hash

    async def remove(self, content_hash: str) -> None:
        """Remove a document record (used when re-ingesting a modified file)."""
        async with AsyncSession(self.engine) as session:
            async with session.begin():
                await session.execute(
                    text("DELETE FROM doc_registry WHERE content_hash = :h"),
                    {"h": content_hash},
                )

    @staticmethod
    def _as_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(v) for v in value)
        return str(value)

    @staticmethod
    def _as_int(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None