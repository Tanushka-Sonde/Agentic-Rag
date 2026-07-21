"""
ingestion/colpali_indexer.py   [CHANGE 1 — modified]
─────────────────────────────
Now indexes VisualElement crops (image / table / chart regions) instead of
full-page screenshots. Everything else — Pinecone namespace, Postgres table
schema, query helpers — is unchanged.

The only modified interface:
  OLD: index_pages(page_vectors: list[tuple[ColPaliPage, list[float]]])
  NEW: index_elements(element_vectors: list[tuple[VisualElement, list[float]]])

Both signatures are kept for backward compatibility; index_pages delegates
to index_elements by converting ColPaliPage → VisualElement internally.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from config.settings import get_settings
from ingestion.element_classifier import VisualElement

settings = get_settings()

CREATE_COLPALI_TABLE = """
CREATE TABLE IF NOT EXISTS colpali_pages (
    page_id         TEXT PRIMARY KEY,
    source_file     TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    page_idx        INTEGER NOT NULL,
    element_idx     INTEGER NOT NULL DEFAULT 0,
    element_kind    TEXT NOT NULL DEFAULT 'image',
    page_image_b64  TEXT NOT NULL,
    engagement_id   TEXT,
    client          TEXT,
    country         TEXT,
    practice        TEXT,
    year            INTEGER,
    extra_metadata  JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
)
"""

CREATE_COLPALI_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_colpali_source     ON colpali_pages(source_file)",
    "CREATE INDEX IF NOT EXISTS idx_colpali_engagement ON colpali_pages(engagement_id)",
    "CREATE INDEX IF NOT EXISTS idx_colpali_country    ON colpali_pages(country)",
    "CREATE INDEX IF NOT EXISTS idx_colpali_kind       ON colpali_pages(element_kind)",
]

COLPALI_DIM = 128


class ColPaliIndexer:
    """
    Stores ColPali visual-element embeddings to Pinecone + Postgres.
    Uses a separate Pinecone index (different dimension: 128 vs 1536).
    """

    UPSERT_BATCH = 50

    def __init__(self, engine: AsyncEngine | None = None) -> None:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        colpali_index_name = f"{settings.pinecone_index_name}-colpali"
        existing = [i.name for i in pc.list_indexes()]
        if colpali_index_name not in existing:
            pc.create_index(
                name=colpali_index_name,
                dimension=COLPALI_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Created Pinecone ColPali index: {colpali_index_name}")
        self.index  = pc.Index(colpali_index_name)
        self.engine = engine or create_async_engine(settings.database_url, echo=False)

    async def init_db(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text(CREATE_COLPALI_TABLE))
            for idx_sql in CREATE_COLPALI_INDEXES:
                await conn.execute(text(idx_sql))
        print("ColPali Postgres table ready.")

    # ── Primary interface (Change 1) ──────────────────────────────────────────

    async def index_elements(
        self,
        element_vectors: list[tuple[VisualElement, list[float]]],
    ) -> None:
        """Index VisualElement crops (image / table / chart) to Pinecone + Postgres."""
        self._upsert_pinecone_elements(element_vectors)
        await self._upsert_postgres_elements(element_vectors)

    # ── Backward-compat shim (keeps old callers working) ─────────────────────

    async def index_pages(self, page_vectors) -> None:
        """
        Legacy interface — converts ColPaliPage list to VisualElement and delegates.
        Kept so any code that still calls index_pages() doesn't break.
        """
        from ingestion.element_classifier import VisualElement as VE
        converted = []
        for page, vec in page_vectors:
            ve = VE(
                kind="image",
                image_b64=page.page_image_b64,
                source_file=page.source_file,
                page_idx=page.page_idx,
                element_idx=0,
                bbox=None,
                metadata=page.metadata,
            )
            converted.append((ve, vec))
        await self.index_elements(converted)

    # ── Pinecone ──────────────────────────────────────────────────────────────

    def _upsert_pinecone_elements(
        self,
        element_vectors: list[tuple[VisualElement, list[float]]],
    ) -> None:
        records = []
        for elem, vector in element_vectors:
            eid = self._make_element_id(elem.source_file, elem.page_idx, elem.element_idx)
            meta = {
                "page_id":       eid,
                "source_file":   elem.source_file,
                "doc_type":      Path(elem.source_file).suffix.lstrip("."),
                "page_idx":      elem.page_idx,
                "element_idx":   elem.element_idx,
                "element_kind":  elem.kind,
                "engagement_id": self._as_text(elem.metadata.get("engagement_id", "")),
                "client":        self._as_text(elem.metadata.get("client", "")),
                "country":       self._as_text(elem.metadata.get("country", "")),
                "practice":      self._as_text(elem.metadata.get("practice", "")),
                "year":          self._as_int(elem.metadata.get("year")) or 0,
            }
            records.append({"id": eid, "values": vector, "metadata": meta})

        for i in range(0, len(records), self.UPSERT_BATCH):
            self.index.upsert(
                vectors=records[i : i + self.UPSERT_BATCH],
                namespace="colpali_pages",
            )

    # ── Postgres ──────────────────────────────────────────────────────────────

    async def _upsert_postgres_elements(
        self,
        element_vectors: list[tuple[VisualElement, list[float]]],
    ) -> None:
        async with AsyncSession(self.engine) as session:
            async with session.begin():
                for elem, _ in element_vectors:
                    eid   = self._make_element_id(elem.source_file, elem.page_idx, elem.element_idx)
                    meta  = elem.metadata
                    known = {"engagement_id", "client", "country", "practice", "year"}
                    extra = {k: v for k, v in meta.items() if k not in known}

                    await session.execute(text("""
                        INSERT INTO colpali_pages (
                            page_id, source_file, doc_type, page_idx, element_idx,
                            element_kind, page_image_b64,
                            engagement_id, client, country, practice, year, extra_metadata
                        ) VALUES (
                            :page_id, :source_file, :doc_type, :page_idx, :element_idx,
                            :element_kind, :page_image_b64,
                            :engagement_id, :client, :country, :practice, :year, :extra_metadata
                        )
                        ON CONFLICT (page_id) DO UPDATE SET
                            page_image_b64 = EXCLUDED.page_image_b64,
                            element_kind   = EXCLUDED.element_kind,
                            extra_metadata = EXCLUDED.extra_metadata
                    """), {
                        "page_id":       eid,
                        "source_file":   elem.source_file,
                        "doc_type":      Path(elem.source_file).suffix.lstrip("."),
                        "page_idx":      elem.page_idx,
                        "element_idx":   elem.element_idx,
                        "element_kind":  elem.kind,
                        "page_image_b64": elem.image_b64,
                        "engagement_id": self._as_text(meta.get("engagement_id", "")),
                        "client":        self._as_text(meta.get("client", "")),
                        "country":       self._as_text(meta.get("country", "")),
                        "practice":      self._as_text(meta.get("practice", "")),
                        "year":          self._as_int(meta.get("year")),
                        "extra_metadata": json.dumps(extra),
                    })

    # ── Query helpers (unchanged from original) ───────────────────────────────

    def query_pages(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace="colpali_pages",
            filter=filters or {},
            include_metadata=True,
        )
        return [
            {
                "page_id":      m.id,
                "score":        m.score,
                "source_file":  m.metadata.get("source_file", ""),
                "doc_type":     m.metadata.get("doc_type", ""),
                "page_idx":     m.metadata.get("page_idx", 0),
                "element_kind": m.metadata.get("element_kind", "image"),
                "engagement_id": m.metadata.get("engagement_id", ""),
                "country":      m.metadata.get("country", ""),
                "practice":     m.metadata.get("practice", ""),
            }
            for m in response.matches
        ]

    @staticmethod
    def _make_element_id(source_file: str, page_idx: int, element_idx: int) -> str:
        raw = f"{source_file}|element|{page_idx}|{element_idx}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

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