"""
ingestion/colpali_indexer.py
─────────────────────────────
Stores ColPali page-level embeddings in:
  1. Pinecone  — namespace "colpali_pages" (separate from text/image namespaces)
  2. PostgreSQL — "colpali_pages" table for page image retrieval at query time

At query time, the retriever fetches matching pages from this table,
and the generator receives the actual page image (base64 PNG) alongside
the chunk-level context — giving GPT-4o both text and visual context.

ColPali vector dimension: 128 (after average pooling from multi-vector)
Note: This requires a SEPARATE Pinecone index from your text chunks
      because ColPali vectors have a different dimension (128 vs 1536).
"""

from __future__ import annotations

import json
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from config.settings import get_settings
from ingestion.colpali_embedder import ColPaliPage

settings = get_settings()

# ── Postgres schema ────────────────────────────────────────────────────────────

CREATE_COLPALI_TABLE = """
CREATE TABLE IF NOT EXISTS colpali_pages (
    page_id         TEXT PRIMARY KEY,   -- source_file|page_idx
    source_file     TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    page_idx        INTEGER NOT NULL,
    page_image_b64  TEXT NOT NULL,      -- full page PNG for GPT-4o at query time
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
    "CREATE INDEX IF NOT EXISTS idx_colpali_source ON colpali_pages(source_file)",
    "CREATE INDEX IF NOT EXISTS idx_colpali_engagement ON colpali_pages(engagement_id)",
    "CREATE INDEX IF NOT EXISTS idx_colpali_country ON colpali_pages(country)",
]

# ColPali average-pooled dimension (128 for colpali-v1.2)
COLPALI_DIM = 128


class ColPaliIndexer:
    """
    Stores ColPali page embeddings to Pinecone (colpali namespace) + Postgres.
    Uses a SEPARATE Pinecone index ("colpali-index") from the text chunk index.
    """

    UPSERT_BATCH = 50

    @staticmethod
    def _as_text(value) -> str:
        """Coerce metadata to a TEXT-safe string (LLM may return lists)."""
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

    def __init__(self, engine: AsyncEngine | None = None) -> None:
        pc = Pinecone(api_key=settings.pinecone_api_key)

        # ColPali gets its own index (different dimension from text chunks)
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

        self.index = pc.Index(colpali_index_name)
        self.engine = engine or create_async_engine(settings.database_url, echo=False)

    async def init_db(self) -> None:
        """Create colpali_pages table if it doesn't exist."""
        async with self.engine.begin() as conn:
            await conn.execute(text(CREATE_COLPALI_TABLE))
            for idx_sql in CREATE_COLPALI_INDEXES:
                await conn.execute(text(idx_sql))
        print("ColPali Postgres table ready.")

    async def index_pages(
        self,
        page_vectors: list[tuple[ColPaliPage, list[float]]],
    ) -> None:
        """Write ColPali page vectors to Pinecone + page images to Postgres."""
        await self._upsert_pinecone(page_vectors)
        await self._upsert_postgres(page_vectors)

    # ── Pinecone ──────────────────────────────────────────────────────────────

    def _upsert_pinecone(
        self,
        page_vectors: list[tuple[ColPaliPage, list[float]]],
    ) -> None:
        records = []
        for page, vector in page_vectors:
            page_id = self._make_page_id(page.source_file, page.page_idx)
            meta = {
                "page_id":      page_id,
                "source_file":  page.source_file,
                "doc_type":     page.doc_type,
                "page_idx":     page.page_idx,
                "engagement_id": self._as_text(page.metadata.get("engagement_id", "")),
                "client":       self._as_text(page.metadata.get("client", "")),
                "country":      self._as_text(page.metadata.get("country", "")),
                "practice":     self._as_text(page.metadata.get("practice", "")),
                "year":         self._as_int(page.metadata.get("year")) or 0,
            }
            records.append({
                "id":       page_id,
                "values":   vector,
                "metadata": meta,
            })

        for i in range(0, len(records), self.UPSERT_BATCH):
            self.index.upsert(
                vectors=records[i : i + self.UPSERT_BATCH],
                namespace="colpali_pages",
            )

    # ── Postgres ──────────────────────────────────────────────────────────────

    async def _upsert_postgres(
        self,
        page_vectors: list[tuple[ColPaliPage, list[float]]],
    ) -> None:
        async with AsyncSession(self.engine) as session:
            async with session.begin():
                for page, _ in page_vectors:
                    page_id = self._make_page_id(page.source_file, page.page_idx)
                    meta    = page.metadata
                    known   = {"engagement_id", "client", "country", "practice", "year"}
                    extra   = {k: v for k, v in meta.items() if k not in known}

                    await session.execute(text("""
                        INSERT INTO colpali_pages (
                            page_id, source_file, doc_type, page_idx, page_image_b64,
                            engagement_id, client, country, practice, year, extra_metadata
                        ) VALUES (
                            :page_id, :source_file, :doc_type, :page_idx, :page_image_b64,
                            :engagement_id, :client, :country, :practice, :year, :extra_metadata
                        )
                        ON CONFLICT (page_id) DO UPDATE SET
                            page_image_b64 = EXCLUDED.page_image_b64,
                            extra_metadata = EXCLUDED.extra_metadata
                    """), {
                        "page_id":       page_id,
                        "source_file":   page.source_file,
                        "doc_type":      page.doc_type,
                        "page_idx":      page.page_idx,
                        "page_image_b64": page.page_image_b64,
                        "engagement_id": self._as_text(meta.get("engagement_id", "")),
                        "client":        self._as_text(meta.get("client", "")),
                        "country":       self._as_text(meta.get("country", "")),
                        "practice":      self._as_text(meta.get("practice", "")),
                        "year":          self._as_int(meta.get("year")),
                        "extra_metadata": json.dumps(extra),
                    })

    # ── Query helpers ──────────────────────────────────────────────────────────

    def query_pages(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Query ColPali Pinecone index. Returns list of page metadata dicts.
        Caller should then fetch page images from Postgres using page_ids.
        """
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
                "engagement_id": m.metadata.get("engagement_id", ""),
                "country":      m.metadata.get("country", ""),
                "practice":     m.metadata.get("practice", ""),
            }
            for m in response.matches
        ]

    @staticmethod
    def _make_page_id(source_file: str, page_idx: int) -> str:
        import hashlib
        raw = f"{source_file}|page|{page_idx}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
