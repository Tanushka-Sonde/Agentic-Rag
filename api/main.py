"""
api/main.py
────────────
FastAPI application.

Endpoints:
  POST /chat          – Main conversational RAG endpoint
  POST /chat/reset    – Clear session memory
  GET  /health        – Liveness probe
  GET  /status        – Shows how many chunks/docs are indexed (no re-embed needed)
  GET  /documents     – List indexed documents with optional filters
  POST /ingest        – Trigger ingestion for a new file (admin)

Auth:
  Pass your shared API key as: Authorization: Bearer <api-key>
  In development (APP_ENV=development) auth is bypassed.

The data lives in Pinecone + Postgres.
Once embedded, anyone with the API key can query — no re-embedding.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import get_graph
from agent.memory import get_memory
from api.auth import get_current_user
from api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentListResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ResetResponse,
    StatusResponse,
)
from config.settings import get_settings

log      = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting EY ME Agentic RAG API")
    get_graph()
    get_memory()

    # Pre-warm the ColPali model at startup (NOT during the first request) so
    # the heavy ~5GB model load can't time out a /chat call. Runs in a thread
    # so it doesn't block the event loop while the model loads.
    if settings.colpali_enabled:
        import asyncio
        from agent.tools import _get_colpali_embedder

        async def _warm():
            try:
                emb = _get_colpali_embedder()
                if emb is not None:
                    log.info("Pre-warming ColPali model (one-time, may take ~1 min)…")
                    await asyncio.to_thread(emb._get_model)
                    log.info("ColPali model ready.")
            except Exception as exc:
                log.warning("ColPali pre-warm failed; visual retrieval disabled", error=str(exc))

        asyncio.create_task(_warm())

    yield
    log.info("Shutting down")


app = FastAPI(
    title="EY Middle East Knowledge Agent",
    description=(
        "Multimodal Agentic RAG for EY consulting knowledge retrieval.\n\n"
        "Data is indexed once in Pinecone + Postgres. "
        "Pass your API key as `Authorization: Bearer <key>` to query."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(status="healthy", version="2.0.0")


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def status(current_user: dict = Depends(get_current_user)):
    """Show how many chunks are indexed — confirms no re-embedding is needed."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    engine = create_async_engine(settings.database_url)
    async with AsyncSession(engine) as session:
        total_chunks = (await session.execute(text("SELECT COUNT(*) FROM chunks"))).scalar()
        total_docs   = (await session.execute(
            text("SELECT COUNT(DISTINCT source_file) FROM chunks")
        )).scalar()
        ns_rows = (await session.execute(
            text("SELECT kind, COUNT(*) FROM chunks GROUP BY kind")
        )).fetchall()

    return StatusResponse(
        total_chunks=total_chunks or 0,
        total_documents=total_docs or 0,
        namespaces={row[0]: row[1] for row in ns_rows},
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(
    request:      ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Main conversational endpoint.
    Runs the full LangGraph Agentic RAG pipeline.
    Returns the answer, extracted tables, and citations.
    """
    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.perf_counter()

    log.info("chat_request", session_id=session_id, user=current_user.get("user"))

    memory  = get_memory()
    history = await memory.get_history(session_id)
    await memory.add(session_id, "user", request.question)

    graph = get_graph()
    try:
        state = await graph.ainvoke({
            "question":          request.question,
            "chat_history":      history,
            "reflection_loops":  0,
            "raw_chunks":        [],
            "colpali_pages":     [],
            "graded_chunks":     [],
            "reranked_chunks":   [],
            "full_chunks":       [],
            "page_images":       [],
            "citations":         [],
            "tables":            [],
            "rewritten_queries": [],
            "context":           "",
            "answer":            "",
            "final_answer":      "",
            "reflection_result": {},
        })
    except Exception as exc:
        import traceback
        log.error("agent_error", error=str(exc), tb=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    answer    = state.get("final_answer") or state.get("answer", "")
    citations = state.get("citations", [])
    tables    = state.get("tables", [])

    await memory.add(session_id, "assistant", answer)

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    log.info("chat_response", session_id=session_id, latency_ms=latency_ms,
             chunks=len(state.get("reranked_chunks", [])), tables=len(tables))

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        citations=[Citation(**c) for c in citations],
        tables=tables,
        queries_used=state.get("rewritten_queries", []),
        chunks_retrieved=len(state.get("reranked_chunks", [])),
        latency_ms=latency_ms,
    )


@app.post("/chat/reset", response_model=ResetResponse, tags=["Agent"])
async def reset_session(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    """Clear the conversation history for a session."""
    await get_memory().clear(session_id)
    return ResetResponse(session_id=session_id, cleared=True)


@app.get("/documents", response_model=DocumentListResponse, tags=["Knowledge Base"])
async def list_documents(
    country:      str | None = None,
    practice:     str | None = None,
    doc_type:     str | None = None,
    year:         int | None = None,
    current_user: dict = Depends(get_current_user),
):
    """List documents indexed in the knowledge base."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    engine     = create_async_engine(settings.database_url)
    conditions = ["1=1"]
    params: dict = {}

    if country:
        conditions.append("country ILIKE :country")
        params["country"] = f"%{country}%"
    if practice:
        conditions.append("practice ILIKE :practice")
        params["practice"] = f"%{practice}%"
    if doc_type:
        conditions.append("doc_type = :doc_type")
        params["doc_type"] = doc_type
    if year:
        conditions.append("year = :year")
        params["year"] = year

    where = " AND ".join(conditions)
    async with AsyncSession(engine) as session:
        result = await session.execute(text(f"""
            SELECT DISTINCT source_file, doc_type, client, country,
                   practice, year, engagement_id
            FROM chunks
            WHERE {where}
            ORDER BY year DESC, source_file
            LIMIT 100
        """), params)
        rows = result.mappings().all()

    return DocumentListResponse(documents=[dict(r) for r in rows])


@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest_document(
    request:      IngestRequest,
    current_user: dict = Depends(get_current_user),
):
    """Trigger ingestion pipeline for a new document (admin only)."""
    from ingestion.chunker import DocumentChunker
    from ingestion.embedder import Embedder
    from ingestion.indexer import DocumentIndexer
    from ingestion.parser import DocumentParser

    async def run_ingestion():
        parser   = DocumentParser()
        chunker  = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            project_metadata=request.metadata or {},
        )
        embedder = Embedder()
        indexer  = DocumentIndexer()
        await indexer.init_db()

        parsed   = parser.parse(request.file_path)
        chunks   = chunker.chunk_document(parsed)
        embedded = await embedder.embed_chunks(chunks)
        await indexer.index_chunks(embedded)
        return len(chunks)

    try:
        n_chunks = await run_ingestion()
        return IngestResponse(file_path=request.file_path, chunks_indexed=n_chunks, success=True)
    except Exception as exc:
        log.error("ingest_error", file=request.file_path, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
