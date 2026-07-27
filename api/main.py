"""
api/main.py   [CHANGE 2 + CHANGE 4 — modified]
────────────
Change 2: /chat now routes through route_and_run() (multi-agent middleware)
          instead of calling rag_graph.ainvoke() directly.
          Response includes agent_name, code, image_b64, chart_type fields.

Change 4: New POST /chat/stop endpoint that cancels an in-progress request.
          Each in-flight /chat call registers a CancellationToken keyed by
          session_id. Calling /chat/stop signals that token, and the
          route_and_run() coroutine checks it between agent steps.

All other endpoints and behaviours are unchanged.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import get_graph
from agent.memory import get_memory
# Change 2
from agent.multi_agent import route_and_run
# Change 4
from agent.cancellation import CancellationRegistry
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
    StopResponse,
)
from config.settings import get_settings

log      = structlog.get_logger()
settings = get_settings()

# Change 4: shared cancellation registry
_cancel_registry = CancellationRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting EY ME Agentic RAG API")
    get_graph()
    get_memory()
    if settings.colpali_enabled:
        from agent.tools import _get_colpali_embedder
        async def _warm():
            try:
                emb = _get_colpali_embedder()
                if emb:
                    log.info("Pre-warming ColPali…")
                    await asyncio.to_thread(emb._get_model)
                    log.info("ColPali ready.")
            except Exception as exc:
                log.warning("ColPali pre-warm failed", error=str(exc))
        asyncio.create_task(_warm())
    yield
    log.info("Shutting down")


app = FastAPI(
    title="EY Middle East Knowledge Agent",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Health / Status (unchanged) ───────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(status="healthy", version="2.1.0")


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def status(current_user: dict = Depends(get_current_user)):
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


# ── Chat — Change 2 (multi-agent) + Change 4 (cancellation) ──────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(
    request:      ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.perf_counter()
    log.info("chat_request", session_id=session_id)

    memory  = get_memory()
    history = await memory.get_history(session_id)
    await memory.add(session_id, "user", request.question)

    # Change 4: register a cancellation token for this session
    token = _cancel_registry.register(session_id)

    graph = get_graph()
    try:
        # Change 2 + 6: delegate to multi-agent router, now WITH the
        # cancellation token forwarded. Previously `token` was registered
        # above but never actually passed in here — /chat/stop only ever
        # aborted the client's own connection while retrieval/rerank/
        # generation/chart-rendering kept running to completion
        # server-side regardless of the stop click. route_and_run forwards
        # this into every agent's run() and into the LangGraph state
        # (checked by every node — see agent/nodes.py's _check_cancelled),
        # and into the chart subprocess (see agent/multi_agent.py's
        # _execute_chart_code), so Stop now actually stops the work.
        result = await route_and_run(
            question=request.question,
            chat_history=history,
            rag_graph=graph,
            cancel_token=token,
        )
    except asyncio.CancelledError:
        # Change 4: request was cancelled via /chat/stop
        log.info("chat_cancelled", session_id=session_id)
        raise HTTPException(status_code=499, detail="Request cancelled by client.")
    except Exception as exc:
        import traceback
        log.error("agent_error", error=str(exc), tb=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")
    finally:
        # Change 4: always clean up the token
        _cancel_registry.unregister(session_id)

    await memory.add(session_id, "assistant", result.answer)

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    log.info("chat_response", session_id=session_id, latency_ms=latency_ms,
             agent=result.agent_name)

    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
        tables=result.tables,
        queries_used=result.raw_state.get("rewritten_queries", []),
        chunks_retrieved=len(result.raw_state.get("reranked_chunks", [])),
        latency_ms=latency_ms,
        # Change 2: new fields
        agent_name=result.agent_name,
        code=result.code,
        image_b64=result.image_b64,
        chart_type=result.chart_type,
    )


# ── Stop endpoint — Change 4 ──────────────────────────────────────────────────

@app.post("/chat/stop", response_model=StopResponse, tags=["Agent"])
async def stop_generation(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    """
    Cancel an in-progress /chat call for the given session_id.
    The partial response already generated remains in session memory.
    """
    cancelled = _cancel_registry.cancel(session_id)
    log.info("stop_requested", session_id=session_id, found=cancelled)
    return StopResponse(session_id=session_id, cancelled=cancelled)


# ── Reset (unchanged) ──────────────────────────────────────────────────────────

@app.post("/chat/reset", response_model=ResetResponse, tags=["Agent"])
async def reset_session(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    await get_memory().clear(session_id)
    return ResetResponse(session_id=session_id, cleared=True)


# ── Documents (unchanged) ─────────────────────────────────────────────────────

@app.get("/documents", response_model=DocumentListResponse, tags=["Knowledge Base"])
async def list_documents(
    country:      str | None = None,
    practice:     str | None = None,
    doc_type:     str | None = None,
    year:         int | None = None,
    current_user: dict = Depends(get_current_user),
):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    engine = create_async_engine(settings.database_url)
    conditions = ["1=1"]
    params: dict = {}
    if country:
        conditions.append("country ILIKE :country"); params["country"] = f"%{country}%"
    if practice:
        conditions.append("practice ILIKE :practice"); params["practice"] = f"%{practice}%"
    if doc_type:
        conditions.append("doc_type = :doc_type"); params["doc_type"] = doc_type
    if year:
        conditions.append("year = :year"); params["year"] = year
    where = " AND ".join(conditions)
    async with AsyncSession(engine) as session:
        result = await session.execute(text(f"""
            SELECT DISTINCT source_file, doc_type, client, country, practice, year, engagement_id
            FROM chunks WHERE {where} ORDER BY year DESC, source_file LIMIT 100
        """), params)
        rows = result.mappings().all()
    return DocumentListResponse(documents=[dict(r) for r in rows])


# ── Ingest (unchanged) ────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest_document(
    request:      IngestRequest,
    current_user: dict = Depends(get_current_user),
):
    from ingestion.chunker import DocumentChunker
    from ingestion.embedder import Embedder
    from ingestion.indexer import DocumentIndexer
    from ingestion.parser import DocumentParser

    async def run_ingestion():
        parser  = DocumentParser()
        chunker = DocumentChunker(
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
        n = await run_ingestion()
        return IngestResponse(file_path=request.file_path, chunks_indexed=n, success=True)
    except Exception as exc:
        log.error("ingest_error", file=request.file_path, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))