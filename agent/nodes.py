"""
agent/nodes.py
───────────────
LangGraph node functions for the hybrid ColPali + chunk retrieval pipeline.

Nodes:
  1. query_rewriter_node   — expand question into multiple search queries
  2. retriever_node        — parallel hybrid (Pinecone) + ColPali retrieval
  3. retrieval_grader_node — pass-through (can be extended for self-RAG grading)
  4. reranker_node         — score-based top-N selection
  5. context_builder_node  — fetch full chunk text + ColPali page images
  6. generator_node        — multimodal GPT-4o generation; extracts inline tables
  7. reflection_grader_node — CRAG quality check
  8. should_retry          — conditional edge
  9. retry_prep_node       — set up retry with refined query
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, TypedDict

import openai
from langchain_openai import ChatOpenAI

from agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    REFLECTION_PROMPT,
    RERANK_PROMPT,
    RETRIEVAL_GRADE_PROMPT,
)
from agent.tools import (
    colpali_retrieval,
    fetch_colpali_page,
    fetch_full_chunk,
    hybrid_retrieval,
)
from config.settings import get_settings

settings = get_settings()

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0,
    openai_api_key=settings.openai_api_key,
)

_llm_json = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
    openai_api_key=settings.openai_api_key,
)

_openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)


# ── State schema ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:          str
    chat_history:      list[dict]
    rewritten_queries: list[str]
    raw_chunks:        list[dict]
    colpali_pages:     list[dict]
    graded_chunks:     list[dict]
    reranked_chunks:   list[dict]
    full_chunks:       list[dict]
    page_images:       list[dict]
    context:           str
    answer:            str
    tables:            list[str]   # markdown tables extracted from the answer
    citations:         list[dict]
    reflection_loops:  int
    reflection_result: dict
    final_answer:      str
    cancel_token:       Any        # CancellationToken | None — checked at every node


async def _check_cancelled(state: AgentState) -> None:
    """
    Call at the top of every node. Previously nothing in this pipeline ever
    looked at cancellation — /chat/stop only aborted the client's own HTTP
    connection, while retrieval/rerank/generation kept running to
    completion on the server regardless. This makes "Stop" actually stop
    work, node by node, instead of just walking away from a socket.
    """
    token = state.get("cancel_token")
    if token is not None:
        await token.raise_if_cancelled()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_tables(text: str) -> list[str]:
    """Extract all markdown table blocks from a text string."""
    lines   = text.splitlines()
    tables  = []
    current = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            current.append(line)
        else:
            if len(current) >= 2:   # header + separator = valid table
                tables.append("\n".join(current))
            current = []

    if len(current) >= 2:
        tables.append("\n".join(current))

    return tables


# ── Node 1: Query Rewriter ─────────────────────────────────────────────────────

async def query_rewriter_node(state: AgentState) -> dict:
    await _check_cancelled(state)
    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in state["chat_history"][-6:]
    )
    prompt   = QUERY_REWRITE_PROMPT.format(
        n=2, history=history_str or "None", question=state["question"],
    )
    response = await _llm.ainvoke(prompt)
    raw      = response.content.strip()

    try:
        queries = json.loads(raw)
        if not isinstance(queries, list):
            queries = [state["question"]]
    except Exception:
        queries = re.findall(r'"([^"]+)"', raw) or [state["question"]]

    if state["question"] not in queries:
        queries.insert(0, state["question"])

    return {"rewritten_queries": queries[:2]}


# ── Node 2: Retriever ──────────────────────────────────────────────────────────

async def retriever_node(state: AgentState) -> dict:
    """Run hybrid chunk + ColPali page retrieval in parallel."""
    await _check_cancelled(state)
    queries = state["rewritten_queries"]

    chunk_tasks  = [hybrid_retrieval(q, top_k=settings.max_retrieval_k) for q in queries]
    colpali_task = colpali_retrieval(queries[0], top_k=5)

    *chunk_results_list, colpali_results = await asyncio.gather(*chunk_tasks, colpali_task)

    merged: dict[str, dict] = {}
    for results in chunk_results_list:
        for chunk in results:
            cid = chunk["chunk_id"]
            if cid not in merged or chunk["score"] > merged[cid]["score"]:
                merged[cid] = chunk

    sorted_chunks = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    return {
        "raw_chunks":    sorted_chunks[:settings.max_retrieval_k],
        "colpali_pages": colpali_results,
    }


# ── Node 3: Retrieval Grader (pass-through) ────────────────────────────────────

async def retrieval_grader_node(state: AgentState) -> dict:
    await _check_cancelled(state)
    question = state["question"]
    chunks   = state["raw_chunks"]

    if not chunks:
        return {"graded_chunks": []}

    async def grade_one(chunk: dict) -> dict | None:
        content = chunk.get("content_preview", "")[:400]
        prompt  = RETRIEVAL_GRADE_PROMPT.format(question=question, chunk=content)
        try:
            response = await _llm_json.ainvoke(prompt)
            result   = json.loads(response.content)
            if result.get("relevant", False):
                return chunk
        except Exception:
            return chunk
        return None

    batch_size = 10
    graded: list[dict] = []
    for i in range(0, len(chunks), batch_size):
        batch   = chunks[i : i + batch_size]
        results = await asyncio.gather(*[grade_one(c) for c in batch])
        graded.extend([r for r in results if r is not None])

    # ← CHANGED: only fall back if at least some chunks scored reasonably well
    # Don't force bad context — return empty and let the generator handle it
    if not graded:
        # Check top chunk score — if it's very low, nothing is relevant
        top_score = chunks[0].get("score", 0) if chunks else 0
        if top_score > 0.4:   # only use fallback if chunks are at least somewhat relevant
            graded = chunks[:3]
        # else: return empty → generator will give a sensible "not found" or general answer

    return {"graded_chunks": graded}

# ── Node 4: Reranker (OpenAI LLM-based cross-encoder) ──────────────────────────

async def reranker_node(state: AgentState) -> dict:
    """
    Rerank candidate chunks for relevance using an OpenAI LLM (replaces Cohere).
    Falls back to embedding-score ordering if the LLM call fails.
    """
    await _check_cancelled(state)
    chunks = state["graded_chunks"]
    if not chunks:
        return {"reranked_chunks": []}

    top_n      = settings.rerank_top_n
    candidates = chunks[:settings.max_retrieval_k]

    # Embedding-score order is the fallback / tie-breaker.
    score_order = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)

    if len(candidates) <= top_n:
        return {"reranked_chunks": score_order[:top_n]}

    passages = "\n\n".join(
        f"[{i}] {(c.get('content_preview') or c.get('metadata', {}).get('content_preview', ''))[:400]}"
        for i, c in enumerate(candidates)
    )
    prompt = RERANK_PROMPT.format(question=state["question"], n=top_n, passages=passages)

    try:
        response = await _llm_json.ainvoke(prompt)
        ranking  = json.loads(response.content).get("ranking", [])
        seen: set[int] = set()
        reranked: list[dict] = []
        for idx in ranking:
            if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                reranked.append(candidates[idx])
        # Backfill from score order if the LLM returned too few
        for c in score_order:
            if len(reranked) >= top_n:
                break
            if c not in reranked:
                reranked.append(c)
    except Exception:
        reranked = score_order

    return {"reranked_chunks": reranked[:top_n]}


# ── Node 5: Context Builder ────────────────────────────────────────────────────

async def context_builder_node(state: AgentState) -> dict:
    """Fetch full chunk text + ColPali page images; build text context."""
    await _check_cancelled(state)
    chunks       = state["reranked_chunks"]
    colpali_hits = state.get("colpali_pages", [])

    if not chunks and not colpali_hits:
        return {
            "context":     "No relevant content found in the EY Middle East knowledge base.",
            "full_chunks": [],
            "citations":   [],
            "page_images": [],
        }

    # Fetch full chunk texts in parallel
    full_data   = await asyncio.gather(*[fetch_full_chunk(c["chunk_id"]) for c in chunks])
    full_chunks = [f for f in full_data if f is not None]

    citations:     list[dict] = []
    context_parts: list[str]  = []

    for i, fc in enumerate(full_chunks, 1):
        source_file   = fc.get("source_file", "Unknown")
        page_or_slide = fc.get("page_or_slide", "N/A")
        section_title = fc.get("section_title", "")
        doc_type      = fc.get("doc_type", "")
        engagement_id = fc.get("engagement_id", "")
        kind          = fc.get("kind", "text")
        content       = fc.get("content", "")[:2000]   # more context for tables

        label      = f"Source {i}"
        page_label = "Page" if doc_type == "pdf" else "Slide" if doc_type == "pptx" else "Section"
        kind_tag   = f" [{kind.upper()}]" if kind in ("table", "image") else ""

        context_parts.append(
            f"[{label}]{kind_tag} {source_file} | {page_label} {page_or_slide}"
            + (f" | {section_title}" if section_title else "")
            + f"\n{content}"
        )
        citations.append({
            "label":         label,
            "source_file":   source_file,
            "page_label":    page_label,
            "page":          page_or_slide,
            "section":       section_title,
            "engagement_id": engagement_id,
            "doc_type":      doc_type,
            "kind":          kind,
        })

    # Fetch ColPali page images in parallel
    page_image_data = await asyncio.gather(*[fetch_colpali_page(h["page_id"]) for h in colpali_hits])
    page_images     = [p for p in page_image_data if p is not None]

    return {
        "context":     "\n\n---\n\n".join(context_parts),
        "full_chunks": full_chunks,
        "citations":   citations,
        "page_images": page_images,
    }


# ── Node 6: Generator ─────────────────────────────────────────────────────────

async def generator_node(state: AgentState) -> dict:
    """
    Multimodal generation:
    - Text context from chunks (always)
    - Up to 3 ColPali page images for visual context (when available)
    Extracts markdown tables from the answer and returns them separately.
    """
    page_images = state.get("page_images", [])
    await _check_cancelled(state)
    prompt_text = GENERATION_PROMPT.format(
        question=state["question"],
        context=state["context"],
    )

    if not page_images:
        # Text-only path
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            *state["chat_history"][-4:],
            {"role": "user", "content": prompt_text},
        ]
        response = await _llm.ainvoke(messages)
        answer   = response.content.strip()
    else:
        # Multimodal path — include up to 3 ColPali page images
        user_content: list[dict] = [{"type": "text", "text": prompt_text}]
        for pg in page_images[:3]:
            img_b64  = pg.get("page_image_b64", "")
            page_idx = pg.get("page_idx", 0)
            src_file = pg.get("source_file", "")
            if not img_b64:
                continue
            user_content.append({"type": "text", "text": f"[Page image: {src_file}, page {page_idx + 1}]"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "low"},
            })

        openai_messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            *state["chat_history"][-4:],
            {"role": "user", "content": user_content},
        ]
        resp   = await _openai_client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=openai_messages,
            max_tokens=1500,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()

    tables = _extract_tables(answer)
    return {"answer": answer, "tables": tables}


# ── Node 7: Reflection Grader ──────────────────────────────────────────────────

async def reflection_grader_node(state: AgentState) -> dict:
    await _check_cancelled(state)
    if settings.max_reflection_loops == 0:
        return {
            "reflection_result": {"quality": "good"},
            "final_answer":      state["answer"],
        }
    prompt = REFLECTION_PROMPT.format(question=state["question"], answer=state["answer"])
    try:
        response = await _llm_json.ainvoke(prompt)
        result   = json.loads(response.content)
    except Exception:
        result = {"quality": "good", "reason": "Evaluation parse error."}
    return {
        "reflection_result": result,
        "final_answer": state["answer"] if result.get("quality") == "good" else "",
    }


# ── Node 8: Conditional edge ───────────────────────────────────────────────────

def should_retry(state: AgentState) -> str:
    if settings.max_reflection_loops == 0:
        return "end"
    loops  = state.get("reflection_loops", 0)
    result = state.get("reflection_result", {})
    if (
        result.get("quality") == "needs_improvement"
        and loops < settings.max_reflection_loops
        and result.get("suggested_followup_query")
    ):
        return "retry"
    return "end"


# ── Node 9: Retry Prep ─────────────────────────────────────────────────────────

async def retry_prep_node(state: AgentState) -> dict:
    await _check_cancelled(state)
    followup = state["reflection_result"].get("suggested_followup_query", state["question"])
    return {
        "question":         followup,
        "reflection_loops": state.get("reflection_loops", 0) + 1,
        "raw_chunks":       [],
        "colpali_pages":    [],
        "graded_chunks":    [],
        "reranked_chunks":  [],
        "page_images":      [],
        "tables":           [],
    }