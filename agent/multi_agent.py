"""
agent/multi_agent.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import openai
import structlog
from langchain_openai import ChatOpenAI

from agent.prompts import (
    CHITCHAT_SYSTEM_PROMPT,
    INTENT_CLASSIFY_PROMPT,
)
from agent.tools import fetch_full_chunk, hybrid_retrieval
from config.settings import get_settings

settings = get_settings()
log = structlog.get_logger()

_llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0, openai_api_key=settings.openai_api_key)
_llm_json = ChatOpenAI(model=settings.openai_chat_model, temperature=0, model_kwargs={"response_format": {"type": "json_object"}}, openai_api_key=settings.openai_api_key)
_openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)


@dataclass
class AgentResult:
    agent_name:   str
    answer:       str
    citations:    list[dict] = field(default_factory=list)
    tables:       list[str]  = field(default_factory=list)
    code:         str | None = None
    image_b64:    str | None = None
    chart_type:   str | None = None
    raw_state:    dict       = field(default_factory=dict)


class SpecialisedAgent(ABC):
    name: str = "base"

    @abstractmethod
    def can_handle(self, query: str) -> bool: ...

    @abstractmethod
    async def run(self, question: str, chat_history: list[dict], rag_graph: Any) -> AgentResult: ...


class ChitchatAgent(SpecialisedAgent):
    name = "chitchat"
    _TRIGGERS = re.compile(
        r"^(hi|hello|hey|hiya|howdy|good\s?(morning|afternoon|evening)|"
        r"thanks?|thank you|thx|bye|goodbye|ok|okay|cool|great|awesome|"
        r"what can you do|who are you|what are you|help me|how does this work|"
        r"what do you know|tell me about yourself|introduce yourself)\W*$",
        re.IGNORECASE,
    )

    def can_handle(self, query: str) -> bool:
        return bool(self._TRIGGERS.match(query.strip()))

    async def run(self, question, chat_history, rag_graph) -> AgentResult:
        messages = [
            {"role": "system", "content": CHITCHAT_SYSTEM_PROMPT},
            *chat_history[-4:],
            {"role": "user", "content": question},
        ]
        response = await _llm.ainvoke(messages)
        return AgentResult(agent_name="chitchat", answer=response.content.strip())


class RetrievalAgent(SpecialisedAgent):
    name = "retrieval"

    def can_handle(self, query: str) -> bool:
        return True

    async def run(self, question, chat_history, rag_graph) -> AgentResult:
        state = await _invoke_rag_graph(rag_graph, question, chat_history)
        return AgentResult(
            agent_name="retrieval",
            answer=state.get("final_answer") or state.get("answer", ""),
            citations=state.get("citations", []),
            tables=state.get("tables", []),
            raw_state=state,
        )


# ── Shared helper — every agent that needs real KB data goes through this ─────
# Pulled out of RetrievalAgent so CodeGraphAgent can reuse the exact same
# retrieval path instead of guessing/inventing numbers of its own.

async def _invoke_rag_graph(rag_graph: Any, question: str, chat_history: list[dict]) -> dict:
    return await rag_graph.ainvoke({
        "question": question, "chat_history": chat_history,
        "reflection_loops": 0, "raw_chunks": [], "colpali_pages": [],
        "graded_chunks": [], "reranked_chunks": [], "full_chunks": [],
        "page_images": [], "citations": [], "tables": [],
        "rewritten_queries": [], "context": "", "answer": "",
        "final_answer": "", "reflection_result": {},
    })


# ── Chart / graph generation — now GROUNDED in retrieved KB data ─────────────
#
# WHY THIS CHANGED:
# The previous version asked an LLM to write matplotlib code straight from
# the user's question, with a system prompt that explicitly told it to
# "include placeholder data if none provided". That's how charts like
# [5, 3, 4, 2] "Deviations from Standard Formula by Ratio" got generated —
# confident-looking numbers that were never in any EY document.
#
# The fix: retrieve real context from the knowledge base FIRST (the same
# rag_graph every other agent uses), then require the chart-writing LLM to
# pull its numbers only from that retrieved text/tables. If the retrieved
# context doesn't actually contain chartable numbers, the agent says so
# instead of inventing a bar chart.

_CODE_GRAPH_SYSTEM = """\
You are a data visualisation expert working on EY consulting projects.
Generate Python code using matplotlib to create the chart the user asked for.

GROUNDING RULES — these override everything else:
- You may ONLY use numeric values, labels, and categories that appear
  explicitly in the "Retrieved knowledge base context" below.
- Never invent, estimate, round, or "fill in" a number that isn't present
  in the context. Do not use placeholder or example data of any kind.
- "Present in the context" includes counts spelled out in prose, not just
  numerals. If a remark says "one company excluded X" or "two companies
  used Y instead", that IS a concrete data point — count it as 1 or 2
  respectively for that category. Extract these the way a careful analyst
  reading the table by hand would: one mention of a specific numbered
  deviation ("one company...") = a count of 1 for that row/category.
- Do NOT convert vague, non-numeric language into a count. Words like
  "majority", "mixed practices", "most companies", "few companies", or
  "some companies" do not have an extractable number — leave that
  category out of the chart rather than guessing a value for it. Only
  chart categories where the context gives you an explicit number
  (numeral OR spelled-out word like "one"/"two"/"three").
- If, after this extraction, fewer than 2 categories have an explicit
  count, do NOT write any chart code — a chart needs multiple comparable
  data points to be meaningful. Instead output exactly one line starting
  with "NO_CHART:" followed by a short, specific explanation of what
  numeric data is missing or too vague to chart.
- When the context contains a markdown table, read values directly from
  the table cells — do not paraphrase or approximate them.
- If several numbers could answer the request, prefer the ones most
  directly tied to what the user asked (matching ratio/category/metric
  names exactly where possible).

Technical requirements (only apply if you ARE generating a chart):
- Self-contained script using matplotlib (the "Agg" backend and BytesIO/
  base64 imports are already set up by the caller — do not re-import or
  reconfigure the backend).
- Use the EY brand colours #FFE600, #2E2E38, #00A3A1, cycling through them
  if there are more categories than colours.
- If any category from the source table was left out because its language
  was too vague to count (e.g. "majority", "mixed practices"), add a
  small ax.text() footnote below the chart naming which category was
  excluded and why — don't silently drop it with no explanation.
- Do NOT call plt.savefig(), plt.show(), or write any file to disk. Just
  build the figure (plt.subplots(), ax.bar()/plot()/etc., labels, title).
  The caller renders and encodes the figure for you after your code runs —
  calling savefig() yourself can crash the render if the working directory
  isn't writable, and adds nothing.
- Return ONLY the Python code (or the single NO_CHART line) — no
  explanations, no markdown code fences.

Retrieved knowledge base context:
{context}
"""

_CODE_GRAPH_TRIGGERS = re.compile(
    r"\b(chart|plot|graph|visuali[sz]|bar chart|line chart|pie chart|scatter|"
    r"histogram|heatmap|treemap|matplotlib|plotly|draw a graph|show me a graph|"
    r"create a graph|generate a (chart|graph|plot)|visualise data)\b",
    re.IGNORECASE,
)

# How much retrieved context to hand the chart-writing LLM. Raised from the
# original 6000 because a real fix requires reassembling a whole multi-page
# table, not just whatever the general-purpose reranker kept.
_MAX_CONTEXT_CHARS = 12000

# How many raw candidates to pull when widening the net for chart requests.
_CHART_RETRIEVAL_TOP_K = 25


class CodeGraphAgent(SpecialisedAgent):
    name = "code_graph"

    def can_handle(self, query: str) -> bool:
        return bool(_CODE_GRAPH_TRIGGERS.search(query))

    async def _gather_chart_context(self, question: str, state: dict) -> str:
        """
        Build a fuller context string for charting than the standard RAG
        pass provides.

        The standard pipeline (retriever -> grader -> reranker) truncates
        down to `settings.rerank_top_n` chunks, which is fine for a text
        answer but can silently drop rows of a table that spans multiple
        pages/chunks. Here we:
          1. Keep whatever context/tables the standard pass already built.
          2. Re-run retrieval with a much wider top_k to surface chunks the
             reranker discarded.
          3. Identify the source document(s) the question is actually about
             (from the reranked chunks + the wide hits).
          4. Fetch the FULL content of every candidate chunk belonging to
             those source documents, so a multi-page table gets
             reassembled instead of truncated.
        """
        parts: list[str] = []

        existing_context = state.get("context", "")
        if existing_context and "No relevant content found" not in existing_context:
            parts.append(existing_context)

        for t in state.get("tables", []):
            if t not in parts:
                parts.append(t)

        # Widen the net beyond the reranked top-N.
        try:
            wide_hits = await hybrid_retrieval(question, top_k=_CHART_RETRIEVAL_TOP_K)
        except Exception:
            wide_hits = []

        # Figure out which source document(s) this question is really about.
        source_files: set[str] = set()
        for c in state.get("reranked_chunks", []):
            sf = (c.get("metadata") or {}).get("source_file")
            if sf:
                source_files.add(sf)
        if not source_files:
            for h in wide_hits:
                sf = (h.get("metadata") or {}).get("source_file")
                if sf:
                    source_files.add(sf)

        # Collect every candidate chunk_id belonging to those source files
        # (falls back to all wide hits if we couldn't pin down a source file).
        seen_ids: set[str] = set()
        chunk_ids: list[str] = []
        for h in wide_hits:
            cid = h.get("chunk_id")
            sf  = (h.get("metadata") or {}).get("source_file")
            if not cid or cid in seen_ids:
                continue
            if source_files and sf not in source_files:
                continue
            seen_ids.add(cid)
            chunk_ids.append(cid)

        if chunk_ids:
            full_chunks = await asyncio.gather(
                *[fetch_full_chunk(cid) for cid in chunk_ids]
            )
            for fc in full_chunks:
                if not fc:
                    continue
                content = fc.get("content", "")
                if content and content not in parts:
                    parts.append(content)

        return "\n\n---\n\n".join(p for p in parts if p).strip()

    async def run(self, question, chat_history, rag_graph) -> AgentResult:
        # 1. Standard retrieval — same path every other agent uses, gives us
        #    citations plus a baseline answer/context.
        state = await _invoke_rag_graph(rag_graph, question, chat_history)
        citations = state.get("citations", [])

        # 2. Widen the net specifically for charting. The standard pipeline
        #    reranks down to a handful of chunks, which is fine for a text
        #    answer but silently truncates a table that spans multiple
        #    pages/chunks (exactly what happened with the Schedule III ratio
        #    table — rows on page 8 and 10 kept getting dropped, so the LLM
        #    only ever saw page 9 and correctly (but needlessly) refused to
        #    chart). Pull more candidates and reassemble every table chunk
        #    from the same source document so the LLM sees the full table.
        context = await self._gather_chart_context(question, state)

        if not context:
            return AgentResult(
                agent_name="code_graph",
                answer=(
                    "I couldn't find any data in the EY knowledge base to build "
                    "this chart from. Try asking the underlying knowledge "
                    "question first (e.g. \"how many companies deviated from "
                    "each ratio formula\") so there's real data to visualise, "
                    "then ask for the chart."
                ),
                citations=citations,
            )

        # 3. Ask the LLM to chart ONLY what's in that (now much fuller) context.
        system_prompt = _CODE_GRAPH_SYSTEM.format(context=context[:_MAX_CONTEXT_CHARS])
        messages = [
            {"role": "system", "content": system_prompt},
            *chat_history[-4:],
            {"role": "user", "content": question},
        ]
        response = await _llm.ainvoke(messages)
        raw = response.content.strip()

        # 4. Model self-reported that the data isn't there — trust it and
        #    say so plainly instead of forcing a chart.
        if raw.upper().startswith("NO_CHART"):
            reason = raw.split(":", 1)[1].strip() if ":" in raw else ""
            answer = (
                "I found relevant information but not enough concrete "
                "numbers to chart it accurately."
                + (f" {reason}" if reason else "")
                + "\n\nHere's what the knowledge base actually says:\n\n"
                + context[:1500]
            )
            return AgentResult(agent_name="code_graph", answer=answer, citations=citations)

        code = re.sub(r"^```(?:python)?\s*", "", raw, flags=re.MULTILINE)
        code = re.sub(r"\s*```$", "", code, flags=re.MULTILINE)

        # Safety net: even though the prompt tells the model not to, strip
        # any stray plt.savefig(...)/plt.show() calls it emits anyway. A
        # savefig to a file path can crash the whole render if the sandbox's
        # working directory isn't writable — we always render from the
        # in-memory figure state ourselves regardless of what the code did.
        code = re.sub(r"^\s*plt\.savefig\([^)]*\)\s*$", "", code, flags=re.MULTILINE)
        code = re.sub(r"^\s*plt\.show\(\)\s*$", "", code, flags=re.MULTILINE)

        image_b64, render_error = await _execute_chart_code(code)

        if image_b64 is None:
            # Rendering failed — don't pretend nothing happened, and don't
            # swallow *why*. Log the real error/traceback server-side so
            # this is diagnosable from logs instead of by guessing, then
            # fall back to the grounded text/table data so the user still
            # gets a correct, truthful answer.
            log.warning(
                "chart_render_failed",
                error=render_error,
                code=code,
            )
            answer = (
                "I generated chart code from the retrieved data, but it "
                "failed to render. Here's the underlying data instead:\n\n"
                + context[:1500]
            )
            return AgentResult(agent_name="code_graph", answer=answer, code=code, citations=citations)

        answer = (
            "Here's the chart, built from data retrieved from the EY "
            "knowledge base (not invented).\n\n"
            f"**Code used:**\n```python\n{code}\n```"
        )
        return AgentResult(
            agent_name="code_graph",
            answer=answer,
            code=code,
            image_b64=image_b64,
            chart_type="matplotlib",
            citations=citations,
        )


async def _execute_chart_code(code: str) -> tuple[str | None, str | None]:
    """
    Execute LLM-generated matplotlib code in an isolated subprocess.

    Returns (image_b64, error):
      - On success:  (base64_png, None)
      - On failure:  (None, human-readable reason — traceback, stderr,
                      timeout message, or launch failure) so the caller can
                      log *why* it failed instead of just that it failed.

    The LLM's own code is now INSIDE the same try/except as the savefig
    step (previously it wasn't — an error in the LLM's code, e.g. a typo'd
    column name, would crash the whole subprocess before the try/except
    even started, and get silently discarded).
    """
    indented_code = textwrap.indent(code, "    ")
    wrapper = textwrap.dedent(f"""
import sys, io, base64, traceback
import matplotlib
matplotlib.use("Agg")
try:
{indented_code}
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close("all")
    buf.seek(0)
    print("BASE64:" + base64.b64encode(buf.read()).decode())
except Exception:
    traceback.print_exc()
    sys.exit(1)
""")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", wrapper,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        stdout_text = stdout.decode(errors="replace")
        for line in stdout_text.splitlines():
            if line.startswith("BASE64:"):
                return line[7:], None
        error = stderr.decode(errors="replace").strip()
        return None, error or "Subprocess exited with no BASE64 output and no stderr."
    except asyncio.TimeoutError:
        return None, "Chart rendering timed out after 30s."
    except Exception as e:
        return None, f"Failed to launch chart subprocess: {e!r}"


_FLOWCHART_SYSTEM = """\
You are a diagram expert. Generate a Mermaid diagram for the requested process/flow.
Return ONLY valid Mermaid syntax — no explanations, no code fences.
Use flowchart TD by default. Keep labels under 40 chars per node.
"""

_FLOWCHART_TRIGGERS = re.compile(
    r"\b(flowchart|flow chart|diagram|process flow|workflow|sequence diagram|"
    r"mermaid|graphviz|architecture diagram|draw a (flow|process|diagram)|"
    r"show (the|a|me) (flow|process|diagram|workflow)|map (the|a) process)\b",
    re.IGNORECASE,
)


class FlowchartAgent(SpecialisedAgent):
    name = "flowchart"

    def can_handle(self, query: str) -> bool:
        return bool(_FLOWCHART_TRIGGERS.search(query))

    async def run(self, question, chat_history, rag_graph) -> AgentResult:
        messages = [{"role": "system", "content": _FLOWCHART_SYSTEM}, *chat_history[-4:], {"role": "user", "content": question}]
        response = await _llm.ainvoke(messages)
        mermaid_code = re.sub(r"^```(?:mermaid)?\s*", "", response.content.strip(), flags=re.MULTILINE)
        mermaid_code = re.sub(r"\s*```$", "", mermaid_code, flags=re.MULTILINE)
        answer = f"Here is the requested diagram (rendered below).\n\n```mermaid\n{mermaid_code}\n```"
        return AgentResult(agent_name="flowchart", answer=answer, code=mermaid_code, chart_type="mermaid")


_IMAGE_GEN_TRIGGERS = re.compile(
    r"\b(generate (an?|the) image|create (an?|the) image|draw (an?|the)|"
    r"make (an?|the) image|dalle|dall-e|image of|picture of|illustration of|"
    r"generate (a )?visual)\b",
    re.IGNORECASE,
)


class ImageGenerationAgent(SpecialisedAgent):
    name = "image_generation"

    def can_handle(self, query: str) -> bool:
        return bool(_IMAGE_GEN_TRIGGERS.search(query))

    async def run(self, question, chat_history, rag_graph) -> AgentResult:
        try:
            resp = await _openai_client.images.generate(model="dall-e-3", prompt=question, size="1024x1024", quality="standard", n=1, response_format="b64_json")
            image_b64 = resp.data[0].b64_json
            answer = "Here is the generated image:"
        except Exception as e:
            image_b64 = None
            answer = f"Image generation failed: {e}"
        return AgentResult(agent_name="image_generation", answer=answer, image_b64=image_b64, chart_type="dalle")


# Registry — order matters, RetrievalAgent always last
AGENT_REGISTRY: list[SpecialisedAgent] = [
    ChitchatAgent(),
    FlowchartAgent(),
    CodeGraphAgent(),
    ImageGenerationAgent(),
    RetrievalAgent(),
]


async def route_and_run(question: str, chat_history: list[dict], rag_graph: Any) -> AgentResult:
    # Fast path — regex triggers
    for agent in AGENT_REGISTRY[:-1]:
        if agent.can_handle(question):
            return await agent.run(question, chat_history, rag_graph)

    # Slow path — LLM intent classification
    try:
        resp = await _llm_json.ainvoke([
            {"role": "system", "content": "You are a query classifier."},
            {"role": "user", "content": INTENT_CLASSIFY_PROMPT.format(question=question)},
        ])
        intent = json.loads(resp.content).get("intent", "knowledge")
    except Exception:
        intent = "knowledge"

    if intent == "chitchat":
        agent = next(a for a in AGENT_REGISTRY if a.name == "chitchat")
    elif intent == "visual":
        for a in AGENT_REGISTRY:
            if a.name in ("flowchart", "code_graph") and a.can_handle(question):
                return await a.run(question, chat_history, rag_graph)
        agent = AGENT_REGISTRY[-1]
    else:
        agent = AGENT_REGISTRY[-1]

    return await agent.run(question, chat_history, rag_graph)