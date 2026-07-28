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
    async def run(
        self,
        question: str,
        chat_history: list[dict],
        rag_graph: Any,
        cancel_token: Any = None,
    ) -> AgentResult: ...


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

    async def run(self, question, chat_history, rag_graph, cancel_token=None) -> AgentResult:
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

    async def run(self, question, chat_history, rag_graph, cancel_token=None) -> AgentResult:
        state = await _invoke_rag_graph(rag_graph, question, chat_history, cancel_token)
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

async def _invoke_rag_graph(
    rag_graph: Any, question: str, chat_history: list[dict], cancel_token: Any = None,
) -> dict:
    return await rag_graph.ainvoke({
        "question": question, "chat_history": chat_history,
        "reflection_loops": 0, "raw_chunks": [], "colpali_pages": [],
        "graded_chunks": [], "reranked_chunks": [], "full_chunks": [],
        "page_images": [], "citations": [], "tables": [],
        "rewritten_queries": [], "context": "", "answer": "",
        "final_answer": "", "reflection_result": {},
        # Wired through so every node in the graph can check it — see
        # agent/nodes.py's _check_cancelled(). Without this, /chat/stop
        # only ever aborted the client's own connection.
        "cancel_token": cancel_token,
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
- Create the figure with a fixed, generous size so labels have room to
  breathe: fig, ax = plt.subplots(figsize=(9, 6)).
- X-axis category labels: rotate with plt.xticks(rotation=30, ha='right')
  (30 degrees, not 45+ — steeper angles push labels further down and are
  more likely to collide with anything placed below the axes).
- If any category from the source table was left out because its language
  was too vague to count (e.g. "majority", "mixed practices"), you MUST
  add a footnote — but it MUST NOT collide with the rotated x-axis labels.
  To guarantee that:
    1. Place the footnote with fig.text(...) in FIGURE coordinates, not
       ax.text() in axes coordinates — e.g.
       fig.text(0.5, 0.02, "Excluded: <category> (<reason>).",
                ha='center', fontsize=8, style='italic')
    2. Reserve space for it explicitly with
       fig.subplots_adjust(bottom=0.30)
       (increase to 0.35+ if the footnote text is long or category names
       are long/rotated) so the rotated tick labels and the footnote each
       get their own vertical band and never overlap.
    3. Do NOT call plt.tight_layout() in the same script as fig.text() —
       tight_layout() recalculates margins and will undo the
       subplots_adjust() reservation, which is what causes the footnote
       to land on top of the tick labels. Use subplots_adjust() alone.
  If there is nothing excluded, skip the footnote entirely — don't add an
  empty or placeholder one.
- Leave adequate left margin for the y-axis label (subplots_adjust(left=...)
  if the number labels are wide) and a clear, non-overlapping title.
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

    async def run(self, question, chat_history, rag_graph, cancel_token=None) -> AgentResult:
        # 1. Standard retrieval — same path every other agent uses, gives us
        #    citations plus a baseline answer/context.
        state = await _invoke_rag_graph(rag_graph, question, chat_history, cancel_token)
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

        image_b64, render_error = await _execute_chart_code(code, cancel_token)

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

        answer = await self._explain_chart(question, context, code)
        return AgentResult(
            agent_name="code_graph",
            answer=answer,
            code=code,
            image_b64=image_b64,
            chart_type="matplotlib",
            citations=citations,
        )

    async def _explain_chart(self, question: str, context: str, code: str) -> str:
        """
        Produce the chat-facing text for a successfully rendered chart.

        Previously this just dumped the raw matplotlib code into the answer
        (wrapped in a ```python fence), which is why the UI showed a code
        block sitting on top of the image — useful for a developer, not for
        a consultant asking "chart X for me". The code is still returned via
        AgentResult.code for anyone who wants it later; the chat answer
        itself should just be the kind of one-paragraph gloss a consulting
        analyst would give when handing over a chart: what it shows, and any
        caveat (e.g. an excluded category) worth flagging.
        """
        prompt = f"""\
A chart has just been generated and will be shown to the user directly below your reply.

User's request:
{question}

Data the chart was built from:
{context[:2500]}

Chart-building code (for your reference only — do NOT reproduce, quote, or mention code, \
Python, or matplotlib in your reply):
{code}

Write a short (2-4 sentence) consulting-style caption for this chart:
- State what the chart shows in plain language.
- Call out the standout data point(s) or comparison.
- If the chart's code excludes a category (e.g. via an ax.text footnote) because the \
source language was too vague to count, mention that exclusion and why in one clause.
- Do NOT mention "source", "document", "page", or "retrieved context" — just describe the data.
- Do NOT include any code, code fences, or the word "matplotlib".
"""
        try:
            response = await _llm.ainvoke([{"role": "user", "content": prompt}])
            explanation = response.content.strip()
            # Belt-and-suspenders: strip any code fence the model adds anyway.
            explanation = re.sub(r"```.*?```", "", explanation, flags=re.DOTALL).strip()
            return explanation or "Here's the chart, built from data in the EY knowledge base."
        except Exception:
            return "Here's the chart, built from data in the EY knowledge base."


async def _execute_chart_code(code: str, cancel_token: Any = None) -> tuple[str | None, str | None]:
    """
    Execute LLM-generated matplotlib code in an isolated subprocess.

    Returns (image_b64, error):
      - On success:  (base64_png, None)
      - On failure:  (None, human-readable reason — traceback, stderr,
                      timeout message, cancellation, or launch failure) so
                      the caller can log *why* it failed instead of just
                      that it failed.

    The LLM's own code is now INSIDE the same try/except as the savefig
    step (previously it wasn't — an error in the LLM's code, e.g. a typo'd
    column name, would crash the whole subprocess before the try/except
    even started, and get silently discarded).

    cancel_token, if given, is raced against the subprocess: previously
    this ran to completion no matter what — clicking Stop only ever
    aborted the browser's own connection while the subprocess (and the
    CPU it was using) kept going server-side regardless.
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
    except Exception as e:
        return None, f"Failed to launch chart subprocess: {e!r}"

    communicate_task = asyncio.ensure_future(proc.communicate())
    waiters = {communicate_task}
    cancel_wait_task = None
    if cancel_token is not None:
        cancel_wait_task = asyncio.ensure_future(cancel_token.wait())
        waiters.add(cancel_wait_task)

    try:
        done, pending = await asyncio.wait(waiters, timeout=30, return_when=asyncio.FIRST_COMPLETED)
    except Exception as e:
        proc.kill()
        return None, f"Chart execution failed: {e!r}"

    if cancel_wait_task is not None and cancel_wait_task in done:
        proc.kill()
        communicate_task.cancel()
        return None, "Chart generation cancelled by user."

    if communicate_task not in done:
        proc.kill()
        if cancel_wait_task is not None:
            cancel_wait_task.cancel()
        return None, "Chart rendering timed out after 30s."

    if cancel_wait_task is not None:
        cancel_wait_task.cancel()

    stdout, stderr = communicate_task.result()
    stdout_text = stdout.decode(errors="replace")
    for line in stdout_text.splitlines():
        if line.startswith("BASE64:"):
            return line[7:], None
    error = stderr.decode(errors="replace").strip()
    return None, error or "Subprocess exited with no BASE64 output and no stderr."


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

    async def run(self, question, chat_history, rag_graph, cancel_token=None) -> AgentResult:
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

    async def run(self, question, chat_history, rag_graph, cancel_token=None) -> AgentResult:
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


async def route_and_run(
    question: str, chat_history: list[dict], rag_graph: Any, cancel_token: Any = None,
) -> AgentResult:
    # Fast path — regex triggers
    for agent in AGENT_REGISTRY[:-1]:
        if agent.can_handle(question):
            return await agent.run(question, chat_history, rag_graph, cancel_token)

    # Slow path — LLM intent classification
    if cancel_token is not None:
        await cancel_token.raise_if_cancelled()
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
                return await a.run(question, chat_history, rag_graph, cancel_token)
        agent = AGENT_REGISTRY[-1]
    else:
        agent = AGENT_REGISTRY[-1]

    return await agent.run(question, chat_history, rag_graph, cancel_token)