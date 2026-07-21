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
from langchain_openai import ChatOpenAI

from agent.prompts import (
    CHITCHAT_SYSTEM_PROMPT,
    INTENT_CLASSIFY_PROMPT,
)
from config.settings import get_settings

settings = get_settings()

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
        state = await rag_graph.ainvoke({
            "question": question, "chat_history": chat_history,
            "reflection_loops": 0, "raw_chunks": [], "colpali_pages": [],
            "graded_chunks": [], "reranked_chunks": [], "full_chunks": [],
            "page_images": [], "citations": [], "tables": [],
            "rewritten_queries": [], "context": "", "answer": "",
            "final_answer": "", "reflection_result": {},
        })
        return AgentResult(
            agent_name="retrieval",
            answer=state.get("final_answer") or state.get("answer", ""),
            citations=state.get("citations", []),
            tables=state.get("tables", []),
            raw_state=state,
        )


_CODE_GRAPH_SYSTEM = """\
You are a data visualisation expert working on EY consulting projects.
Generate Python code using matplotlib or plotly to create the requested chart/graph.
The code must be self-contained, use EY brand colours (#FFE600, #2E2E38, #00A3A1),
save output as PNG to BytesIO and print as base64, include placeholder data if none provided.
Return ONLY the Python code — no explanations, no markdown fences.
"""

_CODE_GRAPH_TRIGGERS = re.compile(
    r"\b(chart|plot|graph|visuali[sz]|bar chart|line chart|pie chart|scatter|"
    r"histogram|heatmap|treemap|matplotlib|plotly|draw a graph|show me a graph|"
    r"create a graph|generate a (chart|graph|plot)|visualise data)\b",
    re.IGNORECASE,
)


class CodeGraphAgent(SpecialisedAgent):
    name = "code_graph"

    def can_handle(self, query: str) -> bool:
        return bool(_CODE_GRAPH_TRIGGERS.search(query))

    async def run(self, question, chat_history, rag_graph) -> AgentResult:
        messages = [{"role": "system", "content": _CODE_GRAPH_SYSTEM}, *chat_history[-4:], {"role": "user", "content": question}]
        response = await _llm.ainvoke(messages)
        code = re.sub(r"^```(?:python)?\s*", "", response.content.strip(), flags=re.MULTILINE)
        code = re.sub(r"\s*```$", "", code, flags=re.MULTILINE)
        image_b64 = await _execute_chart_code(code)
        answer = f"I've generated the requested visualisation.\n\n**Code used:**\n```python\n{code}\n```"
        return AgentResult(agent_name="code_graph", answer=answer, code=code, image_b64=image_b64, chart_type="matplotlib")


async def _execute_chart_code(code: str) -> str | None:
    wrapper = textwrap.dedent(f"""
import sys, io, base64
import matplotlib
matplotlib.use("Agg")
{code}
try:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close("all")
    buf.seek(0)
    print("BASE64:" + base64.b64encode(buf.read()).decode())
except Exception:
    pass
""")
    try:
        proc = await asyncio.create_subprocess_exec(sys.executable, "-c", wrapper, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        for line in stdout.decode().splitlines():
            if line.startswith("BASE64:"):
                return line[7:]
    except Exception:
        pass
    return None


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