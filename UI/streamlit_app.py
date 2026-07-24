"""
ui/streamlit_app.py

A minimal RAG chat UI built entirely on native Streamlit components.

Why this rewrite:
  - Tables were rendering as raw "|a|b|c|" text because the previous version
    piped every answer through a custom regex-based markdown renderer that
    silently kicked in whenever the `markdown` pip package wasn't installed
    — and that fallback never supported tables. Streamlit's own
    st.markdown/st.chat_message already render GFM tables, fenced code
    (with a copy button), and syntax highlighting correctly with zero extra
    dependencies, so we just use those directly. No parsing bugs possible.
  - The old app rebuilt chat bubbles, buttons, and layout from scratch with
    raw HTML + CSS pinned to Streamlit's internal DOM (`[data-testid=...]`
    selectors). That's brittle by nature: any Streamlit version bump changes
    the DOM shape and the CSS silently stops matching (that's what caused
    the button text-wrapping/overflow bugs too). Using st.chat_message,
    st.chat_input, st.feedback, and st.status means Streamlit maintains the
    styling for us, and theming is handled by .streamlit/config.toml
    instead of hand-written CSS variables.

Backend contract (adjust to match your FastAPI service):
  POST /chat/stream  {question, session_id, top_k} -> text/event-stream of
                      JSON lines: {"type":"token","text":...}
                                  {"type":"step","step":...}
                                  {"type":"done","citations":[...],
                                   "image_b64":..., "code":...,
                                   "agent_steps":[...]}
                      (falls back to POST /chat if this 404s)
  POST /chat          {question, session_id}        -> single JSON blob
  POST /feedback       {session_id, message, rating} -> optional, best-effort
  POST /upload         files=...                      -> optional, best-effort
"""

import base64
import json
import os
import time
import uuid
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# THEME — self-writing config.toml
#
# Streamlit's theme system (colors, font, radius) only reads from
# .streamlit/config.toml, and only at server *startup* — there's no
# in-script API to set it. Rather than asking you to hand-maintain that
# file separately, this app writes it next to itself automatically the
# first time it runs. Note: because Streamlit reads config before your
# script executes at all, the theme won't visually apply until the *next*
# time you launch `streamlit run` — after that it's fully self-contained.
# ─────────────────────────────────────────────────────────────────────────

_THEME_TOML = """[theme]
base = "light"
primaryColor = "#F2C94C"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F7F7F8"
textColor = "#1A1A1A"
font = "sans serif"
baseRadius = "medium"

[client]
toolbarMode = "minimal"

[server]
runOnSave = true
"""

_config_dir = Path(__file__).resolve().parent / ".streamlit"
_config_path = _config_dir / "config.toml"
if not _config_path.exists():
    _config_dir.mkdir(parents=True, exist_ok=True)
    _config_path.write_text(_THEME_TOML)

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────

def _new_chat_dict():
    return {"title": "New chat", "messages": []}


if "chats" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.chats = {cid: _new_chat_dict()}
    st.session_state.current_chat = cid

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

if "show_steps" not in st.session_state:
    st.session_state.show_steps = True


def new_chat():
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = _new_chat_dict()
    st.session_state.current_chat = cid


def truncate(text, n=32):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def export_markdown(chat) -> str:
    lines = [f"# {chat['title']}", ""]
    for m in chat["messages"]:
        who = "**You**" if m["role"] == "user" else "**Assistant**"
        lines.append(f"{who}:\n\n{m['content']}\n")
        for c in m.get("citations", []) or []:
            lines.append(f"> Source: {c.get('source_file', 'Source')}")
        lines.append("")
    return "\n".join(lines)


def send_feedback(message_text: str, rating: str):
    try:
        httpx.post(
            f"{API_URL}/feedback",
            json={"session_id": st.session_state.session_id, "message": message_text, "rating": rating},
            timeout=5,
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
# BACKEND CALL — returns (generator_of_text_chunks, result_dict)
# result_dict is filled in-place as the generator is consumed, and holds
# citations / agent_steps / image once streaming finishes.
# ─────────────────────────────────────────────────────────────────────────

def ask_backend(question: str, session_id: str, top_k: int):
    result = {"citations": [], "agent_steps": [], "image_b64": None}

    def gen():
        try:
            with httpx.stream(
                "POST",
                f"{API_URL}/chat/stream",
                json={"question": question, "session_id": session_id, "top_k": top_k},
                timeout=300,
            ) as resp:
                if resp.status_code == 404:
                    raise RuntimeError("no streaming endpoint")
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line[5:].strip() if line.startswith("data:") else line
                    try:
                        evt = json.loads(line)
                    except Exception:
                        continue
                    etype = evt.get("type")
                    if etype == "token":
                        yield evt.get("text", "")
                    elif etype == "step":
                        result["agent_steps"].append(evt.get("step", ""))
                    elif etype == "done":
                        result["citations"] = evt.get("citations", [])
                        result["agent_steps"] = evt.get("agent_steps", result["agent_steps"])
                        result["image_b64"] = evt.get("image_b64")
                return
        except Exception:
            pass  # fall through to non-streaming endpoint below

        try:
            resp = httpx.post(
                f"{API_URL}/chat",
                json={"question": question, "session_id": session_id, "top_k": top_k},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            result["citations"] = data.get("citations", [])
            result["agent_steps"] = data.get("agent_steps", [])
            result["image_b64"] = data.get("image_b64")
            answer = data.get("answer", "No answer returned.")
            for word in answer.split(" "):
                yield word + " "
                time.sleep(0.012)
        except Exception as e:
            yield f"⚠️ Couldn't reach the assistant backend: {e}"

    return gen(), result


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 💬 Knowledge Assistant")

    if st.button("＋ New chat", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    search = st.text_input("Search", placeholder="Search conversations", label_visibility="collapsed")

    st.caption("Recent")
    items = list(reversed(list(st.session_state.chats.items())))
    if search:
        items = [(cid, c) for cid, c in items if search.lower() in c["title"].lower()]

    for chat_id, chat_data in items:
        col1, col2 = st.columns([6, 1])
        active = chat_id == st.session_state.current_chat
        with col1:
            if st.button(
                ("● " if active else "") + chat_data["title"],
                key=f"open_{chat_id}",
                use_container_width=True,
            ):
                st.session_state.current_chat = chat_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{chat_id}"):
                del st.session_state.chats[chat_id]
                if not st.session_state.chats:
                    new_chat()
                elif st.session_state.current_chat == chat_id:
                    st.session_state.current_chat = next(iter(st.session_state.chats))
                st.rerun()

    st.divider()

    with st.expander("⚙️ Settings"):
        st.session_state.top_k = st.slider("Sources to retrieve", 1, 15, st.session_state.top_k)
        st.session_state.show_steps = st.toggle("Show agent / retrieval steps", value=st.session_state.show_steps)

        st.markdown("**Add documents to knowledge base**")
        uploaded = st.file_uploader(
            "Upload PDF / DOCX / PPTX / TXT", type=["pdf", "docx", "pptx", "txt"], label_visibility="collapsed"
        )
        if uploaded is not None and st.button("Ingest into knowledge base", use_container_width=True):
            try:
                httpx.post(f"{API_URL}/upload", files={"file": (uploaded.name, uploaded.getvalue())}, timeout=120)
                st.success(f"Queued “{uploaded.name}” for ingestion.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    current_chat = st.session_state.chats[st.session_state.current_chat]
    if current_chat["messages"]:
        st.download_button(
            "⬇️ Export this chat",
            data=export_markdown(current_chat),
            file_name=f"{truncate(current_chat['title'], 40)}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

current_chat = st.session_state.chats[st.session_state.current_chat]
messages = current_chat["messages"]

st.caption("Ask questions about your knowledge base — answers come with their sources.")

if not messages:
    st.info("Start a conversation below. Try: *“Summarize our latest advisory framework.”*")

for i, msg in enumerate(messages):
    avatar = "🧑" if msg["role"] == "user" else "💬"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("image_b64"):
                try:
                    st.image(base64.b64decode(msg["image_b64"]), use_container_width=True)
                except Exception:
                    pass

            if msg.get("agent_steps") and st.session_state.show_steps:
                with st.expander("🧠 Agent / retrieval steps"):
                    for step in msg["agent_steps"]:
                        st.write(step)

            if msg.get("citations"):
                with st.expander(f"📚 Sources ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.write(f"- {c.get('source_file', 'Source')}")

            rating = st.feedback("thumbs", key=f"fb_{i}")
            if rating is not None and not msg.get("_fb_sent"):
                send_feedback(msg["content"], "up" if rating == 1 else "down")
                msg["_fb_sent"] = True


prompt = st.chat_input("Ask about your documents…")

if prompt and prompt.strip():
    prompt = prompt.strip()
    if current_chat["title"] == "New chat":
        current_chat["title"] = truncate(prompt)
    messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="💬"):
        gen, result = ask_backend(prompt, st.session_state.session_id, st.session_state.top_k)
        full_text = st.write_stream(gen)

        if result["image_b64"]:
            try:
                st.image(base64.b64decode(result["image_b64"]), use_container_width=True)
            except Exception:
                pass

        if result["agent_steps"] and st.session_state.show_steps:
            with st.expander("🧠 Agent / retrieval steps"):
                for step in result["agent_steps"]:
                    st.write(step)

        if result["citations"]:
            with st.expander(f"📚 Sources ({len(result['citations'])})"):
                for c in result["citations"]:
                    st.write(f"- {c.get('source_file', 'Source')}")

    messages.append(
        {
            "role": "assistant",
            "content": full_text,
            "citations": result["citations"],
            "agent_steps": result["agent_steps"],
            "image_b64": result["image_b64"],
        }
    )
    st.rerun()