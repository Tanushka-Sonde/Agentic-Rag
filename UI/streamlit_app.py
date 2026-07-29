"""
ui/streamlit_app.py

A formal, minimal RAG chat interface built on native Streamlit components.

Design notes:
  - Tables/code render via plain st.markdown inside st.chat_message. Streamlit's
    built-in renderer already supports GFM tables and fenced code correctly —
    no custom markdown parser, so no parsing bugs.
  - Theme lives in .streamlit/config.toml, which this script writes next to
    itself on first run (see THEME section) so there's only one file to
    manage. Because Streamlit reads that file at server startup — before your
    script runs — the theme takes effect from the *next* launch onward.
  - Stop-generation is real, not cosmetic: the backend call runs on a
    background thread. A small st.fragment polls it ~10x/second so the Stop
    control stays responsive while tokens are streaming in, and stopping it
    both breaks the read loop and notifies the backend via POST /chat/stop.
  - No decorative avatars or sample prompts — default chat bubbles only.

Backend contract (adjust to match your FastAPI service):
  POST /chat/stream  {question, session_id, top_k} -> text/event-stream of
                      JSON lines: {"type":"token","text":...}
                                  {"type":"step","step":...}
                                  {"type":"done","citations":[...],
                                   "image_b64":..., "agent_steps":[...]}
                      (falls back to POST /chat if this 404s)
  POST /chat          {question, session_id}        -> single JSON blob
  POST /chat/stop     ?session_id=...                -> cancels in-flight call
  POST /feedback      {session_id, message, rating}  -> optional, best-effort
  POST /upload        files=...                        -> optional, best-effort
"""

import base64
import json
import os
import threading
import time
import uuid
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# THEME — self-writing config.toml
#
# Streamlit only reads theme colors/fonts from .streamlit/config.toml, and
# only at server startup — there is no in-script theme API. Rather than
# maintaining that as a separate file, this script writes it next to itself
# automatically the first time it runs. Because config loads before the
# script executes, the theme applies from the next `streamlit run` onward.
# ─────────────────────────────────────────────────────────────────────────

_THEME_TOML = """[theme]
base = "light"
primaryColor = "#2E3B4E"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F4F5F7"
textColor = "#1C2027"
font = "sans serif"
baseRadius = "small"

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
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon=":material/library_books:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Only style the one small custom element (the copy button) — using
# Streamlit's own theme CSS variables rather than hardcoded hex colors, so
# it stays legible whether the active theme is light, dark, or custom.
st.markdown(
    """
    <style>
    .copy-btn {
        border: 1px solid rgba(128,128,128,0.35);
        background: transparent;
        color: inherit;
        font-size: 0.95rem;
        line-height: 1;
        border-radius: 6px;
        padding: 4px 8px;
        cursor: pointer;
        font-family: inherit;
        opacity: 0.75;
        transition: opacity .12s, border-color .12s;
    }
    .copy-btn:hover { opacity: 1; border-color: var(--primary-color, #8A6D3B); }
    </style>
    """,
    unsafe_allow_html=True,
)

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/smart_toy:"


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────

def _new_chat_dict():
    return {"title": "New conversation", "messages": []}


if "chats" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.chats = {cid: _new_chat_dict()}
    st.session_state.current_chat = cid

# NOTE: there used to be a single global st.session_state.session_id sent
# to the backend for every request, regardless of which sidebar
# conversation was active. Since agent/memory.py's SessionMemory keys its
# sliding-window history by session_id, that meant "New conversation" (and
# switching between existing threads) never actually got a clean memory —
# every thread in the sidebar silently shared and polluted the same
# backend history. Each chat's own uuid (the dict key in st.session_state
# .chats) now doubles as its backend session_id, so every thread gets its
# own isolated memory, exactly like ChatGPT's per-conversation memory. A
# full page reload resets st.session_state entirely, so all of it — every
# thread, every session_id — is gone, matching "one session only" scope.

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

if "show_steps" not in st.session_state:
    st.session_state.show_steps = True

if "stream" not in st.session_state:
    st.session_state.stream = None


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
            json={"session_id": st.session_state.current_chat, "message": message_text, "rating": rating},
            timeout=5,
        )
    except Exception:
        pass


def copy_button(text: str, key: str):
    """A real Streamlit component (components.html) rather than an inline
    onclick attribute in st.markdown — inline event-handler attributes get
    silently blocked by Content-Security-Policy in a lot of environments,
    which is why the previous version did nothing. A <script> tag inside
    components.html actually executes, so this works reliably. Falls back
    to execCommand('copy') because navigator.clipboard requires a secure
    context (HTTPS or localhost).

    Uses an inline SVG icon (not a unicode glyph) since glyph rendering is
    inconsistent across fonts, and explicitly transparent backgrounds since
    components.html's iframe otherwise shows as an opaque box that doesn't
    match the page theme."""
    payload_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    html = f"""
    <style>
      html, body {{ margin:0; padding:0; background:transparent; }}
      .copy-btn {{
          display:inline-flex; align-items:center; justify-content:center;
          width:28px; height:28px; border-radius:6px;
          border:1px solid rgba(150,150,150,0.35); background:transparent;
          cursor:pointer; opacity:0.7; transition:opacity .12s, border-color .12s;
          color:#9AA0A6;
      }}
      .copy-btn:hover {{ opacity:1; border-color:#8A6D3B; color:#8A6D3B; }}
      .copy-btn svg {{ width:14px; height:14px; }}
    </style>
    <button id="copy-{key}" class="copy-btn" title="Copy response">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
    </button>
    <script>
      const btn = document.getElementById("copy-{key}");
      btn.addEventListener("click", function() {{
          const txt = decodeURIComponent(escape(window.atob("{payload_b64}")));
          function mark() {{
              btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
              setTimeout(() => {{
                  btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
              }}, 1200);
          }}
          function fallback() {{
              const ta = document.createElement("textarea");
              ta.value = txt;
              document.body.appendChild(ta);
              ta.select();
              try {{ document.execCommand("copy"); mark(); }} catch (e) {{}}
              document.body.removeChild(ta);
          }}
          if (navigator.clipboard) {{
              navigator.clipboard.writeText(txt).then(mark).catch(fallback);
          }} else {{
              fallback();
          }}
      }});
    </script>
    """
    components.html(html, height=32)


# ─────────────────────────────────────────────────────────────────────────
# BACKGROUND STREAMING WORKER — enables a real Stop control
# ─────────────────────────────────────────────────────────────────────────

class StreamState:
    def __init__(self, question: str, session_id: str, top_k: int):
        self.question = question
        self.session_id = session_id
        self.top_k = top_k

        self.lock = threading.Lock()
        self.text = ""
        self.citations = []
        self.agent_steps = []
        self.image_b64 = None
        self.done = False
        self.stopped = False
        self.error = None

        # Owning our own client (instead of one-off httpx.stream/post calls)
        # is what makes an instant hard-stop possible: closing this client
        # from the button-click thread aborts whatever socket read the
        # worker thread is blocked on right now — whether that's mid-token
        # or still waiting on retrieval before the first byte arrives, where
        # checking a flag inside the read loop wouldn't help because the
        # loop never gets a turn to check it.
        self.client = httpx.Client(timeout=300)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def request_stop(self):
        self.stop_event.set()
        try:
            self.client.close()  # forces any in-flight request to abort now
        except Exception:
            pass
        try:
            # best-effort: also tell the backend to cancel retrieval/LLM
            # work server-side, independent of the client-side abort above
            httpx.post(f"{API_URL}/chat/stop", params={"session_id": self.session_id}, timeout=5)
        except Exception:
            pass

    def snapshot(self):
        with self.lock:
            return dict(
                text=self.text,
                citations=list(self.citations),
                agent_steps=list(self.agent_steps),
                image_b64=self.image_b64,
                done=self.done,
                stopped=self.stopped,
                error=self.error,
            )

    def _run(self):
        try:
            self._run_stream()
        except Exception:
            if self.stop_event.is_set():
                # the exception here IS the forced abort from request_stop —
                # not a real backend failure — so don't fall back to a fresh
                # non-streaming call, just record that we stopped.
                with self.lock:
                    self.stopped = True
            else:
                try:
                    self._run_fallback()
                except Exception as e:
                    if self.stop_event.is_set():
                        with self.lock:
                            self.stopped = True
                    else:
                        with self.lock:
                            self.error = str(e)
        finally:
            with self.lock:
                self.done = True
            try:
                self.client.close()
            except Exception:
                pass

    def _run_stream(self):
        if self.stop_event.is_set():
            return
        with self.client.stream(
            "POST",
            f"{API_URL}/chat/stream",
            json={"question": self.question, "session_id": self.session_id, "top_k": self.top_k},
        ) as resp:
            if resp.status_code == 404:
                raise RuntimeError("no streaming endpoint")
            resp.raise_for_status()
            for line in resp.iter_lines():
                if self.stop_event.is_set():
                    with self.lock:
                        self.stopped = True
                    return
                if not line:
                    continue
                line = line[5:].strip() if line.startswith("data:") else line
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                etype = evt.get("type")
                with self.lock:
                    if etype == "token":
                        self.text += evt.get("text", "")
                    elif etype == "step":
                        self.agent_steps.append(evt.get("step", ""))
                    elif etype == "done":
                        self.citations = evt.get("citations", [])
                        self.agent_steps = evt.get("agent_steps", self.agent_steps)
                        self.image_b64 = evt.get("image_b64")

    def _run_fallback(self):
        if self.stop_event.is_set():
            return
        resp = self.client.post(
            f"{API_URL}/chat",
            json={"question": self.question, "session_id": self.session_id, "top_k": self.top_k},
        )
        resp.raise_for_status()
        data = resp.json()
        with self.lock:
            self.citations = data.get("citations", [])
            self.agent_steps = data.get("agent_steps", [])
            self.image_b64 = data.get("image_b64")
        answer = data.get("answer", "No answer returned.")
        for word in answer.split(" "):
            if self.stop_event.is_set():
                with self.lock:
                    self.stopped = True
                return
            with self.lock:
                self.text += word + " "
            time.sleep(0.012)
        with self.lock:
            self.text = self.text.strip()


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Knowledge Assistant**")
    st.caption("Enterprise document intelligence")

    if st.button("New conversation", use_container_width=True, type="primary"):
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
                ("• " if active else "") + chat_data["title"],
                key=f"open_{chat_id}",
                use_container_width=True,
            ):
                st.session_state.current_chat = chat_id
                st.rerun()
        with col2:
            if st.button("×", key=f"del_{chat_id}"):
                del st.session_state.chats[chat_id]
                # Free the backend's in-memory history for this thread too —
                # SessionMemory (agent/memory.py) never expires entries on
                # its own, so without this a deleted sidebar chat would keep
                # its history alive server-side indefinitely.
                try:
                    httpx.post(f"{API_URL}/chat/reset", params={"session_id": chat_id}, timeout=5)
                except Exception:
                    pass
                if not st.session_state.chats:
                    new_chat()
                elif st.session_state.current_chat == chat_id:
                    st.session_state.current_chat = next(iter(st.session_state.chats))
                st.rerun()

    st.divider()

    with st.expander("Settings"):
        st.session_state.top_k = st.slider("Sources to retrieve", 1, 15, st.session_state.top_k)
        st.session_state.show_steps = st.toggle("Show retrieval trace", value=st.session_state.show_steps)

        st.markdown("**Knowledge base**")
        uploaded = st.file_uploader(
            "Add a document", type=["pdf", "docx", "pptx", "txt"], label_visibility="collapsed"
        )
        if uploaded is not None and st.button("Ingest document", use_container_width=True):
            try:
                httpx.post(f"{API_URL}/upload", files={"file": (uploaded.name, uploaded.getvalue())}, timeout=120)
                st.success(f"Queued “{uploaded.name}” for ingestion.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    current_chat = st.session_state.chats[st.session_state.current_chat]
    if current_chat["messages"]:
        st.download_button(
            "Export conversation",
            data=export_markdown(current_chat),
            file_name=f"{truncate(current_chat['title'], 40)}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────────────────────────────────

current_chat = st.session_state.chats[st.session_state.current_chat]
messages = current_chat["messages"]

st.title("Knowledge Assistant")
st.caption("Answers are drawn from your organization's knowledge base, with sources cited.")
st.divider()


# ─────────────────────────────────────────────────────────────────────────
# RENDER STORED MESSAGES
# ─────────────────────────────────────────────────────────────────────────

for i, msg in enumerate(messages):
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("image_b64"):
                try:
                    # NOTE: Streamlit 1.39 does not support use_container_width
                    # on st.image() (only use_column_width) — passing it raised
                    # a TypeError that this try/except was silently swallowing,
                    # so a successfully-generated chart never appeared at all.
                    st.image(base64.b64decode(msg["image_b64"]), use_column_width=True)
                except Exception as e:
                    st.warning(f"Chart image failed to render: {e}")

            if msg.get("agent_steps") and st.session_state.show_steps:
                with st.expander("Retrieval trace"):
                    for step in msg["agent_steps"]:
                        st.write(step)

            if msg.get("citations"):
                with st.expander(f"Sources ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.write(f"- {c.get('source_file', 'Source')}")

            copy_button(msg["content"], key=f"copy_{i}")
            rating = st.feedback("thumbs", key=f"fb_{i}")
            if rating is not None and not msg.get("_fb_sent"):
                send_feedback(msg["content"], "up" if rating == 1 else "down")
                msg["_fb_sent"] = True


# ─────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask about your documents…", disabled=st.session_state.stream is not None)

if prompt and prompt.strip():
    prompt = prompt.strip()
    if current_chat["title"] == "New conversation":
        current_chat["title"] = truncate(prompt)
    messages.append({"role": "user", "content": prompt})
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# STREAMING FRAGMENT — polls the background worker, shows a live Stop
# control, and finalizes the message once generation completes or is
# interrupted.
# ─────────────────────────────────────────────────────────────────────────

@st.fragment(run_every=0.1)
def _streaming_fragment(state: "StreamState"):
    snap = state.snapshot()

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if not snap["text"] and not snap["done"]:
            st.markdown("Thinking…")
        else:
            st.markdown(snap["text"] + (" ▍" if not snap["done"] else ""))

        if not snap["done"]:
            if st.button("■", key="stop_btn", help="Stop generating"):
                state.request_stop()
                st.rerun(scope="fragment")

    if snap["done"]:
        final_text = snap["text"].strip()
        if snap["error"]:
            content = f"Something went wrong: {snap['error']}"
        elif snap["stopped"]:
            content = (final_text + "\n\n*Generation stopped.*") if final_text else "Generation stopped."
        else:
            content = final_text or "No answer returned."

        current_chat["messages"].append(
            {
                "role": "assistant",
                "content": content,
                "citations": snap["citations"],
                "agent_steps": snap["agent_steps"],
                "image_b64": snap["image_b64"],
            }
        )
        st.session_state.stream = None
        st.rerun()


if messages and messages[-1]["role"] == "user" and st.session_state.stream is None:
    st.session_state.stream = StreamState(
        messages[-1]["content"], st.session_state.current_chat, st.session_state.top_k
    )
    st.session_state.stream.start()

if st.session_state.stream is not None:
    _streaming_fragment(st.session_state.stream)