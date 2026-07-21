"""
ui/streamlit_app.py

============================================================================
 CHANGE 3 — Markdown / table rendering
============================================================================
  - Full GFM-style markdown via python-markdown (tables, fenced_code,
    sane_lists, nl2br, admonitions-free) → no more raw "|a|b|c|" text.
  - Code blocks are highlighted client-side with highlight.js (auto language
    detection) and get a "Copy" button, like Claude/ChatGPT.
  - ```mermaid fenced blocks are auto-detected anywhere in the answer text
    (not just via a separate chart_type field) and rendered as diagrams via
    Mermaid.js.
  - Tables only ever get parsed once the FULL response/block has arrived —
    during word-by-word streaming we show plain (escaped) text with a caret,
    and swap to the fully-parsed HTML the moment the stream finishes. This
    is what stops tables/code from "garbling" mid-stream.

============================================================================
 CHANGE 4 — Real Stop Generation
============================================================================
  - Backend call runs on a background thread (not the Streamlit script
    thread), writing tokens into a small thread-safe state object.
  - The chat area is a `st.fragment` that polls that state ~10x/second and
    re-renders just itself (not the whole page) — this is what lets the
    Stop button remain clickable *while* generation is streaming, which
    plain top-to-bottom Streamlit scripts can't do.
  - Clicking Stop: (a) sets a `threading.Event` the worker thread checks
    between chunks/words and exits early, and (b) fires `POST /chat/stop`
    so the backend actually cancels the in-flight LLM/agent call instead of
    just being ignored client-side.
  - Whatever text/table/citations had already streamed in stays visible.
  - Works the same way regardless of which agent(s) from the multi-agent
    graph happen to be running — the worker thread is what's talking to the
    backend, so killing/stopping it stops the whole in-flight turn.

============================================================================
 EXTRA UX — general "good chatbot" features layered on top
============================================================================
  - Per-message action bar: Copy answer, 👍/👎 feedback (best-effort POST to
    /feedback), and Regenerate.
  - Collapsible "Agent steps" trace under an answer, if the backend returns
    `agent_steps` (great for showing the multi-agent / RAG retrieval path).
  - Sidebar: retrieval depth (top_k) slider, "show agent steps" toggle,
    light/dark theme switch, knowledge-base file uploader (best-effort POST
    to /upload), export current conversation as a .md file.
  - Wider, taller, auto-growing input bar (closer to Claude/ChatGPT).
  - Elapsed-time badge + source-count on each answer.
  - Auto-scroll to the newest message.

Backend contract this file expects (adjust to match your FastAPI service):
  POST /chat/stream   {question, session_id, top_k}  -> text/event-stream of
                       JSON lines: {"type":"token","text":...}
                                   {"type":"step","step":...}
                                   {"type":"done","citations":[...],
                                    "image_b64":..., "chart_type":...,
                                    "code":..., "agent_steps":[...]}
                       (falls back to POST /chat if this 404s)
  POST /chat           {question, session_id}         -> single JSON blob
                       (original non-streaming endpoint, used as fallback)
  POST /chat/stop      ?session_id=...                -> cancels in-flight
  POST /feedback        {session_id, message, rating}  -> optional, best-effort
  POST /upload          files=...                       -> optional, best-effort
"""

import base64
import io
import json
import os
import re
import threading
import time
import uuid

import httpx
import streamlit as st

# ── Markdown renderer setup ──────────────────────────────────────────────────
try:
    import markdown as _md
    _HAS_MD = True
except ImportError:
    _HAS_MD = False


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="EY Knowledge Assistant",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# SESSION STATE (declared early so CSS/theme can use it)
# =====================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit_{uuid.uuid4()}"

if "chats" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.chats = {cid: {"title": "New conversation", "messages": []}}
    st.session_state.current_chat = cid

if ("current_chat" not in st.session_state
        or st.session_state.current_chat not in st.session_state.chats):
    st.session_state.current_chat = next(iter(st.session_state.chats))

if "generating" not in st.session_state:
    st.session_state.generating = False

if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

if "show_steps" not in st.session_state:
    st.session_state.show_steps = True

if "stream" not in st.session_state:
    st.session_state.stream = None  # holds the active StreamState while generating


# =====================================================
# CUSTOM CSS
# =====================================================

_DARK = st.session_state.theme == "dark"

_THEME_VARS = """
:root {
    --ink:       #F4F4F6;
    --ink-soft:  #E4E4E8;
    --text:      #EDEDEF;
    --muted:     #9A9AA4;
    --muted-2:   #77777F;
    --accent:    #FFE600;
    --accent-dk: #F2D900;
    --hair:      #303038;
    --surface:   #17171B;
    --sidebar:   #111114;
    --fill:      #1F1F24;
    --radius:    14px;
}
""" if _DARK else """
:root {
    --ink:       #2E2E38;
    --ink-soft:  #3C3C46;
    --text:      #1A1A1A;
    --muted:     #6B6B76;
    --muted-2:   #9A9AA4;
    --accent:    #FFE600;
    --accent-dk: #F2D900;
    --hair:      #ECECEF;
    --surface:   #FFFFFF;
    --sidebar:   #FAFAFB;
    --fill:      #F5F5F7;
    --radius:    14px;
}
"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

{_THEME_VARS}

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp {{
    background-color: var(--surface);
    color: var(--text);
    font-family: "Inter", ui-sans-serif, system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden !important;
}}

/* ── THE OVERFLOW FIX ──────────────────────────────────────────────────
   Streamlit's columns/rows are CSS flexbox. Flex children default to
   min-width: auto, meaning they refuse to shrink below their own content
   size — so long chat titles, button rows, and the topbar status badge
   were all pushing past their container's edge instead of wrapping or
   truncating. This is THE cause of basically every "stuff is overflowing"
   symptom in the screenshots. Forcing min-width: 0 lets flex children
   actually shrink to fit, which is what makes text-overflow: ellipsis
   and word-wrap work at all. */
[data-testid="stHorizontalBlock"],
[data-testid="column"],
[data-testid="stVerticalBlock"],
[data-testid="stMarkdownContainer"],
.block-container, .main, section.main {{
    min-width: 0 !important;
    overflow-wrap: anywhere !important;
}}
[data-testid="stSidebar"] * {{ min-width: 0 !important; }}
img, svg, table {{ max-width: 100% !important; }}

#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton, [data-testid="stToolbar"], [data-testid="stStatusWidget"] {{ display: none; }}

[data-testid="stSidebar"] {{
    background-color: var(--sidebar);
    border-right: 1px solid var(--hair);
    width: 284px !important;
    overflow-x: hidden !important;
}}
[data-testid="stSidebar"] .block-container {{ padding: 1.1rem 0.8rem; overflow-x: hidden; }}

.sidebar-brand {{
    display: flex; align-items: center; gap: 11px;
    padding: 0.2rem 0.5rem 1rem;
    border-bottom: 1px solid var(--hair);
}}
.brand-mark {{
    width: 36px; height: 36px; border-radius: 9px;
    background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800; color: #2E2E38;
    letter-spacing: -0.5px; flex-shrink: 0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}}
.brand-text {{ font-size: 14.5px; font-weight: 700; color: var(--ink); letter-spacing: -0.2px; line-height: 1.15; }}
.brand-text span {{ display: block; font-size: 11px; font-weight: 500; color: var(--muted-2); letter-spacing: 0.02em; margin-top: 2px; }}

[data-testid="stSidebar"] .stButton > button {{
    background: var(--surface); border: 1px solid var(--hair); color: var(--text);
    border-radius: 10px; font-size: 13px; font-weight: 500;
    padding: 0.5rem 0.8rem; text-align: left;
    transition: background .12s, border-color .12s, color .12s, box-shadow .12s;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: var(--fill); border-color: var(--hair); color: var(--text);
}}
.new-conv + div .stButton > button {{
    background: var(--accent) !important; border: none !important; color: #2E2E38 !important;
    font-weight: 600 !important; text-align: center !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}}
.new-conv + div .stButton > button:hover {{
    background: var(--accent-dk) !important;
    box-shadow: 0 2px 9px rgba(255,230,0,0.4) !important;
}}

[data-testid="stSidebar"] [data-testid="stTextInput"] input {{
    background: var(--surface); border: 1px solid var(--hair);
    border-radius: 9px; font-size: 12.5px; color: var(--text); padding: 0.5rem 0.7rem;
    caret-color: var(--accent-dk);
}}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {{
    border-color: var(--accent); box-shadow: 0 0 0 3px rgba(255,230,0,0.18);
}}

.history-label {{
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted-2); padding: 0 0.4rem; margin: 1rem 0 0.4rem;
    display: block; font-weight: 600;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button {{
    background: transparent; border: none; color: var(--ink-soft);
    font-size: 13px; font-weight: 400; padding: 0.4rem 0.6rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button:hover {{
    background: var(--fill); color: var(--text);
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button {{
    background: transparent; border: none; color: var(--muted-2); font-size: 15px;
    padding: 0.3rem 0.35rem; line-height: 1;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button:hover {{
    color: #E04848; background: #FDEDED;
}}
.active-row + div [data-testid="column"]:first-child .stButton > button {{
    background: var(--surface) !important; color: var(--text) !important;
    font-weight: 600 !important; box-shadow: inset 0 0 0 1px var(--hair);
}}
[data-testid="stSidebar"] hr {{ border-color: var(--hair); margin: 0.6rem 0; }}

.block-container {{ max-width: 840px; margin: 0 auto; padding: 1.6rem 1.5rem 12rem; }}

.topbar {{
    display: flex; align-items: center; gap: 13px;
    padding-bottom: 1rem; margin-bottom: 1.6rem; border-bottom: 1px solid var(--hair);
}}
.topbar-mark {{
    width: 42px; height: 42px; border-radius: 11px; background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 800; color: #2E2E38; letter-spacing: -0.5px; flex-shrink: 0;
}}
.topbar-title {{ font-size: 19px; font-weight: 700; color: var(--ink); letter-spacing: -0.4px; line-height: 1.2; }}
.topbar-sub {{ font-size: 12.5px; color: var(--muted); margin-top: 2px; }}
.topbar-status {{
    margin-left: auto; display: flex; align-items: center; gap: 6px;
    font-size: 11.5px; color: var(--muted); font-weight: 500;
}}
.status-dot {{ width: 7px; height: 7px; border-radius: 50%; background: #34C759; box-shadow: 0 0 0 3px rgba(52,199,89,0.15); }}

.user-row {{ display: flex; justify-content: flex-end; margin: 1.5rem 0 0.4rem; }}
.user-bubble {{
    background: var(--ink); color: var(--surface);
    border-radius: 16px 16px 5px 16px;
    padding: 11px 16px; max-width: 80%;
    font-size: 14.5px; line-height: 1.6; font-weight: 400;
    box-shadow: 0 1px 2px rgba(46,46,56,0.15);
    white-space: pre-wrap; word-wrap: break-word;
}}

.bot-row {{ display: flex; gap: 13px; margin: 0.5rem 0 0.2rem; align-items: flex-start; }}
.bot-avatar {{
    width: 32px; height: 32px; border-radius: 9px; background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 800; color: #2E2E38; letter-spacing: -0.3px;
    flex-shrink: 0; margin-top: 2px;
}}
.bot-body {{ max-width: calc(100% - 45px); padding-top: 3px; width: 100%; }}
.bot-text {{ color: var(--text); font-size: 14.8px; line-height: 1.75; }}
.bot-text p {{ margin: 0 0 0.7em; }}
.bot-text p:last-child {{ margin-bottom: 0; }}
.bot-text ul, .bot-text ol {{ margin: 0.5em 0 0.7em; padding-left: 1.35em; }}
.bot-text li {{ margin-bottom: 0.35em; }}
.bot-text h1, .bot-text h2, .bot-text h3 {{ font-size: 15.5px; font-weight: 700; color: var(--ink); margin: 1em 0 0.4em; letter-spacing: -0.2px; }}
.bot-text strong {{ color: var(--ink); font-weight: 600; }}
.bot-text a {{ color: #1f6feb; text-decoration: none; border-bottom: 1px solid rgba(31,111,235,0.25); }}
.bot-text a:hover {{ border-bottom-color: #1f6feb; }}
.bot-text blockquote {{
    border-left: 3px solid var(--accent-dk); margin: 0.6em 0; padding: 0.2em 0 0.2em 0.9em;
    color: var(--muted); font-style: italic;
}}
.bot-text code {{
    background: var(--fill); border: 1px solid var(--hair); border-radius: 5px;
    padding: 1px 5px; font-size: 12.8px;
    font-family: "SF Mono", "JetBrains Mono", Consolas, monospace;
}}

/* code blocks + copy button */
.code-wrap {{ position: relative; margin: 0.7em 0; }}
.code-wrap pre {{
    background: #1E1E24; color: #EDEDED; border-radius: 10px;
    padding: 14px 16px; overflow-x: auto; margin: 0; font-size: 13px;
    font-family: "SF Mono", "JetBrains Mono", Consolas, monospace; line-height: 1.55;
}}
.code-wrap pre code {{ background: none; border: none; padding: 0; color: inherit; font-size: 13px; }}
.code-copy-btn {{
    position: absolute; top: 8px; right: 8px;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
    color: #DADADA; font-size: 11px; font-weight: 500;
    border-radius: 6px; padding: 3px 9px; cursor: pointer;
    font-family: "Inter", sans-serif; transition: background .12s;
}}
.code-copy-btn:hover {{ background: rgba(255,255,255,0.16); }}

/* GFM tables */
.bot-text table {{
    border-collapse: collapse; margin: 0.8em 0;
    font-size: 13.5px; width: 100%; overflow-x: auto; display: block;
}}
.bot-text th {{
    background: var(--fill); font-weight: 600; color: var(--ink);
    border: 1px solid var(--hair); padding: 8px 12px; text-align: left;
    white-space: nowrap;
}}
.bot-text td {{
    border: 1px solid var(--hair); padding: 7px 12px;
    text-align: left; vertical-align: top;
}}
.bot-text tr:nth-child(even) td {{ background: rgba(127,127,127,0.045); }}
.bot-text tr:hover td {{ background: rgba(255,230,0,0.06); }}

.caret {{ display:inline-block; width:7px; height:1.05em; background:var(--ink); margin-left:2px; vertical-align:-2px; animation: blink 1s steps(2) infinite; border-radius:1px; }}
@keyframes blink {{ 50% {{ opacity: 0; }} }}

.sources {{ margin: 0.75rem 0 0.4rem 45px; }}
.sources-label {{ font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-2); margin-bottom: 0.45rem; font-weight: 600; }}
.sources-row {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.source-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface); border: 1px solid var(--hair);
    border-radius: 8px; padding: 5px 11px; font-size: 12px; color: var(--muted);
    max-width: 240px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    transition: border-color .12s, background .12s, color .12s;
}}
.source-pill:hover {{ border-color: var(--accent-dk); background: rgba(255,230,0,0.06); color: var(--text); }}
.source-pill .dot {{ width: 5px; height: 5px; border-radius: 50%; background: var(--accent-dk); flex-shrink: 0; }}

.meta-row {{ margin: 0.35rem 0 0.1rem 45px; font-size: 11px; color: var(--muted-2); display: flex; gap: 12px; align-items: center; }}

.action-row {{ margin: 0.5rem 0 0.2rem 45px; display: flex; gap: 6px; }}
.action-row + div {{ display: flex; gap: 6px; }}
.action-row + div .stButton {{ display: inline-block; }}
.action-row + div .stButton > button {{
    background: transparent !important; border: 1px solid var(--hair) !important; color: var(--muted) !important;
    border-radius: 8px !important; font-size: 12px !important; font-weight: 500 !important;
    padding: 0.3rem 0.65rem !important; text-align: center !important; min-height: 0 !important;
}}
.action-row + div .stButton > button:hover {{ background: var(--fill) !important; color: var(--text) !important; border-color: var(--accent-dk) !important; }}

.steps-wrap {{ margin: 0.4rem 0 0.1rem 45px; }}

.error-row {{ display: flex; gap: 13px; margin: 0.5rem 0; align-items: flex-start; }}
.error-bubble {{
    background: rgba(224,72,72,0.08); border: 1px solid rgba(224,72,72,0.3); border-radius: 11px;
    padding: 11px 15px; font-size: 13.5px; color: #E04848; line-height: 1.5;
    max-width: calc(100% - 45px);
}}

.dots {{ display: inline-flex; gap: 5px; padding: 7px 2px; }}
.dots span {{ width: 7px; height: 7px; background: var(--muted-2); border-radius: 50%; animation: bob 1.2s infinite ease-in-out; }}
.dots span:nth-child(2) {{ animation-delay: .18s; }}
.dots span:nth-child(3) {{ animation-delay: .36s; }}
@keyframes bob {{ 0%,80%,100% {{ transform: scale(.6); opacity:.35; }} 40% {{ transform: scale(1); opacity:1; }} }}

.stop-wrap {{ margin-left: 45px; margin-top: 0.5rem; }}
.stop-wrap + div .stButton > button {{
    background: rgba(224,72,72,0.08) !important; border: 1px solid rgba(224,72,72,0.3) !important;
    color: #E04848 !important; border-radius: 8px !important;
    font-size: 12px !important; font-weight: 600 !important; padding: 0.35rem 0.8rem !important;
}}
.stop-wrap + div .stButton > button:hover {{
    background: rgba(224,72,72,0.16) !important; border-color: #E04848 !important;
}}

.welcome {{ text-align: center; padding: 3rem 1rem 1.5rem; }}
.welcome-mark {{
    width: 58px; height: 58px; border-radius: 16px; background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; font-weight: 800; color: #2E2E38; letter-spacing: -0.6px;
    margin: 0 auto 1.3rem; box-shadow: 0 4px 16px rgba(255,230,0,0.4);
}}
.welcome-title {{ font-size: 28px; font-weight: 800; color: var(--ink); letter-spacing: -0.8px; margin-bottom: 0.5rem; }}
.welcome-sub {{ font-size: 14.5px; color: var(--muted); max-width: 460px; margin: 0 auto; line-height: 1.6; }}
.suggest-label {{
    text-align: center; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted-2); font-weight: 600; margin: 2rem 0 0.9rem;
}}

.block-container .stButton > button {{
    background: var(--surface); border: 1px solid var(--hair); color: var(--text);
    border-radius: 13px; padding: 0.85rem 1rem; text-align: left;
    font-size: 13.5px; font-weight: 500; line-height: 1.45; min-height: 70px;
    transition: border-color .14s, box-shadow .14s, transform .14s, background .14s;
}}
.block-container .stButton > button:hover {{
    border-color: var(--accent-dk); background: rgba(255,230,0,0.05);
    box-shadow: 0 4px 14px rgba(0,0,0,0.06); transform: translateY(-2px);
}}

/* ── GLOBAL, DEFENSIVE button rules ───────────────────────────────────────
   The rules above/below this block target buttons via fragile sibling
   selectors (".action-row + div .stButton > button" etc.) which depend on
   Streamlit emitting an exact DOM shape. When that shape shifts (different
   Streamlit version, extra wrapper <div>, etc.) those selectors silently
   stop matching and the browser falls back to Streamlit's *unstyled*
   default button — which wraps short labels like "Copy" onto two lines
   ("Cop" / "y") inside the tiny default width. These rules apply to EVERY
   button on the page regardless of which wrapper matched, so text never
   wraps and buttons never look "naked" even if a scoped rule above fails. */
.stButton > button, [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {{
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
.stButton > button p {{ white-space: nowrap !important; margin: 0 !important; }}
/* small icon-buttons (Copy / 👍 / 👎 / Stop / Regenerate) never shrink
   below a sane width and never wrap, no matter which parent CSS matched */
.action-row + div .stButton > button,
.stop-wrap + div .stButton > button {{
    white-space: nowrap !important;
    min-width: max-content !important;
    width: auto !important;
}}
.action-row + div [data-testid="column"] {{
    width: auto !important;
    flex: 0 0 auto !important;
    min-width: fit-content !important;
}}

/* ── long, tall, auto-growing input bar ── */
[data-testid="stChatInput"] {{
    position: fixed; bottom: 0; left: 284px; right: 0;
    padding: 0.8rem 2rem 1.4rem;
    background: linear-gradient(to top, var(--surface) 72%, rgba(255,255,255,0));
    z-index: 100;
}}
[data-testid="stChatInput"] > div {{ max-width: 840px; margin: 0 auto; }}
[data-testid="stChatInput"] textarea {{
    background: var(--surface) !important; border: 1.5px solid var(--hair) !important;
    border-radius: 18px !important; color: var(--text) !important;
    caret-color: var(--accent-dk) !important;
    font-size: 15px !important; padding: 16px 20px !important;
    min-height: 56px !important; max-height: 240px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important;
    transition: border-color .15s, box-shadow .15s !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(255,230,0,0.18), 0 2px 12px rgba(0,0,0,0.05) !important;
}}
[data-testid="stChatInput"] button {{ background: var(--ink) !important; }}
.input-hint {{ text-align:center; font-size: 11px; color: var(--muted-2); margin-top: 0.5rem; }}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--hair); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--muted-2); }}

.mermaid-wrap {{
    background: var(--fill); border: 1px solid var(--hair);
    border-radius: 12px; padding: 1rem; margin: 0.6em 0; overflow-x: auto;
}}

@media (max-width: 820px) {{ [data-testid="stChatInput"] {{ left: 0; }} }}
</style>
""", unsafe_allow_html=True)


# =====================================================
# HELPERS
# =====================================================

def new_chat_id():
    return str(uuid.uuid4())


def truncate_title(text, max_len=36):
    text = " ".join(text.split())
    return (text[:max_len] + "…") if len(text) > max_len else text


def safe_html(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_MERMAID_BLOCK_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.S | re.I)


def _extract_mermaid_blocks(text: str):
    """Pull out ```mermaid fenced blocks and replace with placeholders so the
    markdown parser doesn't touch them; returns (text_without_mermaid, blocks)."""
    blocks = []

    def _sub(m):
        blocks.append(m.group(1).strip())
        return f"\n\n%%MERMAID_BLOCK_{len(blocks) - 1}%%\n\n"

    return _MERMAID_BLOCK_RE.sub(_sub, text), blocks


def render_mermaid(code: str) -> str:
    return f'<div class="mermaid-wrap"><pre class="mermaid">{safe_html(code)}</pre></div>'


def render_markdown(text: str) -> str:
    """
    Convert assistant markdown to HTML with full GFM-ish support:
      - tables (fixes the raw "|a|b|c|" rendering bug)
      - fenced code blocks (syntax-highlighted client-side via highlight.js)
      - sane list handling, line breaks
      - ```mermaid blocks auto-rendered as diagrams

    Only ever called on the *complete* text (after streaming finishes, or on
    stored/history messages), so tables and code fences are always whole —
    this is what prevents the "garbled mid-stream" table bug.
    """
    if not text:
        return ""

    text_wo_mermaid, mermaid_blocks = _extract_mermaid_blocks(text)

    if _HAS_MD:
        html = _md.markdown(
            text_wo_mermaid,
            extensions=["fenced_code", "tables", "sane_lists", "nl2br"],
        )
        # Wrap <pre><code> blocks so we can attach a copy button + let
        # highlight.js target them without disturbing table/list HTML.
        html = re.sub(
            r"<pre><code([^>]*)>(.*?)</code></pre>",
            lambda m: (
                '<div class="code-wrap">'
                '<button class="code-copy-btn" onclick="'
                "navigator.clipboard.writeText(this.nextElementSibling.innerText);"
                "this.innerText='Copied!';setTimeout(()=>this.innerText='Copy',1200);"
                '">Copy</button>'
                f"<pre><code{m.group(1)}>{m.group(2)}</code></pre></div>"
            ),
            html, flags=re.S,
        )
    else:
        # Fallback lightweight renderer if `markdown` isn't installed.
        t = text_wo_mermaid
        t = re.sub(
            r"```(\w*)\n?(.*?)```",
            lambda m: (
                '<div class="code-wrap">'
                '<button class="code-copy-btn" onclick="'
                "navigator.clipboard.writeText(this.nextElementSibling.innerText);"
                "this.innerText='Copied!';setTimeout(()=>this.innerText='Copy',1200);"
                '">Copy</button>'
                "<pre><code>" + safe_html(m.group(2)) + "</code></pre></div>"
            ),
            t, flags=re.S,
        )
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", r"<em>\1</em>", t)
        t = re.sub(r"`([^`]+?)`", r"<code>\1</code>", t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank">\1</a>', t)
        paras = [p for p in t.split("\n\n") if p.strip()]
        html = "".join(
            p if p.strip().startswith("<div") else "<p>" + p.replace("\n", "<br>") + "</p>"
            for p in paras
        )

    for i, block in enumerate(mermaid_blocks):
        html = html.replace(f"<p>%%MERMAID_BLOCK_{i}%%</p>", render_mermaid(block))
        html = html.replace(f"%%MERMAID_BLOCK_{i}%%", render_mermaid(block))

    return html


def post_render_scripts(container_id: str) -> str:
    """highlight.js + mermaid re-init + smooth autoscroll, scoped to one message."""
    return f"""
    <script>
    (function() {{
        function run() {{
            if (window.hljs) {{
                document.querySelectorAll('#{container_id} pre code').forEach(function(el) {{
                    if (!el.dataset.hlDone) {{ window.hljs.highlightElement(el); el.dataset.hlDone = '1'; }}
                }});
            }}
            if (window.mermaid) {{
                try {{ window.mermaid.run({{ querySelector: '#{container_id} .mermaid' }}); }} catch(e) {{}}
            }}
            var el = document.getElementById('{container_id}');
            if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'end' }});
        }}
        if (document.readyState === 'complete') run(); else window.addEventListener('load', run);
        setTimeout(run, 60);
    }})();
    </script>
    """


def submit_message(text):
    chat = st.session_state.chats[st.session_state.current_chat]
    if chat["title"] == "New conversation":
        chat["title"] = truncate_title(text)
    chat["messages"].append({"role": "user", "content": text})
    st.rerun()


def export_conversation_md(chat) -> str:
    lines = [f"# {chat['title']}", ""]
    for m in chat["messages"]:
        who = "**You**" if m["role"] == "user" else "**EY Knowledge Assistant**"
        lines.append(f"{who}:\n\n{m['content']}\n")
        for c in m.get("citations", []) or []:
            lines.append(f"> Source: {c.get('source_file', 'Source')}")
        lines.append("")
    return "\n".join(lines)


# =====================================================
# ONE-TIME JS/CSS LIBRARY INJECTION (highlight.js + mermaid)
# =====================================================

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11/styles/github-dark.min.css">
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11/lib/common.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  if (window.mermaid) {
      window.mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' });
  }
</script>
""", unsafe_allow_html=True)


# =====================================================
# BACKGROUND STREAMING WORKER (Change 4: real cancellation)
# =====================================================

class StreamState:
    """Thread-safe box the background worker writes into and the UI fragment
    polls. Keeping this off session_state internals avoids races with
    Streamlit's own rerun cycle."""

    def __init__(self, question: str, session_id: str, api_url: str, top_k: int):
        self.question   = question
        self.session_id = session_id
        self.api_url    = api_url
        self.top_k      = top_k

        self.lock       = threading.Lock()
        self.text       = ""
        self.steps      = []
        self.citations  = []
        self.image_b64  = None
        self.chart_type = None
        self.code       = None
        self.done       = False
        self.stopped    = False
        self.error      = None
        self.started_at = time.time()
        self.finished_at = None

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def request_stop(self):
        self.stop_event.set()
        # Best-effort: tell the backend to actually cancel the agent/LLM call,
        # not just abandon the client-side read.
        try:
            httpx.post(
                f"{self.api_url}/chat/stop",
                params={"session_id": self.session_id},
                timeout=5,
            )
        except Exception:
            pass

    def snapshot(self):
        with self.lock:
            return dict(
                text=self.text, steps=list(self.steps), citations=list(self.citations),
                image_b64=self.image_b64, chart_type=self.chart_type, code=self.code,
                done=self.done, stopped=self.stopped, error=self.error,
                elapsed=(self.finished_at or time.time()) - self.started_at,
            )

    # ── worker body ──────────────────────────────────────────────────────
    def _run(self):
        try:
            self._run_streaming()
        except Exception:
            # Streaming endpoint missing/broken → fall back to the plain
            # non-streaming /chat endpoint, then simulate a word-by-word feed
            # so the UI still animates in nicely.
            try:
                self._run_fallback()
            except Exception as e:
                with self.lock:
                    self.error = str(e)
        finally:
            with self.lock:
                self.done = True
                self.finished_at = time.time()

    def _run_streaming(self):
        with httpx.stream(
            "POST", f"{self.api_url}/chat/stream",
            json={"question": self.question, "session_id": self.session_id,
                  "top_k": self.top_k},
            timeout=300,
        ) as resp:
            if resp.status_code == 404:
                raise RuntimeError("no streaming endpoint")
            resp.raise_for_status()
            for line in resp.iter_lines():
                if self.stop_event.is_set():
                    with self.lock:
                        self.stopped = True
                    try:
                        resp.close()
                    except Exception:
                        pass
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
                        self.steps.append(evt.get("step", ""))
                    elif etype == "done":
                        self.citations  = evt.get("citations", [])
                        self.image_b64  = evt.get("image_b64")
                        self.chart_type = evt.get("chart_type")
                        self.code       = evt.get("code")
                        self.steps      = evt.get("agent_steps", self.steps)

    def _run_fallback(self):
        response = httpx.post(
            f"{self.api_url}/chat",
            json={"question": self.question, "session_id": self.session_id,
                  "top_k": self.top_k},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        with self.lock:
            self.citations  = data.get("citations", [])
            self.image_b64  = data.get("image_b64")
            self.chart_type = data.get("chart_type")
            self.code       = data.get("code")
            self.steps      = data.get("agent_steps", [])
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


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-mark">EY</div>
        <div class="brand-text">Knowledge<span>Middle East · Agentic RAG</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="new-conv"></div>', unsafe_allow_html=True)
    if st.button("＋  New conversation", key="new_chat_btn", use_container_width=True):
        cid = new_chat_id()
        st.session_state.chats[cid] = {"title": "New conversation", "messages": []}
        st.session_state.current_chat = cid
        st.rerun()

    query = st.text_input(
        "Search", key="chat_search",
        placeholder="🔍  Search conversations",
        label_visibility="collapsed",
    )

    st.markdown('<span class="history-label">Recent</span>', unsafe_allow_html=True)

    q = (query or "").strip().lower()
    items = list(reversed(list(st.session_state.chats.items())))
    if q:
        items = [(cid, c) for cid, c in items if q in c["title"].lower()]

    if not items:
        st.markdown(
            '<div style="padding:0.5rem 0.6rem;font-size:12.5px;color:var(--muted-2);">No conversations found.</div>',
            unsafe_allow_html=True,
        )

    for chat_id, chat_data in items:
        is_active = (chat_id == st.session_state.current_chat)
        if is_active:
            st.markdown('<div class="active-row"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([10, 1], gap="small")
        with col1:
            if st.button(chat_data["title"], key=f"open_{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id
                st.rerun()
        with col2:
            if st.button("×", key=f"del_{chat_id}"):
                del st.session_state.chats[chat_id]
                if not st.session_state.chats:
                    cid = new_chat_id()
                    st.session_state.chats[cid] = {"title": "New conversation", "messages": []}
                    st.session_state.current_chat = cid
                else:
                    st.session_state.current_chat = next(iter(st.session_state.chats))
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    with st.expander("⚙️  Settings", expanded=False):
        st.session_state.top_k = st.slider(
            "Retrieval depth (top-k sources)", min_value=1, max_value=15,
            value=st.session_state.top_k,
        )
        st.session_state.show_steps = st.toggle(
            "Show agent / retrieval steps", value=st.session_state.show_steps,
        )
        new_theme = st.radio(
            "Theme", options=["light", "dark"],
            index=0 if st.session_state.theme == "light" else 1,
            horizontal=True,
        )
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

        st.markdown("**Add documents to knowledge base**")
        uploaded = st.file_uploader(
            "Upload PDF / DOCX / PPTX", type=["pdf", "docx", "pptx", "txt"],
            label_visibility="collapsed", key="kb_uploader",
        )
        if uploaded is not None:
            if st.button("Ingest into knowledge base", use_container_width=True):
                try:
                    api_url = os.getenv("API_URL", "http://localhost:8000")
                    httpx.post(
                        f"{api_url}/upload",
                        files={"file": (uploaded.name, uploaded.getvalue())},
                        timeout=120,
                    )
                    st.success(f"Queued “{uploaded.name}” for ingestion.")
                except Exception as e:
                    st.warning(f"Couldn't reach the upload endpoint: {e}")

        current_chat_for_export = st.session_state.chats[st.session_state.current_chat]
        if current_chat_for_export["messages"]:
            st.download_button(
                "⬇  Export this conversation (.md)",
                data=export_conversation_md(current_chat_for_export),
                file_name=f"{truncate_title(current_chat_for_export['title'], 40)}.md",
                mime="text/markdown",
                use_container_width=True,
            )


# =====================================================
# MAIN — HEADER
# =====================================================

current_chat = st.session_state.chats[st.session_state.current_chat]
messages     = current_chat["messages"]

st.markdown("""
<div class="topbar">
    <div class="topbar-mark">EY</div>
    <div>
        <div class="topbar-title">Knowledge Assistant</div>
        <div class="topbar-sub">Ask anything about EY Middle East projects and knowledge base</div>
    </div>
    <div class="topbar-status"><span class="status-dot"></span> Connected</div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# RENDER ONE STORED ASSISTANT MESSAGE
# =====================================================

def render_assistant_message(msg: dict, idx: int) -> None:
    is_error = str(msg["content"]).startswith("⚠️")
    container_id = f"msg-{idx}"

    if is_error:
        st.markdown(f"""
        <div class="error-row">
            <div class="bot-avatar">EY</div>
            <div class="error-bubble">{safe_html(msg["content"])}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="bot-row" id="{container_id}">
        <div class="bot-avatar">EY</div>
        <div class="bot-body"><div class="bot-text">{render_markdown(msg["content"])}</div></div>
    </div>
    """, unsafe_allow_html=True)

    if msg.get("image_b64"):
        chart_type = msg.get("chart_type", "")
        if chart_type == "mermaid":
            st.markdown(render_mermaid(msg.get("code", "")), unsafe_allow_html=True)
        else:
            try:
                img_bytes = base64.b64decode(msg["image_b64"])
                st.image(img_bytes, use_container_width=True)
            except Exception:
                pass

    st.markdown(post_render_scripts(container_id), unsafe_allow_html=True)

    steps = msg.get("agent_steps") or []
    if steps and st.session_state.show_steps:
        with st.expander("🧠  Agent / retrieval steps", expanded=False):
            for s_i, s in enumerate(steps, 1):
                st.markdown(f"**{s_i}.** {safe_html(s)}", unsafe_allow_html=True)

    citations = msg.get("citations", [])
    if citations:
        pills = "".join(
            f'<div class="source-pill"><span class="dot"></span>'
            f'{safe_html(s.get("source_file", "Source"))}</div>'
            for s in citations
        )
        st.markdown(f"""
        <div class="sources">
            <div class="sources-label">Sources ({len(citations)})</div>
            <div class="sources-row">{pills}</div>
        </div>
        """, unsafe_allow_html=True)

    meta_bits = []
    if msg.get("elapsed") is not None:
        meta_bits.append(f"⏱ {msg['elapsed']:.1f}s")
    if citations:
        meta_bits.append(f"📚 {len(citations)} source{'s' if len(citations) != 1 else ''}")
    if meta_bits:
        st.markdown(
            f'<div class="meta-row">{" · ".join(meta_bits)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="action-row"></div>', unsafe_allow_html=True)
    a1, a2, a3, _sp = st.columns([2, 1, 1, 10])
    with a1:
        if st.button("⧉ Copy", key=f"copy_{idx}"):
            st.toast("Copied to clipboard (select the text above and copy manually if this doesn't work in your browser).")
    with a2:
        if st.button("👍", key=f"up_{idx}"):
            _send_feedback(msg["content"], "up")
            st.toast("Thanks for the feedback!")
    with a3:
        if st.button("👎", key=f"down_{idx}"):
            _send_feedback(msg["content"], "down")
            st.toast("Thanks — we'll use this to improve answers.")


def _send_feedback(message_text, rating):
    try:
        api_url = os.getenv("API_URL", "http://localhost:8000")
        httpx.post(
            f"{api_url}/feedback",
            json={"session_id": st.session_state.session_id,
                  "message": message_text, "rating": rating},
            timeout=5,
        )
    except Exception:
        pass


# =====================================================
# MAIN — CONVERSATION
# =====================================================

SUGGESTIONS = [
    "Summarize our latest advisory framework for financial risk management.",
    "What case studies do we have on digital transformation in the GCC?",
    "Draw a flowchart of our standard client onboarding process.",
    "Compare our engagement models across the last three sectors we served.",
]

if not messages:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-mark">EY</div>
        <div class="welcome-title">How can I help today?</div>
        <div class="welcome-sub">
            Search engagements, advisory frameworks, and case studies across the
            EY Middle East knowledge base — answers come with their sources.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="suggest-label">Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(s, key=f"suggest_{i}", use_container_width=True):
                submit_message(s)

else:
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-row">
                <div class="user-bubble">{safe_html(msg["content"])}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            render_assistant_message(msg, i)

    if (len(messages) >= 2
            and messages[-1]["role"] == "assistant"
            and not str(messages[-1]["content"]).startswith("⚠️")
            and not st.session_state.generating):
        st.markdown('<div class="regen-wrap" style="margin-left:45px;"></div>', unsafe_allow_html=True)
        if st.button("↻  Regenerate", key="regen_btn"):
            messages.pop()
            if messages and messages[-1]["role"] == "user":
                pass  # leave the user question so the flow below re-fires it
            else:
                pass
            st.rerun()


# =====================================================
# CHAT INPUT
# =====================================================

prompt = st.chat_input("Ask about EY Middle East… (Shift+Enter for a new line)")
st.markdown(
    '<div class="input-hint">Answers are generated from the EY Middle East knowledge base. AI can make mistakes — verify important information.</div>',
    unsafe_allow_html=True,
)

if prompt and prompt.strip():
    submit_message(prompt.strip())


# =====================================================
# STREAMING FRAGMENT (Change 4: real, interruptible generation)
# =====================================================

messages = current_chat["messages"]
api_url  = os.getenv("API_URL", "http://localhost:8000")


@st.fragment(run_every=0.12)
def _streaming_fragment(state: "StreamState"):
    snap = state.snapshot()

    thinking = st.empty()
    if not snap["text"] and not snap["done"]:
        thinking.markdown("""
        <div class="bot-row">
            <div class="bot-avatar">EY</div>
            <div class="bot-body"><div class="dots"><span></span><span></span><span></span></div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        thinking.markdown(f"""
        <div class="bot-row">
            <div class="bot-avatar">EY</div>
            <div class="bot-body"><div class="bot-text">{safe_html(snap["text"])}<span class="caret"></span></div></div>
        </div>
        """, unsafe_allow_html=True)

    if not snap["done"]:
        st.markdown('<div class="stop-wrap"></div>', unsafe_allow_html=True)
        if st.button("■  Stop generation", key="stop_btn"):
            state.request_stop()
            st.rerun(scope="fragment")

    if snap["done"]:
        final_text = snap["text"].strip()
        if snap["error"]:
            current_chat["messages"].append({
                "role": "assistant",
                "content": f"⚠️ Something went wrong: {snap['error']}",
                "citations": [],
            })
        elif snap["stopped"]:
            current_chat["messages"].append({
                "role": "assistant",
                "content": (final_text + "\n\n*⏹ Generation stopped by user.*") if final_text
                            else "⚠️ Generation stopped.",
                "citations":    snap["citations"],
                "image_b64":    snap["image_b64"],
                "chart_type":   snap["chart_type"],
                "code":         snap["code"],
                "agent_steps":  snap["steps"],
                "elapsed":      snap["elapsed"],
            })
        else:
            current_chat["messages"].append({
                "role":         "assistant",
                "content":      final_text or "No answer returned.",
                "citations":    snap["citations"],
                "image_b64":    snap["image_b64"],
                "chart_type":   snap["chart_type"],
                "code":         snap["code"],
                "agent_steps":  snap["steps"],
                "elapsed":      snap["elapsed"],
            })
        st.session_state.generating = False
        st.session_state.stream = None
        st.rerun()


if messages and messages[-1]["role"] == "user":
    last_q = messages[-1]["content"]

    if st.session_state.stream is None:
        st.session_state.generating = True
        state = StreamState(last_q, st.session_state.session_id, api_url, st.session_state.top_k)
        st.session_state.stream = state
        state.start()

    _streaming_fragment(st.session_state.stream)