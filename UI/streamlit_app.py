import streamlit as st
import httpx
import uuid
import time
import os
import re

# Optional: proper Markdown rendering for assistant answers.
# `pip install markdown` for best results; a light fallback is used otherwise.
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
# CUSTOM CSS — EY brand, light, professional, AI-product feel
# =====================================================
# Design tokens
#   Ink (charcoal)   #2E2E38   EY corporate dark
#   Accent (yellow)  #FFE600   used with restraint
#   Text             #1A1A1A / #6B6B76 muted
#   Hairline         #ECECEF
#   Surfaces         #FFFFFF main / #FAFAFB sidebar / #F5F5F7 fills

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

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

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: var(--surface);
    color: var(--text);
    font-family: "Inter", ui-sans-serif, system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"], [data-testid="stStatusWidget"] { display: none; }

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {
    background-color: var(--sidebar);
    border-right: 1px solid var(--hair);
    width: 272px !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.1rem 0.8rem; }

.sidebar-brand {
    display: flex; align-items: center; gap: 11px;
    padding: 0.2rem 0.5rem 1rem;
    border-bottom: 1px solid var(--hair);
}
.brand-mark {
    width: 36px; height: 36px; border-radius: 9px;
    background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800; color: var(--ink);
    letter-spacing: -0.5px; flex-shrink: 0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.brand-text { font-size: 14.5px; font-weight: 700; color: var(--ink); letter-spacing: -0.2px; line-height: 1.15; }
.brand-text span { display: block; font-size: 11px; font-weight: 500; color: var(--muted-2); letter-spacing: 0.02em; margin-top: 2px; }

/* Sidebar buttons base */
[data-testid="stSidebar"] .stButton > button {
    background: var(--surface);
    border: 1px solid var(--hair);
    color: var(--text);
    border-radius: 10px;
    font-size: 13px; font-weight: 500;
    padding: 0.5rem 0.8rem;
    text-align: left;
    transition: background .12s, border-color .12s, color .12s, box-shadow .12s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--fill);
    border-color: #DEDEE3;
    color: var(--text);
}
/* Primary "new conversation" */
.new-conv + div .stButton > button {
    background: var(--accent) !important;
    border: none !important;
    font-weight: 600 !important;
    text-align: center !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
.new-conv + div .stButton > button:hover {
    background: var(--accent-dk) !important;
    box-shadow: 0 2px 9px rgba(255,230,0,0.4) !important;
}

/* Search box */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: var(--surface);
    border: 1px solid var(--hair);
    border-radius: 9px;
    font-size: 12.5px;
    color: var(--text);
    padding: 0.5rem 0.7rem;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(255,230,0,0.18);
}

.history-label {
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted-2); padding: 0 0.4rem; margin: 1rem 0 0.4rem; display: block; font-weight: 600;
}

/* History row buttons (open vs delete) live in columns */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button {
    background: transparent; border: none; color: var(--ink-soft);
    font-size: 13px; font-weight: 400; padding: 0.4rem 0.6rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button:hover {
    background: var(--fill); color: var(--text);
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button {
    background: transparent; border: none; color: var(--muted-2);
    font-size: 15px; padding: 0.3rem 0.35rem; line-height: 1;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button:hover {
    color: #E04848; background: #FDEDED;
}
.active-row + div [data-testid="column"]:first-child .stButton > button {
    background: var(--surface) !important;
    color: var(--text) !important; font-weight: 600 !important;
    box-shadow: inset 0 0 0 1px var(--hair);
}
[data-testid="stSidebar"] hr { border-color: var(--hair); margin: 0.4rem 0; }

/* ============ MAIN AREA ============ */
.block-container {
    max-width: 800px; margin: 0 auto;
    padding: 1.6rem 1.5rem 10rem;
}

.topbar {
    display: flex; align-items: center; gap: 13px;
    padding-bottom: 1rem; margin-bottom: 1.6rem;
    border-bottom: 1px solid var(--hair);
}
.topbar-mark {
    width: 42px; height: 42px; border-radius: 11px; background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 800; color: var(--ink); letter-spacing: -0.5px; flex-shrink: 0;
}
.topbar-title { font-size: 19px; font-weight: 700; color: var(--ink); letter-spacing: -0.4px; line-height: 1.2; }
.topbar-sub { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
.topbar-status {
    margin-left: auto; display: flex; align-items: center; gap: 6px;
    font-size: 11.5px; color: var(--muted); font-weight: 500;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #34C759; box-shadow: 0 0 0 3px rgba(52,199,89,0.15); }

/* ============ MESSAGES ============ */
.user-row { display: flex; justify-content: flex-end; margin: 1.5rem 0 0.4rem; }
.user-bubble {
    background: var(--ink); color: #fff;
    border-radius: 16px 16px 5px 16px;
    padding: 11px 16px; max-width: 80%;
    font-size: 14.5px; line-height: 1.6; font-weight: 400;
    box-shadow: 0 1px 2px rgba(46,46,56,0.15);
}

.bot-row { display: flex; gap: 13px; margin: 0.5rem 0 0.2rem; align-items: flex-start; }
.bot-avatar {
    width: 32px; height: 32px; border-radius: 9px; background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 800; color: var(--ink); letter-spacing: -0.3px;
    flex-shrink: 0; margin-top: 2px;
}
.bot-body { max-width: calc(100% - 45px); padding-top: 3px; }
.bot-text { color: var(--text); font-size: 14.8px; line-height: 1.75; }
.bot-text p { margin: 0 0 0.7em; }
.bot-text p:last-child { margin-bottom: 0; }
.bot-text ul, .bot-text ol { margin: 0.5em 0 0.7em; padding-left: 1.35em; }
.bot-text li { margin-bottom: 0.35em; }
.bot-text h1, .bot-text h2, .bot-text h3 { font-size: 15.5px; font-weight: 700; color: var(--ink); margin: 1em 0 0.4em; letter-spacing: -0.2px; }
.bot-text strong { color: var(--ink); font-weight: 600; }
.bot-text a { color: #1f6feb; text-decoration: none; border-bottom: 1px solid rgba(31,111,235,0.25); }
.bot-text a:hover { border-bottom-color: #1f6feb; }
.bot-text code {
    background: var(--fill); border: 1px solid var(--hair); border-radius: 5px;
    padding: 1px 5px; font-size: 12.8px;
    font-family: "SF Mono", "JetBrains Mono", Consolas, monospace;
}
.bot-text pre {
    background: #1E1E24; color: #EDEDED; border-radius: 10px;
    padding: 14px 16px; overflow-x: auto; margin: 0.6em 0; font-size: 13px;
    font-family: "SF Mono", "JetBrains Mono", Consolas, monospace; line-height: 1.55;
}
.bot-text pre code { background: none; border: none; padding: 0; color: inherit; font-size: 13px; }
.bot-text table { border-collapse: collapse; margin: 0.6em 0; font-size: 13.5px; }
.bot-text th, .bot-text td { border: 1px solid var(--hair); padding: 7px 11px; text-align: left; }
.bot-text th { background: var(--fill); font-weight: 600; }

.caret { display:inline-block; width:7px; height:1.05em; background:var(--ink); margin-left:2px; vertical-align:-2px; animation: blink 1s steps(2) infinite; border-radius:1px; }
@keyframes blink { 50% { opacity: 0; } }

/* ============ SOURCES ============ */
.sources { margin: 0.75rem 0 0.4rem 45px; }
.sources-label {
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted-2); margin-bottom: 0.45rem; font-weight: 600;
}
.sources-row { display: flex; flex-wrap: wrap; gap: 6px; }
.source-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface); border: 1px solid var(--hair);
    border-radius: 8px; padding: 5px 11px; font-size: 12px; color: var(--muted);
    max-width: 240px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    transition: border-color .12s, background .12s, color .12s;
}
.source-pill:hover { border-color: var(--accent-dk); background: #FFFEF2; color: var(--text); }
.source-pill .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent-dk); flex-shrink: 0; }

/* Regenerate row */
.regen-wrap { margin-left: 45px; }
.regen-wrap + div .stButton > button {
    background: transparent; border: 1px solid var(--hair); color: var(--muted);
    border-radius: 8px; font-size: 12px; font-weight: 500; padding: 0.3rem 0.7rem;
    text-align: center;
}
.regen-wrap + div .stButton > button:hover { background: var(--fill); color: var(--text); border-color: #DEDEE3; }

/* ============ ERROR ============ */
.error-row { display: flex; gap: 13px; margin: 0.5rem 0; align-items: flex-start; }
.error-bubble {
    background: #FEF6F6; border: 1px solid #F6D5D5; border-radius: 11px;
    padding: 11px 15px; font-size: 13.5px; color: #B23B3B; line-height: 1.5;
    max-width: calc(100% - 45px);
}

/* ============ THINKING ============ */
.dots { display: inline-flex; gap: 5px; padding: 7px 2px; }
.dots span { width: 7px; height: 7px; background: var(--muted-2); border-radius: 50%; animation: bob 1.2s infinite ease-in-out; }
.dots span:nth-child(2) { animation-delay: .18s; }
.dots span:nth-child(3) { animation-delay: .36s; }
@keyframes bob { 0%,80%,100% { transform: scale(.6); opacity:.35; } 40% { transform: scale(1); opacity:1; } }

/* ============ WELCOME / EMPTY STATE ============ */
.welcome { text-align: center; padding: 3rem 1rem 1.5rem; }
.welcome-mark {
    width: 58px; height: 58px; border-radius: 16px; background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; font-weight: 800; color: var(--ink); letter-spacing: -0.6px;
    margin: 0 auto 1.3rem; box-shadow: 0 4px 16px rgba(255,230,0,0.4);
}
.welcome-title { font-size: 28px; font-weight: 800; color: var(--ink); letter-spacing: -0.8px; margin-bottom: 0.5rem; }
.welcome-sub { font-size: 14.5px; color: var(--muted); max-width: 420px; margin: 0 auto; line-height: 1.6; }
.suggest-label {
    text-align: center; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted-2); font-weight: 600; margin: 2rem 0 0.9rem;
}

/* Suggestion cards = main-area buttons */
.block-container .stButton > button {
    background: var(--surface); border: 1px solid var(--hair); color: var(--text);
    border-radius: 13px; padding: 0.85rem 1rem; text-align: left;
    font-size: 13.5px; font-weight: 500; line-height: 1.45; min-height: 70px;
    transition: border-color .14s, box-shadow .14s, transform .14s, background .14s;
}
.block-container .stButton > button:hover {
    border-color: var(--accent-dk); background: #FFFEF6;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06); transform: translateY(-2px);
}

/* ============ CHAT INPUT ============ */
[data-testid="stChatInput"] {
    position: fixed; bottom: 0; left: 272px; right: 0;
    padding: 0.8rem 2rem 1.4rem;
    background: linear-gradient(to top, var(--surface) 72%, rgba(255,255,255,0));
    z-index: 100;
}
[data-testid="stChatInput"] > div { max-width: 800px; margin: 0 auto; }
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--hair) !important;
    border-radius: 15px !important;
    color: var(--text) !important; font-size: 14.5px !important;
    padding: 14px 18px !important; box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important;
    transition: border-color .15s, box-shadow .15s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(255,230,0,0.18), 0 2px 12px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatInput"] button { background: var(--ink) !important; }
.input-hint { text-align:center; font-size: 11px; color: var(--muted-2); margin-top: 0.5rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #DEDEE3; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #C4C4CC; }

@media (max-width: 820px) {
    [data-testid="stChatInput"] { left: 0; }
}
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
    """Escape user-supplied content before injecting into HTML blocks."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_markdown(text):
    """Convert an assistant answer (markdown or HTML) to display HTML."""
    if not text:
        return ""
    if _HAS_MD:
        return _md.markdown(
            text,
            extensions=["fenced_code", "tables", "sane_lists", "nl2br"],
        )
    # Lightweight fallback (assistant content is trusted, like the original).
    t = text
    t = re.sub(r"```(.*?)```", lambda m: "<pre><code>" + safe_html(m.group(1)) + "</code></pre>", t, flags=re.S)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", r"<em>\1</em>", t)
    t = re.sub(r"`([^`]+?)`", r"<code>\1</code>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank">\1</a>', t)
    paras = [p for p in t.split("\n\n") if p.strip()]
    return "".join("<p>" + p.replace("\n", "<br>") + "</p>" for p in paras)


def submit_message(text):
    """Append a user message to the active chat and rerun (same path as chat_input)."""
    chat = st.session_state.chats[st.session_state.current_chat]
    if chat["title"] == "New conversation":
        chat["title"] = truncate_title(text)
    chat["messages"].append({"role": "user", "content": text})
    st.rerun()


# =====================================================
# SESSION STATE
# =====================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit_{uuid.uuid4()}"

if "chats" not in st.session_state:
    cid = new_chat_id()
    st.session_state.chats = {cid: {"title": "New conversation", "messages": []}}
    st.session_state.current_chat = cid

if ("current_chat" not in st.session_state
        or st.session_state.current_chat not in st.session_state.chats):
    st.session_state.current_chat = next(iter(st.session_state.chats))


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-mark">EY</div>
        <div class="brand-text">Knowledge<span>Middle East</span></div>
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
            '<div style="padding:0.5rem 0.6rem;font-size:12.5px;color:#9A9AA4;">No conversations found.</div>',
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


# =====================================================
# MAIN — HEADER
# =====================================================

current_chat = st.session_state.chats[st.session_state.current_chat]
messages = current_chat["messages"]

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
# MAIN — WELCOME (empty state) OR CONVERSATION
# =====================================================

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

else:
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-row">
                <div class="user-bubble">{safe_html(msg["content"])}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            is_error = str(msg["content"]).startswith("⚠️")
            if is_error:
                st.markdown(f"""
                <div class="error-row">
                    <div class="bot-avatar">EY</div>
                    <div class="error-bubble">{safe_html(msg["content"])}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-row">
                    <div class="bot-avatar">EY</div>
                    <div class="bot-body"><div class="bot-text">{render_markdown(msg["content"])}</div></div>
                </div>
                """, unsafe_allow_html=True)

            citations = msg.get("citations", [])
            if citations:
                pills = "".join(
                    f'<div class="source-pill"><span class="dot"></span>'
                    f'{safe_html(s.get("source_file", "Source"))}</div>'
                    for s in citations
                )
                st.markdown(f"""
                <div class="sources">
                    <div class="sources-label">Sources</div>
                    <div class="sources-row">{pills}</div>
                </div>
                """, unsafe_allow_html=True)

    # Regenerate the last answer (re-asks the previous user question)
    if (len(messages) >= 2
            and messages[-1]["role"] == "assistant"
            and not str(messages[-1]["content"]).startswith("⚠️")):
        st.markdown('<div class="regen-wrap"></div>', unsafe_allow_html=True)
        if st.button("↻  Regenerate", key="regen_btn"):
            messages.pop()  # drop last assistant turn; user turn re-triggers the call below
            st.rerun()


# =====================================================
# CHAT INPUT
# =====================================================

prompt = st.chat_input("Ask about EY Middle East…")
st.markdown(
    '<div class="input-hint">Answers are generated from the EY Middle East knowledge base.</div>',
    unsafe_allow_html=True,
)

if prompt and prompt.strip():
    submit_message(prompt.strip())


# =====================================================
# CALL BACKEND & STREAM RESPONSE
# =====================================================

messages = current_chat["messages"]

if messages and messages[-1]["role"] == "user":
    last_q = messages[-1]["content"]

    thinking = st.empty()
    thinking.markdown("""
    <div class="bot-row">
        <div class="bot-avatar">EY</div>
        <div class="bot-body"><div class="dots"><span></span><span></span><span></span></div></div>
    </div>
    """, unsafe_allow_html=True)

    try:
        api_url = os.getenv("API_URL", "http://api:8000")
        response = httpx.post(
            f"{api_url}/chat",
            json={"question": last_q, "session_id": st.session_state.session_id},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "No answer returned.")
        citations = data.get("citations", [])

        # Stream word-by-word into the assistant bubble
        words = answer.split()
        streamed = ""
        for word in words:
            streamed += word + " "
            thinking.markdown(f"""
            <div class="bot-row">
                <div class="bot-avatar">EY</div>
                <div class="bot-body"><div class="bot-text">{safe_html(streamed)}<span class="caret"></span></div></div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.012)

        thinking.empty()
        current_chat["messages"].append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
        })

    except httpx.TimeoutException:
        thinking.empty()
        current_chat["messages"].append({
            "role": "assistant",
            "content": "⚠️ The request timed out. Please try again.",
            "citations": [],
        })
    except httpx.ConnectError:
        thinking.empty()
        current_chat["messages"].append({
            "role": "assistant",
            "content": "⚠️ Couldn't reach the knowledge service. Check that the API is running and try again.",
            "citations": [],
        })
    except Exception as e:
        thinking.empty()
        current_chat["messages"].append({
            "role": "assistant",
            "content": f"⚠️ Something went wrong: {str(e)}",
            "citations": [],
        })

    st.rerun()