"""
app.py
Streamlit chat interface for the Conversational Recommender System.
Run with: streamlit run app.py
"""

import copy
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import database
import memory
import ranking
import observability
import llm_client
from state import initial_state
from graph import run_turn

# ── Branding ──────────────────────────────────────────────────────────────────
APP_NAME = "Findora"
APP_TAGLINE = "Your AI assistant for smartphones & headphones"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

    :root {
        --grad-1: #6366f1;
        --grad-2: #8b5cf6;
        --grad-3: #ec4899;
        --ink: #1e293b;
        --muted: #64748b;
        --line: #e8eaf3;
        --card: #ffffff;
    }

    /* ── Global ──────────────────────────────────────────────── */
    html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1200px 600px at 80% -10%, #ede9fe 0%, transparent 55%),
            radial-gradient(1000px 500px at -10% 0%, #e0e7ff 0%, transparent 50%),
            linear-gradient(180deg, #f7f8fc 0%, #f4f6fb 100%);
    }
    .main .block-container { padding-top: 1.5rem; max-width: 1150px; }

    /* Hide default Streamlit chrome (Deploy button, ⋮ menu, status, footer).
       Do NOT hide the whole stToolbar: the collapsed-sidebar expand (») button
       lives inside it, so hiding the toolbar makes a hidden sidebar impossible
       to reopen. Hide only the specific chrome items instead. */
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    [data-testid="stToolbarActions"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stAppDeployButton"] { display: none !important; }
    .stDeployButton { display: none !important; }
    /* keep the header (so the sidebar toggle stays) but make it blend in */
    [data-testid="stHeader"] { background: transparent; }

    /* ── Sticky top toolbar (always-visible brand + actions) ─────
       Streamlit wraps each element in a wrapper only as tall as itself, so
       `position: sticky` on the bar alone can't travel. We pin the WRAPPER
       (its containing block is the full-height main column). The wrapper's own
       box can collapse to a thin strip at the left edge, so we make the wrapper
       click-through and only let the visible bar capture clicks — otherwise that
       invisible strip sits over the collapsed-sidebar expand button and you
       can't reopen the sidebar. z-index stays low (below the sidebar controls). */
    [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {
        position: sticky; top: 0; z-index: 99; pointer-events: none;
    }
    .st-key-topbar {
        pointer-events: auto;
        background: rgba(255,255,255,0.94);
        backdrop-filter: blur(10px);
        border: 1px solid var(--line); border-radius: 16px;
        box-shadow: 0 10px 28px -18px rgba(30,41,59,.45);
        padding: 8px 16px; margin-bottom: 10px;
    }
    /* The collapsed-sidebar expand (») control must always sit above the toolbar
       so the sidebar can be reopened after it's hidden. */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] { z-index: 1000001 !important; }
    .brand {
        font-family: 'Poppins', sans-serif; font-weight: 800; font-size: 1.5rem;
        background: linear-gradient(135deg, var(--grad-1), var(--grad-3));
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; line-height: 1.1;
    }
    .brand-tag { color: var(--muted); font-size: .74rem; margin-top: -2px; margin-bottom: 6px; }
    .st-key-topbar .stButton button {
        padding: .36rem .5rem; font-size: .82rem; font-weight: 600;
        white-space: nowrap; min-height: 0;
    }
    .st-key-topbar .stButton button p { white-space: nowrap; }
    .st-key-topbar .stTooltipHoverTarget { width: 100%; }
    /* tighten the gap the sticky container adds above the button row */
    .st-key-topbar [data-testid="stHorizontalBlock"] { margin-top: 2px; }
    /* greeting sits above the buttons and stays for the whole session */
    .topbar-greeting {
        margin: 6px 0 10px; color: var(--ink); font-size: .92rem; line-height: 1.5;
    }

    /* ── Inline "edit message" affordance — ghost button under user turns ── */
    [class*="st-key-edit_btn_"] button {
        padding: .02rem .45rem; min-height: 0; font-size: .72rem; font-weight: 600;
        background: transparent; border: 1px solid transparent;
        color: var(--muted); box-shadow: none;
    }
    [class*="st-key-edit_btn_"] button:hover {
        background: rgba(99,102,241,.08); border-color: var(--line); color: var(--ink);
    }
    /* Buttons / labels that must never wrap char-by-char in narrow flex columns. */
    [class*="st-key-det_"] button p, [class*="st-key-cmpbtn_"] button p,
    [class*="st-key-edit_save_"] button p, [class*="st-key-edit_cancel_"] button p,
    [class*="st-key-cmp_"] p { white-space: nowrap; }

    /* ── Hero header ─────────────────────────────────────────── */
    .hero {
        position: relative;
        background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 45%, #ec4899 100%);
        background-size: 200% 200%;
        animation: heroShift 12s ease infinite;
        border-radius: 26px;
        padding: 30px 36px;
        margin-bottom: 18px;
        box-shadow: 0 24px 50px -18px rgba(99,102,241,.55);
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(420px 200px at 88% -30%, rgba(255,255,255,.30), transparent 70%);
        pointer-events: none;
    }
    @keyframes heroShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,.18);
        border: 1px solid rgba(255,255,255,.35);
        color: #fff;
        font-size: .74rem; font-weight: 600; letter-spacing: .04em;
        padding: 5px 13px; border-radius: 999px;
        backdrop-filter: blur(6px);
        margin-bottom: 14px;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        color: #fff; font-weight: 800;
        font-size: 2.3rem; line-height: 1.1; margin: 0 0 6px 0;
        letter-spacing: -.02em;
    }
    .hero-sub {
        color: rgba(255,255,255,.92);
        font-size: 1.02rem; margin: 0 0 16px 0; max-width: 640px;
    }
    .hero-chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .hero-chip {
        background: rgba(255,255,255,.16);
        border: 1px solid rgba(255,255,255,.30);
        color: #fff; font-size: .82rem; font-weight: 600;
        padding: 6px 14px; border-radius: 999px;
        backdrop-filter: blur(6px);
    }

    /* ── Chat messages ───────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 6px 18px;
        margin: 8px 0;
        border: 1px solid var(--line);
        box-shadow: 0 8px 24px -16px rgba(30,41,59,.30);
        animation: fadeUp .35s ease both;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color: var(--ink); }
    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: #ffffff;
        border-left: 4px solid var(--grad-2);
    }
    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, #eef2ff 0%, #faf5ff 100%);
        border: 1px solid #e9d5ff;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Buttons (quick-replies + reset) ─────────────────────── */
    .stButton > button {
        border-radius: 999px;
        border: 1.5px solid #c7d2fe;
        background: #ffffff;
        color: #4f46e5;
        font-weight: 600; font-size: .88rem;
        padding: .42rem 1rem;
        transition: all .18s ease;
        box-shadow: 0 2px 6px -3px rgba(99,102,241,.4);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--grad-1), var(--grad-2));
        color: #fff;
        border-color: transparent;
        transform: translateY(-2px);
        box-shadow: 0 10px 22px -10px rgba(99,102,241,.7);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ── Chat input bar ──────────────────────────────────────── */
    [data-testid="stChatInput"] {
        border-radius: 18px;
        border: 1.5px solid var(--line);
        box-shadow: 0 10px 30px -16px rgba(30,41,59,.35);
        background: #fff;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--grad-2);
        box-shadow: 0 0 0 4px rgba(139,92,246,.16);
    }

    /* ── Selectbox ───────────────────────────────────────────── */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-radius: 12px;
        border-color: var(--line);
    }

    /* ── Expanders (product cards) ───────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        background: #fff;
        box-shadow: 0 6px 18px -14px rgba(30,41,59,.35);
        margin-bottom: 8px;
        transition: box-shadow .18s ease, transform .18s ease;
        overflow: hidden;
    }
    [data-testid="stExpander"]:hover {
        box-shadow: 0 16px 34px -18px rgba(99,102,241,.45);
        transform: translateY(-1px);
    }
    [data-testid="stExpander"] summary {
        font-weight: 600; font-size: .95rem; color: var(--ink);
        padding: 6px 4px;
    }
    [data-testid="stExpander"] summary:hover { color: var(--grad-1); }

    /* ── Tables ──────────────────────────────────────────────── */
    [data-testid="stTable"] table {
        border-radius: 12px; overflow: hidden;
        border: 1px solid var(--line);
        font-size: .88rem;
    }
    [data-testid="stTable"] thead th {
        background: linear-gradient(135deg, #eef2ff, #f5f3ff);
        color: #4338ca; font-weight: 700;
        border-bottom: 1px solid var(--line) !important;
    }
    [data-testid="stTable"] tbody th {
        background: #fafafe; color: var(--muted); font-weight: 600;
    }
    [data-testid="stTable"] tbody tr:hover td { background: #f7f8fe; }

    /* ── Sidebar ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #fbfaff 100%);
        border-right: 1px solid var(--line);
    }
    .side-head {
        font-family: 'Poppins', sans-serif;
        font-weight: 700; font-size: 1.15rem;
        background: linear-gradient(135deg, var(--grad-1), var(--grad-3));
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 10px 14px;
        box-shadow: 0 4px 12px -10px rgba(30,41,59,.4);
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-weight: 700; color: var(--ink); font-size: 1.25rem;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
        color: var(--muted); font-weight: 600; font-size: .78rem;
        text-transform: uppercase; letter-spacing: .05em;
    }

    /* ── Product image placeholder (honest miss, not a fake photo) ── */
    .img-placeholder {
        width: 200px; height: 150px;
        display: flex; align-items: center; justify-content: center;
        font-size: 3rem;
        background: linear-gradient(135deg, #eef2ff, #f3e8ff);
        border: 1px dashed #c7d2fe; border-radius: 14px;
        color: #94a3b8;
    }
    .img-credit { color: #94a3b8; font-size: .7rem; margin-top: 2px; }

    /* ── Product grid cards ─────────────────────────────────────── */
    .card-img-wrap {
        width: 100%; height: 140px; display: flex; align-items: center;
        justify-content: center; background: #fff; border-radius: 12px;
        border: 1px solid var(--line); overflow: hidden; margin-bottom: 8px;
    }
    .card-img-wrap img { max-height: 132px; max-width: 92%; object-fit: contain; }
    .card-img-ph {
        width: 100%; height: 140px; display: flex; align-items: center;
        justify-content: center; font-size: 2.6rem; color: #b9c0d4;
        background: linear-gradient(135deg, #eef2ff, #f3e8ff);
        border: 1px dashed #c7d2fe; border-radius: 12px; margin-bottom: 8px;
    }
    .card-title { font-weight: 700; color: var(--ink); font-size: .92rem;
        line-height: 1.2; min-height: 2.4em; }
    .card-meta { color: var(--muted); font-size: .82rem; margin: 2px 0 6px; }
    .card-rank { color: #8b5cf6; font-weight: 700; font-size: .72rem; letter-spacing: .04em; }

    /* ── Winner-highlight comparison table ──────────────────────── */
    table.cmp { width: 100%; border-collapse: collapse; font-size: .86rem; }
    table.cmp th { background: linear-gradient(135deg,#eef2ff,#f5f3ff); color:#4338ca;
        font-weight: 700; padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: center; }
    table.cmp td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: center; color: var(--ink); }
    table.cmp td.cmp-attr { text-align: left; color: var(--muted); font-weight: 600; }
    table.cmp td.cmp-win { background: #dcfce7; color: #166534; font-weight: 700; }

    /* ── Filter pills ────────────────────────────────────────── */
    .filter-pill {
        display: inline-block;
        background: linear-gradient(135deg, #eef2ff, #f3e8ff);
        color: #5b21b6;
        border: 1px solid #e9d5ff;
        border-radius: 999px;
        padding: 4px 12px;
        margin: 3px 3px 0 0;
        font-size: 0.78rem; font-weight: 600;
    }

    /* ── Custom scrollbar ────────────────────────────────────── */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #c7d2fe, #ddd6fe);
        border-radius: 999px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #a5b4fc; }
</style>
""", unsafe_allow_html=True)

# ── Load data once ────────────────────────────────────────────────────────────

@st.cache_resource
def load_database():
    database.load_all()
    return True

load_database()

# ── "How it works" modal + zero-input Top Picks ──────────────────────────────

@st.dialog(f"How {APP_NAME} works")
def show_how_it_works():
    st.markdown(
        f"**{APP_NAME}** is a conversational assistant for **smartphones** and **headphones**. "
        "Here's how to get the most out of it:"
    )
    st.markdown(
        "- 🗣️ **Just describe what you want** in plain language — "
        "*“a cheap Samsung with a good camera”* or *“headphones for the gym”*.\n"
        "- ❓ **I ask only what I need** — a couple of quick questions (you can say *“any”* to skip).\n"
        "- 🏆 **Or let me decide** — hit a **Top picks** button and I'll recommend the best by value.\n"
        "- 📊 **Every pick is explained** — photo, full specs, and a match score you can expand.\n"
        "- ⚖️ **Compare & refine** — tick 2–3 to compare, or say *“cheaper”*, *“bigger battery”*, "
        "*“actually iPhone”*; say *“ignore that”* to undo.\n"
        "- 💡 **Handy commands** — *“clear all filters”* / *“start over”* to reset, "
        "*“what do you have so far?”* for a recap, *“actually, show me headphones”* to switch category.\n"
        "- ✅ **When happy**, say *“I'll take it”* / *“that's all”* to wrap up."
    )
    st.caption("Under the hood: validated NLU · semantic search · TOPSIS ranking · real product photos.")
    if st.button("Got it — let's go", use_container_width=True):
        st.rerun()


def start_top_picks(category):
    """Zero-input recommendation: seed a 'best of' list and hand off to the
    normal refine flow. Deterministic (no LLM call) — instant and free."""
    ranked = ranking.top_picks(category, n=5)
    ds = initial_state()
    ds["category"] = category
    ds["action"] = "recommend"
    ds["candidates"] = ranked
    ds["last_recommend_stats"] = database.candidate_stats(category, ranked)
    st.session_state.dialogue_state = ds
    label = "smartphones" if category == "smartphone" else "headphones"
    intro = (
        f"Here are my **top {label}** right now — ranked by overall value (specs vs. price). "
        f"Browse below and compare a few, or just tell me what matters to you "
        f"(e.g. *“cheaper”*, *“Samsung”*, *“bigger battery”*) and I'll refine."
    )
    st.session_state.chat_messages = [
        {"role": "user", "content": f"Recommend your best {label}",
         "pre_state": initial_state()},   # editing this query rewinds to a clean slate
        {"role": "assistant", "content": intro,
         "recommend_products": ranked, "recommend_category": category},
    ]
    st.session_state.pop("pending_reset", None)
    st.session_state.pop("editing_idx", None)
    memory.record_recommendation(st.session_state.get("user_name", "guest"), category, {})
    st.rerun()


def reset_conversation():
    """Start fresh — clears chat, state, edit mode, and snapshots."""
    st.session_state.dialogue_state = initial_state()
    st.session_state.chat_messages = []
    st.session_state.show_welcome = True
    st.session_state.pop("pending_reset", None)
    st.session_state.pop("editing_idx", None)
    st.rerun()


def begin_edit(i):
    st.session_state.editing_idx = i


def apply_edit(i, new_text):
    """Rewind to before message i (using its saved pre-turn snapshot) and
    re-run with the edited text — instant, no replay of other turns."""
    msgs = st.session_state.chat_messages
    pre = msgs[i].get("pre_state")
    if pre is not None:
        st.session_state.dialogue_state = copy.deepcopy(pre)
    st.session_state.chat_messages = msgs[:i]          # drop this turn + everything after
    st.session_state.pop("editing_idx", None)
    st.session_state.queued_input = new_text           # reprocessed as a fresh turn
    st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────

if "dialogue_state" not in st.session_state:
    st.session_state.dialogue_state = initial_state()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # [{role, content}]

if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

if "user_name" not in st.session_state:
    st.session_state.user_name = "guest"

# Load (or refresh) the persistent profile for the current user.
_profile = memory.load_profile(st.session_state.user_name)

# ── Sidebar: live dialogue state inspector ────────────────────────────────────

@st.cache_data(ttl=20, show_spinner=False)
def _discover_models():
    """Cloud default(s) + locally-installed Ollama models. Cached briefly so we
    don't poll the local Ollama server on every rerun."""
    return llm_client.available_models()

with st.sidebar:
    # ── Identity / personalization ───────────────────────────────────────────
    st.markdown('<div class="side-head">👤 Your Profile</div>', unsafe_allow_html=True)
    name_in = st.text_input("Name", value=st.session_state.user_name,
                            help="Used to remember your preferences across sessions.")
    if name_in and name_in != st.session_state.user_name:
        st.session_state.user_name = name_in
        memory.start_session(name_in)
        st.rerun()

    _summary = memory.profile_summary(_profile)
    if _summary:
        st.caption(f"Welcome back! You usually like: **{_summary}**")
        if memory.has_usual(_profile) and st.button("⭐ Use my usual preferences", use_container_width=True):
            ds_new = initial_state()
            ds_new["category"] = _profile["last_category"]
            ds_new["active_filters"] = dict(_profile["last_filters"])
            ds_new["wants_results"] = True
            st.session_state.dialogue_state = ds_new
            st.session_state.queued_input = "show me something like my usual preferences"
            st.rerun()
    else:
        st.caption("New here — I'll learn your preferences as we go.")

    st.divider()

    # ── Model picker: cloud vs. local Ollama (for side-by-side comparison) ────
    st.markdown('<div class="side-head">🧠 Model</div>', unsafe_allow_html=True)
    _models = _discover_models()
    _ids = [m["id"] for m in _models]
    _labels = {m["id"]: ("💻 " if m["local"] else "☁️ ") + m["label"] for m in _models}
    if st.session_state.get("model_id") not in _ids:
        st.session_state.model_id = _ids[0]
    st.selectbox(
        "Active model", _ids, key="model_id",
        format_func=lambda i: _labels.get(i, i),
        help="Switch between the cloud model and local Ollama models — handy for "
             "comparing answers. Each reply is tagged with the model that produced it.",
    )
    llm_client.set_active_model(st.session_state.model_id)
    _active = llm_client.get_active_model()
    st.caption(("💻 Local · Ollama" if _active["local"] else "☁️ Cloud")
               + f" · `{_active['model']}`")
    if _active["local"]:
        st.caption("⏳ Runs on your machine — replies can take a while. Lighter models "
                   "(e.g. llama3.2, phi4-mini) are much faster than 7B ones.")
    elif not any(m["local"] for m in _models):
        st.caption("💡 Start Ollama (e.g. `ollama run llama3.2`) to add a local model here.")

    st.divider()

    st.markdown('<div class="side-head">🔍 Session State</div>', unsafe_allow_html=True)
    st.caption("Live view of what the system knows about your preferences")

    ds = st.session_state.dialogue_state

    # Category
    cat = ds.get("category") or "—"
    st.metric("Category", cat.replace("_", " ").title())

    # Turn count
    st.metric("Turn", ds.get("turn_count", 0))

    # Intent
    intent = ds.get("intent") or "—"
    st.metric("Last Intent", intent)

    # Active filters
    st.subheader("Active Filters")
    filters = ds.get("active_filters", {})
    if filters:
        PRICE_LABELS = {
            "price_usd_min": "Price min ($)",
            "price_usd_max": "Price max ($)",
        }
        for k, v in filters.items():
            label = PRICE_LABELS.get(k, k.replace("_", " ").title())
            st.markdown(f'<span class="filter-pill">**{label}**: {v}</span>', unsafe_allow_html=True)
    else:
        st.caption("No filters yet")

    # Candidates count
    n_candidates = len(ds.get("candidates", []))
    if n_candidates > 0:
        st.metric("Matching Products", n_candidates)

    # Last action
    action = ds.get("action") or "—"
    action_colors = {
        "recommend":        "🟢",
        "ask_clarification": "🟡",
        "ask_category":     "🔵",
        "no_results":       "🔴",
        "done":             "⚫",
    }
    icon = action_colors.get(action, "⚪")
    st.caption(f"Last action: {icon} {action}")

    st.divider()

    # "Start new conversation" (🔄 New chat) and "How it works" now live in the
    # main top panel, so they're intentionally not duplicated in the sidebar.

    # Debug expander
    with st.expander("🛠 Raw State (debug)"):
        import json
        debug_state = {k: v for k, v in ds.items() if k != "messages"}
        st.code(json.dumps(debug_state, indent=2, default=str), language="json")

# ── Comparison table renderer ───────────────────────────────────────────────

def _render_comparison_table(products, category):
    """Render a side-by-side spec comparison of the top recommended products."""
    import pandas as pd

    if not products:
        return

    def cell(p, key, fmt=str):
        v = p.get(key)
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            return "—"
        try:
            return fmt(v)
        except Exception:
            return "—"

    def truthy(v):
        return str(v).lower() == "true"

    def freq_range(p):
        lo, hi = p.get("freq_low_hz"), p.get("freq_high_hz")
        bad = lambda v: v is None or (isinstance(v, float) and v != v)
        if bad(lo) or bad(hi):
            return "—"
        return f"{int(lo)}–{int(hi)} Hz"

    if category == "smartphone":
        rows = [
            ("Brand",        lambda p: cell(p, "brand_name", lambda v: str(v).title())),
            ("Model",        lambda p: cell(p, "model")),
            ("Price",        lambda p: cell(p, "price_usd", lambda v: f"${int(v):,}")),
            ("OS",           lambda p: cell(p, "os", lambda v: str(v).upper())),
            ("RAM",          lambda p: cell(p, "ram_capacity", lambda v: f"{int(v)} GB")),
            ("Storage",      lambda p: cell(p, "internal_memory", lambda v: f"{int(v)} GB")),
            ("Battery",      lambda p: cell(p, "battery_capacity", lambda v: f"{int(v)} mAh")),
            ("Rear Camera",  lambda p: cell(p, "primary_camera_rear", lambda v: f"{int(v)} MP")),
            ("Front Camera", lambda p: cell(p, "primary_camera_front", lambda v: f"{int(v)} MP")),
            ("Screen",       lambda p: cell(p, "screen_size", lambda v: f'{v}"')),
            ("Score",        lambda p: f"{p.get('_score', '—')} / 100"),
        ]
        def name(p):
            return f"{str(p.get('brand_name', '')).title()} {p.get('model', '')}".strip()
    elif category == "headphones":
        rows = [
            ("Brand",              lambda p: cell(p, "brand")),
            ("Model",              lambda p: cell(p, "model")),
            ("Price",              lambda p: cell(p, "price_usd", lambda v: f"${int(v)}")),
            ("Type",               lambda p: cell(p, "type")),
            ("Form Factor",        lambda p: cell(p, "form_factor")),
            ("Connectivity",       lambda p: cell(p, "connectivity")),
            ("Noise Cancellation", lambda p: "Yes" if truthy(p.get("noise_cancellation")) else "No"),
            ("Microphone",         lambda p: "Yes" if truthy(p.get("microphone")) else "No"),
            ("Foldable",           lambda p: "Yes" if truthy(p.get("foldable")) else "No"),
            ("Battery (hrs)",      lambda p: cell(p, "battery_hrs", lambda v: f"{v}") if str(p.get("type", "")).lower() == "wireless" else "—"),
            ("Freq Range",         freq_range),
            ("Avg Rating",         lambda p: cell(p, "avg_rating", lambda v: f"{v} / 5")),
            ("Release Year",       lambda p: cell(p, "release_year", lambda v: str(int(v)))),
            ("Score",              lambda p: f"{p.get('_score', '—')} / 100"),
        ]
        def name(p):
            return f"{p.get('brand', '')} {p.get('model', '')}".strip()
    else:
        return

    # Column headers from brand+model (deduped via _card_name); disambiguate identical names
    headers = [_card_name(p, category) or "Product" for p in products]
    if len(set(headers)) < len(headers):
        headers = [f"{h} ({i+1})" for i, h in enumerate(headers)]

    data = {col: [fn(p) for _, fn in rows] for col, p in zip(headers, products)}
    df = pd.DataFrame(data, index=[label for label, _ in rows])
    st.table(df)


def _product_header(p, category):
    """One-line summary used as the expander title for a single product."""
    score = p.get("_score", "—")
    if category == "smartphone":
        brand = str(p.get("brand_name", "")).title()
        model = p.get("model", "")
        price = p.get("price_usd")
        price_str = f"${int(price):,}" if isinstance(price, (int, float)) and price == price else "—"
    elif category == "headphones":
        brand = p.get("brand", "")
        model = p.get("model", "")
        price = p.get("price_usd")
        price_str = f"${int(price)}" if isinstance(price, (int, float)) and price == price else "—"
    else:
        brand, model, price_str = "", "", "—"
    header = f"{brand} {model}  ·  {price_str}  ·  ⭐ {score} / 100".strip()
    # Show semantic relevance when a free-text "vibe" query drove this ranking.
    sem = p.get("_semantic")
    if isinstance(sem, (int, float)) and sem > 0:
        header += f"  ·  🔎 {sem:.0f}% match"
    return header


_SCORE_ATTR_LABELS = {
    "price_usd":           "💰 Price",
    "primary_camera_rear": "📸 Rear Camera",
    "battery_capacity":    "🔋 Battery",
    "ram_capacity":        "🧠 RAM",
    "internal_memory":     "💾 Storage",
    "rating":              "⭐ Rating",
    "avg_rating":          "⭐ Avg Rating",
    "battery_hrs":         "🔋 Battery Life",
    "freq_range":          "🎵 Freq Range",
    "noise_cancellation":  "🔇 Noise Cancel.",
}


def _format_breakdown_value(attr, raw, direction):
    """Pretty-print a raw value with the right unit for the breakdown table."""
    if raw is None:
        return "—"
    if direction == "binary":
        return "Yes" if raw == 1.0 else "No"
    if attr == "price_usd":
        return f"${int(raw):,}"
    if attr in ("battery_capacity",):
        return f"{int(raw)} mAh"
    if attr in ("ram_capacity", "internal_memory"):
        return f"{int(raw)} GB"
    if attr == "primary_camera_rear":
        return f"{int(raw)} MP"
    if attr == "battery_hrs":
        return f"{raw} hrs"
    if attr == "freq_range":
        return f"{int(raw)} Hz"
    if attr in ("rating", "avg_rating"):
        return f"{raw}"
    return str(raw)


def _render_score_breakdown(rows, total_count):
    """Render a per-attribute breakdown markdown table for one product."""
    if not rows:
        return
    md = ["| Factor | Value | Score (0-100) | Weight |", "|---|---|---|---|"]
    for r in rows:
        label = _SCORE_ATTR_LABELS.get(r["attr"], r["attr"])
        value = _format_breakdown_value(r["attr"], r["raw"], r["direction"])
        norm = r["norm_0_100"]
        marker = "🟢" if norm >= 75 else "🟡" if norm >= 40 else "🔴"
        md.append(f"| {label} | {value} | {marker} {norm:.0f} | {r['weight_pct']:.0f}% |")
    st.markdown("\n".join(md))
    st.caption(
        f"_Top score uses **TOPSIS** — closeness to the ideal (best specs, lowest price) "
        f"across the {total_count} matches. The Score column here shows where this product "
        f"stands in the **whole catalog** for each attribute (100 = best in the catalog, "
        f"0 = worst, 50 = median or missing)._"
    )


def _render_product_image(p, category):
    """
    Show the product photo if the catalog has one (populated offline by
    enrich_images.py). Otherwise an honest placeholder — never a fake/AI image.
    """
    def _valid(v):
        return isinstance(v, str) and v.startswith("http")

    icon = "📱" if category == "smartphone" else "🎧"
    # Headphone photos in the catalog are unreliable — always show the placeholder.
    if category == "headphones":
        img = None
    else:
        img = next((p[k] for k in ("image_url", "image", "img", "thumbnail")
                    if _valid(p.get(k))), None)
    if img:
        try:
            st.image(img, width=200)
            return
        except Exception:
            pass  # dead/blocked URL → fall through to placeholder
    st.markdown(f'<div class="img-placeholder">{icon}</div>', unsafe_allow_html=True)


def _card_name(p, category):
    brand = str(p.get("brand_name", "") if category == "smartphone" else p.get("brand", ""))
    model = str(p.get("model", ""))
    # The model often already includes the brand ("Xiaomi Redmi Note 12") — don't
    # double it up into "Xiaomi Xiaomi Redmi Note 12".
    if model.lower().startswith(brand.lower()):
        return model.strip()
    brand = brand.title() if category == "smartphone" else brand
    return f"{brand} {model}".strip()


def _card_image_html(p, category):
    """Uniform image tile (HTML) so every card aligns; honest placeholder on miss."""
    def _valid(v):
        return isinstance(v, str) and v.startswith("http")
    icon = "📱" if category == "smartphone" else "🎧"
    # Headphone photos in the catalog are unreliable — always show the placeholder.
    img = None if category == "headphones" else next(
        (p[k] for k in ("image_url", "image", "img", "thumbnail") if _valid(p.get(k))), None)
    if img:
        return f'<div class="card-img-wrap"><img src="{img}"></div>'
    return f'<div class="card-img-ph">{icon}</div>'


@st.dialog("Product details", width="large")
def _product_detail_dialog(p, category, breakdown, total, shown):
    st.markdown(f"### {_card_name(p, category)}")
    ci, cm = st.columns([1, 1])
    with ci:
        _render_product_image(p, category)
    with cm:
        price = p.get("price_usd")
        price_str = f"${int(price):,}" if isinstance(price, (int, float)) and price == price else "—"
        st.metric("Price", price_str)
        st.metric("Match score", f"{p.get('_score', '—')} / 100")
        sem = p.get("_semantic")
        if isinstance(sem, (int, float)) and sem > 0:
            st.caption(f"🔎 {sem:.0f}% match to your request")
    desc = p.get("description")
    if isinstance(desc, str) and desc.strip():
        st.markdown(desc.strip())
    real = p.get("real_review")
    if isinstance(real, str) and real.strip():
        url = p.get("real_review_url")
        link = f"  [source ↗]({url})" if isinstance(url, str) and url.startswith("http") else ""
        st.success(f"📝 **GSMArena review** (excerpt):{link}\n\n{real.strip()}")
    else:
        rev = p.get("review_summary")
        if isinstance(rev, str) and rev.strip():
            st.info(f"🤖 **AI summary** (generated from specs & rating): {rev.strip()}")
    st.markdown("**Specifications**")
    _render_comparison_table([p], category)
    st.markdown("**📊 Why this score?**")
    _render_score_breakdown(breakdown, total_count=total)
    if st.button("👍 This is a strong match", use_container_width=True, key="detail_pick"):
        ranking.record_selection(
            category, shown, p,
            query=st.session_state.dialogue_state.get("semantic_query"),
            session_id=st.session_state.get("session_id"),
        )
        st.toast(f"Thanks! Noted your pick: {p.get('model', '')}", icon="👍")


@st.dialog("Compare products", width="large")
def _product_compare_dialog(products, category):
    names = [_card_name(p, category) for p in products]
    st.markdown("#### " + "  vs  ".join(n[:22] for n in names))
    bds = database.score_breakdown(category, products)
    attrs = [r["attr"] for r in bds[0]] if bds and bds[0] else []
    labels = [_SCORE_ATTR_LABELS.get(a, a).split(" ", 1)[-1] for a in attrs]

    # Layer 1 — radar (the shape of each product's strengths)
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for p, bd in zip(products, bds):
            vals = [r["norm_0_100"] for r in bd]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=labels + [labels[0]],
                fill="toself", name=_card_name(p, category)[:20]))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, height=380, margin=dict(l=30, r=30, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each axis = position vs. the whole catalog (100 = best). Bigger shape = stronger all-round.")
    except Exception:
        st.info("Radar needs plotly (`pip install plotly`). Winner table below.")

    # Layer 2 — winner-highlight table (the facts, with 🏆 per spec)
    ncol = len(products)
    html = ['<table class="cmp"><tr><th>Spec</th>']
    html += [f"<th>{n[:22]}</th>" for n in names] + ["</tr>"]
    for ai, attr in enumerate(attrs):
        norms = [bds[pi][ai]["norm_0_100"] for pi in range(ncol)]
        best = max(range(ncol), key=lambda i: norms[i])
        html.append(f'<tr><td class="cmp-attr">{_SCORE_ATTR_LABELS.get(attr, attr)}</td>')
        for pi in range(ncol):
            r = bds[pi][ai]
            val = _format_breakdown_value(attr, r["raw"], r["direction"])
            cls, tr = (" cmp-win", " 🏆") if pi == best else ("", "")
            html.append(f'<td class="cmp-val{cls}">{val}{tr}</td>')
        html.append("</tr>")
    scores = [p.get("_score", 0) or 0 for p in products]
    bestp = max(range(ncol), key=lambda i: scores[i])
    html.append('<tr><td class="cmp-attr">Overall score</td>')
    for pi in range(ncol):
        cls, tr = (" cmp-win", " 🏆") if pi == bestp else ("", "")
        html.append(f'<td class="cmp-val{cls}">{scores[pi]}/100{tr}</td>')
    html.append("</tr></table>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_recommendation_list(products, category, msg_index):
    """
    Grid of product cards (image · name · price · score). Open a card's Details
    for a full-spec modal; tick Compare on 2–3 cards then open a radar +
    winner-highlight comparison modal.
    """
    if not products:
        return
    total = len(products)
    st.caption(
        f"**{total} matching {category}** — ranked by overall value (TOPSIS). "
        "Tap **Details** for full specs, or tick **Compare** on 2–3 and hit Compare."
    )

    options = [n for n in (3, 6, 12) if n < total] + ["All"]
    default = "All" if total <= 6 else 6
    if default not in options:
        default = options[0]
    sel = st.selectbox("Show", options, index=options.index(default), key=f"show_{msg_index}")
    display = products if sel == "All" else products[: int(sel)]

    breakdowns = database.score_breakdown(category, products)

    ncols = 3
    selected = []
    for start in range(0, len(display), ncols):
        cols = st.columns(ncols)
        for ci in range(ncols):
            idx = start + ci
            if idx >= len(display):
                break
            p = display[idx]
            with cols[ci]:
                with st.container(border=True):
                    st.markdown(_card_image_html(p, category), unsafe_allow_html=True)
                    if idx < 3:
                        st.markdown(f'<span class="card-rank">#{idx+1} TOP PICK</span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-title">{_card_name(p, category)}</div>', unsafe_allow_html=True)
                    price = p.get("price_usd")
                    price_str = f"${int(price):,}" if isinstance(price, (int, float)) and price == price else "—"
                    st.markdown(f'<div class="card-meta">{price_str} · ⭐ {p.get("_score","—")}/100</div>',
                                unsafe_allow_html=True)
                    # Stack the actions: side-by-side columns get too narrow
                    # inside a grid card and the button text wraps char-by-char.
                    if st.button("🔍 Details", key=f"det_{msg_index}_{idx}", use_container_width=True):
                        _product_detail_dialog(p, category, breakdowns[idx], total, products)
                    if st.checkbox("Compare", key=f"cmp_{msg_index}_{idx}"):
                        selected.append(idx)

    if len(selected) >= 2:
        if st.button(f"⚖️ Compare {len(selected)} selected", key=f"cmpbtn_{msg_index}",
                     use_container_width=True):
            _product_compare_dialog([display[i] for i in selected[:3]], category)
        if len(selected) > 3:
            st.caption("_Comparing the first 3 selected._")
    elif len(selected) == 1:
        st.caption("_Tick one or two more to compare._")


# ── Quick-reply suggestions for clarification questions ─────────────────────

# Each label is sent back through the LLM pipeline as if the user typed it.
# "Any" triggers the existing skip-detection in state_updater_node.
ATTRIBUTE_SUGGESTIONS = {
    "smartphone": {
        "os":                  ["Android", "iOS", "Other", "Any"],
        "price_usd":           ["Under $200", "$200-500", "$500-1000", "$1000+", "Any"],
        "battery_capacity":    ["3000+ mAh", "4000+ mAh", "5000+ mAh", "Any"],
        "primary_camera_rear": ["12+ MP", "48+ MP", "64+ MP", "100+ MP", "Any"],
        "ram_capacity":        ["4 GB", "6 GB", "8 GB", "12 GB+", "Any"],
        "internal_memory":     ["64 GB", "128 GB", "256 GB", "512 GB+", "Any"],
    },
    "headphones": {
        "type":                ["Wired", "Wireless", "Any"],
        "form_factor":         ["Over-Ear", "On-Ear", "In-Ear", "Any"],
        "noise_cancellation":  ["Yes", "No", "Any"],
        "price_usd":           ["Under $100", "$100-300", "$300+", "Any"],
    },
}


def _render_suggestion_buttons(category, attribute, msg_index):
    """Render a row of quick-reply buttons; return the clicked label or None."""
    options = ATTRIBUTE_SUGGESTIONS.get(category, {}).get(attribute)
    if not options:
        return None
    cols = st.columns(len(options))
    for i, opt in enumerate(options):
        if cols[i].button(opt, key=f"sugg_{msg_index}_{attribute}_{i}", use_container_width=True):
            return opt
    return None


# ── Chat avatars ──────────────────────────────────────────────────────────────
USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "🛍️"

def _avatar(role):
    return USER_AVATAR if role == "user" else ASSISTANT_AVATAR

# ── Sticky top toolbar — always-visible brand + quick actions ─────────────────
with st.container(key="topbar"):
    st.markdown(
        f'<div class="brand">🛍️ {APP_NAME}</div>',
        unsafe_allow_html=True,
    )
    # Greeting sits ABOVE the buttons and stays visible for the whole session.
    st.markdown(
        f'<div class="topbar-greeting">👋 Hi, I\'m <strong>{APP_NAME}</strong> — your assistant for '
        '<strong>smartphones</strong> and <strong>headphones</strong>. Tell me what you\'re looking for and '
        'I\'ll narrow it down. Not sure where to start? Tap <strong>📱 Top Smartphones</strong> or '
        '<strong>🎧 Top Headphones</strong> below to see my top picks, or <strong>ℹ️ How it works</strong>.</div>',
        unsafe_allow_html=True,
    )
    a1, a2, a3, a4 = st.columns(4, gap="small")
    if a1.button("📱 Top Smartphones", use_container_width=True, key="tb_phone", help="See the best smartphones right now"):
        start_top_picks("smartphone")
    if a2.button("🎧 Top Headphones", use_container_width=True, key="tb_head", help="See the best headphones right now"):
        start_top_picks("headphones")
    if a3.button("ℹ️ How it works", use_container_width=True, key="tb_howto"):
        show_how_it_works()
    if a4.button("🔄 Start New Conversation", use_container_width=True, key="tb_new", help="Start a new conversation"):
        reset_conversation()

# Render chat history (with browsable recommendation lists and quick-reply buttons)
_last_idx = len(st.session_state.chat_messages) - 1
_editing_idx = st.session_state.get("editing_idx")
for i, msg in enumerate(st.session_state.chat_messages):
    with st.chat_message(msg["role"], avatar=_avatar(msg["role"])):
        if msg["role"] == "user" and _editing_idx == i:
            # Inline editor — rewind to this point and re-run with the new text.
            new_text = st.text_area(
                "Edit message", value=msg["content"],
                key=f"edit_box_{i}", label_visibility="collapsed",
            )
            be1, be2, _ = st.columns([1, 1, 2], gap="small")
            if be1.button("💾 Save", key=f"edit_save_{i}", type="primary", use_container_width=True):
                if new_text.strip():
                    apply_edit(i, new_text.strip())
            if be2.button("Cancel", key=f"edit_cancel_{i}", use_container_width=True):
                st.session_state.pop("editing_idx", None)
                st.rerun()
        else:
            st.markdown(msg["content"])
            # Tag past assistant replies with the model that produced them
            # (so different models stay distinguishable when comparing).
            if msg["role"] == "assistant" and msg.get("model"):
                st.caption(f"🧠 {msg['model']}")
            # Edit affordance — on user turns, when nothing else is being edited.
            if msg["role"] == "user" and _editing_idx is None:
                if st.button("✏️ Edit", key=f"edit_btn_{i}",
                             help="Edit this message and rewind the conversation to here"):
                    begin_edit(i)
                    st.rerun()
        # Recommendation list — kept per-message so past recommendations
        # stay browsable; each message has its own checkbox state via msg_index.
        if msg.get("recommend_products"):
            st.divider()
            _render_recommendation_list(
                msg["recommend_products"],
                msg["recommend_category"],
                i,
            )
        # Quick-reply buttons: only on the most recent assistant message that's
        # awaiting an answer to a clarification question.
        if (
            i == _last_idx
            and msg["role"] == "assistant"
            and msg.get("clarification_attribute")
        ):
            clicked = _render_suggestion_buttons(
                msg.get("clarification_category"),
                msg["clarification_attribute"],
                i,
            )
            if clicked:
                st.session_state.queued_input = clicked
                st.rerun()
        # Conversation closed → offer a clean fresh start.
        if i == _last_idx and msg.get("closed"):
            if st.button("🎉 Start a new search", key=f"newsearch_{i}", use_container_width=True):
                reset_conversation()

# ── Chat input ────────────────────────────────────────────────────────────────

# Accept either typed input or a queued quick-reply button click.
typed_input = st.chat_input("Tell me what you're looking for...")
queued_input = st.session_state.pop("queued_input", None)
user_input = typed_input or queued_input

if user_input:
    # If the previous turn ended with action="done" (user said thanks / start
    # over / etc.), wipe the conversation and product state BEFORE processing
    # the new message — so this input starts a completely fresh session.
    if st.session_state.pop("pending_reset", False):
        st.session_state.dialogue_state = initial_state()
        st.session_state.chat_messages = []
        st.session_state.show_welcome = False  # the new input is taking welcome's place
    # Display user message immediately
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)
    # Snapshot the dialogue state BEFORE this turn runs, so the user can later
    # edit this message and rewind the conversation to exactly this point.
    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_input,
        "pre_state": copy.deepcopy(st.session_state.dialogue_state),
    })

    # Run the LangGraph pipeline. The interactive recommendation list and the
    # quick-reply buttons are intentionally NOT rendered inline — they have
    # widget keys that would collide if rendered twice. The rerun + history
    # loop below renders them once per pass.
    _active_spec = llm_client.get_active_model()
    model_label = _active_spec["model"]
    spin = f"Thinking… · {model_label}" + (" (local — can take a while)" if _active_spec["local"] else "")
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner(spin):
            try:
                new_state = run_turn(
                    st.session_state.dialogue_state, user_input,
                    session_id=st.session_state.session_id,
                )
                st.session_state.dialogue_state = new_state
                turn_error = None
            except Exception as e:        # local model down, bad key, timeout, …
                turn_error = e
        if turn_error is not None:
            is_timeout = "timeout" in (type(turn_error).__name__ + str(turn_error)).lower()
            if _active_spec["local"] and is_timeout:
                hint = (" It's taking too long on your hardware — a 7B model is heavy. Try a "
                        "lighter one (e.g. **llama3.2** or **phi4-mini**) from the sidebar, or "
                        "raise `OLLAMA_TIMEOUT` in `.env`.")
            elif _active_spec["local"]:
                hint = f" Is Ollama running? Try `ollama run {model_label}` in a terminal."
            else:
                hint = " Check your API key / connection."
            response = (f"⚠️ I couldn't get a response from **{model_label}**.{hint}\n\n"
                        f"You can pick a different model in the sidebar and try again.")
            st.markdown(response)
        else:
            response = new_state["response"]
            st.markdown(response)
        st.caption(f"🧠 {model_label}")

    # Persist the assistant turn, tagged with the model that produced it. Save the
    # ranked recommendation list (recommend turns) or clarification metadata.
    chat_entry = {"role": "assistant", "content": response, "model": model_label}
    if turn_error is not None:
        pass  # error message already stored in chat_entry; no product state changed
    elif new_state["action"] == "recommend" and new_state["candidates"]:
        chat_entry["recommend_products"] = new_state["candidates"]
        chat_entry["recommend_category"] = new_state["category"]
        # Personalization: remember what this user was shown.
        memory.record_recommendation(
            st.session_state.user_name,
            new_state["category"],
            new_state.get("active_filters", {}),
        )
    elif new_state["action"] == "compare" and new_state["candidates"]:
        # Show the compared products as cards too (image · Details · radar compare).
        chat_entry["recommend_products"] = new_state["candidates"]
        chat_entry["recommend_category"] = new_state["category"]
    elif new_state["action"] in ("ask_clarification", "advise") and new_state.get("clarification_attribute"):
        chat_entry["clarification_attribute"] = new_state["clarification_attribute"]
        chat_entry["clarification_category"] = new_state["category"]
    elif new_state["action"] == "done":
        chat_entry["closed"] = True
    st.session_state.chat_messages.append(chat_entry)

    # If the user signaled they're done / want to start over, queue a full
    # reset that will fire on their next message (see top of input handler).
    if turn_error is None and new_state["action"] == "done":
        st.session_state.pending_reset = True

    st.rerun()
