"""
state.py
Central dialogue state definition shared by all LangGraph nodes.
"""

from typing import TypedDict, Optional, List, Dict, Any


class DialogueState(TypedDict):
    # ── Conversation history ──────────────────────────────────────────────────
    messages: List[Dict[str, str]]   # [{"role": "user"|"assistant", "content": "..."}]
    user_input: str                  # latest raw user message

    # ── Intent & extraction (refreshed every turn) ────────────────────────────
    intent: Optional[str]            # explore | specific | refine | summarize | done | chitchat
    extracted_filters: Dict[str, Any]  # filters extracted from THIS turn only
    extracted_category: Optional[str]  # category mentioned in THIS turn (None if not mentioned)
    also_category: Optional[str]       # a SECOND category mentioned same message (multi-intent), to offer next
    wants_results: bool              # True if the user is asking to SEE recommendations now
                                     # (vs. still describing preferences). Gates premature recommending.
    wants_single: bool               # True if the user asked for ONE item ("the cheapest") → show top 1
    sort_preference: Optional[str]   # price_asc | price_desc | None — explicit ordering
                                     # ("cheapest" / "most expensive"); None = rank by match score
    intent_confidence: float         # router's confidence in the intent (0-1); low → clarify
    raw_extracted_filters: Dict[str, Any]  # pre-validation slots from the extractor (for logging)
    dropped_slots: List[Dict[str, Any]]    # slots dropped by validation: [{key,value,reason}]
    extracted_skips: List[str]             # attrs the user proactively declined ("don't care about storage")
    semantic_query: Optional[str]    # free-text 'vibe' descriptor blended into ranking, if any
    vibe_query: Optional[str]        # PERSISTENT vibe descriptor — survives across turns so a
                                     # "for travel" said early still influences the later recommendation

    # ── Persistent dialogue state (accumulated across turns) ──────────────────
    category: Optional[str]          # smartphone | headphones | None
    active_filters: Dict[str, Any]   # merged filters from all turns so far

    # ── Action & retrieval ────────────────────────────────────────────────────
    action: Optional[str]            # ask_category | ask_clarification | recommend | no_results | done
    candidates: List[Dict[str, Any]] # products matching current filters (up to 10)
    clarification_attribute: Optional[str]  # which attribute to ask about next
    last_asked_attribute: Optional[str]     # attribute asked in the PREVIOUS turn (for skip detection)
    asked_skipped: List[str]                # attributes the user explicitly declined to filter on
    last_recommend_stats: Optional[Dict[str, Dict[str, float]]]  # min/max/median of last recommendation set per scoreable attr — used to anchor refine-intent critiques like "cheaper" / "bigger battery"
    last_filter_delta: List[str]            # filter keys added/changed last turn — lets "undo"/"ignore that" revert them

    # ── Response ──────────────────────────────────────────────────────────────
    response: str                    # final assistant message shown to the user
    turn_count: int


def initial_state() -> DialogueState:
    """Returns a blank starting state for a new session."""
    return DialogueState(
        messages=[],
        user_input="",
        intent=None,
        extracted_filters={},
        extracted_category=None,
        also_category=None,
        wants_results=False,
        wants_single=False,
        sort_preference=None,
        intent_confidence=1.0,
        raw_extracted_filters={},
        dropped_slots=[],
        extracted_skips=[],
        semantic_query=None,
        vibe_query=None,
        category=None,
        active_filters={},
        action=None,
        candidates=[],
        clarification_attribute=None,
        last_asked_attribute=None,
        asked_skipped=[],
        last_recommend_stats=None,
        last_filter_delta=[],
        response="",
        turn_count=0,
    )