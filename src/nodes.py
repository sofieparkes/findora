"""
nodes.py
All LangGraph node functions. Each takes a DialogueState and returns
an updated DialogueState (partial dict — LangGraph merges automatically).
"""

import re
import json
from typing import Any, Dict

import database
import nlu
import schema
import ranking
import semantic
from llm_client import get_llm
from state import DialogueState

# Phrases that mean "undo the change I just made" (mixed-initiative correction).
_UNDO_RE = re.compile(
    r"\b(undo|ignore that|scratch that|never\s?mind|forget that|go back|revert|take that back)\b",
    re.IGNORECASE,
)

# An explicit price mention (so a real "under $300" survives the superlative guard).
_PRICE_PHRASE_RE = re.compile(
    r"(\$|\d+\s*(usd|dollars?)|\bunder\b|\bbelow\b|\bover\b|\babove\b|\bbudget\b|\bprice\b|"
    r"\bbetween\b|less than|more than|cheaper than|up to)",
    re.IGNORECASE,
)

# "Why did you recommend X / how did you rank these" — the user wants the
# reasoning behind a recommendation already on screen (explainability).
_WHY_RE = re.compile(
    r"\bwhy\b.*\b(recommend\w*|suggest\w*|pick\w*|choose|chose|chosen|rank\w*|"
    r"top|best|first|this|these|that)\b"
    r"|\bhow (?:did|do) you (?:pick|choose|chose|rank|decide|come up|get|end up)\b"
    r"|\bwhat makes\b"
    r"|\b(justify|explain)\b.*\b(recommend\w*|pick\w*|choice|ranking|suggestion|this|these)\b",
    re.IGNORECASE,
)


def _is_why_question(text: str) -> bool:
    """True for 'why did you recommend…/how did you rank…' style questions, plus a
    short bare 'why?' (which, right after results, means 'why these')."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if _WHY_RE.search(t):
        return True
    # short, bare "why…?" — "why?", "why this one?", "why these?". Keep it tight
    # (<=3 words, no negation) so complaints like "why don't you have wireless"
    # are NOT swept in.
    if re.match(r"^why\b[\w\s'?.!]*$", t) and len(t.split()) <= 3:
        return not re.search(r"\b(not|don'?t|do not|can'?t|cannot|won'?t|isn'?t|aren'?t|have)\b", t)
    return False


# The user is UNSURE on the pending question and wants GUIDANCE — distinct from a
# hard skip ("any"/"don't care"). e.g. "I have no idea", "what do you recommend?",
# "you decide", "which is better?". Here we should reason + recommend, not skip.
_ADVICE_RE = re.compile(
    r"\b(no idea|not sure|unsure|don'?t know|do not know|dunno|can'?t decide|hard to say|"
    r"what (?:do|would|should) (?:you|i)\b|which (?:one )?(?:is |would be )?(?:better|best)|"
    r"you (?:choose|decide|pick|recommend|tell me)|your (?:recommendation|suggestion|advice|opinion|call)|"
    r"help me (?:choose|decide|pick|out)|what'?s best|i'?m confused|new to this|"
    r"whatever you (?:think|recommend|suggest)|recommend (?:me )?(?:something|one|anything)|up to you)\b",
    re.IGNORECASE,
)


def _is_advice_request(text: str) -> bool:
    """True when the user is asking the assistant to help them decide the pending
    question, rather than answering it or hard-skipping it."""
    return bool(_ADVICE_RE.search((text or "").strip().lower()))


# The user wants a specific product's SPECIFICATIONS/details (not pictures). Used
# only as a backstop — the router is primary. e.g. "show me specs of the Itel A23",
# "detailed specifications of X", "tell me about it", "what are its specs".
_SPECS_RE = re.compile(
    r"\b(spec|specs|specification|specifications|"
    r"detailed (?:spec|info|specification)|full (?:spec|detail|specification)|"
    r"tell me (?:more )?about|what are (?:its|the) (?:spec|feature)|"
    r"details? (?:of|on|about|for)|more (?:info|details?) (?:on|about|for))\b",
    re.IGNORECASE,
)


def _is_specs_request(text: str) -> bool:
    return bool(_SPECS_RE.search((text or "").lower()))


# The user wants a SIDE-BY-SIDE comparison of two (or three) products. Backstop only.
_COMPARE_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|stacks? up|how does it (?:compare|stack)|"
    r"difference between|which is better between|side by side|pit (?:it )?against)\b",
    re.IGNORECASE,
)


def _is_compare_request(text: str) -> bool:
    return bool(_COMPARE_RE.search((text or "").lower()))


# ── Per-attribute explanations (woven into clarification questions) ──────────

ATTRIBUTE_HINTS: Dict[str, Dict[str, str]] = {
    "smartphone": {
        "os": "Operating system — Android (open, wide hardware choice) or iOS (tight integration with Apple devices). 'Other' covers less common OSes.",
        "price_usd": "Budget in US Dollars. Dataset range is roughly $40 to $2,500.",
        "battery_capacity": "Battery in mAh — higher = longer life. 4,000-5,000 mAh comfortably lasts a day.",
        "primary_camera_rear": "Main rear camera in megapixels. 48-64 MP is mid-range, 100+ MP is flagship territory.",
        "ram_capacity": "RAM in GB — affects multitasking smoothness. 4 GB is okay for casual use, 8-12 GB for heavy multitasking and gaming.",
        "internal_memory": "Internal storage in GB. 64 GB is tight for media, 128 GB is comfortable, 256 GB+ is generous.",
    },
    "headphones": {
        "type": "Wired or Wireless? Wireless gives mobility; wired avoids batteries and pairing.",
        "form_factor": "Over-Ear (most immersive), On-Ear (compact and lighter), or In-Ear (most portable, gym-friendly).",
        "noise_cancellation": "Active noise cancellation blocks ambient noise — great for flights and commutes.",
        "price_usd": "Budget in US Dollars. Dataset range is $50 to $940.",
    },
}


# ── Node 1: Intent Classification + Preference Extraction ────────────────────

def intent_and_extract_node(state: DialogueState) -> dict:
    """
    Two-stage NLU:
      1. nlu.route()        — intent, category, wants_results, sort, confidence.
      2. nlu.extract_slots()— filters for the effective category (only when the
                              intent could carry preferences).
    All extracted slots are then validated by schema.validate_filters(), which
    coerces types, resolves brands/categoricals, range-checks numerics, and
    drops anything unknown or invalid. The persistent `category` is NOT updated
    here — state_updater compares old vs new to detect switches.
    """
    routed = nlu.route(state)
    intent = routed["intent"]
    extracted_category = routed["category"]
    effective_category = extracted_category or state.get("category")

    # ── Reconcile the two context-dependent reasoning intents ────────────────
    # The LLM router is the PRIMARY classifier for 'explain' and 'advise'. Here we
    # only (a) downgrade them when their context is absent, and (b) apply a regex
    # BACKSTOP that catches a clear case the router MISSED — it no longer overrides
    # a router that already classified correctly.
    has_results = bool(state.get("candidates")) and bool(state.get("last_recommend_stats"))
    pending_q = state.get("last_asked_attribute")
    uin = state.get("user_input", "")

    if intent == "explain" and not has_results:
        intent = "specific"          # nothing recommended yet → nothing to explain
    if intent == "advise" and not pending_q:
        intent = "explore"           # "help me decide" with no pending question → normal flow

    if intent not in ("explain", "done") and has_results and _is_why_question(uin):
        intent = "explain"           # backstop: router missed a clear "why did you pick X"
    if intent not in ("advise", "done") and pending_q and _is_advice_request(uin):
        intent = "advise"            # backstop: router missed a clear "I'm not sure, you decide"
    # specs backstop: "show me specs of <product>" is often mis-read as out_of_scope
    # (pictures). Only fire when a real product can actually be resolved from the text.
    if (intent not in ("specs", "done") and effective_category
            and _is_specs_request(uin) and database.find_product(effective_category, uin)):
        intent = "specs"
    # compare backstop: "compare it with one more option" must not fall through to
    # a generic re-recommend. Needs a category (there must be products to compare).
    if intent not in ("compare", "done") and effective_category and _is_compare_request(uin):
        intent = "compare"

    raw_filters: Dict[str, Any] = {}
    clean_filters: Dict[str, Any] = {}
    dropped: list = []
    extracted_skips: list = []
    if intent in ("specific", "explore", "refine") and effective_category:
        raw_filters = nlu.extract_slots(effective_category, state, routed["sort_preference"])
        # Proactive skips ("I don't care about storage") arrive as a "skip" list.
        sk = raw_filters.pop("skip", None)
        if isinstance(sk, list):
            extracted_skips = [database.base_attr(str(a)) for a in sk]
        clean_filters, dropped = schema.validate_filters(effective_category, raw_filters)

        # Guard the "cheapest"/"most expensive" leak: when an ordering preference
        # is set, the price is handled by SORTING, not filtering. The extractor
        # sometimes still emits a percentile price cap. Strip price bounds UNLESS
        # the user gave an explicit price ("under $300", "below 500", "budget").
        # NOT for refine — there the price bound is the relative-critique anchor
        # ("cheaper" → below the previous min), which must survive.
        if intent != "refine" and routed["sort_preference"] in ("price_asc", "price_desc"):
            if not _PRICE_PHRASE_RE.search(state.get("user_input", "") or ""):
                for pk in ("price_usd_min", "price_usd_max"):
                    clean_filters.pop(pk, None)

    return {
        "intent": intent,
        "extracted_filters": clean_filters,
        "extracted_category": extracted_category,
        "also_category": routed.get("also_category"),
        "wants_results": routed["wants_results"],
        "wants_single": routed.get("wants_single", False),
        "sort_preference": routed["sort_preference"],
        "intent_confidence": routed["confidence"],
        "raw_extracted_filters": raw_filters,   # pre-validation, for observability
        "dropped_slots": dropped,               # for observability + confidence
        "extracted_skips": extracted_skips,     # attributes the user said they don't care about
    }


# ── Node 2: Dialogue State Updater ────────────────────────────────────────────

def state_updater_node(state: DialogueState) -> dict:
    """
    Merge newly extracted filters into active_filters, with two extra behaviors:
      1. CATEGORY SWITCH: if the user pivots to a different non-null category,
         reset active_filters and asked_skipped (start fresh).
      2. SKIP DETECTION: if the user was asked about an attribute last turn
         and didn't provide a filter for it (and isn't chitchatting), mark
         that attribute as 'skipped' so we don't ask again.

    Pure Python — no LLM call.
    """
    extracted = state["extracted_filters"]
    extracted_category = state.get("extracted_category")
    current_category = state["category"]
    intent = state["intent"]
    last_asked = state.get("last_asked_attribute")
    new_skips = list(state.get("extracted_skips", []))  # proactive "don't care about X"

    # 0. Mixed-initiative UNDO — checked FIRST (deterministic keyword), before any
    #    intent-based short-circuit, so "ignore that"/"undo" works even if the
    #    router happened to label it chitchat/ambiguous.
    if _UNDO_RE.search(state.get("user_input", "") or "") and state.get("last_filter_delta"):
        reverted = dict(state["active_filters"])
        for k in state["last_filter_delta"]:
            reverted.pop(k, None)
        return {
            "active_filters": reverted,
            "last_filter_delta": [],
            "turn_count": state["turn_count"] + 1,
        }

    # 0.5 Chitchat / ambiguous / out-of-scope / explain / advise short-circuit.
    #     Don't touch product state — these turns don't change preferences. ('advise'
    #     answers "I'm not sure about X" and deliberately keeps the SAME question
    #     open, so it must NOT mark the attribute as skipped or advance.)
    if intent in ("chitchat", "ambiguous", "out_of_scope", "explain", "advise",
                  "specs", "compare"):
        return {"turn_count": state["turn_count"] + 1}

    # 1. Category switch → wipe filters + skipped list, keep only THIS turn's filters.
    #    Guard: only switch to a REAL, known category (never a stray "null" string),
    #    otherwise a parsing slip would silently wipe the user's preferences.
    if (
        extracted_category in database.get_categories()
        and current_category
        and extracted_category != current_category
    ):
        switch_filters = {k: v for k, v in extracted.items() if v is not None}
        uin = state.get("user_input", "")
        return {
            "category": extracted_category,
            "active_filters": switch_filters,
            "asked_skipped": new_skips,
            "candidates": [],
            "last_filter_delta": list(switch_filters.keys()),
            # New category → reset the vibe, keeping only one mentioned this turn.
            "vibe_query": uin if semantic.is_vibe_query(uin) else None,
            "turn_count": state["turn_count"] + 1,
        }

    # 2. Otherwise: decide effective category (set first time, or keep current)
    new_category = extracted_category or current_category

    # 1.5 FRESH SEARCH. A self-contained "specific" request that NAMES A NEW
    #     BRAND (e.g. "recommend me the cheapest iPhones" after an Android search)
    #     starts a NEW search — replace the filters so contradictory leftovers
    #     (apple + android) can't zero out the results. Requiring a brand keeps
    #     pure spec additions like "with 8GB RAM" as a normal merge.
    fresh = {k: v for k, v in extracted.items() if v is not None}
    introduces_brand = any(k in ("brand_name", "brand") for k in fresh)
    if intent == "specific" and not last_asked and introduces_brand:
        uin = state.get("user_input", "")
        return {
            "category": new_category,
            "active_filters": fresh,
            "asked_skipped": new_skips,
            "candidates": [],
            "last_filter_delta": list(fresh.keys()),
            "vibe_query": uin if semantic.is_vibe_query(uin) else state.get("vibe_query"),
            "turn_count": state["turn_count"] + 1,
        }

    # 3. Merge filters (None removes a prior filter). Track which keys ACTUALLY
    #    changed this turn (new value, not an echo of an existing one) so a
    #    follow-up "undo"/"ignore that" reverts exactly those — not unchanged
    #    filters the extractor happened to restate.
    prior = state["active_filters"]
    updated = dict(prior)
    delta: list = []
    for key, value in extracted.items():
        if value is None:
            updated.pop(key, None)
        else:
            if prior.get(key) != value:
                delta.append(key)
            updated[key] = value

    # 4. Skip detection — proactive ("don't care about X") + answered-with-skip.
    updated_skipped = list(state.get("asked_skipped", []))
    for a in new_skips:                       # proactive skips this turn
        if a not in updated_skipped:
            updated_skipped.append(a)
    if (
        last_asked
        and intent not in (None, "chitchat", "done")
        and last_asked not in updated_skipped
        and not any(database.base_attr(k) == last_asked for k in updated)
    ):
        updated_skipped.append(last_asked)

    # Persist a vibe descriptor ("for travel", "gaming") across turns so it still
    # influences the recommendation even if mentioned several turns earlier.
    uin = state.get("user_input", "")
    vibe = uin if semantic.is_vibe_query(uin) else state.get("vibe_query")

    return {
        "category": new_category,
        "active_filters": updated,
        "asked_skipped": updated_skipped,
        "last_filter_delta": delta,
        "vibe_query": vibe,
        "turn_count": state["turn_count"] + 1,
    }


# ── Node 3: Product Retriever + Action Selector ───────────────────────────────

# Below this many matches, asking more questions adds little value — just show
# the shortlist. Above it (and with questions still pending), keep narrowing.
MIN_RESULTS_TO_RECOMMEND = 4


def retrieve_and_act_node(state: DialogueState) -> dict:
    """
    Decide the next system action based on intent + dialogue state.

    Policy:
      - no category yet                → ask_category
      - done / chitchat / summarize    → handled directly (no retrieval)
      - refine                         → re-recommend with the updated filters
        (the user is reacting to a list they've already seen)
      - explore / specific            → INFORMATION-GATHERING GATE:
            recommend only when the user asked to see results (wants_results),
            OR the candidate set is already small (<= MIN_RESULTS_TO_RECOMMEND),
            OR there are no more questions to ask. Otherwise ask one more
            clarifying question. This prevents dumping recommendations right
            after the first reply when the result set is still broad.

    Pure Python — no LLM call.
    """
    intent = state["intent"]
    category = state["category"]

    # Explicit non-search intents are handled FIRST — they must not be swallowed
    # by the low-confidence clarify guard below.

    # User signaled they're done / restarting
    if intent == "done":
        return {
            "action": "done",
            "candidates": [],
            "clarification_attribute": None,
            "last_asked_attribute": None,
        }

    # Off-topic / small-talk — preserve candidates + pending question.
    if intent == "chitchat":
        return {"action": "chitchat"}

    # Request we can't fulfill (images, purchase, stock, warranty, reviews…).
    # Preserve all product context so the reply can reference the current pick.
    if intent == "out_of_scope":
        return {"action": "out_of_scope"}

    # Recap request — keep candidates + clarification context so the response
    # generator can summarize the current preferences without changing them.
    if intent == "summarize":
        return {"action": "summarize"}

    # "Why did you recommend X?" — keep all product context so the response
    # generator can ground the rationale in the candidates already shown.
    if intent == "explain":
        return {"action": "explain"}

    # "I'm not sure about <the pending question>" — advise (reason + recommend),
    # and KEEP the same question open so the quick-reply buttons stay and the
    # user's next reply is still understood as answering it.
    if intent == "advise":
        attr = state.get("last_asked_attribute")
        return {
            "action": "advise",
            "clarification_attribute": attr,
            "last_asked_attribute": attr,
        }

    # "Show me the specs of <product>" — resolve the named product (or 'it' = the
    # current top pick) and present its specifications. A pure lookup: it does NOT
    # change filters, and it KEEPS any pending question so the flow can resume.
    if intent == "specs":
        product = database.find_product(category, state.get("user_input", "")) if category else None
        if product is None and state.get("candidates"):
            product = state["candidates"][0]      # "tell me about it" → current top pick
        if product is None:
            return {"action": "clarify_intent",
                    "clarification_attribute": None, "last_asked_attribute": None}
        return {"action": "specs", "candidates": [product]}

    # "Compare it with one more option" / "Itel A23 vs Galaxy A32" — pick the
    # products to compare (named ones + the current top pick + best alternatives)
    # and present a grounded side-by-side. A lookup: filters are untouched.
    if intent == "compare":
        products = _select_compare_products(state)
        if len(products) < 2:
            return {"action": "clarify_intent",
                    "clarification_attribute": None, "last_asked_attribute": None}
        return {"action": "compare", "candidates": products,
                "clarification_attribute": None, "last_asked_attribute": None}

    # Genuinely unclear message (router said so, or very low confidence) → ask
    # the user to clarify instead of guessing an action or a filter.
    if intent == "ambiguous" or state.get("intent_confidence", 1.0) < nlu.CONFIDENCE_THRESHOLD:
        if not (state.get("extracted_filters") or state.get("active_filters")):
            return {
                "action": "clarify_intent",
                "candidates": [],
                "clarification_attribute": None,
                "last_asked_attribute": None,
            }

    # No category yet → must ask
    if not category:
        return {
            "action": "ask_category",
            "candidates": [],
            "clarification_attribute": None,
            "last_asked_attribute": None,
        }

    filters = state["active_filters"]
    candidates = database.retrieve(category, filters, limit=None)

    if len(candidates) == 0:
        return {
            "action": "no_results",
            "candidates": [],
            "clarification_attribute": None,
            "last_asked_attribute": None,
        }

    # Map an explicit ordering preference ("cheapest" / "most expensive") to a
    # sort key; None falls back to learned/TOPSIS match-score ranking.
    sort_by = {
        "price_asc": ("price_usd", "asc"),
        "price_desc": ("price_usd", "desc"),
    }.get(state.get("sort_preference"))

    # Free-text "vibe" descriptor ("good for travel", "gaming") → semantic blend.
    # Use this turn's text if it's a vibe, else the one persisted from earlier.
    user_text = state.get("user_input", "")
    semantic_query = user_text if semantic.is_vibe_query(user_text) else state.get("vibe_query")

    # Singular superlative ("the cheapest", "the best one") → present just 1.
    wants_single = state.get("wants_single", False)

    def _recommend_return(ranked):
        if wants_single:
            ranked = ranked[:1]
        return {
            "action": "recommend",
            "candidates": ranked,
            "clarification_attribute": None,
            "last_asked_attribute": None,
            "semantic_query": semantic_query,
            # Save stats of THIS set so the next turn can anchor relative
            # critiques ('cheaper', 'bigger battery') against it.
            "last_recommend_stats": database.candidate_stats(category, ranked),
        }

    # REFINE intent → only valid if products were ALREADY shown. Otherwise the
    # user is still in the guided flow (e.g. answering a question), so fall
    # through to the information-gathering gate instead of recommending early.
    if intent == "refine" and state.get("last_recommend_stats"):
        ranked = ranking.rank(category, candidates, sort_by=sort_by, semantic_query=semantic_query)
        return _recommend_return(ranked)

    # EXPLORE / SPECIFIC (and refine-with-no-prior-results) → information gate.
    # Keep narrowing unless the user wants results now, asked for a single item,
    # the set is already small, or nothing's left to ask.
    next_q = database.next_question(
        category, filters, state.get("asked_skipped", [])
    )
    # How many concrete criteria the user front-loaded THIS message.
    turn_filters = sum(1 for v in state.get("extracted_filters", {}).values() if v is not None)
    ready_to_recommend = (
        state.get("wants_results", False)
        or wants_single
        or next_q is None
        or len(candidates) <= MIN_RESULTS_TO_RECOMMEND
        # Enough signal to recommend — the user knows what they want; don't
        # interrogate. Either 2+ criteria already active on a specific request,
        # or 2+ criteria given in a single message (robust to intent variance).
        or (intent == "specific" and len(filters) >= 2)
        or turn_filters >= 2
    )
    if ready_to_recommend:
        ranked = ranking.rank(category, candidates, sort_by=sort_by, semantic_query=semantic_query)
        return _recommend_return(ranked)

    return {
        "action": "ask_clarification",
        "candidates": candidates,
        "clarification_attribute": next_q,
        "last_asked_attribute": next_q,
    }


# ── Recommendation explainability ("why did you recommend X?") ────────────────

# Human-readable labels for the scoreable attributes used in the value ranking.
_EXPLAIN_LABELS = {
    "price_usd": "price",
    "primary_camera_rear": "rear camera",
    "battery_capacity": "battery capacity",
    "ram_capacity": "RAM",
    "internal_memory": "storage",
    "rating": "user rating",
    "avg_rating": "average rating",
    "battery_hrs": "battery life",
    "freq_range": "frequency range",
    "noise_cancellation": "noise cancellation",
}


def _fmt_spec(attr: str, raw: Any) -> str:
    """Format a raw attribute value with its unit for human-readable prose."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return "—"
    if attr == "noise_cancellation":
        return "yes" if (raw is True or raw == 1.0 or str(raw).lower() == "true") else "no"
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return str(raw)
    if attr == "price_usd":
        return f"${x:,.0f}"
    if attr == "primary_camera_rear":
        return f"{x:.0f} MP"
    if attr == "battery_capacity":
        return f"{x:,.0f} mAh"
    if attr in ("ram_capacity", "internal_memory"):
        return f"{x:.0f} GB"
    if attr == "battery_hrs":
        return f"{x:.0f} hrs"
    if attr in ("avg_rating", "rating"):
        return f"{x:.1f}"
    if attr == "freq_range":
        return f"{x:,.0f} Hz"
    return f"{x:g}"


def _product_display_name(p: Dict, category: str) -> str:
    brand = str((p.get("brand_name") if category == "smartphone" else p.get("brand")) or "").strip()
    model = str(p.get("model") or "").strip()
    if model and model.lower().startswith(brand.lower()):
        return model
    return f"{brand} {model}".strip() or "this one"


def _resolve_explained_product(state: DialogueState):
    """Pick which shown product the user is asking about (fuzzy match on the
    name/model in their message); default to the top-ranked pick if none named."""
    cands = state.get("candidates") or []
    if not cands:
        return None, 0
    category = state.get("category")
    text = (state.get("user_input") or "").lower()
    text_toks = set(re.findall(r"[a-z0-9]+", text))
    best_i, best_overlap = 0, 0.0
    for i, p in enumerate(cands):
        name = _product_display_name(p, category).lower()
        model = str(p.get("model") or "").lower()
        if model and model in text:          # exact model mention → certain
            return p, i
        if name and name in text:
            return p, i
        name_toks = set(re.findall(r"[a-z0-9]+", name))
        if not name_toks:
            continue
        overlap = len(name_toks & text_toks) / len(name_toks)
        if overlap > best_overlap:
            best_i, best_overlap = i, overlap
    if best_overlap >= 0.5:                   # enough of the name matched
        return cands[best_i], best_i
    return cands[0], 0                         # no specific product → the top pick


def _build_rationale(state: DialogueState) -> str:
    """Grounded facts for the 'why' answer: rank, value score, the attributes
    that make this product strong (vs. the whole catalog), and one honest
    trade-off. Returned as a compact block the response LLM phrases naturally."""
    cands = state.get("candidates") or []
    category = state.get("category")
    if not cands or not category:
        return ""
    product, idx = _resolve_explained_product(state)
    if product is None:
        return ""
    rows = database.score_breakdown(category, [product])
    breakdown = rows[0] if rows else []

    def _qual(norm):
        if norm >= 80:
            return "among the best in the catalog"
        if norm >= 60:
            return "above average"
        if norm >= 45:
            return "around average"
        return "below average"

    ranked_attrs = sorted(breakdown, key=lambda r: r.get("norm_0_100", 50), reverse=True)
    strengths = [r for r in ranked_attrs if r.get("norm_0_100", 50) >= 58][:3]
    weak = [r for r in reversed(ranked_attrs) if r.get("norm_0_100", 50) <= 38][:1]

    name = _product_display_name(product, category)
    score = product.get("_score")
    price = product.get("price_usd")
    rank_txt = f"ranked #{idx + 1} of {len(cands)}" if len(cands) > 1 else "the single best match"
    head = f"PRODUCT: {name} - {rank_txt} by overall value"
    if isinstance(score, (int, float)):
        head += f"; value score {score:.0f}/100"
    if isinstance(price, (int, float)) and price == price:
        head += f"; price ${price:,.0f}"

    lines = [head]
    if strengths:
        lines.append("STRENGTHS (the main reasons it ranks well):")
        for r in strengths:
            label = _EXPLAIN_LABELS.get(r["attr"], r["attr"])
            lines.append(f"  - {label}: {_fmt_spec(r['attr'], r.get('raw'))} ({_qual(r.get('norm_0_100', 50))})")
    if weak:
        r = weak[0]
        label = _EXPLAIN_LABELS.get(r["attr"], r["attr"])
        lines.append(
            f"TRADE-OFF (mention this honestly): {label} is "
            f"{_fmt_spec(r['attr'], r.get('raw'))} ({_qual(r.get('norm_0_100', 50))})."
        )
    lines.append(
        "METHOD: items are ranked by an overall value score that weighs specs against "
        "price - lower price and stronger specs score higher."
    )
    return "\n".join(lines)


# ── Product spec sheet (for the "show me specs of X" lookup) ──────────────────

# Which attributes to show, in order, per category — (column, human label).
_SPEC_FIELDS = {
    "smartphone": [
        ("os", "OS"), ("price_usd", "Price"), ("ram_capacity", "RAM"),
        ("internal_memory", "Storage"), ("battery_capacity", "Battery"),
        ("fast_charging", "Fast charging"), ("primary_camera_rear", "Rear camera"),
        ("primary_camera_front", "Front camera"), ("num_rear_cameras", "Rear cameras"),
        ("screen_size", "Screen"), ("rating", "Rating"),
    ],
    "headphones": [
        ("price_usd", "Price"), ("type", "Type"), ("form_factor", "Form factor"),
        ("connectivity", "Connectivity"), ("noise_cancellation", "Noise cancellation"),
        ("microphone", "Microphone"), ("battery_hrs", "Battery life"),
        ("foldable", "Foldable"), ("avg_rating", "Rating"), ("release_year", "Released"),
    ],
}


def _fmt_spec_full(attr: str, raw: Any):
    """Format any spec value with its unit; None when missing/blank."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return None
    if attr in ("noise_cancellation", "microphone", "foldable"):
        return "Yes" if (raw is True or raw == 1.0 or s.lower() in ("true", "yes", "1")) else "No"
    if attr in ("type", "form_factor", "connectivity"):
        return s
    if attr == "os":
        return "iOS" if s.lower() in ("ios", "apple") else s.title()
    if attr == "release_year":
        try:
            return str(int(float(raw)))
        except (TypeError, ValueError):
            return s
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return s
    return {
        "price_usd": f"${x:,.0f}",
        "ram_capacity": f"{x:.0f} GB",
        "internal_memory": f"{x:.0f} GB",
        "battery_capacity": f"{x:,.0f} mAh",
        "fast_charging": f"{x:.0f} W",
        "primary_camera_rear": f"{x:.0f} MP",
        "primary_camera_front": f"{x:.0f} MP",
        "num_rear_cameras": f"{x:.0f}",
        "screen_size": f'{x:g}"',
        "battery_hrs": f"{x:.0f} hrs",
        "rating": f"{x:.0f}/100",
        "avg_rating": f"{x:.1f}/5",
    }.get(attr, f"{x:g}")


def _build_spec_sheet(product: Dict, category: str) -> str:
    """Grounded, formatted spec list for one product (no LLM, no invention)."""
    name = _product_display_name(product, category)
    lines = [f"PRODUCT: {name}"]
    for attr, label in _SPEC_FIELDS.get(category, []):
        val = _fmt_spec_full(attr, product.get(attr))
        if val is not None:
            lines.append(f"  - {label}: {val}")
    return "\n".join(lines)


# ── Product comparison ("compare it with one more option") ────────────────────

def _product_key(p: Dict) -> str:
    return f"{p.get('brand_name') or p.get('brand', '')}|{p.get('model', '')}".strip().lower()


def _select_compare_products(state: DialogueState) -> list:
    """Choose 2-3 products to compare: any explicitly NAMED in the message, the
    current top pick ('it'), then the best remaining alternatives for the active
    filters (vibe-aware). De-duplicated, order-stable."""
    category = state.get("category")
    if not category:
        return []
    uin = (state.get("user_input") or "").lower()
    want = 3 if re.search(r"\b(three|top\s*3|3)\b", uin) else 2

    selected, have = [], set()
    def _add(p):
        if p is not None and _product_key(p) not in have:
            selected.append(p)
            have.add(_product_key(p))

    # 1) a product explicitly NAMED in the message (require a strong match)
    _add(database.find_product(category, uin, min_score=4.0))
    # 2) the focal / current top pick ("it" / "this one")
    cur = state.get("candidates") or []
    if cur:
        _add(cur[0])
    # 3) fill with the best alternatives from the (vibe-aware) ranked pool
    if len(selected) < want:
        try:
            pool = ranking.rank(
                category,
                database.retrieve(category, state.get("active_filters", {}), limit=None),
                semantic_query=state.get("vibe_query"),
                n=12,
            )
        except Exception:
            pool = cur
        for p in pool:
            if len(selected) >= want:
                break
            _add(p)
    return selected[:want]


def _build_comparison(products: list, category: str) -> str:
    """Grounded side-by-side: each spec for every product, with the per-attribute
    winner (direction-aware) and the overall value score. No LLM, no invention."""
    names = [_product_display_name(p, category) for p in products]
    weights = database._WEIGHTS_BY_CATEGORY.get(category, {})
    lines = ["COMPARING: " + "  vs  ".join(names)]
    for attr, label in _SPEC_FIELDS.get(category, []):
        raws = [p.get(attr) for p in products]
        vals = [_fmt_spec_full(attr, r) for r in raws]
        if all(v is None for v in vals):
            continue
        row = f"  - {label}: " + "  |  ".join(
            f"{n} = {v or '—'}" for n, v in zip(names, vals))
        spec = weights.get(attr)
        if spec:  # a scoreable attribute → mark the winner
            direction = spec[1]
            nums = []
            for r in raws:
                try:
                    nums.append(float(r))
                except (TypeError, ValueError):
                    nums.append(None)
            if all(x is not None for x in nums) and len(set(nums)) > 1:
                best_i = (min if direction == "lower" else max)(
                    range(len(nums)), key=lambda i: nums[i])
                row += f"   -> better: {names[best_i]}"
        lines.append(row)
    if all(isinstance(p.get("_score"), (int, float)) for p in products):
        lines.append("  - Overall value score: " + ", ".join(
            f"{n} {p['_score']:.0f}/100" for n, p in zip(names, products)))
    return "\n".join(lines)


# ── Node 4: Response Generator ────────────────────────────────────────────────

def response_generator_node(state: DialogueState) -> dict:
    """
    Generates the final natural-language assistant reply.
    One LLM call, guided by the action and current state.
    """
    action = state["action"]
    category = state["category"] or "products"
    # Human-readable plural for the assistant prose (avoids "headphoness").
    CATEGORY_LABEL = {
        "smartphone": "smartphones",
        "headphones": "headphones",
    }
    category_label = CATEGORY_LABEL.get(category, category)
    filters_summary = json.dumps(state["active_filters"], indent=2) if state["active_filters"] else "none"

    # For the LLM prompt, only show the top 2 (it doesn't need to know about
    # all 50). The full ranked list is in state["candidates"] for the UI.
    top_candidates = state["candidates"][:2]
    total_matches = len(state["candidates"])
    candidates_text = ""
    if top_candidates:
        lines = []
        for p in top_candidates:
            key_fields = {k: v for k, v in p.items() if k != "id"}
            lines.append(json.dumps(key_fields, default=str))
        candidates_text = "\n".join(lines)

    # Look up the explanation for the attribute we're about to ask about
    attr = state.get("clarification_attribute")
    attr_hint = ATTRIBUTE_HINTS.get(category, {}).get(attr, "") if attr else ""

    # For chitchat: optionally reference the pending clarification question
    pending_q = state.get("last_asked_attribute")
    chitchat_reprompt = (
        f" They were in the middle of choosing — you'd just asked about '{pending_q}'. Gently invite them back to that question."
        if pending_q else
        " Invite them to tell you what they're looking for."
    )

    # Grounded facts for an explainability ("why?") answer.
    rationale = _build_rationale(state) if action == "explain" else ""

    # Recent conversation, so an ADVISORY / COMPARISON reply can reason about the
    # customer's own situation (their job, use-case, budget hints stated earlier).
    history_block = ""
    if action in ("advise", "compare"):
        recent = state.get("messages", [])[-6:]
        convo = "\n".join(
            f"{'Customer' if m.get('role') == 'user' else 'You'}: {m.get('content', '')}"
            for m in recent
        )
        history_block = ("\nConversation so far (reason from what they've told you about "
                         f"themselves):\n{convo}\n")

    # Grounded spec sheet for a "show me the specs of X" lookup.
    spec_sheet = ""
    if action == "specs" and state.get("candidates"):
        spec_sheet = _build_spec_sheet(state["candidates"][0], category)

    # Grounded side-by-side facts for a "compare X with Y" request.
    comparison = ""
    if action == "compare" and len(state.get("candidates", [])) >= 2:
        comparison = _build_comparison(state["candidates"], category)

    # Build action-specific instructions
    action_instructions = {
        "ask_category": (
            "Greet the user warmly and ask which product category they are interested in. "
            "Mention the available categories: smartphones or headphones."
        ),
        "ask_clarification": (
            f"Ask the user about ONE attribute: '{attr}'.\n"
            f"Context to weave in (briefly): {attr_hint}\n"
            "Phrase a single natural question — 2 to 3 sentences total. Briefly explain the trade-off using the context above, "
            "but don't over-list options. Mention casually that the user can say 'any' or 'skip' to move on. "
            "Ask ONLY about this one attribute — do NOT preview or mention any other attribute or 'next option'."
        ),
        "advise": (
            f"The user is UNSURE and wants your help deciding ONE attribute: '{attr}'. They are NOT "
            f"skipping it — they want guidance. Reason it through FOR them using what they've told you "
            f"about themselves (their job, how they'll use the device, any budget hints — see the "
            f"conversation below).\n"
            f"In 2-4 warm, concrete sentences: (1) briefly explain in plain terms what '{attr}' means "
            f"and why it matters for someone like them ({attr_hint}); (2) RECOMMEND one specific, "
            f"sensible option for their situation and say why in a line; (3) invite them to go with "
            f"your suggestion, pick another, or skip. Be decisive — do NOT just list the options or "
            f"hand the question back without making a recommendation."
        ),
        "specs": (
            "The user asked to SEE the specifications of a specific product — you CAN do this. "
            "Open with the product name in bold, then present the specs below as a clean bulleted "
            "list (you MAY include the exact specs and price). Use ONLY the grounded specs provided "
            "— do not invent or drop any. Optionally finish with ONE short sentence on how it suits "
            "their needs. Do NOT claim you can't show specs, and do NOT mention images/pictures.\n"
            f"GROUNDED SPECS:\n{spec_sheet}"
        ),
        "compare": (
            "The user asked to COMPARE products. Using ONLY the grounded comparison below, write a "
            "clear side-by-side: name the products (brand + model), then the 3-5 most decision-"
            "relevant differences (say which one wins each). Finish with a clear VERDICT — which you'd "
            "pick for THEIR situation (use the conversation context, e.g. their job) and why, in one "
            "or two sentences. You MAY cite specs, prices, and the value score. The cards appear below "
            "your reply. Do not invent anything beyond the facts.\n"
            f"GROUNDED COMPARISON:\n{comparison}"
        ),
        "recommend": (
            (
                f"You found the single best match for the user's request. A card with its full "
                f"specs appears below your reply. DO NOT list specs, prices, or scores in your text. "
                f"Write 1-2 sentences presenting this ONE pick (brand + model only) and why it fits, "
                f"then ask if they'd like to go with it, see other options, or change anything."
                if total_matches == 1 else
                f"You found {total_matches} matching {category_label}, ranked best-first. A browsable "
                f"list appears below your reply — each item expands to full specs, and the user can tick "
                f"2-3 boxes to compare side by side. DO NOT list specs, prices, or scores in your text. "
                f"Write 1-2 sentences acknowledging the {total_matches} options and naming the top pick "
                f"(brand + model only) as a starting suggestion."
            )
            + " End by offering the natural next steps in ONE sentence: refine the search, compare a "
              "few, or say they'll take one / they're all set when happy."
        ),
        "no_results": (
            "No products match the current preferences. Apologize briefly. "
            "Identify the most restrictive preference and suggest relaxing it. "
            "Ask if they'd like to adjust."
        ),
        "explain": (
            "The user asked WHY you recommended a product, or how you ranked them. "
            "Using ONLY the grounded facts below, explain your reasoning warmly and clearly: "
            "name the product (brand + model), give the 2-3 concrete reasons it ranks well "
            "(the specs that stand out and its value for the price), and honestly note the one "
            "trade-off. You MAY refer to specific specs and the overall 'value score'. Keep it to "
            "3-4 sentences. End by inviting them to compare it with another option or refine. "
            "Do NOT invent anything beyond the facts below.\n"
            f"GROUNDED FACTS:\n{rationale}"
        ),
        "summarize": (
            f"The user asked you to recap what you understand so far. "
            f"Give a clear plain-English summary of the {category_label} category "
            f"and the active preferences (listed in the session context). "
            f"Format: 1-2 sentences naming the criteria. End with 'Is that right, "
            f"or would you like to change anything?'. Do not list raw filter keys "
            f"like 'price_usd_max' — translate to natural phrasing (e.g. "
            f"'under $300', 'at least 8 GB RAM')."
        ),
        "done": (
            "The user is wrapping up, has chosen a product, or wants to start over. "
            "If they named a product they're taking, warmly congratulate them on the choice "
            "(use brand + model only). Otherwise thank them for shopping with you. "
            "In ONE or TWO short sentences total, then say they can start a fresh search any time. "
            "DON'T mention resets or technical internals — keep it warm and human."
        ),
        "chitchat": (
            "The user said something off-topic — a greeting, thanks, or small talk. "
            "Reply warmly in ONE short sentence acknowledging what they said. "
            "Then in ONE sentence politely remind them that you're here to help find a smartphone or a pair of headphones."
            + chitchat_reprompt
            + " Keep the whole reply to 2-3 sentences. Do NOT pretend to have product results — there are none to present."
        ),
        "clarify_intent": (
            "You're not fully sure what the user meant. In ONE or TWO friendly sentences, "
            "ask them to clarify what they're looking for, and remind them you can help with "
            "smartphones or headphones. Do not guess or present any products."
        ),
        "out_of_scope": (
            "The user asked for something you genuinely CANNOT do — such as showing photos/images/"
            "videos, buying/checkout, stock/availability, warranty/returns/shipping, or external "
            "reviews. Be honest and warm in ONE sentence that you can't do that here. Then, in ONE "
            "sentence, pivot to what you CAN do: share full specifications, compare options side by "
            "side, or narrow down by features. If a product is in the context below, refer to it by "
            "brand + model. Do NOT pretend to provide the unavailable thing, and do NOT invent a link."
        ),
    }.get(action, "Respond helpfully to the user.")

    # Multi-intent: the user also mentioned a SECOND category this message.
    # Acknowledge it so the request isn't dropped, but handle one at a time.
    also = state.get("also_category")
    also_offer = ""
    if also and action in ("recommend", "ask_clarification"):
        also_label = CATEGORY_LABEL.get(also, also)
        also_offer = (
            f"\nThe user ALSO mentioned {also_label} in the same message. Add ONE short "
            f"closing sentence letting them know you can help with {also_label} next — "
            f"they just need to say so. Handle one category at a time."
        )

    prompt = f"""You are a friendly, knowledgeable shop assistant helping a customer find {category_label}.

Your task: {action_instructions}{also_offer}

Current session context:
- Category: {category}
- Active preferences: {filters_summary}
- Turn number: {state['turn_count']}

{"Products currently shown (context):" if top_candidates else ""}
{candidates_text}
{history_block}
Rules:
- Be concise. For ask_clarification: 2-3 sentences. For recommend: 1-2 sentences plus a closing line.
- Never mention "filters", "database", "JSON", raw field/attribute names, or technical internals.
  (For an explanation you MAY refer to the overall "value score" and concrete specs.)
- NEVER write the words "null" or "None".
- Sound like a real shop assistant, not a chatbot.
- Do not repeat the user's exact words back to them.

Write the assistant reply only, no preamble."""

    response = get_llm().invoke(prompt)

    return {"response": response.content.strip()}
