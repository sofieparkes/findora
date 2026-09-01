"""
nlu.py
Natural-language understanding, split into two focused stages instead of one
fragile mega-prompt:

  1. route()        — a small, fast classifier: intent, category, wants_results,
                      sort_preference, and a self-reported confidence.
  2. extract_slots()— given a known category, pull ONLY structured filters.

Keeping routing separate from slot extraction makes each prompt smaller and
more reliable, and lets a low routing confidence trigger a clarifying question
instead of a wrong action. All slot output is still validated downstream by
schema.validate_filters().
"""

import json
from typing import Any, Dict, List

import database
import schema
from llm_client import get_llm, parse_json_response

# Routing confidence below this is treated as "unsure" → ask to clarify.
CONFIDENCE_THRESHOLD = 0.45


def _recent_history(messages: List[Dict[str, str]], n: int = 6) -> str:
    lines = []
    for msg in messages[-n:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) or "(start of conversation)"


# ── Stage 1: Router ──────────────────────────────────────────────────────────

def route(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify the user's latest message. Returns:
      intent, category (or None), wants_results, sort_preference, confidence.
    """
    history = _recent_history(state["messages"])
    pending_q = state.get("last_asked_attribute")
    has_shown_results = bool(state.get("last_recommend_stats"))
    context_note = ""
    if pending_q:
        context_note += (
            f"\nIMPORTANT CONTEXT: the assistant JUST asked the user about '{pending_q}'. "
            "If this message is the user ANSWERING that question (a value, a range, a "
            "preference, or 'any'/'skip'), the intent is 'specific' (or 'explore' if no "
            "concrete value) and wants_results MUST be false — they are still in the "
            "guided flow, NOT asking to see results, and it is NOT 'refine'. "
            "BUT if the user is UNSURE about it or asks YOU to choose/recommend for it "
            "('no idea', 'what do you recommend', 'you decide', 'which is better', "
            "'I'm new to this'), the intent is 'advise' — NOT 'specific'."
        )
    if not has_shown_results:
        context_note += (
            "\nNo products have been recommended yet this conversation, so 'refine' does "
            "NOT apply — there is nothing to refine. Use specific/explore instead. "
            "'explain' also does not apply yet (nothing has been recommended to explain)."
        )
    else:
        context_note += (
            "\nProducts HAVE been recommended. If the user asks WHY you recommended "
            "something, HOW you ranked them, or what makes an item good, the intent is 'explain'."
        )

    prompt = f"""You are the ROUTER of a product-recommendation assistant. You ONLY classify
the user's latest message — you do NOT extract product specifications.

Return a JSON object with EXACTLY these fields:
{{
  "intent": "<explore | specific | refine | summarize | done | chitchat | ambiguous | out_of_scope | explain | advise | specs | compare>",
  "category": "<smartphone | headphones | null>",
  "also_category": "<smartphone | headphones | null>",
  "wants_results": <true | false>,
  "wants_single": <true | false>,
  "sort_preference": "<price_asc | price_desc | null>",
  "confidence": <0.0-1.0>
}}

Intent:
- specific  : names concrete criteria, OR answers the assistant's question. e.g. "a Samsung phone", "8GB RAM", "around $300", "android".
- explore   : category only, no criteria. e.g. "I want a phone", "recommend headphones".
- refine    : reacting to a list ALREADY SHOWN. e.g. "cheaper ones", "better camera", "I prefer iPhone", "forget the brand". ONLY valid after a recommendation was made.
- explain   : asks WHY a shown item was recommended, HOW you ranked them, or what makes one good. e.g. "why this one?", "why did you recommend the Creative Max 1000", "how did you pick these", "what makes it the best". ONLY valid after a recommendation has been shown.
- advise    : while a question is PENDING, the user is UNSURE and wants YOU to help decide it — e.g. "I have no idea", "what do you recommend?", "you choose", "which is better?", "I'm new to this", "not sure, you pick". This is NOT skipping: "any" / "doesn't matter" / "don't care" / "skip" / "no preference" means they're fine with anything → that's 'specific'. Use 'advise' ONLY when a question is pending.
- specs     : the user wants the SPECIFICATIONS / details of a specific product — named, or "it"/"this one". e.g. "show me specs of the Itel A23", "detailed specifications of the Galaxy A32", "tell me about it", "what are its specs", "full specs". Specifications ARE available, so this is 'specs' — NOT out_of_scope.
- compare   : the user wants a SIDE-BY-SIDE comparison of two (or three) products. e.g. "compare it with one more option", "compare these two", "Itel A23 vs Galaxy A32", "how does it stack up against the Samsung", "show me the difference between them", "compare the top 3". Use 'compare' even if only one product is named plus "another"/"one more".
- summarize : asks for a recap. e.g. "what do you have so far?", "summarize".
- done      : finished, satisfied, picking one, or restarting. e.g. "thanks bye", "start over", "reset", "I'll take it", "I'll go with the Samsung", "that's all", "perfect, that works", "this one".
- chitchat  : clearly social — greetings/thanks/jokes/off-topic. NOT about products.
- ambiguous : product-related but too vague to act on. e.g. "hmm idk something", "i need a gadget".
- out_of_scope : a product-related request this assistant CANNOT fulfill with the data it has.
              ONLY: pictures/photos/images/videos; buying/ordering/checkout/cart/payment;
              stock/availability/"where to buy"; warranty/returns/shipping/delivery; written user
              reviews or news; price history/deals/coupons.
              e.g. "show me pictures of this", "can I buy it here", "is it in stock", "what's the warranty".
              NOTE: specs, prices, star-RATINGS, brand, OS, battery, camera, etc. ARE available. Asking
              for a product's SPECIFICATIONS/details ("show me the specs of X", "tell me about it") is
              'specs', NOT out_of_scope. "show me PICTURES/photos/images of X" IS out_of_scope; "show me
              the SPECS of X" is NOT. Other spec/price questions are 'specific' (or 'explore').

category: the product type mentioned THIS turn (the PRIMARY one if several), or null.
  ("iphone"/"galaxy"/"pixel" imply smartphone; "earbuds"/"headphones" imply headphones.)
also_category: a SECOND, different product type mentioned in the same message, else null.
wants_results: true ONLY if the user explicitly asks to SEE recommendations now ("show me",
  "what do you recommend", "just show me options"). false when answering a question or
  still describing preferences. Default false.
wants_single: true if the user asks for ONE specific item — "the cheapest", "the best phone",
  "which one should I get", "recommend me one", "your top pick". false for plural ("cheap phones").
sort_preference: "price_asc" for "cheapest/lowest price"; "price_desc" for "most expensive/priciest"; else null.
confidence: how sure you are about "intent" (0=guessing, 1=certain). Use LOW (<0.45) for vague messages.
{context_note}

Conversation so far:
{history}

User's latest message: "{state['user_input']}"

Return ONLY the JSON object."""

    raw = get_llm().invoke(prompt)
    parsed = parse_json_response(getattr(raw, "content", "") or "")

    intent = parsed.get("intent", "explore")
    if intent not in ("explore", "specific", "refine", "summarize", "done",
                      "chitchat", "ambiguous", "out_of_scope", "explain", "advise",
                      "specs", "compare"):
        intent = "explore"

    sort_pref = parsed.get("sort_preference")
    if sort_pref not in ("price_asc", "price_desc"):
        sort_pref = None

    try:
        confidence = float(parsed.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))

    # LLMs sometimes emit the literal string "null"/"none" instead of JSON null.
    # Coerce those (and any non-category) to a real None so they don't get
    # mistaken for a category switch downstream.
    def _clean_cat(v):
        if v is None:
            return None
        s = str(v).strip().lower()
        return v if s in ("smartphone", "headphones") else None

    category = _clean_cat(parsed.get("category"))
    also = _clean_cat(parsed.get("also_category"))
    if also == category:
        also = None  # not actually a second category

    return {
        "intent": intent,
        "category": category,
        "also_category": also,
        "wants_results": bool(parsed.get("wants_results", False)),
        "wants_single": bool(parsed.get("wants_single", False)),
        "sort_preference": sort_pref,
        "confidence": confidence,
    }


# ── Stage 2: Slot extractor ──────────────────────────────────────────────────

def _vague_terms_block() -> str:
    """Percentile reference + mapping rules so the LLM grounds vague terms."""
    rows = []
    for category in database.get_categories():
        thresholds = database.vague_term_thresholds(category)
        if not thresholds:
            continue
        bits = []
        for attr, p in thresholds.items():
            if attr.startswith("price"):
                bits.append(f"{attr}=${p['p25']:.0f}/${p['p50']:.0f}/${p['p75']:.0f}")
            elif attr in ("avg_rating",):
                bits.append(f"{attr}={p['p25']:.1f}/{p['p50']:.1f}/{p['p75']:.1f}")
            else:
                bits.append(f"{attr}={p['p25']:.0f}/{p['p50']:.0f}/{p['p75']:.0f}")
        rows.append(f"  [{category}] " + ", ".join(bits))
    if not rows:
        return ""
    return (
        "VAGUE TERMS — translate to thresholds grounded in these percentiles (p25/median/p75):\n"
        + "\n".join(rows) + "\n"
        "  - 'cheap'/'budget' -> <price>_max=p25 ; 'premium'/'high-end' -> <price>_min=p75 ;\n"
        "    'mid-range' -> <price>_min=p25 AND <price>_max=p75\n"
        "  - 'good camera'/'big battery'/'lots of RAM'/'highly rated'/'long battery' -> <attr>_min=p75\n"
        "  - 'small battery'/'low rated' -> <attr>_max=p25\n"
        "  - SUPERLATIVES ('cheapest','best','biggest') are SORT prefs, NOT filters — emit NO threshold."
    )


def _critique_block(state: Dict[str, Any]) -> str:
    """Anchor relative critiques against the previous recommendation set."""
    prev = state.get("last_recommend_stats")
    if not prev:
        return ""
    lines = []
    for attr, s in prev.items():
        prefix = "$" if attr.startswith("price") else ""
        lines.append(f"  {attr}: min={prefix}{s['min']:.0f}, median={prefix}{s['median']:.0f}, max={prefix}{s['max']:.0f}")
    return (
        "PREVIOUS RECOMMENDATIONS (anchor relative critiques against these):\n"
        + "\n".join(lines) + "\n"
        "  - 'cheaper' -> <price>_max=previous min ; 'more expensive' -> <price>_min=previous max\n"
        "  - 'bigger battery'/'better X' -> <X>_min=previous max ; 'smaller X' -> <X>_max=previous min\n"
        "  - REFINE adds constraints: return ONLY the new filter(s). Use null ONLY for explicit removal\n"
        "    ('forget about X', 'any brand', 'change X')."
    )


_SLOT_HINTS = {
    "smartphone": (
        "brand_name(str), model_contains(str), price_usd_min/max(int), rating_min(0-100),\n"
        "battery_capacity_min(mAh), fast_charging_min(W), ram_capacity/ram_capacity_min(GB),\n"
        "internal_memory/internal_memory_min(GB), screen_size_min/max(in), num_rear_cameras_min(int),\n"
        "os(android|ios|other), primary_camera_rear_min(MP), primary_camera_front_min(MP)"
    ),
    "headphones": (
        "brand(str), model_contains(str), type(Wired|Wireless), connectivity(3.5mm|Bluetooth),\n"
        "form_factor(In-Ear|On-Ear|Over-Ear), microphone(bool), noise_cancellation(bool),\n"
        "foldable(bool), battery_hrs_min(num), price_usd_min/max(int), avg_rating_min(0-5),\n"
        "release_year_min/max(int)"
    ),
}


def extract_slots(category: str, state: Dict[str, Any], sort_preference: str = None) -> Dict[str, Any]:
    """
    Extract ONLY structured filters for a known category. Returns a raw filter
    dict (NOT yet validated — caller runs schema.validate_filters()).

    sort_preference (if the router already detected one) tells the extractor NOT
    to also emit a price filter for "cheapest"/"most expensive" wording — that's
    handled by sorting, not filtering.
    """
    if category not in _SLOT_HINTS:
        return {}

    history = _recent_history(state["messages"])
    active = json.dumps(state.get("active_filters", {}), indent=2)
    last_asked = state.get("last_asked_attribute") or "nothing in particular"

    sort_note = ""
    if sort_preference in ("price_asc", "price_desc"):
        sort_note = (
            f"IMPORTANT: a sort preference ('{sort_preference}') is ALREADY captured. Do NOT emit a "
            "price filter DERIVED FROM the bare word 'cheapest'/'most expensive' — that's handled by "
            "sorting. HOWEVER, if the user ALSO states an explicit price limit (e.g. 'under $300', "
            "'below 500', 'between 200 and 400'), DO emit that EXACT number as price_usd_max/min.\n"
        )

    prompt = f"""You extract structured product filters for a {category}. Return a JSON object
mapping filter keys to values — NOTHING else (no intent, no prose).

Valid filter keys for {category}:
{_SLOT_HINTS[category]}

Rules:
- Only include keys the user explicitly implied THIS message.
- Use null as a VALUE to REMOVE a previously-set filter.
- brand must be the MANUFACTURER (iphone->apple, galaxy->samsung, pixel->google).
- Do NOT invent values. If unsure, omit the key.
- The assistant's previous question was about: "{last_asked}". If the user
  declined ("any", "doesn't matter", "skip"), omit that key.
- If the user explicitly says they DON'T care about an attribute ("I don't care
  about storage", "storage doesn't matter", "any RAM"), add a "skip" array of the
  affected attribute KEY(S) — e.g. {{"skip": ["internal_memory"]}} — and do NOT
  add those as filters.
{sort_note}
{_vague_terms_block()}

{_critique_block(state)}

Current active filters: {active}

Conversation so far:
{history}

User's latest message: "{state['user_input']}"

Return ONLY the JSON object of filters."""

    raw = get_llm().invoke(prompt)
    parsed = parse_json_response(getattr(raw, "content", "") or "")
    # The extractor should return a flat filter dict; if it wrapped it, unwrap.
    if isinstance(parsed.get("extracted_filters"), dict):
        parsed = parsed["extracted_filters"]
    return parsed if isinstance(parsed, dict) else {}
