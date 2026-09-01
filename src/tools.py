"""
tools.py
The deterministic TOOL LAYER (Phase 1 of the hybrid re-architecture).

Each tool is a pure, validated function over the existing deterministic core
(database / ranking / schema / semantic). No LLM inside — tools return grounded,
JSON-serializable results. The Phase 2 agent will plan and call these via
function-calling; everything here is independently unit-testable without an LLM.

Design goals (and how they fix the measured baseline gaps):
  • search_products(..., n) honours an explicit COUNT       -> fixes "2 best"/"top 3"
  • search degrades gracefully on 0 results (relax hint)    -> fixes over-constraint
  • compare_products reports UNRESOLVED names honestly       -> fixes out-of-catalog grounding
  • every filter is schema-validated before use             -> the agent can't inject bad data

Public surface:
  TOOLS         : name -> callable
  TOOL_SCHEMAS  : OpenAI-style function schemas (for the agent)
  call_tool(name, args) -> dict   (validated dispatch)
"""

import json
import re
from typing import Any, Dict, List, Optional

import database
import ranking
import schema

CATEGORIES = ("smartphone", "headphones")

# Columns surfaced in a compact product view: (column, output_key).
_VIEW_FIELDS = {
    "smartphone": [
        ("os", "os"), ("price_usd", "price_usd"), ("ram_capacity", "ram_gb"),
        ("internal_memory", "storage_gb"), ("battery_capacity", "battery_mah"),
        ("primary_camera_rear", "rear_camera_mp"), ("rating", "rating_100"),
    ],
    "headphones": [
        ("price_usd", "price_usd"), ("type", "type"), ("form_factor", "form_factor"),
        ("noise_cancellation", "noise_cancellation"), ("battery_hrs", "battery_hrs"),
        ("avg_rating", "rating_5"),
    ],
}
_BOOL_COLS = {"noise_cancellation", "microphone", "foldable"}
_STR_COLS = {"os", "type", "form_factor", "connectivity", "brand", "brand_name", "model"}

# Valid filter keys per category (also enforced by schema.validate_filters).
_FILTER_KEYS = {
    "smartphone": ("brand_name, model_contains, os(android|ios|other), price_usd_min, price_usd_max, "
                   "rating_min(0-100), battery_capacity_min(mAh), fast_charging_min(W), "
                   "ram_capacity(GB, EXACT match) OR ram_capacity_min(GB, at-least), "
                   "internal_memory(GB, EXACT match) OR internal_memory_min(GB, at-least), "
                   "screen_size_min, screen_size_max, "
                   "num_rear_cameras_min, primary_camera_rear_min(MP), primary_camera_front_min(MP)"),
    "headphones": ("brand, model_contains, type(Wired|Wireless), connectivity(3.5mm|Bluetooth), "
                   "form_factor(In-Ear|On-Ear|Over-Ear), microphone(bool), noise_cancellation(bool), "
                   "foldable(bool), battery_hrs_min, price_usd_min, price_usd_max, avg_rating_min(0-5), "
                   "release_year_min, release_year_max"),
}

# Guard: every schema-enforced filter key must be advertised to the agent above, so the
# human-readable vocabulary can't silently drift from the real capability — exactly how
# the exact-match `ram_capacity` key once went unlisted and the agent wrongly claimed it
# couldn't filter exact RAM. Fails fast at import (and in test_tools.py).
for _cat in ("smartphone", "headphones"):
    _undocumented = [k for k in schema.valid_slot_keys(_cat)
                     if not re.search(rf"\b{re.escape(k)}\b", _FILTER_KEYS[_cat])]
    assert not _undocumented, f"filter keys not advertised to the agent ({_cat}): {_undocumented}"


# ── value/format helpers ─────────────────────────────────────────────────────

def _is_nan(v):
    return isinstance(v, float) and v != v


def _num(v):
    if v is None or _is_nan(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clean_val(col, v):
    """JSON-friendly attribute value (bool / number / trimmed string / None)."""
    if v is None or _is_nan(v):
        return None
    if col in _BOOL_COLS:
        return v is True or v == 1.0 or str(v).strip().lower() in ("true", "yes", "1")
    n = _num(v)
    if n is not None and col not in _STR_COLS:
        return int(n) if float(n).is_integer() else round(n, 2)
    s = str(v).strip()
    return s or None


def _brand(p, category):
    return str((p.get("brand_name") if category == "smartphone" else p.get("brand")) or "").strip()


def _name(p, category):
    brand, model = _brand(p, category), str(p.get("model") or "").strip()
    if model and model.lower().startswith(brand.lower()):
        return model
    return f"{brand} {model}".strip() or "(unnamed)"


def _value_score(breakdown) -> Optional[float]:
    """Catalog-relative 0-100 value score: weight-averaged market-normalised
    position across the scoreable attributes. Defined for ANY product."""
    if not breakdown:
        return None
    tot = sum(r.get("weight_pct", 0) for r in breakdown)
    if not tot:
        return None
    return round(sum(r["norm_0_100"] * r["weight_pct"] for r in breakdown) / tot, 1)


def _product_view(p, category, breakdown=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": _name(p, category), "brand": _brand(p, category),
                           "model": str(p.get("model") or "").strip()}
    vs = _value_score(breakdown) if breakdown is not None else None
    if vs is not None:
        out["value_score"] = vs
    for col, key in _VIEW_FIELDS.get(category, []):
        cv = _clean_val(col, p.get(col))
        if cv is not None:
            out[key] = cv
    return out


def _err(msg, **extra):
    return {"error": msg, **extra}


def _check_category(category):
    return category in CATEGORIES


def _as_dict(v) -> Dict[str, Any]:
    """Coerce a 'filters' arg to a dict. Tolerates a JSON string or a loose
    'key=value, key=value' string — some models (e.g. Gemini) stringify object
    arguments instead of passing JSON. Unknown/invalid keys are dropped downstream
    by schema.validate_filters, so a best-effort parse is safe."""
    if isinstance(v, dict):
        return v
    if not v:
        return {}
    if isinstance(v, str):
        try:
            j = json.loads(v)
            if isinstance(j, dict):
                return j
        except Exception:
            pass
        out: Dict[str, Any] = {}
        for part in re.split(r"[,;\n]+", v):
            m = re.match(r"\s*['\"]?(\w+)['\"]?\s*[=:]\s*(.+?)\s*$", part)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip().strip("'\"")
            if re.fullmatch(r"-?\d+", val):
                val = int(val)
            elif re.fullmatch(r"-?\d+\.\d+", val):
                val = float(val)
            elif val.lower() in ("true", "false"):
                val = val.lower() == "true"
            out[key] = val
        return out
    return {}


def _as_list(v) -> List[Any]:
    """Coerce a list-typed arg a model may have stringified (JSON or comma-list)."""
    if isinstance(v, list):
        return v
    if not v:
        return []
    if isinstance(v, str):
        try:
            j = json.loads(v)
            if isinstance(j, list):
                return j
        except Exception:
            pass
        return [p.strip().strip("'\"") for p in re.split(r"[,;\n]+", v) if p.strip()]
    return [v]


# ── Tools ─────────────────────────────────────────────────────────────────────

def search_products(category: str, filters: Optional[Dict] = None,
                    sort_by: Optional[str] = None, n: int = 5,
                    vibe: Optional[str] = None,
                    exclude_brands: Optional[List] = None) -> Dict[str, Any]:
    """Search + rank the catalog. Honours an explicit count `n`, validates filters,
    blends a free-text `vibe`, excludes ruled-out brands, and degrades gracefully
    (relax hint) on 0 results."""
    if not _check_category(category):
        return _err(f"unknown category '{category}'", valid=list(CATEGORIES))
    try:
        n = max(1, min(int(n), 50))
    except (TypeError, ValueError):
        n = 5
    clean, dropped = schema.validate_filters(category, _as_dict(filters))
    cands = database.retrieve(category, clean, limit=None)
    exclude_brands = _as_list(exclude_brands)
    if exclude_brands:
        ex = {str(b).strip().lower() for b in exclude_brands}
        bcol = "brand_name" if category == "smartphone" else "brand"
        cands = [c for c in cands if str(c.get(bcol, "")).strip().lower() not in ex]
    total = len(cands)
    sort_key = {"price_asc": ("price_usd", "asc"),
                "price_desc": ("price_usd", "desc")}.get(sort_by)
    ranked = ranking.rank(category, cands, sort_by=sort_key,
                          semantic_query=vibe, n=n) if cands else []
    bds = database.score_breakdown(category, ranked) if ranked else []
    out = {
        "category": category,
        "applied_filters": clean,
        "dropped_filters": [d["key"] for d in dropped],
        "total_matches": total,
        "returned": len(ranked),
        "products": [_product_view(p, category, bds[i] if i < len(bds) else None)
                     for i, p in enumerate(ranked)],
        "_raw": ranked,   # internal: full catalog rows for UI cards (stripped before the LLM)
    }
    if total == 0 and clean:
        out["relax_hint"] = _relax_hint(category, clean)
    return out


def _relax_hint(category, clean_filters) -> str:
    """Identify which single filter, if relaxed, would unlock results."""
    unlockers = []
    for k in list(clean_filters):
        sub = {kk: vv for kk, vv in clean_filters.items() if kk != k}
        if len(database.retrieve(category, sub, limit=None)) > 0:
            unlockers.append(k)
    if unlockers:
        return f"No exact matches. Relaxing any one of these would return results: {unlockers}."
    return "No matches even after dropping a single filter — try broadening several criteria."


def recommend_top_picks(category: str, n: int = 5) -> Dict[str, Any]:
    """Zero-input 'best of' for a category (objective value, popularity-blended)."""
    if not _check_category(category):
        return _err(f"unknown category '{category}'", valid=list(CATEGORIES))
    try:
        n = max(1, min(int(n), 20))
    except (TypeError, ValueError):
        n = 5
    picks = ranking.top_picks(category, n=n)
    bds = database.score_breakdown(category, picks) if picks else []
    return {"category": category, "returned": len(picks),
            "products": [_product_view(p, category, bds[i] if i < len(bds) else None)
                         for i, p in enumerate(picks)],
            "_raw": picks}


def get_product_details(category: str, query: str) -> Dict[str, Any]:
    """Full specifications for one NAMED product (or None if not in the catalog)."""
    if not _check_category(category):
        return _err(f"unknown category '{category}'", valid=list(CATEGORIES))
    p = database.find_product(category, str(query or ""), min_score=2.0)
    if not p:
        return {"found": False, "query": query,
                "note": f"No product matching '{query}' in the {category} catalog."}
    bd = database.score_breakdown(category, [p])
    view = _product_view(p, category, bd[0] if bd else None)
    # add every scoreable/spec column we know about
    for col, key in _VIEW_FIELDS.get(category, []):
        cv = _clean_val(col, p.get(col))
        if cv is not None:
            view[key] = cv
    for extra in ("fast_charging", "primary_camera_front", "num_rear_cameras",
                  "screen_size", "connectivity", "microphone", "foldable", "release_year"):
        cv = _clean_val(extra, p.get(extra))
        if cv is not None:
            view[extra] = cv
    # Enrichment (offline-generated from specs). Key names tell the LLM the review
    # is AI-generated, so it is never presented as a real user review.
    desc = p.get("description")
    if isinstance(desc, str) and desc.strip():
        view["description"] = desc.strip()
    real = p.get("real_review")
    if isinstance(real, str) and real.strip():
        view["gsmarena_review_excerpt"] = real.strip()   # real editorial excerpt (flagships)
    rev = p.get("review_summary")
    if isinstance(rev, str) and rev.strip():
        view["ai_review_summary"] = rev.strip()
    return {"found": True, "product": view}


def compare_products(category: str, queries: List[str]) -> Dict[str, Any]:
    """Side-by-side comparison of 2-3 NAMED products. Reports any names that are
    NOT in the catalog under `unresolved` (never silently substitutes)."""
    if not _check_category(category):
        return _err(f"unknown category '{category}'", valid=list(CATEGORIES))
    products, unresolved, seen = [], [], set()
    for q in _as_list(queries)[:3]:
        p = database.find_product(category, str(q), min_score=4.0)
        key = _name(p, category).lower() if p else None
        if p and key not in seen:
            products.append(p)
            seen.add(key)
        elif not p:
            unresolved.append(q)
    if len(products) < 2:
        return _err("need at least 2 resolvable products to compare",
                    resolved=[_name(p, category) for p in products],
                    unresolved=unresolved)
    bds = database.score_breakdown(category, products)
    weights = database._WEIGHTS_BY_CATEGORY.get(category, {})
    names = [_name(p, category) for p in products]
    rows = []
    for col, key in _VIEW_FIELDS.get(category, []):
        vals = [_clean_val(col, p.get(col)) for p in products]
        if all(v is None for v in vals):
            continue
        row = {"attribute": key, "values": vals}
        spec = weights.get(col)
        if spec:
            nums = [_num(p.get(col)) for p in products]
            if all(x is not None for x in nums) and len(set(nums)) > 1:
                pick = (min if spec[1] == "lower" else max)(
                    range(len(nums)), key=lambda i: nums[i])
                row["winner"] = names[pick]
        rows.append(row)
    for i, p in enumerate(products):
        p["_score"] = _value_score(bds[i])   # so UI cards show a value score
    return {
        "category": category,
        "products": [_product_view(p, category, bds[i]) for i, p in enumerate(products)],
        "comparison_rows": rows,
        "value_scores": {names[i]: _value_score(bds[i]) for i in range(len(products))},
        "unresolved": unresolved,
        "_raw": products,
    }


def explain_ranking(category: str, query: str) -> Dict[str, Any]:
    """Why a product scores well: its strongest attributes (vs the whole catalog),
    its value score, and one honest trade-off."""
    if not _check_category(category):
        return _err(f"unknown category '{category}'", valid=list(CATEGORIES))
    p = database.find_product(category, str(query or ""), min_score=2.0)
    if not p:
        return {"found": False, "query": query,
                "note": f"No product matching '{query}' in the {category} catalog."}
    bd = database.score_breakdown(category, [p])[0] if database.score_breakdown(category, [p]) else []
    ranked = sorted(bd, key=lambda r: r.get("norm_0_100", 50), reverse=True)

    def _entry(r):
        return {"attribute": r["attr"], "value": _clean_val(r["attr"], r.get("raw")),
                "vs_catalog_0_100": round(r.get("norm_0_100", 50)),
                "weight_pct": round(r.get("weight_pct", 0))}

    return {
        "found": True,
        "product": _name(p, category),
        "value_score": _value_score(bd),
        "strengths": [_entry(r) for r in ranked if r.get("norm_0_100", 50) >= 58][:3],
        "tradeoffs": [_entry(r) for r in reversed(ranked) if r.get("norm_0_100", 50) <= 38][:1],
        "method": "value = market-normalised specs weighed against price (lower price + stronger specs score higher).",
    }


def catalog_info(category: Optional[str] = None) -> Dict[str, Any]:
    """Context for grounding vague terms and knowing what is askable: per-category
    size, brands, valid filter keys, and price/spec percentiles (p25/p50/p75)."""
    cats = [category] if category in CATEGORIES else list(CATEGORIES)
    info = {}
    for cat in cats:
        df = database._dataframes.get(cat)
        if df is None:
            continue
        brand_col = "brand_name" if cat == "smartphone" else "brand"
        brands = sorted({str(b).strip() for b in df[brand_col].dropna().unique()})[:40] \
            if brand_col in df.columns else []
        info[cat] = {
            "count": int(len(df)),
            "filter_keys": _FILTER_KEYS[cat],
            "brands": brands,
            "percentiles": database.vague_term_thresholds(cat),
        }
    return {"categories": list(CATEGORIES), "by_category": info}


# ── Dispatch + function-calling schemas ───────────────────────────────────────

TOOLS = {
    "search_products": search_products,
    "recommend_top_picks": recommend_top_picks,
    "get_product_details": get_product_details,
    "compare_products": compare_products,
    "explain_ranking": explain_ranking,
    "catalog_info": catalog_info,
}


def _fn(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


_CAT = {"type": "string", "enum": list(CATEGORIES)}

TOOL_SCHEMAS = [
    _fn("search_products",
        "Search and rank the product catalog. Use `n` for an explicit count "
        "(e.g. the user asked for 2 or 3). `filters` are structured constraints; "
        "`vibe` is a free-text use-case ('for gaming', 'for travel'); `sort_by` "
        "orders by price. Returns ranked products + total_matches; on 0 results it "
        "includes a relax_hint.",
        {"category": _CAT,
         "filters": {"type": "object",
                     "description": "Structured filters. Valid keys (smartphone): "
                                    + _FILTER_KEYS["smartphone"] + ". (headphones): "
                                    + _FILTER_KEYS["headphones"]},
         "sort_by": {"type": "string", "enum": ["price_asc", "price_desc"]},
         "n": {"type": "integer", "description": "How many products to return (default 5)."},
         "vibe": {"type": "string", "description": "Optional free-text use-case/vibe."},
         "exclude_brands": {"type": "array", "items": {"type": "string"},
                            "description": "Brands to EXCLUDE (e.g. user said 'not Samsung')."}},
        ["category"]),
    _fn("recommend_top_picks",
        "Zero-input 'best of' for a category, ranked by objective value. Use when the "
        "user has no criteria yet ('what's good?', the Top-picks buttons).",
        {"category": _CAT, "n": {"type": "integer", "description": "How many (default 5)."}},
        ["category"]),
    _fn("get_product_details",
        "Full specifications for ONE named product. Returns found:false if the named "
        "product is not in the catalog (do not invent it).",
        {"category": _CAT, "query": {"type": "string", "description": "Product name as the user referred to it."}},
        ["category", "query"]),
    _fn("compare_products",
        "Side-by-side comparison of 2-3 named products, with the per-attribute winner. "
        "Any names not in the catalog come back under `unresolved` — tell the user honestly "
        "rather than substituting other products.",
        {"category": _CAT,
         "queries": {"type": "array", "items": {"type": "string"},
                     "description": "2-3 product names to compare."}},
        ["category", "queries"]),
    _fn("explain_ranking",
        "Explain WHY a product scores well: its strongest attributes vs the whole catalog, "
        "its value score, and one honest trade-off.",
        {"category": _CAT, "query": {"type": "string", "description": "Product name."}},
        ["category", "query"]),
    _fn("catalog_info",
        "Context for grounding vague terms ('cheap', 'good camera') and knowing what is "
        "askable: per-category size, brands, valid filter keys, and price/spec percentiles. "
        "Call this to translate fuzzy wording into concrete filters.",
        {"category": _CAT},
        []),
]


def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Validated dispatch: unknown tool or bad args return an error dict instead of raising."""
    fn = TOOLS.get(name)
    if fn is None:
        return _err(f"unknown tool '{name}'", available=list(TOOLS))
    try:
        return fn(**(args or {}))
    except TypeError as e:
        return _err(f"bad arguments for '{name}': {e}")
    except Exception as e:  # tools must never crash the agent loop
        return _err(f"tool '{name}' failed: {e}")
