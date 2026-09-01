"""
ranking.py
Pluggable ranking with a feedback flywheel.

  rank() is the single entry point used by the dialogue engine. It:
    1. scores candidates — with a LEARNED model if one has been trained,
       otherwise the hand-tuned TOPSIS scorer (graceful fallback);
    2. optionally blends in semantic similarity for free-text "vibe" queries;
    3. honors an explicit sort preference ("cheapest") when present.

  record_selection() logs which product the user engaged with, building the
  training data. train() fits a simple logistic model over per-product feature
  vectors (numpy, no sklearn) so the weights stop being guesses — the moat is
  the data, not the formula.
"""

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import database
import observability

MODEL_DIR = Path(os.getenv("CRS_MODEL_DIR", "models"))

# Weight of semantic similarity when blending with the structured score (0-1).
SEMANTIC_BLEND = 0.4

_MODEL_CACHE: Dict[str, Optional[Dict]] = {}


# ── Feature vector (shared by learned scorer + trainer) ──────────────────────

def _feature_vector(category: str, product: Dict[str, Any]) -> Dict[str, float]:
    """
    Per-attribute, market-normalised features in [0,1] — the same signal the
    score breakdown surfaces, reused as model inputs.
    """
    rows = database.score_breakdown(category, [product])
    feats: Dict[str, float] = {}
    if rows and rows[0]:
        for r in rows[0]:
            feats[r["attr"]] = r["norm_0_100"] / 100.0
    return feats


# ── Learned model ────────────────────────────────────────────────────────────

def load_model(category: str) -> Optional[Dict]:
    if category in _MODEL_CACHE:
        return _MODEL_CACHE[category]
    path = MODEL_DIR / f"ranker_{category}.json"
    model = None
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                model = json.load(f)
        except Exception:
            model = None
    _MODEL_CACHE[category] = model
    return model


def _learned_scores(category: str, candidates: List[Dict], model: Dict) -> List[float]:
    attrs = model["attrs"]
    w = np.array(model["weights"], dtype=float)
    b = float(model.get("bias", 0.0))
    out = []
    for c in candidates:
        f = _feature_vector(category, c)
        x = np.array([f.get(a, 0.5) for a in attrs], dtype=float)
        z = float(np.dot(w, x) + b)
        out.append(100.0 / (1.0 + np.exp(-z)))  # sigmoid -> 0-100
    return out


# ── Public: rank ─────────────────────────────────────────────────────────────

def rank(
    category: str,
    candidates: List[Dict],
    sort_by: Optional[tuple] = None,
    semantic_query: Optional[str] = None,
    n: Optional[int] = None,
) -> List[Dict]:
    """
    Return candidates ranked best-first. Each dict gets '_score' (0-100), and
    '_semantic' (0-100) when a semantic query is blended.
    """
    if not candidates:
        return []
    n = len(candidates) if n is None else n

    model = load_model(category)
    if model:
        base = _learned_scores(category, candidates, model)
    else:
        base = database.score_candidates(category, candidates)  # TOPSIS

    scored = [{**c, "_score": round(s, 1)} for c, s in zip(candidates, base)]

    # Blend semantic similarity for vibe queries.
    final = list(base)
    if semantic_query:
        try:
            import semantic
            sem = semantic.semantic_scores(category, semantic_query, candidates)  # 0-1
        except Exception:
            sem = [0.0] * len(candidates)
        if any(sem):
            for i in range(len(scored)):
                scored[i]["_semantic"] = round(sem[i] * 100, 1)
                final[i] = (1 - SEMANTIC_BLEND) * base[i] + SEMANTIC_BLEND * (sem[i] * 100)

    # Order.
    if sort_by:
        attr, direction = sort_by
        reverse = (direction == "desc")

        def _num(c):
            v = c.get(attr)
            try:
                return float(v) if v is not None and not (isinstance(v, float) and v != v) else None
            except (TypeError, ValueError):
                return None

        present = [c for c in scored if _num(c) is not None]
        missing = [c for c in scored if _num(c) is None]
        present.sort(key=lambda c: _num(c), reverse=reverse)
        ordered = present + missing
    else:
        idx = list(range(len(scored)))
        random.shuffle(idx)                       # random tie-break
        idx.sort(key=lambda i: final[i], reverse=True)
        ordered = [scored[i] for i in idx]

    return ordered[:n]


def _pid(p: Dict) -> str:
    return f"{p.get('brand_name') or p.get('brand','')}|{p.get('model','')}".strip()


def popularity_counts(category: str) -> Dict[str, int]:
    """How often each product was picked (from the feedback log). Empty at cold start."""
    from collections import Counter
    counts: Counter = Counter()
    for e in observability._load_jsonl(observability.FEEDBACK_LOG):
        if e.get("type") == "selection" and e.get("category") == category and e.get("selected"):
            counts[e["selected"]] += 1
    return dict(counts)


# Weight of popularity when blending into "top picks" (0 = pure value score).
POP_WEIGHT = 0.25
# Don't trust popularity until enough signals exist — a handful of clicks
# shouldn't reorder the catalog. Below this, "top picks" is pure value score.
MIN_POP_SIGNALS = 10


def top_picks(category: str, n: int = 8) -> List[Dict]:
    """
    Zero-input ‘best of’ for a category: rank the whole catalog by objective
    value (TOPSIS), blended with popularity from the feedback flywheel ONCE there
    are enough signals (MIN_POP_SIGNALS). Cold/low-data → pure value score.
    Used by the welcome 'Top picks' buttons.
    """
    cands = database.retrieve(category, {}, limit=None)
    if not cands:
        return []
    ranked = rank(category, cands, n=len(cands))   # value score, best-first
    pop = popularity_counts(category)
    if pop and sum(pop.values()) >= MIN_POP_SIGNALS:
        mx = max(pop.values()) or 1
        for p in ranked:
            p["_pop"] = pop.get(_pid(p), 0)
        ranked.sort(
            key=lambda p: (1 - POP_WEIGHT) * p["_score"] + POP_WEIGHT * 100 * (p.get("_pop", 0) / mx),
            reverse=True,
        )
    return ranked[:n]


# ── Feedback flywheel ────────────────────────────────────────────────────────

def record_selection(category: str, shown: List[Dict], selected: Dict,
                     query: str = None, session_id: str = None) -> None:
    """Log that, among `shown`, the user engaged with `selected`. Training data."""
    def _id(p):
        return f"{p.get('brand_name') or p.get('brand','')}|{p.get('model','')}".strip()
    observability.log_feedback({
        "type": "selection",
        "category": category,
        "session_id": session_id,
        "query": query,
        "selected": _id(selected),
        "shown": [_id(p) for p in shown],
    })


def record_comparison(category: str, compared: List[Dict], session_id: str = None) -> None:
    """Log that the user compared a set of products (a weaker positive signal)."""
    def _id(p):
        return f"{p.get('brand_name') or p.get('brand','')}|{p.get('model','')}".strip()
    observability.log_feedback({
        "type": "comparison",
        "category": category,
        "session_id": session_id,
        "compared": [_id(p) for p in compared],
    })


# ── Trainer (numpy logistic; runs once enough feedback exists) ────────────────

def _product_by_id(category: str, pid: str) -> Optional[Dict]:
    df = database._dataframes.get(category)
    if df is None:
        return None
    brand_col = "brand_name" if category == "smartphone" else "brand"
    try:
        brand, model = pid.split("|", 1)
    except ValueError:
        return None
    sub = df[(df[brand_col].astype(str).str.lower() == brand.lower())
             & (df["model"].astype(str) == model)]
    if len(sub) == 0:
        return None
    return sub.iloc[0].to_dict()


def train(category: str, min_events: int = 20, epochs: int = 300, lr: float = 0.3) -> Dict:
    """
    Fit a pointwise logistic ranker from selection feedback. Each 'selection'
    event yields one positive (selected) and several negatives (shown but not
    selected). Returns a status dict; writes models/ranker_<category>.json on
    success.
    """
    events = [e for e in observability._load_jsonl(observability.FEEDBACK_LOG)
              if e.get("type") == "selection" and e.get("category") == category]
    if len(events) < min_events:
        return {"trained": False, "reason": f"need >= {min_events} selection events, have {len(events)}"}

    # Attribute order from the scorer's weights.
    weights_map = database._WEIGHTS_BY_CATEGORY.get(category, {})
    attrs = list(weights_map.keys())

    X, y = [], []
    for e in events:
        sel = _product_by_id(category, e.get("selected", ""))
        if not sel:
            continue
        sel_feats = _feature_vector(category, sel)
        X.append([sel_feats.get(a, 0.5) for a in attrs]); y.append(1)
        for sid in e.get("shown", []):
            if sid == e.get("selected"):
                continue
            p = _product_by_id(category, sid)
            if not p:
                continue
            f = _feature_vector(category, p)
            X.append([f.get(a, 0.5) for a in attrs]); y.append(0)

    if not X or sum(y) == 0 or sum(y) == len(y):
        return {"trained": False, "reason": "insufficient class balance"}

    X = np.array(X); y = np.array(y, dtype=float)
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b

    model = {"attrs": attrs, "weights": [round(float(x), 4) for x in w],
             "bias": round(b, 4), "n_events": len(events), "n_examples": len(y)}
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / f"ranker_{category}.json", "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)
        _MODEL_CACHE.pop(category, None)
    except Exception as ex:
        return {"trained": False, "reason": f"write failed: {ex}"}
    return {"trained": True, **model}
