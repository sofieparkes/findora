"""
semantic.py
Hybrid retrieval: structured filters can't express "good for travel", "punchy
bass", or "great for gaming". This module generates a short description per
product (enriched with use-case tags), embeds it, and scores products by
similarity to a free-text query.

Backends (auto-selected, same interface):
  - DENSE (default): static word embeddings via `model2vec` — true semantic
    similarity ("long trips" ≈ "travel") with no torch dependency. The model
    (~30 MB) is downloaded once and cached by Hugging Face.
  - TF-IDF (fallback): pure-numpy lexical index, used automatically if model2vec
    or the model can't be loaded (e.g. offline). Always available.

Override with CRS_SEMANTIC_BACKEND = embeddings | tfidf | auto (default auto).
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import database

# Hugging Face uses symlinks by default, which need elevated privileges on
# Windows. Disable so the model cache works for every user.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_MODEL_NAME = os.getenv("CRS_EMBED_MODEL", "minishlab/potion-base-8M")
_BACKEND_PREF = os.getenv("CRS_SEMANTIC_BACKEND", "auto").lower()

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "with", "to", "in", "on", "is",
    "it", "this", "that", "good", "best", "want", "need", "looking", "phone",
    "headphones", "headphone", "me", "my", "i", "show", "find", "please",
}


# ── Product → searchable description (shared by both backends) ────────────────

def _num(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and v != v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v, unit) -> str:
    n = _num(v)
    return f"{int(n)} {unit}" if n is not None else ""


def _smartphone_tags(p: Dict[str, Any]) -> List[str]:
    tags = []
    ram, batt = _num(p.get("ram_capacity")), _num(p.get("battery_capacity"))
    cam, price = _num(p.get("primary_camera_rear")), _num(p.get("price_usd"))
    fast = _num(p.get("fast_charging"))
    if ram and ram >= 8 and batt and batt >= 5000:
        tags += ["gaming", "performance", "powerful", "multitasking"]
    if cam and cam >= 64:
        tags += ["photography", "camera", "photos", "content", "vlogging"]
    if batt and batt >= 5500:
        tags += ["travel", "long", "battery", "endurance", "outdoor"]
    if fast and fast >= 40:
        tags += ["fast", "charging"]
    if price and price <= 200:
        tags += ["budget", "affordable", "value", "student"]
    if price and price >= 900:
        tags += ["premium", "flagship", "professional"]
    return tags


def _headphone_tags(p: Dict[str, Any]) -> List[str]:
    tags = []
    nc = str(p.get("noise_cancellation", "")).lower() == "true"
    mic = str(p.get("microphone", "")).lower() == "true"
    typ = str(p.get("type", "")).lower()
    form = str(p.get("form_factor", "")).lower()
    low, batt = _num(p.get("freq_low_hz")), _num(p.get("battery_hrs"))
    if nc:
        tags += ["travel", "commute", "flights", "office", "focus", "isolation"]
    if "in-ear" in form and typ == "wireless":
        tags += ["workout", "running", "sports", "gym", "active", "portable"]
    if "over-ear" in form:
        tags += ["immersive", "studio", "home", "comfortable"]
    if mic:
        tags += ["calls", "gaming", "meetings"]
    if low is not None and low <= 18:
        tags += ["bass", "deep", "punchy"]
    if batt and batt >= 40:
        tags += ["long", "battery", "travel"]
    return tags


def _describe(category: str, p: Dict[str, Any]) -> str:
    if category == "smartphone":
        base = (
            f"{p.get('brand_name','')} {p.get('model','')} {p.get('os','')} smartphone "
            f"{_fmt(p.get('ram_capacity'),'gb ram')} {_fmt(p.get('internal_memory'),'gb storage')} "
            f"{_fmt(p.get('battery_capacity'),'mah battery')} {_fmt(p.get('primary_camera_rear'),'mp camera')}"
        )
        tags = _smartphone_tags(p)
    elif category == "headphones":
        base = (
            f"{p.get('brand','')} {p.get('model','')} {p.get('type','')} {p.get('form_factor','')} headphones "
            f"{'noise cancelling' if str(p.get('noise_cancellation','')).lower()=='true' else ''} "
            f"{_fmt(p.get('battery_hrs'),'hours battery')}"
        )
        tags = _headphone_tags(p)
    else:
        return ""
    # Blend in the generated natural-language description (if enriched) so vibe
    # queries can match richer wording, not just the spec tags.
    extra = str(p.get("description") or "")
    return (base + " " + " ".join(tags) + " " + extra).lower()


def is_vibe_query(text: str) -> bool:
    """Does the text contain subjective/use-case language worth a semantic pass?"""
    cues = (
        "travel", "commute", "flight", "trip", "gaming", "game", "bass", "music",
        "workout", "running", "sport", "gym", "office", "work", "call",
        "photo", "camera", "vlog", "content", "study", "student", "kids",
        "everyday", "daily", "durable", "rugged", "immersive", "comfortable",
        "professional", "casual", "portable", "loud", "quiet", "movies", "podcast",
    )
    low = (text or "").lower()
    return any(c in low for c in cues)


# ── Dense backend (model2vec) ────────────────────────────────────────────────

_dense_model = None
_dense_tried = False


def _load_dense():
    global _dense_model, _dense_tried
    if _dense_tried:
        return _dense_model
    _dense_tried = True
    if _BACKEND_PREF == "tfidf":
        return None
    try:
        from model2vec import StaticModel
        _dense_model = StaticModel.from_pretrained(_MODEL_NAME)
    except Exception:
        _dense_model = None  # offline / not installed → TF-IDF fallback
    return _dense_model


def _l2norm(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ── TF-IDF backend (numpy) ───────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    toks = re.split(r"[^a-z0-9]+", (text or "").lower())
    return [t for t in toks if t and t not in _STOPWORDS and len(t) > 1]


class _TfidfIndex:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None

    def fit(self, docs: List[str]):
        vocab: Dict[str, int] = {}
        tokenized = [_tokenize(d) for d in docs]
        for toks in tokenized:
            for t in set(toks):
                vocab.setdefault(t, len(vocab))
        n = len(docs)
        df = np.zeros(len(vocab))
        for toks in tokenized:
            for t in set(toks):
                df[vocab[t]] += 1
        self.vocab = vocab
        self.idf = np.log((1 + n) / (1 + df)) + 1.0
        return self

    def vec(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in _tokenize(text):
            j = self.vocab.get(t)
            if j is not None:
                v[j] += 1.0
        v = v * self.idf
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v


# ── Unified index cache ──────────────────────────────────────────────────────

# category -> (backend, payload)
#   dense:  ("dense", (doc_emb[N,d], records))
#   tfidf:  ("tfidf", (index, doc_mat[N,V], records))
_CACHE: Dict[str, Tuple[str, tuple]] = {}


def reset_cache() -> None:
    _CACHE.clear()


def backend_name() -> str:
    """Which backend is active (after the first build)."""
    if _CACHE:
        return next(iter(_CACHE.values()))[0]
    return "dense" if _load_dense() is not None else "tfidf"


def _get(category: str):
    if category in _CACHE:
        return _CACHE[category]
    df = database._dataframes.get(category)
    if df is None:
        return None
    records = df.to_dict("records")
    docs = [_describe(category, r) for r in records]

    model = _load_dense()
    if model is not None:
        emb = _l2norm(np.asarray(model.encode(docs), dtype=float))
        _CACHE[category] = ("dense", (emb, records))
    else:
        index = _TfidfIndex().fit(docs)
        mat = np.vstack([index.vec(d) for d in docs]) if docs else np.zeros((0, 0))
        _CACHE[category] = ("tfidf", (index, mat, records))
    return _CACHE[category]


def _embed_query(category: str, query: str) -> Optional[np.ndarray]:
    built = _get(category)
    if not built:
        return None
    kind = built[0]
    if kind == "dense":
        model = _load_dense()
        q = np.asarray(model.encode([query])[0], dtype=float)
        n = np.linalg.norm(q)
        return q / n if n > 0 else q
    else:
        index = built[1][0]
        return index.vec(query)


def _embed_products(category: str, products: List[Dict]) -> Optional[np.ndarray]:
    built = _get(category)
    if not built:
        return None
    kind = built[0]
    docs = [_describe(category, p) for p in products]
    if kind == "dense":
        model = _load_dense()
        return _l2norm(np.asarray(model.encode(docs), dtype=float))
    else:
        index = built[1][0]
        return np.vstack([index.vec(d) for d in docs]) if docs else np.zeros((0, 0))


# ── Public API (unchanged signatures) ────────────────────────────────────────

def semantic_scores(category: str, query: str, products: List[Dict]) -> List[float]:
    """Cosine similarity (0-1, min-max scaled within this set) per product."""
    if not products or not (query or "").strip():
        return [0.0] * len(products)
    qv = _embed_query(category, query)
    pv = _embed_products(category, products)
    if qv is None or pv is None or pv.size == 0 or np.linalg.norm(qv) == 0:
        return [0.0] * len(products)
    sims = pv @ qv
    lo, hi = float(sims.min()), float(sims.max())
    if hi - lo < 1e-9:
        return [0.0] * len(products)
    return list((sims - lo) / (hi - lo))


def semantic_search(category: str, query: str, k: int = 10) -> List[Dict]:
    """Top-k products in the whole category by similarity to query."""
    built = _get(category)
    if not built or not (query or "").strip():
        return []
    qv = _embed_query(category, query)
    if qv is None or np.linalg.norm(qv) == 0:
        return []
    if built[0] == "dense":
        emb, records = built[1]
    else:
        _, emb, records = built[1]
    if emb.size == 0:
        return []
    sims = emb @ qv
    order = np.argsort(-sims)[:k]
    return [{**records[i], "_semantic": round(float(sims[i]), 3)} for i in order if sims[i] > 0]
