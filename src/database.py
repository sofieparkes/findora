"""
database.py
Loads product CSVs into pandas DataFrames and exposes a filter function.
All filtering logic lives here — no LLM involved.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

# ── Load data ─────────────────────────────────────────────────────────────────

DATA_DIR = Path("datasets")

_dataframes: Dict[str, pd.DataFrame] = {}

# Headphone CSV has columns with spaces/units. Normalize to snake_case on load
# so filter keys stay clean across categories.
_HEADPHONE_COLUMN_MAP = {
    "Brand": "brand",
    "Model": "model",
    "Type": "type",
    "Connectivity": "connectivity",
    "Freq Low (Hz)": "freq_low_hz",
    "Freq High (Hz)": "freq_high_hz",
    "Cable Length (m)": "cable_length_m",
    "Microphone": "microphone",
    "Noise Cancellation": "noise_cancellation",
    "Foldable": "foldable",
    "Battery (hrs)": "battery_hrs",
    "BT Version": "bt_version",
    "Charging Port": "charging_port",
    "Color": "color",
    "Price (USD)": "price_usd",
    "Release Year": "release_year",
    "Avg Rating": "avg_rating",
    "Form Factor": "form_factor",
}


def load_all() -> None:
    """Call once at startup to load all CSVs."""
    files = {
        "smartphone": "reduced_file_smartphone_500.csv",
        "headphones": "reduced_file_headphones_500.csv",
    }
    for category, filename in files.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  [WARNING] Missing {path} - add your CSV files to the datasets/ folder")
            continue
        df = pd.read_csv(path)
        if category == "headphones":
            df = df.rename(columns=_HEADPHONE_COLUMN_MAP)
        df = _attach_images(category, df)
        df = _attach_enrichment(category, df)
        _dataframes[category] = df
        print(f"  [OK] Loaded {len(df)} {category}")


def _attach_images(category: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an `image_url` column from the offline image cache (datasets/
    image_cache.json), if present. No network here — enrichment is done once by
    enrich_images.py. Missing entries stay null and the UI shows a placeholder.
    """
    cache_path = Path("datasets/image_cache.json")
    if not cache_path.exists():
        return df
    try:
        import json
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        return df
    brand_col = "brand_name" if category == "smartphone" else "brand"
    if brand_col not in df.columns or "model" not in df.columns:
        return df

    def _lookup(row):
        key = f"{category}|{str(row[brand_col]).strip().lower()}|{str(row['model']).strip().lower()}"
        entry = cache.get(key)
        return entry.get("url") if entry else None

    df = df.copy()
    df["image_url"] = df.apply(_lookup, axis=1)
    return df


def _attach_enrichment(category: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add enrichment columns from offline side files (no network), if present:
      - description / review_summary  <- datasets/<category>_enrichment.json
        (AI-generated from specs by enrich_smartphones.py; the summary must be
        labeled AI-generated wherever it is shown)
      - real_review / real_review_url <- datasets/<category>_reviews.json
        (real GSMArena editorial-review excerpts for the flagship subset, scraped
        locally by enrich_reviews.py; absent for most phones, which keep the summary)
    Missing entries / missing files stay null.
    """
    import json
    brand_col = "brand_name" if category == "smartphone" else "brand"
    if brand_col not in df.columns or "model" not in df.columns:
        return df

    def _load(name):
        path = Path(f"datasets/{name}")
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    enr = _load(f"{category}_enrichment.json")
    rev = _load(f"{category}_reviews.json")
    if not enr and not rev:
        return df

    def _get(row, cache, field):
        key = f"{category}|{str(row[brand_col]).strip().lower()}|{str(row['model']).strip().lower()}"
        entry = cache.get(key)
        return entry.get(field) if entry else None

    df = df.copy()
    df["description"] = df.apply(lambda r: _get(r, enr, "description"), axis=1)
    df["review_summary"] = df.apply(lambda r: _get(r, enr, "review_summary"), axis=1)
    df["real_review"] = df.apply(lambda r: _get(r, rev, "review_excerpt"), axis=1)
    df["real_review_url"] = df.apply(lambda r: _get(r, rev, "review_url"), axis=1)
    return df


def get_categories() -> List[str]:
    return list(_dataframes.keys())


# Generic words to ignore when matching a product NAME inside free text.
_NAME_STOPWORDS = {
    "show", "me", "the", "of", "for", "a", "an", "please", "specs", "spec",
    "specification", "specifications", "detail", "details", "detailed", "give",
    "tell", "about", "full", "what", "are", "is", "its", "it", "this", "that",
    "and", "with", "more", "info", "information", "on", "phone", "smartphone",
    "smartphones", "headphone", "headphones", "model", "features", "feature",
    "all", "i", "want", "to", "know", "see", "can", "you", "do", "have",
    # comparison filler — so "compare it with one more option" finds no false name
    "compare", "comparison", "vs", "versus", "option", "options", "another",
    "alternative", "alternatives", "between", "or", "against",
}


def find_product(category: str, text: str, min_score: float = 2.0) -> Optional[Dict[str, Any]]:
    """
    Best-effort resolve a product NAMED in free text (e.g. "specs of the Itel A23
    Pro") to its catalog row. Returns the row dict, or None if nothing matches.
    Scores primarily on MODEL-name token overlap, brand secondarily, and rewards
    matching the full model name — so "Itel A23" picks "Itel A23" over "Itel A23
    Pro" only if the bare model exists, else the closest.
    """
    import re
    df = _dataframes.get(category)
    if df is None or len(df) == 0:
        return None
    brand_col = "brand_name" if category == "smartphone" else "brand"
    q = {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
         if t not in _NAME_STOPWORDS}
    if not q:
        return None

    best, best_score = None, 0.0
    for rec in df.to_dict("records"):
        model = str(rec.get("model", "")).lower()
        brand = str(rec.get(brand_col, "")).lower()
        model_toks = set(re.findall(r"[a-z0-9]+", model))
        brand_toks = set(re.findall(r"[a-z0-9]+", brand))
        if not model_toks:
            continue
        model_hit = len(q & model_toks)
        if model_hit == 0:                      # must match at least one model token
            continue
        brand_hit = len(q & brand_toks)
        coverage = model_hit / len(model_toks)  # 1.0 = matched the whole model name
        score = model_hit * 2 + brand_hit + coverage
        if score > best_score:
            best, best_score = rec, score
    return best if best_score >= min_score else None

# ── Filter helpers ─────────────────────────────────────────────────────────────

def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply structured filters to a DataFrame.
    Supported filter keys follow the convention:
      <field>          → exact match  (e.g. brand_name="samsung")
      <field>_min      → >= threshold (e.g. battery_capacity_min=4000)
      <field>_max      → <= threshold (e.g. price_usd_max=300)
      <field>_contains → case-insensitive substring (e.g. model_contains="pro")
    Booleans: noise_cancellation=True (CSV strings "True"/"False" handled).
    """
    for key, value in filters.items():
        if value is None:
            continue

        if key.endswith("_min"):
            col = key[:-4]
            if col in df.columns:
                df = df[pd.to_numeric(df[col], errors="coerce") >= float(value)]

        elif key.endswith("_max"):
            col = key[:-4]
            if col in df.columns:
                df = df[pd.to_numeric(df[col], errors="coerce") <= float(value)]

        elif key.endswith("_contains"):
            col = key[:-9]
            if col in df.columns:
                df = df[df[col].astype(str).str.lower().str.contains(str(value).lower(), na=False)]

        else:
            # Exact match (case-insensitive for strings)
            if key in df.columns:
                col_dtype = df[key].dtype
                if col_dtype == bool:
                    df = df[df[key] == bool(value)]
                elif col_dtype == object or isinstance(value, str):
                    # Always treat as string if either the column or the value is a string
                    df = df[df[key].astype(str).str.lower() == str(value).lower()]
                else:
                    df = df[pd.to_numeric(df[key], errors="coerce") == float(value)]

    return df


def retrieve(category: str, filters: Dict[str, Any], limit: Optional[int] = 10) -> List[Dict]:
    """
    Return products matching the filters for a given category.
    Pass `limit=None` to get the full filtered set (used during recommendation
    so the scorer sees the true population and the sidebar count is accurate).
    """
    if category not in _dataframes:
        return []
    df = _dataframes[category].copy()
    df = _apply_filters(df, filters)
    # Sort by price ascending. Both categories use `price_usd` after the
    # one-time INR→USD conversion on 2026-05-12 (rate 1 INR = 0.010485 USD).
    if "price_usd" in df.columns:
        df = df.sort_values("price_usd")
    if limit is not None:
        df = df.head(limit)
    return df.to_dict("records")


# ── Dialogue: fixed question order per category ───────────────────────────────
#
# The recommender follows this fixed sequence — picking the first attribute the
# user hasn't either answered (via a filter) or explicitly skipped.

QUESTION_ORDER: Dict[str, List[str]] = {
    "smartphone": [
        "os",
        "price_usd",
        "battery_capacity",
        "primary_camera_rear",
        "ram_capacity",
        "internal_memory",
    ],
    "headphones": [
        "type",
        "form_factor",
        "noise_cancellation",
        "price_usd",
    ],
}


def base_attr(filter_key: str) -> str:
    """Strip _min / _max / _contains suffix to get the underlying attribute name."""
    for suffix in ("_min", "_max", "_contains"):
        if filter_key.endswith(suffix):
            return filter_key[: -len(suffix)]
    return filter_key


# ── Vague-term grounding: dataset percentiles per scoreable attribute ────────
#
# Used by the intent-extraction prompt so the LLM can translate terms like
# "cheap", "premium", "big battery", "good camera" into concrete filter
# thresholds derived from the actual dataset distribution.

_VAGUE_GROUND_ATTRS: Dict[str, List[str]] = {
    "smartphone": [
        "price_usd", "battery_capacity", "primary_camera_rear",
        "ram_capacity", "internal_memory", "rating",
    ],
    "headphones": [
        "price_usd", "avg_rating", "battery_hrs",
    ],
}

_VAGUE_THRESHOLDS_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {}


def candidate_stats(category: str, candidates: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    For a given (already-filtered) candidate set, compute min/median/max
    of each scoreable numeric attribute. Used to anchor RELATIVE critiques
    ('cheaper', 'bigger battery', 'better camera') against what the user
    has just been shown — so 'cheaper' means 'less than the current min'.
    """
    attrs = _VAGUE_GROUND_ATTRS.get(category, [])
    if not candidates or not attrs:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for attr in attrs:
        vals: List[float] = []
        for c in candidates:
            v = c.get(attr)
            if v is None or _is_nan(v):
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if not vals:
            continue
        vals.sort()
        n = len(vals)
        median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        out[attr] = {"min": vals[0], "median": median, "max": vals[-1]}
    return out


def vague_term_thresholds(category: str) -> Dict[str, Dict[str, float]]:
    """
    Returns {attribute: {"p25": v, "p50": v, "p75": v}} computed once per
    category and cached. Numeric columns only; NaNs are ignored.
    """
    if category in _VAGUE_THRESHOLDS_CACHE:
        return _VAGUE_THRESHOLDS_CACHE[category]

    attrs = _VAGUE_GROUND_ATTRS.get(category, [])
    df = _dataframes.get(category)
    out: Dict[str, Dict[str, float]] = {}
    if df is None:
        _VAGUE_THRESHOLDS_CACHE[category] = out
        return out

    for attr in attrs:
        if attr not in df.columns:
            continue
        col = pd.to_numeric(df[attr], errors="coerce").dropna()
        if len(col) == 0:
            continue
        out[attr] = {
            "p25": float(col.quantile(0.25)),
            "p50": float(col.quantile(0.50)),
            "p75": float(col.quantile(0.75)),
        }
    _VAGUE_THRESHOLDS_CACHE[category] = out
    return out


def next_question(
    category: str,
    filters: Dict[str, Any],
    asked_skipped: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Return the next attribute to ask about, or None if nothing useful is left.

    Skips attributes that are: already answered, explicitly skipped, OR
    non-discriminative given the current filters — i.e. every remaining
    candidate shares the same value (e.g. don't ask "which OS?" once the brand
    is Samsung, since all matches are Android). Asking those can't narrow.
    """
    order = QUESTION_ORDER.get(category, [])
    answered = {base_attr(k) for k in (filters or {}).keys()}
    skipped = set(asked_skipped or [])

    df = None
    if category in _dataframes:
        df = _apply_filters(_dataframes[category].copy(), filters or {})

    for attr in order:
        if attr in answered or attr in skipped:
            continue
        # Non-discriminative? (0 or 1 distinct value among current candidates)
        if df is not None and attr in df.columns and df[attr].nunique(dropna=True) <= 1:
            continue
        return attr
    return None


# ── Scoring: weighted-feature score (0-100) for picking top recommendations ───

_SMARTPHONE_WEIGHTS = {
    # attribute              (weight, direction)
    "price_usd":            (25, "lower"),
    "primary_camera_rear":  (20, "higher"),
    "battery_capacity":     (20, "higher"),
    "ram_capacity":         (15, "higher"),
    "internal_memory":      (15, "higher"),
    "rating":               (5,  "higher"),
}

_HEADPHONES_WEIGHTS = {
    "price_usd":            (25, "lower"),
    "avg_rating":           (30, "higher"),
    "battery_hrs":          (20, "higher"),
    "freq_range":           (10, "higher"),   # derived: freq_high_hz - freq_low_hz
    "noise_cancellation":   (15, "binary"),
}

_WEIGHTS_BY_CATEGORY = {
    "smartphone": _SMARTPHONE_WEIGHTS,
    "headphones": _HEADPHONES_WEIGHTS,
}


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and v != v


def _normalize_to_score(values: List[Any], mode: str) -> List[float]:
    """Map values to a 0-100 score using min-max within the candidate set."""
    if mode == "binary":
        # Accept both string "True" (from CSV) and numeric 1.0 (from _value_for)
        def _is_true(v):
            return v is True or v == 1.0 or str(v).lower() == "true"
        return [100.0 if _is_true(v) else 0.0 for v in values]

    nums: List[float] = []
    for v in values:
        if v is None or _is_nan(v):
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if not nums:
        return [50.0] * len(values)

    vmin, vmax = min(nums), max(nums)
    if vmax == vmin:
        return [50.0] * len(values)

    out: List[float] = []
    for v in values:
        if v is None or _is_nan(v):
            out.append(50.0)
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            out.append(50.0)
            continue
        if mode == "higher":
            out.append(100.0 * (x - vmin) / (vmax - vmin))
        else:  # "lower"
            out.append(100.0 * (vmax - x) / (vmax - vmin))
    return out


def _value_for(category: str, candidate: Dict, attr: str, mode: str) -> Optional[float]:
    """Extract a single numeric value for an attribute from a candidate (or None)."""
    if attr == "freq_range":
        lo, hi = candidate.get("freq_low_hz"), candidate.get("freq_high_hz")
        if lo is None or hi is None or _is_nan(lo) or _is_nan(hi):
            return None
        return float(hi) - float(lo)
    if mode == "binary":
        return 1.0 if str(candidate.get(attr, "")).lower() == "true" else 0.0
    v = candidate.get(attr)
    if v is None or _is_nan(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score_candidates(category: str, candidates: List[Dict]) -> List[float]:
    """
    TOPSIS scoring (Hwang & Yoon, 1981) — Technique for Order of Preference by
    Similarity to Ideal Solution. Returns 0-100 closeness coefficients.

    Steps:
      1. Build decision matrix (impute missing values with column median).
      2. Vector-normalise each column (divide by its Euclidean norm).
      3. Apply normalised attribute weights.
      4. Determine positive ideal A* (best per attribute) and negative ideal A-.
         For benefit attrs (higher better): A* = max, A- = min. For cost: reversed.
      5. For each candidate, compute Euclidean distances d+ to A* and d- to A-.
      6. Closeness coefficient C = d- / (d+ + d-), scaled to 0-100.
    """
    import math

    weights = _WEIGHTS_BY_CATEGORY.get(category)
    if not weights or not candidates:
        return [0.0] * len(candidates)

    n = len(candidates)
    if n == 1:
        return [100.0]  # only candidate is trivially both best and worst → return 100

    attrs = list(weights.keys())
    m = len(attrs)

    # 1. Build decision matrix; impute missing with per-column median.
    matrix = [[0.0] * m for _ in range(n)]
    for j, attr in enumerate(attrs):
        mode = weights[attr][1]
        col = [_value_for(category, c, attr, mode) for c in candidates]
        clean = [v for v in col if v is not None]
        median = (sorted(clean)[len(clean) // 2] if clean else 0.0)
        for i in range(n):
            matrix[i][j] = col[i] if col[i] is not None else median

    # 2. Vector-normalise each column.
    for j in range(m):
        norm = math.sqrt(sum(matrix[i][j] ** 2 for i in range(n)))
        if norm > 0:
            for i in range(n):
                matrix[i][j] /= norm

    # 3. Apply normalised weights.
    total_weight = sum(w for w, _ in weights.values())
    w_arr = [weights[attrs[j]][0] / total_weight for j in range(m)]
    for i in range(n):
        for j in range(m):
            matrix[i][j] *= w_arr[j]

    # 4. Positive and negative ideals per attribute.
    is_benefit = [weights[attrs[j]][1] != "lower" for j in range(m)]
    ideal_pos = [
        (max(matrix[i][j] for i in range(n)) if is_benefit[j]
         else min(matrix[i][j] for i in range(n)))
        for j in range(m)
    ]
    ideal_neg = [
        (min(matrix[i][j] for i in range(n)) if is_benefit[j]
         else max(matrix[i][j] for i in range(n)))
        for j in range(m)
    ]

    # 5-6. Closeness coefficient per candidate.
    scores: List[float] = []
    for i in range(n):
        d_pos = math.sqrt(sum((matrix[i][j] - ideal_pos[j]) ** 2 for j in range(m)))
        d_neg = math.sqrt(sum((matrix[i][j] - ideal_neg[j]) ** 2 for j in range(m)))
        denom = d_pos + d_neg
        scores.append((d_neg / denom * 100) if denom > 0 else 50.0)
    return scores


def score_breakdown(category: str, candidates: List[Dict]) -> List[List[Dict]]:
    """
    Per-product, per-attribute breakdown for the transparency UI.

    Returns a list (one per candidate, same order as input) of lists of dicts:
        {
            "attr":        attribute name,
            "raw":         raw value (may be None for missing),
            "norm_0_100":  direction-aware position vs the WHOLE category (not the
                           filtered set) — 100 = best in the entire catalog on
                           this attribute, 0 = worst, 50 = median or missing.
            "weight_pct":  percent weight (0-100, sums to 100 across attributes),
            "direction":   "higher" | "lower" | "binary",
        }

    Normalising against the whole category (rather than the filtered candidates)
    keeps the breakdown meaningful even when only 1-2 items match the filter —
    otherwise every attribute would collapse to a self-referential 50.

    The TOPSIS score in `score_candidates` still operates on the filtered set
    (that's the right "best of my matches" ranking question).
    """
    weights = _WEIGHTS_BY_CATEGORY.get(category)
    if not weights or not candidates:
        return [[] for _ in candidates]

    # Reference distribution = the full category dataframe (independent of filters)
    full_df = _dataframes.get(category)
    full_records = full_df.to_dict("records") if full_df is not None else candidates

    attrs = list(weights.keys())
    total_weight = sum(w for w, _ in weights.values())

    # Per-attribute min/max across the WHOLE category
    per_attr_minmax: Dict[str, tuple] = {}
    for attr in attrs:
        mode = weights[attr][1]
        vals = [_value_for(category, c, attr, mode) for c in full_records]
        clean = [v for v in vals if v is not None]
        per_attr_minmax[attr] = (min(clean), max(clean)) if clean else (0.0, 0.0)

    def _market_norm(raw, attr, mode):
        if raw is None:
            return 50.0
        if mode == "binary":
            return 100.0 if (raw is True or raw == 1.0 or str(raw).lower() == "true") else 0.0
        vmin, vmax = per_attr_minmax[attr]
        if vmax == vmin:
            return 50.0
        if mode == "higher":
            x = 100.0 * (raw - vmin) / (vmax - vmin)
        else:  # "lower"
            x = 100.0 * (vmax - raw) / (vmax - vmin)
        return max(0.0, min(100.0, x))

    breakdowns: List[List[Dict]] = []
    for c in candidates:
        rows = []
        for attr in attrs:
            w, mode = weights[attr]
            raw = _value_for(category, c, attr, mode)
            rows.append({
                "attr": attr,
                "raw": raw,
                "norm_0_100": round(_market_norm(raw, attr, mode), 1),
                "weight_pct": round(100 * w / total_weight, 1),
                "direction": mode,
            })
        breakdowns.append(rows)
    return breakdowns


def top_n_by_score(
    category: str,
    candidates: List[Dict],
    n: int = 2,
    sort_by: Optional[tuple] = None,
) -> List[Dict]:
    """
    Rank candidates and return the top n. Each returned dict includes a
    '_score' key (TOPSIS closeness, rounded to 1 decimal) for display.

    Ordering:
      - sort_by=None (default): order by TOPSIS score, random tie-break.
      - sort_by=(attr, "asc"|"desc"): honor an explicit user sort preference
        (e.g. "cheapest" → ("price_usd", "asc")). Products missing that
        attribute are pushed to the end regardless of direction. The _score
        is still attached so the UI shows match quality.
    """
    import random

    if not candidates:
        return []
    scores = score_candidates(category, candidates)
    scored = [{**c, "_score": round(s, 1)} for c, s in zip(candidates, scores)]

    if sort_by:
        attr, direction = sort_by
        reverse = (direction == "desc")

        def _num(c):
            v = c.get(attr)
            if v is None or _is_nan(v):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        present = [c for c in scored if _num(c) is not None]
        missing = [c for c in scored if _num(c) is None]
        present.sort(key=lambda c: _num(c), reverse=reverse)
        ordered = present + missing
    else:
        random.shuffle(scored)                       # random order for ties
        scored.sort(key=lambda c: c["_score"], reverse=True)
        ordered = scored

    return ordered[:n]