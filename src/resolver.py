"""
resolver.py
Entity-resolution layer. Maps messy, free-text user values to the canonical
values that actually exist in the dataset — backed by explicit alias
dictionaries plus a difflib fuzzy fallback, NOT by ever-growing prompt rules.

This is the single biggest robustness win: "iphone" -> brand "apple",
"over ear" -> "Over-Ear", "bluetooth" -> "Wireless", typos -> nearest brand.
"""

import difflib
from typing import Optional, List, Dict

import database

# ── Brand / product-line aliases ─────────────────────────────────────────────
# Map a product LINE or common nickname to the manufacturer brand as stored in
# the dataset. Only include mappings whose target is an actual brand value.
_BRAND_ALIASES: Dict[str, str] = {
    "iphone": "apple",
    "apple iphone": "apple",
    "galaxy": "samsung",
    "samsung galaxy": "samsung",
    "pixel": "google",
    "google pixel": "google",
    "moto": "motorola",
    "redmi note": "redmi",
    "mi": "xiaomi",
    "oneplus nord": "oneplus",
}

# ── Categorical value synonyms, per (category, column) ───────────────────────
_VALUE_ALIASES: Dict[tuple, Dict[str, str]] = {
    ("smartphone", "os"): {
        "android": "android", "google": "android", "google android": "android",
        "ios": "ios", "iphone": "ios", "apple": "ios", "apple ios": "ios",
        "other": "other", "others": "other", "harmonyos": "other", "kaios": "other",
    },
    ("headphones", "type"): {
        "wireless": "Wireless", "bluetooth": "Wireless", "wire-free": "Wireless",
        "true wireless": "Wireless", "tws": "Wireless", "cordless": "Wireless",
        "wired": "Wired", "cabled": "Wired", "with cable": "Wired", "corded": "Wired",
    },
    ("headphones", "connectivity"): {
        "bluetooth": "Bluetooth", "bt": "Bluetooth", "wireless": "Bluetooth",
        "3.5mm": "3.5mm", "3.5 mm": "3.5mm", "aux": "3.5mm", "jack": "3.5mm",
        "headphone jack": "3.5mm", "wired": "3.5mm", "cable": "3.5mm",
    },
    ("headphones", "form_factor"): {
        "over-ear": "Over-Ear", "over ear": "Over-Ear", "overear": "Over-Ear",
        "around-ear": "Over-Ear", "full size": "Over-Ear", "circumaural": "Over-Ear",
        "on-ear": "On-Ear", "on ear": "On-Ear", "onear": "On-Ear", "supra-aural": "On-Ear",
        "in-ear": "In-Ear", "in ear": "In-Ear", "inear": "In-Ear", "iem": "In-Ear",
        "earbud": "In-Ear", "earbuds": "In-Ear", "buds": "In-Ear", "earphones": "In-Ear",
    },
}

# Which column holds the brand, per category.
_BRAND_COL = {"smartphone": "brand_name", "headphones": "brand"}

_DISTINCT_CACHE: Dict[tuple, List[str]] = {}


def _distinct_values(category: str, col: str) -> List[str]:
    """Distinct canonical values for a column (cached)."""
    key = (category, col)
    if key in _DISTINCT_CACHE:
        return _DISTINCT_CACHE[key]
    df = database._dataframes.get(category)
    vals: List[str] = []
    if df is not None and col in df.columns:
        vals = [str(v) for v in df[col].dropna().unique()]
    _DISTINCT_CACHE[key] = vals
    return vals


def reset_cache() -> None:
    """Call after (re)loading data so distinct-value caches refresh."""
    _DISTINCT_CACHE.clear()


def resolve_brand(category: str, value) -> Optional[str]:
    """
    Resolve a free-text brand/product-line to a canonical brand in the dataset.
    Returns None if it can't be confidently resolved (caller should drop it).
    """
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None

    col = _BRAND_COL.get(category)
    if not col:
        return None
    known = {v.lower(): v for v in _distinct_values(category, col)}  # lower -> canonical

    # 1. exact (case-insensitive)
    if raw in known:
        return known[raw]
    # 2. explicit alias
    if raw in _BRAND_ALIASES and _BRAND_ALIASES[raw] in known:
        return known[_BRAND_ALIASES[raw]]
    # 3. product-line contained in the phrase ("galaxy s22 ultra" -> samsung)
    for line, brand in _BRAND_ALIASES.items():
        if line in raw and brand in known:
            return known[brand]
    # 4. a known brand contained in the phrase ("samsung phone" -> samsung)
    for low, canon in known.items():
        if low in raw:
            return canon
    # 5. fuzzy (typos: "samsng" -> "samsung")
    match = difflib.get_close_matches(raw, list(known.keys()), n=1, cutoff=0.82)
    if match:
        return known[match[0]]
    return None


def resolve_categorical(category: str, base_col: str, value) -> Optional[str]:
    """
    Resolve a free-text categorical value (os/type/connectivity/form_factor)
    to a canonical dataset value. Returns None if unresolvable.
    """
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None

    table = _VALUE_ALIASES.get((category, base_col), {})
    known = {v.lower(): v for v in _distinct_values(category, base_col)}

    # 1. alias table
    if raw in table:
        return table[raw]
    # 2. exact against dataset values
    if raw in known:
        return known[raw]
    # 3. alias/known substring (e.g. "over ear headphones")
    for syn, canon in table.items():
        if syn in raw:
            return canon
    # 4. fuzzy against alias keys + known values
    candidates = list(table.keys()) + list(known.keys())
    match = difflib.get_close_matches(raw, candidates, n=1, cutoff=0.80)
    if match:
        m = match[0]
        return table.get(m) or known.get(m)
    return None
