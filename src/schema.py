"""
schema.py
Declarative slot schema + validation. Every filter the NLU is allowed to emit
is described here. validate_filters() coerces values to the right type,
resolves categoricals/brands via the resolver, range-checks numerics against
the dataset, and DROPS anything unknown or invalid — reporting why (for
observability and confidence scoring).

This turns "trust the LLM's JSON" into "trust the schema": misplaced control
fields, wrong types, hallucinated brands, and out-of-range numbers can no
longer reach the query engine.
"""

from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

import database
import resolver

# kind:
#   'min' / 'max'  -> numeric threshold on `col`
#   'exact_num'    -> numeric equality on `col`
#   'exact_cat'    -> categorical equality on `col` (resolved)
#   'brand'        -> categorical brand on `col` (resolved via resolve_brand)
#   'contains'     -> case-insensitive substring on `col`
#   'bool'         -> boolean on `col`
_SMARTPHONE_SLOTS: Dict[str, Dict[str, str]] = {
    "brand_name":               {"kind": "brand",     "col": "brand_name"},
    "model_contains":           {"kind": "contains",  "col": "model"},
    "price_usd_min":            {"kind": "min",       "col": "price_usd"},
    "price_usd_max":            {"kind": "max",       "col": "price_usd"},
    "rating_min":               {"kind": "min",       "col": "rating"},
    "battery_capacity_min":     {"kind": "min",       "col": "battery_capacity"},
    "fast_charging_min":        {"kind": "min",       "col": "fast_charging"},
    "ram_capacity":             {"kind": "exact_num", "col": "ram_capacity"},
    "ram_capacity_min":         {"kind": "min",       "col": "ram_capacity"},
    "internal_memory":          {"kind": "exact_num", "col": "internal_memory"},
    "internal_memory_min":      {"kind": "min",       "col": "internal_memory"},
    "screen_size_min":          {"kind": "min",       "col": "screen_size"},
    "screen_size_max":          {"kind": "max",       "col": "screen_size"},
    "num_rear_cameras_min":     {"kind": "min",       "col": "num_rear_cameras"},
    "os":                       {"kind": "exact_cat", "col": "os"},
    "primary_camera_rear_min":  {"kind": "min",       "col": "primary_camera_rear"},
    "primary_camera_front_min": {"kind": "min",       "col": "primary_camera_front"},
}

_HEADPHONE_SLOTS: Dict[str, Dict[str, str]] = {
    "brand":                {"kind": "brand",     "col": "brand"},
    "model_contains":       {"kind": "contains",  "col": "model"},
    "type":                 {"kind": "exact_cat", "col": "type"},
    "connectivity":         {"kind": "exact_cat", "col": "connectivity"},
    "form_factor":          {"kind": "exact_cat", "col": "form_factor"},
    "microphone":           {"kind": "bool",      "col": "microphone"},
    "noise_cancellation":   {"kind": "bool",      "col": "noise_cancellation"},
    "foldable":             {"kind": "bool",      "col": "foldable"},
    "battery_hrs_min":      {"kind": "min",       "col": "battery_hrs"},
    "price_usd_min":        {"kind": "min",       "col": "price_usd"},
    "price_usd_max":        {"kind": "max",       "col": "price_usd"},
    "avg_rating_min":       {"kind": "min",       "col": "avg_rating"},
    "release_year_min":     {"kind": "min",       "col": "release_year"},
    "release_year_max":     {"kind": "max",       "col": "release_year"},
}

SLOTS: Dict[str, Dict[str, Dict[str, str]]] = {
    "smartphone": _SMARTPHONE_SLOTS,
    "headphones": _HEADPHONE_SLOTS,
}

_RANGE_CACHE: Dict[tuple, Optional[Tuple[float, float]]] = {}


def reset_cache() -> None:
    _RANGE_CACHE.clear()


def valid_slot_keys(category: str) -> List[str]:
    return list(SLOTS.get(category, {}).keys())


def _col_range(category: str, col: str) -> Optional[Tuple[float, float]]:
    key = (category, col)
    if key in _RANGE_CACHE:
        return _RANGE_CACHE[key]
    df = database._dataframes.get(category)
    rng: Optional[Tuple[float, float]] = None
    if df is not None and col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0:
            rng = (float(s.min()), float(s.max()))
    _RANGE_CACHE[key] = rng
    return rng


def _to_number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        # tolerate "8gb", "$300", "256 GB"
        import re
        m = re.search(r"-?\d+(\.\d+)?", str(v).replace(",", ""))
        return float(m.group()) if m else None


def _to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "yes", "1", "y", "with", "required", "needed"):
        return True
    if s in ("false", "no", "0", "n", "without", "none"):
        return False
    return None


def validate_filters(category: str, filters: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Validate a raw filter dict against the schema.

    Returns (clean_filters, dropped) where:
      - clean_filters keeps only valid, coerced, resolved entries. A value of
        None is preserved (it means "explicitly remove this filter").
      - dropped is a list of {key, value, reason} for logging / confidence.
    """
    slots = SLOTS.get(category, {})
    clean: Dict[str, Any] = {}
    dropped: List[Dict] = []

    for key, value in (filters or {}).items():
        # Explicit removal — always allowed (state updater handles the pop).
        if value is None:
            clean[key] = None
            continue

        spec = slots.get(key)
        if spec is None:
            dropped.append({"key": key, "value": value, "reason": "unknown_slot"})
            continue

        kind, col = spec["kind"], spec["col"]

        if kind in ("min", "max", "exact_num"):
            num = _to_number(value)
            if num is None:
                dropped.append({"key": key, "value": value, "reason": "not_numeric"})
                continue
            if num < 0:
                dropped.append({"key": key, "value": value, "reason": "negative"})
                continue
            rng = _col_range(category, col)
            if rng:
                lo, hi = rng
                # A threshold absurdly beyond the data (e.g. battery_min=99999)
                # would zero results — almost always a hallucination. Drop it.
                if kind == "min" and num > hi * 1.5:
                    dropped.append({"key": key, "value": value, "reason": "above_max"})
                    continue
                if kind == "max" and num < lo * 0.5:
                    dropped.append({"key": key, "value": value, "reason": "below_min"})
                    continue
            clean[key] = num

        elif kind == "brand":
            r = resolver.resolve_brand(category, value)
            if r is None:
                dropped.append({"key": key, "value": value, "reason": "unresolved_brand"})
                continue
            clean[key] = r

        elif kind == "exact_cat":
            r = resolver.resolve_categorical(category, col, value)
            if r is None:
                dropped.append({"key": key, "value": value, "reason": "unresolved_value"})
                continue
            clean[key] = r

        elif kind == "bool":
            b = _to_bool(value)
            if b is None:
                dropped.append({"key": key, "value": value, "reason": "not_boolean"})
                continue
            clean[key] = b

        elif kind == "contains":
            clean[key] = str(value)

    return clean, dropped
