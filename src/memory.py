"""
memory.py
Lightweight persistent user profiles. Turns a stateless search box into an
assistant that remembers: preferred brand/OS, typical budget, category history,
and the last set of filters — persisted to disk as JSON, keyed by a username.

Identity here is a simple username (no auth) — enough to demonstrate cross-
session personalization. In production this would be a real user id from auth.
"""

import os
import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROFILE_DIR = Path(os.getenv("CRS_PROFILE_DIR", "profiles"))
_lock = threading.Lock()


def _safe_name(user: str) -> str:
    keep = "".join(c for c in (user or "guest").strip().lower() if c.isalnum() or c in "-_")
    return keep or "guest"


def _path(user: str) -> Path:
    return PROFILE_DIR / f"{_safe_name(user)}.json"


def load_profile(user: str) -> Dict[str, Any]:
    """Load a user's profile, or a fresh empty one."""
    p = _path(user)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "user": _safe_name(user),
        "sessions": 0,
        "recommendations": 0,
        "category_counts": {},
        "brand_counts": {},
        "os_counts": {},
        "budgets_usd": [],
        "last_category": None,
        "last_filters": {},
        "updated": None,
    }


def save_profile(profile: Dict[str, Any]) -> None:
    try:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        profile["updated"] = datetime.now(timezone.utc).isoformat()
        with _lock:
            with open(_path(profile["user"]), "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, default=str)
    except Exception:
        pass


def start_session(user: str) -> Dict[str, Any]:
    """Increment the session counter and persist."""
    profile = load_profile(user)
    profile["sessions"] = profile.get("sessions", 0) + 1
    save_profile(profile)
    return profile


def record_recommendation(user: str, category: str, filters: Dict[str, Any]) -> None:
    """Update the profile from a recommendation the user was shown."""
    if not category:
        return
    profile = load_profile(user)
    profile["recommendations"] = profile.get("recommendations", 0) + 1
    profile["category_counts"][category] = profile["category_counts"].get(category, 0) + 1
    profile["last_category"] = category
    profile["last_filters"] = dict(filters or {})

    brand = filters.get("brand_name") or filters.get("brand")
    if brand:
        profile["brand_counts"][str(brand)] = profile["brand_counts"].get(str(brand), 0) + 1
    if filters.get("os"):
        profile["os_counts"][str(filters["os"])] = profile["os_counts"].get(str(filters["os"]), 0) + 1
    budget = filters.get("price_usd_max")
    if isinstance(budget, (int, float)):
        profile["budgets_usd"] = (profile.get("budgets_usd", []) + [round(float(budget))])[-20:]

    save_profile(profile)


def _top(counter: Dict[str, int]) -> Optional[str]:
    if not counter:
        return None
    return Counter(counter).most_common(1)[0][0]


def typical_budget(profile: Dict[str, Any]) -> Optional[int]:
    vals = profile.get("budgets_usd") or []
    if not vals:
        return None
    return round(sum(vals) / len(vals))


def is_returning(profile: Dict[str, Any]) -> bool:
    return profile.get("recommendations", 0) > 0 or profile.get("sessions", 0) > 1


def profile_summary(profile: Dict[str, Any]) -> Optional[str]:
    """
    One-line summary of the user's last search — describing exactly what
    "Use my usual preferences" will apply (so it's always self-consistent).
    Derived from last_filters, NOT independent aggregates (which could show a
    contradictory mix like "Apple · Android" pulled from different sessions).
    """
    if not is_returning(profile):
        return None
    lf = profile.get("last_filters", {}) or {}
    cat = profile.get("last_category")
    bits = []
    if cat:
        bits.append(str(cat))
    brand = lf.get("brand_name") or lf.get("brand")
    if brand:
        bits.append(str(brand).title())
    if lf.get("os"):
        bits.append(str(lf["os"]).upper())
    pmin, pmax = lf.get("price_usd_min"), lf.get("price_usd_max")
    if pmin and pmax:
        bits.append(f"${int(pmin)}-{int(pmax)}")
    elif pmax:
        bits.append(f"under ${int(pmax)}")
    elif pmin:
        bits.append(f"over ${int(pmin)}")
    if not bits:
        return None
    return " · ".join(bits)


def has_usual(profile: Dict[str, Any]) -> bool:
    return bool(profile.get("last_filters")) and bool(profile.get("last_category"))
