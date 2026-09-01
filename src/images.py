"""
images.py
Product-image enrichment — the CTO-correct pattern: images are sourced OFFLINE
into the catalog (a URL cache), never fetched live per request.

`fetch_product_image()` is a pluggable, tiered resolver. It tries the best
source available and returns (url, source). Swap/add a source — e.g. a licensed
commerce feed in production — by editing one function; nothing else changes.

Source tiers (first hit wins):
  1. SerpAPI Google Images   — broad coverage, needs SERPAPI_API_KEY (free 100/mo)
  2. Google Custom Search    — needs GOOGLE_API_KEY + GOOGLE_CSE_ID (free 100/day)
  3. Wikipedia (CC)          — free, no key, flagships only
  4. Openverse (CC)          — free, no key, aggregates CC media
  (miss → None → the UI shows an honest placeholder, never a fake/AI photo)

Licensing note: Wikipedia/Openverse are Creative-Commons (attribution stored).
For commercial use, prefer a licensed feed (Icecat / Best Buy / Amazon PA-API).
"""

import os
import re
import json
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import requests

import database

CACHE_PATH = Path(os.getenv("CRS_IMAGE_CACHE", "datasets/image_cache.json"))
_HEADERS = {"User-Agent": "CRS-demo/1.0 (educational project)"}
_TIMEOUT = 12
_lock = threading.Lock()

_cache: Optional[Dict[str, Dict]] = None


# ── Cache ────────────────────────────────────────────────────────────────────

def _key(category: str, brand: str, model: str) -> str:
    return f"{category}|{str(brand).strip().lower()}|{str(model).strip().lower()}"


def load_cache() -> Dict[str, Dict]:
    global _cache
    if _cache is not None:
        return _cache
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    else:
        _cache = {}
    return _cache


def save_cache() -> None:
    if _cache is None:
        return
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_cache, f, indent=2)
    except Exception:
        pass


def cached_url(category: str, brand: str, model: str) -> Optional[str]:
    entry = load_cache().get(_key(category, brand, model))
    return entry.get("url") if entry else None


# ── Query helpers ────────────────────────────────────────────────────────────

def _query(brand: str, model: str) -> str:
    brand, model = str(brand).strip(), str(model).strip()
    # Drop variant parentheticals like "(8GB RAM + 128GB)" — the photo is the
    # same across storage/RAM variants, and they hurt search matching.
    model = re.sub(r"\(.*?\)", "", model).strip()
    # Avoid "Samsung Samsung Galaxy ..." when the model already includes the brand.
    q = model if model.lower().startswith(brand.lower()) else f"{brand} {model}"
    return re.sub(r"\s+", " ", q).strip()


# ── Sources (each returns a direct image URL or None) ────────────────────────

def _serpapi(query: str) -> Optional[str]:
    key = os.getenv("SERPAPI_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_images", "q": query, "api_key": key, "num": 1},
            timeout=_TIMEOUT,
        )
        imgs = r.json().get("images_results") or []
        return imgs[0].get("original") or imgs[0].get("thumbnail") if imgs else None
    except Exception:
        return None


def _google_cse(query: str) -> Optional[str]:
    key, cx = os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CSE_ID")
    if not (key and cx):
        return None
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": key, "cx": cx, "q": query, "searchType": "image", "num": 1},
            timeout=_TIMEOUT,
        )
        items = r.json().get("items") or []
        return items[0].get("link") if items else None
    except Exception:
        return None


def _wikipedia(query: str) -> Optional[str]:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": query, "gsrlimit": 1, "prop": "pageimages",
                "piprop": "thumbnail", "pithumbsize": 400, "redirects": 1,
            },
            timeout=_TIMEOUT, headers=_HEADERS,
        )
        pages = r.json().get("query", {}).get("pages", {})
        for _, pg in pages.items():
            if "thumbnail" in pg:
                return pg["thumbnail"]["source"]
    except Exception:
        return None
    return None


def _openverse(query: str) -> Optional[str]:
    try:
        r = requests.get(
            "https://api.openverse.org/v1/images/",
            params={"q": query, "page_size": 1},
            timeout=_TIMEOUT, headers=_HEADERS,
        )
        res = r.json().get("results") or []
        return (res[0].get("thumbnail") or res[0].get("url")) if res else None
    except Exception:
        return None


# Browser-like UA for GSMArena (which serves a phone product database).
_BROWSER_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}
# Toggle off if it gets blocked or for licensing reasons (default on — best
# coverage for budget phone brands). Production: replace with a licensed feed.
_GSMARENA_ENABLED = os.getenv("CRS_ENABLE_GSMARENA", "1") not in ("0", "false", "False")


def _gsmarena(query: str) -> Optional[str]:
    """
    Resolve a phone product photo from GSMArena (best coverage for budget brands).
    NOTE: GSMArena has no public API; this scrapes search → device page → CDN
    image. Polite (cached offline, rate-limited). Against their ToS for commercial
    use — swap for a licensed feed in production.
    """
    if not _GSMARENA_ENABLED:
        return None
    try:
        s = requests.get(
            "https://www.gsmarena.com/results.php3",
            params={"sQuickSearch": "yes", "sName": query},
            headers=_BROWSER_UA, timeout=_TIMEOUT,
        )
        if s.status_code != 200:
            return None
        m = re.search(r'<div class="makers">.*?<a href="([^"]+\.php)"', s.text, re.DOTALL)
        if not m:
            return None
        time.sleep(0.3)  # be polite between the two requests
        d = requests.get("https://www.gsmarena.com/" + m.group(1),
                         headers=_BROWSER_UA, timeout=_TIMEOUT)
        if d.status_code != 200:
            return None
        img = re.search(r'https://fdn[0-9]?\.gsmarena\.com/vv/bigpic/[^"\' >]+\.(?:jpg|png)', d.text)
        return img.group(0) if img else None
    except Exception:
        return None


# Free / keyed sources tried for every category.
_SOURCES = [
    ("serpapi", _serpapi),
    ("google_cse", _google_cse),
    ("wikipedia", _wikipedia),
    ("openverse", _openverse),
]


def fetch_product_image(brand: str, model: str, category: str = "") -> Tuple[Optional[str], Optional[str]]:
    """Resolve a product image URL via the first source that returns one."""
    q = _query(brand, model)
    # GSMArena first for smartphones (uniform product shots, covers budget brands);
    # it has no headphone data, so it's skipped for that category.
    sources = ([("gsmarena", _gsmarena)] if category == "smartphone" else []) + _SOURCES
    for name, fn in sources:
        url = fn(q)
        if url:
            return url, name
    return None, None


# ── Offline enrichment ───────────────────────────────────────────────────────

def enrich(categories: Optional[List[str]] = None, limit: Optional[int] = None,
           sleep: float = 0.4, verbose: bool = True, force: bool = False) -> Dict[str, int]:
    """
    Populate the image cache for unique products. Resumable (skips cached keys
    unless force=True, which re-fetches everything — e.g. to switch all phone
    photos to uniform GSMArena shots). Returns counts {found, missed, skipped}.
    """
    cache = load_cache()
    cats = categories or database.get_categories()
    counts = {"found": 0, "missed": 0, "skipped": 0}
    processed = 0

    for category in cats:
        df = database._dataframes.get(category)
        if df is None:
            continue
        brand_col = "brand_name" if category == "smartphone" else "brand"
        seen = set()
        for _, row in df.iterrows():
            brand, model = row.get(brand_col), row.get("model")
            if not brand or not model:
                continue
            k = _key(category, brand, model)
            if k in seen:
                continue
            seen.add(k)
            if not force and k in cache and cache[k].get("url"):
                counts["skipped"] += 1
                continue
            if limit is not None and processed >= limit:
                break
            url, source = fetch_product_image(brand, model, category)
            cache[k] = {"url": url, "source": source, "query": _query(brand, model)}
            counts["found" if url else "missed"] += 1
            processed += 1
            if verbose:
                tag = source if url else "—"
                print(f"  [{tag:9}] {_query(brand, model)[:46]}")
            save_cache()  # incremental, so an interrupted run isn't lost
            time.sleep(sleep)
    save_cache()
    return counts
