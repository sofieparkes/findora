"""
enrich_images.py
One-time (resumable) offline pass that populates datasets/image_cache.json with
a product-image URL per item. Run whenever the catalog changes.

Usage:
    python enrich_images.py                 # all products, free sources
    python enrich_images.py --limit 50      # first 50 uncached (good for a quick demo)
    python enrich_images.py --category smartphone

For best coverage, set a key first (free tiers):
    SERPAPI_API_KEY=...     (https://serpapi.com, 100 searches/mo)
  or
    GOOGLE_API_KEY=... GOOGLE_CSE_ID=...   (Google Custom Search, 100/day)
Without a key it falls back to Wikipedia + Openverse (Creative-Commons, partial
coverage); misses render an honest placeholder in the app.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

import database
import images


def main():
    ap = argparse.ArgumentParser(description="Populate product-image cache.")
    ap.add_argument("--category", choices=["smartphone", "headphones"], help="limit to one category")
    ap.add_argument("--limit", type=int, default=None, help="max NEW lookups this run")
    ap.add_argument("--sleep", type=float, default=0.4, help="seconds between lookups")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even cached items (e.g. switch phones to GSMArena shots)")
    args = ap.parse_args()

    database.load_all()
    cats = [args.category] if args.category else None

    src = "SerpAPI" if __import__("os").getenv("SERPAPI_API_KEY") else (
        "Google CSE" if __import__("os").getenv("GOOGLE_API_KEY") else "free CC (Wikipedia/Openverse)")
    print(f"Image source: {src}")
    print("Enriching (resumable; Ctrl-C is safe — progress is saved incrementally)...\n")

    counts = images.enrich(categories=cats, limit=args.limit, sleep=args.sleep, force=args.force)
    total = counts["found"] + counts["missed"]
    print(f"\nDone. found={counts['found']} missed={counts['missed']} "
          f"skipped(cached)={counts['skipped']}")
    if total:
        print(f"Coverage this run: {100*counts['found']/total:.0f}%")
    print(f"Cache: {images.CACHE_PATH}")


if __name__ == "__main__":
    main()
