"""
enrich_reviews.py   —   RUN THIS ON YOUR LOCAL MACHINE (not a server/sandbox).

Fetches a short REAL editorial-review excerpt from GSMArena for the phones that
have one (notable/flagship models) into a side file datasets/smartphone_reviews.json,
which database.load_all() joins into the catalog (same pattern as image_cache.json).
Phones without a GSMArena review simply keep their generated summary.

Why local: GSMArena serves a Cloudflare Turnstile bot-check to datacenter/cloud IPs,
so this does NOT work from a server/sandbox. The script auto-detects the challenge
and stops with a clear message. It worked for the image scrape on a real machine.

  python enrich_reviews.py                  # try all phones (resumable, slow)
  python enrich_reviews.py --min-price 250  # only phones >= $250  (flagships; far fewer requests)
  python enrich_reviews.py --limit 40       # cap NEW lookups this run
  python enrich_reviews.py --force          # re-fetch even cached

ToS note: GSMArena has no public API and its ToS disallows scraping; the Turnstile
is a clear "no bots" signal. This is a polite, cached, rate-limited, one-time
ACADEMIC pass over a SUBSET. For production, use a licensed feed. Reviews exist for
only a minority of models (flagships) — that is expected, not a bug.
"""
import os
import re
import json
import time
import argparse

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

CSV = "datasets/reduced_file_smartphone_500.csv"
OUT = "datasets/smartphone_reviews.json"
CATEGORY = "smartphone"
BASE = "https://www.gsmarena.com/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
T = 15


def _key(b, m):
    return f"{CATEGORY}|{str(b).strip().lower()}|{str(m).strip().lower()}"


def _blocked(text):
    t = text.lower()
    return ("turnstile" in t) or ("just a moment" in t) or ("challenges.cloudflare.com" in t)


def _resolve_page(query, sess):
    r = sess.get(BASE + "results.php3", params={"sQuickSearch": "yes", "sName": query},
                 headers=UA, timeout=T)
    if _blocked(r.text):
        return None, "BLOCKED"
    if r.status_code != 200:
        return None, f"search {r.status_code}"
    m = re.search(r'<div class="makers">.*?<a href="([^"]+\.php)"', r.text, re.DOTALL)
    return (m.group(1) if m else None), ("ok" if m else "no match")


def _fetch_review(page_path, sess):
    time.sleep(0.5)
    d = sess.get(BASE + page_path, headers=UA, timeout=T)
    if _blocked(d.text):
        return None, None, "BLOCKED"
    if d.status_code != 200:
        return None, None, f"page {d.status_code}"
    rm = re.search(r'href="([^"]*-review-[0-9]+\.php)"', d.text)
    if not rm:
        return None, None, "no review"
    time.sleep(0.5)
    r = sess.get(BASE + rm.group(1), headers=UA, timeout=T)
    if _blocked(r.text):
        return None, None, "BLOCKED"
    if r.status_code != 200:
        return None, None, f"review {r.status_code}"
    for p in re.findall(r"<p[^>]*>(.*?)</p>", r.text, re.DOTALL):
        txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", p)).strip()
        if len(txt) > 90:
            return txt[:480], BASE + rm.group(1), "ok"
    return None, None, "no intro text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-price", type=float, default=0.0, help="only phones >= this USD price")
    ap.add_argument("--limit", type=int, default=None, help="cap NEW lookups this run")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.7, help="seconds between phones (be polite)")
    args = ap.parse_args()

    rows = pd.read_csv(CSV).to_dict("records")
    out = {}
    if os.path.exists(OUT) and not args.force:
        out = json.load(open(OUT, encoding="utf-8"))

    cands = [r for r in rows
             if (args.force or _key(r.get("brand_name"), r.get("model")) not in out)
             and float(r.get("price_usd") or 0) >= args.min_price]
    if args.limit:
        cands = cands[:args.limit]
    print(f"{len(rows)} phones | {len(out)} cached | trying {len(cands)} (price >= ${args.min_price:g})")
    print("GSMArena scrape (LOCAL only; Ctrl-C is safe, progress saved each phone)...\n")

    sess = requests.Session()
    found = misses = 0
    for i, r in enumerate(cands, 1):
        model, brand = str(r.get("model", "")), str(r.get("brand_name", ""))
        q = model if brand.lower() in model.lower() else f"{brand} {model}"
        q = re.sub(r"\(.*?\)", "", q).strip()                 # drop storage notes
        try:
            page, info = _resolve_page(q, sess)
            if info == "BLOCKED":
                print("\n*** Cloudflare bot-check on THIS machine too — GSMArena is not "
                      "scriptable from here. Stopping (nothing lost). Try a residential "
                      "network, or keep the generated summaries. ***")
                break
            if not page:
                misses += 1
            else:
                excerpt, url, status = _fetch_review(page, sess)
                if status == "BLOCKED":
                    print("\n*** Cloudflare bot-check — stopping. ***")
                    break
                if excerpt:
                    out[_key(r.get("brand_name"), r.get("model"))] = {
                        "review_excerpt": excerpt, "review_url": url, "source": "gsmarena"}
                    found += 1
                    print(f"  + {q[:40]}")
                else:
                    misses += 1
        except Exception as e:
            print(f"  [{q[:40]}] ERROR {type(e).__name__}")
            misses += 1
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        if i % 20 == 0:
            print(f"  ... {i}/{len(cands)} | found={found} misses={misses}")
        time.sleep(args.sleep)

    print(f"\nDone. reviews found this run: {found} | total in file: {len(out)}")
    print(f"Saved {OUT}  (phones without a review keep their generated summary)")


if __name__ == "__main__":
    main()
