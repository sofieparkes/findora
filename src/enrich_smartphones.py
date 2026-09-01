"""
enrich_smartphones.py
Generate a grounded DESCRIPTION and a clearly-labeled AI REVIEW SUMMARY for every
phone in datasets/reduced_file_smartphone_500.csv, written to a side file
datasets/smartphone_enrichment.json — joined into the catalog by
database.load_all() (mirroring the image_cache.json pattern). The original CSV is
never modified.

Both fields are generated ONLY from the structured specs (+ the numeric rating),
so they stay grounded — no invented facts. The review summary is surfaced with an
explicit "AI summary" label wherever it is shown (UI + get_product_details).

  python enrich_smartphones.py             # generate for all (resumable)
  python enrich_smartphones.py --limit 8   # quick test on the first 8 phones
  python enrich_smartphones.py --force     # regenerate everything from scratch

Cloud-primary (uses the configured default model via llm_client).
"""
import os
import sys
import json

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import llm_client
from llm_client import get_llm, parse_json_response

CSV = "datasets/reduced_file_smartphone_500.csv"
OUT = "datasets/smartphone_enrichment.json"
CATEGORY = "smartphone"
BATCH = 8


def _key(brand, model):
    return f"{CATEGORY}|{str(brand).strip().lower()}|{str(model).strip().lower()}"


def _g(r, k):
    v = r.get(k)
    return "" if v is None or (isinstance(v, float) and v != v) else v


def _spec_line(idx, r):
    return (f"{idx}. {_g(r,'brand_name')} {_g(r,'model')} | os={_g(r,'os')} | "
            f"price=${_g(r,'price_usd')} | rating={_g(r,'rating')}/100 | "
            f"ram={_g(r,'ram_capacity')}GB | storage={_g(r,'internal_memory')}GB | "
            f"battery={_g(r,'battery_capacity')}mAh | fast_charging={_g(r,'fast_charging')}W | "
            f"rear_cam={_g(r,'primary_camera_rear')}MP | front_cam={_g(r,'primary_camera_front')}MP | "
            f"screen={_g(r,'screen_size')}in | rear_cameras={_g(r,'num_rear_cameras')}")


PROMPT = """You write concise, factual copy for an online phone shop. For each numbered phone below, \
use ONLY the specs given — never invent brands, prices, or features. Return for each phone:
- "description": ONE sentence (<= ~28 words), natural language, describing the phone from its specs.
- "review_summary": ONE sentence (<= ~28 words) in the style of aggregated owner feedback, grounded in \
the specs and the rating (higher rating = more positive; if rating is below ~70, include a mild caveat). \
Cite 1-2 concrete specs as strengths.

Return ONLY a JSON object mapping each phone number (as a string) to its two fields, e.g.:
{{"1": {{"description": "...", "review_summary": "..."}}, "2": {{"description": "...", "review_summary": "..."}}}}

Phones:
{block}
"""


def main():
    force = "--force" in sys.argv
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    llm_client.set_active_model(None)            # cloud default
    rows = pd.read_csv(CSV).to_dict("records")
    if limit:
        rows = rows[:limit]

    out = {}
    if os.path.exists(OUT) and not force:
        with open(OUT, encoding="utf-8") as f:
            out = json.load(f)

    todo = [r for r in rows if force or _key(r.get("brand_name"), r.get("model")) not in out]
    print(f"{len(rows)} phones | {len(out)} already done | {len(todo)} to generate")

    llm = get_llm()
    done = 0
    n_batches = (len(todo) + BATCH - 1) // BATCH
    for bi in range(n_batches):
        batch = todo[bi * BATCH: (bi + 1) * BATCH]
        block = "\n".join(_spec_line(i + 1, r) for i, r in enumerate(batch))
        try:
            resp = llm.invoke(PROMPT.format(block=block))
            data = parse_json_response(getattr(resp, "content", "") or "")
        except Exception as e:
            print(f"  batch {bi + 1}/{n_batches} ERROR: {e}")
            data = {}
        for i, r in enumerate(batch):
            entry = data.get(str(i + 1)) if isinstance(data, dict) else None
            if isinstance(entry, dict) and entry.get("description"):
                out[_key(r.get("brand_name"), r.get("model"))] = {
                    "description": str(entry.get("description", "")).strip(),
                    "review_summary": str(entry.get("review_summary", "")).strip(),
                    "generated": True,
                }
                done += 1
        with open(OUT, "w", encoding="utf-8") as f:        # write incrementally (crash-safe)
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  batch {bi + 1}/{n_batches} -> {done}/{len(todo)} generated")

    print(f"Done. {len(out)} total entries in {OUT}")


if __name__ == "__main__":
    main()
