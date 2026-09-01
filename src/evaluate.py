"""
evaluate.py
Quantitative evaluation harness for the conversational recommender.

Four test suites, from cheap+deterministic to LLM-backed:
  1. Entity resolution   (resolver.py)  — deterministic, free
  2. Slot validation     (schema.py)    — deterministic, free
  3. Intent routing      (nlu.route)    — LLM
  4. End-to-end task success (full graph)— LLM, multi-turn

Run:  python evaluate.py
Writes eval_results.json (consumed by make_poster.py).
"""

import json
import time
from dotenv import load_dotenv

load_dotenv()

import database
database.load_all()

import resolver
import schema
import nlu
from state import initial_state
from graph import run_turn


# ── 1. Entity resolution (deterministic) ─────────────────────────────────────
# (category, kind, input, expected)  kind: "brand" or a categorical column
RESOLUTION_CASES = [
    ("smartphone", "brand", "iphone", "apple"),
    ("smartphone", "brand", "Galaxy S22", "samsung"),
    ("smartphone", "brand", "pixel", "google"),
    ("smartphone", "brand", "samsng", "samsung"),        # typo
    ("smartphone", "brand", "APPLE", "apple"),
    ("smartphone", "brand", "redmi note 12", "redmi"),
    ("smartphone", "brand", "xiomi", "xiaomi"),           # typo
    ("headphones", "brand", "sony", "Sony"),
    ("headphones", "brand", "sennheiser", "Sennheiser"),
    ("smartphone", "os", "android", "android"),
    ("smartphone", "os", "Android", "android"),
    ("smartphone", "os", "apple", "ios"),
    ("smartphone", "os", "iphone", "ios"),
    ("headphones", "form_factor", "over ear", "Over-Ear"),
    ("headphones", "form_factor", "in-ear", "In-Ear"),
    ("headphones", "form_factor", "earbuds", "In-Ear"),
    ("headphones", "type", "bluetooth", "Wireless"),
    ("headphones", "type", "wired", "Wired"),
    ("headphones", "connectivity", "aux", "3.5mm"),
    ("headphones", "connectivity", "bt", "Bluetooth"),
]


def eval_resolution():
    correct, fails = 0, []
    for cat, kind, inp, exp in RESOLUTION_CASES:
        got = (resolver.resolve_brand(cat, inp) if kind == "brand"
               else resolver.resolve_categorical(cat, kind, inp))
        ok = got is not None and str(got).lower() == str(exp).lower()
        correct += ok
        if not ok:
            fails.append(f"{inp!r}->{got!r} (exp {exp!r})")
    return {"name": "Entity resolution", "correct": correct,
            "total": len(RESOLUTION_CASES), "fails": fails}


# ── 2. Slot validation (deterministic) ───────────────────────────────────────
# (category, raw_filters, must_keep_keys, must_drop_keys)
VALIDATION_CASES = [
    ("smartphone", {"brand_name": "iphone", "ram_capacity_min": 8}, ["brand_name", "ram_capacity_min"], []),
    ("smartphone", {"sort_preference": "price_asc"}, [], ["sort_preference"]),           # control field
    ("smartphone", {"wants_results": True}, [], ["wants_results"]),                       # control field
    ("smartphone", {"brand_name": "Zyxxon"}, [], ["brand_name"]),                         # unresolvable brand
    ("smartphone", {"battery_capacity_min": 99999}, [], ["battery_capacity_min"]),        # impossible value
    ("smartphone", {"price_usd_max": "$300"}, ["price_usd_max"], []),                     # coercible string
    ("headphones", {"noise_cancellation": "yes"}, ["noise_cancellation"], []),            # bool coercion
    ("headphones", {"form_factor": "over ear"}, ["form_factor"], []),                     # resolved categorical
    ("smartphone", {"os": "windows phone"}, [], ["os"]),                                  # invalid os
    ("smartphone", {"made_up_key": 5}, [], ["made_up_key"]),                              # unknown slot
]


def eval_validation():
    correct, fails = 0, []
    for cat, raw, keep, drop in VALIDATION_CASES:
        clean, dropped = schema.validate_filters(cat, raw)
        dropped_keys = {d["key"] for d in dropped}
        ok = all(k in clean for k in keep) and all(k in dropped_keys for k in drop)
        correct += ok
        if not ok:
            fails.append(f"{raw} -> kept {list(clean)} dropped {list(dropped_keys)}")
    return {"name": "Slot validation", "correct": correct,
            "total": len(VALIDATION_CASES), "fails": fails}


# ── 3. Intent routing (LLM) ──────────────────────────────────────────────────
# (utterance, expected_intent, prior_recommended?, pending_attr?)
#   prior  -> seeds last_recommend_stats (a recommendation was shown)
#   pending-> seeds last_asked_attribute (a clarification question is open)
#   expected_intent may be a tuple = any of these is acceptable.
INTENT_CASES = [
    ("I want a smartphone", "explore", False),
    ("recommend me some headphones", "explore", False),
    ("a Samsung phone with 8GB RAM", "specific", False),
    ("show me the cheapest iphone", "specific", False),
    ("wireless over-ear headphones under $200", "specific", False),
    ("cheaper ones", "refine", True),
    ("anything with a better camera", "refine", True),
    ("actually I prefer Apple", "refine", True),
    ("what do you have so far?", "summarize", False),
    ("recap my preferences", "summarize", False),
    ("thanks, that's all", "done", False),
    ("let's start over", "done", False),
    ("I'll take the Samsung", "done", True),
    ("good morning", "chitchat", False),
    ("thank you so much", "chitchat", False),
    ("can you show me pictures of this", "out_of_scope", True),
    ("is it in stock?", "out_of_scope", True),
    ("what's the warranty", "out_of_scope", True),
    ("hmm idk just something", "ambiguous", False),
    ("android", "specific", False),
    ("headphones for the gym", "specific", False),
    ("I need a phone for gaming under $300", "specific", False),
    # explain — the user wants reasoning behind a shown recommendation
    ("why did you recommend the Galaxy A32?", "explain", True),
    ("why this one?", "explain", True),
    ("how did you pick these", "explain", True),
    ("what makes it a good choice", "explain", True),
    # advise — unsure on a PENDING question, wants help deciding (4th field = attr)
    ("I have no idea about the operating system", "advise", False, "os"),
    ("what do you recommend?", "advise", False, "os"),
    ("you decide", "advise", False, "price_usd"),
    ("which is better for me?", "advise", False, "os"),
    ("I'm new to this, not sure", "advise", False, "os"),
    # negatives: a hard skip with a pending question must NOT become 'advise'
    ("any", ("specific", "explore"), False, "os"),
    ("doesn't matter", ("specific", "explore"), False, "os"),
    # specs (product details) — specs ARE available, so NOT out_of_scope
    ("show me the specs of the Galaxy A32", "specs", True),
    ("detailed specifications of the Itel A23", "specs", True),
    ("what are its specifications?", "specs", True),
    # pictures are still out_of_scope (must NOT be confused with specs)
    ("show me pictures of it", "out_of_scope", True),
    # compare — side-by-side of two/three products
    ("compare it with one more option", "compare", True),
    ("how does it compare to the Samsung", "compare", True),
    ("Itel A23 vs Galaxy A32", "compare", True),
]


def eval_intents():
    correct, fails = 0, []
    for case in INTENT_CASES:
        utt, exp = case[0], case[1]
        prior = case[2] if len(case) > 2 else False
        pending = case[3] if len(case) > 3 else None
        s = initial_state()
        s["user_input"] = utt
        if prior:
            s["last_recommend_stats"] = {"price_usd": {"min": 200, "median": 400, "max": 900}}
            s["category"] = "smartphone"
        if pending:
            s["last_asked_attribute"] = pending
            s["category"] = s.get("category") or "smartphone"
        got = nlu.route(s)["intent"]
        ok = (got == exp) if isinstance(exp, str) else (got in exp)
        correct += ok
        if not ok:
            fails.append(f"{utt!r}: got {got}, exp {exp}")
    return {"name": "Intent routing", "correct": correct,
            "total": len(INTENT_CASES), "fails": fails}


# ── 4. End-to-end task success (LLM, multi-turn) ─────────────────────────────
SCENARIOS = [
    ("Specific lookup",
     ["a Samsung Android phone with at least 8GB RAM"],
     lambda s: s["action"] == "recommend" and s["active_filters"].get("brand_name") == "samsung"
               and len(s["candidates"]) > 0),
    ("Guided exploration asks first",
     ["I want a smartphone"],
     lambda s: s["action"] == "ask_clarification"),
    ("Vague term grounding",
     ["I want a cheap android phone"],
     lambda s: s["active_filters"].get("price_usd_max") is not None and len(s["candidates"]) > 0),
    ("Cheapest = sorted, fresh search",
     ["show me android phones under 1000", "recommend me the cheapest iphones"],
     lambda s: s["active_filters"].get("brand_name") == "apple" and len(s["candidates"]) > 0),
    ("Refine narrows prior results",
     ["show me samsung phones now", "cheaper ones"],
     lambda s: s["action"] == "recommend" and "price_usd_max" in s["active_filters"]),
    ("Category switch resets",
     ["I want a samsung phone", "actually show me headphones"],
     lambda s: s["category"] == "headphones" and "brand_name" not in s["active_filters"]),
    ("Mixed-initiative undo",
     ["I want a samsung android phone", "with 8gb ram", "ignore that"],
     lambda s: "ram_capacity_min" not in s["active_filters"]
               and s["active_filters"].get("brand_name") == "samsung"),
    ("Semantic vibe retrieval",
     ["headphones for travel and flights"],
     lambda s: len(s["candidates"]) > 0),
    ("Out-of-scope handled honestly",
     ["show me sony headphones", "can I buy this here"],
     lambda s: s["action"] == "out_of_scope"),
    ("Chitchat preserves state",
     ["I want a phone", "good morning"],
     lambda s: s["action"] == "chitchat"),
]


def eval_scenarios():
    correct, fails, lat = 0, [], []
    for name, turns, predicate in SCENARIOS:
        s = initial_state()
        for t in turns:
            t0 = time.perf_counter()
            s = run_turn(s, t)
            lat.append(time.perf_counter() - t0)
        try:
            ok = bool(predicate(s))
        except Exception:
            ok = False
        correct += ok
        if not ok:
            fails.append(f"{name}: action={s['action']} filters={s['active_filters']}")
    return ({"name": "End-to-end task success", "correct": correct,
             "total": len(SCENARIOS), "fails": fails},
            round(sum(lat) / len(lat), 2) if lat else None)


# ── Runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  CONVERSATIONAL RECOMMENDER — EVALUATION")
    print("=" * 64)
    suites = []

    for fn in (eval_resolution, eval_validation, eval_intents):
        r = fn()
        suites.append(r)
        pct = 100 * r["correct"] / r["total"]
        print(f"\n{r['name']:24} {r['correct']}/{r['total']}  ({pct:.0f}%)")
        for f in r["fails"]:
            print(f"   miss: {f}")

    sc, avg_latency = eval_scenarios()
    suites.append(sc)
    pct = 100 * sc["correct"] / sc["total"]
    print(f"\n{sc['name']:24} {sc['correct']}/{sc['total']}  ({pct:.0f}%)")
    for f in sc["fails"]:
        print(f"   miss: {f}")

    # Image coverage (from the catalog).
    img = {}
    for cat in database.get_categories():
        df = database._dataframes[cat]
        if "image_url" in df.columns:
            img[cat] = round(100 * df["image_url"].notna().sum() / len(df))
        else:
            img[cat] = 0

    overall = {
        "suites": {s["name"]: {"correct": s["correct"], "total": s["total"],
                               "pct": round(100 * s["correct"] / s["total"])} for s in suites},
        "avg_latency_s_per_turn": avg_latency,
        "image_coverage_pct": img,
    }
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\n" + "=" * 64)
    print(f"  Avg latency/turn: {avg_latency}s   Image coverage: {img}")
    print("  Saved eval_results.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
