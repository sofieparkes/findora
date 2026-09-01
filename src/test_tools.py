"""
test_tools.py
Deterministic unit tests for the Phase 1 tool layer (no LLM, fast, free).
Run: python test_tools.py
"""

import database
database.load_all()
import tools

_passed = 0


def check(label, cond, detail=""):
    global _passed
    assert cond, f"FAIL: {label}  {detail}"
    _passed += 1
    print(f"  ok  {label}")


def run():
    # count: explicit n is honoured (fixes "2 best"/"top 3")
    check("search honours n=2", tools.search_products("smartphone", n=2)["returned"] == 2)
    check("top_picks honours n=3", tools.recommend_top_picks("headphones", n=3)["returned"] == 3)

    # filters validated + respected
    r = tools.search_products("smartphone", filters={"brand_name": "samsung", "price_usd_max": 300}, n=5)
    check("filter applied (brand)", r["applied_filters"].get("brand_name") == "samsung")
    check("filter respected (price<=300)", all((p.get("price_usd", 0) or 0) <= 300 for p in r["products"]))
    check("invalid filter dropped",
          "made_up_key" in tools.search_products("smartphone", filters={"made_up_key": 5})["dropped_filters"])

    # over-constraint: valid-but-empty -> 0 + relax hint (fixes the over-filter dead end)
    r = tools.search_products("smartphone", filters={"brand_name": "apple", "os": "android"})
    check("over-constraint -> 0 + relax_hint", r["total_matches"] == 0 and "relax_hint" in r)

    # grounding: genuinely-absent product is reported, never substituted
    r = tools.compare_products("smartphone", ["Itel A23 Pro", "Acme Foobar 9000"])
    check("compare reports unresolved",
          "Acme Foobar 9000" in (r.get("unresolved") or []) and "error" in r)

    # compare two real products -> rows + per-attribute winners
    names = [p["name"] for p in tools.search_products("smartphone", n=2)["products"]]
    r = tools.compare_products("smartphone", names)
    check("compare two real -> rows", len(r["products"]) == 2 and len(r["comparison_rows"]) > 0)
    check("compare marks a winner", any("winner" in row for row in r["comparison_rows"]))

    # details: real vs fake
    check("details found (real)", tools.get_product_details("smartphone", "Itel A23 Pro")["found"] is True)
    check("details not-found (fake)",
          tools.get_product_details("smartphone", "Quantum Phone 9000")["found"] is False)

    # explain
    r = tools.explain_ranking("smartphone", "Itel A23 Pro")
    check("explain returns strengths", r["found"] and isinstance(r["strengths"], list))

    # catalog_info
    ci = tools.catalog_info("smartphone")["by_category"]["smartphone"]
    check("catalog_info size", ci["count"] == 500 and len(ci["brands"]) > 0)
    check("catalog_info percentiles", "price_usd" in ci["percentiles"])

    # dispatch + error handling (tools must never raise)
    check("dispatch unknown tool -> error", "error" in tools.call_tool("nope", {}))
    check("dispatch unknown category -> error", "error" in tools.call_tool("search_products", {"category": "laptops"}))
    check("dispatch valid call", tools.call_tool("search_products", {"category": "smartphone", "n": 3})["returned"] == 3)


if __name__ == "__main__":
    print("Tool layer tests:")
    run()
    print(f"\nALL {_passed} TOOL TESTS PASSED")
