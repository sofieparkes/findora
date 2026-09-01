# CHANGES: Findora (after the poster session)

Feedback came from **6 peer reviewers**. Below we mention every piece of feedback, what we changed in response and why — and,
where we choose **not** to change something, the reasoning. The post-change **evaluation** is at the end.

---

## 1. What we changed

## "Expand the data: add product details, descriptions and reviews." (reviewer 2; also our own stated limitation)
**Changed: ** Every phone now has a grounded **description** and a clearly-labeled **AI review summary**,
generated *only* from its real specs + rating (`enrich_smartphones.py` →
`datasets/smartphone_enrichment.json`, joined into the catalog at load time, similar to image cache). They appear in a product's **Details** view and feed the semantic "vibe" search. Coverage
is **100% of the 500 phones**, and the summary is explicitly marked *"AI summary (generated from specs &
rating)"* so it is never mistaken for a real user review.

*Why generated rather than scraped real reviews:* we made an attempt to get the reviews for the matching products but could not find the reviews for all the products in one dataset.


## "Show one more real-world example of the full recommendation process." (reviewer 7)
**Changed:** Added in the updated poster.

## "Address the local-model limitation and the agent's failure handling more concretely." (reviewer 6)
**Changed:**  Bad tool arguments are validated and **never crash** (tools never raise); a request needing more than the 4-round budget is forced to a final grounded answer; local function-calling is unreliable on small models, which is *why* the system is cloud-primary.

## "Be honest about / improve loading times." (reviewer 4)
**Changed:** We tried through prompt caching but still depends on 2 API calls. 

### "Surface the less-intuitive commands (e.g. you can tell it to clear all filters)." (reviewer 4)
**Changed:** Added a **"Handy commands"** section to the in-app *"ℹ️ How it works"* panel listing the
non-obvious natural-language commands — *"clear all filters", "ignore that" (undo), "cheaper ones",
"compare those", "why this one?", "actually, show me headphones"* — so users discover them.

### "The '700 True' tooltip was confusing." (reviewer 5)
**Resolved:** We could not reproduce it in the current build: the post-session UI rework relabeled every
spec/score value with a unit or **Yes/No** (verified across the spec table, the score breakdown, and the
compare view), so no raw value renders without a label. It appears the rework already eliminated it.

## 2. Considered but not changed (with reasoning)

- **Cross-shop price comparison:** Out of scope: the reviewer agreed. Findora is a
  single-catalog recommender; live multi-retailer pricing needs commercial price-feed integrations well
  beyond this project.
- **Expanding the catalog to ~1,000 phones:** We deliberately **enriched the existing 500
  in place** rather than swap the dataset: it keeps our validated evaluation baseline comparable, keeps
  the clean USD prices the ranking depends on, and the enrichment (not the row count) was the actual gap.
- **Lower loading time via response streaming:** Acknowledged, not implemented. The agent is
  already faster on average and we added the honest latency analysis; streaming the final reply (lower
  *perceived* latency) is the clear next step but was deprioritised in favour of the data work.
---