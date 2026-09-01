"""
make_poster.py
Generates poster.pdf, three vertically-stacked A3-landscape pages
(page 1 = top, 2 = middle, 3 = bottom), as required for submission.

  python make_poster.py

Design goal: the "2-metre test". A passer-by should grasp WHAT it is, HOW it
is evaluated, and the RESULT from across the room, so each page leads with one
big visual (headline numbers / architecture diagram / evaluation-setup diagram)
and keeps prose to short captions.

Quantitative metrics are pulled live from eval_agent.json + eval_baseline.json
(run `python eval_harness.py` and `ENGINE=pipeline python eval_harness.py`).
Edit the CONFIG block below to set group members / title.
"""

import json
import os

from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.colors import HexColor, white, Color
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

# ── CONFIG ───────────────────────────────────────────────────────────────────
BRAND = "Findora"
TAGLINE = "Conversational shop assistant for smartphones & headphones"
POSITIONING = ("A multi-turn LLM recommender, re-architected from a structured pipeline into a "
               "hybrid tool-calling agent and proven on an evaluation gate.")
GROUP = "Group 26"
COURSE = "Applied Generative AI · SS 2026"
MEMBERS = ["Arthur Galois", "Jakob Meltzer", "Sofie Parkesaas", "Muhammad Azeem Shahzad"]
# ─────────────────────────────────────────────────────────────────────────────

PAGE = landscape(A3)            # (1190.55, 841.89) pts
PW, PH = PAGE

INDIGO = HexColor(0x6366F1)
VIOLET = HexColor(0x8B5CF6)
PINK   = HexColor(0xEC4899)
INK    = HexColor(0x1E293B)
MUTED  = HexColor(0x64748B)
LINE   = HexColor(0xE2E6F0)
SOFT   = HexColor(0xF5F6FC)
LAV    = HexColor(0xEEF2FF)
GREEN  = HexColor(0x16A34A)
RED    = HexColor(0xDC2626)
TEAL   = HexColor(0x0EA5E9)


def load_metrics():
    def _load(f):
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                return json.load(fh)
        return {}
    return {"agent": _load("eval_agent.json"),
            "baseline": _load("eval_baseline.json"),
            "legacy": _load("eval_results.json")}


def g(d, *keys, default=None):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return default
    return d


# ── Low-level drawing helpers (top-left coordinate system) ───────────────────

def Y(top):
    return PH - top


def gradient_band(c, x, top, w, h, c1, c2, steps=120):
    strip = w / steps
    for i in range(steps):
        t = i / (steps - 1)
        col = Color(c1.red + (c2.red - c1.red) * t,
                    c1.green + (c2.green - c1.green) * t,
                    c1.blue + (c2.blue - c1.blue) * t)
        c.setFillColor(col)
        c.rect(x + i * strip, Y(top + h), strip + 1, h, stroke=0, fill=1)


def panel(c, x, top, w, h, title, accent=VIOLET, fill=white):
    c.setFillColor(fill)
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.roundRect(x, Y(top + h), w, h, 10, stroke=1, fill=1)
    c.setFillColor(accent)
    c.roundRect(x, Y(top + h), 6, h, 3, stroke=0, fill=1)
    c.rect(x + 3, Y(top + h), 4, h, stroke=0, fill=1)
    if title:
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(x + 18, Y(top + 22), title)
    return top + 34


def wrapped(c, text, x, top, w, font="Helvetica", size=10.5, leading=14,
            color=INK):
    c.setFillColor(color)
    c.setFont(font, size)
    for ln in simpleSplit(text, font, size, w):
        c.drawString(x, Y(top), ln)
        top += leading
    return top


def bullets(c, items, x, top, w, size=10.5, leading=14, gap=4, color=INK):
    for it in items:
        c.setFillColor(VIOLET)
        c.setFont("Helvetica-Bold", size)
        c.drawString(x, Y(top), "•")
        c.setFillColor(color)
        c.setFont("Helvetica", size)
        for ln in simpleSplit(it, "Helvetica", size, w - 14):
            c.drawString(x + 14, Y(top), ln)
            top += leading
        top += gap
    return top


def harrow(c, x, y, length, color=VIOLET, width=2.0):
    """Left-to-right arrow from (x,y) spanning `length` pts (top-down y)."""
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x, Y(y), x + length, Y(y))
    c.setFillColor(color)
    p = c.beginPath()
    p.moveTo(x + length, Y(y))
    p.lineTo(x + length - 7, Y(y - 4.5))
    p.lineTo(x + length - 7, Y(y + 4.5))
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def flow_node(c, x, top, w, h, title, sub=None, fill=LAV, stroke=VIOLET,
              tcol=INK, tsize=12, bold=True):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.5)
    c.roundRect(x, Y(top + h), w, h, 9, stroke=1, fill=1)
    c.setFillColor(tcol)
    tlines = title.split("\n")
    c.setFont("Helvetica-Bold" if bold else "Helvetica", tsize)
    ty = top + 19
    for ln in tlines:
        c.drawCentredString(x + w / 2, Y(ty), ln)
        ty += tsize + 2
    if sub:
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8.6)
        for ln in sub.split("\n"):
            for wl in simpleSplit(ln, "Helvetica", 8.6, w - 16):
                c.drawCentredString(x + w / 2, Y(ty), wl)
                ty += 10.5


def metric_tile(c, x, top, w, h, after, before, label):
    gradient_band(c, x, top, w, h, INDIGO, VIOLET)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 34)
    c.drawCentredString(x + w / 2, Y(top + 44), str(after))
    c.setFont("Helvetica", 10.5)
    c.drawCentredString(x + w / 2, Y(top + 60), f"was {before}")
    c.setFont("Helvetica-Bold", 10)
    ly = top + 76
    for ln in simpleSplit(label, "Helvetica-Bold", 10, w - 14):
        c.drawCentredString(x + w / 2, Y(ly), ln)
        ly += 12


def _ctx(m):
    A = (m or {}).get("agent", {}) or {}
    B = (m or {}).get("baseline", {}) or {}

    def cc(d):
        bt = g(d, "robustness", "by_tag", default={}) or {}
        co, cp = bt.get("count", {}), bt.get("compound", {})
        return (co.get("pass", 0) + cp.get("pass", 0), co.get("total", 0) + cp.get("total", 0))

    cca, cct = cc(A)
    ccb, cctb = cc(B)
    cct = cct or cctb

    def q(d):
        v = g(d, "personas", "overall", default=None)
        return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"

    headline = [
        (q(A), q(B), "Conversation quality (LLM-judge)"),
        (f"{g(A, 'robustness', 'pct', default='n/a')}%", f"{g(B, 'robustness', 'pct', default='n/a')}%",
         "Adversarial & safety pass-rate"),
        (f"{cca}/{cct}" if cct else "n/a", f"{ccb}/{cct}" if cct else "n/a",
         "Count & compound requests"),
    ]
    return A, B, headline


# ── PAGE 1 - Brand · Headline results · Abstract · Problem · Use cases ────────

def page1(c, m):
    A, B, headline = _ctx(m)

    # Hero
    gradient_band(c, 0, 0, PW, 150, INDIGO, PINK)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 46)
    c.drawString(40, Y(66), BRAND)
    c.setFont("Helvetica", 15)
    c.drawString(40, Y(92), TAGLINE)
    c.setFont("Helvetica", 11)
    for i, ln in enumerate(simpleSplit(POSITIONING, "Helvetica", 11, PW - 540)):
        c.drawString(40, Y(114 + i * 15), ln)
    c.setFont("Helvetica-Bold", 20)
    c.drawRightString(PW - 40, Y(52), GROUP)
    c.setFont("Helvetica", 12)
    c.drawRightString(PW - 40, Y(72), COURSE)
    c.setFont("Helvetica", 10.5)
    c.drawRightString(PW - 40, Y(92), " · ".join(MEMBERS))

    top = 172
    colw = (PW - 80 - 30) / 2

    # Abstract (left)
    by = panel(c, 40, top, colw, 330, "Abstract", INDIGO)
    abstract = (
        "Findora is a conversational recommender that acts as a shop assistant for tech products, taken "
        "from prototype to a deployable design. v1 was a structured LangGraph pipeline: split NLU (router "
        "plus slot-filling) guarded by schema validation and entity resolution, vague language ('cheap', "
        "'good camera') grounded in dataset percentiles, hybrid pandas plus dense-embedding retrieval, and "
        "TOPSIS ranking. Adding intents one by one proved brittle (whack-a-mole), so we built an evaluation "
        "harness (persona simulation, LLM-judge, adversarial / safety suites) and re-architected the "
        "dialogue brain into a hybrid tool-calling AGENT over the same deterministic core. On the shared "
        "gate the agent roughly doubles conversation quality, reaches 100% on adversarial / safety, and "
        "fixes count / compound requests, with no per-scenario code. The pipeline is kept as a fallback."
    )
    wrapped(c, abstract, 58, by + 8, colw - 36, leading=20, size=15)

    # Problem (right)
    px = 40 + colw + 30
    py = panel(c, px, top, colw, 330, "Initial Situation & Problem", PINK)
    py = wrapped(c, "Traditional recommenders work silently (ranked lists, 'people also bought'). Users "
                    "often start vague and refine as they talk. A conversational recommender (CRS) must "
                    "instead elicit preferences in real time and adapt.",
                 px + 18, py + 6, colw - 36, leading=19.5, size=14)
    py += 12
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(px + 18, Y(py), "Core challenges:")
    py += 24
    bullets(c, [
        "Robust intent detection and preference elicitation from messy natural language.",
        "Reasoning over a dialogue state to pick the next best action, not blindly answering.",
        "Safely translating critiques ('cheaper but better camera') into database queries.",
        "What information does the system actually need to reliably recommend a set of items?",
    ], px + 18, py, colw - 36, size=13.5, leading=18, gap=6)

    # Sample use cases strip (full width)
    uy = 514
    by = panel(c, 40, uy, PW - 80, 288, "Sample Use Cases", VIOLET)
    cards = [
        ("1 · Specific lookup", "'A Samsung Android phone with 8GB RAM'",
         "Pre-fills brand, OS and spec, then recommends immediately, ranked, with no needless questions."),
        ("2 · Guided exploration", "'I want a smartphone'",
         "Asks a focused sequence: OS -> price -> battery -> camera -> RAM, honouring 'any' / 'skip'."),
        ("3 · Refinement / critique", "'cheaper ones' · 'bigger battery'",
         "Anchors relative terms to the last results; refinements are additive; 'forget X' removes a filter."),
        ("4 · Vague & vibe language", "'cheap phone for gaming'",
         "Maps 'cheap' to the p25 price; reads 'gaming' via semantic embeddings over spec descriptions."),
    ]
    cw = (PW - 80 - 36 - 30) / 4
    for i, (h, ex, d) in enumerate(cards):
        cx = 58 + i * (cw + 10)
        c.setFillColor(SOFT)
        c.setStrokeColor(LINE)
        c.roundRect(cx, Y(by + 242), cw, 242, 8, stroke=1, fill=1)
        c.setFillColor(HexColor(0x4338CA))
        c.setFont("Helvetica-Bold", 12)
        c.drawString(cx + 12, Y(by + 24), h)
        c.setFillColor(VIOLET)
        c.setFont("Helvetica-Oblique", 11.5)
        ey = by + 46
        for ln in simpleSplit(ex, "Helvetica-Oblique", 11.5, cw - 24):
            c.drawString(cx + 12, Y(ey), ln)
            ey += 15
        c.setFillColor(INK)
        c.setFont("Helvetica", 11)
        ey += 5
        for ln in simpleSplit(d, "Helvetica", 11, cw - 24):
            c.drawString(cx + 12, Y(ey), ln)
            ey += 14.5

    _footer(c, 1)
    c.showPage()


# ── PAGE 2 - Architecture (visual centrepiece) ───────────────────────────────

def page2(c, m):
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 25)
    c.drawString(40, Y(46), "Approach: From Pipeline to Tool-Calling Agent")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 12.5)
    c.drawString(40, Y(68), "v2 is a hybrid tool-calling agent over a deterministic core (the v1 pipeline, "
                            "kept as the ENGINE=pipeline fallback).")

    # ── v2 AGENT LOOP (the centrepiece) ──
    c.setFillColor(VIOLET)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, Y(94), "v2 · Agent loop (default engine)")
    nodes = [
        ("User\nmessage", None, SOFT, LINE),
        ("LLM PLANS", "policy + tool schemas\ndecides which tool to call", LAV, VIOLET),
        ("DETERMINISTIC TOOLS", "search · details · compare\nexplain · top-picks · catalog", HexColor(0xDCFCE7), GREEN),
        ("GROUNDED REPLY", "natural text + product cards\naction: recommend / compare / respond", HexColor(0xFCE7F3), PINK),
    ]
    bw, bh, gap = 232, 86, 52
    total = len(nodes) * bw + (len(nodes) - 1) * gap
    x0 = (PW - total) / 2
    ytop = 104
    for i, (t, s, fl, st) in enumerate(nodes):
        x = x0 + i * (bw + gap)
        flow_node(c, x, ytop, bw, bh, t, s, fill=fl, stroke=st,
                  tsize=12 if i else 13)
        if i < len(nodes) - 1:
            harrow(c, x + bw, ytop + bh / 2, gap, color=VIOLET)
    # return loop arrow (tools -> LLM): "iterate up to 4 rounds"
    lx1 = x0 + 1 * (bw + gap) + bw / 2          # under LLM PLANS
    lx2 = x0 + 2 * (bw + gap) + bw / 2          # under TOOLS
    ly = ytop + bh + 20
    c.setStrokeColor(TEAL)
    c.setLineWidth(2)
    c.line(lx1, Y(ytop + bh), lx1, Y(ly))
    c.line(lx1, Y(ly), lx2, Y(ly))
    c.line(lx2, Y(ly), lx2, Y(ytop + bh))
    c.setFillColor(TEAL)
    p = c.beginPath()
    p.moveTo(lx1, Y(ytop + bh))
    p.lineTo(lx1 - 4.5, Y(ytop + bh + 7))
    p.lineTo(lx1 + 4.5, Y(ytop + bh + 7))
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString((lx1 + lx2) / 2, Y(ly - 4), "iterate up to 4 LLM <-> tool rounds")
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 9.5)
    c.drawCentredString(PW / 2, Y(ytop + bh + 40),
                        "Grounding lives in the tools: they validate every filter, return only real catalog "
                        "rows, and honour an explicit count. The LLM only plans and composes (no fabricated specs).")

    # ── v1 PIPELINE (the deterministic core, also the fallback) ──
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, Y(254), "v1 · Deterministic core  (the agent calls these as tools; also runs standalone as the fallback)")
    pipe = ["User", "ROUTER\nintent · conf", "SLOT\nEXTRACTOR", "VALIDATE\n+ RESOLVE",
            "STATE\nmerge · undo", "RETRIEVE\nfilter -> rank", "RESPONSE"]
    pbw, pbh, pgap = 138, 58, 14
    ptotal = len(pipe) * pbw + (len(pipe) - 1) * pgap
    px0 = (PW - ptotal) / 2
    pytop = 266
    pfill = [SOFT, LAV, LAV, HexColor(0xFCE7F3), LAV, HexColor(0xDCFCE7), LAV]
    for i, label in enumerate(pipe):
        x = px0 + i * (pbw + pgap)
        lines = label.split("\n")
        flow_node(c, x, pytop, pbw, pbh, lines[0],
                  "\n".join(lines[1:]) if len(lines) > 1 else None,
                  fill=pfill[i], stroke=VIOLET if i in (1, 2) else LINE, tsize=11)
        if i < len(pipe) - 1:
            harrow(c, x + pbw, pytop + pbh / 2, pgap, color=MUTED, width=1.6)

    # ── Cross-cutting platform rails ──
    rails = [
        ("MEMORY", "per-user profiles · 'use my usual' · personalised greeting", INDIGO),
        ("RANKING", "TOPSIS (MCDM) <-> learned LTR · semantic blend · feedback", PINK),
        ("OBSERVABILITY", "LangSmith tracing · per-turn JSONL · tokens & latency", VIOLET),
        ("IMAGES", "offline enrichment -> GSMArena / Wikipedia -> card image", INDIGO),
    ]
    ry = 336
    rw = (PW - 80 - 3 * 14) / 4
    for i, (h, d, col) in enumerate(rails):
        rx = 40 + i * (rw + 14)
        c.setFillColor(white)
        c.setStrokeColor(col)
        c.setLineWidth(1.4)
        c.roundRect(rx, Y(ry + 60), rw, 60, 8, stroke=1, fill=1)
        c.setFillColor(col)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(rx + 12, Y(ry + 22), h)
        c.setFillColor(INK)
        c.setFont("Helvetica", 9.3)
        ey = ry + 38
        for ln in simpleSplit(d, "Helvetica", 9.3, rw - 24):
            c.drawString(rx + 12, Y(ey), ln)
            ey += 12

    # ── Key design decisions (left) + Models & Tools (right) ──
    dtop = 406
    colw = (PW - 80 - 30) / 2
    dy = panel(c, 40, dtop, colw, 396, "Key Design Decisions", INDIGO)
    bullets(c, [
        "Plan-and-call agent instead of enumerating intents: open-ended and compound requests compose "
        "from tools, with no per-scenario code.",
        "Validate every LLM-proposed slot against a schema; drop unknown / out-of-range / wrong-type, "
        "so bad arguments never reach the query engine.",
        "Grounding lives in deterministic tools (real rows only, explicit counts, honest 'not in catalog').",
        "Entity resolution as a real layer (alias dictionary plus fuzzy match), not ever-growing prompt rules.",
        "Deterministic guardrails for the precise bits: undo ('ignore that') and brand exclusion.",
        "Hybrid retrieval: exact pandas filters plus dense embeddings (model2vec) for 'vibe' queries.",
        "Transparent TOPSIS ranking; a pluggable learned ranker trains from click feedback.",
        "Prompt caching on the static system-plus-tools prefix keeps cost and latency down.",
    ], 58, dy + 6, colw - 36, size=11.5, leading=15.5, gap=6)

    tx = 40 + colw + 30
    ty = panel(c, tx, dtop, colw, 396, "Models & Tools", PINK)
    ty += 4
    rows = [
        ("Engine", "tool-calling agent (default) <-> LangGraph pipeline (fallback)"),
        ("Tools", "search · details · compare · explain · top-picks · catalog"),
        ("LLM", "OpenRouter · GPT-4o-mini (function-calling) · local Ollama opt."),
        ("Embeddings", "model2vec · potion-base-8M (static, no GPU)"),
        ("Ranking", "TOPSIS (Hwang & Yoon) · numpy logistic LTR"),
        ("Data / UI", "pandas · 2 x 500 catalog · Streamlit chat plus cards"),
        ("Observability", "LangSmith · per-turn JSONL · tokens & latency"),
        ("Evaluation", "persona sim plus LLM-judge plus 28 adversarial / safety checks"),
    ]
    for k, v in rows:
        c.setFillColor(HexColor(0xBE185D))
        c.setFont("Helvetica-Bold", 12)
        c.drawString(tx + 18, Y(ty), k)
        c.setFillColor(INK)
        c.setFont("Helvetica", 11.5)
        vlines = simpleSplit(v, "Helvetica", 11.5, colw - 172)
        for j, ln in enumerate(vlines):
            c.drawString(tx + 152, Y(ty + j * 14.5), ln)
        ty += 26 + 14.5 * (len(vlines) - 1)

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 9.5)
    c.drawString(tx + 18, Y(dtop + 384),
                 "Cross-cutting: skip · undo · multi-intent · category-switch · percentile grounding · memory.")

    _footer(c, 2)
    c.showPage()


# ── PAGE 3 - Results · Evaluation setup · Methodology · Learnings ─────────────

def page3(c, m):
    A, B, headline = _ctx(m)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 25)
    c.drawString(40, Y(46), "Results, Evaluation & Learnings")

    # Hero metrics - three big before/after tiles (the judge-corroborated headliners)
    mtop, mh = 58, 84
    tw = (PW - 80 - 2 * 16) / 3
    for i, (af, bf, lab) in enumerate(headline):
        metric_tile(c, 40 + i * (tw + 16), mtop, tw, mh, af, bf, lab)

    # Honest latency & cost strip (full width)
    la = g(A, "latency", default={}) or {}
    lb = g(B, "latency", default={}) or {}
    cache, tin = g(A, "tokens", "avg_cached"), g(A, "tokens", "avg_in")
    cache_pct = f"{round(100 * cache / tin)}%" if cache and tin else "n/a"
    lat_top = 148
    ly = panel(c, 40, lat_top, PW - 80, 60, "Latency & cost per turn  (report the distribution, not just the average)",
               TEAL)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(58, Y(ly + 4),
                 f"agent: avg {la.get('avg_s', '?')}s  ·  p95 {la.get('p95_s', '?')}s  ·  max {la.get('max_s', '?')}s "
                 f"   |   pipeline: avg {lb.get('avg_s', '?')}s  ·  p95 {lb.get('p95_s', '?')}s  ·  "
                 f"max {lb.get('max_s', '?')}s   |   {cache_pct} input tokens cached")
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 9.3)
    c.drawString(58, Y(ly + 20),
                 "The agent wins on PREDICTABILITY and COST, not raw speed: its p95 is higher on genuinely "
                 "multi-step turns, while the pipeline average is inflated by API-retry outliers (max ~222s).")

    # Evaluation-setup diagram (full width) - answers 'how is it measured' visually
    ev_top = 214
    panel(c, 40, ev_top, PW - 80, 132, "How we measure it (engine-agnostic A/B gate)", INDIGO)
    ny = ev_top + 52
    nh = 44
    flow_node(c, 60, ny, 150, nh, "10 PERSONAS", "LLM 'customer'\nmulti-turn dialogues",
              fill=LAV, stroke=VIOLET, tsize=11)
    flow_node(c, 60 + 150 + 40, ny, 150, nh, "FINDORA", "system under test\n(agent or pipeline)",
              fill=SOFT, stroke=INK, tsize=11)
    jx = 60 + 2 * (150 + 40)
    flow_node(c, jx, ny - 24, 210, nh, "LLM JUDGE", "5 dims, 0-1, unweighted",
              fill=HexColor(0xEDE9FE), stroke=VIOLET, tsize=11)
    flow_node(c, jx, ny + 28, 210, nh, "28 DETERMINISTIC CHECKS", "judge-free assertions",
              fill=HexColor(0xDCFCE7), stroke=GREEN, tsize=10)
    flow_node(c, jx + 210 + 40, ny, 150, nh, "SCORECARD", "eval_*.json\nquality · safety · cost",
              fill=HexColor(0xFCE7F3), stroke=PINK, tsize=11)
    harrow(c, 60 + 150, ny + nh / 2, 40, color=VIOLET)
    harrow(c, 60 + 2 * 150 + 40, ny + nh / 2, 40, color=VIOLET)
    harrow(c, jx + 210, ny + nh / 2, 40, color=PINK)

    # Demo conversation (left)
    ctop = 352
    colw = (PW - 80 - 30) / 2
    cy = panel(c, 40, ctop, colw, 268, "Demo: example conversation", VIOLET)
    convo = [
        ("u", "suggest me 2 best phones"),
        ("a", "Two top picks by value: the Redmi Note 12 Pro+ and the Explorer. (count = 2 honoured)"),
        ("u", "compare them"),
        ("a", "Side-by-side: the Pro+ leads on RAM and battery; the Explorer is cheaper. Cards below."),
        ("u", "compare it with the iPhone 15 Pro Max"),
        ("a", "I do not carry the iPhone 15 Pro Max; the closest I have is the iPhone 15 Pro. (no fabrication)"),
    ]
    yy = cy + 8
    for who, txt in convo:
        is_u = who == "u"
        bw = colw - 56
        lines = simpleSplit(txt, "Helvetica", 9.5, bw - 20)
        bh = 11 + len(lines) * 12
        bx = (40 + colw - 18 - bw) if is_u else 58
        c.setFillColor(LAV if is_u else SOFT)
        c.setStrokeColor(LINE)
        c.roundRect(bx, Y(yy + bh), bw, bh, 8, stroke=1, fill=1)
        c.setFillColor(HexColor(0x4338CA) if is_u else INK)
        c.setFont("Helvetica-Bold", 8.3)
        c.drawString(bx + 10, Y(yy + 10.5), "You" if is_u else "Findora")
        c.setFillColor(INK)
        c.setFont("Helvetica", 9.5)
        for j, ln in enumerate(lines):
            c.drawString(bx + 10, Y(yy + 22 + j * 12), ln)
        yy += bh + 7

    # Evaluation methodology (right) - the rigorous detail reviewers want
    lx = 40 + colw + 30
    var = g(A, "personas", "overall_std", default=None)
    runs = g(A, "personas", "runs", default=None)
    var_line = (f"Variance: persona quality is {g(A, 'personas', 'overall', default='n/a')} "
                f"+/- {var} over {runs} runs (--runs N)." if var is not None else
                "Variance: --runs N repeats the sims and reports mean +/- std (observed run-to-run sigma ~0.02).")
    my = panel(c, lx, ctop, colw, 268, "Evaluation Methodology", PINK)
    bullets(c, [
        "Quality = mean of 5 judge dims (goal, honored, grounded, helpful, safe), each 0-1, UNWEIGHTED, "
        "averaged over 10 personas.",
        "Judge-free corroboration: 28 deterministic assertions move 82% -> 100% (count / compound 1/6 -> 6/6) "
        "with no LLM judging, so the gain is not a judge artifact.",
        "Bias guard: --judge-model grades with a different model family than the system under test.",
        var_line,
        "Reproducible: python eval_harness.py  vs  ENGINE=pipeline python eval_harness.py.",
    ], lx + 18, my + 4, colw - 36, size=9.8, leading=12.3, gap=3)

    # Learnings & limitations (full width, two sub-columns)
    btop = 624
    panel(c, 40, btop, PW - 80, 182, "Learnings & Limitations", GREEN)
    half = (PW - 80 - 36) / 2
    ly2 = btop + 40
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(58, Y(ly2), "What worked")
    ly2 += 16
    bullets(c, [
        "Build the eval gate FIRST: persona sim plus LLM-judge turned 'feels better' into a defensible number.",
        "The tool-calling agent fixed open-world gaps (counts, compound, anaphora) with NO per-scenario code.",
        "Grounding in deterministic tools means the LLM only plans: honest replies, no fabricated specs.",
        "Hybrid wins: deterministic guardrails plus an LLM brain beat either alone.",
    ], 58, ly2, half - 10, size=11, leading=13.8, gap=3)
    rx2 = 58 + half + 18
    ry3 = btop + 40
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(rx2, Y(ry3), "What was hard / we would change")
    ry3 += 16
    bullets(c, [
        "Enumerating intents was whack-a-mole: fixing one scenario broke another. The agent removed that ceiling.",
        "Optimising latency over-tightened replies and dropped quality; the gate caught it and we re-balanced.",
        "Agent is cloud-first: function-calling on small local models is unreliable (a known trade-off).",
        "A few personas are capped by the dataset (obscure products), not the system; richer data would lift them.",
    ], rx2, ry3, half - 10, size=11, leading=13.8, gap=3)

    _footer(c, 3)
    c.showPage()


def _footer(c, n):
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.line(40, Y(PH - 28), PW - 40, Y(PH - 28))
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(40, Y(PH - 16), f"{BRAND}  ·  {GROUP}")
    c.drawCentredString(PW / 2, Y(PH - 16), COURSE)
    c.drawRightString(PW - 40, Y(PH - 16), f"Page {n} of 3")


def main():
    m = load_metrics()
    c = canvas.Canvas("poster.pdf", pagesize=PAGE)
    page1(c, m)
    page2(c, m)
    page3(c, m)
    c.save()
    note = "" if (m.get("agent") or m.get("baseline")) else "  [no eval_*.json: run eval_harness.py for live metrics]"
    print("Wrote poster.pdf (3 x A3 landscape)" + note)


if __name__ == "__main__":
    main()
