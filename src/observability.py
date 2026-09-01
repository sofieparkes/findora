"""
observability.py
Structured, append-only turn logging + lightweight analytics + LangSmith status.

Every turn is logged as one JSON line: utterance -> intent -> confidence ->
raw/validated/dropped slots -> action -> candidate count -> latency. This is
the raw material for dashboards (intent distribution, drop reasons, drop-off)
and for the future learning-to-rank flywheel. Logging never raises — a logging
failure must not break a conversation.

LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true and
LANGCHAIN_API_KEY are set in the environment (langchain picks them up); this
module just reports whether it's active.
"""

import os
import json
import time
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR = Path(os.getenv("CRS_LOG_DIR", "logs"))
TURNS_LOG = LOG_DIR / "turns.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"

_lock = threading.Lock()


def langsmith_status() -> str:
    """Human-readable LangSmith tracing status for display."""
    on = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true")
    has_key = bool(os.getenv("LANGCHAIN_API_KEY"))
    if on and has_key:
        return f"on · project={os.getenv('LANGCHAIN_PROJECT', 'default')}"
    if on and not has_key:
        return "enabled but LANGCHAIN_API_KEY missing"
    return "off"


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass  # never let logging break a turn


def log_turn(record: Dict[str, Any]) -> None:
    """Append one turn record (timestamp added automatically)."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    _append_jsonl(TURNS_LOG, rec)


def log_feedback(event: Dict[str, Any]) -> None:
    """Append a user-feedback event (selection / comparison) for the flywheel."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    _append_jsonl(FEEDBACK_LOG, rec)


class Timer:
    """`with Timer() as t: ...; t.ms` — wall-clock milliseconds."""
    def __enter__(self):
        self._t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc):
        self.ms = round((time.perf_counter() - self._t0) * 1000, 1)
        return False


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return rows[-limit:] if limit else rows


def load_turns(limit: Optional[int] = None) -> List[Dict]:
    return _load_jsonl(TURNS_LOG, limit)


def funnel(rows: List[Dict]) -> Dict[str, Any]:
    """
    Conversation-level drop-off / conversion metrics, grouped by session_id:
      - sessions
      - reached_recommend (%): conversations that produced a recommendation
      - abandoned_at_question (%): conversations whose LAST action was a
        clarification question (user dropped off mid-funnel)
      - avg_turns_to_recommend: how many turns until the first recommendation
    """
    by_session: Dict[str, List[Dict]] = {}
    for r in rows:
        sid = r.get("session_id") or "_nosession"
        by_session.setdefault(sid, []).append(r)

    n = len(by_session)
    if n == 0:
        return {"sessions": 0}

    reached = 0
    abandoned_q = 0
    turns_to_rec = []
    for sid, turns in by_session.items():
        actions = [t.get("action") for t in turns]
        if "recommend" in actions:
            reached += 1
            turns_to_rec.append(actions.index("recommend") + 1)
        elif actions and actions[-1] == "ask_clarification":
            abandoned_q += 1

    return {
        "sessions": n,
        "reached_recommend_pct": round(100 * reached / n, 1),
        "abandoned_at_question_pct": round(100 * abandoned_q / n, 1),
        "avg_turns_to_recommend": round(sum(turns_to_rec) / len(turns_to_rec), 1) if turns_to_rec else None,
    }


def analytics() -> Dict[str, Any]:
    """Aggregate logged turns into a small dashboard-ready summary."""
    rows = load_turns()
    intents, actions, drops = Counter(), Counter(), Counter()
    confs, lats = [], []
    for r in rows:
        intents[r.get("intent")] += 1
        actions[r.get("action")] += 1
        for d in (r.get("dropped_slots") or []):
            drops[d.get("reason")] += 1
        if isinstance(r.get("confidence"), (int, float)):
            confs.append(r["confidence"])
        if isinstance(r.get("latency_ms"), (int, float)):
            lats.append(r["latency_ms"])
    return {
        "turns": len(rows),
        "intents": dict(intents),
        "actions": dict(actions),
        "drop_reasons": dict(drops),
        "avg_confidence": round(sum(confs) / len(confs), 3) if confs else None,
        "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else None,
        "funnel": funnel(rows),
        "langsmith": langsmith_status(),
    }
