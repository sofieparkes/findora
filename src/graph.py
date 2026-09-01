"""
graph.py
Builds and compiles the LangGraph conversation graph.
Import `run_turn()` to process one user message and get back an updated state.
"""

import os

from langgraph.graph import StateGraph, END

import observability
from state import DialogueState, initial_state
from nodes import (
    intent_and_extract_node,
    state_updater_node,
    retrieve_and_act_node,
    response_generator_node,
)

# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(DialogueState)

    # Register nodes
    builder.add_node("intent_extract",  intent_and_extract_node)
    builder.add_node("state_update",    state_updater_node)
    builder.add_node("retrieve_act",    retrieve_and_act_node)
    builder.add_node("respond",         response_generator_node)

    # Linear pipeline (all turns follow the same path)
    builder.set_entry_point("intent_extract")
    builder.add_edge("intent_extract", "state_update")
    builder.add_edge("state_update",   "retrieve_act")
    builder.add_edge("retrieve_act",   "respond")
    builder.add_edge("respond",        END)

    return builder.compile()


# Compile once at import time
_graph = build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_turn(state: DialogueState, user_message: str, session_id: str = None) -> DialogueState:
    """
    Process one user turn and return the updated state.

    Args:
        state:        Current dialogue state (from initial_state() or previous turn)
        user_message: Raw text from the user
        session_id:   Optional id for observability grouping.

    Returns:
        Updated DialogueState with new response, filters, candidates, etc.
    """
    # Engine: the tool-calling AGENT is the default (it beat the pipeline on the
    # eval gate — persona quality 0.41 -> 0.85, robustness 82% -> 93%). The legacy
    # LangGraph pipeline below remains the fallback via ENGINE=pipeline.
    if os.getenv("ENGINE", "agent").lower() != "pipeline":
        from agent import run_turn_agent
        return run_turn_agent(state, user_message, session_id)

    # Append user message to history before invoking
    updated_messages = state["messages"] + [{"role": "user", "content": user_message}]

    input_state = {
        **state,
        "user_input": user_message,
        "messages": updated_messages,
    }

    with observability.Timer() as t:
        new_state: DialogueState = _graph.invoke(input_state)

    # Append assistant response to history
    new_state["messages"] = new_state["messages"] + [
        {"role": "assistant", "content": new_state["response"]}
    ]

    # Structured turn log: utterance -> intent/confidence -> slots -> action.
    observability.log_turn({
        "session_id": session_id,
        "utterance": user_message,
        "intent": new_state.get("intent"),
        "confidence": new_state.get("intent_confidence"),
        "category": new_state.get("category"),
        "raw_slots": new_state.get("raw_extracted_filters"),
        "validated_slots": new_state.get("extracted_filters"),
        "dropped_slots": new_state.get("dropped_slots"),
        "active_filters": new_state.get("active_filters"),
        "action": new_state.get("action"),
        "n_candidates": len(new_state.get("candidates", [])),
        "latency_ms": t.ms,
    })

    return new_state