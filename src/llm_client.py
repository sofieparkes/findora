"""
llm_client.py
Provider-agnostic chat-model layer used by the NLU router, the slot extractor,
and the response generator. Centralising this gives one place to:

  • register the available models (a cloud default + any LOCAL Ollama models),
  • switch the ACTIVE model at runtime (the UI sets it each rerun),
  • build + cache one client per model.

Ollama is wired through its OpenAI-COMPATIBLE endpoint (http://localhost:11434/v1)
so it reuses the already-installed `langchain-openai` — no extra dependency. The
model id format is "<provider>:<model>" (e.g. "ollama:mistral",
"openrouter:openai/gpt-4o-mini"), which the active selection round-trips without
needing a network call.
"""

import os
import re
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

# Active-model id (module-level → shared by all nodes in the process). The UI
# calls set_active_model() each rerun; None means "use the cloud default".
_active_id: Optional[str] = None
_cache: Dict[str, Any] = {}   # model id -> built LangChain chat model


# ── Model registry ────────────────────────────────────────────────────────────

def _cloud_models() -> List[Dict[str, Any]]:
    """The configured cloud model(s). For OpenRouter you can list several via
    OPENROUTER_MODELS=comma,separated to compare cloud models too."""
    provider = os.getenv("LLM_PROVIDER", "openrouter")
    if provider == "anthropic":
        m = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        return [{"id": f"anthropic:{m}", "provider": "anthropic", "model": m,
                 "label": m, "local": False}]
    if provider == "openai":
        m = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return [{"id": f"openai:{m}", "provider": "openai", "model": m,
                 "label": m, "local": False}]
    if provider in ("gemini", "vertex", "google"):
        m = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return [{"id": f"gemini:{m}", "provider": "gemini", "model": m,
                 "label": m, "local": False}]
    # openrouter (default)
    listed = os.getenv("OPENROUTER_MODELS", "").strip()
    names = [m.strip() for m in listed.split(",") if m.strip()]
    if not names:
        names = [os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")]
    return [{"id": f"openrouter:{m}", "provider": "openrouter", "model": m,
             "label": m, "local": False} for m in names]


def list_ollama_models(timeout: float = 1.0) -> List[Dict[str, Any]]:
    """Installed local models from the Ollama server. Empty list if Ollama isn't
    running/reachable (so the app degrades gracefully to cloud-only)."""
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        r.raise_for_status()
        out: List[Dict[str, Any]] = []
        for m in r.json().get("models", []):
            name = m.get("name") or m.get("model")
            if name:
                out.append({"id": f"ollama:{name}", "provider": "ollama",
                            "model": name, "label": name, "local": True})
        return out
    except Exception:
        return []


def available_models() -> List[Dict[str, Any]]:
    """Cloud default(s) + any locally-installed Ollama models. Used by the UI to
    populate the model picker (it makes a quick local HTTP call — cache upstream
    if calling frequently)."""
    return _cloud_models() + list_ollama_models()


def ollama_running() -> bool:
    return len(list_ollama_models()) > 0


# ── Active selection ──────────────────────────────────────────────────────────

def _spec_from_id(model_id: str) -> Dict[str, Any]:
    """Reconstruct a model spec from its id ('<provider>:<model>') — no network."""
    provider, _, model = model_id.partition(":")
    return {"id": model_id, "provider": provider, "model": model,
            "label": model, "local": provider == "ollama"}


def set_active_model(model_id: Optional[str]) -> None:
    global _active_id
    _active_id = model_id or None


def get_active_model() -> Dict[str, Any]:
    """The currently-selected model spec (defaults to the first cloud model)."""
    if _active_id:
        return _spec_from_id(_active_id)
    return _cloud_models()[0]


# ── Build + cache ─────────────────────────────────────────────────────────────

def _build(spec: Dict[str, Any]):
    provider = spec["provider"]
    model = spec["model"]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0)
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
        )
    if provider in ("gemini", "vertex", "google"):
        # Gemini on Vertex AI in EXPRESS mode: authenticate with a Vertex API key
        # (no service account / ADC needed). The unified google-genai SDK handles
        # this when GOOGLE_GENAI_USE_VERTEXAI=True + GOOGLE_API_KEY are set.
        from langchain_google_genai import ChatGoogleGenerativeAI
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("key")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            max_output_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "2048")),
        )
    if provider == "ollama":
        # Ollama speaks the OpenAI chat API at /v1 — reuse ChatOpenAI, no new dep.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            openai_api_key="ollama",                       # ignored by Ollama
            openai_api_base=f"{OLLAMA_BASE_URL}/v1",
            temperature=0,
            # Local models on CPU are slow for big prompts (a 7B model measured
            # ~540s for one router call on this box). Use a generous, configurable
            # per-call timeout and—crucially—NO retries, so a slow call isn't
            # silently attempted 3x (which made the wait ~3x longer → the failure).
            timeout=float(os.getenv("OLLAMA_TIMEOUT", "900")),
            max_retries=0,
            max_tokens=512,                                # short JSON / prose — caps generation time
        )
    raise ValueError(
        f"Unknown LLM provider: '{provider}'. Choose from: openai, anthropic, "
        f"openrouter, ollama."
    )


def get_llm():
    """Return the cached chat model for the ACTIVE selection (building on first use)."""
    spec = get_active_model()
    mid = spec["id"]
    if mid not in _cache:
        _cache[mid] = _build(spec)
    return _cache[mid]


def cap_tokens(llm, max_tokens: int):
    """Bind a generation-length cap using the parameter the ACTIVE provider accepts.
    OpenAI-family uses `max_tokens`; Gemini/Vertex uses `max_output_tokens` and
    rejects `max_tokens`. For Gemini the cap is already applied at build time, so we
    return the model unchanged to avoid an invalid-argument error. Keeps the agent
    engine-agnostic — it just calls cap_tokens() instead of .bind(max_tokens=...)."""
    if get_active_model().get("provider") in ("gemini", "vertex", "google"):
        return llm
    return llm.bind(max_tokens=max_tokens)


def get_llm_for(model_id: str):
    """Build + cache a chat model for a SPECIFIC model id ('<provider>:<model>'),
    independent of the active selection. Lets a caller run a second model alongside
    the active one, e.g. an evaluation judge from a DIFFERENT family than the
    system-under-test (to rule out shared-model bias)."""
    if model_id not in _cache:
        _cache[model_id] = _build(_spec_from_id(model_id))
    return _cache[model_id]


# ── JSON parsing helper (unchanged) ───────────────────────────────────────────

def parse_json_response(text: str) -> Dict[str, Any]:
    """Robustly extract a JSON object from an LLM response."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
