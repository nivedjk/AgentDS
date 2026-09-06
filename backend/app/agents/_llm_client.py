"""Provider-agnostic LLM client, shared by every module that talks to an LLM.

Two distinct call shapes both go through this module:

  - Modules 1 (Data Understanding) and 2 (Cleaning) run an agentic tool-use
    loop directly against ``self.client.messages.create(model=..., tools=...,
    messages=...)``, reading back ``response.stop_reason`` / multi-block
    ``response.content`` across several turns.
  - Modules 3-7 (Visualization, Recommendation, Training, Explainability,
    Report) make exactly one forced-tool call each through
    ``app/agents/_llm.py``'s ``narrate()`` helper, which also calls
    ``self.client.messages.create(...)`` — same shape, single turn.

Both are Anthropic's Messages API shape. ``get_client()`` returns something
that satisfies that exact surface for either provider:

  - ``AGENTDS_LLM_PROVIDER=anthropic`` -> a real ``anthropic.Anthropic()``
    (unchanged, existing behavior).
  - ``AGENTDS_LLM_PROVIDER=ollama`` (the default) -> ``OllamaClient``, which
    translates the same Anthropic-shaped request into Ollama's ``/api/chat``
    tool-calling API and translates the response back. Ollama connection
    details reused from the standalone probes
    (``phase/qwen3-module1-probe/``, ``phase/qwen3-module2-probe/``): 13/13
    successful runs, 0 malformed tool calls across both, is the basis for
    this integration.

Ollama's ``/api/chat`` has no ``tool_choice`` equivalent — it lets the model
choose freely among the tools it's given, it cannot be forced to call one.
``narrate()`` always offers exactly one tool, so it compensates by stating
in the prompt that the tool must be called; this module's ``create()``
still accepts an Anthropic-shaped ``tool_choice`` kwarg (for the real
Anthropic path) and silently ignores it for Ollama rather than erroring.

Scope decision: ``check_llm_available()`` is a pre-flight check only (same
tier as the ``if not os.environ.get("ANTHROPIC_API_KEY")`` gates it
replaces across routers) — it confirms the provider is usable *right now*,
before a run starts. A provider that goes down mid-run is not specially
handled, for either provider — that already wasn't handled before this
change (an Anthropic API failure mid-loop/mid-call was never caught
either), so this preserves the existing behavior tier rather than
expanding it. Module 5 (Training)'s narration is the one exception worth
calling out explicitly: it's optional and best-effort by design (see
``app/routers/train.py``), so there a pre-flight failure and a mid-call
failure are handled identically — both are swallowed and just mean no
narrative, never a failed training run.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"

# Validated against phase/qwen3-module2-probe/ (the larger of the two tool
# schemas, 8 tools vs Module 1's 5): peak observed prompt across all 5 probe
# runs stayed under 3000 tokens against an 8192 ceiling. temperature=0.7 is
# the exact setting both probes ran their real 10+3 runs under.
OLLAMA_TEMPERATURE = 0.7
OLLAMA_NUM_CTX = 8192

_CHAT_TIMEOUT_S = 900.0
_REACHABILITY_TIMEOUT_S = 2.0


class LLMUnavailableError(RuntimeError):
    """The configured LLM provider (for Module 1's /analyze and Module 2's
    /clean only) isn't usable right now. Routers turn this into a 503 with
    ``str(exc)`` as the detail — never a raw connection-error traceback."""


def _provider() -> str:
    return os.environ.get("AGENTDS_LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()


def get_client() -> Any:
    """Returns an object exposing ``.messages.create(...)`` in Anthropic's
    shape, backed by whichever provider ``AGENTDS_LLM_PROVIDER`` selects."""
    provider = _provider()
    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic()
    if provider == "ollama":
        return OllamaClient()
    raise LLMUnavailableError(
        f"Unknown AGENTDS_LLM_PROVIDER {provider!r} (expected 'ollama' or 'anthropic')."
    )


def check_llm_available() -> None:
    """Raises ``LLMUnavailableError`` with a human-readable reason if the
    configured provider isn't usable right now. Cheap, no model call."""
    provider = _provider()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set.")
        return
    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
        try:
            resp = httpx.get(f"{host}/api/tags", timeout=_REACHABILITY_TIMEOUT_S)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama not reachable at {host}: {exc}") from exc
        return
    raise LLMUnavailableError(
        f"Unknown AGENTDS_LLM_PROVIDER {provider!r} (expected 'ollama' or 'anthropic')."
    )


# ---------------------------------------------------------------------------
# Anthropic-shaped response objects the two agents' loops already expect
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class LLMResponse:
    stop_reason: str
    content: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Anthropic <-> Ollama translation (reused/generalized from the probes'
# to_ollama_tools / message-building logic)
# ---------------------------------------------------------------------------


def _to_ollama_tools(anthropic_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in anthropic_tools
    ]


def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    """Read an attribute off either a dataclass block (this module's own
    TextBlock/ToolUseBlock, re-appended by the agent between turns) or a
    plain dict (defensive — nothing in this codebase appends dict-shaped
    assistant content today, but tool_result content always is)."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _to_ollama_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-shaped ``messages`` (built by DataUnderstandingAgent.run /
    CleaningAgent.run, unchanged) -> Ollama ``/api/chat`` message list."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]

    for m in messages:
        role = m["role"]
        content = m["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                btype = _block_attr(block, "type")
                if btype == "text":
                    text_parts.append(_block_attr(block, "text", "") or "")
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "function": {
                                "name": _block_attr(block, "name"),
                                "arguments": _block_attr(block, "input") or {},
                            }
                        }
                    )
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
            continue

        # role == "user" with list content: Anthropic tool_result dicts.
        # Each becomes its own Ollama "tool" message, in order. Anthropic's
        # `is_error` is a structural flag with no Ollama /api/chat equivalent
        # — the only way the model can tell a call failed is from the text,
        # so a failed result is prefixed to make that explicit rather than
        # silently dropping the signal.
        for tr in content:
            text = str(_block_attr(tr, "content", ""))
            if _block_attr(tr, "is_error", False) and not text.startswith("Error:"):
                text = f"Error: {text}"
            out.append({"role": "tool", "content": text})

    return out


def _from_ollama_response(data: dict[str, Any]) -> LLMResponse:
    message = data.get("message", {}) or {}
    content_text = message.get("content", "") or ""
    tool_calls = message.get("tool_calls", []) or []

    blocks: list[Any] = []
    if content_text:
        blocks.append(TextBlock(text=content_text))
    for tc in tool_calls:
        fn = tc.get("function", {}) or {}
        blocks.append(
            ToolUseBlock(
                id=f"ollama_call_{uuid.uuid4().hex[:10]}",
                name=fn.get("name"),
                input=fn.get("arguments") or {},
            )
        )

    stop_reason = "tool_use" if tool_calls else "end_turn"
    return LLMResponse(stop_reason=stop_reason, content=blocks)


class _OllamaMessages:
    def __init__(self, host: str, model: str):
        self._host = host.rstrip("/")
        self._model = model

    def create(
        self,
        *,
        model: str | None = None,  # Anthropic-signature compat; ignored, see below
        max_tokens: int = 4096,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,  # Anthropic-signature compat; ignored, see below
    ) -> LLMResponse:
        # `model` is accepted so call sites (`self.client.messages.create(model=MODEL, ...)`)
        # don't need a provider-specific branch — but it's always the caller's
        # Anthropic model id (e.g. "claude-sonnet-5"), so it's ignored in favor
        # of this client's own configured OLLAMA_MODEL.
        #
        # `tool_choice` is accepted for the same reason (narrate() passes it
        # unconditionally): Ollama's /api/chat has no way to force a specific
        # tool call, so it's silently ignored here rather than raising. The
        # caller (narrate()) compensates by stating in the prompt that the
        # single available tool must be called.
        payload = {
            "model": self._model,
            "messages": _to_ollama_messages(system, messages),
            "tools": _to_ollama_tools(tools),
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = httpx.post(f"{self._host}/api/chat", json=payload, timeout=_CHAT_TIMEOUT_S)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama request failed at {self._host}: {exc}") from exc
        return _from_ollama_response(resp.json())


class OllamaClient:
    """Drop-in stand-in for ``anthropic.Anthropic()``: exposes the same
    ``.messages.create(...)`` surface, backed by a local Ollama server."""

    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.messages = _OllamaMessages(self.host, self.model)
