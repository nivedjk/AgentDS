"""Tests for the provider-agnostic LLM client (`get_client()` /
`check_llm_available()` / `OllamaClient`), exercised directly here via
Module 1 (Data Understanding) and Module 2 (Cleaning)'s call shape. These
same functions now also back Modules 3/4/5/6/7's `narrate()` helper
(app/agents/_llm.py) — see `tests/test_visualization.py`,
`test_recommendation.py`, `test_training.py`, `test_explainability.py`,
and `test_report.py` for that call path's own gating/fallback tests.

None of these tests require a live Ollama server: the Ollama HTTP calls are
mocked via monkeypatching `httpx.get`/`httpx.post`.
"""

from __future__ import annotations

import httpx
import pytest

from app.agents._llm_client import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    LLMUnavailableError,
    OllamaClient,
    check_llm_available,
    get_client,
)

# ---------------------------------------------------------------------------
# get_client() — provider selection
# ---------------------------------------------------------------------------


def test_get_client_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("AGENTDS_LLM_PROVIDER", raising=False)
    client = get_client()
    assert isinstance(client, OllamaClient)


def test_get_client_explicit_ollama(monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "ollama")
    client = get_client()
    assert isinstance(client, OllamaClient)


def test_get_client_anthropic_provider_returns_anthropic_client(monkeypatch):
    import anthropic

    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = get_client()
    assert isinstance(client, anthropic.Anthropic)


def test_get_client_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "bogus")
    with pytest.raises(LLMUnavailableError):
        get_client()


def test_ollama_client_uses_env_host_and_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://example-host:9999")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:custom")
    client = OllamaClient()
    assert client.host == "http://example-host:9999"
    assert client.model == "qwen3:custom"


def test_ollama_client_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    client = OllamaClient()
    assert client.host == DEFAULT_OLLAMA_HOST
    assert client.model == DEFAULT_OLLAMA_MODEL


# ---------------------------------------------------------------------------
# check_llm_available() — the router pre-flight gate
# ---------------------------------------------------------------------------


def test_check_llm_available_anthropic_missing_key_raises(monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMUnavailableError, match="ANTHROPIC_API_KEY"):
        check_llm_available()


def test_check_llm_available_anthropic_with_key_passes(monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    check_llm_available()  # must not raise


def test_check_llm_available_ollama_unreachable_names_the_host(monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

    def fake_get(url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(LLMUnavailableError, match="http://localhost:11434"):
        check_llm_available()


def test_check_llm_available_ollama_reachable_passes(monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "ollama")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: FakeResponse())

    check_llm_available()  # must not raise


def test_check_llm_available_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "bogus")
    with pytest.raises(LLMUnavailableError):
        check_llm_available()


# ---------------------------------------------------------------------------
# OllamaClient.messages.create() — request/response translation
# ---------------------------------------------------------------------------

ANTHROPIC_TOOLS = [
    {
        "name": "check_duplicates",
        "description": "Check for duplicates.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "submit_report",
        "description": "Submit.",
        "input_schema": {
            "type": "object",
            "properties": {"narrative": {"type": "string"}},
            "required": ["narrative"],
        },
    },
]


def test_ollama_client_create_sends_translated_tools_and_seed_message(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "no tool call here", "tool_calls": []}}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    client = OllamaClient(host="http://localhost:11434", model="qwen3:8b")
    response = client.messages.create(
        model="claude-sonnet-5",  # Anthropic-signature kwarg; must be ignored
        max_tokens=4096,
        system="You are a test agent.",
        tools=ANTHROPIC_TOOLS,
        messages=[{"role": "user", "content": "Here is the dataset overview."}],
    )

    assert captured["url"] == "http://localhost:11434/api/chat"
    payload = captured["payload"]
    # The client's own configured model is used, NOT the Anthropic `model` kwarg.
    assert payload["model"] == "qwen3:8b"
    assert payload["messages"][0] == {"role": "system", "content": "You are a test agent."}
    assert payload["messages"][1] == {"role": "user", "content": "Here is the dataset overview."}
    tool_names = [t["function"]["name"] for t in payload["tools"]]
    assert tool_names == ["check_duplicates", "submit_report"]
    submit_tool = payload["tools"][1]["function"]
    assert submit_tool["parameters"] == ANTHROPIC_TOOLS[1]["input_schema"]

    # No tool_calls in the mocked response -> a plain text block, non-tool_use stop_reason.
    assert response.stop_reason != "tool_use"
    assert len(response.content) == 1
    assert response.content[0].type == "text"
    assert response.content[0].text == "no tool call here"


def test_ollama_client_create_returns_tool_use_block(monkeypatch):
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "check_duplicates", "arguments": {}}}
                    ],
                }
            }

    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: FakeResponse())

    client = OllamaClient(host="http://localhost:11434", model="qwen3:8b")
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4096,
        system="sys",
        tools=ANTHROPIC_TOOLS,
        messages=[{"role": "user", "content": "seed"}],
    )

    assert response.stop_reason == "tool_use"
    assert len(response.content) == 1
    block = response.content[0]
    assert block.type == "tool_use"
    assert block.name == "check_duplicates"
    assert block.input == {}
    assert isinstance(block.id, str) and block.id  # a synthetic id was assigned


def test_ollama_client_create_translates_multiturn_history_with_tool_results(monkeypatch):
    """Mirrors exactly what DataUnderstandingAgent/CleaningAgent append to
    `messages` between calls: an assistant turn carrying a prior
    response's tool_use content, followed by a user turn carrying
    Anthropic-style tool_result dicts (including an is_error one)."""
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "done", "tool_calls": []}}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    client = OllamaClient(host="http://localhost:11434", model="qwen3:8b")

    # First call to get a real ToolUseBlock in the shape the agent re-appends.
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, json, timeout: type(
            "R",
            (),
            {
                "status_code": 200,
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "check_duplicates", "arguments": {}}}
                        ],
                    }
                },
            },
        )(),
    )
    first_messages = [{"role": "user", "content": "seed"}]
    first_response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4096,
        system="sys",
        tools=ANTHROPIC_TOOLS,
        messages=first_messages,
    )
    tool_use_block = first_response.content[0]

    # Now build the next turn's messages exactly like the real agents do,
    # including a failed tool call (is_error: True) alongside a successful
    # one, since a single turn can contain both.
    messages = [
        {"role": "user", "content": "seed"},
        {"role": "assistant", "content": first_response.content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": '{"n_duplicate_rows": 0}',
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "Column 'bogus' not found.",
                    "is_error": True,
                },
            ],
        },
    ]

    monkeypatch.setattr(httpx, "post", fake_post)
    client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4096,
        system="sys",
        tools=ANTHROPIC_TOOLS,
        messages=messages,
    )

    sent = captured["payload"]["messages"]
    assert sent[0] == {"role": "system", "content": "sys"}
    assert sent[1] == {"role": "user", "content": "seed"}
    assert sent[2]["role"] == "assistant"
    assert sent[2]["tool_calls"][0]["function"]["name"] == "check_duplicates"
    assert sent[3] == {"role": "tool", "content": '{"n_duplicate_rows": 0}'}
    # is_error must survive translation as an explicit signal in the text —
    # Ollama's /api/chat tool messages have no separate error flag, so the
    # only way qwen3 can tell a tool call failed is from the content itself.
    assert sent[4]["role"] == "tool"
    assert sent[4]["content"].startswith("Error:")
    assert "Column 'bogus' not found." in sent[4]["content"]


def test_ollama_client_create_raises_llm_unavailable_on_connection_failure(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    client = OllamaClient(host="http://localhost:11434", model="qwen3:8b")
    with pytest.raises(LLMUnavailableError, match="http://localhost:11434"):
        client.messages.create(
            model="claude-sonnet-5",
            max_tokens=4096,
            system="sys",
            tools=ANTHROPIC_TOOLS,
            messages=[{"role": "user", "content": "seed"}],
        )
