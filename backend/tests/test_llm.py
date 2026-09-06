"""Tests for the shared single-shot narration helper."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.agents._llm import narrate


class DemoSchema(BaseModel):
    label: str
    score: int


def _tool_use(name, data, block_id="c1"):
    return SimpleNamespace(type="tool_use", name=name, input=data, id=block_id)


def _text(t="hi"):
    return SimpleNamespace(type="text", text=t)


def _resp(content, stop_reason="tool_use"):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


DEFAULT = DemoSchema(label="fallback", score=0)


def _call(client):
    return narrate(
        client,
        system="sys",
        user="usr",
        schema=DemoSchema,
        tool_name="submit_demo",
        tool_description="desc",
        default=DEFAULT,
    )


def test_narrate_returns_validated_model():
    client = FakeClient([_resp([_tool_use("submit_demo", {"label": "a", "score": 3})])])
    out = _call(client)
    assert isinstance(out, DemoSchema)
    assert out.label == "a" and out.score == 3
    assert len(client.calls) == 1


def test_narrate_forces_the_tool_and_passes_the_schema():
    client = FakeClient([_resp([_tool_use("submit_demo", {"label": "a", "score": 3})])])
    _call(client)
    kw = client.calls[0]
    assert kw["tool_choice"] == {"type": "tool", "name": "submit_demo"}
    assert kw["tools"][0]["input_schema"] == DemoSchema.model_json_schema()
    assert kw["model"] == "claude-sonnet-5"


def test_narrate_retries_once_then_succeeds():
    client = FakeClient([
        _resp([_tool_use("submit_demo", {"label": "a"})]),           # missing score -> invalid
        _resp([_tool_use("submit_demo", {"label": "b", "score": 9})]),
    ])
    out = _call(client)
    assert out.score == 9
    assert len(client.calls) == 2
    # the retry message carries the validation error back to the model
    assert client.calls[1]["messages"][-1]["role"] == "user"


def test_narrate_falls_back_to_default_after_two_validation_failures():
    client = FakeClient([
        _resp([_tool_use("submit_demo", {"label": "a"})]),
        _resp([_tool_use("submit_demo", {"score": "not-an-int"})]),
    ])
    out = _call(client)
    assert out is DEFAULT
    assert len(client.calls) == 2


def test_narrate_falls_back_when_no_tool_use_block():
    client = FakeClient([_resp([_text("no tool call")], stop_reason="end_turn")])
    out = _call(client)
    assert out is DEFAULT
    assert len(client.calls) == 1
