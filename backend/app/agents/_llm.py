"""Shared single-shot narration helper.

Modules that compute every factual field deterministically and need an LLM
only for prose + enumerated choices use ``narrate``: one forced-tool
``messages.create`` call, the tool input validated against a Pydantic schema,
one retry with the validation error fed back, then a caller-supplied
schema-valid default. ``client`` defaults to ``app.agents._llm_client.get_client()``
(provider selected by ``AGENTDS_LLM_PROVIDER``, same as Modules 1/2); tests
inject a fake.

``tool_choice`` is passed to every provider, but Ollama has no way to force
a specific tool call (see ``_llm_client``'s module docstring) — the system
prompt is given an explicit instruction to call the tool as a result, so a
model that can't be forced still has the strongest possible nudge to.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from app.agents._llm_client import get_client

MODEL = "claude-sonnet-5"


def narrate(
    client: Any | None = None,
    *,
    system: str,
    user: str,
    schema: type[BaseModel],
    tool_name: str,
    tool_description: str,
    default: BaseModel,
    model: str = MODEL,
    max_tokens: int = 4096,
) -> BaseModel:
    client = client if client is not None else get_client()
    tools = [
        {
            "name": tool_name,
            "description": tool_description,
            "input_schema": schema.model_json_schema(),
        }
    ]
    system = (
        f"{system}\n\n"
        f"You must respond by calling the `{tool_name}` tool — it is the "
        "only tool available and the only way to answer. Some providers "
        "cannot force this choice, so call it yourself rather than "
        "replying in plain text."
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]

    for _ in range(2):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            tool_choice={"type": "tool", "name": tool_name},
            messages=messages,
        )
        block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if block is None:
            return default
        try:
            return schema.model_validate(block.input)
        except ValidationError as exc:
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That {tool_name} call failed schema validation:\n{exc}\n"
                        f"Call {tool_name} again with a valid payload."
                    ),
                }
            )

    return default
