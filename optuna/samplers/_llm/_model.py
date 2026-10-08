from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any
from typing import Protocol
from typing import runtime_checkable


# Approximate list prices in USD per million tokens (input, output) for models LiteLLM does not
# price itself. Override per model with ``Model(price_per_m=(in, out))``.
PRICE_TABLE_PER_M: dict[str, tuple[float, float]] = {
    "qwen3.7-plus": (0.4, 1.2),
    "qwen3.5-plus": (0.4, 1.2),
    "qwen3.6-plus": (0.4, 1.2),
}

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ModelResponse:
    """One completion: the raw text, the parsed JSON if any, and what it cost."""

    text: str
    parsed: dict[str, Any] | None
    model: str
    input_tokens: int
    output_tokens: int
    usd: float | None
    latency_s: float
    error: str | None = None


@runtime_checkable
class ModelLike(Protocol):
    """What :class:`LLMSampler` and :class:`LLMPruner` need from a model object.

    Any object with a ``name`` attribute and a ``complete(system, user, schema)`` method
    returning a :class:`ModelResponse` can replace :class:`Model`.
    """

    name: str

    def complete(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> ModelResponse: ...


def parse_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from ``text``; ``None`` if there is none."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.DOTALL)
    for candidate in (stripped, *(m.group(0) for m in _JSON_BLOCK.finditer(stripped))):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class Model:
    """A chat model behind LiteLLM, with JSON output and cost accounting.

    ``model`` is any LiteLLM model string: ``"openrouter/anthropic/claude-sonnet-5.5"``,
    ``"vertex_ai/gemini-3.5-flash"``, ``"openai/qwen3.7-plus"`` with ``api_base`` pointing at an
    OpenAI-compatible endpoint, ``"ollama/..."`` and so on. Keys come from the environment
    (``OPENROUTER_API_KEY``, ``GEMINI_API_KEY``, application default credentials, ...) or from
    ``api_key``. Extra keyword arguments are passed to ``litellm.completion`` unchanged, for
    example ``vertex_project`` and ``vertex_location``.

    ``complete`` asks for JSON-schema output when the provider supports it, then plain JSON
    mode, then free text with a parse-and-repair pass. ``usd`` is taken from LiteLLM's price
    table, then :data:`PRICE_TABLE_PER_M`, then ``price_per_m``; it is ``None`` when none of
    them knows the model.

    Args:
        model: LiteLLM model string.
        api_key: Overrides the environment key for the provider.
        api_base: Base URL for OpenAI-compatible endpoints.
        temperature: Sampling temperature; ``None`` sends the provider default (some models
            reject any other value).
        max_tokens: Output token cap per call.
        timeout: Seconds per call.
        max_retries: LiteLLM retries on transient errors.
        price_per_m: ``(input, output)`` USD per million tokens, used when nothing else prices
            the model.
        kwargs: Passed through to ``litellm.completion``.
    """

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-sonnet-5.5",
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 8192,
        timeout: float = 180.0,
        max_retries: int = 2,
        price_per_m: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.name = model
        self._api_key = api_key
        self._api_base = api_base
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._price_per_m = price_per_m
        self._kwargs = kwargs
        self.calls = 0
        self.usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self._schema_mode: str | None = None

    def __repr__(self) -> str:
        return f"Model({self.name!r})"

    def _call(self, messages: list[dict[str, str]], response_format: Any) -> Any:
        import litellm

        litellm.suppress_debug_info = True
        kwargs: dict[str, Any] = dict(
            model=self.name,
            messages=messages,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
            num_retries=self._max_retries,
            **self._kwargs,
        )
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._api_base is not None:
            kwargs["api_base"] = self._api_base
        if response_format is not None:
            kwargs["response_format"] = response_format
        return litellm.completion(**kwargs)

    def _cost(self, response: Any) -> float | None:
        try:
            import litellm

            usd = float(litellm.completion_cost(completion_response=response))
            if usd > 0:
                return usd
        except Exception:
            pass
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prices = self._price_per_m
        if prices is None:
            for key, value in PRICE_TABLE_PER_M.items():
                if key in self.name:
                    prices = value
                    break
        if prices is None:
            return None
        return (
            (usage.prompt_tokens or 0) * prices[0] + (usage.completion_tokens or 0) * prices[1]
        ) / 1e6

    def _complete_once(
        self, messages: list[dict[str, str]], schema: dict[str, Any] | None
    ) -> tuple[Any, str]:
        """Try structured output modes from strictest to loosest; remember what worked."""
        modes: list[tuple[str, Any]]
        if schema is None:
            modes = [("text", None)]
        else:
            modes = [
                (
                    "json_schema",
                    {
                        "type": "json_schema",
                        "json_schema": {"name": "proposal", "schema": schema, "strict": False},
                    },
                ),
                ("json_object", {"type": "json_object"}),
                ("text", None),
            ]
            if self._schema_mode is not None:
                modes = [m for m in modes if m[0] == self._schema_mode] + [
                    m for m in modes if m[0] != self._schema_mode
                ]
        last_error: Exception | None = None
        for mode, response_format in modes:
            try:
                response = self._call(messages, response_format)
                self._schema_mode = mode
                return response, mode
            except Exception as e:  # provider rejected the format or the call failed
                last_error = e
                message = str(e).lower()
                transient = any(
                    word in message
                    for word in ("rate", "timeout", "timed out", "overloaded", "503", "502")
                )
                if transient:
                    raise
        assert last_error is not None
        raise last_error

    def complete(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> ModelResponse:
        """Run one completion; when ``schema`` is given, parse JSON and repair once if needed."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t0 = time.perf_counter()
        try:
            response, _ = self._complete_once(messages, schema)
        except Exception as e:
            return ModelResponse(
                text="",
                parsed=None,
                model=self.name,
                input_tokens=0,
                output_tokens=0,
                usd=None,
                latency_s=time.perf_counter() - t0,
                error=f"{type(e).__name__}: {str(e)[:300]}",
            )
        text = response.choices[0].message.content or ""
        usage = response.usage
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        usd = self._cost(response)
        parsed = parse_json(text) if schema is not None else None
        if schema is not None and parsed is None and text:
            repair = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "That was not a single valid JSON object. Return only the JSON.",
                },
            ]
            try:
                response2, _ = self._complete_once(repair, schema)
                text = response2.choices[0].message.content or ""
                usage2 = response2.usage
                tokens_in += int(getattr(usage2, "prompt_tokens", 0) or 0)
                tokens_out += int(getattr(usage2, "completion_tokens", 0) or 0)
                usd2 = self._cost(response2)
                if usd is not None and usd2 is not None:
                    usd += usd2
                parsed = parse_json(text)
            except Exception:
                pass
        self.calls += 1
        self.input_tokens += tokens_in
        self.output_tokens += tokens_out
        if usd is not None:
            self.usd += usd
        return ModelResponse(
            text=text,
            parsed=parsed,
            model=self.name,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            usd=usd,
            latency_s=time.perf_counter() - t0,
        )
