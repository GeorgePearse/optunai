from __future__ import annotations

import json
import os
import time
from typing import Any
import urllib.error
import urllib.request


URL = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
MODEL = "typesafe-ai/jev"
RETRY_DELAYS_S = (1.0, 4.0, 12.0)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def boolean(instructions: str, true: str = "", false: str = "") -> dict[str, Any]:
    q: dict[str, Any] = {"type": "boolean", "instructions": instructions}
    if true or false:
        q["criteria"] = {"true": true, "false": false}
    return q


class Jev:
    """Typed evaluation calls to Jev on the Vercel AI Gateway (synchronous).

    ``ask`` sends one text ``state`` and a dict of named questions and returns, per question,
    ``{"choice", "p", "probabilities"}``. The key is ``AI_GATEWAY_API_KEY`` unless given.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 60.0) -> None:
        self.api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY", "")
        self.timeout = timeout
        self.name = MODEL
        self.calls = 0
        self.usd = 0.0
        self.input_tokens = 0
        self.last_latency_s = 0.0

    @staticmethod
    def configured() -> bool:
        return bool(os.environ.get("AI_GATEWAY_API_KEY"))

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "ai-gateway-auth-method": "api-key",
            "ai-gateway-protocol-version": "0.0.1",
            "ai-evaluation-model-specification-version": "4",
            "ai-model-id": MODEL,
            "content-type": "application/json",
        }
        body = json.dumps({"state": state, "questions": questions}).encode()
        t0 = time.perf_counter()
        data: dict[str, Any] | None = None
        for i, delay in enumerate((0.0, *RETRY_DELAYS_S)):
            time.sleep(delay)
            req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code in RETRY_STATUSES and i < len(RETRY_DELAYS_S):
                    continue
                raise RuntimeError(f"Jev HTTP {e.code}: {e.read()[:300]!r}") from e
            except urllib.error.URLError:
                if i == len(RETRY_DELAYS_S):
                    raise
        assert data is not None
        self.last_latency_s = time.perf_counter() - t0
        meta = data.get("providerMetadata") or {}
        self.calls += 1
        self.usd += float((meta.get("gateway") or {}).get("cost") or 0.0)
        self.input_tokens += int((data.get("usage") or {}).get("inputTokens") or 0)
        out: dict[str, dict[str, Any]] = {}
        for name, raw in (data.get("answers") or {}).items():
            if raw.get("type") == "boolean":
                p = float(raw.get("probability") or 0.0)
                out[name] = {
                    "choice": p >= 0.5,
                    "p": p,
                    "probabilities": {"true": p, "false": 1 - p},
                }
            else:
                probs = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
                pick = raw.get("choice")
                out[name] = {
                    "choice": pick,
                    "p": probs.get(str(pick), 0.0),
                    "probabilities": probs,
                }
        return out
