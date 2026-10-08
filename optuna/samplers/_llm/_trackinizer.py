from __future__ import annotations

import json
from typing import Any


class TrackinizerSink:
    """Ledger sink that mirrors each proposal into trackinizer.

    A ``proposal`` event becomes a Belief (the hypothesis) and an Experiment (the trial) with a
    ``proves`` edge from the experiment to the belief. A ``trial_end`` event sets the
    experiment's outcome and the edge valence (+1 when the trial improved on the best value
    known at proposal time, -1 otherwise). Failures are logged by the ledger and never raised.

    Args:
        url: Base URL of a running trackinizer, for example ``http://127.0.0.1:8090``.
        labels: Labels attached to every record.
        timeout: Seconds per request.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8090",
        *,
        labels: tuple[str, ...] = ("optuna", "llm-sampler"),
        timeout: float = 10.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._labels = list(labels)
        self._timeout = timeout
        self._experiments: dict[tuple[str | None, int], str] = {}
        self._beliefs: dict[tuple[str | None, int], str] = {}

    def _request(self, method: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        req = urllib.request.Request(
            self._url + path,
            data=json.dumps(body, default=str).encode(),
            headers={"content-type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            payload = resp.read().decode()
        return json.loads(payload) if payload else {}

    def __call__(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        study = event.get("study")
        number = event.get("trial")
        if kind == "proposal" and isinstance(number, int):
            hypothesis = str(event.get("hypothesis") or "no hypothesis stated")
            labels = self._labels + [f"study:{study}"]
            belief = self._request(
                "POST",
                "/api/inquiries/belief",
                {
                    "title": hypothesis[:140],
                    "description": hypothesis
                    + "\n\nevidence: trials "
                    + ", ".join(str(n) for n in event.get("evidence") or [])
                    + f"\nmodel: {event.get('model')} prompt: {event.get('prompt_hash')}",
                    "labels": labels,
                },
            )
            experiment = self._request(
                "POST",
                "/api/inquiries/experiment",
                {
                    "title": f"{study} trial {number}",
                    "description": f"source: {event.get('source')}\n{hypothesis}",
                    "labels": labels,
                    "config": {
                        "params": event.get("params"),
                        "new_params": event.get("new_params"),
                        "model": event.get("model"),
                        "prompt_hash": event.get("prompt_hash"),
                    },
                },
            )
            self._beliefs[(study, number)] = belief["id"]
            self._experiments[(study, number)] = experiment["id"]
            self._request(
                "POST",
                f"/api/edges/{experiment['id']}/proves/{belief['id']}",
                {"note": "proposed by LLMSampler; valence set at trial end"},
            )
        elif kind == "trial_end" and isinstance(number, int):
            experiment_id = self._experiments.get((study, number))
            belief_id = self._beliefs.get((study, number))
            if experiment_id is None:
                return
            values = event.get("values")
            improved = event.get("improved")
            self._request(
                "PUT",
                f"/api/experiment/{experiment_id}/outcome",
                {
                    "value": f"state={event.get('state')} values={values} improved={improved}",
                    "reason": "LLMSampler.after_trial",
                },
            )
            if belief_id is not None and improved is not None:
                self._request(
                    "PUT",
                    f"/api/edges/{experiment_id}/proves/{belief_id}/valence",
                    {"value": 1.0 if improved else -1.0, "reason": "trial outcome"},
                )
