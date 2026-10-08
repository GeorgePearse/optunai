from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
import datetime
import json
import os
from pathlib import Path
import threading
from typing import Any
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from optuna.study import Study


Sink = Callable[[dict[str, Any]], None]


class Ledger:
    """Append-only JSONL record of everything the LLM sampler and pruner did and spent.

    One line per event. Kinds written by the library: ``call`` (one model call: prompt hash,
    model, tokens, USD, latency), ``proposal`` (params, hypothesis, evidence, source),
    ``new_param`` (an open-vocabulary dimension and its rationale), ``fallback`` (why a trial
    went to the fallback sampler), ``violation`` (a coerced or rejected value), ``prune``
    (a pruning decision with its probability) and ``trial_end`` (state and values).

    ``path`` defaults to ``<study_name>.ledger.jsonl`` beside the storage file (SQLite or
    journal file storage), or in the working directory for other storages. ``sink`` is any
    callable that receives each event dict; see ``TrackinizerSink`` in this package for one
    that writes trials as Experiments proving Beliefs.

    Args:
        path: File to append to; ``None`` derives it from the study on first use.
        sink: Optional callable invoked with every event after it is written.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None, *, sink: Sink | None = None):
        self._path = Path(path) if path is not None else None
        self._sink = sink
        self._lock = threading.Lock()
        self._study_name: str | None = None
        self.usd = 0.0
        self.calls = 0
        self.fallbacks = 0
        self.violations = 0
        self.proposals = 0

    @property
    def path(self) -> Path | None:
        return self._path

    def bind(self, study: Study) -> None:
        """Derive the file path from ``study`` if none was given."""
        self._study_name = study.study_name
        if self._path is not None:
            return
        base = Path.cwd()
        storage = study._storage
        inner = getattr(storage, "_backend", storage)  # unwrap _CachedStorage / JournalStorage
        url = getattr(inner, "url", None)
        if isinstance(url, str) and url.startswith("sqlite:///"):
            base = Path(url[len("sqlite:///") :]).resolve().parent
        file_path = getattr(inner, "_file_path", None)
        if isinstance(file_path, str):
            base = Path(file_path).resolve().parent
        self._path = base / f"{study.study_name}.ledger.jsonl"

    def write(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Append one event and return it."""
        event: dict[str, Any] = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
            "kind": kind,
            "study": self._study_name,
            **fields,
        }
        usd = fields.get("usd")
        with self._lock:
            if kind == "call":
                self.calls += 1
                if isinstance(usd, (int, float)):
                    self.usd += float(usd)
            elif kind == "fallback":
                self.fallbacks += 1
            elif kind == "violation":
                self.violations += 1
            elif kind == "proposal":
                self.proposals += 1
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, default=str) + "\n")
        if self._sink is not None:
            try:
                self._sink(event)
            except Exception as e:  # a sink must never break the study
                import optuna

                optuna.logging.get_logger(__name__).warning(f"ledger sink failed: {e}")
        return event

    def lines(self) -> Iterator[dict[str, Any]]:
        """Read every event back from the file."""
        if self._path is None or not self._path.exists():
            return iter(())
        with self._path.open(encoding="utf-8") as f:
            return iter([json.loads(line) for line in f if line.strip()])

    def totals(self) -> dict[str, Any]:
        """Counters for this process: USD, calls, proposals, fallbacks, violations."""
        return {
            "usd": round(self.usd, 6),
            "calls": self.calls,
            "proposals": self.proposals,
            "fallbacks": self.fallbacks,
            "violations": self.violations,
            "path": str(self._path) if self._path else None,
        }
