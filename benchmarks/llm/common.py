"""Shared harness for the LLMSampler benchmarks.

An *arm* is a sampler configuration; a *problem* is an objective with a direction. Each
(problem, arm, seed) run writes one JSON file with the per-trial record (value, best-so-far,
cumulative USD, source, violations) so runs are resumable and the report is a pure read.

Keys are read from the environment, or from ``OPTUNAI_ENV_FILE`` (a ``KEY=value`` file) when
set; nothing is written back.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import time
from typing import Any

import optuna
from optuna.samplers import Ledger
from optuna.samplers import LLMSampler
from optuna.samplers import Model
from optuna.trial import TrialState


RESULTS_ROOT = Path(os.environ.get("OPTUNAI_RESULTS", "/var/tmp/optunai/results"))

MODELS: dict[str, Callable[[], Model]] = {
    "claude": lambda: Model("openrouter/anthropic/claude-sonnet-5.5"),
    "haiku": lambda: Model("openrouter/anthropic/claude-haiku-5.5"),
    "gemini": lambda: Model(
        "vertex_ai/gemini-3.5-flash",
        vertex_project=os.environ.get("GOOGLE_CLOUD_PROJECT", "binit-244703"),
        vertex_location="global",
    ),
    "qwen": lambda: Model(
        "openai/qwen3.7-plus",
        api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY_INTL"),
    ),
}


def load_env() -> None:
    path = os.environ.get("OPTUNAI_ENV_FILE")
    if not path or not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def make_sampler(
    arm: str, seed: int, context: Any, ledger_path: Path
) -> optuna.samplers.BaseSampler:
    """``random`` | ``tpe`` | ``cmaes`` | ``llm[-noctx|-new]:<model key>``."""
    if arm == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if arm == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if arm == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    if arm == "gp":
        return optuna.samplers.GPSampler(seed=seed)
    kind, _, model_key = arm.partition(":")
    if not kind.startswith("llm"):
        raise ValueError(f"unknown arm {arm!r}")
    model = MODELS[model_key]()
    return LLMSampler(
        model,
        context=None if kind == "llm-noctx" else context,
        allow_new_params=kind == "llm-new",
        seed=seed,
        ledger=Ledger(ledger_path),
    )


def run(
    *,
    bench: str,
    problem: str,
    arm: str,
    seed: int,
    n_trials: int,
    objective: Callable[[optuna.Trial], float],
    direction: str,
    context: Any = None,
    pruner: optuna.pruners.BasePruner | None = None,
    extra_record: Callable[[optuna.Study], dict[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = RESULTS_ROOT / bench
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{problem}__{arm.replace(':', '-')}__seed{seed}.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    ledger_path = out.with_suffix(".ledger.jsonl")
    if ledger_path.exists():
        ledger_path.unlink()
    sampler = make_sampler(arm, seed, context, ledger_path)
    study = optuna.create_study(
        study_name=f"{bench}/{problem}/{arm}/seed{seed}",
        direction=direction,
        sampler=sampler,
        pruner=pruner if pruner is not None else optuna.pruners.NopPruner(),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    wall = time.time() - t0
    sign = 1.0 if direction == "minimize" else -1.0
    best = float("inf")
    cum_usd = 0.0
    trials: list[dict[str, Any]] = []
    for t in study.trials:
        attrs = t.system_attrs
        usd = float(attrs.get("llm:usd") or 0.0)
        cum_usd += usd
        if t.state == TrialState.COMPLETE and t.value is not None:
            best = min(best, sign * t.value)
        trials.append(
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "best": sign * best if best != float("inf") else None,
                "params": t.params,
                "source": attrs.get("llm:source"),
                "usd": usd,
                "cum_usd": cum_usd,
                "tokens_in": attrs.get("llm:tokens_in"),
                "tokens_out": attrs.get("llm:tokens_out"),
                "violations": attrs.get("llm:violations") or [],
                "new_params": attrs.get("llm:new_params"),
                "hypothesis": attrs.get("llm:hypothesis"),
                "evidence": attrs.get("llm:evidence"),
                "intermediate": t.intermediate_values,
                "user_attrs": t.user_attrs,
                "duration_s": (t.duration.total_seconds() if t.duration else None),
            }
        )
    calls = []
    if ledger_path.exists():
        calls = [
            json.loads(line)
            for line in ledger_path.read_text().splitlines()
            if line.strip() and '"kind": "call"' in line
        ]
    record = {
        "bench": bench,
        "problem": problem,
        "arm": arm,
        "seed": seed,
        "direction": direction,
        "n_trials": n_trials,
        "wall_s": wall,
        "best_value": study.best_value
        if any(t.state == TrialState.COMPLETE for t in study.trials)
        else None,
        "best_params": study.best_params
        if any(t.state == TrialState.COMPLETE for t in study.trials)
        else None,
        "trials": trials,
        "calls": [
            {
                k: c.get(k)
                for k in (
                    "trial",
                    "purpose",
                    "model",
                    "tokens_in",
                    "tokens_out",
                    "usd",
                    "latency_s",
                    "error",
                )
            }
            for c in calls
        ],
        "n_fallback": sum(t["source"] == "fallback" for t in trials),
        "n_llm": sum(t["source"] in ("llm", "repair", "batch") for t in trials),
        "n_violation_trials": sum(bool(t["violations"]) for t in trials),
        "n_repair": sum(c.get("purpose") == "repair" for c in calls),
        "usd": cum_usd,
    }
    if extra_record is not None:
        record.update(extra_record(study))
    out.write_text(json.dumps(record, default=str))
    return record


def summarise(record: dict[str, Any]) -> str:
    return (
        f"{record['bench']}/{record['problem']}/{record['arm']}/seed{record['seed']}: "
        f"best={record['best_value']} usd={record['usd']:.4f} fallback={record['n_fallback']} "
        f"violations={record['n_violation_trials']} wall={record['wall_s']:.0f}s"
    )
