"""Prompt assembly, output schema, and validation of LLM proposals against Optuna distributions."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_INTERMEDIATE_POINTS = 8


def load_prompt(name: str) -> tuple[str, str]:
    """Return ``(text, hash)`` of a versioned prompt file; the hash changes with the text."""
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]


class ProposalError(ValueError):
    """A proposal could not be made valid without another model call."""


def describe_distribution(dist: BaseDistribution) -> dict[str, Any]:
    if isinstance(dist, FloatDistribution):
        d: dict[str, Any] = {"type": "float", "low": dist.low, "high": dist.high}
        if dist.log:
            d["log"] = True
        if dist.step is not None:
            d["step"] = dist.step
        return d
    if isinstance(dist, IntDistribution):
        d = {"type": "int", "low": dist.low, "high": dist.high}
        if dist.log:
            d["log"] = True
        if dist.step != 1:
            d["step"] = dist.step
        return d
    if isinstance(dist, CategoricalDistribution):
        return {"type": "categorical", "choices": list(dist.choices)}
    return {"type": type(dist).__name__, "repr": repr(dist)}


def describe_search_space(search_space: dict[str, BaseDistribution]) -> dict[str, Any]:
    return {name: describe_distribution(dist) for name, dist in search_space.items()}


def _round(x: float, digits: int = 6) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    magnitude = int(math.floor(math.log10(abs(x))))
    return round(x, max(digits - 1 - magnitude, 0))


def _compact_intermediate(values: dict[int, float]) -> list[list[float]]:
    if not values:
        return []
    steps = sorted(values)
    if len(steps) > MAX_INTERMEDIATE_POINTS:
        idx = [
            round(i * (len(steps) - 1) / (MAX_INTERMEDIATE_POINTS - 1))
            for i in range(MAX_INTERMEDIATE_POINTS)
        ]
        steps = [steps[i] for i in idx]
    return [[s, _round(values[s])] for s in steps]


def trial_record(trial: FrozenTrial) -> dict[str, Any]:
    rec: dict[str, Any] = {"n": trial.number, "state": trial.state.name}
    if trial.values is not None:
        rec["values"] = [_round(v) for v in trial.values]
    rec["params"] = {
        k: (_round(v) if isinstance(v, float) else v) for k, v in trial.params.items()
    }
    if trial.intermediate_values:
        rec["intermediate"] = _compact_intermediate(trial.intermediate_values)
    if trial.user_attrs:
        rec["user_attrs"] = trial.user_attrs
    hypothesis = trial.system_attrs.get("llm:hypothesis")
    if hypothesis:
        rec["hypothesis"] = hypothesis
    new_params = trial.system_attrs.get("llm:new_params")
    if new_params:
        rec["new_params"] = new_params
    return rec


def read_context(
    context: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None, max_chars: int
) -> str:
    """Join free text and file contents into one context block, capped at ``max_chars``."""
    if context is None:
        return ""
    items: Sequence[str | os.PathLike[str]]
    if isinstance(context, (str, os.PathLike)):
        items = [context]
    else:
        items = context
    parts: list[str] = []
    for item in items:
        path = Path(item)
        try:
            is_file = path.is_file() and len(str(item)) < 4096
        except (OSError, ValueError):
            is_file = False
        if is_file:
            parts.append(
                f"## file: {path}\n\n{path.read_text(encoding='utf-8', errors='replace')}"
            )
        else:
            parts.append(str(item))
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[context truncated at {max_chars} characters]"
    return text


def _param_schema(dist: BaseDistribution) -> dict[str, Any]:
    if isinstance(dist, FloatDistribution):
        return {"type": "number", "minimum": dist.low, "maximum": dist.high}
    if isinstance(dist, IntDistribution):
        return {"type": "integer", "minimum": dist.low, "maximum": dist.high}
    if isinstance(dist, CategoricalDistribution):
        return {"enum": list(dist.choices)}
    return {}


def proposal_schema(
    search_space: dict[str, BaseDistribution], n: int, allow_new_params: bool
) -> dict[str, Any]:
    proposal: dict[str, Any] = {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "properties": {k: _param_schema(d) for k, d in search_space.items()},
                "required": list(search_space),
                "additionalProperties": False,
            },
            "hypothesis": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["params", "hypothesis", "evidence"],
    }
    if allow_new_params:
        proposal["properties"]["new_params"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["float", "int", "categorical"]},
                    "low": {"type": "number"},
                    "high": {"type": "number"},
                    "log": {"type": "boolean"},
                    "step": {"type": "number"},
                    "choices": {"type": "array"},
                    "value": {},
                    "rationale": {"type": "string"},
                },
                "required": ["name", "type", "value", "rationale"],
            },
        }
    return {
        "type": "object",
        "properties": {
            "proposals": {"type": "array", "items": proposal, "minItems": n, "maxItems": n}
        },
        "required": ["proposals"],
    }


def build_user_prompt(
    study: Study,
    trials: Sequence[FrozenTrial],
    search_space: dict[str, BaseDistribution],
    *,
    n: int,
    context_text: str,
    allow_new_params: bool,
    max_trials: int,
) -> str:
    directions = [d.name.lower() for d in study.directions]
    best: list[dict[str, Any]] = []
    try:
        if len(directions) == 1:
            best = [trial_record(study.best_trial)]
        else:
            best = [trial_record(t) for t in study.best_trials[:5]]
    except ValueError:
        best = []
    shown = list(trials)
    if len(shown) > max_trials:
        # Keep the best, the most recent, and a spread of the rest.
        completed = sorted(
            (t for t in shown if t.state == TrialState.COMPLETE and t.values is not None),
            key=lambda t: t.values[0] if directions[0] == "minimize" else -t.values[0],
        )
        keep = {t.number for t in completed[: max_trials // 3]}
        keep.update(t.number for t in shown[-(max_trials // 3) :])
        rest = [t for t in shown if t.number not in keep]
        stride = max(1, len(rest) // max(1, max_trials - len(keep)))
        keep.update(t.number for t in rest[::stride])
        shown = [t for t in shown if t.number in keep]
    n_complete = sum(t.state == TrialState.COMPLETE for t in trials)
    n_pruned = sum(t.state == TrialState.PRUNED for t in trials)
    n_failed = sum(t.state == TrialState.FAIL for t in trials)
    sections = [
        f"# Study: {study.study_name}",
        f"directions: {json.dumps(directions)}",
        f"trials so far: {len(trials)} "
        f"({n_complete} complete, {n_pruned} pruned, {n_failed} failed)",
        "",
        "# Search space",
        json.dumps(describe_search_space(search_space), indent=1),
        "",
        "# Best so far",
        json.dumps(best),
        "",
        f"# Trial history ({len(shown)} of {len(trials)} shown)",
        "\n".join(json.dumps(trial_record(t)) for t in shown),
    ]
    if context_text:
        sections += ["", "# Context", context_text]
    sections += [
        "",
        "# Request",
        f"Propose {n} trial(s). allow_new_params: {str(allow_new_params).lower()}.",
        'Return JSON: {"proposals": [{"params": {...}, "hypothesis": "...", '
        '"evidence": [trial numbers]'
        + (', "new_params": [...]' if allow_new_params else "")
        + "}]}",
    ]
    return "\n".join(sections)


def validate_params(
    params: Any, search_space: dict[str, BaseDistribution]
) -> tuple[dict[str, Any], list[str]]:
    """Coerce ``params`` into the search space.

    Returns the coerced values and the list of violations that were repaired in place. Raises
    :class:`ProposalError` when a parameter is missing or cannot be coerced.
    """
    if not isinstance(params, dict):
        raise ProposalError(f"params is not an object: {type(params).__name__}")
    out: dict[str, Any] = {}
    violations: list[str] = []
    for name, dist in search_space.items():
        if name not in params:
            raise ProposalError(f"missing parameter {name!r}")
        raw = params[name]
        if isinstance(dist, CategoricalDistribution):
            choices = list(dist.choices)
            if raw in choices and type(raw) in {type(c) for c in choices}:
                out[name] = raw
                continue
            matches = [c for c in choices if str(c) == str(raw)]
            if len(matches) == 1:
                out[name] = matches[0]
                violations.append(f"{name}: {raw!r} matched choice {matches[0]!r} by string")
                continue
            raise ProposalError(f"{name}: {raw!r} is not one of {choices!r}")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            try:
                raw = float(raw)
            except (TypeError, ValueError):
                raise ProposalError(f"{name}: {raw!r} is not a number") from None
        if not math.isfinite(raw):
            raise ProposalError(f"{name}: {raw!r} is not finite")
        if isinstance(dist, IntDistribution):
            value_i = int(round(raw))
            if value_i != raw:
                violations.append(f"{name}: {raw!r} rounded to {value_i}")
            if value_i < dist.low or value_i > dist.high:
                clipped = min(max(value_i, dist.low), dist.high)
                violations.append(f"{name}: {value_i} clipped to [{dist.low}, {dist.high}]")
                value_i = clipped
            if dist.step != 1:
                snapped = dist.low + round((value_i - dist.low) / dist.step) * dist.step
                snapped = min(snapped, dist.high)
                if snapped != value_i:
                    violations.append(f"{name}: {value_i} snapped to step {dist.step}")
                value_i = snapped
            out[name] = value_i
        elif isinstance(dist, FloatDistribution):
            value_f = float(raw)
            if value_f < dist.low or value_f > dist.high:
                clipped_f = min(max(value_f, dist.low), dist.high)
                violations.append(f"{name}: {value_f} clipped to [{dist.low}, {dist.high}]")
                value_f = clipped_f
            if dist.step is not None:
                snapped_f = dist.low + round((value_f - dist.low) / dist.step) * dist.step
                snapped_f = min(snapped_f, dist.high)
                if abs(snapped_f - value_f) > 1e-12:
                    violations.append(f"{name}: {value_f} snapped to step {dist.step}")
                value_f = snapped_f
            out[name] = value_f
        else:
            raise ProposalError(f"{name}: unsupported distribution {type(dist).__name__}")
    extra = sorted(set(params) - set(search_space))
    if extra:
        violations.append(f"ignored parameters outside the search space: {extra}")
    return out, violations


def materialise_new_param(spec: Any) -> tuple[str, BaseDistribution, Any, str]:
    """Turn a ``new_params`` entry into ``(name, distribution, value, rationale)``."""
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str) or not spec["name"]:
        raise ProposalError(f"new_params entry without a name: {spec!r}")
    name = spec["name"]
    kind = spec.get("type")
    rationale = str(spec.get("rationale", ""))
    if "value" not in spec:
        raise ProposalError(f"new parameter {name!r} has no value")
    dist: BaseDistribution
    if kind == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProposalError(f"new parameter {name!r} needs non-empty choices")
        dist = CategoricalDistribution(choices)
    elif kind in ("float", "int"):
        try:
            low, high = float(spec["low"]), float(spec["high"])
        except (KeyError, TypeError, ValueError):
            raise ProposalError(f"new parameter {name!r} needs numeric low and high") from None
        if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
            raise ProposalError(f"new parameter {name!r} has an invalid range [{low}, {high}]")
        log = bool(spec.get("log", False))
        if log and low <= 0:
            raise ProposalError(f"new parameter {name!r}: log scale needs low > 0")
        if kind == "int":
            dist = IntDistribution(int(low), int(high), log=log, step=int(spec.get("step") or 1))
        else:
            step = spec.get("step")
            dist = FloatDistribution(low, high, log=log, step=float(step) if step else None)
    else:
        raise ProposalError(f"new parameter {name!r} has unknown type {kind!r}")
    value, _ = validate_params({name: spec["value"]}, {name: dist})
    return name, dist, value[name], rationale
