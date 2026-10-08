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
DECISIONS = ("select", "run_more", "insufficient_evidence", "defer")
INVESTIGATION_STEPS = ("measurement", "config", "data", "implementation", "hardware", "recipe")


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
    search_space: dict[str, BaseDistribution],
    n: int,
    allow_new_params: bool,
    *,
    ladder: bool = False,
    expandable: Sequence[str] = (),
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
    if ladder:
        # Post §2.3 / §6.7: every hypothesis states the effect it expects against the frozen
        # threshold, and the proposal carries the prediction it is making.
        proposal["properties"].update(
            {
                "expected_effect": {"type": "number"},
                "predicted_target_value": {"type": "number"},
                "interval": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "equivalent_compute_multiplier": {"type": ["number", "null"]},
                "decision": {"type": "string", "enum": list(DECISIONS)},
                "diagnosis": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string", "enum": list(INVESTIGATION_STEPS)},
                        "note": {"type": "string"},
                    },
                    "required": ["step", "note"],
                },
            }
        )
        proposal["required"] += [
            "expected_effect",
            "predicted_target_value",
            "interval",
            "decision",
        ]
    if expandable:
        proposal["properties"]["expand_bounds"] = {
            "type": "object",
            "properties": {
                name: {
                    "type": "object",
                    "properties": {"low": {"type": "number"}, "high": {"type": "number"}},
                    "required": ["low", "high"],
                }
                for name in expandable
            },
            "additionalProperties": False,
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


def ladder_section(study: Study) -> tuple[str, list[str]]:
    """The ladder block of the prompt and the names a bound expansion may touch.

    Empty when the study registered nothing (no threshold, no ladder, no near-optimal check).
    """
    from optuna.ladder._study import ladder_summary

    summary = ladder_summary(study)
    fit_present = bool((summary.get("fit") or {}).get("n_units"))
    if (
        summary.get("threshold") is None
        and not fit_present
        and summary.get("near_optimal") is None
    ):
        return "", []
    lines = ["# Ladder (pre-registered; every hypothesis is judged against this)"]
    if summary.get("threshold") is not None:
        lines.append(
            f"seed noise floor (SD of the incumbent over seeds): {summary.get('seed_sd')}; "
            f"decision threshold: {summary['threshold']} "
            f"(frozen at {summary.get('threshold_frozen_at')}). A difference smaller than the "
            "threshold is not a win; say in each hypothesis how large an effect you expect "
            "relative to it (expected_effect, in objective units, positive = better)."
        )
    if summary.get("incumbent"):
        lines.append(f"incumbent: {json.dumps(summary['incumbent'])}")
    fit = summary.get("fit")
    if fit:
        lines.append(
            "rung ladder: budgets "
            + json.dumps(fit.get("rungs"))
            + f", fitted form {fit.get('form')}, pooled exponent "
            + f"{_round(float(fit.get('pooled_gamma') or 0.0))}, "
            + f"{fit.get('n_units')} candidates completed all rungs. "
            + ("incumbent fit: " + json.dumps(fit["incumbent"]) if fit.get("incumbent") else "")
        )
        lines.append(
            "Each trial's user_attrs carry ladder:rungs (budget, value), ladder:predicted "
            "(predicted target value, interval, decision per rung) and ladder:status_reason. "
            "A candidate is killed only when the better end of its interval does not reach the "
            "incumbent; early rank alone never kills."
        )
    expandable: list[str] = []
    check = summary.get("near_optimal")
    if check:
        lines.append(f"near-optimal check of the selected point: {json.dumps(check)}")
        expandable = list(check.get("on_boundary") or [])
        if expandable:
            lines.append(
                f"These parameters sit on a search boundary: {expandable}. You may propose "
                "expand_bounds for them only (new low/high); any other expansion is refused."
            )
        else:
            lines.append("No parameter is on a boundary; do not propose expand_bounds.")
    if summary.get("bounds"):
        lines.append(f"bounds already expanded: {json.dumps(summary['bounds'])}")
    if summary.get("holdout"):
        lines.append(
            f"holdout: {summary['holdout'].get('name')}, consumed: "
            f"{bool(summary.get('holdout_consumed'))}. You never see holdout values."
        )
    lines.append(
        "For every proposal return predicted_target_value (your prediction of the objective at "
        "the target budget), interval [low, high] (your 90% interval for a single run), "
        "equivalent_compute_multiplier (null unless the ladder gives you a fit to convert "
        "with) and decision (select | run_more | insufficient_evidence | defer: what you expect "
        "the comparison against the incumbent to conclude once the trial finishes)."
    )
    return "\n".join(lines), expandable


def deviation_section(trials: Sequence[FrozenTrial]) -> str:
    """Post §12.2: when the last trial deviated, the proposal must say what it is checking."""
    for t in reversed(list(trials)):
        attrs = t.user_attrs
        reason = attrs.get("ladder:status_reason")
        outside = attrs.get("ladder:inside_interval") is False
        if (
            reason in ("algorithmic_divergence", "infrastructure_failure", "implementation_error")
            or outside
        ):
            desc = {
                "trial": t.number,
                "status_reason": reason,
                "cause_unknown": attrs.get("ladder:cause_unknown"),
                "error": attrs.get("ladder:error"),
                "outside_frozen_interval": outside,
                "prediction_error": attrs.get("ladder:prediction_error"),
            }
            return (
                "# Deviation to diagnose\n"
                + json.dumps(desc)
                + "\nFollow the investigation order in the system prompt: include a "
                "`diagnosis` with the step you are testing (measurement, config, data, "
                "implementation, hardware, recipe) and a note. Do not change the recipe "
                "(the hyperparameters that the deviating trial used) before the earlier steps "
                "are addressed, unless the status is algorithmic_divergence with a known cause. "
                "An infrastructure failure is not evidence against the configuration; propose "
                "to re-run it."
            )
        if t.state == TrialState.COMPLETE:
            break
    return ""


def build_user_prompt(
    study: Study,
    trials: Sequence[FrozenTrial],
    search_space: dict[str, BaseDistribution],
    *,
    n: int,
    context_text: str,
    allow_new_params: bool,
    max_trials: int,
    ladder_text: str = "",
    expandable: Sequence[str] = (),
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
    if ladder_text:
        sections += ["", ladder_text]
    deviation = deviation_section(trials)
    if deviation:
        sections += ["", deviation]
    ladder_fields = (
        ', "expected_effect": <number>, "predicted_target_value": <number>, '
        '"interval": [low, high], "equivalent_compute_multiplier": <number or null>, '
        '"decision": "select|run_more|insufficient_evidence|defer"'
        if ladder_text
        else ""
    )
    sections += [
        "",
        "# Request",
        f"Propose {n} trial(s). allow_new_params: {str(allow_new_params).lower()}.",
        'Return JSON: {"proposals": [{"params": {...}, "hypothesis": "...", '
        '"evidence": [trial numbers]'
        + (', "new_params": [...]' if allow_new_params else "")
        + ladder_fields
        + (', "expand_bounds": {name: {"low": .., "high": ..}}' if expandable else "")
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
            # Bounds left out: take a range around the proposed value and say so.
            try:
                v = float(spec["value"])
            except (TypeError, ValueError):
                raise ProposalError(f"new parameter {name!r} needs numeric low and high") from None
            low, high = min(0.0, v), max(1.0, 2.0 * v)
            rationale += f" [bounds inferred: {low}, {high}]"
        if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
            raise ProposalError(f"new parameter {name!r} has an invalid range [{low}, {high}]")
        log = bool(spec.get("log", False))
        if log and low <= 0:
            raise ProposalError(f"new parameter {name!r}: log scale needs low > 0")
        step = spec.get("step")
        try:
            step_f = float(step) if step is not None else None
        except (TypeError, ValueError):
            step_f = None
        if step_f is not None and step_f <= 0:
            step_f = None  # a zero or negative step means "no step"
        if kind == "int":
            dist = IntDistribution(
                int(low), int(high), log=log, step=max(1, int(step_f)) if step_f else 1
            )
        else:
            dist = FloatDistribution(low, high, log=log, step=step_f)
    else:
        raise ProposalError(f"new parameter {name!r} has unknown type {kind!r}")
    value, _ = validate_params({name: spec["value"]}, {name: dist})
    return name, dist, value[name], rationale


def parse_ladder_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Tolerant read of the ladder fields of one proposal; missing ones become ``None``."""

    def num(x: Any) -> float | None:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            try:
                x = float(x)
            except (TypeError, ValueError):
                return None
        return float(x) if math.isfinite(float(x)) else None

    interval_raw = raw.get("interval")
    interval: list[float] | None = None
    if isinstance(interval_raw, (list, tuple)) and len(interval_raw) == 2:
        lo, hi = num(interval_raw[0]), num(interval_raw[1])
        if lo is not None and hi is not None:
            interval = [min(lo, hi), max(lo, hi)]
    decision = raw.get("decision")
    diagnosis = raw.get("diagnosis")
    if not (isinstance(diagnosis, dict) and diagnosis.get("step") in INVESTIGATION_STEPS):
        diagnosis = None
    else:
        diagnosis = {"step": diagnosis["step"], "note": str(diagnosis.get("note", ""))[:500]}
    expand: dict[str, dict[str, float]] = {}
    for name, spec in (
        (raw.get("expand_bounds") or {}).items()
        if isinstance(raw.get("expand_bounds"), dict)
        else []
    ):
        if isinstance(spec, dict):
            lo, hi = num(spec.get("low")), num(spec.get("high"))
            if lo is not None and hi is not None and lo < hi:
                expand[str(name)] = {"low": lo, "high": hi}
    return {
        "expected_effect": num(raw.get("expected_effect")),
        "predicted_target_value": num(raw.get("predicted_target_value")),
        "interval": interval,
        "equivalent_compute_multiplier": num(raw.get("equivalent_compute_multiplier")),
        "decision": decision if decision in DECISIONS else None,
        "diagnosis": diagnosis,
        "expand_bounds": expand,
    }
