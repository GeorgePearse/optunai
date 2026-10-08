"""Study-level ladder discipline: noise floor, frozen threshold, holdout acceptance, the
near-optimal test. Attached to :class:`~optuna.study.Study` as thin delegations.

Evaluations made here (noise seeds, fresh-seed re-evaluations, holdout runs, perturbations) run
through :class:`~optuna.trial.FixedTrial` and never enter the study's trial table, so
``best_trial`` keeps its meaning as the development best-of-N. The objective receives its seed
in ``trial.user_attrs["ladder:seed"]`` and its role in ``trial.user_attrs["ladder:role"]``.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import math
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.ladder._decision import decide
from optuna.ladder._decision import Decision
from optuna.ladder._decision import DecisionRecord
from optuna.ladder._decision import now_iso
from optuna.samplers._llm._ledger import Ledger
from optuna.study._study_direction import StudyDirection
from optuna.trial import BaseTrial
from optuna.trial import FixedTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


def _sys(study: "Study") -> dict[str, Any]:
    return study._storage.get_study_system_attrs(study._study_id)


Objective = Callable[[BaseTrial], float]

_NOISE_SEED_BASE = 100_003
_ACCEPT_SEED_BASE = 200_003


def _state(study: Study) -> dict[str, Any]:
    state = getattr(study, "_ladder_state", None)
    if state is None:
        state = {"holdout": None, "holdout_name": None, "objective": None, "ledger": None}
        study._ladder_state = state  # type: ignore[attr-defined]
    return state


def _ledger(study: Study) -> Ledger:
    state = _state(study)
    if state["ledger"] is None:
        pruner_ledger = getattr(study.pruner, "_ledger", None)
        sampler_ledger = getattr(study.sampler, "ledger", None)
        ledger = (
            pruner_ledger
            if isinstance(pruner_ledger, Ledger)
            else sampler_ledger
            if isinstance(sampler_ledger, Ledger)
            else Ledger()
        )
        ledger.bind(study)
        state["ledger"] = ledger
    return state["ledger"]  # type: ignore[no-any-return]


def _set(study: Study, key: str, value: Any) -> None:
    study._storage.set_study_system_attr(study._study_id, key, value)


def _sign(study: Study) -> float:
    if len(study.directions) != 1:
        raise ValueError("ladder methods support single-objective studies only")
    return 1.0 if study.direction == StudyDirection.MINIMIZE else -1.0


def _fixed_trial(params: dict[str, Any], seed: int, role: str, number: int = 0) -> FixedTrial:
    trial = FixedTrial(dict(params), number=number)
    trial.set_user_attr("ladder:seed", int(seed))
    trial.set_user_attr("ladder:role", role)
    return trial


def _evaluate(objective: Objective, params: dict[str, Any], seed: int, role: str) -> float:
    value = float(objective(_fixed_trial(params, seed, role)))
    if not math.isfinite(value):
        raise ValueError(f"{role} evaluation returned a non-finite value {value!r}")
    return value


def _name(fn: Any) -> str:
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
    return str(name)


def _fingerprint(fn: Any) -> str:
    code = getattr(fn, "__code__", None)
    payload = (
        f"{_name(fn)}:{getattr(code, 'co_filename', '')}:{getattr(code, 'co_firstlineno', '')}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def decision_threshold(study: Study) -> float | None:
    """The frozen threshold, or ``None`` before :func:`register_noise`."""
    t = _sys(study).get("ladder:threshold")
    return float(t) if isinstance(t, (int, float)) else None


def register_noise(
    study: Study,
    objective: Objective,
    *,
    n_seeds: int = 3,
    params: dict[str, Any] | None = None,
    min_effect: float = 0.0,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Measure seed noise on the incumbent and freeze the decision threshold (post §2.3, §5.5).

    Runs ``objective`` at ``params`` (default: the study's current best parameters) with
    ``n_seeds`` fresh seeds, stores the mean and SD, and freezes
    ``decision_threshold = max(min_effect, 2 * seed_SD)`` with a timestamp. The threshold cannot
    be re-registered once a holdout has been consumed by :func:`accept`.

    Returns the record that was stored in ``_sys(study)["ladder:noise"]``.
    """
    if n_seeds < 2 and seeds is None:
        raise ValueError("n_seeds must be at least 2 to estimate a SD")
    if _sys(study).get("ladder:holdout_consumed"):
        raise ValueError("a holdout has already been consumed; the threshold is frozen")
    if params is None:
        completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if not completed:
            raise ValueError("no completed trial to use as the incumbent; pass params=")
        params = dict(study.best_trial.params)
    seed_list = (
        list(seeds) if seeds is not None else [_NOISE_SEED_BASE + i for i in range(n_seeds)]
    )
    values = [_evaluate(objective, params, s, "noise") for s in seed_list]
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    threshold = max(float(min_effect), 2.0 * sd)
    frozen_at = now_iso()
    record = {
        "params": params,
        "seeds": seed_list,
        "values": values,
        "mean": float(arr.mean()),
        "seed_sd": sd,
        "min_effect": float(min_effect),
        "threshold": threshold,
        "frozen_at": frozen_at,
        "objective": _name(objective),
    }
    _set(study, "ladder:noise", record)
    _set(study, "ladder:seed_sd", sd)
    _set(study, "ladder:threshold", threshold)
    _set(study, "ladder:threshold_frozen_at", frozen_at)
    _set(study, "ladder:incumbent", {"params": params, "mean": float(arr.mean()), "seed_sd": sd})
    _state(study)["objective"] = objective
    _ledger(study).write("threshold_frozen", **record)
    return record


def holdout(study: Study, objective: Objective, *, name: str | None = None) -> None:
    """Register the acceptance objective; only :func:`accept` evaluates it (post §10.1)."""
    state = _state(study)
    state["holdout"] = objective
    state["holdout_name"] = name or _name(objective)
    fp = _fingerprint(objective)
    _set(study, "ladder:holdout", {"name": state["holdout_name"], "fingerprint": fp})
    _set(study, "ladder:holdout_consumed", False)
    _ledger(study).write("holdout_registered", name=state["holdout_name"], fingerprint=fp)


def accept(
    study: Study,
    objective: Objective | None = None,
    *,
    n_seeds: int = 1,
    seed: int | None = None,
    max_rounds: int = 1,
    min_effect: float | None = None,
    trial_number: int | None = None,
    incumbent: dict[str, Any] | None = None,
) -> DecisionRecord:
    """Acceptance with fresh seeds on the holdout (post §6.7, §10.1).

    Re-evaluates the selected candidate (``trial_number`` or the study's best trial) on the
    development ``objective`` with a new seed (the fresh-seed value that is reported instead of
    the best-of-N), then evaluates candidate and incumbent on the holdout with the same seeds
    (paired). The difference, its interval, the frozen threshold and the equivalent compute
    multiplier (when the study's pruner is a fitted :class:`Ladder`) go into the returned
    :class:`DecisionRecord`, the ledger (``acceptance``) and
    ``_sys(study)["ladder:acceptance"]``. Returning the record shows the holdout result
    to the caller, so the holdout is marked consumed: a second call returns
    ``insufficient_evidence`` until :func:`holdout` registers a fresh one.

    With one seed the interval of the paired difference uses ``√2 · seed_SD``; with more it
    uses the paired differences' standard error. ``max_rounds`` is the number of acceptance
    rounds the pre-registered plan allows; ``run_more`` is only returned while a round is left.
    ``incumbent`` names the parameters to compare against; it defaults to the configuration
    :func:`register_noise` measured, and to the candidate itself (difference zero, recorded as
    such) when nothing was registered.
    """
    state = _state(study)
    holdout_fn = state["holdout"]
    if holdout_fn is None:
        raise ValueError("register a holdout objective with study.holdout(fn) first")
    objective = objective or state["objective"]
    if objective is None:
        raise ValueError("pass the development objective, or call register_noise first")
    threshold = decision_threshold(study)
    if threshold is None:
        if min_effect is None:
            raise ValueError("call register_noise first, or pass min_effect= explicitly")
        threshold = float(min_effect)
    sign = _sign(study)
    rounds = list(_sys(study).get("ladder:acceptance") or [])
    round_index = len(rounds)
    if _sys(study).get("ladder:holdout_consumed"):
        record = DecisionRecord(
            Decision.INSUFFICIENT_EVIDENCE,
            "acceptance",
            math.nan,
            (math.nan, math.nan),
            threshold,
            reason="holdout already shown; it is development data now, register a fresh one",
        )
        refused = record.as_dict()
        refused["record_kind"] = refused.pop("kind")
        _ledger(study).write("acceptance", **refused)
        return record
    selected = study.best_trial if trial_number is None else study.trials[trial_number]
    params = dict(selected.params)
    noise = _sys(study).get("ladder:noise") or {}
    incumbent_params = (
        incumbent
        or (noise.get("params") if isinstance(noise, dict) else None)
        or dict(_sys(study).get("ladder:incumbent", {}).get("params") or {})
        or params
    )
    if incumbent_params is params:
        _ledger(study).write(
            "acceptance_note",
            note="no incumbent registered; the candidate is compared against itself",
        )
    seed_sd = _sys(study).get("ladder:seed_sd")
    base = seed if seed is not None else _ACCEPT_SEED_BASE + 1000 * round_index
    seeds = [base + i for i in range(max(1, n_seeds))]
    reeval = [_evaluate(objective, params, s, "accept_dev") for s in seeds]
    cand = [_evaluate(holdout_fn, params, s, "accept_holdout") for s in seeds]
    inc = [_evaluate(holdout_fn, incumbent_params, s, "accept_holdout_incumbent") for s in seeds]
    diffs = np.asarray([sign * (i - c) for c, i in zip(cand, inc)], dtype=float)
    diff = float(diffs.mean())
    if len(diffs) > 1:
        se = float(diffs.std(ddof=1) / math.sqrt(len(diffs)))
        method = "paired_seeds"
    elif isinstance(seed_sd, (int, float)) and seed_sd > 0:
        se = math.sqrt(2.0) * float(seed_sd)
        method = "sqrt2_seed_sd"
    else:
        se = math.inf
        method = "none"
    lo, hi = diff - 2.0 * se, diff + 2.0 * se
    decision = decide(diff, lo, hi, threshold, can_run_more=round_index + 1 < max_rounds)
    multiplier = None
    mult_fn = getattr(study.pruner, "multiplier", None)
    if callable(mult_fn):
        try:
            multiplier = mult_fn(study, diff)
        except Exception:  # a pruner that is not a Ladder, or no fit yet
            multiplier = None
    best_of_n = float(selected.value) if selected.value is not None else math.nan
    reevaluated = float(np.mean(reeval))
    record = DecisionRecord(
        decision,
        "acceptance",
        diff,
        (lo, hi),
        threshold,
        equivalent_compute_multiplier=multiplier,
        trial_number=selected.number,
        best_of_n_value=best_of_n,
        reevaluated_value=reevaluated,
        holdout_value=float(np.mean(cand)),
        incumbent_holdout_value=float(np.mean(inc)),
        selection_bias_gap=sign * (reevaluated - best_of_n),
        seeds=seeds,
        interval_method=method,
        reason=f"round {round_index + 1} of {max_rounds}, holdout {state['holdout_name']}",
    )
    entry = record.as_dict()
    entry["record_kind"] = entry.pop("kind")
    entry.update(
        {
            "reevaluated_values": reeval,
            "holdout_values": cand,
            "incumbent_holdout_values": inc,
            "incumbent_params": incumbent_params,
            "params": params,
        }
    )
    rounds.append(entry)
    _set(study, "ladder:acceptance", rounds)
    _set(study, "ladder:holdout_consumed", True)
    _set(
        study,
        "ladder:reported_best",
        {"trial": selected.number, "value": reevaluated, "holdout": float(np.mean(cand))},
    )
    ledger = _ledger(study)
    ledger.write("acceptance", **entry)
    ledger.write(
        "holdout_consumed",
        name=state["holdout_name"],
        shown_to="caller",
        note="holdout result returned; it is development data from here on",
    )
    return record


def _perturb(
    params: dict[str, Any], dists: dict[str, BaseDistribution], signs: dict[str, int]
) -> dict[str, Any]:
    out = dict(params)
    for name, dist in dists.items():
        s = signs.get(name, 0)
        if s == 0 or name not in params:
            continue
        v = params[name]
        if isinstance(dist, FloatDistribution):
            if dist.log:
                nv = v * (math.sqrt(2.0) ** s)
            else:
                nv = v + s * 0.1 * (dist.high - dist.low)
            if dist.step is not None:
                nv = dist.low + round((nv - dist.low) / dist.step) * dist.step
            out[name] = float(min(max(nv, dist.low), dist.high))
        elif isinstance(dist, IntDistribution):
            delta = max(dist.step, int(round(abs(v) * 0.25)))
            nv_i = v + s * delta
            nv_i = dist.low + round((nv_i - dist.low) / dist.step) * dist.step
            out[name] = int(min(max(nv_i, dist.low), dist.high))
    return out


def _on_boundary(params: dict[str, Any], dists: dict[str, BaseDistribution]) -> list[str]:
    names: list[str] = []
    for name, dist in dists.items():
        if name not in params or isinstance(dist, CategoricalDistribution):
            continue
        v = params[name]
        if isinstance(dist, FloatDistribution):
            if dist.log:
                hit = v <= dist.low * math.sqrt(2.0) or v >= dist.high / math.sqrt(2.0)
            else:
                margin = 0.1 * (dist.high - dist.low)
                hit = v <= dist.low + margin or v >= dist.high - margin
        elif isinstance(dist, IntDistribution):
            margin = max(dist.step, int(round(abs(v) * 0.25)))
            hit = v <= dist.low + margin or v >= dist.high - margin
        else:
            hit = False
        if hit:
            names.append(name)
    return names


def near_optimal_check(
    study: Study,
    objective: Objective | None = None,
    *,
    n_perturbations: int = 2,
    seed: int | None = None,
    tolerance: float | None = None,
    trial_number: int | None = None,
) -> dict[str, Any]:
    """The post's operational "fully tuned" test (§6.1) on the selected point.

    Perturbs every numeric parameter jointly (log floats ×/÷√2, linear floats ±10 % of the
    range, ints ±25 %, at least one step; categoricals untouched), the first two perturbations
    all-up and all-down, the rest with random signs, evaluates the development objective at the
    same fresh seed as the base point, and reports whether the largest move is below the
    tolerance (the frozen threshold by default) and which parameters sit on a search boundary.
    Stored in ``_sys(study)["ladder:near_optimal"]``; the LLM sampler may propose a
    bound expansion only for the names in ``on_boundary``.
    """
    state = _state(study)
    objective = objective or state["objective"]
    if objective is None:
        raise ValueError("pass the development objective, or call register_noise first")
    threshold = tolerance if tolerance is not None else decision_threshold(study)
    if threshold is None:
        raise ValueError("call register_noise first, or pass tolerance=")
    sign = _sign(study)
    selected = study.best_trial if trial_number is None else study.trials[trial_number]
    params = dict(selected.params)
    dists = dict(selected.distributions)
    numeric = [n for n, d in dists.items() if not isinstance(d, CategoricalDistribution)]
    rng = np.random.default_rng(seed if seed is not None else 7)
    base_seed = seed if seed is not None else _ACCEPT_SEED_BASE + 500
    base = _evaluate(objective, params, base_seed, "perturb_base")
    sign_sets: list[dict[str, int]] = [{n: 1 for n in numeric}, {n: -1 for n in numeric}]
    while len(sign_sets) < n_perturbations:
        sign_sets.append({n: int(rng.choice([-1, 1])) for n in numeric})
    results: list[dict[str, Any]] = []
    deltas: list[float] = []
    for signs in sign_sets[: max(2, n_perturbations)]:
        p = _perturb(params, dists, signs)
        if p == params:
            continue
        v = _evaluate(objective, p, base_seed, "perturb")
        deltas.append(sign * (v - base))
        results.append({"params": p, "value": v, "delta": deltas[-1]})
    max_abs = max((abs(d) for d in deltas), default=0.0)
    best_delta = min(deltas, default=0.0)
    record = {
        "trial": selected.number,
        "seed": base_seed,
        "base_value": base,
        "perturbations": results,
        "max_abs_delta": max_abs,
        "best_perturbation_improvement": -best_delta if best_delta < 0 else 0.0,
        "tolerance": float(threshold),
        "near_optimal": bool(max_abs < threshold),
        "on_boundary": _on_boundary(params, dists),
        "at": now_iso(),
    }
    _set(study, "ladder:near_optimal", record)
    _ledger(study).write("near_optimal_check", **record)
    return record


def expand_bounds(study: Study, name: str, low: float, high: float, *, by: str) -> bool:
    """Record a bound expansion for ``name`` if the near-optimal check put it on a boundary.

    Returns ``False`` (and records a refusal) otherwise. Objectives that opt in read the
    expansion with ``trial._sys(study)["ladder:bounds"][name]``.
    """
    check = _sys(study).get("ladder:near_optimal") or {}
    allowed = name in (check.get("on_boundary") or [])
    ledger = _ledger(study)
    if not allowed:
        ledger.write(
            "bounds_refused", name=name, low=low, high=high, by=by, reason="not on a boundary"
        )
        return False
    bounds = dict(_sys(study).get("ladder:bounds") or {})
    bounds[name] = {"low": low, "high": high, "by": by, "at": now_iso()}
    _set(study, "ladder:bounds", bounds)
    ledger.write("bounds_expanded", name=name, low=low, high=high, by=by)
    return True


def ladder_summary(study: Study) -> dict[str, Any]:
    """Everything the sampler prompt needs, read from the study's system attrs."""
    attrs = _sys(study)
    out: dict[str, Any] = {
        "seed_sd": attrs.get("ladder:seed_sd"),
        "threshold": attrs.get("ladder:threshold"),
        "threshold_frozen_at": attrs.get("ladder:threshold_frozen_at"),
        "incumbent": attrs.get("ladder:incumbent"),
        "holdout": attrs.get("ladder:holdout"),
        "holdout_consumed": attrs.get("ladder:holdout_consumed"),
        "near_optimal": None,
        "bounds": attrs.get("ladder:bounds"),
        "fit": None,
    }
    check = attrs.get("ladder:near_optimal")
    if isinstance(check, dict):
        out["near_optimal"] = {
            k: check.get(k) for k in ("near_optimal", "max_abs_delta", "tolerance", "on_boundary")
        }
    report = getattr(study.pruner, "fit_report", None)
    if callable(report):
        try:
            out["fit"] = report(study)
        except Exception:
            out["fit"] = None
    return out
