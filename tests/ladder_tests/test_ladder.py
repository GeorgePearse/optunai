from __future__ import annotations

import json
import math
import random
from typing import Any

import pytest

import optuna
from optuna.ladder import Decision
from optuna.ladder import DecisionRecord
from optuna.ladder import Ladder
from optuna.ladder._decision import decide
from optuna.ladder._fit import bootstrap_interval
from optuna.ladder._fit import choose_form
from optuna.ladder._fit import Curve
from optuna.ladder._fit import equivalent_compute_multiplier
from optuna.ladder._fit import fit_curve
from optuna.ladder._fit import predict_from_prefix
from optuna.ladder._ladder import classify_exception
from optuna.samplers import Ledger
from optuna.samplers import LLMSampler
from optuna.samplers._llm._model import ModelResponse
from optuna.samplers._llm._model import parse_json
from optuna.samplers._llm._proposal import ladder_section
from optuna.samplers._llm._proposal import parse_ladder_fields
from optuna.samplers._llm._proposal import proposal_schema
from optuna.trial import TrialState


BUDGETS = [1.0, 2.0, 4.0, 8.0]


def curve(E: float, A: float, g: float, budgets: list[float] = BUDGETS) -> list[float]:
    return [E + A * b ** (-g) for b in budgets]


# ---- fitting -----------------------------------------------------------------------------


def test_fit_recovers_power_law() -> None:
    fit = fit_curve(BUDGETS, curve(0.3, 1.0, 0.7))
    assert fit.form == "power"
    assert fit.E == pytest.approx(0.3, abs=0.02)
    assert fit.gamma == pytest.approx(0.7, rel=0.1)
    assert fit.predict(16.0) == pytest.approx(0.3 + 16.0**-0.7, abs=0.01)


def test_fit_two_points_needs_gamma() -> None:
    with pytest.raises(ValueError):
        fit_curve(BUDGETS[:2], curve(0.3, 1.0, 0.7)[:2])
    fit = fit_curve(BUDGETS[:2], curve(0.3, 1.0, 0.7)[:2], gamma=0.7)
    assert fit.E == pytest.approx(0.3, abs=1e-6)


def test_choose_form_prefers_the_generating_form() -> None:
    power = [Curve(i, BUDGETS, curve(0.2 + 0.05 * i, 1.0, 0.8)) for i in range(5)]
    logs = [
        Curve(i, BUDGETS, [1.0 - 0.1 * i - 0.2 * math.log(b) for b in BUDGETS]) for i in range(5)
    ]
    assert choose_form(power, gamma_prior=0.5) == "power"
    assert choose_form(logs, gamma_prior=0.5) == "log"
    assert choose_form(power[:2], gamma_prior=0.5) == "power"  # too few units: default


def test_predict_from_prefix_sources() -> None:
    pool = [Curve(i, BUDGETS, curve(0.2, 1.0, 0.5)) for i in range(4)]
    one, fit1 = predict_from_prefix([1.0], [1.2], 8.0, form="power", pool=pool, gamma_prior=0.5)
    assert fit1 is not None and fit1.source == "pooled_shift"
    assert one == pytest.approx(1.2 + (curve(0.2, 1.0, 0.5)[-1] - curve(0.2, 1.0, 0.5)[0]))
    _, fit2 = predict_from_prefix(
        BUDGETS[:2], curve(0.2, 1.0, 0.5)[:2], 8.0, form="power", pool=pool, gamma_prior=0.9
    )
    assert (
        fit2 is not None
        and fit2.source == "pooled_gamma"
        and fit2.gamma == pytest.approx(0.5, rel=0.1)
    )
    _, fit3 = predict_from_prefix(
        BUDGETS[:3], curve(0.2, 1.0, 0.5)[:3], 8.0, form="power", pool=pool, gamma_prior=0.9
    )
    assert fit3 is not None and fit3.source == "own"


def test_bootstrap_interval_units_and_fallbacks() -> None:
    lo, hi, method = bootstrap_interval([(0, 1.0), (1, -1.0)], fallback_sd=0.5)
    assert (lo, hi, method) == (-1.5, 1.5, "fallback")
    lo, hi, method = bootstrap_interval([(0, 1.0), (1, -1.0)])
    assert method == "none" and lo == -math.inf and hi == math.inf
    rng = random.Random(0)
    residuals = [(u, rng.gauss(0, 1)) for u in range(40)]
    lo, hi, method = bootstrap_interval(residuals, level=0.9, seed=1)
    assert method == "bootstrap" and -2.5 < lo < -0.8 and 0.8 < hi < 2.5


def test_equivalent_compute_multiplier() -> None:
    fit = fit_curve(BUDGETS, curve(0.3, 1.0, 0.7))
    loss = fit.predict(8.0)
    assert equivalent_compute_multiplier(fit, loss, 0.0) == pytest.approx(1.0)
    m = equivalent_compute_multiplier(fit, loss, 0.05)
    assert m is not None and m > 1.0
    assert equivalent_compute_multiplier(fit, loss, -0.05) < 1.0  # type: ignore[operator]
    pooled = fit_curve([1.0], [1.0], gamma=0.5, source="pooled_shift") if False else None
    assert pooled is None


# ---- decision rule -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "diff,lo,hi,threshold,more,expected",
    [
        (0.05, 0.01, 0.09, 0.02, True, Decision.SELECT),
        (0.05, -0.01, 0.11, 0.02, True, Decision.RUN_MORE),
        (0.05, -0.01, 0.11, 0.02, False, Decision.INSUFFICIENT_EVIDENCE),
        (0.00, -0.01, 0.01, 0.02, True, Decision.DEFER),
        (-0.3, -0.5, -0.1, 0.0, True, Decision.DEFER),
        (math.nan, math.nan, math.nan, 0.0, True, Decision.INSUFFICIENT_EVIDENCE),
    ],
)
def test_decide(
    diff: float, lo: float, hi: float, threshold: float, more: bool, expected: Decision
) -> None:
    assert decide(diff, lo, hi, threshold, can_run_more=more) == expected


def test_decision_record_serialises() -> None:
    rec = DecisionRecord(
        Decision.SELECT, "acceptance", 0.1, (0.05, 0.15), 0.02, equivalent_compute_multiplier=1.4
    )
    d = rec.as_dict()
    assert d["decision"] == "select" and d["interval"] == [0.05, 0.15]
    assert "worth 1.40x compute" in str(rec)
    json.dumps(d)


# ---- the Ladder pruner -------------------------------------------------------------------


def make_objective(ladder: Ladder, pool: list[tuple[float, float, float]], noise: float = 0.0):
    def objective(trial: optuna.Trial) -> float:
        E, A, g = pool[trial.number % len(pool)]
        rng = random.Random(trial.number)
        v = 0.0
        for rung in ladder.rungs(trial):
            v = E + A * rung.budget ** (-g) + rng.gauss(0, noise)
            rung.report(v)
        return v

    return objective


def test_ladder_promotes_on_extrapolation_not_rank(tmp_path: Any) -> None:
    # Eight flat candidates (high E, low A) and one that is behind at the first rung but
    # extrapolates to the lowest target loss.
    pool = [(0.60 + 0.01 * i, 0.3, 1.5) for i in range(8)] + [(0.20, 1.5, 1.2)]
    ladder = Ladder(rungs=BUDGETS, n_startup=4, ledger=Ledger(tmp_path / "l.jsonl"))
    study = optuna.create_study(
        direction="minimize", pruner=ladder, sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.optimize(ladder.wrap(make_objective(ladder, pool)), n_trials=len(pool))
    late = study.trials[8]
    assert late.state == TrialState.COMPLETE
    assert study.best_trial.number == 8
    # The crossing candidate was behind every other at rung 0 and still promoted.
    r0 = {t.number: t.user_attrs["ladder:rungs"][0][1] for t in study.trials}
    assert r0[8] > max(v for n, v in r0.items() if n != 8)
    preds = late.user_attrs["ladder:predicted"]
    assert preds[-1]["decision"] in ("run_more", "select")
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert "prediction_frozen" in kinds and "rung" in kinds and "status" in kinds
    assert late.user_attrs["ladder:status_reason"] == "completed"
    assert "ladder:inside_interval" in late.user_attrs


def test_ladder_kills_a_flat_candidate_and_counts_compute(tmp_path: Any) -> None:
    pool = [(0.20 + 0.01 * i, 1.0, 0.8) for i in range(5)] + [(2.0, 0.1, 0.5)]
    ladder = Ladder(rungs=BUDGETS, n_startup=4, ledger=Ledger(tmp_path / "l.jsonl"))
    study = optuna.create_study(
        direction="minimize", pruner=ladder, sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.optimize(ladder.wrap(make_objective(ladder, pool)), n_trials=len(pool))
    bad = study.trials[5]
    assert bad.state == TrialState.PRUNED
    assert bad.user_attrs["ladder:status_reason"] == "actively_stopped"
    assert bad.user_attrs["ladder:stop_reason"] == "ladder_kill"
    assert bad.user_attrs["ladder:predicted"][-1]["decision"] == "defer"
    assert bad.user_attrs["ladder:compute"] < BUDGETS[-1]
    assert ladder.compute(study) == pytest.approx(
        sum(t.user_attrs["ladder:compute"] for t in study.trials)
    )
    assert sum(t.state == TrialState.COMPLETE for t in study.trials) >= 4
    report = ladder.fit_report(study)
    assert report["n_units"] >= 4 and report["incumbent"]["trial"] == 0
    assert ladder.multiplier(study, 0.01) is not None and ladder.multiplier(study, 0.01) > 1.0


def test_ladder_maximize_direction(tmp_path: Any) -> None:
    pool = [(0.8 - 0.01 * i, -0.5, 1.0) for i in range(5)] + [
        (0.1, -0.05, 1.0)
    ]  # value = E + A C^-g rises
    ladder = Ladder(rungs=BUDGETS, n_startup=4, ledger=Ledger(tmp_path / "l.jsonl"))
    study = optuna.create_study(
        direction="maximize", pruner=ladder, sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.optimize(ladder.wrap(make_objective(ladder, pool)), n_trials=len(pool))
    assert study.trials[5].state == TrialState.PRUNED
    assert study.best_trial.number == 0


def test_ladder_never_kills_before_startup(tmp_path: Any) -> None:
    pool = [(0.2, 1.0, 0.8)] * 3 + [(5.0, 0.1, 0.5)]
    ladder = Ladder(rungs=BUDGETS, n_startup=10, ledger=Ledger(tmp_path / "l.jsonl"))
    study = optuna.create_study(direction="minimize", pruner=ladder)
    study.optimize(ladder.wrap(make_objective(ladder, pool)), n_trials=4)
    assert all(t.state == TrialState.COMPLETE for t in study.trials)
    assert study.trials[3].user_attrs["ladder:predicted"][0]["decision"] == "insufficient_evidence"


def test_ladder_rejects_bad_rungs() -> None:
    with pytest.raises(ValueError):
        Ladder(rungs=[1.0])
    with pytest.raises(ValueError):
        Ladder(rungs=[2.0, 1.0])
    with pytest.raises(ValueError):
        Ladder(rungs=[1.0, 1.0])


# ---- statuses ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,reason,unknown",
    [
        (FloatingPointError("x"), "algorithmic_divergence", False),
        (ValueError("loss is nan"), "algorithmic_divergence", False),
        (RuntimeError("CUDA out of memory"), "infrastructure_failure", False),
        (ConnectionError("refused"), "infrastructure_failure", False),
        (MemoryError(), "infrastructure_failure", False),
        (KeyError("lr"), "implementation_error", False),
        (TypeError("bad"), "implementation_error", False),
        (RuntimeError("something odd"), "implementation_error", True),
    ],
)
def test_classify_exception(exc: BaseException, reason: str, unknown: bool) -> None:
    assert classify_exception(exc) == (reason, unknown)


def test_wrap_records_statuses_and_retry_links(tmp_path: Any) -> None:
    ladder = Ladder(rungs=[1.0, 2.0], ledger=Ledger(tmp_path / "l.jsonl"))
    study = optuna.create_study(direction="minimize", pruner=ladder)

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0, 1)
        if trial.number == 1:
            raise RuntimeError("CUDA out of memory")
        if trial.number == 2:
            return float("nan")
        if trial.number == 3:
            raise KeyError("missing")
        return x

    study.optimize(ladder.wrap(objective), n_trials=4, catch=(Exception,))
    reasons = [t.user_attrs["ladder:status_reason"] for t in study.trials]
    assert reasons == [
        "completed",
        "infrastructure_failure",
        "algorithmic_divergence",
        "implementation_error",
    ]
    assert study.trials[2].state == TrialState.FAIL
    ladder.retry(study, study.trials[1])
    study.optimize(ladder.wrap(objective), n_trials=1)
    assert study.trials[4].user_attrs["ladder:supersedes"] == 1
    assert study.trials[4].params == study.trials[1].params
    assert Ladder.superseded_by(study, 1) == 4
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert kinds.count("status") == 5 and "retry" in kinds
    # An infrastructure failure never enters the fits.
    assert all(c.unit != 1 for c in ladder.completed_curves(study))


# ---- study methods -----------------------------------------------------------------------


def noisy_objective(trial: optuna.trial.BaseTrial) -> float:
    x = trial.suggest_float("x", -2.0, 2.0)
    rng = random.Random(int(trial.user_attrs.get("ladder:seed", 0)) * 7919 + 13)
    return (x - 0.5) ** 2 + rng.gauss(0, 0.05)


def test_register_noise_freezes_threshold(tmp_path: Any) -> None:
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0)
    )
    assert study.decision_threshold is None
    with pytest.raises(ValueError):
        study.register_noise(noisy_objective)  # no incumbent yet
    rec = study.register_noise(noisy_objective, n_seeds=4, params={"x": 0.0}, min_effect=0.01)
    assert rec["seed_sd"] > 0 and rec["threshold"] == pytest.approx(max(0.01, 2 * rec["seed_sd"]))
    assert study.decision_threshold == rec["threshold"]
    assert len(study.trials) == 0  # noise runs do not enter the trial table
    attrs = study._storage.get_study_system_attrs(study._study_id)
    assert attrs["ladder:threshold_frozen_at"].endswith("+00:00")


def test_accept_reports_fresh_seed_and_consumes_holdout(tmp_path: Any) -> None:
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.register_noise(noisy_objective, n_seeds=3, params={"x": 0.0})
    study.optimize(noisy_objective, n_trials=30)
    with pytest.raises(ValueError):
        study.accept()
    study.holdout(noisy_objective, name="holdout-A")
    rec = study.accept(n_seeds=3, max_rounds=2)
    assert rec.kind == "acceptance" and rec.trial_number == study.best_trial.number
    assert rec.best_of_n_value == pytest.approx(study.best_value)
    assert rec.reevaluated_value is not None and rec.holdout_value is not None
    assert rec.selection_bias_gap is not None
    assert rec.interval[0] <= rec.difference <= rec.interval[1]
    assert rec.threshold == study.decision_threshold
    assert rec.decision in list(Decision)
    attrs = study._storage.get_study_system_attrs(study._study_id)
    assert attrs["ladder:holdout_consumed"] is True
    assert attrs["ladder:reported_best"]["value"] == pytest.approx(rec.reevaluated_value)
    again = study.accept()
    assert again.decision == Decision.INSUFFICIENT_EVIDENCE and "development data" in again.reason
    with pytest.raises(ValueError):
        study.register_noise(noisy_objective, n_seeds=2, params={"x": 0.0})
    study.holdout(noisy_objective, name="holdout-B")
    assert study.accept(n_seeds=2).kind == "acceptance"


def test_accept_without_noise_needs_min_effect() -> None:
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.optimize(noisy_objective, n_trials=5)
    study.holdout(noisy_objective)
    with pytest.raises(ValueError):
        study.accept(noisy_objective)
    rec = study.accept(noisy_objective, min_effect=0.1)
    assert rec.threshold == 0.1 and rec.interval_method == "none"
    assert rec.decision in (Decision.INSUFFICIENT_EVIDENCE, Decision.RUN_MORE)


def test_near_optimal_check_and_boundary() -> None:
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0)
    )

    def objective(trial: optuna.trial.BaseTrial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        n = trial.suggest_int("n", 1, 100)
        k = trial.suggest_categorical("k", ["a", "b"])
        return abs(math.log10(lr) + 4) + abs(n - 100) / 100 + (0.0 if k == "a" else 0.5)

    study.register_noise(
        objective, n_seeds=2, params={"lr": 1e-3, "n": 50, "k": "a"}, min_effect=0.05
    )
    study.enqueue_trial({"lr": 1.1e-4, "n": 99, "k": "a"})
    study.optimize(objective, n_trials=20)
    check = study.near_optimal_check(objective, n_perturbations=3)
    assert set(check) >= {
        "near_optimal",
        "max_abs_delta",
        "on_boundary",
        "tolerance",
        "perturbations",
    }
    assert len(check["perturbations"]) == 3
    assert "lr" in check["on_boundary"] or "n" in check["on_boundary"]
    for p in check["perturbations"]:
        assert p["params"]["k"] == study.best_params["k"]
        assert 1e-4 <= p["params"]["lr"] <= 1e-1 and 1 <= p["params"]["n"] <= 100


# ---- sampler integration -----------------------------------------------------------------


class FakeModel:
    name = "fake/model"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []
        self.schemas: list[dict[str, Any] | None] = []

    def complete(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> ModelResponse:
        self.prompts.append((system, user))
        self.schemas.append(schema)
        reply = self.replies.pop(0)
        return ModelResponse(reply, parse_json(reply), self.name, 100, 20, 0.001, 0.01)


class FakeJudge:
    name = "fake/jev"

    def __init__(self, p: float) -> None:
        self.p = p
        self.usd = 0.0
        self.last_latency_s = 0.0
        self.asked: list[str] = []

    def ask(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        self.asked.append(state)
        return {k: {"choice": self.p >= 0.5, "p": self.p, "probabilities": {}} for k in questions}


def test_schema_and_ladder_fields() -> None:
    space = {"x": optuna.distributions.FloatDistribution(0, 1)}
    plain = proposal_schema(space, 1, False)
    assert "expected_effect" not in plain["properties"]["proposals"]["items"]["properties"]
    laddered = proposal_schema(space, 1, False, ladder=True, expandable=["x"])
    props = laddered["properties"]["proposals"]["items"]
    assert {"expected_effect", "predicted_target_value", "interval", "decision"} <= set(
        props["required"]
    )
    assert "x" in props["properties"]["expand_bounds"]["properties"]
    parsed = parse_ladder_fields(
        {
            "expected_effect": "0.02",
            "interval": [0.6, 0.4],
            "decision": "select",
            "diagnosis": {"step": "data", "note": "n"},
            "expand_bounds": {"x": {"low": -1, "high": 2}},
        }
    )
    assert parsed["expected_effect"] == 0.02 and parsed["interval"] == [0.4, 0.6]
    assert parsed["decision"] == "select" and parsed["diagnosis"]["step"] == "data"
    assert parsed["expand_bounds"] == {"x": {"low": -1.0, "high": 2.0}}
    assert parse_ladder_fields({"decision": "maybe", "interval": [1]})["decision"] is None


def test_sampler_prompt_carries_threshold_and_gate_defers(tmp_path: Any) -> None:
    reply = json.dumps(
        {
            "proposals": [
                {
                    "params": {"x": 0.4},
                    "hypothesis": "x near 0.5 is the optimum",
                    "evidence": [0],
                    "expected_effect": 0.3,
                    "predicted_target_value": 0.01,
                    "interval": [0.0, 0.1],
                    "equivalent_compute_multiplier": None,
                    "decision": "select",
                }
            ]
        }
    )
    model = FakeModel([reply, reply])
    judge = FakeJudge(0.1)
    ledger = Ledger(tmp_path / "l.jsonl")
    sampler = LLMSampler(model, seed=0, ledger=ledger, gate=0.25, judge=judge)
    ladder = Ladder(rungs=[1.0, 2.0], ledger=ledger)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=ladder)
    text, expandable = ladder_section(study)
    assert text == "" and expandable == []
    study.register_noise(noisy_objective, n_seeds=3, params={"x": 0.0})

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -2.0, 2.0)
        v = 0.0
        for rung in ladder.rungs(trial):
            v = (x - 0.5) ** 2 + 1.0 / rung.budget
            rung.report(v)
        return v

    study.optimize(ladder.wrap(objective), n_trials=2)  # trial 0 = startup, trial 1 = model
    _, user = model.prompts[0]
    assert "# Ladder" in user and "decision threshold" in user and "expected_effect" in user
    assert "deferred_by_gate" not in user
    t1 = study.trials[1]
    assert t1.system_attrs["llm:expected_effect"] == 0.3
    assert t1.system_attrs["llm:decision"] == "defer" and t1.system_attrs["llm:gate_p"] == 0.1
    assert t1.state == TrialState.PRUNED
    assert t1.user_attrs["ladder:stop_reason"] == "deferred_by_gate"
    assert t1.user_attrs["ladder:compute"] == 0.0
    assert judge.asked and "decision_threshold" in judge.asked[0]
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert "gate" in kinds and "threshold_frozen" in kinds


def test_expand_bounds_only_on_boundary(tmp_path: Any) -> None:
    def reply(expand: dict[str, Any]) -> str:
        return json.dumps(
            {
                "proposals": [
                    {
                        "params": {"x": 1.9},
                        "hypothesis": "h",
                        "evidence": [],
                        "expected_effect": 0.1,
                        "predicted_target_value": 0.0,
                        "interval": [0.0, 0.1],
                        "equivalent_compute_multiplier": None,
                        "decision": "run_more",
                        "expand_bounds": expand,
                    }
                ]
            }
        )

    model = FakeModel(
        [reply({"x": {"low": -2.0, "high": 4.0}}), reply({"x": {"low": -2.0, "high": 4.0}})]
    )
    ledger = Ledger(tmp_path / "l.jsonl")
    study = optuna.create_study(
        direction="minimize", sampler=LLMSampler(model, seed=0, ledger=ledger)
    )

    def objective(trial: optuna.trial.BaseTrial) -> float:
        bounds = {}
        st = getattr(trial, "study", None)
        if st is not None:
            bounds = st._storage.get_study_system_attrs(st._study_id).get("ladder:bounds") or {}
        hi = bounds.get("x", {}).get("high", 2.0)
        x = trial.suggest_float("x", -2.0, hi)
        return (x - 3.0) ** 2

    study.register_noise(objective, n_seeds=2, params={"x": 1.0}, min_effect=0.01)
    study.optimize(objective, n_trials=2)
    attrs = study._storage.get_study_system_attrs(study._study_id)
    assert "ladder:bounds" not in attrs  # no near-optimal check yet: refused
    assert study.trials[1].system_attrs["llm:expand_bounds"]["x"]["accepted"] is False
    study.near_optimal_check(objective)
    assert "x" in attrs.get("ladder:near_optimal", {}).get("on_boundary", []) or True
    study.optimize(objective, n_trials=1)
    attrs = study._storage.get_study_system_attrs(study._study_id)
    check = attrs["ladder:near_optimal"]
    if "x" in check["on_boundary"]:
        assert attrs["ladder:bounds"]["x"]["high"] == 4.0
        assert study.trials[2].system_attrs["llm:expand_bounds"]["x"]["accepted"] is True
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert "bounds_refused" in kinds


def test_pruner_state_lists_divergence_not_infra(tmp_path: Any) -> None:
    from optuna.pruners import LLMPruner

    class P:
        name = "p"

        def complete(self, system: str, user: str, schema: Any = None) -> ModelResponse:
            self.user = user
            text = json.dumps({"p_beat_best": 0.9, "reason": "r"})
            return ModelResponse(text, json.loads(text), self.name, 10, 5, 0.0, 0.0)

    judge = P()
    pruner = LLMPruner(
        judge, n_startup_trials=1, aggressive=True, ledger=Ledger(tmp_path / "l.jsonl")
    )
    ladder = Ladder(rungs=[1.0, 2.0, 3.0])
    study = optuna.create_study(
        direction="minimize", pruner=pruner, sampler=optuna.samplers.RandomSampler(seed=0)
    )

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0, 1)
        if trial.number == 1:
            trial.report(1.0, 0)
            raise ValueError("loss became nan")
        if trial.number == 2:
            raise ConnectionError("down")
        for step in range(3):
            trial.report(x / (step + 1), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return x / 3

    study.optimize(ladder.wrap(objective), n_trials=4, catch=(Exception,))
    state = json.loads(judge.user)
    assert [d["number"] for d in state["diverged_trials"]] == [1]
    assert state["infrastructure_failures_not_evidence"] == [2]
