from __future__ import annotations

import json
from typing import Any

import pytest

import optuna
from optuna.pruners import LLMPruner
from optuna.samplers import Ledger
from optuna.samplers._llm._model import ModelResponse
from optuna.samplers._llm._model import parse_json
from optuna.trial import TrialState


class ConstantJudge:
    name = "fake/judge"

    def __init__(self, p: float) -> None:
        self.p = p
        self.calls = 0

    def complete(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> ModelResponse:
        self.calls += 1
        assert "running_trial" in user and "best_completed_trial" in user
        text = json.dumps({"p_beat_best": self.p, "reason": "constant"})
        return ModelResponse(text, parse_json(text), self.name, 50, 10, 0.0001, 0.0)


def curve_objective(trial: optuna.Trial) -> float:
    slope = trial.suggest_float("slope", 0.0, 1.0)
    value = 0.0
    for step in range(10):
        value = slope * (step + 1)
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return value


def test_llm_pruner_prunes_more_than_median_not_less(tmp_path: Any) -> None:
    judge = ConstantJudge(p=0.0)  # certain the trial will not beat the best
    pruner = LLMPruner(judge, n_startup_trials=2, ledger=Ledger(tmp_path / "l.jsonl"))
    study = optuna.create_study(
        direction="maximize", pruner=pruner, sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.optimize(curve_objective, n_trials=8)
    states = [t.state for t in study.trials]
    assert states[:2] == [TrialState.COMPLETE, TrialState.COMPLETE]
    assert all(s == TrialState.PRUNED for s in states[2:])
    assert judge.calls >= 1
    judged = [t for t in study.trials if "llm:prune" in t.system_attrs]
    assert judged, "the model was never asked"
    assert judged[0].system_attrs["llm:prune"][-1]["prune"] is True
    assert judged[0].system_attrs["llm:prune"][-1]["p_beat_best"] == 0.0


def test_llm_pruner_keeps_median_floor(tmp_path: Any) -> None:
    judge = ConstantJudge(p=1.0)  # the model never wants to prune
    pruner = LLMPruner(judge, n_startup_trials=2, ledger=Ledger(tmp_path / "l.jsonl"))
    study = optuna.create_study(
        direction="maximize", pruner=pruner, sampler=optuna.samplers.RandomSampler(seed=1)
    )
    study.optimize(curve_objective, n_trials=12)
    median_only = optuna.pruners.MedianPruner(n_startup_trials=2)
    reference = optuna.create_study(
        direction="maximize", pruner=median_only, sampler=optuna.samplers.RandomSampler(seed=1)
    )
    reference.optimize(curve_objective, n_trials=12)
    ours = [t.state for t in study.trials]
    theirs = [t.state for t in reference.trials]
    assert ours == theirs  # floor only: identical decisions when the model never prunes


def test_llm_pruner_aggressive_can_override_floor(tmp_path: Any) -> None:
    judge = ConstantJudge(p=1.0)
    pruner = LLMPruner(
        judge, n_startup_trials=2, aggressive=True, ledger=Ledger(tmp_path / "l.jsonl")
    )
    study = optuna.create_study(
        direction="maximize", pruner=pruner, sampler=optuna.samplers.RandomSampler(seed=1)
    )
    study.optimize(curve_objective, n_trials=12)
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


def test_llm_pruner_survives_judge_errors(tmp_path: Any) -> None:
    class Broken:
        name = "broken"

        def complete(
            self, system: str, user: str, schema: dict[str, Any] | None = None
        ) -> ModelResponse:
            return ModelResponse("", None, self.name, 0, 0, None, 0.0, error="boom")

    pruner = LLMPruner(Broken(), n_startup_trials=1, ledger=Ledger(tmp_path / "l.jsonl"))
    study = optuna.create_study(
        direction="maximize", pruner=pruner, sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.optimize(curve_objective, n_trials=4)
    assert all(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in study.trials)


def test_threshold_validation() -> None:
    with pytest.raises(ValueError):
        LLMPruner(ConstantJudge(0.5), threshold=0.0)
