from __future__ import annotations

import json
from typing import Any

import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import Ledger
from optuna.samplers import LLMSampler
from optuna.samplers._llm._model import ModelResponse
from optuna.samplers._llm._model import parse_json
from optuna.samplers._llm._proposal import materialise_new_param
from optuna.samplers._llm._proposal import ProposalError
from optuna.samplers._llm._proposal import validate_params
from optuna.trial import TrialState


class FakeModel:
    """Replies with the queued texts in order; records every prompt it saw."""

    name = "fake/model"

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> ModelResponse:
        self.prompts.append((system, user))
        reply = self.replies.pop(0) if self.replies else '{"proposals": []}'
        if isinstance(reply, Exception):
            return ModelResponse("", None, self.name, 0, 0, None, 0.0, error=str(reply))
        return ModelResponse(reply, parse_json(reply), self.name, 100, 20, 0.001, 0.01)


def proposal(
    params: dict[str, Any], hypothesis: str = "h", evidence: list[int] | None = None, **extra: Any
) -> str:
    return json.dumps(
        {
            "proposals": [
                {"params": params, "hypothesis": hypothesis, "evidence": evidence or [], **extra}
            ]
        }
    )


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    n = trial.suggest_int("n", 1, 10)
    kind = trial.suggest_categorical("kind", ["a", "b"])
    return x**2 + n + (0 if kind == "a" else 1)


def test_parse_json_handles_fences_and_prose() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    assert parse_json("no json here") is None


def test_validate_params_coerces_and_reports() -> None:
    space = {
        "x": FloatDistribution(0.0, 1.0),
        "n": IntDistribution(1, 10, step=2),
        "k": CategoricalDistribution(["a", "b"]),
    }
    params, violations = validate_params({"x": 1.5, "n": 4.2, "k": "b", "zzz": 1}, space)
    assert params == {"x": 1.0, "n": 5, "k": "b"}
    assert any("clipped" in v for v in violations)
    assert any("snapped" in v or "rounded" in v for v in violations)
    assert any("outside the search space" in v for v in violations)
    with pytest.raises(ProposalError):
        validate_params({"x": 0.5, "n": 3}, space)
    with pytest.raises(ProposalError):
        validate_params({"x": "wide", "n": 3, "k": "a"}, space)
    with pytest.raises(ProposalError):
        validate_params({"x": 0.5, "n": 3, "k": "c"}, space)


def test_materialise_new_param() -> None:
    name, dist, value, rationale = materialise_new_param(
        {
            "name": "lr",
            "type": "float",
            "low": 1e-5,
            "high": 1e-1,
            "log": True,
            "value": 1e-3,
            "rationale": "r",
        }
    )
    assert name == "lr" and isinstance(dist, FloatDistribution) and dist.log and value == 1e-3
    with pytest.raises(ProposalError):
        materialise_new_param({"name": "bad", "type": "float", "low": 1, "high": 0, "value": 0.5})


def test_sampler_uses_model_then_records_attrs(tmp_path: Any) -> None:
    model = FakeModel([proposal({"x": 0.5, "n": 3, "kind": "a"}, "small x", [0])])
    ledger = Ledger(tmp_path / "ledger.jsonl")
    sampler = LLMSampler(model, ledger=ledger, seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=2)
    first, second = study.trials
    assert first.system_attrs["llm:source"] == "startup"
    assert second.system_attrs["llm:source"] == "llm"
    assert second.params == {"x": 0.5, "n": 3, "kind": "a"}
    assert second.system_attrs["llm:hypothesis"] == "small x"
    assert second.system_attrs["llm:evidence"] == [0]
    assert second.system_attrs["llm:model"] == "fake/model"
    assert second.system_attrs["llm:usd"] == 0.001
    kinds = [line["kind"] for line in ledger.lines()]
    assert (
        kinds.count("call") == 1 and kinds.count("proposal") == 1 and kinds.count("trial_end") == 2
    )
    system, user = model.prompts[0]
    assert "Search space" in user and '"x"' in user and "Trial history" in user


def test_sampler_repairs_once_then_falls_back(tmp_path: Any) -> None:
    model = FakeModel(
        [
            proposal({"x": 0.5, "n": 3}),  # missing kind -> repair
            proposal({"x": 0.5, "n": 3, "kind": "b"}),  # repaired
            "not json at all",  # next trial: no JSON -> fallback
            RuntimeError("provider down"),  # next trial: error -> fallback
        ]
    )
    ledger = Ledger(tmp_path / "ledger.jsonl")
    sampler = LLMSampler(model, ledger=ledger, seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=4)
    sources = [t.system_attrs["llm:source"] for t in study.trials]
    assert sources == ["startup", "repair", "fallback", "fallback"]
    assert study.trials[1].params["kind"] == "b"
    assert "Previous reply" in model.prompts[1][1]
    assert ledger.fallbacks == 2
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


def test_sampler_coerces_out_of_range_values(tmp_path: Any) -> None:
    model = FakeModel([proposal({"x": 99.0, "n": 2.7, "kind": "a"})])
    sampler = LLMSampler(model, ledger=Ledger(tmp_path / "l.jsonl"), seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=2)
    t = study.trials[1]
    assert t.params["x"] == 5.0 and t.params["n"] == 3
    assert len(t.system_attrs["llm:violations"]) == 2


def test_batch_mode_queues_proposals(tmp_path: Any) -> None:
    reply = json.dumps(
        {
            "proposals": [
                {"params": {"x": 0.1, "n": 1, "kind": "a"}, "hypothesis": "one", "evidence": []},
                {"params": {"x": 0.2, "n": 2, "kind": "b"}, "hypothesis": "two", "evidence": []},
            ]
        }
    )
    model = FakeModel([reply, reply])
    sampler = LLMSampler(model, n_parallel=2, ledger=Ledger(tmp_path / "l.jsonl"), seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=4)
    assert len(model.prompts) == 2
    assert [t.system_attrs["llm:source"] for t in study.trials] == [
        "startup",
        "llm",
        "batch",
        "llm",
    ]
    assert study.trials[2].params["x"] == 0.2


def test_new_params_are_gated_and_recorded(tmp_path: Any) -> None:
    new = [
        {
            "name": "boost",
            "type": "float",
            "low": 0.0,
            "high": 1.0,
            "value": 0.25,
            "rationale": "why",
        }
    ]
    model = FakeModel(
        [
            proposal({"x": 0.5, "n": 3, "kind": "a"}, new_params=new),
            proposal({"x": 0.5, "n": 3, "kind": "a"}, new_params=new),
        ]
    )

    def objective_with_optional(trial: optuna.Trial) -> float:
        base = objective(trial)
        return base - trial.params.get("boost", 0.0)

    ledger = Ledger(tmp_path / "l.jsonl")
    sampler = LLMSampler(model, ledger=ledger, seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective_with_optional, n_trials=2)
    assert "boost" not in study.trials[1].params
    assert any("new_params ignored" in v for v in study.trials[1].system_attrs["llm:violations"])
    study.optimize(objective_with_optional, n_trials=1, allow_new_params=True)
    t = study.trials[2]
    assert t.params["boost"] == 0.25
    assert isinstance(t.distributions["boost"], FloatDistribution)
    assert t.system_attrs["llm:new_params"]["boost"]["rationale"] == "why"
    assert t.value == pytest.approx(0.25 + 3 - 0.25)
    assert any(line["kind"] == "new_param" for line in ledger.lines())
    # The new dimension is visible to the next proposal.
    model.replies.append(proposal({"x": 0.5, "n": 3, "kind": "a"}))
    study.optimize(objective_with_optional, n_trials=1)
    assert '"new_params": {"boost"' in model.prompts[-1][1]


def test_conditional_params_go_to_fallback(tmp_path: Any) -> None:
    def conditional(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0, 1)
        if trial.number % 2 == 0:
            x += trial.suggest_float("y", 0, 1)
        return x

    model = FakeModel([proposal({"x": 0.5})] * 5)
    sampler = LLMSampler(model, ledger=Ledger(tmp_path / "l.jsonl"), seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(conditional, n_trials=5)
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


def test_ledger_path_beside_sqlite_storage(tmp_path: Any) -> None:
    storage = f"sqlite:///{tmp_path / 'db.sqlite3'}"
    model = FakeModel([proposal({"x": 0.5, "n": 3, "kind": "a"})])
    sampler = LLMSampler(model, seed=0)
    study = optuna.create_study(study_name="s", storage=storage, sampler=sampler)
    study.optimize(objective, n_trials=2)
    assert sampler.ledger.path == tmp_path / "s.ledger.jsonl"
    assert sampler.ledger.path.exists()
    reloaded = optuna.load_study(study_name="s", storage=storage)
    assert reloaded.trials[1].system_attrs["llm:hypothesis"] == "h"


def test_multi_objective_is_accepted(tmp_path: Any) -> None:
    model = FakeModel([proposal({"x": 0.5, "n": 3, "kind": "a"})] * 3)
    sampler = LLMSampler(model, ledger=Ledger(tmp_path / "l.jsonl"), seed=0)
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
    study.optimize(lambda t: (objective(t), t.params["n"]), n_trials=3)
    assert len(study.best_trials) >= 1
