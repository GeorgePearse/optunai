from __future__ import annotations

from collections.abc import Sequence
import os
from typing import Any
from typing import TYPE_CHECKING

import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.samplers._llm._ledger import Ledger
from optuna.samplers._llm._model import Model
from optuna.samplers._llm._model import ModelLike
from optuna.samplers._llm._proposal import build_user_prompt
from optuna.samplers._llm._proposal import load_prompt
from optuna.samplers._llm._proposal import materialise_new_param
from optuna.samplers._llm._proposal import proposal_schema
from optuna.samplers._llm._proposal import ProposalError
from optuna.samplers._llm._proposal import read_context
from optuna.samplers._llm._proposal import validate_params
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_logger = optuna.logging.get_logger(__name__)

ALLOW_NEW_PARAMS_KEY = "llm:allow_new_params"


class LLMSampler(BaseSampler):
    """Sampler that asks a language model for the next trial.

    Each trial, the model receives the study directions, the search space, the trial history
    (parameters, values, states, intermediate values, user attributes), the best trial, and
    the ``context`` you pass (free text or file paths: the training script, a config, a
    README, prior results). It returns the parameter values with a one-paragraph hypothesis and
    the trial numbers it used as evidence. Proposals are validated against the distributions:
    out-of-range or off-grid values are coerced and counted as violations, a missing or
    unparseable value triggers one repair call, and if that fails the trial is sampled by the
    ``fallback`` sampler (:class:`~optuna.samplers.TPESampler` by default) and counted as a
    fallback.

    The first trial, and any parameter outside the intersection search space (conditional
    parameters), are sampled by the fallback sampler.

    The hypothesis, evidence, model id, prompt hash, tokens and USD are stored in
    ``trial.system_attrs`` under ``llm:*`` keys, so any storage and the dashboard carry them.
    Every model call, proposal, fallback, violation and trial end is appended to the
    :class:`~optuna.samplers.Ledger`.

    Example:

        .. code::

            import optuna

            sampler = optuna.samplers.LLMSampler(
                "openrouter/anthropic/claude-sonnet-5.5",
                context=["train.py", "README.md"],
            )
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=40)

    Args:
        model: A LiteLLM model string, a :class:`~optuna.samplers.Model`, or any object with a
            ``name`` and ``complete(system, user, schema)``. ``None`` uses the default model.
        context: Free text, a file path, or a sequence of them, shown to the model with every
            request. ``None`` gives the model only the numbers.
        n_parallel: Number of diverse proposals requested per model call; the extra ones are
            queued for the following trials.
        allow_new_params: Let the model add a parameter the objective did not ask for. The new
            dimension is written to the storage with its distribution and is readable with
            ``trial.params.get(name)``. ``study.optimize(..., allow_new_params=True)`` turns
            this on per study.
        fallback: Sampler used for startup trials, conditional parameters and failed
            proposals. Defaults to ``TPESampler(seed=seed)``.
        ledger: :class:`~optuna.samplers.Ledger` to append to. Defaults to
            ``<study_name>.ledger.jsonl`` beside the storage.
        seed: Seed for the fallback sampler.
        max_trials_in_prompt: Cap on the number of trials shown; the best, the most recent and a
            spread of the rest are kept.
        max_context_chars: Cap on the context text.
    """

    def __init__(
        self,
        model: ModelLike | str | None = None,
        *,
        context: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
        n_parallel: int = 1,
        allow_new_params: bool = False,
        fallback: BaseSampler | None = None,
        ledger: Ledger | None = None,
        seed: int | None = None,
        max_trials_in_prompt: int = 60,
        max_context_chars: int = 60_000,
    ) -> None:
        if model is None:
            self._model: ModelLike = Model()
        elif isinstance(model, str):
            self._model = Model(model)
        else:
            self._model = model
        if n_parallel < 1:
            raise ValueError("n_parallel must be at least 1")
        self._context_text = read_context(context, max_context_chars)
        self._n_parallel = n_parallel
        self._allow_new_params = allow_new_params
        self._fallback = (
            fallback if fallback is not None else optuna.samplers.TPESampler(seed=seed)
        )
        self._ledger = ledger if ledger is not None else Ledger()
        self._max_trials_in_prompt = max_trials_in_prompt
        self._search_space = IntersectionSearchSpace(include_pruned=True)
        self._system_prompt, self._prompt_hash = load_prompt("sampler_system.md")
        self._repair_prompt, _ = load_prompt("sampler_repair.md")
        self._queue: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self._bound = False

    @property
    def model(self) -> ModelLike:
        return self._model

    @property
    def ledger(self) -> Ledger:
        return self._ledger

    def reseed_rng(self) -> None:
        self._fallback.reseed_rng()

    def _bind(self, study: Study) -> None:
        if not self._bound:
            self._ledger.bind(study)
            self._bound = True

    @staticmethod
    def _set_attr(study: Study, trial: FrozenTrial, key: str, value: Any) -> None:
        study._storage.set_trial_system_attr(trial._trial_id, key, value)
        trial.system_attrs[key] = value

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {
            name: dist
            for name, dist in self._search_space.calculate(study).items()
            if not dist.single()
        }

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if not search_space:
            return {}
        self._bind(study)
        study_attrs = study._storage.get_study_system_attrs(study._study_id)
        allow_new = self._allow_new_params or bool(study_attrs.get(ALLOW_NEW_PARAMS_KEY))
        key = tuple(sorted(search_space))
        proposal: dict[str, Any] | None = None
        while self._queue and proposal is None:
            queued_key, queued = self._queue.pop(0)
            if queued_key == key:
                proposal = queued
                proposal["source"] = "batch"
        if proposal is None:
            try:
                proposals = self._propose(study, trial, search_space, allow_new)
            except Exception as e:  # model error, bad JSON, or an unrepairable proposal
                reason = f"{type(e).__name__}: {str(e)[:300]}"
                _logger.warning(
                    f"LLMSampler fell back to {self._fallback} for trial {trial.number}: {reason}"
                )
                self._ledger.write(
                    "fallback", trial=trial.number, reason=reason, model=self._model.name
                )
                self._set_attr(study, trial, "llm:source", "fallback")
                self._set_attr(study, trial, "llm:fallback_reason", reason)
                return {
                    name: self._fallback.sample_independent(study, trial, name, dist)
                    for name, dist in search_space.items()
                }
            proposal = proposals[0]
            for extra in proposals[1:]:
                self._queue.append((key, extra))
        return self._apply(study, trial, search_space, proposal, allow_new)

    def _propose(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
        allow_new: bool,
    ) -> list[dict[str, Any]]:
        trials = study.get_trials(
            deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
        )
        user = build_user_prompt(
            study,
            trials,
            search_space,
            n=self._n_parallel,
            context_text=self._context_text,
            allow_new_params=allow_new,
            max_trials=self._max_trials_in_prompt,
        )
        schema = proposal_schema(search_space, self._n_parallel, allow_new)
        response = self._model.complete(self._system_prompt, user, schema)
        self._ledger.write(
            "call",
            trial=trial.number,
            purpose="propose",
            model=response.model,
            prompt_hash=self._prompt_hash,
            tokens_in=response.input_tokens,
            tokens_out=response.output_tokens,
            usd=response.usd,
            latency_s=round(response.latency_s, 3),
            error=response.error,
            reply_head=None if response.parsed is not None else response.text[:400],
        )
        if response.error:
            raise RuntimeError(response.error)
        if response.parsed is None:
            raise ProposalError(
                f"no JSON object in the model reply ({response.output_tokens} output tokens)"
            )
        tokens = [response.input_tokens, response.output_tokens]
        usd = response.usd or 0.0
        try:
            proposals = self._validate_all(response.parsed, search_space, allow_new)
        except ProposalError as first_error:
            self._ledger.write(
                "violation", trial=trial.number, repaired=True, problem=str(first_error)
            )
            repair_user = (
                user
                + "\n\n# Previous reply\n"
                + response.text[:8000]
                + "\n\n"
                + self._repair_prompt.format(problems=str(first_error))
            )
            response2 = self._model.complete(self._system_prompt, repair_user, schema)
            self._ledger.write(
                "call",
                trial=trial.number,
                purpose="repair",
                model=response2.model,
                prompt_hash=self._prompt_hash,
                tokens_in=response2.input_tokens,
                tokens_out=response2.output_tokens,
                usd=response2.usd,
                latency_s=round(response2.latency_s, 3),
                error=response2.error,
            )
            tokens = [tokens[0] + response2.input_tokens, tokens[1] + response2.output_tokens]
            usd += response2.usd or 0.0
            if response2.error:
                raise RuntimeError(response2.error)
            if response2.parsed is None:
                raise ProposalError("no JSON object in the repair reply")
            proposals = self._validate_all(response2.parsed, search_space, allow_new)
            for p in proposals:
                p["source"] = "repair"
        for i, p in enumerate(proposals):
            p.setdefault("source", "llm")
            p["tokens"] = tokens if i == 0 else [0, 0]
            p["usd"] = usd if i == 0 else 0.0
            p["model"] = response.model
        return proposals

    def _validate_all(
        self, parsed: dict[str, Any], search_space: dict[str, BaseDistribution], allow_new: bool
    ) -> list[dict[str, Any]]:
        raw_list = parsed.get("proposals")
        if isinstance(parsed.get("params"), dict) and raw_list is None:
            raw_list = [parsed]
        if not isinstance(raw_list, list) or not raw_list:
            raise ProposalError("reply has no non-empty 'proposals' list")
        out: list[dict[str, Any]] = []
        for raw in raw_list[: self._n_parallel]:
            if not isinstance(raw, dict):
                raise ProposalError("a proposal is not an object")
            params, violations = validate_params(raw.get("params"), search_space)
            evidence_raw = raw.get("evidence") or []
            evidence = [
                int(e)
                for e in evidence_raw
                if isinstance(e, (int, float)) and not isinstance(e, bool)
            ]
            new_params: list[tuple[str, BaseDistribution, Any, str]] = []
            if allow_new:
                for spec in raw.get("new_params") or []:
                    name, dist, value, rationale = materialise_new_param(spec)
                    if name in search_space:
                        violations.append(f"new parameter {name!r} already exists; ignored")
                        continue
                    new_params.append((name, dist, value, rationale))
            elif raw.get("new_params"):
                violations.append("new_params ignored: allow_new_params is off")
            out.append(
                {
                    "params": params,
                    "hypothesis": str(raw.get("hypothesis", "")).strip(),
                    "evidence": evidence,
                    "violations": violations,
                    "new_params": new_params[:1],
                }
            )
        if len(out) < self._n_parallel:
            raise ProposalError(f"expected {self._n_parallel} proposals, got {len(out)}")
        return out

    def _apply(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
        proposal: dict[str, Any],
        allow_new: bool,
    ) -> dict[str, Any]:
        best_value: float | None = None
        if len(study.directions) == 1:
            try:
                best_value = study.best_value
            except ValueError:
                best_value = None
        attrs = {
            "llm:source": proposal["source"],
            "llm:hypothesis": proposal["hypothesis"],
            "llm:evidence": proposal["evidence"],
            "llm:model": proposal["model"],
            "llm:prompt_hash": self._prompt_hash,
            "llm:tokens_in": proposal["tokens"][0],
            "llm:tokens_out": proposal["tokens"][1],
            "llm:usd": proposal["usd"],
            "llm:violations": proposal["violations"],
            "llm:best_at_proposal": best_value,
        }
        for violation in proposal["violations"]:
            self._ledger.write("violation", trial=trial.number, repaired=False, problem=violation)
        new_params_attr: dict[str, Any] = {}
        for name, dist, value, rationale in proposal["new_params"]:
            study._storage.set_trial_param(
                trial._trial_id, name, dist.to_internal_repr(value), dist
            )
            trial.distributions[name] = dist
            trial.params[name] = value
            new_params_attr[name] = {
                "value": value,
                "distribution": repr(dist),
                "rationale": rationale,
            }
            self._ledger.write(
                "new_param",
                trial=trial.number,
                name=name,
                value=value,
                distribution=repr(dist),
                rationale=rationale,
            )
        if new_params_attr:
            attrs["llm:new_params"] = new_params_attr
        for key, value in attrs.items():
            self._set_attr(study, trial, key, value)
        self._ledger.write(
            "proposal",
            trial=trial.number,
            source=proposal["source"],
            params=proposal["params"],
            new_params={k: v["value"] for k, v in new_params_attr.items()},
            hypothesis=proposal["hypothesis"],
            evidence=proposal["evidence"],
            model=proposal["model"],
            prompt_hash=self._prompt_hash,
            best_at_proposal=best_value,
        )
        return proposal["params"]

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._bind(study)
        if "llm:source" not in trial.system_attrs:
            self._set_attr(study, trial, "llm:source", "startup")
        return self._fallback.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._fallback.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._fallback.after_trial(study, trial, state, values)
        self._bind(study)
        improved: bool | None = None
        best_before = trial.system_attrs.get("llm:best_at_proposal")
        if values is not None and len(values) == 1 and state == TrialState.COMPLETE:
            if best_before is None:
                improved = True
            elif study.direction == optuna.study.StudyDirection.MINIMIZE:
                improved = values[0] < best_before
            else:
                improved = values[0] > best_before
        self._ledger.write(
            "trial_end",
            trial=trial.number,
            state=state.name,
            values=list(values) if values is not None else None,
            source=trial.system_attrs.get("llm:source"),
            improved=improved,
        )
