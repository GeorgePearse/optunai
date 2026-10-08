from __future__ import annotations

import json
from typing import Any
from typing import TYPE_CHECKING

import optuna
from optuna.pruners._base import BasePruner
from optuna.pruners._median import MedianPruner
from optuna.samplers._llm._jev import boolean
from optuna.samplers._llm._jev import Jev
from optuna.samplers._llm._ledger import Ledger
from optuna.samplers._llm._model import Model
from optuna.samplers._llm._model import ModelLike
from optuna.samplers._llm._proposal import load_prompt
from optuna.study import StudyDirection
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_logger = optuna.logging.get_logger(__name__)

MAX_COMPARISON_TRIALS = 12
MAX_CURVE_POINTS = 12


def _compact_curve(values: dict[int, float]) -> list[list[float]]:
    steps = sorted(values)
    if len(steps) > MAX_CURVE_POINTS:
        idx = [
            round(i * (len(steps) - 1) / (MAX_CURVE_POINTS - 1)) for i in range(MAX_CURVE_POINTS)
        ]
        steps = [steps[i] for i in idx]
    return [[s, round(values[s], 6)] for s in steps]


class LLMPruner(BasePruner):
    """Pruner that asks a model whether the running trial will beat the best completed trial.

    At each reported step (after ``n_warmup_steps``, every ``interval_steps``), the pruner
    shows the running trial's learning curve, the best completed trial's curve and final
    value, and a sample of other completed and pruned curves, and asks for the probability
    that the running trial will finish better than the best. The trial is pruned when that
    probability is at most ``1 - threshold``.

    A :class:`~optuna.pruners.MedianPruner` with the same warm-up settings runs as a floor: a
    trial the median pruner would stop is stopped before the model is asked, so the
    ``LLMPruner`` can only prune more than the classic pruner, never less. ``aggressive=True``
    removes the floor and lets the model's verdict stand alone.

    The question goes to Jev (a typed evaluation model on the Vercel AI Gateway) when
    ``AI_GATEWAY_API_KEY`` is set and no ``model`` is given; otherwise to the chat ``model``,
    which answers with JSON. Each decision is written to the ledger with its probability and
    stored in ``trial.system_attrs["llm:prune"]``.

    Args:
        model: A LiteLLM model string, a :class:`~optuna.samplers.Model`, or any object with
            ``name`` and ``complete(system, user, schema)``. ``None`` selects Jev when it is
            configured, else the default :class:`~optuna.samplers.Model`.
        threshold: Prune when the probability that the trial will *not* beat the best is at
            least this value.
        aggressive: Drop the median-pruner floor.
        n_startup_trials: Do not ask the model until this many trials have completed.
        n_warmup_steps: Do not prune before this step.
        interval_steps: Ask the model every this many steps after warm-up.
        ledger: :class:`~optuna.samplers.Ledger` to append decisions to.
    """

    def __init__(
        self,
        model: ModelLike | str | None = None,
        *,
        threshold: float = 0.8,
        aggressive: bool = False,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        ledger: Ledger | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._jev: Jev | None = None
        self._model: ModelLike | None = None
        if model is None and Jev.configured():
            self._jev = Jev()
        elif model is None:
            self._model = Model()
        elif isinstance(model, str):
            self._model = Model(model)
        else:
            self._model = model
        self._threshold = threshold
        self._aggressive = aggressive
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = max(1, interval_steps)
        self._ledger = ledger if ledger is not None else Ledger()
        self._median = MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=self._interval_steps,
        )
        self._system_prompt, self._prompt_hash = load_prompt("pruner_system.md")
        self._bound = False
        self.usd = 0.0
        self.calls = 0

    @property
    def ledger(self) -> Ledger:
        return self._ledger

    @property
    def judge_name(self) -> str:
        return self._jev.name if self._jev is not None else self._model.name  # type: ignore[union-attr]

    def _state_text(self, study: Study, trial: FrozenTrial, step: int) -> str | None:
        direction = study.direction
        completed = [
            t
            for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if t.value is not None
        ]
        if len(completed) < self._n_startup_trials:
            return None
        sign = 1.0 if direction == StudyDirection.MINIMIZE else -1.0
        completed.sort(key=lambda t: sign * float(t.value))  # type: ignore[arg-type]
        best = completed[0]
        others = completed[1:]
        if len(others) > MAX_COMPARISON_TRIALS:
            stride = max(1, len(others) // MAX_COMPARISON_TRIALS)
            others = others[::stride][:MAX_COMPARISON_TRIALS]
        pruned = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))[-4:]
        payload: dict[str, Any] = {
            "direction": direction.name.lower(),
            "running_trial": {
                "number": trial.number,
                "current_step": step,
                "params": trial.params,
                "curve": _compact_curve(trial.intermediate_values),
            },
            "best_completed_trial": {
                "number": best.number,
                "final_value": best.value,
                "params": best.params,
                "curve": _compact_curve(best.intermediate_values),
                "value_at_same_step": best.intermediate_values.get(step),
            },
            "other_completed_trials": [
                {
                    "number": t.number,
                    "final_value": t.value,
                    "value_at_same_step": t.intermediate_values.get(step),
                    "curve": _compact_curve(t.intermediate_values),
                }
                for t in others
            ],
            "recently_pruned_trials": [
                {"number": t.number, "curve": _compact_curve(t.intermediate_values)}
                for t in pruned
            ],
        }
        return json.dumps(payload)

    def _probability(self, state: str) -> tuple[float, str, float | None, float]:
        """Return (p_beat_best, reason, usd, latency_s)."""
        if self._jev is not None:
            answer = self._jev.ask(
                state,
                {
                    "beats_best": boolean(
                        "Will the running trial, if allowed to finish, end with a final value "
                        "better than the best completed trial's final value? Judge from the "
                        "curve shapes at matching steps, in the study's direction.",
                        true="the running trial's trajectory is on course to beat the best",
                        false="the running trial is flat or behind the best at the same step "
                        "and unlikely to overtake",
                    )
                },
            )
            usd_before = self._jev.usd
            p = float(answer["beats_best"]["p"])
            return (
                p,
                "jev",
                self._jev.usd - usd_before if self._jev.usd else None,
                self._jev.last_latency_s,
            )
        assert self._model is not None
        response = self._model.complete(
            self._system_prompt,
            state,
            {
                "type": "object",
                "properties": {
                    "p_beat_best": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["p_beat_best", "reason"],
            },
        )
        if response.error or response.parsed is None:
            raise RuntimeError(response.error or "no JSON in pruner reply")
        p = float(response.parsed.get("p_beat_best", 0.5))
        p = min(max(p, 0.0), 1.0)
        return p, str(response.parsed.get("reason", ""))[:300], response.usd, response.latency_s

    def prune(self, study: Study, trial: FrozenTrial) -> bool:
        if not self._bound:
            self._ledger.bind(study)
            self._bound = True
        floor = self._median.prune(study, trial)
        if floor and not self._aggressive:
            self._ledger.write(
                "prune", trial=trial.number, step=trial.last_step, decision=True, by="median"
            )
            return True
        step = trial.last_step
        if step is None or step < self._n_warmup_steps:
            return floor
        if (step - self._n_warmup_steps) % self._interval_steps != 0:
            return floor
        state = self._state_text(study, trial, step)
        if state is None:
            return floor
        try:
            p, reason, usd, latency = self._probability(state)
        except Exception as e:
            _logger.warning(f"LLMPruner judge failed at trial {trial.number} step {step}: {e}")
            self._ledger.write(
                "prune",
                trial=trial.number,
                step=step,
                decision=floor,
                by="floor",
                error=str(e)[:300],
            )
            return floor
        self.calls += 1
        if usd:
            self.usd += usd
        decision = (1.0 - p) >= self._threshold
        self._ledger.write(
            "call",
            trial=trial.number,
            purpose="prune",
            model=self.judge_name,
            prompt_hash=self._prompt_hash,
            usd=usd,
            latency_s=round(latency, 3),
        )
        self._ledger.write(
            "prune",
            trial=trial.number,
            step=step,
            decision=decision,
            by="llm",
            p_beat_best=round(p, 4),
            median_floor=floor,
            reason=reason,
        )
        record = trial.system_attrs.get("llm:prune") or []
        record = list(record) + [{"step": step, "p_beat_best": round(p, 4), "prune": decision}]
        study._storage.set_trial_system_attr(trial._trial_id, "llm:prune", record[-20:])
        return decision if self._aggressive else (floor or decision)
