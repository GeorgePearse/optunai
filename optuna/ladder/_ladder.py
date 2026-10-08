from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
import math
from typing import Any
from typing import TYPE_CHECKING

import optuna
from optuna.ladder._decision import decide
from optuna.ladder._decision import Decision
from optuna.ladder._decision import DecisionRecord
from optuna.ladder._fit import bootstrap_interval
from optuna.ladder._fit import choose_form
from optuna.ladder._fit import Curve
from optuna.ladder._fit import equivalent_compute_multiplier
from optuna.ladder._fit import Fit
from optuna.ladder._fit import fit_curve
from optuna.ladder._fit import pooled_gamma
from optuna.ladder._fit import predict_from_prefix
from optuna.ladder._fit import prefix_residuals
from optuna.pruners._base import BasePruner
from optuna.samplers._llm._ledger import Ledger
from optuna.study import StudyDirection
from optuna.trial import BaseTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


def _sys(study: "Study") -> dict[str, Any]:
    return study._storage.get_study_system_attrs(study._study_id)


_logger = optuna.logging.get_logger(__name__)

STATUS_REASONS = (
    "completed",
    "actively_stopped",
    "algorithmic_divergence",
    "infrastructure_failure",
    "implementation_error",
)
_DIVERGENCE_WORDS = ("nan", "inf", "diverge", "overflow", "not finite", "exploded")
_INFRA_WORDS = (
    "cuda",
    "out of memory",
    "oom",
    "nccl",
    "connection",
    "timeout",
    "timed out",
    "disk",
    "no space",
    "broken pipe",
    "rate limit",
    "503",
    "502",
    "unavailable",
)
_IMPLEMENTATION_TYPES = (
    TypeError,
    AttributeError,
    KeyError,
    IndexError,
    NameError,
    ImportError,
    AssertionError,
    NotImplementedError,
)


def classify_exception(e: BaseException) -> tuple[str, bool]:
    """Map an exception from the objective to ``(status_reason, cause_unknown)`` (post §12.1).

    Divergence: floating-point errors, or a ``ValueError``/``RuntimeError`` that talks about
    NaN, infinity or overflow. Infrastructure: memory, OS and network errors, CUDA/NCCL/OOM
    messages, anything whose class name says timeout, connection or HTTP. Implementation:
    type, attribute, key, index, name, import and assertion errors. Anything else is recorded as
    an implementation error with ``cause_unknown`` set, which the ledger keeps until someone
    resolves it; it is never silently counted as a divergence.
    """
    name = type(e).__name__.lower()
    text = str(e).lower()
    if isinstance(e, (FloatingPointError, OverflowError, ZeroDivisionError)):
        return "algorithmic_divergence", False
    if isinstance(e, (MemoryError, OSError)):
        return "infrastructure_failure", False
    if any(w in name for w in ("timeout", "connection", "http", "resource", "unavailable")):
        return "infrastructure_failure", False
    if isinstance(e, (ValueError, RuntimeError, ArithmeticError)):
        if any(w in text for w in _INFRA_WORDS):
            return "infrastructure_failure", False
        if any(w in text for w in _DIVERGENCE_WORDS):
            return "algorithmic_divergence", False
    if isinstance(e, _IMPLEMENTATION_TYPES):
        return "implementation_error", False
    return "implementation_error", True


class Rung:
    """One rung handed to the objective by :meth:`Ladder.rungs`; call :meth:`report` once."""

    def __init__(self, ladder: Ladder, trial: BaseTrial, index: int, budget: float, n: int):
        self.ladder = ladder
        self.trial = trial
        self.index = index
        self.budget = budget
        self.n = n
        self.reported = False

    @property
    def is_last(self) -> bool:
        return self.index == self.n - 1

    @property
    def fraction(self) -> float:
        return self.budget / self.ladder.target

    def report(self, value: float) -> None:
        """Record ``value`` for this rung, and stop the trial if the ladder kills it."""
        self.reported = True
        trial = self.trial
        rungs = list(trial.user_attrs.get("ladder:rungs") or [])
        rungs.append([self.budget, value])
        trial.set_user_attr("ladder:rungs", rungs)
        spent = max(b for b, _ in rungs) if self.ladder.cumulative else sum(b for b, _ in rungs)
        trial.set_user_attr("ladder:compute", spent)
        if self.is_last:
            self.ladder._check_frozen(trial, value)
            return
        trial.report(value, step=self.index)
        if trial.should_prune():
            trial.set_user_attr("ladder:status_reason", "actively_stopped")
            trial.set_user_attr("ladder:cause_unknown", False)
            trial.set_user_attr("ladder:stop_reason", "ladder_kill")
            raise optuna.TrialPruned(f"ladder killed the candidate after rung {self.index}")


class Ladder(BasePruner):
    """Multi-rung fidelity with fitted extrapolation (post §5-§8, §10.6).

    Each candidate runs at ``rungs`` of increasing budget; the objective asks for them with
    :meth:`rungs` and reports each rung's value. After every rung but the last the ladder fits
    ``L(C) = E + A C^-γ`` (or ``E + A ln C``, whichever has the lower leave-last-rung-out error
    over the completed candidates) to the candidate's rungs, extrapolates to the target budget,
    attaches a prediction interval built from the extrapolation errors of the completed
    candidates (bootstrapped with the candidate as the unit), and kills the candidate only when
    even the better end of that interval does not reach the incumbent's target value minus the
    study's frozen threshold. A candidate that is behind at the current rung but extrapolates to
    win is promoted; ``HyperbandPruner`` would drop it on rank. With one rung the prediction is
    the pooled rung-to-target shift; with two, the pooled exponent is used.

    Before ``n_startup`` candidates have completed all rungs nothing is killed (decision
    ``insufficient_evidence``). The prediction and interval at the moment a candidate is
    promoted to its final rung are frozen in the ledger (``prediction_frozen``) and the final
    result is checked against them (``ladder:inside_interval``).

    Example:

        .. code::

            ladder = optuna.ladder.Ladder(rungs=[0.25, 0.5, 1.0])
            study = optuna.create_study(direction="maximize", pruner=ladder)


            def objective(trial):
                lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
                for rung in ladder.rungs(trial):
                    score = train(lr, data_fraction=rung.fraction)
                    rung.report(score)
                return score


            study.optimize(ladder.wrap(objective), n_trials=40)

    Args:
        rungs: Budgets in increasing order; the last one is the target. Fractions, steps or
            any positive number the objective understands.
        n_startup: Completed candidates needed before a kill is possible.
        level: Coverage of the prediction interval.
        gamma_prior: Exponent used for two-rung predictions until a pooled exponent exists.
        n_bootstrap: Bootstrap resamples for the interval.
        min_rungs_to_kill: Rungs a candidate must have reported before it can be killed. The
            default of two means a kill always rests on a slope, never on one point; with one
            point the "prediction" is the pooled shift, which is early rank in disguise.
        seed: Seed for the bootstrap.
        ledger: :class:`~optuna.samplers.Ledger`; defaults to the sampler's ledger when the
            study's sampler has one, else a new one.
        fallback_sd: Scale for the interval while fewer than four candidates have completed;
            defaults to the seed SD from :meth:`Study.register_noise` when present.
        cumulative: Rungs continue the previous one (a warm start), so the compute a trial
            spends is the largest budget it reached. ``False`` means every rung restarts and
            compute is the sum of the budgets run.
    """

    def __init__(
        self,
        rungs: list[float] | tuple[float, ...] = (0.25, 0.5, 1.0),
        *,
        n_startup: int = 4,
        level: float = 0.9,
        gamma_prior: float = 0.5,
        n_bootstrap: int = 200,
        min_rungs_to_kill: int = 2,
        seed: int = 0,
        ledger: Ledger | None = None,
        fallback_sd: float | None = None,
        cumulative: bool = True,
    ) -> None:
        budgets = [float(b) for b in rungs]
        if len(budgets) < 2 or any(b <= 0 for b in budgets) or budgets != sorted(budgets):
            raise ValueError("rungs must be at least two positive budgets in increasing order")
        if len(set(budgets)) != len(budgets):
            raise ValueError("rungs must be distinct")
        self._budgets = budgets
        self._n_startup = n_startup
        self._level = level
        self._gamma_prior = gamma_prior
        self._n_bootstrap = n_bootstrap
        self._min_rungs_to_kill = max(1, min_rungs_to_kill)
        self._seed = seed
        self._ledger = ledger
        self._fallback_sd = fallback_sd
        self.cumulative = cumulative
        self._bound = False

    @property
    def budgets(self) -> list[float]:
        return list(self._budgets)

    @property
    def target(self) -> float:
        return self._budgets[-1]

    def ledger_for(self, study: Study) -> Ledger:
        if self._ledger is None:
            sampler_ledger = getattr(study.sampler, "ledger", None)
            self._ledger = sampler_ledger if isinstance(sampler_ledger, Ledger) else Ledger()
        if not self._bound:
            self._ledger.bind(study)
            self._bound = True
            study._storage.set_study_system_attr(study._study_id, "ladder:rungs", self._budgets)
        return self._ledger

    # ---- objective side -------------------------------------------------------------------

    def rungs(self, trial: BaseTrial) -> Iterator[Rung]:
        """Yield the rungs for ``trial``; the objective reports each one.

        A trial the sampler marked ``llm:decision == "defer"`` (the Jev gate) is stopped here,
        before any compute is spent, with status ``actively_stopped`` / ``deferred_by_gate``.
        """
        frozen = getattr(trial, "_cached_frozen_trial", None)
        system_attrs = getattr(frozen, "system_attrs", None) or {}
        if system_attrs.get("llm:decision") == "defer":
            trial.set_user_attr("ladder:status_reason", "actively_stopped")
            trial.set_user_attr("ladder:cause_unknown", False)
            trial.set_user_attr("ladder:stop_reason", "deferred_by_gate")
            trial.set_user_attr("ladder:compute", 0.0)
            raise optuna.TrialPruned("deferred: expected effect judged below the threshold")
        n = len(self._budgets)
        for i, budget in enumerate(self._budgets):
            rung = Rung(self, trial, i, budget, n)
            yield rung
            if not rung.reported:
                raise RuntimeError(f"rung {i} was not reported; call rung.report(value)")

    def wrap(self, objective: Callable[[optuna.Trial], float]) -> Callable[[optuna.Trial], float]:
        """Return ``objective`` with run-status classification (post §12.1).

        A normal return sets ``ladder:status_reason = completed``; a non-finite value sets
        ``algorithmic_divergence`` (Optuna then fails the trial); an exception is classified by
        :func:`classify_exception` and re-raised. Nothing is swallowed.
        """

        def wrapped(trial: optuna.Trial) -> float:
            try:
                value = objective(trial)
            except optuna.TrialPruned:
                if "ladder:status_reason" not in trial.user_attrs:
                    trial.set_user_attr("ladder:status_reason", "actively_stopped")
                    trial.set_user_attr("ladder:cause_unknown", False)
                    trial.set_user_attr("ladder:stop_reason", "pruned")
                self._status_line(trial)
                raise
            except Exception as e:
                reason, unknown = classify_exception(e)
                trial.set_user_attr("ladder:status_reason", reason)
                trial.set_user_attr("ladder:cause_unknown", unknown)
                trial.set_user_attr("ladder:error", f"{type(e).__name__}: {str(e)[:300]}")
                self._status_line(trial)
                raise
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                trial.set_user_attr("ladder:status_reason", "algorithmic_divergence")
                trial.set_user_attr("ladder:cause_unknown", False)
                trial.set_user_attr("ladder:error", f"non-finite value {value!r}")
            else:
                trial.set_user_attr("ladder:status_reason", "completed")
                trial.set_user_attr("ladder:cause_unknown", False)
            self._status_line(trial)
            return value

        wrapped.__name__ = getattr(objective, "__name__", "objective")
        return wrapped

    def _status_line(self, trial: optuna.Trial) -> None:
        attrs = trial.user_attrs
        self.ledger_for(trial.study).write(
            "status",
            trial=trial.number,
            status_reason=attrs.get("ladder:status_reason"),
            cause_unknown=attrs.get("ladder:cause_unknown"),
            stop_reason=attrs.get("ladder:stop_reason"),
            error=attrs.get("ladder:error"),
            supersedes=attrs.get("ladder:supersedes"),
            compute=attrs.get("ladder:compute"),
        )

    def retry(self, study: Study, trial: FrozenTrial) -> None:
        """Enqueue the same parameters again and link both records; nothing is deleted.

        Meant for ``infrastructure_failure`` trials. The new trial carries
        ``ladder:supersedes``; :meth:`superseded_by` resolves the other direction.
        """
        user_attrs = {
            "ladder:supersedes": trial.number,
            "ladder:seed": trial.user_attrs.get("ladder:seed", 0),
        }
        study.enqueue_trial(trial.params, user_attrs=user_attrs)
        self.ledger_for(study).write(
            "retry",
            trial=trial.number,
            status_reason=trial.user_attrs.get("ladder:status_reason"),
            params=trial.params,
        )

    @staticmethod
    def superseded_by(study: Study, number: int) -> int | None:
        for t in study.get_trials(deepcopy=False):
            if t.user_attrs.get("ladder:supersedes") == number:
                return t.number
        return None

    # ---- pruner side ----------------------------------------------------------------------

    @staticmethod
    def _sign(study: Study) -> float:
        if len(study.directions) != 1:
            raise ValueError("Ladder supports single-objective studies only")
        return 1.0 if study.direction == StudyDirection.MINIMIZE else -1.0

    def completed_curves(self, study: Study, *, exclude: int | None = None) -> list[Curve]:
        """Loss-scale rung curves of every candidate that completed all rungs."""
        sign = self._sign(study)
        out: list[Curve] = []
        for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
            if t.number == exclude or t.value is None:
                continue
            if t.user_attrs.get("ladder:status_reason") == "infrastructure_failure":
                continue
            rungs = t.user_attrs.get("ladder:rungs")
            if not rungs or len(rungs) != len(self._budgets):
                continue
            out.append(
                Curve(t.number, [float(b) for b, _ in rungs], [sign * float(v) for _, v in rungs])
            )
        return out

    def _fallback_scale(self, study: Study) -> float | None:
        if self._fallback_sd is not None:
            return self._fallback_sd
        sd = _sys(study).get("ladder:seed_sd")
        return float(sd) if isinstance(sd, (int, float)) else None

    def incumbent_fit(self, study: Study) -> tuple[Fit, float, int] | None:
        """``(fit, loss_at_target, trial_number)`` of the best completed candidate."""
        curves = self.completed_curves(study)
        if not curves:
            return None
        best = min(curves, key=lambda c: c.losses[-1])
        if len(best.budgets) >= 3:
            fit = fit_curve(
                best.budgets, best.losses, choose_form(curves, gamma_prior=self._gamma_prior)
            )
        else:
            fit = fit_curve(
                best.budgets,
                best.losses,
                "power",
                gamma=pooled_gamma(curves, self._gamma_prior),
                source="pooled_gamma",
            )
        return fit, best.losses[-1], best.unit

    def multiplier(self, study: Study, delta_improvement: float) -> float | None:
        """Equivalent compute multiplier of an improvement over the incumbent (post §2.3)."""
        inc = self.incumbent_fit(study)
        if inc is None:
            return None
        fit, loss, _ = inc
        return equivalent_compute_multiplier(fit, loss, delta_improvement)

    def fit_report(self, study: Study) -> dict[str, Any]:
        curves = self.completed_curves(study)
        form = choose_form(curves, gamma_prior=self._gamma_prior)
        inc = self.incumbent_fit(study)
        report: dict[str, Any] = {
            "rungs": self._budgets,
            "form": form,
            "pooled_gamma": pooled_gamma(curves, self._gamma_prior),
            "n_units": len(curves),
            "incumbent": None,
        }
        if inc is not None:
            fit, loss, number = inc
            sign = self._sign(study)
            report["incumbent"] = {"trial": number, "value": sign * loss, "fit": fit.as_dict()}
        return report

    def prune(self, study: Study, trial: FrozenTrial) -> bool:
        step = trial.last_step
        if step is None:
            return False
        k = step + 1
        if k >= len(self._budgets):
            return False
        ledger = self.ledger_for(study)
        sign = self._sign(study)
        rungs = trial.user_attrs.get("ladder:rungs") or []
        if len(rungs) < k:  # reported through trial.report without the Rung helper
            iv = trial.intermediate_values
            rungs = [[self._budgets[s], iv[s]] for s in sorted(iv) if s < len(self._budgets)]
        budgets = [float(b) for b, _ in rungs][:k]
        losses = [sign * float(v) for _, v in rungs][:k]
        curves = self.completed_curves(study, exclude=trial.number)
        threshold = float(_sys(study).get("ladder:threshold") or 0.0)
        if len(curves) < self._n_startup or not budgets:
            record = DecisionRecord(
                Decision.INSUFFICIENT_EVIDENCE,
                "rung",
                math.nan,
                (-math.inf, math.inf),
                threshold,
                trial_number=trial.number,
                rung=step,
                reason=f"{len(curves)} completed candidates, {self._n_startup} needed",
            )
            self._record(study, trial, record, ledger)
            return False
        form = choose_form(curves, gamma_prior=self._gamma_prior)
        pred, fit = predict_from_prefix(
            budgets, losses, self.target, form=form, pool=curves, gamma_prior=self._gamma_prior
        )
        residuals = prefix_residuals(
            curves, k, self.target, form=form, gamma_prior=self._gamma_prior
        )
        lo_r, hi_r, method = bootstrap_interval(
            residuals,
            level=self._level,
            n_bootstrap=self._n_bootstrap,
            seed=self._seed + trial.number,
            fallback_sd=self._fallback_scale(study),
        )
        pred_lo, pred_hi = pred + lo_r, pred + hi_r  # loss scale
        incumbent = min(curves, key=lambda c: c.losses[-1])
        diff = incumbent.losses[-1] - pred
        lo, hi = incumbent.losses[-1] - pred_hi, incumbent.losses[-1] - pred_lo
        decision = decide(diff, lo, hi, threshold, can_run_more=True)
        reason = f"fit={fit.source if fit else 'none'} form={form} units={len(curves)}"
        if decision == Decision.DEFER and k < self._min_rungs_to_kill:
            decision = Decision.RUN_MORE
            reason += f"; kill withheld, {k} rung(s) < min_rungs_to_kill={self._min_rungs_to_kill}"
        record = DecisionRecord(
            decision,
            "rung",
            diff,
            (lo, hi),
            threshold,
            equivalent_compute_multiplier=self.multiplier(study, diff),
            trial_number=trial.number,
            incumbent_trial_number=incumbent.unit,
            predicted_target_value=sign * pred,
            rung=step,
            interval_method=method,
            reason=reason,
        )
        value_interval = (min(sign * pred_lo, sign * pred_hi), max(sign * pred_lo, sign * pred_hi))
        self._record(study, trial, record, ledger, fit=fit, predicted_interval=value_interval)
        if decision != Decision.DEFER and k == len(self._budgets) - 1:
            ledger.write(
                "prediction_frozen",
                trial=trial.number,
                rung=step,
                predicted_target_value=sign * pred,
                interval=list(value_interval),
                interval_method=method,
                form=form,
                fit=fit.as_dict() if fit else None,
                incumbent_trial=incumbent.unit,
                incumbent_value=sign * incumbent.losses[-1],
                threshold=threshold,
            )
        return decision == Decision.DEFER

    def _record(
        self,
        study: Study,
        trial: FrozenTrial,
        record: DecisionRecord,
        ledger: Ledger,
        *,
        fit: Fit | None = None,
        predicted_interval: tuple[float, float] | None = None,
    ) -> None:
        entry = record.as_dict()
        entry["predicted_interval"] = list(predicted_interval) if predicted_interval else None
        entry["fit"] = fit.as_dict() if fit else None
        history = list(trial.user_attrs.get("ladder:predicted") or [])
        history.append(entry)
        study._storage.set_trial_user_attr(trial._trial_id, "ladder:predicted", history)
        trial.user_attrs["ladder:predicted"] = history
        ledger.write("rung", **{k: v for k, v in entry.items() if k != "kind"})

    def _check_frozen(self, trial: BaseTrial, value: float) -> None:
        history = trial.user_attrs.get("ladder:predicted") or []
        if not history:
            return
        last = history[-1]
        if last.get("rung") != len(self._budgets) - 2 or not last.get("predicted_interval"):
            return
        lo, hi = last["predicted_interval"]
        inside = bool(lo <= value <= hi) if all(math.isfinite(x) for x in (lo, hi)) else None
        trial.set_user_attr("ladder:inside_interval", inside)
        trial.set_user_attr(
            "ladder:prediction_error", value - float(last["predicted_target_value"])
        )

    def compute(self, study: Study) -> float:
        """Total rung budget spent by every trial of the study."""
        return float(
            sum(
                float(t.user_attrs.get("ladder:compute") or 0.0)
                for t in study.get_trials(deepcopy=False)
            )
        )

    def calibration(self, study: Study) -> dict[str, Any]:
        """How often the final result fell inside its frozen interval."""
        checked = [
            t.user_attrs.get("ladder:inside_interval")
            for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if "ladder:inside_interval" in t.user_attrs
        ]
        known = [c for c in checked if c is not None]
        return {
            "n_frozen": len(checked),
            "n_inside": sum(bool(c) for c in known),
            "coverage": (sum(bool(c) for c in known) / len(known)) if known else None,
            "level": self._level,
        }
