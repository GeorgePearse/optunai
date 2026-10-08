"""Rung-curve fitting for the ladder: two functional forms, pooled exponent, residual intervals.

Everything here works on a *loss* scale (lower is better); the caller negates values of a
maximised objective. The fitting protocol is fixed (post §8.3): fit on the value scale, exponent
on a log-spaced grid with a linear solve for the remaining two coefficients, no outlier removal,
no weights. Prediction intervals come from leave-last-rungs-out residuals over completed
candidates, bootstrapped with the candidate as the resampling unit (post §5.4, §8.3): all rungs
of one candidate are one unit because a warm-started objective makes them dependent.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


FORMS = ("power", "log")
GAMMA_GRID = np.exp(np.linspace(math.log(0.05), math.log(4.0), 48))


@dataclass(frozen=True)
class Fit:
    """One fitted curve.

    ``power``: ``L(C) = E + A * C ** -gamma``. ``log``: ``L(C) = E + A * ln(C)`` (``gamma`` is
    0), the no-asymptote alternative. ``source`` says how the coefficients were obtained:
    ``own`` (all three from this candidate's rungs), ``pooled_gamma`` (two rungs, exponent from
    the pooled fit), ``pooled_shift`` (one rung, the pooled rung-to-target shift).
    """

    form: str
    E: float
    A: float
    gamma: float
    rmse: float
    n_points: int
    source: str

    def predict(self, budget: float) -> float:
        if self.form == "power":
            return self.E + self.A * budget ** (-self.gamma)
        return self.E + self.A * math.log(budget)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("E", "A", "gamma", "rmse"):
            d[k] = _r(d[k])
        return d


def _r(x: float) -> float:
    return float(f"{x:.6g}") if math.isfinite(x) else x


def _design(budgets: np.ndarray, form: str, gamma: float) -> np.ndarray:
    x = budgets ** (-gamma) if form == "power" else np.log(budgets)
    return np.column_stack([np.ones_like(x), x])


def _solve(
    budgets: np.ndarray, losses: np.ndarray, form: str, gamma: float
) -> tuple[float, float, float]:
    """Least-squares ``(E, A, rmse)`` for a fixed exponent."""
    X = _design(budgets, form, gamma)
    coef, *_ = np.linalg.lstsq(X, losses, rcond=None)
    resid = losses - X @ coef
    rmse = float(np.sqrt(np.mean(resid**2))) if len(losses) > 2 else 0.0
    return float(coef[0]), float(coef[1]), rmse


def fit_curve(
    budgets: list[float] | np.ndarray,
    losses: list[float] | np.ndarray,
    form: str = "power",
    *,
    gamma: float | None = None,
    source: str = "own",
) -> Fit:
    """Fit one form to one candidate's ``(budget, loss)`` rungs.

    With ``gamma`` given (or the ``log`` form) the fit is linear and needs two points. Without,
    the exponent is chosen on :data:`GAMMA_GRID` by least squares and needs three.
    """
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    b = np.asarray(budgets, dtype=float)
    y = np.asarray(losses, dtype=float)
    if len(b) != len(y) or len(b) < 2:
        raise ValueError("need at least two (budget, loss) points")
    if np.any(b <= 0):
        raise ValueError("budgets must be positive")
    if form == "log":
        E, A, rmse = _solve(b, y, form, 0.0)
        return Fit("log", E, A, 0.0, rmse, len(b), source)
    if gamma is not None:
        E, A, rmse = _solve(b, y, form, gamma)
        return Fit("power", E, A, float(gamma), rmse, len(b), source)
    if len(b) < 3:
        raise ValueError("fitting the exponent needs three points; pass gamma for two")
    best: tuple[float, float, float, float] | None = None
    for g in GAMMA_GRID:
        E, A, rmse = _solve(b, y, form, float(g))
        if best is None or rmse < best[3]:
            best = (E, A, float(g), rmse)
    assert best is not None
    return Fit("power", best[0], best[1], best[2], best[3], len(b), source)


@dataclass
class Curve:
    """A candidate's rung results: ``budgets[i]`` was run and gave ``losses[i]``."""

    unit: int
    budgets: list[float]
    losses: list[float]


def pooled_gamma(completed: list[Curve], prior: float) -> float:
    """Median exponent over completed candidates with three or more rungs; ``prior`` if none."""
    gammas = [
        fit_curve(c.budgets, c.losses, "power").gamma for c in completed if len(c.budgets) >= 3
    ]
    return float(np.median(gammas)) if gammas else prior


def predict_from_prefix(
    budgets: list[float],
    losses: list[float],
    target: float,
    *,
    form: str,
    pool: list[Curve],
    gamma_prior: float,
) -> tuple[float, Fit | None]:
    """Predict the loss at ``target`` from the first rungs of one candidate.

    Three or more rungs: the candidate's own fit. Two: a linear fit with the pooled exponent
    (power form) or the candidate's own log fit. One: the pooled rung-to-target shift, the
    median over ``pool`` of ``loss(target) - loss(first rung)``; a pooled shift of zero when
    the pool is empty, which is the "no information" prediction.
    """
    k = len(budgets)
    if k >= 3:
        fit = fit_curve(budgets, losses, form)
        return fit.predict(target), fit
    if k == 2:
        gamma = pooled_gamma(pool, gamma_prior) if form == "power" else None
        fit = fit_curve(budgets, losses, form, gamma=gamma, source="pooled_gamma")
        return fit.predict(target), fit
    shifts = [
        c.losses[-1] - c.losses[0]
        for c in pool
        if len(c.budgets) >= 2 and abs(c.budgets[0] - budgets[0]) < 1e-12
    ]
    shift = float(np.median(shifts)) if shifts else 0.0
    return losses[0] + shift, Fit("power", losses[0] + shift, 0.0, 0.0, 0.0, 1, "pooled_shift")


def choose_form(completed: list[Curve], *, gamma_prior: float, min_units: int = 3) -> str:
    """Pick the form with the lower leave-last-rung-out error over completed candidates.

    Development data only: the caller never passes acceptance (holdout) results here. Ties and
    too few units keep ``power``.
    """
    usable = [c for c in completed if len(c.budgets) >= 3]
    if len(usable) < min_units:
        return "power"
    errors: dict[str, float] = {}
    for form in FORMS:
        se = []
        for c in usable:
            pool = [o for o in usable if o.unit != c.unit]
            pred, _ = predict_from_prefix(
                c.budgets[:-1],
                c.losses[:-1],
                c.budgets[-1],
                form=form,
                pool=pool,
                gamma_prior=gamma_prior,
            )
            se.append((pred - c.losses[-1]) ** 2)
        errors[form] = float(np.mean(se))
    return "log" if errors["log"] < errors["power"] else "power"


def prefix_residuals(
    completed: list[Curve], k: int, target: float, *, form: str, gamma_prior: float
) -> list[tuple[int, float]]:
    """``(unit, actual - predicted)`` at the target for every completed candidate, predicted
    from its first ``k`` rungs with the pool being the other candidates (leave-one-unit-out)."""
    out: list[tuple[int, float]] = []
    for c in completed:
        if len(c.budgets) <= k:
            continue
        pool = [o for o in completed if o.unit != c.unit]
        pred, _ = predict_from_prefix(
            c.budgets[:k], c.losses[:k], target, form=form, pool=pool, gamma_prior=gamma_prior
        )
        out.append((c.unit, c.losses[-1] - pred))
    return out


def bootstrap_interval(
    residuals: list[tuple[int, float]],
    *,
    level: float = 0.9,
    n_bootstrap: int = 200,
    seed: int = 0,
    fallback_sd: float | None = None,
) -> tuple[float, float, str]:
    """Quantiles of the residual distribution, bagged over bootstrap resamples of the units.

    Returns ``(lo, hi, method)``. With fewer than four units the empirical quantiles are
    meaningless, so the interval is ``±3 * fallback_sd`` (``method="fallback"``), or
    ``(-inf, +inf)`` (``method="none"``) when no scale is known; an unbounded interval never
    kills anything. With enough units the interval is never narrower than ``±2 * fallback_sd``
    when that scale is known: a single run cannot be predicted better than seed noise.
    """
    units = sorted({u for u, _ in residuals})
    by_unit = {u: [r for uu, r in residuals if uu == u] for u in units}
    if len(units) < 4:
        if fallback_sd is not None and fallback_sd > 0:
            return -3.0 * fallback_sd, 3.0 * fallback_sd, "fallback"
        return -math.inf, math.inf, "none"
    rng = np.random.default_rng(seed)
    alpha = (1.0 - level) / 2.0
    los, his = [], []
    for _ in range(n_bootstrap):
        pick = rng.choice(units, size=len(units), replace=True)
        sample = np.array([r for u in pick for r in by_unit[u]])
        los.append(np.quantile(sample, alpha))
        his.append(np.quantile(sample, 1.0 - alpha))
    lo, hi = float(np.mean(los)), float(np.mean(his))
    if fallback_sd is not None and fallback_sd > 0:
        lo, hi = min(lo, -2.0 * fallback_sd), max(hi, 2.0 * fallback_sd)
    return lo, hi, "bootstrap"


def equivalent_compute_multiplier(
    fit: Fit, loss_at_target: float, delta_loss: float
) -> float | None:
    """``exp(ΔL / (γ (L − E)))`` for the power form, ``exp(ΔL / |A|)`` for the log form.

    ``delta_loss`` is the loss the candidate saves relative to the curve's owner (positive =
    candidate better); the result is the factor by which the owner's compute would have to grow
    to match it (post §2.3). ``None`` when the curve has no slope to convert with, or when the
    fitted exponent sits on the search grid's boundary (the form does not describe the curve and
    the multiplier would be arbitrary).
    """
    if fit.form == "power":
        denom = fit.gamma * (loss_at_target - fit.E)
        if fit.gamma >= float(GAMMA_GRID[-1]) * 0.999 or fit.gamma <= float(GAMMA_GRID[0]) * 1.001:
            return None  # exponent on the grid boundary: the form does not describe this curve
    else:
        denom = abs(fit.A)
    if not math.isfinite(denom) or denom <= 0 or fit.source == "pooled_shift":
        return None
    try:
        return float(math.exp(delta_loss / denom))
    except OverflowError:
        return None
