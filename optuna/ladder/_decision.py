"""The formal decision enum and record every ladder comparison writes."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import datetime
import enum
import math
from typing import Any


class Decision(str, enum.Enum):
    """Outcome of a comparison against a pre-registered threshold (post §2.3).

    ``select``: the difference exceeds the threshold and its interval excludes zero.
    ``run_more``: the interval straddles the threshold and another round is allowed.
    ``insufficient_evidence``: the interval straddles the threshold and no round is left; a
    formal, terminal outcome, never a silent pick.
    ``defer``: even the optimistic end of the interval does not reach the threshold; keep the
    incumbent.
    """

    SELECT = "select"
    RUN_MORE = "run_more"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DEFER = "defer"


def decide(
    difference: float, lo: float, hi: float, threshold: float, *, can_run_more: bool
) -> Decision:
    """Apply the rule. ``difference`` and the interval are in improvement units (positive =
    candidate better); ``lo <= difference <= hi``."""
    if math.isnan(difference):
        return Decision.INSUFFICIENT_EVIDENCE
    if difference >= threshold and lo > 0:
        return Decision.SELECT
    if hi < threshold:
        return Decision.DEFER
    return Decision.RUN_MORE if can_run_more else Decision.INSUFFICIENT_EVIDENCE


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class DecisionRecord:
    """What a comparison reported: the difference, its paired interval, the threshold it was
    tested against, the equivalent compute multiplier when a fitted ladder exists, and the
    numbers behind it. ``kind`` is ``acceptance`` (from :meth:`Study.accept`), ``rung`` (a
    ladder promotion or kill) or ``final`` (a candidate finishing its last rung)."""

    decision: Decision
    kind: str
    difference: float
    interval: tuple[float, float]
    threshold: float
    equivalent_compute_multiplier: float | None = None
    trial_number: int | None = None
    incumbent_trial_number: int | None = None
    best_of_n_value: float | None = None
    reevaluated_value: float | None = None
    holdout_value: float | None = None
    incumbent_holdout_value: float | None = None
    selection_bias_gap: float | None = None
    predicted_target_value: float | None = None
    rung: int | None = None
    seeds: list[int] = field(default_factory=list)
    interval_method: str | None = None
    reason: str = ""
    at: str = field(default_factory=now_iso)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        d["interval"] = [self.interval[0], self.interval[1]]
        return d

    def __str__(self) -> str:
        lo, hi = self.interval
        mult = (
            f", worth {self.equivalent_compute_multiplier:.2f}x compute"
            if self.equivalent_compute_multiplier is not None
            else ""
        )
        return (
            f"{self.decision.value}: difference {self.difference:+.4g} "
            f"[{lo:.4g}, {hi:.4g}] against threshold {self.threshold:.4g}{mult}"
            + (f" ({self.reason})" if self.reason else "")
        )
