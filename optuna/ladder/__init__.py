"""Scaling-ladder discipline for a study: rung extrapolation, pre-registered thresholds,
holdout acceptance, run statuses. See ``docs/scaling-ladder.md``.

Public names: :class:`Ladder`, :class:`Decision`, and the ``Study`` methods
``register_noise``, ``accept`` (with ``holdout`` and ``near_optimal_check``).
"""

from optuna.ladder._decision import Decision
from optuna.ladder._decision import DecisionRecord
from optuna.ladder._ladder import Ladder


__all__ = ["Decision", "DecisionRecord", "Ladder"]
