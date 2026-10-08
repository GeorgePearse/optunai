"""Synthetic benchmark: Ackley, Rosenbrock and a mixed int/categorical toy, minimised.

Usage: python -m benchmarks.llm.synthetic --arms random,tpe,cmaes,llm:claude --seeds 0-4 --n-trials 30
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from benchmarks.llm.common import load_env
from benchmarks.llm.common import parse_seeds
from benchmarks.llm.common import run
from benchmarks.llm.common import summarise
import optuna


def ackley(trial: optuna.Trial) -> float:
    xs = [trial.suggest_float(f"x{i}", -32.768, 32.768) for i in range(4)]
    n = len(xs)
    s1 = sum(x * x for x in xs) / n
    s2 = sum(math.cos(2 * math.pi * x) for x in xs) / n
    return -20 * math.exp(-0.2 * math.sqrt(s1)) - math.exp(s2) + 20 + math.e


def rosenbrock(trial: optuna.Trial) -> float:
    xs = [trial.suggest_float(f"x{i}", -5.0, 10.0) for i in range(4)]
    return sum(100 * (xs[i + 1] - xs[i] ** 2) ** 2 + (1 - xs[i]) ** 2 for i in range(3))


def mixed_toy(trial: optuna.Trial) -> float:
    """A toy with an int, a log float, two categoricals and an interaction.

    kernel chooses the shape; the optimum sits at kernel="cosine", n_layers=3, lr=1e-2,
    activation="gelu", width=0.5 with value 0.
    """
    lr = trial.suggest_float("lr", 1e-5, 1.0, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 8)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "cosine", "poly"])
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "tanh"])
    width = trial.suggest_float("width", 0.0, 1.0, step=0.05)
    penalty = {"linear": 3.0, "rbf": 1.0, "cosine": 0.0, "poly": 2.0}[kernel]
    act = {"relu": 0.5, "gelu": 0.0, "tanh": 1.0}[activation]
    layers = (n_layers - 3) ** 2 * (0.5 if kernel == "cosine" else 1.5)
    lr_term = (math.log10(lr) + 2) ** 2
    return penalty + act + layers + lr_term + 4 * (width - 0.5) ** 2


PROBLEMS: dict[str, tuple[Any, str]] = {
    "ackley": (ackley, "4-D Ackley function, minimised; global minimum 0 at the origin."),
    "rosenbrock": (
        rosenbrock,
        "4-D Rosenbrock function, minimised; global minimum 0 at x_i = 1. The valley is curved: "
        "follow y = x^2.",
    ),
    "mixed_toy": (mixed_toy, open(__file__).read().split("def mixed_toy")[1].split("PROBLEMS")[0]),
}


def main() -> None:
    load_env()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="random,tpe,cmaes")
    ap.add_argument("--problems", default=",".join(PROBLEMS))
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--n-trials", type=int, default=30)
    args = ap.parse_args()
    for problem in args.problems.split(","):
        objective, context = PROBLEMS[problem]
        for arm in args.arms.split(","):
            for seed in parse_seeds(args.seeds):
                rec = run(
                    bench="synthetic",
                    problem=problem,
                    arm=arm,
                    seed=seed,
                    n_trials=args.n_trials,
                    objective=objective,
                    direction="minimize",
                    context="def " + context if problem == "mixed_toy" else context,
                )
                print(summarise(rec), flush=True)


if __name__ == "__main__":
    main()
