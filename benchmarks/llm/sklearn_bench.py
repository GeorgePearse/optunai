"""sklearn benchmark: HistGradientBoosting on two OpenML-style datasets, 7-dim mixed space.

Datasets are the ones scikit-learn ships offline (no download): ``digits`` (1797 x 64, 10
classes) and ``covtype`` subsample is not offline, so the second is ``breast_cancer`` reshaped
into a harder problem by using 3-fold CV on 20% label noise. Score = mean 3-fold macro F1,
maximised. Each trial reports a per-fold intermediate value so pruners can act on it.

Usage: python -m benchmarks.llm.sklearn_bench --arms random,tpe,llm:claude --seeds 0-4 --n-trials 30
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from benchmarks.llm.common import load_env
from benchmarks.llm.common import parse_seeds
from benchmarks.llm.common import run
from benchmarks.llm.common import summarise
import optuna


def dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "digits":
        d = load_digits()
        return d.data, d.target
    if name == "breast_cancer_noisy":
        d = load_breast_cancer()
        rng = np.random.default_rng(0)
        y = d.target.copy()
        flip = rng.random(len(y)) < 0.2
        y[flip] = 1 - y[flip]
        return d.data, y
    raise KeyError(name)


SPACE_DOC = """
Search space (HistGradientBoostingClassifier):
  learning_rate: float, log, [1e-3, 1.0]
  max_iter: int, [20, 400]
  max_leaf_nodes: int, [4, 128]
  max_depth: int, [2, 16]
  min_samples_leaf: int, [2, 100]
  l2_regularization: float, log, [1e-6, 10.0]
  max_bins: categorical, [32, 64, 128, 255]
Score: mean macro F1 over 3 stratified folds (maximise). Each fold's running mean is
reported as an intermediate value at steps 0, 1, 2.
"""


def make_objective(name: str, seed: int) -> Any:
    x, y = dataset(name)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(x, y))

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            max_iter=trial.suggest_int("max_iter", 20, 400),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 4, 128),
            max_depth=trial.suggest_int("max_depth", 2, 16),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 100),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            max_bins=trial.suggest_categorical("max_bins", [32, 64, 128, 255]),
        )
        scores = []
        for step, (tr, te) in enumerate(folds):
            clf = HistGradientBoostingClassifier(random_state=seed, **params)
            clf.fit(x[tr], y[tr])
            scores.append(f1_score(y[te], clf.predict(x[te]), average="macro"))
            trial.report(float(np.mean(scores)), step)
            trial.set_user_attr(f"fold{step}_f1", round(scores[-1], 4))
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def context_for(name: str) -> str:
    x, y = dataset(name)
    classes, counts = np.unique(y, return_counts=True)
    return (
        f"Dataset {name}: {x.shape[0]} rows, {x.shape[1]} numeric features, "
        f"{len(classes)} classes with counts {counts.tolist()}."
        + (
            " Labels carry 20% uniform noise, so regularisation matters."
            if "noisy" in name
            else ""
        )
        + "\n"
        + SPACE_DOC
    )


def main() -> None:
    load_env()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="random,tpe")
    ap.add_argument("--datasets", default="digits,breast_cancer_noisy")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--n-trials", type=int, default=30)
    args = ap.parse_args()
    for name in args.datasets.split(","):
        for arm in args.arms.split(","):
            for seed in parse_seeds(args.seeds):
                rec = run(
                    bench="sklearn",
                    problem=name,
                    arm=arm,
                    seed=seed,
                    n_trials=args.n_trials,
                    objective=make_objective(name, seed),
                    direction="maximize",
                    context=context_for(name),
                )
                print(summarise(rec), flush=True)


if __name__ == "__main__":
    main()
