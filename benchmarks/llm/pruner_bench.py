"""Pruner benchmark: HistGradientBoosting learning curves on the sklearn datasets.

Each trial grows a HistGradientBoostingClassifier with ``warm_start`` in chunks of 10
iterations up to ``max_iter``, reporting the validation macro F1 after every chunk, so a trial
has up to 15 intermediate steps. Pruners compared with the same TPE sampler and seeds:
``none``, ``median`` (MedianPruner) and ``llm`` (LLMPruner, Jev judge with the median floor).

Recorded per run: final best value, total boosting steps spent, pruned count, and the
false-prune rate: every pruned trial is re-run to completion afterwards and counted as a false
prune when its full value would have beaten the study's final best.

Usage: python -m benchmarks.llm.pruner_bench --pruners none,median,llm --seeds 0-4 --n-trials 30
"""

from __future__ import annotations

import argparse
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from benchmarks.llm.common import load_env
from benchmarks.llm.common import parse_seeds
from benchmarks.llm.common import RESULTS_ROOT
from benchmarks.llm.common import run
from benchmarks.llm.common import summarise
from benchmarks.llm.sklearn_bench import dataset
import optuna


CHUNK = 10


def make_problem(name: str, seed: int) -> tuple[Any, Any]:
    x, y = dataset(name)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)

    def train(params: dict[str, Any], trial: optuna.Trial | None) -> tuple[float, int]:
        clf = HistGradientBoostingClassifier(
            random_state=seed,
            warm_start=True,
            early_stopping=False,
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            max_iter=CHUNK,
        )
        steps = 0
        score = 0.0
        for step, iters in enumerate(range(CHUNK, params["max_iter"] + 1, CHUNK)):
            clf.set_params(max_iter=iters)
            clf.fit(x_tr, y_tr)
            steps = iters
            score = float(f1_score(y_te, clf.predict(x_te), average="macro"))
            if trial is not None:
                trial.report(score, step)
                trial.set_user_attr("steps", steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return score, steps

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            max_iter=trial.suggest_int("max_iter", 40, 150, step=CHUNK),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 4, 64),
            max_depth=trial.suggest_int("max_depth", 2, 16),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 100),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        )
        score, _ = train(params, trial)
        return score

    return objective, train


def make_pruner(kind: str) -> optuna.pruners.BasePruner:
    if kind == "none":
        return optuna.pruners.NopPruner()
    if kind == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    if kind == "llm":
        return optuna.pruners.LLMPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=3)
    if kind == "llm-aggressive":
        return optuna.pruners.LLMPruner(
            n_startup_trials=5, n_warmup_steps=3, interval_steps=3, aggressive=True
        )
    raise KeyError(kind)


def main() -> None:
    load_env()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruners", default="none,median,llm")
    ap.add_argument("--datasets", default="digits,breast_cancer_noisy")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--n-trials", type=int, default=30)
    args = ap.parse_args()
    for name in args.datasets.split(","):
        for kind in args.pruners.split(","):
            for seed in parse_seeds(args.seeds):
                objective, train = make_problem(name, seed)
                pruner = make_pruner(kind)

                def extra(
                    study: optuna.Study, train: Any = train, pruner: Any = pruner
                ) -> dict[str, Any]:
                    best = study.best_value
                    false_prunes = 0
                    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
                    full_values = {}
                    for t in pruned:
                        value, _ = train(t.params, None)
                        full_values[t.number] = value
                        if value > best:
                            false_prunes += 1
                    steps = sum(int(t.user_attrs.get("steps", 0)) for t in study.trials)
                    return {
                        "pruner": kind,
                        "total_steps": steps,
                        "n_pruned": len(pruned),
                        "false_prunes": false_prunes,
                        "false_prune_rate": (false_prunes / len(pruned)) if pruned else 0.0,
                        "pruned_full_values": full_values,
                        "judge_usd": getattr(pruner, "usd", 0.0),
                        "judge_calls": getattr(pruner, "calls", 0),
                    }

                rec = run(
                    bench="pruner",
                    problem=name,
                    arm=f"tpe+{kind}",
                    seed=seed,
                    n_trials=args.n_trials,
                    objective=objective,
                    direction="maximize",
                    pruner=pruner,
                    extra_record=extra,
                )
                print(
                    summarise(rec),
                    f"steps={rec['total_steps']} pruned={rec['n_pruned']} false={rec['false_prunes']}",
                    flush=True,
                )
    print("results in", RESULTS_ROOT / "pruner")


if __name__ == "__main__":
    main()
