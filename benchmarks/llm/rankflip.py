"""Rank-flip benchmark: do rung-extrapolation kills beat rank-based kills when curves cross?

A pool of ``N`` synthetic candidates per seed, each a loss curve ``L(C) = E + A C^-γ`` plus
seed-dependent noise at every rung, with ``E``, ``A`` and ``γ`` drawn so that the candidate
with the lowest target loss is usually a slow starter (high ``A``): it is near the bottom at the
first rung and overtakes later. Every arm sees the same pool in the same order (candidate =
``trial.number``), the same rung budgets ``1, 2, 4, 8, 16`` and the same noise draws.

Arms:
  full          every candidate at the full budget (no pruning); the ceiling given the noise
  ladder        ``optuna.ladder.Ladder`` (kills on the predicted target value with its interval)
  sha-<η>       ``SuccessiveHalvingPruner`` (ASHA): promotes the top 1/η by rank at each rung
  hb-<η>        ``HyperbandPruner`` with reduction factor η
  median        ``MedianPruner`` (rank against the median at the same rung)

Metrics per (arm, seed): whether the selected candidate (best observed target value among those
that reached the target) is the true target-scale best (noise-free), its true regret, and the
compute spent (the budget each candidate reached; rungs continue one another). ``P(select true best)`` is the mean over seeds. Because the
ladder and the rank pruners spend different compute, the table also reports the rank pruner
whose compute is closest to the ladder's (the equal-compute comparison).

Usage: python -m benchmarks.llm.rankflip --seeds 0-9 --n-candidates 24
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from benchmarks.llm.common import parse_seeds
import optuna
from optuna.ladder import Ladder
from optuna.samplers import Ledger
from optuna.trial import TrialState


optuna.logging.set_verbosity(optuna.logging.WARNING)
RESULTS = Path(os.environ.get("OPTUNAI_RESULTS", "/var/tmp/optunai-ladder/results")) / "rankflip"
RUNGS = [1.0, 2.0, 4.0, 8.0, 16.0]
NOISE_SD = 0.01


def make_pool(seed: int, n: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    E = rng.uniform(0.20, 0.45, n)
    A = rng.uniform(0.2, 2.5, n)
    g = rng.uniform(0.5, 1.3, n)
    noise = rng.normal(0.0, NOISE_SD, (n, len(RUNGS)))
    clean = np.array([[E[i] + A[i] * c ** (-g[i]) for c in RUNGS] for i in range(n)])
    target = clean[:, -1]
    first = clean[:, 0]
    true_best = int(np.argmin(target))
    rank_first = int(np.argsort(np.argsort(first))[true_best])  # 0 = best at the first rung
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    flips = sum(((first[i] - first[j]) * (target[i] - target[j])) < 0 for i, j in pairs)
    return {
        "seed": seed,
        "n": n,
        "E": E.tolist(),
        "A": A.tolist(),
        "gamma": g.tolist(),
        "noise": noise.tolist(),
        "clean": clean.tolist(),
        "true_best": true_best,
        "true_best_rank_at_first_rung": rank_first,
        "pair_flip_rate": flips / len(pairs),
    }


def observed(pool: dict[str, Any], i: int, rung: int) -> float:
    return float(pool["clean"][i][rung] + pool["noise"][i][rung])


def run_arm(arm: str, pool: dict[str, Any], ledger_dir: Path) -> dict[str, Any]:
    n = pool["n"]
    seed = pool["seed"]
    pruner: optuna.pruners.BasePruner
    ladder: Ladder | None = None
    if arm == "full":
        pruner = optuna.pruners.NopPruner()
    elif arm == "ladder":
        ladder = Ladder(
            rungs=RUNGS,
            n_startup=4,
            ledger=Ledger(ledger_dir / f"rankflip_seed{seed}.ledger.jsonl"),
            fallback_sd=NOISE_SD,
        )
        pruner = ladder
    elif arm.startswith("sha-"):
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=int(arm.split("-")[1]), min_early_stopping_rate=0
        )
    elif arm.startswith("hb-"):
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=int(RUNGS[-1]), reduction_factor=int(arm.split("-")[1])
        )
    elif arm == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0)
    else:
        raise ValueError(arm)
    study = optuna.create_study(
        study_name=f"rankflip/{arm}/seed{seed}",
        direction="minimize",
        pruner=pruner,
        sampler=optuna.samplers.RandomSampler(seed=seed),
    )

    def objective(trial: optuna.Trial) -> float:
        i = trial.number % n
        trial.suggest_int("candidate", 0, n - 1)  # a nominal dimension; the pool fixes the curve
        v = 0.0
        if ladder is not None:
            for rung in ladder.rungs(trial):
                v = observed(pool, i, rung.index)
                rung.report(v)
            return v
        for r, budget in enumerate(RUNGS):
            v = observed(pool, i, r)
            trial.set_user_attr(
                "ladder:compute", budget
            )  # rungs continue: compute = budget reached
            if r < len(RUNGS) - 1:
                trial.report(v, step=int(budget))
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return v

    t0 = time.time()
    study.optimize(ladder.wrap(objective) if ladder is not None else objective, n_trials=n)
    wall = time.time() - t0
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    selected = min(completed, key=lambda t: float(t.value)).number % n if completed else None  # type: ignore[arg-type]
    target = [row[-1] for row in pool["clean"]]
    regret = (target[selected] - target[pool["true_best"]]) if selected is not None else None
    compute = sum(float(t.user_attrs.get("ladder:compute") or 0.0) for t in study.trials)
    rec: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "n": n,
        "selected": selected,
        "true_best": pool["true_best"],
        "hit": selected == pool["true_best"],
        "regret": regret,
        "compute": compute,
        "compute_full": n * RUNGS[-1],
        "n_completed": len(completed),
        "n_pruned": sum(t.state == TrialState.PRUNED for t in study.trials),
        "true_best_completed": any(t.number % n == pool["true_best"] for t in completed),
        "true_best_rank_at_first_rung": pool["true_best_rank_at_first_rung"],
        "pair_flip_rate": pool["pair_flip_rate"],
        "wall_s": wall,
    }
    if ladder is not None:
        rec["calibration"] = ladder.calibration(study)
        rec["fit"] = ladder.fit_report(study)
        tb = [t for t in study.trials if t.number % n == pool["true_best"]]
        rec["true_best_decisions"] = (
            [
                {
                    "rung": p["rung"],
                    "decision": p["decision"],
                    "difference": p["difference"],
                    "interval": p["interval"],
                }
                for p in (tb[0].user_attrs.get("ladder:predicted") or [])
            ]
            if tb
            else []
        )
    return rec


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted({r["arm"] for r in records}, key=lambda a: (a != "full", a != "ladder", a))
    out: dict[str, Any] = {}
    for arm in arms:
        rs = [r for r in records if r["arm"] == arm]
        regrets = [r["regret"] for r in rs if r["regret"] is not None]
        out[arm] = {
            "n_seeds": len(rs),
            "p_select_true_best": float(np.mean([r["hit"] for r in rs])),
            "p_true_best_reached_target": float(np.mean([r["true_best_completed"] for r in rs])),
            "regret_median": float(np.median(regrets)) if regrets else None,
            "regret_iqr": [float(np.percentile(regrets, 25)), float(np.percentile(regrets, 75))]
            if regrets
            else None,
            "compute_median": float(np.median([r["compute"] for r in rs])),
            "compute_fraction_of_full": float(
                np.median([r["compute"] / r["compute_full"] for r in rs])
            ),
            "n_completed_median": float(np.median([r["n_completed"] for r in rs])),
        }
        if arm == "ladder":
            cal = [r["calibration"] for r in rs if r.get("calibration")]
            known = [c for c in cal if c.get("coverage") is not None]
            out[arm]["interval_coverage"] = (
                float(np.mean([c["coverage"] for c in known])) if known else None
            )
            out[arm]["n_frozen"] = int(sum(c["n_frozen"] for c in cal))
    if "ladder" in out:
        lc = out["ladder"]["compute_median"]
        rank_arms = [a for a in out if a not in ("full", "ladder")]
        if rank_arms:
            nearest = min(rank_arms, key=lambda a: abs(out[a]["compute_median"] - lc))
            out["equal_compute_comparison"] = {
                "ladder": {
                    k: out["ladder"][k]
                    for k in ("p_select_true_best", "compute_median", "regret_median")
                },
                "nearest_rank_arm": nearest,
                nearest: {
                    k: out[nearest][k]
                    for k in ("p_select_true_best", "compute_median", "regret_median")
                },
            }
    out["pool"] = {
        "true_best_rank_at_first_rung_median": float(
            np.median([r["true_best_rank_at_first_rung"] for r in records])
        ),
        "pair_flip_rate_median": float(np.median([r["pair_flip_rate"] for r in records])),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-candidates", type=int, default=24)
    ap.add_argument("--arms", default="full,ladder,sha-2,sha-3,sha-4,hb-2,hb-3,hb-4,median")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in parse_seeds(args.seeds):
        pool = make_pool(seed, args.n_candidates)
        for arm in args.arms.split(","):
            out = RESULTS / f"{arm}__seed{seed}.json"
            if out.exists() and not args.force:
                rec = json.loads(out.read_text())
            else:
                rec = run_arm(arm, pool, RESULTS)
                out.write_text(json.dumps(rec))
            records.append(rec)
            print(
                f"{arm:8s} seed{seed}: hit={rec['hit']} regret={rec['regret']:.4f} compute={rec['compute']:.0f}"
                f"/{rec['compute_full']} completed={rec['n_completed']} true_best_rank0={rec['true_best_rank_at_first_rung']}",
                flush=True,
            )
    summary = summarise(records)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
