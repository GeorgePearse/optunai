"""Few-shot MLP head with the scaling-ladder discipline: rungs, noise floor, acceptance.

Same objective, cases, search space and context as ``benchmarks/llm/fewshot.py`` (see its
docstring), with three changes:

1. **Rungs.** Training steps are the fidelity: every head is trained to 25 %, 50 % and 100 % of
   its step budget and scored at each rung without restarting (a warm-started ladder, so the
   rungs of one trial share a prefix and are one bootstrap unit). A killed trial stops its
   training at the kill; its compute is the fraction of steps it ran.
2. **Seeds.** The head seed and the data permutation come from ``trial.user_attrs["ladder:seed"]``
   (0 for ordinary trials), so ``register_noise`` and ``accept`` can run fresh seeds.
3. **Decisions.** Every study registers the noise floor on a fixed incumbent, registers the
   held-out cases as the holdout, and after search calls ``accept`` (fresh-seed re-evaluation
   and paired holdout comparison against the incumbent) and ``near_optimal_check``.

Arms (``--arms``):
  tpe          TPESampler, 40 full trials; noise/threshold registered (TPE ignores it)
  llm          LLMSampler (Gemini, context), 40 full trials, no noise registered: no threshold
               in the prompt. The "LLMSampler alone" arm and the no-threshold half of the ablation.
  llm-noise    LLMSampler with the frozen threshold in the prompt (gate annotates only), 40 trials
  llm-ladder   LLMSampler + Ladder (rungs 0.25/0.5/1.0, Jev gate 0.25), same compute budget of
               40 full-trial units; stops when the ladder's compute reaches it (at most 70 trials)

Usage: python -m benchmarks.llm.fewshot_ladder --arms tpe,llm-ladder --seeds 0-2
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from benchmarks.llm import fewshot
from benchmarks.llm.common import load_env
from benchmarks.llm.common import MODELS
from benchmarks.llm.common import parse_seeds
import optuna
from optuna.ladder import Ladder
from optuna.samplers import Ledger
from optuna.samplers import LLMSampler
from optuna.trial import TrialState


RESULTS = (
    Path(os.environ.get("OPTUNAI_RESULTS", "/var/tmp/optunai-ladder/results")) / "fewshot_ladder"
)
RUNGS = [0.25, 0.5, 1.0]
BUDGET_UNITS = float(os.environ.get("OPTUNAI_BUDGET_UNITS", "40"))
MAX_TRIALS = int(os.environ.get("OPTUNAI_MAX_TRIALS", "70"))
NOISE_SEEDS = int(os.environ.get("OPTUNAI_NOISE_SEEDS", "3"))
fewshot.CACHE = Path(
    os.environ.get("OPTUNAI_FEWSHOT_CACHE", "/var/tmp/optunai-ladder/fewshot-cache")
)
INCUMBENT = dict(
    space="concat",
    cradio_weight=1.0,
    centre=True,
    hidden=256,
    activation="gelu",
    dropout=0.2,
    lr=2e-3,
    weight_decay=1e-4,
    epochs=10,
    min_steps=200,
    batch=64,
    cw=0.5,
    label_smoothing=0.1,
)


class HeadTrainer:
    """One MLP head on one case, advanced in step fractions; the state persists across rungs."""

    def __init__(self, case: dict[str, Any], params: dict[str, Any], seed: int) -> None:
        import torch

        torch.set_num_threads(int(os.environ.get("OPTUNAI_TORCH_THREADS", "1")))
        self.case = case
        self.params = params
        x_train, x_query = fewshot.features(case, params)
        labels = case["train_labels"]
        self.classes = sorted(set(labels.tolist()))
        self.index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([self.index[c] for c in labels])
        self.x_query = x_query
        self.x_train = x_train
        self.y = y
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        d, hidden = x_train.shape[1], int(params["hidden"])
        n_classes = len(self.classes)
        act = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU, "silu": torch.nn.SiLU}[
            params["activation"]
        ]
        self.model = (
            torch.nn.Linear(d, n_classes)
            if hidden == 0
            else torch.nn.Sequential(
                torch.nn.Dropout(params["dropout"]),
                torch.nn.Linear(d, hidden),
                act(),
                torch.nn.Dropout(params["dropout"]),
                torch.nn.Linear(hidden, n_classes),
            )
        )
        n = len(x_train)
        self.n = n
        self.batch = min(int(params["batch"]), n)
        self.steps = max(int(params["epochs"]) * -(-n // self.batch), int(params["min_steps"]))
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=params["lr"], total_steps=self.steps
        )
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        w = (counts.max() / np.maximum(counts, 1)) ** params["cw"]
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32), label_smoothing=params["label_smoothing"]
        )
        self.xt = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
        self.yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.noise = float(params.get("input_noise", 0.0))
        self.mixup = float(params.get("mixup_alpha", 0.0))
        self.seed = seed
        self.done = 0
        self.perm = torch.randperm(n, generator=self.gen)
        self.pos = 0

    def advance(self, fraction: float) -> None:
        import torch

        target = min(self.steps, int(math.ceil(fraction * self.steps)))
        self.model.train()
        while self.done < target:
            if self.pos >= self.n:
                self.perm = torch.randperm(self.n, generator=self.gen)
                self.pos = 0
            b = self.perm[self.pos : self.pos + self.batch]
            self.pos += self.batch
            xb, yb = self.xt[b], self.yt[b]
            if self.noise > 0:
                xb = xb + self.noise * torch.randn(xb.shape, generator=self.gen)
            self.opt.zero_grad()
            if self.mixup > 0 and len(b) > 1:
                lam = float(
                    np.random.default_rng(self.seed + self.done).beta(self.mixup, self.mixup)
                )
                idx = torch.randperm(len(b), generator=self.gen)
                logits = self.model(lam * xb + (1 - lam) * xb[idx])
                loss = lam * self.loss_fn(logits, yb) + (1 - lam) * self.loss_fn(logits, yb[idx])
            else:
                loss = self.loss_fn(self.model(xb), yb)
            loss.backward()
            self.opt.step()
            self.sched.step()
            self.done += 1

    def probs(self) -> np.ndarray:
        import torch

        temperature = float(self.params.get("temperature", 1.0))
        self.model.eval()
        with torch.no_grad():
            logits = (
                self.model(torch.from_numpy(np.asarray(self.x_query, dtype=np.float32)))
                / temperature
            )
            out = torch.softmax(logits, dim=1).numpy()
        self.model.train()
        return out


def score_from_probs(case: dict[str, Any], trainer: HeadTrainer, probs: np.ndarray) -> float:
    mix = float(trainer.params.get("prototype_mix", 0.0))
    if mix > 0:
        x_train, y = trainer.x_train, trainer.y
        protos = np.stack([x_train[y == i].mean(0) for i in range(len(trainer.classes))])
        cos = fewshot.unit_rows(trainer.x_query) @ fewshot.unit_rows(protos).T
        proto_probs = np.exp(cos * 20)
        proto_probs /= proto_probs.sum(1, keepdims=True)
        probs = (1 - mix) * probs + mix * proto_probs
    aps = []
    offsets = case["query_offsets"]
    for j, cls in enumerate(case["query_classes"].tolist()):
        lo, hi = int(offsets[j]), int(offsets[j + 1])
        positive = case["query_positive"][lo:hi]
        if cls in trainer.index:
            aps.append(fewshot.average_precision(probs[lo:hi, trainer.index[cls]], positive))
        else:
            aps.append(float(positive.mean()))
    return float(np.mean(aps))


def evaluate_rungs(
    params: dict[str, Any], cases: list[tuple[str, int, int]], seed: int, fractions: list[float]
) -> list[dict[str, float]]:
    """Per-case scores at each fraction of the step budget; heads are advanced, not restarted."""
    ensemble = max(1, int(params.get("ensemble", 1)))
    trainers: dict[str, list[HeadTrainer]] = {}
    for d, s, n in cases:
        case = fewshot.load_case(d, s, n)
        trainers[f"{d}/s{s}/n{n}"] = [
            HeadTrainer(case, params, seed + 17 * k) for k in range(ensemble)
        ]
    out: list[dict[str, float]] = []
    for frac in fractions:
        per_case: dict[str, float] = {}
        for key, heads in trainers.items():
            probs = np.zeros_like(heads[0].probs())
            for h in heads:
                h.advance(frac)
                probs += h.probs()
            probs /= len(heads)
            per_case[key] = round(score_from_probs(heads[0].case, heads[0], probs), 4)
        out.append(per_case)
    return out


def full_params(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    params = fewshot.suggest(trial)  # type: ignore[arg-type]
    for name in ("prototype_mix", "input_noise", "ensemble", "mixup_alpha", "temperature"):
        if name in trial.params:
            params[name] = trial.params[name]
    return params


def make_objective(ladder: Ladder | None) -> Any:
    def objective(trial: optuna.trial.BaseTrial) -> float:
        params = full_params(trial)
        seed = int(trial.user_attrs.get("ladder:seed", 0))
        t0 = time.time()
        if ladder is None or not isinstance(trial, optuna.Trial):
            per_case = evaluate_rungs(params, fewshot.SEEN_CASES, seed, [1.0])[-1]
            trial.set_user_attr("per_case", per_case)
            trial.set_user_attr("seconds", round(time.time() - t0, 1))
            return float(np.mean(list(per_case.values())))
        # Rung-by-rung, with the kill inside the training loop so compute is really saved.
        ensemble = max(1, int(params.get("ensemble", 1)))
        trainers: dict[str, list[HeadTrainer]] = {}
        for d, s, n in fewshot.SEEN_CASES:
            case = fewshot.load_case(d, s, n)
            trainers[f"{d}/s{s}/n{n}"] = [
                HeadTrainer(case, params, seed + 17 * k) for k in range(ensemble)
            ]
        value = 0.0
        for rung in ladder.rungs(trial):
            per_case = {}
            for key, heads in trainers.items():
                probs = np.zeros_like(heads[0].probs())
                for h in heads:
                    h.advance(rung.budget)
                    probs += h.probs()
                probs /= len(heads)
                per_case[key] = round(score_from_probs(heads[0].case, heads[0], probs), 4)
            value = float(np.mean(list(per_case.values())))
            trial.set_user_attr("per_case", per_case)
            trial.set_user_attr("seconds", round(time.time() - t0, 1))
            rung.report(value)
        return value

    return objective


def holdout_objective(trial: optuna.trial.BaseTrial) -> float:
    params = full_params(trial)
    seed = int(trial.user_attrs.get("ladder:seed", 0))
    per_case = evaluate_rungs(params, fewshot.HELDOUT_CASES, seed, [1.0])[-1]
    trial.set_user_attr("per_case", per_case)
    return float(np.mean(list(per_case.values())))


def make_sampler(
    arm: str, seed: int, context: list[str], ledger: Ledger
) -> optuna.samplers.BaseSampler:
    if arm == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    model = MODELS[os.environ.get("OPTUNAI_MODEL", "gemini")]()
    return LLMSampler(
        model,
        context=context,
        allow_new_params=True,
        seed=seed,
        ledger=ledger,
        gate=0.25 if arm == "llm-ladder" else 0.0,
    )


def trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    rows = []
    for t in study.trials:
        a, u = t.system_attrs, t.user_attrs
        rows.append(
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
                "source": a.get("llm:source"),
                "usd": float(a.get("llm:usd") or 0.0),
                "hypothesis": a.get("llm:hypothesis"),
                "expected_effect": a.get("llm:expected_effect"),
                "predicted_target_value": a.get("llm:predicted_target_value"),
                "llm_interval": a.get("llm:interval"),
                "llm_decision": a.get("llm:decision"),
                "gate_p": a.get("llm:gate_p"),
                "new_params": a.get("llm:new_params"),
                "status_reason": u.get("ladder:status_reason"),
                "stop_reason": u.get("ladder:stop_reason"),
                "rungs": u.get("ladder:rungs"),
                "compute": u.get(
                    "ladder:compute", 1.0 if t.state == TrialState.COMPLETE else None
                ),
                "predicted": u.get("ladder:predicted"),
                "inside_interval": u.get("ladder:inside_interval"),
                "per_case": u.get("per_case"),
                "seconds": u.get("seconds"),
            }
        )
    return rows


def run_study(arm: str, seed: int, context: list[str], force: bool) -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"mlp_head__{arm}__seed{seed}.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    ledger_path = out.with_suffix(".ledger.jsonl")
    if ledger_path.exists():
        ledger_path.unlink()
    ledger = Ledger(ledger_path)
    ladder = Ladder(rungs=RUNGS, n_startup=4, ledger=ledger) if arm == "llm-ladder" else None
    study = optuna.create_study(
        study_name=f"fewshot_ladder/mlp_head/{arm}/seed{seed}",
        direction="maximize",
        sampler=make_sampler(arm, seed, context, ledger),
        pruner=ladder if ladder is not None else optuna.pruners.NopPruner(),
    )
    objective = make_objective(ladder)
    t0 = time.time()
    noise = None
    if arm != "llm":
        noise = study.register_noise(objective, n_seeds=NOISE_SEEDS, params=INCUMBENT)
        print(
            f"  noise floor sd={noise['seed_sd']:.4f} threshold={noise['threshold']:.4f}",
            flush=True,
        )
    study.holdout(holdout_objective, name="heldout cases, world seed 2")

    def stop_at_budget(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
        if ladder is not None and ladder.compute(st) >= BUDGET_UNITS:
            st.stop()

    n_trials = MAX_TRIALS if ladder is not None else int(BUDGET_UNITS)
    func = ladder.wrap(objective) if ladder is not None else objective
    study.optimize(func, n_trials=n_trials, callbacks=[stop_at_budget], catch=(Exception,))
    search_wall = time.time() - t0
    # Acceptance and the near-optimal test need a threshold; the no-noise arm borrows the seed SD
    # measured by its sibling arms (same objective) through min_effect, recorded as such.
    accept_kwargs: dict[str, Any] = {"n_seeds": 2, "max_rounds": 1}
    if arm == "llm":
        sibling = RESULTS / f"mlp_head__tpe__seed{seed}.json"
        borrowed = (
            json.loads(sibling.read_text())["noise"]["threshold"] if sibling.exists() else 0.0
        )
        accept_kwargs["min_effect"] = borrowed
    record_accept = study.accept(objective, **accept_kwargs)
    check = study.near_optimal_check(
        objective, n_perturbations=2, tolerance=record_accept.threshold
    )
    attrs = study._storage.get_study_system_attrs(study._study_id)
    rows = trial_rows(study)
    rec: dict[str, Any] = {
        "bench": "fewshot_ladder",
        "problem": "mlp_head",
        "arm": arm,
        "seed": seed,
        "model": getattr(getattr(study.sampler, "model", None), "name", None),
        "n_trials": len(rows),
        "search_wall_s": search_wall,
        "wall_s": time.time() - t0,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "noise": noise,
        "threshold": record_accept.threshold,
        "acceptance": record_accept.as_dict(),
        "acceptance_rounds": attrs.get("ladder:acceptance"),
        "near_optimal": check,
        "trials": rows,
        "usd": sum(r["usd"] for r in rows),
        "compute_units": sum(float(r["compute"] or 0.0) for r in rows),
        "n_llm": sum(r["source"] in ("llm", "repair", "batch") for r in rows),
        "n_fallback": sum(r["source"] == "fallback" for r in rows),
        "n_pruned": sum(r["state"] == "PRUNED" for r in rows),
        "n_deferred": sum(r["stop_reason"] == "deferred_by_gate" for r in rows),
        "status_counts": {
            k: sum(r["status_reason"] == k for r in rows)
            for k in (
                "completed",
                "actively_stopped",
                "algorithmic_divergence",
                "infrastructure_failure",
                "implementation_error",
            )
        },
        "ledger_totals": ledger.totals(),
    }
    if ladder is not None:
        rec["ladder_fit"] = ladder.fit_report(study)
        rec["calibration"] = ladder.calibration(study)
    out.write_text(json.dumps(rec, default=str))
    return rec


def main() -> None:
    load_env()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="tpe,llm-ladder")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--context-dir", default="/var/tmp/optunai/context")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    context = fewshot.build_context(Path(args.context_dir))
    for arm in args.arms.split(","):
        for seed in parse_seeds(args.seeds):
            print(f"== {arm} seed{seed}", flush=True)
            rec = run_study(arm, seed, context, args.force)
            acc = rec["acceptance"]
            print(
                f"{arm} seed{seed}: best_of_n={rec['best_value']:.4f} reevaluated={acc['reevaluated_value']:.4f} "
                f"holdout={acc['holdout_value']:.4f} incumbent_holdout={acc['incumbent_holdout_value']:.4f} "
                f"decision={acc['decision']} diff={acc['difference']:+.4f} thr={acc['threshold']:.4f} "
                f"near_optimal={rec['near_optimal']['near_optimal']} boundary={rec['near_optimal']['on_boundary']} "
                f"usd={rec['usd']:.2f} trials={rec['n_trials']} pruned={rec['n_pruned']} deferred={rec['n_deferred']} "
                f"compute={rec['compute_units']:.1f} wall={rec['wall_s']:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
