"""Few-shot MLP head over cached crop embeddings: the in-house demo target.

Objective
---------
For each benchmark *case* ``(dataset, world seed, n labels)`` the trial trains one MLP (softmax)
head on ``n`` randomly labelled crops from a cached pool (C-RADIO v4 and DINOv3 embeddings,
``visia_data_curve`` worlds) and scores eight frozen query pools (200 candidates each, 40 of the
query class). The metric per pool is average precision; macro over the eight classes; the trial
value is the mean over the seen cases (maximise). Held-out cases (world seed 2) are scored only
for the final best trial, to measure how much the optimiser overfits the seen cases.

Each trial sets ``user_attr["per_case"]`` with the score of every seen case, so a sampler that
reads attributes can see where a configuration wins or loses.

Search space (13 dims, all read with ``trial.suggest_*``):
  space:            categorical  cradio | dinov3 | concat | wstack2  (which embedders, raw or PCA-whitened)
  cradio_weight:    float  [0.25, 4.0] log   block weight of C-RADIO relative to DINOv3 in concat/wstack2
  centre:           categorical  True | False  subtract the pool mean before unit-normalising
  hidden:           categorical  0 | 128 | 256 | 512 | 1024   (0 = linear head)
  activation:       categorical  gelu | relu | silu
  dropout:          float  [0.0, 0.6]
  lr:               float  [1e-4, 2e-2] log   AdamW peak learning rate (OneCycle)
  weight_decay:     float  [1e-6, 1e-1] log
  epochs:           int    [2, 40]
  min_steps:        int    [50, 1200] log  floor on optimiser steps (small n gives few steps per epoch)
  batch:            int    [16, 1024] log
  cw:               float  [0.0, 1.0]   class-weight exponent, weight = (max_count / count) ** cw
  label_smoothing:  float  [0.0, 0.3]

Optional parameters the objective reads with ``trial.params.get(name, default)`` when a sampler
proposes them (open-vocabulary search space):
  prototype_mix:  float [0, 1]   blend the MLP probability with a cosine-to-class-prototype softmax
  input_noise:    float [0, 0.5] gaussian noise std added to training rows (after unit norm)
  ensemble:       int   [1, 5]   average the probabilities of this many differently seeded heads
  mixup_alpha:    float [0, 1]   mixup on rows and one-hot labels
  temperature:    float [0.1, 10] softmax temperature at prediction time

Usage: python -m benchmarks.llm.fewshot --arms random,tpe,llm:claude --seeds 0-4 --n-trials 40
"""

from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from benchmarks.llm.common import load_env
from benchmarks.llm.common import parse_seeds
from benchmarks.llm.common import run
from benchmarks.llm.common import summarise
import optuna


WORLDS = Path(os.environ.get("DATA_CURVE_CACHE", Path.home() / ".cache/visia_data_curve/worlds"))
CACHE = Path("/var/tmp/optunai/fewshot-cache")
SEEN_CASES = [(d, s, n) for d in ("municipals", "vmi_ewaste") for s in (0, 1) for n in (80, 400)]
HELDOUT_CASES = [(d, 2, n) for d in ("municipals", "vmi_ewaste") for n in (80, 400)]
WHITEN_DIM = 256
WHITEN_EPS = 1e-3
SOURCES = {"cradio": 0, "dinov3": 1}


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    hits = positive[order]
    if hits.sum() == 0:
        return 0.0
    precision = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((precision * hits).sum() / hits.sum())


@functools.lru_cache(maxsize=None)
def load_case(dataset: str, seed: int, n: int) -> dict[str, Any]:
    """Labelled subsample, query pools, pool means and whiteners for one case (cached on disk)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{dataset}-seed{seed}-n{n}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    world = np.load(WORLDS / f"{dataset}-seed{seed}.npz", allow_pickle=True)
    rng = np.random.default_rng(1000 * seed + n)
    pick = rng.choice(len(world["pool_labels"]), n, replace=False)
    out: dict[str, Any] = {
        "train_labels": world["pool_labels"][pick],
        "query_classes": world["query_classes"],
        "query_offsets": world["query_offsets"],
        "query_positive": world["query_positive"],
    }
    for name, k in SOURCES.items():
        pool = unit_rows(world[f"pool_vectors_{k}"].astype(np.float32))
        mu = pool.mean(0)
        sample = pool[rng.choice(len(pool), min(len(pool), 8000), replace=False)] - mu
        _, s, vt = np.linalg.svd(sample, full_matrices=False)
        var = (s[:WHITEN_DIM] ** 2) / len(sample)
        proj = (vt[:WHITEN_DIM].T / np.sqrt(var + WHITEN_EPS * var.mean())).astype(np.float32)
        out[f"train_{name}"] = pool[pick]
        out[f"query_{name}"] = unit_rows(world[f"query_vectors_{k}"].astype(np.float32))
        out[f"mu_{name}"] = mu
        out[f"proj_{name}"] = proj
    np.savez(path, **out)
    return out


def features(case: dict[str, Any], params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    space = params["space"]
    names = {
        "cradio": ["cradio"],
        "dinov3": ["dinov3"],
        "concat": ["cradio", "dinov3"],
        "wstack2": ["cradio", "dinov3"],
    }[space]
    blocks_train, blocks_query = [], []
    for name in names:
        tr, q = case[f"train_{name}"], case[f"query_{name}"]
        if space == "wstack2":
            tr = unit_rows((tr - case[f"mu_{name}"]) @ case[f"proj_{name}"])
            q = unit_rows((q - case[f"mu_{name}"]) @ case[f"proj_{name}"])
        elif params["centre"]:
            tr = unit_rows(tr - case[f"mu_{name}"])
            q = unit_rows(q - case[f"mu_{name}"])
        w = float(params["cradio_weight"]) if (name == "cradio" and len(names) == 2) else 1.0
        blocks_train.append(tr * w)
        blocks_query.append(q * w)
    return np.concatenate(blocks_train, axis=1), np.concatenate(blocks_query, axis=1)


def train_head(
    x: np.ndarray, y: np.ndarray, n_classes: int, params: dict[str, Any], seed: int
) -> Any:
    import torch

    torch.set_num_threads(int(os.environ.get("OPTUNAI_TORCH_THREADS", "4")))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    d, hidden = x.shape[1], int(params["hidden"])
    act = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU, "silu": torch.nn.SiLU}[
        params["activation"]
    ]
    model = (
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
    n = len(x)
    batch = min(int(params["batch"]), n)
    steps = max(int(params["epochs"]) * -(-n // batch), int(params["min_steps"]))
    opt = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=params["lr"], total_steps=steps)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = (counts.max() / np.maximum(counts, 1)) ** params["cw"]
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32), label_smoothing=params["label_smoothing"]
    )
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    noise = float(params.get("input_noise", 0.0))
    mixup = float(params.get("mixup_alpha", 0.0))
    model.train()
    done = 0
    while done < steps:
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            if done >= steps:
                break
            b = perm[s : s + batch]
            xb, yb = xt[b], yt[b]
            if noise > 0:
                xb = xb + noise * torch.randn(xb.shape, generator=gen)
            opt.zero_grad()
            if mixup > 0 and len(b) > 1:
                lam = float(np.random.default_rng(seed + done).beta(mixup, mixup))
                idx = torch.randperm(len(b), generator=gen)
                logits = model(lam * xb + (1 - lam) * xb[idx])
                loss = lam * loss_fn(logits, yb) + (1 - lam) * loss_fn(logits, yb[idx])
            else:
                loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            done += 1
    model.eval()
    temperature = float(params.get("temperature", 1.0))

    def predict(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.from_numpy(np.asarray(z, dtype=np.float32))) / temperature
            return torch.softmax(logits, dim=1).numpy()

    return predict


def score_case(case: dict[str, Any], params: dict[str, Any], seed: int) -> float:
    x_train, x_query = features(case, params)
    labels = case["train_labels"]
    classes = sorted(set(labels.tolist()))
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[c] for c in labels])
    ensemble = max(1, int(params.get("ensemble", 1)))
    probs = np.zeros((len(x_query), len(classes)), dtype=np.float64)
    for k in range(ensemble):
        probs += train_head(x_train, y, len(classes), params, seed + 17 * k)(x_query)
    probs /= ensemble
    mix = float(params.get("prototype_mix", 0.0))
    if mix > 0:
        protos = np.stack([x_train[y == i].mean(0) for i in range(len(classes))])
        cos = unit_rows(x_query) @ unit_rows(protos).T
        proto_probs = np.exp(cos * 20)
        proto_probs /= proto_probs.sum(1, keepdims=True)
        probs = (1 - mix) * probs + mix * proto_probs
    aps = []
    offsets = case["query_offsets"]
    for j, cls in enumerate(case["query_classes"].tolist()):
        lo, hi = int(offsets[j]), int(offsets[j + 1])
        positive = case["query_positive"][lo:hi]
        if cls in index:
            aps.append(average_precision(probs[lo:hi, index[cls]], positive))
        else:
            aps.append(float(positive.mean()))  # no label for this class: chance
    return float(np.mean(aps))


def suggest(trial: optuna.Trial) -> dict[str, Any]:
    return dict(
        space=trial.suggest_categorical("space", ["cradio", "dinov3", "concat", "wstack2"]),
        cradio_weight=trial.suggest_float("cradio_weight", 0.25, 4.0, log=True),
        centre=trial.suggest_categorical("centre", [True, False]),
        hidden=trial.suggest_categorical("hidden", [0, 128, 256, 512, 1024]),
        activation=trial.suggest_categorical("activation", ["gelu", "relu", "silu"]),
        dropout=trial.suggest_float("dropout", 0.0, 0.6),
        lr=trial.suggest_float("lr", 1e-4, 2e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        epochs=trial.suggest_int("epochs", 2, 40),
        min_steps=trial.suggest_int("min_steps", 50, 1200, log=True),
        batch=trial.suggest_int("batch", 16, 1024, log=True),
        cw=trial.suggest_float("cw", 0.0, 1.0),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.3),
    )


def evaluate(
    params: dict[str, Any], cases: list[tuple[str, int, int]], seed: int
) -> dict[str, float]:
    return {
        f"{d}/s{s}/n{n}": round(score_case(load_case(d, s, n), params, seed), 4)
        for d, s, n in cases
    }


def objective(trial: optuna.Trial) -> float:
    params = suggest(trial)
    for name in ("prototype_mix", "input_noise", "ensemble", "mixup_alpha", "temperature"):
        if name in trial.params:
            params[name] = trial.params[name]
    t0 = time.time()
    per_case = evaluate(params, SEEN_CASES, seed=0)
    trial.set_user_attr("per_case", per_case)
    trial.set_user_attr("seconds", round(time.time() - t0, 1))
    return float(np.mean(list(per_case.values())))


def heldout_record(study: optuna.Study) -> dict[str, Any]:
    best = study.best_trial
    params = suggest_from_params(best.params)
    heldout = evaluate(params, HELDOUT_CASES, seed=0)
    return {"heldout_per_case": heldout, "heldout_mean": float(np.mean(list(heldout.values())))}


def suggest_from_params(p: dict[str, Any]) -> dict[str, Any]:
    return dict(p)


def build_context(context_dir: Path) -> list[str]:
    files = (
        [Path(__file__)] + sorted(context_dir.glob("*"))
        if context_dir.exists()
        else [Path(__file__)]
    )
    return [str(f) for f in files if f.is_file()]


def main() -> None:
    load_env()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="random,tpe")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--context-dir", default="/var/tmp/optunai/context")
    ap.add_argument("--warm", action="store_true", help="only build the case cache")
    args = ap.parse_args()
    if args.warm:
        for c in SEEN_CASES + HELDOUT_CASES:
            t0 = time.time()
            load_case(*c)
            print("cached", c, f"{time.time() - t0:.1f}s", flush=True)
        return
    context = build_context(Path(args.context_dir))
    for arm in args.arms.split(","):
        for seed in parse_seeds(args.seeds):
            rec = run(
                bench="fewshot",
                problem="mlp_head",
                arm=arm,
                seed=seed,
                n_trials=args.n_trials,
                objective=objective,
                direction="maximize",
                context=context,
                extra_record=heldout_record,
            )
            print(summarise(rec), "heldout", rec.get("heldout_mean"), flush=True)


if __name__ == "__main__":
    main()
