# optunai: Optuna with a language-model sampler and pruner

optunai is a fork of [optuna/optuna](https://github.com/optuna/optuna) (v5.0.0, MIT). It keeps
the `optuna` import path, every storage, the dashboard and every integration, and adds four
public types plus one keyword, and a second layer (`optuna.ladder`, below) for ladder discipline:

| Added | What it does |
|---|---|
| `optuna.samplers.LLMSampler` | Asks a model for the next trial: it sees the search space, the trial history (values, intermediate values, user attributes), the best trial and any `context` you pass (code, README, prior results). Each proposal carries a `hypothesis` and the trial numbers it cites as `evidence`, stored in `trial.system_attrs` under `llm:*`. Invalid values are coerced or repaired once; a second failure hands the trial to TPE and is counted. |
| `optuna.pruners.LLMPruner` | Reads the learning curves of the running and completed trials and asks "will this trial beat the best" as a probability (Jev typed evaluation, or any chat model). Runs `MedianPruner` as a floor so it can only prune more, unless `aggressive=True`. |
| `optuna.samplers.Model` | One LiteLLM-backed model: `Model("openrouter/anthropic/claude-sonnet-5.5")`, `"vertex_ai/gemini-3.5-flash"`, `"openai/qwen3.7-plus"` with an `api_base`, Ollama, and so on. Any object with `name` and `complete(system, user, schema)` works in its place. |
| `optuna.samplers.Ledger` | Append-only JSONL beside the storage: one line per model call (prompt hash, model, tokens, USD, latency), proposal, fallback, violation, prune decision and trial end. Optional sink that writes each trial into trackinizer as an Experiment with a `proves` edge to its hypothesis Belief. |
| `study.optimize(..., allow_new_params=True)` | Lets the sampler add a parameter the objective did not declare. It is written to the storage with a declared distribution and read with `trial.params.get(name, default)`. Off by default. |

Design, including what the model is not allowed to do: [docs/design.md](docs/design.md).

## Install

```bash
pip install "git+https://github.com/GeorgePearse/optunai.git@feat/llm-sampler"
```

Keys come from the environment (`OPENROUTER_API_KEY`, `GEMINI_API_KEY` or application default
credentials for Vertex, `AI_GATEWAY_API_KEY` for Jev, provider keys as LiteLLM expects them).

## Example

```python
import optuna

sampler = optuna.samplers.LLMSampler(
    "openrouter/anthropic/claude-sonnet-5.5",
    context=["train.py", "README.md"],   # what the model may read; None = numbers only
)
study = optuna.create_study(direction="maximize", sampler=sampler,
                            pruner=optuna.pruners.LLMPruner())
study.optimize(objective, n_trials=40)

best = study.best_trial
print(best.params, best.system_attrs["llm:hypothesis"], best.system_attrs["llm:evidence"])
print(sampler.ledger.totals())   # USD, calls, proposals, fallbacks, violations
```

## Benchmarks

Five seeds per arm (fewer where noted), same trial budget per problem, median [IQR] over seeds. Hardware: one shared 16-core CPU VM (no GPU), load 20-80 from other jobs throughout. Models: Claude Sonnet 5.5 via OpenRouter, Gemini 3.5 Flash via Vertex AI, Qwen 3.7 Plus via DashScope, Jev via the Vercel AI Gateway. USD is per study as reported by LiteLLM (list price for Qwen).

| benchmark | arm | best (median [IQR]) | seeds | USD / study | USD / proposal | latency s | fallback | coerced |
|---|---|---|---|---|---|---|---|---|
| fewshot/mlp_head (max) | Random | 0.4654 [0.4632, 0.4685] held-out 0.5550 | 5 | 0.00 | – | – | – | – |
| fewshot/mlp_head (max) | TPE | 0.4704 [0.4663, 0.4710] held-out 0.5616 | 5 | 0.00 | – | – | – | – |
| fewshot/mlp_head (max) | LLMSampler, context, Claude Sonnet 5.5 (OpenRouter) | 0.4721 [0.4721, 0.4721] held-out 0.5686 | 1 | 3.59 | 0.092 | 8.8 | 0.00 | 0.00 |
| fewshot/mlp_head (max) | LLMSampler, no context, Gemini 3.5 Flash | 0.4674 [0.4672, 0.4708] held-out 0.5604 | 5 | 1.69 | 0.042 | 14.8 | 0.00 | 0.00 |
| fewshot/mlp_head (max) | LLMSampler, context, Gemini 3.5 Flash (Vertex) | 0.4698 [0.4697, 0.4707] held-out 0.5614 | 5 | 3.36 | 0.083 | 20.4 | 0.00 | 0.00 |
| fewshot/mlp_head (max) | LLMSampler, context + allow_new_params, Gemini 3.5 Flash | 0.4724 [0.4711, 0.4727] held-out 0.5626 | 5 | 4.05 | 0.098 | 26.6 | 0.01 | 0.00 |
| fewshot/mlp_head (max) | LLMSampler, context, Qwen 3.7 Plus (DashScope) | 0.4716 [0.4688, 0.4716] held-out 0.5673 | 5 | 0.74 | 0.020 | 126.7 | 0.00 | 0.00 |
| sklearn/breast_cancer_noisy (max) | Random | 0.7459 [0.7456, 0.7588] | 5 | 0.00 | – | – | – | – |
| sklearn/breast_cancer_noisy (max) | TPE | 0.7516 [0.7510, 0.7624] | 5 | 0.00 | – | – | – | – |
| sklearn/breast_cancer_noisy (max) | LLMSampler, no context, Gemini 3.5 Flash | 0.7565 [0.7537, 0.7662] | 5 | 1.08 | 0.034 | 13.9 | 0.00 | 0.00 |
| sklearn/breast_cancer_noisy (max) | LLMSampler, context, Gemini 3.5 Flash (Vertex) | 0.7566 [0.7516, 0.7589] | 5 | 1.08 | 0.037 | 15.0 | 0.00 | 0.00 |
| sklearn/digits (max) | Random | 0.9672 [0.9650, 0.9677] | 5 | 0.00 | – | – | – | – |
| sklearn/digits (max) | TPE | 0.9688 [0.9661, 0.9710] | 5 | 0.00 | – | – | – | – |
| sklearn/digits (max) | LLMSampler, no context, Gemini 3.5 Flash | 0.9610 [0.9609, 0.9638] | 5 | 1.06 | 0.036 | 15.1 | 0.00 | 0.00 |
| sklearn/digits (max) | LLMSampler, context, Gemini 3.5 Flash (Vertex) | 0.9609 [0.9604, 0.9625] | 5 | 1.10 | 0.037 | 15.0 | 0.00 | 0.00 |
| synthetic/ackley (min) | Random | 17.1400 [15.6976, 17.4129] | 5 | 0.00 | – | – | – | – |
| synthetic/ackley (min) | TPE | 12.8421 [12.0622, 14.8578] | 5 | 0.00 | – | – | – | – |
| synthetic/ackley (min) | CMA-ES | 9.6841 [6.8890, 12.1864] | 5 | 0.00 | – | – | – | – |
| synthetic/ackley (min) | LLMSampler, no context, Claude Sonnet 5.5 | 0 (exact) [0, 0] | 4 | 0.44 | 0.015 | 2.9 | 0.01 | 0.00 |
| synthetic/ackley (min) | LLMSampler, context, Claude Sonnet 5.5 (OpenRouter) | 0 (exact) [0, 0] | 5 | 0.48 | 0.016 | 3.2 | 0.12 | 0.00 |
| synthetic/ackley (min) | LLMSampler, no context, Gemini 3.5 Flash | 0 (exact) [0, 0] | 5 | 0.51 | 0.016 | 8.3 | 0.00 | 0.00 |
| synthetic/ackley (min) | LLMSampler, context, Gemini 3.5 Flash (Vertex) | 0 (exact) [0, 0] | 5 | 0.57 | 0.018 | 8.5 | 0.00 | 0.00 |
| synthetic/mixed_toy (min) | Random | 2.0044 [1.4804, 2.2273] | 5 | 0.00 | – | – | – | – |
| synthetic/mixed_toy (min) | TPE | 1.0942 [0.7044, 1.6189] | 5 | 0.00 | – | – | – | – |
| synthetic/mixed_toy (min) | CMA-ES | 0.7233 [0.6122, 0.8776] | 5 | 0.00 | – | – | – | – |
| synthetic/mixed_toy (min) | LLMSampler, no context, Gemini 3.5 Flash | 0 (exact) [0, 0] | 5 | 0.85 | 0.025 | 10.8 | 0.00 | 0.00 |
| synthetic/mixed_toy (min) | LLMSampler, context, Gemini 3.5 Flash (Vertex) | 0 (exact) [0, 0] | 5 | 0.60 | 0.021 | 8.5 | 0.00 | 0.00 |
| synthetic/rosenbrock (min) | Random | 4744.6603 [2913.1496, 8313.3837] | 5 | 0.00 | – | – | – | – |
| synthetic/rosenbrock (min) | TPE | 327.8316 [302.0974, 720.8814] | 5 | 0.00 | – | – | – | – |
| synthetic/rosenbrock (min) | CMA-ES | 352.5756 [269.7666, 352.9732] | 5 | 0.00 | – | – | – | – |
| synthetic/rosenbrock (min) | LLMSampler, context, Claude Sonnet 5.5 (OpenRouter) | 0 (exact) [0, 0] | 3 | 0.38 | 0.013 | 2.6 | 0.15 | 0.00 |
| synthetic/rosenbrock (min) | LLMSampler, no context, Gemini 3.5 Flash | 0 (exact) [0, 0] | 5 | 0.59 | 0.019 | 7.2 | 0.00 | 0.00 |
| synthetic/rosenbrock (min) | LLMSampler, context, Gemini 3.5 Flash (Vertex) | 0 (exact) [0, 0] | 5 | 0.67 | 0.020 | 7.7 | 0.00 | 0.00 |

| pruner benchmark | pruner | final best (median [IQR]) | boosting steps (median) | pruned | false-prune rate |
|---|---|---|---|---|---|
| breast_cancer_noisy (max) | TPE, no pruner | 0.7692 [0.7162, 0.7832] | 3500 | 0 | 0.00 |
| breast_cancer_noisy (max) | TPE + MedianPruner | 0.7692 [0.7047, 0.7832] | 2770 | 45 | 0.09 |
| breast_cancer_noisy (max) | TPE + LLMPruner (Jev, median floor) | 0.7637 [0.7232, 0.7832] | 1810 | 113 | 0.02 |
| digits (max) | TPE, no pruner | 0.9737 [0.9722, 0.9833] | 2760 | 0 | 0.00 |
| digits (max) | TPE + MedianPruner | 0.9722 [0.9703, 0.9814] | 2300 | 67 | 0.03 |
| digits (max) | TPE + LLMPruner (Jev, median floor) | 0.9703 [0.9701, 0.9814] | 1950 | 102 | 0.02 |

Reading: on the textbook functions every LLM arm lands on the exact optimum at its first proposal (recall, not search); on the unpublished mixed toy the numbers-only LLM arm beats TPE's final value at trial 4 and reaches 0 by trial 12 by changing one factor at a time. On gradient boosting it edges TPE on the noisy dataset and loses to TPE and random on digits. On the few-shot head the open-vocabulary arm is the best sampler (it adds `ensemble`, `input_noise`, `mixup_alpha` from the objective's docstring) and the context arm is far ahead over the first ten trials; held-out scores are level across arms. The LLM pruner spends 29-35% fewer steps than the median pruner with a 2% false-prune rate. Claude few-shot and Rosenbrock rows have fewer seeds because OpenRouter credit ran out; its synthetic fallback rate is OpenRouter 402 refusals under parallel load, not model output. Full report with regret curves: gs://visia-agent-artifacts/analysis/optunai-llm-sampler-20261007.html (portal: /admin/analytics-artefacts, name optunai-llm-sampler-20261007; DB registration retrying in tmux `optunai-register`).

## Scaling-ladder discipline

The second layer, `optuna.ladder`, turns a study into a ladder in the sense of Zou's
[scaling-ladder post](https://jiaxuanzou0714.github.io/en/blog/2026/how-to-build-scientific-scaling-ladder/):
pre-registered thresholds set from seed noise, a formal `insufficient_evidence` outcome,
selection-bias control, a holdout that becomes development data once looked at, equivalent
compute as the currency, rung fidelity with fitted extrapolation and frozen prediction
intervals, and run statuses that are never deleted. Mapping from the post's sections to these
mechanisms, and what was deliberately not ported: [docs/scaling-ladder.md](docs/scaling-ladder.md).

| Added | What it does |
|---|---|
| `optuna.ladder.Ladder` | A pruner. The objective asks it for rungs (`for rung in ladder.rungs(trial): ...; rung.report(value)`). After each rung it fits `E + A·C^-γ` (or `E + A·ln C`, chosen on development data by leave-last-rung-out error) to the candidate's rungs, extrapolates to the target budget, builds a prediction interval from the extrapolation errors of completed candidates (bootstrapped by candidate, since rungs share a prefix), and kills only when even the better end of that interval does not reach the incumbent minus the frozen threshold. A kill needs at least two rungs: one point is early rank in disguise. The prediction made when a candidate is promoted to its last rung is frozen in the ledger and checked against the final value. `ladder.wrap(objective)` classifies failures into `completed / actively_stopped / algorithmic_divergence / infrastructure_failure / implementation_error` with a `cause_unknown` flag; infrastructure failures never enter the fits; `ladder.retry` re-enqueues and links the records. |
| `optuna.ladder.Decision` | `select / run_more / insufficient_evidence / defer`, plus the record each comparison writes: the difference, its paired interval, the threshold it was tested against, and the equivalent compute multiplier `exp(ΔL / (γ (L − E)))` when a fitted ladder exists. |
| `study.register_noise(objective, n_seeds=3, params=incumbent)` | Runs the incumbent at fresh seeds (outside the trial table), stores the seed SD and freezes `study.decision_threshold = max(min_effect, 2·SD)` with a timestamp, before any holdout result is read. |
| `study.holdout(fn)`, `study.accept()`, `study.near_optimal_check()` | `accept` re-evaluates the selected trial with a new seed on the development objective (the number that is reported, next to the best-of-N and the gap between them) and compares candidate and incumbent on the holdout at the same fresh seeds; returning the record consumes the holdout, so a second `accept` is `insufficient_evidence` until a fresh holdout is registered. `near_optimal_check` perturbs the selected point jointly (log floats ×/÷√2, linear floats ±10 % of range, ints ±25 %) and reports whether the objective moves by less than the threshold and which parameters sit on a search boundary; the sampler may propose `expand_bounds` only for those. |

The LLM sampler sees all of it as context, not prose: the prompt shows the noise floor, the
frozen threshold, the incumbent, the fitted ladder and the near-optimal result; every proposal
must state `expected_effect`, `predicted_target_value`, `interval` and `decision`; when the
previous trial deviated it must carry a `diagnosis` following the post's investigation order
(measurement, config, data, implementation, hardware, recipe). With `gate=p`, Jev scores
whether the stated effect is plausibly above the threshold and a proposal below `p` is deferred
before its first rung runs.

```python
ladder = optuna.ladder.Ladder(rungs=[0.25, 0.5, 1.0])
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.LLMSampler(gate=0.25),
                            pruner=ladder)
study.register_noise(objective, n_seeds=3, params=incumbent)   # freezes the threshold
study.holdout(holdout_objective)
study.optimize(ladder.wrap(objective), n_trials=60)
print(study.accept(objective))        # select | run_more | insufficient_evidence | defer, with the numbers
print(study.near_optimal_check(objective)["on_boundary"])
```

Benchmarks for this layer (rank-flip, few-shot ladder, noise-floor ablation):
[benchmarks/llm/results/ladder/summary.md](benchmarks/llm/results/ladder/summary.md).

## Sync with upstream

```bash
git remote add upstream https://github.com/optuna/optuna.git   # once
git fetch upstream && git merge upstream/master
```

All additions are new modules (`optuna/samplers/_llm/`, `optuna/pruners/_llm.py`,
`benchmarks/llm/`) plus four small edits: two import lines in `optuna/samplers/__init__.py`, one
in `optuna/pruners/__init__.py`, one keyword on `Study.optimize`, and package metadata.

---

The upstream README follows.

<div align="center"><img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width="800"/></div>

# Optuna: A hyperparameter optimization framework

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/)

:link: [**Website**](https://optuna.org/)
| :page_with_curl: [**Docs**](https://optuna.readthedocs.io/en/stable/)
| :gear: [**Install Guide**](https://optuna.readthedocs.io/en/stable/installation.html)
| :pencil: [**Tutorial**](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
| :bulb: [**Examples**](https://github.com/optuna/optuna-examples)
| [**Twitter**](https://twitter.com/OptunaAutoML)
| [**LinkedIn**](https://www.linkedin.com/showcase/optuna/)
| [**Medium**](https://medium.com/optuna)

*Optuna* is an automatic hyperparameter optimization software framework, particularly designed
for machine learning. It features an imperative, *define-by-run* style user API. Thanks to our
*define-by-run* API, the code written with Optuna enjoys high modularity, and the user of
Optuna can dynamically construct the search spaces for the hyperparameters.

## :loudspeaker: News

<!-- TODO: when you add a new line, please delete the oldest line -->
* **Sep 7, 2026**: Optuna v5 and [Rustuna](https://github.com/optuna/rustuna/) v0.1.0 have been released! Check out the release blog posts.
  * [Optuna v5.0.0 release blog post](https://medium.com/optuna/releasing-optuna-v5-0-an-open-source-black-box-optimization-library-9bff8b4587ba)
  * [Rustuna v0.1.0 release blog post](https://medium.com/optuna/announcing-rustuna-cc82a6815bf7)
* **Aug 3, 2026**: Release candidate of Optuna v5 is available! Check out [the release note](https://github.com/optuna/optuna/releases/tag/v5.0.0-rc1) for details.
* **Jun 1, 2026**: Optuna 4.9.0 is out! Check out [the release note](https://github.com/optuna/optuna/releases/tag/v4.9.0) for details.
* **Mar 16, 2026**: Optuna 4.8.0 is out! Check out [the release note](https://github.com/optuna/optuna/releases/tag/v4.8.0) for details.
* **Jan 19, 2026**: Optuna 4.7.0 is out! Check out [the release note](https://github.com/optuna/optuna/releases/tag/v4.7.0) for details.
* **Nov 10, 2025**: A new article [Announcing Optuna 4.6](https://medium.com/optuna/announcing-optuna-4-6-a9e82183ab07) has been published.

## :fire: Key Features

Optuna has modern functionalities as follows:

- [Lightweight, versatile, and platform agnostic architecture](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html)
  - Handle a wide variety of tasks with a simple installation that has few requirements.
- [Pythonic search spaces](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html)
  - Define search spaces using familiar Python syntax including conditionals and loops.
- [Efficient optimization algorithms](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
  - Adopt state-of-the-art algorithms for sampling hyperparameters and efficiently pruning unpromising trials.
- [Easy parallelization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)
  - Scale studies to tens or hundreds of workers with little or no changes to the code.
- [Quick visualization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html)
  - Inspect optimization histories from a variety of plotting functions.


## Basic Concepts

We use the terms *study* and *trial* as follows:

- Study: optimization based on an objective function
- Trial: a single execution of the objective function

Please refer to the sample code below. The goal of a *study* is to find out the optimal set of
hyperparameter values (e.g., `regressor` and `svr_c`) through multiple *trials* (e.g.,
`n_trials=100`). Optuna is a framework designed for automation and acceleration of
optimization *studies*.

<details open>
<summary>Sample code with scikit-learn</summary>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/optuna/optuna-examples/blob/main/quickstart.ipynb)

```python
import optuna
import sklearn


# Define an objective function to be minimized.
def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical("regressor", ["SVR", "RandomForest"])
    if regressor_name == "SVR":
        svr_c = trial.suggest_float("svr_c", 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    return error  # An objective value linked with the Trial object.


study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
```
</details>

> [!NOTE]
> More examples can be found in [optuna/optuna-examples](https://github.com/optuna/optuna-examples).
>
> The examples cover diverse problem setups such as multi-objective optimization, constrained optimization, pruning, and distributed optimization.

## Installation

Optuna is available at [the Python Package Index](https://pypi.org/project/optuna/) and on [Anaconda Cloud](https://anaconda.org/conda-forge/optuna).

```bash
# PyPI
$ pip install optuna
```

```bash
# Anaconda Cloud
$ conda install -c conda-forge optuna
```

> [!IMPORTANT]
> Optuna supports Python 3.9 or newer.

## Rustuna

[Rustuna](https://github.com/optuna/rustuna) is a faster implementation of Optuna written in Rust.
It keeps the API you already know, and rewrites the parts that start to hurt at scale: sampling speed and memory efficiency.

1. **Faster sampler implementations**: Rustuna supports TPE, MOTPE, NSGA-II, and CMA-ES, and finishes the same study several times to several hundred times faster for cheap objective functions.
2. **Memory-efficient storage**: Several design choices to make Rustuna's storage memory-efficient. Furthermore, discarding unnecessary trial history prevents memory usage and runtime from increasing as the number of trials grows.
3. **Zero Python runtime dependencies**: No Python runtime dependencies by default ー dramatically faster imports, and far less exposure to supply chain attacks.

> [!NOTE]
> Rustuna is currently experimental. Compared with Optuna, it still lacks some features and APIs, and it has not yet been optimized enough to deliver better performance for every use case. Since the project has not had the same level of maturity as Optuna, bugs and rough edges likely remain. We appreciate your understanding when using it.

## Web Dashboard

[Optuna Dashboard](https://github.com/optuna/optuna-dashboard) is a real-time web dashboard for Optuna.
You can check the optimization history, hyperparameter importance, etc. in graphs and tables.
You don't need to create a Python script to call [Optuna's visualization](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html) functions.
Feature requests and bug reports are welcome!

![optuna-dashboard](https://user-images.githubusercontent.com/5564044/204975098-95c2cb8c-0fb5-4388-abc4-da32f56cb4e5.gif)

`optuna-dashboard` can be installed via pip:

```shell
$ pip install optuna-dashboard
```

> [!TIP]
> Please check out the convenience of Optuna Dashboard using the sample code below.

<details>
<summary>Sample code to launch Optuna Dashboard</summary>

Save the following code as `optimize_toy.py`.

```python
import optuna


def objective(trial):
    x1 = trial.suggest_float("x1", -100, 100)
    x2 = trial.suggest_float("x2", -100, 100)
    return x1**2 + 0.01 * x2**2


study = optuna.create_study(storage="sqlite:///db.sqlite3")  # Create a new study with database.
study.optimize(objective, n_trials=100)
```

Then try the commands below:

```shell
# Run the study specified above
$ python optimize_toy.py

# Launch the dashboard based on the storage `sqlite:///db.sqlite3`
$ optuna-dashboard sqlite:///db.sqlite3
...
Listening on http://localhost:8080/
Hit Ctrl-C to quit.
```

</details>


## OptunaHub

[OptunaHub](https://hub.optuna.org/) is a feature-sharing platform for Optuna.
You can use the registered features and publish your packages.

### Use registered features

`optunahub` can be installed via pip:

```shell
$ pip install optunahub
# Install AutoSampler dependencies (CPU only is sufficient for PyTorch)
$ pip install cmaes scipy torch --extra-index-url https://download.pytorch.org/whl/cpu
```

You can load registered module with `optunahub.load_module`.

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


module = optunahub.load_module(package="samplers/auto_sampler")
# See https://hub.optuna.org/samplers/auto_sampler/ to know the ``AutoSampler`` API.
study = optuna.create_study(sampler=module.AutoSampler())
study.optimize(objective, n_trials=10)

print(study.best_trial.value, study.best_trial.params)
```

For more details, please refer to [the optunahub documentation](https://optuna.github.io/optunahub/).

### Publish your packages

You can publish your package via [optunahub-registry](https://github.com/optuna/optunahub-registry).
See the [Tutorials for Contributors](https://optuna.github.io/optunahub/tutorials_for_contributors.html) in OptunaHub.


## Communication

- [GitHub Discussions] for questions.
- [GitHub Issues] for bug reports and feature requests.

[GitHub Discussions]: https://github.com/optuna/optuna/discussions
[GitHub issues]: https://github.com/optuna/optuna/issues


## Contribution

Any contributions to Optuna are more than welcome!

If you are new to Optuna, please check the [good first issues](https://github.com/optuna/optuna/labels/good%20first%20issue). They are relatively simple, well-defined, and often good starting points for you to get familiar with the contribution workflow and other developers.

If you already have contributed to Optuna, we recommend the other [contribution-welcome issues](https://github.com/optuna/optuna/labels/contribution-welcome).

For general guidelines on how to contribute to the project, take a look at [CONTRIBUTING.md](./CONTRIBUTING.md).


## Reference

If you use Optuna in one of your research projects, please cite [our KDD paper](https://doi.org/10.1145/3292500.3330701) "Optuna: A Next-generation Hyperparameter Optimization Framework":

<details open>
<summary>BibTeX</summary>

```bibtex
@inproceedings{akiba2019optuna,
  title={{O}ptuna: A Next-Generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={The 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2623--2631},
  year={2019}
}
```
</details>


## License

MIT License (see [LICENSE](./LICENSE)).

Optuna uses the codes from SciPy and fdlibm projects (see [LICENSE_THIRD_PARTY](./LICENSE_THIRD_PARTY)).
