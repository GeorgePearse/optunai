# Scaling-ladder discipline inside optunai

This document maps the experimental discipline of Zou's
["How to Build a Scientific Scaling Ladder"](https://jiaxuanzou0714.github.io/en/blog/2026/how-to-build-scientific-scaling-ladder/)
(September 2026) onto optunai, a fork of Optuna with a language-model sampler and pruner. The
post is written for LLM pretraining ladders; most of its material (parameter counting, MoE
axes, data mixtures, tokenizers, LR/BSZ scaling rules, downstream prediction) does not transfer
to a hyperparameter study. What transfers is how it turns "run experiments, pick the best" into
a procedure with pre-registered thresholds, formal outcomes, selection-bias control and an audit
trail. Four public names carry that into optunai:

| Public name | What it is |
|---|---|
| `optuna.ladder.Ladder` | A pruner that runs each candidate at rungs of increasing budget, fits `L(C) = E + A C^-γ` (or a plain power law, chosen on development data) to its rung results, and promotes or kills on the *predicted target value with its bootstrap interval*, never on the raw rank at the current rung. |
| `optuna.ladder.Decision` | The formal outcome enum `select / run_more / insufficient_evidence / defer` and the record every comparison writes: difference, paired interval, the threshold it was tested against, and the equivalent compute multiplier when a fitted ladder exists. |
| `Study.register_noise` | Runs the incumbent at fresh seeds before search, stores the seed SD, and freezes `study.decision_threshold = max(min_effect, 2·seed_SD)` with a timestamped ledger line. |
| `Study.accept` | Re-runs the selected candidate with a new seed on a holdout registered with `Study.holdout`, records acceptance evidence separately from development evidence, reports the fresh-seed value (not the best-of-N) and the gap between the two, and consumes the holdout. `Study.near_optimal_check` and `Study.holdout` are part of the same surface. |

Everything else is in `trial.user_attrs`, `study.system_attrs` and the ledger, under `ladder:*`
keys, so any storage and the dashboard carry it with no schema change.

## Mapping: post section → optunai mechanism

| Post section | What the post says | optunai mechanism |
|---|---|---|
| §2.3 Acceptance thresholds | Thresholds come from the smallest difference that must be resolved and from seed variance; set them *before* observing holdout results. | `study.register_noise(objective, n_seeds=3, min_effect=...)` runs the incumbent at `n_seeds` fresh seeds and freezes `decision_threshold = max(min_effect, 2·seed_SD)` with an ISO timestamp in `study.system_attrs["ladder:threshold_frozen_at"]` and a `threshold_frozen` ledger line. `accept()` refuses to run before the threshold is frozen unless `min_effect` is given explicitly. |
| §2.3 Equivalent compute multiplier | Express thresholds and differences as `ln(C2/C1) ≈ ΔL / (γ (L − E))`, not only as absolute loss. | Every `Decision` record carries `equivalent_compute_multiplier` computed from the incumbent's fitted `(E, γ, L)` when a `Ladder` is attached to the study (`None` otherwise, never a made-up number). The LLM prompt shows it per trial. |
| §2.3 Three decision categories; "insufficient evidence" is a formal outcome | Select, run more, or defer; never a silent pick. | `optuna.ladder.Decision` enum with four values (`select`, `run_more`, `insufficient_evidence`, `defer`) and a rule that reads the difference, its interval and the threshold. Recorded on every acceptance and every ladder promotion. |
| §2.3 Uncertainty of a difference, shared error cancels | Two independent error bars cannot be added; estimate the paired difference. | `accept()` evaluates candidate and incumbent at the *same* fresh seeds and uses the paired differences (or `√2·seed_SD` with one seed). |
| §5.4 Shared trajectories; branches sharing a prefix are not independent | Resampling units must match the dependency structure. | The `Ladder` bootstrap resamples *candidates*, never individual rung points: all rungs of one trial are one unit, which is what a warm-started objective produces. |
| §5.5 Run a few trials first to estimate seed noise, then allocate | Seed noise is measured before the search budget is allocated. | `register_noise` is the first call on a study; its cost is written to the ledger as `role=noise` trials, which stay in the history (tagged) rather than being hidden. |
| §6.1 Near-optimal region; optimum must not lie on the search boundary | Operational definition of "fully tuned": joint perturbation moves loss less than a tolerance ≥ seed variance; optimum not on a boundary. | `study.near_optimal_check(objective)` perturbs the selected point jointly (log floats ×/÷√2, linear floats ±10 % of range, ints ±25 %), evaluates, and reports `near_optimal` (max |Δ| < threshold) and `on_boundary` (names within one step of a bound). The LLM sampler may propose `expand_bounds` only for the names in `on_boundary`; the expansion is written to `study.system_attrs["ladder:bounds"]` for the objective to read. |
| §6.7 Selection bias (Cawley & Talbot); re-verify the final candidate with a new seed; separate selection evaluation from reporting evaluation | The lowest of N noisy trials overstates the gain. | `accept()` reports `reevaluated_value` (fresh seed, development objective) and `holdout_value` (fresh seed, holdout objective) next to `best_of_n_value`; the gap is `selection_bias_gap`. The reported best is the fresh-seed number. |
| §6.7 Do not use early rankings to eliminate configurations; rankings flip | Early rank is not a kill criterion. | `Ladder.prune` kills only when the *better* end of the candidate's predicted-target interval does not reach the incumbent's target value; a candidate that is behind at the current rung but extrapolates to win is promoted. Compared with `HyperbandPruner` on a constructed rank-flip benchmark. |
| §6.7, §12.1 Run statuses | Actively stopped, algorithmic divergence, infrastructure failure, implementation error are separate statuses; nothing is deleted. | `Ladder.wrap(objective)` classifies exceptions and non-finite values into `ladder:status_reason` ∈ {`completed`, `actively_stopped`, `algorithmic_divergence`, `infrastructure_failure`, `implementation_error`} with `ladder:cause_unknown`. Infrastructure failures are excluded from the ladder's fits and from kill counts; `Ladder.retry` re-enqueues the same parameters and links the records with `ladder:supersedes` / `ladder:superseded_by`. `LLMPruner` shows diverged trials as divergence, not as "pruned". |
| §8.1 Functional form | Compare candidate forms on development data; holdout does not participate. | Two forms, `E + A C^-γ` and `A C^-γ`, scored by leave-last-rung-out error over completed development candidates; the chosen form is recorded in `ladder:fit`. The acceptance holdout is never used for form selection. |
| §8.3 Fitting protocol fixed in advance; prediction intervals for a single future run; bootstrap by dependency unit | Fit settings are recorded; intervals are reported separately from parameter errors. | `optuna.ladder._fit` fixes the protocol: fit on the value scale, exponent on a grid with a linear solve for `(E, A)`, `E ≤ min` constraint, no outlier removal. Intervals are prediction intervals from the empirical leave-last-rung-out residuals, bootstrapped by candidate, and are frozen in the ledger (`prediction_frozen`) before the final rung is read. |
| §8.4 Failure handling; "cause unknown" | When the fit is poor, list what cannot be decided. | Fit diagnostics (`rmse`, number of units, chosen form) go into `ladder:fit`; a candidate with fewer than two rungs gets the pooled rung-to-target shift with its wide interval and a `fit=pooled` tag, and a ladder with fewer than `n_startup` completed candidates kills nothing. |
| §10.1 Holdout; once used for adjustment it becomes development data | Acceptance needs evidence that selection never saw. | `study.holdout(fn)` registers an objective that only `accept()` evaluates. After `accept()` returns, the holdout is marked consumed (`holdout_consumed` ledger line); a second `accept()` on it returns `insufficient_evidence` with the reason until a fresh holdout is registered. |
| §10.6 Freeze the predicted curve and interval before the run; pass if inside | Predictions are committed before the result. | `prediction_frozen` ledger lines carry the predicted target value and interval at the moment of promotion; `trial_end` records `inside_interval`. The calibration of those intervals is reported in the benchmark. |
| §12.2 Investigation order for deviations | Measurements and config first, then data, implementation, hardware, and only last the recipe. | The sampler system prompt contains the order; when the previous trial deviated (`status_reason` not `completed`, or a value outside its frozen interval) the proposal must carry a `diagnosis` with the step it is testing, and the prompt forbids changing the recipe before the earlier steps are addressed. |
| Proposal discipline (§2.3, §6.7) | Every hypothesis states the effect it expects relative to the threshold. | The proposal schema gains `expected_effect`, `predicted_target_value`, `interval`, `equivalent_compute_multiplier`, `decision`. Jev scores "is this proposal's expected effect plausibly above the threshold" as a typed probability; below `gate` the trial is deferred (status `actively_stopped`, reason `deferred_by_gate`) before its first rung runs. |

## What was deliberately not ported, and why

- **Parameter counting, FLOPs accounting, `N_body` vs `N_total` (§3).** A study's compute axis is
  whatever the objective declares as a rung budget (data fraction, steps, epochs). The Ladder
  takes rung budgets as numbers and does not interpret them.
- **Architecture consistency, width-depth, MoE scaling rules (§4, §5.6, §8.5).** Not a
  hyperparameter-study concern.
- **LR/BSZ/WD scaling laws and µ-transfer (§6.2-6.6).** These are priors for *where* to search at
  a larger scale; the sampler already receives context and the trial history, and a scaling rule
  is something the objective's author encodes in the search space. Fitting hyperparameter power
  laws across scales needs several model sizes per candidate, which a study does not have.
- **Data ladder, mixtures, repetition laws (§7).** Out of scope.
- **Chinchilla/Skaling joint forms in `(N, D)` (§8.1).** A rung ladder is one-dimensional in
  budget; the post itself says the one-dimensional `E + G/C^γ` form is the one to use when the
  ratio is fixed. Two candidate forms are kept so the form choice is still a recorded decision.
- **Loss-curve-with-schedule fitting (§8.2).** Needs the LR schedule as an input; objectives do
  not expose it.
- **Downstream task prediction, COD (§9).** No proxy-to-target mapping exists in a study; the
  objective is the target.
- **Stability stress testing, implementation consistency, realized efficiency (§10.3-10.5).**
  Infrastructure checks that live in the training code, not in the optimiser. The status taxonomy
  (§12.1) is the part that reaches the study.
- **Medium-scale trial run (§10.6)** is ported only as the frozen-prediction mechanism; there is
  no separate "medium scale" in a study.
- **Regression ladder and coefficient versioning (§12.3).** The ledger records fit coefficients
  and the prompt hash per study; a cross-study regression ladder is a workflow around optunai,
  not inside it.

## Where the pieces live

- `optuna/ladder/_fit.py`: the two functional forms, the grid-plus-linear fit, pooled exponent,
  leave-last-rung-out residuals, bootstrap by candidate, equivalent compute multiplier.
- `optuna/ladder/_decision.py`: `Decision` enum, `DecisionRecord`, the rule.
- `optuna/ladder/_ladder.py`: `Ladder` (a `BasePruner`), `Ladder.rungs(trial)`, `Ladder.wrap`,
  `Ladder.retry`, status classification.
- `optuna/ladder/_study.py`: `register_noise`, `holdout`, `accept`, `near_optimal_check`, bound
  expansion; `Study` gets four one-line delegations and the `decision_threshold` property.
- `optuna/samplers/_llm/_proposal.py` and `prompts/sampler_system.md`: the ladder block of the
  prompt, the extra schema fields, the Jev gate; `optuna/pruners/_llm.py`: status-aware state.
- `benchmarks/llm/rankflip.py`, `benchmarks/llm/fewshot_ladder.py`,
  `benchmarks/llm/noise_ablation.py`: the three evaluations.
