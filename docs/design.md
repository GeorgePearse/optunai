# optunai design: an LLM as a sampler and pruner inside Optuna

optunai is a fork of Optuna. It keeps the `optuna` import path, every storage, the dashboard
and every integration, and adds four public types: `optuna.samplers.LLMSampler`,
`optuna.pruners.LLMPruner`, `optuna.samplers.Model` and `optuna.samplers.Ledger`. The only
change a user makes is `sampler=optuna.samplers.LLMSampler(...)`.

## What a language model can do that TPE cannot

TPE, CMA-ES and GP samplers see one thing: the table of `(params, value)` pairs. A model that
is given the same table can also read

1. **The code and the documentation of the objective.** Which parameter is a learning rate,
   which one is a regulariser, which pair interacts, what the README says has already been
   tried. TPE learns this from samples; the model knows it before trial 1.
2. **Per-trial breakdowns, not just the scalar.** Intermediate values, user attributes such as a
   per-dataset or per-class score, the shape of a loss curve. TPE reduces every trial to one
   number.
3. **Structure the objective did not ask for.** A new parameter ("try label smoothing") or a
   different family of values. Every classical sampler is confined to the declared space.
4. **A stated reason.** Each proposal carries a hypothesis and the trials it cites, so the
   study is auditable afterwards and can be pushed into a tracking system as experiments that
   prove or refute beliefs.

Points 1 and 2 should show up as fewer trials to reach a given value when context is rich and
the space is small (tens of trials). Point 3 is the one capability no classical sampler has.
Point 4 costs nothing and is always on.

## What the model must not be allowed to do

- **Break the contract of a distribution.** Every proposal is validated against Optuna's
  distributions. Out-of-range floats are clamped and ints and steps snapped (counted as a
  schema violation); a missing or unparseable parameter triggers one repair call; if the
  repair also fails the trial is sampled by the fallback sampler (TPE by default) and the
  fallback is counted. The objective never sees an invalid value.
- **Change the search space silently.** New parameters are off by default
  (`allow_new_params=False`). When on, each new dimension is written to the storage as an
  ordinary Optuna parameter with a declared distribution, recorded in the ledger with its
  rationale, and visible in `trial.params`. The objective reads it with `trial.params.get`.
- **Prune less than the classic pruner.** `LLMPruner` runs `MedianPruner` as a floor and can
  only add prunes on top of it, unless `aggressive=True` is set explicitly.
- **Spend without a record.** Every model call writes one ledger line: prompt hash, model id,
  tokens, USD, latency. The sampler also stores hypothesis, evidence, model, prompt hash,
  tokens and USD in `trial.system_attrs` so storages and the dashboard carry them with no
  schema change.
- **Be the only sampler.** The first `n_startup_trials` trials and any parameter outside the
  intersection search space (conditional parameters) are sampled by the fallback sampler.
  The model proposes over the relative search space only.

## Shape of the implementation

- `optuna/samplers/_llm/`: `_model.py` (LiteLLM-backed `Model`, duck-typed so any object with
  `complete(system, user, schema)` works), `_ledger.py` (`Ledger`, JSONL plus optional sink),
  `_trackinizer.py` (sink that writes Experiments with a `proves` edge to a Belief),
  `_jev.py` (typed evaluation calls), `_proposal.py` (prompt assembly, validation, repair),
  `_sampler.py` (`LLMSampler`), `prompts/` (versioned prompt files, hashed into every ledger
  line).
- `optuna/pruners/_llm.py`: `LLMPruner`.
- Registration edits: two import lines in `optuna/samplers/__init__.py`, one in
  `optuna/pruners/__init__.py`, and one keyword (`allow_new_params`) on `Study.optimize`.
  Everything else is new files, so `git merge upstream/master` stays trivial.

Multi-objective studies are accepted: the prompt lists all directions and values, and the
fallback sampler handles what it handles. No Pareto-specific logic was added; it costs nothing
and is not benchmarked here.

## Evaluation

Benchmarks compare `RandomSampler`, `TPESampler`, `CmaEsSampler` (where the space allows) and
`LLMSampler` with and without context, 5 seeds each, same trial budget, best-so-far against
trials and against USD. The pruner is compared with `MedianPruner` and no pruning on steps
spent, final best and false-prune rate. Costs, fallback rates and schema-violation rates are
reported per model. See the README for the table and `benchmarks/` for the scripts.
