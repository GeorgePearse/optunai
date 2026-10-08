# Scaling-ladder benchmarks

## Rank-flip (synthetic, 24 candidates per pool, rungs 1/2/4/8/16, noise SD 0.01)

30 seeds (pools). True best is at median rank 3 of 24 at the first rung; 32% of candidate pairs flip order between the first rung and the target.

| arm | P(select true best) | P(true best reaches target) | regret median [IQR] | compute (fraction of full) | candidates completed |
|---|---|---|---|---|---|
| full budget, no pruning | 0.90 | 1.00 | 0 [0, 0] | 384 (1.00) | 24 |
| Ladder (extrapolation) | 0.90 | 0.97 | 0 [0, 0] | 158 (0.41) | 6 |
| Hyperband η=2 | 0.73 | 0.83 | 0 [0, 0.0019] | 208 (0.54) | 11 |
| Hyperband η=3 | 0.70 | 0.77 | 0 [0, 0.0045] | 190 (0.50) | 10 |
| Hyperband η=4 | 0.63 | 0.67 | 0 [0, 0.0131] | 160 (0.42) | 8 |
| MedianPruner | 0.70 | 0.77 | 0 [0, 0.0051] | 172 (0.45) | 9 |
| ASHA η=2 | 0.60 | 0.67 | 0 [0, 0.0121] | 105 (0.27) | 4 |
| ASHA η=3 | 0.57 | 0.63 | 0 [0, 0.0132] | 102 (0.27) | 4 |
| ASHA η=4 | 0.50 | 0.57 | 0.0013 [0, 0.0145] | 87 (0.23) | 3 |

Equal compute: Ladder 0.90 at 158 vs Hyperband η=4 0.63 at 160.
Frozen 90% prediction intervals covered the final value in 79% of 68 promotions.

## Few-shot MLP head with the ladder (higher is better)

Seed SD of the incumbent (3 fresh seeds per study, pooled median): 0.0017; frozen threshold = 2·SD = 0.0035.

| arm | seeds | best-of-N median [IQR] | fresh-seed re-evaluation | selection-bias gap | holdout (fresh seed) | incumbent holdout | decision per study | near-optimal | USD/study | trials | pruned | deferred | frozen intervals inside / n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TPE (40 full trials) | 3 | 0.4701 [0.4693, 0.4717] | 0.4683 [0.4677, 0.4701] | 0.0014 | 0.5674 [0.5649, 0.5685] | 0.5686 | defer, defer, defer | 1/3 | 0 | 40 | 0 | 0 | – |
| LLMSampler alone (no threshold in prompt) | 3 | 0.4715 [0.4712, 0.4715] | 0.4687 [0.4686, 0.4688] | 0.0026 | 0.5599 [0.5599, 0.5617] | 0.5686 | defer, defer, defer | 1/3 | 4.82 | 40 | 0 | 0 | – |
| LLMSampler + register_noise (threshold in prompt) | 3 | 0.4727 [0.4719, 0.4731] | 0.4689 [0.4673, 0.4691] | 0.0034 | 0.5651 [0.5587, 0.5659] | 0.5686 | defer, defer, defer | 0/3 | 5.28 | 40 | 0 | 0 | – |
| LLMSampler + Ladder (rungs, gate, threshold) | 3 | 0.4719 [0.4717, 0.472] | 0.4693 [0.4686, 0.4695] | 0.0027 | 0.5655 [0.5652, 0.5659] | 0.5686 | defer, defer, defer | 0/3 | 8.67 | 46 | 12 | 0 | 92 / 92 |

## Noise-floor ablation

A 'declared win' is every new best-so-far in the study. The thresholded rule tests each against the frozen threshold with a √2·seed_SD interval.

| arm | declared wins | below threshold | select | insufficient_evidence | defer |
|---|---|---|---|---|---|
| TPE (40 full trials) | 13 | 6 | 6 | 7 | 0 |
| LLMSampler alone (no threshold in prompt) | 22 | 18 | 3 | 19 | 0 |
| LLMSampler + register_noise (threshold in prompt) | 38 | 36 | 2 | 36 | 0 |
| LLMSampler + Ladder (rungs, gate, threshold) | 22 | 19 | 2 | 20 | 0 |
