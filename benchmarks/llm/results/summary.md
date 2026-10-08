
### fewshot / mlp_head (higher is better)

| arm | model | best median [IQR] | held-out | USD/study | USD/proposal | latency s | fallback | violations |
|---|---|---|---|---|---|---|---|---|
| llm:claude | openrouter/anthropic/claude-sonnet-5.5 | 0.4721 [0.4721, 0.4721] | 0.5686 [0.5686, 0.5686] | 3.592 | 0.0922 | 8.8 | 0.00 | 0.00 |
| llm:gemini | vertex_ai/gemini-3.5-flash | 0.4698 [0.4697, 0.4707] | 0.5614 [0.5575, 0.5627] | 3.357 | 0.0826 | 20.4 | 0.00 | 0.00 |
| llm-new:gemini | vertex_ai/gemini-3.5-flash | 0.4724 [0.4711, 0.4727] | 0.5626 [0.5609, 0.5638] | 4.046 | 0.0978 | 26.6 | 0.01 | 0.00 |
| llm-noctx:gemini | vertex_ai/gemini-3.5-flash | 0.4674 [0.4672, 0.4708] | 0.5604 [0.5583, 0.5651] | 1.694 | 0.0424 | 14.8 | 0.00 | 0.00 |
| llm:qwen | openai/qwen3.7-plus | 0.4716 [0.4688, 0.4716] | 0.5673 [0.5658, 0.5702] | 0.741 | 0.0195 | 126.7 | 0.00 | 0.00 |
| random | - | 0.4654 [0.4632, 0.4685] | 0.5550 [0.5534, 0.5704] | 0.000 | - | - | - | - |
| tpe | - | 0.4704 [0.4663, 0.4710] | 0.5616 [0.5601, 0.5623] | 0.000 | - | - | - | - |

### pruner / breast_cancer_noisy (higher is better)

| pruner | final best median [IQR] | steps median | pruned | false-prune rate | judge calls | judge USD |
|---|---|---|---|---|---|---|
| tpe+llm | 0.7637 [0.7232, 0.7832] | 1810 | 113 | 0.02 | 147 | 0.0000 |
| tpe+median | 0.7692 [0.7047, 0.7832] | 2770 | 45 | 0.09 | 0 | 0.0000 |
| tpe+none | 0.7692 [0.7162, 0.7832] | 3500 | 0 | 0.00 | 0 | 0.0000 |

### pruner / digits (higher is better)

| pruner | final best median [IQR] | steps median | pruned | false-prune rate | judge calls | judge USD |
|---|---|---|---|---|---|---|
| tpe+llm | 0.9703 [0.9701, 0.9814] | 1950 | 102 | 0.02 | 155 | 0.0000 |
| tpe+median | 0.9722 [0.9703, 0.9814] | 2300 | 67 | 0.03 | 0 | 0.0000 |
| tpe+none | 0.9737 [0.9722, 0.9833] | 2760 | 0 | 0.00 | 0 | 0.0000 |

### sklearn / breast_cancer_noisy (higher is better)

| arm | model | best median [IQR] | held-out | USD/study | USD/proposal | latency s | fallback | violations |
|---|---|---|---|---|---|---|---|---|
| llm:gemini | vertex_ai/gemini-3.5-flash | 0.7566 [0.7516, 0.7589] | - | 1.085 | 0.0371 | 15.0 | 0.00 | 0.00 |
| llm-noctx:gemini | vertex_ai/gemini-3.5-flash | 0.7565 [0.7537, 0.7662] | - | 1.078 | 0.0344 | 13.9 | 0.00 | 0.00 |
| random | - | 0.7459 [0.7456, 0.7588] | - | 0.000 | - | - | - | - |
| tpe | - | 0.7516 [0.7510, 0.7624] | - | 0.000 | - | - | - | - |

### sklearn / digits (higher is better)

| arm | model | best median [IQR] | held-out | USD/study | USD/proposal | latency s | fallback | violations |
|---|---|---|---|---|---|---|---|---|
| llm:gemini | vertex_ai/gemini-3.5-flash | 0.9609 [0.9604, 0.9625] | - | 1.099 | 0.0366 | 15.0 | 0.00 | 0.00 |
| llm-noctx:gemini | vertex_ai/gemini-3.5-flash | 0.9610 [0.9609, 0.9638] | - | 1.062 | 0.0360 | 15.1 | 0.00 | 0.00 |
| random | - | 0.9672 [0.9650, 0.9677] | - | 0.000 | - | - | - | - |
| tpe | - | 0.9688 [0.9661, 0.9710] | - | 0.000 | - | - | - | - |

### synthetic / ackley (lower is better)

| arm | model | best median [IQR] | held-out | USD/study | USD/proposal | latency s | fallback | violations |
|---|---|---|---|---|---|---|---|---|
| cmaes | - | 9.6841 [6.8890, 12.1864] | - | 0.000 | - | - | - | - |
| llm:claude | openrouter/anthropic/claude-sonnet-5.5 | 0.0000 [0.0000, 0.0000] | - | 0.484 | 0.0159 | 3.2 | 0.12 | 0.00 |
| llm:gemini | vertex_ai/gemini-3.5-flash | 0.0000 [0.0000, 0.0000] | - | 0.575 | 0.0182 | 8.5 | 0.00 | 0.00 |
| llm-noctx:claude | openrouter/anthropic/claude-sonnet-5.5 | 0.0000 [0.0000, 0.0000] | - | 0.436 | 0.0148 | 2.9 | 0.01 | 0.00 |
| llm-noctx:gemini | vertex_ai/gemini-3.5-flash | 0.0000 [0.0000, 0.0000] | - | 0.509 | 0.0164 | 8.3 | 0.00 | 0.00 |
| random | - | 17.1400 [15.6976, 17.4129] | - | 0.000 | - | - | - | - |
| tpe | - | 12.8421 [12.0622, 14.8578] | - | 0.000 | - | - | - | - |

### synthetic / mixed_toy (lower is better)

| arm | model | best median [IQR] | held-out | USD/study | USD/proposal | latency s | fallback | violations |
|---|---|---|---|---|---|---|---|---|
| cmaes | - | 0.7233 [0.6122, 0.8776] | - | 0.000 | - | - | - | - |
| llm:gemini | vertex_ai/gemini-3.5-flash | 0.0000 [0.0000, 0.0000] | - | 0.598 | 0.0214 | 8.5 | 0.00 | 0.00 |
| llm-noctx:gemini | vertex_ai/gemini-3.5-flash | 0.0000 [0.0000, 0.0000] | - | 0.849 | 0.0248 | 10.8 | 0.00 | 0.00 |
| random | - | 2.0044 [1.4804, 2.2273] | - | 0.000 | - | - | - | - |
| tpe | - | 1.0942 [0.7044, 1.6189] | - | 0.000 | - | - | - | - |

### synthetic / rosenbrock (lower is better)

| arm | model | best median [IQR] | held-out | USD/study | USD/proposal | latency s | fallback | violations |
|---|---|---|---|---|---|---|---|---|
| cmaes | - | 352.5756 [269.7666, 352.9732] | - | 0.000 | - | - | - | - |
| llm:claude | openrouter/anthropic/claude-sonnet-5.5 | 0.0000 [0.0000, 0.0000] | - | 0.380 | 0.0126 | 2.6 | 0.15 | 0.00 |
| llm:gemini | vertex_ai/gemini-3.5-flash | 0.0000 [0.0000, 0.0000] | - | 0.665 | 0.0197 | 7.7 | 0.00 | 0.00 |
| llm-noctx:gemini | vertex_ai/gemini-3.5-flash | 0.0000 [0.0000, 0.0000] | - | 0.590 | 0.0187 | 7.2 | 0.00 | 0.00 |
| random | - | 4744.6603 [2913.1496, 8313.3837] | - | 0.000 | - | - | - | - |
| tpe | - | 327.8316 [302.0974, 720.8814] | - | 0.000 | - | - | - | - |
