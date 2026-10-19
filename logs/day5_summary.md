## μ=1：相对 baseline
| run | acc | lenient | len_ok | len_bad | gap | +/- vs baseline | p | ent | zstd | clip_hi | clip_all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 74.8% | 74.8% | 151 | 192 | +40 | — | — | 0.162 | 0.53 | 0.0000 | 0.0000 |
| dapo_token | 73.1% | 73.1% | 140 | 176 | +36 | +77 / -99 | 0.11 | 0.149 | 0.57 | 0.0000 | 0.0000 |
| dr_grpo | 76.2% | 76.3% | 151 | 201 | +50 | +95 / -76 | 0.17 | 0.128 | 0.57 | 0.0000 | 0.0000 |

## μ=2：相对 dapo_token_mu2
| run | acc | lenient | len_ok | len_bad | gap | +/- vs dapo_token_mu2 | p | ent | zstd | clip_hi | clip_all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dapo_token_mu2 | 74.4% | 74.5% | 167 | 217 | +50 | — | — | 0.111 | 0.67 | 0.0005 | 0.0006 |
| dapo_clip_higher | 75.5% | 75.5% | 170 | 217 | +47 | +93 / -78 | 0.28 | 0.137 | 0.63 | 0.0001 | 0.0003 |
| dapo_full | 没有 results/dapo_full.jsonl |
