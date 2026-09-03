# CompactifAI-aligned evaluation

Status: complete (15/15)

Scores and deltas are percentage points. Recovery is the fraction of the baseline-to-compressed loss recovered by retraining.

| Task / metric | baseline | mpo | retrained_mpo | MPO delta | retrained delta | healing gain | recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| mmlu_5shot / acc,none | 74.80 | 25.23 | 23.42 | -49.57 | -51.38 | -1.82 | -3.7% |
| hellaswag_0shot / acc_norm,none | 74.96 | 30.19 | 35.48 | -44.76 | -39.47 | +5.29 | 11.8% |
| boolq_0shot / acc,none | 86.73 | 40.70 | 61.53 | -46.02 | -25.20 | +20.83 | 45.2% |
| triviaqa_0shot / exact_match,remove_whitespace | 32.03 | 0.00 | 0.20 | -32.03 | -31.83 | +0.20 | 0.6% |
| gsm8k_5shot / exact_match,strict-match | 87.87 | 0.00 | 0.00 | -87.87 | -87.87 | +0.00 | 0.0% |

Raw lm-eval outputs are stored under each variant directory.
