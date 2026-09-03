# CompactifAI-aligned evaluation

Status: complete (3/3)

Scores and deltas are percentage points. Recovery is the fraction of the baseline-to-compressed loss recovered by retraining.

| Task / metric | baseline | mpo | retrained_mpo | MPO delta | retrained delta | healing gain | recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| hellaswag_smoke / acc_norm,none | 50.00 | 50.00 | 18.75 | +0.00 | -31.25 | -31.25 | — |

Raw lm-eval outputs are stored under each variant directory.
