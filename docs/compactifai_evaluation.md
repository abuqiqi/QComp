# CompactifAI-aligned standard evaluation

CompactifAI reports five downstream tasks: MMLU, HellaSwag, BoolQ, TriviaQA,
and GSM8K. The paper identifies LM Evaluation Harness as the evaluator, but does
not publish the exact harness version, prompts, few-shot counts, or metric filters.
Consequently, an exact reproduction of the paper protocol is not possible from the
paper alone.

This repository fixes the following auditable `lm-eval==0.4.12` protocol for all
three Qwen3-8B variants:

| Task | Setting | Reported metric |
|---|---:|---|
| MMLU | 5-shot | `acc,none` |
| HellaSwag | 0-shot | `acc_norm,none` |
| BoolQ | 0-shot | `acc,none` |
| TriviaQA (`rc.nocontext`) | 0-shot | `exact_match,remove_whitespace` |
| GSM8K | 5-shot | `exact_match,strict-match` |

All tasks use seed 42, batch size 8, maximum context length 4096, no chat
template, and no bootstrap resampling. The comparison configuration is
`configs/experiments/qwen3_8b_compactifai_standard_eval.json`.

Run a 16-example HellaSwag end-to-end check first:

```bash
python -m qwen3_tn.cli.evaluate_suite \
  configs/experiments/qwen3_8b_compactifai_standard_eval_smoke.json
```

Then run the full resumable suite:

```bash
bash scripts/run_compactifai_standard_eval.sh start
bash scripts/run_compactifai_standard_eval.sh status
bash scripts/run_compactifai_standard_eval.sh tail
```

Each task result is cached independently under
`results/evaluations/compactifai-standard/<variant>/<task>.json`. Restarting the
suite skips results whose model/module-set and evaluation signatures still match.
The compact outputs are `summary.json` and `comparison.md` in the same directory.
