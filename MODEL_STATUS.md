# Model Status

Current model status summary for the `leetcode_evaluator` pipeline.

## Fully usable now

These models have completed at least one real end-to-end run:
- code generation
- LeetCode submission
- result collection

### Realtime evaluation

- `qwen3-coder-plus`
  - Realtime evaluation works.
  - 10-problem run completed successfully.
  - Best-performing Qwen-family model tested so far.

- `qwen-max`
  - Realtime evaluation works.
  - 10-problem run completed successfully.
  - Usable, but weaker than `qwen3-coder-plus` on the current 10-problem slice.

- `qwen-plus-2025-07-28`
  - Realtime evaluation works.
  - Verified on Easy / Medium / Hard single-problem checks.

- `qwen-plus`
  - Realtime evaluation works.
  - Qwen batch generation path also works end-to-end.

### Qwen batch evaluation

- `qwen-plus`
  - Confirmed working with Alibaba Batch API.
  - Verified flow:
    - batch job created
    - batch job completed
    - generated outputs normalized
    - LeetCode submissions executed sequentially
    - results/report produced

## Smoke-test usable

These models completed the `qwen` client smoke test on `EASY + 1 problem`.
This means the provider/client path works, but they have not yet been validated with a larger evaluation slice.

- `qwen-coder-turbo-0919`
- `qwen3-vl-235b-a22b-thinking`
- `glm-4.5-air`
- `qwen3-8b`
- `opennlu-v1`
- `qwen3-coder-flash`
- `tongyi-xiaomi-analysis-flash`
- `deepseek-r1-distill-qwen-7b`
- `MiniMax-M2.1`

## Not recommended currently

These models have already shown clear problems in the current setup.

### Realtime path problems

- `qwen3.5-35b-a3b`
  - Hung in generation stage under the `qwen` provider.

- `qwen3.5-flash-2026-02-23`
  - Hung in generation stage under the `qwen` provider.

- `deepseek-r1-distill-qwen-32b`
  - Timed out in smoke test.

### Batch API not supported

- `qwen3-coder-plus`
  - Realtime path works.
  - Alibaba Batch API rejected the model as unsupported.

## Batch API support notes

Based on current validation:

- Confirmed batch-compatible:
  - `qwen-plus`

- Confirmed batch-incompatible:
  - `qwen3-coder-plus`

Known likely batch-supported families to test next:
- `qwen-max`
- `qwen-plus`
- `qwen-flash`
- `qwen-turbo`
- `qwen-long`
- `qwq-plus`
- `qwq-32b-preview`
- `deepseek-r1`
- `deepseek-v3`

## Recommended next choices

If the goal is paper data using the current realtime evaluator:

1. `qwen3-coder-plus`
2. `qwen-max`
3. `glm-4.5-air`

If the goal is lower-cost batch generation using Alibaba Batch API:

1. `qwen-plus`
2. `qwen-max`
3. `deepseek-v3`
