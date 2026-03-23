# LeetCode LLM Evaluator

A research evaluation framework for benchmarking LLM performance on LeetCode problems. Supports large-scale batch generation (Qwen/DashScope, AWS Bedrock, and Google Vertex AI) followed by automated LeetCode submission, with paper-quality metrics (RLPR, FPA, PSR, AV + 95% CI).

## Features

- **Multi-provider support**: OpenRouter, OpenAI, Gemini, AWS Bedrock, Qwen/DashScope
- **Batch inference**: Async batch generation via DashScope, AWS Bedrock, and Google Vertex AI batch APIs
- **Dual-prompt evaluation**: Compares `detailed` vs `minimal` prompts on every problem
- **Stability runs**: Multiple trials per problem for PSR and variance metrics
- **Resumable pipeline**: Submit → Collect → Resume across machines/sessions
- **Aggregated reporting**: Cross-model leaderboard tables and paper-ready figures

## Setup

```bash
cd leetcode_evaluator
pip install -r requirements.txt
cp .env.example .env   # fill in credentials
```

Key `.env` variables:

```env
# LeetCode (get from browser DevTools → Application → Cookies → leetcode.com)
LEETCODE_SESSION_COOKIE=<full cookie string>
LEETCODE_CSRF_TOKEN=<csrftoken value>

# Provider (pick one)
LLM_PROVIDER=openrouter        # recommended for multi-model
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL_ID=openai/gpt-4o-mini

# Qwen batch
DASHSCOPE_API_KEY=sk-...
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# AWS Bedrock batch
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
BEDROCK_BATCH_S3_BUCKET=your-bucket
BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::ACCOUNT:role/BedrockBatchRole

# Gemini batch (Vertex AI)
GCP_PROJECT=your-gcp-project-id
GCP_LOCATION=us-central1
GEMINI_BATCH_GCS_BUCKET=your-gcs-bucket

# Throttling
LEETCODE_SUBMISSION_DELAY_S=10   # seconds between submissions
```

## Quick Start

```bash
# Sanity check: 1 problem, 1 trial
python main.py --provider openrouter --model openai/gpt-4o-mini --num-problems 1

# Standard run: 10 problems, 3 stability runs
python main.py --num-problems 10 --stability-runs 3 --workers 2

# Generate aggregated leaderboard from all finished runs
python main.py --generate-report
```

## Batch Experiment Workflow

Large-scale paper experiments use async batch APIs to generate solutions offline, then submit to LeetCode separately. The pipeline has three phases:

```
submit  →  collect  →  resume
 (batch API)   (download)   (LeetCode submission)
```

### Qwen Batch (DashScope)

Config file: `experiments/paper_qwen_batch.json`

```json
{
  "dataset": "main",
  "num_problems": 100,
  "stability_runs": 5,
  "prompt_types": ["detailed", "minimal"],
  "models": ["qwen-max", "qwq-plus", "qwen-plus", "qwen-turbo"],
  "temperature": 0.3,
  "top_p": 0.9,
  "max_tokens": 4096
}
```

```bash
# Phase 1: Submit all batch jobs (exits immediately, does not wait)
python main.py --qwen-batch-config experiments/paper_qwen_batch.json --qwen-batch-mode submit

# Phase 2: Download completed outputs and normalize (run after jobs finish)
python main.py --qwen-batch-config experiments/paper_qwen_batch.json --qwen-batch-mode collect

# Check local status without calling the API
python main.py --qwen-batch-config experiments/paper_qwen_batch.json --qwen-batch-mode status

# Phase 3: Submit normalized solutions to LeetCode and generate reports
python main.py --qwen-batch-config experiments/paper_qwen_batch.json --qwen-batch-mode resume
```

### AWS Bedrock Batch

Config file: `experiments/paper_bedrock_batch.json`

```bash
python main.py --bedrock-batch-config experiments/paper_bedrock_batch.json --bedrock-batch-mode submit
python main.py --bedrock-batch-config experiments/paper_bedrock_batch.json --bedrock-batch-mode collect
python main.py --bedrock-batch-config experiments/paper_bedrock_batch.json --bedrock-batch-mode resume
```

### Gemini Batch (Vertex AI)

Config file: `experiments/paper_gemini_batch.json`

```json
{
  "dataset": "main",
  "num_problems": 100,
  "stability_runs": 5,
  "prompt_types": ["detailed", "minimal"],
  "models": [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.0-flash-001"
  ],
  "temperature": 0.3,
  "top_p": 0.9,
  "max_tokens": 4096
}
```

Requires GCP setup: a project with Vertex AI API enabled, a GCS bucket, and `gcloud` auth (`gcloud auth application-default login` or a service account key).

```bash
# Phase 1: Submit batch prediction jobs to Vertex AI
python main.py --gemini-batch-config experiments/paper_gemini_batch.json --gemini-batch-mode submit

# Phase 2: Download completed outputs from GCS
python main.py --gemini-batch-config experiments/paper_gemini_batch.json --gemini-batch-mode collect

# Check local status
python main.py --gemini-batch-config experiments/paper_gemini_batch.json --gemini-batch-mode status

# Phase 3: Submit to LeetCode and generate reports
python main.py --gemini-batch-config experiments/paper_gemini_batch.json --gemini-batch-mode resume
```

### Filtering by Model

Use `--batch-models` to process only a subset of models:

```bash
# Resume only qwen-max
python main.py --qwen-batch-config experiments/paper_qwen_batch.json \
  --qwen-batch-mode resume --batch-models qwen-max

# Resume multiple models
python main.py --qwen-batch-config experiments/paper_qwen_batch.json \
  --qwen-batch-mode resume --batch-models qwen-plus qwen-turbo
```

> **Do not use `--batch-prompts` to split by prompt type.** Each model's `detailed` and `minimal` results are paired together during LeetCode submission — splitting them would leave `with_prompt` or `without_prompt` empty, breaking all comparison metrics (RLPR, FPA, etc.) and producing incomplete result files.

### Multi-Person Distribution

To distribute LeetCode submission work across multiple people (each with their own account), each person runs `resume` with `--batch-models` pointing to their assigned models. The `output/batch_jobs/` directory must be synced to each machine first (e.g. via `rsync` or shared network drive):

```bash
# Person A — qwen-max (~3h, 1000 submissions)
python main.py --qwen-batch-config experiments/paper_qwen_batch.json \
  --qwen-batch-mode resume --batch-models qwen-max

# Person B — qwq-plus (~3h, 1000 submissions)
python main.py --qwen-batch-config experiments/paper_qwen_batch.json \
  --qwen-batch-mode resume --batch-models qwq-plus

# Person C — qwen-plus (~3h, 1000 submissions)
python main.py --qwen-batch-config experiments/paper_qwen_batch.json \
  --qwen-batch-mode resume --batch-models qwen-plus

# Person D — qwen-turbo (~3h, 1000 submissions)
python main.py --qwen-batch-config experiments/paper_qwen_batch.json \
  --qwen-batch-mode resume --batch-models qwen-turbo
```

After all runs complete, sync `output/aggregate/` back to one machine and generate the combined report:

```bash
python main.py --generate-report
```

### Running in Background

```bash
nohup caffeinate -i python3 main.py \
  --qwen-batch-config experiments/paper_qwen_batch.json \
  --qwen-batch-mode resume \
  --batch-models qwen-max \
  > qwen_resume_qwen_max.log 2>&1 &

tail -f qwen_resume_qwen_max.log
```

> Note: `caffeinate -i` prevents idle sleep but closing the laptop lid can still suspend the machine.

### Pipeline Notes

- `collect` is the only phase that calls the batch API after `submit`
- `status` reads local artifacts only — no API calls
- `resume` automatically runs `collect` first, then submits ready models
- A model is "ready" only when both `detailed` and `minimal` normalized outputs exist
- Re-running any phase is safe — results are cached and skipped if already done
- The `run_id` is derived from the config file path + content hash, so the same config always maps to the same output directory

## Output Structure

```
output/
├── batch_jobs/{run_id}/       # Batch artifacts per run
│   └── {model_id}/
│       ├── evaluation_state.json      # Model-level state machine
│       ├── detailed/
│       │   ├── input.jsonl            # Batch requests sent to API
│       │   ├── requests_manifest.json # custom_id → problem metadata
│       │   ├── batch_job.json         # API job metadata (job_id/arn)
│       │   ├── output.jsonl           # Raw batch responses
│       │   └── normalized_generations.json  # Extracted + validated code
│       └── minimal/
│           └── (same structure)
├── aggregate/                 # Cross-run, cross-model results
│   ├── summary/               # Per-run JSON summaries
│   ├── tables/leaderboard.md  # Paper-ready leaderboard
│   └── figures/               # Paper-ready plots
├── results/                   # Per-run evaluation JSONs
├── reports/                   # Per-run markdown reports
└── experiments/               # Per-run JSONL trial data
```

## CLI Reference

```
python main.py [OPTIONS]

Core options:
  --num-problems INT        Problems to evaluate (default: 5)
  --dataset NAME|FILE       Dataset: main, test, or a JSON path
  --difficulty EASY|MEDIUM|HARD
  --stability-runs INT      Trials per problem (default: 1)
  --workers INT             Concurrent LLM workers (default: 4)
  --provider PROVIDER       openrouter|openai|gemini|bedrock|qwen
  --model MODEL_ID          Override model ID

Batch options:
  --qwen-batch-config FILE      Qwen batch experiment config JSON
  --qwen-batch-mode MODE        submit|collect|resume|status (default: resume)
  --bedrock-batch-config FILE   Bedrock batch experiment config JSON
  --bedrock-batch-mode MODE     submit|collect|resume|status (default: resume)
  --gemini-batch-config FILE    Gemini (Vertex AI) batch experiment config JSON
  --gemini-batch-mode MODE      submit|collect|resume|status (default: resume)
  --batch-models MODEL [...]    Filter: only process these model IDs

Reporting:
  --generate-report             Aggregate all finished runs into leaderboard
  --report FILE                 Generate report from a specific results JSON
```

## Architecture

```
leetcode_evaluator/
├── main.py                         # CLI entry point
└── leetcode_evaluator/
    ├── core/
    │   ├── config.py               # All settings (env vars, prompts, pricing)
    │   ├── evaluator.py            # LLM generation + LeetCode submission
    │   ├── qwen_batch_runner.py    # Qwen batch pipeline (submit/collect/resume)
    │   ├── bedrock_batch_runner.py # Bedrock batch pipeline
    │   ├── gemini_batch_runner.py  # Gemini/Vertex AI batch pipeline
    │   ├── experiment_manager.py   # Per-run JSONL/CSV logging
    │   ├── aggregation_manager.py  # Cross-run aggregation
    │   ├── report_generator.py     # Metrics, tables, figures
    │   └── stability_metrics.py    # PSR, AV, 95% CI computation
    └── clients/
        ├── leetcode.py             # LeetCode GraphQL API
        └── llm/
            ├── base.py             # Factory + abstract base (code extraction)
            ├── openrouter.py       # OpenRouter
            ├── openai.py           # OpenAI
            ├── gemini.py           # Google Gemini
            ├── bedrock.py          # AWS Bedrock (real-time)
            ├── bedrock_batch.py    # AWS Bedrock batch client
            ├── gemini_batch.py     # Google Vertex AI batch client
            ├── qwen.py             # Qwen/DashScope (real-time)
            └── qwen_batch.py       # Qwen/DashScope batch client
```

## Troubleshooting

**LeetCode auth fails**: Session cookies expire — refresh from browser DevTools → Application → Cookies → leetcode.com. Copy the full cookie string into `LEETCODE_SESSION_COOKIE` and the `csrftoken` value into `LEETCODE_CSRF_TOKEN`.

**Rate limited (429)**: The client auto-retries after `LEETCODE_RATE_LIMIT_COOLDOWN_S` (default 60s). Increase `LEETCODE_SUBMISSION_DELAY_S` to 15-20s for safer operation.

**Qwen batch "Arrearage" error**: DashScope account overdue. Top up balance, delete the `batch_job.json` for the affected model/prompt, and re-run `submit`.

**Bedrock batch "account not authorized"**: Bedrock batch inference (`CreateModelInvocationJob`) requires account-level authorization. Submit an AWS support case to enable it.

**Bedrock `iam:PassRole` denied**: The IAM user needs a `PassRole` policy to pass the batch role to Bedrock. Add an inline policy allowing `iam:PassRole` on the `BEDROCK_BATCH_ROLE_ARN`.

**Gemini batch "Permission denied"**: Ensure the Vertex AI API is enabled in your GCP project (`gcloud services enable aiplatform.googleapis.com`), and authenticate via `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS` to a service account key. The GCS bucket must be in the same project.

**Gemini batch no output files**: Vertex AI writes output to a subdirectory of `output_uri_prefix`. If the download fails, check the GCS bucket at `gs://BUCKET/RUN_ID/MODEL/PROMPT_TYPE/output/` for the actual output path.

## License

Research and educational use only. Please respect LeetCode's terms of service — use conservative submission delays and avoid bulk automated usage.
