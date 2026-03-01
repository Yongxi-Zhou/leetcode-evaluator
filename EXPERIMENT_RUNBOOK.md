# LeetCode LLM Evaluation Runbook

## Overview
Evaluate LLM performance on LeetCode problems with two approaches:
1. **OpenRouter (Recommended)** - Single API key, multiple models
2. **Provider-Specific Clients** - Direct API access per provider

## Prerequisites
- Python 3.8+
- LeetCode account (for session cookies)
- API keys for chosen providers

## Setup
1. Clone repository and install dependencies:
```bash
cd leetcode_evaluator
pip install -r requirements.txt
```

2. Configure `.env` file:
```env
# LeetCode authentication
LEETCODE_SESSION_COOKIE=your_session_cookie
LEETCODE_CSRF_TOKEN=your_csrf_token

# OpenRouter (for Approach 1)
OPENROUTER_API_KEY=your_openrouter_key
LLM_PROVIDER=openrouter

# OR Provider-specific (for Approach 2)
# OPENAI_API_KEY=your_openai_key
# GEMINI_API_KEY=your_gemini_key
# GROK_API_KEY=your_grok_key
# DASHSCOPE_API_KEY=your_qwen_key
```

## Approach 1: OpenRouter (Single API Key)

### Step 1: Configure Experiments
Edit `experiments/config.json`:
```json
[
  {
    "name": "gpt-4o-mini",
    "params": {
      "model": "openai/gpt-4o-mini"
    }
  },
  {
    "name": "claude-3-5-sonnet",
    "params": {
      "model": "anthropic/claude-3.5-sonnet"
    }
  }
]
```

### Step 2: Run Batch Experiments
```bash
# Run all experiments in parallel
python main.py --provider openrouter --experiment-config experiments/config.json \
  --num-problems 10 --difficulty MEDIUM --attempts 3

# Options:
# --num-problems N    Number of problems (default: 5)
# --difficulty LEVEL  EASY/MEDIUM/HARD
# --attempts N        Attempts per problem (default: 1)
# --stability-runs N  Repeated runs for stability (default: 1)
```

### Step 3: View Results
Results are saved to:
- `experiments/YYYYMMDD_HHMMSS_*/` - Individual experiment data
- `reports/YYYYMMDD_HHMMSS_*/` - HTML/PDF reports with metrics
- `results/raw/` - Raw JSONL data for analysis

### Step 4: Generate Aggregated Report
```bash
python main.py --generate-report
```
Creates aggregated leaderboard in `results/summary/`

## Approach 2: Provider-Specific Clients

### Step 1: Configure Provider
Set in `.env`:
```env
LLM_PROVIDER=openai  # or gemini, grok, qwen, bedrock
```

### Step 2: Run Single Experiment
```bash
# OpenAI
python main.py --provider openai --model gpt-4o-mini --num-problems 10

# Gemini
python main.py --provider gemini --model gemini-1.5-flash --num-problems 10

# Grok
python main.py --provider grok --model grok-beta --num-problems 10

# Qwen
python main.py --provider qwen --model qwen-plus --num-problems 10
```

### Step 3: Batch Experiments (Multiple Providers)
Create `experiments/multi_provider.json`:
```json
[
  {
    "name": "openai-gpt4o",
    "params": {
      "provider": "openai",
      "model": "gpt-4o"
    }
  },
  {
    "name": "gemini-flash",
    "params": {
      "provider": "gemini",
      "model": "gemini-1.5-flash"
    }
  }
]
```

Run with:
```bash
python main.py --experiment-config experiments/multi_provider.json --num-problems 10
```

## Available Metrics

### 1. **Correctness Metrics**
- Pass rates (with/without prompts)
- Pass@1 (first attempt success)
- Error type breakdown
- Difficulty-specific performance

### 2. **Stability Metrics**
- Perfect stability rate (% problems where all trials pass)
- First-pass accuracy
- Run-level pass rate with 95% CI
- Variance across trials

### 3. **Performance Metrics**
- Runtime percentiles (mean, median, top 25%)
- Memory percentiles
- Latency (p50, p90)

### 4. **Cost & Efficiency**
- Mean tokens (input/output/total)
- Cost per evaluation
- Token usage by prompt type

### 5. **Comparison Metrics**
- Pass rate improvement with prompts
- Model comparison across experiments
- Cost-effectiveness analysis

## Output Files Structure
```
results/
├── evaluations/           # Evaluation results JSON
├── problems/             # Problem metadata
├── raw/                  # Raw JSONL trial data
├── summary/              # Aggregated metrics
└── tables/               # Paper-ready tables

reports/
└── YYYYMMDD_HHMMSS_*/   # Individual experiment reports
    ├── analysis_report.md
    ├── metrics.json
    └── plots/           # Visualizations

experiments/
└── YYYYMMDD_HHMMSS_*/   # Experiment metadata
    ├── summary.csv
    └── detailed.jsonl
```

## Quick Reference

### OpenRouter Model IDs
- `openai/gpt-4o-mini`
- `anthropic/claude-3.5-sonnet`
- `google/gemini-1.5-flash`
- `mistralai/mistral-7b-instruct`
- `meta-llama/llama-3.1-8b-instruct`

### Common Commands
```bash
# Quick test (1 problem)
python main.py --provider openrouter --model openai/gpt-4o-mini --num-problems 1

# Full evaluation (10 problems, 3 attempts)
python main.py --provider openrouter --experiment-config experiments/config.json \
  --num-problems 10 --attempts 3

# Generate report from existing results
python main.py --report results/evaluations/evaluation_results_*.json

# Aggregate all results
python main.py --generate-report
```

## Troubleshooting
- **Rate limiting**: Add `--workers 2` to reduce concurrent requests
- **Authentication errors**: Verify LeetCode cookies in `.env`
- **API errors**: Check API key and model availability
- **Memory issues**: Reduce `--num-problems` or `--workers`
