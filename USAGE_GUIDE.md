# LeetCode AI Evaluator - Quick Start Guide

## Setup (5 minutes)

### Step 1: Install Dependencies

```bash
cd leetcode_evaluator
pip install -r requirements.txt
```

### Step 2: Configure Credentials

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
LEETCODE_USERNAME=your_leetcode_username
LEETCODE_PASSWORD=your_leetcode_password
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
```

### Step 3: Test Installation

```bash
python main.py --help
```

## Basic Usage

### Example 1: Quick Test (5 Easy Problems)

```bash
python main.py --num-problems 5 --difficulty EASY --attempts 1
```

**What happens:**
1. Logs into LeetCode
2. Fetches 5 easy problems
3. For each problem:
   - Generates solution WITH detailed prompt
   - Submits to LeetCode and gets result
   - Generates solution WITHOUT detailed prompt
   - Submits to LeetCode and gets result
4. Generates comprehensive report with metrics
5. Creates visualizations

**Expected output:**
- `results/evaluation_results_TIMESTAMP.json`
- `reports/analysis_report_TIMESTAMP.md`
- Multiple PNG visualization files

### Example 2: Medium Problems with Multiple Attempts

```bash
python main.py --num-problems 10 --difficulty MEDIUM --attempts 3
```

This runs 3 attempts per approach (with/without prompt) for each problem, giving better statistical significance.

### Example 3: Just Fetch Problems (No Evaluation)

```bash
python main.py --fetch-only --num-problems 20 --difficulty HARD
```

Useful for collecting problems without using API credits.

### Example 4: Generate Report from Existing Results

```bash
python main.py --report results/evaluation_results_20250101_120000.json
```

Re-generate visualizations and analysis from previous run.

## Understanding Your Results

### 1. Results JSON File

Contains raw data for each problem:

```json
{
  "problem_id": "1",
  "title": "Two Sum",
  "difficulty": "Easy",
  "with_prompt": [
    {
      "status": "Accepted",
      "runtime_percentile": 75.2,
      "memory_percentile": 68.5
    }
  ],
  "without_prompt": [
    {
      "status": "Accepted", 
      "runtime_percentile": 52.1,
      "memory_percentile": 45.3
    }
  ]
}
```

### 2. Analysis Report (Markdown)

Key sections:

**Executive Summary**
- Overall pass rates
- Performance comparison
- Key improvements

**Correctness Analysis**
- Pass@k metrics
- Success by difficulty
- Error breakdown

**Performance Analysis**
- Runtime percentiles
- Memory efficiency
- Top 25% rates

**Conclusions**
- Actionable insights
- Recommendations

### 3. Visualizations

**Pass Rate Chart** - Bar chart comparing success rates
**Runtime Distribution** - Histogram of execution times
**Memory Distribution** - Histogram of memory usage
**Runtime vs Memory** - Scatter plot showing correlation
**Error Types** - Breakdown of failure modes
**Topic Heatmap** - Performance by problem category

## Interpreting Metrics

### Pass Rate

```
With Prompt: 68%
Without Prompt: 42%
Improvement: +26 percentage points
```

**Meaning:** Detailed prompts improve success by 26%. Model needs guidance for complex problems.

### Runtime Percentile

```
Mean Runtime Percentile: 63.5%
```

**Meaning:** Solutions are faster than 63.5% of all LeetCode submissions. Above 75% indicates expert-level optimization.

### Statistical Significance

```
P-Value: 0.023
Statistically Significant: Yes
```

**Meaning:** Results are unlikely due to chance (p < 0.05). Improvements are real.

## Common Workflows

### Research Paper Replication

```bash
# 1. Collect problems (same as paper)
python main.py --fetch-only --num-problems 200

# 2. Run evaluation
python main.py --num-problems 200 --attempts 5

# 3. Generate final report
python main.py --report results/evaluation_results_TIMESTAMP.json
```

### Compare Different Models

```bash
# Claude 3.5 Sonnet
python main.py --num-problems 10 --model anthropic.claude-3-5-sonnet-20241022-v2:0

# Claude 3 Haiku (faster, cheaper)
python main.py --num-problems 10 --model anthropic.claude-3-haiku-20240307-v1:0

# Compare reports
```

### Difficulty-Specific Analysis

```bash
# Test on easy problems
python main.py --num-problems 20 --difficulty EASY

# Test on hard problems
python main.py --num-problems 20 --difficulty HARD

# Compare pass rates across difficulties
```

## Expected Performance

Based on research paper results:

| Difficulty | Expected Pass Rate | Runtime Percentile |
|------------|-------------------|-------------------|
| Easy | 70-85% | 55-70% |
| Medium | 40-55% | 50-65% |
| Hard | 15-30% | 45-60% |

Your actual results will vary based on:
- Model version
- Prompt quality
- Problem selection
- Random variation

## Troubleshooting

### Issue: Login Failed

**Solution:**
1. Verify credentials in `.env`
2. Try manual login at leetcode.com
3. Check for 2FA (not supported)

### Issue: Bedrock Access Denied

**Solution:**
1. Verify AWS credentials
2. Check IAM permissions for Bedrock
3. Ensure model is available in your region

### Issue: Submission Timeout

**Solution:**
1. Increase timeout in `config.py`
2. Check internet connectivity
3. LeetCode may be under maintenance

### Issue: Too Many Requests

**Solution:**
1. Reduce `--num-problems`
2. Increase delays in `config.py`
3. Wait before retrying

## Tips for Best Results

1. **Start Small**: Test with 5 problems first
2. **Use Easy Problems**: Higher success rate for initial validation
3. **Multiple Attempts**: Use `--attempts 3` for better statistics
4. **Save Results**: Keep JSON files for later analysis
5. **Monitor Costs**: Each problem costs 2 API calls (with/without prompt)

## Cost Estimation

**Per Problem:**
- 2 Bedrock API calls (~8K tokens total)
- 2 LeetCode submissions
- ~$0.02-0.05 depending on model

**For 100 Problems:**
- ~$2-5 total cost
- ~30-60 minutes runtime
- Comprehensive dataset

## Next Steps

After running evaluation:

1. Review markdown report
2. Analyze visualizations
3. Identify patterns in failures
4. Adjust prompts if needed
5. Re-run on different problems
6. Compare across models
7. Write up findings

## Advanced Usage

### Custom Prompts

Edit `config.py` to modify:
- `DETAILED_PROMPT` - Full optimization guidance
- `MINIMAL_PROMPT` - Baseline comparison

### Custom Metrics

Extend `report_generator.py`:
- Add new calculations
- Create custom visualizations
- Export to different formats

### Batch Processing

```python
from evaluator import LeetCodeEvaluator

evaluator = LeetCodeEvaluator()
evaluator.initialize()

# Evaluate multiple difficulty levels
for difficulty in ['EASY', 'MEDIUM', 'HARD']:
    evaluator.run_evaluation(
        num_problems=50,
        difficulty=difficulty,
        attempts=3
    )
```

## Support

Questions? Issues?

1. Check README.md for detailed docs
2. Review config.py for settings
3. Examine output JSON for raw data
4. Verify environment variables
5. Test with smaller datasets first

---

**Happy Evaluating! 🚀**
