# LeetCode AI Solution Evaluator

A comprehensive evaluation framework for comparing AI-generated LeetCode solutions with and without detailed prompts. This tool evaluates solutions using AWS Bedrock models and provides detailed performance analysis.

## Features

- **Experiment Archiving**: Every run is saved in a timestamped folder with full metrics and code solutions
- **Batch Experiment Runner**: Run multiple configurations (temp, top-p, etc.) sequentially via JSON config
- **Model Parameters**: Fine-grained control over Temperature, Top-P, and Max Tokens
- **Multi-Provider Support**: AWS Bedrock, OpenAI, Google Gemini, and xAI Grok
- **Problem Fetching**: Automatically fetch LeetCode problems by difficulty
- **Dual Evaluation**: Compare solutions generated with detailed vs minimal prompts
- **Comprehensive Metrics**: 
  - Token usage & cost estimation
  - Pass@k success rates
  - Runtime & Memory percentile rankings
- **Rich Visualizations**: Charts, graphs, and performance heatmaps

## Installation

### 1. Install Dependencies

```bash
cd leetcode_evaluator
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# LeetCode Credentials
LEETCODE_USERNAME=your_username
LEETCODE_PASSWORD=your_password

# LLM Provider Configuration
LLM_PROVIDER=bedrock  # bedrock, openai, gemini, grok

# AWS Bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL_ID=gpt-4o

# Model Parameters (Defaults)
MODEL_TEMPERATURE=0.3
MODEL_TOP_P=0.9
MODEL_MAX_TOKENS=4096

# Concurrency & Throttling
WORKER_THREADS=4
LEETCODE_SUBMISSION_DELAY_S=10
LEETCODE_RATE_LIMIT_COOLDOWN_S=60
```

## Usage

### Quick Start

Evaluate 5 easy problems:

```bash
python main.py --num-problems 5 --difficulty EASY --attempts 1
```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --num-problems INT       Number of problems to evaluate (default: 5)
  --difficulty EASY|MEDIUM|HARD
  --attempts INT          Number of attempts per approach (default: 1)
  --temperature FLOAT     LLM temperature override
  --max-tokens INT        LLM max tokens override
  --workers INT           Number of concurrent LLM generative workers (default: 4)
  --experiment-config FILE Run multiple experiments defined in a JSON file
  --report FILE           Generate report from existing results
  --provider PROVIDER     bedrock|openai|gemini|grok
  --model MODEL_ID        Override model ID
```

### Examples

#### 1. Evaluate 10 Medium Problems

```bash
python main.py --num-problems 10 --difficulty MEDIUM --attempts 2
```

#### 2. Fetch Problems Only

```bash
python main.py --fetch-only --num-problems 20 --difficulty HARD
```

#### 3. Generate Report from Existing Results

```bash
python main.py --report results/evaluation_results_20250101_120000.json
```

#### 4. Batch Experiment Runner

Run multiple configurations sequentially:
```bash
python main.py --experiment-config experiments.json
```

**experiments.json example:**
```json
[
  { "name": "baseline", "params": { "temperature": 0.1 } },
  { "name": "creative", "params": { "temperature": 0.8 } }
]
```

## Output Files

The tool generates several output files:

### Experiment Directory (`experiments/YYYYMMDD_HHMMSS/`)
- `detailed.jsonl` - Raw LLM responses, tokens, and metadata
- `summary.csv` - High-level metrics (Pass Rate, Cost, Latency)
- `solutions.md` - Formatted code blocks of all generated solutions
- `execution.log` - Application logs for the run

### Reports Directory (`reports/`)
- `analysis_report_TIMESTAMP.md` - Comprehensive performance breakdown
- `*.png` - Visualizations (Success rates, Runtime/Memory distributions)

## Metrics Explained

### Correctness Metrics

- **Pass@1**: Success rate on first attempt
- **Pass@k**: Success rate within k attempts
- **Pass Rate by Difficulty**: Success rates for Easy/Medium/Hard problems
- **Error Breakdown**: Distribution of syntax, runtime, and logical errors

### Performance Metrics

- **Runtime Percentile**: How fast your solution is compared to all submissions (higher is better)
- **Memory Percentile**: How memory-efficient your solution is (higher is better)
- **Top 25% Rate**: Percentage of solutions in the top performance quartile

### Comparison Metrics

- **Pass Rate Improvement**: Percentage improvement with detailed prompts
- **Runtime Improvement**: Performance difference between approaches
- **Statistical Significance**: T-test results for statistical validity

## Understanding the Results

### Sample Report Summary

```
Overall Pass Rate (With Prompt): 68.0%
Overall Pass Rate (Without Prompt): 42.0%
Improvement: +26.0 percentage points

Mean Runtime Percentile (With Prompt): 63.5%
Mean Runtime Percentile (Without Prompt): 48.2%
```

### Interpretation

- **Positive Pass Rate Improvement**: Detailed prompts help the model solve more problems
- **Higher Runtime Percentile**: Solutions are faster than average
- **Top 25% Rate**: Indicates production-ready performance level

## Architecture

The project has been reorganized into a structured package for better maintainability:

```
leetcode-evaluator/
├── main.py                    # CLI entry point
├── leetcode_evaluator/        # Main package
│   ├── core/                  # Core logic and configuration
│   │   ├── config.py          # Settings and environment variables
│   │   ├── evaluator.py       # Main evaluation logic
│   │   └── report_generator.py # Metrics and visualizations
│   └── clients/               # External service clients
│       ├── leetcode.py        # LeetCode API client
│       └── llm/               # LLM provider clients
│           ├── base.py        # Abstract base client
│           ├── bedrock.py     # AWS Bedrock client
│           ├── gemini.py      # Google Gemini client
│           ├── grok.py        # xAI Grok client
│           └── openai.py      # OpenAI client
├── tests/                     # Unit and integration tests
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (not in git)
└── .env.example               # Example environment file
```

## Workflow

1. **Fetch Problems**: Query LeetCode API for problems
2. **Generate Solutions**: 
   - With detailed prompt (optimization hints)
   - Without detailed prompt (minimal guidance)
3. **Submit & Verify**: Submit to LeetCode and poll for results
4. **Analyze Results**: Calculate metrics and generate visualizations
5. **Generate Report**: Create comprehensive markdown report

## Limitations

- **Rate Limiting**: LeetCode may rate limit frequent submissions
- **Premium Problems**: Only free problems are evaluated
- **Language**: Currently supports Python 3 only
- **Authentication**: Requires valid LeetCode account

## Troubleshooting

### Login Failed

- Verify credentials in `.env` file
- Check if LeetCode account is active
- Try logging in manually on leetcode.com first

### AWS Bedrock Errors

- Verify AWS credentials have Bedrock access
- Check if model ID is available in your region
- Ensure sufficient quota for API calls

### Submission Timeout

- Increase `SUBMISSION_TIMEOUT` in config.py
- Check network connectivity
- Verify LeetCode API is accessible

## Research Use Case

This tool is designed based on the methodology in the research paper on LLM performance for solving LeetCode problems. It enables:

1. Reproducible experiments with controlled conditions
2. Quantitative comparison of different prompt strategies
3. Statistical validation of prompt engineering effectiveness
4. Comprehensive performance benchmarking

## Contributing

Contributions are welcome! Areas for improvement:

- Support for more programming languages
- Additional metrics and visualizations
- Integration with more LLM providers
- Enhanced error handling and recovery

## License

This project is intended for research and educational purposes.

## Citation

If you use this tool in your research, please cite the original paper:

```
Performance Review on LLM for solving leetcode problems
Wang et al., 2025
arXiv:2502.15770v2
```

## Support

For issues and questions:
- Check the troubleshooting section
- Review LeetCode API documentation
- Verify AWS Bedrock configuration
- Check environment variable setup

---

**Note**: This tool respects LeetCode's terms of service. Use responsibly and avoid excessive API calls that may impact the platform.
