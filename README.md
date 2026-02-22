# LeetCode AI Solution Evaluator

A comprehensive evaluation framework for comparing AI-generated LeetCode solutions with and without detailed prompts. This tool evaluates solutions using AWS Bedrock models and provides detailed performance analysis.

## Features

- **Problem Fetching**: Automatically fetch LeetCode problems by difficulty
- **Dual Evaluation**: Compare solutions generated with detailed vs minimal prompts
- **Automated Submission**: Submit solutions to LeetCode and retrieve results
- **Comprehensive Metrics**: 
  - Pass@k success rates
  - Runtime percentile rankings
  - Memory usage analysis
  - Error breakdown by type
  - Topic-specific performance
- **Statistical Analysis**: T-tests for significance, effect sizes
- **Rich Visualizations**: Charts, graphs, and heatmaps
- **Detailed Reports**: Markdown reports with actionable insights

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

# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Bedrock Model ID (optional, default: Claude 3.5 Sonnet)
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
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
                          Filter problems by difficulty
  --attempts INT          Number of attempts per approach (default: 1)
  --fetch-only            Only fetch problems, don't evaluate
  --report FILE           Generate report from existing results
  --model MODEL_ID        Override Bedrock model ID
  --help                  Show this help message
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

#### 4. Use Different Bedrock Model

```bash
python main.py --num-problems 5 --model anthropic.claude-3-haiku-20240307-v1:0
```

## Output Files

The tool generates several output files:

### Results Directory (`results/`)
- `problems_TIMESTAMP.json` - Fetched problem data
- `evaluation_results_TIMESTAMP.json` - Detailed evaluation results

### Reports Directory (`reports/`)
- `analysis_report_TIMESTAMP.md` - Comprehensive markdown report
- `pass_rate_by_difficulty_TIMESTAMP.png` - Success rate visualization
- `runtime_distribution_TIMESTAMP.png` - Runtime percentile histogram
- `memory_distribution_TIMESTAMP.png` - Memory usage histogram
- `runtime_vs_memory_TIMESTAMP.png` - Performance scatter plot
- `error_types_TIMESTAMP.png` - Error breakdown chart
- `topic_performance_TIMESTAMP.png` - Topic-specific heatmap

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
