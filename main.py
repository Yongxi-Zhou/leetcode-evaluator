"""
Main entry point for LeetCode Evaluator
"""
import argparse
import sys
import os
import json
import hashlib
from datetime import datetime
from leetcode_evaluator.core.evaluator import LeetCodeEvaluator
from leetcode_evaluator.core.report_generator import ReportGenerator
from leetcode_evaluator.core.config import Config
from leetcode_evaluator.core.aggregation_manager import AggregationManager
from leetcode_evaluator.core.stability_metrics import StabilityAnalyzer
from leetcode_evaluator.core.qwen_batch_runner import QwenBatchRunner
from leetcode_evaluator.core.bedrock_batch_runner import BedrockBatchRunner
from leetcode_evaluator.core.gemini_batch_runner import GeminiBatchRunner
from leetcode_evaluator.core.azure_batch_runner import AzureBatchRunner
from leetcode_evaluator.core.openai_batch_runner import OpenAIBatchRunner
from leetcode_evaluator.core.anthropic_batch_runner import AnthropicBatchRunner


def main():
    def build_bedrock_batch_run_id(config_path: str) -> str:
        abs_path = os.path.abspath(config_path)
        with open(abs_path, 'rb') as f:
            content = f.read()
        digest = hashlib.sha1(abs_path.encode('utf-8') + b'\0' + content).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
        return f"bedrock_batch_{safe_stem}_{digest}"

    def build_openai_batch_run_id(config_path: str) -> str:
        abs_path = os.path.abspath(config_path)
        with open(abs_path, 'rb') as f:
            content = f.read()
        digest = hashlib.sha1(content).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
        return f"openai_batch_{safe_stem}_{digest}"

    def build_azure_batch_run_id(config_path: str) -> str:
        abs_path = os.path.abspath(config_path)
        with open(abs_path, 'rb') as f:
            content = f.read()
        digest = hashlib.sha1(content).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
        return f"azure_batch_{safe_stem}_{digest}"

    def build_anthropic_batch_run_id(config_path: str) -> str:
        abs_path = os.path.abspath(config_path)
        with open(abs_path, 'rb') as f:
            content = f.read()
        digest = hashlib.sha1(content).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
        return f"anthropic_batch_{safe_stem}_{digest}"

    def build_gemini_batch_run_id(config_path: str) -> str:
        abs_path = os.path.abspath(config_path)
        with open(abs_path, 'rb') as f:
            content = f.read()
        digest = hashlib.sha1(content).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
        return f"gemini_batch_{safe_stem}_{digest}"

    def build_qwen_batch_run_id(config_path: str) -> str:
        abs_path = os.path.abspath(config_path)
        with open(abs_path, 'rb') as f:
            content = f.read()
        # Hash only file content (not path) so the same config produces the same
        # run_id regardless of working directory or environment (local vs Docker).
        digest = hashlib.sha1(content).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
        return f"qwen_batch_{safe_stem}_{digest}"

    parser = argparse.ArgumentParser(
        description='LeetCode AI Solution Evaluator - Compare prompted vs non-prompted solutions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate 5 easy problems with 1 attempt each
  python main.py --num-problems 5 --difficulty EASY --attempts 1
  
  # Evaluate 10 medium problems with 3 attempts each
  python main.py --num-problems 10 --difficulty MEDIUM --attempts 3
  
  # Generate report from existing results
  python main.py --report results/evaluation_results_20250101_120000.json
  
  # Fetch problems only (no evaluation)
  python main.py --fetch-only --num-problems 20 --difficulty HARD

  # Create Qwen batch jobs only
  python main.py --qwen-batch-config experiments/paper_qwen_batch.json --qwen-batch-mode submit

  # Collect finished outputs without submitting to LeetCode
  python main.py --qwen-batch-config experiments/paper_qwen_batch.json --qwen-batch-mode collect

  # Check Qwen batch status later from local artifacts
  python main.py --qwen-batch-config experiments/paper_qwen_batch.json --qwen-batch-mode status

  # Resume from ready models only and submit sequentially
  python main.py --qwen-batch-config experiments/paper_qwen_batch.json --qwen-batch-mode resume
        """
    )
    
    # Evaluation options
    parser.add_argument(
        '--num-problems',
        type=int,
        default=5,
        help='Number of problems to evaluate (default: 5)'
    )
    
    parser.add_argument(
        '--difficulty',
        choices=['EASY', 'MEDIUM', 'HARD'],
        help='Filter problems by difficulty'
    )
    
    parser.add_argument(
        '--attempts',
        type=int,
        default=1,
        help='Number of attempts per approach (default: 1)'
    )
    
    parser.add_argument(
        '--num-trials',
        type=int,
        dest='attempts',
        help='Alias for --attempts, number of evaluation trials per problem'
    )
    
    parser.add_argument(
        '--fetch-only',
        action='store_true',
        help='Only fetch problems without evaluation'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='main',
        help='Dataset selector: e.g. main, test, or a direct JSON path'
    )

    parser.add_argument(
        '--selection',
        choices=['LATEST', 'RANDOM'],
        default='LATEST',
        help='Selection strategy for problems (default: LATEST)'
    )
    
    # Report generation
    parser.add_argument(
        '--report',
        type=str,
        help='Generate report from existing results file'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate aggregated tables and plots from archived summaries; combine with --qwen-batch-config or --summary-dir to scope the aggregation'
    )
    parser.add_argument(
        '--summary-dir',
        type=str,
        help='Optional summary directory to aggregate instead of the global archive'
    )
    
    # LLM configuration
    parser.add_argument(
        '--provider',
        type=str,
        choices=['bedrock', 'openai', 'gemini', 'grok', 'qwen', 'openrouter'],
        help='LLM provider to use (default: from config or bedrock)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Override model ID from config'
    )
    
    # New Model Parameters
    parser.add_argument(
        '--temperature',
        type=float,
        help='LLM temperature override'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        help='LLM Top-P override'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='LLM Max Tokens override'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=Config.WORKER_THREADS,
        help=f'Number of concurrent workers for LLM generation (default: {Config.WORKER_THREADS})'
    )
    
    parser.add_argument(
        '--stability-runs',
        type=int,
        default=Config.DEFAULT_STABILITY_RUNS,
        help=f'Number of repeated runs per problem for stability analysis (default: {Config.DEFAULT_STABILITY_RUNS})'
    )
    
    # Batch Experiment Runner
    parser.add_argument(
        '--experiment-config',
        type=str,
        help='Path to a JSON file defining multiple experiments'
    )

    parser.add_argument(
        '--qwen-batch-config',
        type=str,
        help='Path to a JSON file defining one or more Qwen batch-generation jobs'
    )

    parser.add_argument(
        '--qwen-batch-mode',
        choices=['submit', 'collect', 'resume', 'status'],
        default='resume',
        help='Qwen batch lifecycle mode: submit only, collect outputs only, resume ready models, or local status-only (default: resume)'
    )

    parser.add_argument(
        '--batch-models',
        type=str,
        nargs='+',
        help='Filter: only process these model IDs (e.g. --batch-models qwen-max qwq-plus)'
    )

    parser.add_argument(
        '--batch-num-problems',
        type=int,
        default=None,
        help='Limit number of problems submitted to LeetCode (for testing, e.g. --batch-num-problems 3)'
    )

    parser.add_argument(
        '--batch-num-trials',
        type=int,
        default=None,
        help='Limit number of stability trials per problem (for testing, e.g. --batch-num-trials 3)'
    )

    parser.add_argument(
        '--batch-prompts',
        type=str,
        nargs='+',
        choices=['detailed', 'minimal'],
        help='Filter: only process these prompt types (e.g. --batch-prompts detailed)'
    )

    parser.add_argument(
        '--batch-problem-offset',
        type=int,
        default=None,
        help='Skip first N problems before submitting (for splitting 100 problems across accounts, e.g. --batch-problem-offset 50)'
    )

    parser.add_argument(
        '--skip-premium',
        action='store_true',
        default=False,
        help='Skip problems marked isPaidOnly=true in the dataset'
    )

    parser.add_argument(
        '--qwen-batch-run-id',
        type=str,
        default=None,
        help='Override the auto-generated run ID (useful when resuming across environments, e.g. --qwen-batch-run-id qwen_batch_paper_qwen_batch_608ffde791)'
    )

    parser.add_argument(
        '--bedrock-batch-config',
        type=str,
        help='Path to a JSON file defining one or more AWS Bedrock batch-generation jobs'
    )

    parser.add_argument(
        '--bedrock-batch-mode',
        choices=['submit', 'collect', 'resume', 'status'],
        default='resume',
        help='Bedrock batch lifecycle mode: submit only, collect outputs only, resume ready models, or status-only (default: resume)'
    )

    parser.add_argument(
        '--gemini-batch-config',
        type=str,
        help='Path to a JSON file defining one or more Gemini (Vertex AI) batch-generation jobs'
    )

    parser.add_argument(
        '--gemini-batch-mode',
        choices=['submit', 'collect', 'resume', 'status'],
        default='resume',
        help='Gemini batch lifecycle mode: submit only, collect outputs only, resume ready models, or status-only (default: resume)'
    )

    parser.add_argument(
        '--gemini-batch-run-id',
        type=str,
        default=None,
        help='Override the auto-generated Gemini batch run ID'
    )

    parser.add_argument(
        '--openai-batch-config',
        type=str,
        help='Path to a JSON file defining one or more OpenAI batch-generation jobs'
    )

    parser.add_argument(
        '--openai-batch-mode',
        choices=['submit', 'collect', 'resume', 'status'],
        default='resume',
        help='OpenAI batch lifecycle mode (default: resume)'
    )

    parser.add_argument(
        '--openai-batch-run-id',
        type=str,
        default=None,
        help='Override the auto-generated OpenAI batch run ID'
    )

    parser.add_argument(
        '--azure-batch-config',
        type=str,
        help='Path to a JSON file defining one or more Azure OpenAI batch-generation jobs'
    )

    parser.add_argument(
        '--azure-batch-mode',
        choices=['submit', 'collect', 'resume', 'status'],
        default='resume',
        help='Azure batch lifecycle mode: submit only, collect outputs only, resume ready models, or status-only (default: resume)'
    )

    parser.add_argument(
        '--azure-batch-run-id',
        type=str,
        default=None,
        help='Override the auto-generated Azure batch run ID'
    )

    parser.add_argument(
        '--anthropic-batch-config',
        type=str,
        help='Path to a JSON file defining one or more Anthropic batch-generation jobs'
    )

    parser.add_argument(
        '--anthropic-batch-mode',
        choices=['submit', 'collect', 'resume', 'status'],
        default='resume',
        help='Anthropic batch lifecycle mode (default: resume)'
    )

    parser.add_argument(
        '--anthropic-batch-run-id',
        type=str,
        default=None,
        help='Override the auto-generated Anthropic batch run ID'
    )

    args = parser.parse_args()
    
    # Generate run ID for new experiments
    # For --report or --generate-report, we might want to skip this or handle it differently
    if not args.report and not args.generate_report:
        if args.qwen_batch_config:
            run_id = args.qwen_batch_run_id or build_qwen_batch_run_id(args.qwen_batch_config)
        elif args.bedrock_batch_config:
            run_id = build_bedrock_batch_run_id(args.bedrock_batch_config)
        elif args.gemini_batch_config:
            run_id = args.gemini_batch_run_id or build_gemini_batch_run_id(args.gemini_batch_config)
        elif args.openai_batch_config:
            run_id = args.openai_batch_run_id or build_openai_batch_run_id(args.openai_batch_config)
        elif args.azure_batch_config:
            run_id = args.azure_batch_run_id or build_azure_batch_run_id(args.azure_batch_config)
        elif args.anthropic_batch_config:
            run_id = args.anthropic_batch_run_id or build_anthropic_batch_run_id(args.anthropic_batch_config)
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model:
            run_id += f"_{args.model}"
        Config.set_run_id(run_id)

    try:
        # Report generation mode
        if args.report:
            print("Generating report from existing results...")
            generator = ReportGenerator(args.report)
            report_file = generator.generate_full_report()
            report_dir = os.path.dirname(report_file)
            print(f"\n✓ Report generation complete!")
            print(f"✓ Report saved to: {report_file}")
            print(f"✓ Visualizations saved to: {report_dir}/")
            return 0
            
        # Global aggregation mode
        if args.generate_report:
            print("🚀 Generating paper-ready aggregated report...")
            summary_dirs = None
            raw_dirs = None
            tables_dir = None
            figures_dir = None

            if args.summary_dir:
                summary_dirs = [args.summary_dir]

            if args.qwen_batch_config:
                batch_run_id = build_qwen_batch_run_id(args.qwen_batch_config)
                batch_output_root = os.path.join("output", batch_run_id)
                summary_dirs = [os.path.join(batch_output_root, "results", "summary")]
                raw_dirs = [os.path.join(batch_output_root, "results", "raw")]
                tables_dir = os.path.join(batch_output_root, "results", "tables")
                print(f"  Scope: qwen batch run `{batch_run_id}`")
                print(f"  Summary dir: {summary_dirs[0]}")
                print(f"  Raw dir: {raw_dirs[0]}")
            elif args.bedrock_batch_config:
                batch_run_id = build_bedrock_batch_run_id(args.bedrock_batch_config)
                batch_output_root = os.path.join("output", batch_run_id)
                summary_dirs = [os.path.join(batch_output_root, "results", "summary")]
                raw_dirs = [os.path.join(batch_output_root, "results", "raw")]
                tables_dir = os.path.join(batch_output_root, "results", "tables")
                print(f"  Scope: bedrock batch run `{batch_run_id}`")
                print(f"  Summary dir: {summary_dirs[0]}")
                print(f"  Raw dir: {raw_dirs[0]}")
            elif args.gemini_batch_config:
                batch_run_id = args.gemini_batch_run_id or build_gemini_batch_run_id(args.gemini_batch_config)
                batch_output_root = os.path.join("output", batch_run_id)
                summary_dirs = [os.path.join(batch_output_root, "results", "summary")]
                raw_dirs = [os.path.join(batch_output_root, "results", "raw")]
                tables_dir = os.path.join(batch_output_root, "results", "tables")
                print(f"  Scope: gemini batch run `{batch_run_id}`")
                print(f"  Summary dir: {summary_dirs[0]}")
                print(f"  Raw dir: {raw_dirs[0]}")
            elif summary_dirs:
                print(f"  Scope: explicit summary dir `{summary_dirs[0]}`")

            agg_manager = AggregationManager(
                summary_dirs=summary_dirs,
                raw_dirs=raw_dirs,
                aggregate_tables_dir=tables_dir,
                aggregate_figures_dir=figures_dir,
            )
            dataset_stats_path = agg_manager.generate_dataset_stats(args.dataset)
            leaderboard_path = agg_manager.aggregate_leaderboard()
            agg_manager.generate_plots()
            print(f"\n✓ Aggregation complete!")
            print(f"✓ Dataset stats saved to: {dataset_stats_path}")
            print(f"✓ Leaderboard saved to: {leaderboard_path}")
            print(f"✓ Tables saved to: {agg_manager.aggregate_tables_dir}/")
            print(f"✓ Figures saved to: {agg_manager.aggregate_figures_dir}/")
            return 0

        if args.qwen_batch_config:
            print(f"🚀 Running Qwen batch generation from: {args.qwen_batch_config}")
            print(f"Mode: {args.qwen_batch_mode}")
            batch_runner = QwenBatchRunner(args.qwen_batch_config, run_id)
            batch_results = batch_runner.run(
                mode=args.qwen_batch_mode,
                filter_models=args.batch_models,
                filter_prompts=args.batch_prompts,
                num_problems=args.batch_num_problems,
                num_trials=args.batch_num_trials,
                problem_offset=args.batch_problem_offset,
                skip_premium=args.skip_premium,
            )
            print(f"\n{'='*60}")
            if args.qwen_batch_mode == 'submit':
                print("✓ Qwen batch jobs submitted!")
                print("Use --qwen-batch-mode collect to download finished outputs later.")
                print("Use --qwen-batch-mode status to inspect local artifacts and ready models.")
                print("Use --qwen-batch-mode resume to submit ready models to LeetCode.")
            elif args.qwen_batch_mode == 'collect':
                print("✓ Qwen batch output collection completed!")
            elif args.qwen_batch_mode == 'status':
                print("✓ Qwen batch status check completed!")
            else:
                print("✓ Qwen batch resume completed!")
            print(f"{'='*60}")
            for result in batch_results:
                print(f"Model: {result['model_id']}")
                print(f"Results: {result['results_file']}")
                print(f"Report: {result['report_file']}")
                print("-" * 60)
            return 0

        if args.bedrock_batch_config:
            print(f"🚀 Running Bedrock batch generation from: {args.bedrock_batch_config}")
            print(f"Mode: {args.bedrock_batch_mode}")
            batch_runner = BedrockBatchRunner(args.bedrock_batch_config, run_id)
            batch_results = batch_runner.run(
                mode=args.bedrock_batch_mode,
                filter_models=args.batch_models,
                filter_prompts=args.batch_prompts,
            )
            print(f"\n{'='*60}")
            if args.bedrock_batch_mode == 'submit':
                print("✓ Bedrock batch jobs submitted!")
                print("Use --bedrock-batch-mode collect to download finished outputs later.")
                print("Use --bedrock-batch-mode status to inspect local artifacts and ready models.")
                print("Use --bedrock-batch-mode resume to submit ready models to LeetCode.")
            elif args.bedrock_batch_mode == 'collect':
                print("✓ Bedrock batch output collection completed!")
            elif args.bedrock_batch_mode == 'status':
                print("✓ Bedrock batch status check completed!")
            else:
                print("✓ Bedrock batch resume completed!")
            print(f"{'='*60}")
            for result in batch_results:
                print(f"Model: {result['model_id']}")
                print(f"Results: {result['results_file']}")
                print(f"Report: {result['report_file']}")
                print("-" * 60)
            return 0

        if args.gemini_batch_config:
            print(f"Running Gemini batch generation from: {args.gemini_batch_config}")
            print(f"Mode: {args.gemini_batch_mode}")
            batch_runner = GeminiBatchRunner(args.gemini_batch_config, run_id)
            batch_results = batch_runner.run(
                mode=args.gemini_batch_mode,
                filter_models=args.batch_models,
                filter_prompts=args.batch_prompts,
                num_problems=args.batch_num_problems,
                num_trials=args.batch_num_trials,
                problem_offset=args.batch_problem_offset,
                skip_premium=args.skip_premium,
            )
            print(f"\n{'='*60}")
            if args.gemini_batch_mode == 'submit':
                print("Gemini batch jobs submitted!")
                print("Use --gemini-batch-mode collect to download finished outputs later.")
                print("Use --gemini-batch-mode status to inspect local artifacts and ready models.")
                print("Use --gemini-batch-mode resume to submit ready models to LeetCode.")
            elif args.gemini_batch_mode == 'collect':
                print("Gemini batch output collection completed!")
            elif args.gemini_batch_mode == 'status':
                print("Gemini batch status check completed!")
            else:
                print("Gemini batch resume completed!")
            print(f"{'='*60}")
            for result in batch_results:
                print(f"Model: {result['model_id']}")
                print(f"Results: {result['results_file']}")
                print(f"Report: {result['report_file']}")
                print("-" * 60)
            return 0

        if args.openai_batch_config:
            print(f"Running OpenAI batch generation from: {args.openai_batch_config}")
            print(f"Mode: {args.openai_batch_mode}")
            batch_runner = OpenAIBatchRunner(args.openai_batch_config, run_id)
            batch_results = batch_runner.run(
                mode=args.openai_batch_mode,
                filter_models=args.batch_models,
                filter_prompts=args.batch_prompts,
                num_problems=args.batch_num_problems,
                num_trials=args.batch_num_trials,
                problem_offset=args.batch_problem_offset,
                skip_premium=args.skip_premium,
            )
            print(f"\n{'='*60}")
            if args.openai_batch_mode == 'submit':
                print("OpenAI batch jobs submitted!")
                print("Use --openai-batch-mode collect to download finished outputs later.")
                print("Use --openai-batch-mode resume to submit ready models to LeetCode.")
            elif args.openai_batch_mode == 'collect':
                print("OpenAI batch output collection completed!")
            elif args.openai_batch_mode == 'status':
                print("OpenAI batch status check completed!")
            else:
                print("OpenAI batch resume completed!")
            print(f"{'='*60}")
            for result in batch_results:
                print(f"Model: {result['model_id']}\nResults: {result['results_file']}\nReport: {result['report_file']}\n{'-'*60}")
            return 0

        if args.azure_batch_config:
            print(f"Running Azure OpenAI batch generation from: {args.azure_batch_config}")
            print(f"Mode: {args.azure_batch_mode}")
            batch_runner = AzureBatchRunner(args.azure_batch_config, run_id)
            batch_results = batch_runner.run(
                mode=args.azure_batch_mode,
                filter_models=args.batch_models,
                filter_prompts=args.batch_prompts,
                num_problems=args.batch_num_problems,
                num_trials=args.batch_num_trials,
                problem_offset=args.batch_problem_offset,
                skip_premium=args.skip_premium,
            )
            print(f"\n{'='*60}")
            if args.azure_batch_mode == 'submit':
                print("Azure batch jobs submitted!")
                print("Use --azure-batch-mode collect to download finished outputs later.")
                print("Use --azure-batch-mode status to inspect local artifacts and ready models.")
                print("Use --azure-batch-mode resume to submit ready models to LeetCode.")
            elif args.azure_batch_mode == 'collect':
                print("Azure batch output collection completed!")
            elif args.azure_batch_mode == 'status':
                print("Azure batch status check completed!")
            else:
                print("Azure batch resume completed!")
            print(f"{'='*60}")
            for result in batch_results:
                print(f"Model: {result['model_id']}")
                print(f"Results: {result['results_file']}")
                print(f"Report: {result['report_file']}")
                print("-" * 60)
            return 0

        if args.anthropic_batch_config:
            print(f"Running Anthropic batch generation from: {args.anthropic_batch_config}")
            print(f"Mode: {args.anthropic_batch_mode}")
            batch_runner = AnthropicBatchRunner(args.anthropic_batch_config, run_id)
            batch_results = batch_runner.run(
                mode=args.anthropic_batch_mode,
                filter_models=args.batch_models,
                filter_prompts=args.batch_prompts,
                num_problems=args.batch_num_problems,
                num_trials=args.batch_num_trials,
                problem_offset=args.batch_problem_offset,
                skip_premium=args.skip_premium,
            )
            print(f"\n{'='*60}")
            if args.anthropic_batch_mode == 'submit':
                print("Anthropic batch jobs submitted!")
                print("Use --anthropic-batch-mode collect to download finished outputs later.")
                print("Use --anthropic-batch-mode resume to submit ready models to LeetCode.")
            elif args.anthropic_batch_mode == 'collect':
                print("Anthropic batch output collection completed!")
            elif args.anthropic_batch_mode == 'status':
                print("Anthropic batch status check completed!")
            else:
                print("Anthropic batch resume completed!")
            print(f"{'='*60}")
            for result in batch_results:
                print(f"Model: {result['model_id']}\nResults: {result['results_file']}\nReport: {result['report_file']}\n{'-'*60}")
            return 0

        # Initialize evaluator with provider and model
        evaluator = LeetCodeEvaluator(
            provider=args.provider,
            model_id=args.model
        )
        
        # Fetch only mode
        if args.fetch_only:
            if not evaluator.initialize():
                print("✗ Initialization failed")
                return 1
            
            problems = evaluator.fetch_problems(
                count=args.num_problems,
                difficulty=args.difficulty,
                selection=args.selection
            )
            
            print(f"\n✓ Successfully fetched {len(problems)} problems")
            return 0
        
        # Run evaluation (Batch or Single)
        
        # Collect base parameters
        base_params = {
            'num_problems': args.num_problems,
            'difficulty': args.difficulty,
            'attempts': args.attempts,
            'stability_runs': args.stability_runs,
            'workers': args.workers,
            'selection': args.selection,
            'dataset': args.dataset
        }
        
        # Collect model overrides
        run_params = {
            'temperature': args.temperature,
            'top_p': args.top_p,
            'max_tokens': args.max_tokens
        }
        
        if args.experiment_config:
            # Batch Mode
            print(f"🚀 Loading batch experiments from: {args.experiment_config}")
            with open(args.experiment_config, 'r') as f:
                experiments = json.load(f)
                
            print(f"Found {len(experiments)} experiments to run.")
            
            import time
            
            def run_single_experiment(exp, i, args, base_params):
                name = exp.get('name', f"exp_{i}")
                params = exp.get('params', {})
                
                print(f"\n{'-'*60}")
                print(f"STARTING EXPERIMENT {i+1}/{len(experiments)}: {name}")
                print(f"Parameters: {params}")
                print(f"{'-'*60}")
                
                # Re-initialize evaluator for each experiment
                exp_evaluator = LeetCodeEvaluator(
                    provider=params.get('provider', args.provider),
                    model_id=params.get('model', args.model),
                    experiment_name=name
                )
                
                results_file = exp_evaluator.run_evaluation(
                    **base_params,
                    **params
                )
                
                if results_file:
                    print(f"✓ Experiment {name} completed. Results: {results_file}")
                    agg_manager = AggregationManager()
                    analyzer = StabilityAnalyzer(detailed_jsonl_path=exp_evaluator.experiment_manager.jsonl_path)
                    prompt_metrics = analyzer.compute_metrics_by_strategy()

                    for prompt_type, metrics in prompt_metrics.items():
                        config_data = {
                            'model_name': exp_evaluator.llm_client.model_id,
                            'prompt_type': prompt_type,
                            'temperature': params.get('temperature', Config.MODEL_TEMPERATURE),
                            'top_p': params.get('top_p', Config.MODEL_TOP_P),
                            'max_tokens': params.get('max_tokens', Config.MODEL_MAX_TOKENS),
                            'provider': params.get('provider', args.provider or Config.LLM_PROVIDER),
                            'dataset': base_params.get('dataset'),
                            'num_problems': base_params.get('num_problems'),
                            'stability_runs': base_params.get('stability_runs'),
                            'generation_mode': 'realtime',
                        }
                        agg_manager.save_experiment_summary(
                            f"{name}_{prompt_type}",
                            metrics,
                            config_data
                        )
                    agg_manager.copy_raw_data(exp_evaluator.experiment_manager)

                    # Use the same directory name for reports as for experiments
                    exp_dir_name = os.path.basename(exp_evaluator.experiment_manager.experiment_dir)
                    report_dir = os.path.join(Config.REPORTS_DIR, exp_dir_name)
                    
                    # Set matplotlib backend to non-interactive for thread safety
                    import matplotlib
                    matplotlib.use('Agg')
                    
                    generator = ReportGenerator(results_file, output_dir=report_dir)
                    report_file = generator.generate_full_report()
                    print(f"✓ Report generated: {report_file}")
                    # Save summary data for report generator and global aggregation
                    agg_manager = AggregationManager()
                    analyzer = StabilityAnalyzer(detailed_jsonl_path=exp_evaluator.experiment_manager.jsonl_path)
                    metrics = analyzer.compute_metrics()
                    
                    config_data = {
                        'experiment_name': name,
                        'model_name': exp_evaluator.llm_client.model_id,
                        'prompt_type': name, # Use experiment name as prompt type for batch
                        'temperature': params.get('temperature', Config.MODEL_TEMPERATURE),
                        'top_p': params.get('top_p', Config.MODEL_TOP_P),
                        'max_tokens': params.get('max_tokens', Config.MODEL_MAX_TOKENS),
                        'num_problems': base_params.get('num_problems'),
                        'attempts': base_params.get('attempts'),
                        'stability_runs': base_params.get('stability_runs'),
                        'workers': base_params.get('workers'),
                        'selection': base_params.get('selection'),
                        'dataset': base_params.get('dataset'),
                        'generation_mode': 'realtime',
                    }
                    agg_manager.save_experiment_summary(exp_dir_name, metrics, config_data)
                    agg_manager.copy_raw_data(exp_evaluator.experiment_manager)

                    # Set matplotlib backend to non-interactive for thread safety
                    import matplotlib
                    matplotlib.use('Agg')
                    
                    generator = ReportGenerator(
                        results_file, output_dir=report_dir,
                        model_name=exp_evaluator.llm_client.model_id,
                        provider=args.provider)
                    report_file = generator.generate_full_report()
                    print(f"✓ Report generated: {report_file}")
                    
                    return name, True
                else:
                    print(f"✗ Experiment {name} failed.")
                    return name, False
            
            # Run experiments sequentially to keep LeetCode-side rate limiting from
            # contaminating model/config comparisons.
            print(f"\n🚀 Running {len(experiments)} experiments sequentially...")
            results = []
            for i, exp in enumerate(experiments):
                if i > 0:
                    time.sleep(5)
                try:
                    name, success = run_single_experiment(exp, i, args, base_params)
                    results.append((name, success))
                except Exception as e:
                    exp_name = exp.get('name', f"exp_{i}")
                    print(f"✗ Experiment {exp_name} failed with error: {str(e)}")
                    results.append((exp_name, False))
            
            # Summary
            successful = sum(1 for _, success in results if success)
            failed = len(results) - successful
            
            print(f"\n{'='*60}")
            print(f"✓ Batch experiments completed!")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"{'='*60}")
            return 0 if failed == 0 else 1
            
        else:
            # Single Evaluation Mode
            print("Starting LeetCode evaluation...")
            print(f"Configuration:")
            print(f"  - Problems: {args.num_problems}")
            print(f"  - Dataset: {args.dataset}")
            print(f"  - Difficulty: {args.difficulty or 'All'}")
            print(f"  - Selection: {args.selection}")
            print(f"  - Base Attempts/Stability Runs: {max(args.attempts, args.stability_runs)}")
            
            if run_params:
                # Filter out None values from run_params for display
                display_params = {k: v for k, v in run_params.items() if v is not None}
                if display_params:
                    print(f"  - Overrides: {display_params}")
                
            # Run evaluation
            results_file = evaluator.run_evaluation(
                **base_params,
                **run_params
            )
            
            if not results_file:
                print("✗ Evaluation failed")
                return 1
            
            # Archive for paper publication and report generation
            exp_name_base = os.path.basename(evaluator.experiment_manager.experiment_dir)
            agg_manager = AggregationManager()
            analyzer = StabilityAnalyzer(detailed_jsonl_path=evaluator.experiment_manager.jsonl_path)
            metrics = analyzer.compute_metrics()
            
            config_data = {
                'experiment_name': exp_name_base,
                'model_name': evaluator.llm_client.model_id,
                'prompt_type': 'standard',
                'temperature': args.temperature or Config.MODEL_TEMPERATURE,
                'top_p': args.top_p or Config.MODEL_TOP_P,
                'max_tokens': args.max_tokens or Config.MODEL_MAX_TOKENS,
                'num_problems': args.num_problems,
                'attempts': args.attempts,
                'stability_runs': args.stability_runs,
                'workers': args.workers,
                'selection': args.selection,
                'dataset': args.dataset,
                'generation_mode': 'realtime',
            }
            agg_manager.save_experiment_summary(exp_name_base, metrics, config_data)
            agg_manager.copy_raw_data(evaluator.experiment_manager)

            # Generate report
            print("\nGenerating comprehensive report...")
            # Use the same directory name for reports as for experiments
            report_dir = os.path.join(Config.REPORTS_DIR, exp_name_base)
            
            generator = ReportGenerator(
                results_file, output_dir=report_dir,
                model_name=evaluator.llm_client.model_id,
                provider=args.provider)
            report_file = generator.generate_full_report()
            report_dir = os.path.dirname(report_file)
            
            print(f"\n{'='*60}")
            print("✓ Tasks completed successfully!")
            print(f"{'='*60}")
            print(f"Results: {results_file}")
            print(f"Report: {report_file}")
            print(f"Visualizations: {report_dir}/")
            print(f"{'='*60}")
            
            agg_manager = AggregationManager()
            analyzer = StabilityAnalyzer(detailed_jsonl_path=evaluator.experiment_manager.jsonl_path)
            exp_name = os.path.basename(evaluator.experiment_manager.experiment_dir)
            prompt_metrics = analyzer.compute_metrics_by_strategy()
            for prompt_type, metrics in prompt_metrics.items():
                config_data = {
                    'model_name': evaluator.llm_client.model_id,
                    'prompt_type': prompt_type,
                    'temperature': args.temperature or Config.MODEL_TEMPERATURE,
                    'top_p': args.top_p or Config.MODEL_TOP_P,
                    'max_tokens': args.max_tokens or Config.MODEL_MAX_TOKENS,
                    'provider': args.provider or Config.LLM_PROVIDER,
                    'dataset': args.dataset,
                    'num_problems': args.num_problems,
                    'stability_runs': args.stability_runs,
                    'generation_mode': 'realtime',
                }
                agg_manager.save_experiment_summary(
                    f"{exp_name}_{prompt_type}",
                    metrics,
                    config_data
                )
            agg_manager.copy_raw_data(evaluator.experiment_manager)
            
            return 0
        
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
