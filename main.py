"""
Main entry point for LeetCode Evaluator
"""
import argparse
import sys
import os
import json
from leetcode_evaluator.core.evaluator import LeetCodeEvaluator
from leetcode_evaluator.core.report_generator import ReportGenerator
from leetcode_evaluator.core.config import Config
from leetcode_evaluator.core.aggregation_manager import AggregationManager
from leetcode_evaluator.core.stability_metrics import StabilityAnalyzer


def main():
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
        help='Aggregate all results from results/summary and generate paper-ready tables and plots'
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
    
    args = parser.parse_args()
    
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
            agg_manager = AggregationManager()
            leaderboard_path = agg_manager.aggregate_leaderboard()
            agg_manager.generate_plots()
            print(f"\n✓ Aggregation complete!")
            print(f"✓ Leaderboard saved to: {leaderboard_path}")
            print(f"✓ Figures saved to: {Config.RESULTS_FIGURES}/")
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
            'selection': args.selection
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
            
            import concurrent.futures
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
                    provider=args.provider,
                    model_id=params.get('model', args.model),
                    experiment_name=name
                )
                
                results_file = exp_evaluator.run_evaluation(
                    **base_params,
                    **params
                )
                
                if results_file:
                    print(f"✓ Experiment {name} completed. Results: {results_file}")
                    # Use the same directory name for reports as for experiments
                    exp_dir_name = os.path.basename(exp_evaluator.experiment_manager.experiment_dir)
                    report_dir = os.path.join(Config.REPORTS_DIR, exp_dir_name)
                    
                    # Set matplotlib backend to non-interactive for thread safety
                    import matplotlib
                    matplotlib.use('Agg')
                    
                    generator = ReportGenerator(results_file, output_dir=report_dir)
                    report_file = generator.generate_full_report()
                    print(f"✓ Report generated: {report_file}")
                    
                    # Archive for paper publication
                    agg_manager = AggregationManager()
                    analyzer = StabilityAnalyzer(detailed_jsonl_path=exp_evaluator.experiment_manager.jsonl_path)
                    metrics = analyzer.compute_metrics()
                    
                    config_data = {
                        'model_name': exp_evaluator.llm_client.model_id,
                        'prompt_type': name, # Use experiment name as prompt type for batch
                        'temperature': params.get('temperature', Config.MODEL_TEMPERATURE),
                        'top_p': params.get('top_p', Config.MODEL_TOP_P)
                    }
                    agg_manager.save_experiment_summary(name, metrics, config_data)
                    agg_manager.copy_raw_data(exp_evaluator.experiment_manager)
                    return name, True
                else:
                    print(f"✗ Experiment {name} failed.")
                    return name, False
            
            # Run experiments in parallel with staggered start to avoid rate limits
            max_workers = min(len(experiments), Config.WORKER_THREADS)
            print(f"\n🚀 Running {len(experiments)} experiments in parallel with {max_workers} workers...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, exp in enumerate(experiments):
                    # Stagger start times to avoid rate limits
                    if i > 0:
                        time.sleep(5)
                    future = executor.submit(run_single_experiment, exp, i, args, base_params)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        name, success = future.result()
                        results.append((name, success))
                    except Exception as e:
                        print(f"✗ Experiment failed with error: {str(e)}")
                        results.append(("unknown", False))
            
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
            
            # Generate report
            print("\nGenerating comprehensive report...")
            # Use the same directory name for reports as for experiments
            exp_dir_name = os.path.basename(evaluator.experiment_manager.experiment_dir)
            report_dir = os.path.join(Config.REPORTS_DIR, exp_dir_name)
            
            generator = ReportGenerator(results_file, output_dir=report_dir)
            report_file = generator.generate_full_report()
            report_dir = os.path.dirname(report_file)
            
            print(f"\n{'='*60}")
            print("✓ Tasks completed successfully!")
            print(f"{'='*60}")
            print(f"Results: {results_file}")
            print(f"Report: {report_file}")
            print(f"Visualizations: {report_dir}/")
            print(f"{'='*60}")
            
            # Archive for paper publication
            agg_manager = AggregationManager()
            analyzer = StabilityAnalyzer(detailed_jsonl_path=evaluator.experiment_manager.jsonl_path)
            metrics = analyzer.compute_metrics()
            
            config_data = {
                'model_name': evaluator.llm_client.model_id,
                'prompt_type': 'standard', # Default prompt type
                'temperature': args.temperature or Config.MODEL_TEMPERATURE,
                'top_p': args.top_p or Config.MODEL_TOP_P
            }
            exp_name = os.path.basename(evaluator.experiment_manager.experiment_dir)
            agg_manager.save_experiment_summary(exp_name, metrics, config_data)
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
