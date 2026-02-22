"""
Main entry point for LeetCode Evaluator
"""
import argparse
import sys
from leetcode_evaluator.core.evaluator import LeetCodeEvaluator
from leetcode_evaluator.core.report_generator import ReportGenerator
from leetcode_evaluator.core.config import Config


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
        '--fetch-only',
        action='store_true',
        help='Only fetch problems without evaluation'
    )
    
    # Report generation
    parser.add_argument(
        '--report',
        type=str,
        help='Generate report from existing results file'
    )
    
    # LLM configuration
    parser.add_argument(
        '--provider',
        type=str,
        choices=['bedrock', 'openai', 'gemini', 'grok'],
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
            print(f"\n✓ Report generation complete!")
            print(f"✓ Report saved to: {report_file}")
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
                difficulty=args.difficulty
            )
            
            print(f"\n✓ Successfully fetched {len(problems)} problems")
            return 0
        
        # Run evaluation (Batch or Single)
        import json
        
        # Collect base parameters
        base_params = {
            'num_problems': args.num_problems,
            'difficulty': args.difficulty,
            'attempts': args.attempts,
            'stability_runs': args.stability_runs,
            'workers': args.workers
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
            
            for i, exp in enumerate(experiments):
                name = exp.get('name', f"exp_{i}")
                params = exp.get('params', {})
                
                print(f"\n{'-'*60}")
                print(f"RUNNING EXPERIMENT {i+1}/{len(experiments)}: {name}")
                print(f"Parameters: {params}")
                print(f"{'-'*60}")
                
                # Re-initialize evaluator for each experiment
                exp_evaluator = LeetCodeEvaluator(
                    provider=args.provider,
                    model_id=args.model,
                    experiment_name=name
                )
                
                results_file = exp_evaluator.run_evaluation(
                    **base_params,
                    **params
                )
                
                if results_file:
                    print(f"✓ Experiment {name} completed. Results: {results_file}")
                    # Optional: generate report for each
                    generator = ReportGenerator(results_file)
                    generator.generate_full_report()
                else:
                    print(f"✗ Experiment {name} failed.")
            
            print(f"\n{'='*60}")
            print("✓ All batch experiments completed!")
            print(f"{'='*60}")
            return 0
            
        else:
            # Single Evaluation Mode
            print("Starting LeetCode evaluation...")
            print(f"Configuration:")
            print(f"  - Problems: {args.num_problems}")
            print(f"  - Difficulty: {args.difficulty or 'All'}")
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
            generator = ReportGenerator(results_file)
            report_file = generator.generate_full_report()
            
            print(f"\n{'='*60}")
            print("✓ Tasks completed successfully!")
            print(f"{'='*60}")
            print(f"Results: {results_file}")
            print(f"Report: {report_file}")
            print(f"Visualizations: {Config.REPORT_DIR}/")
            print(f"{'='*60}")
            
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
