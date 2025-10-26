"""
Main entry point for LeetCode Evaluator
"""
import argparse
import sys
from evaluator import LeetCodeEvaluator
from report_generator import ReportGenerator
from config import Config


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
        
        # Full evaluation mode
        print("Starting LeetCode evaluation...")
        print(f"Configuration:")
        print(f"  - Problems: {args.num_problems}")
        print(f"  - Difficulty: {args.difficulty or 'All'}")
        print(f"  - Attempts per approach: {args.attempts}")
        provider = args.provider or Config.LLM_PROVIDER
        model = args.model or getattr(Config, f"{provider.upper()}_MODEL_ID", "default")
        print(f"  - Provider: {provider}")
        print(f"  - Model: {model}")
        print()
        
        # Run evaluation
        results_file = evaluator.run_evaluation(
            num_problems=args.num_problems,
            difficulty=args.difficulty,
            attempts=args.attempts
        )
        
        if not results_file:
            print("✗ Evaluation failed")
            return 1
        
        # Generate report
        print("\nGenerating comprehensive report...")
        generator = ReportGenerator(results_file)
        report_file = generator.generate_full_report()
        
        print(f"\n{'='*60}")
        print("✓ All tasks completed successfully!")
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
