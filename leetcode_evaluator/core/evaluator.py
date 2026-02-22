"""
Main evaluation logic for comparing prompted vs non-prompted solutions
"""
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

from leetcode_evaluator.clients.leetcode import LeetCodeClient
from leetcode_evaluator.clients.llm.base import LLMClientFactory
from leetcode_evaluator.core.config import Config
from leetcode_evaluator.core.experiment_manager import ExperimentManager


class LeetCodeEvaluator:
    """Main evaluator for comparing AI-generated solutions"""

    def __init__(self, provider: str = None, model_id: str = None):
        self.leetcode_client = LeetCodeClient()
        self.llm_client = LLMClientFactory.create_client(
            provider=provider, model_id=model_id)
        # Store the actual provider being used
        self.provider = provider or Config.LLM_PROVIDER
        self.results = []
        
        # Initialize Experiment Manager
        self.experiment_manager = ExperimentManager(
            provider=self.provider,
            model_id=self.llm_client.model_id
        )

    def initialize(self) -> bool:
        """Initialize clients and authenticate"""
        print("=" * 60)
        print("LeetCode AI Solution Evaluator")
        print("=" * 60)

        # Validate configuration for the actual provider being used
        try:
            Config.validate_provider(self.provider)
        except ValueError as e:
            print(f"✗ Configuration error: {str(e)}")
            return False

        # Login to LeetCode
        if not self.leetcode_client.login():
            print("✗ Failed to authenticate with LeetCode")
            return False

        print(
            f"✓ Using {self.provider} provider with model: {self.llm_client.model_id}")
        self.experiment_manager.info(f"Initialized evaluator with {self.provider}")
        print()
        return True

    def evaluate_problem(self, problem: Dict, attempts: int = 1) -> Dict:
        """
        Evaluate a single problem with both prompted and non-prompted approaches

        Args:
            problem: Problem dictionary
            attempts: Number of solution attempts per approach

        Returns:
            Evaluation results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Problem: {problem['title']} ({problem['difficulty']})")
        print(f"Topics: {', '.join(problem['topics'][:3])}")
        print(f"{'='*60}")

        result = {
            'problem_id': problem['question_id'],
            'title': problem['title'],
            'title_slug': problem['title_slug'],
            'difficulty': problem['difficulty'],
            'topics': problem['topics'],
            'timestamp': datetime.now().isoformat(),
            'with_prompt': [],
            'without_prompt': []
        }

        # Evaluate with detailed prompt
        print("\n[1/2] Evaluating WITH detailed prompt...")
        for attempt in range(attempts):
            print(f"  Attempt {attempt + 1}/{attempts}")
            eval_data = self._evaluate_single_solution(
                problem,
                use_detailed_prompt=True
            )
            if eval_data:
                result['with_prompt'].append(eval_data)
                print(f"    ✓ Status: {eval_data['status']}")
                if eval_data.get('runtime_percentile'):
                    print(
                        f"    Runtime: {eval_data['runtime_percentile']:.1f}th percentile")
                
                # Log to experiment manager
                self.experiment_manager.log_attempt({
                    'problem': problem['title'],
                    'strategy': 'detailed',
                    'attempt': attempt + 1,
                    **eval_data
                })
                self.experiment_manager.log_solution(
                    problem['title'], 
                    'detailed', 
                    eval_data.get('code')
                )
            time.sleep(2)  # Rate limiting

        # Evaluate without detailed prompt
        print("\n[2/2] Evaluating WITHOUT detailed prompt...")
        for attempt in range(attempts):
            print(f"  Attempt {attempt + 1}/{attempts}")
            eval_data = self._evaluate_single_solution(
                problem,
                use_detailed_prompt=False
            )
            if eval_data:
                result['without_prompt'].append(eval_data)
                print(f"    ✓ Status: {eval_data['status']}")
                if eval_data.get('runtime_percentile'):
                    print(
                        f"    Runtime: {eval_data['runtime_percentile']:.1f}th percentile")
                
                # Log to experiment manager
                self.experiment_manager.log_attempt({
                    'problem': problem['title'],
                    'strategy': 'minimal',
                    'attempt': attempt + 1,
                    **eval_data
                })
                self.experiment_manager.log_solution(
                    problem['title'], 
                    'minimal', 
                    eval_data.get('code')
                )
            time.sleep(2)  # Rate limiting

        return result

    def _evaluate_single_solution(self, problem: Dict,
                                  use_detailed_prompt: bool) -> Optional[Dict]:
        """
        Generate and evaluate a single solution with metadata tracking
        """
        # Generate solution
        gen_result = self.llm_client.generate_solution(
            problem,
            use_detailed_prompt=use_detailed_prompt
        )

        metadata = {
            'input_tokens': gen_result.input_tokens,
            'output_tokens': gen_result.output_tokens,
            'latency_ms': gen_result.latency_ms,
            'cost': gen_result.cost,
            'gen_status': gen_result.status
        }

        if gen_result.status != "Success":
            return {
                'status': 'Generation Failed',
                'error': gen_result.error,
                **metadata
            }

        code = gen_result.code
        if not code:
            return {
                'status': 'Extraction Failed',
                'error': 'Failed to extract code from response',
                'raw_response': gen_result.raw_response,
                **metadata
            }

        # Validate syntax
        if not self.llm_client.validate_code_syntax(code):
            return {
                'status': 'Syntax Error',
                'error': 'Invalid Python syntax',
                'code': code,
                **metadata
            }

        # Submit to LeetCode
        submission_id = self.leetcode_client.submit_solution(
            title_slug=problem['title_slug'],
            code=code,
            question_id=problem['question_id']
        )

        if not submission_id:
            return {
                'status': 'Submission Failed',
                'error': 'Failed to submit to LeetCode',
                'code': code,
                **metadata
            }

        # Check submission result
        result = self.leetcode_client.check_submission(submission_id)

        if result:
            result['code'] = code
            result['prompt_type'] = 'detailed' if use_detailed_prompt else 'minimal'
            result.update(metadata)

        return result

    def evaluate_batch(self, problems: List[Dict], attempts: int = 1) -> List[Dict]:
        """
        Evaluate multiple problems with Circuit Breaker mechanism

        Args:
            problems: List of problem dictionaries
            attempts: Number of attempts per approach per problem

        Returns:
            List of evaluation results
        """
        results = []
        consecutive_errors = 0

        print(
            f"\nEvaluating {len(problems)} problems with {attempts} attempt(s) each")
        print(f"Total evaluations: {len(problems) * attempts * 2}")

        for i, problem in enumerate(tqdm(problems, desc="Problems")):
            print(f"\nProgress: {i+1}/{len(problems)}")

            try:
                result = self.evaluate_problem(problem, attempts)
                results.append(result)
                
                # Check for "Generation Failed" in all attempts
                all_failed = True
                for approach in ['with_prompt', 'without_prompt']:
                    for attempt_res in result.get(approach, []):
                        if attempt_res.get('status') != 'Generation Failed':
                            all_failed = False
                            break
                    if not all_failed:
                        break
                
                if all_failed and (result.get('with_prompt') or result.get('without_prompt')):
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                # Save intermediate results
                self._save_results(results)

            except Exception as e:
                self.experiment_manager.error(f"Error evaluating {problem['title']}: {str(e)}")
                results.append({
                    'problem_id': problem['question_id'],
                    'title': problem['title'],
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                consecutive_errors += 1

            # Circuit Breaker check
            if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                msg = f"CIRCUIT BREAKER TRIGGERED: {consecutive_errors} consecutive failures."
                self.experiment_manager.error(msg)
                print(f"\n{'!'*60}")
                print(f"⚠ {msg}")
                print("Aborting evaluation to prevent token wastage.")
                print(f"{'!'*60}")
                break

            time.sleep(3)  # Rate limiting between problems

        # Save summary CSV
        summary_data = []
        for res in results:
            for strategy_key, strategy_name in [('with_prompt', 'detailed'), ('without_prompt', 'minimal')]:
                for attempt in res.get(strategy_key, []):
                    summary_data.append({
                        'Timestamp': attempt.get('timestamp', datetime.now().isoformat()),
                        'Problem': res.get('title'),
                        'Difficulty': res.get('difficulty'),
                        'Model': self.llm_client.model_id,
                        'Strategy': strategy_name,
                        'Status': attempt.get('status'),
                        'Runtime(ms)': attempt.get('runtime'),
                        'Memory(MB)': attempt.get('memory'),
                        'Gen Time(s)': attempt.get('latency_ms', 0) / 1000,
                        'Tokens(In/Out)': f"{attempt.get('input_tokens', 0)}/{attempt.get('output_tokens', 0)}",
                        'Est Cost($)': attempt.get('cost', 0)
                    })
        
        self.experiment_manager.save_summary(summary_data)
        return results

    def _save_results(self, results: List[Dict], filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{Config.RESULTS_DIR}/evaluation_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def load_results(self, filename: str) -> List[Dict]:
        """Load results from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)

    def fetch_problems(self, count: int = 10, difficulty: str = None) -> List[Dict]:
        """
        Fetch problems from LeetCode

        Args:
            count: Number of problems to fetch
            difficulty: Filter by difficulty (EASY, MEDIUM, HARD)

        Returns:
            List of problem dictionaries
        """
        print(f"\nFetching {count} problems from LeetCode...")
        if difficulty:
            print(f"Difficulty filter: {difficulty}")

        problems = self.leetcode_client.get_random_problems(count, difficulty)

        print(f"✓ Fetched {len(problems)} problems")

        # Save problems to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{Config.RESULTS_DIR}/problems_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(problems, f, indent=2)
        print(f"✓ Problems saved to {filename}")

        return problems

    def run_evaluation(self, num_problems: int = 10, difficulty: str = None,
                       attempts: int = 1) -> str:
        """
        Run complete evaluation workflow

        Args:
            num_problems: Number of problems to evaluate
            difficulty: Optional difficulty filter
            attempts: Number of attempts per approach

        Returns:
            Path to results file
        """
        # Initialize
        if not self.initialize():
            return None

        # Fetch problems
        problems = self.fetch_problems(num_problems, difficulty)

        if not problems:
            print("✗ No problems fetched")
            return None

        # Evaluate
        results = self.evaluate_batch(problems, attempts)

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{Config.RESULTS_DIR}/evaluation_results_{timestamp}.json"
        self._save_results(results, results_file)

        print(f"\n{'='*60}")
        print(f"✓ Evaluation complete!")
        print(f"✓ Results saved to: {results_file}")
        print(f"{'='*60}")

        return results_file


def calculate_pass_at_k(results: List[Dict], k: int = 1) -> float:
    """
    Calculate pass@k metric

    Args:
        results: List of evaluation results for a set of problems
        k: Number of attempts to consider

    Returns:
        Pass@k percentage
    """
    if not results:
        return 0.0

    total_problems = len(results)
    successful_problems = 0

    for result in results:
        # Check if at least one of the first k attempts was successful
        attempts = result[:k] if len(result) >= k else result
        if any(attempt.get('status') == 'Accepted' for attempt in attempts):
            successful_problems += 1

    return (successful_problems / total_problems) * 100
