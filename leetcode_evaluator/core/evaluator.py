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


class LeetCodeEvaluator:
    """Main evaluator for comparing AI-generated solutions"""

    def __init__(self, provider: str = None, model_id: str = None):
        self.leetcode_client = LeetCodeClient()
        self.llm_client = LLMClientFactory.create_client(
            provider=provider, model_id=model_id)
        # Store the actual provider being used
        self.provider = provider or Config.LLM_PROVIDER
        self.results = []

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
            evaluation = self._evaluate_single_solution(
                problem,
                use_detailed_prompt=True
            )
            if evaluation:
                result['with_prompt'].append(evaluation)
                print(f"    ✓ Status: {evaluation['status']}")
                if evaluation.get('runtime_percentile'):
                    print(
                        f"    Runtime: {evaluation['runtime_percentile']:.1f}th percentile")
            time.sleep(2)  # Rate limiting

        # Evaluate without detailed prompt
        print("\n[2/2] Evaluating WITHOUT detailed prompt...")
        for attempt in range(attempts):
            print(f"  Attempt {attempt + 1}/{attempts}")
            evaluation = self._evaluate_single_solution(
                problem,
                use_detailed_prompt=False
            )
            if evaluation:
                result['without_prompt'].append(evaluation)
                print(f"    ✓ Status: {evaluation['status']}")
                if evaluation.get('runtime_percentile'):
                    print(
                        f"    Runtime: {evaluation['runtime_percentile']:.1f}th percentile")
            time.sleep(2)  # Rate limiting

        return result

    def _evaluate_single_solution(self, problem: Dict,
                                  use_detailed_prompt: bool) -> Optional[Dict]:
        """
        Generate and evaluate a single solution

        Args:
            problem: Problem dictionary
            use_detailed_prompt: Whether to use detailed prompt

        Returns:
            Evaluation results or None if failed
        """
        # Generate solution
        code = self.llm_client.generate_solution(
            problem,
            use_detailed_prompt=use_detailed_prompt
        )

        if not code:
            return {
                'status': 'Generation Failed',
                'error': 'Failed to generate code'
            }

        # Validate syntax
        if not self.llm_client.validate_code_syntax(code):
            return {
                'status': 'Syntax Error',
                'error': 'Invalid Python syntax',
                'code': code
            }

        # Submit to LeetCode (REST API requires question_id)
        submission_id = self.leetcode_client.submit_solution(
            title_slug=problem['title_slug'],
            code=code,
            question_id=problem['question_id']
        )

        if not submission_id:
            return {
                'status': 'Submission Failed',
                'error': 'Failed to submit to LeetCode',
                'code': code
            }

        # Check submission result
        result = self.leetcode_client.check_submission(submission_id)

        if result:
            result['code'] = code
            result['prompt_type'] = 'detailed' if use_detailed_prompt else 'minimal'

        return result

    def evaluate_batch(self, problems: List[Dict], attempts: int = 1) -> List[Dict]:
        """
        Evaluate multiple problems

        Args:
            problems: List of problem dictionaries
            attempts: Number of attempts per approach per problem

        Returns:
            List of evaluation results
        """
        results = []

        print(
            f"\nEvaluating {len(problems)} problems with {attempts} attempt(s) each")
        print(f"Total evaluations: {len(problems) * attempts * 2}")

        for i, problem in enumerate(tqdm(problems, desc="Problems")):
            print(f"\nProgress: {i+1}/{len(problems)}")

            try:
                result = self.evaluate_problem(problem, attempts)
                results.append(result)

                # Save intermediate results
                self._save_results(results)

            except Exception as e:
                print(f"✗ Error evaluating {problem['title']}: {str(e)}")
                results.append({
                    'problem_id': problem['question_id'],
                    'title': problem['title'],
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

            time.sleep(3)  # Rate limiting between problems

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
