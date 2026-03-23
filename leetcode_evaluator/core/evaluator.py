import os
import json
import time
import queue
import threading
import concurrent.futures
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

from leetcode_evaluator.clients.leetcode import LeetCodeClient


class RateLimitExhaustedException(RuntimeError):
    """Raised when consecutive LeetCode 429 failures exceed the configured threshold.

    The calling process should exit so a different account can continue.
    """
from leetcode_evaluator.clients.llm.base import LLMClientFactory
from leetcode_evaluator.core.config import Config
from leetcode_evaluator.core.experiment_manager import ExperimentManager
from leetcode_evaluator.core.stability_metrics import StabilityAnalyzer


class LeetCodeEvaluator:
    """Main evaluator for comparing AI-generated solutions"""

    def __init__(self, provider: str = None, model_id: str = None, experiment_name: str = None):
        self.global_rate_limit_pause = threading.Event()
        self.global_rate_limit_pause.set()  # Initial state: Allow submissions
        self.leetcode_client = LeetCodeClient(rate_limit_pause=self.global_rate_limit_pause)
        self.llm_client = LLMClientFactory.create_client(
            provider=provider, model_id=model_id)
        # Store the actual provider being used
        self.provider = provider or Config.LLM_PROVIDER
        self.results = []
        
        # Initialize Experiment Manager
        self.experiment_manager = ExperimentManager(
            provider=self.provider,
            model_id=self.llm_client.model_id,
            experiment_name=experiment_name
        )
        # Stable path for incremental saves and resume (set via run_generated_evaluation)
        self._active_results_file = None
        # Counter for consecutive submission API failures (null submission_id → likely 429)
        self._consecutive_rate_limit_failures = 0

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

    def _evaluate_single_solution(self, problem: Dict,
                                  use_detailed_prompt: bool, **kwargs) -> Optional[Dict]:
        """
        Generate and validate a single solution (part of concurrent workflow)
        """
        # Generate solution
        gen_result = self.llm_client.generate_solution(
            problem,
            use_detailed_prompt=use_detailed_prompt,
            **kwargs
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

        return {
            'status': 'Success',
            'code': code,
            **metadata
        }

    def _producer_worker(self, tasks_queue: queue.Queue, submissions_queue: queue.Queue, 
                         attempts: int, **kwargs):
        """Worker function for LLM generation (Producer)"""
        # Normalize generation params once so downstream logging always has concrete values.
        run_params = dict(kwargs)
        if run_params.get('temperature') is None:
            run_params['temperature'] = Config.MODEL_TEMPERATURE
        if run_params.get('top_p') is None:
            run_params['top_p'] = Config.MODEL_TOP_P
        if run_params.get('max_tokens') is None:
            run_params['max_tokens'] = Config.MODEL_MAX_TOKENS

        while True:
            try:
                problem = tasks_queue.get_nowait()
            except queue.Empty:
                break
            
            # Generate solutions for both strategies
            result = {
                'problem': problem,
                'with_prompt': [],
                'without_prompt': []
            }
            
            # Detailed prompt attempts
            for attempt in range(attempts):
                eval_data = self._evaluate_single_solution(
                    problem, use_detailed_prompt=True, **run_params
                )
                if eval_data:
                    result['with_prompt'].append(eval_data)
            
            # Minimal prompt attempts
            for attempt in range(attempts):
                eval_data = self._evaluate_single_solution(
                    problem, use_detailed_prompt=False, **run_params
                )
                if eval_data:
                    result['without_prompt'].append(eval_data)
                    
            result['params'] = run_params  # Store normalized params for logging
            submissions_queue.put(result)
            tasks_queue.task_done()

    def _consumer_worker(self, submissions_queue: queue.Queue, results_list: List[Dict], 
                         results_lock: threading.Lock, pbar: tqdm):
        """Worker function for LeetCode submission (Consumer)"""
        while True:
            item = submissions_queue.get()
            if item is None: # Sentinel
                submissions_queue.task_done()
                break
            self._process_submission_item(item, results_list, results_lock, pbar)
            submissions_queue.task_done()

    def _process_submission_item(
        self,
        item: Dict,
        results_list: List[Dict],
        results_lock: Optional[threading.Lock] = None,
        pbar: Optional[tqdm] = None
    ):
        problem = item['problem']
        aggregated_result = {
            'problem_id': problem.get('frontend_question_id') or problem['question_id'],
            'backend_question_id': problem.get('question_id'),
            'title': problem['title'],
            'title_slug': problem['title_slug'],
            'difficulty': problem['difficulty'],
            'topics': problem['topics'],
            'timestamp': datetime.now().isoformat(),
            'with_prompt': [],
            'without_prompt': []
        }

        for strategy in ['with_prompt', 'without_prompt']:
            for attempt_idx, gen_data in enumerate(item.get(strategy, [])):
                self.global_rate_limit_pause.wait()

                normalized_prompt = 'detailed' if strategy == 'with_prompt' else 'minimal'
                attempt_result = {
                    'status': gen_data.get('status', 'Unknown'),
                    'prompt_type': normalized_prompt,
                    'input_tokens': gen_data.get('input_tokens', 0),
                    'output_tokens': gen_data.get('output_tokens', 0),
                    'latency_ms': gen_data.get('latency_ms', 0),
                    'cost': gen_data.get('cost', 0),
                    'gen_status': gen_data.get('gen_status', 'Unknown')
                }

                if gen_data.get('error'):
                    attempt_result['error'] = gen_data.get('error')
                if gen_data.get('raw_response'):
                    attempt_result['raw_response'] = gen_data.get('raw_response')

                code = gen_data.get('code')
                if code:
                    attempt_result['code'] = code
                    submission_id = self.leetcode_client.submit_solution(
                        title_slug=problem['title_slug'],
                        code=code,
                        question_id=problem['question_id']
                    )

                    if submission_id:
                        self._consecutive_rate_limit_failures = 0  # reset on real submission
                        res = self.leetcode_client.check_submission(submission_id)
                        if res:
                            attempt_result.update(res)
                        else:
                            attempt_result['status'] = 'Submission Result Missing'
                    else:
                        attempt_result['status'] = 'Submission Failed'
                        self._consecutive_rate_limit_failures += 1
                        threshold = Config.MAX_CONSECUTIVE_RATE_LIMIT_FAILURES
                        if self._consecutive_rate_limit_failures >= threshold:
                            raise RateLimitExhaustedException(
                                f"LeetCode rate limit exhausted: {self._consecutive_rate_limit_failures} "
                                f"consecutive submission failures (no submission_id). "
                                f"Switch LeetCode account and re-run."
                            )

                aggregated_result[strategy].append(attempt_result)

                status = attempt_result.get('status', 'Unknown')
                self.experiment_manager.log_attempt({
                    'problem_id': problem.get('frontend_question_id') or problem.get('question_id'),
                    'backend_question_id': problem.get('question_id'),
                    'problem': problem['title'],
                    'strategy': normalized_prompt,
                    'trial_index': attempt_idx,
                    'provider': self.provider,
                    'model_name': self.llm_client.model_id,
                    'temperature': (
                        item.get('params', {}).get('temperature')
                        if item.get('params', {}).get('temperature') is not None
                        else Config.MODEL_TEMPERATURE
                    ),
                    'top_p': (
                        item.get('params', {}).get('top_p')
                        if item.get('params', {}).get('top_p') is not None
                        else Config.MODEL_TOP_P
                    ),
                    'max_tokens': (
                        item.get('params', {}).get('max_tokens')
                        if item.get('params', {}).get('max_tokens') is not None
                        else Config.MODEL_MAX_TOKENS
                    ),
                    'verdict': status,
                    'accepted_bool': 1 if status == 'Accepted' else 0,
                    'prompt_tokens': gen_data.get('input_tokens', 0),
                    'completion_tokens': gen_data.get('output_tokens', 0),
                    **attempt_result
                })
                if code:
                    self.experiment_manager.log_solution(
                        problem['title'],
                        normalized_prompt,
                        code
                    )
                    time.sleep(Config.LEETCODE_SUBMISSION_DELAY_S)

        if results_lock:
            with results_lock:
                results_list.append(aggregated_result)
                self._save_results(results_list)
        else:
            results_list.append(aggregated_result)
            self._save_results(results_list)

        if pbar:
            pbar.update(1)

    def evaluate_batch(self, problems: List[Dict], attempts: int = 1, **kwargs) -> List[Dict]:
        """
        Evaluate multiple problems concurrently using Producer-Consumer pattern
        """
        results = []
        results_lock = threading.Lock()
        tasks_queue = queue.Queue()
        submissions_queue = queue.Queue()
        
        # Determine number of workers
        num_workers = kwargs.get('workers', Config.WORKER_THREADS)
        
        print(f"\n🚀 Starting concurrent evaluation:")
        print(f"  - Problems: {len(problems)}")
        print(f"  - LLM Workers: {num_workers}")
        print(f"  - Submission Delay: {Config.LEETCODE_SUBMISSION_DELAY_S}s")
        print(f"  - Total Evaluations: {len(problems) * attempts * 2}")
        
        # Populate tasks
        for prob in problems:
            tasks_queue.put(prob)
            
        pbar = tqdm(total=len(problems), desc="Progress")
        
        # Start Consumer thread
        consumer_thread = threading.Thread(
            target=self._consumer_worker,
            args=(submissions_queue, results, results_lock, pbar),
            daemon=True
        )
        consumer_thread.start()
        
        # Start Producers using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._producer_worker, tasks_queue, submissions_queue, attempts, **kwargs)
                for _ in range(num_workers)
            ]
            concurrent.futures.wait(futures)
            
        # Signal consumer to finish
        submissions_queue.put(None)
        consumer_thread.join()
        pbar.close()

        # Save summary CSV
        self._save_summary_csv(results)
        return results

    def evaluate_generated_batch(self, generated_items: List[Dict], results_file: str = None) -> List[Dict]:
        """Submit a set of pre-generated attempts to LeetCode and reuse normal logging/reporting.

        If results_file points to an existing file, problems whose every trial already has a real
        submission_id (i.e. not a 429-caused Submission Failed) are skipped automatically — enabling
        seamless account-switching resume.
        """
        results = []
        completed_slugs: set = set()

        # Resume: load existing results and skip already-submitted problems
        if results_file and os.path.exists(results_file):
            with open(results_file) as f:
                existing = json.load(f)
            for r in existing:
                slug = r.get('title_slug')
                # A problem needs retry if any trial has no submission_id (API-level failure)
                needs_retry = any(
                    attempt.get('submission_id') is None
                    and attempt.get('status') == 'Submission Failed'
                    for strategy in ['with_prompt', 'without_prompt']
                    for attempt in r.get(strategy, [])
                )
                if not needs_retry:
                    completed_slugs.add(slug)
                    results.append(r)
                    # Re-log to current JSONL so StabilityAnalyzer sees complete data
                    for strategy in ['with_prompt', 'without_prompt']:
                        prompt_type = 'detailed' if strategy == 'with_prompt' else 'minimal'
                        for trial_idx, attempt in enumerate(r.get(strategy, [])):
                            self.experiment_manager.log_attempt({
                                'problem_id': r.get('problem_id'),
                                'backend_question_id': r.get('backend_question_id'),
                                'problem': r.get('title'),
                                'strategy': attempt.get('prompt_type', prompt_type),
                                'trial_index': trial_idx,
                                'provider': self.provider,
                                'model_name': self.llm_client.model_id,
                                'temperature': Config.MODEL_TEMPERATURE,
                                'top_p': Config.MODEL_TOP_P,
                                'max_tokens': Config.MODEL_MAX_TOKENS,
                                'verdict': attempt.get('status'),
                                'accepted_bool': 1 if attempt.get('status') == 'Accepted' else 0,
                                **{k: v for k, v in attempt.items()
                                   if k not in ('prompt_type', 'gen_status')},
                            })

        pending = [item for item in generated_items
                   if item['problem']['title_slug'] not in completed_slugs]

        print(f"\n🚀 Starting generated-result submission batch:")
        print(f"  - Total Problems: {len(generated_items)}")
        if completed_slugs:
            print(f"  - Already complete (resuming): {len(completed_slugs)}, remaining: {len(pending)}")
        print(f"  - Submission Delay: {Config.LEETCODE_SUBMISSION_DELAY_S}s")
        total_attempts = sum(
            len(item.get('with_prompt', [])) + len(item.get('without_prompt', []))
            for item in pending
        )
        print(f"  - Total Generated Attempts: {total_attempts}")

        pbar = tqdm(total=len(pending), desc="Progress")
        for item in pending:
            self._process_submission_item(item, results, None, pbar)
        pbar.close()

        self._save_summary_csv(results)
        return results

    def _save_summary_csv(self, results: List[Dict]):
        """Save summary CSV from aggregated evaluation results."""
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

    def _save_results(self, results: List[Dict], filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = self._active_results_file
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(Config.RESULTS_EVALUATIONS, exist_ok=True)
            filename = f"{Config.RESULTS_EVALUATIONS}/evaluation_results_{timestamp}.json"
            self._active_results_file = filename

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def load_results(self, filename: str) -> List[Dict]:
        """Load results from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)

    def load_problems(self, filename: str) -> List[Dict]:
        """Load a fixed problem set from a JSON file."""
        with open(filename, 'r') as f:
            problems = json.load(f)

        if not isinstance(problems, list):
            raise ValueError(f"Problems file must contain a JSON list: {filename}")

        missing_backend_ids = []
        for problem in problems:
            if 'frontend_question_id' not in problem and 'question_id' in problem:
                problem['frontend_question_id'] = str(problem['question_id'])
            if 'question_id' in problem:
                problem['question_id'] = str(problem['question_id'])
            if 'frontend_question_id' in problem:
                problem['frontend_question_id'] = str(problem['frontend_question_id'])
            if 'backend_question_id' in problem and not problem.get('question_id'):
                problem['question_id'] = str(problem['backend_question_id'])
            if 'backend_question_id' in problem:
                problem['backend_question_id'] = str(problem['backend_question_id'])
            elif 'question_id' not in problem or problem.get('question_id') == problem.get('frontend_question_id'):
                problem['backend_question_id'] = None
            if not problem.get('backend_question_id'):
                missing_backend_ids.append(problem)

        if missing_backend_ids:
            print(f"⚠ Refreshing backend question IDs for {len(missing_backend_ids)} dataset problems")
            refreshed = 0
            for problem in missing_backend_ids:
                details = self.leetcode_client.get_problem_details(problem['title_slug'])
                if not details:
                    continue
                problem['backend_question_id'] = str(details['question_id'])
                problem['question_id'] = str(details['question_id'])
                problem['frontend_question_id'] = str(
                    details.get('frontend_question_id') or problem.get('frontend_question_id')
                )
                refreshed += 1

            if refreshed:
                with open(filename, 'w') as f:
                    json.dump(problems, f, indent=2)
                print(f"✓ Updated dataset with backend question IDs: {filename}")

        print(f"✓ Loaded {len(problems)} problems from {filename}")
        return problems

    def fetch_problems(self, count: int = 10, difficulty: str = None, selection: str = 'LATEST') -> List[Dict]:
        """
        Fetch problems from LeetCode

        Args:
            count: Number of problems to fetch
            difficulty: Filter by difficulty (EASY, MEDIUM, HARD)
            selection: Selection strategy ('LATEST' or 'RANDOM')

        Returns:
            List of problem dictionaries
        """
        print(f"\nFetching {count} problems from LeetCode...")
        if difficulty:
            print(f"Difficulty filter: {difficulty}")
        print(f"Selection strategy: {selection}")

        if selection.upper() == 'RANDOM':
            problems = self.leetcode_client.get_random_problems(count, difficulty)
        else:
            problems = self.leetcode_client.get_latest_problems(count, difficulty)

        print(f"✓ Fetched {len(problems)} problems")

        # Save problems to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(Config.RESULTS_PROBLEMS, exist_ok=True)
        filename = f"{Config.RESULTS_PROBLEMS}/problems_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(problems, f, indent=2)
        print(f"✓ Problems saved to {filename}")

        return problems

    def run_evaluation(self, num_problems: int = 10, difficulty: str = None,
                       attempts: int = 1, selection: str = 'LATEST', **kwargs) -> str:
        """
        Run complete evaluation workflow

        Args:
            num_problems: Number of problems to evaluate
            difficulty: Optional difficulty filter
            attempts: Number of attempts per approach
            selection: Selection strategy ('LATEST' or 'RANDOM')

        Returns:
            Path to results file
        """
        # Initialize
        if not self.initialize():
            return None

        dataset_name = kwargs.get('dataset', 'main')
        problems_file = Config.resolve_dataset_file(dataset_name)
        problems = []
        if problems_file and os.path.exists(problems_file):
            print(f"\nLoading fixed problem set from: {problems_file}")
            problems = self.load_problems(problems_file)
            if difficulty:
                problems = [p for p in problems if str(p.get('difficulty', '')).upper() == difficulty.upper()]
            if num_problems:
                problems = problems[:num_problems]
            print(f"✓ Using {len(problems)} fixed problems for evaluation")
        else:
            print(f"\nFixed problems file not found, falling back to fetch: {problems_file}")
            # Fetch problems
            problems = self.fetch_problems(num_problems, difficulty, selection)

        if not problems:
            print("✗ No problems fetched")
            return None

        # Evaluate
        eval_attempts = kwargs.get('stability_runs', attempts)
        results = self.evaluate_batch(problems, eval_attempts, **kwargs)
 
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(Config.RESULTS_EVALUATIONS, exist_ok=True)
        results_file = f"{Config.RESULTS_EVALUATIONS}/evaluation_results_{timestamp}.json"
        self._save_results(results, results_file)
        
        # Stability Analysis
        if eval_attempts > 1:
            analyzer = StabilityAnalyzer(self.experiment_manager.jsonl_path)
            summary = analyzer.compute_metrics()
            if summary:
                self.experiment_manager.save_stability_summary(summary)
                analyzer.print_report(summary)
 
        print(f"\n{'='*60}")
        print(f"✓ Evaluation complete!")
        print(f"✓ Results saved to: {results_file}")
        print(f"{'='*60}")

        return results_file

    def run_generated_evaluation(self, generated_items: List[Dict], stability_runs: int = 1,
                                  results_file: str = None) -> str:
        """Run evaluation using pre-generated attempts instead of live model generation.

        Pass results_file to reuse an existing path across resume runs so that incremental saves
        always go to the same file and the resume logic can load prior progress.
        """
        if not self.initialize():
            return None

        if results_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(Config.RESULTS_EVALUATIONS, exist_ok=True)
            results_file = f"{Config.RESULTS_EVALUATIONS}/evaluation_results_{timestamp}.json"

        self._active_results_file = results_file
        results = self.evaluate_generated_batch(generated_items, results_file=results_file)
        self._save_results(results, results_file)

        if stability_runs > 1:
            analyzer = StabilityAnalyzer(self.experiment_manager.jsonl_path)
            summary = analyzer.compute_metrics()
            if summary:
                self.experiment_manager.save_stability_summary(summary)
                analyzer.print_report(summary)

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
