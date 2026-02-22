"""
Main evaluation logic for comparing prompted vs non-prompted solutions
"""
import json
import time
import queue
import threading
import concurrent.futures
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

from leetcode_evaluator.clients.leetcode import LeetCodeClient
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
                    problem, use_detailed_prompt=True, **kwargs
                )
                if eval_data:
                    result['with_prompt'].append(eval_data)
            
            # Minimal prompt attempts
            for attempt in range(attempts):
                eval_data = self._evaluate_single_solution(
                    problem, use_detailed_prompt=False, **kwargs
                )
                if eval_data:
                    result['without_prompt'].append(eval_data)
                    
            result['params'] = kwargs # Store params for logging
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
                
            problem = item['problem']
            aggregated_result = {
                'problem_id': problem['question_id'],
                'title': problem['title'],
                'title_slug': problem['title_slug'],
                'difficulty': problem['difficulty'],
                'topics': problem['topics'],
                'timestamp': datetime.now().isoformat(),
                'with_prompt': [],
                'without_prompt': []
            }
            
            # Process submissions one by one with delay
            for strategy in ['with_prompt', 'without_prompt']:
                for attempt_idx, gen_data in enumerate(item[strategy]):
                    # Check global rate limit pause
                    self.global_rate_limit_pause.wait()
                    
                    code = gen_data.get('code')
                    if code:
                        # Submit to LeetCode
                        submission_id = self.leetcode_client.submit_solution(
                            title_slug=problem['title_slug'],
                            code=code,
                            question_id=problem['question_id']
                        )
                        
                        if submission_id:
                            # Check result
                            res = self.leetcode_client.check_submission(submission_id)
                            if res:
                                res['code'] = code
                                res['prompt_type'] = 'detailed' if strategy == 'with_prompt' else 'minimal'
                                # Update with LLM metadata
                                res.update({
                                    'input_tokens': gen_data.get('input_tokens', 0),
                                    'output_tokens': gen_data.get('output_tokens', 0),
                                    'latency_ms': gen_data.get('latency_ms', 0),
                                    'cost': gen_data.get('cost', 0),
                                    'gen_status': gen_data.get('gen_status', 'Success')
                                })
                                aggregated_result[strategy].append(res)
                                
                                # Log to experiment manager with stability-specific fields
                                status = res.get('status', 'Unknown')
                                self.experiment_manager.log_attempt({
                                    'problem_id': problem.get('question_id'),
                                    'problem': problem['title'],
                                    'strategy': 'detailed' if strategy == 'with_prompt' else 'minimal',
                                    'trial_index': attempt_idx,
                                    'model_name': self.llm_client.model_id,
                                    'temperature': item.get('params', {}).get('temperature', Config.MODEL_TEMPERATURE),
                                    'top_p': item.get('params', {}).get('top_p', Config.MODEL_TOP_P),
                                    'verdict': status,
                                    'accepted_bool': 1 if status == 'Accepted' else 0,
                                    'prompt_tokens': gen_data.get('input_tokens', 0),
                                    'completion_tokens': gen_data.get('output_tokens', 0),
                                    **res
                                })
                                self.experiment_manager.log_solution(
                                    problem['title'], 
                                    'detailed' if strategy == 'with_prompt' else 'minimal', 
                                    code
                                )
                    
                    # Mandatory delay between submissions
                    time.sleep(Config.LEETCODE_SUBMISSION_DELAY_S)
            
            with results_lock:
                results_list.append(aggregated_result)
                self._save_results(results_list)
            
            pbar.update(1)
            submissions_queue.task_done()

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

        problems = self.leetcode_client.get_latest_problems(count, difficulty)

        print(f"✓ Fetched {len(problems)} problems")

        # Save problems to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{Config.RESULTS_DIR}/problems_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(problems, f, indent=2)
        print(f"✓ Problems saved to {filename}")

        return problems

    def run_evaluation(self, num_problems: int = 10, difficulty: str = None,
                       attempts: int = 1, **kwargs) -> str:
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
        eval_attempts = kwargs.get('stability_runs', attempts)
        results = self.evaluate_batch(problems, eval_attempts, **kwargs)
 
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{Config.RESULTS_DIR}/evaluation_results_{timestamp}.json"
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
