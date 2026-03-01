import os
import math
import logging
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np

class StabilityAnalyzer:
    """Analyzes repeated-run experiment results to compute stability and correctness metrics."""
    
    def __init__(self, detailed_jsonl_path: str = None, trials: List[Dict] = None):
        self.detailed_jsonl_path = detailed_jsonl_path
        self.trials = trials or []
        self.logger = logging.getLogger("StabilityAnalyzer")
        if not self.trials and self.detailed_jsonl_path:
            self.load_trials()
        
    def load_trials(self):
        """Loads raw trial data from the detailed.jsonl file."""
        if not os.path.exists(self.detailed_jsonl_path):
            self.logger.error(f"Results file not found: {self.detailed_jsonl_path}")
            return
            
        with open(self.detailed_jsonl_path, 'r') as f:
            for line in f:
                try:
                    self.trials.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                    
    def compute_metrics(self) -> Dict[str, Any]:
        """Computes all stability metrics from the loaded trials."""
        if not self.trials:
            return {}
            
        # Group trials by problem and strategy
        problem_stats = defaultdict(lambda: {"accepted_count": 0, "total": 0, "first_pass_accepted": False, "latencies": []})
        
        total_accepted_runs = 0
        total_runs = len(self.trials)
        all_latencies = []
        
        for trial in self.trials:
            pid = trial.get('problem_id')
            strategy = trial.get('strategy') or trial.get('prompt_type', 'default')
            key = (pid, strategy)
            
            is_accepted = 1 if trial.get('status') == 'Accepted' else 0
            latency = trial.get('latency_ms', 0)
            
            problem_stats[key]["total"] += 1
            problem_stats[key]["accepted_count"] += is_accepted
            if latency:
                problem_stats[key]["latencies"].append(latency)
                all_latencies.append(latency)
            
            if trial.get('trial_index') == 0:
                problem_stats[key]["first_pass_accepted"] = bool(is_accepted)
                
            total_accepted_runs += is_accepted
            
        # 1) Run-Level Pass Rate
        run_level_pass_rate = (total_accepted_runs / total_runs) * 100 if total_runs > 0 else 0
        
        # 2) Perfect Stability Rate & 3) First-Pass Accuracy & 4) Variance
        perfect_stable_count = 0
        first_pass_correct_count = 0
        variances = []
        
        for key, stats in problem_stats.items():
            n = stats["total"]
            acc = stats["accepted_count"]
            
            # Perfect Stability (all N trials accepted)
            if acc == n:
                perfect_stable_count += 1
                
            # First-Pass Accuracy
            if stats["first_pass_accepted"]:
                first_pass_correct_count += 1
                
            # Variance per problem: p(1-p)
            p = acc / n if n > 0 else 0
            variances.append(p * (1 - p))
            
        total_problems = len(problem_stats)
        perfect_stability_rate = (perfect_stable_count / total_problems) * 100 if total_problems > 0 else 0
        first_pass_accuracy = (first_pass_correct_count / total_problems) * 100 if total_problems > 0 else 0
        average_variance = sum(variances) / len(variances) if variances else 0
        
        # 5) 95% Confidence Interval for Run-Level Pass Rate (Wilson score interval)
        def wilson_interval(successes, total, z=1.96):
            if total == 0:
                return 0.0, 0.0
            p = successes / total
            denominator = 1 + z**2 / total
            centre_adj_p = p + z**2 / (2 * total)
            adj_p_delta = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
            lower = (centre_adj_p - adj_p_delta) / denominator
            upper = (centre_adj_p + adj_p_delta) / denominator
            return round(max(0, lower) * 100, 2), round(min(1, upper) * 100, 2)
            
        ci_lower, ci_upper = wilson_interval(total_accepted_runs, total_runs)
        
        # Latency Percentiles
        latencies = np.array(all_latencies) if all_latencies else np.array([])
        p50_latency = np.percentile(latencies, 50) if latencies.size > 0 else 0
        p90_latency = np.percentile(latencies, 90) if latencies.size > 0 else 0
        
        summary = {
            "total_problems": total_problems,
            "total_runs": total_runs,
            "run_level_pass_rate": round(run_level_pass_rate, 2),
            "run_level_ci_95": [ci_lower, ci_upper],
            "perfect_stability_rate": round(perfect_stability_rate, 2),
            "first_pass_accuracy": round(first_pass_accuracy, 2),
            "average_variance": round(average_variance, 4),
            "p50_latency_ms": round(p50_latency, 2),
            "p90_latency_ms": round(p90_latency, 2)
        }
        
        return summary

    def print_report(self, summary: Dict[str, Any]):
        """Prints a human-readable report of the metrics."""
        print("\n" + "="*40)
        print("STABILITY EVALUATION SUMMARY")
        print("="*40)
        print(f"Total Problems:          {summary.get('total_problems')}")
        print(f"Total Runs:              {summary.get('total_runs')}")
        print(f"Run-Level Pass Rate:     {summary.get('run_level_pass_rate')}%")
        ci = summary.get('run_level_ci_95', [0, 0])
        print(f"95% CI (Wilson):         [{ci[0]}%, {ci[1]}%]")
        print(f"Perfect Stability Rate:  {summary.get('perfect_stability_rate')}%")
        print(f"First-Pass Accuracy:     {summary.get('first_pass_accuracy')}%")
        print(f"Average Variance:        {summary.get('average_variance')}")
        print(f"p50 Latency:             {summary.get('p50_latency_ms')} ms")
        print(f"p90 Latency:             {summary.get('p90_latency_ms')} ms")
        print("="*40 + "\n")
