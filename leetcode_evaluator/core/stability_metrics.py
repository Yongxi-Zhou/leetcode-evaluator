import os
import math
import logging
import json
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

class StabilityAnalyzer:
    """Analyzes repeated-run experiment results to compute stability and correctness metrics."""
    DEFAULT_BOOTSTRAP_SAMPLES = 1000
    
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

    def _normalize_strategy(self, trial: Dict[str, Any]) -> str:
        """Normalizes prompt/strategy labels across raw trial sources."""
        strategy = trial.get('strategy') or trial.get('prompt_type', 'default')
        mapping = {
            'with_prompt': 'detailed',
            'without_prompt': 'minimal',
        }
        return mapping.get(strategy, strategy)

    def _compute_summary(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes all stability metrics for a homogeneous set of trials."""
        if not trials:
            return {}

        # Group trials by problem and strategy
        problem_stats = defaultdict(lambda: {
            "accepted_count": 0, 
            "total": 0, 
            "first_pass_accepted": False, 
            "latencies": [],
            "accepted_in_first_k": 0  # 0 = not accepted yet, N = accepted in Nth attempt
        })

        total_accepted_runs = 0
        total_runs = len(trials)
        all_latencies = []

        for trial in trials:
            pid = trial.get('problem_id')
            strategy = self._normalize_strategy(trial)
            key = (pid, strategy)

            is_accepted = 1 if trial.get('status') == 'Accepted' else 0
            latency = trial.get('latency_ms', 0)

            problem_stats[key]["total"] += 1
            problem_stats[key]["accepted_count"] += is_accepted

            # Track if accepted in first k attempts
            if is_accepted and problem_stats[key]["accepted_in_first_k"] == 0:
                problem_stats[key]["accepted_in_first_k"] = problem_stats[key]["total"]

            if latency:
                problem_stats[key]["latencies"].append(latency)
                all_latencies.append(latency)

            if trial.get('trial_index') == 0:
                problem_stats[key]["first_pass_accepted"] = bool(is_accepted)

            total_accepted_runs += is_accepted

        # 1) Run-Level Pass Rate
        run_level_pass_rate = (total_accepted_runs / total_runs) * 100 if total_runs > 0 else 0

        # 2) Perfect Stability Rate & 3) First-Pass Accuracy & 4) Variance & 5) Pass@k
        perfect_stable_count = 0
        first_pass_correct_count = 0
        variances = []
        pass_at_k_counts = {1: 0, 3: 0, 5: 0}  # pass@1, pass@3, pass@5

        for key, stats in problem_stats.items():
            n = stats["total"]
            acc = stats["accepted_count"]

            # Perfect Stability (all N trials accepted)
            if acc == n:
                perfect_stable_count += 1

            # First-Pass Accuracy
            if stats["first_pass_accepted"]:
                first_pass_correct_count += 1

            # Pass@k calculation
            for k in [1, 3, 5]:
                accepted_in_first_k = stats.get('accepted_in_first_k', 0)
                if accepted_in_first_k > 0 and accepted_in_first_k <= k:
                    pass_at_k_counts[k] += 1

            # Variance per problem: p(1-p)
            p = acc / n if n > 0 else 0
            variances.append(p * (1 - p))

        total_problems = len(problem_stats)
        perfect_stability_rate = (perfect_stable_count / total_problems) * 100 if total_problems > 0 else 0
        first_pass_accuracy = (first_pass_correct_count / total_problems) * 100 if total_problems > 0 else 0
        average_variance = sum(variances) / len(variances) if variances else 0

        # Pass@k rates
        pass_at_k_rates = {}
        for k, count in pass_at_k_counts.items():
            pass_at_k_rates[f'pass@{k}'] = (count / total_problems) * 100 if total_problems > 0 else 0

        # 5) 95% Confidence intervals (Wilson score interval)
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
            
        rlpr_ci_lower, rlpr_ci_upper = wilson_interval(total_accepted_runs, total_runs)
        fpa_ci_lower, fpa_ci_upper = wilson_interval(first_pass_correct_count, total_problems)
        psr_ci_lower, psr_ci_upper = wilson_interval(perfect_stable_count, total_problems)

        # AV uncertainty via bootstrap over problems.
        av_ci_lower = 0.0
        av_ci_upper = 0.0
        if variances:
            rng = np.random.default_rng(0)
            variance_array = np.array(variances, dtype=float)
            bootstrap_samples = []
            for _ in range(self.DEFAULT_BOOTSTRAP_SAMPLES):
                sample = rng.choice(variance_array, size=len(variance_array), replace=True)
                bootstrap_samples.append(float(sample.mean()))
            av_ci_lower, av_ci_upper = np.percentile(bootstrap_samples, [2.5, 97.5])
        
        # Latency Percentiles
        latencies = np.array(all_latencies) if all_latencies else np.array([])
        p50_latency = np.percentile(latencies, 50) if latencies.size > 0 else 0
        p90_latency = np.percentile(latencies, 90) if latencies.size > 0 else 0
        
        summary = {
            "total_problems": total_problems,
            "total_runs": total_runs,
            "run_level_pass_rate": round(run_level_pass_rate, 2),
            "run_level_ci_95": [rlpr_ci_lower, rlpr_ci_upper],
            "perfect_stability_rate": round(perfect_stability_rate, 2),
            "perfect_stability_ci_95": [psr_ci_lower, psr_ci_upper],
            "first_pass_accuracy": round(first_pass_accuracy, 2),
            "first_pass_ci_95": [fpa_ci_lower, fpa_ci_upper],
            "average_variance": round(average_variance, 4),
            "average_variance_ci_95": [float(round(av_ci_lower, 4)), float(round(av_ci_upper, 4))],
            "p50_latency_ms": float(round(p50_latency, 2)),
            "p90_latency_ms": float(round(p90_latency, 2)),
            "pass@1": round(pass_at_k_rates.get('pass@1', 0), 2),
            "pass@3": round(pass_at_k_rates.get('pass@3', 0), 2),
            "pass@5": round(pass_at_k_rates.get('pass@5', 0), 2)
        }

        return summary

    def compute_metrics(self) -> Dict[str, Any]:
        """Computes all stability metrics from the loaded trials."""
        return self._compute_summary(self.trials)

    def compute_metrics_by_strategy(self) -> Dict[str, Dict[str, Any]]:
        """Computes prompt-specific stability metrics for paper-ready aggregation."""
        grouped_trials = defaultdict(list)
        for trial in self.trials:
            grouped_trials[self._normalize_strategy(trial)].append(trial)

        return {
            strategy: self._compute_summary(strategy_trials)
            for strategy, strategy_trials in grouped_trials.items()
            if strategy_trials
        }

    def print_report(self, summary: Dict[str, Any]):
        """Prints a human-readable report of the metrics."""
        print("\n" + "="*40)
        print("STABILITY EVALUATION SUMMARY")
        print("="*40)
        print(f"Total Problems:          {summary.get('total_problems')}")
        print(f"Total Runs:              {summary.get('total_runs')}")
        print(f"Run-Level Pass Rate:     {summary.get('run_level_pass_rate')}%")
        ci = summary.get('run_level_ci_95', [0, 0])
        print(f"95% CI (RLPR):           [{ci[0]}%, {ci[1]}%]")
        psr_ci = summary.get('perfect_stability_ci_95', [0, 0])
        print(f"Perfect Stability Rate:  {summary.get('perfect_stability_rate')}%")
        print(f"95% CI (PSR):            [{psr_ci[0]}%, {psr_ci[1]}%]")
        fpa_ci = summary.get('first_pass_ci_95', [0, 0])
        print(f"First-Pass Accuracy:     {summary.get('first_pass_accuracy')}%")
        print(f"95% CI (FPA):            [{fpa_ci[0]}%, {fpa_ci[1]}%]")
        print(f"Pass@1:                  {summary.get('pass@1', 0):.2f}%")
        print(f"Pass@3:                  {summary.get('pass@3', 0):.2f}%")
        print(f"Pass@5:                  {summary.get('pass@5', 0):.2f}%")
        av_ci = summary.get('average_variance_ci_95', [0, 0])
        print(f"Average Variance:        {summary.get('average_variance')}")
        print(f"95% CI (AV):             [{av_ci[0]}, {av_ci[1]}]")
        print(f"p50 Latency:             {summary.get('p50_latency_ms')} ms")
        print(f"p90 Latency:             {summary.get('p90_latency_ms')} ms")
        print("="*40 + "\n")
