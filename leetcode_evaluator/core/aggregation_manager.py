import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from typing import List, Dict, Any
from leetcode_evaluator.core.config import Config

class AggregationManager:
    """Manages aggregation of results across multiple experiments and generation of paper-ready tables."""
    
    def __init__(self):
        self.summary_dir = Config.RESULTS_SUMMARY
        self.tables_dir = Config.RESULTS_TABLES
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)

    def _load_single_summary(self, path: str) -> Dict:
        """Helper to load a single summary file with error handling."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def aggregate_leaderboard(self) -> str:
        """Scan results/summary and generate leaderboard tables."""
        all_summaries = []
        
        # 1. Scan default/legacy location
        if os.path.exists(self.summary_dir):
            for filename in os.listdir(self.summary_dir):
                if filename.endswith(".json"):
                    all_summaries.append(self._load_single_summary(os.path.join(self.summary_dir, filename)))
        
        # 2. Scan run-specific nested locations
        output_root = Config.OUTPUT_ROOT  # Usually "output" or "output/<run_id>"
        # If we are in a run-specific root, also look at the literal "output" folder if we are one level down
        scan_roots = ["output"]
        if Config.OUTPUT_ROOT != "output" and os.path.dirname(Config.OUTPUT_ROOT):
            scan_roots.append("output") # redundant but safe

        scanned_paths = set()
        for root in scan_roots:
            if not os.path.exists(root): continue
            for run_id in os.listdir(root):
                run_path = os.path.join(root, run_id)
                if not os.path.isdir(run_path): continue
                summary_dir = os.path.join(run_path, "results", "summary")
                if os.path.isdir(summary_dir):
                    for filename in os.listdir(summary_dir):
                        if filename.endswith(".json"):
                            path = os.path.join(summary_dir, filename)
                            if path not in scanned_paths:
                                all_summaries.append(self._load_single_summary(path))
                                scanned_paths.add(path)

        all_summaries = [s for s in all_summaries if s]
        if not all_summaries:
            return "No summary data found to aggregate."

        df = pd.DataFrame(all_summaries)

        def format_ci(value):
            if isinstance(value, list) and len(value) == 2:
                return f"[{value[0]}, {value[1]}]"
            return ""

        ci_columns = {
            'run_level_ci_95': 'RLPR 95% CI',
            'first_pass_ci_95': 'FPA 95% CI',
            'perfect_stability_ci_95': 'PSR 95% CI',
            'average_variance_ci_95': 'AV 95% CI',
        }
        for raw_col, formatted_col in ci_columns.items():
            if raw_col in df.columns:
                df[formatted_col] = df[raw_col].apply(format_ci)
        
        # Columns requested by paper requirements:
        # Model, Prompt, point estimates, confidence intervals, and latency.
        
        column_mapping = {
            'model_name': 'Model',
            'prompt_type': 'Prompt',
            'run_level_pass_rate': 'Run-Level Pass Rate',
            'first_pass_accuracy': 'First-Pass Accuracy',
            'perfect_stability_rate': 'Perfect Stability Rate',
            'average_variance': 'Average Variance',
            'RLPR 95% CI': 'RLPR 95% CI',
            'FPA 95% CI': 'FPA 95% CI',
            'PSR 95% CI': 'PSR 95% CI',
            'AV 95% CI': 'AV 95% CI',
            'p90_latency_ms': 'p90 Latency'
        }
        
        # Filter and rename columns if they exist
        existing_cols = [c for c in column_mapping.keys() if c in df.columns]
        leaderboard = df[existing_cols].rename(columns=column_mapping)
        
        # Sort by Pass Rate descending
        if 'Run-Level Pass Rate' in leaderboard.columns:
            leaderboard = leaderboard.sort_values(by='Run-Level Pass Rate', ascending=False)

        # Export CSV
        csv_path = os.path.join(self.tables_dir, "leaderboard.csv")
        leaderboard.to_csv(csv_path, index=False)
        
        # Export Markdown
        md_path = os.path.join(self.tables_dir, "leaderboard.md")
        with open(md_path, 'w') as f:
            f.write("# Overall Performance by Configuration\n\n")
            f.write(leaderboard.to_markdown(index=False))

        return md_path

    def generate_plots(self):
        """Generate multi-configuration plots for paper-ready aggregation."""
        all_summaries = []
        for filename in os.listdir(self.summary_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.summary_dir, filename), 'r') as f:
                    all_summaries.append(json.load(f))
                    
        if not all_summaries: return
        df = pd.DataFrame(all_summaries)
        fig_dir = Config.RESULTS_FIGURES
        os.makedirs(fig_dir, exist_ok=True)
        
        # Figure 1: Accuracy vs Stability Scatter Plot
        if 'run_level_pass_rate' in df.columns and 'perfect_stability_rate' in df.columns:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='run_level_pass_rate', y='perfect_stability_rate', 
                            hue='model_name', style='prompt_type', s=100, data=df)
            plt.title('Accuracy vs Stability (Across Configurations)')
            plt.xlabel('Run-Level Pass Rate (%)')
            plt.ylabel('Perfect Stability Rate (%)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(fig_dir, "figure1_accuracy_vs_stability.png"), dpi=300)
            plt.close()

    def copy_raw_data(self, experiment_manager):
        """Copy experiment raw data to results/raw."""
        raw_dir = Config.RESULTS_RAW
        os.makedirs(raw_dir, exist_ok=True)
        
        exp_name = os.path.basename(experiment_manager.experiment_dir)
        dest_path = os.path.join(raw_dir, f"{exp_name}.jsonl")
        
        if os.path.exists(experiment_manager.jsonl_path):
            shutil.copy2(experiment_manager.jsonl_path, dest_path)
            print(f"✓ Raw data archived to {dest_path}")

    def save_experiment_summary(self, experiment_name: str, metrics: Dict[str, Any], config: Dict[str, Any]):
        """Save a single experiment's metrics and config to the summary directory."""
        summary = {**config, **metrics}
        summary['experiment_name'] = experiment_name
        
        output_path = os.path.join(self.summary_dir, f"{experiment_name}.json")
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=4)
