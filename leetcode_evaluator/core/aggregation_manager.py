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

    def aggregate_leaderboard(self) -> str:
        """Scan results/summary and generate leaderboard tables."""
        all_summaries = []
        
        if not os.path.exists(self.summary_dir):
            return "Summary directory not found."

        for filename in os.listdir(self.summary_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.summary_dir, filename)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        # We expect the summary to contain model/prompt config info
                        all_summaries.append(data)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        if not all_summaries:
            return "No summary data found to aggregate."

        df = pd.DataFrame(all_summaries)
        
        # Columns requested by paper requirements:
        # Model, Prompt, Temperature, Run-Level Pass Rate, First-Pass Accuracy, 
        # Perfect Stability Rate, Average Variance, p90 Latency
        
        column_mapping = {
            'model_name': 'Model',
            'prompt_type': 'Prompt',
            'temperature': 'Temperature',
            'run_level_pass_rate': 'Run-Level Pass Rate',
            'first_pass_accuracy': 'First-Pass Accuracy',
            'perfect_stability_rate': 'Perfect Stability Rate',
            'average_variance': 'Average Variance',
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
        """Generate multi-configuration plots (Figure 1 and Figure 2)."""
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

        # Figure 2: Temperature vs Variance Curve
        if 'temperature' in df.columns and 'average_variance' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='temperature', y='average_variance', hue='model_name', marker='o', data=df)
            plt.title('Temperature vs Solution Variance')
            plt.xlabel('Temperature')
            plt.ylabel('Average Variance')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(fig_dir, "figure2_temperature_vs_variance.png"), dpi=300)
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
