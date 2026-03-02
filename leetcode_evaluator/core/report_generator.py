"""
Report generator for comprehensive analysis of evaluation results
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime
from scipy import stats
from collections import defaultdict

from leetcode_evaluator.core.config import Config
from leetcode_evaluator.core.stability_metrics import StabilityAnalyzer


class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, results_file: str, output_dir: str = None,
                 model_name: str = None, provider: str = None):
        self.results_file = results_file
        self.results = self._load_results()
        self.df = self._create_dataframe()
        self.output_dir = output_dir
        self.model_name = model_name
        self.provider = provider
        self.experiment_metadata = {}
        
    def _load_results(self) -> List[Dict]:
        """Load results from JSON file"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from results"""
        rows = []
        
        for result in self.results:
            problem_id = result.get('problem_id')
            title = result.get('title')
            difficulty = result.get('difficulty')
            topics = result.get('topics', [])
            
            # Process with_prompt results
            for attempt in result.get('with_prompt', []):
                rows.append({
                    'problem_id': problem_id,
                    'title': title,
                    'difficulty': difficulty,
                    'topics': ', '.join(topics),
                    'prompt_type': 'with_prompt',
                    'status': attempt.get('status'),
                    'runtime': attempt.get('runtime'),
                    'memory': attempt.get('memory'),
                    'runtime_percentile': attempt.get('runtime_percentile'),
                    'memory_percentile': attempt.get('memory_percentile'),
                    'total_correct': attempt.get('total_correct'),
                    'total_testcases': attempt.get('total_testcases'),
                    'latency_ms': attempt.get('latency_ms'),
                    'input_tokens': attempt.get('input_tokens'),
                    'output_tokens': attempt.get('output_tokens'),
                    'cost': attempt.get('cost'),
                })
            
            # Process without_prompt results
            for attempt in result.get('without_prompt', []):
                rows.append({
                    'problem_id': problem_id,
                    'title': title,
                    'difficulty': difficulty,
                    'topics': ', '.join(topics),
                    'prompt_type': 'without_prompt',
                    'status': attempt.get('status'),
                    'runtime': attempt.get('runtime'),
                    'memory': attempt.get('memory'),
                    'runtime_percentile': attempt.get('runtime_percentile'),
                    'memory_percentile': attempt.get('memory_percentile'),
                    'total_correct': attempt.get('total_correct'),
                    'total_testcases': attempt.get('total_testcases'),
                    'latency_ms': attempt.get('latency_ms'),
                    'input_tokens': attempt.get('input_tokens'),
                    'output_tokens': attempt.get('output_tokens'),
                    'cost': attempt.get('cost'),
                })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Normalize numeric fields; LeetCode/API responses may store numbers as strings/nulls.
        numeric_cols = [
            'runtime_percentile', 'memory_percentile', 'total_correct', 'total_testcases',
            'latency_ms', 'input_tokens', 'output_tokens', 'cost'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _load_experiment_metadata(self) -> Dict:
        """
        Load experiment config/summary metadata from results/summary/<experiment_name>.json
        when the report output directory follows reports/<experiment_name>/...
        """
        if not self.output_dir:
            return {}

        experiment_name = os.path.basename(os.path.normpath(self.output_dir))
        if not experiment_name:
            return {}

        # 1. Try finding in Config.RESULTS_SUMMARY (current run)
        summary_path = os.path.join(Config.RESULTS_SUMMARY, f"{experiment_name}.json")
        
        # 2. Try sibling path (portable/independent report mode)
        # self.output_dir = 'output/<run_id>/reports/<exp_id>'
        # Try: 'output/<run_id>/results/summary/<exp_id>.json'
        if not os.path.exists(summary_path):
            try:
                # If output_dir is 'output/run_id/reports/exp_id', then base_run_dir is 'output/run_id'
                base_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.output_dir)))
                alt_path = os.path.join(base_run_dir, "results", "summary", f"{experiment_name}.json")
                if os.path.exists(alt_path):
                    summary_path = alt_path
            except:
                pass
                
        if not os.path.exists(summary_path):
            return {}

        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
            data['_summary_path'] = summary_path
            return data
        except Exception as e:
            print(f"Warning: failed to load experiment metadata from {summary_path}: {e}")
            return {}
    
    def calculate_correctness_metrics(self) -> Dict:
        """Calculate correctness and success rate metrics"""
        metrics = {}
        
        # Overall pass rates
        for prompt_type in ['with_prompt', 'without_prompt']:
            df_subset = self.df[self.df['prompt_type'] == prompt_type]
            total = len(df_subset)
            accepted = len(df_subset[df_subset['status'] == 'Accepted'])
            
            metrics[f'{prompt_type}_total'] = total
            metrics[f'{prompt_type}_accepted'] = accepted
            metrics[f'{prompt_type}_pass_rate'] = (accepted / total * 100) if total > 0 else 0
        
        # Pass rates by difficulty
        for difficulty in ['Easy', 'Medium', 'Hard']:
            for prompt_type in ['with_prompt', 'without_prompt']:
                df_subset = self.df[
                    (self.df['difficulty'] == difficulty) & 
                    (self.df['prompt_type'] == prompt_type)
                ]
                total = len(df_subset)
                accepted = len(df_subset[df_subset['status'] == 'Accepted'])
                
                key = f'{prompt_type}_{difficulty.lower()}_pass_rate'
                metrics[key] = (accepted / total * 100) if total > 0 else 0
        
        # Pass@k metrics
        metrics['with_prompt_pass@1'] = self._calculate_pass_at_k('with_prompt', k=1)
        metrics['without_prompt_pass@1'] = self._calculate_pass_at_k('without_prompt', k=1)
        
        # Error breakdown
        for prompt_type in ['with_prompt', 'without_prompt']:
            df_subset = self.df[self.df['prompt_type'] == prompt_type]
            error_counts = df_subset['status'].value_counts()
            
            for status, count in error_counts.items():
                if status != 'Accepted':
                    key = f'{prompt_type}_{status.lower().replace(" ", "_")}_count'
                    metrics[key] = int(count)
        
        return metrics
    
    def _calculate_pass_at_k(self, prompt_type: str, k: int = 1) -> float:
        """Calculate pass@k metric for a prompt type"""
        problems = self.df[self.df['prompt_type'] == prompt_type]['problem_id'].unique()
        successful = 0
        
        for problem_id in problems:
            attempts = self.df[
                (self.df['problem_id'] == problem_id) & 
                (self.df['prompt_type'] == prompt_type)
            ]['status'].head(k)
            
            if any(attempts == 'Accepted'):
                successful += 1
        
        return (successful / len(problems) * 100) if len(problems) > 0 else 0
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate runtime and memory performance metrics"""
        metrics = {}

        # Generation metrics are available even for failed submissions
        for prompt_type in ['with_prompt', 'without_prompt']:
            all_df_subset = self.df[self.df['prompt_type'] == prompt_type]
            if len(all_df_subset) > 0:
                latency_values = all_df_subset['latency_ms'].dropna() if 'latency_ms' in all_df_subset else pd.Series(dtype=float)
                if len(latency_values) > 0:
                    metrics[f'{prompt_type}_mean_latency_ms'] = float(latency_values.mean())
                    metrics[f'{prompt_type}_median_latency_ms'] = float(latency_values.median())
                    metrics[f'{prompt_type}_p90_latency_ms'] = float(latency_values.quantile(0.9))

                input_tokens = all_df_subset['input_tokens'].dropna() if 'input_tokens' in all_df_subset else pd.Series(dtype=float)
                output_tokens = all_df_subset['output_tokens'].dropna() if 'output_tokens' in all_df_subset else pd.Series(dtype=float)
                if len(input_tokens) > 0:
                    metrics[f'{prompt_type}_mean_input_tokens'] = float(input_tokens.mean())
                if len(output_tokens) > 0:
                    metrics[f'{prompt_type}_mean_output_tokens'] = float(output_tokens.mean())
                if len(input_tokens) > 0 or len(output_tokens) > 0:
                    total_tokens = all_df_subset[['input_tokens', 'output_tokens']].fillna(0).sum(axis=1)
                    total_tokens = total_tokens[total_tokens > 0]
                    if len(total_tokens) > 0:
                        metrics[f'{prompt_type}_mean_total_tokens'] = float(total_tokens.mean())

                cost_values = all_df_subset['cost'].dropna() if 'cost' in all_df_subset else pd.Series(dtype=float)
                if len(cost_values) > 0:
                    metrics[f'{prompt_type}_mean_cost'] = float(cost_values.mean())

        # Judge runtime/memory percentiles only exist for accepted submissions
        accepted_df = self.df[self.df['status'] == 'Accepted']
        for prompt_type in ['with_prompt', 'without_prompt']:
            df_subset = accepted_df[accepted_df['prompt_type'] == prompt_type]
            if len(df_subset) == 0:
                continue

            runtime_percentiles = df_subset['runtime_percentile'].dropna()
            if len(runtime_percentiles) > 0:
                metrics[f'{prompt_type}_mean_runtime_percentile'] = float(runtime_percentiles.mean())
                metrics[f'{prompt_type}_median_runtime_percentile'] = float(runtime_percentiles.median())
                metrics[f'{prompt_type}_std_runtime_percentile'] = float(runtime_percentiles.std())
                metrics[f'{prompt_type}_top_25_runtime_pct'] = float(
                    (runtime_percentiles >= 75).sum() / len(runtime_percentiles) * 100
                )

            memory_percentiles = df_subset['memory_percentile'].dropna()
            if len(memory_percentiles) > 0:
                metrics[f'{prompt_type}_mean_memory_percentile'] = float(memory_percentiles.mean())
                metrics[f'{prompt_type}_median_memory_percentile'] = float(memory_percentiles.median())
                metrics[f'{prompt_type}_std_memory_percentile'] = float(memory_percentiles.std())

        return metrics
    
    def calculate_comparison_metrics(self) -> Dict:
        """Calculate metrics comparing prompted vs non-prompted"""
        metrics = {}
        
        # Pass rate improvement
        with_pass = self._calculate_pass_at_k('with_prompt', k=1)
        without_pass = self._calculate_pass_at_k('without_prompt', k=1)
        
        if without_pass > 0:
            metrics['pass_rate_improvement'] = ((with_pass - without_pass) / without_pass) * 100
        else:
            metrics['pass_rate_improvement'] = float('inf') if with_pass > 0 else 0
        
        metrics['pass_rate_difference'] = with_pass - without_pass
        
        # Runtime improvement (lower percentile is better for LeetCode ranking, but higher means faster)
        accepted_df = self.df[self.df['status'] == 'Accepted']
        
        with_runtime = accepted_df[accepted_df['prompt_type'] == 'with_prompt']['runtime_percentile'].dropna()
        without_runtime = accepted_df[accepted_df['prompt_type'] == 'without_prompt']['runtime_percentile'].dropna()
        
        if len(with_runtime) > 0 and len(without_runtime) > 0:
            metrics['runtime_percentile_difference'] = float(with_runtime.mean() - without_runtime.mean())
            
            # Statistical significance test
            t_stat, p_value = stats.ttest_ind(with_runtime, without_runtime)
            metrics['runtime_ttest_statistic'] = float(t_stat)
            metrics['runtime_ttest_pvalue'] = float(p_value)
            metrics['runtime_statistically_significant'] = p_value < 0.05
        
        return metrics
    
    def calculate_topic_metrics(self) -> Dict:
        """Calculate performance by topic"""
        topic_metrics = defaultdict(lambda: {'with_prompt': [], 'without_prompt': []})
        
        for _, row in self.df.iterrows():
            if row['status'] == 'Accepted' and pd.notna(row['topics']):
                topics = [t.strip() for t in str(row['topics']).split(',')]
                for topic in topics:
                    topic_metrics[topic][row['prompt_type']].append(1)
            elif pd.notna(row['topics']):
                topics = [t.strip() for t in str(row['topics']).split(',')]
                for topic in topics:
                    topic_metrics[topic][row['prompt_type']].append(0)
        
        # Calculate success rates
        results = {}
        for topic, data in topic_metrics.items():
            if data['with_prompt']:
                results[f'{topic}_with_prompt_rate'] = np.mean(data['with_prompt']) * 100
            if data['without_prompt']:
                results[f'{topic}_without_prompt_rate'] = np.mean(data['without_prompt']) * 100
        
        return results

    def calculate_stability_metrics(self) -> Dict:
        """Calculate stability metrics using trials from results"""
        trials = []
        for result in self.results:
            pid = result.get('problem_id')
            for strategy, key in [('with_prompt', 'detailed'), ('without_prompt', 'minimal')]:
                for idx, attempt in enumerate(result.get(strategy, [])):
                    trials.append({
                        'problem_id': pid,
                        'prompt_type': key,
                        'trial_index': idx,
                        'status': attempt.get('status')
                    })
        
        if not trials:
            return {}
            
        analyzer = StabilityAnalyzer(trials=trials)
        return analyzer.compute_metrics()
    
    def generate_visualizations(self, output_dir: str = None):
        """Generate all visualization plots"""
        if output_dir is None:
            output_dir = self.output_dir or Config.REPORTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Pass rate comparison by difficulty
        self._plot_pass_rate_by_difficulty(f"{output_dir}/pass_rate_by_difficulty.png")
        
        # 2. Runtime percentile distribution
        self._plot_runtime_distribution(f"{output_dir}/runtime_distribution.png")
        
        # 3. Memory percentile distribution
        self._plot_memory_distribution(f"{output_dir}/memory_distribution.png")
        
        # 4. Runtime vs Memory scatter
        self._plot_runtime_vs_memory(f"{output_dir}/runtime_vs_memory.png")
        
        # 5. Error type comparison
        self._plot_error_types(f"{output_dir}/error_types.png")
        
        # 6. Topic performance heatmap
        self._plot_topic_heatmap(f"{output_dir}/topic_performance.png")
        
        # 7. Prompt Comparison Bar Chart (Figure 3)
        self._plot_prompt_comparison(f"{output_dir}/figure3_prompt_comparison.png")
        
        # 8. Per-Problem Stability Heatmap (Figure 4)
        self._plot_stability_heatmap(f"{output_dir}/figure4_stability_heatmap.png")
        
        # 9. Distribution of Problem-Level Success (Figure 5)
        self._plot_success_distribution(f"{output_dir}/figure5_success_distribution.png")
        
        print(f"✓ Visualizations saved to {output_dir}")
    
    def _plot_pass_rate_by_difficulty(self, filename: str):
        """Plot pass rates by difficulty level"""
        difficulties = ['Easy', 'Medium', 'Hard']
        with_prompt_rates = []
        without_prompt_rates = []
        
        for diff in difficulties:
            for prompt_type, rates_list in [('with_prompt', with_prompt_rates), 
                                            ('without_prompt', without_prompt_rates)]:
                df_subset = self.df[
                    (self.df['difficulty'] == diff) & 
                    (self.df['prompt_type'] == prompt_type)
                ]
                total = len(df_subset)
                accepted = len(df_subset[df_subset['status'] == 'Accepted'])
                rate = (accepted / total * 100) if total > 0 else 0
                rates_list.append(rate)
        
        x = np.arange(len(difficulties))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, with_prompt_rates, width, label='With Detailed Prompt', color='#2ecc71')
        ax.bar(x + width/2, without_prompt_rates, width, label='Without Detailed Prompt', color='#e74c3c')
        
        ax.set_xlabel('Difficulty Level', fontsize=12)
        ax.set_ylabel('Pass Rate (%)', fontsize=12)
        ax.set_title('Pass Rate Comparison by Difficulty', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_runtime_distribution(self, filename: str):
        """Plot runtime percentile distribution"""
        accepted_df = self.df[self.df['status'] == 'Accepted']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for prompt_type, color in [('with_prompt', '#3498db'), ('without_prompt', '#e67e22')]:
            data = accepted_df[accepted_df['prompt_type'] == prompt_type]['runtime_percentile'].dropna()
            if len(data) > 0:
                label = 'With Detailed Prompt' if prompt_type == 'with_prompt' else 'Without Detailed Prompt'
                ax.hist(data, bins=20, alpha=0.6, label=label, color=color, edgecolor='black')
        
        ax.set_xlabel('Runtime Percentile', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Runtime Percentile Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_distribution(self, filename: str):
        """Plot memory percentile distribution"""
        accepted_df = self.df[self.df['status'] == 'Accepted']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for prompt_type, color in [('with_prompt', '#9b59b6'), ('without_prompt', '#e74c3c')]:
            data = accepted_df[accepted_df['prompt_type'] == prompt_type]['memory_percentile'].dropna()
            if len(data) > 0:
                label = 'With Detailed Prompt' if prompt_type == 'with_prompt' else 'Without Detailed Prompt'
                ax.hist(data, bins=20, alpha=0.6, label=label, color=color, edgecolor='black')
        
        ax.set_xlabel('Memory Percentile', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Memory Percentile Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_runtime_vs_memory(self, filename: str):
        """Scatter plot of runtime vs memory percentiles"""
        accepted_df = self.df[self.df['status'] == 'Accepted']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for prompt_type, color, marker in [('with_prompt', '#2ecc71', 'o'), 
                                           ('without_prompt', '#e74c3c', 's')]:
            df_subset = accepted_df[accepted_df['prompt_type'] == prompt_type]
            data = df_subset[['runtime_percentile', 'memory_percentile']].dropna()
            
            if len(data) > 0:
                label = 'With Detailed Prompt' if prompt_type == 'with_prompt' else 'Without Detailed Prompt'
                ax.scatter(data['runtime_percentile'], data['memory_percentile'], 
                          alpha=0.6, label=label, color=color, marker=marker, s=100)
        
        ax.set_xlabel('Runtime Percentile', fontsize=12)
        ax.set_ylabel('Memory Percentile', fontsize=12)
        ax.set_title('Runtime vs Memory Percentile', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_types(self, filename: str):
        """Plot error type distribution"""
        error_data = []
        
        for prompt_type in ['with_prompt', 'without_prompt']:
            df_subset = self.df[self.df['prompt_type'] == prompt_type]
            status_counts = df_subset['status'].value_counts()
            
            label = 'With Prompt' if prompt_type == 'with_prompt' else 'Without Prompt'
            for status, count in status_counts.items():
                error_data.append({
                    'Status': status,
                    'Count': count,
                    'Type': label
                })
        
        error_df = pd.DataFrame(error_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        statuses = error_df['Status'].unique()
        x = np.arange(len(statuses))
        width = 0.35
        
        with_counts = [error_df[(error_df['Status'] == s) & (error_df['Type'] == 'With Prompt')]['Count'].sum() 
                      for s in statuses]
        without_counts = [error_df[(error_df['Status'] == s) & (error_df['Type'] == 'Without Prompt')]['Count'].sum() 
                         for s in statuses]
        
        ax.bar(x - width/2, with_counts, width, label='With Detailed Prompt', color='#3498db')
        ax.bar(x + width/2, without_counts, width, label='Without Detailed Prompt', color='#e67e22')
        
        ax.set_xlabel('Status', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Submission Status Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(statuses, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_topic_heatmap(self, filename: str):
        """Plot topic performance heatmap"""
        # Get top topics
        all_topics = []
        for topics_str in self.df['topics'].dropna():
            all_topics.extend([t.strip() for t in str(topics_str).split(',')])
        
        from collections import Counter
        top_topics = [topic for topic, _ in Counter(all_topics).most_common(10)]
        
        # Calculate success rates
        heatmap_data = []
        for topic in top_topics:
            with_rate = 0
            without_rate = 0
            
            for prompt_type, rate_var in [('with_prompt', 'with_rate'), ('without_prompt', 'without_rate')]:
                df_topic = self.df[self.df['topics'].str.contains(topic, na=False) & 
                                  (self.df['prompt_type'] == prompt_type)]
                total = len(df_topic)
                accepted = len(df_topic[df_topic['status'] == 'Accepted'])
                rate = (accepted / total * 100) if total > 0 else 0
                
                if prompt_type == 'with_prompt':
                    with_rate = rate
                else:
                    without_rate = rate
            
            heatmap_data.append([with_rate, without_rate])
        
        fig, ax = plt.subplots(figsize=(8, 10))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['With Prompt', 'Without Prompt'])
        ax.set_yticks(range(len(top_topics)))
        ax.set_yticklabels(top_topics)
        
        ax.set_title('Success Rate by Topic (%)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Rate (%)', fontsize=12)
        
        # Add text annotations
        for i in range(len(top_topics)):
            for j in range(2):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prompt_comparison(self, filename: str):
        """Figure 3: Prompt Comparison Bar Chart (Accuracy and Stability)"""
        if self.df.empty: return
        
        metrics = []
        for ptype in self.df['prompt_type'].unique():
            subset = self.df[self.df['prompt_type'] == ptype]
            
            # Use original results structure for stability calculation if needed, 
            # but here we can approx or use StabilityAnalyzer
            analyzer = StabilityAnalyzer(trials=subset.to_dict('records'))
            stats = analyzer.compute_metrics()
            
            metrics.append({
                'Prompt': 'Detailed' if ptype == 'with_prompt' else 'Minimal',
                'Metric': 'First-Pass Accuracy',
                'Value': stats.get('first_pass_accuracy', 0)
            })
            metrics.append({
                'Prompt': 'Detailed' if ptype == 'with_prompt' else 'Minimal',
                'Metric': 'Perfect Stability Rate',
                'Value': stats.get('perfect_stability_rate', 0)
            })
            
        plot_df = pd.DataFrame(metrics)
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Metric', y='Value', hue='Prompt', data=plot_df, palette='viridis')
        plt.title('Prompt Strategy Comparison: Accuracy vs Stability')
        plt.ylabel('Rate (%)')
        plt.ylim(0, 105)
        
        # Add values on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', fontsize=11, color='gray', xytext=(0, 5),
                        textcoords='offset points')
            
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_stability_heatmap(self, filename: str):
        """Figure 4: Per-Problem Stability Heatmap"""
        if self.df.empty: return
        
        data_matrix = []
        problem_titles = []
        
        for result in self.results:
            title = result.get('title')
            # For simplicity, we just show 'with_prompt' stability
            attempts = result.get('with_prompt', [])
            if not attempts: continue
            
            row = [1 if a.get('status') == 'Accepted' else 0 for a in attempts]
            data_matrix.append(row)
            problem_titles.append(title[:30] + '...' if len(title) > 30 else title)
            
        if not data_matrix: return
        
        # Pad rows to same length if inconsistent
        max_len = max(len(r) for r in data_matrix)
        data_matrix = [r + [0]*(max_len - len(r)) for r in data_matrix]
        
        plt.figure(figsize=(12, min(len(problem_titles) * 0.4, 15)))
        sns.heatmap(data_matrix, annot=False, cmap='RdYlGn', cbar=False, 
                    yticklabels=problem_titles, xticklabels=range(max_len))
        plt.title('Per-Problem Stability Heatmap (Green=Accepted, Red=Failed)')
        plt.xlabel('Trial Index')
        plt.ylabel('Problem')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_success_distribution(self, filename: str):
        """Figure 5: Distribution of Problem-Level Success Counts"""
        if self.df.empty: return
        
        success_counts = []
        for result in self.results:
            attempts = result.get('with_prompt', [])
            if not attempts: continue
            count = sum(1 for a in attempts if a.get('status') == 'Accepted')
            success_counts.append(count)
            
        if not success_counts: return
        
        max_attempts = max(len(result.get('with_prompt', [])) for result in self.results)
        
        plt.figure(figsize=(10, 6))
        plt.hist(success_counts, bins=range(max_attempts + 2), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
        plt.xticks(range(max_attempts + 1))
        plt.title('Distribution of Problem-Level Success Counts')
        plt.xlabel('Number of Successful Runs (out of N)')
        plt.ylabel('Number of Problems')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_full_report(self, output_dir: str = None) -> str:
        """Generate complete analysis report"""
        if output_dir is None:
            if self.output_dir:
                output_dir = self.output_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(Config.REPORTS_DIR, timestamp)
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.experiment_metadata = self._load_experiment_metadata()
        
        output_file = os.path.join(output_dir, "analysis_report.md")
        
        # Calculate all metrics
        correctness = self.calculate_correctness_metrics()
        performance = self.calculate_performance_metrics()
        comparison = self.calculate_comparison_metrics()
        topic_metrics = self.calculate_topic_metrics()
        stability = self.calculate_stability_metrics()
        
        # Generate report
        report = self._generate_markdown_report(correctness, performance, comparison, topic_metrics, stability)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        # Generate visualizations
        self.generate_visualizations(output_dir)
        
        print(f"✓ Report generated: {output_file}")
        return output_file
    
    def _generate_markdown_report(self, correctness: Dict, performance: Dict, 
                                  comparison: Dict, topic_metrics: Dict, stability: Dict) -> str:
        """Generate markdown formatted report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        model_line = ""
        if self.model_name:
            if self.provider:
                model_line = f"\n**Model:** {self.model_name} (provider: {self.provider})  "
            else:
                model_line = f"\n**Model:** {self.model_name}  "

        report = f"""# LeetCode AI Solution Evaluation Report

**Generated:** {timestamp}  {model_line}
**Results File:** {self.results_file}  
**Total Problems Evaluated:** {len(self.results)}

---

## Executive Summary

### Key Findings

"""
        
        # Add key metrics
        with_pass = correctness.get('with_prompt_pass@1', 0)
        without_pass = correctness.get('without_prompt_pass@1', 0)
        improvement = comparison.get('pass_rate_difference', 0)
        
        report += f"""
- **Overall Pass Rate (With Prompt):** {with_pass:.1f}%
- **Overall Pass Rate (Without Prompt):** {without_pass:.1f}%
- **Improvement:** {improvement:+.1f} percentage points
"""

        if self.experiment_metadata:
            report += """

### Experiment Configuration

| Variable | Value |
|---|---|
"""
            config_rows = [
                ("Experiment Name", self.experiment_metadata.get('experiment_name')),
                ("Model", self.experiment_metadata.get('model_name')),
                ("Prompt Profile", self.experiment_metadata.get('prompt_type')),
                ("Temperature", self.experiment_metadata.get('temperature')),
                ("Top-p", self.experiment_metadata.get('top_p')),
                ("Max Tokens", self.experiment_metadata.get('max_tokens')),
                ("Total Problems", self.experiment_metadata.get('num_problems') or self.experiment_metadata.get('total_problems')),
                ("Attempts", self.experiment_metadata.get('attempts')),
                ("Stability Runs", self.experiment_metadata.get('stability_runs')),
                ("Workers", self.experiment_metadata.get('workers')),
                ("Selection", self.experiment_metadata.get('selection')),
            ]
            for key, value in config_rows:
                if value is not None:
                    report += f"| {key} | {value} |\n"

        if 'with_prompt_mean_runtime_percentile' in performance:
            report += f"""
- **Mean Runtime Percentile (With Prompt):** {performance['with_prompt_mean_runtime_percentile']:.1f}%
- **Mean Runtime Percentile (Without Prompt):** {performance.get('without_prompt_mean_runtime_percentile', 0):.1f}%
"""
        elif 'with_prompt_mean_latency_ms' in performance or 'without_prompt_mean_latency_ms' in performance:
            report += f"""
- **Mean Generation Latency (With Prompt):** {performance.get('with_prompt_mean_latency_ms', 0):.0f} ms
- **Mean Generation Latency (Without Prompt):** {performance.get('without_prompt_mean_latency_ms', 0):.0f} ms
"""
        
        report += """
---

## 1. Correctness Analysis

### 1.1 Overall Success Rates

| Metric | With Prompt | Without Prompt |
|--------|-------------|----------------|
"""
        
        report += f"""| Pass@1 | {correctness.get('with_prompt_pass@1', 0):.1f}% | {correctness.get('without_prompt_pass@1', 0):.1f}% |
| Total Attempts | {correctness.get('with_prompt_total', 0)} | {correctness.get('without_prompt_total', 0)} |
| Accepted | {correctness.get('with_prompt_accepted', 0)} | {correctness.get('without_prompt_accepted', 0)} |
"""
        
        report += """
### 1.2 Success Rate by Difficulty

| Difficulty | With Prompt | Without Prompt | Difference |
|------------|-------------|----------------|------------|
"""
        
        for diff in ['easy', 'medium', 'hard']:
            with_rate = correctness.get(f'with_prompt_{diff}_pass_rate', 0)
            without_rate = correctness.get(f'without_prompt_{diff}_pass_rate', 0)
            diff_val = with_rate - without_rate
            report += f"""| {diff.capitalize()} | {with_rate:.1f}% | {without_rate:.1f}% | {diff_val:+.1f}% |
"""
        
        report += """
---

## 2. Performance Analysis

### 2.1 Generation Performance (LLM Call)

"""

        if ('with_prompt_mean_latency_ms' in performance or
                'without_prompt_mean_latency_ms' in performance or
                'with_prompt_mean_total_tokens' in performance or
                'without_prompt_mean_total_tokens' in performance):
            report += f"""| Metric | With Prompt | Without Prompt |
|--------|-------------|----------------|
| Mean Latency (ms) | {performance.get('with_prompt_mean_latency_ms', 0):.0f} | {performance.get('without_prompt_mean_latency_ms', 0):.0f} |
| Median Latency (ms) | {performance.get('with_prompt_median_latency_ms', 0):.0f} | {performance.get('without_prompt_median_latency_ms', 0):.0f} |
| P90 Latency (ms) | {performance.get('with_prompt_p90_latency_ms', 0):.0f} | {performance.get('without_prompt_p90_latency_ms', 0):.0f} |
| Mean Input Tokens | {performance.get('with_prompt_mean_input_tokens', 0):.1f} | {performance.get('without_prompt_mean_input_tokens', 0):.1f} |
| Mean Output Tokens | {performance.get('with_prompt_mean_output_tokens', 0):.1f} | {performance.get('without_prompt_mean_output_tokens', 0):.1f} |
| Mean Total Tokens | {performance.get('with_prompt_mean_total_tokens', 0):.1f} | {performance.get('without_prompt_mean_total_tokens', 0):.1f} |
| Mean Cost | {performance.get('with_prompt_mean_cost', 0):.6f} | {performance.get('without_prompt_mean_cost', 0):.6f} |
"""
        else:
            report += "_No generation latency/token/cost data found in the results file._\n"

        report += """

### 2.2 Judge Runtime Performance (Accepted Only)

"""

        if 'with_prompt_mean_runtime_percentile' in performance:
            report += f"""| Metric | With Prompt | Without Prompt |
|--------|-------------|----------------|
| Mean Runtime Percentile | {performance.get('with_prompt_mean_runtime_percentile', 0):.1f}% | {performance.get('without_prompt_mean_runtime_percentile', 0):.1f}% |
| Median Runtime Percentile | {performance.get('with_prompt_median_runtime_percentile', 0):.1f}% | {performance.get('without_prompt_median_runtime_percentile', 0):.1f}% |
| Std Dev | {performance.get('with_prompt_std_runtime_percentile', 0):.1f} | {performance.get('without_prompt_std_runtime_percentile', 0):.1f} |
| Top 25% Rate | {performance.get('with_prompt_top_25_runtime_pct', 0):.1f}% | {performance.get('without_prompt_top_25_runtime_pct', 0):.1f}% |
"""
        else:
            report += "_No accepted submissions with runtime percentile data in this run._\n"

        report += """

### 2.3 Judge Memory Performance (Accepted Only)

"""

        if ('with_prompt_mean_memory_percentile' in performance or
                'without_prompt_mean_memory_percentile' in performance):
            report += f"""| Metric | With Prompt | Without Prompt |
|--------|-------------|----------------|
| Mean Memory Percentile | {performance.get('with_prompt_mean_memory_percentile', 0):.1f}% | {performance.get('without_prompt_mean_memory_percentile', 0):.1f}% |
| Median Memory Percentile | {performance.get('with_prompt_median_memory_percentile', 0):.1f}% | {performance.get('without_prompt_median_memory_percentile', 0):.1f}% |
"""
        else:
            report += "_No accepted submissions with memory percentile data in this run._\n"

        report += f"""
---

## 3. Stability Analysis

| Metric | Value |
|--------|-------|
| Total Problems | {stability.get('total_problems', 0)} |
| Total Runs | {stability.get('total_runs', 0)} |
| Run-Level Pass Rate | {stability.get('run_level_pass_rate', 0):.1f}% |
| Perfect Stability Rate | {stability.get('perfect_stability_rate', 0):.1f}% |
| First-Pass Accuracy | {stability.get('first_pass_accuracy', 0):.1f}% |
| Average Variance | {stability.get('average_variance', 0):.4f} |

---

## 4. Prompt Impact Analysis

"""
        
        report += f"""
### 4.1 Effectiveness Metrics

- **Pass Rate Improvement:** {comparison.get('pass_rate_improvement', 0):.1f}%
- **Pass Rate Difference:** {comparison.get('pass_rate_difference', 0):+.1f} percentage points
"""
        
        if 'runtime_ttest_pvalue' in comparison:
            sig = "Yes" if comparison.get('runtime_statistically_significant', False) else "No"
            report += f"""
### 4.2 Statistical Significance

- **Runtime T-Test P-Value:** {comparison.get('runtime_ttest_pvalue', 1):.4f}
- **Statistically Significant (p<0.05):** {sig}
"""
        
        report += """
---

## 5. Conclusions

### 5.1 Key Insights

"""
        
        # Generate insights based on metrics
        if with_pass > without_pass:
            report += f"""
1. **Prompt Engineering Impact:** Detailed prompts improve success rates by {improvement:.1f} percentage points, demonstrating the value of structured guidance in complex problem-solving.
"""
        
        if 'with_prompt_mean_runtime_percentile' in performance:
            runtime_diff = comparison.get('runtime_percentile_difference', 0)
            if runtime_diff > 5:
                report += """
2. **Algorithm Quality:** Solutions generated with detailed prompts show better runtime performance, suggesting improved algorithmic choices.
"""
            elif runtime_diff < -5:
                report += """
2. **Algorithm Quality:** Solutions without prompts occasionally achieve better runtime, indicating potential over-optimization in prompted solutions.
"""
        
        report += """
### 5.2 Recommendations

1. **Use Detailed Prompts** for complex algorithmic problems where optimization matters
2. **Multiple Attempts** can significantly improve success rates (Pass@k > Pass@1)
3. **Human Review** remains essential for production deployment
4. **Performance Testing** should be conducted on all AI-generated solutions

---

## 6. Visualizations

See the generated PNG files in the reports directory for:
- Pass rate comparison by difficulty
- Runtime percentile distributions
- Memory usage analysis
- Error type breakdown
- Topic-specific performance

---

*Report generated by LeetCode AI Evaluator*
"""
        
        return report
