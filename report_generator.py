"""
Report generator for comprehensive analysis of evaluation results
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime
from scipy import stats
from collections import defaultdict

from config import Config


class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = self._load_results()
        self.df = self._create_dataframe()
        
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
                })
        
        return pd.DataFrame(rows)
    
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
        
        # Only consider accepted solutions
        accepted_df = self.df[self.df['status'] == 'Accepted']
        
        for prompt_type in ['with_prompt', 'without_prompt']:
            df_subset = accepted_df[accepted_df['prompt_type'] == prompt_type]
            
            if len(df_subset) > 0:
                # Runtime metrics
                runtime_percentiles = df_subset['runtime_percentile'].dropna()
                if len(runtime_percentiles) > 0:
                    metrics[f'{prompt_type}_mean_runtime_percentile'] = float(runtime_percentiles.mean())
                    metrics[f'{prompt_type}_median_runtime_percentile'] = float(runtime_percentiles.median())
                    metrics[f'{prompt_type}_std_runtime_percentile'] = float(runtime_percentiles.std())
                    metrics[f'{prompt_type}_top_25_runtime_pct'] = float(
                        (runtime_percentiles >= 75).sum() / len(runtime_percentiles) * 100
                    )
                
                # Memory metrics
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
    
    def generate_visualizations(self, output_dir: str = None):
        """Generate all visualization plots"""
        if output_dir is None:
            output_dir = Config.REPORT_DIR
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Pass rate comparison by difficulty
        self._plot_pass_rate_by_difficulty(f"{output_dir}/pass_rate_by_difficulty_{timestamp}.png")
        
        # 2. Runtime percentile distribution
        self._plot_runtime_distribution(f"{output_dir}/runtime_distribution_{timestamp}.png")
        
        # 3. Memory percentile distribution
        self._plot_memory_distribution(f"{output_dir}/memory_distribution_{timestamp}.png")
        
        # 4. Runtime vs Memory scatter
        self._plot_runtime_vs_memory(f"{output_dir}/runtime_vs_memory_{timestamp}.png")
        
        # 5. Error type comparison
        self._plot_error_types(f"{output_dir}/error_types_{timestamp}.png")
        
        # 6. Topic performance heatmap
        self._plot_topic_heatmap(f"{output_dir}/topic_performance_{timestamp}.png")
        
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
    
    def generate_full_report(self, output_file: str = None) -> str:
        """Generate complete analysis report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{Config.REPORT_DIR}/analysis_report_{timestamp}.md"
        
        # Calculate all metrics
        correctness = self.calculate_correctness_metrics()
        performance = self.calculate_performance_metrics()
        comparison = self.calculate_comparison_metrics()
        topic_metrics = self.calculate_topic_metrics()
        
        # Generate report
        report = self._generate_markdown_report(correctness, performance, comparison, topic_metrics)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        # Generate visualizations
        self.generate_visualizations()
        
        print(f"✓ Report generated: {output_file}")
        return output_file
    
    def _generate_markdown_report(self, correctness: Dict, performance: Dict, 
                                  comparison: Dict, topic_metrics: Dict) -> str:
        """Generate markdown formatted report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# LeetCode AI Solution Evaluation Report

**Generated:** {timestamp}  
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
        
        if 'with_prompt_mean_runtime_percentile' in performance:
            report += f"""
- **Mean Runtime Percentile (With Prompt):** {performance['with_prompt_mean_runtime_percentile']:.1f}%
- **Mean Runtime Percentile (Without Prompt):** {performance.get('without_prompt_mean_runtime_percentile', 0):.1f}%
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

### 2.1 Runtime Performance

"""
        
        if 'with_prompt_mean_runtime_percentile' in performance:
            report += f"""
| Metric | With Prompt | Without Prompt |
|--------|-------------|----------------|
| Mean Runtime Percentile | {performance.get('with_prompt_mean_runtime_percentile', 0):.1f}% | {performance.get('without_prompt_mean_runtime_percentile', 0):.1f}% |
| Median Runtime Percentile | {performance.get('with_prompt_median_runtime_percentile', 0):.1f}% | {performance.get('without_prompt_median_runtime_percentile', 0):.1f}% |
| Std Dev | {performance.get('with_prompt_std_runtime_percentile', 0):.1f} | {performance.get('without_prompt_std_runtime_percentile', 0):.1f} |
| Top 25% Rate | {performance.get('with_prompt_top_25_runtime_pct', 0):.1f}% | {performance.get('without_prompt_top_25_runtime_pct', 0):.1f}% |
"""
        
        report += """
### 2.2 Memory Performance

"""
        
        if 'with_prompt_mean_memory_percentile' in performance:
            report += f"""
| Metric | With Prompt | Without Prompt |
|--------|-------------|----------------|
| Mean Memory Percentile | {performance.get('with_prompt_mean_memory_percentile', 0):.1f}% | {performance.get('without_prompt_mean_memory_percentile', 0):.1f}% |
| Median Memory Percentile | {performance.get('with_prompt_median_memory_percentile', 0):.1f}% | {performance.get('without_prompt_median_memory_percentile', 0):.1f}% |
"""
        
        report += """
---

## 3. Prompt Impact Analysis

"""
        
        report += f"""
### 3.1 Effectiveness Metrics

- **Pass Rate Improvement:** {comparison.get('pass_rate_improvement', 0):.1f}%
- **Pass Rate Difference:** {comparison.get('pass_rate_difference', 0):+.1f} percentage points
"""
        
        if 'runtime_ttest_pvalue' in comparison:
            sig = "Yes" if comparison.get('runtime_statistically_significant', False) else "No"
            report += f"""
### 3.2 Statistical Significance

- **Runtime T-Test P-Value:** {comparison.get('runtime_ttest_pvalue', 1):.4f}
- **Statistically Significant (p<0.05):** {sig}
"""
        
        report += """
---

## 4. Conclusions

### 4.1 Key Insights

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
### 4.2 Recommendations

1. **Use Detailed Prompts** for complex algorithmic problems where optimization matters
2. **Multiple Attempts** can significantly improve success rates (Pass@k > Pass@1)
3. **Human Review** remains essential for production deployment
4. **Performance Testing** should be conducted on all AI-generated solutions

---

## 5. Visualizations

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
