import os
import json
import pandas as pd
import shutil
import tempfile
import re
from collections import Counter
from typing import List, Dict, Any, Optional
from leetcode_evaluator.core.config import Config

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "leetcode-evaluator-mpl")
if not os.environ.get("MPLBACKEND"):
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AggregationManager:
    """Manages aggregation of results across multiple experiments and generation of paper-ready tables."""

    PAPER_MIN_NUM_PROBLEMS = 10
    PAPER_MIN_STABILITY_RUNS = 3
    PAPER_ALLOWED_GENERATION_MODES = {
        'realtime', 'qwen_batch', 'openai_batch', 'anthropic_batch',
        'gemini_batch', 'azure_batch', 'bedrock_batch',
    }
    
    def __init__(
        self,
        summary_dirs: Optional[List[str]] = None,
        raw_dirs: Optional[List[str]] = None,
        aggregate_tables_dir: Optional[str] = None,
        aggregate_figures_dir: Optional[str] = None,
    ):
        self.summary_dir = Config.RESULTS_SUMMARY
        self.tables_dir = Config.RESULTS_TABLES
        self.aggregate_summary_dir = Config.AGGREGATE_SUMMARY
        self.aggregate_raw_dir = Config.AGGREGATE_RAW
        self.aggregate_tables_dir = aggregate_tables_dir or Config.AGGREGATE_TABLES
        self.aggregate_figures_dir = aggregate_figures_dir or Config.AGGREGATE_FIGURES
        self.scan_summary_dirs = summary_dirs or [self.aggregate_summary_dir, self.summary_dir]
        self.scan_raw_dirs = raw_dirs or [self.aggregate_raw_dir, Config.RESULTS_RAW]
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.aggregate_summary_dir, exist_ok=True)
        os.makedirs(self.aggregate_raw_dir, exist_ok=True)
        os.makedirs(self.aggregate_tables_dir, exist_ok=True)
        os.makedirs(self.aggregate_figures_dir, exist_ok=True)

    def _load_single_summary(self, path: str) -> Dict:
        """Helper to load a single summary file with error handling."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """Render a Markdown table without requiring optional pandas dependencies."""
        if df.empty:
            return ""

        headers = [str(col) for col in df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in df.itertuples(index=False, name=None):
            cells = [str(cell) for cell in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _escape_latex(self, value: Any) -> str:
        text = str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _dataframe_to_latex(self, df: pd.DataFrame) -> str:
        """Render a simple LaTeX tabular without pandas optional dependencies."""
        if df.empty:
            return ""

        cols = "l" * len(df.columns)
        lines = [
            r"\begin{tabular}{" + cols + "}",
            r"\hline",
            " & ".join(self._escape_latex(col) for col in df.columns) + r" \\",
            r"\hline",
        ]
        for row in df.itertuples(index=False, name=None):
            lines.append(" & ".join(self._escape_latex(cell) for cell in row) + r" \\")
        lines.extend([r"\hline", r"\end{tabular}"])
        return "\n".join(lines)

    def _load_dataset(self, dataset: str = "main") -> List[Dict[str, Any]]:
        dataset_path = Config.resolve_dataset_file(dataset)
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Dataset file must contain a JSON list: {dataset_path}")
        return data

    def _load_archived_summaries(self) -> List[Dict[str, Any]]:
        """Load prompt-specific summaries from the global aggregate archive."""
        all_summaries = []
        scanned_paths = set()
        for scan_dir in self.scan_summary_dirs:
            if not os.path.isdir(scan_dir):
                continue
            for filename in os.listdir(scan_dir):
                if not filename.endswith(".json"):
                    continue
                path = os.path.join(scan_dir, filename)
                if path in scanned_paths:
                    continue
                summary = self._load_single_summary(path)
                scanned_paths.add(path)
                if not summary:
                    continue
                if summary.get('prompt_type') not in {'detailed', 'minimal'}:
                    continue
                summary.setdefault('dataset', None)
                summary.setdefault('num_problems', None)
                summary.setdefault('stability_runs', None)
                summary.setdefault('generation_mode', 'realtime')
                all_summaries.append(summary)

        return all_summaries

    def _extract_sort_key(self, experiment_name: str) -> str:
        if not experiment_name:
            return ""
        match = re.match(r"(\d{8}_\d{6})", str(experiment_name))
        if match:
            return match.group(1)
        return str(experiment_name)

    def _dedupe_summaries(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only the latest run for each equivalent evaluation configuration."""
        grouped: Dict[Any, Dict[str, Any]] = {}
        for summary in summaries:
            key = (
                summary.get('model_name'),
                summary.get('prompt_type'),
                summary.get('dataset'),
                summary.get('num_problems'),
                summary.get('stability_runs'),
                summary.get('generation_mode'),
                summary.get('temperature'),
                summary.get('top_p'),
                summary.get('max_tokens'),
            )
            current = grouped.get(key)
            if current is None or self._extract_sort_key(summary.get('experiment_name')) > self._extract_sort_key(current.get('experiment_name')):
                grouped[key] = summary
        return list(grouped.values())

    def _filter_paper_summaries(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter summaries down to paper-eligible rows only."""
        filtered = []
        for summary in summaries:
            dataset = summary.get('dataset')
            num_problems = summary.get('num_problems')
            stability_runs = summary.get('stability_runs')
            generation_mode = summary.get('generation_mode')
            if dataset != 'main':
                continue
            if generation_mode not in self.PAPER_ALLOWED_GENERATION_MODES:
                continue
            if num_problems is None:
                continue
            if int(num_problems) < self.PAPER_MIN_NUM_PROBLEMS:
                continue
            if stability_runs is None:
                continue
            if int(stability_runs) < self.PAPER_MIN_STABILITY_RUNS:
                continue
            filtered.append(summary)
        return filtered

    def _load_archived_trials(self) -> List[Dict[str, Any]]:
        """Load raw repeated-run trial rows from archived jsonl files."""
        trials: List[Dict[str, Any]] = []
        scanned_paths = set()

        for raw_dir in self.scan_raw_dirs:
            if not os.path.isdir(raw_dir):
                continue
            for filename in os.listdir(raw_dir):
                if not filename.endswith(".jsonl"):
                    continue
                path = os.path.join(raw_dir, filename)
                if path in scanned_paths:
                    continue
                scanned_paths.add(path)
                try:
                    with open(path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            trial = json.loads(line)
                            strategy = trial.get('strategy') or trial.get('prompt_type', 'default')
                            if strategy == 'with_prompt':
                                strategy = 'detailed'
                            elif strategy == 'without_prompt':
                                strategy = 'minimal'
                            trial['prompt_type'] = strategy
                            trial['accepted_bool'] = int(trial.get('accepted_bool', 0))
                            trials.append(trial)
                except Exception as e:
                    print(f"Error loading raw trials from {path}: {e}")

        return trials

    def _build_leaderboard(self, summaries: List[Dict[str, Any]]) -> pd.DataFrame:
        if not summaries:
            return pd.DataFrame()

        df = pd.DataFrame(summaries)

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

        column_mapping = {
            'model_name': 'Model',
            'prompt_type': 'Prompt',
            'dataset': 'Dataset',
            'num_problems': 'Num Problems',
            'stability_runs': 'Stability Runs',
            'generation_mode': 'Generation Mode',
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

        existing_cols = [c for c in column_mapping.keys() if c in df.columns]
        leaderboard = df[existing_cols].rename(columns=column_mapping)
        if 'Run-Level Pass Rate' in leaderboard.columns:
            leaderboard = leaderboard.sort_values(by='Run-Level Pass Rate', ascending=False)
        return leaderboard

    def generate_dataset_stats(self, dataset: str = "main") -> str:
        """Generate dataset statistics JSON/Markdown for the fixed evaluation dataset."""
        dataset_path = Config.resolve_dataset_file(dataset)
        dataset_rows = self._load_dataset(dataset)

        difficulty_counts = Counter(str(row.get('difficulty', 'Unknown')).title() for row in dataset_rows)
        topic_counts = Counter()
        for row in dataset_rows:
            for topic in row.get('topics', []):
                topic_counts[str(topic).strip()] += 1

        stats = {
            'dataset_name': dataset,
            'dataset_file': dataset_path,
            'num_problems': len(dataset_rows),
            'difficulty_counts': dict(sorted(difficulty_counts.items())),
            'top_topics': topic_counts.most_common(15),
        }

        json_path = os.path.join(self.aggregate_tables_dir, "dataset_stats.json")
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)

        md_path = os.path.join(self.aggregate_tables_dir, "dataset_stats.md")
        with open(md_path, 'w') as f:
            f.write("# Dataset Statistics\n\n")
            f.write(f"- Dataset: `{dataset}`\n")
            f.write(f"- Source file: `{dataset_path}`\n")
            f.write(f"- Number of problems: {len(dataset_rows)}\n\n")
            f.write("## Difficulty Distribution\n\n")
            f.write("| Difficulty | Count |\n|---|---:|\n")
            for difficulty, count in sorted(difficulty_counts.items()):
                f.write(f"| {difficulty} | {count} |\n")
            f.write("\n## Top Topics\n\n")
            f.write("| Topic | Count |\n|---|---:|\n")
            for topic, count in topic_counts.most_common(15):
                f.write(f"| {topic} | {count} |\n")

        return md_path

    def aggregate_leaderboard(self) -> str:
        """Scan results/summary and generate leaderboard tables."""
        all_summaries = self._dedupe_summaries(self._load_archived_summaries())
        if not all_summaries:
            return "No summary data found to aggregate."

        leaderboard = self._build_leaderboard(all_summaries)

        # Export CSV
        csv_path = os.path.join(self.aggregate_tables_dir, "leaderboard.csv")
        leaderboard.to_csv(csv_path, index=False)
        
        # Export Markdown
        md_path = os.path.join(self.aggregate_tables_dir, "leaderboard.md")
        with open(md_path, 'w') as f:
            f.write("# Overall Performance by Configuration\n\n")
            f.write(self._dataframe_to_markdown(leaderboard))

        tex_path = os.path.join(self.aggregate_tables_dir, "leaderboard.tex")
        with open(tex_path, 'w') as f:
            f.write(self._dataframe_to_latex(leaderboard))

        paper_inputs = {
            'num_rows': int(len(leaderboard)),
            'columns': list(leaderboard.columns),
            'rows': leaderboard.to_dict(orient='records'),
        }
        paper_inputs_path = os.path.join(self.aggregate_tables_dir, "paper_inputs.json")
        with open(paper_inputs_path, 'w') as f:
            json.dump(paper_inputs, f, indent=2)

        paper_summaries = self._filter_paper_summaries(all_summaries)
        paper_leaderboard = self._build_leaderboard(paper_summaries)

        paper_csv_path = os.path.join(self.aggregate_tables_dir, "paper_leaderboard.csv")
        paper_md_path = os.path.join(self.aggregate_tables_dir, "paper_leaderboard.md")
        paper_tex_path = os.path.join(self.aggregate_tables_dir, "paper_leaderboard.tex")
        paper_inputs_path = os.path.join(self.aggregate_tables_dir, "paper_leaderboard_inputs.json")

        paper_leaderboard.to_csv(paper_csv_path, index=False)
        with open(paper_md_path, 'w') as f:
            f.write("# Paper-Eligible Performance by Configuration\n\n")
            f.write(
                f"_Filters: dataset=`main`, num_problems>={self.PAPER_MIN_NUM_PROBLEMS}, "
                f"stability_runs>={self.PAPER_MIN_STABILITY_RUNS}, "
                f"generation_mode in {sorted(self.PAPER_ALLOWED_GENERATION_MODES)}._\n\n"
            )
            f.write(self._dataframe_to_markdown(paper_leaderboard))
        with open(paper_tex_path, 'w') as f:
            f.write(self._dataframe_to_latex(paper_leaderboard))
        with open(paper_inputs_path, 'w') as f:
            json.dump({
                'num_rows': int(len(paper_leaderboard)),
                'columns': list(paper_leaderboard.columns),
                'rows': paper_leaderboard.to_dict(orient='records'),
                'filters': {
                    'dataset': 'main',
                    'min_num_problems': self.PAPER_MIN_NUM_PROBLEMS,
                    'min_stability_runs': self.PAPER_MIN_STABILITY_RUNS,
                    'generation_modes': sorted(self.PAPER_ALLOWED_GENERATION_MODES),
                },
            }, f, indent=2)

        return md_path

    def generate_plots(self):
        """Generate multi-configuration plots for paper-ready aggregation."""
        all_summaries = self._dedupe_summaries(self._load_archived_summaries())
        if not all_summaries: return
        df = pd.DataFrame(all_summaries)
        paper_df = pd.DataFrame(self._filter_paper_summaries(all_summaries))
        fig_dir = self.aggregate_figures_dir
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

        if not paper_df.empty and 'run_level_pass_rate' in paper_df.columns and 'perfect_stability_rate' in paper_df.columns:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x='run_level_pass_rate',
                y='perfect_stability_rate',
                hue='model_name',
                style='prompt_type',
                s=100,
                data=paper_df
            )
            plt.title('Accuracy vs Stability (Paper-Eligible Configurations)')
            plt.xlabel('Run-Level Pass Rate (%)')
            plt.ylabel('Perfect Stability Rate (%)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(fig_dir, "paper_figure1_accuracy_vs_stability.png"), dpi=300)
            plt.close()

        self._generate_diagnostic_plots(all_summaries, fig_dir)

    def _generate_diagnostic_plots(self, all_summaries: List[Dict[str, Any]], fig_dir: str):
        trials = self._load_archived_trials()
        if not trials:
            return

        trial_df = pd.DataFrame(trials)
        if trial_df.empty:
            return

        summary_df = pd.DataFrame(self._filter_paper_summaries(all_summaries))
        if summary_df.empty:
            return

        # Figure 2: cross-model stability heatmap (problems × models)
        # Use "detailed" prompt; rows = models sorted by RLPR desc; cols = problems sorted by difficulty then pass rate
        detailed_trials = trial_df[trial_df['prompt_type'] == 'detailed'].copy()

        # Load difficulty labels from dataset
        difficulty_map: Dict[str, str] = {}
        dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'main-dataset.json')
        dataset_path = os.path.normpath(dataset_path)
        if os.path.exists(dataset_path):
            with open(dataset_path) as _f:
                for _p in json.load(_f):
                    difficulty_map[str(_p.get('frontend_question_id', ''))] = _p.get('difficulty', 'Unknown')

        # Paper-eligible model names (detailed prompt)
        paper_models = (
            summary_df[summary_df['prompt_type'] == 'detailed']
            .sort_values('run_level_pass_rate', ascending=False)['model_name']
            .tolist()
        )
        paper_models = list(dict.fromkeys(paper_models))  # deduplicate, preserve order

        if paper_models and not detailed_trials.empty:
            heatmap_trials = detailed_trials[detailed_trials['model_name'].isin(paper_models)].copy()
            heatmap_trials['problem_id'] = heatmap_trials['problem_id'].astype(str)

            # Compute mean pass rate per model × problem
            model_problem_rate = (
                heatmap_trials
                .groupby(['model_name', 'problem_id'])['accepted_bool']
                .mean()
                .reset_index()
            )
            matrix = model_problem_rate.pivot(index='model_name', columns='problem_id', values='accepted_bool').fillna(0)

            # Sort columns: by difficulty tier (Easy < Medium < Hard) then mean pass rate desc within tier
            diff_order_map = {'Easy': 0, 'Medium': 1, 'Hard': 2, 'Unknown': 3}
            col_sort_key = pd.DataFrame({
                'pid': matrix.columns,
                'diff_order': [diff_order_map.get(difficulty_map.get(c, 'Unknown'), 3) for c in matrix.columns],
                'mean_rate': matrix.mean(axis=0).values,
            })
            col_sort_key = col_sort_key.sort_values(['diff_order', 'mean_rate'], ascending=[True, False])
            matrix = matrix[col_sort_key['pid'].tolist()]
            matrix = matrix.reindex([m for m in paper_models if m in matrix.index])

            # Difficulty tier boundaries for vertical separator lines
            tier_boundaries = []
            prev_tier = col_sort_key.iloc[0]['diff_order']
            for i, row in enumerate(col_sort_key.itertuples()):
                if row.diff_order != prev_tier:
                    tier_boundaries.append(i)
                    prev_tier = row.diff_order

            # Short display names for y-axis
            model_display = {
                'gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
                'deepseek-r1': 'DeepSeek-R1',
                'gemini-3-flash-preview': 'Gemini 3 Flash',
                'qwq-plus': 'QwQ-Plus',
                'gemini-3.1-flash-lite-preview': 'Gemini 3.1 Flash-Lite',
                'qwen-plus': 'Qwen-Plus',
                'gpt-4.1': 'GPT-4.1',
                'gpt-4.1-mini': 'GPT-4.1-mini',
                'claude-sonnet-4-5-20250929': 'Claude Sonnet 4.5',
                'deepseek-v3.2': 'DeepSeek-V3.2',
                'claude-haiku-4-5-20251001': 'Claude Haiku 4.5',
                'qwen-turbo': 'Qwen-Turbo',
                'qwen-max': 'Qwen-Max',
            }
            matrix.index = [model_display.get(m, m) for m in matrix.index]

            fig, ax = plt.subplots(figsize=(20, 5))
            cmap = sns.color_palette(["#d73027", "#ffffbf", "#1a9850"], as_cmap=True)
            sns.heatmap(
                matrix, ax=ax, cmap=cmap, vmin=0, vmax=1, linewidths=0,
                cbar_kws={'label': 'Pass rate (0=always fail, 1=always pass)', 'shrink': 0.8, 'pad': 0.01},
            )

            for xpos in tier_boundaries:
                ax.axvline(x=xpos, color='gray', linewidth=1.5, linestyle='--')

            tier_names = {0: 'Easy', 1: 'Medium', 2: 'Hard'}
            tier_starts = [0] + tier_boundaries
            tier_ends = tier_boundaries + [len(matrix.columns)]
            tier_diff_orders = [
                int(col_sort_key[col_sort_key['pid'] == col_sort_key.iloc[s]['pid']]['diff_order'].iloc[0])
                for s in tier_starts
            ]
            for s, e, td in zip(tier_starts, tier_ends, tier_diff_orders):
                mid = (s + e) / 2
                ax.text(mid, -0.7, tier_names.get(td, ''), ha='center', va='bottom',
                        fontsize=11, fontweight='bold', transform=ax.get_xaxis_transform())

            ax.set_xlabel('Problems sorted by difficulty tier, then pass rate (high→low within tier)', fontsize=9)
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "paper_figure2_stability_heatmap.png"), dpi=150, bbox_inches='tight')
            plt.close()

        # Figure 3: problem-level success distribution across all paper-eligible models (detailed)
        if paper_models and not detailed_trials.empty:
            heatmap_trials = detailed_trials[detailed_trials['model_name'].isin(paper_models)].copy()
            problem_success = heatmap_trials.groupby(['problem_id', 'problem'])['accepted_bool'].mean()
            success_dist = problem_success.reset_index(name='success_rate')
            labels_5 = ['0/5', '1/5', '2/5', '3/5', '4/5', '5/5']
            success_dist['bucket'] = pd.cut(
                success_dist['success_rate'].round(4),
                bins=[-0.01, 0.01, 0.21, 0.41, 0.61, 0.81, 1.01],
                labels=labels_5,
                include_lowest=True
            )
            bucket_counts = success_dist['bucket'].value_counts().reindex(labels_5, fill_value=0)

            plt.figure(figsize=(7, 4.5))
            sns.barplot(x=bucket_counts.index, y=bucket_counts.values, color='#4c78a8')
            plt.title('Problem-Level Success Distribution (all models, detailed prompt)')
            plt.xlabel(r'Empirical mean pass rate $\bar{p}_i$ across models')
            plt.ylabel('Number of problems')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "paper_figure3_success_distribution.png"), dpi=300)
            plt.close()

        # Figure 4 compares prompts per model for RLPR and PSR with 95% CI.
        prompt_df = summary_df[summary_df['prompt_type'].isin(['detailed', 'minimal'])].copy()
        if prompt_df.empty:
            return

        model_order = (
            prompt_df.groupby('model_name')['run_level_pass_rate']
            .max()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        prompt_order = ['detailed', 'minimal']
        x = np.arange(len(model_order))
        width = 0.36
        colors = {'detailed': '#1f77b4', 'minimal': '#ff7f0e'}

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

        metric_specs = [
            ('run_level_pass_rate', 'run_level_ci_95', 'Run-Level Pass Rate (%)'),
            ('perfect_stability_rate', 'perfect_stability_ci_95', 'Perfect Stability Rate (%)'),
        ]

        for ax, (metric, ci_col, ylabel) in zip(axes, metric_specs):
            for idx, prompt in enumerate(prompt_order):
                subset = (
                    prompt_df[prompt_df['prompt_type'] == prompt]
                    .set_index('model_name')
                    .reindex(model_order)
                )
                values = subset[metric].astype(float).fillna(0).to_numpy()
                ci_bounds = subset[ci_col].tolist()
                lower_err = []
                upper_err = []
                for value, bounds in zip(values, ci_bounds):
                    if isinstance(bounds, list) and len(bounds) == 2:
                        lower_err.append(max(0.0, value - float(bounds[0])))
                        upper_err.append(max(0.0, float(bounds[1]) - value))
                    else:
                        lower_err.append(0.0)
                        upper_err.append(0.0)
                offsets = x + (idx - 0.5) * width
                ax.bar(
                    offsets,
                    values,
                    width=width,
                    label=prompt.capitalize(),
                    color=colors[prompt],
                    yerr=np.array([lower_err, upper_err]),
                    capsize=4,
                    alpha=0.9,
                )
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 100)
            ax.grid(True, axis='y', linestyle='--', alpha=0.4)

        axes[0].set_title('Prompt Effect on RLPR')
        axes[1].set_title('Prompt Effect on PSR')
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(model_order, rotation=15, ha='right')
        axes[1].legend(frameon=False, loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "paper_figure4_prompt_comparison.png"), dpi=300)
        plt.close(fig)

    def copy_raw_data(self, experiment_manager):
        """Copy experiment raw data to run-local and global raw archives."""
        raw_dir = Config.RESULTS_RAW
        os.makedirs(raw_dir, exist_ok=True)
        
        exp_name = os.path.basename(experiment_manager.experiment_dir)
        if os.path.exists(experiment_manager.jsonl_path):
            local_dest = os.path.join(raw_dir, f"{exp_name}.jsonl")
            global_dest = os.path.join(self.aggregate_raw_dir, f"{exp_name}.jsonl")
            shutil.copy2(experiment_manager.jsonl_path, local_dest)
            shutil.copy2(experiment_manager.jsonl_path, global_dest)
            print(f"✓ Raw data archived to {local_dest}")
            print(f"✓ Raw data archived to {global_dest}")

    def save_experiment_summary(self, experiment_name: str, metrics: Dict[str, Any], config: Dict[str, Any]):
        """Save a single experiment's metrics and config to run-local and global summary archives."""
        summary = {**config, **metrics}
        summary['experiment_name'] = experiment_name
        summary.setdefault('dataset', None)
        summary.setdefault('num_problems', None)
        summary.setdefault('stability_runs', None)
        summary.setdefault('generation_mode', 'realtime')
        
        local_path = os.path.join(self.summary_dir, f"{experiment_name}.json")
        global_path = os.path.join(self.aggregate_summary_dir, f"{experiment_name}.json")
        with open(local_path, 'w') as f:
            json.dump(summary, f, indent=4)
        with open(global_path, 'w') as f:
            json.dump(summary, f, indent=4)
