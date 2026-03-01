import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from leetcode_evaluator.core.config import Config

class ExperimentManager:
    """Handles structured logging and archiving of experiment results"""
    
    def __init__(self, provider: str, model_id: str, experiment_name: str = None):
        # Create experiment directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = self.timestamp
        if experiment_name:
            dir_name = f"{self.timestamp}_{experiment_name}"
            
        self.experiment_dir = os.path.join(Config.EXPERIMENTS_DIR, dir_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup files
        self.jsonl_path = os.path.join(self.experiment_dir, "detailed.jsonl")
        self.solutions_path = os.path.join(self.experiment_dir, "solutions.md")
        self.summary_path = os.path.join(self.experiment_dir, "summary.csv")
        self.stability_summary_path = os.path.join(self.experiment_dir, "stability_summary.json")
        self.log_path = os.path.join(self.experiment_dir, "execution.log")
        
        # Setup logging
        self._setup_logging()
        self.logger.info(f"Experiment started: {dir_name}")
        self.logger.info(f"Provider: {provider}, Model: {model_id}")
        if experiment_name:
            self.logger.info(f"Experiment Name: {experiment_name}")
        
        # Initialize solutions.md
        with open(self.solutions_path, 'w') as f:
            f.write(f"# Experiment Solutions: {self.timestamp}\n")
            f.write(f"- Provider: {provider}\n")
            f.write(f"- Model: {model_id}\n\n")

    def _setup_logging(self):
        """Setup standard application logging to execution.log"""
        self.logger = logging.getLogger("ExperimentManager")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_path)
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        
        # Also print to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def log_attempt(self, data: Dict[str, Any]):
        """Append a machine-readable JSON object to detailed.jsonl"""
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def log_solution(self, problem_title: str, strategy: str, code: str):
        """Append a formatted code block to solutions.md"""
        with open(self.solutions_path, 'a') as f:
            f.write(f"## {problem_title} ({strategy})\n")
            f.write("```python\n")
            f.write(code if code else "# No code generated")
            f.write("\n```\n\n")

    def save_summary(self, results_list: List[Dict[str, Any]]):
        """Save final metrics table to summary.csv"""
        if not results_list:
            return
            
        df = pd.DataFrame(results_list)
        df.to_csv(self.summary_path, index=False)
        self.logger.info(f"Summary saved to {self.summary_path}")

    def save_stability_summary(self, summary_data: Dict[str, Any]):
        """Save aggregated stability metrics to stability_summary.json"""
        with open(self.stability_summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        self.logger.info(f"Stability summary saved to {self.stability_summary_path}")

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)
