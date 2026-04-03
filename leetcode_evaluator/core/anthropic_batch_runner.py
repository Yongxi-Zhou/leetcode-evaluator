"""
Coordinates multi-model Anthropic batch generation and sequential LeetCode submission.

Mirrors OpenAIBatchRunner lifecycle:
  submit → poll → collect → normalize → evaluate via LeetCode
"""
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any

from leetcode_evaluator.clients.llm.anthropic_batch import AnthropicBatchClient
from leetcode_evaluator.core.aggregation_manager import AggregationManager
from leetcode_evaluator.core.config import Config
from leetcode_evaluator.core.evaluator import LeetCodeEvaluator, RateLimitExhaustedException
from leetcode_evaluator.core.report_generator import ReportGenerator
from leetcode_evaluator.core.stability_metrics import StabilityAnalyzer


class AnthropicBatchRunner:
    """Coordinates multi-model Anthropic batch generation and sequential LeetCode submission."""

    MODEL_STATES = {
        'not_started', 'submitted', 'generation_in_progress',
        'generation_completed', 'submission_in_progress', 'completed', 'failed',
    }

    def __init__(self, config_path: str, run_id: str):
        self.config_path = config_path
        self.run_id = run_id
        self.batch_root = os.path.join(Config.BATCH_JOBS_ROOT, run_id)
        os.makedirs(self.batch_root, exist_ok=True)
        self.config = self._load_config(config_path)
        self.dataset_file = Config.resolve_dataset_file(self.config.get('dataset', 'main'))
        self.problems = self._load_problems()
        self.prompt_types = self.config.get('prompt_types', ['detailed', 'minimal'])
        self.stability_runs = int(self.config.get('stability_runs', Config.DEFAULT_STABILITY_RUNS))
        self.generation_params = {
            'temperature': self.config.get('temperature', Config.MODEL_TEMPERATURE),
            'top_p': self.config.get('top_p', Config.MODEL_TOP_P),
            'max_tokens': self.config.get('max_tokens', Config.MODEL_MAX_TOKENS),
        }

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            config = json.load(f)
        if 'models' not in config:
            raise ValueError(f"Missing required key 'models' in anthropic batch config: {path}")
        return config

    def _load_problems(self) -> List[Dict[str, Any]]:
        with open(self.dataset_file, 'r') as f:
            problems = json.load(f)
        if not isinstance(problems, list):
            raise ValueError(f"Dataset file must contain a JSON list: {self.dataset_file}")
        difficulty = self.config.get('difficulty')
        if difficulty:
            problems = [p for p in problems if str(p.get('difficulty', '')).upper() == difficulty.upper()]
        num_problems = int(self.config.get('num_problems', len(problems)))
        problems = problems[:num_problems]
        for problem in problems:
            backend = problem.get('backend_question_id') or problem.get('question_id')
            frontend = problem.get('frontend_question_id') or problem.get('problem_id')
            if not backend:
                raise ValueError(f"Problem missing backend question_id: {problem.get('title_slug')}")
            problem['question_id'] = str(backend)
            if frontend:
                problem['frontend_question_id'] = str(frontend)
        return problems

    def _safe_name(self, value: str) -> str:
        return re.sub(r'[^A-Za-z0-9._-]+', "_", str(value))

    def _job_dir(self, model_id: str, prompt_type: str) -> str:
        path = os.path.join(self.batch_root, self._safe_name(model_id), prompt_type)
        os.makedirs(path, exist_ok=True)
        return path

    def _job_paths(self, model_id: str, prompt_type: str) -> Dict[str, str]:
        job_dir = self._job_dir(model_id, prompt_type)
        return {
            'dir': job_dir,
            'input_jsonl': os.path.join(job_dir, 'input.jsonl'),
            'requests_manifest': os.path.join(job_dir, 'requests_manifest.json'),
            'job_metadata': os.path.join(job_dir, 'batch_job.json'),
            'output_jsonl': os.path.join(job_dir, 'output.jsonl'),
            'normalized_json': os.path.join(job_dir, 'normalized_generations.json'),
        }

    def _model_state_path(self, model_id: str) -> str:
        model_dir = os.path.join(self.batch_root, self._safe_name(model_id))
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, 'evaluation_state.json')

    def _write_json(self, path: str, data: Any):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_json(self, path: str) -> Any:
        with open(path, 'r') as f:
            return json.load(f)

    def _default_model_state(self, model_id: str) -> Dict[str, Any]:
        return {'model_id': model_id, 'status': 'not_started'}

    def _load_model_state(self, model_id: str) -> Dict[str, Any]:
        path = self._model_state_path(model_id)
        if os.path.exists(path):
            state = self._load_json(path)
            if state.get('status') not in self.MODEL_STATES:
                state['status'] = 'not_started'
            return state
        return self._default_model_state(model_id)

    def _save_model_state(self, model_id: str, state: Dict[str, Any]):
        self._write_json(self._model_state_path(model_id), state)

    def _job_status_local(self, model_id: str, prompt_type: str) -> Dict[str, Any]:
        paths = self._job_paths(model_id, prompt_type)
        metadata = self._load_json(paths['job_metadata']) if os.path.exists(paths['job_metadata']) else {}
        return {
            'job_id': metadata.get('job_id'),
            'status': str(metadata.get('processing_status', 'not_created')).lower(),
            'normalized_exists': os.path.exists(paths['normalized_json']),
            'paths': paths,
        }

    def _model_ready_state(self, model_id: str) -> str:
        if self._load_model_state(model_id).get('status') == 'completed':
            return 'EVALUATION_COMPLETED'
        detailed = self._job_status_local(model_id, 'detailed')
        minimal = self._job_status_local(model_id, 'minimal')
        dr = detailed['status'] == 'ended' and detailed['normalized_exists']
        mr = minimal['status'] == 'ended' and minimal['normalized_exists']
        if dr and mr:
            return 'READY_FOR_SUBMISSION'
        if dr:
            return 'WAITING_FOR_MINIMAL'
        if mr:
            return 'WAITING_FOR_DETAILED'
        return 'WAITING_FOR_BOTH'

    def _build_job_requests(self, client: AnthropicBatchClient, prompt_type: str):
        requests, manifest = [], []
        for trial_index in range(self.stability_runs):
            for problem in self.problems:
                req, man = client.build_request_entry(
                    problem=problem, prompt_type=prompt_type,
                    trial_index=trial_index, generation_params=self.generation_params,
                )
                requests.append(req)
                manifest.append(man)
        return requests, manifest

    def _finalize_job(self, client: AnthropicBatchClient, paths: Dict, metadata: Dict, manifest_entries: List):
        batch_id = metadata['job_id']
        if not os.path.exists(paths['output_jsonl']):
            print(f"Downloading batch results for job {batch_id}...")
            records = client.download_results(batch_id)
            client.save_results_jsonl(records, paths['output_jsonl'])
        output_records = client.load_jsonl_records(paths['output_jsonl'])
        normalized = client.normalize_batch_outputs(output_records, manifest_entries)
        self._write_json(paths['normalized_json'], normalized)
        return normalized

    def _ensure_job_state(self, model_id: str, prompt_type: str, mode: str = 'resume') -> Dict[str, Any]:
        client = AnthropicBatchClient(model_id=model_id)
        paths = self._job_paths(model_id, prompt_type)

        if os.path.exists(paths['requests_manifest']):
            manifest_entries = self._load_json(paths['requests_manifest'])
        else:
            request_entries, manifest_entries = self._build_job_requests(client, prompt_type)
            client.write_requests_jsonl(request_entries, paths['input_jsonl'])
            self._write_json(paths['requests_manifest'], manifest_entries)

        metadata = self._load_json(paths['job_metadata']) if os.path.exists(paths['job_metadata']) else None

        if metadata:
            status = str(metadata.get('processing_status', '')).lower()
            print(f"Resuming Anthropic batch job: model={model_id}, prompt={prompt_type}, status={status}, job_id={metadata.get('job_id')}")
            if mode == 'submit':
                return metadata
            if mode in ('collect', 'resume'):
                batch = client.retrieve_batch_job(metadata['job_id'])
                metadata.update({
                    'processing_status': batch.get('processing_status'),
                    'request_counts': batch.get('request_counts'),
                    'ended_at': batch.get('ended_at'),
                })
                self._write_json(paths['job_metadata'], metadata)
                if mode == 'collect':
                    return metadata
                status = str(metadata.get('processing_status', '')).lower()
            if mode == 'status':
                return metadata
            if status not in client.TERMINAL_STATES:
                batch = client.poll_batch_job(metadata['job_id'])
                metadata.update({
                    'processing_status': batch.get('processing_status'),
                    'request_counts': batch.get('request_counts'),
                    'ended_at': batch.get('ended_at'),
                })
                self._write_json(paths['job_metadata'], metadata)
            elif status in {'errored', 'canceled', 'expired'}:
                raise RuntimeError(f"Anthropic batch job for {model_id}/{prompt_type} in terminal failure: {status}")
        else:
            if mode in {'status', 'collect'}:
                raise RuntimeError(f"No batch job metadata found for {model_id}/{prompt_type}")
            print(f"Creating new Anthropic batch job: model={model_id}, prompt={prompt_type}")
            request_entries, manifest_entries = self._build_job_requests(client, prompt_type)
            client.write_requests_jsonl(request_entries, paths['input_jsonl'])
            self._write_json(paths['requests_manifest'], manifest_entries)
            batch = client.create_batch_job(request_entries)
            metadata = {
                'run_id': self.run_id, 'model_id': model_id,
                'prompt_type': prompt_type, 'num_problems': len(self.problems),
                'stability_runs': self.stability_runs,
                'job_id': batch['id'],
                'processing_status': batch.get('processing_status'),
                'request_counts': batch.get('request_counts'),
                'created_at': batch.get('created_at') or datetime.now().isoformat(),
                'ended_at': batch.get('ended_at'),
            }
            self._write_json(paths['job_metadata'], metadata)
            if mode == 'submit':
                return metadata
            if str(metadata.get('processing_status', '')).lower() not in client.TERMINAL_STATES:
                batch = client.poll_batch_job(metadata['job_id'])
                metadata.update({
                    'processing_status': batch.get('processing_status'),
                    'request_counts': batch.get('request_counts'),
                    'ended_at': batch.get('ended_at'),
                })
                self._write_json(paths['job_metadata'], metadata)

        return self._load_json(paths['job_metadata'])

    def _collect_job_outputs_if_ready(self, model_id: str, prompt_type: str):
        paths = self._job_paths(model_id, prompt_type)
        if os.path.exists(paths['normalized_json']):
            return self._job_status_local(model_id, prompt_type)
        metadata = self._ensure_job_state(model_id, prompt_type, mode='collect')
        status = str(metadata.get('processing_status', '')).lower()
        if status in {'errored', 'canceled', 'expired'}:
            raise RuntimeError(f"Anthropic batch job for {model_id}/{prompt_type} failed: {status}")
        if status != 'ended':
            return self._job_status_local(model_id, prompt_type)
        manifest_entries = self._load_json(paths['requests_manifest'])
        client = AnthropicBatchClient(model_id=model_id)
        self._finalize_job(client, paths, metadata, manifest_entries)
        return self._job_status_local(model_id, prompt_type)

    def _collect_model_outputs(self, model_id: str) -> str:
        model_state = self._load_model_state(model_id)
        if model_state.get('status') == 'completed':
            return 'EVALUATION_COMPLETED'
        failed = False
        for prompt_type in self.prompt_types:
            try:
                self._collect_job_outputs_if_ready(model_id, prompt_type)
            except Exception as exc:
                failed = True
                model_state.update({'status': 'failed', 'failed_at': datetime.now().isoformat(), 'error': str(exc)})
                self._save_model_state(model_id, model_state)
                print(f"✗ Failed collecting {model_id}/{prompt_type}: {exc}")
        ready_state = self._model_ready_state(model_id)
        if failed:
            return 'FAILED'
        if ready_state == 'READY_FOR_SUBMISSION':
            model_state.update({'status': 'generation_completed', 'generated_at': datetime.now().isoformat()})
        else:
            model_state.update({'status': 'generation_in_progress' if model_state.get('status') != 'submitted' else 'submitted'})
        self._save_model_state(model_id, model_state)
        return ready_state

    def _ensure_job_outputs(self, model_id: str, prompt_type: str):
        client = AnthropicBatchClient(model_id=model_id)
        paths = self._job_paths(model_id, prompt_type)
        if os.path.exists(paths['normalized_json']):
            print(f"Reusing normalized batch output: model={model_id}, prompt={prompt_type}")
            return self._load_json(paths['normalized_json'])
        metadata = self._ensure_job_state(model_id, prompt_type, mode='resume')
        manifest_entries = self._load_json(paths['requests_manifest'])
        status = str(metadata.get('processing_status', '')).lower()
        if status in {'errored', 'canceled', 'expired'}:
            raise RuntimeError(f"Anthropic batch job for {model_id}/{prompt_type} failed: {status}")
        if status != 'ended':
            raise RuntimeError(f"Anthropic batch job for {model_id}/{prompt_type} not completed yet: {status}")
        return self._finalize_job(client, paths, metadata, manifest_entries)

    def _build_generated_items_for_model(self, model_id: str) -> List[Dict[str, Any]]:
        normalized_by_prompt, manifest_by_prompt = {}, {}
        for prompt_type in self.prompt_types:
            normalized_by_prompt[prompt_type] = self._ensure_job_outputs(model_id, prompt_type)
            manifest_by_prompt[prompt_type] = self._load_json(self._job_paths(model_id, prompt_type)['requests_manifest'])
        by_problem = {
            p['title_slug']: {'problem': p, 'with_prompt': [], 'without_prompt': [], 'params': dict(self.generation_params)}
            for p in self.problems
        }
        for prompt_type, normalized_results in normalized_by_prompt.items():
            strategy_key = 'with_prompt' if prompt_type == 'detailed' else 'without_prompt'
            for entry in manifest_by_prompt[prompt_type]:
                by_problem[entry['title_slug']][strategy_key].append(
                    (int(entry['trial_index']), normalized_results[entry['custom_id']])
                )
        generated_items = []
        for problem in self.problems:
            item = by_problem[problem['title_slug']]
            for key in ('with_prompt', 'without_prompt'):
                item[key] = [a for _, a in sorted(item[key], key=lambda x: x[0])]
            generated_items.append(item)
        return generated_items

    def _archive_model_results(self, evaluator: LeetCodeEvaluator, results_file: str) -> str:
        exp_name_base = os.path.basename(evaluator.experiment_manager.experiment_dir)
        agg_manager = AggregationManager()
        analyzer = StabilityAnalyzer(detailed_jsonl_path=evaluator.experiment_manager.jsonl_path)
        metrics = analyzer.compute_metrics()
        config_data = {
            'experiment_name': exp_name_base, 'model_name': evaluator.llm_client.model_id,
            'prompt_type': 'standard', 'temperature': self.generation_params['temperature'],
            'top_p': self.generation_params['top_p'], 'max_tokens': self.generation_params['max_tokens'],
            'num_problems': len(self.problems), 'attempts': 1, 'stability_runs': self.stability_runs,
            'workers': 0, 'selection': 'BATCH', 'dataset': self.config.get('dataset', 'main'),
            'generation_mode': 'anthropic_batch',
        }
        agg_manager.save_experiment_summary(exp_name_base, metrics, config_data)
        agg_manager.copy_raw_data(evaluator.experiment_manager)
        report_dir = os.path.join(Config.REPORTS_DIR, exp_name_base)
        generator = ReportGenerator(results_file, output_dir=report_dir, model_name=evaluator.llm_client.model_id, provider='anthropic')
        report_file = generator.generate_full_report()
        prompt_metrics = analyzer.compute_metrics_by_strategy()
        for prompt_type, prompt_summary in prompt_metrics.items():
            agg_manager.save_experiment_summary(
                f"{exp_name_base}_{prompt_type}", prompt_summary,
                {**config_data, 'prompt_type': prompt_type},
            )
        agg_manager.copy_raw_data(evaluator.experiment_manager)
        return report_file

    def _results_coverage(self, results_file: str) -> int:
        if not os.path.exists(results_file):
            return 0
        with open(results_file) as f:
            saved = json.load(f)
        count = 0
        for r in saved:
            needs_retry = any(
                attempt.get('submission_id') is None and attempt.get('status') == 'Submission Failed'
                for strategy in ['with_prompt', 'without_prompt']
                for attempt in r.get(strategy, [])
            )
            if not needs_retry:
                count += 1
        return count

    def _submit_jobs_for_model(self, model_id: str):
        model_state = self._load_model_state(model_id)
        if model_state.get('status') == 'completed':
            print(f"Skipping completed model run: {model_id}")
            return
        for prompt_type in self.prompt_types:
            self._ensure_job_state(model_id, prompt_type, mode='submit')
            info = self._job_paths(model_id, prompt_type)
            meta = self._load_json(info['job_metadata']) if os.path.exists(info['job_metadata']) else {}
            print(f"  - model={model_id} prompt={prompt_type} job_id={meta.get('job_id', 'n/a')} status={meta.get('processing_status', 'n/a')}")
        model_state.update({'status': 'submitted', 'submitted_at': datetime.now().isoformat()})
        self._save_model_state(model_id, model_state)

    def print_status(self):
        print(f"Run id: {self.run_id}\nBatch root: {self.batch_root}")
        for model_id in self.config['models']:
            model_state = self._load_model_state(model_id)
            ready_state = self._model_ready_state(model_id)
            print(f"\n{'-'*60}\nANTHROPIC BATCH MODEL: {model_id}")
            print(f"State: {model_state.get('status')} | Ready: {ready_state}\n{'-'*60}")
            for prompt_type in self.prompt_types:
                paths = self._job_paths(model_id, prompt_type)
                meta = self._load_json(paths['job_metadata']) if os.path.exists(paths['job_metadata']) else {}
                normalized = os.path.exists(paths['normalized_json'])
                print(f"  - prompt={prompt_type} job_id={meta.get('job_id', 'n/a')} status={meta.get('processing_status', 'n/a')} normalized={'yes' if normalized else 'no'}")

    def run(self, mode: str = 'resume', filter_models=None, filter_prompts=None,
            num_problems=None, num_trials=None, problem_offset=None, skip_premium=False):
        results = []
        print(f"Run id: {self.run_id}\nBatch root: {self.batch_root}")
        if filter_models:
            print(f"Filter models: {filter_models}")
        if filter_prompts:
            self.prompt_types = [p for p in self.prompt_types if p in filter_prompts]
        models_to_run = [m for m in self.config['models'] if m in filter_models] if filter_models else self.config['models']

        if mode == 'status':
            self.print_status()
            return results

        for model_id in models_to_run:
            print(f"\n{'-'*60}\nANTHROPIC BATCH MODEL: {model_id}\n{'-'*60}")
            model_state = self._load_model_state(model_id)
            try:
                if mode == 'submit':
                    self._submit_jobs_for_model(model_id)
                    continue
                if mode == 'collect':
                    ready_state = self._collect_model_outputs(model_id)
                    print(f"Model ready state after collect: {ready_state}")
                    continue
                if model_state.get('status') == 'completed':
                    print(f"Skipping completed model run: {model_id}")
                    results.append({'model_id': model_id, 'results_file': model_state.get('results_file'), 'report_file': model_state.get('report_file')})
                    continue

                ready_state = self._collect_model_outputs(model_id)
                if ready_state != 'READY_FOR_SUBMISSION':
                    print(f"Skipping model not ready: {model_id} ({ready_state})")
                    continue

                model_state.update({'status': 'submission_in_progress', 'submission_started_at': datetime.now().isoformat()})
                self._save_model_state(model_id, model_state)
                generated_items = self._build_generated_items_for_model(model_id)

                if skip_premium:
                    before = len(generated_items)
                    generated_items = [i for i in generated_items if not i['problem'].get('isPaidOnly', False)]
                    print(f"Premium filter: {before} → {len(generated_items)} problems")
                if problem_offset is not None:
                    generated_items = generated_items[problem_offset:]
                    print(f"Problem offset {problem_offset}: {len(generated_items)} problems remaining")
                if num_problems is not None:
                    generated_items = generated_items[:num_problems]
                    print(f"Limiting to {num_problems} problems")
                if num_trials is not None:
                    for item in generated_items:
                        item['with_prompt'] = item['with_prompt'][:num_trials]
                        item['without_prompt'] = item['without_prompt'][:num_trials]
                    print(f"Limiting to {num_trials} trials per problem")

                active_results_file = model_state.get('results_file_in_progress')
                if not active_results_file:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs(Config.RESULTS_EVALUATIONS, exist_ok=True)
                    active_results_file = os.path.join(Config.RESULTS_EVALUATIONS, f"evaluation_results_{timestamp}.json")
                    model_state['results_file_in_progress'] = active_results_file
                    self._save_model_state(model_id, model_state)
                    print(f"Results file: {active_results_file}")
                else:
                    print(f"Resuming from results file: {active_results_file}")

                evaluator = LeetCodeEvaluator(provider='anthropic', model_id=model_id, experiment_name=self._safe_name(model_id))
                results_file = evaluator.run_generated_evaluation(generated_items, stability_runs=self.stability_runs, results_file=active_results_file)
                if not results_file:
                    raise RuntimeError(f"Generated evaluation failed for model: {model_id}")

                coverage = self._results_coverage(results_file)
                total = len(self.problems)
                if coverage >= total:
                    report_file = self._archive_model_results(evaluator, results_file)
                    model_state.update({'status': 'completed', 'completed_at': datetime.now().isoformat(), 'results_file': results_file, 'report_file': report_file})
                    self._save_model_state(model_id, model_state)
                    results.append({'model_id': model_id, 'results_file': results_file, 'report_file': report_file})
                    print(f"✓ Model fully complete: {coverage}/{total} problems")
                else:
                    model_state.update({'status': 'generation_completed', 'results_file_in_progress': results_file})
                    self._save_model_state(model_id, model_state)
                    results.append({'model_id': model_id, 'results_file': results_file, 'report_file': None})
                    print(f"Partial run: {coverage}/{total} done. Re-run with --batch-problem-offset {coverage} to continue.")
            except RateLimitExhaustedException as exc:
                model_state.update({'status': 'generation_completed', 'rate_limited_at': datetime.now().isoformat(), 'error': str(exc)})
                self._save_model_state(model_id, model_state)
                print(f"\n{'!'*60}\nRATE LIMIT EXHAUSTED for model: {model_id}\nProgress saved. Switch LeetCode account and re-run.\n{'!'*60}\n")
                raise
            except Exception as exc:
                if mode == 'resume':
                    model_state.update({'status': 'failed', 'failed_at': datetime.now().isoformat(), 'error': str(exc)})
                    self._save_model_state(model_id, model_state)
                print(f"✗ Model failed: {model_id}: {exc}")
        return results
