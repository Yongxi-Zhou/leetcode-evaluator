import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any

from leetcode_evaluator.clients.llm.qwen_batch import QwenBatchClient
from leetcode_evaluator.core.aggregation_manager import AggregationManager
from leetcode_evaluator.core.config import Config
from leetcode_evaluator.core.evaluator import LeetCodeEvaluator, RateLimitExhaustedException
from leetcode_evaluator.core.report_generator import ReportGenerator
from leetcode_evaluator.core.stability_metrics import StabilityAnalyzer


class QwenBatchRunner:
    """Coordinates multi-model Qwen batch generation and sequential LeetCode submission."""

    MODEL_STATES = {
        'not_started',
        'submitted',
        'generation_in_progress',
        'generation_completed',
        'submission_in_progress',
        'completed',
        'failed',
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
        required = ['models']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required key in qwen batch config: {key}")
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
                raise ValueError(
                    f"Problem is missing backend question_id required for submission: {problem.get('title_slug')}"
                )
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
            'error_jsonl': os.path.join(job_dir, 'error.jsonl'),
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
        return {
            'model_id': model_id,
            'status': 'not_started',
        }

    def _job_status_local(self, model_id: str, prompt_type: str) -> Dict[str, Any]:
        paths = self._job_paths(model_id, prompt_type)
        metadata = self._load_json(paths['job_metadata']) if os.path.exists(paths['job_metadata']) else {}
        return {
            'job_id': metadata.get('job_id'),
            'status': str(metadata.get('status', 'not_created')).lower(),
            'normalized_exists': os.path.exists(paths['normalized_json']),
            'metadata_exists': bool(metadata),
            'paths': paths,
        }

    def _model_ready_state(self, model_id: str) -> str:
        model_state = self._load_model_state(model_id)
        if model_state.get('status') == 'completed':
            return 'EVALUATION_COMPLETED'

        detailed = self._job_status_local(model_id, 'detailed')
        minimal = self._job_status_local(model_id, 'minimal')
        detailed_ready = detailed['status'] == 'completed' and detailed['normalized_exists']
        minimal_ready = minimal['status'] == 'completed' and minimal['normalized_exists']

        if detailed_ready and minimal_ready:
            return 'READY_FOR_SUBMISSION'
        if detailed_ready and not minimal_ready:
            return 'WAITING_FOR_MINIMAL'
        if minimal_ready and not detailed_ready:
            return 'WAITING_FOR_DETAILED'
        return 'WAITING_FOR_BOTH'

    def _build_job_requests(
        self,
        client: QwenBatchClient,
        prompt_type: str,
    ) -> List[Dict[str, Any]]:
        requests = []
        manifest = []
        for trial_index in range(self.stability_runs):
            for problem in self.problems:
                request_entry, manifest_entry = client.build_request_entry(
                    problem=problem,
                    prompt_type=prompt_type,
                    trial_index=trial_index,
                    generation_params=self.generation_params,
                )
                requests.append(request_entry)
                manifest.append(manifest_entry)
        return requests, manifest

    def _finalize_job(
        self,
        client: QwenBatchClient,
        paths: Dict[str, str],
        metadata: Dict[str, Any],
        manifest_entries: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        output_file_id = metadata.get('output_file_id')
        error_file_id = metadata.get('error_file_id')

        if output_file_id and not os.path.exists(paths['output_jsonl']):
            client.download_file(output_file_id, paths['output_jsonl'])
        if error_file_id and not os.path.exists(paths['error_jsonl']):
            client.download_file(error_file_id, paths['error_jsonl'])

        output_records = client.load_jsonl_records(paths['output_jsonl'])
        error_records = client.load_jsonl_records(paths['error_jsonl'])
        normalized = client.normalize_batch_outputs(output_records, error_records, manifest_entries)
        self._write_json(paths['normalized_json'], normalized)
        return normalized

    def _collect_job_metadata(self, model_id: str, prompt_type: str) -> Dict[str, Any]:
        paths = self._job_paths(model_id, prompt_type)
        metadata = self._load_json(paths['job_metadata']) if os.path.exists(paths['job_metadata']) else {}
        normalized_exists = os.path.exists(paths['normalized_json'])
        return {
            'model_id': model_id,
            'prompt_type': prompt_type,
            'job_id': metadata.get('job_id'),
            'status': metadata.get('status', 'not_created'),
            'normalized_exists': normalized_exists,
            'evaluation_completed': self._load_model_state(model_id).get('status') == 'completed',
            'paths': paths,
        }

    def _print_job_metadata(self, model_id: str, prompt_type: str):
        info = self._collect_job_metadata(model_id, prompt_type)
        print(
            f"  - model={model_id} prompt={prompt_type} "
            f"job_id={info['job_id'] or 'n/a'} status={info['status']} "
            f"normalized={'yes' if info['normalized_exists'] else 'no'} "
            f"evaluation_completed={'yes' if info['evaluation_completed'] else 'no'}"
        )

    def _refresh_job_status(self, model_id: str, prompt_type: str) -> Dict[str, Any]:
        paths = self._job_paths(model_id, prompt_type)
        if not os.path.exists(paths['job_metadata']):
            raise RuntimeError(f"No batch job metadata found for {model_id}/{prompt_type}")
        metadata = self._load_json(paths['job_metadata'])
        client = QwenBatchClient(model_id=model_id)
        batch = client.retrieve_batch_job(metadata['job_id'])
        metadata.update({
            'status': batch.get('status'),
            'output_file_id': batch.get('output_file_id'),
            'error_file_id': batch.get('error_file_id'),
            'completed_at': batch.get('completed_at'),
            'batch': batch,
        })
        self._write_json(paths['job_metadata'], metadata)
        return metadata

    def _ensure_job_state(self, model_id: str, prompt_type: str, mode: str = 'resume') -> Dict[str, Any]:
        client = QwenBatchClient(model_id=model_id)
        paths = self._job_paths(model_id, prompt_type)

        if os.path.exists(paths['requests_manifest']):
            manifest_entries = self._load_json(paths['requests_manifest'])
        else:
            request_entries, manifest_entries = self._build_job_requests(client, prompt_type)
            client.write_requests_jsonl(request_entries, paths['input_jsonl'])
            self._write_json(paths['requests_manifest'], manifest_entries)

        metadata = self._load_json(paths['job_metadata']) if os.path.exists(paths['job_metadata']) else None
        if metadata:
            status = str(metadata.get('status', '')).lower()
            print(f"Resuming batch job: model={model_id}, prompt={prompt_type}, status={status}, job_id={metadata.get('job_id')}")
            if mode == 'submit':
                return metadata
            if mode == 'collect':
                batch = client.retrieve_batch_job(metadata['job_id'])
                metadata.update({
                    'status': batch.get('status'),
                    'output_file_id': batch.get('output_file_id'),
                    'error_file_id': batch.get('error_file_id'),
                    'completed_at': batch.get('completed_at'),
                    'batch': batch,
                })
                self._write_json(paths['job_metadata'], metadata)
                return metadata
            if mode == 'status':
                return metadata
            if status in {'pending', 'validating', 'in_progress', 'finalizing'}:
                batch = client.poll_batch_job(metadata['job_id'])
                metadata.update({
                    'status': batch.get('status'),
                    'output_file_id': batch.get('output_file_id'),
                    'error_file_id': batch.get('error_file_id'),
                    'completed_at': batch.get('completed_at'),
                    'batch': batch,
                })
                self._write_json(paths['job_metadata'], metadata)
            elif status == 'completed' and not metadata.get('output_file_id'):
                batch = client.retrieve_batch_job(metadata['job_id'])
                metadata.update({
                    'status': batch.get('status'),
                    'output_file_id': batch.get('output_file_id'),
                    'error_file_id': batch.get('error_file_id'),
                    'completed_at': batch.get('completed_at'),
                    'batch': batch,
                })
                self._write_json(paths['job_metadata'], metadata)
            elif status in {'failed', 'expired', 'cancelled'}:
                raise RuntimeError(f"Batch job for {model_id}/{prompt_type} is in terminal failure state: {status}")
        else:
            if mode in {'status', 'collect'}:
                raise RuntimeError(f"No batch job metadata found for {model_id}/{prompt_type}")
            print(f"Creating new batch job: model={model_id}, prompt={prompt_type}")
            request_entries, manifest_entries = self._build_job_requests(client, prompt_type)
            client.write_requests_jsonl(request_entries, paths['input_jsonl'])
            self._write_json(paths['requests_manifest'], manifest_entries)
            uploaded_file = client.upload_batch_file(paths['input_jsonl'])
            batch = client.create_batch_job(
                input_file_id=uploaded_file['id'],
                metadata={
                    'run_id': self.run_id,
                    'dataset_file': self.dataset_file,
                    'num_problems': len(self.problems),
                    'difficulty': self.config.get('difficulty'),
                    'stability_runs': self.stability_runs,
                    'model_id': model_id,
                    'prompt_type': prompt_type,
                }
            )
            metadata = {
                'run_id': self.run_id,
                'dataset_file': self.dataset_file,
                'num_problems': len(self.problems),
                'difficulty': self.config.get('difficulty'),
                'stability_runs': self.stability_runs,
                'model_id': model_id,
                'prompt_type': prompt_type,
                'job_id': batch['id'],
                'input_file_id': uploaded_file['id'],
                'output_file_id': batch.get('output_file_id'),
                'error_file_id': batch.get('error_file_id'),
                'status': batch.get('status'),
                'created_at': batch.get('created_at') or datetime.now().isoformat(),
                'completed_at': batch.get('completed_at'),
                'batch': batch,
            }
            self._write_json(paths['job_metadata'], metadata)
            if mode == 'submit':
                return metadata
            if mode in {'status', 'collect'}:
                return metadata
            if str(metadata.get('status', '')).lower() not in client.TERMINAL_STATES:
                batch = client.poll_batch_job(metadata['job_id'])
                metadata.update({
                    'status': batch.get('status'),
                    'output_file_id': batch.get('output_file_id'),
                    'error_file_id': batch.get('error_file_id'),
                    'completed_at': batch.get('completed_at'),
                    'batch': batch,
                })
                self._write_json(paths['job_metadata'], metadata)

        return self._load_json(paths['job_metadata'])

    def _collect_job_outputs_if_ready(self, model_id: str, prompt_type: str) -> Dict[str, Any]:
        paths = self._job_paths(model_id, prompt_type)
        if os.path.exists(paths['normalized_json']):
            return self._job_status_local(model_id, prompt_type)

        metadata = self._ensure_job_state(model_id, prompt_type, mode='collect')
        status = str(metadata.get('status', '')).lower()
        if status in {'failed', 'expired', 'cancelled'}:
            raise RuntimeError(f"Batch job for {model_id}/{prompt_type} failed: {status}")
        if status != 'completed':
            return self._job_status_local(model_id, prompt_type)

        manifest_entries = self._load_json(paths['requests_manifest'])
        client = QwenBatchClient(model_id=model_id)
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
                model_state.update({
                    'status': 'failed',
                    'failed_at': datetime.now().isoformat(),
                    'error': str(exc),
                })
                self._save_model_state(model_id, model_state)
                print(f"✗ Failed collecting {model_id}/{prompt_type}: {exc}")

        ready_state = self._model_ready_state(model_id)
        if failed:
            return 'FAILED'
        if ready_state == 'READY_FOR_SUBMISSION':
            model_state.update({
                'status': 'generation_completed',
                'generated_at': datetime.now().isoformat(),
            })
        else:
            model_state.update({
                'status': 'generation_in_progress' if model_state.get('status') != 'submitted' else 'submitted'
            })
        self._save_model_state(model_id, model_state)
        return ready_state

    def _ensure_job_outputs(self, model_id: str, prompt_type: str) -> Dict[str, Dict[str, Any]]:
        client = QwenBatchClient(model_id=model_id)
        paths = self._job_paths(model_id, prompt_type)

        if os.path.exists(paths['normalized_json']):
            print(f"Reusing normalized batch output: model={model_id}, prompt={prompt_type}")
            return self._load_json(paths['normalized_json'])

        metadata = self._ensure_job_state(model_id, prompt_type, mode='resume')
        manifest_entries = self._load_json(paths['requests_manifest'])
        status = str(metadata.get('status', '')).lower()
        if status in {'failed', 'expired', 'cancelled'}:
            raise RuntimeError(f"Batch job for {model_id}/{prompt_type} failed: {status}")
        if status != 'completed':
            raise RuntimeError(
                f"Batch job for {model_id}/{prompt_type} is not completed yet: {status}"
            )
        return self._finalize_job(client, paths, metadata, manifest_entries)

    def _build_generated_items_for_model(self, model_id: str) -> List[Dict[str, Any]]:
        normalized_by_prompt = {}
        manifest_by_prompt = {}
        for prompt_type in self.prompt_types:
            normalized_by_prompt[prompt_type] = self._ensure_job_outputs(model_id, prompt_type)
            manifest_by_prompt[prompt_type] = self._load_json(self._job_paths(model_id, prompt_type)['requests_manifest'])

        by_problem = {
            problem['title_slug']: {
                'problem': problem,
                'with_prompt': [],
                'without_prompt': [],
                'params': dict(self.generation_params),
            }
            for problem in self.problems
        }

        for prompt_type, normalized_results in normalized_by_prompt.items():
            strategy_key = 'with_prompt' if prompt_type == 'detailed' else 'without_prompt'
            for manifest_entry in manifest_by_prompt[prompt_type]:
                custom_id = manifest_entry['custom_id']
                attempt = normalized_results[custom_id]
                by_problem[manifest_entry['title_slug']][strategy_key].append(
                    (int(manifest_entry['trial_index']), attempt)
                )

        generated_items = []
        for problem in self.problems:
            item = by_problem[problem['title_slug']]
            for key in ('with_prompt', 'without_prompt'):
                item[key] = [attempt for _, attempt in sorted(item[key], key=lambda pair: pair[0])]
            generated_items.append(item)
        return generated_items

    def _archive_model_results(self, evaluator: LeetCodeEvaluator, results_file: str):
        exp_name_base = os.path.basename(evaluator.experiment_manager.experiment_dir)
        agg_manager = AggregationManager()
        analyzer = StabilityAnalyzer(detailed_jsonl_path=evaluator.experiment_manager.jsonl_path)
        metrics = analyzer.compute_metrics()

        config_data = {
            'experiment_name': exp_name_base,
            'model_name': evaluator.llm_client.model_id,
            'prompt_type': 'standard',
            'temperature': self.generation_params['temperature'],
            'top_p': self.generation_params['top_p'],
            'max_tokens': self.generation_params['max_tokens'],
            'num_problems': len(self.problems),
            'attempts': 1,
            'stability_runs': self.stability_runs,
            'workers': 0,
            'selection': 'BATCH',
            'dataset': self.config.get('dataset', 'main'),
            'generation_mode': 'qwen_batch'
        }
        agg_manager.save_experiment_summary(exp_name_base, metrics, config_data)
        agg_manager.copy_raw_data(evaluator.experiment_manager)

        report_dir = os.path.join(Config.REPORTS_DIR, exp_name_base)
        generator = ReportGenerator(
            results_file, output_dir=report_dir,
            model_name=evaluator.llm_client.model_id,
            provider='qwen')
        report_file = generator.generate_full_report()

        prompt_metrics = analyzer.compute_metrics_by_strategy()
        for prompt_type, prompt_summary in prompt_metrics.items():
            prompt_config = {
                'model_name': evaluator.llm_client.model_id,
                'prompt_type': prompt_type,
                'temperature': self.generation_params['temperature'],
                'top_p': self.generation_params['top_p'],
                'max_tokens': self.generation_params['max_tokens'],
                'provider': 'qwen',
                'num_problems': len(self.problems),
                'stability_runs': self.stability_runs,
                'dataset': self.config.get('dataset', 'main'),
                'generation_mode': 'qwen_batch'
            }
            agg_manager.save_experiment_summary(
                f"{exp_name_base}_{prompt_type}",
                prompt_summary,
                prompt_config
            )
        agg_manager.copy_raw_data(evaluator.experiment_manager)
        return report_file

    def _results_coverage(self, results_file: str) -> int:
        """Count problems in results_file where every trial has a real submission_id.

        A problem with any trial returning submission_id=None (API/rate-limit failure)
        is NOT counted as covered, so that round will be retried.
        """
        if not os.path.exists(results_file):
            return 0
        with open(results_file) as f:
            saved = json.load(f)
        count = 0
        for r in saved:
            needs_retry = any(
                attempt.get('submission_id') is None
                and attempt.get('status') == 'Submission Failed'
                for strategy in ['with_prompt', 'without_prompt']
                for attempt in r.get(strategy, [])
            )
            if not needs_retry:
                count += 1
        return count

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

    def _submit_jobs_for_model(self, model_id: str):
        model_state = self._load_model_state(model_id)
        if model_state.get('status') == 'completed':
            print(f"Skipping completed model run: {model_id}")
            return
        for prompt_type in self.prompt_types:
            self._ensure_job_state(model_id, prompt_type, mode='submit')
            self._print_job_metadata(model_id, prompt_type)
        model_state.update({
            'status': 'submitted',
            'submitted_at': datetime.now().isoformat(),
        })
        self._save_model_state(model_id, model_state)

    def print_status(self) -> List[Dict[str, Any]]:
        statuses = []
        print(f"Stable run id: {self.run_id}")
        print(f"Batch root: {self.batch_root}")
        ready_count = 0
        completed_count = 0
        pending_count = 0
        for model_id in self.config['models']:
            model_state = self._load_model_state(model_id)
            ready_state = self._model_ready_state(model_id)
            print(f"\n{'-' * 60}")
            print(f"QWEN BATCH MODEL: {model_id}")
            print(f"Model evaluation state: {model_state.get('status')}")
            print(f"Model ready state: {ready_state}")
            print(f"{'-' * 60}")
            model_status = {
                'model_id': model_id,
                'evaluation_state': model_state.get('status'),
                'ready_state': ready_state,
                'jobs': [],
            }
            if ready_state == 'READY_FOR_SUBMISSION':
                ready_count += 1
            elif ready_state == 'EVALUATION_COMPLETED':
                completed_count += 1
            else:
                pending_count += 1
            for prompt_type in self.prompt_types:
                try:
                    metadata = self._load_json(self._job_paths(model_id, prompt_type)['job_metadata'])
                    info = self._collect_job_metadata(model_id, prompt_type)
                    info['status'] = metadata.get('status', info['status'])
                    self._print_job_metadata(model_id, prompt_type)
                except Exception as exc:
                    info = {
                        'model_id': model_id,
                        'prompt_type': prompt_type,
                        'job_id': None,
                        'status': f"error: {exc}",
                        'normalized_exists': False,
                        'evaluation_completed': model_state.get('status') == 'completed',
                    }
                    print(
                        f"  - model={model_id} prompt={prompt_type} "
                        f"job_id=n/a status=error: {exc} normalized=no "
                        f"evaluation_completed={'yes' if info['evaluation_completed'] else 'no'}"
                    )
                model_status['jobs'].append(info)
            statuses.append(model_status)
        print(f"\nRun summary: ready_models={ready_count} completed_models={completed_count} pending_models={pending_count}")
        return statuses

    def run(self, mode: str = 'resume', filter_models: List[str] = None, filter_prompts: List[str] = None, num_problems: int = None, num_trials: int = None, problem_offset: int = None, skip_premium: bool = False) -> List[Dict[str, Any]]:
        results = []
        print(f"Stable run id: {self.run_id}")
        print(f"Batch root: {self.batch_root}")

        # Apply filters
        if filter_models:
            print(f"Filter models: {filter_models}")
        if filter_prompts:
            print(f"Filter prompts: {filter_prompts}")
            self.prompt_types = [p for p in self.prompt_types if p in filter_prompts]
        if skip_premium:
            print(f"Skipping isPaidOnly problems")
        if problem_offset is not None:
            print(f"Problem offset: {problem_offset}")

        models_to_run = [m for m in self.config['models'] if m in filter_models] if filter_models else self.config['models']

        if mode == 'status':
            self.print_status()
            return results

        for model_id in models_to_run:
            print(f"\n{'-' * 60}")
            print(f"QWEN BATCH MODEL: {model_id}")
            print(f"{'-' * 60}")
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
                    results.append({
                        'model_id': model_id,
                        'results_file': model_state.get('results_file'),
                        'report_file': model_state.get('report_file'),
                    })
                    continue

                ready_state = self._collect_model_outputs(model_id)
                if ready_state != 'READY_FOR_SUBMISSION':
                    print(f"Skipping model not ready for submission: {model_id} ({ready_state})")
                    continue

                model_state.update({
                    'status': 'submission_in_progress',
                    'submission_started_at': datetime.now().isoformat(),
                })
                self._save_model_state(model_id, model_state)
                generated_items = self._build_generated_items_for_model(model_id)

                # Filter out premium-only problems when requested
                if skip_premium:
                    before = len(generated_items)
                    generated_items = [
                        item for item in generated_items
                        if not item['problem'].get('isPaidOnly', False)
                    ]
                    print(f"Premium filter: {before} → {len(generated_items)} problems")

                # Apply problem offset and count (for splitting across accounts)
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

                # Establish a stable results file path for resume support across runs/accounts
                active_results_file = model_state.get('results_file_in_progress')
                if not active_results_file:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs(Config.RESULTS_EVALUATIONS, exist_ok=True)
                    active_results_file = os.path.join(
                        Config.RESULTS_EVALUATIONS,
                        f"evaluation_results_{timestamp}.json"
                    )
                    model_state['results_file_in_progress'] = active_results_file
                    self._save_model_state(model_id, model_state)
                    print(f"Results file: {active_results_file}")
                else:
                    print(f"Resuming from results file: {active_results_file}")

                evaluator = LeetCodeEvaluator(
                    provider='qwen',
                    model_id=model_id,
                    experiment_name=self._safe_name(model_id)
                )
                results_file = evaluator.run_generated_evaluation(
                    generated_items,
                    stability_runs=self.stability_runs,
                    results_file=active_results_file,
                )
                if not results_file:
                    raise RuntimeError(f"Generated evaluation failed for model: {model_id}")

                # Only mark completed when ALL problems in the config have valid results.
                # This allows multi-round submission (e.g. 50 + 50 across two accounts)
                # without prematurely closing the model.
                coverage = self._results_coverage(results_file)
                total = len(self.problems)
                if coverage >= total:
                    report_file = self._archive_model_results(evaluator, results_file)
                    model_state.update({
                        'status': 'completed',
                        'completed_at': datetime.now().isoformat(),
                        'results_file': results_file,
                        'report_file': report_file,
                    })
                    self._save_model_state(model_id, model_state)
                    results.append({
                        'model_id': model_id,
                        'results_file': results_file,
                        'report_file': report_file,
                    })
                    print(f"✓ Model fully complete: {coverage}/{total} problems")
                else:
                    # Partial run finished — keep state open for next round
                    model_state.update({
                        'status': 'generation_completed',
                        'results_file_in_progress': results_file,
                    })
                    self._save_model_state(model_id, model_state)
                    results.append({
                        'model_id': model_id,
                        'results_file': results_file,
                        'report_file': None,
                    })
                    print(f"Partial run: {coverage}/{total} problems done. "
                          f"Re-run with --batch-problem-offset {coverage} to continue.")
            except RateLimitExhaustedException as exc:
                # Keep state resumable (generation_completed) so the next account can continue.
                # results_file_in_progress is already set, so progress is preserved.
                model_state.update({
                    'status': 'generation_completed',
                    'rate_limited_at': datetime.now().isoformat(),
                    'error': str(exc),
                })
                self._save_model_state(model_id, model_state)
                print(f"\n{'!'*60}")
                print(f"RATE LIMIT EXHAUSTED for model: {model_id}")
                print(f"Progress saved. Switch LeetCode account and re-run the same command.")
                print(f"{'!'*60}\n")
                raise  # propagate → process exits → Docker container stops
            except Exception as exc:
                if mode == 'resume':
                    model_state.update({
                        'status': 'failed',
                        'failed_at': datetime.now().isoformat(),
                        'error': str(exc),
                    })
                    self._save_model_state(model_id, model_state)
                print(f"✗ Model failed: {model_id}: {exc}")
        return results
