"""
Anthropic official batch client built on the Anthropic Message Batches API.

Mirrors OpenAIBatchClient — same lifecycle:
  build → upload → create → poll → download → normalize.
"""
import hashlib
import json
import os
import re
import time
from typing import Dict, Any, List, Tuple

import anthropic

from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient


class AnthropicBatchClient(LLMClient):
    """Client for Anthropic batch generation via the official Message Batches API."""

    TERMINAL_STATES = {"ended", "errored", "canceled", "expired"}

    _CODE_TOOL = {
        "name": "submit_solution",
        "description": "Submit the Python solution code for the LeetCode problem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python solution code, ready to run.",
                }
            },
            "required": ["code"],
        },
    }

    def __init__(self, model_id: str = None, api_key: str = None):
        super().__init__(model_id=model_id or Config.ANTHROPIC_MODEL_ID)
        self.client = anthropic.Anthropic(
            api_key=api_key or Config.ANTHROPIC_API_KEY,
        )

    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("AnthropicBatchClient only supports offline batch generation")

    def _sanitize_component(self, value: str) -> str:
        return re.sub(r'[^A-Za-z0-9._-]+', "_", str(value))

    def build_custom_id(self, problem: Dict[str, Any], prompt_type: str, trial_index: int) -> str:
        frontend_id = problem.get('frontend_question_id') or problem.get('problem_id') or problem.get('question_id')
        full = (
            f"model={self._sanitize_component(self.model_id)}"
            f"|problem={self._sanitize_component(frontend_id)}"
            f"|slug={self._sanitize_component(problem['title_slug'])}"
            f"|prompt={self._sanitize_component(prompt_type)}"
            f"|trial={trial_index}"
        )
        # Anthropic custom_id limit is 64 chars — use sha1 hash
        return hashlib.sha1(full.encode()).hexdigest()

    def build_request_entry(
        self,
        problem: Dict[str, Any],
        prompt_type: str,
        trial_index: int,
        generation_params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        use_detailed_prompt = prompt_type == 'detailed'
        prompt = self._prepare_prompt(problem, use_detailed_prompt)
        custom_id = self.build_custom_id(problem, prompt_type, trial_index)
        params = {
            "model": self.model_id,
            "max_tokens": generation_params.get('max_tokens', Config.MODEL_MAX_TOKENS),
            "temperature": generation_params.get('temperature', Config.MODEL_TEMPERATURE),
            "system": "You are an expert Python programmer specializing in algorithmic problem solving.",
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "tools": [self._CODE_TOOL],
            "tool_choice": {"type": "tool", "name": "submit_solution"},
        }
        request_entry = {
            "custom_id": custom_id,
            "params": params,
        }
        manifest_entry = {
            "custom_id": custom_id,
            "model_id": self.model_id,
            "problem_id": problem.get('frontend_question_id') or problem.get('question_id'),
            "backend_question_id": problem.get('question_id'),
            "title": problem['title'],
            "title_slug": problem['title_slug'],
            "difficulty": problem.get('difficulty'),
            "topics": problem.get('topics', []),
            "prompt_type": prompt_type,
            "trial_index": trial_index,
            "input_file_body": params,
        }
        return request_entry, manifest_entry

    def write_requests_jsonl(self, request_entries: List[Dict[str, Any]], output_path: str):
        with open(output_path, 'w') as f:
            for entry in request_entries:
                f.write(json.dumps(entry, ensure_ascii=True) + '\n')

    def _batch_to_dict(self, batch) -> Dict[str, Any]:
        if hasattr(batch, "model_dump"):
            return json.loads(batch.model_dump_json())
        return json.loads(json.dumps(dict(batch), default=str))

    def create_batch_job(self, request_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a batch job directly from request entries (no file upload needed)."""
        requests = [
            {"custom_id": entry["custom_id"], "params": entry["params"]}
            for entry in request_entries
        ]
        batch = self.client.messages.batches.create(requests=requests)
        return self._batch_to_dict(batch)

    def retrieve_batch_job(self, batch_id: str) -> Dict[str, Any]:
        batch = self.client.messages.batches.retrieve(batch_id)
        return self._batch_to_dict(batch)

    def poll_batch_job(self, batch_id: str) -> Dict[str, Any]:
        while True:
            batch = self.retrieve_batch_job(batch_id)
            status = str(batch.get('processing_status', '')).lower()
            counts = batch.get('request_counts', {})
            print(
                f"Anthropic batch status: model={self.model_id}, batch_id={batch_id}, "
                f"status={status}, processing={counts.get('processing', 0)}, "
                f"succeeded={counts.get('succeeded', 0)}, errored={counts.get('errored', 0)}"
            )
            if status in self.TERMINAL_STATES:
                return batch
            time.sleep(Config.ANTHROPIC_BATCH_POLL_INTERVAL_S)

    def download_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Download batch results and return as list of records."""
        records = []
        for result in self.client.messages.batches.results(batch_id):
            r = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            records.append(r)
        return records

    def save_results_jsonl(self, records: List[Dict[str, Any]], output_path: str):
        with open(output_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True, default=str) + '\n')

    def load_jsonl_records(self, path: str) -> List[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return []
        records = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _extract_text_from_result(self, result: Dict[str, Any]) -> str:
        """Extract code from tool_use block, fallback to text block."""
        result_obj = result.get('result') or {}
        if result_obj.get('type') == 'errored':
            return ""
        message = result_obj.get('message') or {}
        content = message.get('content') or []
        # Prefer tool_use submit_solution block (structured output)
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_use' and block.get('name') == 'submit_solution':
                code = (block.get('input') or {}).get('code', '')
                return code
        # Fallback to plain text block
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                return block.get('text', '')
        return ""

    def _extract_usage_from_result(self, result: Dict[str, Any]) -> Dict[str, int]:
        result_obj = result.get('result') or {}
        message = result_obj.get('message') or {}
        usage = message.get('usage') or {}
        return {
            'input_tokens': int(usage.get('input_tokens', 0) or 0),
            'output_tokens': int(usage.get('output_tokens', 0) or 0),
        }

    def normalize_batch_outputs(
        self,
        output_records: List[Dict[str, Any]],
        manifest_entries: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}

        for record in output_records:
            custom_id = record.get('custom_id')
            if not custom_id:
                continue

            result_obj = record.get('result') or {}
            result_type = result_obj.get('type', '')

            if result_type == 'errored':
                error_obj = result_obj.get('error') or {}
                error_text = json.dumps(error_obj) if isinstance(error_obj, dict) else str(error_obj)
                normalized[custom_id] = {
                    'status': 'Generation Failed', 'error': error_text,
                    'gen_status': 'Error', 'input_tokens': 0, 'output_tokens': 0,
                    'latency_ms': 0.0, 'cost': 0.0, 'raw_response': None,
                }
                continue

            # tool_use returns code directly; _extract_text_from_result returns the code string
            code = self._extract_text_from_result(record)
            usage = self._extract_usage_from_result(record)
            input_tokens = usage['input_tokens']
            output_tokens = usage['output_tokens']
            pricing = Config.MODEL_PRICING.get(self.model_id, Config.MODEL_PRICING['default'])
            cost = (input_tokens * pricing['input'] + output_tokens * pricing['output']) / 1000

            if not code:
                normalized[custom_id] = {
                    'status': 'Generation Failed', 'error': 'Empty batch response',
                    'gen_status': 'Error', 'input_tokens': input_tokens,
                    'output_tokens': output_tokens, 'latency_ms': 0.0,
                    'cost': cost, 'raw_response': None,
                }
                continue

            if not self.validate_code_syntax(code):
                normalized[custom_id] = {
                    'status': 'Syntax Error', 'error': 'Invalid Python syntax',
                    'code': code, 'raw_response': code, 'gen_status': 'Success',
                    'input_tokens': input_tokens, 'output_tokens': output_tokens,
                    'latency_ms': 0.0, 'cost': cost,
                }
                continue

            normalized[custom_id] = {
                'status': 'Success', 'code': code, 'raw_response': code,
                'gen_status': 'Success', 'input_tokens': input_tokens,
                'output_tokens': output_tokens, 'latency_ms': 0.0, 'cost': cost,
            }

        for entry in manifest_entries:
            cid = entry['custom_id']
            if cid not in normalized:
                normalized[cid] = {
                    'status': 'Generation Failed', 'error': 'Missing from batch output',
                    'gen_status': 'Error', 'input_tokens': 0, 'output_tokens': 0,
                    'latency_ms': 0.0, 'cost': 0.0, 'raw_response': None,
                }

        return normalized
