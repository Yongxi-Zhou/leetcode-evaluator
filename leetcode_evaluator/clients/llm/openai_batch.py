"""
OpenAI batch client built on the official OpenAI Batch File API.

Mirrors QwenBatchClient/AzureBatchClient — same lifecycle:
  upload → create → poll → download → normalize.
"""
import json
import os
import re
import time
from typing import Dict, Any, List, Tuple

from openai import OpenAI

from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient


class OpenAIBatchClient(LLMClient):
    """Client for OpenAI batch generation via the official Batch File API."""

    TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}

    def __init__(self, model_id: str = None, api_key: str = None):
        super().__init__(model_id=model_id or Config.OPENAI_MODEL_ID)
        self.client = OpenAI(
            api_key=api_key or Config.OPENAI_API_KEY,
            base_url="https://api.openai.com/v1",  # explicit to avoid OPENAI_BASE_URL override
            timeout=Config.LLM_REQUEST_TIMEOUT_S,
            max_retries=0,
        )

    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("OpenAIBatchClient only supports offline batch generation")

    def _sanitize_component(self, value: str) -> str:
        return re.sub(r'[^A-Za-z0-9._-]+', "_", str(value))

    def build_custom_id(self, problem: Dict[str, Any], prompt_type: str, trial_index: int) -> str:
        frontend_id = problem.get('frontend_question_id') or problem.get('problem_id') or problem.get('question_id')
        return (
            f"model={self._sanitize_component(self.model_id)}"
            f"|problem={self._sanitize_component(frontend_id)}"
            f"|slug={self._sanitize_component(problem['title_slug'])}"
            f"|prompt={self._sanitize_component(prompt_type)}"
            f"|trial={trial_index}"
        )

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
        # o-series reasoning models (o1, o3, o4-mini, etc.) do not support
        # temperature / top_p and require max_completion_tokens instead of max_tokens.
        is_o_series = re.match(r'^o\d', self.model_id) is not None
        max_tok = generation_params.get('max_tokens', Config.MODEL_MAX_TOKENS)
        body = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python programmer specializing in algorithmic problem solving.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        if is_o_series:
            body["max_completion_tokens"] = max_tok
        else:
            body["temperature"] = generation_params.get('temperature', Config.MODEL_TEMPERATURE)
            body["max_tokens"] = max_tok
            body["top_p"] = generation_params.get('top_p', Config.MODEL_TOP_P)
        request_entry = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
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
            "input_file_body": body,
        }
        return request_entry, manifest_entry

    def write_requests_jsonl(self, request_entries: List[Dict[str, Any]], output_path: str):
        with open(output_path, 'w') as f:
            for entry in request_entries:
                f.write(json.dumps(entry, ensure_ascii=True) + '\n')

    def upload_batch_file(self, input_path: str) -> Dict[str, Any]:
        print(f"Uploading batch input file for model {self.model_id}: {input_path}")
        with open(input_path, 'rb') as f:
            response = self.client.files.create(file=f, purpose="batch")
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return dict(response)

    def create_batch_job(self, input_file_id: str, metadata: Dict[str, Any], completion_window: str = None) -> Dict[str, Any]:
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window or "24h",
            metadata=metadata,
        )
        if hasattr(batch, "model_dump"):
            return batch.model_dump()
        return dict(batch)

    def retrieve_batch_job(self, batch_id: str) -> Dict[str, Any]:
        batch = self.client.batches.retrieve(batch_id)
        if hasattr(batch, "model_dump"):
            return batch.model_dump()
        return dict(batch)

    def poll_batch_job(self, batch_id: str) -> Dict[str, Any]:
        while True:
            batch = self.retrieve_batch_job(batch_id)
            status = str(batch.get('status', '')).lower()
            print(f"OpenAI batch status: model={self.model_id}, batch_id={batch_id}, status={status}")
            if status in self.TERMINAL_STATES:
                return batch
            time.sleep(Config.OPENAI_BATCH_POLL_INTERVAL_S)

    def download_file(self, file_id: str, output_path: str) -> str:
        content = self.client.files.content(file_id)
        if hasattr(content, "write_to_file"):
            content.write_to_file(output_path)
        else:
            payload = getattr(content, "text", None)
            if payload is None and hasattr(content, "read"):
                payload = content.read()
            if isinstance(payload, bytes):
                with open(output_path, 'wb') as f:
                    f.write(payload)
            else:
                with open(output_path, 'w') as f:
                    f.write(payload or "")
        return output_path

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

    def _extract_response_text(self, response_body: Dict[str, Any]) -> str:
        choices = response_body.get('choices') or []
        if not choices:
            return ""
        message = choices[0].get('message') or {}
        content = message.get('content')
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text']
            return "\n".join(parts)
        return ""

    def normalize_batch_outputs(
        self,
        output_records: List[Dict[str, Any]],
        error_records: List[Dict[str, Any]],
        manifest_entries: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}

        for record in error_records:
            custom_id = record.get('custom_id')
            if not custom_id:
                continue
            error_text = record.get('error')
            if isinstance(error_text, dict):
                error_text = json.dumps(error_text, ensure_ascii=True)
            normalized[custom_id] = {
                'status': 'Generation Failed',
                'error': error_text or 'Batch error',
                'gen_status': 'Error',
                'input_tokens': 0, 'output_tokens': 0,
                'latency_ms': 0.0, 'cost': 0.0, 'raw_response': None,
            }

        for record in output_records:
            custom_id = record.get('custom_id')
            if not custom_id:
                continue
            response = record.get('response') or {}
            response_body = response.get('body') or {}
            error = record.get('error') or response_body.get('error')
            if error:
                normalized[custom_id] = {
                    'status': 'Generation Failed',
                    'error': json.dumps(error, ensure_ascii=True) if isinstance(error, dict) else str(error),
                    'gen_status': 'Error',
                    'input_tokens': 0, 'output_tokens': 0,
                    'latency_ms': 0.0, 'cost': 0.0, 'raw_response': None,
                }
                continue

            text = self._extract_response_text(response_body)
            usage = response_body.get('usage') or {}
            input_tokens = int(usage.get('prompt_tokens', 0) or 0)
            output_tokens = int(usage.get('completion_tokens', 0) or 0)
            pricing = Config.MODEL_PRICING.get(self.model_id, Config.MODEL_PRICING['default'])
            cost = (input_tokens * pricing['input'] + output_tokens * pricing['output']) / 1000
            code = self._extract_code(text) if text else None

            if not text:
                normalized[custom_id] = {
                    'status': 'Generation Failed', 'error': 'Empty batch response',
                    'gen_status': 'Error', 'input_tokens': input_tokens,
                    'output_tokens': output_tokens, 'latency_ms': 0.0,
                    'cost': cost, 'raw_response': text,
                }
                continue

            if not code:
                normalized[custom_id] = {
                    'status': 'Extraction Failed', 'error': 'Failed to extract code from batch response',
                    'raw_response': text, 'gen_status': 'Success',
                    'input_tokens': input_tokens, 'output_tokens': output_tokens,
                    'latency_ms': 0.0, 'cost': cost,
                }
                continue

            if not self.validate_code_syntax(code):
                normalized[custom_id] = {
                    'status': 'Syntax Error', 'error': 'Invalid Python syntax',
                    'code': code, 'raw_response': text, 'gen_status': 'Success',
                    'input_tokens': input_tokens, 'output_tokens': output_tokens,
                    'latency_ms': 0.0, 'cost': cost,
                }
                continue

            normalized[custom_id] = {
                'status': 'Success', 'code': code, 'raw_response': text,
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
