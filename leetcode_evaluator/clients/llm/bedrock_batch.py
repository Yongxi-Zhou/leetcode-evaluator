"""
AWS Bedrock Batch Client using CreateModelInvocationJob (S3-based batch inference).
"""
import json
import os
import re
import time
from typing import Dict, Any, List, Tuple

import boto3
from botocore.config import Config as BotoConfig

from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient


class BedrockBatchClient(LLMClient):
    """Client for AWS Bedrock batch generation via CreateModelInvocationJob."""

    TERMINAL_STATES = {"Completed", "Failed", "Stopped", "Expired"}
    IN_PROGRESS_STATES = {"Submitted", "InProgress", "Stopping"}

    def __init__(self, model_id: str = None, region: str = None, profile: str = None):
        super().__init__(model_id=model_id or Config.BEDROCK_MODEL_ID)
        self.region = region or Config.AWS_REGION
        self.profile = profile or Config.AWS_PROFILE
        self.s3_bucket = Config.BEDROCK_BATCH_S3_BUCKET
        self.role_arn = Config.BEDROCK_BATCH_ROLE_ARN

        boto_config = BotoConfig(
            region_name=self.region,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        if self.profile:
            session = boto3.Session(profile_name=self.profile, region_name=self.region)
            self.bedrock_client = session.client("bedrock", config=boto_config)
            self.s3_client = session.client("s3")
        else:
            self.bedrock_client = boto3.client("bedrock", region_name=self.region, config=boto_config)
            self.s3_client = boto3.client("s3", region_name=self.region)

    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("BedrockBatchClient only supports offline batch generation")

    def _sanitize_component(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))

    def _build_model_input(self, prompt: str, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Build the modelInput payload depending on model family."""
        temperature = generation_params.get("temperature", Config.MODEL_TEMPERATURE)
        top_p = generation_params.get("top_p", Config.MODEL_TOP_P)
        max_tokens = generation_params.get("max_tokens", Config.MODEL_MAX_TOKENS)
        system_msg = "You are an expert Python programmer specializing in algorithmic problem solving."

        if "anthropic.claude" in self.model_id:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system_msg,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
            }
        elif "amazon.nova" in self.model_id:
            return {
                "schemaVersion": "messages-v1",
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "system": [{"text": system_msg}],
                "inferenceConfig": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
        elif "meta.llama" in self.model_id:
            full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            return {
                "prompt": full_prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        elif "deepseek" in self.model_id.lower():
            return {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        else:
            # Generic fallback
            return {
                "prompt": f"{system_msg}\n\n{prompt}",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

    def build_custom_id(self, problem: Dict[str, Any], prompt_type: str, trial_index: int) -> str:
        frontend_id = problem.get("frontend_question_id") or problem.get("problem_id") or problem.get("question_id")
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
        use_detailed_prompt = prompt_type == "detailed"
        prompt = self._prepare_prompt(problem, use_detailed_prompt)
        custom_id = self.build_custom_id(problem, prompt_type, trial_index)
        model_input = self._build_model_input(prompt, generation_params)

        # Bedrock batch JSONL format: recordId + modelInput
        request_entry = {
            "recordId": custom_id,
            "modelInput": model_input,
        }
        manifest_entry = {
            "custom_id": custom_id,
            "model_id": self.model_id,
            "problem_id": problem.get("frontend_question_id") or problem.get("question_id"),
            "backend_question_id": problem.get("question_id"),
            "title": problem["title"],
            "title_slug": problem["title_slug"],
            "difficulty": problem.get("difficulty"),
            "topics": problem.get("topics", []),
            "prompt_type": prompt_type,
            "trial_index": trial_index,
            "input_model_input": model_input,
        }
        return request_entry, manifest_entry

    def write_requests_jsonl(self, request_entries: List[Dict[str, Any]], output_path: str):
        with open(output_path, "w") as f:
            for entry in request_entries:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def upload_input_to_s3(self, input_path: str, s3_prefix: str) -> str:
        """Upload local JSONL to S3 and return the S3 URI."""
        filename = os.path.basename(input_path)
        s3_key = f"{s3_prefix.rstrip('/')}/{filename}"
        print(f"Uploading batch input to s3://{self.s3_bucket}/{s3_key}")
        self.s3_client.upload_file(input_path, self.s3_bucket, s3_key)
        return f"s3://{self.s3_bucket}/{s3_key}"

    def create_batch_job(
        self,
        input_s3_uri: str,
        output_s3_prefix: str,
        job_name: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = self.bedrock_client.create_model_invocation_job(
            jobName=job_name,
            roleArn=self.role_arn,
            modelId=self.model_id,
            inputDataConfig={"s3InputDataConfig": {"s3Uri": input_s3_uri}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_s3_prefix}},
        )
        return {
            "job_arn": response["jobArn"],
            "job_name": job_name,
            "status": "Submitted",
            "input_s3_uri": input_s3_uri,
            "output_s3_prefix": output_s3_prefix,
        }

    def retrieve_batch_job(self, job_arn: str) -> Dict[str, Any]:
        response = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
        return {
            "job_arn": response.get("jobArn"),
            "job_name": response.get("jobName"),
            "status": response.get("status"),
            "output_data_config": response.get("outputDataConfig", {}),
            "submit_time": str(response.get("submitTime", "")),
            "end_time": str(response.get("endTime", "")),
            "message": response.get("message", ""),
        }

    def poll_batch_job(self, job_arn: str) -> Dict[str, Any]:
        while True:
            info = self.retrieve_batch_job(job_arn)
            status = info.get("status", "")
            print(f"Bedrock batch status: model={self.model_id}, job_arn={job_arn}, status={status}")
            if status in self.TERMINAL_STATES:
                return info
            time.sleep(Config.BEDROCK_BATCH_POLL_INTERVAL_S)

    def download_output_from_s3(self, output_s3_prefix: str, job_name: str, local_dir: str) -> str:
        """Download output JSONL from S3 and return the local path."""
        # Bedrock writes output to: {output_s3_prefix}/{job_name}/{model_id}.jsonl.out
        # or it may vary – we list objects under the prefix to find the output
        prefix = output_s3_prefix.replace(f"s3://{self.s3_bucket}/", "").rstrip("/")
        paginator = self.s3_client.get_paginator("list_objects_v2")
        output_keys = []
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".jsonl.out") or key.endswith(".out"):
                    output_keys.append(key)

        if not output_keys:
            raise RuntimeError(
                f"No output files found in s3://{self.s3_bucket}/{prefix}"
            )

        local_path = os.path.join(local_dir, "output.jsonl")
        # Concatenate all output chunks into one file
        with open(local_path, "wb") as out_f:
            for key in sorted(output_keys):
                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
                out_f.write(response["Body"].read())
        return local_path

    def load_jsonl_records(self, path: str) -> List[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return []
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _extract_text_from_model_output(self, model_output: Dict[str, Any]) -> str:
        """Extract generated text from Bedrock model output based on model family."""
        if "anthropic.claude" in self.model_id:
            content = model_output.get("content") or []
            parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            return "\n".join(parts)
        elif "amazon.nova" in self.model_id:
            output = model_output.get("output") or {}
            message = output.get("message") or {}
            content = message.get("content") or []
            parts = [c.get("text", "") for c in content if isinstance(c, dict)]
            return "\n".join(parts)
        elif "meta.llama" in self.model_id:
            return model_output.get("generation", "")
        elif "deepseek" in self.model_id.lower():
            choices = model_output.get("choices") or []
            if choices:
                return (choices[0].get("message") or {}).get("content", "")
            return ""
        else:
            return (
                model_output.get("completion")
                or model_output.get("generated_text")
                or model_output.get("text", "")
            )

    def _extract_usage_from_model_output(self, model_output: Dict[str, Any]) -> Tuple[int, int]:
        """Return (input_tokens, output_tokens)."""
        usage = model_output.get("usage") or {}
        if "anthropic.claude" in self.model_id:
            return (
                int(usage.get("input_tokens", 0) or 0),
                int(usage.get("output_tokens", 0) or 0),
            )
        elif "amazon.nova" in self.model_id:
            nova_usage = model_output.get("usage") or {}
            return (
                int(nova_usage.get("inputTokens", 0) or 0),
                int(nova_usage.get("outputTokens", 0) or 0),
            )
        elif "meta.llama" in self.model_id:
            return (
                int(model_output.get("prompt_token_count", 0) or 0),
                int(model_output.get("generation_token_count", 0) or 0),
            )
        else:
            return (
                int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0),
            )

    def normalize_batch_outputs(
        self,
        output_records: List[Dict[str, Any]],
        error_records: List[Dict[str, Any]],
        manifest_entries: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Normalize Bedrock batch outputs to the same structure as QwenBatchClient."""
        manifest_by_id = {entry["custom_id"]: entry for entry in manifest_entries}
        normalized: Dict[str, Dict[str, Any]] = {}

        for record in error_records:
            custom_id = record.get("custom_id") or record.get("recordId")
            if not custom_id:
                continue
            error_text = record.get("error")
            if isinstance(error_text, dict):
                error_text = json.dumps(error_text, ensure_ascii=True)
            normalized[custom_id] = {
                "status": "Generation Failed",
                "error": error_text or "Batch error",
                "gen_status": "Error",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
                "cost": 0.0,
                "raw_response": None,
            }

        for record in output_records:
            # Bedrock output line: {"recordId": "...", "modelOutput": {...}}
            custom_id = record.get("recordId") or record.get("custom_id")
            if not custom_id:
                continue

            error = record.get("error")
            if error:
                normalized[custom_id] = {
                    "status": "Generation Failed",
                    "error": json.dumps(error, ensure_ascii=True) if isinstance(error, dict) else str(error),
                    "gen_status": "Error",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0.0,
                    "cost": 0.0,
                    "raw_response": None,
                }
                continue

            model_output = record.get("modelOutput") or {}
            text = self._extract_text_from_model_output(model_output)
            input_tokens, output_tokens = self._extract_usage_from_model_output(model_output)

            pricing = Config.MODEL_PRICING.get(self.model_id, Config.MODEL_PRICING["default"])
            cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
            code = self._extract_code(text) if text else None

            if not text:
                normalized[custom_id] = {
                    "status": "Generation Failed",
                    "error": "Empty batch response",
                    "gen_status": "Error",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": 0.0,
                    "cost": cost,
                    "raw_response": text,
                }
                continue

            if not code:
                normalized[custom_id] = {
                    "status": "Extraction Failed",
                    "error": "Failed to extract code from batch response",
                    "raw_response": text,
                    "gen_status": "Success",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": 0.0,
                    "cost": cost,
                }
                continue

            if not self.validate_code_syntax(code):
                normalized[custom_id] = {
                    "status": "Syntax Error",
                    "error": "Invalid Python syntax",
                    "code": code,
                    "raw_response": text,
                    "gen_status": "Success",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": 0.0,
                    "cost": cost,
                }
                continue

            normalized[custom_id] = {
                "status": "Success",
                "code": code,
                "raw_response": text,
                "gen_status": "Success",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": 0.0,
                "cost": cost,
            }

        for custom_id in manifest_by_id:
            if custom_id not in normalized:
                normalized[custom_id] = {
                    "status": "Generation Failed",
                    "error": "Missing batch result",
                    "gen_status": "Error",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0.0,
                    "cost": 0.0,
                    "raw_response": None,
                }

        return normalized
