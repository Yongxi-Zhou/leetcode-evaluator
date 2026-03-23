"""
Google Vertex AI Batch Prediction Client for Gemini models.

Uses Vertex AI BatchPredictionJob with GCS for input/output.

Requires:
  pip install google-cloud-aiplatform google-cloud-storage

Environment:
  GCP_PROJECT               — GCP project ID
  GCP_LOCATION              — Region (default: us-central1)
  GEMINI_BATCH_GCS_BUCKET   — GCS bucket for batch input/output
"""
import hashlib
import json
import os
import re
import time
from typing import Dict, Any, List, Tuple

from google.cloud import aiplatform
from google.cloud import storage as gcs

from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient


class GeminiBatchClient(LLMClient):
    """Client for Gemini batch generation via Vertex AI BatchPredictionJob."""

    TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
    IN_PROGRESS_STATES = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_UPDATING"}

    def __init__(self, model_id: str = None, project: str = None,
                 location: str = None, gcs_bucket: str = None):
        super().__init__(model_id=model_id or Config.GEMINI_MODEL_ID)
        self.project = project or Config.GCP_PROJECT
        self.location = location or Config.GCP_LOCATION
        self.gcs_bucket = gcs_bucket or Config.GEMINI_BATCH_GCS_BUCKET

        if not self.project:
            raise ValueError("GCP_PROJECT must be set for Gemini batch inference")
        if not self.gcs_bucket:
            raise ValueError("GEMINI_BATCH_GCS_BUCKET must be set for Gemini batch inference")

        aiplatform.init(project=self.project, location=self.location)
        self.storage_client = gcs.Client(project=self.project)

    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("GeminiBatchClient only supports offline batch generation")

    def _sanitize_component(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))

    def _source_model(self) -> str:
        """Convert model_id to Vertex AI publisher model resource name."""
        model = self.model_id
        if model.startswith("publishers/"):
            return model
        return f"publishers/google/models/{model}"

    def build_custom_id(self, problem: Dict[str, Any], prompt_type: str, trial_index: int) -> str:
        frontend_id = (
            problem.get("frontend_question_id")
            or problem.get("problem_id")
            or problem.get("question_id")
        )
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
        system_msg = "You are an expert Python programmer specializing in algorithmic problem solving."

        # Vertex AI GenerateContent batch format
        request_body = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "systemInstruction": {
                "parts": [{"text": system_msg}]
            },
            "generationConfig": {
                "temperature": generation_params.get("temperature", Config.MODEL_TEMPERATURE),
                "maxOutputTokens": generation_params.get("max_tokens", Config.MODEL_MAX_TOKENS),
                "topP": generation_params.get("top_p", Config.MODEL_TOP_P),
            },
        }

        request_entry = {"request": request_body}
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
            # fingerprint for matching output to input when order is uncertain
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
        }
        return request_entry, manifest_entry

    def write_requests_jsonl(self, request_entries: List[Dict[str, Any]], output_path: str):
        with open(output_path, "w") as f:
            for entry in request_entries:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    # ── GCS operations ──────────────────────────────────────────

    def upload_to_gcs(self, local_path: str, gcs_prefix: str) -> str:
        """Upload local file to GCS and return the gs:// URI."""
        filename = os.path.basename(local_path)
        gcs_key = f"{gcs_prefix.rstrip('/')}/{filename}"
        bucket = self.storage_client.bucket(self.gcs_bucket)
        blob = bucket.blob(gcs_key)
        print(f"Uploading batch input to gs://{self.gcs_bucket}/{gcs_key}")
        blob.upload_from_filename(local_path)
        return f"gs://{self.gcs_bucket}/{gcs_key}"

    def download_output_from_gcs(self, output_gcs_prefix: str, local_dir: str) -> str:
        """Download batch output JSONL from GCS. Returns local file path."""
        prefix = output_gcs_prefix.replace(f"gs://{self.gcs_bucket}/", "").rstrip("/")
        bucket = self.storage_client.bucket(self.gcs_bucket)
        blobs = list(bucket.list_blobs(prefix=prefix))
        output_blobs = [
            b for b in blobs
            if b.name.endswith(".jsonl") or b.name.endswith(".jsonl.out")
        ]
        if not output_blobs:
            # Vertex AI may write to a subdirectory; try all files
            output_blobs = [b for b in blobs if not b.name.endswith("/")]

        if not output_blobs:
            raise RuntimeError(f"No output files found in gs://{self.gcs_bucket}/{prefix}")

        local_path = os.path.join(local_dir, "output.jsonl")
        with open(local_path, "wb") as out_f:
            for blob in sorted(output_blobs, key=lambda b: b.name):
                print(f"  Downloading gs://{self.gcs_bucket}/{blob.name}")
                out_f.write(blob.download_as_bytes())
        return local_path

    # ── Batch job lifecycle ─────────────────────────────────────

    def create_batch_job(
        self,
        input_gcs_uri: str,
        output_gcs_prefix: str,
        display_name: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit a Vertex AI BatchPredictionJob."""
        job = aiplatform.BatchPredictionJob.submit(
            source_model=self._source_model(),
            input_dataset=input_gcs_uri,
            output_uri_prefix=output_gcs_prefix,
            display_name=display_name,
        )
        return {
            "job_name": job.resource_name,
            "display_name": display_name,
            "status": job.state.name,
            "input_gcs_uri": input_gcs_uri,
            "output_gcs_prefix": output_gcs_prefix,
        }

    def retrieve_batch_job(self, job_name: str) -> Dict[str, Any]:
        """Get current status of a batch prediction job."""
        job = aiplatform.BatchPredictionJob(batch_prediction_job_name=job_name)
        output_info = {}
        if job.output_info:
            output_info = {
                "gcs_output_directory": getattr(job.output_info, "gcs_output_directory", ""),
                "bigquery_output_dataset": getattr(job.output_info, "bigquery_output_dataset", ""),
            }
        return {
            "job_name": job.resource_name,
            "display_name": job.display_name,
            "status": job.state.name,
            "create_time": str(job.create_time) if job.create_time else "",
            "end_time": str(job.end_time) if job.end_time else "",
            "output_info": output_info,
            "error": str(job.error) if job.error else "",
        }

    def poll_batch_job(self, job_name: str) -> Dict[str, Any]:
        """Poll until the job reaches a terminal state."""
        while True:
            info = self.retrieve_batch_job(job_name)
            status = info.get("status", "")
            print(f"Gemini batch status: model={self.model_id}, job={job_name}, status={status}")
            if status in self.TERMINAL_STATES:
                return info
            time.sleep(Config.GEMINI_BATCH_POLL_INTERVAL_S)

    # ── Output parsing ──────────────────────────────────────────

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

    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract generated text from Vertex AI GenerateContent response."""
        candidates = response.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "\n".join(texts)

    def _extract_usage_from_response(self, response: Dict[str, Any]) -> Tuple[int, int]:
        """Return (input_tokens, output_tokens) from usageMetadata."""
        usage = response.get("usageMetadata") or {}
        return (
            int(usage.get("promptTokenCount", 0) or 0),
            int(usage.get("candidatesTokenCount", 0) or 0),
        )

    def _prompt_hash_from_request(self, request: Dict[str, Any]) -> str:
        """Extract prompt text from echoed request and compute hash for matching."""
        contents = request.get("contents") or []
        for item in contents:
            if item.get("role") == "user":
                parts = item.get("parts") or []
                for part in parts:
                    text = part.get("text", "")
                    if text:
                        return hashlib.md5(text.encode()).hexdigest()
        return ""

    def normalize_batch_outputs(
        self,
        output_records: List[Dict[str, Any]],
        error_records: List[Dict[str, Any]],
        manifest_entries: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Normalize Vertex AI batch outputs to the standard structure.

        Vertex AI output preserves line order and echoes the request.
        We map by line index (primary) with prompt-hash fallback.
        """
        normalized: Dict[str, Dict[str, Any]] = {}

        # Build prompt_hash → custom_id lookup for fallback matching
        hash_to_custom_id = {}
        for entry in manifest_entries:
            h = entry.get("prompt_hash", "")
            if h:
                hash_to_custom_id[h] = entry["custom_id"]

        for idx, record in enumerate(output_records):
            # Determine custom_id: by line index or by prompt hash
            if idx < len(manifest_entries):
                custom_id = manifest_entries[idx]["custom_id"]
            else:
                # fallback: match by prompt hash
                request = record.get("request") or {}
                h = self._prompt_hash_from_request(request)
                custom_id = hash_to_custom_id.get(h)
                if not custom_id:
                    continue

            status_text = record.get("status", "")
            response = record.get("response")

            # Non-empty status field indicates an error
            if status_text or response is None:
                normalized[custom_id] = {
                    "status": "Generation Failed",
                    "error": status_text or "No response from Vertex AI",
                    "gen_status": "Error",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0.0,
                    "cost": 0.0,
                    "raw_response": None,
                }
                continue

            text = self._extract_text_from_response(response)
            input_tokens, output_tokens = self._extract_usage_from_response(response)
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

        # Mark any manifest entries without results
        for entry in manifest_entries:
            if entry["custom_id"] not in normalized:
                normalized[entry["custom_id"]] = {
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
