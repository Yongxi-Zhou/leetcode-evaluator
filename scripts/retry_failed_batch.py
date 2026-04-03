#!/usr/bin/env python3
"""
Retry failed requests from DeepSeek-R1 batch jobs.

Reads the error files from the two original batches, extracts the failed
custom_ids, rebuilds their request entries from the original input.jsonl,
and submits two new retry batches (one per prompt type).

After the retry batches complete, merges their output into the original
batch job directory so the main runner can collect seamlessly.

Usage:
    # Submit retry batches
    python scripts/retry_failed_batch.py submit

    # Check status
    python scripts/retry_failed_batch.py status

    # Collect results and merge into original batch output
    python scripts/retry_failed_batch.py collect
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leetcode_evaluator.clients.llm.qwen_batch import QwenBatchClient
from leetcode_evaluator.core.config import Config

# --- Original batch info ---
ORIGINAL_RUN_ID = "qwen_batch_paper_qwen_batch_deepseek_f372dc2788"
MODEL_ID = "deepseek-r1"
BATCH_ROOT = os.path.join(Config.BATCH_JOBS_ROOT, ORIGINAL_RUN_ID)

ORIGINAL_BATCHES = {
    "detailed": {
        "job_id": "batch_c19a4022-06e3-41d5-b317-b5f9bc621188",
        "error_file_id": "file-batch_output-a5b76889a7334e6fa2e015ae",
        "input_jsonl": os.path.join(BATCH_ROOT, MODEL_ID, "detailed", "input.jsonl"),
    },
    "minimal": {
        "job_id": "batch_973b6c26-3fd3-4ce8-8016-66cb2cabb566",
        "error_file_id": "file-batch_output-2b5545b4dbc24297ba1c0b5d",
        "input_jsonl": os.path.join(BATCH_ROOT, MODEL_ID, "minimal", "input.jsonl"),
    },
}

RETRY_DIR = os.path.join(BATCH_ROOT, MODEL_ID, "_retry")
RETRY_STATE_FILE = os.path.join(RETRY_DIR, "retry_state.json")


def _load_state():
    if os.path.exists(RETRY_STATE_FILE):
        with open(RETRY_STATE_FILE) as f:
            return json.load(f)
    return {}


def _save_state(state):
    os.makedirs(RETRY_DIR, exist_ok=True)
    with open(RETRY_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _get_failed_custom_ids(client, error_file_id):
    """Download error file and return set of failed custom_ids."""
    content = client.client.files.content(error_file_id)
    failed = set()
    for line in content.text.strip().split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        cid = record.get("custom_id")
        if cid:
            failed.add(cid)
    return failed


def _filter_input_jsonl(input_jsonl_path, failed_ids):
    """Read original input.jsonl and return only entries matching failed_ids."""
    entries = []
    with open(input_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("custom_id") in failed_ids:
                entries.append(entry)
    return entries


def cmd_submit():
    client = QwenBatchClient(model_id=MODEL_ID)
    state = _load_state()
    os.makedirs(RETRY_DIR, exist_ok=True)

    for prompt_type, info in ORIGINAL_BATCHES.items():
        if state.get(prompt_type, {}).get("job_id"):
            print(f"[{prompt_type}] Retry batch already submitted: {state[prompt_type]['job_id']}")
            continue

        print(f"\n[{prompt_type}] Fetching failed custom_ids from error file...")
        failed_ids = _get_failed_custom_ids(client, info["error_file_id"])
        print(f"[{prompt_type}] {len(failed_ids)} failed requests found")

        if not failed_ids:
            print(f"[{prompt_type}] No failures, skipping")
            continue

        # Filter original input.jsonl to only failed entries
        retry_entries = _filter_input_jsonl(info["input_jsonl"], failed_ids)
        print(f"[{prompt_type}] Matched {len(retry_entries)} entries from input.jsonl")

        if len(retry_entries) != len(failed_ids):
            missing = failed_ids - {e["custom_id"] for e in retry_entries}
            print(f"[{prompt_type}] WARNING: {len(missing)} failed IDs not found in input.jsonl:")
            for m in sorted(missing):
                print(f"  - {m}")

        # Write retry input file
        retry_jsonl = os.path.join(RETRY_DIR, f"{prompt_type}_retry_input.jsonl")
        with open(retry_jsonl, "w") as f:
            for entry in retry_entries:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
        print(f"[{prompt_type}] Wrote {retry_jsonl}")

        # Upload and create batch
        uploaded = client.upload_batch_file(retry_jsonl)
        print(f"[{prompt_type}] Uploaded file: {uploaded['id']}")

        batch = client.create_batch_job(
            input_file_id=uploaded["id"],
            metadata={
                "run_id": ORIGINAL_RUN_ID,
                "model_id": MODEL_ID,
                "prompt_type": prompt_type,
                "retry": "true",
                "original_job_id": info["job_id"],
                "num_retries": str(len(retry_entries)),
            },
        )
        print(f"[{prompt_type}] Retry batch created: {batch['id']} (status: {batch.get('status')})")

        state[prompt_type] = {
            "job_id": batch["id"],
            "input_file_id": uploaded["id"],
            "num_retries": len(retry_entries),
            "failed_custom_ids": sorted(failed_ids),
            "status": batch.get("status"),
        }

    _save_state(state)
    print("\nRetry batches submitted. Run 'status' to check progress.")


def cmd_status():
    state = _load_state()
    if not state:
        print("No retry state found. Run 'submit' first.")
        return

    client = QwenBatchClient(model_id=MODEL_ID)
    for prompt_type in ["detailed", "minimal"]:
        info = state.get(prompt_type, {})
        if not info.get("job_id"):
            print(f"[{prompt_type}] No retry batch submitted")
            continue

        batch = client.retrieve_batch_job(info["job_id"])
        counts = batch.get("request_counts", {})
        print(
            f"[{prompt_type}] job={info['job_id']}  "
            f"status={batch.get('status')}  "
            f"completed={counts.get('completed', 0)}/{counts.get('total', 0)}  "
            f"failed={counts.get('failed', 0)}"
        )
        # Update state
        info["status"] = batch.get("status")
        info["output_file_id"] = batch.get("output_file_id")
        info["error_file_id"] = batch.get("error_file_id")

    _save_state(state)


def cmd_collect():
    """Download retry outputs and merge into original batch output directory."""
    state = _load_state()
    if not state:
        print("No retry state found. Run 'submit' first.")
        return

    client = QwenBatchClient(model_id=MODEL_ID)

    for prompt_type in ["detailed", "minimal"]:
        info = state.get(prompt_type, {})
        if not info.get("job_id"):
            print(f"[{prompt_type}] No retry batch, skipping")
            continue

        # Refresh status
        batch = client.retrieve_batch_job(info["job_id"])
        status = batch.get("status", "")
        print(f"[{prompt_type}] Retry batch status: {status}")

        if status != "completed":
            print(f"[{prompt_type}] Not completed yet, skipping. Current status: {status}")
            continue

        # Download retry output
        retry_output_file = os.path.join(RETRY_DIR, f"{prompt_type}_retry_output.jsonl")
        output_file_id = batch.get("output_file_id")
        if output_file_id and not os.path.exists(retry_output_file):
            client.download_file(output_file_id, retry_output_file)
            print(f"[{prompt_type}] Downloaded retry output -> {retry_output_file}")

        # Download retry errors (if any)
        retry_error_file = os.path.join(RETRY_DIR, f"{prompt_type}_retry_error.jsonl")
        error_file_id = batch.get("error_file_id")
        if error_file_id and not os.path.exists(retry_error_file):
            client.download_file(error_file_id, retry_error_file)
            print(f"[{prompt_type}] Downloaded retry errors -> {retry_error_file}")

        # Load retry results
        retry_records = client.load_jsonl_records(retry_output_file)
        retry_by_id = {r["custom_id"]: r for r in retry_records if r.get("custom_id")}
        print(f"[{prompt_type}] Retry output has {len(retry_by_id)} records")

        # Load original output (download first if needed)
        orig_dir = os.path.join(BATCH_ROOT, MODEL_ID, prompt_type)
        orig_output = os.path.join(orig_dir, "output.jsonl")
        orig_error = os.path.join(orig_dir, "error.jsonl")

        # Download original output if not already there
        orig_meta = json.load(open(os.path.join(orig_dir, "batch_job.json")))
        orig_batch = client.retrieve_batch_job(orig_meta["job_id"])
        if orig_batch.get("output_file_id") and not os.path.exists(orig_output):
            client.download_file(orig_batch["output_file_id"], orig_output)
            print(f"[{prompt_type}] Downloaded original output -> {orig_output}")
        if orig_batch.get("error_file_id") and not os.path.exists(orig_error):
            client.download_file(orig_batch["error_file_id"], orig_error)
            print(f"[{prompt_type}] Downloaded original errors -> {orig_error}")

        # Merge: start from original successful outputs
        orig_records = client.load_jsonl_records(orig_output)
        orig_by_id = {r["custom_id"]: r for r in orig_records if r.get("custom_id")}
        orig_error_records = client.load_jsonl_records(orig_error)

        merged = dict(orig_by_id)  # 470/478 successful records
        replaced = 0
        still_failed_ids = []

        # For every originally-failed custom_id, try to fill from retry
        for cid, record in retry_by_id.items():
            resp = record.get("response", {})
            body = resp.get("body", {})
            err = record.get("error") or body.get("error")
            if not err:
                merged[cid] = record  # retry succeeded -> add/replace
                replaced += 1
            else:
                still_failed_ids.append(cid)
                print(f"[{prompt_type}] Retry also failed for {cid}: {err}")

        print(f"[{prompt_type}] {replaced} failed entries replaced with successful retries, {len(still_failed_ids)} still failing")

        # Write merged output
        merged_output = os.path.join(orig_dir, "output.jsonl")
        # Backup original
        backup = os.path.join(orig_dir, "output_original.jsonl")
        if os.path.exists(orig_output) and not os.path.exists(backup):
            os.rename(orig_output, backup)
            print(f"[{prompt_type}] Backed up original -> {backup}")

        with open(merged_output, "w") as f:
            for record in merged.values():
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
        print(f"[{prompt_type}] Wrote merged output ({len(merged)} records) -> {merged_output}")

        # Write cleaned error file: keep original errors not retried + retry errors
        retry_error_records = client.load_jsonl_records(retry_error_file) if os.path.exists(retry_error_file) else []
        retry_error_by_id = {r.get("custom_id"): r for r in retry_error_records if r.get("custom_id")}

        still_failed = []
        for r in orig_error_records:
            cid = r.get("custom_id")
            if cid and cid in retry_by_id:
                # Was retried — only keep if retry also failed
                if cid in retry_error_by_id or cid in still_failed_ids:
                    still_failed.append(retry_error_by_id.get(cid, r))
            else:
                still_failed.append(r)  # Not retried, keep original error

        error_backup = os.path.join(orig_dir, "error_original.jsonl")
        if os.path.exists(orig_error) and not os.path.exists(error_backup):
            os.rename(orig_error, error_backup)
            print(f"[{prompt_type}] Backed up original errors -> {error_backup}")

        with open(orig_error, "w") as f:
            for record in still_failed:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
        print(f"[{prompt_type}] Wrote updated error file ({len(still_failed)} records)")

        # Remove normalized cache so runner re-processes
        normalized = os.path.join(orig_dir, "normalized.json")
        if os.path.exists(normalized):
            os.remove(normalized)
            print(f"[{prompt_type}] Removed cached normalized.json (will be regenerated)")

        # Update original batch_job.json metadata
        orig_meta["status"] = "completed"
        orig_meta["output_file_id"] = orig_batch.get("output_file_id")
        orig_meta["error_file_id"] = orig_batch.get("error_file_id")
        orig_meta["completed_at"] = orig_batch.get("completed_at")
        orig_meta["batch"] = orig_batch
        with open(os.path.join(orig_dir, "batch_job.json"), "w") as f:
            json.dump(orig_meta, f, indent=2)
        print(f"[{prompt_type}] Updated batch_job.json metadata")

    # Check retry errors
    print("\n=== Retry Summary ===")
    for prompt_type in ["detailed", "minimal"]:
        retry_error_file = os.path.join(RETRY_DIR, f"{prompt_type}_retry_error.jsonl")
        retry_errors = client.load_jsonl_records(retry_error_file) if os.path.exists(retry_error_file) else []
        info = state.get(prompt_type, {})
        print(
            f"[{prompt_type}] Retried {info.get('num_retries', '?')} requests, "
            f"still failing: {len(retry_errors)}"
        )

    print("\nDone. You can now run the main runner with mode=collect to proceed.")


def main():
    parser = argparse.ArgumentParser(description="Retry failed DeepSeek-R1 batch requests")
    parser.add_argument("command", choices=["submit", "status", "collect"],
                        help="submit=create retry batches, status=check progress, collect=merge results")
    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit()
    elif args.command == "status":
        cmd_status()
    elif args.command == "collect":
        cmd_collect()


if __name__ == "__main__":
    main()
