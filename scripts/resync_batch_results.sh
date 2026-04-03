#!/bin/bash
# Resync batch run results from another machine's leetcode_evaluator directory.
#
# Usage:
#   ./scripts/resync_batch_results.sh <source_path> [run_id]
#
# Examples:
#   # Sync a specific run from another copy of the repo
#   ./scripts/resync_batch_results.sh /Volumes/USB/leetcode_evaluator qwen_batch_paper_qwen_batch_deepseek_f372dc2788
#   ./scripts/resync_batch_results.sh ~/Desktop/leetcode_evaluator qwen_batch_paper_qwen_batch_deepseek_f372dc2788
#
#   # Sync ALL output
#   ./scripts/resync_batch_results.sh /Volumes/USB/leetcode_evaluator
#
# What gets synced (all paths relative to leetcode_evaluator/):
#
# 1. output/batch_jobs/<run_id>/        — batch job metadata, outputs, normalized, evaluation_state
# 2. output/<run_id>_<model>/           — per-model experiment dir (results JSON, JSONL, solutions.md)
# 3. output/experiments/                — experiment logs and detailed JSONL
# 4. output/aggregate/                  — cross-run summary JSONs and raw data
# 5. output/reports/                    — generated report markdowns and figures
# 6. output/results/                    — evaluation results, summaries, tables

set -euo pipefail

SOURCE_DIR="$1"
RUN_ID="${2:-}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

RSYNC_OPTS="-av --progress --update"

# Validate source
if [ ! -d "$SOURCE_DIR/output" ]; then
    echo "ERROR: $SOURCE_DIR/output does not exist."
    echo "Make sure the path points to the leetcode_evaluator directory."
    exit 1
fi

echo "Source: $SOURCE_DIR"
echo "Local:  $LOCAL_DIR"
echo ""

if [ -n "$RUN_ID" ]; then
    echo "=== Syncing run: $RUN_ID ==="
    echo ""

    # 1. Batch jobs dir (generation outputs, normalized, evaluation state)
    echo "--- [1/5] Batch jobs ---"
    if [ -d "$SOURCE_DIR/output/batch_jobs/$RUN_ID" ]; then
        rsync $RSYNC_OPTS \
            "$SOURCE_DIR/output/batch_jobs/$RUN_ID/" \
            "$LOCAL_DIR/output/batch_jobs/$RUN_ID/"
    else
        echo "  (not found, skipping)"
    fi

    # 2. Run output dir (output/<run_id>/) and per-model dirs (output/<run_id>_<model>/)
    echo ""
    echo "--- [2/5] Run & per-model output dirs ---"
    FOUND=0
    # The run's own output dir (results, reports, etc. nested under run_id)
    if [ -d "$SOURCE_DIR/output/$RUN_ID" ]; then
        echo "  Syncing $RUN_ID ..."
        rsync $RSYNC_OPTS "$SOURCE_DIR/output/$RUN_ID/" "$LOCAL_DIR/output/$RUN_ID/"
        FOUND=1
    fi
    # Per-model dirs (<run_id>_<model>)
    for d in "$SOURCE_DIR"/output/${RUN_ID}_*/; do
        [ -d "$d" ] || continue
        dirname=$(basename "$d")
        echo "  Syncing $dirname ..."
        rsync $RSYNC_OPTS "$d" "$LOCAL_DIR/output/$dirname/"
        FOUND=1
    done
    [ $FOUND -eq 0 ] && echo "  (none found)"

    # 3. Experiment logs
    echo ""
    echo "--- [3/5] Experiment logs ---"
    if [ -d "$SOURCE_DIR/output/experiments" ]; then
        rsync $RSYNC_OPTS \
            "$SOURCE_DIR/output/experiments/" \
            "$LOCAL_DIR/output/experiments/"
    else
        echo "  (not found, skipping)"
    fi

    # 4. Aggregate summaries and raw data
    echo ""
    echo "--- [4/5] Aggregate data ---"
    if [ -d "$SOURCE_DIR/output/aggregate" ]; then
        rsync $RSYNC_OPTS \
            "$SOURCE_DIR/output/aggregate/" \
            "$LOCAL_DIR/output/aggregate/"
    else
        echo "  (not found, skipping)"
    fi

    # 5. Reports and results
    echo ""
    echo "--- [5/5] Reports & results ---"
    for subdir in reports results; do
        if [ -d "$SOURCE_DIR/output/$subdir" ]; then
            rsync $RSYNC_OPTS \
                "$SOURCE_DIR/output/$subdir/" \
                "$LOCAL_DIR/output/$subdir/"
        else
            echo "  output/$subdir (not found, skipping)"
        fi
    done

else
    echo "=== Syncing ALL output ==="
    echo ""
    rsync $RSYNC_OPTS \
        "$SOURCE_DIR/output/" \
        "$LOCAL_DIR/output/"
fi

echo ""
echo "=== Resync complete ==="
