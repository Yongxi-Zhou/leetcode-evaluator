#!/usr/bin/env python3
"""
Re-normalize 'Syntax Error' entries in normalized_generations.json files.

The old _extract_code stopped at \\n\\n inside class bodies, so multi-method
classes were truncated to just the first line. The fixed version handles blank
lines correctly. This script re-applies the current extraction logic to every
entry that has status='Syntax Error', updating them in-place.
"""
import json
import os
import re
import sys
from glob import glob


def extract_code(response: str):
    """Current (fixed) _extract_code logic from base.py."""
    if not response:
        return None

    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<\|thinking\|>.*?<\|/thinking\|>', '', clean, flags=re.DOTALL | re.IGNORECASE)
    clean = clean.strip() or response

    code_block_pattern = r'```(?:python|python3)?\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, clean, re.DOTALL)
    if matches:
        return matches[0].strip()

    class_pattern = r'(class\s+\w+.*?)(?=\n(?:class|def)\s|\Z)'
    class_matches = re.findall(class_pattern, clean, re.DOTALL)
    if class_matches:
        return class_matches[0].strip()

    lines = clean.split('\n')
    code_indicators = ['def ', 'class ', 'import ', 'from ', 'return ', '    ']
    if any(any(line.strip().startswith(ind) for ind in code_indicators) for line in lines):
        return clean

    if 'class ' in clean:
        start_idx = clean.find('class ')
        return clean[start_idx:].strip()

    return None


def validate_syntax(code: str) -> bool:
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def fix_file(path: str) -> tuple[int, int, int]:
    """Returns (fixed, still_broken, skipped) counts."""
    with open(path) as f:
        data = json.load(f)

    fixed = still_broken = skipped = 0
    changed = False

    for custom_id, entry in data.items():
        if entry.get('status') != 'Syntax Error':
            skipped += 1
            continue

        raw = entry.get('raw_response')
        if not raw:
            still_broken += 1
            continue

        code = extract_code(raw)
        if code and validate_syntax(code):
            entry['status'] = 'Success'
            entry['code'] = code
            entry.pop('error', None)
            fixed += 1
            changed = True
        else:
            # Update code field to at least show the re-extracted version
            if code:
                entry['code'] = code
                entry['error'] = 'Invalid Python syntax (re-checked)'
                changed = True
            still_broken += 1

    if changed:
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return fixed, still_broken, skipped


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(root, 'output', 'batch_jobs', '**', 'normalized_generations.json')
    files = glob(pattern, recursive=True)

    if not files:
        print("No normalized_generations.json files found.")
        sys.exit(0)

    total_fixed = total_broken = 0
    for path in sorted(files):
        rel = os.path.relpath(path, root)
        fixed, broken, skipped = fix_file(path)
        if fixed or broken:
            print(f"{rel}: fixed={fixed}, still_broken={broken}, skipped={skipped}")
        total_fixed += fixed
        total_broken += broken

    print(f"\nTotal fixed: {total_fixed}, still broken: {total_broken}")


if __name__ == '__main__':
    main()
