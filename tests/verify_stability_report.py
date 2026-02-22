import os
import json
import sys
from leetcode_evaluator.core.report_generator import ReportGenerator
from leetcode_evaluator.core.config import Config

def verify():
    # 1. Create mock results
    mock_results = [
        {
            "problem_id": "1",
            "title": "Two Sum",
            "difficulty": "Easy",
            "topics": ["Array", "Hash Table"],
            "with_prompt": [
                {"status": "Accepted", "timestamp": "2026-02-21T18:00:00"},
                {"status": "Accepted", "timestamp": "2026-02-21T18:01:00"}
            ],
            "without_prompt": [
                {"status": "Accepted", "timestamp": "2026-02-21T18:02:00"},
                {"status": "Wrong Answer", "timestamp": "2026-02-21T18:03:00"}
            ]
        }
    ]
    
    mock_file = "tests/mock_results_stability.json"
    os.makedirs("tests", exist_ok=True)
    with open(mock_file, 'w') as f:
        json.dump(mock_results, f)
    
    # 2. Run ReportGenerator
    report_file = "tests/mock_report_stability.md"
    generator = ReportGenerator(mock_file)
    actual_report_file = generator.generate_full_report(output_file=report_file)
    
    # 3. Check if Stability Analysis is in the report
    with open(actual_report_file, 'r') as f:
        content = f.read()
        
    if "## 3. Stability Analysis" in content:
        print("✓ Stability Analysis section found in report")
        # Check some metrics
        if "Total Problems | 1" in content:
            print("✓ Correct Total Problems")
        if "Total Runs | 4" in content:
            print("✓ Correct Total Runs")
        print("\nFull Report Content Preview:")
        print("-" * 20)
        print(content[:500] + "...")
        print("-" * 20)
        return True
    else:
        print("✗ Stability Analysis section NOT found in report")
        return False

if __name__ == "__main__":
    if verify():
        sys.exit(0)
    else:
        sys.exit(1)
