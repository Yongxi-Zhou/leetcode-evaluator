"""
Test script to verify LeetCode API is working correctly
"""
from leetcode_evaluator.clients.leetcode import LeetCodeClient
from leetcode_evaluator.core.config import Config

def test_authentication():
    """Test if session cookies work"""
    print("Testing authentication...")
    client = LeetCodeClient()
    success = client.login()
    return success

def test_fetch_problems():
    """Test fetching problems"""
    print("\nTesting problem fetching...")
    client = LeetCodeClient()
    
    if not client.login():
        return False
    
    # Test fetching 3 easy problems
    problems = client.get_problem_list(limit=3, skip=0, difficulty="EASY")
    
    if problems:
        print(f"✓ Successfully fetched {len(problems)} problems")
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem['title']} ({problem['difficulty']})")
        return True
    else:
        print("✗ Failed to fetch problems")
        return False

def test_problem_details():
    """Test fetching problem details"""
    print("\nTesting problem details...")
    client = LeetCodeClient()
    
    if not client.login():
        return False
    
    # Try to get details for "two-sum"
    details = client.get_problem_details("two-sum")
    
    if details:
        print(f"✓ Successfully fetched problem: {details['title']}")
        print(f"  Question ID: {details['question_id']}")
        print(f"  Difficulty: {details['difficulty']}")
        print(f"  Topics: {', '.join(details['topics'][:3])}")
        print(f"  Has code template: {'Yes' if details['code_template'] else 'No'}")
        return True
    else:
        print("✗ Failed to fetch problem details")
        return False

def test_submission():
    """Test submitting a solution"""
    print("\nTesting solution submission...")
    client = LeetCodeClient()
    
    if not client.login():
        return False
    
    # Get problem details first
    details = client.get_problem_details("two-sum")
    
    if not details:
        print("✗ Failed to fetch problem for submission test")
        return False
    
    # Submit a simple (likely incorrect) solution
    test_code = """class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # Test submission - intentionally simple
        return [0, 1]
"""
    
    submission_id = client.submit_solution(
        title_slug=details['title_slug'],
        code=test_code,
        question_id=details['question_id']
    )
    
    if submission_id:
        print(f"✓ Successfully created submission: {submission_id}")
        print(f"  (Note: This is just a test - solution may not be correct)")
        return True
    else:
        print("✗ Failed to submit solution")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("LeetCode API Test")
    print("=" * 60)
    
    try:
        Config.validate()
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        exit(1)
    
    results = {
        'Authentication': test_authentication(),
        'Fetch Problems': test_fetch_problems(),
        'Problem Details': test_problem_details(),
        'Submission': test_submission()
    }
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("=" * 60)
    
    if all_passed:
        print("✓ All tests passed! System is ready to use.")
        exit(0)
    else:
        print("✗ Some tests failed. Check configuration and cookies.")
        exit(1)
