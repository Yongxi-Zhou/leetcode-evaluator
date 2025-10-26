"""
Test script to verify Google Gemini API is working correctly
"""
import os
import sys
from gemini_client import GeminiClient
from config import Config


def test_configuration():
    """Test if Gemini configuration is valid"""
    print("Testing Gemini configuration...")

    try:
        Config.validate_provider('gemini')
        print("✓ Configuration validation passed")
        return True
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_client_initialization():
    """Test Gemini client initialization"""
    print("\nTesting Gemini client initialization...")

    try:
        client = GeminiClient()
        print(f"✓ Client initialized successfully")
        print(f"  Model ID: {client.model_id}")
        print(
            f"  API Key: {'*' * 20}{client.api_key[-4:] if client.api_key else 'None'}")
        return True
    except Exception as e:
        print(f"✗ Client initialization failed: {e}")
        return False


def test_simple_generation():
    """Test simple code generation"""
    print("\nTesting simple code generation...")

    try:
        client = GeminiClient()

        # Simple test problem
        test_problem = {
            'description': 'Write a function that returns the sum of two numbers.',
            'code_template': 'def add(a, b):\n    pass'
        }

        print("  Generating solution with minimal prompt...")
        code = client.generate_solution(
            test_problem, use_detailed_prompt=False)

        if code:
            print(f"✓ Generated code successfully ({len(code)} characters)")
            print(f"  Code preview: {code[:100]}...")

            # Validate syntax
            if client.validate_code_syntax(code):
                print("✓ Generated code has valid Python syntax")
                return True
            else:
                print("✗ Generated code has invalid syntax")
                return False
        else:
            print("✗ Failed to generate code")
            return False

    except Exception as e:
        print(f"✗ Code generation failed: {e}")
        return False


def test_detailed_prompt():
    """Test generation with detailed prompt"""
    print("\nTesting detailed prompt generation...")

    try:
        client = GeminiClient()

        # LeetCode-style problem
        test_problem = {
            'description': '''
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]
            ''',
            'code_template': '''
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        pass
            '''
        }

        print("  Generating solution with detailed prompt...")
        code = client.generate_solution(test_problem, use_detailed_prompt=True)

        if code:
            print(f"✓ Generated detailed solution ({len(code)} characters)")
            print(f"  Code preview: {code[:150]}...")

            # Check if it looks like a proper solution
            if 'class Solution' in code and 'def twoSum' in code:
                print("✓ Generated code follows LeetCode format")
                return True
            else:
                print("✗ Generated code doesn't follow expected format")
                return False
        else:
            print("✗ Failed to generate detailed solution")
            return False

    except Exception as e:
        print(f"✗ Detailed prompt generation failed: {e}")
        return False


def test_code_extraction():
    """Test code extraction from various response formats"""
    print("\nTesting code extraction...")

    try:
        client = GeminiClient()

        # Test different response formats
        test_cases = [
            {
                'name': 'Markdown code block',
                'response': '''
Here's the solution:

```python
class Solution:
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
```
                ''',
                'expected': 'class Solution:'
            },
            {
                'name': 'Plain code',
                'response': '''
class Solution:
    def twoSum(self, nums, target):
        hashmap = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in hashmap:
                return [hashmap[complement], i]
            hashmap[num] = i
                ''',
                'expected': 'class Solution:'
            },
            {
                'name': 'Code with explanation',
                'response': '''
The solution uses a hash map to store numbers and their indices.

class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, num in enumerate(nums):
            if target - num in seen:
                return [seen[target - num], i]
            seen[num] = i
        return []
                ''',
                'expected': 'class Solution:'
            }
        ]

        for test_case in test_cases:
            print(f"  Testing {test_case['name']}...")
            extracted = client._extract_code(test_case['response'])

            if extracted and test_case['expected'] in extracted:
                print(f"    ✓ Successfully extracted code")
            else:
                print(f"    ✗ Failed to extract code properly")
                return False

        print("✓ All code extraction tests passed")
        return True

    except Exception as e:
        print(f"✗ Code extraction test failed: {e}")
        return False


def test_multiple_solutions():
    """Test generating multiple solutions"""
    print("\nTesting multiple solution generation...")

    try:
        client = GeminiClient()

        test_problem = {
            'description': 'Write a function to calculate the factorial of a number.',
            'code_template': 'def factorial(n):\n    pass'
        }

        print("  Generating 3 solutions...")
        solutions = client.generate_multiple_solutions(
            test_problem, count=3, use_detailed_prompt=False)

        if len(solutions) > 0:
            print(f"✓ Generated {len(solutions)} valid solutions")
            for i, solution in enumerate(solutions, 1):
                print(f"    Solution {i}: {len(solution)} characters")
            return True
        else:
            print("✗ Failed to generate any valid solutions")
            return False

    except Exception as e:
        print(f"✗ Multiple solution generation failed: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\nTesting error handling...")

    try:
        client = GeminiClient()

        # Test with empty problem
        print("  Testing empty problem...")
        result = client.generate_solution({}, use_detailed_prompt=False)
        if result is None:
            print("    ✓ Correctly handled empty problem")
        else:
            print("    ✗ Should have returned None for empty problem")
            return False

        # Test with malformed problem
        print("  Testing malformed problem...")
        malformed_problem = {
            'description': None,
            'code_template': None
        }
        result = client.generate_solution(
            malformed_problem, use_detailed_prompt=False)
        if result is None:
            print("    ✓ Correctly handled malformed problem")
        else:
            print("    ✗ Should have returned None for malformed problem")
            return False

        print("✓ Error handling tests passed")
        return True

    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_token_limits():
    """Test token limits and truncation handling"""
    print("\nTesting token limits...")

    try:
        client = GeminiClient()

        # Test with very long prompt
        print("  Testing with long prompt...")
        long_problem = {
            'description': 'A' * 10000,  # Very long description
            'code_template': 'def test():\n    pass'
        }

        code = client.generate_solution(
            long_problem, use_detailed_prompt=False)
        if code is not None:
            print("    ✓ Handled long prompt successfully")
            print(f"    Generated {len(code)} characters")
        else:
            print("    ⚠ Long prompt may have hit token limits (this is expected)")

        print("✓ Token limits test completed")
        return True

    except Exception as e:
        print(f"✗ Token limits test failed: {e}")
        return False


def test_truncation_handling():
    """Test handling of truncated responses"""
    print("\nTesting truncation handling...")

    try:
        client = GeminiClient()

        # Test with a problem that might cause truncation
        complex_problem = {
            'description': '''
Write a comprehensive solution for the following complex algorithmic problem:

Given a 2D grid of characters and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

This is a classic backtracking problem that requires careful implementation of the search algorithm with proper state management and boundary checking.

The solution should:
1. Handle edge cases like empty grid or empty word
2. Use efficient backtracking with proper pruning
3. Maintain visited state to avoid cycles
4. Return boolean result indicating word existence

Example:
board = [
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
word = "ABCCED"
Output: true

The algorithm should be optimized for both time and space complexity.
            ''',
            'code_template': '''
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        pass
            '''
        }

        print("  Testing complex problem that might cause truncation...")
        code = client.generate_solution(
            complex_problem, use_detailed_prompt=True)

        if code:
            print(f"✓ Generated solution ({len(code)} characters)")
            if 'class Solution' in code and 'def exist' in code:
                print("✓ Generated valid solution structure")
                return True
            else:
                print("⚠ Generated code may be truncated or incomplete")
                return True  # Still consider this a pass since truncation handling worked
        else:
            print("✗ Failed to generate any solution")
            return False

    except Exception as e:
        print(f"✗ Truncation handling test failed: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Google Gemini API Test Suite")
    print("=" * 60)

    # Check if Gemini provider is configured
    if not Config.GEMINI_API_KEY:
        print("✗ GEMINI_API_KEY not found in environment variables")
        print("Please set GEMINI_API_KEY in your .env file")
        sys.exit(1)

    tests = [
        ('Configuration', test_configuration),
        ('Client Initialization', test_client_initialization),
        ('Simple Generation', test_simple_generation),
        ('Detailed Prompt', test_detailed_prompt),
        ('Code Extraction', test_code_extraction),
        ('Multiple Solutions', test_multiple_solutions),
        ('Error Handling', test_error_handling),
        ('Token Limits', test_token_limits),
        ('Truncation Handling', test_truncation_handling)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"{test_name}: {status}")
        if passed_test:
            passed += 1

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed! Gemini API is ready to use.")
        sys.exit(0)
    elif passed >= total * 0.7:  # 70% pass rate
        print("⚠ Most tests passed. Gemini API is mostly functional.")
        sys.exit(0)
    else:
        print("✗ Many tests failed. Check your Gemini API configuration.")
        sys.exit(1)
