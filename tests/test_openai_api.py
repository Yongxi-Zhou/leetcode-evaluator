"""
Test script to verify OpenAI API is working correctly
"""
import os
import sys
from leetcode_evaluator.clients.llm.openai import OpenAIClient
from leetcode_evaluator.core.config import Config


def test_configuration():
    """Test if OpenAI configuration is valid"""
    print("Testing OpenAI configuration...")

    try:
        Config.validate_provider('openai')
        print("✓ Configuration validation passed")
        return True
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_client_initialization():
    """Test OpenAI client initialization"""
    print("\nTesting OpenAI client initialization...")

    try:
        client = OpenAIClient()
        print(f"✓ Client initialized successfully")
        print(f"  Model ID: {client.model_id}")
        print(
            f"  API Key: {'*' * 20}{client.api_key[-4:] if client.api_key else 'None'}")
        return True
    except Exception as e:
        print(f"✗ Client initialization failed: {e}")
        return False


def test_list_models():
    """Test listing available OpenAI models"""
    print("\nTesting OpenAI model listing...")

    try:
        client = OpenAIClient()

        print("  Fetching available models...")
        models = client.client.models.list()
        if models and models.data:
            print(f"✓ Successfully retrieved {len(models.data)} models")

            # Filter for GPT models (most relevant for this use case)
            gpt_models = [
                model for model in models.data if 'gpt' in model.id.lower()]

            print(f"  Found {len(gpt_models)} GPT models:")
            for model in gpt_models:  # Show first 10 GPT models
                print(f"    - {model.id}")

            if len(gpt_models) > 10:
                print(f"    ... and {len(gpt_models) - 10} more GPT models")

            # Check if the configured model is available
            configured_model = client.model_id
            available_model_ids = [model.id for model in models.data]

            if configured_model in available_model_ids:
                print(f"✓ Configured model '{configured_model}' is available")
            else:
                print(
                    f"⚠ Configured model '{configured_model}' not found in available models")
                print("  Available models include:")
                for model_id in available_model_ids[:5]:
                    print(f"    - {model_id}")
                if len(available_model_ids) > 5:
                    print(f"    ... and {len(available_model_ids) - 5} more")

            return True
        else:
            print("✗ Failed to retrieve models or no models found")
            return False

    except Exception as e:
        print(f"✗ Model listing failed: {e}")
        return False


def test_simple_generation():
    """Test simple code generation"""
    print("\nTesting simple code generation...")

    try:
        client = OpenAIClient()

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
        client = OpenAIClient()

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
        client = OpenAIClient()

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
        client = OpenAIClient()

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
        client = OpenAIClient()

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


def test_api_limits():
    """Test API rate limits and token limits"""
    print("\nTesting API limits...")

    try:
        client = OpenAIClient()

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
        else:
            print("    ⚠ Long prompt may have hit token limits (this is expected)")

        print("✓ API limits test completed")
        return True

    except Exception as e:
        print(f"✗ API limits test failed: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("OpenAI API Test Suite")
    print("=" * 60)

    # Check if OpenAI provider is configured
    if not Config.OPENAI_API_KEY:
        print("✗ OPENAI_API_KEY not found in environment variables")
        print("Please set OPENAI_API_KEY in your .env file")
        sys.exit(1)

    tests = [
        # ('Configuration', test_configuration),
        # ('Client Initialization', test_client_initialization),
        # ('List Models', test_list_models),
        ('Simple Generation', test_simple_generation),
        # ('Detailed Prompt', test_detailed_prompt),
        # ('Code Extraction', test_code_extraction),
        # ('Multiple Solutions', test_multiple_solutions),
        # ('Error Handling', test_error_handling),
        # ('API Limits', test_api_limits)
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
        print("✓ All tests passed! OpenAI API is ready to use.")
        sys.exit(0)
    elif passed >= total * 0.7:  # 70% pass rate
        print("⚠ Most tests passed. OpenAI API is mostly functional.")
        sys.exit(0)
    else:
        print("✗ Many tests failed. Check your OpenAI API configuration.")
        sys.exit(1)
