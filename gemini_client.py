"""
Google Gemini Client for generating code solutions
"""
import re
from typing import Dict, Optional
import google.generativeai as genai
from config import Config
from llm_client import LLMClient


class GeminiClient(LLMClient):
    """Client for interacting with Google Gemini models"""

    def __init__(self, model_id: str = None, api_key: str = None):
        super().__init__(model_id=model_id or Config.GEMINI_MODEL_ID)
        self.api_key = api_key or Config.GEMINI_API_KEY

        # Initialize Gemini client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_id)

    def generate_solution(self, problem: Dict, use_detailed_prompt: bool = True) -> Optional[str]:
        """
        Generate a solution for the given problem

        Args:
            problem: Problem dictionary with description and code_template
            use_detailed_prompt: Whether to use detailed prompt or minimal

        Returns:
            Generated Python code solution
        """
        # Prepare prompt
        prompt = self._prepare_prompt(problem, use_detailed_prompt)

        try:
            # Call Gemini API
            response = self._invoke_model(prompt)

            if response:
                # Extract code from response
                code = self._extract_code(response)
                return code

        except Exception as e:
            print(f"Error generating solution: {str(e)}")

        return None

    def _invoke_model(self, prompt: str) -> Optional[str]:
        """Invoke the Gemini model with the given prompt"""

        try:
            generation_config = {
                "temperature": 0.3,
                "top_p": 0.9,
                # "max_output_tokens": 8192,  # Increased from 4096
            }
            print('prompt')
            print(prompt)

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            print(response)

            # Check if response was truncated
            if response and response.candidates:
                candidate = response.candidates[0]
                if candidate.finish_reason == "MAX_TOKENS":
                    print("⚠ Warning: Response was truncated due to token limit")
                    # Try to extract partial content if available
                    if hasattr(candidate, 'content') and candidate.content:
                        try:
                            return candidate.content.parts[0].text
                        except (AttributeError, IndexError):
                            pass

                # Normal response handling
                if response.text:
                    return response.text

            return None

        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            # If it's a truncated response error, try to get partial content
            if "finish_reason" in str(e) and "MAX_TOKENS" in str(e):
                print("Attempting to extract partial content from truncated response...")
                try:
                    if response and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                            partial_text = candidate.content.parts[0].text
                            if partial_text:
                                print(
                                    f"✓ Extracted {len(partial_text)} characters of partial content")
                                return partial_text
                except Exception as extract_error:
                    print(
                        f"Failed to extract partial content: {extract_error}")
            return None

    def _extract_code(self, response: str) -> Optional[str]:
        """
        Extract Python code from model response

        Handles various response formats:
        - Markdown code blocks (```python ... ```)
        - Plain code
        - Code with explanations
        """
        if not response:
            return None

        # Try to find code in markdown blocks
        code_block_pattern = r'```(?:python|python3)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Return the first code block found
            return matches[0].strip()

        # If no markdown blocks, look for class definitions (common in LeetCode)
        class_pattern = r'(class\s+\w+.*?)(?=\n\n|\Z)'
        class_matches = re.findall(class_pattern, response, re.DOTALL)

        if class_matches:
            return class_matches[0].strip()

        # Last resort: check if the entire response looks like code
        lines = response.strip().split('\n')
        code_indicators = ['def ', 'class ',
                           'import ', 'from ', 'return ', '    ']

        if any(any(line.strip().startswith(indicator) for indicator in code_indicators)
               for line in lines):
            return response.strip()

        # If we still can't find code, try to extract everything between first 'class' and end
        if 'class ' in response:
            start_idx = response.find('class ')
            return response[start_idx:].strip()

        return None

    def generate_multiple_solutions(self, problem: Dict, count: int = 5,
                                    use_detailed_prompt: bool = True) -> list:
        """
        Generate multiple solutions for the same problem

        Args:
            problem: Problem dictionary
            count: Number of solutions to generate
            use_detailed_prompt: Whether to use detailed prompt

        Returns:
            List of generated code solutions
        """
        solutions = []

        for i in range(count):
            print(f"Generating solution {i+1}/{count}...")
            solution = self.generate_solution(problem, use_detailed_prompt)

            if solution and self.validate_code_syntax(solution):
                solutions.append(solution)
            else:
                print(f"  ✗ Solution {i+1} failed validation")

        return solutions
