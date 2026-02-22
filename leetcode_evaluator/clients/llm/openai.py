"""
OpenAI Client for generating code solutions
"""
import re
from typing import Dict, Optional
from openai import OpenAI
from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient


class OpenAIClient(LLMClient):
    """Client for interacting with OpenAI models (GPT-4, GPT-3.5, etc.)"""
    
    def __init__(self, model_id: str = None, api_key: str = None):
        super().__init__(model_id=model_id or Config.OPENAI_MODEL_ID)
        self.api_key = api_key or Config.OPENAI_API_KEY
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
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
            # Call OpenAI API
            response = self._invoke_model(prompt)
            
            if response:
                # Extract code from response
                code = self._extract_code(response)
                return code
            
        except Exception as e:
            print(f"Error generating solution: {str(e)}")
        
        return None
    
    def _invoke_model(self, prompt: str) -> Optional[str]:
        """Invoke the OpenAI model with the given prompt"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Python programmer specializing in algorithmic problem solving."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=4096,
                top_p=0.9
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            
            return None
            
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
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
        code_indicators = ['def ', 'class ', 'import ', 'from ', 'return ', '    ']
        
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
