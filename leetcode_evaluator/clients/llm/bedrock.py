"""
AWS Bedrock Client for generating code solutions
"""
import json
import re
import boto3
from typing import Dict, Optional
from botocore.config import Config as BotoConfig
from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient


class BedrockClient(LLMClient):
    """Client for interacting with AWS Bedrock models"""
    
    def __init__(self, model_id: str = None, region: str = None, profile: str = None):
        super().__init__(model_id=model_id or Config.BEDROCK_MODEL_ID)
        self.region = region or Config.AWS_REGION
        self.profile = profile or Config.AWS_PROFILE
        
        # Initialize Bedrock client
        boto_config = BotoConfig(
            region_name=self.region,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        # Use AWS profile if specified, otherwise use default credentials or explicit keys
        if self.profile:
            # Use profile from ~/.aws/credentials
            session = boto3.Session(profile_name=self.profile, region_name=self.region)
            self.client = session.client('bedrock-runtime', config=boto_config)
            print(f"✓ Using AWS profile: {self.profile}")
        else:
            # Use default credentials chain (env vars, instance profile, etc.)
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=self.region,
                config=boto_config
            )
        
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
            # Call Bedrock API
            print(f"Using model: {self.model_id}")
            print(f"With prompt: {prompt}")
            response = self._invoke_model(prompt)
            
            if response:
                # Extract code from response
                code = self._extract_code(response)
                return code
            
        except Exception as e:
            print(f"Error generating solution: {str(e)}")
        
        return None
    
    def _invoke_model(self, prompt: str) -> Optional[str]:
        """Invoke the Bedrock model with the given prompt"""
        
        # Prepare request based on model family
        if 'anthropic.claude' in self.model_id:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "top_p": 0.9
            }
        else:
            # Generic format for other models
            body = {
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.3
            }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract text based on model response format
            if 'anthropic.claude' in self.model_id:
                if 'content' in response_body and len(response_body['content']) > 0:
                    return response_body['content'][0]['text']
            elif 'completion' in response_body:
                return response_body['completion']
            elif 'generated_text' in response_body:
                return response_body['generated_text']
            
            return None
            
        except Exception as e:
            print(f"Model invocation error: {str(e)}")
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
