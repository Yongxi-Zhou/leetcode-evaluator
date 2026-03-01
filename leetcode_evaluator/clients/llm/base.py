import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from leetcode_evaluator.core.config import Config


@dataclass
class GenerationResult:
    """Standardized result for LLM generation"""
    code: Optional[str] = None
    raw_response: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    status: str = "Success"
    error: Optional[str] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients with centralized reliability and logging logic"""
    
    def __init__(self, model_id: str = None):
        self.model_id = model_id
    
    def generate_solution(self, problem: Dict, use_detailed_prompt: bool = True, **kwargs) -> GenerationResult:
        """
        Main entry point for generating a solution.
        Orchestrates prompt preparation, model invocation with retries, and code extraction.
        
        Args:
            problem: Problem dictionary
            use_detailed_prompt: Whether to use detailed prompt
            **kwargs: Generation parameters (temperature, top_p, max_tokens)
        """
        prompt = self._prepare_prompt(problem, use_detailed_prompt)
        start_time = time.time()
        
        # Merge kwargs with defaults
        gen_params = {
            'temperature': kwargs.get('temperature', Config.MODEL_TEMPERATURE),
            'top_p': kwargs.get('top_p', Config.MODEL_TOP_P),
            'max_tokens': kwargs.get('max_tokens', Config.MODEL_MAX_TOKENS)
        }
        
        try:
            # The abstract _invoke_model will be called here via retry wrapper.
            # Response should contain 'text' and 'usage' (input_tokens, output_tokens).
            response_data = self._invoke_model_with_retry(prompt, **gen_params)
            latency_ms = (time.time() - start_time) * 1000
            
            text = response_data.get('text')
            usage = response_data.get('usage', {})
            
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            
            # Calculate cost
            pricing = Config.MODEL_PRICING.get(self.model_id, Config.MODEL_PRICING['default'])
            cost = (input_tokens * pricing['input'] + output_tokens * pricing['output']) / 1000
            
            code = self._extract_code(text) if text else None
            
            return GenerationResult(
                code=code,
                raw_response=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost=cost
            )
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"✗ Failed to generate solution after retries: {str(e)}")
            return GenerationResult(status="Error", error=str(e), latency_ms=latency_ms)

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(multiplier=Config.RETRY_DELAY, max=Config.RETRY_MAX_DELAY),
        reraise=True
    )
    def _invoke_model_with_retry(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Wrapper around _invoke_model with retry logic"""
        return self._invoke_model(prompt, **kwargs)

    @abstractmethod
    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Abstract method to be implemented by subclasses.
        Should return a dict with:
        {
            'text': str,
            'usage': {
                'input_tokens': int,
                'output_tokens': int
            }
        }
        """
        pass
    
    def _prepare_prompt(self, problem: Dict, use_detailed_prompt: bool) -> str:
        """Prepare the prompt based on the problem and prompt type"""
        if use_detailed_prompt:
            return Config.DETAILED_PROMPT.format(
                problem_description=problem['description'],
                code_template=problem['code_template']
            )
        else:
            return Config.MINIMAL_PROMPT.format(
                problem_description=problem['description'],
                code_template=problem['code_template']
            )
    
    def _extract_code(self, response: str) -> Optional[str]:
        """
        Centralized method to extract Python code from model response
        """
        if not response:
            return None
        
        # Try to find code in markdown blocks
        code_block_pattern = r'```(?:python|python3)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Look for class definitions (common in LeetCode)
        class_pattern = r'(class\s+\w+.*?)(?=\n\n|\Z)'
        class_matches = re.findall(class_pattern, response, re.DOTALL)
        
        if class_matches:
            return class_matches[0].strip()
        
        # Final fallback check
        lines = response.strip().split('\n')
        code_indicators = ['def ', 'class ', 'import ', 'from ', 'return ', '    ']
        
        if any(any(line.strip().startswith(indicator) for indicator in code_indicators) 
               for line in lines):
            return response.strip()
        
        if 'class ' in response:
            start_idx = response.find('class ')
            return response[start_idx:].strip()
            
        return None

    def validate_code_syntax(self, code: str) -> bool:
        """Validate if the code has valid Python syntax"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def generate_multiple_solutions(self, problem: Dict, count: int = 5, 
                                   use_detailed_prompt: bool = True) -> List[str]:
        """Generate multiple solutions for the same problem"""
        solutions = []
        for i in range(count):
            print(f"  Generating solution {i+1}/{count}...")
            solution = self.generate_solution(problem, use_detailed_prompt)
            if solution and self.validate_code_syntax(solution):
                solutions.append(solution)
        return solutions


class LLMClientFactory:
    """Factory for creating LLM clients based on provider"""
    
    @staticmethod
    def create_client(provider: str = None, model_id: str = None) -> LLMClient:
        """
        Create an LLM client based on provider
        
        Args:
            provider: Provider name (bedrock, openai, gemini, grok, qwen)
            model_id: Model ID to use
        
        Returns:
            LLM client instance
        """
        provider = provider or Config.LLM_PROVIDER
        
        if provider == 'bedrock':
            from leetcode_evaluator.clients.llm.bedrock import BedrockClient
            return BedrockClient(model_id=model_id)
        elif provider == 'openai':
            from leetcode_evaluator.clients.llm.openai import OpenAIClient
            return OpenAIClient(model_id=model_id)
        elif provider == 'gemini':
            from leetcode_evaluator.clients.llm.gemini import GeminiClient
            return GeminiClient(model_id=model_id)
        elif provider == 'grok':
            from leetcode_evaluator.clients.llm.grok import GrokClient
            return GrokClient(model_id=model_id)
        elif provider == 'openrouter':
            from leetcode_evaluator.clients.llm.openrouter import OpenRouterClient
            return OpenRouterClient(model_id=model_id)
        elif provider == 'qwen':
            from leetcode_evaluator.clients.llm.qwen import QwenClient
            return QwenClient(model_id=model_id)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
