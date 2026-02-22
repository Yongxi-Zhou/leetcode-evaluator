"""
Base LLM Client interface and factory for multiple providers
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
from leetcode_evaluator.core.config import Config


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model_id: str = None):
        self.model_id = model_id
    
    @abstractmethod
    def generate_solution(self, problem: Dict, use_detailed_prompt: bool = True) -> Optional[str]:
        """
        Generate a solution for the given problem
        
        Args:
            problem: Problem dictionary with description and code_template
            use_detailed_prompt: Whether to use detailed prompt or minimal
        
        Returns:
            Generated Python code solution
        """
        pass
    
    @abstractmethod
    def _invoke_model(self, prompt: str) -> Optional[str]:
        """Invoke the model with the given prompt"""
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
    
    def validate_code_syntax(self, code: str) -> bool:
        """Validate if the code has valid Python syntax"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False


class LLMClientFactory:
    """Factory for creating LLM clients based on provider"""
    
    @staticmethod
    def create_client(provider: str = None, model_id: str = None) -> LLMClient:
        """
        Create an LLM client based on provider
        
        Args:
            provider: Provider name (bedrock, openai, gemini, grok)
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
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
