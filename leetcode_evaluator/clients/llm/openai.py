"""
OpenAI Client for generating code solutions
"""
import re
from typing import Dict, Optional, Any
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
        
    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Invoke the OpenAI model and return text with usage metadata"""
        
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
                temperature=kwargs.get('temperature', 0.3),
                max_tokens=kwargs.get('max_tokens', 4096),
                top_p=kwargs.get('top_p', 0.9)
            )
            
            usage = {'input_tokens': 0, 'output_tokens': 0}
            if response.usage:
                usage['input_tokens'] = response.usage.prompt_tokens
                usage['output_tokens'] = response.usage.completion_tokens
            
            if response.choices and len(response.choices) > 0:
                return {
                    'text': response.choices[0].message.content,
                    'usage': usage
                }
            
            raise ValueError(f"OpenAI API returned no content. Response: {response}")
            
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            raise e
