"""
xAI Grok Client for generating code solutions
"""
import re
import requests
from typing import Dict, Optional, Any
from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient


class GrokClient(LLMClient):
    """Client for interacting with xAI Grok models"""
    
    def __init__(self, model_id: str = None, api_key: str = None):
        super().__init__(model_id=model_id or Config.GROK_MODEL_ID)
        self.api_key = api_key or Config.GROK_API_KEY
        self.base_url = Config.GROK_API_BASE_URL
        
    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Invoke the Grok model and return text with usage metadata"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python programmer specializing in algorithmic problem solving."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": kwargs.get('temperature', 0.3),
            "max_tokens": kwargs.get('max_tokens', 4096),
            "top_p": kwargs.get('top_p', 0.9)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            usage = {'input_tokens': 0, 'output_tokens': 0}
            
            if 'usage' in data:
                usage['input_tokens'] = data['usage'].get('prompt_tokens', 0)
                usage['output_tokens'] = data['usage'].get('completion_tokens', 0)
            
            if data.get('choices') and len(data['choices']) > 0:
                return {
                    'text': data['choices'][0]['message']['content'],
                    'usage': usage
                }
            
            raise ValueError(f"Grok API returned no content. Response: {data}")
            
        except Exception as e:
            print(f"Grok API error: {str(e)}")
            raise e
