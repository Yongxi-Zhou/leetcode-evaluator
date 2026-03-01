"""
OpenRouter client for accessing multiple LLM providers via a single API key.
Supports OpenAI, Anthropic, Google, Mistral, and more.
"""
from typing import Dict, Any
from openai import OpenAI

from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient


class OpenRouterClient(LLMClient):
    """Client for OpenRouter - centralized API for multiple LLM providers."""

    def __init__(self, model_id: str = None, api_key: str = None, base_url: str = None):
        super().__init__(model_id=model_id or Config.OPENROUTER_MODEL_ID)
        self.api_key = api_key or Config.OPENROUTER_API_KEY
        self.base_url = base_url or Config.OPENROUTER_API_BASE_URL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Invoke OpenRouter model via OpenAI-compatible chat completions API."""
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
                top_p=kwargs.get('top_p', 0.9),
            )

            usage = {'input_tokens': 0, 'output_tokens': 0}
            if getattr(response, 'usage', None):
                usage['input_tokens'] = getattr(response.usage, 'prompt_tokens', 0) or 0
                usage['output_tokens'] = getattr(response.usage, 'completion_tokens', 0) or 0

            if response.choices and len(response.choices) > 0:
                return {
                    'text': response.choices[0].message.content,
                    'usage': usage
                }

            raise ValueError(f"OpenRouter API returned no content. Response: {response}")

        except Exception as e:
            print(f"OpenRouter API error: {str(e)}")
            raise e
