"""Anthropic real-time client for single-request generation."""
import time
from typing import Dict, Any

import anthropic

from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient

_CODE_TOOL = {
    "name": "submit_solution",
    "description": "Submit the Python solution code for the LeetCode problem.",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python solution code, ready to run.",
            }
        },
        "required": ["code"],
    },
}


class AnthropicClient(LLMClient):
    """Real-time Anthropic client using the Messages API with tool_use structured output."""

    def __init__(self, model_id: str = None, api_key: str = None):
        super().__init__(model_id=model_id or Config.ANTHROPIC_MODEL_ID)
        self.client = anthropic.Anthropic(
            api_key=api_key or Config.ANTHROPIC_API_KEY,
        )

    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get('temperature', Config.MODEL_TEMPERATURE)
        max_tokens = kwargs.get('max_tokens', Config.MODEL_MAX_TOKENS)
        top_p = kwargs.get('top_p', Config.MODEL_TOP_P)

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system="You are an expert Python programmer specializing in algorithmic problem solving.",
            messages=[{"role": "user", "content": prompt}],
            tools=[_CODE_TOOL],
            tool_choice={"type": "tool", "name": "submit_solution"},
        )

        text = ""
        for block in response.content:
            if hasattr(block, 'type') and block.type == 'tool_use' and block.name == 'submit_solution':
                text = block.input.get('code', '')
                break

        return {
            'text': text,
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
            },
        }
