"""
Qwen (Alibaba DashScope) client via OpenAI-compatible API.
"""
import json
import time
from typing import Dict, Any
import httpx
from openai import OpenAI

from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient


class QwenClient(LLMClient):
    """Client for Qwen models using DashScope OpenAI-compatible endpoint."""

    def __init__(self, model_id: str = None, api_key: str = None, base_url: str = None):
        super().__init__(model_id=model_id or Config.QWEN_MODEL_ID)
        self.api_key = api_key or Config.DASHSCOPE_API_KEY or Config.OPENAI_API_KEY
        self.base_url = base_url or Config.QWEN_API_BASE_URL
        self.request_timeout_s = Config.LLM_REQUEST_TIMEOUT_S
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(
                timeout=self.request_timeout_s,
                connect=min(10.0, float(self.request_timeout_s)),
                read=float(self.request_timeout_s),
                write=min(30.0, float(self.request_timeout_s)),
                pool=min(10.0, float(self.request_timeout_s)),
            )
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_s,
            max_retries=0,
            http_client=self.http_client,
        )

    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Invoke Qwen model via OpenAI-compatible chat completions API."""
        request_timeout_s = kwargs.get('timeout', self.request_timeout_s)
        start_time = time.time()
        print(
            f"Qwen request started: model={self.model_id}, "
            f"timeout={request_timeout_s}s, prompt_chars={len(prompt)}"
        )
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
                timeout=request_timeout_s,
            )
            elapsed_s = time.time() - start_time

            usage = {'input_tokens': 0, 'output_tokens': 0}
            if getattr(response, 'usage', None):
                usage['input_tokens'] = getattr(response.usage, 'prompt_tokens', 0) or 0
                usage['output_tokens'] = getattr(response.usage, 'completion_tokens', 0) or 0

            if response.choices and len(response.choices) > 0:
                print(
                    f"Qwen request completed: model={self.model_id}, "
                    f"elapsed={elapsed_s:.2f}s, input_tokens={usage['input_tokens']}, "
                    f"output_tokens={usage['output_tokens']}"
                )
                return {
                    'text': response.choices[0].message.content,
                    'usage': usage
                }

            raise ValueError(f"Qwen API returned no content. Response: {response}")

        except Exception as e:
            elapsed_s = time.time() - start_time
            response = getattr(e, 'response', None)
            response_preview = None
            if response is not None:
                try:
                    response_preview = response.text
                except Exception:
                    response_preview = repr(response)
            body = getattr(e, 'body', None)
            body_preview = None
            if body is not None:
                try:
                    body_preview = json.dumps(body, ensure_ascii=True)[:1000]
                except Exception:
                    body_preview = repr(body)
            print(
                f"Qwen API error for model {self.model_id} "
                f"after {elapsed_s:.2f}s "
                f"(timeout={request_timeout_s}s): {type(e).__name__}: {str(e)}"
            )
            if response_preview:
                print(f"Qwen API error response: {response_preview[:1000]}")
            if body_preview:
                print(f"Qwen API error body: {body_preview}")
            raise e
