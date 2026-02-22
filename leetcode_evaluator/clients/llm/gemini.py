"""
Google Gemini Client for generating code solutions
"""
import re
from typing import Dict, Optional
import google.generativeai as genai
from leetcode_evaluator.core.config import Config
from leetcode_evaluator.clients.llm.base import LLMClient


class GeminiClient(LLMClient):
    """Client for interacting with Google Gemini models"""

    def __init__(self, model_id: str = None, api_key: str = None):
        super().__init__(model_id=model_id or Config.GEMINI_MODEL_ID)
        self.api_key = api_key or Config.GEMINI_API_KEY

        # Initialize Gemini client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_id)

    def _invoke_model(self, prompt: str) -> Dict[str, Any]:
        """Invoke the Gemini model and return text with usage metadata"""

        try:
            generation_config = {
                "temperature": 0.3,
                "top_p": 0.9,
            }

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            text = None
            usage = {'input_tokens': 0, 'output_tokens': 0}
            
            # Extract usage metadata
            if hasattr(response, 'usage_metadata'):
                usage['input_tokens'] = response.usage_metadata.prompt_token_count
                usage['output_tokens'] = response.usage_metadata.candidates_token_count

            # Check if response was truncated
            if response and response.candidates:
                candidate = response.candidates[0]
                if candidate.finish_reason == "MAX_TOKENS":
                    print("⚠ Warning: Response was truncated due to token limit")
                    if hasattr(candidate, 'content') and candidate.content:
                        try:
                            text = candidate.content.parts[0].text
                        except (AttributeError, IndexError):
                            pass

                # Normal response handling
                if not text and response.text:
                    text = response.text

            if text is None:
                raise ValueError(f"Gemini API returned no content or failed. Response: {response}")

            return {
                'text': text,
                'usage': usage
            }

        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            # Try to get usage if available even on error
            usage = {'input_tokens': 0, 'output_tokens': 0}
            try:
                if 'response' in locals() and hasattr(response, 'usage_metadata'):
                    usage['input_tokens'] = response.usage_metadata.prompt_token_count
                    usage['output_tokens'] = response.usage_metadata.candidates_token_count
            except:
                pass
            
            # If it's a truncated response error, try to get partial content
            if "finish_reason" in str(e) and "MAX_TOKENS" in str(e):
                print("Attempting to extract partial content from truncated response...")
                try:
                    if response and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                            partial_text = candidate.content.parts[0].text
                            if partial_text:
                                return {
                                    'text': partial_text,
                                    'usage': usage
                                }
                except Exception as extract_error:
                    print(f"Failed to extract partial content: {extract_error}")
            raise e
