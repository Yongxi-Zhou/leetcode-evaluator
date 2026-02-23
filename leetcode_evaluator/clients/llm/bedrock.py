"""
AWS Bedrock Client for generating code solutions
"""
import json
import re
import boto3
from typing import Dict, Optional, Any, List
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
        
    def _invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Invoke the Bedrock model and return text with usage metadata"""
        
        temperature = kwargs.get('temperature', 0.3)
        top_p = kwargs.get('top_p', 0.9)
        max_tokens = kwargs.get('max_tokens', 4096)
        
        # Prepare request based on model family
        if 'anthropic.claude' in self.model_id:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "top_p": top_p
            }
        else:
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            text = None
            usage = {'input_tokens': 0, 'output_tokens': 0}
            
            # Extract text and usage based on model response format
            if 'anthropic.claude' in self.model_id:
                if 'content' in response_body and len(response_body['content']) > 0:
                    text = response_body['content'][0]['text']
                
                # Check for usage metadata in Claude responses
                if 'usage' in response_body:
                    usage['input_tokens'] = response_body['usage'].get('input_tokens', 0)
                    usage['output_tokens'] = response_body['usage'].get('output_tokens', 0)
                    
            elif 'completion' in response_body:
                text = response_body['completion']
            elif 'generated_text' in response_body:
                text = response_body['generated_text']
            
            if text is None:
                raise ValueError(f"Could not extract response from Bedrock: {response_body}")
                
            return {
                'text': text,
                'usage': usage
            }
            
        except Exception as e:
            print(f"Bedrock invocation error: {str(e)}")
            raise e
