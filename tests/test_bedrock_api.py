"""
Quick test to verify AWS Bedrock connectivity and Claude model response.
Usage: python tests/test_bedrock_api.py
"""
import json
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")

MODELS = [
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
]

PROMPT = "Write a Python function that returns the sum of two integers. Return only the code, no explanation."

TOOL = {
    "name": "submit_solution",
    "description": "Submit the Python solution code.",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Complete Python code."}
        },
        "required": ["code"],
    },
}


def test_model(client, model_id: str):
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"{'='*60}")
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.3,
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": "submit_solution"},
    }
    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        response_body = json.loads(response["body"].read())
        code = None
        for block in response_body.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "submit_solution":
                code = block.get("input", {}).get("code", "")
                break
        usage = response_body.get("usage", {})
        print(f"✓ Success!")
        print(f"  Input tokens:  {usage.get('input_tokens', '?')}")
        print(f"  Output tokens: {usage.get('output_tokens', '?')}")
        print(f"  Code:\n---\n{code}\n---")
    except Exception as e:
        print(f"✗ Failed: {e}")


def main():
    print(f"Region: {REGION}")
    client = boto3.client("bedrock-runtime", region_name=REGION)
    for model_id in MODELS:
        test_model(client, model_id)


if __name__ == "__main__":
    main()
