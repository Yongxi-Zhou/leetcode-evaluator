"""
Configuration module for LeetCode Evaluator
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for all settings"""

    # LeetCode Session Cookies (from browser)
    LEETCODE_SESSION_COOKIE = os.getenv('LEETCODE_SESSION_COOKIE')
    LEETCODE_CSRF_TOKEN = os.getenv('LEETCODE_CSRF_TOKEN')

    # LLM Provider Configuration
    # bedrock, openai, gemini, grok, qwen
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'bedrock')

    # AWS Bedrock Configuration
    AWS_PROFILE = os.getenv('AWS_PROFILE')  # Preferred: use AWS profile
    AWS_ACCESS_KEY_ID = os.getenv(
        'AWS_ACCESS_KEY_ID')  # Fallback: explicit keys
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    BEDROCK_MODEL_ID = os.getenv(
        'BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL_ID = os.getenv('OPENAI_MODEL_ID', 'gpt-4-turbo-preview')

    # Qwen / DashScope (OpenAI-compatible) Configuration
    DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
    QWEN_API_BASE_URL = os.getenv(
        'QWEN_API_BASE_URL',
        os.getenv('OPENAI_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    )
    QWEN_MODEL_ID = os.getenv('QWEN_MODEL_ID', 'qwen-plus')

    # Google Gemini Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL_ID = os.getenv('GEMINI_MODEL_ID', 'gemini-pro')

    # xAI Grok Configuration
    GROK_API_KEY = os.getenv('GROK_API_KEY')
    GROK_MODEL_ID = os.getenv('GROK_MODEL_ID', 'grok-beta')
    GROK_API_BASE_URL = os.getenv('GROK_API_BASE_URL', 'https://api.x.ai/v1')

    # OpenRouter Configuration
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    OPENROUTER_MODEL_ID = os.getenv('OPENROUTER_MODEL_ID', 'openai/gpt-4o')
    OPENROUTER_API_BASE_URL = os.getenv('OPENROUTER_API_BASE_URL', 'https://openrouter.ai/api/v1')

    # LeetCode API Endpoints
    LEETCODE_BASE_URL = "https://leetcode.com"
    LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"
    LEETCODE_LOGIN_URL = "https://leetcode.com/accounts/login/"

    # Evaluation Settings
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    RETRY_DELAY = int(os.getenv('RETRY_DELAY', 2))  # initial delay in seconds
    RETRY_MAX_DELAY = int(os.getenv('RETRY_MAX_DELAY', 10))  # max delay for exponential backoff
    MAX_CONSECUTIVE_ERRORS = int(os.getenv('MAX_CONSECUTIVE_ERRORS', 3))  # circuit breaker threshold

    # LLM Parameters
    MODEL_TEMPERATURE = float(os.getenv('MODEL_TEMPERATURE', 0.3))
    MODEL_TOP_P = float(os.getenv('MODEL_TOP_P', 0.9))
    MODEL_MAX_TOKENS = int(os.getenv('MODEL_MAX_TOKENS', 4096))
    
    # Concurrency and Throttling
    WORKER_THREADS = int(os.getenv('WORKER_THREADS', 4))
    LEETCODE_SUBMISSION_DELAY_S = int(os.getenv('LEETCODE_SUBMISSION_DELAY_S', 10))
    LEETCODE_RATE_LIMIT_COOLDOWN_S = int(os.getenv('LEETCODE_RATE_LIMIT_COOLDOWN_S', 60))
    
    SUBMISSION_POLL_INTERVAL = 2  # seconds
    SUBMISSION_TIMEOUT = 60  # seconds
    
    # Stability Evaluation
    DEFAULT_STABILITY_RUNS = int(os.getenv('DEFAULT_STABILITY_RUNS', 1))

    # Prompts
    DETAILED_PROMPT = """You are an expert algorithm and data structure engineer. 
Solve the following LeetCode problem with optimal time and space complexity.

Requirements:
1. Analyze the problem carefully and identify the optimal approach
2. Consider time complexity - aim for the most efficient solution
3. Consider space complexity - optimize memory usage
4. Write clean, readable Python code with proper variable names
5. Add comments only for complex logic
6. Ensure edge cases are handled
7. The solution must pass all test cases

Problem:
{problem_description}

Code Template:
{code_template}

Provide ONLY the complete Python code solution, no explanations or markdown formatting.
"""

    MINIMAL_PROMPT = """Solve this LeetCode problem in Python:

{problem_description}

Code Template:
{code_template}

Provide ONLY the complete Python code solution, no explanations or markdown formatting.
"""

    # Consolidated Output Directory
    OUTPUT_ROOT = "output"
    DATASET_DIR = "dataset"
    DEFAULT_DATASET_FILE = os.path.join(DATASET_DIR, "main-dataset.json")
    RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
    REPORTS_DIR = os.path.join(OUTPUT_ROOT, "reports")
    EXPERIMENTS_DIR = os.path.join(OUTPUT_ROOT, "experiments")
    
    # Paper-ready Result Structure
    RESULTS_RAW = os.path.join(RESULTS_DIR, "raw")
    RESULTS_SUMMARY = os.path.join(RESULTS_DIR, "summary")
    RESULTS_TABLES = os.path.join(RESULTS_DIR, "tables")
    RESULTS_FIGURES = os.path.join(RESULTS_DIR, "figures")
    RESULTS_EVALUATIONS = os.path.join(RESULTS_DIR, "evaluations")
    RESULTS_PROBLEMS = os.path.join(RESULTS_DIR, "problems")

    @classmethod
    def resolve_dataset_file(cls, dataset: str = "main") -> str:
        """Resolve a dataset selector to a concrete JSON file path."""
        if not dataset:
            return cls.DEFAULT_DATASET_FILE

        if os.path.sep in dataset or dataset.endswith(".json"):
            return dataset

        return os.path.join(cls.DATASET_DIR, f"{dataset}-dataset.json")
    

    @classmethod
    def set_run_id(cls, run_id: str):
        """Update output directories to be nested under a run-specific folder"""
        cls.OUTPUT_ROOT = os.path.join("output", run_id)
        cls.RESULTS_DIR = os.path.join(cls.OUTPUT_ROOT, "results")
        cls.REPORTS_DIR = os.path.join(cls.OUTPUT_ROOT, "reports")
        cls.EXPERIMENTS_DIR = os.path.join(cls.OUTPUT_ROOT, "experiments")
        
        # Update nested result paths
        cls.RESULTS_RAW = os.path.join(cls.RESULTS_DIR, "raw")
        cls.RESULTS_SUMMARY = os.path.join(cls.RESULTS_DIR, "summary")
        cls.RESULTS_TABLES = os.path.join(cls.RESULTS_DIR, "tables")
        cls.RESULTS_FIGURES = os.path.join(cls.RESULTS_DIR, "figures")
        cls.RESULTS_EVALUATIONS = os.path.join(cls.RESULTS_DIR, "evaluations")
        cls.RESULTS_PROBLEMS = os.path.join(cls.RESULTS_DIR, "problems")
        
        # Create directories
        for d in [cls.OUTPUT_ROOT, cls.RESULTS_DIR, cls.REPORTS_DIR, cls.EXPERIMENTS_DIR]:
            os.makedirs(d, exist_ok=True)
    
    # Model Pricing (per 1k tokens)
    MODEL_PRICING = {
        'anthropic.claude-3-5-sonnet-20240620-v1:0': {'input': 0.003, 'output': 0.015},
        'anthropic.claude-3-haiku-20240307-v1:0': {'input': 0.00025, 'output': 0.00125},
        'gpt-4o': {'input': 0.005, 'output': 0.015},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'gemini-1.5-pro': {'input': 0.0035, 'output': 0.0105},
        'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
        'grok-beta': {'input': 0.005, 'output': 0.015},
        'qwen-plus': {'input': 0.0, 'output': 0.0},
        'default': {'input': 0.0, 'output': 0.0}
    }

    @classmethod
    def validate(cls):
        """Validate required configuration based on default LLM provider"""
        return cls.validate_provider(cls.LLM_PROVIDER)

    @classmethod
    def validate_provider(cls, provider: str):
        """Validate required configuration for a specific LLM provider"""
        # LeetCode session cookies required
        required = {
            'LEETCODE_SESSION_COOKIE': cls.LEETCODE_SESSION_COOKIE,
            'LEETCODE_CSRF_TOKEN': cls.LEETCODE_CSRF_TOKEN,
        }

        # Add provider-specific requirements
        provider = provider.lower()

        if provider == 'bedrock':
            # Require either AWS profile OR access keys
            # if not cls.AWS_PROFILE and not (cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY):
            #     raise ValueError(
            #         "Bedrock requires either AWS_PROFILE or both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            #     )
            print(f"no using bedrock")
        elif provider == 'openai':
            required.update({
                'OPENAI_API_KEY': cls.OPENAI_API_KEY,
            })
        elif provider == 'gemini':
            required.update({
                'GEMINI_API_KEY': cls.GEMINI_API_KEY,
            })
        elif provider == 'grok':
            required.update({
                'GROK_API_KEY': cls.GROK_API_KEY,
            })
        elif provider == 'openrouter':
            required.update({
                'OPENROUTER_API_KEY': cls.OPENROUTER_API_KEY,
            })
        elif provider == 'qwen':
            required.update({
                # Allow either explicit DashScope key or OPENAI_API_KEY fallback for compatibility
                'DASHSCOPE_API_KEY/OPENAI_API_KEY': cls.DASHSCOPE_API_KEY or cls.OPENAI_API_KEY,
            })
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. Choose from: bedrock, openai, gemini, grok, qwen")

        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}")

        return True
