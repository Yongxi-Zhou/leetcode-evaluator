# LLM Provider Guide

The LeetCode Evaluator supports multiple LLM providers. Choose the one that best fits your needs.

## Supported Providers

1. **AWS Bedrock** (Claude models)
2. **OpenAI** (GPT-4, GPT-3.5)
3. **Google Gemini** (Gemini Pro, Ultra)
4. **xAI Grok** (Grok Beta)

## Configuration

### Method 1: Environment Variables (.env file)

Set the `LLM_PROVIDER` variable in your `.env` file:

```bash
# Choose your provider
LLM_PROVIDER=openai  # or bedrock, gemini, grok

# Add corresponding API keys
OPENAI_API_KEY=your_key_here
```

### Method 2: Command Line Arguments

Override the provider at runtime:

```bash
python main.py --provider openai --model gpt-4-turbo-preview
```

## Provider Details

### 1. AWS Bedrock (Default)

**Best for:** Enterprise deployments, Claude models, AWS-integrated workflows

**Setup (Option 1 - Recommended: AWS Profile):**
```bash
# In .env
LLM_PROVIDER=bedrock
AWS_PROFILE=default  # or your profile name
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
```

**Setup (Option 2 - Explicit Keys):**
```bash
# In .env
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
```

**Prerequisites:**
- AWS CLI configured with credentials: `aws configure`
- Or AWS profile in `~/.aws/credentials`
- Bedrock access enabled in your AWS account

**Available Models:**
- `anthropic.claude-3-5-sonnet-20241022-v2:0` (Recommended)
- `anthropic.claude-3-opus-20240229-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`

**Usage:**
```bash
python main.py --num-problems 5 --provider bedrock
```

**Pricing:** ~$3 per 1M input tokens, ~$15 per 1M output tokens (Claude 3.5 Sonnet)

---

### 2. OpenAI

**Best for:** GPT-4 access, wide model selection, robust API

**Setup:**
```bash
# In .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL_ID=gpt-4-turbo-preview
```

**Available Models:**
- `gpt-4-turbo-preview` (Recommended for quality)
- `gpt-4` (Stable, high quality)
- `gpt-3.5-turbo` (Fast, cost-effective)
- `gpt-4o` (Latest multimodal model)

**Usage:**
```bash
python main.py --num-problems 5 --provider openai --model gpt-4-turbo-preview
```

**Pricing:** 
- GPT-4 Turbo: ~$10 per 1M input tokens, ~$30 per 1M output tokens
- GPT-3.5 Turbo: ~$0.50 per 1M input tokens, ~$1.50 per 1M output tokens

**Get API Key:** https://platform.openai.com/api-keys

---

### 3. Google Gemini

**Best for:** Google ecosystem integration, competitive pricing

**Setup:**
```bash
# In .env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL_ID=gemini-pro
```

**Available Models:**
- `gemini-pro` (Recommended for most use cases)
- `gemini-pro-vision` (Multimodal capabilities)
- `gemini-ultra` (Highest capability, limited access)

**Usage:**
```bash
python main.py --num-problems 5 --provider gemini --model gemini-pro
```

**Pricing:** 
- Gemini Pro: Free tier available, then ~$0.50 per 1M tokens
- Very competitive for high-volume use

**Get API Key:** https://makersuite.google.com/app/apikey

---

### 4. xAI Grok

**Best for:** Latest xAI technology, early access features

**Setup:**
```bash
# In .env
LLM_PROVIDER=grok
GROK_API_KEY=your_key
GROK_MODEL_ID=grok-beta
GROK_API_BASE_URL=https://api.x.ai/v1
```

**Available Models:**
- `grok-beta` (Current generation)
- More models coming as xAI expands

**Usage:**
```bash
python main.py --num-problems 5 --provider grok --model grok-beta
```

**Pricing:** Contact xAI for current pricing

**Get API Key:** https://x.ai/ (requires access)

---

## Comparing Providers

### Performance Comparison

| Provider | Speed | Code Quality | Cost | Availability |
|----------|-------|--------------|------|--------------|
| Bedrock (Claude 3.5) | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| OpenAI (GPT-4) | ★★★★☆ | ★★★★★ | ★★☆☆☆ | ★★★★★ |
| OpenAI (GPT-3.5) | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ |
| Gemini Pro | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ |
| Grok | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ |

### Use Case Recommendations

**Research/Academic:**
- Use GPT-4 or Claude 3.5 Sonnet for highest quality
- Multiple providers for comparison studies

**Cost-Conscious:**
- Start with Gemini Pro (best free tier)
- GPT-3.5 Turbo for paid but affordable option

**Enterprise:**
- AWS Bedrock for AWS-integrated environments
- OpenAI for general enterprise use

**Experimentation:**
- Try multiple providers with same problems
- Compare results using the built-in metrics

---

## Multi-Provider Evaluation

### Compare Different Models

Run the same evaluation with different providers:

```bash
# Test with GPT-4
python main.py --provider openai --model gpt-4 \
  --num-problems 10 --difficulty MEDIUM

# Test with Claude
python main.py --provider bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0 \
  --num-problems 10 --difficulty MEDIUM

# Test with Gemini
python main.py --provider gemini --model gemini-pro \
  --num-problems 10 --difficulty MEDIUM

# Compare reports
```

### Batch Comparison Script

Create `compare_providers.sh`:

```bash
#!/bin/bash

PROBLEMS=10
DIFFICULTY=MEDIUM

echo "Running multi-provider comparison..."

providers=("openai" "bedrock" "gemini")
models=("gpt-4-turbo-preview" "anthropic.claude-3-5-sonnet-20241022-v2:0" "gemini-pro")

for i in ${!providers[@]}; do
    provider=${providers[$i]}
    model=${models[$i]}
    
    echo "Testing $provider with $model..."
    python main.py \
        --provider $provider \
        --model $model \
        --num-problems $PROBLEMS \
        --difficulty $DIFFICULTY
    
    sleep 5  # Rate limiting
done

echo "Comparison complete! Check reports/ directory"
```

---

## Troubleshooting

### Provider Not Working

1. **Check API Key:**
   ```bash
   # Verify key is set
   echo $OPENAI_API_KEY  # or other provider
   ```

2. **Verify Configuration:**
   ```python
   from config import Config
   Config.validate()
   ```

3. **Test API Access:**
   ```bash
   # OpenAI
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   
   # Gemini
   curl "https://generativelanguage.googleapis.com/v1/models?key=$GEMINI_API_KEY"
   ```

### Rate Limiting

If you hit rate limits:

1. **Reduce concurrent requests** in `config.py`
2. **Increase delays** between API calls
3. **Use a higher-tier API plan**
4. **Switch to a different provider** temporarily

### Model Not Available

Some models may not be available in all regions or accounts:

1. **Check model availability** in your region
2. **Request access** from the provider
3. **Use an alternative model** from the same provider

---

## Best Practices

1. **Start Small:** Test with 5 problems before scaling up
2. **Monitor Costs:** Track API usage and set billing alerts
3. **Use Appropriate Models:** Don't use GPT-4 if GPT-3.5 suffices
4. **Cache Results:** Save evaluation results to avoid re-running
5. **Compare Providers:** Run same problems on multiple providers
6. **Document Choices:** Note which provider/model was used in reports

---

## Cost Optimization Tips

1. **Use Minimal Prompts** for simple problems
2. **Batch API calls** where possible
3. **Start with cheaper models** (GPT-3.5, Gemini Pro)
4. **Use provider free tiers** for testing
5. **Cache successful solutions** to avoid regeneration
6. **Set cost limits** in provider dashboards

---

## Provider Status & Updates

Check provider status pages:
- **AWS Bedrock:** https://status.aws.amazon.com/
- **OpenAI:** https://status.openai.com/
- **Google Cloud:** https://status.cloud.google.com/
- **xAI:** https://status.x.ai/ (when available)

---

## Support

For provider-specific issues:
- **AWS Bedrock:** AWS Support Console
- **OpenAI:** https://help.openai.com/
- **Google Gemini:** Google Cloud Support
- **xAI Grok:** https://x.ai/support

For evaluation tool issues:
- Check README.md
- Review USAGE_GUIDE.md
- Open an issue on the repository
