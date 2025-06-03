# OpenAI MMLU Evaluation

Simple script to test OpenAI models on MMLU dataset using lm_eval framework.

## Quick Start

```bash
# Install dependencies
pip install lm_eval PyYAML

# Set API key
export OPENAI_API_KEY="your_api_key_here"

# Run evaluation
python openai_mmlu_api_call.py
```

## Key Features

- ✅ Custom MMLU task configuration
- ✅ OpenAI Chat API compatible
- ✅ Cost-controlled testing (default: 5 samples)
- ✅ JSON result export

## Technical Solution

### Fixed Issues
1. **Model name**: `openai-chat-completions` (not `openai-chat`)
2. **Required parameter**: `apply_chat_template=True`
3. **Output type**: `generate_until` (not `multiple_choice`)
4. **Metric**: `exact_match` (not `acc`)

### Task Configuration
```yaml
output_type: generate_until
doc_to_target: "{{ ['A', 'B', 'C', 'D'][answer] }}"
generation_kwargs:
  until: ['\n', '.', '!', '?']
  max_gen_toks: 1
metric_list:
  - metric: exact_match
```

### Evaluation Call
```python
results = evaluator.simple_evaluate(
    model="openai-chat-completions",
    model_args="model=gpt-3.5-turbo",
    tasks=["mmlu_openai_test"],
    apply_chat_template=True,
    task_manager=task_manager
)
```

## Customization

- **Subject**: Change `dataset_name` in task config
- **Samples**: Modify `limit` parameter
- **Model**: Use `gpt-4` instead of `gpt-3.5-turbo`
- **Few-shot**: Adjust `num_fewshot` parameter

## Output Example

```
🎯 Testing OpenAI gpt-3.5-turbo on MMLU
📱 Model: gpt-3.5-turbo
📋 Task: mmlu_openai_test (abstract_algebra)
🔢 Sample count: 5

✅ Evaluation completed!

📊 Evaluation Results:
   Accuracy: 0.6000
   Std Error: 0.2191

💾 Detailed results saved: results/openai_gpt_3_5_turbo_mmlu_results.json
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `insufficient_quota` | Check OpenAI account balance |
| `OPENAI_API_KEY not set` | Export API key environment variable |
| `No module named 'yaml'` | `pip install PyYAML` |
| Task loading failed | Check YAML syntax in task config |