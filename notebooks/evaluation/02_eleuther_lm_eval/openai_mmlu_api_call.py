#!/usr/bin/env python3
"""
OpenAI MMLU Evaluation Script

Test OpenAI models on MMLU dataset using custom task configuration
"""

import os
import json
import yaml
from lm_eval import evaluator
from lm_eval.tasks import TaskManager


def create_mmlu_custom_task():
    """Create custom MMLU task configuration"""
    print("📝 Creating custom MMLU task...")

    # Create task directory
    task_dir = "./tasks/mmlu_openai_test"
    os.makedirs(task_dir, exist_ok=True)

    # Custom MMLU task configuration
    task_config = {
        'task': 'mmlu_openai_test',
        'dataset_path': 'cais/mmlu',
        'dataset_name': 'abstract_algebra',  # Using abstract algebra as test subject
        'test_split': 'test',
        'output_type': 'generate_until',  # Use generate_until instead of multiple_choice
        'doc_to_text': 'Question: {{question}}\n\nChoices:\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\n\nAnswer:',
        'doc_to_target': "{{ ['A', 'B', 'C', 'D'][answer] }}",  # Output letter choices
        'generation_kwargs': {
            'until': ['\n', '.', '!', '?'],
            'max_gen_toks': 1  # Generate only one token
        },
        'metric_list': [
            {
                'metric': 'exact_match',
                'aggregation': 'mean',
                'higher_is_better': True
            }
        ],
        'metadata': {
            'version': 1.0,
            'description': 'Custom MMLU task for OpenAI API testing with generate_until'
        }
    }

    # Save task configuration
    config_file = os.path.join(task_dir, "mmlu_openai_test.yaml")
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(task_config, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ Task configuration saved: {config_file}")
    return os.path.dirname(task_dir)


def test_openai_mmlu(model_name="gpt-3.5-turbo", limit=5):
    """Test OpenAI model performance on MMLU"""

    print(f"🎯 Testing OpenAI {model_name} on MMLU")
    print("=" * 50)

    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your_api_key_here'")
        return None

    # Create custom task
    custom_task_path = create_mmlu_custom_task()

    # Create task manager
    print(f"🔧 Loading custom task: {custom_task_path}")
    task_manager = TaskManager(include_path=custom_task_path)

    # Verify task loading
    custom_tasks = [task for task in task_manager.all_tasks if 'mmlu_openai_test' in task]
    if not custom_tasks:
        print("❌ Failed to load custom task")
        return None

    print(f"✅ Found custom tasks: {custom_tasks}")

    # Run evaluation
    print(f"\n🚀 Starting evaluation...")
    print(f"📱 Model: {model_name}")
    print(f"📋 Task: mmlu_openai_test (abstract_algebra)")
    print(f"🔢 Sample count: {limit}")

    try:
        results = evaluator.simple_evaluate(
            model="openai-chat-completions",
            model_args=f"model={model_name}",
            tasks=["mmlu_openai_test"],
            num_fewshot=5,  # 5-shot
            batch_size=1,
            limit=limit,
            write_out=True,
            log_samples=True,
            task_manager=task_manager,
            apply_chat_template=True
        )

        print("✅ Evaluation completed!")

        # Display results
        task_results = results['results']['mmlu_openai_test']
        print(f"\n📊 Evaluation Results:")
        print(f"   Accuracy: {task_results.get('exact_match', 'N/A'):.4f}")
        print(f"   Std Error: {task_results.get('exact_match_stderr', 'N/A'):.4f}")

        # Save results
        os.makedirs("results", exist_ok=True)
        result_file = f"results/openai_{model_name.replace('-', '_')}_mmlu_results.json"

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 Detailed results saved: {result_file}")

        return results

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_openai_models():
    """Compare different OpenAI model performances"""
    print("\n🔄 Comparing different OpenAI models...")

    models = ["gpt-3.5-turbo", "gpt-4"]
    results = {}

    for model in models:
        print(f"\n{'='*20} {model} {'='*20}")
        result = test_openai_mmlu(model, limit=3)  # Small sample for quick test
        if result:
            acc = result['results']['mmlu_openai_test'].get('exact_match', 0)
            results[model] = acc

    # Display comparison results
    if results:
        print(f"\n📊 Model Comparison Results:")
        print("-" * 30)
        for model, acc in results.items():
            print(f"{model:15}: {acc:.4f}")


def main():
    """Main function"""
    print("🎯 OpenAI MMLU Evaluation Script")
    print("=" * 40)

    # Check dependencies
    try:
        import yaml
        print("✅ PyYAML installed")
    except ImportError:
        print("❌ Please install PyYAML: pip install PyYAML")
        return

    # Single model test
    print("\n1️⃣ Testing GPT-3.5-turbo")
    test_openai_mmlu("gpt-3.5-turbo", limit=5)

    # Optional GPT-4 test
    test_gpt4 = input("\nTest GPT-4? (y/n): ").lower().strip()
    if test_gpt4 == 'y':
        print("\n2️⃣ Testing GPT-4")
        test_openai_mmlu("gpt-4", limit=3)  # GPT-4 is more expensive, use fewer samples

    print("\n🎉 Testing completed!")
    print("\n💡 Tips:")
    print("- Increase limit parameter to test more samples")
    print("- Modify dataset_name to test other MMLU subjects")
    print("- Results are saved in results/ directory")


if __name__ == "__main__":
    main()