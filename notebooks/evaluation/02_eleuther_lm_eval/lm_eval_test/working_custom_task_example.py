#!/usr/bin/env python3
"""
EleutherAI LM Evaluation Harness - Complete Custom Task Working Example

This script demonstrates how to properly register and use custom tasks.
"""

import os
import json
from lm_eval import evaluator
from lm_eval.tasks import TaskManager

def setup_and_test_custom_task():
    """Setup and test custom tasks"""

    print("🎯 EleutherAI LM Evaluation Harness - Complete Custom Task Example")
    print("=" * 60)

    # 1. Setup custom task path
    TASK_REGISTRY_PATH = os.path.abspath("./tasks")
    print(f"🔧 Custom task path: {TASK_REGISTRY_PATH}")
    print(f"📁 Path exists: {os.path.exists(TASK_REGISTRY_PATH)}")

    # 2. Create task manager and register custom tasks
    print("\n📋 Registering custom tasks...")
    task_manager = TaskManager(include_path=TASK_REGISTRY_PATH)

    # 3. Verify custom task registration
    all_tasks = task_manager.all_tasks
    custom_tasks = [task for task in all_tasks if 'my_mmlu_gen' in task]
    print(f"✅ Found custom tasks: {custom_tasks}")

    if not custom_tasks:
        print("❌ No custom tasks found!")
        return

    # 4. Get task details
    print(f"\n📝 Task details for '{custom_tasks[0]}':")
    try:
        # Use correct API to get tasks
        task_dict = task_manager.get_task_dict(custom_tasks)
        task_obj = task_dict[custom_tasks[0]]
        print(f"   - Task type: {type(task_obj)}")
        print(f"   - Task config: {hasattr(task_obj, 'config')}")

        # Display task configuration info
        if hasattr(task_obj, 'config'):
            config = task_obj.config
            print(f"   - Dataset: {getattr(config, 'dataset_path', 'N/A')}")
            print(f"   - Output type: {getattr(config, 'output_type', 'N/A')}")
            print(f"   - Metrics: {getattr(config, 'metric_list', 'N/A')}")

    except Exception as e:
        print(f"⚠️  Error getting task details: {e}")

    # 5. Run custom task evaluation (using small model for quick testing)
    print(f"\n🚀 Running evaluation with custom task '{custom_tasks[0]}'...")

    try:
        # Use simple_evaluate to run evaluation
        results = evaluator.simple_evaluate(
            model="hf",
            model_args="pretrained=gpt2,device=cpu",  # Use small model for quick testing
            tasks=custom_tasks[0],
            num_fewshot=0,
            batch_size=1,
            limit=3,  # Only run 3 samples for testing
            task_manager=task_manager  # Pass task manager containing custom tasks
        )

        print("✅ Evaluation completed successfully!")

        # Display results
        print(f"\n📊 Results for '{custom_tasks[0]}':")
        task_results = results['results'][custom_tasks[0]]
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")

        # Save results
        os.makedirs("results", exist_ok=True)
        result_file = f"results/{custom_tasks[0]}_test_results.json"

        # Custom JSON encoder to handle numpy data types
        import numpy as np
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'dtype'):
                    return str(obj)
                return super(NumpyEncoder, self).default(obj)

        with open(result_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"\n💾 Full results saved to: {result_file}")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("\n💡 This might be expected if:")
        print("1. The model needs more memory")
        print("2. The dataset is not accessible")
        print("3. There are configuration issues")

        # Display detailed error information
        import traceback
        print(f"\n🔍 Detailed error:")
        traceback.print_exc()

def create_improved_task_config():
    """Create an improved task configuration example"""

    print("\n📝 Creating improved task configuration example...")

    # Create a more complete task configuration
    improved_config = {
        'task': 'my_mmlu_gen_improved',
        'group': 'my_mmlu_gen',
        'dataset_path': 'cais/mmlu',
        'dataset_name': 'abstract_algebra',
        'test_split': 'test',
        'output_type': 'generate_until',
        'doc_to_text': 'Q: {{question}}\nOptions:\n(A) {{choices[0]}}\n(B) {{choices[1]}}\n(C) {{choices[2]}}\n(D) {{choices[3]}}\nA:',
        'doc_to_target': "{{ ['A', 'B', 'C', 'D'][answer] }}",
        'generation_kwargs': {
            'until': ['\n', '<|im_end|>'],
            'max_gen_toks': 5
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
            'description': 'Improved custom MMLU generation task for abstract algebra'
        }
    }

    # Save improved configuration
    import yaml
    improved_dir = "./tasks/my_mmlu_gen_improved"
    os.makedirs(improved_dir, exist_ok=True)

    improved_file = os.path.join(improved_dir, "my_mmlu_gen_improved.yaml")
    with open(improved_file, 'w', encoding='utf-8') as f:
        yaml.dump(improved_config, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ Improved task config saved to: {improved_file}")
    return improved_file

def main():
    """Main function"""

    # 1. Test existing custom tasks
    setup_and_test_custom_task()

    # 2. Create improved task configuration example
    create_improved_task_config()

    print("\n" + "=" * 60)
    print("🎉 Custom task example completed!")
    print("\n💡 Key points summary:")
    print("1. ✅ Use TaskManager(include_path=path) to register custom tasks")
    print("2. ✅ Ensure YAML file contains all necessary fields")
    print("3. ✅ Pass task manager to simple_evaluate()")
    print("4. ✅ Use small models and few samples for quick testing")

if __name__ == "__main__":
    main()