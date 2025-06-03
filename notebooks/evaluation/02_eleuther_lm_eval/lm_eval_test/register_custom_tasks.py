#!/usr/bin/env python3
"""
EleutherAI LM Evaluation Harness - Custom Task Registration Example

This script demonstrates how to register and use custom evaluation tasks.
"""

import os
import json
import yaml
from lm_eval import evaluator
from lm_eval.tasks import TaskManager

def setup_custom_tasks():
    """Setup custom task path and register tasks"""

    # Setup custom task path
    TASK_REGISTRY_PATH = os.path.abspath("./tasks")
    os.environ["EVAL_HARNESS_TASK_PATHS"] = TASK_REGISTRY_PATH

    print(f"🔧 Custom task path: {TASK_REGISTRY_PATH}")
    print(f"📁 Path exists: {os.path.exists(TASK_REGISTRY_PATH)}")

    return TASK_REGISTRY_PATH

def list_available_tasks():
    """List all available tasks, including custom tasks"""

    print("\n📋 Listing all available tasks...")

    try:
        # Create task manager instance
        task_manager = TaskManager()

        # Get all tasks
        all_tasks = task_manager.all_tasks
        print(f"Total available tasks: {len(all_tasks)}")

        # Find custom tasks
        custom_tasks = [task for task in all_tasks if 'my_mmlu_gen' in task]
        print(f"Custom tasks found: {custom_tasks}")

        # Display first 20 tasks as example
        print(f"\nFirst 20 tasks:")
        for i, task in enumerate(list(all_tasks)[:20]):
            print(f"  {i+1}. {task}")

        if len(all_tasks) > 20:
            print(f"  ... and {len(all_tasks) - 20} more tasks")

        return all_tasks, custom_tasks

    except Exception as e:
        print(f"❌ Error listing tasks: {e}")
        return [], []

def verify_custom_task():
    """Verify custom task registration is correct"""

    print("\n🔍 Verifying custom task registration...")

    try:
        # Create task manager and include custom path
        TASK_REGISTRY_PATH = os.path.abspath("./tasks")
        task_manager = TaskManager(include_path=TASK_REGISTRY_PATH)

        # Check custom tasks
        all_tasks = task_manager.all_tasks
        custom_tasks = [task for task in all_tasks if 'my_mmlu_gen' in task]

        if custom_tasks:
            print(f"✅ Found custom tasks: {custom_tasks}")

            # Try to get task details
            for task_name in custom_tasks:
                try:
                    task_dict = task_manager.get_task_dict([task_name])
                    print(f"📝 Task '{task_name}' details:")
                    print(f"   - Task type: {type(task_dict[task_name])}")
                    print(f"   - Task config available: {hasattr(task_dict[task_name], 'config')}")
                except Exception as e:
                    print(f"⚠️  Error getting task details for '{task_name}': {e}")
        else:
            print("❌ No custom tasks found")

            # Check if task file exists
            task_file = os.path.join(TASK_REGISTRY_PATH, "my_mmlu_gen", "my_mmlu_gen.yaml")
            if os.path.exists(task_file):
                print(f"📄 Task file exists: {task_file}")
                with open(task_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"📄 Task file content preview:\n{content[:200]}...")
            else:
                print(f"❌ Task file not found: {task_file}")

    except Exception as e:
        print(f"❌ Error verifying custom task: {e}")

def run_custom_task_evaluation():
    """Run custom task evaluation example"""

    print("\n🚀 Running custom task evaluation...")

    try:
        # Setup task path
        TASK_REGISTRY_PATH = os.path.abspath("./tasks")

        # Use simple_evaluate to run evaluation
        results = evaluator.simple_evaluate(
            model="hf",
            model_args="pretrained=gpt2,device=cpu",
            tasks=["my_mmlu_gen"],
            num_fewshot=0,
            batch_size=1,
            limit=5,  # Only run 5 samples for testing
            task_manager=TaskManager(include_path=TASK_REGISTRY_PATH)
        )

        print("✅ Evaluation completed!")
        print(f"📊 Results: {json.dumps(results['results'], indent=2)}")

    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        print("💡 This might be expected if the model or task configuration needs adjustment")

def main():
    """Main function"""

    print("🎯 EleutherAI LM Evaluation Harness - Custom Task Registration Example")
    print("=" * 60)

    # 1. Setup custom task path
    task_path = setup_custom_tasks()

    # 2. List all available tasks
    all_tasks, custom_tasks = list_available_tasks()

    # 3. Verify custom tasks
    verify_custom_task()

    # 4. Optional: Run custom task evaluation (commented out to avoid long runtime)
    # run_custom_task_evaluation()

    print("\n" + "=" * 60)
    print("✅ Custom task registration check completed!")

    if custom_tasks:
        print(f"🎉 Successfully found {len(custom_tasks)} custom tasks")
    else:
        print("⚠️  No custom tasks found, please check task configuration files")

if __name__ == "__main__":
    main()