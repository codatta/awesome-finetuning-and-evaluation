#!/usr/bin/env python3
"""
Simplified custom task registration test script
"""

import os
import sys
from lm_eval.tasks import TaskManager

def test_task_registration():
    """Test custom task registration"""

    print("🔧 Testing custom task registration...")

    # Setup custom task path
    TASK_REGISTRY_PATH = os.path.abspath("./tasks")
    print(f"📁 Task path: {TASK_REGISTRY_PATH}")
    print(f"📁 Path exists: {os.path.exists(TASK_REGISTRY_PATH)}")

    # Set environment variable
    os.environ["EVAL_HARNESS_TASK_PATHS"] = TASK_REGISTRY_PATH

    try:
        # Method 1: Using include_path parameter
        print("\n🔍 Method 1: Using include_path parameter")
        task_manager = TaskManager(include_path=TASK_REGISTRY_PATH)
        all_tasks = task_manager.all_tasks
        custom_tasks = [task for task in all_tasks if 'my_mmlu_gen' in task]
        print(f"Found custom tasks: {custom_tasks}")

        # Method 2: Using environment variable
        print("\n🔍 Method 2: Using environment variable")
        task_manager2 = TaskManager()
        all_tasks2 = task_manager2.all_tasks
        custom_tasks2 = [task for task in all_tasks2 if 'my_mmlu_gen' in task]
        print(f"Found custom tasks: {custom_tasks2}")

        # Method 3: Direct task loading
        print("\n🔍 Method 3: Direct task loading")
        try:
            task_dict = task_manager.get_task_dict(["my_mmlu_gen"])
            print(f"✅ Successfully loaded task: {list(task_dict.keys())}")
        except Exception as e:
            print(f"❌ Failed to load task: {e}")

        # Check task files
        print("\n📄 Checking task files:")
        task_file = os.path.join(TASK_REGISTRY_PATH, "my_mmlu_gen", "my_mmlu_gen.yaml")
        print(f"Task file: {task_file}")
        print(f"File exists: {os.path.exists(task_file)}")

        if os.path.exists(task_file):
            with open(task_file, 'r') as f:
                content = f.read()
                print(f"File content:\n{content}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_task_registration()