#!/usr/bin/env python3
"""Comprehensive validation for OpenEnv submission."""

import yaml
import importlib.util
import sys

def validate_openenv_yaml():
    print("=" * 60)
    print("VALIDATING openenv.yaml")
    print("=" * 60)
    
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    assert "tasks" in config, "Missing 'tasks' key"
    tasks = config["tasks"]
    assert len(tasks) >= 3, f"Need at least 3 tasks, found {len(tasks)}"
    
    print(f"[OK] Found {len(tasks)} tasks")
    
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}:")
        print(f"  - id: {task.get('id')}")
        print(f"  - name: {task.get('name')}")
        print(f"  - difficulty: {task.get('difficulty')}")
        print(f"  - grader: {task.get('grader')}")
        
        assert "id" in task, f"Task {i+1} missing 'id'"
        assert "grader" in task, f"Task {i+1} missing 'grader'"
    
    print("\n[OK] openenv.yaml structure valid")
    return tasks

def validate_graders(tasks):
    print("\n" + "=" * 60)
    print("VALIDATING GRADERS")
    print("=" * 60)
    
    for task in tasks:
        grader_path = task["grader"]
        module_path, func_name = grader_path.split(":")
        
        print(f"\nTesting {task['id']} grader: {grader_path}")
        
        module = importlib.import_module(module_path.replace("/", "."))
        grader_func = getattr(module, func_name)
        
        if task["id"] == "task_easy":
            score = grader_func("billing", "billing")
        elif task["id"] == "task_medium":
            score = grader_func("high", "high")
        elif task["id"] == "task_hard":
            pred = {"tone": "apologetic", "resolution_steps": "1. Apologize 2. Refund", "escalation": True}
            gt = {"tone": "apologetic", "resolution_steps": "1. Apologize 2. Refund", "escalation": True}
            score = grader_func(pred, gt)
        
        print(f"  Score: {score}")
        assert isinstance(score, (int, float)), f"Grader must return number, got {type(score)}"
        assert 0.0 < score < 1.0, f"Score must be in (0, 1), got {score}"
        print(f"  [OK] Grader returns valid score")
    
    print("\n[OK] All graders validated")

def main():
    print("\n" + "=" * 60)
    print("OPENENV VALIDATION SCRIPT")
    print("=" * 60 + "\n")
    
    try:
        tasks = validate_openenv_yaml()
        validate_graders(tasks)
        
        print("\n" + "=" * 60)
        print("VALIDATION PASSED")
        print("=" * 60)
        print(f"\n{len(tasks)} tasks with graders detected")
        print("Ready for OpenEnv submission!")
        
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
