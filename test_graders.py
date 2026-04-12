#!/usr/bin/env python3
"""Test script to verify all graders work correctly."""

from graders.grader_easy import grade as grade_easy
from graders.grader_medium import grade as grade_medium
from graders.grader_hard import grade as grade_hard

def test_grader_easy():
    score = grade_easy("billing", "billing")
    assert 0.0 < score < 1.0, f"Easy grader returned {score}, must be in (0, 1)"
    print(f"[OK] Easy grader: {score}")

def test_grader_medium():
    score = grade_medium("high", "high")
    assert 0.0 < score < 1.0, f"Medium grader returned {score}, must be in (0, 1)"
    print(f"[OK] Medium grader: {score}")

def test_grader_hard():
    pred = {"tone": "apologetic", "resolution_steps": "1. Apologize 2. Refund 3. Confirm", "escalation": True}
    gt = {"tone": "apologetic", "resolution_steps": "1. Apologize 2. Refund 3. Confirm", "escalation": True}
    score = grade_hard(pred, gt)
    assert 0.0 < score < 1.0, f"Hard grader returned {score}, must be in (0, 1)"
    print(f"[OK] Hard grader: {score}")

if __name__ == "__main__":
    test_grader_easy()
    test_grader_medium()
    test_grader_hard()
    print("\n[OK] All graders pass validation!")
