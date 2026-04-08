"""
smoke_test.py — Quick local validation without any external API calls.

Tests:
  1. Environment reset returns a valid Observation
  2. All 28 samples can be stepped through with mock actions
  3. Graders return [0.0, 1.0] scores
  4. env.state() reflects correct progress
  5. env.final_score() returns a float
  6. done=True after all samples
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SupportOpsEnv, Action, Observation, Reward


def mock_action_for(obs: Observation) -> Action:
    """Generate a valid mock action for each task."""
    task = obs.task_name

    if task == "email_classification":
        return Action(task_name=task, payload={"label": "billing"})

    elif task == "ticket_prioritization":
        return Action(task_name=task, payload={"priority": "high"})

    elif task == "response_generation":
        return Action(task_name=task, payload={
            "tone": "empathetic",
            "resolution_steps": (
                "1. Acknowledge the issue and apologize sincerely. "
                "2. Investigate the root cause immediately. "
                "3. Provide a resolution or workaround. "
                "4. Follow up within 24 hours."
            ),
            "escalation": True,
        })

    raise ValueError(f"Unknown task: {task}")


def run_smoke_test():
    print("=" * 55)
    print("  SupportOpsEnv — Smoke Test")
    print("=" * 55)

    env = SupportOpsEnv()

    # ── Test 1: Reset ─────────────────────────────────────────────
    obs = env.reset()
    assert isinstance(obs, Observation), "reset() must return Observation"
    assert obs.task_name == "email_classification", f"Expected email_classification, got {obs.task_name}"
    assert not obs.done, "Initial obs should not be done"
    print("✅ Test 1 PASS: reset() returns valid Observation")

    # ── Test 2: Full episode with mock actions ────────────────────
    step_count = 0
    scores = []
    current_tasks = []

    while not obs.done:
        task_name = obs.task_name
        if not current_tasks or current_tasks[-1] != task_name:
            current_tasks.append(task_name)

        action = mock_action_for(obs)
        obs, reward, done, info = env.step(action)

        assert isinstance(reward, Reward), "step() must return Reward"
        assert 0.0 <= reward.raw_score <= 1.0, f"raw_score out of range: {reward.raw_score}"
        assert isinstance(reward.shaped_reward, float), "shaped_reward must be float"

        scores.append(reward.raw_score)
        step_count += 1

        if step_count % 5 == 0:
            print(f"   ... {step_count} steps done, current cumulative: {reward.cumulative_score:.4f}")

    print(f"✅ Test 2 PASS: Completed {step_count} steps across {len(current_tasks)} tasks")

    # ── Test 3: done flag ─────────────────────────────────────────
    assert obs.done, "Final obs should be done=True"
    assert done, "done flag should be True at end"
    print("✅ Test 3 PASS: done=True after all samples")

    # ── Test 4: Tasks visited ─────────────────────────────────────
    expected_tasks = ["email_classification", "ticket_prioritization", "response_generation"]
    assert current_tasks == expected_tasks, f"Expected tasks {expected_tasks}, got {current_tasks}"
    print(f"✅ Test 4 PASS: All 3 tasks visited in order")

    # ── Test 5: Scores in range ───────────────────────────────────
    out_of_range = [s for s in scores if not (0.0 <= s <= 1.0)]
    assert not out_of_range, f"Out-of-range scores: {out_of_range}"
    print(f"✅ Test 5 PASS: All {len(scores)} scores in [0.0, 1.0]")

    # ── Test 6: Final score ───────────────────────────────────────
    final = env.final_score()
    assert isinstance(final, float), f"final_score() must return float, got {type(final)}"
    assert 0.0 <= final <= 1.0, f"final_score out of range: {final}"
    print(f"✅ Test 6 PASS: final_score() = {final:.4f}")

    # ── Test 7: Task scores ───────────────────────────────────────
    task_scores = env.task_scores()
    for t in expected_tasks:
        assert t in task_scores, f"Missing task in task_scores: {t}"
        assert isinstance(task_scores[t], float), f"task_score for {t} must be float"
    print(f"✅ Test 7 PASS: task_scores() = {task_scores}")

    # ── Test 8: State dict ────────────────────────────────────────
    state = env.state()
    assert isinstance(state, dict), "state() must return dict"
    required_keys = ["task_index", "task_name", "sample_index", "total_steps", "done"]
    for k in required_keys:
        assert k in state, f"Missing key in state(): {k}"
    print(f"✅ Test 8 PASS: state() returns valid dict with required keys")

    # ── Test 9: Graders ───────────────────────────────────────────
    from graders.grader_easy import grade as ge
    from graders.grader_medium import grade as gm
    from graders.grader_hard import grade as gh

    assert ge("billing", "billing") == 1.0
    assert ge("general", "billing") == 0.5
    assert ge("technical", "billing") == 0.0
    assert ge("INVALID", "billing") == -0.2
    print("✅ Test 9a PASS: grader_easy correct")

    assert gm("high", "high") == 1.0
    assert gm("low", "high") < 0.0       # underestimate penalty
    assert gm("medium", "medium") == 1.0
    assert gm("INVALID", "medium") == -0.2
    print("✅ Test 9b PASS: grader_medium correct")

    hard_perfect = {
        "tone": "apologetic",
        "resolution_steps": "1. Refund. 2. Apologize. 3. Follow up. 4. Close ticket.",
        "escalation": True,
    }
    hard_ref = {
        "tone": "apologetic",
        "resolution_steps": "1. Apologize. 2. Refund. 3. Confirm.",
        "escalation": True,
    }
    s = gh(hard_perfect, hard_ref)
    assert 0.0 <= s <= 1.0, f"hard grader out of range: {s}"
    print(f"✅ Test 9c PASS: grader_hard = {s:.4f}")

    # ── Test 10: Invalid action handling ─────────────────────────
    env2 = SupportOpsEnv()
    env2.reset()
    _, reward2, _, info2 = env2.step(Action(
        task_name="ticket_prioritization",  # wrong task (should be email_classification)
        payload={"priority": "high"}
    ))
    assert reward2.shaped_reward < 0, "Invalid action must yield negative shaped_reward"
    assert "wrong_task_name" in info2.get("error", "")
    print("✅ Test 10 PASS: Invalid (wrong task_name) action penalized correctly")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  ALL TESTS PASSED ✅")
    print(f"  Total samples : {step_count}")
    print(f"  Final score   : {final:.4f} ({final*100:.1f}%)")
    print(f"  Per-task      : {task_scores}")
    print("=" * 55)

    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
