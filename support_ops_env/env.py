"""
SupportOpsEnv — Main Environment

Implements the OpenEnv spec:
  - Observation  (Pydantic model)
  - Action       (Pydantic model)
  - Reward       (Pydantic model)
  - step(action) -> (Observation, Reward, done, info)
  - reset()      -> Observation
  - state()      -> dict
"""

import json
import copy
from typing import Any, Optional, Union
from pydantic import BaseModel, Field

# Task modules
from tasks.task_easy import (
    MOCK_EMAILS,
    EasyObservation,
    EasyAction,
    get_observation as easy_obs,
)
from tasks.task_medium import (
    MOCK_TICKETS,
    MediumObservation,
    MediumAction,
    get_observation as medium_obs,
)
from tasks.task_hard import (
    MOCK_COMPLAINTS,
    HardObservation,
    HardAction,
    get_observation as hard_obs,
)

# Grader modules
from graders.grader_easy import grade as grade_easy
from graders.grader_medium import grade as grade_medium
from graders.grader_hard import grade as grade_hard


# ─────────────────────────────────────────────────────────────────────────────
# OpenEnv Core Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """Unified observation returned by the environment."""
    task_name: str                          # "email_classification" | "ticket_prioritization" | "response_generation"
    task_difficulty: str                    # "easy" | "medium" | "hard"
    step_number: int
    sample_index: int
    total_samples: int
    payload: dict                           # task-specific observation fields
    done: bool = False
    metadata: dict = Field(default_factory=dict)


class Action(BaseModel):
    """Unified action submitted to the environment."""
    task_name: str                          # must match current task
    payload: dict                           # task-specific action fields


class Reward(BaseModel):
    """Reward signal returned after each step."""
    raw_score: float                        # grader output [0.0, 1.0]
    shaped_reward: float                    # dense reward after shaping
    penalties_applied: list[str] = Field(default_factory=list)
    bonuses_applied: list[str] = Field(default_factory=list)
    cumulative_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Task Configurations
# ─────────────────────────────────────────────────────────────────────────────

TASKS = [
    {
        "name": "email_classification",
        "difficulty": "easy",
        "samples": MOCK_EMAILS,
        "get_obs": easy_obs,
        "grade_fn": grade_easy,
        "action_keys": ["label"],
    },
    {
        "name": "ticket_prioritization",
        "difficulty": "medium",
        "samples": MOCK_TICKETS,
        "get_obs": medium_obs,
        "grade_fn": grade_medium,
        "action_keys": ["priority"],
    },
    {
        "name": "response_generation",
        "difficulty": "hard",
        "samples": MOCK_COMPLAINTS,
        "get_obs": hard_obs,
        "grade_fn": grade_hard,
        "action_keys": ["tone", "resolution_steps", "escalation"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SupportOpsEnv
# ─────────────────────────────────────────────────────────────────────────────

class SupportOpsEnv:
    """
    OpenEnv-compliant environment for customer support operations.

    Tasks (in order):
      1. Email Classification   (easy)
      2. Ticket Prioritization  (medium)
      3. Response Generation    (hard)

    Usage:
        env = SupportOpsEnv()
        obs = env.reset()
        while not obs.done:
            action = Action(task_name=obs.task_name, payload={...})
            obs, reward, done, info = env.step(action)
    """

    # Reward shaping constants
    INVALID_ACTION_PENALTY = -0.25
    REPEATED_ACTION_PENALTY = -0.15
    LONG_LOOP_PENALTY = -0.10
    MAX_STEPS_PER_TASK = 30        # prevent infinite loops
    LOOP_THRESHOLD = 5             # steps without progress = long loop

    def __init__(self):
        self._task_index: int = 0
        self._sample_index: int = 0
        self._step_number: int = 0
        self._cumulative_score: float = 0.0
        self._total_steps: int = 0
        self._scores: list[float] = []
        self._action_history: list[str] = []
        self._steps_without_progress: int = 0
        self._last_score: float = -999.0
        self._done: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment to the beginning of all tasks."""
        self._task_index = 0
        self._sample_index = 0
        self._step_number = 0
        self._cumulative_score = 0.0
        self._total_steps = 0
        self._scores = []
        self._action_history = []
        self._steps_without_progress = 0
        self._last_score = -999.0
        self._done = False
        return self._build_observation()

    def step(self, action: Union[Action, dict]) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action (or dict) with 'task_name' and 'payload'.

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() to start over.")

        # Accept dict for convenience
        if isinstance(action, dict):
            action = Action(**action)

        self._step_number += 1
        self._total_steps += 1

        current_task = TASKS[self._task_index]
        penalties = []
        bonuses = []

        # ── Validate task_name ────────────────────────────────────────────
        if action.task_name != current_task["name"]:
            reward = Reward(
                raw_score=0.0,
                shaped_reward=self.INVALID_ACTION_PENALTY,
                penalties_applied=[f"wrong_task_name: expected '{current_task['name']}'"],
                cumulative_score=self._cumulative_score,
            )
            obs = self._build_observation()
            return obs, reward, False, {"error": "wrong_task_name"}

        # ── Validate action payload keys ─────────────────────────────────
        required_keys = current_task["action_keys"]
        missing_keys = [k for k in required_keys if k not in action.payload]
        if missing_keys:
            reward = Reward(
                raw_score=0.0,
                shaped_reward=self.INVALID_ACTION_PENALTY,
                penalties_applied=[f"missing_keys: {missing_keys}"],
                cumulative_score=self._cumulative_score,
            )
            obs = self._build_observation()
            return obs, reward, False, {"error": f"missing_keys: {missing_keys}"}

        # ── Repetition detection ─────────────────────────────────────────
        action_fingerprint = json.dumps(action.payload, sort_keys=True)
        if action_fingerprint in self._action_history[-5:]:
            penalties.append("repeated_action")

        self._action_history.append(action_fingerprint)

        # ── Grade the action ──────────────────────────────────────────────
        sample = current_task["samples"][self._sample_index]
        grade_fn = current_task["grade_fn"]

        if current_task["difficulty"] == "easy":
            raw_score = grade_fn(
                predicted_label=action.payload.get("label", ""),
                ground_truth_label=sample["label"],
            )
        elif current_task["difficulty"] == "medium":
            raw_score = grade_fn(
                predicted_priority=action.payload.get("priority", ""),
                ground_truth_priority=sample["priority"],
            )
        elif current_task["difficulty"] == "hard":
            raw_score = grade_fn(
                predicted_action=action.payload,
                ground_truth=sample["reference_response"],
            )
        else:
            raw_score = 0.0

        # ── Shape reward ──────────────────────────────────────────────────
        shaped = raw_score

        if "repeated_action" in penalties:
            shaped += self.REPEATED_ACTION_PENALTY

        # Long loop detection (steps without score improvement)
        if raw_score <= self._last_score:
            self._steps_without_progress += 1
        else:
            self._steps_without_progress = 0
            if raw_score == 1.0:
                bonuses.append("perfect_score")
                shaped += 0.05  # small bonus for perfect score

        if self._steps_without_progress >= self.LOOP_THRESHOLD:
            shaped += self.LONG_LOOP_PENALTY
            penalties.append("long_loop_detected")

        # Max steps safeguard per task
        if self._step_number > self.MAX_STEPS_PER_TASK:
            shaped += self.INVALID_ACTION_PENALTY
            penalties.append("max_steps_exceeded")

        shaped = round(max(-1.0, min(1.05, shaped)), 4)
        self._last_score = raw_score

        # ── Accumulate score ──────────────────────────────────────────────
        self._scores.append(raw_score)
        self._cumulative_score = round(sum(self._scores) / len(self._scores), 4)

        reward = Reward(
            raw_score=round(raw_score, 4),
            shaped_reward=shaped,
            penalties_applied=penalties,
            bonuses_applied=bonuses,
            cumulative_score=self._cumulative_score,
        )

        # ── Advance to next sample / task ──────────────────────────────────
        self._sample_index += 1
        self._step_number = 0
        self._steps_without_progress = 0
        self._last_score = -999.0

        done = False
        if self._sample_index >= len(current_task["samples"]):
            self._sample_index = 0
            self._task_index += 1
            if self._task_index >= len(TASKS):
                done = True
                self._done = True

        obs = self._build_observation(done=done)
        info = {
            "sample_id": sample.get("id", ""),
            "raw_score": raw_score,
            "cumulative_score": self._cumulative_score,
        }

        return obs, reward, done, info

    def state(self) -> dict:
        """Return the current state of the environment."""
        if self._task_index >= len(TASKS):
            task_name = "completed"
            difficulty = "n/a"
            total_samples = 0
        else:
            current_task = TASKS[self._task_index]
            task_name = current_task["name"]
            difficulty = current_task["difficulty"]
            total_samples = len(current_task["samples"])

        return {
            "task_index": self._task_index,
            "task_name": task_name,
            "difficulty": difficulty,
            "sample_index": self._sample_index,
            "total_samples": total_samples,
            "step_number": self._step_number,
            "total_steps": self._total_steps,
            "cumulative_score": self._cumulative_score,
            "scores": copy.copy(self._scores),
            "done": self._done,
        }

    # ── Convenience methods ───────────────────────────────────────────────────

    def final_score(self) -> float:
        """Return normalized final score across all tasks."""
        if not self._scores:
            return 0.0
        return round(sum(self._scores) / len(self._scores), 4)

    def task_scores(self) -> dict:
        """Return per-task average scores."""
        result = {}
        offset = 0
        for task in TASKS:
            n = len(task["samples"])
            task_scores_ = self._scores[offset: offset + n]
            if task_scores_:
                result[task["name"]] = round(sum(task_scores_) / len(task_scores_), 4)
            else:
                result[task["name"]] = None
            offset += n
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_observation(self, done: bool = False) -> Observation:
        """Build the current Observation object."""
        if self._task_index >= len(TASKS) or done:
            return Observation(
                task_name="completed",
                task_difficulty="n/a",
                step_number=self._step_number,
                sample_index=self._sample_index,
                total_samples=0,
                payload={},
                done=True,
                metadata={"final_score": self.final_score()},
            )

        current_task = TASKS[self._task_index]
        sample = current_task["samples"][self._sample_index]
        task_obs = current_task["get_obs"](sample)

        return Observation(
            task_name=current_task["name"],
            task_difficulty=current_task["difficulty"],
            step_number=self._step_number,
            sample_index=self._sample_index,
            total_samples=len(current_task["samples"]),
            payload=task_obs.model_dump(),
            done=False,
            metadata={
                "cumulative_score": self._cumulative_score,
                "task_index": self._task_index,
            },
        )
