"""
Baseline Inference Script for SupportOpsEnv
Meta OpenEnv Hackathon Submission

Uses an OpenAI-compatible client to run all three tasks and prints structured logs.

Environment variables:
  - API_BASE_URL    : OpenAI-compatible base URL  (default: "https://api-inference.huggingface.co/v1")
  - MODEL_NAME      : model identifier            (default: "meta-llama/Llama-3.3-70B-Instruct")
  - HF_TOKEN        : HuggingFace / API token     (NO default — must be set by caller)
  - LOCAL_IMAGE_NAME: (optional) Docker image name when using from_docker_image()

Usage:
  export API_BASE_URL="https://api-inference.huggingface.co/v1"
  export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
  export HF_TOKEN="hf_..."
  python inference.py
"""

import os
import sys
import json
import time
import traceback
from typing import Optional

from openai import OpenAI

# Add project root to path so tasks / graders / env are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SupportOpsEnv, Action


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (checklist requirement: only API_BASE_URL and MODEL_NAME have
#                 defaults; HF_TOKEN must NOT have a default value)
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL     = os.environ.get("API_BASE_URL",     "https://api-inference.huggingface.co/v1")
MODEL_NAME       = os.environ.get("MODEL_NAME",       "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN         = os.environ.get("HF_TOKEN")          # NO default — intentional
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")  # optional, used with from_docker_image()

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Client  (checklist requirement: all LLM calls must use OpenAI client)
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Structured Logger  (checklist requirement: stdout must follow START/STEP/END)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(env_name: str, model: str, base_url: str) -> None:
    print(f"[START] task={env_name} model={model} base_url={base_url}", flush=True)


def log_step(
    step: int,
    task: str,
    difficulty: str,
    sample: int,
    total: int,
    action: dict,
    raw_score: float,
    shaped_reward: float,
    cumulative: float,
    penalties: list,
    bonuses: list,
) -> None:
    print(
        f"[STEP] step={step} task={task} difficulty={difficulty} "
        f"sample={sample}/{total} reward={round(shaped_reward, 4)} "
        f"raw_score={round(raw_score, 4)} cumulative={round(cumulative, 4)}",
        flush=True,
    )


def log_end(task_scores: dict, overall_score: float, total_steps: int) -> None:
    scores_str = " ".join(f"{k}={v}" for k, v in task_scores.items())
    print(
        f"[END] {scores_str} score={round(overall_score, 4)} steps={total_steps}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_easy_prompt(payload: dict) -> str:
    return f"""You are a customer support email classification expert.

Classify the following customer email into exactly one of these categories:
- billing
- technical
- general

Email:
{payload["email_text"]}

Respond with ONLY the category label (billing, technical, or general). No explanation."""


def build_medium_prompt(payload: dict) -> str:
    signals = ", ".join(payload.get("urgency_signals", []))
    return f"""You are a customer support ticket prioritization expert.

Assign a priority level to the following support ticket.
Priority options: low, medium, high

Urgency signals detected: {signals}

Ticket:
{payload["ticket_text"]}

RULES:
- Return ONLY the priority label: low, medium, or high
- Never underestimate high-priority issues (security, outages, revenue impact)
- No explanation, just the label."""


def build_hard_prompt(payload: dict) -> str:
    return f"""You are an expert customer support agent.

Generate a structured response to the following customer complaint.

Complaint:
{payload["complaint_text"]}

Return your response as a valid JSON object with EXACTLY these fields:
{{
  "tone": "<one of: empathetic, formal, apologetic, assertive>",
  "resolution_steps": "<numbered list of actionable steps as a single string>",
  "escalation": <true or false>
}}

RULES:
- tone must be exactly one of: empathetic, formal, apologetic, assertive
- resolution_steps must be a string with clear numbered steps (minimum 3 steps)
- escalation must be true if a human agent is needed, false otherwise
- Return ONLY the JSON object — no markdown, no extra text."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Caller  (uses the OpenAI client configured above)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(prompt: str, max_retries: int = 3) -> Optional[str]:
    """Call the model via OpenAI-compatible client and return the response text."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful customer support AI assistant. "
                            "Follow all instructions precisely and return only what is asked."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            sys.stderr.write(f"[LLM Error attempt {attempt + 1}/{max_retries}]: {e}\n")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Action Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_easy_action(llm_output: str, task_name: str) -> Action:
    label = llm_output.strip().lower().split()[0] if llm_output else "general"
    label = label.rstrip(".,;:")
    if label not in {"billing", "technical", "general"}:
        label = "general"
    return Action(task_name=task_name, payload={"label": label})


def parse_medium_action(llm_output: str, task_name: str) -> Action:
    priority = llm_output.strip().lower().split()[0] if llm_output else "medium"
    priority = priority.rstrip(".,;:")
    if priority not in {"low", "medium", "high"}:
        priority = "medium"
    return Action(task_name=task_name, payload={"priority": priority})


def parse_hard_action(llm_output: str, task_name: str) -> Action:
    """Parse JSON response from LLM for the hard task."""
    default_payload = {
        "tone": "empathetic",
        "resolution_steps": (
            "1. Acknowledge the issue and sincerely apologize. "
            "2. Investigate the root cause immediately. "
            "3. Provide a clear resolution or workaround. "
            "4. Follow up within 24 hours to confirm resolution."
        ),
        "escalation": False,
    }
    if not llm_output:
        return Action(task_name=task_name, payload=default_payload)

    # Strip markdown code fences if present
    text = llm_output.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(text)
        tone = str(parsed.get("tone", "empathetic")).strip().lower()
        steps = str(parsed.get("resolution_steps", default_payload["resolution_steps"])).strip()
        escalation = bool(parsed.get("escalation", False))

        if tone not in {"empathetic", "formal", "apologetic", "assertive"}:
            tone = "empathetic"

        return Action(task_name=task_name, payload={
            "tone": tone,
            "resolution_steps": steps,
            "escalation": escalation,
        })
    except (json.JSONDecodeError, KeyError, TypeError):
        return Action(task_name=task_name, payload=default_payload)


# ─────────────────────────────────────────────────────────────────────────────
# Task Dispatch Tables
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_BUILDERS = {
    "email_classification":  build_easy_prompt,
    "ticket_prioritization": build_medium_prompt,
    "response_generation":   build_hard_prompt,
}

ACTION_PARSERS = {
    "email_classification":  parse_easy_action,
    "ticket_prioritization": parse_medium_action,
    "response_generation":   parse_hard_action,
}

DEFAULT_ACTIONS = {
    "email_classification":  {"label": "general"},
    "ticket_prioritization": {"priority": "medium"},
    "response_generation": {
        "tone": "empathetic",
        "resolution_steps": (
            "1. Acknowledge the issue. "
            "2. Investigate and identify root cause. "
            "3. Provide resolution or escalate. "
            "4. Follow up with customer."
        ),
        "escalation": False,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference() -> float:
    # ── START log (required structured format) ────────────────────────────────
    log_start(
        env_name="SupportOpsEnv",
        model=MODEL_NAME,
        base_url=API_BASE_URL,
    )

    env = SupportOpsEnv()
    obs = env.reset()

    step_count = 0

    while not obs.done:
        task_name  = obs.task_name
        difficulty = obs.task_difficulty
        sample_idx = obs.sample_index
        total      = obs.total_samples
        payload    = obs.payload

        # Build prompt
        prompt_builder = PROMPT_BUILDERS.get(task_name)
        if not prompt_builder:
            sys.stderr.write(f"[SKIP] Unknown task: {task_name}\n")
            break

        prompt = prompt_builder(payload)

        # Call LLM via OpenAI client
        llm_output = call_llm(prompt)

        # Parse action
        parser = ACTION_PARSERS.get(task_name)
        if llm_output and parser:
            action = parser(llm_output, task_name)
        else:
            action = Action(
                task_name=task_name,
                payload=DEFAULT_ACTIONS[task_name],
            )

        # Step environment
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            sys.stderr.write(f"[ENV ERROR]: {e}\n")
            traceback.print_exc(file=sys.stderr)
            break

        step_count += 1

        # ── STEP log (required structured format) ─────────────────────────────
        log_step(
            step=step_count,
            task=task_name,
            difficulty=difficulty,
            sample=sample_idx + 1,
            total=total,
            action=action.payload,
            raw_score=reward.raw_score,
            shaped_reward=reward.shaped_reward,
            cumulative=reward.cumulative_score,
            penalties=reward.penalties_applied,
            bonuses=reward.bonuses_applied,
        )

    # ── END log (required structured format) ──────────────────────────────────
    log_end(
        task_scores=env.task_scores(),
        overall_score=env.final_score(),
        total_steps=step_count,
    )

    return env.final_score()


if __name__ == "__main__":
    score = run_inference()
    sys.exit(0 if score >= 0 else 1)
