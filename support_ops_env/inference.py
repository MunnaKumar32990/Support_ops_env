"""
Baseline Inference Script for SupportOpsEnv

Uses an OpenAI-compatible client to run all three tasks and print final scores.

Environment variables:
  - API_BASE_URL   : OpenAI-compatible base URL
  - MODEL_NAME     : model to use (e.g. "meta-llama/Llama-3-8b-instruct")
  - HF_TOKEN       : HuggingFace token (used as API key if applicable)

Usage:
  export API_BASE_URL="https://api-inference.huggingface.co/v1"
  export MODEL_NAME="meta-llama/Llama-3-8b-instruct"
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SupportOpsEnv, Action


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))

if not HF_TOKEN:
    print("[WARNING] No API token found. Set HF_TOKEN or OPENAI_API_KEY.")


# ─────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────

client = OpenAI(
    api_key=HF_TOKEN or "dummy",
    base_url=API_BASE_URL,
)


# ─────────────────────────────────────────────
# Prompt Builders
# ─────────────────────────────────────────────

def build_easy_prompt(payload: dict) -> str:
    return f"""You are a customer support email classification expert.

Classify the following customer email into exactly one of these categories:
- billing
- technical
- general

Email:
{payload['email_text']}

Respond with ONLY the category label (billing, technical, or general). No explanation."""


def build_medium_prompt(payload: dict) -> str:
    signals = ", ".join(payload.get("urgency_signals", []))
    return f"""You are a customer support ticket prioritization expert.

Assign a priority level to the following support ticket.
Priority options: low, medium, high

Urgency signals detected: {signals}

Ticket:
{payload['ticket_text']}

RULES:
- Return ONLY the priority label: low, medium, or high
- Be careful not to underestimate high-priority issues (security vulnerabilities, system outages, revenue impact)
- No explanation, just the label."""


def build_hard_prompt(payload: dict) -> str:
    return f"""You are an expert customer support agent.

Generate a structured response to the following customer complaint.

Complaint:
{payload['complaint_text']}

Return your response as a valid JSON object with EXACTLY these fields:
{{
  "tone": "<one of: empathetic, formal, apologetic, assertive>",
  "resolution_steps": "<numbered list of actionable resolution steps as a string>",
  "escalation": <true or false>
}}

RULES:
- tone must be exactly one of: empathetic, formal, apologetic, assertive
- resolution_steps must be a string with clear numbered steps
- escalation must be true if a human agent should handle this, false otherwise
- Do NOT include any text outside the JSON object"""


# ─────────────────────────────────────────────
# LLM Caller
# ─────────────────────────────────────────────

def call_llm(prompt: str, max_retries: int = 3) -> Optional[str]:
    """Call the LLM and return the response text."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful customer support AI assistant. Follow instructions precisely.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [LLM Error attempt {attempt+1}/{max_retries}]: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


# ─────────────────────────────────────────────
# Action Parsers
# ─────────────────────────────────────────────

def parse_easy_action(llm_output: str, task_name: str) -> Action:
    label = llm_output.strip().lower().split()[0] if llm_output else "general"
    # Clean punctuation
    label = label.rstrip(".,;:")
    valid = {"billing", "technical", "general"}
    if label not in valid:
        label = "general"
    return Action(task_name=task_name, payload={"label": label})


def parse_medium_action(llm_output: str, task_name: str) -> Action:
    priority = llm_output.strip().lower().split()[0] if llm_output else "medium"
    priority = priority.rstrip(".,;:")
    valid = {"low", "medium", "high"}
    if priority not in valid:
        priority = "medium"
    return Action(task_name=task_name, payload={"priority": priority})


def parse_hard_action(llm_output: str, task_name: str) -> Action:
    """Parse JSON response from LLM for the hard task."""
    default = {
        "tone": "empathetic",
        "resolution_steps": "1. Acknowledge the issue. 2. Investigate and resolve. 3. Follow up with customer.",
        "escalation": False,
    }
    if not llm_output:
        return Action(task_name=task_name, payload=default)

    # Extract JSON from output (handle markdown code blocks)
    text = llm_output.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(text)
        tone = str(parsed.get("tone", "empathetic")).strip().lower()
        steps = str(parsed.get("resolution_steps", default["resolution_steps"])).strip()
        escalation = bool(parsed.get("escalation", False))

        valid_tones = {"empathetic", "formal", "apologetic", "assertive"}
        if tone not in valid_tones:
            tone = "empathetic"

        return Action(task_name=task_name, payload={
            "tone": tone,
            "resolution_steps": steps,
            "escalation": escalation,
        })
    except (json.JSONDecodeError, KeyError, TypeError):
        return Action(task_name=task_name, payload=default)


# ─────────────────────────────────────────────
# Task Dispatch
# ─────────────────────────────────────────────

PROMPT_BUILDERS = {
    "email_classification":   build_easy_prompt,
    "ticket_prioritization":  build_medium_prompt,
    "response_generation":    build_hard_prompt,
}

ACTION_PARSERS = {
    "email_classification":   parse_easy_action,
    "ticket_prioritization":  parse_medium_action,
    "response_generation":    parse_hard_action,
}


# ─────────────────────────────────────────────
# Main Inference Loop
# ─────────────────────────────────────────────

def run_inference():
    print("=" * 65)
    print("  SupportOpsEnv — Baseline Inference")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Base URL : {API_BASE_URL}")
    print("=" * 65)

    env = SupportOpsEnv()
    obs = env.reset()

    task_results: dict[str, list[float]] = {}
    step_count = 0

    while not obs.done:
        task_name  = obs.task_name
        difficulty = obs.task_difficulty
        sample_idx = obs.sample_index
        total      = obs.total_samples
        payload    = obs.payload

        print(f"\n[Task: {task_name} | {difficulty.upper()} | Sample {sample_idx+1}/{total}]")

        # Build prompt
        prompt_builder = PROMPT_BUILDERS.get(task_name)
        if not prompt_builder:
            print(f"  [SKIP] Unknown task: {task_name}")
            break

        prompt = prompt_builder(payload)

        # Call LLM
        llm_output = call_llm(prompt)
        print(f"  LLM Output: {repr(llm_output[:100]) if llm_output else '[None]'}")

        # Parse action
        parser = ACTION_PARSERS.get(task_name)
        if llm_output and parser:
            action = parser(llm_output, task_name)
        else:
            # Fallback default action
            if task_name == "email_classification":
                action = Action(task_name=task_name, payload={"label": "general"})
            elif task_name == "ticket_prioritization":
                action = Action(task_name=task_name, payload={"priority": "medium"})
            else:
                action = Action(task_name=task_name, payload={
                    "tone": "empathetic",
                    "resolution_steps": "We are looking into your issue and will respond shortly.",
                    "escalation": False,
                })

        # Step environment
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            print(f"  [ENV ERROR]: {e}")
            traceback.print_exc()
            break

        step_count += 1
        print(f"  Raw Score     : {reward.raw_score:.4f}")
        print(f"  Shaped Reward : {reward.shaped_reward:.4f}")
        if reward.penalties_applied:
            print(f"  Penalties     : {reward.penalties_applied}")
        if reward.bonuses_applied:
            print(f"  Bonuses       : {reward.bonuses_applied}")
        print(f"  Cumulative    : {reward.cumulative_score:.4f}")

        # Collect per-task results
        if task_name not in task_results:
            task_results[task_name] = []
        task_results[task_name].append(reward.raw_score)

    # ── Final Results ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)

    task_score_map = env.task_scores()
    overall_score = env.final_score()

    for task_name, avg in task_score_map.items():
        difficulty_map = {
            "email_classification":  "Easy",
            "ticket_prioritization": "Medium",
            "response_generation":   "Hard",
        }
        diff = difficulty_map.get(task_name, "")
        score_display = f"{avg:.4f}" if avg is not None else "N/A"
        print(f"  {task_name:<30} [{diff:<6}]: {score_display}")

    print("-" * 65)
    print(f"  OVERALL SCORE : {overall_score:.4f}  ({overall_score*100:.1f}%)")
    print(f"  Total Steps   : {step_count}")
    print("=" * 65)

    return overall_score


if __name__ == "__main__":
    score = run_inference()
    sys.exit(0 if score >= 0 else 1)
