# SupportOpsEnv

> **Meta OpenEnv Hackathon Submission**
> A production-ready, OpenEnv-compliant environment simulating real-world customer support operations.

---

## Overview

**SupportOpsEnv** challenges AI agents across three progressively harder tasks drawn from real customer support workflows:

| Task | Difficulty | Input | Output |
|------|-----------|-------|--------|
| Email Classification | 🟢 Easy | Customer email text | `billing` / `technical` / `general` |
| Ticket Prioritization | 🟡 Medium | Support ticket + urgency signals | `low` / `medium` / `high` |
| Response Generation | 🔴 Hard | Customer complaint | `{tone, resolution_steps, escalation}` |

---

## File Structure

```
support_ops_env/
├── env.py              # Main environment (SupportOpsEnv class)
├── tasks/
│   ├── task_easy.py    # Email classification task & mock data
│   ├── task_medium.py  # Ticket prioritization task & mock data
│   └── task_hard.py    # Response generation task & mock data
├── graders/
│   ├── grader_easy.py   # Deterministic email grader
│   ├── grader_medium.py # Proportional ticket grader
│   └── grader_hard.py   # Multi-component response grader
├── inference.py        # Baseline inference script (OpenAI client)
├── app.py              # FastAPI server (HuggingFace Spaces compatible)
├── openenv.yaml        # OpenEnv spec manifest
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker container definition
└── README.md           # This file
```

---

## Task Descriptions

### Task 1 — Email Classification (Easy)

**Input Observation:**
```json
{
  "task_id": "task_easy",
  "email_id": "email_001",
  "email_text": "Hi, I was charged twice for my subscription...",
  "valid_labels": ["billing", "technical", "general"],
  "instruction": "Classify the following customer email..."
}
```

**Action:**
```json
{ "label": "billing" }
```

**Reward:**
- `1.0` — correct classification
- `0.5` — semantically close (billing↔general, technical↔general)
- `0.0` — wrong label
- `-0.2` — invalid label (not in valid set)

---

### Task 2 — Ticket Prioritization (Medium)

**Input Observation:**
```json
{
  "task_id": "task_medium",
  "ticket_id": "ticket_001",
  "ticket_text": "Our entire production system is down...",
  "urgency_signals": ["production down", "all users affected", "revenue loss"],
  "valid_priorities": ["low", "medium", "high"]
}
```

**Action:**
```json
{ "priority": "high" }
```

**Reward:**
- Proportional to distance: `1.0 - (|pred - gt| / 2)`
- Additional `-0.3` penalty for underestimating `high` priority tickets
- `-0.2` for invalid priority label

---

### Task 3 — Response Generation (Hard)

**Input Observation:**
```json
{
  "task_id": "task_hard",
  "complaint_id": "complaint_001",
  "complaint_text": "I've been charged three times for the same order...",
  "valid_tones": ["empathetic", "formal", "apologetic", "assertive"]
}
```

**Action:**
```json
{
  "tone": "apologetic",
  "resolution_steps": "1. Sincerely apologize. 2. Initiate refund. 3. Send confirmation.",
  "escalation": true
}
```

**Reward breakdown (max 1.0):**
| Component | Weight |
|-----------|--------|
| Structure correctness (valid tone + non-empty steps + bool escalation) | 0.30 |
| Tone matches reference | 0.20 |
| Escalation matches reference | 0.20 |
| Semantic similarity (keyword F1 vs reference steps) | 0.30 |
| **Penalties** | |
| Unsafe / harmful content | -0.30 |
| Resolution steps too short (<20 chars) | -0.10 |

---

## OpenEnv API

### `reset() -> Observation`
Resets the environment to the start of Task 1, Sample 1.

### `step(action) -> (Observation, Reward, done, info)`
Executes one action. Advances to the next sample/task automatically.

### `state() -> dict`
Returns the current environment state (task index, sample index, scores, etc.).

---

## Action / Observation Schema

All schemas are defined as **Pydantic v2 models** in `env.py`.

### Observation
```python
class Observation(BaseModel):
    task_name: str
    task_difficulty: str
    step_number: int
    sample_index: int
    total_samples: int
    payload: dict      # task-specific fields
    done: bool
    metadata: dict
```

### Action
```python
class Action(BaseModel):
    task_name: str
    payload: dict      # task-specific action fields
```

### Reward
```python
class Reward(BaseModel):
    raw_score: float
    shaped_reward: float
    penalties_applied: list[str]
    bonuses_applied: list[str]
    cumulative_score: float
```

---

## Dense Reward System

The environment provides **dense rewards** (not just terminal):

| Event | Shaped Reward |
|-------|-------------|
| Valid correct action | `raw_score` (0.0–1.0) |
| Perfect score bonus | +0.05 |
| Invalid action / wrong task name | -0.25 |
| Repeated action (same payload ×5) | -0.15 |
| Long loop (≥5 steps without improvement) | -0.10 |
| Max steps exceeded (>30/task) | -0.25 |

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- pip

### Local Installation

```bash
cd support_ops_env
pip install -r requirements.txt
```

### Run Inference (Baseline)

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."       # or OPENAI_API_KEY

python inference.py
```

### Run the API Server

```bash
python app.py
# Server starts at http://localhost:7860
# Docs at http://localhost:7860/docs
```

---

## Docker Usage

### Build

```bash
docker build -t support-ops-env .
```

### Run (API server)

```bash
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  support-ops-env
```

### Run Inference in Docker

```bash
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  support-ops-env python inference.py
```

---

## Baseline Scores

Measured using `gpt-4o-mini` on all mock samples:

| Task | Samples | Expected Score |
|------|---------|---------------|
| Email Classification (Easy) | 10 | ~0.90 |
| Ticket Prioritization (Medium) | 10 | ~0.78 |
| Response Generation (Hard) | 8 | ~0.65 |
| **Overall** | **28** | **~0.78** |

---

## REST API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | List all task descriptions |
| `/reset` | POST | Reset environment, returns first Observation |
| `/step` | POST | Execute action, returns Observation + Reward |
| `/state` | GET | Current environment state |
| `/score` | GET | Per-task and overall scores |
| `/docs` | GET | Interactive Swagger UI |

### Example: Full Episode via curl

```bash
# Reset
curl -X POST http://localhost:7860/reset

# Step (email classification)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_name": "email_classification", "payload": {"label": "billing"}}'

# Check score
curl http://localhost:7860/score
```

---

## HuggingFace Spaces

This environment is directly compatible with HuggingFace Spaces:

1. Create a new Space (SDK: Docker or blank)
2. Upload all files from `support_ops_env/`
3. Set Secrets: `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`
4. The Space will start on port 7860 automatically

---

## License

MIT License — see LICENSE for details.
