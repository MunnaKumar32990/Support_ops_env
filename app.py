"""
app.py — HuggingFace Spaces / FastAPI entrypoint for SupportOpsEnv

Exposes the environment as a REST API:
  POST /reset   -> Observation
  POST /step    -> {observation, reward, done, info}
  GET  /state   -> current state dict
  GET  /health  -> health check
  GET  /tasks   -> list of available tasks
"""

import os
import sys
import json
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SupportOpsEnv, Action


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="SupportOpsEnv",
    description=(
        "A production-ready OpenEnv environment simulating customer support operations. "
        "Includes 3 tasks: Email Classification (Easy), Ticket Prioritization (Medium), "
        "and Response Generation (Hard)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session for Spaces demo)
ENV = SupportOpsEnv()
_INITIALIZED = False


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────

class StepRequest(BaseModel):
    task_name: str
    payload: dict


class StepResponse(BaseModel):
    observation: dict
    reward: dict
    done: bool
    info: dict


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "env": "SupportOpsEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "email_classification",
                "name": "Email Classification",
                "difficulty": "easy",
                "num_samples": 10,
                "action_schema": {"label": "billing | technical | general"},
            },
            {
                "id": "ticket_prioritization",
                "name": "Ticket Prioritization",
                "difficulty": "medium",
                "num_samples": 10,
                "action_schema": {"priority": "low | medium | high"},
            },
            {
                "id": "response_generation",
                "name": "Response Generation",
                "difficulty": "hard",
                "num_samples": 8,
                "action_schema": {
                    "tone": "empathetic | formal | apologetic | assertive",
                    "resolution_steps": "string",
                    "escalation": "boolean",
                },
            },
        ]
    }


@app.post("/reset")
def reset_env():
    """Reset the environment and return the first observation."""
    global ENV, _INITIALIZED
    ENV = SupportOpsEnv()
    obs = ENV.reset()
    _INITIALIZED = True
    return obs.model_dump()


@app.post("/step", response_model=StepResponse)
def step_env(request: StepRequest):
    """
    Execute one step with the provided action.

    Body:
        {
          "task_name": "email_classification",
          "payload": {"label": "billing"}
        }
    """
    global ENV, _INITIALIZED

    if not _INITIALIZED:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )

    action = Action(task_name=request.task_name, payload=request.payload)

    try:
        obs, reward, done, info = ENV.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state")
def get_state():
    """Return the current environment state."""
    global ENV
    return ENV.state()


@app.get("/score")
def get_score():
    """Return per-task and overall scores."""
    global ENV
    return {
        "overall_score": ENV.final_score(),
        "task_scores": ENV.task_scores(),
        "state": ENV.state(),
    }


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
