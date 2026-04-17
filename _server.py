"""
OpenEnv Starter Kit — FastAPI Server
======================================

This is the CRITICAL file. The OpenEnv validator checks:
  1. POST /reset returns 200 with an observation
  2. POST /step returns (observation, reward, done, truncated, info)
  3. GET /tasks returns tasks with graders
  4. GET /metadata returns environment metadata
  5. POST /grade/{task_id} grades an episode

DO NOT rename this file. The server/ package re-exports `app` from here.
"""

from __future__ import annotations
from typing import Any, Literal
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from environment import Environment, StepResult
from models import AgentAction, EnvironmentObservation, EnvironmentState, ActionType
from grader import GRADERS, grade_episode as _grade_episode, list_graders, GraderResult
from tasks import TASK_REGISTRY, get_task, list_tasks, list_tasks_with_graders


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class ResetRequest(BaseModel):
    task_level: str = "easy"
    seed: int | None = None

class StepRequest(BaseModel):
    action_type: str
    target: str = ""
    task_id: str | None = None
    reason: str = ""

class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    truncated: bool
    info: dict

class ObservationResponse(BaseModel):
    observation: dict

class StateResponse(BaseModel):
    state: dict

class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


# =============================================================================
# ENVIRONMENT WRAPPER
# =============================================================================

_env: Environment | None = None

def get_env() -> Environment:
    global _env
    if _env is None:
        _env = Environment()
    return _env


# =============================================================================
# FASTAPI APP
# =============================================================================

# TODO: Update title and description for your problem.
app = FastAPI(
    title="OpenEnv Starter Kit",
    description="OpenEnv-compliant RL environment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# =============================================================================
# CORE ENDPOINTS (Required by OpenEnv validator)
# =============================================================================

@app.get("/")
async def root():
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists() and (static_dir / "index.html").exists():
        return RedirectResponse(url="/dashboard/")
    return {
        "name": "OpenEnv Starter Kit",  # TODO: Change
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health", "reset": "POST /reset",
            "step": "POST /step", "observation": "GET /observation",
            "tasks": "GET /tasks", "metadata": "GET /metadata",
            "grade": "POST /grade/{task_id}",
        },
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", environment="openenv-starter", version="1.0.0")

@app.post("/reset", response_model=ObservationResponse)
async def reset_environment(request: ResetRequest | None = None):
    try:
        env = get_env()
        if request is None:
            request = ResetRequest()
        obs = env.reset(task_level=request.task_level, seed=request.seed)
        return ObservationResponse(observation=obs.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/step", response_model=StepResponse)
async def step_environment(request: StepRequest):
    try:
        env = get_env()
        try:
            action_type = ActionType(request.action_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid action_type: {request.action_type}")

        action = AgentAction(action_type=action_type, target=request.target,
                             task_id=request.task_id, reason=request.reason)
        result = env.step(action)
        return StepResponse(observation=result.observation.model_dump(), reward=result.reward,
                            done=result.done, truncated=result.truncated, info=result.info)
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/observation", response_model=ObservationResponse)
async def get_observation():
    try:
        env = get_env()
        obs = env.get_observation()
        return ObservationResponse(observation=obs.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=StateResponse)
async def get_state():
    try:
        env = get_env()
        return StateResponse(state=env.state.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/render")
async def render_environment():
    try:
        return {"render": get_env().render()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema/action")
async def get_action_schema():
    return AgentAction.model_json_schema()

@app.get("/schema/observation")
async def get_observation_schema():
    return EnvironmentObservation.model_json_schema()

@app.get("/schema")
async def get_schemas():
    return {
        "action": AgentAction.model_json_schema(),
        "observation": EnvironmentObservation.model_json_schema(),
        "state": EnvironmentState.model_json_schema(),
    }


# =============================================================================
# TASKS & GRADERS ENDPOINTS (Required for Phase 2 validation)
# =============================================================================

def _serialize_task(task: dict) -> dict:
    return {
        "id": task["id"], "name": task["name"], "description": task["description"],
        "max_turns": task["max_turns"], "difficulty": task["difficulty"],
        "has_grader": task.get("has_grader", False), "grader": task.get("grader", {}),
        "success_criteria": task.get("success_criteria", {}),
    }

@app.get("/tasks")
async def get_tasks_endpoint():
    all_tasks = list_tasks()
    serialized = [_serialize_task(t) for t in all_tasks]
    with_graders = [t for t in serialized if t.get("has_grader")]
    return {"tasks": serialized, "count": len(serialized),
            "tasks_with_graders": len(with_graders), "graders_available": len(with_graders)}

@app.get("/tasks/{task_id}")
async def get_task_endpoint(task_id: str):
    try:
        return _serialize_task(get_task(task_id))
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

@app.get("/graders")
async def get_graders_endpoint():
    graders = list_graders()
    return {"graders": graders, "count": len(graders)}

@app.post("/grade/{task_id}")
async def grade_episode_endpoint(task_id: str, episode_data: dict):
    try:
        result = _grade_episode(task_id, episode_data)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading error: {str(e)}")

@app.get("/metadata")
async def get_metadata():
    all_tasks = list_tasks()
    serialized = [_serialize_task(t) for t in all_tasks]
    with_graders = [t for t in serialized if t.get("has_grader")]
    graders = list_graders()
    return {
        "name": "openenv-starter",           # TODO: Change
        "display_name": "OpenEnv Starter Kit", # TODO: Change
        "description": "An OpenEnv-compliant RL environment.",  # TODO: Change
        "version": "1.0.0",
        "author": "Your Team Name",           # TODO: Change
        "license": "Apache-2.0",
        "tasks": serialized,
        "tasks_count": len(serialized),
        "tasks_with_graders": len(with_graders),
        "graders_count": len(graders),
        "graders": graders,
        "evaluation": {"default_task": "medium", "scoring_range": [0.0, 1.0], "primary_metric": "normalized_score"},
        "tags": ["openenv", "hackathon"],
    }


# Static files (dashboard)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_static_dir), html=True), name="dashboard")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
