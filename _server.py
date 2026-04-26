"""
The Panopticon Protocol v3 — FastAPI Server
=============================================

OpenEnv-compliant REST API for the counter-espionage RL environment.

Required endpoints:
  POST /reset — Reset to new episode
  POST /step  — Execute agent action
  GET  /tasks — List tasks with graders
  GET  /metadata — Environment metadata
  POST /grade/{task_id} — Grade an episode
"""

from __future__ import annotations
from typing import Any
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from environment import Environment, StepResult
from models import AgentAction, EnvironmentObservation, EnvironmentState, ActionType, SubAction, validate_action
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
    sub_action: str = "none"
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


class AgentStatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    policy: str
    model_ref: str
    local_model_present: bool
    loaded: bool
    error: str | None = None


class AgentStepResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    observation: dict
    reward: float
    done: bool
    truncated: bool
    info: dict
    agent_action: dict
    agent_policy: str
    agent_raw_text: str = ""
    model_info: dict | None = None


# =============================================================================
# ENVIRONMENT WRAPPER
# =============================================================================

_env: Environment | None = None
_agent_policy: Any | None = None
_agent_load_error: str | None = None


def get_env() -> Environment:
    global _env
    if _env is None:
        _env = Environment()
    return _env


def resolve_agent_model_ref() -> str:
    explicit_ref = os.environ.get("ARGUS_MODEL_REF")
    if explicit_ref:
        return explicit_ref

    local_model = Path(__file__).parent / "trained_model"
    if local_model.exists():
        return str(local_model)

    return "Ayush-Kumar0207/panopticon-argus-qwen-1.5B"


def get_agent_policy():
    global _agent_policy, _agent_load_error
    if _agent_policy is not None:
        return _agent_policy

    try:
        from inference_local import LocalModelPolicy

        _agent_policy = LocalModelPolicy(
            resolve_agent_model_ref(),
            deterministic=True,
            temperature=0.0,
            top_p=1.0,
        )
        _agent_load_error = None
        return _agent_policy
    except Exception as exc:  # pragma: no cover - depends on local model/runtime availability
        _agent_load_error = str(exc)
        raise


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="The Panopticon Protocol v3",
    description=(
        "Counter-espionage RL environment where ARGUS (AI agent) defends a "
        "corporate network against HYDRA (adaptive adversary) using canary traps, "
        "double agents, and disinformation campaigns. Among Us… for AIs."
    ),
    version="3.0.0",
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
        "name": "The Panopticon Protocol v3",
        "tagline": "Among Us… for AIs",
        "version": "3.0.0",
        "mechanics": [
            "Canary Traps", "Multi-gen Sleepers", "False Flags",
            "Dead-man's Switches", "Double Agent Turning",
            "Disinformation Campaigns", "HYDRA Adaptive Memory",
        ],
        "endpoints": {
            "health": "GET /health", "reset": "POST /reset",
            "step": "POST /step", "observation": "GET /observation",
            "tasks": "GET /tasks", "metadata": "GET /metadata",
            "grade": "POST /grade/{task_id}",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        environment="panopticon-protocol-v3",
        version="3.0.0",
    )


@app.get("/agent/status", response_model=AgentStatusResponse)
async def get_agent_status():
    model_ref = resolve_agent_model_ref()
    local_model_present = Path(model_ref).exists()

    if _agent_policy is not None:
        return AgentStatusResponse(
            status="ready",
            policy="trained",
            model_ref=model_ref,
            local_model_present=local_model_present,
            loaded=True,
            error=None,
        )

    if _agent_load_error:
        return AgentStatusResponse(
            status="error",
            policy="trained",
            model_ref=model_ref,
            local_model_present=local_model_present,
            loaded=False,
            error=_agent_load_error,
        )

    return AgentStatusResponse(
        status="configured",
        policy="trained",
        model_ref=model_ref,
        local_model_present=local_model_present,
        loaded=False,
        error=None,
    )


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

        # Validate action type
        try:
            action_type = ActionType(request.action_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action_type: '{request.action_type}'. "
                       f"Valid: {[a.value for a in ActionType]}"
            )

        # Validate sub-action
        try:
            sub_action = SubAction(request.sub_action)
        except ValueError:
            sub_action = SubAction.NONE

        action = AgentAction(
            action_type=action_type.value,
            target=request.target,
            sub_action=sub_action.value,
            reason=request.reason,
        )
        result = env.step(action)
        return StepResponse(
            observation=result.observation.model_dump(),
            reward=result.reward,
            done=result.done,
            truncated=result.truncated,
            info=result.info,
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/agent/step", response_model=AgentStepResponse)
async def step_with_trained_agent():
    try:
        env = get_env()
        obs = env.get_observation()
        policy = get_agent_policy()
        decision = policy.act(obs)

        is_valid, validation_error = validate_action(decision.action, obs)
        if not is_valid:
            decision.action = AgentAction(
                action_type=ActionType.NOOP.value,
                reason=f"Model produced invalid action: {validation_error}",
            )

        result = env.step(decision.action)
        info = dict(result.info)
        info["agent_action"] = decision.action.model_dump()
        if decision.action.reason:
            info["agent_reason"] = decision.action.reason

        model_info = policy.model_info() if hasattr(policy, "model_info") else None
        return AgentStepResponse(
            observation=result.observation.model_dump(),
            reward=result.reward,
            done=result.done,
            truncated=result.truncated,
            info=info,
            agent_action=decision.action.model_dump(),
            agent_policy=getattr(policy, "policy_name", "trained"),
            agent_raw_text=decision.raw_text,
            model_info=model_info,
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Trained agent unavailable: {str(e)}")


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
# TASKS & GRADERS ENDPOINTS
# =============================================================================

def _serialize_task(task: dict) -> dict:
    return {
        "id": task["id"], "name": task["name"],
        "description": task["description"],
        "max_turns": task["max_turns"],
        "difficulty": task["difficulty"],
        "has_grader": task.get("has_grader", False),
        "grader": task.get("grader", {}),
        "success_criteria": task.get("success_criteria", {}),
    }


@app.get("/tasks")
async def get_tasks_endpoint():
    all_tasks = list_tasks()
    serialized = [_serialize_task(t) for t in all_tasks]
    with_graders = [t for t in serialized if t.get("has_grader")]
    return {
        "tasks": serialized,
        "count": len(serialized),
        "tasks_with_graders": len(with_graders),
        "graders_available": len(with_graders),
    }


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
        "name": "panopticon-protocol-v3",
        "display_name": "The Panopticon Protocol v3",
        "description": (
            "Counter-espionage RL environment where ARGUS defends a corporate "
            "network against HYDRA using canary traps, double agents, and "
            "disinformation campaigns. 7 stacking mechanics, 5 sleeper generations, "
            "6-phase narrative arc with adaptive adversary. Among Us… for AIs."
        ),
        "version": "3.0.0",
        "author": "Team Panopticon",
        "license": "Apache-2.0",
        "tasks": serialized,
        "tasks_count": len(serialized),
        "tasks_with_graders": len(with_graders),
        "graders_count": len(graders),
        "graders": graders,
        "evaluation": {
            "default_task": "medium",
            "scoring_range": [0.0, 1.0],
            "primary_metric": "normalized_score",
        },
        "mechanics": [
            "canary_traps", "multi_gen_sleepers", "false_flags",
            "dead_man_switches", "double_agent_turning",
            "disinformation_campaigns", "hydra_adaptive_memory",
        ],
        "tags": [
            "openenv", "hackathon", "counter-espionage", "adversarial-rl",
            "multi-agent", "deception-games", "turn-based",
            "json-observation", "json-action", "llm-agent",
            "multi-dimensional-grading",
        ],
    }


# ── Static files (dashboard) ──
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_static_dir), html=True), name="dashboard")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
