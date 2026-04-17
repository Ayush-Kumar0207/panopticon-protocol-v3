# OpenEnv Starter Kit

> **Ready-to-deploy** boilerplate for the Meta PyTorch OpenEnv Hackathon.
> All infrastructure is pre-built. You just need to fill in your problem domain.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test
python smoke_test.py

# Start server
uvicorn server:app --host 0.0.0.0 --port 8000

# Run inference (requires HF_TOKEN)
HF_TOKEN=your_token python inference.py
```

## What To Customize (TODO markers)

1. **`models.py`** — Define your Enums (states, actions) and Pydantic models
2. **`environment.py`** — Implement `reset()` scenario generation and `step()` action effects
3. **`grader.py`** — Tune scoring dimensions and weights per difficulty
4. **`openenv.yaml`** — Update metadata, descriptions, and task definitions
5. **`inference.py`** — Write the LLM system prompt for your domain

## Architecture

```
├── models.py          # Pydantic data models (State, Action, Observation)
├── environment.py     # Core simulation engine (reset, step)
├── grader.py          # Multi-dimensional programmatic graders (5 levels)
├── _server.py         # FastAPI server (all required endpoints)
├── server/            # Package re-export (uvicorn server:app)
├── tasks/             # Task definitions with grader instances
├── client.py          # HTTP + Local client for testing
├── inference.py       # LLM agent inference script
├── smoke_test.py      # Quick end-to-end validation
├── openenv.yaml       # OpenEnv configuration
├── Dockerfile         # Production container
├── requirements.txt   # Python dependencies
└── pyproject.toml     # Package configuration
```

## Validation Checklist

- [ ] `python smoke_test.py` passes
- [ ] `uvicorn server:app` starts and `/health` returns 200
- [ ] `POST /reset` with `{}` returns observation
- [ ] `POST /step` processes actions correctly
- [ ] `GET /tasks` lists 5 tasks with graders
- [ ] `GET /metadata` returns environment info
- [ ] `docker build .` succeeds
- [ ] Deploy to HuggingFace Spaces
- [ ] `openenv validate` passes
