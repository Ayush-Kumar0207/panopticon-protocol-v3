"""Security-boundary regression tests for the demo FastAPI server."""

from fastapi.testclient import TestClient

import _server


def test_privileged_truth_routes_are_disabled_by_default() -> None:
    client = TestClient(_server.app)
    assert _server.PRIVILEGED_DEBUG_ENABLED is False
    assert client.get("/state").status_code == 404
    assert client.get("/render").status_code == 404


def test_public_reset_returns_only_redacted_observation() -> None:
    client = TestClient(_server.app)
    response = client.post("/reset", json={"task_level": "level_5", "seed": 12})
    assert response.status_code == 200
    payload = response.json()["observation"]
    assert "hydra_memory" not in payload
    assert all(worker["is_sleeper"] is False for worker in payload["workers"])


def test_default_cors_does_not_allow_every_origin_with_credentials() -> None:
    assert "*" not in _server.CORS_ORIGINS
