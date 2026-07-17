"""Static packaging checks that do not require a Docker daemon."""

from __future__ import annotations

import shlex
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_local_docker_copy_source_exists() -> None:
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY "):
            continue
        parts = shlex.split(stripped)
        assert len(parts) == 3, f"unsupported COPY form in static validator: {line}"
        source = parts[1].rstrip("/")
        assert (ROOT / source).exists(), f"Docker COPY source is missing: {source}"


def test_server_entrypoints_use_the_checked_in_fastapi_module() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["entry-points"]["openenv.server"]["panopticon"] == "_server:app"
    assert project["project"]["scripts"]["server"] == "_server:main"
    openenv = (ROOT / "openenv.yaml").read_text(encoding="utf-8")
    assert "app: _server:app" in openenv
    assert 'CMD ["uvicorn", "_server:app"' in (ROOT / "Dockerfile").read_text(encoding="utf-8")
