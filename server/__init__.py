# server package - re-exports from _server.py so that
# "uvicorn server:app" and "server:main" still resolve correctly.
from _server import app, main  # noqa: F401

__all__ = ["app", "main"]
