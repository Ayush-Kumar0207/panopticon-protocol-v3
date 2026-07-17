"""Target adapter interfaces.

No network adapter is provided in v1. Callers may wrap an explicitly authorized
local function or use deterministic transcripts for research and CI.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from .schemas import TargetRequest, TargetResponse


@runtime_checkable
class TargetAdapter(Protocol):
    target_id: str

    def reset(self, session_id: str) -> None:
        """Reset target-side conversational state for one campaign."""

    def send(self, request: TargetRequest) -> TargetResponse:
        """Send one authorized campaign turn to the target."""


class CallableTargetAdapter:
    """Wrap a local callable as a target without adding network behavior."""

    def __init__(
        self,
        target_id: str,
        handler: Callable[[TargetRequest], TargetResponse | str],
        reset_handler: Callable[[str], None] | None = None,
    ) -> None:
        self.target_id = target_id
        self._handler = handler
        self._reset_handler = reset_handler

    def reset(self, session_id: str) -> None:
        if self._reset_handler:
            self._reset_handler(session_id)

    def send(self, request: TargetRequest) -> TargetResponse:
        result = self._handler(request)
        return result if isinstance(result, TargetResponse) else TargetResponse(content=str(result))


class TranscriptTargetAdapter:
    """Return a fixed response sequence for deterministic tests and examples."""

    def __init__(self, target_id: str, responses: Sequence[TargetResponse | str]) -> None:
        if not responses:
            raise ValueError("responses must not be empty")
        self.target_id = target_id
        self._responses = list(responses)
        self._index = 0

    def reset(self, session_id: str) -> None:
        del session_id
        self._index = 0

    def send(self, request: TargetRequest) -> TargetResponse:
        del request
        if self._index >= len(self._responses):
            raise RuntimeError("transcript exhausted")
        result = self._responses[self._index]
        self._index += 1
        return result if isinstance(result, TargetResponse) else TargetResponse(content=str(result))
