"""
agent/cancellation.py   [CHANGE 4 — new file]
──────────────────────
Lightweight in-process cancellation registry.

Each active /chat call registers a CancellationToken keyed by session_id.
POST /chat/stop signals the token, which causes the running coroutine
to raise asyncio.CancelledError at the next check point.

Thread-safe via asyncio primitives (all access is on the event loop thread).
"""

from __future__ import annotations

import asyncio


class CancellationToken:
    """A simple event-based cancellation flag."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Signal cancellation."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def raise_if_cancelled(self) -> None:
        """Call this at checkpoints inside long-running agent steps."""
        if self._event.is_set():
            raise asyncio.CancelledError("Generation stopped by user.")


class CancellationRegistry:
    """Maps session_id → CancellationToken for all in-flight requests."""

    def __init__(self) -> None:
        self._tokens: dict[str, CancellationToken] = {}

    def register(self, session_id: str) -> CancellationToken:
        token = CancellationToken()
        self._tokens[session_id] = token
        return token

    def cancel(self, session_id: str) -> bool:
        """Signal cancellation for a session. Returns True if a token was found."""
        token = self._tokens.get(session_id)
        if token:
            token.cancel()
            return True
        return False

    def unregister(self, session_id: str) -> None:
        self._tokens.pop(session_id, None)