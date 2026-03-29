"""Async state slot: bridges the processing thread to asyncio SSE handlers."""

import asyncio


class StateSlot:
    """Latest-value holder that signals a waiting coroutine.

    Calling put_threadsafe() from any thread overwrites the current value and
    wakes the consumer. Intermediate frames are silently dropped — no queue.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._value: dict | None = None
        self._event = asyncio.Event()

    def put_threadsafe(self, value: dict) -> None:
        self._value = value
        self._loop.call_soon_threadsafe(self._event.set)

    async def get(self) -> dict:
        await self._event.wait()
        self._event.clear()
        return self._value  # type: ignore[return-value]
