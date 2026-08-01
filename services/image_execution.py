from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Iterator


DEFAULT_IMAGE_MAX_WORKERS = 100
IMAGE_MAX_WORKERS_ENV = "CHATGPT2API_IMAGE_MAX_WORKERS"


class ImageCapacityExceeded(RuntimeError):
    """Raised when every image worker is already reserved."""


class ImageSlot:
    def __init__(self, semaphore: threading.BoundedSemaphore):
        self._semaphore = semaphore
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        """Release at most once so overlapping cleanup paths are safe."""
        with self._lock:
            if self._released:
                return
            self._released = True
            self._semaphore.release()


class ImageExecution:
    """No-wait admission control backed by a dedicated image thread pool."""

    def __init__(self, max_workers: int = DEFAULT_IMAGE_MAX_WORKERS):
        self.max_workers = max(1, int(max_workers))
        self._slots = threading.BoundedSemaphore(self.max_workers)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="image-worker",
        )

    def acquire(self) -> ImageSlot:
        if not self._slots.acquire(blocking=False):
            raise ImageCapacityExceeded("all image workers are busy")
        return ImageSlot(self._slots)

    async def run(self, handler: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, partial(handler, *args))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Threads cannot be killed safely. Do not let callers release the slot
            # until the worker has actually stopped after a client disconnect.
            try:
                await future
            except Exception:
                pass
            raise

    async def stream(self, iterator: Iterator[Any], slot: ImageSlot):
        try:
            while True:
                found, item = await self.run(_next_item, iterator)
                if not found:
                    return
                yield item
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            slot.release()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _next_item(iterator: Iterator[Any]) -> tuple[bool, Any]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _configured_max_workers() -> int:
    try:
        return max(1, int(os.getenv(IMAGE_MAX_WORKERS_ENV, DEFAULT_IMAGE_MAX_WORKERS)))
    except (TypeError, ValueError):
        return DEFAULT_IMAGE_MAX_WORKERS


image_execution = ImageExecution(_configured_max_workers())
