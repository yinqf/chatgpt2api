from __future__ import annotations

import asyncio
import threading
import unittest

from services.image_execution import ImageCapacityExceeded, ImageExecution


class ImageExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_capacity_is_rejected_without_waiting(self) -> None:
        execution = ImageExecution(max_workers=1)
        slot = execution.acquire()
        try:
            with self.assertRaises(ImageCapacityExceeded):
                execution.acquire()
        finally:
            slot.release()
            execution.shutdown()

    async def test_release_is_idempotent_and_capacity_is_reusable(self) -> None:
        execution = ImageExecution(max_workers=1)
        first = execution.acquire()
        first.release()
        first.release()
        second = execution.acquire()
        second.release()
        execution.shutdown()

    async def test_handler_runs_on_dedicated_image_thread(self) -> None:
        execution = ImageExecution(max_workers=1)
        try:
            thread_name = await execution.run(lambda: threading.current_thread().name)
        finally:
            execution.shutdown()

        self.assertTrue(thread_name.startswith("image-worker"))

    async def test_failed_handler_does_not_break_executor(self) -> None:
        execution = ImageExecution(max_workers=1)

        def fail() -> None:
            raise RuntimeError("boom")

        try:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await execution.run(fail)
            thread_name = await execution.run(lambda: threading.current_thread().name)
        finally:
            execution.shutdown()

        self.assertTrue(thread_name.startswith("image-worker"))

    async def test_stream_releases_slot_when_iteration_ends(self) -> None:
        execution = ImageExecution(max_workers=1)
        slot = execution.acquire()
        try:
            items = [item async for item in execution.stream(iter(["chunk"]), slot)]
            replacement = execution.acquire()
            replacement.release()
        finally:
            slot.release()
            execution.shutdown()

        self.assertEqual(items, ["chunk"])

    async def test_cancelled_request_keeps_capacity_until_worker_stops(self) -> None:
        execution = ImageExecution(max_workers=1)
        slot = execution.acquire()
        started = threading.Event()
        finish = threading.Event()

        def wait_for_finish() -> None:
            started.set()
            finish.wait(timeout=2)

        task = asyncio.create_task(execution.run(wait_for_finish))
        try:
            await asyncio.to_thread(started.wait, 1)
            task.cancel()
            await asyncio.sleep(0)
            with self.assertRaises(ImageCapacityExceeded):
                execution.acquire()
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            finish.set()
            slot.release()
            execution.shutdown()


if __name__ == "__main__":
    unittest.main()
