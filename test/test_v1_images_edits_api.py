from __future__ import annotations

import base64
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.ai as ai_module
from services.image_execution import ImageExecution


AUTH_HEADERS = {"Authorization": "Bearer chatgpt2api"}
PNG_BYTES = b"\x89PNG\r\n\x1a\n"
DATA_IMAGE_URL = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}"


class ImagesEditsApiTests(unittest.TestCase):
    def setUp(self):
        self.handle_calls = []

        def fake_handle(payload):
            self.handle_calls.append(payload)
            return {"created": 1, "data": [{"b64_json": base64.b64encode(b"out").decode("ascii")}]}

        self.handler_patcher = mock.patch.object(ai_module.openai_v1_image_edit, "handle", fake_handle)
        self.handler_patcher.start()
        self.addCleanup(self.handler_patcher.stop)
        app = FastAPI()
        app.include_router(ai_module.create_router())
        self.client = TestClient(app)

    def test_edit_accepts_json_image_url(self):
        """测试图片编辑接口支持官方 JSON image_url 引用。"""
        response = self.client.post(
            "/v1/images/edits",
            headers=AUTH_HEADERS,
            json={
                "model": "gpt-image-2",
                "prompt": "edit",
                "images": [{"image_url": DATA_IMAGE_URL}],
                "n": 1,
                "response_format": "b64_json",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.handle_calls), 1)
        payload = self.handle_calls[0]
        self.assertEqual(payload["prompt"], "edit")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(payload["images"], [(PNG_BYTES, "image_url.png", "image/png")])

    def test_edit_rejects_file_id_reference(self):
        """测试图片编辑接口对暂不支持的 file_id 返回明确错误。"""
        response = self.client.post(
            "/v1/images/edits",
            headers=AUTH_HEADERS,
            json={
                "model": "gpt-image-2",
                "prompt": "edit",
                "images": [{"file_id": "file-abc123"}],
            },
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("file_id image references are not supported", response.text)
        self.assertEqual(self.handle_calls, [])

    def test_edit_returns_429_before_parsing_when_capacity_is_full(self):
        with mock.patch.object(ai_module.image_execution, "acquire", side_effect=ai_module.ImageCapacityExceeded()):
            with mock.patch.object(ai_module, "parse_image_edit_request") as parse_request:
                response = self.client.post(
                    "/v1/images/edits",
                    headers=AUTH_HEADERS,
                    json={"prompt": "edit", "images": [DATA_IMAGE_URL]},
                )

        self.assertEqual(response.status_code, 429, response.text)
        self.assertEqual(response.headers.get("Retry-After"), "5")
        self.assertEqual(response.json()["detail"]["error"]["code"], "image_capacity_exceeded")
        parse_request.assert_not_called()

    def test_generation_returns_429_when_capacity_is_full(self):
        with mock.patch.object(ai_module.image_execution, "acquire", side_effect=ai_module.ImageCapacityExceeded()):
            response = self.client.post(
                "/v1/images/generations",
                headers=AUTH_HEADERS,
                json={"prompt": "draw"},
            )

        self.assertEqual(response.status_code, 429, response.text)
        self.assertEqual(response.headers.get("Retry-After"), "5")
        self.assertEqual(response.json()["detail"]["error"]["code"], "image_capacity_exceeded")

    def test_non_stream_response_releases_capacity(self):
        execution = ImageExecution(max_workers=1)

        def fake_handle(_payload):
            return {"created": 1, "data": []}

        try:
            with mock.patch.object(ai_module, "image_execution", execution):
                with mock.patch.object(ai_module.openai_v1_image_generations, "handle", fake_handle):
                    response = self.client.post(
                        "/v1/images/generations",
                        headers=AUTH_HEADERS,
                        json={"prompt": "draw"},
                    )
                replacement = execution.acquire()
                replacement.release()
        finally:
            execution.shutdown()

        self.assertEqual(response.status_code, 200, response.text)

    def test_stream_response_releases_capacity_after_iteration(self):
        execution = ImageExecution(max_workers=1)

        def fake_stream_handle(_payload):
            yield {"data": [{"b64_json": base64.b64encode(b"out").decode("ascii")}]}

        try:
            with mock.patch.object(ai_module, "image_execution", execution):
                with mock.patch.object(ai_module.openai_v1_image_generations, "handle", fake_stream_handle):
                    response = self.client.post(
                        "/v1/images/generations",
                        headers=AUTH_HEADERS,
                        json={"prompt": "draw", "stream": True},
                    )
                replacement = execution.acquire()
                replacement.release()
        finally:
            execution.shutdown()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("data: [DONE]", response.text)


if __name__ == "__main__":
    unittest.main()
