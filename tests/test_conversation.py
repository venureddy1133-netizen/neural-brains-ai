import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from neural_brains.agent import BrainAgent
from neural_brains.conversation import ConversationEngine


class _FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


class ConversationTests(unittest.TestCase):
    def test_fallback_reply_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = ConversationEngine(
                agent=BrainAgent(directory),
                ollama_url="http://127.0.0.1:1",
                gemini_api_key="",
                timeout=0.2,
            )
            result = engine.respond("Who are you?")
            self.assertTrue(result["answer"])
            self.assertEqual(result["engine"], "built-in-fallback")
            self.assertGreaterEqual(engine.agent.memory.count(), 3)

    def test_empty_message_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = ConversationEngine(
                agent=BrainAgent(directory),
                ollama_url="http://127.0.0.1:1",
                gemini_api_key="",
                timeout=0.2,
            )
            with self.assertRaises(ValueError):
                engine.respond("   ")

    def test_gemini_backend_uses_key_header_and_returns_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = ConversationEngine(
                agent=BrainAgent(directory),
                gemini_api_key="test-secret-key",
                gemini_model="gemini-test-model",
                include_memory=False,
                ollama_url="http://127.0.0.1:1",
            )
            response = _FakeResponse({
                "candidates": [{
                    "content": {"parts": [{"text": "A neural network learns from examples."}]}
                }]
            })
            with patch("neural_brains.conversation.urllib.request.urlopen", return_value=response) as mocked:
                result = engine.respond("How does a neural network learn?")

            self.assertEqual(result["engine"], "gemini:gemini-test-model")
            self.assertEqual(result["answer"], "A neural network learns from examples.")
            request = mocked.call_args.args[0]
            request_payload = json.loads(request.data.decode("utf-8"))
            self.assertIn("systemInstruction", request_payload)
            self.assertEqual(request.get_header("X-goog-api-key"), "test-secret-key")
            self.assertNotIn("test-secret-key", request.full_url)
            self.assertNotIn("test-secret-key", json.dumps(engine.status()))

    def test_default_gemini_model_is_current_flash_lite(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                engine = ConversationEngine(
                    agent=BrainAgent(directory), gemini_api_key="test-key"
                )
            self.assertEqual(engine.gemini_model, "gemini-3.5-flash-lite")

    def test_gemini_status_never_exposes_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = ConversationEngine(
                agent=BrainAgent(directory), gemini_api_key="secret-value"
            )
            status = engine.status()
            self.assertTrue(status["gemini_configured"])
            self.assertNotIn("secret-value", repr(status))
            self.assertFalse(status["gemini_memory_context"])

    def test_project_env_file_loads_gemini_key(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / ".env").write_text(
                "GEMINI_API_KEY=file-test-key\nGEMINI_MODEL=gemini-test-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "GEMINI_MODEL": ""}):
                with patch("neural_brains.conversation.Path.cwd", return_value=Path(directory)):
                    engine = ConversationEngine(agent=BrainAgent(Path(directory) / "data"))
            self.assertEqual(engine.gemini_api_key, "file-test-key")
            self.assertEqual(engine.gemini_model, "gemini-test-model")

    def test_gemini_quota_error_falls_back_with_clear_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = ConversationEngine(
                agent=BrainAgent(directory),
                gemini_api_key="test-key",
                ollama_url="http://127.0.0.1:1",
                timeout=0.2,
            )
            error = urllib.error.HTTPError(
                "https://example.invalid", 429, "quota", {},
                __import__("io").BytesIO(b'{"error":{"message":"quota reached"}}'),
            )
            with patch("neural_brains.conversation.urllib.request.urlopen", side_effect=error):
                result = engine.respond("Explain machine learning")
            self.assertEqual(result["engine"], "built-in-fallback")
            self.assertIn("quota or rate limit", result["provider_warning"])


if __name__ == "__main__":
    unittest.main()
