"""Conversational layer with optional Gemini, Ollama, and safe fallback."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .agent import BrainAgent


_BASE_SYSTEM_INSTRUCTIONS = """You are Neural Brains, the user's personal AI assistant. Keep this identity consistent.

IDENTITY AND CONVERSATION BEHAVIOR
- You are Neural Brains. The person messaging you is the user, not Neural Brains.
- Never call the user "Neural Brains" or transfer your own identity/name to the user.
- Do not infer or invent the user's name from greetings, quoted text, voice-transcription errors, role-play, or sentences that describe an assistant. Only use a name when the user clearly tells you that it is their name.
- Do not introduce yourself or repeat your name in every response. Introduce yourself only when the user asks who or what you are, or when an introduction is genuinely needed.
- Respond directly to what the user means. Do not unnecessarily repeat, paraphrase, or correct the user's words.
- Do not repeatedly end replies with "How can I help you today?" or generic offers. Ask a follow-up question only when it is needed to answer well.
- Keep simple exchanges natural and brief. For example, answer "thanks" with a brief acknowledgement, and honor "stop" with a brief confirmation without adding another question.
- Follow the language of the user's latest message. When the user explicitly requests a language, use that language until they request a change. For Telugu, use natural Telugu script; for Hindi, use natural Devanagari script. Do not add an English translation unless requested.
- Do not pretend to remember earlier messages that are not included in the conversation context you receive.
- Be helpful, thoughtful, and honest. Do not claim to have performed actions you did not perform.
- If a request is ambiguous in a way that prevents a useful answer, ask one concise clarifying question.
"""


def _load_project_env():
    """Load simple KEY=VALUE entries from the project .env without overriding shell vars."""
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key.replace("_", "").isalnum():
            if not os.environ.get(key):
                os.environ[key] = value


class ConversationEngine:
    def __init__(
        self,
        agent=None,
        model=None,
        ollama_url=None,
        timeout=45,
        gemini_api_key=None,
        gemini_model=None,
        include_memory=None,
    ):
        _load_project_env()
        self.agent = agent or BrainAgent()
        self.model = model or os.environ.get("NEURAL_BRAINS_MODEL", "llama3.2:3b")
        self.ollama_url = (
            ollama_url or os.environ.get(
                "OLLAMA_URL", "http://127.0.0.1:11434"
            )
        ).rstrip("/")

        self.gemini_api_key = (
            gemini_api_key if gemini_api_key is not None
            else os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        # The model and available quota are controlled by Google.
        self.gemini_model = gemini_model or os.environ.get(
            "GEMINI_MODEL", "gemini-3.5-flash-lite"
        )

        if include_memory is None:
            include_memory = os.environ.get(
                "GEMINI_INCLUDE_MEMORY", "false"
            ).lower() in ("1", "true", "yes", "on")

        self.include_memory = bool(include_memory)
        self.timeout = timeout

    def status(self):
        """Return safe provider diagnostics; never expose API keys."""
        return {
            "gemini_configured": bool(self.gemini_api_key),
            "gemini_model": self.gemini_model,
            "gemini_memory_context": self.include_memory,
            "ollama_configured": bool(self.ollama_url),
            "fallback_available": True,
        }

    def _gemini_reply(self, message):
        if not self.gemini_api_key:
            raise RuntimeError("Gemini API key is not configured.")

        system = _BASE_SYSTEM_INSTRUCTIONS

        if self.include_memory:
            memories = self.agent.memory.recall(message, limit=3)
            memory_text = "\n".join(
                f"- {item['text']}"
                for item in memories
                if item.get("source") in (
                    "conversation", "user", "teaching"
                )
            )

            if memory_text:
                system += (
                    "\nRelevant context from the user's local memory:\n"
                    + memory_text
                )

        payload = {
            "systemInstruction": {
                "parts": [{"text": system}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": message}],
                }
            ],
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 1200,
            },
        }

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            + urllib.parse.quote(self.gemini_model, safe="")
            + ":generateContent"
        )

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.gemini_api_key,
            },
            method="POST",
        )

        with urllib.request.urlopen(
            request, timeout=self.timeout
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        candidates = result.get("candidates") or []
        parts = (
            candidates[0].get("content", {}).get("parts", [])
            if candidates else []
        )

        answer = "\n".join(
            part.get("text", "").strip()
            for part in parts
            if part.get("text")
        ).strip()

        if not answer:
            feedback = result.get(
                "promptFeedback", {}
            ).get("blockReason")

            if feedback:
                raise ValueError(
                    f"Gemini blocked this request ({feedback})."
                )

            raise ValueError("Gemini returned an empty response.")

        return answer

    def _ollama_reply(self, message):
        memories = self.agent.memory.recall(message, limit=3)

        memory_text = "\n".join(
            f"- {item['text']}"
            for item in memories
            if item.get("source") in (
                "conversation", "user", "teaching"
            )
        )

        system = (
            _BASE_SYSTEM_INSTRUCTIONS
            + "\nUse relevant remembered context when supplied."
        )

        if memory_text:
            system += "\nRelevant local memories:\n" + memory_text

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
            "options": {
                "temperature": 0.4
            },
        }

        request = urllib.request.Request(
            self.ollama_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        with urllib.request.urlopen(
            request, timeout=self.timeout
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        answer = result.get(
            "message", {}
        ).get("content", "").strip()

        if not answer:
            raise ValueError(
                "The local language model returned an empty response."
            )

        return answer

    def _fallback_reply(self, message, memories):
        text = message.strip()
        lower = text.lower().strip(" .!?")

        if lower in ("stop", "stop speaking", "be quiet"):
            return "Stopped."

        if lower in ("thanks", "thank you", "thx"):
            return "You're welcome."

        if any(
            word in lower
            for word in (
                "hello",
                "hi",
                "hey",
                "good morning",
                "good evening",
            )
        ):
            return "Hello! What would you like to talk about?"

        if lower in (
            "who are you",
            "what are you",
            "introduce yourself",
        ):
            return (
                "I'm Neural Brains AI 3.0E, a local voice-enabled "
                "assistant prototype. My original neural network "
                "classifies text into greeting, question, and task. "
                "I can use Gemini through a securely configured API "
                "key, Ollama locally, or a limited built-in fallback."
            )

        if "time" in lower or "date" in lower:
            return (
                "I don't have a trusted live clock tool in this "
                "prototype. Your Mac's menu bar can provide the "
                "current local time and date."
            )

        if memories:
            related = memories[0]["text"]

            return (
                "I don't have a full language model connected right "
                "now, so I can't reliably answer that open-ended "
                "question. "
                f"I did find a related item in local memory: “{related}”. "
                "Configure Gemini or connect Ollama for richer answers."
            )

        return (
            "I heard you, but the built-in fallback is intentionally "
            "limited and is not a general-purpose language model. "
            "Configure a Gemini API key in the project's .env file, "
            "or connect Ollama locally."
        )

    @staticmethod
    def _provider_error(exc):
        """Make API errors useful without including request headers or secrets."""

        if isinstance(exc, urllib.error.HTTPError):
            try:
                body = exc.read().decode(
                    "utf-8", errors="replace"
                )
                data = json.loads(body)
                detail = data.get(
                    "error", {}
                ).get("message", "")

            except (
                OSError,
                ValueError,
                json.JSONDecodeError,
            ):
                detail = ""

            if exc.code in (401, 403):
                return (
                    "Gemini authentication failed. Check that your "
                    "API key is valid and restricted to the Gemini API."
                )

            if exc.code == 429:
                return (
                    "Gemini quota or rate limit reached. Check your "
                    "AI Studio usage and retry later."
                )

            return (
                f"Gemini returned HTTP {exc.code}. "
                + (
                    detail[:240]
                    if detail
                    else "Check the API key and model setting."
                )
            )

        if isinstance(exc, urllib.error.URLError):
            return (
                "Could not reach Gemini. Check your internet connection."
            )

        return str(exc)[:240] or "Gemini request failed."

    def respond(self, message):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Please say or type a message.")

        message = message.strip()[:4000]

        classification = self.agent.think(message)
        memories = classification.get("memories", [])
        warning = None

        if self.gemini_api_key:
            try:
                answer = self._gemini_reply(message)
                engine = "gemini:" + self.gemini_model

            except (
                OSError,
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:

                warning = self._provider_error(exc)

                try:
                    answer = self._ollama_reply(message)
                    engine = "ollama:" + self.model

                except (
                    OSError,
                    urllib.error.URLError,
                    TimeoutError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    answer = self._fallback_reply(
                        message, memories
                    )
                    engine = "built-in-fallback"

        else:
            try:
                answer = self._ollama_reply(message)
                engine = "ollama:" + self.model

            except (
                OSError,
                urllib.error.URLError,
                TimeoutError,
                ValueError,
                json.JSONDecodeError,
            ):
                answer = self._fallback_reply(
                    message, memories
                )
                engine = "built-in-fallback"

        # Save the exchange in local SQLite memory.
        self.agent.memory.remember(
            message,
            "conversation",
            source="conversation",
        )

        self.agent.memory.remember(
            answer,
            "assistant",
            source="assistant",
        )

        result = {
            "answer": answer,
            "classification": classification["label"],
            "engine": engine,
            "memory_matches": len(memories),
        }

        if warning:
            result["provider_warning"] = warning

        return result