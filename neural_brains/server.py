"""Local-only HTTP server for the Neural Brains browser voice interface."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agent import BrainAgent
from .conversation import ConversationEngine
from .dashboard import build_dashboard

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_BODY = 32 * 1024

class Handler(BaseHTTPRequestHandler):
    engine = None

    def _send(self, status, payload, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, (WEB_DIR / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif path == "/api/health":
            self._send(200, {"ok": True, "name": "Neural Brains AI 3.0E", "ai": self.engine.status()})
        elif path == "/api/stats":
            self._send(200, self.engine.agent.stats())
        elif path == "/api/memory":
            self._send(200, {"items": self.engine.agent.memory.recent(30)})
        elif path == "/api/dashboard":
            output = build_dashboard(self.engine.agent)
            self._send(200, {"path": str(output.resolve())})
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            self._send(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                self._send(413, {"error": "Message is empty or too large."})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.engine.respond(payload.get("message", ""))
            self._send(200, result)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception:
            self._send(500, {"error": "The assistant encountered an internal error. Check the VS Code terminal."})

    def log_message(self, fmt, *args):
        print("[Neural Brains server] " + (fmt % args))

def run(host="127.0.0.1", port=8765, data_dir="data"):
    agent = BrainAgent(data_dir)
    Handler.engine = ConversationEngine(agent)
    server = ThreadingHTTPServer((host, port), Handler)
    print("\nNEURAL BRAINS AI 3.0E — VOICE CONVERSATION")
    if Handler.engine.status()["gemini_configured"]:
        print("Gemini API key detected; Gemini will be tried first.")
    else:
        print("Gemini API key not found; checking optional Ollama, then built-in fallback.")
    print(f"Open http://{host}:{port} in Google Chrome")
    print("This server listens only on your Mac (127.0.0.1). Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nStopping Neural Brains...")
    finally:
        server.server_close()

if __name__ == "__main__":
    run()
