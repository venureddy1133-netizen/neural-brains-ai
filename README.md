# Neural Brains AI 3.0 — Voice Conversation

This release preserves the original v2 classifier, SQLite experience memory, persistent weights, dashboard generator, and tests. It adds a local Python HTTP server and a browser voice interface.

## Requirements
- Python 3.10 or newer (the core server uses only the Python standard library).
- A recent Google Chrome browser for the Web Speech API microphone experience.
- Microphone permission for `http://127.0.0.1:8765`.

## Run on macOS
From the project root in VS Code Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m unittest discover -s tests -v
python3 -m neural_brains --voice
```

Open **http://127.0.0.1:8765** in Google Chrome. The page is the interactive 3.0E command center with Overview, Voice & Chat, Experience Memory, Neural Network, and Settings & Status sections. It refreshes live statistics and recent memory every 15 seconds. Keep the terminal running while using the assistant. Press the microphone button and speak. Enable wake-word mode to say **“Hey Neural Brains”** followed by your question. The page must remain open; it does not listen while Chrome is closed.

Stop the server with `Ctrl+C`.

## Voice and privacy notes
- Speech recognition uses the browser's Web Speech API. Chrome may send audio to an online speech recognition service; this is not guaranteed to be offline.
- Speech output uses the browser's built-in speech synthesis.
- The Python API server binds to `127.0.0.1` only and is not exposed to your local network.
- Conversation text and responses are stored in `data/brain_memory.sqlite3` as plaintext. Do not store passwords, financial details, or other sensitive information.
- If microphone permission is denied, allow it in Chrome's site settings and reload the page.

## Optional: connect a local conversational model with Ollama
Without Ollama, the assistant uses a deliberately limited fallback. For broader open-ended answers, install Ollama for macOS, then run:

```bash
ollama pull llama3.2:3b
ollama serve
```

In another terminal, start Neural Brains as above. The Python app will use `llama3.2:3b` at `http://127.0.0.1:11434` when it is available. To select another model:

```bash
NEURAL_BRAINS_MODEL=qwen2.5:3b python3 -m neural_brains --voice
```

The model is optional; the base voice interface does not require it or any pip packages.

## Existing v2 commands
```bash
python3 -m neural_brains --demo
python3 -m neural_brains --dashboard
python3 -m neural_brains
```

Interactive commands: `think`, `teach`, `memory`, `dashboard`, `stats`, `quit`.

## Tests
```bash
python3 -m unittest discover -s tests -v
```

## Interactive dashboard (3.0E)
- Overview metrics for stored experiences, training updates, output classes, and model persistence.
- Experience distribution chart based on the latest 30 memory records.
- Recent activity feed and conceptual network visualization.
- Voice & Chat page with microphone controls, typed chat, and speech output.
- Searchable, filterable, read-only memory browser.
- Neural Network architecture view and Settings & Status diagnostics.
- Manual refresh and automatic refresh every 15 seconds.

## Architecture
- Existing `BrainAgent` and `NeuralNetwork`: three-class intent classifier.
- Existing `MemoryStore`: SQLite experience store.
- New `ConversationEngine`: optional local Ollama chat plus explicit limited fallback.
- New `server.py`: loopback-only HTTP API.
- New `web/index.html`: browser microphone, optional wake-word phrase matching, chat UI, and spoken responses.

This is a voice-enabled assistant prototype, not AGI, not a human brain, and not an autonomous robot controller.
